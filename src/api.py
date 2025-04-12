"""InsightFace Ray Serve API for face detection and comparison.

This module implements a FastAPI server using Ray Serve for scalable
face detection, attribute extraction, and comparison using the InsightFace library.
The API provides endpoints for detecting faces in images and comparing faces
between two images.

Typical usage:
    The API is deployed using Ray Serve and can be accessed via HTTP requests
    to the endpoints defined in the InsightFaceDeployment class.

Attributes:
    app (FastAPI): The FastAPI application.
    RACE_TAGS (List[str]): Tag names for race classification.
    GENDER_TAGS (List[str]): Tag names for gender classification.
    AGE_BRACKET_TAGS (List[str]): Tag names for age bracket classification.
"""

import base64
import binascii
import io
import logging
import time
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import cv2
import inspireface as isf
import numpy as np
import ray
import requests
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.handle import DeploymentHandle
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from src.models import (
    ComparisonResponse,
    DetectionResponse,
    FaceDetectionResult,
    ImageComparisonInput,
    ImageInput,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Classification tag mappings
RACE_TAGS = ["Black", "Asian", "Latino/Hispanic", "Middle Eastern", "White"]
GENDER_TAGS = ["Female", "Male"]
AGE_BRACKET_TAGS = [
    "0-2 years old",
    "3-9 years old",
    "10-19 years old",
    "20-29 years old",
    "30-39 years old",
    "40-49 years old",
    "50-59 years old",
    "60-69 years old",
    "more than 70 years old",
]



@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=4,
        target_num_ongoing_requests_per_replica=2
    )
)
@serve.ingress(app := FastAPI(
    title="Face Analysis API",
    description="API for face detection, attribute extraction, and comparison.",
    version="1.0.0",
))
class InsightFaceDeployment:
    """Ray Serve deployment for face analysis using InsightFace.
    
    This class implements a scalable service that provides face detection,
    attribute extraction, and face comparison functionality. It uses the
    inspireface library for face processing and is deployed using Ray Serve
    with autoscaling capabilities.
    
    The deployment exposes FastAPI endpoints for detection and comparison
    operations, with full OpenAPI documentation.
    """
    
    def __init__(self):
        """Initialize the InsightFace deployment.
        
        Sets up the inspireface session with all necessary capabilities enabled,
        including face recognition, quality assessment, mask detection,
        liveness detection, and attribute extraction.
        """
        logger.info("Initializing face analysis session...")
        options = (
            isf.HF_ENABLE_FACE_RECOGNITION
            | isf.HF_ENABLE_QUALITY
            | isf.HF_ENABLE_MASK_DETECT
            | isf.HF_ENABLE_LIVENESS
            | isf.HF_ENABLE_INTERACTION
            | isf.HF_ENABLE_FACE_ATTRIBUTE
        )
        self.session = isf.InspireFaceSession(options, isf.HF_DETECT_MODE_ALWAYS_DETECT)
        self.session.set_detection_confidence_threshold(0.5)
        logger.info("Face analysis session initialized successfully.")

    async def _get_image_from_input(
        self, image_base64: Optional[str], image_url: Optional[str]
    ) -> np.ndarray:
        """Process image input from either base64 string or URL.
        
        Handles downloading from HTTP/HTTPS URLs or S3 buckets, or decoding
        from a base64 string, and converts the image to a numpy array for
        processing by OpenCV and InsightFace.
        
        Args:
            image_base64: Optional base64 encoded image string.
            image_url: Optional URL to an image (http, https, or s3).
            
        Returns:
            A numpy array containing the decoded image in BGR format (OpenCV).
            
        Raises:
            ValueError: If no image source is provided or image data is invalid.
            HTTPException: If there's an error retrieving or processing the image.
        """
        image_bytes: Optional[bytes] = None
        if image_base64:
            logger.info("Decoding image from Base64 string.")
            try:
                image_bytes = base64.b64decode(image_base64)
            except binascii.Error as e:
                logger.error(f"Base64 decoding error: {e}")
                raise ValueError(f"Invalid base64 string: {e}") from e
        elif image_url:
            logger.info(f"Fetching image from URL: {image_url}")
            try:
                parsed_url = urlparse(image_url)
                if parsed_url.scheme in ["http", "https"]:
                    response = requests.get(image_url, stream=True, timeout=30)
                    response.raise_for_status()
                    image_bytes = response.content
                elif parsed_url.scheme == "s3":
                    s3 = boto3.client("s3")
                    bucket = parsed_url.netloc
                    key = parsed_url.path.lstrip('/')
                    try:
                        s3_object = s3.get_object(Bucket=bucket, Key=key)
                        image_bytes = s3_object['Body'].read()
                    except NoCredentialsError:
                         logger.error("AWS credentials not found.")
                         raise HTTPException(status_code=500, detail="AWS credentials not configured.")
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'NoSuchKey':
                            logger.error(f"S3 object not found: s3://{bucket}/{key}")
                            raise HTTPException(status_code=404, detail=f"S3 object not found: {image_url}")
                        elif e.response['Error']['Code'] == 'NoSuchBucket':
                             logger.error(f"S3 bucket not found: {bucket}")
                             raise HTTPException(status_code=404, detail=f"S3 bucket not found: {bucket}")
                        else:
                            logger.error(f"S3 ClientError fetching {image_url}: {e}")
                            raise HTTPException(status_code=500, detail=f"Error accessing S3 object: {e}")
                else:
                    raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}")
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP request failed for {image_url}: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error fetching image from URL {image_url}: {e}")
                raise HTTPException(status_code=500, detail=f"Internal error fetching image: {e}")
        else:
            raise ValueError("No image source provided (base64 or url).")
        if image_bytes is None:
             raise ValueError("Failed to retrieve image bytes.")
        try:
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                logger.error(
                    "cv2.imdecode returned None. Invalid image data or format."
                )
                raise ValueError("Could not decode image data.")
            logger.info(f"Image decoded successfully, shape: {image.shape}")
            return image
        except Exception as e:
            logger.exception(f"Unexpected error decoding image bytes: {e}")
            raise ValueError(f"Error processing image data: {e}") from e

    def _format_detection_result(
        self, image: np.ndarray, face: Any, ext: Any, idx: int
    ) -> FaceDetectionResult:
        """Format raw detection data into a structured result model.
        
        Extracts facial landmarks, feature vectors, and demographic attributes
        from the raw detection results and formats them into a structured
        FaceDetectionResult model.
        
        Args:
            image: The source image as a numpy array.
            face: The detected face object from InsightFace.
            ext: The extended face attributes object from InsightFace.
            idx: The index of this face in the detection results.
            
        Returns:
            A FaceDetectionResult object with all available face attributes.
        """
        landmarks_raw = self.session.get_face_dense_landmark(face)
        landmarks: Optional[List[Tuple[int, int]]] = None
        if landmarks_raw is not None:
             processed_landmarks: List[Tuple[int, int]] = []
             for lm in landmarks_raw:
                 if len(lm) >= 2:
                     processed_landmarks.append( (int(lm[0]), int(lm[1])) )
             landmarks = processed_landmarks
        
        # Extract feature vector for face recognition/comparison
        feature_vector_np = self.session.face_feature_extract(image, face)
        feature_vector = (
            feature_vector_np.tolist() if feature_vector_np is not None else None
        )
        if feature_vector is None:
            logger.warning(
                f"Feature vector could not be extracted for face {idx}."
            )
        
        # Map demographic attribute indices to human-readable labels
        gender = GENDER_TAGS[ext.gender] if 0 <= ext.gender < len(GENDER_TAGS) else None
        age_bracket = (
            AGE_BRACKET_TAGS[ext.age_bracket]
            if 0 <= ext.age_bracket < len(AGE_BRACKET_TAGS)
            else None
        )
        race = RACE_TAGS[ext.race] if 0 <= ext.race < len(RACE_TAGS) else None

        # Extract bounding box coordinates
        loc = face.location
        if len(loc) >= 4:
            bbox: Tuple[int, int, int, int] = (int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3]))
        else:
            logger.error(f"Face location has unexpected length: {len(loc)}. Using default bbox.")
            bbox = (0, 0, 0, 0) 

        return FaceDetectionResult(
            bounding_box=bbox,
            confidence=face.detection_confidence,
            landmarks=landmarks,
            feature=feature_vector,
            roll=face.roll,
            yaw=face.yaw,
            pitch=face.pitch,
            quality=ext.quality_confidence,
            mask_confidence=ext.mask_confidence,
            liveness_confidence=ext.rgb_liveness_confidence,
            gender=gender,
            age_bracket=age_bracket,
            race=race,
            )

    async def detect_and_extract(self, image_input: ImageInput) -> DetectionResponse:
        """Detect faces and extract attributes from an image.
        
        This is the main method for face detection that:
        1. Gets the image from the provided input (base64 or URL)
        2. Detects faces in the image
        3. Runs the analysis pipeline to extract face attributes
        4. Formats the results into structured response objects
        
        Args:
            image_input: The ImageInput object containing the image source.
            
        Returns:
            A DetectionResponse object containing all detected faces and their attributes.
            
        Raises:
            ValueError: If image input is invalid.
            HTTPException: If there's an error retrieving the image.
            RuntimeError: If there's an internal processing error.
        """
        logger.info("Received detection request in deployment.")
        try:
            image = await self._get_image_from_input(
                image_input.image_base64, image_input.image_url
            )
        except (ValueError, HTTPException) as e:
            logger.error(f"Failed to get image for detection: {e}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error getting image for detection: {e}")
            raise RuntimeError("Internal server error retrieving image.") from e

        try:
            # Detect faces in the image
            faces = self.session.face_detection(image)
            logger.info(f"Detected {len(faces)} potential faces.")
            if not faces:
                return DetectionResponse(faces=[])
                
            # Run face attribute extraction pipeline
            select_exec_func = (
                isf.HF_ENABLE_FACE_RECOGNITION
                | isf.HF_ENABLE_QUALITY
                | isf.HF_ENABLE_MASK_DETECT
                | isf.HF_ENABLE_LIVENESS
                | isf.HF_ENABLE_INTERACTION
                | isf.HF_ENABLE_FACE_ATTRIBUTE
            )
            extends = self.session.face_pipeline(image, faces, select_exec_func)
            logger.info(f"Ran face pipeline for {len(extends)} faces.")
            
            # Format results for each detected face
            results: List[FaceDetectionResult] = [
                self._format_detection_result(image, face, ext, idx)
                for idx, (face, ext) in enumerate(zip(faces, extends))
            ]
            logger.info(f"Processed {len(results)} faces successfully.")
            return DetectionResponse(faces=results)
        except Exception as e:
            logger.exception(
                f"Internal server error during face detection/pipeline: {e}"
            )
            raise RuntimeError("Internal server error during face processing.") from e



    @app.post("/v1/models/insightface:detect", response_model=DetectionResponse,
              summary="Detect Faces and Extract Attributes",
              tags=["Face Analysis"])
    async def handle_detect(self, image_input: ImageInput):
        """Process a single image for face detection and attribute extraction.
        
        This endpoint accepts an image (via base64 or URL) and returns all detected
        faces with their attributes including:
        - Face bounding box and landmarks
        - Face pose (roll, yaw, pitch)
        - Face quality and mask detection scores
        - Demographic attributes (gender, age, race)
        - Feature vector for face recognition/comparison
        
        Args:
            image_input: The ImageInput object containing the image source.
            
        Returns:
            A DetectionResponse object containing all detected faces and attributes.
            
        Raises:
            HTTPException: On error with appropriate status code and error details.
        """
        logger.info("Received detection request in deployment route.")
        try:
            return await self._detect_and_extract_logic(image_input)
        except (ValueError, HTTPException) as e:
             detail = e.detail if isinstance(e, HTTPException) else str(e)
             status_code = e.status_code if isinstance(e, HTTPException) else 400
             logger.error(f"Error in detect handler: {detail} (Status: {status_code})")
             raise HTTPException(status_code=status_code, detail=detail)
        except Exception as e:
            logger.exception(f"Unexpected error in detect handler: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during detection.")

    @app.post("/v1/models/insightface:compare", response_model=ComparisonResponse,
              summary="Compare Faces in Two Images",
              tags=["Face Analysis"])
    async def handle_compare(self, images: ImageComparisonInput):
        """Compare the primary faces detected in two images.
        
        This endpoint accepts two images (each via base64 or URL) and returns
        a similarity score between the primary (first detected) face in each image.
        The similarity score is a value between 0 and 1, where higher values
        indicate greater similarity between the faces.
        
        Args:
            images: The ImageComparisonInput object containing both image sources.
            
        Returns:
            A ComparisonResponse object with the similarity score.
            
        Raises:
            HTTPException: On error with appropriate status code and error details.
        """
        logger.info("Received image comparison request in deployment route.")
        try:
            return await self._compare_images_logic(images)
        except (ValueError, HTTPException) as e:
             detail = e.detail if isinstance(e, HTTPException) else str(e)
             status_code = e.status_code if isinstance(e, HTTPException) else 400
             logger.error(f"Error in compare handler: {detail} (Status: {status_code})")
             raise HTTPException(status_code=status_code, detail=detail)
        except Exception as e:
            logger.exception(f"Error in compare handler: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during comparison.")



    async def _detect_and_extract_logic(self, image_input: ImageInput) -> DetectionResponse:
        """Internal implementation of face detection and attribute extraction.
        
        This method handles the core logic for face detection and attribute extraction,
        separate from the API endpoint handler to allow for better error handling
        and code organization.
        
        Args:
            image_input: The ImageInput object containing the image source.
            
        Returns:
            A DetectionResponse object containing all detected faces and attributes.
            
        Raises:
            ValueError: If image input is invalid.
            HTTPException: If there's an error retrieving the image.
            RuntimeError: If there's an internal processing error.
        """
        logger.info("Executing internal detection logic.")
        try:
            image = await self._get_image_from_input(
                image_input.image_base64, image_input.image_url
            )
        except (ValueError, HTTPException) as e:
            logger.error(f"Failed to get image for detection logic: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error getting image for detection logic: {e}")
            raise RuntimeError("Internal server error retrieving image.") from e

        try:
            faces = self.session.face_detection(image)
            logger.info(f"Detected {len(faces)} potential faces.")
            if not faces:
                return DetectionResponse(faces=[])
            select_exec_func = (
                isf.HF_ENABLE_FACE_RECOGNITION
                | isf.HF_ENABLE_QUALITY
                | isf.HF_ENABLE_MASK_DETECT
                | isf.HF_ENABLE_LIVENESS
                | isf.HF_ENABLE_INTERACTION
                | isf.HF_ENABLE_FACE_ATTRIBUTE
            )
            extends = self.session.face_pipeline(image, faces, select_exec_func)
            logger.info(f"Ran face pipeline for {len(extends)} faces.")
            results: List[FaceDetectionResult] = [
                self._format_detection_result(image, face, ext, idx)
                for idx, (face, ext) in enumerate(zip(faces, extends))
            ]
            logger.info(f"Processed {len(results)} faces successfully.")
            return DetectionResponse(faces=results)
        except Exception as e:
            logger.exception(
                f"Internal server error during face detection/pipeline logic: {e}"
            )
            raise RuntimeError("Internal server error during face processing.") from e


    async def _get_primary_feature(
        self, image_base64: Optional[str], image_url: Optional[str]
    ) -> Optional[np.ndarray]:
        """Extract the primary face feature vector from an image.
        
        This method:
        1. Gets the image from the provided input (base64 or URL)
        2. Detects faces in the image
        3. Extracts the feature vector from the first/primary detected face
        
        This is used primarily for face comparison operations.
        
        Args:
            image_base64: Optional base64 encoded image string.
            image_url: Optional URL to an image (http, https, or s3).
            
        Returns:
            A numpy array containing the face feature vector, or None if 
            no face is detected or feature extraction fails.
            
        Raises:
            ValueError: If image input is invalid.
            HTTPException: If there's an error retrieving the image.
            RuntimeError: If there's an internal processing error.
        """
        try:
            image = await self._get_image_from_input(image_base64, image_url)
        except (ValueError, HTTPException) as e:
            logger.error(f"Failed to get image for feature extraction: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error getting image for feature extraction: {e}")
            raise RuntimeError(f"Unexpected error retrieving image: {e}") from e

        try:
            faces = self.session.face_detection(image)
            if not faces:
                logger.warning("No faces detected in image for feature extraction.")
                return None
            logger.info(
                f"Detected {len(faces)} faces, extracting feature from the first one."
            )
            primary_face = faces[0]
            feature_vector_np = self.session.face_feature_extract(image, primary_face)
            if feature_vector_np is None:
                logger.warning(
                    "Feature vector could not be extracted."
                )
                return None
            logger.info("Primary feature extracted successfully.")
            return feature_vector_np
        except Exception as e:
            logger.exception(
                f"Internal server error during primary feature extraction: {e}"
            )

            raise RuntimeError(f"Failed to extract primary feature: {e}") from e


    async def compare_images(self, images: ImageComparisonInput) -> ComparisonResponse:
        """Compare the primary faces detected in two images.
        
        This method extracts feature vectors from the primary (first detected) face
        in each of the two provided images, then computes a similarity score between them.
        
        Args:
            images: The ImageComparisonInput object containing both image sources.
            
        Returns:
            A ComparisonResponse object with the similarity score.
            
        Raises:
            ValueError: If images are invalid or faces cannot be detected.
            HTTPException: If there's an error retrieving the images.
            RuntimeError: If there's an internal processing error.
        """
        logger.info("Received image comparison request in deployment.")
        feature1_np: Optional[np.ndarray] = None
        feature2_np: Optional[np.ndarray] = None

        try:
            feature1_np = await self._get_primary_feature(
                images.image1_base64, images.image1_url
            )

        except HTTPException as e:
             raise e
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to get feature from image 1: {e}")

            raise ValueError(f"Could not process image 1: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error processing image 1: {e}")
            raise RuntimeError("Internal server error processing image 1.") from e


        try:
            feature2_np = await self._get_primary_feature(
                images.image2_base64, images.image2_url
            )

        except HTTPException as e:
             raise e
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to get feature from image 2: {e}")

            raise ValueError(f"Could not process image 2: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error processing image 2: {e}")
            raise RuntimeError("Internal server error processing image 2.") from e


        if feature1_np is None:
             logger.error("Feature extraction failed for image 1 (no face detected or error).")

             raise ValueError("Could not extract feature from image 1 (e.g., no face found).")
        if feature2_np is None:
             logger.error("Feature extraction failed for image 2 (no face detected or error).")

             raise ValueError("Could not extract feature from image 2 (e.g., no face found).")


        try:
            similarity = isf.feature_comparison(feature1_np, feature2_np)
            logger.info(f"Comparison completed. Similarity: {similarity}")
            return ComparisonResponse(similarity=similarity)
        except Exception as e:
            logger.exception(f"Internal server error during feature comparison: {e}")

            raise RuntimeError("Internal server error during feature comparison.") from e


    @app.get("/-/healthz", tags=["Health"])
    async def health_check(self):
        """Basic health check."""
        if self.session is None:
             raise HTTPException(status_code=503, detail="InspireFace session not initialized.")

        logger.debug("Health check endpoint called successfully.")
        return {"status": "ok"}


    async def _compare_images_logic(self, images: ImageComparisonInput) -> ComparisonResponse:
        """Internal implementation of face comparison logic.
        
        This method handles the core logic for face comparison, separate from
        the API endpoint handler to allow for better error handling and code
        organization.
        
        Args:
            images: The ImageComparisonInput object containing both image sources.
            
        Returns:
            A ComparisonResponse object with the similarity score.
            
        Raises:
            ValueError: If images are invalid or faces cannot be detected.
            HTTPException: If there's an error retrieving the images.
            RuntimeError: If there's an internal processing error.
        """
        logger.info("Executing internal image comparison logic.")
        feature1_np: Optional[np.ndarray] = None
        feature2_np: Optional[np.ndarray] = None

        try:
            feature1_np = await self._get_primary_feature(
                images.image1_base64, images.image1_url
            )
        except HTTPException as e:
             raise e
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to get feature from image 1 logic: {e}")
            raise ValueError(f"Could not process image 1: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error processing image 1 logic: {e}")
            raise RuntimeError("Internal server error processing image 1.") from e


        try:
            feature2_np = await self._get_primary_feature(
                images.image2_base64, images.image2_url
            )
        except HTTPException as e:
             raise e
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to get feature from image 2 logic: {e}")
            raise ValueError(f"Could not process image 2: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error processing image 2 logic: {e}")
            raise RuntimeError("Internal server error processing image 2.") from e


        if feature1_np is None:
             logger.error("Feature extraction failed for image 1 logic.")
             raise ValueError("Could not extract feature from image 1 (e.g., no face found).")
        if feature2_np is None:
             logger.error("Feature extraction failed for image 2 logic.")
             raise ValueError("Could not extract feature from image 2 (e.g., no face found).")


        try:
            similarity = isf.feature_comparison(feature1_np, feature2_np)
            logger.info(f"Comparison logic completed. Similarity: {similarity}")
            return ComparisonResponse(similarity=similarity)
        except Exception as e:
            logger.exception(f"Internal server error during feature comparison logic: {e}")
            raise RuntimeError("Internal server error during feature comparison.") from e



app_instance = InsightFaceDeployment.bind()
