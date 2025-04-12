"""Data models for the InsightFace API.

This module defines the Pydantic models used for request validation,
data serialization, and API responses in the InsightFace Ray Serve API.

These models include:
- Input models for image uploads (via base64 or URL)
- Face detection result models
- Comparison response models

"""

import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError, root_validator, validator


class ImageInput(BaseModel):
    """Model for accepting image input via base64 string or URL.
    
    Only one of image_base64 or image_url must be provided. The model
    validates proper URL schemes and base64 formatting.
    """
    
    image_base64: Optional[str] = Field(
        None, description="Base64 encoded image string. Provide either this or image_url."
    )
    image_url: Optional[str] = Field(
        None, description="URL (http, https, s3) to the image. Provide either this or image_base64."
    )
    @validator('image_url')
    def check_url_scheme(cls, v):
        """Validate that the image URL uses an accepted scheme.
        
        Args:
            v: The URL to validate.
            
        Returns:
            The validated URL.
            
        Raises:
            ValueError: If URL scheme is not http, https, or s3.
        """
        if v is not None:
            parsed = urlparse(v)
            if parsed.scheme not in ['http', 'https', 's3']:
                raise ValueError('URL scheme must be http, https, or s3')
        return v
    @validator('image_base64')
    def check_base64_format(cls, v):
        """Validate that the image base64 string has correct format.
        
        Args:
            v: The base64 string to validate.
            
        Returns:
            The validated base64 string.
            
        Raises:
            ValueError: If string is not properly base64 encoded.
        """
        if v is not None:
            if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', v):
                raise ValueError('Invalid Base64 format')
        return v
    @root_validator(pre=False, skip_on_failure=True)
    def check_image_source(cls, values):
        """Validate that exactly one image source is provided.
        
        Args:
            values: The field values to validate.
            
        Returns:
            The validated field values.
            
        Raises:
            ValueError: If neither or both image sources are provided.
        """
        base64_provided = values.get('image_base64') is not None and values.get('image_base64') != ""
        url_provided = values.get('image_url') is not None and values.get('image_url') != ""
        if not (base64_provided ^ url_provided):
            raise ValueError('Provide exactly one of image_base64 or image_url')
        return values
class FaceDetectionResult(BaseModel):
    """Result model for a single detected face with attributes.
    
    Contains geometric information (bounding box, landmarks, pose),
    biometric features, and demographic attributes.
    """
    
    bounding_box: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box coordinates (x1, y1, x2, y2)."
    )
    confidence: float = Field(..., description="Detection confidence score.")
    landmarks: Optional[List[Tuple[int, int]]] = Field(
        None, description="List of facial landmark coordinates (x, y)."
    )
    feature: Optional[List[float]] = Field(
        None, description="512-dimensional face feature vector (embedding)."
    )
    roll: Optional[float] = Field(None, description="Roll angle of the face.")
    yaw: Optional[float] = Field(None, description="Yaw angle of the face.")
    pitch: Optional[float] = Field(None, description="Pitch angle of the face.")
    quality: Optional[float] = Field(None, description="Face quality score.")
    mask_confidence: Optional[float] = Field(None, description="Mask confidence score.")
    liveness_confidence: Optional[float] = Field(
        None, description="RGB liveness confidence score."
    )
    gender: Optional[str] = Field(
        None, description="Predicted gender ('Female' or 'Male')."
    )
    age_bracket: Optional[str] = Field(None, description="Predicted age bracket.")
    race: Optional[str] = Field(None, description="Predicted race.")
class DetectionResponse(BaseModel):
    """Response model for face detection API endpoint.
    
    Contains a list of detected faces with their attributes.
    """
    
    faces: List[FaceDetectionResult] = Field(
        ..., description="List of detected faces and their attributes."
    )
class ImageComparisonInput(BaseModel):
    """Input model for comparing faces in two images.
    
    Each image can be provided via base64 string or URL.
    For each image, exactly one source type must be provided.
    """
    
    image1_base64: Optional[str] = Field(
        None, description="Base64 encoded string of the first image. Provide either this or image1_url."
    )
    image1_url: Optional[str] = Field(
        None, description="URL (http, https, s3) to the first image. Provide either this or image1_base64."
    )
    image2_base64: Optional[str] = Field(
        None, description="Base64 encoded string of the second image. Provide either this or image2_url."
    )
    image2_url: Optional[str] = Field(
        None, description="URL (http, https, s3) to the second image. Provide either this or image2_base64."
    )
    @validator('image1_url', 'image2_url')
    def check_url_scheme(cls, v):
        """Validate that the image URLs use accepted schemes.
        
        Args:
            v: The URL to validate.
            
        Returns:
            The validated URL.
            
        Raises:
            ValueError: If URL scheme is not http, https, or s3.
        """
        if v is not None:
            parsed = urlparse(v)
            if parsed.scheme not in ['http', 'https', 's3']:
                raise ValueError('URL scheme must be http, https, or s3')
        return v
    @validator('image1_base64', 'image2_base64')
    def check_base64_format(cls, v):
        """Validate that the image base64 strings have correct format.
        
        Args:
            v: The base64 string to validate.
            
        Returns:
            The validated base64 string.
            
        Raises:
            ValueError: If string is not properly base64 encoded.
        """
        if v is not None:
            if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', v):
                raise ValueError('Invalid Base64 format')
        return v
    @root_validator(pre=False, skip_on_failure=True)
    def check_sources(cls, values):
        """Validate that exactly one source is provided for each image.
        
        Args:
            values: The field values to validate.
            
        Returns:
            The validated field values.
            
        Raises:
            ValueError: If neither or both sources are provided for either image.
        """
        base64_1_provided = values.get('image1_base64') is not None and values.get('image1_base64') != ""
        url_1_provided = values.get('image1_url') is not None and values.get('image1_url') != ""
        if not (base64_1_provided ^ url_1_provided):
            raise ValueError('Provide exactly one of image1_base64 or image1_url')
        base64_2_provided = values.get('image2_base64') is not None and values.get('image2_base64') != ""
        url_2_provided = values.get('image2_url') is not None and values.get('image2_url') != ""
        if not (base64_2_provided ^ url_2_provided):
            raise ValueError('Provide exactly one of image2_base64 or image2_url')
        return values
class ComparisonResponse(BaseModel):
    """Response model for face comparison API endpoint.
    
    Contains the similarity score between the two primary faces.
    The similarity is a cosine similarity value typically between 0 and 1,
    where higher values indicate greater similarity.
    """
    
    similarity: float = Field(
        ..., description="Cosine similarity score between the two faces."
    )
