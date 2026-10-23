from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractEmbeddingRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64 encoded JPEG/PNG face image. Accepts raw base64 string or data URI format (e.g. data:image/jpeg;base64,...)."
    )
    enforce_liveness: Optional[bool] = Field(
        True,
        description="Enable Fasnet anti-spoofing deep learning check (default True for one-time registration)."
    )
    strict_single_face: Optional[bool] = Field(
        True,
        description="Require exactly one face in the frame. Rejects if multiple faces or zero faces are detected."
    )
    check_quality: Optional[bool] = Field(
        True,
        description="Check minimum face dimensions (>=90px), centering within 40% margin, and lighting quality."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP...",
                "enforce_liveness": True,
                "strict_single_face": True,
                "check_quality": True
            }
        }
    }

class ExtractEmbeddingResponse(BaseModel):
    status: str = Field("success", description="Status string: 'success'")
    embedding: List[float] = Field(..., description="512-dimensional normalized biometric embedding array")
    face_confidence: float = Field(..., description="Face detection confidence score (0.0 - 1.0)")
    is_real: Optional[bool] = Field(True, description="Liveness check result (True = genuine live human, False = spoof attempt)")
    antispoof_score: Optional[float] = Field(None, description="Anti-spoofing confidence score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "embedding": [0.0124, -0.0482, 0.0911, -0.0156, 0.0381],
                "face_confidence": 0.985,
                "is_real": True,
                "antispoof_score": 0.962
            }
        }
    }

class VerifyEmbeddingRequest(BaseModel):
    live_image_base64: str = Field(
        ...,
        description="Live clock-in camera frame as base64 string (raw string or data URI)"
    )
    stored_embedding: List[float] = Field(
        ...,
        description="512-dimensional reference embedding vector loaded from employee_face_biometrics table"
    )
    threshold: Optional[float] = Field(
        0.28,
        description="Cosine distance threshold for match. Calibrated at 0.28 (match if distance <= 0.28)."
    )
    enforce_liveness: Optional[bool] = Field(
        True,
        description="Enable anti-spoofing liveness check during verification."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "live_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQE...",
                "stored_embedding": [0.0124, -0.0482, 0.0911, -0.0156, 0.0381],
                "threshold": 0.28,
                "enforce_liveness": True
            }
        }
    }

class VerifyEmbeddingResponse(BaseModel):
    matched: bool = Field(..., description="True if cosine distance <= threshold and liveness is verified")
    distance: float = Field(..., description="Calculated cosine distance between live and stored vectors (lower is closer)")
    threshold: float = Field(..., description="Cosine threshold applied for comparison (default 0.28)")
    similarity_metric: str = Field("cosine", description="Mathematical similarity metric: 'cosine'")
    is_liveness_verified: Optional[bool] = Field(True, description="Liveness verification status")

    model_config = {
        "json_schema_extra": {
            "example": {
                "matched": True,
                "distance": 0.1624,
                "threshold": 0.28,
                "similarity_metric": "cosine",
                "is_liveness_verified": True
            }
        }
    }

class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service health status")
    service: str = Field("propjini-face-service", description="Service name")
    version: str = Field("2.1.0", description="Service API version")
    default_model: str = Field("Facenet512", description="Active facial recognition deep learning model configured on server")
    default_detector: str = Field("opencv", description="Active face detector backend configured on server")
    default_threshold: float = Field(0.28, description="Calibrated cosine distance threshold")
    default_enforce_liveness: bool = Field(True, description="Active liveness enforcement default")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "service": "propjini-face-service",
                "version": "2.1.0",
                "default_model": "Facenet512",
                "default_detector": "opencv",
                "default_threshold": 0.28,
                "default_enforce_liveness": True
            }
        }
    }

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error explanation or validation issue")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "No face detected in camera. Please ensure your face is clearly visible inside the oval."
            }
        }
    }
