from pydantic import BaseModel, Field


# Response returned by the prediction endpoint.
class PredictResponse(BaseModel):
    grade: int = Field(ge=0, le=4)
    grade_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    gradcam_image: str
    report: str


# Small health payload for uptime checks.
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
