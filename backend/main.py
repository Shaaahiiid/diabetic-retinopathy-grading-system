from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError
from backend.config import settings
from backend.gradcam import make_gradcam
from backend.model import load_model, predict, read_image
from backend.report import build_client, generate_report
from backend.schemas import HealthResponse, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load expensive resources once when the API starts.
    app.state.model = app.state.device = None
    app.state.client = build_client()
    try:
        app.state.model, app.state.device = load_model()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(title="DR Grading Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
async def health():
    # Let the frontend know whether the checkpoint loaded correctly.
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # Validate the upload before running inference.
    if app.state.model is None:
        raise HTTPException(503, "Model checkpoint not found.")
    if app.state.client is None:
        raise HTTPException(503, "GEMINI_API_KEY is not configured.")
    if file.content_type not in settings.allowed_types:
        raise HTTPException(400, "Upload a PNG or JPEG retinal image.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    try:
        image = read_image(data)
    except UnidentifiedImageError as exc:
        raise HTTPException(400, "Invalid image file.") from exc
    grade, grade_name, confidence = predict(app.state.model, app.state.device, image)
    try:
        report = generate_report(app.state.client, grade, grade_name, confidence)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"grade": grade, "grade_name": grade_name, "confidence": confidence,
            "gradcam_image": make_gradcam(app.state.model, app.state.device, image, grade),
            "report": report}
