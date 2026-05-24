import base64
import json
import mimetypes
import os
import tempfile
import urllib.error
import urllib.request
import uuid
from io import BytesIO
from google import genai
from PIL import Image
import streamlit as st
import torch
from dataset import get_transforms
from gradcam import make_gradcam
from model import get_model

BACKEND_URL = os.getenv("BACKEND_URL")
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# Send the uploaded image to the FastAPI backend as multipart form-data.
def predict_via_api(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    content_type = mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{uploaded_file.name}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n").encode()
    body = head + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{BACKEND_URL}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode() or "Backend returned an error.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Backend is not running at http://127.0.0.1:8000.") from exc


@st.cache_resource
def load_model():
    # Hugging Face Spaces runs this once, then reuses the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def generate_report(grade, grade_name, confidence):
    # Gemini key should be configured as a Hugging Face Space secret.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API key is not configured."
    prompt = (
        "Write a concise clinical support note for a diabetic retinopathy AI screen. "
        f"Predicted grade: {grade} ({grade_name}). Confidence: {confidence:.2%}. "
        "Include likely severity, one caution, and one follow-up recommendation. "
        "Do not claim a confirmed diagnosis."
    )
    response = genai.Client(api_key=api_key).models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return (response.text or "No report generated.").strip()


def predict_locally(image):
    model, device = load_model()
    input_tensor = get_transforms()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
    grade = int(probs.argmax().item())
    names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        heatmap = make_gradcam(model, temp_file.name, device)
    buffer = BytesIO()
    heatmap.save(buffer, format="PNG")
    return {
        "grade": grade,
        "grade_name": names[grade],
        "confidence": float(probs[grade].item()),
        "gradcam_image": base64.b64encode(buffer.getvalue()).decode(),
        "report": generate_report(grade, names[grade], float(probs[grade].item())),
    }


st.set_page_config(page_title="DR Grading System", layout="centered")
st.markdown("""
<style>
[data-testid="stFileUploader"] section {
    background: #161922;
    border: 1px solid #313646;
    border-radius: 14px;
    padding: 0.4rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent;
    border: 1.5px dashed #4a5164;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("Diabetic Retinopathy Grading System")
uploaded_file = st.file_uploader("Upload a retinal image", ["png", "jpg", "jpeg"])

if uploaded_file:
    # Show the user image first, then ask the backend for results.
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    try:
        result = predict_via_api(uploaded_file) if BACKEND_URL else predict_locally(image)
    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        heatmap = Image.open(BytesIO(base64.b64decode(result["gradcam_image"])))
        st.subheader(f"Predicted Grade: {result['grade']} - {result['grade_name']}")
        st.write(f"Confidence: {result['confidence']:.2%}")
        st.image(heatmap, caption="Grad-CAM heatmap", use_container_width=True)
        st.subheader("Clinical Report")
        st.write(result["report"])
