import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from io import BytesIO
from PIL import Image
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


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
        result = predict_via_api(uploaded_file)
    except RuntimeError as exc:
        st.error(str(exc))
    else:
        heatmap = Image.open(BytesIO(base64.b64decode(result["gradcam_image"])))
        st.subheader(f"Predicted Grade: {result['grade']} - {result['grade_name']}")
        st.write(f"Confidence: {result['confidence']:.2%}")
        st.image(heatmap, caption="Grad-CAM heatmap", use_container_width=True)
        st.subheader("Clinical Report")
        st.write(result["report"])
