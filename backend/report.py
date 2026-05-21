from google import genai
from backend.config import settings


# Create the Gemini client from the API key loaded via .env.
def build_client():
    return genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None


# Ask Gemini for a short clinical-style summary from the prediction.
def generate_report(client, grade, grade_name, confidence):
    prompt = (
        "Write a concise clinical support note for a diabetic retinopathy AI screen. "
        f"Predicted grade: {grade} ({grade_name}). Confidence: {confidence:.2%}. "
        "Include likely severity, one caution, and one follow-up recommendation. "
        "Do not claim a confirmed diagnosis."
    )
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc
    return (response.text or "No report generated.").strip()
