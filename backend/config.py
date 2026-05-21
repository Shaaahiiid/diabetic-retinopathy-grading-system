import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    # Centralize model paths, env vars, and labels in one place.
    def __init__(self):
        self.model_path = BASE_DIR / os.getenv("MODEL_PATH", "best_model.pth")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.allowed_types = {"image/png", "image/jpeg"}
        self.grade_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


settings = Settings()
