# Hugging Face Spaces Deployment

This project can be deployed as a single Streamlit app on Hugging Face Spaces.

## Free Deployment Path

Use a Hugging Face Space with:

- SDK: `Streamlit`
- Hardware: `CPU Basic`
- App file: `app.py`

## Required Files

The Space needs these files in the repository:

- `app.py`
- `requirements.txt`
- `model.py`
- `dataset.py`
- `gradcam.py`
- `.streamlit/config.toml`
- `best_model.pth`

## Secrets

Set these in the Space settings:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
MODEL_PATH=best_model.pth
```

Do not set `BACKEND_URL` on Hugging Face Spaces. If `BACKEND_URL` is empty, the Streamlit app runs inference directly inside the Space.

## Model File

`best_model.pth` is ignored in the GitHub repo. For Hugging Face Spaces, upload it directly in the Space UI or use Git LFS.

## Local Test

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
