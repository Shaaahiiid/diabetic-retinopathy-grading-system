# Diabetic Retinopathy Grading System

A beginner-friendly computer vision project that grades retinal fundus images into 5 diabetic retinopathy classes using PyTorch, visualizes model attention with Grad-CAM, and serves predictions through both Streamlit and FastAPI.

## Features

- EfficientNet-B3 based diabetic retinopathy grading
- 5 output classes: `0` to `4`
- Grad-CAM heatmap for visual explainability
- Streamlit frontend for quick interactive testing
- FastAPI backend with `/predict` and `/health`
- Gemini-generated clinical support report
- Kaggle notebook workflow for training on APTOS 2019

## Tech Stack

- Python
- PyTorch, torchvision
- FastAPI, uvicorn
- Streamlit
- OpenCV, Pillow
- Google Gemini API

## Project Structure

```text
DiabeticRetina/
├── app.py
├── train.py
├── dataset.py
├── model.py
├── gradcam.py
├── kaggle_retina_notebook.ipynb
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── gradcam.py
│   ├── report.py
│   ├── schemas.py
│   ├── config.py
│   └── requirements.txt
└── .streamlit/config.toml
```

## Dataset

This project was trained using the `APTOS 2019 Blindness Detection` retinal image dataset on Kaggle.

## Local Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install backend dependencies:

```bash
python -m pip install -r backend/requirements.txt
python -m pip install streamlit pandas
```

3. Add your environment variables in `.env`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
MODEL_PATH=best_model.pth
```

4. Make sure `best_model.pth` is in the project root.

## Run the Backend

```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Run the Streamlit App

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

## API Endpoints

- `GET /health`
- `POST /predict`

The prediction response includes:

- predicted grade
- grade name
- confidence
- base64 Grad-CAM image
- Gemini-generated clinical report

## Training

Model training was done in Kaggle using GPU because the APTOS dataset is large. The provided notebook:

- loads the dataset
- trains EfficientNet-B3
- saves `best_model.pth`
- runs a sample Grad-CAM prediction

## Resume Bullet

`Built a Diabetic Retinopathy Grading System using PyTorch, EfficientNet-B3, FastAPI, and Streamlit; trained on the APTOS 2019 dataset, achieved 83.6% validation accuracy, and added Grad-CAM plus Gemini-generated clinical support reports for explainable predictions.`

## Important Note

This is a portfolio and educational project, not a clinical diagnostic tool. Predictions, confidence scores, and generated reports should not be treated as medical advice.
