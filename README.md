# Diabetic Retinopathy Grading System

A computer vision system for grading retinal fundus images into diabetic retinopathy severity levels. The project uses an EfficientNet-B3 PyTorch model for image classification, Grad-CAM for visual explainability, a FastAPI backend for inference, and a Streamlit interface for image upload and result display.

## Features

- Retinal image classification into 5 diabetic retinopathy grades
- EfficientNet-B3 inference with PyTorch
- Grad-CAM heatmap generation using pure PyTorch hooks
- FastAPI backend with image upload prediction endpoint
- Streamlit frontend for interactive image assessment
- Gemini API integration for clinical report generation
- Dark Streamlit UI configuration

## Severity Classes

| Grade | Class |
| --- | --- |
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Tech Stack

- Python
- PyTorch, torchvision
- FastAPI, uvicorn
- Streamlit
- OpenCV, Pillow
- Pydantic
- Google Gemini API

## Project Structure

```text
DiabeticRetina/
├── app.py
├── dataset.py
├── model.py
├── train.py
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
└── .streamlit/
    └── config.toml
```

## Dataset

The model was trained on the APTOS 2019 retinal fundus image dataset from Kaggle. The dataset contains retinal images labeled from grade `0` to grade `4`.

## Model

- Architecture: EfficientNet-B3
- Output classes: 5
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Validation metric: accuracy
- Best validation accuracy: 83.6%

The trained checkpoint should be placed in the project root as:

```text
best_model.pth
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
MODEL_PATH=best_model.pth
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m pip install streamlit pandas
```

## Run Backend

```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Run Streamlit App

Start the backend first, then run:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

## API Reference

### GET `/health`

Returns backend status and model loading state.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST `/predict`

Accepts a retinal image as multipart form data.

Request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample_retina.png"
```

Response:

```json
{
  "grade": 2,
  "grade_name": "Moderate",
  "confidence": 0.87,
  "gradcam_image": "<base64_png>",
  "report": "Clinical report text"
}
```

## Training

Training can be run locally if the dataset is available:

```bash
python train.py \
  --csv_file train.csv \
  --image_dir train_images \
  --save_path best_model.pth
```

For Kaggle-based training, use:

```text
kaggle_retina_notebook.ipynb
```

## Notes

- `best_model.pth` is not included in the repository.
- `.env` is ignored and should not be committed.
- The clinical report is generated from model output and should be reviewed by qualified professionals before use in any clinical workflow.
