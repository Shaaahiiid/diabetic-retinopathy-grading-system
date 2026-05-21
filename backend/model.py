from io import BytesIO
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3
from backend.config import settings

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Load the checkpoint once and keep the model ready for inference.
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(settings.model_path, map_location=device))
    model.eval()
    return model.to(device), device


# Convert uploaded bytes into a clean RGB image.
def read_image(data: bytes):
    return Image.open(BytesIO(data)).convert("RGB")


# Run a forward pass and return the top class details.
def predict(model, device, image):
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    grade = int(probs.argmax().item())
    return grade, settings.grade_names[grade], float(probs[grade].item())
