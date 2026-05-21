import base64
from io import BytesIO
import cv2
import numpy as np
import torch
from PIL import Image
from backend.model import TRANSFORM


# Build Grad-CAM with plain PyTorch hooks and return it as base64 PNG.
def make_gradcam(model, device, image, class_id):
    activations, gradients = [], []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    hook1 = model.features[-1].register_forward_hook(forward_hook)
    hook2 = model.features[-1].register_full_backward_hook(backward_hook)
    output = model(TRANSFORM(image).unsqueeze(0).to(device))
    model.zero_grad()
    output[0, class_id].backward()
    hook1.remove()
    hook2.remove()
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=1)).squeeze()
    cam = cv2.resize((cam / (cam.max() + 1e-8)).cpu().numpy(), (224, 224))
    rgb = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = Image.fromarray(cv2.addWeighted(rgb, 0.6, heatmap, 0.4, 0))
    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
