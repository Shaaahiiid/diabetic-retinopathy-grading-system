import cv2
import torch
from PIL import Image
from dataset import get_transforms

def make_gradcam(model, image_path, device):
    activations, gradients = [], []
    
    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, __, output):
        gradients.append(output[0].detach())

    hook1 = model.features[-1].register_forward_hook(forward_hook)
    hook2 = model.features[-1].register_full_backward_hook(backward_hook)
    image = Image.open(image_path).convert("RGB")
    input_tensor = get_transforms()(image).unsqueeze(0).to(device)
    model.eval()
    output = model(input_tensor)
    class_id = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, class_id].backward()
    hook1.remove()
    hook2.remove()
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=1)).squeeze()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam.cpu().numpy(), (224, 224))
    rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (224, 224))
    heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(rgb_image, 0.6, heatmap, 0.4, 0))
