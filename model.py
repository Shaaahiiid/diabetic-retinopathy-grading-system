import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


# Loads EfficientNet-B3 and swaps the classifier for 5 grades.
def get_model(pretrained=False):
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 5)
    return model
