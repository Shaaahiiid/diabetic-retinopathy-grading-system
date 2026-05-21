import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Shared image transform used by training and inference.
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class RetinaDataset(Dataset):
    # Reads image names and labels from a CSV file.
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = get_transforms()
        self.image_col = "id_code" if "id_code" in self.data.columns else self.data.columns[0]
        self.label_col = "diagnosis" if "diagnosis" in self.data.columns else self.data.columns[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = str(row[self.image_col])
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        label = int(row[self.label_col])
        return self.transform(image), label
