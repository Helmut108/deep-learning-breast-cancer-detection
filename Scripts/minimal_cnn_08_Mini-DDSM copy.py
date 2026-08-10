import os
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# To build paths inside a project like this: BASE_DIR / "subdir".

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = BASE_DIR / "miniddsm_image_labels.csv"
LABEL_TO_IDX = {"Benign": 0, "Cancer": 1}


def get_device():
    """
    Return the best available device:
    CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    return device


class MiniDDSMDataset(Dataset):
    def __init__(self, dataframe, label_map, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = DATA_DIR / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        label = self.label_map[row["label"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def prepare_data():
    csv_df = pd.read_csv(CSV_PATH)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = MiniDDSMDataset(csv_df, LABEL_TO_IDX, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    return loader, LABEL_TO_IDX

