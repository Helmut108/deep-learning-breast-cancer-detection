import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import pydicom
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2

from cbismodules.transforms import ResizeAndPad

def create_dataframe_from_csv(csv_path_train):

    df = pd.read_csv(csv_path_train)

    df["Label"] = df["pathology"].map({
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    })

    # df = pd.read_csv("./data/mass_case_description_train_set.csv")
    df["folder"] = df["image file path"].str.split("/", n=1).str[0]
    df["study"] = df["image file path"].str.split("/").str[1].str[-5:]
    df["series"] = df["image file path"].str.split("/").str[2].str[-5:]

    return df



class CBISDataset(Dataset):
    def __init__(self, dataframe, ddsm_path, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.ddsm_path = Path(ddsm_path)    
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        series_path = self.ddsm_path / row["folder"] / row["study"] / row["series"]
        dicom_path = next(series_path.glob("*.dcm"))
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array
        label = int(row["Label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label



def prepare_datasets(csv_path_train, ddsm_path, batch_size=8, sample_size=None):
    df_train = create_dataframe_from_csv(csv_path_train)
    print(len(df_train))

    if sample_size is not None:
        nbr_per_class = sample_size // 2
    
        df_train = (
            df_train
            .groupby("Label", group_keys=False)
            .sample(n=nbr_per_class, random_state=42)
            .reset_index(drop=True)
        )

    # print(f"Training dataset size: {len(df_train)}")

    # sys.exit("Stopping here for now")
    # df_val = df_train.sample(frac=0.25, random_state=42)
    # df_train = df_train.drop(df_val.index).reset_index(drop=True)
    # print(f"Training dataset size: {len(df_train)}")
    # print(f"Validation dataset size: {len(df_val)}")
    
    # df_train, df_val = train_test_split(
    #     df_train,
    #     test_size=0.25,
    #     random_state=42,
    #     stratify=df['Label']
    # )

    

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeAndPad(512),
        ])  

    train_dataset = CBISDataset(df_train, ddsm_path, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return df_train, train_dataset, train_loader


