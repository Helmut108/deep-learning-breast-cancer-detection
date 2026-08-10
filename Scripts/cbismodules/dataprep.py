import pandas as pd
from pathlib import Path
import torch
import pydicom
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2


from cbismodules.transforms import ResizeAndPad
# from Scripts.main import CSV_PATH_TRAIN

# ddsm_path = Path("/Volumes/Samsung_T7/Not_Sync/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/cbis_ddsm")

def create_dataframe_from_csv(CSV_PATH_TRAIN):

    df = pd.read_csv(CSV_PATH_TRAIN)
    print(f"DataFrame created from {CSV_PATH_TRAIN}. Shape: {df.shape}")

    df["Label"] = df["pathology"].map({
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    })

    # df = pd.read_csv("./data/mass_case_description_train_set.csv")
    df["folder"] = df["image file path"].str.split("/", n=1).str[0]
    df["study"] = df["image file path"].str.split("/").str[1].str[-5:]
    df["series"] = df["image file path"].str.split("/").str[2].str[-5:]

    # df.info 
    # print(df.head(2))
    # print(df["Label"].value_counts(dropna=False))
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
        # print(f"Series path: {series_path}")
        dicom_path = next(series_path.glob("*.dcm"))
        # print(f"Loading DICOM image from: {dicom_path}")
        # image = Image.open(image_path).convert("RGB")
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array
        label = int(row["Label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def prepare_datasets(CSV_PATH_TRAIN, ddsm_path, batch_size=8):
    df_train = create_dataframe_from_csv(CSV_PATH_TRAIN)

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


