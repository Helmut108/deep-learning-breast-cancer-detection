import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import pydicom
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from cbismodules.transforms import ResizeAndPad, create_transforms

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




def prepare_datasets(csv_path_train, ddsm_path, batch_size=8, sample_size=None, use_augmentation=False, image_size = 512):
    df_train = create_dataframe_from_csv(csv_path_train)
    print(f"Length of training dataframe: {len(df_train)}")
    # print(df_train["Label"].value_counts())
    # print(df_train["Label"].value_counts(normalize=True))

    if sample_size is not None:
        nbr_per_class = sample_size // 2
        # print(f"Sampling {nbr_per_class} samples per class for a total of {nbr_per_class * 2} samples.")    
    
        df_train = (
            df_train
            .groupby("Label", group_keys=False)
            .sample(n=nbr_per_class, random_state=42)
            .reset_index(drop=True)
        )

    df_train, df_val = train_test_split(
        df_train,
        test_size=0.25,
        random_state=42,
        stratify=df_train["Label"]
    )

    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # print("\ntrain labels:")
    # print(df_train["Label"].value_counts())
    # print("\nValidation labels:")
    # print(df_val["Label"].value_counts())

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
    #     stratify=df_train['Label']
    # )

    

    # transform = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     ResizeAndPad(512),
    #     ])  



    train_transform, val_transform = create_transforms(use_augmentation, image_size)

    train_dataset = CBISDataset(df_train, ddsm_path, transform=train_transform)
    val_dataset = CBISDataset(df_val, ddsm_path, transform=val_transform)


    # print("Training transform:")
    # print(train_transform)
    # print("Length of training dataframe: ", len(df_train))
    # print()

    # print("Validation transform:")
    # print(val_transform)
    # print("Length of validation dataframe: ", len(df_val))

    image, label = train_dataset[0]
    print("\nSample train image data train_dataset[0]:")
    print("Shape: ",image.shape)
    print("Type: ", image.dtype)
    print("Min/Max: ", image.min().item(), image.max().item())
    print("Label: ", label)

    print()
    print("Get 3 images from the training dataset to check if they are the same or different:")
    image1, label1 = train_dataset[0]
    image2, label2 = train_dataset[0]
    image3, label3 = train_dataset[0]

    print(torch.equal(image1, image2))
    print(torch.equal(image2, image3))

    print("3 Labels: ", label1, label2, label3)

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # for ax, image in zip(axes, [image1, image2, image3]):
    #     ax.imshow(image.squeeze(), cmap="gray")
    #     ax.axis("off")

    # plt.show()
    # sys.exit("Stopping here for now")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    print()
    # print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    # print(f"Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    # print(f"Training DataLoader batch size: {train_loader.batch_size}")
    # print(f"Validation DataLoader batch size: {val_loader.batch_size}")
    # print()      
    # print(f"Training dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(val_dataset)}")
    # print(f"Training DataLoader size: {len(train_loader)}")
    # print(f"Validation DataLoader size: {len(val_loader)}")
    # print(f"Training DataLoader batch size: {train_loader.batch_size}")
    # print(f"Validation DataLoader batch size: {val_loader.batch_size}")
    # print(f"Training DataLoader number of batches: {len(train_loader)}")
    # print(f"Validation DataLoader number of batches: {len(val_loader)}")
    # print(f"Training DataLoader number of samples: {len(train_loader.dataset)}")
    # print(f"Validation DataLoader number of samples: {len(val_loader.dataset)}")
    # print()
    # print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    # print(f"Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    # print(type(train_dataset))
    # print(train_dataset["Label"].value_counts())
    # print(df_val["Label"].value_counts())
    # print(f"Training dataset label counts:\n{train_dataset['Labels'].value_counts()}")
    # print(f"Validation dataset label counts:\n{val_dataset['Labels'].value_counts()}")

    # sys.exit("Stopping here for now")``
    # return df_train, train_dataset, train_loader, df_val, val_dataset, val_loader  
    return train_loader, val_loader   
 


