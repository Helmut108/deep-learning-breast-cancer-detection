import sys
import torch
from pathlib import Path
from config import csv_path_train, ddsm_path
from cbismodules import utils
# from cbismodules.transforms import ResizeAndPad
# from cbismodules.dataprep import create_dataframe_from_csv
from cbismodules.dataprep import prepare_datasets
from cbismodules.model import MinimalCNN
from cbismodules.training import train_model

device = utils.get_device()

# sys.exit("Stopping here for now")

# df_train = create_dataframe_from_csv(csv_path_train)
# assert len(df_train) == 1318
# assert "Label" in df_train.columns
# assert df_train["Label"].isna().sum() == 0


df_train, train_dataset, train_loader = prepare_datasets(csv_path_train, ddsm_path, batch_size=8, sample_size=64)

model = MinimalCNN().to(device)
print(model)

images, labels = next(iter(train_loader))

images = images.to(device)

outputs = model(images)

print(f"Input batch shape: {images.shape}")
print(f"Output shape: {outputs.shape}")

# model = MinimalCNN().to(device)
# train_model(model, train_loader, val_loader,device, epochs=2)
train_model(model, train_loader, device, epochs=30)