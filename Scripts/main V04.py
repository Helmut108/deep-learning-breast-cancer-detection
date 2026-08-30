import sys
import torch
from torchinfo import summary
from pathlib import Path
from config import csv_path_train, ddsm_path
from cbismodules import utils
# from cbismodules.transforms import ResizeAndPad
# from cbismodules.dataprep import create_dataframe_from_csv
from cbismodules.dataprep import prepare_datasets
from cbismodules.model import DDSMCNN, DDSMCNN
from cbismodules.training import train_model

device = utils.get_device()

sample_size = 512
batch_size = 8
epochs = 30
learning_rate = 0.0001
print(f"Sample size: {sample_size}, Batch size: {batch_size}, Epochs: {epochs}, Learning rate: {learning_rate}")


# sys.exit("Stopping here for now")

# df_train = create_dataframe_from_csv(csv_path_train)
# assert len(df_train) == 1318
# assert "Label" in df_train.columns
# assert df_train["Label"].isna().sum() == 0


# df_train, train_dataset, train_loader = prepare_datasets(csv_path_train, ddsm_path, batch_size=8, sample_size=64)
train_loader, val_loader = prepare_datasets(csv_path_train, ddsm_path, batch_size=batch_size, sample_size=sample_size)

model = DDSMCNN().to(device)
print(model)
print("Device:", device)
model_device = next(model.parameters()).device
print("Model parameter device:", model_device)

# total_parameters = sum(
#     p.numel() for p in model.parameters()
# )
# print(f"Total parameters: {total_parameters:,}")

summary(model, input_size=(1, 1, 512, 512), device=device)  # (batch_size, channels, height, width)
print("Device after summary:", device)
print(model)
model_device = next(model.parameters()).device
print("Model parameter device:", model_device)
# sys.exit("Stopping here for now")

# images, labels = next(iter(train_loader))
# images = images.to(device)
# outputs = model(images)
# print(f"Input batch shape: {images.shape}")
# print(f"Output shape: {outputs.shape}")

# train_model(model, train_loader, val_loader,device, epochs=2)
train_model(model, train_loader, val_loader, device, epochs=epochs, learning_rate=learning_rate)