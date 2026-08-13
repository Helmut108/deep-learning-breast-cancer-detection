import sys
import torch
from pathlib import Path
from config import CSV_PATH_TRAIN, DDSM_PATH
from cbismodules import utils
from cbismodules.transforms import ResizeAndPad
from cbismodules.dataprep import create_dataframe_from_csv
from cbismodules.dataprep import prepare_datasets

# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = BASE_DIR / "Scripts" / "data"
# CSV_PATH_TRAIN = DATA_DIR / 'DDSM'/ 'CSV' / "mass_case_description_train_set.csv"
# ddsm_path = Path("/Volumes/Samsung_T7/Not_Sync/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/cbis_ddsm")

device = utils.get_device()
print(device)

# print(f"CSV_PATH_TRAIN: {CSV_PATH_TRAIN}")
# print(f"DDSM_PATH: {DDSM_PATH}")

# sys.exit("Stopping here for now")

test_image = torch.zeros((1, 4808, 3024))
output = ResizeAndPad(512)(test_image)
assert output.shape == (1, 512, 512)
print(output.shape)


df_train = create_dataframe_from_csv(CSV_PATH_TRAIN)
assert len(df_train) == 1318
assert "Label" in df_train.columns
assert df_train["Label"].isna().sum() == 0

print(df_train[["folder", "study", "series", "Label"]].head())
# df_train.info()
print(df_train.head(2))

df_train_small = df_train.head(8)
print(df_train_small)
print(len(df_train_small))
print(len(df_train))


df_train, train_dataset, train_loader = prepare_datasets(CSV_PATH_TRAIN, DDSM_PATH, batch_size=8, sample_size=64)

print(f"\nLength of train_dataset: {len(train_dataset)}")
print(f"Length of train_loader: {len(train_loader)}")
first_image, first_label = train_dataset[0]
print(f"First image shape: {first_image.shape}, First label: {first_label}")
print(f"First image dtype: {first_image.dtype}")
print(f"First image min: {first_image.min()}, First image max: {first_image.max()}")
images, labels = next(iter(train_loader))

print(f"Batch image shape: {images.shape}")
print(f"Batch label shape: {labels.shape}")
print(f"Batch image dtype: {images.dtype}")
print(f"Labels: {labels}")
print
last_image, last_label = train_dataset[-1]
print(f"Last image shape: {last_image.shape}, Last label: {last_label}")

last_images = None
last_labels = None

for images, labels in train_loader:
    last_images = images
    last_labels = labels

print(last_images.shape)
print(last_labels.shape)
