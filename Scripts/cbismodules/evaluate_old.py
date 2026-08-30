import sys
import torch
from cbismodules.dataprep import prepare_datasets
from config import csv_path_train, ddsm_path, checkpoint_path
from cbismodules.model import DDSMCNN, DDSMCNN
from cbismodules import utils

device = utils.get_device()

sample_size = None
batch_size = 8

print("checkpoint_path:", checkpoint_path)
print("sample_size:", sample_size)
print("batch_size:", batch_size)

# train_loader, val_loader = prepare_datasets(csv_path_train, ddsm_path, batch_size=batch_size, sample_size=sample_size)

# model = DDSMCNN().to(device)


sys.exit("Stopping here for now")

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

