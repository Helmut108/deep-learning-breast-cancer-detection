import sys
import torch
from cbismodules.dataprep import prepare_datasets
from config import csv_path_train, ddsm_path, checkpoint_path
from cbismodules.model import DDSMCNN
from cbismodules import utils
from cbismodules.evaluation import evaluate_model

device = utils.get_device()

sample_size = None
batch_size = 8


train_loader, val_loader = prepare_datasets(csv_path_train, ddsm_path, batch_size=batch_size, sample_size=sample_size)

model = DDSMCNN().to(device)

# sys.exit("Stopping here for now")

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

evaluate_model(model, val_loader, device)

