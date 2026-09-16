import sys
import torch
from cbismodules.dataprep import prepare_datasets
from config import BASE_DIR, csv_path_train, ddsm_path
from cbismodules.model import DDSMCNN
from cbismodules import utils
from cbismodules.evaluation import evaluate_model

device = utils.get_device()

sample_size = None
batch_size = 8
use_augmentation = True
model_name = "augmentation_model_3"
model_path = BASE_DIR / f"{model_name}.pth"
print(f"Model path: {model_path}")
print(f"Model name: {model_name}")
# sys.exit("Stopping here for now")

train_loader, val_loader = prepare_datasets(csv_path_train, ddsm_path, batch_size=batch_size, sample_size=sample_size, use_augmentation=use_augmentation)

model = DDSMCNN().to(device)
print(model)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)
print(model)
# sys.exit("Stopping here for now")
evaluate_model(model, val_loader, device)

