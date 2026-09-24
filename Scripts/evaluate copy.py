import sys
import torch
from cbismodules.dataprep import prepare_datasets
from config import BASE_DIR, csv_path_train, ddsm_path
from cbismodules.model import DDSMCNN, build_model
from cbismodules import utils
from cbismodules.evaluation import evaluate_model

device = utils.get_device()

sample_size = None
batch_size = 8
use_augmentation = False
architecture = "resnet18"
model_name = "resnet_model_1"
model_path = BASE_DIR / f"{model_name}.pth"
print(f"Model path: {model_path}")
print(f"Model name: {model_name}")
# sys.exit("Stopping here for now")

train_loader, val_loader = prepare_datasets(csv_path_train, ddsm_path, batch_size=batch_size, sample_size=sample_size, use_augmentation=use_augmentation)

# model = DDSMCNN().to(device)
model = build_model(
    architecture=architecture,
    num_classes=2,
    pretrained=False,
    freeze_backbone=True,
    dropout_p=0.0,
    ).to(device)
# print(model)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

load_result = model.load_state_dict(state_dict)

print("Model path:", model_path)
print("Architecture:", architecture)
print("Load result:", load_result)
print("Final layer:", model.fc)

# sys.exit("Stopping here for now")
evaluate_model(model, val_loader, device)

