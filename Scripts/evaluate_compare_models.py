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
model_name = "scheduler_model"
model_path = BASE_DIR / f"{model_name}.pth"
print(f"Model path: {model_path}")
print(f"Model name: {model_name}")
# sys.exit("Stopping here for now")

train_loader, val_loader = prepare_datasets(csv_path_train, ddsm_path, batch_size=batch_size, sample_size=sample_size)

model = DDSMCNN().to(device)
print(model)

state_baseline = torch.load(
    BASE_DIR / "baseline_model.pth",
    map_location="cpu",
    weights_only=True
)


state_dropout = torch.load(
    BASE_DIR / "dropout_model.pth",
    map_location="cpu",
    weights_only=True
)

same = all(
    torch.equal(state_baseline[key], state_dropout[key])
    for key in state_baseline
)

print("Models identical:", same)

sys.exit("Stopping here for now")

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)



print(model)

evaluate_model(model, val_loader, device)

