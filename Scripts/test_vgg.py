# import os
# import certifi

# os.environ["SSL_CERT_FILE"] = certifi.where()

from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights

import torch
import torch.nn as nn
from cbismodules import utils
device = utils.get_device()

# load pretrained model
model_vgg = vgg16(weights=VGG16_Weights.DEFAULT)

model_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

print(model_vgg)
print(model_resnet)

# Freeze all layers
for p in model_vgg.parameters():
    p.requires_grad = False

# change final layer
model_vgg.fc = nn.Linear(4096,2)

model_vgg.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_vgg.parameters(), lr=learning_rate, weight_decay = weight_decay)



transfer_learning = "VGG"

if transfer_learning == "VGG":

if transfer_learning == "ResNet":