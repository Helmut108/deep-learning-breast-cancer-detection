# model.py

import torch
from torch import nn
import torch.nn.functional as F

class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # conv1_out= 1x 512 x 512
        # H_out = ((512 + 2*1 - 3) / 1) + 1 = 512
        # W_out = ((512 + 2*1 - 3) / 1) + 1 = 512
        # So the tensor shape becomes: conv1_out= 16 x 512 x 512

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after  pooling: (batch_size, 16, 256, 256)
        # H_out = ((512 - 2) / 2) + 1 = 256
        # W_out = ((512 - 2) / 2) + 1 = 256
        # So the tensor shape becomes: pool1_out= 16 x 256 x 256

        self.fc = nn.Linear(16 * 256 * 256, 2)  # Output layer for 2 classes: benign and malignant

    def forward(self, x):
        # print("Input shape:", x.shape)

        x= self.conv1(x)
        # print("After conv1:", x.shape)

        x= F.relu(x)
        # print("After ReLU:", x.shape)

        x= self.pool1(x)    
        # print("After pool1:", x.shape)

        # Now flatten the 3D tensor (16 channels, 256 height, 256 width) into a 1D vector
        # x = x.view(-1, 16 * 256 * 256) 
        x = torch.flatten(x, 1)
        # print("After flattening:", x.shape)

        # Pass the flattened vector into your linear layer
        x = self.fc(x)
        # print ("After fully connected layer:", x.shape)

        return x