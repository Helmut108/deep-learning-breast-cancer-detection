# model.py

import torch
from torch import nn
import torch.nn.functional as F

class DDSMCNN(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # conv1_in= 1x 512 x 512
        # H_out = ((512 + 2*1 - 3) / 1) + 1 = 512
        # W_out = ((512 + 2*1 - 3) / 1) + 1 = 512
        # So the tensor shape becomes: conv1_out= 16 x 512 x 512

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after  pooling: (batch_size, 16, 256, 256)
        # H_out = ((512 - 2) / 2) + 1 = 256
        # W_out = ((512 - 2) / 2) + 1 = 256
        # So the tensor shape becomes: pool1_out= 16 x 256 x 256

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # conv2_in= 16 x 256 x 256
        # H_out = ((256 + 2*1 - 3) / 1) + 1 = 256
        # W_out = ((256 + 2*1 - 3) / 1) + 1 = 256
        # So the tensor shape becomes: conv2_out= 32 x 256 x 256

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after  pooling: (batch_size, 32, 128, 128)
        # H_out = ((256 - 2) / 2) + 1 = 128
        # W_out = ((256 - 2) / 2) + 1 = 128
        # So the tensor shape becomes: pool2_out= 32 x 128 x 128

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # conv3_in= 32 x 128 x 128
        # H_out = ((128 + 2*1 - 3) / 1) + 1 = 128
        # W_out = ((128 + 2*1 - 3) / 1) + 1 = 128
        # So the tensor shape becomes: conv3_out= 64 x 128 x 128

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after  pooling: (batch_size, 64, 64, 64)
        # H_out = ((128 - 2) / 2) + 1 = 64
        # W_out = ((128 - 2) / 2) + 1 = 64
        # So the tensor shape becomes: pool3_out= 64 x 64 x 64

        self.dropout = nn.Dropout(p=dropout_p)

        self.fc = nn.Linear(64 * 64 * 64 , 2)  # Output layer for 2 classes: benign and malignant

    def forward(self, x):
        # print("Input shape:", x.shape)

        x= self.conv1(x)
        # print("After conv1:", x.shape)

        x= F.relu(x)
        # print("After ReLU:", x.shape)

        x= self.pool1(x)    
        # print("After pool1:", x.shape)

        x= self.conv2(x)
        # print("After conv2:", x.shape)

        x= F.relu(x)
        # print("After ReLU:", x.shape)

        x= self.pool2(x)
        # print("After pool2:", x.shape)  

        x= self.conv3(x)
        # print("After conv3:", x.shape)

        x= F.relu(x)
        # print("After ReLU:", x.shape)

        x= self.pool3(x)
        # print("After pool3:", x.shape)

        # Now flatten the 3D tensor (64 channels, 64 height, 64 width) into a 1D vector
        # x = x.view(-1, 64 * 64 * 64)  # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)
        # print("After flattening:", x.shape)

        x = self.dropout(x)
        # print("After dropout:", x.shape)

        # Pass the flattened vector into your linear layer
        x = self.fc(x)
        # print ("After fully connected layer:", x.shape)

        return x