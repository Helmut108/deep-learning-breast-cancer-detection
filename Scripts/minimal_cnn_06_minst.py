import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# add data loader

def get_device():
    """
    Return the best available device:
    CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    return device

def prepare_data():
    # Using transformation to scale up the resolution
    # To match the 128x128 architecture size of the MVP
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Load 60000 images and labels from the FashionMNIST training dataset 
    # and transform to 128*128
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=False, 
        transform=transform_pipeline
    )

    # Create a data loader that serves the data in batches, e.g. 8, 16, 32, 64, 128) 
    # and shuffles the data for each epoch
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # # 2. Directly grab the first 8 samples from the dat`aset object
    # images = []
    # labels = []
    # for i in range(8):
    #     img, lbl = train_dataset[i] # Unpacks the image tensor and integer label
    #     images.append(img)
    #     labels.append(lbl)

    # # 3. Stack them together into a proper 4D mini-batch tensor 
    # # This manually handles the [Batch Size, Channels, Height, Width] shape
    # x = torch.stack(images) 
    # y = torch.tensor(labels)

    return train_loader

class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # conv1_out= 16 x 128 x 128

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after  pooling: (batch_size, 16, 64, 64)

        self.fc = nn.Linear(16 * 64 * 64, 10)  # Output layer for 10 classes of the FashionMNIST dataset

    def forward(self, x):
        # print("Input shape:", x.shape)

        x= self.conv1(x)
        # print("After conv1:", x.shape)

        x= F.relu(x)
        # print("After ReLU:", x.shape)

        x= self.pool1(x)    
        # print("After pool1:", x.shape)

        # Now flatten the 3D tensor (16 channels, 64 height, 64 width) into a 1D vector
        x = x.view(-1, 16 * 64 * 64) 
        # print("After flattening:", x.shape)

        # Pass the flattened vector into your linear layer
        x = self.fc(x)
        # print ("After fully connected layer:", x.shape)

        return x


def train_model(model, train_loader, device, epochs=10, learning_rate=0.0001):
    # x = x.to(device)
    # y = y.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("--- BASIC TRAINING LOOP MINI BATCH---")

    model.train()

    for epoch in range(epochs):
        for batch_idx, (real_images, real_targets) in enumerate(train_loader):

            optimizer.zero_grad()

            # Move batch tensors to the selected device once per batch
            real_images = real_images.to(device)
            real_targets = real_targets.to(device)

            # Forward pass: pass data forward through the model to get predictions
            output = model(real_images)

            # Calculate the loss value (the difference between the predicted logits and the true label)
            loss = criterion(output, real_targets)

            # Backward pass: calculate the new gradients based on the current loss
            loss.backward()

            # Update the weights based on the new gradients
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    print("The basic mini batch training loop functional!")


def main():
    device = get_device()
    train_loader = prepare_data()

    model = MinimalCNN().to(device)
    train_model(model, train_loader, device)


if __name__ == "__main__":
    main()


