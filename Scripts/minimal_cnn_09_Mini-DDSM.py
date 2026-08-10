from pathlib import Path
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# This version: train/test split

# To build paths inside a project like this: BASE_DIR / "subdir".
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Scripts" / "data"
CSV_PATH_TRAIN = BASE_DIR / "Scripts" / "miniddsm_image_labels_train.csv"
CSV_PATH_VAL = BASE_DIR / "Scripts" / "miniddsm_image_labels_test.csv"
LABEL_TO_IDX = {"Benign": 0, "Cancer": 1}

# print(f"BASE_DIR: {BASE_DIR}")
# print(f"DATA_DIR: {DATA_DIR}")
# print(f"CSV_PATH: {CSV_PATH}")
# print(f"LABEL_TO_IDX: {LABEL_TO_IDX}")

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


class MiniDDSMDataset(Dataset):
    def __init__(self, dataframe, label_map, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = DATA_DIR / row["image_path"]
        # image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path).convert("L") # Convert to grayscale
        label = self.label_map[row["label"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def prepare_data():
    csv_df_train = pd.read_csv(CSV_PATH_TRAIN)
    csv_df_val = pd.read_csv(CSV_PATH_VAL)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),
    ])

    train_dataset = MiniDDSMDataset(csv_df_train, LABEL_TO_IDX, transform=transform)
    val_dataset = MiniDDSMDataset(csv_df_val, LABEL_TO_IDX, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_loader, val_loader, LABEL_TO_IDX


class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # conv1_out= 16 x 128 x 128

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after  pooling: (batch_size, 16, 64, 64)

        self.fc = nn.Linear(16 * 64 * 64, 2)  # Output layer for 2 classes of the Mini-DDSM dataset

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


def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=0.0001):

    print(epochs)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("\n--- BASIC TRAINING AND EVALUATION LOOP ---\n")



    for epoch in range(epochs):
        
        # *** Training Loop ***

        model.train() # Set the model to training mode for this epoch
        running_loss = 0.0
        
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

            running_loss += loss.item()
            if batch_idx % 1 == 0 and batch_idx >= 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}, RunningLoss: {running_loss / 1:.4f}")
                running_loss = 0.0


        # *** Evaluation Loop ***

        model.eval()  # Set the model to evaluation mode for validation
        accumulated_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0  

        with torch.no_grad():  # No gradient calculation for evaluation
            for val_images, val_targets in val_loader:
                val_images = val_images.to(device)
                val_targets = val_targets.to(device)

                # Forward pass only to get predictions for validation data
                val_output = model(val_images)

                # Calculate the loss for the validation data
                batch_val_loss = criterion(val_output, val_targets)
                accumulated_val_loss += batch_val_loss.item()

                # Get the predicted class labels
                # predicted_labels = torch.max(val_output, 1)
                predicted_labels = torch.argmax(val_output, dim=1)

                # Count correct predictions
                correct_predictions += (predicted_labels == val_targets).sum().item()
                total_samples += val_targets.size(0) 

    # Print out comprehensive epoch stats
        epoch_avg_val_loss = accumulated_val_loss / len(val_loader)
        # print(len(val_loader))
        # print(val_targets.size(0))
        accuracy_percentage = (correct_predictions / total_samples) * 100
        print(f"\n[EPOCH {epoch+1} VALIDATION SUMMARY]")
        print(f"--> Average Val Loss: {epoch_avg_val_loss:.4f}")
        print(f"--> Accuracy on unseen data: {accuracy_percentage:.2f}%\n")
        print("-" * 50)

    print("\n", "Basic training and evaluation loop functional!")



def main():
    device = get_device()
    train_loader, val_loader, label_map = prepare_data()
    print(f"Data loaders and label map prepared. Number of batches: {len(train_loader)}, {len(val_loader)}")
    print(f"Label map: {label_map}")

    model = MinimalCNN().to(device)
    train_model(model, train_loader, val_loader,device, epochs=2)


if __name__ == "__main__":
    main()


# check output classes
# update: if batch_idx % 1 == 0 and batch_idx >= 0 and {running_loss / 1:.4f}")
