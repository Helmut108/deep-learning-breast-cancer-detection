import sys
from pathlib import Path
from PIL import Image
import pydicom
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# This version: use CBIS-DDSM dataset with DICOM format

# To build paths inside a project like this: BASE_DIR / "subdir".
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Scripts" / "data"
CSV_PATH_TRAIN = DATA_DIR / 'DDSM'/ 'CSV' / "mass_case_description_train_set.csv"
ddsm_path = Path("/Volumes/Samsung_T7/Not_Sync/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/cbis_ddsm")
image_path = Path("/Volumes/Samsung_T7/Not_Sync/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/cbis_ddsm/Mass-Training_P_00001_LEFT_CC/74994/24515/e35e53fd-5312-406e-ad9f-4c11e725e53f.dcm")

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"CSV_PATH_TRAIN: {CSV_PATH_TRAIN}")
print(f"ddsm_path: {ddsm_path}")
# print(f"CSV_PATH_VAL: {CSV_PATH_VAL}")
# print(f"LABEL_TO_IDX: {LABEL_TO_IDX}")


def load_dicom_image(image_path):
    """
    Load a DICOM image from the given file path and return it as a PIL Image.
    """
    dicom_data = pydicom.dcmread(image_path)
    image = dicom_data.pixel_array

    return image

my_image    = load_dicom_image(image_path)
print(f"Loaded DICOM image shape: {my_image.shape}")

class ResizeAndPad:
    #constructor
    def __init__(self, size):
        self.size = size

    # runs when caalled
    def __call__(self, image):
        # channel, height, width = image.shape
        _, height, width = image.shape

        scale = min(
            self.size / height,
            self.size / width
        )

        # calc the scale factor
        new_height = round(height * scale)
        new_width = round(width * scale)

        # resize
        image = TF.resize(
            image,
            size=[new_height, new_width],
            antialias=True
        )
        # print(f"Resized shape before padding: {image.shape}")
        # plt.imshow(image.squeeze(0), cmap="gray")
        # plt.title(f"Before padding: {tuple(image.shape)}")
        # plt.axis("off")
        # plt.show()

        # calculate necessary paddig
        pad_height = self.size - new_height
        pad_width = self.size - new_width

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # pad the image
        image = TF.pad(
            image,
            padding=[
                pad_left,
                pad_top,
                pad_right,
                pad_bottom
            ],
            fill=0
        )
        print(f"Resized shape after padding: {image.shape}")
        return image



image = v2.ToImage()(my_image)
print(f"Image shape after ToImage: {image.shape}")  # PIL Image size    
image = v2.ToDtype(torch.float32, scale=True)(image)
print(f"Image shape after ToDtype: {image.shape}")  # PIL Image size
# print(image.dtype)
# print(image.min(), image.max())
custom_transform = ResizeAndPad(512)
output_image = custom_transform(image)
# print(f"Transformed image shape: {output_image.shape}")  # PIL Image size



# plt.imshow(output_image.squeeze(0), cmap="gray")
# plt.title(f"Transformed image: {tuple(output_image.shape)}")
# plt.axis("off")
# plt.show()






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


def create_dataframe_from_csv(CSV_PATH_TRAIN):

    df = pd.read_csv(CSV_PATH_TRAIN)
    print(f"DataFrame created from {CSV_PATH_TRAIN}. Shape: {df.shape}")

    df["Label"] = df["pathology"].map({
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    })

    # df = pd.read_csv("./data/mass_case_description_train_set.csv")
    df["folder"] = df["image file path"].str.split("/", n=1).str[0]
    df["study"] = df["image file path"].str.split("/").str[1].str[-5:]
    df["series"] = df["image file path"].str.split("/").str[2].str[-5:]

    df.info 
    print(df.head(2))
    print(df["Label"].value_counts(dropna=False))
    return df


df_train = create_dataframe_from_csv(CSV_PATH_TRAIN)




class CBISDataset(Dataset):
    def __init__(self, dataframe, ddsm_path, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.ddsm_path = Path(ddsm_path)    
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        series_path = self.ddsm_path / row["folder"] / row["study"] / row["series"]
        print(f"Series path: {series_path}")
        dicom_path = next(series_path.glob("*.dcm"))
        print(f"Loading DICOM image from: {dicom_path}")
        # image = Image.open(image_path).convert("RGB")
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array
        label = int(row["Label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# dataset = CBISDataset(df_train, ddsm_path, transform=transforms.ToTensor())
train_dataset = CBISDataset(df_train, ddsm_path)
first_image, first_label = train_dataset[0]
print(f"First image shape: {first_image.shape}, First label: {first_label}")

sys.exit("Stopping here for now")

def prepare_data():
    csv_df_train = pd.read_csv(CSV_PATH_TRAIN)
    # csv_df_val = pd.read_csv(CSV_PATH_VAL)

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
