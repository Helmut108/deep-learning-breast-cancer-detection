import torch
import torch.nn as nn
import torch.nn.functional as F

# formulas for calculating output dimensions of convolutional and pooling layers:
# output = floor((input_size + 2*padding - kernel_size) / stride) + 1
# height_out = floor((height_in + 2*padding - kernel_size) / stride) + 1
# width_out = floor((width_in + 2*padding - kernel_size) / stride) + 1

class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Create the layers for a minimal CNN architecture

        # First convolution block
        # Input: 1 channel (grayscale image)
        # Output: 16 learned feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # H_out = ((128 + 2*1 - 3) / 1) + 1 = 128
        # W_out = ((128 + 2*1 - 3) / 1) + 1 = 128
        # So the tensor shape becomes: conv1_out= 16 x 128 x 128

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # H_out = ((128 - 2) / 2) + 1 = 64
        # W_out = ((128 - 2) / 2) + 1 = 64
        # So the tensor shape becomes: pool1_out= 16 x 64 x 64

        self.fc = nn.Linear(16 * 64 * 64, 2)

    def forward(self, x):
        print("Input shape:", x.shape)
        
        # Step 1: The input image goes through the convolution layer like this :
        # Input shape: [1, 1, 128, 128] -> Output shape: [1, 16, 128, 128]
        x= self.conv1(x)
        print("After conv1:", x.shape)

        # Then we apply the ReLU activation function
        # Shape stays exactly the same: [1, 16, 128, 128]
        x= F.relu(x)
        print("After ReLU:", x.shape)

        # Finally we apply the Max Pooling layer
        # Spatial dimensions are halved -> Output shape: [1, 16, 64, 64]
        x= self.pool1(x)    
        print("After pool1:", x.shape)

        # Now flatten the 3D tensor (16 channels, 64 height, 64 width) into a 1D vector
        x = x.view(-1, 16 * 64 * 64) 
        print("After flattening:", x.shape)

        # 3And pass the flattened vector into your linear layer
        x = self.fc(x)
        print ("After fully connected layer:", x.shape)

        return x
    
if __name__ == "__main__":
    # Create a dummy batch of 1 grayscale image with 128x128 pixels
    x = torch.randn(1, 1, 128, 128)
    # And a dummy label for this single image, class 1 (e.g., malignant)
    y = torch.tensor([1])  
        
    # Initialize the model
    model = MinimalCNN()
        
    # Loss function for binary classification
    criterion = nn.CrossEntropyLoss()

    # Basic Optimizer with learning rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    print("--- BASIC TRAINING STEP ---")
        
    # Zero the parameter gradients. Mandatory step in PyTorch
    optimizer.zero_grad()

    # Run the forward pass to get the model's predictions (logits)
    output = model(x)
    print("Logits output shape:", output.shape)

    # Compute the loss value (the difference between the predicted logits and the true label)
    loss = criterion(output, y)
    print(f"Calculated Loss: {loss.item():.4f}")
        
    # The backward pass calculates the gradients
    loss.backward()
    print("Backward pass successful. Gradients calculated.")

    # The Optimizer updates the weights based on gradients)
    optimizer.step()
    print("Optimizer step successful. Network weights updated.")

    print("The basic deep learning training pipeline functional!")
