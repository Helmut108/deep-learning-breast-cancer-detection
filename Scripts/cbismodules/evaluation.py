import sys
import torch
from torch import nn


def evaluate_model(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    accumulated_val_loss = 0.0
    avg_val_loss = 0.0
    accuracy = 0.0
    correct_val_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images = val_images.to(device)
            val_targets = val_targets.to(device)

            val_output = model(val_images)

            # Calculate the loss for the validation data
            val_loss = criterion(val_output, val_targets)
            accumulated_val_loss += val_loss.item()

            # Calculate the accuracy for the validation data
            probabilities = torch.softmax(val_output, dim=1)
            print(f"Probabilities: {probabilities}")
            print(f"Probabilities: {probabilities.shape}")
            print(f"Probabilities: {probabilities[:,1]}")
            probabilities_malignant = probabilities[:, 1]  # probabilities for the positive class
            print(f"Probabilities for malignant class: {probabilities_malignant}")
            predicted_labels = torch.argmax(val_output, dim=1)
            # Print the probabilities and predicted labels for each sample in the batch
            for i in range(val_targets.size(0)):
                # print(f"Sample {i}: Probability = {probabilities[i].max().item():.4f}, Predicted = {predicted_labels[i].item()}, Actual = {val_targets[i].item()}")
                print(f"Sample {i}: Probability Malignant = {probabilities_malignant[i].item():.4f}, Predicted = {predicted_labels[i].item()}, Actual = {val_targets[i].item()}")
            total_samples += val_targets.size(0)
            correct_val_predictions += (predicted_labels == val_targets).sum().item()
            sys.exit("Stopping here for now")

    # Print out comprehensive epoch stats
    print(f"Total validation samples: {total_samples}")
    print(f"Length of validation loader: {len(val_loader)}")
    avg_val_loss = accumulated_val_loss / len(val_loader)
    accuracy = correct_val_predictions / total_samples 
    accuracy_percentage = accuracy * 100
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy_percentage:.2f}%")
    # return accuracy