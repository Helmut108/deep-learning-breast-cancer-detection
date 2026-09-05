import sys
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import roc_auc_score
import subprocess
# sys.exit("Stopping here for now")

def evaluate_model(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    accumulated_val_loss = 0.0
    avg_val_loss = 0.0
    accuracy = 0.0
    correct_val_predictions = 0
    total_samples = 0

    tp_total = 0
    fn_total = 0
    tn_total = 0
    fp_total = 0
    auroc = BinaryAUROC().to(device)

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images = val_images.to(device)
            val_targets = val_targets.to(device)

            val_output = model(val_images)

            # Calculate the loss for the validation data
            val_loss = criterion(val_output, val_targets)
            accumulated_val_loss += val_loss.item()* val_targets.size(0)
            # print("val_targets_size:", val_targets.size(0))

            # Calculate the accuracy for the validation data
            probabilities = torch.softmax(val_output, dim=1)
            probabilities_malignant = probabilities[:, 1]  # probabilities for the positive class
            # print(f"Probabilities: {probabilities}")
            # print(f"Probabilities shape: {probabilities.shape}")
            # print(f"Probabilities for malignant class: {probabilities[:, 1]}")
            # print(f"Predicted labels: {torch.argmax(val_output, dim=1)}")
            # print(f"Actual labels: {val_targets}")
            # sys.exit("Stopping here for now")

            # print(f"Probabilities: {probabilities}")
            # print(f"Probabilities: {probabilities.shape}")
            # print(f"Probabilities: {probabilities[:,1]}")
            probabilities_malignant = probabilities[:, 1]  # probabilities for the positive class
            # print(f"Probabilities for malignant class: {probabilities_malignant}")
            predicted_labels = torch.argmax(val_output, dim=1)
            # Print the probabilities and predicted labels for each sample in the batch
            # for i in range(val_targets.size(0)):
                # print(f"Sample {i}: Probability = {probabilities[i].max().item():.4f}, Predicted = {predicted_labels[i].item()}, Actual = {val_targets[i].item()}")
                # print(f"Sample {i}: Probability Malignant = {probabilities_malignant[i].item():.4f}, Predicted = {predicted_labels[i].item()}, Actual = {val_targets[i].item()}")
                # Calculate confusion matrix elements
                # if predicted_labels[i].item() == 1 and val_targets[i].item() == 1:
                #     tp_total += 1
                # elif predicted_labels[i].item() == 0 and val_targets[i].item() == 1:
                #     fn_total += 1
                # elif predicted_labels[i].item() == 0 and val_targets[i].item() == 0:
                #     tn_total += 1
                # elif predicted_labels[i].item() == 1 and val_targets[i].item() == 0:
                #     fp_total += 1

            total_samples += val_targets.size(0)
            correct_val_predictions += (predicted_labels == val_targets).sum().item()
            # tp_total += ((predicted_labels == 1) & (val_targets == 1)).sum().item()
            # fn_total += ((predicted_labels == 0) & (val_targets == 1)).sum().item()
            # tn_total += ((predicted_labels == 0) & (val_targets == 0)).sum().item()
            # fp_total += ((predicted_labels == 1) & (val_targets == 0)).sum().item()

            tp_total += ((predicted_labels == 1) & (val_targets == 1)).sum().item()
            fn_total += ((predicted_labels == 0) & (val_targets == 1)).sum().item()

            tn_total += ((predicted_labels == 0) & (val_targets == 0)).sum().item()
            fp_total += ((predicted_labels == 1) & (val_targets == 0)).sum().item()
            auroc.update(probabilities_malignant, val_targets)
            all_labels.extend(val_targets.cpu().numpy())
            all_probs.extend(probabilities_malignant.cpu().numpy())
            # sys.exit("Stopping here for now")

    # Print out comprehensive epoch stats
    print(f"Total validation samples: {total_samples}")
    print(f"Length of validation loader: {len(val_loader)}")
    # avg_val_loss = accumulated_val_loss / len(val_loader)
    avg_val_loss = accumulated_val_loss / total_samples
    accuracy = correct_val_predictions / total_samples 
    accuracy_percentage = accuracy * 100
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy_percentage:.2f}%")
    # balanced_accuracy = (tp_total / (tp_total + fn_total) + tn_total / (tn_total + fp_total)) / 2
    # print(f"Validation Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    sensitivity = tp_total / (tp_total + fn_total)
    specificity = tn_total / (tn_total + fp_total)
    balanced_accuracy = (sensitivity + specificity) / 2
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"TP: {tp_total}")
    print(f"FN: {fn_total}")
    print(f"TN: {tn_total}")
    print(f"FP: {fp_total}")
    # print(f"Confusion Matrix - TP: {tp_total}, FN: {fn_total}, TN: {tn_total}, FP: {fp_total}")

    print(f"Sensitivity: {sensitivity * 100:.2f}%")
    print(f"Specificity: {specificity * 100:.2f}%")

    auroc_value = auroc.compute().item()
    print(f"AUROC: {auroc_value:.4f}")

    auc_sklearn = roc_auc_score(all_labels, all_probs)
    print(f"AUROC (sklearn): {auc_sklearn:.4f}")

    subprocess.run(["afplay", "/System/Library/Sounds/Hero.aiff"])
    subprocess.run(["say", "Process completed"])