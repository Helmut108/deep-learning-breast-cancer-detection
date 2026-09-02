# def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=0.0001):
import sys
import torch
import torch.nn as nn
import subprocess
import time
import matplotlib.pyplot as plt

# 1. Start the stopwatch
start_time = time.perf_counter()
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []
learning_rate_history = []


def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=0.0001, early_stopping=False, patience=3):
# def train_model(model, train_loader, device, epochs=10, learning_rate=0.0001):
    print(f"Epochs: {epochs}, Learning rate: {learning_rate}")
    print(f"len(train_loader): {len(train_loader)}")
    print(f"len(val_loader): {len(val_loader)}")
    print("Device:", device)
    print(f"Training for {epochs} epochs")
    # sys.exit("Stopping here for now")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=5
    )
    print("Scheduler:", scheduler)
    # sys.exit("Stopping here for now")

    print("\n--- BASIC TRAINING AND EVALUATION LOOP ---\n")

    epochs_without_improvement = 0
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"Early stopping: {early_stopping}, Patience: {patience} type: {type(patience)}")
    print(f"Initial best validation loss: {best_val_loss} type: {type(best_val_loss)}")
    print(f"Initial epochs without improvement: {epochs_without_improvement} type: {type(epochs_without_improvement)}")
    # sys.exit("Stopping here for now")

    # **********************************************
    #           *** Training Loop ***
    # **********************************************

    for epoch in range(epochs):
        
        # *** Training Loop ***
        # print(f"\n[EPOCH {epoch+1}/{epochs} TRAINING]")
        model.train() # Set the model to training mode for this epoch
        running_loss = 0.0
        total_train_samples = 0
        correct_train_predictions_sofar = 0
        
        for batch_idx, (real_images, real_targets) in enumerate(train_loader):
            train_samples = real_images.size(0)  # Get the number of samples in the current batch
            # print(f"Train samples: {train_samples}")
            optimizer.zero_grad()

            # Move batch tensors to the selected device once per batch
            real_images = real_images.to(device)
            real_targets = real_targets.to(device)

            # print(real_images.shape)
            # Forward pass: pass data forward through the model to get predictions
            train_output = model(real_images)

            # Calculate the loss value (the difference between the predicted logits and the true label)
            loss = criterion(train_output, real_targets)

            predicted_train_labels = torch.argmax(train_output, dim=1)
            # print(f"{batch_idx} Batch index: {batch_idx}")
            # print(f".  Predicted train labels: {predicted_train_labels}")
            total_train_samples += real_targets.size(0)  # Update the total number of training samples processed
            correct_train_predictions = (predicted_train_labels == real_targets).sum().item()
            # print(f". Actual train labels:    {real_targets}")
            # print(f". Correct predictions in this batch: {correct_train_predictions} / {train_samples}")
            correct_train_predictions_sofar += correct_train_predictions
            # print(f".  Total correct predictions so far: {correct_train_predictions_sofar} / {total_train_samples}")

            # Backward pass: calculate the new gradients based on the current loss
            loss.backward()

            # Update the weights based on the new gradients
            optimizer.step()

            running_loss += loss.item()
            # if batch_idx % 10 == 0 and batch_idx >= 0:
            #     print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}, RunningLoss: {running_loss / (batch_idx + 1):.4f}")
                # running_loss = 0.0
        
        # print(f"\n[EPOCH {epoch+1}/{epochs} TRAINING SUMMARY]")
        epoch_loss = running_loss / len(train_loader)
        # print(f"len(train_loader): {len(train_loader)}")
        # print(f"Batch index: {batch_idx}")
        # print(f"--> Total training samples: {len(train_loader.dataset)}")
        # print(f"--> Total training samples processed: {total_train_samples}")

        # accuracy_percentage = (correct_predictions / total_samples) * 100
        # print(f"--> Correct predictions: {correct_train_predictions_sofar}")
        epoch_accuracy = (correct_train_predictions_sofar / total_train_samples) * 100
        # print(f"--> Training accuracy for epoch {epoch + 1}: {epoch_accuracy:.2f}%")
        # print(f"--> Average training loss: {epoch_loss:.4f}")

        # print(f"E{epoch + 1} Avg Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}%")
        train_print = f"E{(epoch + 1):02d} Avg Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}%"
        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)
        # print(
        #     f"Epoch {epoch + 1}/{epochs} | "
        #     f"Average training loss: {epoch_loss:.4f}"
        # )

        # **********************************************
        #           *** Evaluation Loop ***
        # **********************************************

        # print(f"\n[EPOCH {epoch+1}/{epochs} VALIDATION]")
        model.eval()  # Set the model to evaluation mode for validation
        accumulated_val_loss = 0.0
        correct_val_predictions = 0
        total_samples = 0  

        with torch.no_grad():  # No gradient calculation for evaluation
            for val_images, val_targets in val_loader:
                val_images = val_images.to(device)
                val_targets = val_targets.to(device)
                # print(f"Val samples: {val_targets.size(0)}")  # Print the number of validation samples in the current batch

                # Forward pass only to get predictions for validation data
                val_output = model(val_images)

                # Calculate the loss for the validation data
                batch_val_loss = criterion(val_output, val_targets)
                accumulated_val_loss += batch_val_loss.item()

                # Get the predicted class labels
                # predicted_labels = torch.max(val_output, 1)
                predicted_labels = torch.argmax(val_output, dim=1)
                # print(f"Predicted labels: {predicted_labels}")

                # Count correct predictions
                correct_val_predictions += (predicted_labels == val_targets).sum().item()
                total_samples += val_targets.size(0) 
                # print(f".  Predicted: {predicted_labels}")
                # print(f"   Actual:   {val_targets}")
                # print(f"   Correct predictions in this batch: {(predicted_labels == val_targets).sum().item()} / {val_targets.size(0)}")
                # print(f"   Total correct predictions so far: {correct_val_predictions} / {total_samples}")
                # print(f"Total samples processed so far: {total_samples}")
                # print(f"Correct predictions so far: {correct_val_predictions} / {total_samples}")

        # print(f"\n[EPOCH {epoch+1}/{epochs} VALIDATION SUMMARY]")
        # Print out comprehensive epoch stats
        epoch_avg_val_loss = accumulated_val_loss / len(val_loader)
        scheduler.step(epoch_avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Average validation loss: {epoch_avg_val_loss:.4f}")
        learning_rate_history.append(scheduler.optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch + 1}/{epochs} | Learning rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
        # print("Scheduler:", scheduler)
        # sys.exit("Stopping here for now")
        # print(len(val_loader))
        # print(val_targets.size(0))
        accuracy_percentage = (correct_val_predictions / total_samples) * 100
        # print(f"\n[EPOCH {epoch+1} VALIDATION SUMMARY]")
        # print(f"len(val_loader): {len(val_loader)}")
        # print(f"--> Total validation samples: {total_samples}")
        # print(f"--> Correct predictions: {correct_val_predictions}")
        # print(f"--> Accuracy on unseen data: {accuracy_percentage:.2f}%")
        # print(f"--> Average Val Loss: {epoch_avg_val_loss:.4f}")

        # print(f"E{epoch + 1} Avg Val Loss:   {epoch_avg_val_loss:.4f} | Val Accuracy:   {accuracy_percentage:.2f}%")
        val_print = f"E{(epoch + 1):02d} Avg Val Loss:   {epoch_avg_val_loss:.2f} | Val Accuracy:   {accuracy_percentage:.2f}%"
        val_loss_history.append(epoch_avg_val_loss)
        val_accuracy_history.append(accuracy_percentage)
        print(f"{train_print} | {val_print}")
        # print("-" * 50)

        if epoch_avg_val_loss < best_val_loss:
            best_val_loss = epoch_avg_val_loss
            epochs_without_improvement = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
        elif early_stopping:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs \n")
                print(f"due to no improvement in validation loss for {patience} epochs.")
                print(
                f"Best model saved at epoch {best_epoch} "
                f"with validation loss {best_val_loss:.4f}"
    )
                break

    print("\n", "Basic training and evaluation loop functional!\n")
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Completed {epochs} epochs in {elapsed:.2f} seconds.")
    subprocess.run(["afplay", "/System/Library/Sounds/Hero.aiff"])
    subprocess.run(["say", "Process completed"])
    print(f"Training Loss History: {train_loss_history}")
    print(f"Training Accuracy History: {train_accuracy_history}")
    print(f"Validation Loss History: {val_loss_history}")
    print(f"Validation Accuracy History: {val_accuracy_history}")

    print(
    f"Best model saved at epoch {best_epoch} "
    f"with validation loss {best_val_loss:.4f}"
    )

    epochs_range = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, train_loss_history, marker='o', label='Train Loss')
    plt.plot(epochs_range, val_loss_history, marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(x = best_epoch, color = 'g', label = 'best epoch', linestyle='--')

    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, train_accuracy_history, marker='o', label='Train Accuracy')
    plt.plot(epochs_range, val_accuracy_history, marker='o', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(x = best_epoch, color = 'g', label = 'best epoch', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # model = MinimalCNN().to(device)
    # train_model(model, train_loader, val_loader,device, epochs=2)
