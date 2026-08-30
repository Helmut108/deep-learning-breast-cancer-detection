# Save the first part earlier in your code
e=1
x=0.5
part_one = f"E{e} Avg Train Loss: {x:.4f} | Train Accuracy: 94.2%"

# ... other code runs here ...

v=2
y=0.3
# Save the second part later
part_two = f"E{v} Avg Val Loss:   {y:.4f} | Val Accuracy:   94.2%"

# Print both together at the end
print(f"{part_one} | {part_two}")

# print("\a")


import subprocess

# Play a native macOS system sound (e.g., Glass, Ping, Sosumi, Hero)
# subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])
# subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"])
# subprocess.run(["afplay", "/System/Library/Sounds/Hero.aiff"])
# # Or announce completion with speech
# subprocess.run(["say", "Process completed"])


import subprocess
import time

# 1. Start the stopwatch
start_time = time.perf_counter()

# --- Your Loop / Workload ---
total_iterations = 50
for epoch in range(1, total_iterations + 1):
    # Simulating work (replace with your actual code)
    time.sleep(0.05)
# ----------------------------

# 2. Stop the stopwatch
end_time = time.perf_counter()

# 3. Calculate elapsed time
elapsed = end_time - start_time

# Print formatted summary using field width and decimal precision
print(f"Completed {total_iterations:02d} iterations in {elapsed:.2f} seconds.")

# Play completion chime
subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])



    # epochs_range = range(1, len(train_loss_history) + 1)
    # plt.figure(figsize=(8, 4))
    # plt.plot(epochs_range, train_loss_history, marker='o', label='Train Loss')
    # plt.plot(epochs_range, val_loss_history, marker='o', label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training vs Validation Loss')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()