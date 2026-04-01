import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from floppy import FLOPpyTracker, WandbConfiguration


# Set device and seed for reproducibility
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ==========================================
# DATA AND MODEL SETUP
# ==========================================

# Create a dummy dataset for classification
X = torch.randn(64, 10).to(device)
y = torch.randint(0, 3, (64,)).to(device)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
num_epochs = 15

# Define a simple neural network
model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to(device)
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer.
# FLOPpy natively supports tracking various optimizers:
# SGD, AdamW, RMSprop, Adagrad, etc. We use Adam in this example.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Optional configuration for real-time telemetry on Weights & Biases
wandb_config = WandbConfiguration(
    project_name="torch_test",
    group_name="eDPO",
    reporter_key="your_wandb_key_here",
)
# or
wandb_config = None

# ==========================================
# FLOPpy INTEGRATION
# ==========================================

# 1. Initialize the tracker and assign an experiment name
tracker = FLOPpyTracker(run_name="torch_test")

# 2. Start monitoring by "hooking" into the model and its components.
# By passing export_path and wandb_config, FLOPpy will save a local CSV
# and stream the logs (FLOPs/BOPs) to the dashboard in real-time.
tracker.run(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    export_path="torch_test.csv",
    wandb_config=wandb_config,
)

# ==========================================
# TRAINING LOOP
# ==========================================

model.train()

for _ in range(num_epochs):
    for xb, yb in loader:
        optimizer.zero_grad()
        y_hat = model(xb)

        loss = loss_fn(y_hat, yb)
        loss.backward()
        optimizer.step()

        # Signal the end of a single batch to FLOPpy (not mandatory).
        # Crucial for accumulating the incremental computational workload.
        tracker.batch()

    # Signal the end of an entire epoch to FLOPpy (not mandatory).
    # This trigger aggregates the statistics and synchronizes data with W&B.
    tracker.epoch()

# ==========================================
# FINAL REPORT
# ==========================================

# 3. Generate and print the detailed hardware and algorithmic workload summary
report = tracker.report()
print(report)
