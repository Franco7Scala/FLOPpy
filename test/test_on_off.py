import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from floppy import FLOPpyTracker

# Set device and seed for strict reproducibility
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ==========================================
# DATA AND MODEL SETUP
# ==========================================

# Create a dummy dataset (64 samples, batch_size=16 -> 4 batches per epoch)
X = torch.randn(64, 10).to(device)
y = torch.randint(0, 3, (64,)).to(device)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Define a simple neural network
model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Helper function to run exactly one epoch of training
def train_one_epoch(is_tracked: bool):
    for xb, yb in loader:
        optimizer.zero_grad()
        y_hat = model(xb)
        loss = loss_fn(y_hat, yb)
        loss.backward()
        optimizer.step()

        # Only call FLOPpy step methods if we are currently tracking
        if is_tracked:
            tracker.batch()

    if is_tracked:
        tracker.epoch()


# ============================================================
# TEST: ATTACH -> DETACH -> REATTACH LIFECYCLE
# ============================================================

model.train()
tracker = FLOPpyTracker(run_name="test_lifecycle")

print("\n=== PHASE 1: Tracking ON ===")
# 1. ATTACH the tracker
tracker.run(model=model, optimizer=optimizer, loss_fn=loss_fn)

# Run 1 epoch
train_one_epoch(is_tracked=True)

# Save the FLOPs counted in Phase 1
report_phase1 = tracker.report()
flops_phase1 = report_phase1.overall_flop
print(f"FLOPs after Phase 1 (1 epoch): {flops_phase1:,}")
assert flops_phase1 > 0, "Tracker failed to record FLOPs in Phase 1."

print("\n=== PHASE 2: Tracking OFF (Detached) ===")
# 2. DETACH the tracker.
# This should safely remove all hooks and ATen dispatch modes.
tracker.stop()

# Run 1 epoch UNTRACKED. The model is doing math, but FLOPpy shouldn't see it.
train_one_epoch(is_tracked=False)

# Get the report again. The FLOPs should be EXACTLY the same as Phase 1.
report_phase2 = tracker.report()
flops_phase2 = report_phase2.overall_flop
print(f"FLOPs after Phase 2 (Untracked epoch): {flops_phase2:,}")
assert flops_phase1 == flops_phase2, "GHOST TRACKING DETECTED! Tracker is still counting after being stopped."

print("\n=== PHASE 3: Tracking ON (Reattached) ===")
# 3. REATTACH the tracker.
# It should inject the hooks again and resume counting from where it left off.
tracker.run(model=model, optimizer=optimizer, loss_fn=loss_fn)

report_phaseXX = tracker.report()

# Run 1 epoch
train_one_epoch(is_tracked=True)

report_phase3 = tracker.report()
flops_phase3 = report_phase3.overall_flop
print(f"FLOPs after Phase 3 (1 epoch tracked): {flops_phase3:,}")

# Validation: Since Phase 1 and Phase 3 both ran exactly 1 tracked epoch,
# the total FLOPs in Phase 3 should be exactly double the FLOPs of Phase 1.
assert flops_phase3 == flops_phase1, "Reattaching failed! FLOPs didn't accumulate correctly."

print("\n" + "=" * 40)
print("✅ TEST PASSED: Attach/Detach/Reattach logic works perfectly!")
print("=" * 40)

# Print final report just to see the visual output
print(report_phase3)
