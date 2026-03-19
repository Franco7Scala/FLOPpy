import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from floppy import FLOPpyTracker

# ------------------------------------------------------------
# Shared setup
# ------------------------------------------------------------

torch.manual_seed(42)

X = torch.randn(64, 10)
y = torch.randint(0, 3, (64,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

num_epochs = 2

# ============================================================
# MODE 1
# ============================================================

model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

tracker = FLOPpyTracker(
    run_name="torch_test_mode1",
    print_summary=True,
    print_hardware=True,
)

tracker.run(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    export_path="torch_test_mode1.csv",
)

model.train()

for _ in range(num_epochs):
    for xb, yb in loader:
        optimizer.zero_grad()
        y_hat = model(xb)
        loss = loss_fn(y_hat, yb)
        loss.backward()
        optimizer.step()

        tracker.batch()

    tracker.epoch()

print(tracker.report())

# ============================================================
# MODE 2: context-manager style
# ============================================================

model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

with FLOPpyTracker(
    run_name="torch_test_mode2",
    print_summary=True,
    print_hardware=True,
) as tracker:

    tracker.start(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        export_path="torch_test_mode2.csv",
    )

    model.train()

    for _ in range(num_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            y_hat = model(xb)
            loss = loss_fn(y_hat, yb)
            loss.backward()
            optimizer.step()

            tracker.batch()

        tracker.epoch()

    tracker.stop()

print(tracker.report())
