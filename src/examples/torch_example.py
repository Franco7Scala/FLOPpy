import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from floppy import FLOPpyTracker
from floppy.utils.wandb_configuration import WandbConfiguration

# ------------------------------------------------------------
# Shared setup
# ------------------------------------------------------------

torch.manual_seed(42)

X = torch.randn(64, 10)
y = torch.randint(0, 3, (64,))

X = X.to("cuda" if torch.cuda.is_available() else "cpu")
y = y.to("cuda" if torch.cuda.is_available() else "cpu")

wandb_config = WandbConfiguration(project_name="torch_test_mode1",group_name="eDPO", reporter_key="your_wandb_key_here")

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

num_epochs = 15

# ============================================================
# MODE 1
# ============================================================

model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)

model.to("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

tracker = FLOPpyTracker(
    run_name="torch_test_mode1"
)

tracker.run(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    export_path="torch_test_mode1.csv",
    wandb_config=wandb_config
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

model.to("cuda" if torch.cuda.is_available() else "cpu")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

with FLOPpyTracker(
    run_name="torch_test_mode2",
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

report = tracker.report()
print(report)
