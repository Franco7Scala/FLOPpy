import torch
import torch.nn as nn
from floppy import FLOPpyTracker

# ------------------------------------------------------------
# Shared setup
# ------------------------------------------------------------

input_data = torch.randn(32, 10)
labels = torch.randint(0, 10, (32,))

# ============================================================
# MODE 1
# ============================================================

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
)

model.to("cuda" if torch.cuda.is_available() else "cpu")
input_data = input_data.to("cuda" if torch.cuda.is_available() else "cpu")
labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

tracker = FLOPpyTracker(
    run_name="pytorch_experiment",
    print_summary=True,
    print_hardware=True,
)

tracker.run(model=model, optimizer=optimizer, loss_fn=loss_fn)

y_hat = model(input_data)
loss = loss_fn(y_hat, labels)
loss.backward()
optimizer.step()

print(tracker.report())

# ============================================================
# MODE 2: context manager style
# ============================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

with FLOPpyTracker(
    run_name="pytorch_experiment_with",
    print_summary=True,
    print_hardware=True,
) as tracker:

    tracker.start(model=model, optimizer=optimizer, loss_fn=loss_fn)

    y_hat = model(input_data)
    loss = loss_fn(y_hat, labels)
    loss.backward()
    optimizer.step()

    tracker.stop()

print(tracker.report())
