import torch
import torch.nn as nn
from floppy.floppy_tracker import FLOPpyTracker


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)

    x = torch.randn(64, 20)
    y = torch.randint(0, 4, (64,))

    model = TinyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    tracker = FLOPpyTracker(
        run_name="torch_training_test",
        print_summary=True,
    )

    with tracker.run(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        export_path="torch_training_test.csv",
    ):
        model.train()

        for _ in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    rep = tracker.report

    assert rep.model_flop > 0
    assert rep.optimizer_flop > 0
    assert rep.loss_forward_flop > 0
    assert rep.overall_flop > rep.model_flop
if __name__ == "__main__":
    main()
