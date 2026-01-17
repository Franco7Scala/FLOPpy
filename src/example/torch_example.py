import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flop_tracker import FlopTracker
from trainers import train_torch


class ManyLayersNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.PReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LayerNorm([64, 8, 8]),
            nn.Tanh(),
        )

        # RMSNorm (se disponibile)
        self.rms = nn.RMSNorm(64) if hasattr(nn, "RMSNorm") else None
        self.softmax2d = nn.Softmax2d()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block(x)                 # conv/norm/act/pool
        x = self.softmax2d(x)             # softmax family
        if self.rms is not None:
            # RMSNorm lavora su last dim -> reshape in (N*H*W, C)
            N, C, H, W = x.shape
            z = x.permute(0, 2, 3, 1).contiguous().view(N * H * W, C)
            z = self.rms(z)
            x = z.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        return self.head(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 10, (256,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    model = ManyLayersNet(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    ft = FlopTracker(run_name="torch_many_layers_facade", print_summary=True).run(
        model=model,
        backend="torch",
        train_fn=train_torch,
        train_kwargs=dict(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=loader,
            device=device,
            epochs=2,
        ),
        log_per_batch=True,
        log_per_epoch=True,
        export_path="torch_many_layers_facade.csv",
        use_wandb=False,
    )

    print("REPORT:", ft.report)


if __name__ == "__main__":
    main()
