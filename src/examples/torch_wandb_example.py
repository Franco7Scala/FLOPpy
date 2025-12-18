import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flop_tracker import FlopTracker


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset 
    x = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 10, (256,))
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = SimpleCNN(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    #  CONFIGURAZIONE WANDB
    WANDB_API_KEY = "70496d4cf370b6213bcccec19098c5167a8f1a9c"  
    WANDB_PROJECT = "flop-tracker-demo"

    # FlopTracker + wandb 
    ft = FlopTracker(run_name="torch_cnn_wandb_silent").torch_bind(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device,
        epochs=3,
        backend="torch",
        log_per_batch=True,           # log per batch su wandb
        log_per_epoch=True,           # log per epoch su wandb
        export_path="torch_cnn_wandb_flop.csv",  # CSV locale opzionale
        use_wandb=True,               # attiva wandb
        wandb_project=WANDB_PROJECT,
        wandb_token=WANDB_API_KEY,    # login tramite API key
    )

    # OPZIONALI, il tracker stampa già i FLOP totali
    print("Raw FLOP:", ft.raw_flop)
    print("Total FLOP:", ft.total_flop)


if __name__ == "__main__":
    main()
