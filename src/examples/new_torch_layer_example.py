import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flop_tracker import FlopTracker


class NewLayersNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
        )

        # Softmax2d lavora bene su (N,C,H,W)
        self.softmax2d = nn.Softmax2d()

        # RMSNorm
        self.use_rms = hasattr(nn, "RMSNorm")
        if self.use_rms:
            # RMSNorm normalizza sull'ultima dimensione -> applichiamo dopo un reshape
            self.rms = nn.RMSNorm(32)

        # RNNCell/LSTMCell/GRUCell: usiamo feature vettoriali
        self.rnn_cell = nn.RNNCell(input_size=32, hidden_size=32, nonlinearity="tanh")
        self.lstm_cell = nn.LSTMCell(input_size=32, hidden_size=32)
        self.gru_cell = nn.GRUCell(input_size=32, hidden_size=32)

        # TransformerEncoderLayer (container): i FLOP arrivano dai suoi Linear/MHA/Norm
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.LogSoftmax(dim=-1),  # test LogSoftmax
        )

    def forward(self, x):
        x = self.conv(x)               # (B,32,16,16)
        x = self.softmax2d(x)          # softmax2d

        # riduco a vettori (B, 16*16, 32)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # (B,256,32)

        if self.use_rms:
            x = self.rms(x)            # RMSNorm su ultima dim

        # Transformer encoder su sequenza (B,256,32)
        x = self.encoder(x)            # container -> leaf modules già hookati

        # prendiamo un singolo vettore (B,32) per testare le Cells
        v = x[:, 0, :]                 # (B,32)

        # inizializzo hidden states a zero 
        h0 = torch.zeros(B, 32, device=v.device)
        h1 = self.rnn_cell(v, h0)

        h2, c2 = self.lstm_cell(v, (h0, h0))
        h3 = self.gru_cell(v, h0)

        # mix non-linearities per testare Sigmoid/Tanh
        v = torch.tanh(h1 + h2 + h3)
        v = torch.sigmoid(v)

        out = self.head(v)             # LogSoftmax
        return out


def main():
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Dataset fittizio
    x = torch.randn(128, 3, 32, 32)
    y = torch.randint(0, 10, (128,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)

    model = NewLayersNet(num_classes=10)
    model.to(device)

    # test DataParallel
    if "cuda" in device:
        model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Loss: CrossEntropyLoss (coerente con LogSoftmax)
    loss_fn = nn.NLLLoss()  

    ft = FlopTracker(run_name="torch_new_layers_test").torch_bind(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        device=device,
        epochs=1,
        log_per_batch=True,
        log_per_epoch=True,
        export_path="torch_new_layers_test.csv",
        use_wandb=False,
    )

    print("Raw FLOP:", ft.raw_flop)
    print("Total FLOP:", ft.total_flop)


if __name__ == "__main__":
    main()
