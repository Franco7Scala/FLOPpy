import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flop_tracker import FlopTracker
from trainers import train_torch


class ManyLayersNet(nn.Module):
    def __init__(self, vocab_size=500, emb_dim=64, num_classes=10):
        super().__init__()

        # conv stack
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.PReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(),
        )

        # token branch
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, 64, batch_first=True)
        self.lstm_cell = nn.LSTMCell(64, 64)
        self.gru_cell = nn.GRUCell(64, 64)

        # attention + transformer
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=False)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=False)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.Tanh(),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=-1),   # log-softmax (per NLLLoss, opzionale)
        )

    def forward(self, x_img, x_tok):
        # img path
        z = self.conv(x_img)
        out = self.fc(z)  # (B, C)

        # token path
        e = self.emb(x_tok)           # (B, T, E)
        r, _ = self.rnn(e)            # (B, T, 64)

        # cells step (usiamo solo un time-step per esercitare i cell)
        h0 = r[:, 0, :]               # (B, 64)
        h1, c1 = self.lstm_cell(h0)   # (B, 64)
        h2 = self.gru_cell(h1)        # (B, 64)

        # MHA expects (L, N, E)
        q = h2.unsqueeze(0)           # (1, B, 64)
        attn_out, _ = self.mha(q, q, q)  # (1, B, 64)
        enc_out = self.encoder(attn_out) # (1, B, 64)

        out = out + enc_out.squeeze(0)[:, : out.shape[1]]
        return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fake dataset
    B = 256
    x_img = torch.randn(B, 3, 32, 32)
    x_tok = torch.randint(0, 500, (B, 12))
    y = torch.randint(0, 10, (B,))

    ds = TensorDataset(x_img, x_tok, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = ManyLayersNet(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_fn = nn.NLLLoss()

    def train_fn_wrapped(*, model, optimizer, loss_fn, train_loader, device=None, epochs=1, observers=None):
        observers = observers or []
        if device is not None:
            model.to(device)

        for obs in observers:
            obs.on_train_start({"backend": "torch"})

        for epoch in range(epochs):
            for obs in observers:
                obs.on_epoch_start(epoch)

            for batch_idx, (img, tok, yb) in enumerate(train_loader):
                if device is not None:
                    img, tok, yb = img.to(device), tok.to(device), yb.to(device)

                from observers import BatchContext
                bc = BatchContext(epoch=epoch, batch_idx=batch_idx, batch_size=int(img.shape[0]))

                for obs in observers:
                    obs.on_batch_start(bc)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(img, tok)

                for obs in observers:
                    obs.on_after_forward(bc, outputs)

                loss = loss_fn(outputs, yb
                for obs in observers:
                    obs.on_after_loss(bc, loss, outputs, yb)

                loss.backward()
                for obs in observers:
                    obs.on_after_backward(bc)

                optimizer.step()
                for obs in observers:
                    obs.on_after_step(bc)

                for obs in observers:
                    obs.on_batch_end(bc)

            for obs in observers:
                obs.on_epoch_end(epoch)

        for obs in observers:
            obs.on_train_end({"backend": "torch"})

    FlopTracker(run_name="torch_many_layers_observer", print_summary=True, print_hardware=True).run(
        model=model,
        backend="torch",
        train_fn=train_fn_wrapped,
        train_kwargs={
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "train_loader": loader,
            "device": device,
            "epochs": 2,
        },
        log_per_batch=True,
        log_per_epoch=True,
        export_path="torch_many_layers_flop.csv",
        use_wandb=False,
    )


if __name__ == "__main__":
    main()
