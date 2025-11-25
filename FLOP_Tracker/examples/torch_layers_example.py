import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flop_tracker import FlopTracker

class ComplexNet(nn.Module):
    """
    Modello PyTorch "dimostrativo" che include molti layer diversi:
    - Padding (ZeroPad2d)
    - Convoluzioni (Conv2d)
    - Pooling (MaxPool2d, AdaptiveAvgPool2d)
    - Normalizzazione (BatchNorm2d, LayerNorm)
    - Dropout (Dropout2d, Dropout)
    - Vision / pixel shuffle (PixelShuffle)
    - Trasformazioni di shape (Flatten, view/permute)
    - Transformer Encoder layer (Multi-Head Attention + FFN)
    - LSTM ricorrente
    - Fully connected finale (Linear)
    """

    def __init__(self, num_classes: int = 10, img_size: int = 32):
        super().__init__()

       
        self.pad = nn.ZeroPad2d(1)  

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
        )

        self.pre_shuffle = nn.Conv2d(32, 32 * 4, kernel_size=1)  
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)   

        self.pool2 = nn.AdaptiveAvgPool2d((8, 8))

        conv_out_dim = 32 * 8 * 8
        self.flatten = nn.Flatten()
        self.fc_to_seq = nn.Linear(conv_out_dim, 128)  # 128 = L * d_model

        self.d_model = 32
        self.seq_len = 4  # 4 * 32 = 128
        self.layernorm = nn.LayerNorm(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,  # (batch, seq, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(p=0.2)

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        
        # 1) CNN + padding + pooling + pixel shuffle
        x = self.pad(x)               
        x = self.conv_block1(x)       
        x = self.conv_block2(x)       

        x = self.pre_shuffle(x)        
        x = self.pixel_shuffle(x)      
        x = self.pool2(x)              

        # 2) Flatten + Linear 
        x = self.flatten(x)            
        x = self.fc_to_seq(x)         

        # 3) Interpretiamo il vettore come sequenza (B, L, d_model)
        x = x.view(-1, self.seq_len, self.d_model) 
        x = self.layernorm(x)        
        x = self.transformer_encoder(x)  

        # 4) LSTM sulla sequenza
        x, (h_n, c_n) = self.lstm(x)  
        # Prendiamo lo stato nascosto finale
        h_last = h_n[-1]              

        h_last = self.dropout(h_last)

        # 5) Classificatore finale
        logits = self.classifier(h_last)  
        return logits


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset: immagini 32x32 RGB, 10 classi
    x = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 10, (256,))
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = ComplexNet(num_classes=10)
    # Supporto opzionale DataParallel 
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    ft = FlopTracker(run_name="torch_complex_model").torch_bind(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device,
        epochs=2,
        backend="torch",
        log_per_batch=True,
        log_per_epoch=True,
        export_path="torch_complex_model_flop.csv",
        use_wandb=False,
    )


if __name__ == "__main__":
    main()
