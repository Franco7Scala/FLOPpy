import torch
import torch.nn as nn
from floppy import FLOPpyTracker

# Verifica che ci siano almeno 2 GPU
num_gpus = torch.cuda.device_count()
if num_gpus < 2:
    print(f"⚠️ Attenzione: hai solo {num_gpus} GPU disponibili.")
    print("Il codice userà la CPU per il secondo device per permetterti di testarlo comunque.")
    device0 = torch.device("cuda:0" if num_gpus > 0 else "cpu")
    device1 = torch.device("cpu")
else:
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

print("\n" + "=" * 50)
print(f" FLOPpyTracker: Multi-GPU Model Parallelism Test")
print(f" Device 0: {device0}")
print(f" Device 1: {device1}")
print("=" * 50)


# ============================================================
# 1. SETUP DEL MODELLO DISTRIBUITO
# ============================================================
class SplitNet(nn.Module):
    def __init__(self):
        super(SplitNet, self).__init__()

        # Prima metà della rete: piazzata sul device 0
        self.part1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU()
        ).to(device0)

        # Seconda metà della rete: piazzata sul device 1
        self.part2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),  # Output 10 classi
        ).to(device1)

    def forward(self, x):
        # Assicuriamoci che l'input sia sul primo device
        x = x.to(device0)

        # Calcolo sulla prima GPU
        out = self.part1(x)

        # Trasferimento del tensore intermedio sulla seconda GPU
        out = out.to(device1)

        # Calcolo sulla seconda GPU
        out = self.part2(out)

        return out


print("\n--- Inizializzazione Rete e Ottimizzatore ---")
model = SplitNet()
model.train()

# L'ottimizzatore raccoglie i parametri da entrambi i device
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ============================================================
# 2. DUMMY DATA & LOSS
# ============================================================
# Batch di 128 elementi
inputs = torch.randn(128, 1024)
labels = torch.empty(128, dtype=torch.long).random_(10).to(device1)  # Le label vanno sul device 1 (dove finisce la rete)

criterion = nn.CrossEntropyLoss()

# ============================================================
# 3. FLOPpy INTEGRATION & TRAINING STEP
# ============================================================
print("\n--- Esecuzione Training Step Multi-GPU ---")

tracker = FLOPpyTracker(run_name="multi_gpu_test")
tracker.run(model=model, optimizer=optimizer)

optimizer.zero_grad()

# A. Forward Pass (attraverserà device0 -> device1)
outputs = model(inputs)

# B. Calcolo della Loss (sul device1)
loss = criterion(outputs, labels)

# C. Backward Pass (attraverserà device1 -> device0)
loss.backward()

# D. Optimizer Step (aggiornerà i pesi su entrambi i device)
optimizer.step()

# ============================================================
# 4. FINAL REPORT
# ============================================================
report = tracker.report()

print("\n" + "=" * 50)
print(" 📊 FLOPpy MULTI-GPU REPORT")
print("=" * 50)
print(f"Forward FLOPs  : {report.model_forward_flop / 1e6:.2f} MFLOPs")
print(f"Backward FLOPs : {report.model_backward_flop / 1e6:.2f} MFLOPs")
print(f"Optimizer FLOPs: {report.optimizer_flop / 1e6:.2f} MFLOPs")
print("-" * 50)
print(f"🔥 TOTAL COST  : {report.overall_flop / 1e6:.2f} MFLOPs")
