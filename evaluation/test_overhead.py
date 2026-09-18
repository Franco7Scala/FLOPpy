import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import warnings

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoProcessor, BlipForConditionalGeneration, ViTForImageClassification
)
import torchvision.models as models
from peft import LoraConfig, get_peft_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from floppy import FLOPpyTracker

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL CONFIGURATION AND UTILITIES
# ==============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_time_in_millis():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return int(round(time.time() * 1000))


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_header(title):
    print("\n" + "=" * 80)
    print(f" ⏱️ OVERHEAD BENCHMARK: {title}")
    print("=" * 80)


if device == "cpu":
    print("⚠️ WARNING: This test suite requires CUDA for some tests (e.g., BitsAndBytes, Multi-GPU).")

# Data structure to accumulate results
benchmark_results = []


def record_result(name, time_no, time_with):
    overhead_ms = time_with - time_no
    overhead_pct = (overhead_ms / time_no) * 100 if time_no > 0 else 0
    benchmark_results.append({
        "Architecture": name,
        "Without Tracker (ms)": time_no,
        "With Tracker (ms)": time_with,
        "Overhead (ms)": overhead_ms,
        "Overhead (%)": f"+{overhead_pct:.2f}%"
    })
    print(f"Result: {time_no}ms vs {time_with}ms -> Overhead: +{overhead_pct:.2f}%")


# ==============================================================================
# 1. CNN (ResNet-18) TRAINING OVERHEAD
# ==============================================================================
print_header("CNN (ResNet-18) Training")
cycles = 50
model_cnn = models.resnet18(weights=None).to(device).train()
opt_cnn = torch.optim.AdamW(model_cnn.parameters(), lr=1e-3)
criterion_cnn = nn.CrossEntropyLoss()
img_cnn = torch.randn(16, 3, 224, 224).to(device)
lbl_cnn = torch.randint(0, 1000, (16,)).to(device)

# Warmup
out = model_cnn(img_cnn)
criterion_cnn(out, lbl_cnn).backward()

# Without Tracker
t0 = get_time_in_millis()
for _ in range(cycles):
    opt_cnn.zero_grad()
    criterion_cnn(model_cnn(img_cnn), lbl_cnn).backward()
    opt_cnn.step()
t_no = get_time_in_millis() - t0

# With Tracker
tracker = FLOPpyTracker(run_name="cnn_over")
tracker.run(model=model_cnn, optimizer=opt_cnn, criterion=criterion_cnn)
t0 = get_time_in_millis()
for _ in range(cycles):
    opt_cnn.zero_grad()
    criterion_cnn(model_cnn(img_cnn), lbl_cnn).backward()
    opt_cnn.step()
t_with = get_time_in_millis() - t0

record_result("CNN (ResNet-18) Training", t_no, t_with)
del model_cnn, tracker
clear_vram()

# ==============================================================================
# 2. AUTOREGRESSIVE DECODER (GPT-2) INFERENCE
# ==============================================================================
print_header("Autoregressive Decoder (GPT-2) Generation")
cycles = 10
model_name_dec = "gpt2"
tokenizer_dec = AutoTokenizer.from_pretrained(model_name_dec)
model_dec = AutoModelForCausalLM.from_pretrained(model_name_dec).to(device).eval()
inputs_dec = tokenizer_dec("The future of green AI is", return_tensors="pt").to(device)

# Warmup
with torch.no_grad():
    model_dec.generate(**inputs_dec, max_new_tokens=5)

t0 = get_time_in_millis()
with torch.no_grad():
    for _ in range(cycles):
        model_dec.generate(**inputs_dec, max_new_tokens=50)
t_no = get_time_in_millis() - t0

tracker = FLOPpyTracker(run_name="dec_over")
tracker.run(model=model_dec, tokenizer=tokenizer_dec)
t0 = get_time_in_millis()
with torch.no_grad():
    for _ in range(cycles):
        model_dec.generate(**inputs_dec, max_new_tokens=50)
t_with = get_time_in_millis() - t0

record_result("Autoregressive Decoder (GPT-2) Gen", t_no, t_with)
del model_dec, tracker
clear_vram()

# ==============================================================================
# 3. ENCODER (BERT) TRAINING
# ==============================================================================
print_header("Encoder (BERT) Training")
cycles = 100
model_name_enc = "bert-base-uncased"
tokenizer_enc = AutoTokenizer.from_pretrained(model_name_enc)
model_enc = AutoModelForSequenceClassification.from_pretrained(model_name_enc, num_labels=2).to(device).train()
opt_enc = torch.optim.AdamW(model_enc.parameters(), lr=2e-5)
inputs_enc = tokenizer_enc(["Green AI is FLOPpy!"] * 8, return_tensors="pt", padding=True, truncation=True).to(device)
inputs_enc["labels"] = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0]).to(device)

# Warmup
model_enc(**inputs_enc).loss.backward()

t0 = get_time_in_millis()
for _ in range(cycles):
    opt_enc.zero_grad()
    model_enc(**inputs_enc).loss.backward()
    opt_enc.step()
t_no = get_time_in_millis() - t0

tracker = FLOPpyTracker(run_name="enc_over")
tracker.run(model=model_enc, optimizer=opt_enc)
t0 = get_time_in_millis()
for _ in range(cycles):
    opt_enc.zero_grad()
    model_enc(**inputs_enc).loss.backward()
    opt_enc.step()
t_with = get_time_in_millis() - t0

record_result("Encoder (BERT) Training", t_no, t_with)
del model_enc, tracker
clear_vram()

# ==============================================================================
# 4. GNN (GCN) TRAINING
# ==============================================================================
print_header("GNN (GCN) Training")
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data


    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(32, 16)
            self.conv2 = GCNConv(16, 3)

        def forward(self, data):
            x = F.relu(self.conv1(data.x, data.edge_index))
            return F.log_softmax(self.conv2(x, data.edge_index), dim=1)


    cycles = 1000
    model_gnn = GCN().to(device).train()
    opt_gnn = torch.optim.Adam(model_gnn.parameters(), lr=0.01)
    data_gnn = Data(
        x=torch.randn((100, 32), dtype=torch.float),
        edge_index=torch.randint(0, 100, (2, 300), dtype=torch.long),
        y=torch.randint(0, 3, (100,), dtype=torch.long)
    ).to(device)

    # Warmup
    F.nll_loss(model_gnn(data_gnn), data_gnn.y).backward()

    t0 = get_time_in_millis()
    for _ in range(cycles):
        opt_gnn.zero_grad()
        F.nll_loss(model_gnn(data_gnn), data_gnn.y).backward()
        opt_gnn.step()
    t_no = get_time_in_millis() - t0

    tracker = FLOPpyTracker(run_name="gnn_over")
    tracker.run(model=model_gnn, optimizer=opt_gnn)
    t0 = get_time_in_millis()
    for _ in range(cycles):
        opt_gnn.zero_grad()
        F.nll_loss(model_gnn(data_gnn), data_gnn.y).backward()
        opt_gnn.step()
    t_with = get_time_in_millis() - t0

    record_result("GNN (GCN) Training", t_no, t_with)
    del model_gnn, tracker
except ImportError:
    print("torch_geometric is not installed. Skipping GNN test.")
clear_vram()

# ==============================================================================
# 5. MIXTURE OF EXPERTS (MoE) + LoRA TRAINING
# ==============================================================================
print_header("MoE (Qwen Safetensors) + LoRA Training")
cycles = 30
model_name_moe = "Qwen/Qwen1.5-MoE-A2.7B"
try:
    base_moe = AutoModelForCausalLM.from_pretrained(model_name_moe, torch_dtype="auto", use_safetensors=True).to(device)
    lora_cfg_moe = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
    model_moe = get_peft_model(base_moe, lora_cfg_moe).train()
    opt_moe = torch.optim.AdamW(model_moe.parameters(), lr=1e-4)
    inputs_moe = AutoTokenizer.from_pretrained(model_name_moe)("Sustainable AI.", return_tensors="pt").to(device)
    inputs_moe["labels"] = inputs_moe["input_ids"].clone()

    # Warmup
    model_moe(**inputs_moe).loss.backward()

    t0 = get_time_in_millis()
    for _ in range(cycles):
        opt_moe.zero_grad()
        model_moe(**inputs_moe).loss.backward()
        opt_moe.step()
    t_no = get_time_in_millis() - t0

    tracker = FLOPpyTracker(run_name="moe_over")
    tracker.run(model=model_moe, optimizer=opt_moe)
    t0 = get_time_in_millis()
    for _ in range(cycles):
        opt_moe.zero_grad()
        model_moe(**inputs_moe).loss.backward()
        opt_moe.step()
    t_with = get_time_in_millis() - t0

    record_result("MoE + LoRA Training", t_no, t_with)
    del model_moe, base_moe, tracker
except Exception as e:
    print(f"Skipping MoE due to memory/loading limits: {e}")
clear_vram()

# ==============================================================================
# 6. MULTIMODAL (BLIP) INFERENCE
# ==============================================================================
print_header("Multimodal (BLIP) Inference")
cycles = 10
model_name_blip = "Salesforce/blip-image-captioning-base"
proc_blip = AutoProcessor.from_pretrained(model_name_blip, use_safetensors=True)
model_blip = BlipForConditionalGeneration.from_pretrained(model_name_blip, use_safetensors=True).to(device).eval()
inputs_blip = {
    "pixel_values": torch.randn(1, 3, 384, 384).to(device),
    "input_ids": proc_blip.tokenizer("A picture of", return_tensors="pt").input_ids.to(device)
}

# Warmup
with torch.no_grad():
    model_blip.generate(**inputs_blip, max_new_tokens=5)

t0 = get_time_in_millis()
with torch.no_grad():
    for _ in range(cycles):
        model_blip.generate(**inputs_blip, max_new_tokens=20)
t_no = get_time_in_millis() - t0

tracker = FLOPpyTracker(run_name="mult_over")
tracker.run(model=model_blip)
t0 = get_time_in_millis()
with torch.no_grad():
    for _ in range(cycles):
        model_blip.generate(**inputs_blip, max_new_tokens=20)
t_with = get_time_in_millis() - t0

record_result("Multimodal (BLIP) Inference", t_no, t_with)
del model_blip, tracker
clear_vram()

# ==============================================================================
# 7. RNN (LSTM) SEQUENCE TRAINING
# ==============================================================================
print_header("RNN (LSTM) Sequence Training")


class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


cycles = 1000
model_rnn = SimpleRNN().to(device).train()
opt_rnn = torch.optim.Adam(model_rnn.parameters(), lr=1e-3)
in_rnn = torch.randn(1, 20, 10).to(device)
lbl_rnn = torch.tensor([1]).to(device)

# Warmup
F.cross_entropy(model_rnn(in_rnn), lbl_rnn).backward()

t0 = get_time_in_millis()
for _ in range(cycles):
    opt_rnn.zero_grad()
    F.cross_entropy(model_rnn(in_rnn), lbl_rnn).backward()
    opt_rnn.step()
t_no = get_time_in_millis() - t0

tracker = FLOPpyTracker(run_name="rnn_over")
tracker.run(model=model_rnn, optimizer=opt_rnn)
t0 = get_time_in_millis()
for _ in range(cycles):
    opt_rnn.zero_grad()
    F.cross_entropy(model_rnn(in_rnn), lbl_rnn).backward()
    opt_rnn.step()
t_with = get_time_in_millis() - t0

record_result("RNN (LSTM) Training", t_no, t_with)
del model_rnn, tracker
clear_vram()

# ==============================================================================
# 8. SCIKIT-LEARN (Random Forest) INFERENCE & TRAINING
# ==============================================================================
print_header("Scikit-Learn (Random Forest) Overhead")
X_sk, y_sk = make_classification(n_samples=5000, n_features=100, n_classes=2, random_state=42)
cycles_fit, cycles_pred = 5, 50

model_sk_no = RandomForestClassifier(n_estimators=50, random_state=42)
t0 = get_time_in_millis()
for _ in range(cycles_fit): model_sk_no.fit(X_sk, y_sk)
for _ in range(cycles_pred): model_sk_no.predict(X_sk)
t_no = get_time_in_millis() - t0

model_sk_with = RandomForestClassifier(n_estimators=50, random_state=42)
tracker = FLOPpyTracker(run_name="sk_over")
tracker.run(model=model_sk_with)
t0 = get_time_in_millis()
for _ in range(cycles_fit): model_sk_with.fit(X_sk, y_sk)
for _ in range(cycles_pred): model_sk_with.predict(X_sk)
t_with = get_time_in_millis() - t0

record_result("Scikit-Learn (Random Forest)", t_no, t_with)
del model_sk_no, model_sk_with, tracker

# ==============================================================================
# 9. ViT + LoRA TRAINING OVERHEAD
# ==============================================================================
print_header("ViT + LoRA Training")
cycles = 100
model_name_vit = "google/vit-base-patch16-224"
base_vit = ViTForImageClassification.from_pretrained(model_name_vit).to(device)

target_modules = set()
for name, module in base_vit.named_modules():
    if isinstance(module, torch.nn.Linear) and "classifier" not in name:
        target_modules.add(name.split(".")[-1])

lora_cfg_vit = LoraConfig(r=16, target_modules=list(target_modules))
model_vit = get_peft_model(base_vit, lora_cfg_vit).train()
opt_vit = torch.optim.AdamW(model_vit.parameters(), lr=1e-4)
inputs_vit = {"pixel_values": torch.randn(1, 3, 224, 224).to(device), "labels": torch.tensor([1]).to(device)}

# Warmup
model_vit(**inputs_vit).loss.backward()

t0 = get_time_in_millis()
for _ in range(cycles):
    opt_vit.zero_grad()
    model_vit(**inputs_vit).loss.backward()
    opt_vit.step()
t_no = get_time_in_millis() - t0

tracker = FLOPpyTracker(run_name="vit_over")
tracker.run(model=model_vit, optimizer=opt_vit)
t0 = get_time_in_millis()
for _ in range(cycles):
    opt_vit.zero_grad()
    model_vit(**inputs_vit).loss.backward()
    opt_vit.step()
t_with = get_time_in_millis() - t0

record_result("ViT + LoRA Training", t_no, t_with)
del model_vit, base_vit, tracker
clear_vram()

# ==============================================================================
# SUMMARY MARKDOWN TABLE OUTPUT
# ==============================================================================
print("\n" + "=" * 115)
print(" 📊 OVERHEAD BENCHMARK RESULTS (MARKDOWN FORMAT)")
print("=" * 115)

w1 = 36  # Architecture
w2 = 20  # Without Tracker (ms)
w3 = 17  # With Tracker (ms)
w4 = 13  # Overhead (ms)
w5 = 12  # Overhead (%)

print(f"| {'Architecture':<{w1}} | {'Without Tracker (ms)':>{w2}} | {'With Tracker (ms)':>{w3}} | {'Overhead (ms)':>{w4}} | {'Overhead (%)':>{w5}} |")

print(f"|{'-' * (w1 + 2)}|{'-' * (w2 + 2)}|{'-' * (w3 + 2)}|{'-' * (w4 + 2)}|{'-' * (w5 + 2)}|")

for res in benchmark_results:
    print(f"| {res['Architecture']:<{w1}} | {res['Without Tracker (ms)']:>{w2}} | {res['With Tracker (ms)']:>{w3}} | {res['Overhead (ms)']:>{w4}} | {res['Overhead (%)']:>{w5}} |")

print("=" * 115)
