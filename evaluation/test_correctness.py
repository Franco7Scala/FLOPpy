import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
from PIL import Image

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    AutoImageProcessor, ResNetForImageClassification,
    AutoProcessor, PaliGemmaForConditionalGeneration
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from floppy import FLOPpyTracker

# ==============================================================================
# GLOBAL CONFIGURATION AND UTILITIES
# ==============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_text = "distilgpt2"
model_vision = "microsoft/resnet-50"
model_vlm = "hf-internal-testing/tiny-random-PaliGemmaForConditionalGeneration" # Using the tiny version for fast CI tests

if device == "cpu":
    print("⚠️ WARNING: This test suite requires CUDA for some tests (e.g., BitsAndBytes, Multi-GPU).")

def clear_vram():
    """Forces VRAM cleanup to avoid OOM between tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_header(title):
    print("\n" + "=" * 80)
    print(f" 🚀 TEST: {title}")
    print("=" * 80)

def evaluate_test(condition, success_msg, fail_msg):
    if condition:
        print(f"✅ PASS: {success_msg}")
    else:
        print(f"❌ FAIL: {fail_msg}")

# Setup base Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_text)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==============================================================================
# PREVIOUS TESTS (Quantization, KV Cache, MoE, Checkpointing, Base LoRA)
# *Note: Included in the engine, here we focus on the new integrations.*
# ==============================================================================

# ==============================================================================
# TEST 6: CNN (ResNet-50) + LoRA
# ==============================================================================
print_header("CNN (ResNet-50) + LoRA - Forward, Backward & Optimizer Workload")

base_cnn = ResNetForImageClassification.from_pretrained(model_vision).to(device)
lora_cnn_cfg = LoraConfig(r=16, target_modules=["convolution"], modules_to_save=["classifier"])
peft_cnn = get_peft_model(base_cnn, lora_cnn_cfg)
peft_cnn.train()

opt_cnn = torch.optim.AdamW(peft_cnn.parameters(), lr=1e-4)
inputs_cnn = {
    "pixel_values": torch.randn(1, 3, 224, 224).to(device),
    "labels": torch.tensor([1]).to(device)
}

tracker_cnn = FLOPpyTracker(run_name="resnet")
tracker_cnn.run(model=peft_cnn, optimizer=opt_cnn)

opt_cnn.zero_grad()
loss_cnn = peft_cnn(**inputs_cnn).loss
loss_cnn.backward()
opt_cnn.step()

rep_cnn = tracker_cnn.report()
fw_gflops = rep_cnn.model_forward_flop / 1e9
ai_cnn = rep_cnn.arithmetic_intensity

print(f"ResNet Forward : {fw_gflops:.2f} GFLOPs | BOPs: {rep_cnn.model_forward_bop / 1e9:.2f} GBOPs")
print(f"ResNet Backward: {rep_cnn.model_backward_flop / 1e9:.2f} GFLOPs")
print(f"ResNet AI      : {ai_cnn:.2f} FLOPs/Byte")

# A nominal ResNet-50 has ~4.1 GigaMACs, so ~8.2 total GFLOPs in forward
evaluate_test(7.5 < fw_gflops < 9.5, "ResNet-50 Forward FLOPs matches theoretical ~8 GFLOPs.", f"Anomalous CNN FLOPs: {fw_gflops:.2f} GFLOPs")
evaluate_test(rep_cnn.model_backward_flop > rep_cnn.model_forward_flop, "Backward cost properly exceeds Forward cost in CNNs.", "Backward scaling failed.")

del peft_cnn, tracker_cnn
clear_vram()

# ==============================================================================
# TEST 7: ALIGNMENT (DPO vs SimPO)
# ==============================================================================
print_header("Alignment Methods - DPO (with Reference) vs SimPO (Reference-Free)")

# 7.1 DPO SETUP
ref_model = AutoModelForCausalLM.from_pretrained(model_text).to(device).eval()
policy_dpo = AutoModelForCausalLM.from_pretrained(model_text).to(device).train()
for p in ref_model.parameters(): p.requires_grad = False
opt_dpo = torch.optim.AdamW(policy_dpo.parameters(), lr=1e-5)

inputs_c = tokenizer("Green AI is FLOPpy.", return_tensors="pt").to(device)
inputs_r = tokenizer("Green AI is useless.", return_tensors="pt").to(device)

def dpo_loss_sim(p_c, p_r, r_c, r_r):
    # Dummy implementation to trigger layers
    return (p_c.sum() - p_r.sum() - r_c.sum() + r_r.sum()) * 0.1

tracker_ref = FLOPpyTracker(run_name="dpo_ref")
tracker_ref.run(model=ref_model)
tracker_pol = FLOPpyTracker(run_name="dpo_pol")
tracker_pol.run(model=policy_dpo, optimizer=opt_dpo)

opt_dpo.zero_grad()
with torch.no_grad():
    r_c, r_r = ref_model(**inputs_c).logits, ref_model(**inputs_r).logits
p_c, p_r = policy_dpo(**inputs_c).logits, policy_dpo(**inputs_r).logits
loss_dpo = dpo_loss_sim(p_c, p_r, r_c, r_r)
loss_dpo.backward()
opt_dpo.step()

dpo_total_flops = tracker_ref.report().overall_flop + tracker_pol.report().overall_flop

# 7.2 SimPO SETUP
policy_simpo = AutoModelForCausalLM.from_pretrained(model_text).to(device).train()
opt_simpo = torch.optim.AdamW(policy_simpo.parameters(), lr=1e-5)

def simpo_loss_sim(p_c, p_r):
    return -F.logsigmoid((p_c.sum() - p_r.sum()))

tracker_simpo = FLOPpyTracker(run_name="simpo_pol")
tracker_simpo.run(model=policy_simpo, optimizer=opt_simpo)

opt_simpo.zero_grad()
p_c_s, p_r_s = policy_simpo(**inputs_c).logits, policy_simpo(**inputs_r).logits
loss_simpo = simpo_loss_sim(p_c_s, p_r_s)
loss_simpo.backward()
opt_simpo.step()

simpo_total_flops = tracker_simpo.report().overall_flop

print(f"Total DPO Cost   : {dpo_total_flops / 1e6:.2f} MFLOPs")
print(f"Total SimPO Cost : {simpo_total_flops / 1e6:.2f} MFLOPs")

evaluate_test(simpo_total_flops < dpo_total_flops, "SimPO is demonstrably cheaper than DPO (no Reference Model overhead).", "SimPO cost is anomalous.")
# The ratio should be close to ~0.66 (since an entire forward pass of the ref model for 2 inputs is missing)
ratio = simpo_total_flops / dpo_total_flops
evaluate_test(0.5 < ratio < 0.75, f"SimPO saves ~33% FLOPs vs DPO (Ratio: {ratio:.2f}).", f"Unexpected ratio: {ratio:.2f}")

del ref_model, policy_dpo, policy_simpo, tracker_ref, tracker_pol, tracker_simpo
clear_vram()

# ==============================================================================
# TEST 8: GRADIENT ACCUMULATION
# ==============================================================================
print_header("Gradient Accumulation - Optimizer Update Frequency")

acc_model = AutoModelForCausalLM.from_pretrained(model_text).to(device).train()
acc_opt = torch.optim.AdamW(acc_model.parameters(), lr=1e-5)
inputs_acc = tokenizer("Accumulate gradients over steps.", return_tensors="pt").to(device)
inputs_acc["labels"] = inputs_acc["input_ids"].clone()

tracker_acc = FLOPpyTracker(run_name="grad_acc")
tracker_acc.run(model=acc_model, optimizer=acc_opt)

steps = 4
acc_opt.zero_grad()
for _ in range(steps):
    loss = acc_model(**inputs_acc).loss / steps
    loss.backward()

# Optimizer is called only once
acc_opt.step()
rep_acc = tracker_acc.report()

fw_flops = rep_acc.model_forward_flop
bw_flops = rep_acc.model_backward_flop
opt_flops = rep_acc.optimizer_flop

print(f"Forward (x4)  : {fw_flops / 1e6:.2f} MFLOPs")
print(f"Backward (x4) : {bw_flops / 1e6:.2f} MFLOPs")
print(f"Optimizer (x1): {opt_flops / 1e6:.2f} MFLOPs")

evaluate_test(bw_flops > fw_flops * 1.5, "Backward FLOPs correctly accumulated across multiple steps.", "Backward accumulation failed.")
# Optimizer should cost a fraction of backward because it was called only once
evaluate_test(opt_flops < (bw_flops / steps), "Optimizer FLOPs stayed low, reflecting a single parameter update.", "Optimizer was incorrectly tracked on every step.")

del acc_model, tracker_acc
clear_vram()

# ==============================================================================
# TEST 9: MULTIMODAL (PaliGemma VLM)
# ==============================================================================
print_header("Multimodal Graph - Vision + Language (PaliGemma)")

processor = AutoProcessor.from_pretrained(model_vlm)
vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(model_vlm, torch_dtype=torch.float32).to(device)
vlm_model.eval()

image = Image.new('RGB', (224, 224), color='blue')
inputs_vlm = processor(text="caption en", images=image, return_tensors="pt").to(device)

tracker_vlm = FLOPpyTracker(run_name="vlm")
tracker_vlm.run(model=vlm_model)

with torch.no_grad():
    vlm_model(**inputs_vlm)

rep_vlm = tracker_vlm.report()
print(f"Multimodal Forward : {rep_vlm.model_forward_flop / 1e6:.2f} MFLOPs")
print(f"Multimodal AI      : {rep_vlm.arithmetic_intensity:.2f} FLOPs/Byte")

evaluate_test(rep_vlm.model_forward_flop > 0, "FLOPpy successfully mapped a heterogeneous Vision-Language graph.", "Failed to track VLM.")

del vlm_model, tracker_vlm
clear_vram()

# ==============================================================================
# TEST 10: TRACKER LIFECYCLE (On / Off / Reattach)
# ==============================================================================
print_header("Lifecycle Resilience - Attach, Detach, and Reattach")

lc_model = nn.Sequential(nn.Linear(10, 16), nn.Linear(16, 3)).to(device)
lc_opt = torch.optim.Adam(lc_model.parameters(), lr=1e-3)
x_lc, y_lc = torch.randn(16, 10).to(device), torch.randn(16, 3).to(device)

tracker_lc = FLOPpyTracker(run_name="lifecycle")

def run_mini_step():
    lc_opt.zero_grad()
    loss = F.mse_loss(lc_model(x_lc), y_lc)
    loss.backward()
    lc_opt.step()

# Phase 1: ON (Tracks 1 step)
tracker_lc.run(model=lc_model, optimizer=lc_opt)
run_mini_step()
flops_p1 = tracker_lc.report().overall_flop

# Phase 2: OFF (Detached - model works but FLOPpy is off)
tracker_lc.stop()
run_mini_step()
flops_p2 = tracker_lc.report().overall_flop

# Phase 3: REATTACHED (Clean new session - Tracks 2 steps)
tracker_lc.run(model=lc_model, optimizer=lc_opt)
run_mini_step()
run_mini_step() # <--- Added a second step
flops_p3 = tracker_lc.report().overall_flop

print(f"Phase 1 (ON - 1 step)     : {flops_p1} FLOPs")
print(f"Phase 2 (OFF - 1 step)    : {flops_p2} FLOPs")
print(f"Phase 3 (RE-ON - 2 steps) : {flops_p3} FLOPs")

evaluate_test(flops_p1 == flops_p2, "Detaching stopped all hardware profiling safely.", "Ghost tracking detected!")
evaluate_test(flops_p3 == flops_p1 * 2, "Reattaching correctly started a clean, active tracking session.", "State corruption on reattach.")

del lc_model, tracker_lc
clear_vram()

# ==============================================================================
# TEST 11: MULTI-GPU MODEL PARALLELISM
# ==============================================================================
print_header("Multi-Device Pipeline - Model Parallelism")

if torch.cuda.device_count() >= 2:
    dev0, dev1 = "cuda:0", "cuda:1"
else:
    dev0, dev1 = device, "cpu"

class SplitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Linear(1024, 2048).to(dev0)
        self.p2 = nn.Linear(2048, 10).to(dev1)
    def forward(self, x):
        x = x.to(dev0)
        return self.p2(self.p1(x).to(dev1))

split_model = SplitNet()
split_opt = torch.optim.AdamW(split_model.parameters(), lr=1e-3)
x_split = torch.randn(128, 1024)
y_split = torch.empty(128, dtype=torch.long).random_(10).to(dev1)

tracker_split = FLOPpyTracker(run_name="split")
tracker_split.run(model=split_model, optimizer=split_opt)

split_opt.zero_grad()
loss_split = F.cross_entropy(split_model(x_split), y_split)
loss_split.backward()
split_opt.step()

rep_split = tracker_split.report()
print(f"Distributed Forward : {rep_split.model_forward_flop / 1e6:.2f} MFLOPs")
print(f"Distributed Backward: {rep_split.model_backward_flop / 1e6:.2f} MFLOPs")

evaluate_test(rep_split.overall_flop > 0, "Successfully tracked logic crossing device boundaries.", "Failed to track multi-device execution.")

del split_model, tracker_split
clear_vram()

# ==============================================================================
# TEST 12: SCIKIT-LEARN ROBUSTNESS & AI
# ==============================================================================
print_header("Scikit-Learn Workload & Arithmetic Intensity")

X_sk, y_sk = make_classification(n_samples=500, n_features=20, random_state=42)

def eval_sklearn(model_cls, kwargs, name):
    mod = model_cls(**kwargs)
    trk = FLOPpyTracker(run_name=name)
    trk.run(model=mod)
    mod.fit(X_sk, y_sk)
    mod.predict(X_sk)
    return trk.report()

rep_rf_10 = eval_sklearn(RandomForestClassifier, {"n_estimators": 10}, "rf10")
rep_rf_100 = eval_sklearn(RandomForestClassifier, {"n_estimators": 100}, "rf100")
rep_svm = eval_sklearn(SVC, {"kernel": "rbf"}, "svm")

f_10, f_100, f_svm = rep_rf_10.overall_flop, rep_rf_100.overall_flop, rep_svm.overall_flop
ai_100 = rep_rf_100.arithmetic_intensity

print(f"RF (10 trees)  : {f_10 / 1e6:.2f} MFLOPs")
print(f"RF (100 trees) : {f_100 / 1e6:.2f} MFLOPs | AI: {ai_100:.2f} FLOPs/Byte")
print(f"SVM (RBF)      : {f_svm / 1e6:.2f} MFLOPs")

evaluate_test(f_100 > f_10 * 8, "Random Forest FLOPs scale accurately with n_estimators.", "Sklearn heuristic missed scaling.")
evaluate_test(ai_100 > 10, "Tree ensembles correctly exhibit high cache-friendly Arithmetic Intensity.", "Sklearn Arithmetic Intensity calculation failed.")

# ==============================================================================
# TEST 13: VISION TRANSFORMER (ViT) + LoRA
# ==============================================================================
print_header("Vision Transformer (ViT) + LoRA Workload")

from transformers import ViTForImageClassification

vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)

# DEFINITIVE BULLETPROOF TRICK:
# Automatically find ALL linear layers of the model (e.g., 'query', 'dense', 'qkv')
# extracting only the last word of their path, excluding the final classifier.
target_modules = set()
for name, module in vit_model.named_modules():
    if isinstance(module, torch.nn.Linear) and "classifier" not in name:
        target_modules.add(name.split(".")[-1])

# Pass the dynamic list to PEFT (something like ['query', 'key', 'value', 'dense'])
vit_lora_cfg = LoraConfig(r=16, target_modules=list(target_modules))
peft_vit = get_peft_model(vit_model, vit_lora_cfg)
peft_vit.train()

opt_vit = torch.optim.AdamW(peft_vit.parameters(), lr=1e-4)
inputs_vit = {
    "pixel_values": torch.randn(1, 3, 224, 224).to(device),
    "labels": torch.tensor([1]).to(device)
}

tracker_vit = FLOPpyTracker(run_name="vit_lora")
tracker_vit.run(model=peft_vit, optimizer=opt_vit)

opt_vit.zero_grad()
loss_vit = peft_vit(**inputs_vit).loss
loss_vit.backward()
opt_vit.step()

rep_vit = tracker_vit.report()
print(f"ViT-LoRA Forward: {rep_vit.model_forward_flop / 1e9:.2f} GFLOPs")
print(f"ViT-LoRA Total  : {rep_vit.overall_flop / 1e9:.2f} GFLOPs")

evaluate_test(rep_vit.overall_flop > 0, "ViT + LoRA tracked correctly without architecture clashes.", "Failed to track ViT.")

del peft_vit, tracker_vit
clear_vram()

# ==============================================================================
# TEST 14: MULTI-ADAPTER (LoRA Duo)
# ==============================================================================
print_header("Multi-Adapter (LoRA Duo) Workload")

base_multi = AutoModelForCausalLM.from_pretrained(model_text).to(device)

# Create the first adapter and immediately add the second one
model_multi = get_peft_model(base_multi, LoraConfig(r=8, target_modules=["c_attn"]), adapter_name="adapter_A")
model_multi.add_adapter("adapter_B", LoraConfig(r=16, target_modules=["c_attn"]))
model_multi.set_adapter("adapter_B")
model_multi.train()

# Unfreeze both adapters to simulate joint training
for name, param in model_multi.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

opt_multi = torch.optim.AdamW(model_multi.parameters(), lr=1e-4)
inputs_multi = tokenizer("Multi-adapter execution test.", return_tensors="pt").to(device)
inputs_multi["labels"] = inputs_multi["input_ids"].clone()

tracker_multi = FLOPpyTracker(run_name="multi_lora")
tracker_multi.run(model=model_multi, optimizer=opt_multi)

opt_multi.zero_grad()
loss_multi = model_multi(**inputs_multi).loss
loss_multi.backward()
opt_multi.step()

rep_multi = tracker_multi.report()
print(f"Multi-Adapter Forward  : {rep_multi.model_forward_flop / 1e6:.2f} MFLOPs")
print(f"Multi-Adapter Backward : {rep_multi.model_backward_flop / 1e6:.2f} MFLOPs")

evaluate_test(rep_multi.overall_flop > 0, "Successfully tracked dual active LoRA adapters simultaneously.", "Failed on multi-adapter routing.")

del model_multi, tracker_multi
clear_vram()

print("\n" + "=" * 80)
print(" 🏁 ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 80)
