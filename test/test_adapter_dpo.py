import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2"

print("\n" + "=" * 50)
print(" FLOPpyTracker: DPO + LoRA PEFT Alignment Test")
print("=" * 50)

# 1. Setup Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 2. SETUP MODELS (Policy with LoRA vs Reference Frozen)
# ============================================================
print("\n--- Preparing Policy (LoRA) and Reference (Frozen) ---")

# A. Reference Model: Base model, strictly frozen
ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# B. Policy Model: Base model + LoRA Adapters
base_policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    target_modules=["c_attn"]
)
policy_model = get_peft_model(base_policy_model, lora_config)
policy_model.train()

# Print trainable parameters to verify PEFT status
policy_model.print_trainable_parameters()

# Optimizer (only for the Policy's adapters)
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-4)


# ============================================================
# 3. DPO LOSS & DUMMY DATA
# ============================================================
class DPOLoss(torch.nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, p_chosen, p_rejected, r_chosen, r_rejected):
        p_chosen_logps = F.log_softmax(p_chosen, dim=-1).mean()
        p_rejected_logps = F.log_softmax(p_rejected, dim=-1).mean()
        r_chosen_logps = F.log_softmax(r_chosen, dim=-1).mean()
        r_rejected_logps = F.log_softmax(r_rejected, dim=-1).mean()

        logits_p = p_chosen_logps - p_rejected_logps
        logits_r = r_chosen_logps - r_rejected_logps
        return -F.logsigmoid(self.beta * (logits_p - logits_r))


dpo_loss_fn = DPOLoss()

prompt = "The future of AI efficiency is"
c_res = " based on precise FLOP counting."
r_res = " just about buying more GPUs."

c_in = tokenizer(prompt + c_res, return_tensors="pt").to(device)
r_in = tokenizer(prompt + r_res, return_tensors="pt").to(device)

# ============================================================
# 4. RUN TRACKERS
# ============================================================
print("\n--- Running DPO+LoRA Step ---")

tracker_ref = FLOPpyTracker(run_name="dpo_ref_base")
tracker_ref.run(model=ref_model)

tracker_policy = FLOPpyTracker(run_name="dpo_policy_lora")
tracker_policy.run(model=policy_model, optimizer=optimizer, loss_fn=dpo_loss_fn)

optimizer.zero_grad()

# Reference Forward (Frozen)
with torch.no_grad():
    ref_c_logits = ref_model(**c_in).logits
    ref_r_logits = ref_model(**r_in).logits

# Policy Forward (With Adapters)
pol_c_logits = policy_model(**c_in).logits
pol_r_logits = policy_model(**r_in).logits

# Loss & Backward (Only Policy triggers the escape hatch)
loss = dpo_loss_fn(pol_c_logits, pol_r_logits, ref_c_logits, ref_r_logits)
loss.backward()

# Optimizer Step
optimizer.step()

# ============================================================
# 5. FINAL COMPARISON REPORT
# ============================================================
rep_ref = tracker_ref.report()
rep_pol = tracker_policy.report()

print("\n" + "=" * 50)
print(" 📊 FLOPpy DPO + LoRA FINAL REPORT")
print("=" * 50)
print(f"REFERENCE (Base Model): {rep_ref.overall_flop / 1e6:.2f} MFLOPs (Forward Only)")
print(f"POLICY (LoRA Model)   : {rep_pol.overall_flop / 1e6:.2f} MFLOPs (Full Loop)")
print("-" * 50)
print(f"Forward (Policy)      : {rep_pol.model_forward_flop / 1e6:.2f} MFLOPs")
print(f"Backward (Policy)     : {rep_pol.model_backward_flop / 1e6:.2f} MFLOPs")
print(f"Optimizer (Policy)    : {rep_pol.optimizer_flop / 1e6:.2f} MFLOPs")
print("-" * 50)
print(f"🔥 TOTAL DPO STEP COST : {(rep_ref.overall_flop + rep_pol.overall_flop) / 1e6:.2f} MFLOPs")
