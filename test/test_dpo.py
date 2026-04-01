import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2"

print("\n" + "=" * 50)
print(" FLOPpyTracker: DPO Alignment Test (Policy vs Ref)")
print("=" * 50)

# Load the Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 1. SETUP MODELS & OPTIMIZER
# ============================================================
print("\n--- Loading Policy and Reference Models ---")
# Policy Model (The one we are training)
policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
policy_model.train()

# Reference Model (Frozen, used for KL penalty baseline)
ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Optimizer for the Policy Model
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

# ============================================================
# 2. DUMMY DPO DATA (Chosen vs Rejected)
# ============================================================
prompt = "The future of green AI is"
chosen_response = " highly dependent on efficient profiling tools like FLOPpy."
rejected_response = " not something we should worry about right now."

# Tokenize inputs
chosen_inputs = tokenizer(prompt + chosen_response, return_tensors="pt").to(device)
rejected_inputs = tokenizer(prompt + rejected_response, return_tensors="pt").to(device)


# ============================================================
# 3. DUMMY DPO LOSS FUNCTION (Ora è un vero nn.Module!)
# ============================================================
class DPOLoss(torch.nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, policy_chosen, policy_rejected, ref_chosen, ref_rejected):
        policy_chosen_logps = F.log_softmax(policy_chosen, dim=-1).mean()
        policy_rejected_logps = F.log_softmax(policy_rejected, dim=-1).mean()

        ref_chosen_logps = F.log_softmax(ref_chosen, dim=-1).mean()
        ref_rejected_logps = F.log_softmax(ref_rejected, dim=-1).mean()

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        return -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))


# Istanziamo la loss
dpo_loss_fn = DPOLoss(beta=0.1)

# ============================================================
# 4. FLOPpy INTEGRATION & TRAINING STEP
# ============================================================
print("\n--- Running DPO Step with Dual FLOPpyTrackers ---")

tracker_ref = FLOPpyTracker(run_name="dpo_reference")
tracker_ref.run(model=ref_model)

tracker_policy = FLOPpyTracker(run_name="dpo_policy")
# Passiamo la classe DPO Loss al tracker
tracker_policy.run(model=policy_model, optimizer=optimizer, loss_fn=dpo_loss_fn)

optimizer.zero_grad()

# A. Forward Pass: Reference Model
with torch.no_grad():
    ref_chosen_logits = ref_model(**chosen_inputs).logits
    ref_rejected_logits = ref_model(**rejected_inputs).logits

# B. Forward Pass: Policy Model
policy_chosen_logits = policy_model(**chosen_inputs).logits
policy_rejected_logits = policy_model(**rejected_inputs).logits

# C. Loss Computation
# TRASPARENZA TOTALE: Usiamo la loss originaria, FLOPpy fa il resto in background!
loss = dpo_loss_fn(
    policy_chosen_logits, policy_rejected_logits,
    ref_chosen_logits, ref_rejected_logits
)

# D. Backward Pass
loss.backward()

# E. Optimizer Step
optimizer.step()

# ============================================================
# 5. FINAL REPORT AGGREGATION
# ============================================================
report_ref = tracker_ref.report()
report_policy = tracker_policy.report()

# Calcolo del costo reale totale dell'allineamento DPO
total_dpo_flops = report_ref.overall_flop + report_policy.overall_flop
total_dpo_bops = report_ref.overall_bop + report_policy.overall_bop

print(f"\n[ Reference Model Cost ]: {report_ref.overall_flop / 1e6:.2f} MFLOPs (Forward Only)")
print(f"[ Policy Model Cost    ]: {report_policy.overall_flop / 1e6:.2f} MFLOPs (Full Loop)")
print("-" * 50)
print(f"🔥 TOTAL DPO STEP COST  : {total_dpo_flops / 1e6:.2f} MFLOPs")
