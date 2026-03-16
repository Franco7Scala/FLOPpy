import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy.floppy_tracker import FLOPpyTracker


model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The future of artificial intelligence"

tracker = FLOPpyTracker(
    run_name="hf_generate_example",
    print_summary=True
)

with tracker.run(
    model=model,
    export_path="hf_generate_example.csv"
):

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=20
        )

print(tokenizer.decode(generated[0]))
