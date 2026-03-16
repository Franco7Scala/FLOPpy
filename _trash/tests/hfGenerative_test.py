import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy.floppy_tracker import FLOPpyTracker
from floppy.utils.tokenizer_ops import wrap_tokenizer


def main():
    model_name = "distilgpt2"

    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tracker = FLOPpyTracker(
        run_name="hf_generate_test",
        print_summary=True,
    )

    with tracker.run(
        model=model,
        export_path="hf_generate_test.csv",
    ):
        tokenizer = wrap_tokenizer(base_tokenizer, tracker=tracker._tracker)
        prompt = "The future of artificial intelligence"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10)

    rep = tracker.report

    assert rep.model_flop > 0


if __name__ == "__main__":
    main()
