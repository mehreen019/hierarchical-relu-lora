# eval/negative_transfer.py

import torch
import re


def _generate(model, tokenizer, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_python_accuracy(model, tokenizer, py_eval_examples):
    """
    Proxy metric: does the model generate a code block containing 'return'?
    Replace with actual HumanEval pass@1 if compute allows.
    """
    model.eval()
    correct = 0
    for ex in py_eval_examples:
        prompt = str(ex.get("instruction", ex.get("prompt", "")))[:300]
        generated = _generate(model, tokenizer, f"Write Python code:\n{prompt}\n\n```python\n")
        # Simple but consistent proxy
        has_return = "return" in generated
        has_code = len(generated.strip()) > 15
        if has_return and has_code:
            correct += 1
    model.train()
    return correct / len(py_eval_examples) if py_eval_examples else 0.0


def eval_medical_accuracy(model, tokenizer, med_eval_examples):
    """
    PubMedQA yes/no/maybe accuracy.
    """
    model.eval()
    correct = 0
    for ex in med_eval_examples:
        ctx = ex["context"]["contexts"][0][:200] if ex["context"]["contexts"] else ""
        prompt = f"Question: {ex['question']}\nContext: {ctx}\nAnswer (yes/no/maybe):"
        true_label = ex["final_decision"].lower().strip()
        generated = _generate(model, tokenizer, prompt, max_new_tokens=5).lower()

        pred = None
        for label in ["yes", "no", "maybe"]:
            if label in generated:
                pred = label
                break
        if pred == true_label:
            correct += 1
    model.train()
    return correct / len(med_eval_examples) if med_eval_examples else 0.0


def compute_negative_transfer(run_a_acc, run_b_acc, run_c_acc):
    return {
        "ceiling": run_a_acc,
        "NT_run_b": run_a_acc - run_b_acc,
        "NT_run_c": run_a_acc - run_c_acc
    }
