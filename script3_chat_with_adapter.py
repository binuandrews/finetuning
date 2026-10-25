"""
This script loads:

- The same base model (microsoft/phi-1_5)
- The LoRA adapter you trained (models/adapters/lora_acme)
- Runs inference on your evaluation questions
- Shows updated / correct answers

This is the “after LoRA” demonstration.
"""

import os
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import warnings
warnings.filterwarnings('ignore')

BASE_MODEL = "microsoft/phi-1_5"
ADAPTER_PATH = "models/adapters/lora_adap"
EVAL_FILE = "data/eval_questions.jsonl"


def load_eval_questions():
    questions = []
    with jsonlines.open(EVAL_FILE, "r") as reader:
        for obj in reader:
            questions.append(obj["question"])
    return questions


def clean_answer(text):
    """
    Removes echo and hallucinated filler.
    Keeps only the first clean answer segment.
    """

    # Remove the Instruction/Response prompt echo
    if "Response:" in text:
        text = text.split("Response:", 1)[1].strip()

    # Cut at first newline or enumeration markers
    for tok in ["\n", "(1)", "(2)", "(3)", "1.", "2.", "3."]:
        if tok in text:
            text = text.split(tok)[0].strip()

    # One clean line
    return text.strip()


def chat(model, tokenizer, question):
    # MUST MATCH TRAINING FORMAT
    prompt = f"Instruction: {question}\nResponse:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return clean_answer(decoded)


def main():
    print("\n=== Loading base model... ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print("\n=== Loading LoRA adapter (kept active, not merged)... ===")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    '''
    model.eval() puts the model into “evaluation mode.”
    This tells PyTorch to turn off certain layers/features used only during training (like dropout and batch normalization), making the model’s behavior consistent and deterministic for inference (prediction).
    '''

    print("\n=== Running Evaluation ===\n")
    questions = load_eval_questions()

    for q in questions:
        print(f"\nQuestion: {q}")
        ans = chat(model, tokenizer, q)
        print(f"Model Answer: {ans}")
        print("-" * 80)


if __name__ == "__main__":
    main()
