import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import warnings

warnings.filterwarnings(("ignore"))

DATA_DIR = "data"
EVAL_FILE = os.path.join(DATA_DIR, "eval_questions.jsonl")

BASE_MODEL = "microsoft/phi-1_5"

def load_eval_questions():
  questions = []
  with jsonlines.open(EVAL_FILE, "r") as reader:
    for obj in reader:
      questions.append(obj["question"])
    return questions

def chat(model, tokenizer, question):
  prompt = f"Questions: {question} \nAnswer"
  input = tokenizer(prompt, return_tensors="pt")

  with torch.no_grad():
    output_ids = model.generate(
        **input,
        max_new_tokens = 80,
        do_sample = False,
        temperature = 0.2,
        pad_token_id = tokenizer.eos_token_id
    )

  decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  return decoded


def run_baselin_evaluation():
  print(f"\nLoading base model: {BASE_MODEL}... CPU May take some time.....")
  tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
  model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
  questions = load_eval_questions()
  for q in questions:
    print(f"\nQuestion to the model: -> {q}")
    answer = chat(model, tokenizer, q)
    print(f"Answer from the model: -> {answer}")
    print("-" * 80)

if __name__ == "__main__":
  run_baselin_evaluation()



