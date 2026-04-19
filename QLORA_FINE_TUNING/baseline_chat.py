import os
import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
EVAL_FILE = os.path.join(DATA_DIR, "eval_questions.jsonl")
BASE_MODEL = "microsoft/phi-1_5" #CPU_FRIENDLY

def load_eval_questions():
    questions = []
    with jsonlines.open(EVAL_FILE, 'r') as reader:
        for obj in reader:
            questions.append(obj["question"])
    return questions

def chat(model, tokenizer, question):
    prompt = f"""### Instruction:
{question}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens = True)
     
def run_baseline_evaluation():
    print("\n===== Loading base model =====\n");
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    
    questions = load_eval_questions()
    
    print("\n===== Baseline Reponse (Before Finetuning) =====\n\n")
    for q in questions:
        print(f"Question: {q}\n")
        answer = chat(model, tokenizer, q)
        print(f"Answer: {answer}\n")
        print("-"*60)

if __name__=='__main__':
    run_baseline_evaluation()