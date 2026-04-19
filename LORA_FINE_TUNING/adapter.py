import os
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import warnings
warnings.filterwarnings("ignore")

BASE_MODEL = "microsoft/phi-1_5"
ADAPTER_PATH = os.path.join("models","adapters","lora_adapter")
EVAL_FILE = os.path.join("data","eval_questions.jsonl")

def load_eval_questions():
    questions = []
    with jsonlines.open(EVAL_FILE,'r') as reader:
        for l in reader:
            questions.append(l['question'])
    return questions

def clean_answer(text):
    if "Response:" in text:
        text = text.split("Response:", 1)[1].strip()
    
    for tok in ["\n", "(1)","(2)","(3)","1.","2.","3."]:
        if tok in text:
            text = text.split(tok)[0].strip()
    return text

def chat(model, tokenizer, question):
    prompt = f"Instruction: {question}\nResponse:"
    input = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **input,
            do_sample=False,
            temperature=0.1,
            pad_token_id = tokenizer.eos_token_id,
            max_new_tokens = 50
        )
        
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens = True)
    return clean_answer(decoded)

def main():
    print("\n===== Loading the base model tokenizer =====\n")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n===== Loading the base model LLM =====\n")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map=None,
        torch_dtype=torch.float32 #CPU
    )
    
    print("\n===== Loading the LoRA Adapter =====\n")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    print("\n===== Running Evaluation =====\n")
    questions = load_eval_questions()
    
    for q in questions:
        print(f"Question: {q}\n")
        response = chat(model, tokenizer, q)
        print(f"Model's Response: {response}\n")
        print("-"*60)

if __name__ == "__main__":
    main()