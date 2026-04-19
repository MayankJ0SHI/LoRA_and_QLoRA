# LoRA Fine-Tuning Project (Phi-1.5)

This project demonstrates **parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)** on the `microsoft/phi-1_5` model for a credit-card recommendation use case.

---

# 📁 Project Structure

```
LORA_FINE_TUNING/
│── data/
│   ├── train.jsonl
│   ├── eval_questions.jsonl
│
│── models/
│   ├── adapters/
│   │   ├── lora_adapter/   # Saved LoRA weights
│
│── myenv/                  # Virtual environment
│
│── adapter.py             # Run inference using LoRA adapter
│── baseline_chat.py       # Base model inference (no LoRA)
│── finetune.py            # Training script (LoRA fine-tuning)
│── requirements.txt
│── README.md
```

---

# 🚀 Project Overview

This project fine-tunes a lightweight LLM using LoRA to improve domain-specific responses (credit card recommendations).

### Base Model

* `microsoft/phi-1_5`

### Fine-tuning Method

* LoRA (PEFT - Parameter Efficient Fine Tuning)
* CPU-based training supported

---

# 🧠 Features

* Instruction-based fine-tuning dataset
* LoRA adapter training (efficient, low compute)
* CPU-compatible training setup
* Evaluation script for testing responses
* Comparison between base model vs fine-tuned model

---

# 📊 Dataset Format

## Training Data (`train.jsonl`)

```json
{
  "instruction": "Recommend a credit card for Amazon spending",
  "output": "Amazon Pay ICICI Credit Card is best for Amazon purchases."
}
```

## Evaluation Data (`eval_questions.jsonl`)

```json
{
  "question": "Best credit card for fuel expenses in India"
}
```

---

# ⚙️ Setup Instructions

## 1. Create Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate   # Linux/Mac
myenv\Scripts\activate    # Windows
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️‍♂️ Training (LoRA Fine-Tuning)

Run training:

```bash
python finetune.py
```

### What happens during training:

* Loads `microsoft/phi-1_5`
* Applies LoRA adapters to attention layers
* Trains on instruction-output pairs
* Saves adapter weights in:

```
models/adapters/lora_adapter/
```

---

# 💬 Running Inference (Fine-tuned Model)

```bash
python adapter.py
```

### Flow:

* Loads base model
* Loads LoRA adapter
* Runs evaluation questions
* Prints generated responses

---

# 🆚 Baseline Comparison

To compare with base model (no fine-tuning):

```bash
python baseline_chat.py
```

---

# ⚠️ Known Observations

* Model may overfit to frequently seen patterns (e.g., repeated credit card recommendations)
* Responses may lack diversity if dataset is small
* LoRA performance depends heavily on dataset quality

---

# 🔧 Key Hyperparameters

| Parameter     | Value             |
| ------------- | ----------------- |
| Model         | microsoft/phi-1_5 |
| LoRA rank (r) | 8                 |
| LoRA alpha    | 128               |
| Batch size    | 1                 |
| Epochs        | 10                |
| Learning rate | 2e-4              |
| Max length    | 128               |

---

# 📌 LoRA Configuration

* Target modules: `q_proj`, `k_proj`, `v_proj`, `dense`
* Dropout: 0.05
* Task type: Causal Language Modeling

---

# 📈 Improvements (Recommended)

To improve model quality:

* Add more diverse training examples
* Reduce repeated card bias in dataset
* Include multi-option reasoning answers
* Add uncertainty-based responses
* Try QLoRA for better efficiency

---

# 🧪 Example Output

**Input:**

```
Recommend a credit card for Amazon spending
```

**Output:**

```
Amazon Pay ICICI Credit Card is ideal for Amazon purchases due to cashback benefits and rewards.
```

---

# 👨‍💻 Author Notes

This project demonstrates practical implementation of:

* PEFT (LoRA)
* HuggingFace Transformers
* Instruction tuning
* Lightweight LLM fine-tuning on CPU

---

# 📜 License

For educational purposes only.
