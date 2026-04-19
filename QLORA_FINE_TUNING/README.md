# ⚡ QLoRA Fine-Tuning Project (Phi-1.5)

This project demonstrates **memory-efficient fine-tuning using QLoRA (Quantized Low-Rank Adaptation)** on the `microsoft/phi-1_5` model for a **credit-card recommendation system**.

QLoRA enables training large language models on **low-resource hardware (CPU/GPU with limited VRAM)** by combining:

* 4-bit quantization
* LoRA adapters
* Efficient training pipelines

---

# 📁 Project Structure

```
QLORA_FINE_TUNING/
│── data/
│   ├── train.jsonl
│   ├── eval_questions.jsonl
│
│── models/
│   ├── adapters/
│   │   ├── qlora_adapter/    # Saved QLoRA weights
│
│── myenv/                   # Virtual environment
│
│── adapter.py              # Inference using QLoRA adapter
│── baseline_chat.py        # Base model inference (no fine-tuning)
│── finetune.py             # QLoRA training script
│── requirements.txt
│── README.md
```

---

# 🚀 Project Overview

This project fine-tunes a lightweight LLM using **QLoRA**, making it significantly more **memory-efficient** than standard LoRA while maintaining strong performance.

### 🔹 Base Model

* `microsoft/phi-1_5`

### 🔹 Fine-tuning Method

* QLoRA (4-bit quantization + LoRA)
* Built using:

  * HuggingFace Transformers
  * PEFT
  * BitsAndBytes

---

# 🧠 Key Features

* 4-bit quantized model loading (low memory usage)
* LoRA adapters for efficient training
* Works on limited GPU / CPU (slow on CPU)
* Instruction-tuning dataset
* Base vs Fine-tuned comparison
* Modular training + inference scripts

---

# 📊 Dataset Format

## Training Data (`train.jsonl`)

```json
{
  "instruction": "Recommend a credit card for travel rewards",
  "output": "HDFC Regalia Credit Card is ideal for travel rewards and lounge access."
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
myenv\\Scripts\\activate      # Windows
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️‍♂️ Training (QLoRA Fine-Tuning)

```bash
python finetune.py
```

### What happens during training:

* Loads model in 4-bit quantized mode
* Applies LoRA adapters
* Freezes base model weights
* Trains only adapter layers
* Saves adapters to:

```
models/adapters/qlora_adapter/
```

---

# 💬 Running Inference

```bash
python adapter.py
```

### Flow:

* Loads quantized base model
* Loads QLoRA adapter
* Runs evaluation questions
* Prints responses

---

# 🆚 Baseline Comparison

```bash
python baseline_chat.py
```

---

# ⚠️ Known Observations

* Quantization may slightly reduce precision
* Small dataset → repetitive outputs
* CPU training is slow

---

# 🔧 Key Hyperparameters

| Parameter     | Value             |
| ------------- | ----------------- |
| Model         | microsoft/phi-1_5 |
| Quantization  | 4-bit (nf4)       |
| LoRA rank (r) | 8                 |
| LoRA alpha    | 16 / 32           |
| Batch size    | 1                 |
| Epochs        | 5–10              |
| Learning rate | 2e-4              |
| Max length    | 128               |

---

# 📌 QLoRA Configuration

* Quantization type: nf4
* Compute dtype: float16 / bfloat16
* Target modules: q_proj, k_proj, v_proj, dense
* Dropout: 0.05
* Task: Causal Language Modeling

---

# 🧠 How QLoRA Works

QLoRA combines:

1. Quantization (reduces model size)
2. LoRA adapters (trainable layers)
3. Frozen base model

Result: Efficient training with minimal memory usage.

---

# 📈 Improvements

* Add more diverse dataset
* Reduce repetitive outputs
* Improve prompt design
* Tune LoRA parameters

---

# 🧪 Example Output

**Input:**

```
Recommend a credit card for travel
```

**Output:**

```
HDFC Regalia Credit Card is a strong option due to travel rewards and lounge access.
```

---

# 👨‍💻 Author Notes

This project demonstrates:

* QLoRA implementation
* Efficient LLM fine-tuning
* Domain adaptation

---

# 📜 License

For educational purposes only.
