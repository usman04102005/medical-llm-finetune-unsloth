# 🏥 Medical LLM Fine-Tuning using Unsloth + LLaMA 2

This project demonstrates fine-tuning a Large Language Model (LLaMA 2 7B) using **Unsloth**, **LoRA**, and **4-bit quantization** on a custom medical Q&A dataset.

The project was built and trained using Google Colab with a Tesla T4 GPU for efficient low-memory fine-tuning.

---

# 🚀 Features

- Fine-tuned LLaMA 2 7B model
- LoRA (Low-Rank Adaptation) training
- 4-bit quantization for memory efficiency
- Custom medical instruction dataset
- Fast inference using Unsloth
- Hugging Face + TRL integration

---

# 🧠 Technologies Used

- Python
- PyTorch
- Unsloth
- Hugging Face Transformers
- TRL (SFTTrainer)
- PEFT / LoRA
- Google Colab

---

# 📂 Project Structure

```bash
medical-llm-finetune-unsloth/
│
├── Arch_Medical_Finetuning.ipynb
├── medical_data.json
├── training_config.json
├── inference.py
├── requirements.txt
├── README.md
├── .gitignore
└── medical_model_lora/
```

---

# 📊 Dataset

The model was trained on a custom medical Q&A dataset containing topics such as:

- Hypertension
- Diabetes
- Pneumonia
- Heart Failure
- Atrial Fibrillation
- ACE Inhibitors
- Cortisol
- Renal Function Tests
- Statins
- Insulin Regulation

Dataset format:

```json
{
  "instruction": "What is hypertension?",
  "input": "",
  "output": "Hypertension is a chronic condition..."
}
```

---

# ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Base Model | LLaMA 2 7B |
| Quantization | 4-bit |
| Fine-Tuning Method | LoRA |
| Epochs | 3 |
| Learning Rate | 2e-4 |
| Batch Size | 2 |
| GPU | Tesla T4 |
| Framework | Unsloth |

---

# 🚀 How to Run

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Open Notebook

Run the training notebook:

```bash
Arch_Medical_Finetuning.ipynb
```

---

## 3️⃣ Run Inference

```bash
python inference.py
```

---

# 🧪 Example Output

## Input

```text
What is hypertension?
```

## Output

```text
Hypertension is a chronic condition where blood pressure is consistently elevated. It increases the risk of heart disease, stroke, and kidney disease.
```

---

# 📈 Results

- Successfully fine-tuned a 7B parameter LLM on medical instructions
- Reduced VRAM usage using 4-bit quantization + LoRA
- Achieved efficient inference on consumer-grade GPUs

---

# 📷 Screenshots
## Training process
<img width="1914" height="575" alt="Screenshot 2026-05-25 222041" src="https://github.com/user-attachments/assets/c62f2f67-7d1e-4f63-8c93-b3ab6a05b16c" />
## GPU utilization
<img width="660" height="328" alt="Screenshot 2026-05-25 222543" src="https://github.com/user-attachments/assets/d681508c-1d20-42cd-8d31-eeb0d81de323" />
## Inference results
<img width="1687" height="499" alt="Screenshot 2026-05-25 223214" src="https://github.com/user-attachments/assets/9269f967-5966-405b-9a60-6d2a125d9401" />

---

# ⚠️ Disclaimer

This project is for educational and research purposes only.  
It is **not intended for real-world medical diagnosis or treatment**.

---

# 👨‍💻 Author

Usman Chughtai

---

# ⭐ Future Improvements

- Larger medical dataset
- Better response formatting
- Streamlit / Web UI deployment
- Hugging Face deployment
- Conversational memory support
- Retrieval-Augmented Generation (RAG)

```
