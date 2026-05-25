🏥 Medical LLM Fine-Tuning using Unsloth + LLaMA 2



This project demonstrates fine-tuning a large language model (LLaMA 2 7B) using Unsloth + LoRA + 4-bit quantization on a custom medical Q\&A dataset.



🚀 Features

Fine-tuned LLaMA 2 7B using LoRA (efficient training)

4-bit quantization for low GPU memory usage

Custom medical Q\&A dataset

Instruction-following format training

Fast inference on Tesla T4 GPU

🧠 Tech Stack

Python

PyTorch

HuggingFace Transformers

Unsloth

TRL (SFTTrainer)

LoRA (PEFT)

📊 Dataset



Custom medical dataset including:



Hypertension

Diabetes

Heart failure

Pneumonia

Statins, insulin, ACE inhibitors, etc.



Format:



{

&#x20; "instruction": "What is hypertension?",

&#x20; "output": "Hypertension is a chronic condition..."

}

⚙️ Training Details

Base Model: LLaMA 2 7B (4-bit)

Method: LoRA fine-tuning

Epochs: 3

Batch Size: 2

Learning Rate: 2e-4

Hardware: NVIDIA Tesla T4 (Google Colab)

🧪 Example Output



Input:



What is hypertension?



Output:



Hypertension is a chronic condition where blood pressure is consistently elevated...



🚀 How to Run

pip install unsloth transformers datasets trl



Run training:



Arch\_Medical\_Finetuning.ipynb



Run inference:



python inference.py

📌 Results

Successfully fine-tuned LLM using low VRAM (4-bit)

Model responds to medical questions in instruction format

Efficient training using LoRA adapters

📷 Screenshots



(Add Colab training + inference screenshots here)

