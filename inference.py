
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-2-7b-bnb-4bit",
    load_in_4bit=True,
)

model.load_adapter("medical_model_lora")
FastLanguageModel.for_inference(model)

question = input("Ask: ")

prompt = f"""### Instruction:
{question}

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
