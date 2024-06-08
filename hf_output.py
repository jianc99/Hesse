import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", torch_dtype=torch.float16, device_map="auto")

model.eval()
max_length=128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prompt = "Pittsburgh is a city located in "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate logits
with torch.no_grad():
    outputs = model.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)