from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto")

input_text = "Hello, "
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print("Generating...")
outputs = model.generate(
    **inputs,
    max_new_tokens=5,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True
)

# Process and print top-k probabilities for each step
print("\nTop-k Token Probabilities:")
for i, score in enumerate(outputs.scores):
    probs = torch.nn.functional.softmax(score, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, 5)

    print(f"\nStep {i+1}:")
    for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
        token = tokenizer.decode(idx)
        print(f"  {token:<10} : {prob:.4f}")

generated_text = tokenizer.decode(
    outputs.sequences[0], skip_special_tokens=True)
print(f"\nFinal Output:\n{generated_text}")
