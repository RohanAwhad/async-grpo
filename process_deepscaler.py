##
from datasets import load_dataset, Dataset
import json

with open("deepscaler.json", "r") as f:
    data = json.load(f)
error_indices = [6303, 6328, 6336, 6408, 6415, 38209, 38211]

data = [d for i, d in enumerate(data) if i not in error_indices]

data = Dataset.from_list(data)

##
from transformers import AutoTokenizer
model_path = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def make_initial_prompt(sample):
    system_prompt = r"""Let's think step by step and output the final answer within \boxed{}."""
    messages = [
        {"role": "user", "content": system_prompt + sample['problem']},
    ]
    sample['messages'] = messages
    sample['input'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sample['input'] = sample['input'] + 'Let me solve this step by step.\n'
    sample['input_token_ids'] = tokenizer.encode(sample['input'])
    return sample

data = data.map(make_initial_prompt, num_proc=16)

data.to_json("deepscaler_initial_prompt.jsonl", lines=True)

data