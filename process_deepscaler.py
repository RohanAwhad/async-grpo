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
# model_path = "Qwen/Qwen2.5-1.5B-Instruct"
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_path = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, token= "***REMOVED***")
tokenizer.add_bos_token = False

def make_initial_prompt(sample):
    system_prompt = r"""Let's think step by step and output the final answer within \boxed{}. """
    # system_prompt = ""
    messages = [
        {"role": "user", "content": system_prompt + sample['problem']},
    ]
    sample['messages'] = messages
    sample['input'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # sample['input'] = messages[0]['content']
    sample['input_token_ids'] = tokenizer.encode(sample['input'])
    if tokenizer.decode(sample['input_token_ids']) != sample['input']:
        decoded = tokenizer.decode(sample['input_token_ids'])
        print(sample['input'])
        for i in range(len(decoded)):
            if decoded[i] != sample['input'][i]:
                print(f"{i}: {decoded[i]} {sample['input'][i]}")
        sample['error'] = True
    else:
        sample['error'] = False
    return sample

data = data.map(make_initial_prompt, num_proc=16)
data = data.filter(lambda x: not x['error'])

data.to_json("deepscaler_qwen1.5b_r1_distill.jsonl", lines=True)

print(data[1]['input'])
print(tokenizer.decode(data[1]['input_token_ids']))
print(tokenizer.decode(data[1]['input_token_ids']) == data[1]['input'])
