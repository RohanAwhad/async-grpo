##
from datasets import load_dataset

data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

##
from transformers import AutoTokenizer
model_path = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def make_initial_prompt(sample):
    user_text = USER_TEMPLATE.format(numbers=sample['nums'], target=sample['target'])
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_text},
    ]
    sample['messages'] = messages
    input_ = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ += RESPONSE_PROMPT
    sample['input'] = input_
    sample['input_token_ids'] = tokenizer.encode(sample['input'], add_special_tokens=False)
    sample['answer'] = sample['target']
    sample['end_token'] = tokenizer.eos_token
    return sample

data = data.map(make_initial_prompt, num_proc=16)
# from IPython import embed; embed()
data.to_json("count_down_tasks_3to4.jsonl", lines=True)
