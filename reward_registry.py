"""
Reward function adapters registry.

Defines adapters mapping simple string keys to functions that compute
a {"reward": float, "reward_info": {...}} dict given a sample.
"""

from typing import Dict, Callable, Any
from enum import Enum

from deepscaler_math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from countdown_reward import format_reward_function, answer_reward_function


def _extract_reference_and_answer(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts 'parsed_gt_answer' and 'parsed_attempt' fields into the sample dict
    by splitting sample['sample_text'] on sample['input'].
    """
    original_input = sample['input']
    output = sample['sample_text'].split(original_input)[1]
    # Ground truth answer
    if "\\boxed" in sample.get('answer', ''):
        parsed_gt = extract_answer(sample['answer'])
    else:
        parsed_gt = sample.get('answer')
    # Model attempt
    try:
        parsed_attempt = extract_answer(output)
    except Exception:
        parsed_attempt = ''
    # Annotate sample
    sample['parsed_gt_answer'] = parsed_gt
    sample['parsed_attempt'] = parsed_attempt or ''
    return sample


def mathd_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Grades the sample using mathd (string match) from deepscaler_math_utils.
    Returns a dict with 'reward' (1.0 or 0.0) and 'reward_success'.
    """
    sample = _extract_reference_and_answer(sample)
    correct = grade_answer_mathd(sample['parsed_attempt'], sample['parsed_gt_answer'])
    reward = float(correct)
    return {"reward": reward}


def sympy_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Grades the sample using sympy-based checker from deepscaler_math_utils.
    Returns a dict with 'reward' and 'reward_success'.
    """
    sample = _extract_reference_and_answer(sample)
    correct = grade_answer_sympy(sample['parsed_attempt'], sample['parsed_gt_answer'])
    reward = float(correct)
    return {"reward": reward}


def countdown_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Adapter for the Countdown Tasks reward.
    """
    RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"
    # Isolate model's generated text by splitting off the prompt
    full_text = sample.get('sample_text', '')
    prompt = sample.get('input', '')
    if prompt and prompt in full_text:
        output = full_text.split(prompt, 1)[1]
    else:
        output = full_text
    # Prepend the RESPONSE_PROMPT to reconstruct the opening <think> tag
    response = RESPONSE_PROMPT + output
    format_r = format_reward_function(response)
    answer_r = answer_reward_function(response, numbers=sample.get('nums'), target=sample.get('target'))
    reward = format_r * 0.1 + answer_r
    success = True
    return {"reward": reward, "format_reward": format_r}


class RewardType(str, Enum):
    """Enum of available reward adapter names."""
    MATHD = "mathd"
    SYMPY = "sympy"
    COUNTDOWN = "countdown"


REWARD_ADAPTERS: Dict[RewardType, Callable[..., Dict[str, Any]]] = {
    RewardType.MATHD: mathd_adapter,
    RewardType.SYMPY: sympy_adapter,
    RewardType.COUNTDOWN: countdown_adapter,
}


def get_reward_adapter(name: RewardType) -> Callable[..., Dict[str, Any]]:
    """
    Look up and return the reward adapter function by RewardType.
    Raises ValueError if not found.
    """
    try:
        return REWARD_ADAPTERS[name]
    except KeyError as e:
        raise ValueError(f"Unknown reward adapter: {name}") from e 