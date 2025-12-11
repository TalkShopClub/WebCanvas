import json
from typing import Union, List, Dict, Optional

import tiktoken
from .token_utils import is_model_supported

def calculation_of_token(
    messages: Union[str, List[Dict]], 
    model: str = 'gpt-3.5-turbo', 
    max_tokens: int = 4096
) -> int:
    """
    Calculate the number of tokens in the messages.
    
    Args:
        messages: List of messages or string to calculate tokens for
        model: Model to use for tokenization
        max_tokens: Maximum number of tokens allowed
    
    Returns:
        int: Number of tokens in the messages
    """
    if not is_model_supported(model):
        print(f"Message: Model {model} not in pricing configuration. Skipping token calculation.")
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: Model not found. Using default encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    current_tokens = 0

    if isinstance(messages, str):
        tokens = encoding.encode(messages)
        current_tokens += len(tokens)
        return current_tokens

    for message in messages:
        content = message.get('content')
        if content is None:
            print("Warning: Message content is None. Skipping.")
            break

        if isinstance(content, list):
            # Process list of prompt elements
            for element in content:
                if 'text' in element.get('type', ''):
                    tokens = encoding.encode(element['text'])
                    current_tokens += len(tokens)
        else:
            # Process direct text content
            tokens = encoding.encode(content)
            current_tokens += len(tokens)

    return current_tokens

def save_token_count_to_file(
    filename: str,
    step_tokens: Dict,
    task_name: str,
    global_reward_text_model: str,
    planning_text_model: str,
    token_pricing: Dict = None
) -> None:
    """
    Save token count to a file in JSON format.

    After OpenRouter migration, prefers API-provided cost data over static pricing.

    Args:
        filename: Name of the file to save the token count
        step_tokens: Number of tokens used in steps (may include API usage_data)
        task_name: Name of the task associated with the token count
        global_reward_text_model: Model used for reward modeling
        planning_text_model: Model used for planning
        token_pricing: Optional pricing information for models (legacy, for backward compatibility)
    """
    # Note: After OpenRouter migration, all models are supported
    # Token pricing is optional and only used as fallback

    # Initialize or load existing data
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {
            "calls": [],
            "total_planning_input_tokens": 0,
            "total_planning_output_tokens": 0,
            "total_reward_input_tokens": 0,
            "total_reward_output_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        }

    # Update call records
    call_record = {
        "task_name": task_name,
        "step_tokens": step_tokens
    }
    data["calls"].append(call_record)

    # Update token counts
    data["total_planning_input_tokens"] += step_tokens["steps_planning_input_token_counts"]
    data["total_planning_output_tokens"] += step_tokens["steps_planning_output_token_counts"]
    data["total_reward_input_tokens"] += step_tokens["steps_reward_input_token_counts"]
    data["total_reward_output_tokens"] += step_tokens["steps_reward_output_token_counts"]
    data["total_input_tokens"] += step_tokens["steps_input_token_counts"]
    data["total_output_tokens"] += step_tokens["steps_output_token_counts"]
    data["total_tokens"] += step_tokens["steps_token_counts"]

    # Initialize cost fields if not present
    if "total_planning_input_token_cost" not in data:
        data["total_planning_input_token_cost"] = 0
    if "total_planning_output_token_cost" not in data:
        data["total_planning_output_token_cost"] = 0
    if "total_reward_input_token_cost" not in data:
        data["total_reward_input_token_cost"] = 0
    if "total_reward_output_token_cost" not in data:
        data["total_reward_output_token_cost"] = 0
    if "total_input_token_cost" not in data:
        data["total_input_token_cost"] = 0
    if "total_output_token_cost" not in data:
        data["total_output_token_cost"] = 0
    if "total_token_cost" not in data:
        data["total_token_cost"] = 0

    # Try to extract API-provided costs from step_tokens
    planning_api_cost = step_tokens.get("planning_total_cost", 0.0)
    reward_api_cost = step_tokens.get("reward_total_cost", 0.0)

    if planning_api_cost > 0 or reward_api_cost > 0:
        # Use API-provided costs (OpenRouter migration)
        data["total_planning_cost"] = data.get("total_planning_cost", 0) + planning_api_cost
        data["total_reward_cost"] = data.get("total_reward_cost", 0) + reward_api_cost
        data["total_token_cost"] += planning_api_cost + reward_api_cost
    elif token_pricing and planning_text_model in token_pricing.get("pricing_models", []):
        # Fallback to legacy pricing calculation
        data["total_planning_input_token_cost"] += (
            step_tokens["steps_planning_input_token_counts"] *
            token_pricing[f"{planning_text_model}_input_price"]
        )
        data["total_planning_output_token_cost"] += (
            step_tokens["steps_planning_output_token_counts"] *
            token_pricing[f"{planning_text_model}_output_price"]
        )

        if global_reward_text_model in token_pricing.get("pricing_models", []):
            data["total_reward_input_token_cost"] += (
                step_tokens["steps_reward_input_token_counts"] *
                token_pricing[f"{global_reward_text_model}_input_price"]
            )
            data["total_reward_output_token_cost"] += (
                step_tokens["steps_reward_output_token_counts"] *
                token_pricing[f"{global_reward_text_model}_output_price"]
            )

        data["total_input_token_cost"] += (
            data["total_planning_input_token_cost"] +
            data["total_reward_input_token_cost"]
        )
        data["total_output_token_cost"] += (
            data["total_planning_output_token_cost"] +
            data["total_reward_output_token_cost"]
        )
        data["total_token_cost"] += (
            data["total_input_token_cost"] +
            data["total_output_token_cost"]
        )

    # Save updated data
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
