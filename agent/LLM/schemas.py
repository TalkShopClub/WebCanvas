"""
Pydantic schemas for structured JSON output from LLMs.
These schemas match the JSON structures defined in agent/Prompt/base_prompts.py
"""
from pydantic import BaseModel, Field
from typing import Optional, Union


class AgentActionSchema(BaseModel):
    """
    Schema for agent planning and action responses.
    Used in dom, vision, and other planning modes.

    Matches the schema defined in base_prompts.py lines 77-84.
    """
    thought: str = Field(
        ...,
        description="The reasoning about what action is needed to complete the task"
    )
    action: str = Field(
        ...,
        description="The action to perform (goto, fill_form, google_search, click, select_option, go_back, cache_data, get_final_answer)"
    )
    action_input: str = Field(
        ...,
        description="The input for the action (e.g., URL for goto, search query for google_search, text for fill_form)"
    )
    element_id: Optional[Union[str, int]] = Field(
        None,
        description="The element ID from the accessibility tree (should be an integer, but can be null for some actions like goto)"
    )
    description: str = Field(
        ...,
        description="A description of the current execution action including what website it is and which action was chosen"
    )


class RewardSchema(BaseModel):
    """
    Schema for reward/evaluation responses.
    Used in global_reward and current_reward modes.

    Matches the schema defined in base_prompts.py lines 135-141.
    """
    status: Optional[str] = Field(
        None,
        description="Task completion status: 'doing', 'finished', or 'loop' (only for global_reward)"
    )
    score: str = Field(
        ...,
        description="Score from [1, 3, 7, 9, 10] representing task completion quality"
    )
    reason: Optional[str] = Field(
        None,
        description="Evidence and reasoning for the given score (for global_reward)"
    )
    description: str = Field(
        ...,
        description="Description of current completion status and future plan"
    )
