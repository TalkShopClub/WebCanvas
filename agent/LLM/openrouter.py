import os
import asyncio
from functools import partial
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from sanic.log import logger
from openai import AsyncOpenAI
from .schemas import AgentActionSchema, RewardSchema


class OpenRouterGenerator:
    """
    Unified LLM generator using OpenRouter API.
    Supports any model available on OpenRouter with smart JSON mode degradation.
    """

    def __init__(self, model=None, json_mode=False):
        self.model = model
        self.json_mode = json_mode
        self.json_mode_strategy = None  # Will be set during first request
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    async def request(self, messages: list = None, max_tokens: int = 500, temperature: float = 0.7) -> tuple:
        """
        Make a request to OpenRouter API with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (response_content, error_message, usage_data)
            usage_data dict contains: prompt_tokens, completion_tokens, total_tokens, total_cost
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Call chat() directly - it already handles async execution via run_in_executor
                response = await self.chat(messages, max_tokens, temperature)

                # Extract content
                choice = response.choices[0]
                if choice.finish_reason == 'length':
                    logger.warning("Response may be truncated due to length. Be cautious when parsing JSON.")
                content = choice.message.content

                # Extract usage data
                usage_data = self._extract_usage(response)

                return content, "", usage_data

            except Exception as e:
                logger.error(f"OpenRouter request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # Final failure - return empty with zero usage
                    return "", str(e), {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "total_cost": 0.0
                    }
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

    async def chat(self, messages, max_tokens=500, temperature=0.7):
        """
        Execute the actual API call with JSON mode handling.

        Implements smart JSON mode degradation:
        1. Try json_schema (structured outputs)
        2. Fall back to json_object (legacy JSON mode)
        3. Fall back to plain text with JSON instruction
        """
        loop = asyncio.get_event_loop()

        if self.json_mode:
            return await self._chat_with_json_mode(messages, max_tokens, temperature, loop)
        else:
            return await self._chat_plain(messages, max_tokens, temperature, loop)

    async def _chat_with_json_mode(self, messages, max_tokens, temperature, loop):
        """Handle JSON mode with graceful degradation."""

        # Strategy 1: Try json_schema (most strict, structured outputs)
        if self.json_mode_strategy is None or self.json_mode_strategy == "json_schema":
            try:
                logger.info(f"Attempting JSON mode with json_schema for model {self.model}")
                response = await self._request_with_json_schema(messages, max_tokens, temperature, loop)
                self.json_mode_strategy = "json_schema"
                logger.info(f"JSON mode: using json_schema for model {self.model}")
                return response
            except Exception as e:
                logger.warning(f"json_schema not supported for {self.model}: {e}")
                self.json_mode_strategy = "json_object"

        # Strategy 2: Try json_object (legacy JSON mode)
        if self.json_mode_strategy == "json_object":
            try:
                logger.info(f"Attempting JSON mode with json_object for model {self.model}")
                response = await self._request_with_json_object(messages, max_tokens, temperature, loop)
                logger.info(f"JSON mode: using json_object for model {self.model}")
                return response
            except Exception as e:
                logger.warning(f"json_object not supported for {self.model}: {e}")
                self.json_mode_strategy = "disabled"

        # Strategy 3: Disable JSON mode and use prompt instructions
        # Note: This is essentially the same as json_mode=False since the prompts
        # already ask for JSON output. We log a warning so users know JSON mode
        # degraded to instruction-based parsing.
        logger.warning(
            f"JSON mode disabled for model {self.model} - neither json_schema nor json_object supported. "
            f"Relying on prompt instructions for JSON formatting."
        )
        return await self._request_plain_no_json_mode(messages, max_tokens, temperature, loop)

    async def _request_with_json_schema(self, messages, max_tokens, temperature, loop):
        """
        Request with structured JSON schema (most strict).

        Uses AgentActionSchema as the default schema, which matches the planning/action
        format defined in agent/Prompt/base_prompts.py.
        """
        # Ensure there's a JSON instruction in messages
        messages = self._prepare_messages_for_json_mode(messages)

        # Use AgentActionSchema as the default (covers most use cases)
        # This matches the JSON structure defined in the prompts
        schema = AgentActionSchema.model_json_schema()

        data = {
            'model': self.model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': messages,
            'response_format': {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_action",
                    "strict": True,
                    "schema": schema
                }
            }
        }

        # Add usage tracking
        data['extra_body'] = {
            "usage": {
                "include": True
            }
        }

        # AsyncOpenAI client - await directly, no executor needed
        return await self.client.chat.completions.create(**data)

    async def _request_with_json_object(self, messages, max_tokens, temperature, loop):
        """Request with json_object response format (legacy JSON mode)."""
        # Ensure there's a JSON instruction in messages
        messages = self._prepare_messages_for_json_mode(messages)

        data = {
            'model': self.model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': messages,
            'response_format': {"type": "json_object"}
        }

        # Add usage tracking
        data['extra_body'] = {
            "usage": {
                "include": True
            }
        }

        # AsyncOpenAI client - await directly, no executor needed
        return await self.client.chat.completions.create(**data)

    async def _request_plain_no_json_mode(self, messages, max_tokens, temperature, loop):
        """
        Plain request without JSON mode enforcement.

        This is used when JSON mode is requested but not supported by the model.
        The prompts already ask for JSON output, so this relies on instruction-following.
        """
        data = {
            'model': self.model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': messages  # Don't modify messages - prompts already ask for JSON
        }

        # Add usage tracking
        data['extra_body'] = {
            "usage": {
                "include": True
            }
        }

        # AsyncOpenAI client - await directly, no executor needed
        return await self.client.chat.completions.create(**data)

    async def _chat_plain(self, messages, max_tokens, temperature, loop):
        """Plain text request without JSON mode."""
        data = {
            'model': self.model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': messages
        }

        # Add usage tracking
        data['extra_body'] = {
            "usage": {
                "include": True
            }
        }

        # AsyncOpenAI client - await directly, no executor needed
        return await self.client.chat.completions.create(**data)

    @staticmethod
    def _prepare_messages_for_json_mode(messages):
        """Ensure there's a system message instructing the model to generate JSON."""
        # Check if there's already a JSON instruction
        has_json_instruction = any(
            "json" in message.get('content', '').lower()
            for message in messages
            if message.get('role') == 'system'
        )

        if not has_json_instruction:
            # Insert JSON instruction at the beginning
            json_instruction = {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON."
            }
            messages.insert(0, json_instruction)

        return messages

    def _extract_usage(self, response) -> dict:
        """
        Extract token usage and cost data from OpenRouter response.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens, total_cost
        """
        if hasattr(response, 'usage'):
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(usage, 'completion_tokens', 0),
                "total_tokens": getattr(usage, 'total_tokens', 0),
                # OpenRouter may provide cost in different fields
                # Check common field names
                "total_cost": getattr(usage, 'cost', 0.0)
            }

        # No usage data available
        logger.warning(f"No usage data in response for model {self.model}")
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }