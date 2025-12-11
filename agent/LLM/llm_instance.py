import os
from .openrouter import OpenRouterGenerator

# Legacy imports kept for backward compatibility (not used)
# from .openai import GPTGenerator, GPTGeneratorWithJSON
# from .claude import ClaudeGenerator
# from .gemini import GeminiGenerator
# from .togetherai import TogetherAIGenerator


def create_llm_instance(model, json_mode=False, all_json_models=None):
    """
    Create an LLM instance for any OpenRouter-supported model.

    After OpenRouter migration, all models are routed through OpenRouterGenerator.
    This provides unified access to OpenAI, Anthropic, Google, Meta, and other models.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet")
        json_mode: Whether to attempt JSON mode (will gracefully degrade if unsupported)
        all_json_models: Deprecated parameter, kept for backward compatibility

    Returns:
        OpenRouterGenerator instance configured for the specified model
    """
    return OpenRouterGenerator(model=model, json_mode=json_mode)


async def semantic_match_llm_request(messages: list = None):
    """
    Make a semantic matching request using the default model.

    Uses OpenRouter with a configurable default model.
    """
    default_model = os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-3.5-turbo")
    llm = OpenRouterGenerator(model=default_model)
    response, error, usage_data = await llm.request(messages)
    return response