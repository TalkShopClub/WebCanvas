import os
from .openrouter import OpenRouterGenerator
from .schemas import SemanticMatchSchema

# Legacy imports kept for backward compatibility (not used)
# from .openai import GPTGenerator, GPTGeneratorWithJSON
# from .claude import ClaudeGenerator
# from .gemini import GeminiGenerator
# from .togetherai import TogetherAIGenerator


def create_llm_instance(model, json_mode=False, all_json_models=None, schema=None, use_custom_endpoint=False):
    """
    Create an LLM instance for any OpenRouter-supported model or custom endpoint.

    After OpenRouter migration, all models are routed through OpenRouterGenerator.
    This provides unified access to OpenAI, Anthropic, Google, Meta, and other models.
    Can also work with custom OpenAI-compatible endpoints (e.g., vLLM).

    Args:
        model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet", or custom model name)
        json_mode: Whether to attempt JSON mode (will gracefully degrade if unsupported)
        all_json_models: Deprecated parameter, kept for backward compatibility
        schema: Pydantic BaseModel schema class for structured output (optional)
        use_custom_endpoint: If True, use CUSTOM_LLM_API_KEY and CUSTOM_LLM_BASE_URL env vars

    Returns:
        OpenRouterGenerator instance configured for the specified model
    """
    return OpenRouterGenerator(model=model, json_mode=json_mode, schema=schema, use_custom_endpoint=use_custom_endpoint)


async def semantic_match_llm_request(messages: list = None):
    """
    Make a semantic matching request using the default model.

    Uses OpenRouter with a configurable default model and SemanticMatchSchema for structured output.

    Returns:
        tuple: (response, usage_data) for backward compatibility with existing code
    """
    default_model = os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-5-mini")
    llm = OpenRouterGenerator(model=default_model, json_mode=True, schema=SemanticMatchSchema)
    response, error, usage_data = await llm.request(messages)
    return response, usage_data