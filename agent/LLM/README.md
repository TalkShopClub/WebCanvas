### Setup Your API Keys

After migrating to OpenRouter, WebCanvas now uses a unified API to access models from multiple providers including OpenAI, Anthropic, Google, Meta, and many others. This simplifies configuration and enables access to 200+ models through a single API key.

#### OpenRouter API Key (Required)

OpenRouter provides unified access to all supported LLM providers. This is the primary API key you need.

**MacOS/Linux:**

```bash
export OPENROUTER_API_KEY='your-openrouter-api-key-here'
```

**Windows:**

```text
setx OPENROUTER_API_KEY "your-openrouter-api-key-here"
```

**Getting Your API Key:**
1. Sign up at [https://openrouter.ai](https://openrouter.ai)
2. Navigate to your account settings
3. Generate an API key
4. Add credits to your account to use the API

**Available Models:**
- OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-3.5-turbo`, etc.
- Anthropic: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`, etc.
- Google: `google/gemini-pro-1.5`, `google/gemini-flash-1.5`, etc.
- Meta: `meta-llama/llama-3.1-70b-instruct`, `meta-llama/llama-3.1-405b-instruct`, etc.
- And many more! See [OpenRouter Models](https://openrouter.ai/models) for the complete list.

#### Custom LLM Endpoint (Optional)

If you want to use a custom LLM endpoint (e.g., vLLM, text-generation-inference, or other OpenAI-compatible servers), you can configure these environment variables:

**MacOS/Linux:**

```bash
export CUSTOM_LLM_API_KEY='your-custom-api-key-here'
export CUSTOM_LLM_BASE_URL='your-custom-base-url-here'
```

**Windows:**

```text
setx CUSTOM_LLM_API_KEY "your-custom-api-key-here"
setx CUSTOM_LLM_BASE_URL "your-custom-base-url-here"
```

**Example with vLLM on RunPod:**

```bash
export CUSTOM_LLM_API_KEY='your-runpod-api-key'
export CUSTOM_LLM_BASE_URL='https://api.runpod.ai/v2/your-endpoint-id/openai/v1'
```

Then run your evaluation with the `--custom_model` flag:

```bash
python evaluate.py \
    --planning_text_model your-model-name \
    --global_reward_text_model openai/gpt-4o-mini \
    --custom_model
```

**Important Notes:**
- When using `--custom_model`, only the planning model uses the custom endpoint
- Vision models (gpt-4-turbo) and reward models always use OpenRouter
- Your custom endpoint must be OpenAI-compatible (support the `/v1/chat/completions` endpoint)

---

### Migration Notes

**If you're upgrading from an older version** that used provider-specific API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.), those keys are no longer needed. Simply set up your OpenRouter API key and update your model names to use the OpenRouter format (`provider/model-name`).

**Benefits of OpenRouter:**
- Single API key for all providers
- Automatic cost tracking via API responses
- Access to 200+ models including newest releases
- Fallback to alternative providers if primary is unavailable
- Unified rate limiting and usage monitoring

Make sure to replace `your-openrouter-api-key-here` and other placeholder values with your actual credentials.
