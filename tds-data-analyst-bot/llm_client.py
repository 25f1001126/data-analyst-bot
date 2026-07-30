"""
Unified LLM client with Groq -> OpenRouter fallback.

Both providers expose an OpenAI-compatible /chat/completions endpoint, so we
reuse the `openai` SDK for both and just swap base_url / api_key / model.

Order of attempts:
  1. Groq, pinned model, with one retry on rate limit / transient error.
  2. OpenRouter, pinned free model.
  3. OpenRouter, "openrouter/auto" router (in case the pinned free model was
     delisted -- OpenRouter's free catalog rotates).

Raises RuntimeError only if every attempt above fails.
"""
import os
import time
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_AUTO_MODEL = "openrouter/auto"

_OPENROUTER_HEADERS = {
    "HTTP-Referer": os.environ.get("REPO_URL", "https://github.com/your-user/your-repo"),
    "X-Title": "data-analyst-telegram-bot",
}

groq_client = (
    OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    if GROQ_API_KEY else None
)
openrouter_client = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    if OPENROUTER_API_KEY else None
)


def _call(client, model, messages, tools, extra_headers=None):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=2048,
        extra_headers=extra_headers or {},
    )


def call_llm(messages, tools):
    """
    Returns (response, provider_name) so the caller can log which provider
    actually answered. Raises RuntimeError if all providers/attempts fail.
    """
    errors = []

    if groq_client:
        for attempt in range(2):
            try:
                resp = _call(groq_client, GROQ_MODEL, messages, tools)
                return resp, f"groq:{GROQ_MODEL}"
            except RateLimitError as e:
                errors.append(f"groq rate limit (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
            except (APIStatusError, APIConnectionError) as e:
                errors.append(f"groq error: {e}")
                break
    else:
        errors.append("groq not configured (no GROQ_API_KEY)")

    if openrouter_client:
        try:
            resp = _call(openrouter_client, OPENROUTER_MODEL, messages, tools, _OPENROUTER_HEADERS)
            return resp, f"openrouter:{OPENROUTER_MODEL}"
        except Exception as e:
            errors.append(f"openrouter ({OPENROUTER_MODEL}) error: {e}")
            try:
                resp = _call(openrouter_client, OPENROUTER_AUTO_MODEL, messages, tools, _OPENROUTER_HEADERS)
                return resp, f"openrouter:{OPENROUTER_AUTO_MODEL}"
            except Exception as e2:
                errors.append(f"openrouter (auto) error: {e2}")
    else:
        errors.append("openrouter not configured (no OPENROUTER_API_KEY)")

    raise RuntimeError("All LLM providers failed: " + " | ".join(errors))
