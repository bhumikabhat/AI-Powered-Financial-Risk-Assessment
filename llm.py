"""LLM explanation layer. Works with ANY OpenAI-compatible endpoint, so you can
swap providers by changing three env vars. No paid OpenAI key needed.

Provider presets (all have free tiers; limits change, check each dashboard):
  groq        -> https://console.groq.com/keys          (fast, no credit card)
  gemini      -> https://aistudio.google.com/apikey     (generous free tier)
  openrouter  -> https://openrouter.ai/keys             (use models ending :free)
  ollama      -> https://ollama.com  (runs locally, truly unlimited & private)
"""
import json
import os

from openai import OpenAI

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "ollama": ("http://localhost:11434/v1", None, "llama3.2"),
}

SYSTEM = (
    "You are a careful financial risk analyst. You are given the output of a "
    "statistical credit-risk model. Explain it in plain English for a loan "
    "officer. Use ONLY the numbers provided; never invent figures. Be concise "
    "(under 150 words): 1) overall risk level, 2) the main drivers, 3) 2-3 "
    "concrete mitigation or follow-up suggestions. End with one line noting "
    "this is decision support, not a final credit decision."
)


def get_client():
    name = os.getenv("LLM_PROVIDER", "groq").lower()
    base, key_env, default_model = PROVIDERS[name]
    key = os.getenv(key_env) if key_env else "ollama"
    if not key:
        return None, None, name
    model = os.getenv("LLM_MODEL", default_model)
    return OpenAI(base_url=base, api_key=key), model, name


def template_explanation(result: dict) -> str:
    """Offline fallback so the app still works with no API key at all."""
    d = result["top_drivers"]
    lines = [f"Risk level: {result['risk_band']} "
             f"({result['default_probability']:.1%} estimated default probability)."]
    for x in d[:3]:
        lines.append(f"- {x['feature'].replace('_', ' ')} = {x['value']:.2f} {x['direction']} "
                     f"(impact {x['impact']:+.1%}).")
    lines.append("Decision support only, not a final credit decision.")
    return "\n".join(lines)


def explain(result: dict) -> tuple[str, str]:
    """Returns (text, source). Falls back to template on any failure."""
    client, model, name = get_client()
    if client is None:
        return template_explanation(result), "offline template (no API key set)"
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(result)},
            ],
        )
        return resp.choices[0].message.content.strip(), f"{name} / {model}"
    except Exception as e:  # rate limit, network, bad key...
        cause = f" | cause: {e.__cause__!r}" if getattr(e, "__cause__", None) else ""
        msg = f"{type(e).__name__}: {e}{cause}"[:400]
        return template_explanation(result), f"offline template -> {msg}"
