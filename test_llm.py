"""Diagnose LLM connection problems:  python test_llm.py
Reads GROQ_API_KEY (or whichever provider you set) from the environment."""
import os

from llm import PROVIDERS, get_client

client, model, name = get_client()
print(f"provider={name} model={model} key_set={client is not None}")
if client is None:
    raise SystemExit("No API key found in this terminal. Set it first (see README).")

print("base_url:", client.base_url)
try:
    ids = [m.id for m in client.models.list().data]
    print("Models available to you:", ids)
    if model not in ids:
        print(f"\n>>> '{model}' is NOT in that list. Set LLM_MODEL to one of the ids above.")
except Exception as e:
    print("Listing models failed ->", type(e).__name__, e, "| cause:", repr(e.__cause__))

try:
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Say hello in 3 words."}]
    )
    print("\nSUCCESS:", r.choices[0].message.content)
except Exception as e:
    print("\nChat call failed ->", type(e).__name__, e, "| cause:", repr(e.__cause__))
