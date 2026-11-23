import os
import time

from dotenv import load_dotenv


MODELS = [
    "gemini-2.5-flash",        # primary
    "gemini-2.5-flash-lite",   # fallback
]


def call_model(model: str, api_key: str):
    from google import genai

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents="Say 'API key works' and name yourself.",
        )
        return resp.text
    except Exception as e:
        print(f"[ERROR] {model}: {e}")
        return None


def run_with_fallback():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment / .env")

    for model in MODELS:
        print(f"→ Trying {model}")
        output = call_model(model, api_key)

        if output:
            print(f"[SUCCESS] {model} response:\n{output}")
            return

        # small backoff before fallback
        time.sleep(1)

    raise RuntimeError("All models failed. Check API key / quota.")


if __name__ == "__main__":
    run_with_fallback()
