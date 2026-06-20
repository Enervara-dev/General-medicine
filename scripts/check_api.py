import os

from dotenv import load_dotenv


def check_gemini_api():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file.")
        return

    try:
        from google import genai
    except ImportError:
        print("❌ google-genai is not installed. Run: pip install -e .")
        return

    model = "gemini-2.5-flash-lite"
    print(f"🔄 Testing Gemini API with model: {model}")
    print("⏳ Sending request (NO retries)...")

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents="Hello, this is a quick API test.",
        )
        print("\n📝 Response text:")
        print(resp.text)
        print("\n✅ API is working correctly!")
    except Exception as e:
        print(f"\n❌ API call failed: {e}")


if __name__ == "__main__":
    check_gemini_api()
