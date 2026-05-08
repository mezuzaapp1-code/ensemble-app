from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GEMINI_KEY", "").strip()
print("Loaded key:", key[:10])

client = genai.Client(api_key=key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello"
)

print(response.text)