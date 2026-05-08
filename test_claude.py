from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=200,
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence"
        }
    ]
)

print(message.content[0].text)