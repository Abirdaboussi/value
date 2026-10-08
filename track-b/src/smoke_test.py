from ollama import chat

import sys
sys.path.append('.')
from src.adapter import generate

response = chat(
    model="phi4-mini",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Ollama is working."
        }
    ]
)

print(response.message.content)


cfg = {"temperature": 0.2, "seed": 42, "max_tokens": 200}
result = generate("phi4-mini", "Summarize: The sky is blue because of Rayleigh scattering.", cfg)
print(result)