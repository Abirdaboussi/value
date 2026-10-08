import os, time, requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def generate_groq(model_id, prompt, cfg):
    api_key = os.environ["GROQ_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.2),
        "max_tokens": cfg.get("max_tokens", 1600),
    }
    start = time.time()
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print("FULL RESPONSE JSON:", data)
    latency_ms = int((time.time() - start) * 1000)

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {
        "text": text,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "latency_ms": latency_ms,
    }