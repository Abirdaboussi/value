import time
import ollama

def generate(model_id: str, prompt: str, cfg: dict) -> dict:
    """
    Calls a local Ollama model with fixed decoding settings.
    Returns text + token counts + latency for logging.
    """
    start = time.time()

    response = ollama.chat(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        options={
    "temperature": cfg["temperature"],
    "seed": cfg["seed"],
    "num_predict": cfg["max_tokens"],
    "stop": ["\nSummarize", "\n###", "\nInstruction"],
},
    )

    latency_ms = (time.time() - start) * 1000

    return {
        "text": response["message"]["content"],
        "input_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "latency_ms": latency_ms,
    }