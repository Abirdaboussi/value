import sys
sys.path.append('.')
from src.adapter import generate

PROMPT_PATH = "prompts/summarize_v1.txt"

def load_prompt_template():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def split_system_user(template: str):
    """Splits the frozen prompt file into system and user parts."""
    system_part = template.split("### USER ###")[0].replace("### SYSTEM ###", "").strip()
    user_part = template.split("### USER ###")[1].strip()
    return system_part, user_part

def chunk_text(text: str, tokenizer_count_fn, budget: int, overlap: int):
    """
    Splits text into chunks that fit the token budget, with overlap.
    tokenizer_count_fn: a function that counts tokens for a string.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        # naive word-based windowing; approximate but simple and safe
        chunk_words = words[start:start + budget]
        chunk_text_str = " ".join(chunk_words)
        chunks.append(chunk_text_str)
        if start + budget >= len(words):
            break
        start += (budget - overlap)
    return chunks

def build_prompt(system_part: str, user_part: str, target_words: int, report_text: str):
    filled_system = system_part.replace("{target_words}", str(target_words))
    filled_user = user_part.replace("{target_words}", str(target_words)) \
                            .replace("{report_text}", report_text)
    return filled_system, filled_user

def summarize_report(model_id: str, report_text: str, cfg: dict, token_count_fn):
    """
    Summarizes one report with single-pass or map-reduce, depending on token budget.
    Returns dict with final summary text, strategy used, and n_calls.
    """
    template = load_prompt_template()
    system_part, user_part = split_system_user(template)
    target_words = cfg["target_summary_words"]
    ctx_budget = cfg["ctx_budget"]
    overlap = cfg["chunk_overlap"]

    report_tokens = token_count_fn(report_text)

    # Case 1: fits in one pass
    if report_tokens <= ctx_budget:
        sys_p, user_p = build_prompt(system_part, user_part, target_words, report_text)
        full_prompt = sys_p + "\n\n" + user_p
        result = generate(model_id, full_prompt, cfg)
        return {
            "summary": result["text"],
            "strategy": "single_pass",
            "n_calls": 1,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_ms": result["latency_ms"],
        }

    # Case 2: map-reduce
    chunks = chunk_text(report_text, token_count_fn, ctx_budget, overlap)
    chunk_summaries = []
    n_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0

    for chunk in chunks:
        sys_p, user_p = build_prompt(system_part, user_part, target_words, chunk)
        full_prompt = sys_p + "\n\n" + user_p
        result = generate(model_id, full_prompt, cfg)
        chunk_summaries.append(result["text"])
        n_calls += 1
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_latency += result["latency_ms"]

    # Reduce step: summarize the concatenated chunk-summaries
    combined = "\n\n".join(chunk_summaries)
    sys_p, user_p = build_prompt(system_part, user_part, target_words, combined)
    full_prompt = sys_p + "\n\n" + user_p
    final_result = generate(model_id, full_prompt, cfg)
    n_calls += 1
    total_input_tokens += final_result["input_tokens"]
    total_output_tokens += final_result["output_tokens"]
    total_latency += final_result["latency_ms"]

    return {
        "summary": final_result["text"],
        "strategy": "map_reduce",
        "n_calls": n_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "latency_ms": total_latency,
    }