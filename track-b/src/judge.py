import sys
sys.path.append('.')
from src.adapter import generate
import os
from dotenv import load_dotenv
from src.groq_adapter import generate_groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Check your .env file.")


def load_judge_prompt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_system_user(template: str):
    system_part = template.split("### USER ###")[0].replace("### SYSTEM ###", "").strip()
    user_part = template.split("### USER ###")[1].strip()
    return system_part, user_part

def extract_claims(summary: str) -> list:
    """
    Simple claim splitter: treats each sentence as one claim.
    Good enough for a v1 faithfulness check.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def check_faithfulness(source_text: str, summary: str, cfg: dict) -> dict:
    """
    Extracts claims from the summary, asks the judge model whether each
    is supported by the source. Returns support fraction + hallucination rate.
    """
    claims = extract_claims(summary)
    claims_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

    template = load_judge_prompt("prompts/judge_faithful.txt")
    system_part, user_part = split_system_user(template)
    filled_system = system_part
    truncated_source = truncate_to_tokens(source_text, max_tokens=4000)
    filled_user = user_part.replace("{source_text}", truncated_source).replace("{claims_list}", claims_list)    
    full_prompt = filled_system + "\n\n" + filled_user
    print(f"PROMPT SIZE: {len(full_prompt)} chars, {len(full_prompt.encode('utf-8'))} bytes")
    result = generate_groq("openai/gpt-oss-120b", full_prompt, cfg)
    judge_response = result["text"]

    # count verdicts line by line, not substring counting
    supported_count = 0
    unsupported_count = 0
    for line in judge_response.splitlines():
        line_lower = line.lower()
        if "unsupported" in line_lower:
            unsupported_count += 1
        elif "supported" in line_lower:
            supported_count += 1

    total_claims = len(claims)
    support_fraction = supported_count / total_claims if total_claims > 0 else 0
    hallucination_rate = 1 - support_fraction

    return {
        "n_claims": total_claims,
        "support_fraction": support_fraction,
        "hallucination_rate": hallucination_rate,
        "raw_verdict": judge_response,
    }
    return {
        "n_claims": total_claims,
        "support_fraction": support_fraction,
        "hallucination_rate": hallucination_rate,
        "raw_verdict": verdict_text,
    }

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncates text to roughly max_tokens using the same tokenizer as elsewhere."""
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])