import json
import os
import csv
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def compute_rouge(generated: str, reference: str) -> dict:
    """Returns ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
    }

def compute_bertscore(generated: str, reference: str) -> float:
    """Returns BERTScore F1 (single pair)."""
    P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)
    return F1.item()

def compute_length_adherence(generated: str, target_words: int, tolerance: float = 0.15) -> dict:
    """
    Returns word count and whether it falls within tolerance of the target.
    tolerance=0.15 means +/-15% of target_words is considered 'on target'.
    """
    word_count = len(generated.split())
    lower_bound = target_words * (1 - tolerance)
    upper_bound = target_words * (1 + tolerance)
    within_target = lower_bound <= word_count <= upper_bound

    return {
        "word_count": word_count,
        "within_target_length": within_target,
    }


def score_summary(doc_id: str, model_id: str, generated: str, reference: str, target_words: int) -> dict:
    """Combines ROUGE + BERTScore + length adherence into one row for metrics.csv."""
    rouge = compute_rouge(generated, reference)
    bert_f1 = compute_bertscore(generated, reference)
    length = compute_length_adherence(generated, target_words)
    return {
        "doc_id": doc_id,
        "model_id": model_id,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bertscore_f1": bert_f1,
        "word_count": length["word_count"],
        "within_target_length": length["within_target_length"],
    }

def append_to_csv(row: dict, csv_path: str):
    """Appends one row to metrics.csv, writing header if file is new."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)