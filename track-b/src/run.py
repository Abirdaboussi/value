import sys
sys.path.append('.')
import csv
import yaml
import os
from datetime import datetime
from src.logger import log_run, VRAMMonitor
from src.summarize import summarize_report
from src.ingest import count_tokens
from src.metrics_auto import score_summary, append_to_csv
from src.judge import check_faithfulness

# --- load config ---
with open("config/run.yaml", "r", encoding="utf-8") as f:
    run_cfg = yaml.safe_load(f)

with open("config/models.yaml", "r", encoding="utf-8") as f:
    models_cfg = yaml.safe_load(f)

    # --- resume into the existing run directory ---
run_dir = os.path.join("runs", "20260818_100736")
os.makedirs(run_dir, exist_ok=True)
print(f"Run dir (resuming): {run_dir}")

# --- figure out which (doc_id, model_id) pairs already succeeded ---
completed_pairs = set()
logs_path = os.path.join(run_dir, "logs.jsonl")
if os.path.exists(logs_path):
    import json
    with open(logs_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            completed_pairs.add((rec["doc_id"], rec["model_id"]))
print(f"Already completed: {len(completed_pairs)} pairs")

# --- load manifest into a dict keyed by doc_id ---
manifest = {}
with open("data/processed/manifest.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["doc_id"]:  # skip blank trailing row
            manifest[row["doc_id"]] = row

# --- load dev split ---
with open("data/splits/dev.txt", "r", encoding="utf-8") as f:
    dev_ids = [line.strip() for line in f if line.strip()]

# --- loop over all dev docs x all models ---
for doc_id in dev_ids[:1]:
    report_path = manifest[doc_id]["report_path"]
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    for model_entry in models_cfg["models"][:2]:
        model_id = f"{model_entry['id']}:{model_entry['tag']}"
        if (doc_id, model_id) in completed_pairs:
            print(f"SKIP (already done): doc={doc_id} model={model_id}")
            continue
        print(f"\n=== Running doc={doc_id} model={model_id} ===", flush=True)
        try:
            with VRAMMonitor() as mon:
                result = summarize_report(model_id, report_text, run_cfg, count_tokens)
        except Exception as e:
            import traceback
            print(f"ERROR on doc={doc_id} model={model_id}: {e}", flush=True)
            traceback.print_exc()
            continue

        log_record = log_run(run_dir, doc_id, model_id, result, peak_vram_mb=mon.peak_mb)
        print("LOGGED:", log_record, flush=True)

    # --- load reference summary and score ---
    highlight_path = manifest[doc_id]["highlight_path"]
    with open(highlight_path, "r", encoding="utf-8") as f:
        reference_text = f.read()

    target_words = run_cfg["target_summary_words"]
    metrics_row = score_summary(doc_id, model_id, result["summary"], reference_text, target_words)

    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    append_to_csv(metrics_row, metrics_csv_path)
    print("METRICS:", metrics_row)

    # --- faithfulness judge ---
    judge_cfg = {"temperature": 0.2, "max_tokens": 2800}
    faith_result = check_faithfulness(report_text, result["summary"], judge_cfg)

    faith_row = {
        "doc_id": doc_id,
        "model_id": model_id,
        "n_claims": faith_result["n_claims"],
        "support_fraction": faith_result["support_fraction"],
        "hallucination_rate": faith_result["hallucination_rate"],
    }
    faith_csv_path = os.path.join(run_dir, "faithfulness.csv")
    append_to_csv(faith_row, faith_csv_path)
    print("FAITHFULNESS:", faith_row)



    print("STRATEGY:", result["strategy"])
    print("N_CALLS:", result["n_calls"])
    print("SUMMARY (first 300 chars):")
    print(result["summary"][:300])