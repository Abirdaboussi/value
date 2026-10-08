import json
import os
import subprocess
from datetime import datetime
import threading
import time

def get_run_dir():
    """Creates (once) and returns today's run directory: runs/<timestamp>/"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_gpu_stats():
    """
    Queries nvidia-smi for peak VRAM usage (MB).
    Returns None if nvidia-smi isn't available (e.g. CPU-only run).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        vram_mb = int(result.stdout.strip().split("\n")[0])
        return vram_mb
    except Exception:
        return None

def log_run(run_dir: str, doc_id: str, model_id: str, result: dict, peak_vram_mb=None):
    """
    Appends one JSONL record for a (doc, model) run to runs/<ts>/logs.jsonl.
    'result' is the dict returned by summarize_report().
    'peak_vram_mb' is the max VRAM seen during generation (from VRAMMonitor).
    """
    latency_s = result["latency_ms"] / 1000
    tokens_per_sec = (result["output_tokens"] / latency_s) if latency_s > 0 else None

    record = {
        "doc_id": doc_id,
        "model_id": model_id,
        "strategy": result["strategy"],
        "n_calls": result["n_calls"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "latency_ms": result["latency_ms"],
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_mb": peak_vram_mb,
        "timestamp": datetime.now().isoformat(),
    }

    log_path = os.path.join(run_dir, "logs.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record



class VRAMMonitor:
    """
    Polls nvidia-smi every ~0.5s on a background thread while a model runs,
    and keeps the highest VRAM value seen. Use as a context manager around
    the summarize_report() call.
    """
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mb = 0
        self._stop_flag = False
        self._thread = None

    def _poll_loop(self):
        while not self._stop_flag:
            current = get_gpu_stats()
            if current is not None:
                self.peak_mb = max(self.peak_mb, current)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_flag = True
        self._thread.join(timeout=2)