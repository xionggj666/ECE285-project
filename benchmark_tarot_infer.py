import csv
import os
import subprocess
import threading
import time
from pathlib import Path

PYTHON_BIN = "python"
SCRIPT_PATH = "/home/gaxiong/tarot_pipeline/tarot_copyplace_symbol_library_full.py"

PROMPT = "impressionist tarot deck, dark blue and gold, antique mystical atmosphere"
CARD = "The Fool"

LAYOUT_DIR = "/home/gaxiong/data/layouts"
SYMBOL_CACHE_DIR = "outputs/symbol_cache"
SYMBOL_LIBRARY_DIR = "outputs/symbol_library"
CONTROLNET_PATH = "lllyasviel/sd-controlnet-scribble"
LORA_PATH = "loras/watercolor_light"

RESULT_CSV = "benchmark_tarot_results.csv"


def sanitize_name(s: str) -> str:
    return s.lower().replace(" ", "_").replace("/", "_")


def get_gpu_mem_mb() -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()
        vals = [int(x.strip()) for x in out if x.strip().isdigit()]
        return max(vals) if vals else 0
    except Exception:
        return 0


def monitor_gpu(stop_event, interval=0.2):
    peak = 0
    while not stop_event.is_set():
        peak = max(peak, get_gpu_mem_mb())
        time.sleep(interval)
    peak = max(peak, get_gpu_mem_mb())
    return peak


def run_case(case_name, use_lora=True, use_controlnet=True):
    out_dir = Path("outputs") / f"bench_{sanitize_name(CARD)}_{sanitize_name(case_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_BIN,
        SCRIPT_PATH,
        "--prompt", PROMPT,
        "--card", CARD,
        "--out_dir", str(out_dir),
        "--layout_dir", LAYOUT_DIR,
        "--symbol_cache_dir", SYMBOL_CACHE_DIR,
        "--symbol_library_dir", SYMBOL_LIBRARY_DIR,
        "--warmup_symbols_per_suit", "8",
        "--symbol_candidates", "8",
        "--store_top_k", "3",
        "--final_select_top_k", "3",
    ]

    if use_controlnet:
        cmd += ["--controlnet_path", CONTROLNET_PATH]

    if use_lora:
        cmd += ["--lora_path", LORA_PATH]

    stop_event = threading.Event()
    peak_holder = {"peak": 0}

    def worker():
        peak = 0
        while not stop_event.is_set():
            peak = max(peak, get_gpu_mem_mb())
            time.sleep(0.2)
        peak = max(peak, get_gpu_mem_mb())
        peak_holder["peak"] = peak

    monitor_thread = threading.Thread(target=worker, daemon=True)

    print(f"\n=== Running case: {case_name} ===")
    print(" ".join(cmd))

    t0 = time.time()
    monitor_thread.start()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - t0
    stop_event.set()
    monitor_thread.join()

    log_path = out_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout or "")
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr or "")

    row = {
        "case_name": case_name,
        "card": CARD,
        "use_controlnet": use_controlnet,
        "use_lora": use_lora,
        "elapsed_sec": round(elapsed, 2),
        "peak_gpu_mem_mb": peak_holder["peak"],
        "returncode": result.returncode,
        "out_dir": str(out_dir),
        "log_path": str(log_path),
    }
    return row


def main():
    cases = [
        ("base", False, False),
        ("lora_only", False, True),
        ("controlnet_lora", True, True),
    ]

    rows = []
    for case_name, use_controlnet, use_lora in cases:
        row = run_case(
            case_name=case_name,
            use_lora=use_lora,
            use_controlnet=use_controlnet,
        )
        rows.append(row)
        print(row)

    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark results to {RESULT_CSV}")


if __name__ == "__main__":
    main()