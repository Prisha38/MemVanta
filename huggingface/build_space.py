from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "openllama-7b-v2-ab"
OUT = ROOT / "hf-space" / "data.json"


def _read_text(name: str) -> str:
    return (RESULTS / name).read_text(encoding="utf-8").strip()


def _source_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    model_sha_line = _read_text("model.sha256").split(maxsplit=1)
    model_sha = model_sha_line[0]
    model_file = model_sha_line[1] if len(model_sha_line) > 1 else "unknown"

    payload = {
        "benchmark": summary["benchmark"],
        "model": {
            "name": "OpenLLaMA 7B v2",
            "file": model_file,
            "quantization": "Q4_0",
            "sha256": model_sha,
            "size": _read_text("model-size.txt"),
        },
        "memvanta": {
            "peak_rss_kib": summary["memvanta_peak_rss_kib"],
            "peak_rss_gib": summary["memvanta_peak_rss_kib"] / (1024 * 1024),
            "prompt_tps_mean": summary["memvanta_pp"]["mean"],
            "prompt_tps_sd": summary["memvanta_pp"]["sd"],
            "prompt_tps_cv_pct": summary["memvanta_pp"]["cv_pct"],
            "generation_tps_mean": summary["memvanta_tg"]["mean"],
            "generation_tps_sd": summary["memvanta_tg"]["sd"],
            "generation_tps_cv_pct": summary["memvanta_tg"]["cv_pct"],
        },
        "llama_cpp": {
            "peak_rss_kib": summary["llama_peak_rss_kib"],
            "peak_rss_gib": summary["llama_peak_rss_kib"] / (1024 * 1024),
            "prompt_tps_mean": summary["llama_pp"]["mean"],
            "prompt_tps_sd": summary["llama_pp"]["sd"],
            "prompt_tps_cv_pct": summary["llama_pp"]["cv_pct"],
            "generation_tps_mean": summary["llama_tg"]["mean"],
            "generation_tps_sd": summary["llama_tg"]["sd"],
            "generation_tps_cv_pct": summary["llama_tg"]["cv_pct"],
            "commit": _read_text("llama_cpp_commit.txt"),
        },
        "comparison": {
            "rss_reduction_pct": summary["memvanta_rss_reduction_pct"],
            "llama_vs_memvanta_prompt_speedup": summary[
                "llama_vs_memvanta_pp_speedup"
            ],
            "llama_vs_memvanta_generation_speedup": summary[
                "llama_vs_memvanta_tg_speedup"
            ],
        },
        "provenance": {
            "memvanta_source_commit": _source_commit(),
            "source_artifact": "results/openllama-7b-v2-ab/summary.json",
            "methodology": "docs/MEMORY_BENCHMARKING.md",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
