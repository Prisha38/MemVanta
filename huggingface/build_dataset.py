from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "openllama-7b-v2-ab"
OUT = ROOT / "hf-dataset"


def _read_text(name: str) -> str:
    return (SOURCE / name).read_text(encoding="utf-8").strip()


def _dataset_card(summary: dict, source_commit: str, llama_commit: str) -> str:
    mem_rss_gib = summary["memvanta_peak_rss_kib"] / 1024 / 1024
    llama_rss_gib = summary["llama_peak_rss_kib"] / 1024 / 1024
    return f"""---
license: apache-2.0
pretty_name: MemVanta CPU LLM Memory Benchmark
size_categories:
- n<1K
tags:
- tabular
- llm
- gguf
- cpu-inference
- memory-efficiency
- quantization
- systems-benchmark
- llm-inference
- edge-ai
- reproducibility
configs:
- config_name: default
  data_files:
  - split: train
    path: data/openllama_7b_v2_q4_0.parquet
---

# MemVanta CPU LLM Memory Benchmark

**Reproducible CPU LLM inference benchmark evidence comparing memory usage and throughput for MemVanta and a pinned comparison runtime on the same GGUF artifact.**

The initial configuration is the committed **OpenLLaMA 7B v2 Q4_0 same-model CPU A/B** benchmark from the public MemVanta repository. This dataset publishes benchmark evidence and provenance only; it does **not** redistribute model weights.

## Canonical result

| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| Peak RSS | **{mem_rss_gib:.2f} GiB** | {llama_rss_gib:.2f} GiB |
| Prompt processing | {summary['memvanta_pp']['mean']:.2f} ± {summary['memvanta_pp']['sd']:.2f} tok/s | **{summary['llama_pp']['mean']:.2f} ± {summary['llama_pp']['sd']:.2f} tok/s** |
| Token generation | {summary['memvanta_tg']['mean']:.2f} ± {summary['memvanta_tg']['sd']:.2f} tok/s | **{summary['llama_tg']['mean']:.2f} ± {summary['llama_tg']['sd']:.2f} tok/s** |
| Peak-RSS reduction | **{summary['memvanta_rss_reduction_pct']:.2f}%** | baseline |

The result shows the measured trade-off for this specific benchmark: lower resident memory for MemVanta, with lower throughput than the pinned `llama.cpp` baseline. It is not a universal claim across models, machines, context sizes, quantizations, or runtimes.

## Dataset rows

The canonical Parquet contains one row per runtime with:

- benchmark and model identifiers;
- quantization and runtime role;
- peak RSS in KiB and GiB;
- prompt-processing throughput mean, standard deviation, and coefficient of variation;
- token-generation throughput mean, standard deviation, and coefficient of variation;
- measured MemVanta RSS reduction relative to the pinned comparison runtime;
- model artifact URL/hash metadata;
- MemVanta source commit and pinned `llama.cpp` commit;
- source evidence location.

## Evidence files

Selected raw evidence from the public benchmark is included under `evidence/openllama-7b-v2-ab/`, including:

- `summary.json`
- `environment.txt`
- `memvanta.csv`
- `memvanta.time.txt`
- `llama-bench.json`
- `llama-bench.time.txt`
- `llama_cpp_commit.txt`
- `model-size.txt`
- `model-url.txt`
- `model.sha256`
- `SUMMARY.md`
- `WORKFLOW_RUN.md`

The benchmark methodology is included as `evidence/MEMORY_BENCHMARKING.md`.

## Reproducibility

Generated from MemVanta source commit `{source_commit}`. The pinned comparison-runtime commit recorded by the benchmark is `{llama_commit}`.

Source repository: https://github.com/sauravsingla/MemVanta

Benchmark documentation: https://sauravsingla.github.io/MemVanta/benchmark/

Reproduction guide: https://sauravsingla.github.io/MemVanta/reproduce/

Zenodo DOI: https://doi.org/10.5281/zenodo.22886357

## Important limits

This dataset is systems benchmark evidence for a fixed model artifact, workload, host, and pinned comparison runtime. Peak RSS is not the same thing as a universal physical-RAM requirement. Throughput and memory measurements can vary with hardware, operating system, compiler, model file, context length, thread count, runtime revision, and benchmark procedure.

MemVanta is memory-first and does not claim to be faster than `llama.cpp`.

## License and upstream artifacts

MemVanta-generated code and benchmark artifacts are published under Apache-2.0. The model itself is not included here; its upstream license and distribution terms remain applicable.
"""


def main() -> None:
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    llama_commit = _read_text("llama_cpp_commit.txt")
    model_url = _read_text("model-url.txt")
    model_sha256 = _read_text("model.sha256").split()[0]
    model_size = _read_text("model-size.txt")
    source_commit = os.environ.get("GITHUB_SHA", "local")

    common = {
        "benchmark_id": "openllama-7b-v2-q4_0-same-model-cpu-ab",
        "benchmark_name": summary["benchmark"],
        "model": "OpenLLaMA 7B v2",
        "model_architecture": "llama",
        "quantization": "Q4_0",
        "model_url": model_url,
        "model_sha256": model_sha256,
        "model_size_recorded": model_size,
        "memvanta_source_commit": source_commit,
        "llama_cpp_commit": llama_commit,
        "source_artifact": "results/openllama-7b-v2-ab/summary.json",
    }

    rows = [
        {
            **common,
            "runtime": "MemVanta",
            "role": "candidate_memory_first_runtime",
            "peak_rss_kib": int(summary["memvanta_peak_rss_kib"]),
            "peak_rss_gib": summary["memvanta_peak_rss_kib"] / 1024 / 1024,
            "prompt_tokens_per_sec_mean": float(summary["memvanta_pp"]["mean"]),
            "prompt_tokens_per_sec_sd": float(summary["memvanta_pp"]["sd"]),
            "prompt_cv_pct": float(summary["memvanta_pp"]["cv_pct"]),
            "generation_tokens_per_sec_mean": float(summary["memvanta_tg"]["mean"]),
            "generation_tokens_per_sec_sd": float(summary["memvanta_tg"]["sd"]),
            "generation_cv_pct": float(summary["memvanta_tg"]["cv_pct"]),
            "rss_reduction_vs_llama_cpp_pct": float(summary["memvanta_rss_reduction_pct"]),
        },
        {
            **common,
            "runtime": "llama.cpp",
            "role": "pinned_comparison_runtime",
            "peak_rss_kib": int(summary["llama_peak_rss_kib"]),
            "peak_rss_gib": summary["llama_peak_rss_kib"] / 1024 / 1024,
            "prompt_tokens_per_sec_mean": float(summary["llama_pp"]["mean"]),
            "prompt_tokens_per_sec_sd": float(summary["llama_pp"]["sd"]),
            "prompt_cv_pct": float(summary["llama_pp"]["cv_pct"]),
            "generation_tokens_per_sec_mean": float(summary["llama_tg"]["mean"]),
            "generation_tokens_per_sec_sd": float(summary["llama_tg"]["sd"]),
            "generation_cv_pct": float(summary["llama_tg"]["cv_pct"]),
            "rss_reduction_vs_llama_cpp_pct": 0.0,
        },
    ]

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    evidence = OUT / "evidence" / "openllama-7b-v2-ab"
    evidence.mkdir(parents=True)

    pd.DataFrame(rows).to_parquet(
        OUT / "data" / "openllama_7b_v2_q4_0.parquet", index=False
    )

    selected = [
        "summary.json",
        "environment.txt",
        "memvanta.csv",
        "memvanta.time.txt",
        "llama-bench.json",
        "llama-bench.time.txt",
        "llama_cpp_commit.txt",
        "model-size.txt",
        "model-url.txt",
        "model.sha256",
        "SUMMARY.md",
        "WORKFLOW_RUN.md",
    ]
    for name in selected:
        shutil.copy2(SOURCE / name, evidence / name)

    methodology = ROOT / "docs" / "MEMORY_BENCHMARKING.md"
    if methodology.exists():
        shutil.copy2(methodology, OUT / "evidence" / methodology.name)

    (OUT / "README.md").write_text(
        _dataset_card(summary, source_commit, llama_commit), encoding="utf-8"
    )

    print(f"Wrote {len(rows)} canonical benchmark rows to {OUT}")
    print(f"MemVanta peak-RSS reduction: {summary['memvanta_rss_reduction_pct']:.2f}%")


if __name__ == "__main__":
    main()
