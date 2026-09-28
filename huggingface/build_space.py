from __future__ import annotations

import html
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "openllama-7b-v2-ab"
OUT = ROOT / "hf-space" / "data.json"
INDEX = ROOT / "hf-space" / "index.html"


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


def _replace_element_text(document: str, element_id: str, value: str) -> str:
    """Replace the initial text node for a simple element identified by id.

    The Space still hydrates from data.json in JavaScript. Pre-rendering the same
    canonical values here makes benchmark evidence visible to crawlers, link
    previewers, accessibility tools, and browsers before JavaScript executes.
    """
    pattern = re.compile(
        rf'(<[^>]+\bid=["\']{re.escape(element_id)}["\'][^>]*>)(.*?)(</[^>]+>)',
        flags=re.DOTALL,
    )
    escaped = html.escape(value, quote=False)
    rendered, count = pattern.subn(rf"\g<1>{escaped}\g<3>", document, count=1)
    if count != 1:
        raise RuntimeError(f"Could not pre-render element #{element_id}")
    return rendered


def _render_index(payload: dict) -> None:
    document = INDEX.read_text(encoding="utf-8")
    mv = payload["memvanta"]
    lc = payload["llama_cpp"]
    comparison = payload["comparison"]
    provenance = payload["provenance"]

    values = {
        "rssReduction": f"{comparison['rss_reduction_pct']:.2f}%",
        "mvRss": f"{mv['peak_rss_gib']:.2f} GiB",
        "lcRss": f"{lc['peak_rss_gib']:.2f} GiB",
        "mvPrompt": f"{mv['prompt_tps_mean']:.2f} tok/s",
        "mvGen": f"{mv['generation_tps_mean']:.2f} tok/s",
        "mvRssBarText": f"{mv['peak_rss_gib']:.2f} GiB",
        "lcRssBarText": f"{lc['peak_rss_gib']:.2f} GiB",
        "mvPromptBarText": f"{mv['prompt_tps_mean']:.2f}",
        "lcPromptBarText": f"{lc['prompt_tps_mean']:.2f}",
        "mvGenBarText": f"{mv['generation_tps_mean']:.2f}",
        "lcGenBarText": f"{lc['generation_tps_mean']:.2f}",
        "memorySentence": (
            f"MemVanta measured {comparison['rss_reduction_pct']:.2f}% lower peak RSS."
        ),
        "promptSentence": (
            f"llama.cpp was {comparison['llama_vs_memvanta_prompt_speedup']:.2f}× "
            "faster in prompt processing."
        ),
        "genSentence": (
            f"llama.cpp was {comparison['llama_vs_memvanta_generation_speedup']:.2f}× "
            "faster in token generation."
        ),
        "benchmarkName": str(payload["benchmark"]),
        "modelFile": str(payload["model"]["file"]),
        "modelSha": str(payload["model"]["sha256"]),
        "llamaCommit": str(lc["commit"]),
        "mvCommit": str(provenance["memvanta_source_commit"]),
        "sourceArtifact": str(provenance["source_artifact"]),
    }

    for element_id, value in values.items():
        document = _replace_element_text(document, element_id, value)

    INDEX.write_text(document, encoding="utf-8")


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
    _render_index(payload)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Pre-rendered benchmark evidence into {INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
