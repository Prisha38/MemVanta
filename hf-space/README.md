---
title: MemVanta — CPU LLM Memory Benchmark
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Interactive CPU LLM memory and throughput benchmark explorer
---

# MemVanta — CPU LLM Memory Benchmark

Interactive browser-only explorer for MemVanta's canonical OpenLLaMA 7B v2 Q4_0 same-model CPU A/B benchmark against pinned `llama.cpp`.

The evidence payload is generated directly from committed benchmark artifacts before each publish. The Space visualizes peak RSS, prompt-processing throughput, token-generation throughput, reproducibility metadata, and the measured memory/throughput trade-off.

Relevant Space and benchmark-evidence changes on `main` are synchronized automatically through Hugging Face Trusted Publishing.

This Space does **not** run the 7B model in the browser and does not claim the measured result generalizes to every model, CPU, workload, runtime revision, or memory configuration.

- Source: https://github.com/sauravsingla/MemVanta
- Dataset: https://huggingface.co/datasets/sauravsingla08/MemVanta-CPU-LLM-Memory-Benchmark
- PyPI: https://pypi.org/project/memvanta/
- DOI: https://doi.org/10.5281/zenodo.22886357

Apache-2.0. Upstream model and dataset licenses/terms remain applicable to their original artifacts.
