# 🧠  BRAIN_DMP_V2

A lightweight diagnostic tool for inspecting the internal activations of locally-run GGUF language models — think of it as an fMRI machine for an LLM's "brain." It runs a prompt through the model and records per-layer, per-token activation statistics, then visualizes them as heatmaps.

This version is **GPU-accelerated**, offloading model layers to an NVIDIA GPU via `llama-cpp-python`'s CUDA backend for significantly faster inference during capture.

## ✨ What it does

- Loads any local `.gguf` model and runs it on an NVIDIA GPU
- Captures activation statistics (mean/max magnitude, L2 norm) for every tensor in the network, layer by layer, token by token
- Falls back gracefully to a stable public-API capture mode if low-level hooks aren't available in your `llama-cpp-python` build
- Saves raw data (`.npz` + `.json`) and renders two heatmaps per run:
  - 📊 Activation magnitude across all layers over generation steps
  - 🔥 Top-N most "active" layers/tensors for the prompt

## ⚙️ Setup

**1. Install dependencies:**
```bash
pip install numpy matplotlib
```

**2. Install `llama-cpp-python` with CUDA support** (the default pip wheel is CPU-only):
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir --upgrade
```
Requires the NVIDIA CUDA toolkit and `nvcc` available on `PATH`. On older `llama-cpp-python` versions, use `-DLLAMA_CUBLAS=on` instead.

**3. Add models:**
Place `.gguf` model files in an `llms/` folder next to the script.

## 🚀 Usage

```bash
python gguf_neuron_fmri_gpu.py
```

You'll be prompted to select a model and either run an interactive prompt or a predefined batch of test questions. Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | (interactive picker) | Path to a specific `.gguf` file |
| `--n-ctx` | `2048` | Context window size |
| `--max-tokens` | `128` | Max tokens to generate |
| `--n-gpu-layers` | `-1` | Layers to offload to GPU (`-1` = all, `0` = CPU only) |

On startup, the script checks for an NVIDIA GPU and confirms whether `llama-cpp-python` was built with CUDA support, so any GPU misconfiguration is reported immediately rather than failing silently.

## 📁 Output

Each run produces a timestamped folder under `fmri_output/` containing:
- `activations.npz` — raw activation statistics
- `metadata.json` — prompt, response, and tensor metadata
- `heatmap_layers_by_step.png` — activation over time, all layers
- `heatmap_top_layers.png` — most active layers, ranked

## 🔍 Why this might be useful

This was built out of curiosity about what's actually happening inside an LLM as it generates a response — which layers light up for which kinds of prompts, how activation patterns shift token-by-token, and whether that's visible without needing access to model internals beyond what's exposed in a standard GGUF runtime. Happy to walk through the approach or extend it for a specific research question.
