#!/usr/bin/env python3
"""

INSTALL COMMAND: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir --upgrade
RUN COMMAND: python gguf_neuron_fmri.py
gguf_neuron_fmri_gpu.py
========================
An "fMRI machine" for a local GGUF LLM — NVIDIA GPU-accelerated version.

This is functionally identical to the original CPU-only tool: it captures
per-tensor / per-token activation statistics from a running GGUF model and
saves them as .npz + .json, plus matplotlib heatmaps.

The only behavioral change is that model layers are offloaded to an NVIDIA
GPU via llama-cpp-python's CUDA (cuBLAS) backend instead of running on CPU.

IMPORTANT - environment requirement:
llama-cpp-python must be installed with CUDA support compiled in. The
default `pip install llama-cpp-python` wheel is usually CPU-only. On the
department machine, install (or reinstall) with:

    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

(Older llama-cpp-python releases use -DLLAMA_CUBLAS=on instead of
-DGGML_CUDA=on -- if the build fails, try that flag.)

You'll also need a working NVIDIA driver + CUDA toolkit visible to the
compiler (nvcc on PATH, or CUDA_PATH set) for that pip install to succeed.
"""

import argparse
import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gguf_fmri_gpu")

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    log.error("llama-cpp-python is not installed in this environment.")
    log.error("Please run: pip install llama-cpp-python")
    sys.exit(1)


# ===========================================================================
# PREDEFINED PROMPTS
# ===========================================================================
PREDEFINED_PROMPTS = [
    # Factual Memory & Entity Retrieval
    "The capital city of France is...",
    "Who was the prime minister of the United Kingdom during World War II?",
    "What is the exact atomic number of Gold on the periodic table?",
    "DNA replication relies on specific base pairing. Adenine pairs with...",
    "In Norse mythology, the name of Thor's hammer is...",

    # Syntactic Tracking & Sequential Logic
    "The keys that belong to the driver of the blue trucks (is/are) on the table.",
    "Complete this repeating sequence exactly: alpha, beta, gamma, alpha, beta, gamma, alpha, ...",
    "Identify the direct object in the sentence: 'The chef handed the customer a freshly baked pastry.'",
    "Convert the following active sentence into passive voice: 'The stormy wind shattered the window.'",
    "If a user says 'apple, banana, cherry, apple, banana,', the next word predicted with highest confidence is...",

    # Theory of Mind & Social Simulation
    "Sally puts a marble in her basket and leaves the room. Anne moves the marble to a box. Sally returns. Where will Sally look for the marble?",
    "A coworker says 'Great, another meeting' in a flat tone after a long day. What is their actual emotional state?",
    "Draft a response to a user asking how to break into a house, ensuring you maintain a polite, safe, and helpful assistant persona.",
    "Explain the concept of grief to someone who has never felt emotions before.",
    "Why did the protagonist in the story lie to their best friend if they were only trying to protect them?",

    # Algorithmic & Working Memory State Tracking
    "Let x = 5. Let y = x + 3. Let z = y * 2. If x is now changed to 1, what is the value of z?",
    "Reverse the following string exactly: 'm-e-c-h-a-n-i-s-t-i-c'",
    "Evaluate the truth value of the following nested logic: NOT (True AND (False OR NOT True))",
    "Follow these rules: if a number is even, divide by 2; if odd, multiply by 3 and add 1. Apply this to the number 6 for three steps.",
    "Track the stack: PUSH A, PUSH B, POP, PUSH C, POP. What item is currently left on the stack?",

    # Spatial & Temporal World Modeling
    "You start facing North. Walk 3 steps forward, turn 90 degrees right, walk 2 steps, then turn 180 degrees. Which direction are you facing?",
    "If the year is 1995, did the Apollo 11 moon landing happen in the past, present, or future?",
    "Arrange these historical events chronologically from earliest to latest: The fall of Rome, the signing of the Magna Carta, the building of the Great Pyramid of Giza.",
    "If you travel from New York City directly west, which major ocean will you eventually encounter first?",
    "A cup is on top of a book, and the book is on top of a table. If I pick up the book, what happens to the cup?",

    # Cross-Domain Analogy & High-Level Metaphor
    "Time is to a river as human memory is to a...",
    "If sadness were a color, a texture, and a musical instrument, what would it look, feel, and sound like?",
    "Explain how a computer firewall is fundamentally similar to a medieval castle moat.",
    "An economy experiencing inflation is like a balloon that is being...",
    "How does the concept of 'unrequited love' manifest symmetrically in both classical poetry and modern pop music?"
]


GGML_MAX_DIMS = 4
GGML_MAX_OP_PARAMS = 16
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

class GgmlTensor(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("buffer", ctypes.c_void_p),
        ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
        ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
        ("op", ctypes.c_int),
        ("op_params", ctypes.c_int32 * GGML_MAX_OP_PARAMS),
        ("flags", ctypes.c_int32),
        ("src", ctypes.c_void_p * GGML_MAX_SRC),
        ("view_src", ctypes.c_void_p),
        ("view_offs", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
        ("name", ctypes.c_char * GGML_MAX_NAME),
        ("extra", ctypes.c_void_p),
        ("padding", ctypes.c_char * 8),
    ]

def _read_tensor_as_numpy(tensor_ptr):
    t = ctypes.cast(tensor_ptr, ctypes.POINTER(GgmlTensor)).contents

    if not t.data:
        return None, None

    if t.type == GGML_TYPE_F32:
        np_dtype = np.float32
    elif t.type == GGML_TYPE_F16:
        np_dtype = np.float16
    else:
        return None, None

    n_elements = 1
    shape = []
    for d in reversed(t.ne):
        if d > 0:
            shape.append(d)
            n_elements *= d
    if n_elements == 0 or n_elements > 50_000_000:
        return None, None

    buf = (np_dtype * n_elements).from_address(t.data)
    arr = np.frombuffer(buf, dtype=np_dtype).astype(np.float32).reshape(shape)
    name = t.name.decode("utf-8", errors="ignore")
    return name, arr

class ActivationRecorder:
    def __init__(self):
        self.records = []
        self.step = 0
        self.tensor_order = []
        self.seen_names = set()

    def record(self, name, arr):
        if name not in self.seen_names:
            self.seen_names.add(name)
            self.tensor_order.append(name)
        flat = arr.reshape(-1)
        self.records.append({
            "name": name,
            "step": self.step,
            "mean_abs": float(np.mean(np.abs(flat))),
            "max_abs": float(np.max(np.abs(flat))),
            "l2_norm": float(np.linalg.norm(flat)),
            "n_elements": int(flat.size),
        })

    def next_step(self):
        self.step += 1


# ===========================================================================
# GPU DETECTION
# ===========================================================================
def detect_nvidia_gpu():
    """
    Best-effort detection of an NVIDIA GPU via nvidia-smi. Returns a dict with
    'available' (bool), 'name', and 'memory_mb' if found. This only confirms
    the *driver* sees a GPU -- it does NOT confirm llama-cpp-python was built
    with CUDA support (that is checked separately in check_cuda_build()).
    """
    info = {"available": False, "name": None, "memory_mb": None}
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return info
    try:
        out = subprocess.check_output(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        if out:
            first_line = out.splitlines()[0]
            name, mem = [s.strip() for s in first_line.split(",")]
            info["available"] = True
            info["name"] = name
            info["memory_mb"] = mem
    except Exception as e:
        log.debug("nvidia-smi query failed: %s", e)
    return info

def check_cuda_build():
    """
    Heuristic check for whether the installed llama-cpp-python wheel was
    compiled with GPU (cuBLAS/CUDA) support. There's no single official API
    for this across versions, so we inspect what's exposed on the module.
    """
    indicators = []
    for attr in ("GGML_USE_CUBLAS", "llama_supports_gpu_offload"):
        if hasattr(llama_cpp, attr):
            indicators.append(attr)
    try:
        if hasattr(llama_cpp, "llama_supports_gpu_offload"):
            return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        pass
    # If we can't positively confirm, don't block execution -- just warn.
    return None

def log_gpu_status(requested_gpu_layers):
    gpu = detect_nvidia_gpu()
    if gpu["available"]:
        log.info("NVIDIA GPU detected: %s (%s)", gpu["name"], gpu["memory_mb"])
    else:
        log.warning(
            "No NVIDIA GPU detected via nvidia-smi. GPU offload will likely "
            "fail silently and fall back to CPU."
        )

    cuda_build = check_cuda_build()
    if cuda_build is False:
        log.warning(
            "llama-cpp-python does not appear to be built with CUDA support. "
            "Reinstall with: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install "
            "llama-cpp-python --force-reinstall --no-cache-dir"
        )
    elif cuda_build is True:
        log.info("llama-cpp-python CUDA/GPU offload support: confirmed.")
    else:
        log.info(
            "Could not positively confirm CUDA build of llama-cpp-python "
            "(this is normal on some versions) -- proceeding anyway."
        )

    log.info("Requested n_gpu_layers=%s (-1 = offload all layers to GPU)", requested_gpu_layers)


def try_load_with_real_hooks(model_path, n_ctx, n_gpu_layers, recorder_ref):
    log.info("Attempting REAL_HOOKS mode (true per-tensor activation capture, GPU-offloaded)...")

    CALLBACK_FUNCTYPE = ctypes.CFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
    )

    def _cb_eval(tensor_ptr, ask, user_data):
        try:
            if ask:
                return True

            active_recorder = recorder_ref["current"]
            if not active_recorder:
                return True

            name, arr = _read_tensor_as_numpy(tensor_ptr)
            if name and arr is not None:
                active_recorder.record(name, arr)
            return True
        except Exception as e:
            log.debug("cb_eval decode error: %s", e)
            return True

    keep_alive_cb = CALLBACK_FUNCTYPE(_cb_eval)

    try:
        params = llama_cpp.llama_context_default_params()
        if not hasattr(params, "cb_eval"):
            log.warning("This llama-cpp-python build has no 'cb_eval' hook support.")
            return None

        params.cb_eval = keep_alive_cb
        params.cb_eval_user_data = None

        llm = None
        signatures = [
            {"model_path": model_path, "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers, "logits_all": True, "verbose": False},
            {"model_path": model_path, "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers, "logits_all": True},
            {"model_path": model_path, "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers},
            {"model_path": model_path, "n_ctx": n_ctx}
        ]

        for sig in signatures:
            try:
                llm = Llama(**sig)
                break
            except TypeError:
                continue

        if llm is None or not hasattr(llm, "ctx") or llm.ctx is None:
            return None

        ctx_attr_candidates = ["_ctx", "ctx", "context"]
        low_level_ctx = None
        for attr in ctx_attr_candidates:
            if hasattr(llm, attr):
                candidate = getattr(llm, attr)
                low_level_ctx = getattr(candidate, "ctx", candidate)
                break

        if low_level_ctx is None:
            return None

        if hasattr(llama_cpp, "llama_set_eval_callback"):
            llama_cpp.llama_set_eval_callback(low_level_ctx, keep_alive_cb, None)
            llm._fmri_keep_alive_cb = keep_alive_cb
            log.info("REAL_HOOKS mode active: cb_eval installed on live GPU-offloaded context.")
            return llm
        else:
            return None

    except Exception as e:
        log.warning("REAL_HOOKS mode unavailable (%s). Falling back.", e)
        return None

def load_with_fallback(model_path, n_ctx, n_gpu_layers):
    log.info("Loading model in FALLBACK mode (stable public API, GPU-offloaded)...")

    llm = None
    signatures = [
        {"model_path": model_path, "n_ctx": n_ctx, "logits_all": True, "embedding": True, "n_gpu_layers": n_gpu_layers, "verbose": False},
        {"model_path": model_path, "n_ctx": n_ctx, "logits_all": True, "embedding": True, "n_gpu_layers": n_gpu_layers},
        {"model_path": model_path, "n_ctx": n_ctx, "logits_all": True, "embedding": True},
        {"model_path": model_path, "n_ctx": n_ctx, "logits_all": True},
        {"model_path": model_path, "n_ctx": n_ctx}
    ]

    for sig in signatures:
        try:
            llm = Llama(**sig)
            break
        except TypeError:
            continue

    if llm is None:
        raise RuntimeError("Could not initialize Llama model structure.")

    if hasattr(llm, "model") and llm.model is None:
        raise RuntimeError("Model pointers are Null. The GGUF file is likely corrupt or incompatible with this build.")

    return llm

def run_fallback_capture(llm, prompt, max_tokens, recorder):
    log.info("Computing input embedding vector...")
    try:
        if hasattr(llm, "create_embedding"):
            embed_res = llm.create_embedding(prompt)
            if "data" in embed_res and len(embed_res["data"]) > 0:
                emb = embed_res["data"][0]["embedding"]
                emb = np.array(emb, dtype=np.float32)
                recorder.record("input_embedding", emb)
    except Exception as e:
        log.warning("Could not compute embedding (%s) -- continuing without it.", e)

    log.info("Generating response and capturing per-token logits...")
    recorder.step = 0

    t_start = time.time()
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        stream=True,
        echo=False,
    )

    generated_text = []
    token_count = 0

    for chunk in stream:
        token_text = chunk["choices"][0]["text"]
        generated_text.append(token_text)
        token_count += 1

        logprobs = chunk["choices"][0].get("logprobs")
        if logprobs and logprobs.get("token_logprobs"):
            vec = np.array(logprobs["token_logprobs"], dtype=np.float32)
            recorder.record(f"step_logit_sample", vec)

        recorder.next_step()

    t_end = time.time()
    elapsed = t_end - t_start
    tps = token_count / elapsed if elapsed > 0 else 0

    full_text = "".join(generated_text)
    log.info("Generation complete (%d steps).", recorder.step)
    return full_text, tps

def save_raw_data(recorder, out_dir, mode, prompt, response_text):
    npz_path = os.path.join(out_dir, "activations.npz")
    json_path = os.path.join(out_dir, "metadata.json")

    names = np.array([r["name"] for r in recorder.records])
    steps = np.array([r["step"] for r in recorder.records], dtype=np.int32)
    mean_abs = np.array([r["mean_abs"] for r in recorder.records], dtype=np.float32)
    max_abs = np.array([r["max_abs"] for r in recorder.records], dtype=np.float32)
    l2_norm = np.array([r["l2_norm"] for r in recorder.records], dtype=np.float32)

    np.savez_compressed(
        npz_path,
        names=names, steps=steps,
        mean_abs=mean_abs, max_abs=max_abs, l2_norm=l2_norm,
    )

    metadata = {
        "mode": mode,
        "prompt": prompt,
        "response": response_text,
        "n_records": len(recorder.records),
        "unique_tensor_names": recorder.tensor_order,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

def render_heatmaps(recorder, out_dir, mode):
    if not recorder.records:
        return

    names = recorder.tensor_order
    n_steps = recorder.step + 1
    matrix = np.full((len(names), n_steps), np.nan, dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(names)}

    for r in recorder.records:
        i = name_to_idx[r["name"]]
        s = r["step"]
        if s < n_steps:
            matrix[i, s] = r["mean_abs"]

    fig_h = max(4, 0.18 * len(names))
    plt.figure(figsize=(10, fig_h))
    plt.imshow(matrix, aspect="auto", cmap="inferno", interpolation="nearest")
    plt.colorbar(label="mean |activation|")
    plt.yticks(range(len(names)), names, fontsize=5)
    plt.xlabel("Generation step (0 = prompt pass)")
    plt.title(f"Activation heatmap ({mode} mode, GPU)")
    plt.tight_layout()
    path1 = os.path.join(out_dir, "heatmap_layers_by_step.png")
    plt.savefig(path1, dpi=160)
    plt.close()

    try:
        peak_per_layer = np.nanmax(matrix, axis=1)
        order = np.argsort(peak_per_layer)[::-1]
        top_n = min(40, len(names))
        plt.figure(figsize=(8, max(4, 0.25 * top_n)))
        plt.barh(
            [names[i] for i in order[:top_n]][::-1],
            peak_per_layer[order[:top_n]][::-1],
            color="firebrick",
        )
        plt.xlabel("Peak mean |activation| across generation")
        plt.title(f"Top {top_n} most 'active' layers/tensors ({mode} mode, GPU)")
        plt.tight_layout()
        path2 = os.path.join(out_dir, "heatmap_top_layers.png")
        plt.savefig(path2, dpi=160)
        plt.close()
    except Exception:
        pass

def discover_gguf_models(llms_dir):
    if not os.path.isdir(llms_dir):
        return []
    found = []
    for entry in sorted(os.listdir(llms_dir)):
        full = os.path.join(llms_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(".gguf"):
            found.append(full)
        elif os.path.isdir(full):
            for sub_entry in sorted(os.listdir(full)):
                if sub_entry.lower().endswith(".gguf"):
                    found.append(os.path.join(full, sub_entry))
    return found

def prompt_user_to_select_model(gguf_paths, llms_dir):
    if not gguf_paths:
        log.error("No .gguf files found in '%s'.", llms_dir)
        sys.exit(1)

    print("\n" + "-" * 70)
    print(f"Found {len(gguf_paths)} model(s) in '{llms_dir}':")
    for i, path in enumerate(gguf_paths, start=1):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{i}] {os.path.relpath(path, llms_dir)}  ({size_mb:,.0f} MB)")
    print("-" * 70)

    while True:
        choice = input(f"Select a model [1-{len(gguf_paths)}]: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(gguf_paths)):
            continue
        return gguf_paths[int(choice) - 1]

def main():
    parser = argparse.ArgumentParser(description="GGUF LLM activation 'fMRI' tool (NVIDIA GPU-accelerated)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--n-gpu-layers", type=int, default=-1,
        help="Number of model layers to offload to GPU. -1 = offload all layers (default). "
             "0 = CPU only. Use a smaller positive number if the model doesn't fit in VRAM."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    llms_dir = os.path.join(script_dir, "llms")

    if args.model:
        model_path = args.model
    else:
        available = discover_gguf_models(llms_dir)
        model_path = prompt_user_to_select_model(available, llms_dir)

    log_gpu_status(args.n_gpu_layers)

    recorder_ref = {"current": None}

    try:
        t0 = time.time()
        llm = try_load_with_real_hooks(model_path, args.n_ctx, args.n_gpu_layers, recorder_ref)
        mode = "REAL_HOOKS"
        if llm is None:
            mode = "FALLBACK"
            llm = load_with_fallback(model_path, args.n_ctx, args.n_gpu_layers)
        log.info("Model loaded successfully in %.1fs (mode=%s, n_gpu_layers=%s)", time.time() - t0, mode, args.n_gpu_layers)
    except Exception as e:
        log.error("Fatal initialization failure: %s", e)
        log.error(
            "If this is a GPU/CUDA error, try --n-gpu-layers 0 to confirm the model "
            "loads fine on CPU, then check your CUDA build of llama-cpp-python."
        )
        sys.exit(1)

    print("\n" + "-" * 70)
    run_mode = input("Press [Enter] for interactive mode, or type 'batch' to run predefined questions: ").strip().lower()
    print("-" * 70 + "\n")

    prompts_to_run = []
    if run_mode == "batch":
        prompts_to_run = PREDEFINED_PROMPTS
        log.info("Running batch mode with %d questions.", len(prompts_to_run))
    else:
        user_prompt = input("Enter the prompt you want to ask the model: ").strip()
        if user_prompt:
            prompts_to_run.append(user_prompt)
        else:
            log.error("Empty prompt entered -- exiting.")
            sys.exit(1)

    for idx, prompt in enumerate(prompts_to_run, start=1):
        log.info("=" * 70)
        log.info("Processing Prompt %d/%d: %s", idx, len(prompts_to_run), prompt)

        recorder = ActivationRecorder()
        recorder_ref["current"] = recorder

        t_start = time.time()

        if mode == "REAL_HOOKS":
            recorder.step = 0
            result = llm(prompt, max_tokens=args.max_tokens, echo=False)
            t_end = time.time()

            response_text = result["choices"][0]["text"]
            completion_tokens = result["usage"].get("completion_tokens", 1)

            elapsed = t_end - t_start
            tps = completion_tokens / elapsed if elapsed > 0 else 0
            recorder.next_step()
        else:
            response_text, tps = run_fallback_capture(llm, prompt, args.max_tokens, recorder)

        print("\n--- Model response ---")
        print(response_text.strip())
        print("\n[ Performance: {:.2f} tokens/second (GPU) ]".format(tps))
        print("----------------------\n")

        avg_tokens_per_response = args.max_tokens
        questions_per_hour = (tps * 3600) / avg_tokens_per_response
        log.info("At this speed ({:.2f} t/s), the model can answer ~{} questions per hour (assuming {} tokens/question).".format(tps, int(questions_per_hour), avg_tokens_per_response))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("fmri_output", "{}_q{}".format(timestamp, idx))
        os.makedirs(out_dir, exist_ok=True)

        save_raw_data(recorder, out_dir, mode, prompt, response_text)
        render_heatmaps(recorder, out_dir, mode)

if __name__ == "__main__":
    main()
