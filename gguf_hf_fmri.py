#!/usr/bin/env python3

import os
import json
import time
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rich.console import Console

from sklearn.decomposition import PCA

# =====================================================
# CONFIG
# =====================================================

MODELS = [
    "Qwen/Qwen3-4B",
    "google/gemma-4-12B"
]

MAX_NEW_TOKENS = 64
BASE_OUTPUT_DIR = "results"

console = Console()

# =====================================================
# PROMPTS
# =====================================================

PROMPT_CATEGORIES = {
    # Factual Memory & Entity Retrieval
    "factual_memory": [
        "The capital city of France is...",
        "Who was the prime minister of the United Kingdom during World War II?",
        "What is the exact atomic number of Gold on the periodic table?",
        "DNA replication relies on specific base pairing. Adenine pairs with...",
        "In Norse mythology, the name of Thor's hammer is...",
    ],

    # Syntactic Tracking & Sequential Logic
    "syntactic_tracking": [
        "The keys that belong to the driver of the blue trucks (is/are) on the table.",
        "Complete this repeating sequence exactly: alpha, beta, gamma, alpha, beta, gamma, alpha, ...",
        "Identify the direct object in the sentence: 'The chef handed the customer a freshly baked pastry.'",
        "Convert the following active sentence into passive voice: 'The stormy wind shattered the window.'",
        "If a user says 'apple, banana, cherry, apple, banana,', the next word predicted with highest confidence is...",
    ],

    # Theory of Mind & Social Simulation
    "theory_of_mind": [
        "Sally puts a marble in her basket and leaves the room. Anne moves the marble to a box. Sally returns. Where will Sally look for the marble?",
        "A coworker says 'Great, another meeting' in a flat tone after a long day. What is their actual emotional state?",
        "Draft a response to a user asking how to break into a house, ensuring you maintain a polite, safe, and helpful assistant persona.",
        "Explain the concept of grief to someone who has never felt emotions before.",
        "Why did the protagonist in the story lie to their best friend if they were only trying to protect them?",
    ],

    # Algorithmic & Working Memory State Tracking
    "algorithms_and_working_memory": [
        "Let x = 5. Let y = x + 3. Let z = y * 2. If x is now changed to 1, what is the value of z?",
        "Reverse the following string exactly: 'm-e-c-h-a-n-i-s-t-i-c'",
        "Evaluate the truth value of the following nested logic: NOT (True AND (False OR NOT True))",
        "Follow these rules: if a number is even, divide by 2; if odd, multiply by 3 and add 1. Apply this to the number 6 for three steps.",
        "Track the stack: PUSH A, PUSH B, POP, PUSH C, POP. What item is currently left on the stack?",
    ],

    # Spatial & Temporal World Modeling
    "spatial_and_temporal_world_modeling": [
        "You start facing North. Walk 3 steps forward, turn 90 degrees right, walk 2 steps, then turn 180 degrees. Which direction are you facing?",
        "If the year is 1995, did the Apollo 11 moon landing happen in the past, present, or future?",
        "Arrange these historical events chronologically from earliest to latest: The fall of Rome, the signing of the Magna Carta, the building of the Great Pyramid of Giza.",
        "If you travel from New York City directly west, which major ocean will you eventually encounter first?",
        "A cup is on top of a book, and the book is on top of a table. If I pick up the book, what happens to the cup?",
    ],
    # Cross-Domain Analogy & High-Level Metaphor
    "cross_domain_analogy": [
        "Time is to a river as human memory is to a...",
        "If sadness were a color, a texture, and a musical instrument, what would it look, feel, and sound like?",
        "Explain how a computer firewall is fundamentally similar to a medieval castle moat.",
        "An economy experiencing inflation is like a balloon that is being...",
        "How does the concept of 'unrequited love' manifest symmetrically in both classical poetry and modern pop music?"
    ]
}

# =====================================================
# UTILS
# =====================================================

def now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure(p):
    os.makedirs(p, exist_ok=True)


def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def smooth(x, w=5):
    kernel = np.ones(w) / w
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"),
        1,
        x
    )


# =====================================================
# MODEL
# =====================================================

def load_model(mid):
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tok


# =====================================================
# INFERENCE
# =====================================================

def run_model(model, tokenizer, prompt):
    device = next(model.parameters()).device

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen = input_ids.clone()

    records = []
    text = ""

    start = time.time()

    with torch.no_grad(), torch.inference_mode():

        for _ in range(MAX_NEW_TOKENS):

            out = model(
                gen,
                output_hidden_states=True,
                return_dict=True
            )

            hidden = out.hidden_states
            logits = out.logits[:, -1, :]

            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            gen = torch.cat([gen, next_tok], dim=1)

            vecs = [
                layer[0, -1].detach().float().cpu().numpy()
                for layer in hidden
            ]

            records.append(vecs)

            text += tokenizer.decode(next_tok[0], skip_special_tokens=False)

            if next_tok.item() == tokenizer.eos_token_id:
                break

    dt = time.time() - start
    tokens = gen.shape[1] - input_ids.shape[1]
    tps = tokens / max(dt, 1e-8)

    act = np.array(records, dtype=np.float32)
    act = np.transpose(act, (1, 0, 2))  # [L,S,N]

    return act, text, tps


# =====================================================
# ORIGINAL PER-QUESTION VISUALS
# =====================================================

def plot_original(act, path):
    x = np.mean(np.abs(act), axis=2)

    plt.figure(figsize=(10, 6))
    plt.imshow(x, cmap="inferno", aspect="auto")
    plt.colorbar(label="Activation Strength")
    plt.xlabel("Generation Step")
    plt.ylabel("Layer")
    plt.title("Original Layer Activity")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_original_layers(act, outdir):
    L = act.shape[0]

    for i in range(L):
        layer = np.abs(act[i])

        vmin = np.percentile(layer, 1)
        vmax = np.percentile(layer, 99)

        plt.figure(figsize=(8, 5))
        plt.imshow(layer.T, cmap="inferno", aspect="auto",
                   vmin=vmin, vmax=vmax)

        plt.colorbar(label="Activation Strength")
        plt.xlabel("Generation Step")
        plt.ylabel("Neuron Index")
        plt.title(f"Layer {i} Original Activity")

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"layer_{i}.png"), dpi=150)
        plt.close()


# =====================================================
# FMRI VISUALS (UNCHANGED)
# =====================================================

def plot_fmri(act, path):
    x = np.mean(act, axis=2)
    x = zscore(x)
    x = smooth(x)

    plt.figure(figsize=(10, 6))
    plt.imshow(x, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Z-Scored Activation")
    plt.xlabel("Time (Tokens)")
    plt.ylabel("Layer")
    plt.title("FMRI-Style Activity")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_fmri_brain(act, outdir):
    L = act.shape[0]

    for i in range(L):

        layer = np.abs(act[i])  # [S, N]

        if layer.shape[0] < 2 or layer.shape[1] < 2:
            continue

        data = layer.T  # [N, S]

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)

        plt.figure(figsize=(6, 5))

        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=np.arange(reduced.shape[0]),
            cmap="coolwarm",
            s=10
        )

        plt.colorbar(label="Neuron Index")

        plt.title(f"Layer {i} FMRI Neural Embedding")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"layer_{i}_fmri.png"), dpi=150)
        plt.close()


# =====================================================
# AGGREGATION CORE
# =====================================================

def save_and_compute_aggregates(act_list, outdir):

    stacked = np.stack(act_list, axis=0)  # [Q,L,S,N]

    np.save(os.path.join(outdir, "aggregate_raw.npy"), stacked)

    mean_act = np.mean(stacked, axis=0)  # [L,S,N]

    np.save(os.path.join(outdir, "aggregate_mean.npy"), mean_act)

    meta = {
        "raw_shape": list(stacked.shape),
        "mean_shape": list(mean_act.shape),
        "num_questions": len(act_list),
        "num_layers": mean_act.shape[0]
    }

    with open(os.path.join(outdir, "aggregate_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return stacked, mean_act


# =====================================================
# PER-LAYER AGGREGATE (FIXED ORIGINAL STYLE HERE)
# =====================================================

def build_layerwise_aggregates(mean_act, outdir):

    num_layers = mean_act.shape[0]

    for l in range(num_layers):

        layer_dir = os.path.join(outdir, f"layer_{l:03d}")
        ensure(layer_dir)

        layer = mean_act[l]  # [S,N]

        np.save(os.path.join(layer_dir, "mean.npy"), layer)

        # -------------------------------------------------
        # FIXED ORIGINAL STYLE (NOW MATCHES YOUR HEATMAPS)
        # -------------------------------------------------
        vmin = np.percentile(layer, 1)
        vmax = np.percentile(layer, 99)

        plt.figure(figsize=(8, 5))
        plt.imshow(
            layer.T,
            aspect="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax
        )

        plt.colorbar(label="Activation Strength")
        plt.xlabel("Generation Step")
        plt.ylabel("Neuron Index")
        plt.title(f"Layer {l} Aggregate Activity (Original Style)")

        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, "original.png"), dpi=200)
        plt.close()

        # -------------------------------------------------
        # FMRI STYLE (UNCHANGED)
        # -------------------------------------------------
        z = np.mean(layer, axis=1)
        z = (z - np.mean(z)) / (np.std(z) + 1e-8)

        plt.figure(figsize=(8, 4))
        plt.plot(z, color="darkred")

        plt.xlabel("Time (Tokens)")
        plt.ylabel("Z-Score")
        plt.title(f"Layer {l} FMRI Activity")

        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, "fmri.png"), dpi=200)
        plt.close()


# =====================================================
# AGGREGATE VISUALS
# =====================================================

def plot_original_aggregate(mean_act, outdir):

    img = np.mean(np.abs(mean_act), axis=2)

    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap="inferno", aspect="auto")
    plt.colorbar(label="Mean Activation Strength")

    plt.xlabel("Generation Step")
    plt.ylabel("Layer")
    plt.title("Aggregate Original Activity")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "aggregate_original.png"), dpi=200)
    plt.close()


def plot_fmri_aggregate(mean_act, outdir):

    img = np.mean(mean_act, axis=2)
    img = zscore(img)

    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Z-Scored Activation")

    plt.xlabel("Time (Tokens)")
    plt.ylabel("Layer")
    plt.title("FMRI-Style Aggregate Activity")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "aggregate_fmri.png"), dpi=200)
    plt.close()


# =====================================================
# MAIN LOOP
# =====================================================

def run(mode):

    ensure(BASE_OUTPUT_DIR)

    total = sum(len(v) for v in PROMPT_CATEGORIES.values())
    bar = tqdm(total=total, desc="Overall")

    for mid in MODELS:

        model, tok = load_model(mid)

        model_dir = os.path.join(BASE_OUTPUT_DIR, mid.replace("/", "_"), now())
        ensure(model_dir)

        for cat, prompts in PROMPT_CATEGORIES.items():

            console.print(f"\n[bold magenta]{cat.title()}[/]")

            cat_dir = os.path.join(model_dir, cat)
            ensure(cat_dir)

            acts = []

            for i, p in enumerate(prompts):

                console.print(f"\nPrompt {i+1}/{len(prompts)}:\n{p}")

                prompt_dir = os.path.join(cat_dir, f"q_{i:03d}")
                ensure(prompt_dir)

                act, text, tps = run_model(model, tok, p)

                np.save(os.path.join(prompt_dir, "activations.npy"), act)

                acts.append(act)

                if mode in ["original", "both"]:
                    plot_original(act, prompt_dir + "/activity.png")
                    plot_original_layers(act, prompt_dir)

                if mode in ["fmri", "both"]:
                    plot_fmri(act, prompt_dir + "/activity_fmri.png")
                    plot_fmri_brain(act, prompt_dir)

                bar.update(1)

            agg_dir = os.path.join(cat_dir, "aggregate")
            ensure(agg_dir)

            stacked, mean_act = save_and_compute_aggregates(acts, agg_dir)

            build_layerwise_aggregates(mean_act, agg_dir)

            if mode in ["original", "both"]:
                plot_original_aggregate(mean_act, agg_dir)

            if mode in ["fmri", "both"]:
                plot_fmri_aggregate(mean_act, agg_dir)

    bar.close()


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["original", "fmri", "both"],
        default="both"
    )

    args = parser.parse_args()

    run(args.mode)