"""
run_gpt2_benchmarks.py
======================
Automated benchmark runner for GPT-2 Small circuit interpretability.
Executes quantitative evaluations:
1. Induction Head Detection across all 144 heads (12 layers x 12 heads) on repeated tokens.
2. Low-Rank OV Copying Eigenvalue Spectrum analysis across all 144 heads.
3. Direct Logit Attribution (DLA) on prompt "The cat sat on the mat. The cat" -> " sat".
4. Saves full empirical metrics to results/gpt2_metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import torch as t
from einops import rearrange
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2-small"
SEQ_LEN = 50
SEED = 0


def generate_repeated_tokens(model: HookedTransformer, seq_len: int = SEQ_LEN) -> t.Tensor:
    t.manual_seed(SEED)
    bos = int(model.tokenizer.bos_token_id)
    random_tokens = t.randint(
        low=0,
        high=model.cfg.d_vocab,
        size=(seq_len,),
        dtype=t.long,
        device=model.cfg.device,
    )
    tokens = t.cat([
        t.tensor([bos], dtype=t.long, device=model.cfg.device),
        random_tokens,
        random_tokens.clone(),
    ])
    return tokens.unsqueeze(0)


def compute_induction_scores(model: HookedTransformer, tokens: t.Tensor, seq_len: int) -> dict:
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_scores = {}
    for layer in range(n_layers):
        pattern = cache["pattern", layer]  # (n_heads, T, T)
        stripe = pattern.diagonal(1 - seq_len, dim1=-2, dim2=-1)  # (n_heads, valid_pos)
        scores = stripe[..., 2:-1].mean(dim=-1)
        for head in range(n_heads):
            head_scores[f"L{layer}H{head}"] = float(scores[head].item())

    sorted_heads = sorted(head_scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "scores": head_scores,
        "top_5": sorted_heads[:5],
        "target_L7H10": head_scores.get("L7H10", 0.0),
    }


def compute_ov_copying_scores(model: HookedTransformer) -> dict:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_E = model.W_E
    W_U = model.W_U

    copying_scores = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            W_V = model.W_V[layer, head]
            W_O = model.W_O[layer, head]
            small_op = W_O @ W_U @ W_E @ W_V  # (d_head, d_head) = (64, 64)
            num = small_op.diagonal().sum()
            ev = t.linalg.eigvals(small_op)
            den = abs(ev).sum()
            score = float((num / den).item())
            copying_scores[f"L{layer}H{head}"] = score

    sorted_heads = sorted(copying_scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "scores": copying_scores,
        "top_5": sorted_heads[:5],
        "L7H10_score": copying_scores.get("L7H10", 0.0),
    }


def compute_direct_logit_attribution(model: HookedTransformer) -> dict:
    prompt = "The cat sat on the mat. The cat"
    target_token = " sat"
    tokens = model.to_tokens(prompt)
    target_id = int(model.to_tokens(target_token, prepend_bos=False).flatten().item())

    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    full_target_logit = float(logits[0, -1, target_id].item())

    per_head_residual, _ = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    per_head_residual = rearrange(per_head_residual, "(l h) x -> l h x", l=model.cfg.n_layers)
    per_head_residual_ln = cache.apply_ln_to_stack(per_head_residual, layer=-1, pos_slice=-1, has_batch_dim=False)
    target_dir = model.W_U[:, target_id]
    dla = t.einsum("l h x, x -> l h", per_head_residual_ln, target_dir)

    dla_dict = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            dla_dict[f"L{layer}H{head}"] = float(dla[layer, head].item())

    sorted_pos = sorted(dla_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_neg = sorted(dla_dict.items(), key=lambda kv: kv[1])

    return {
        "prompt": prompt,
        "target_token": target_token,
        "full_target_logit": full_target_logit,
        "top_positive_5": sorted_pos[:5],
        "top_negative_5": sorted_neg[:5],
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    print("Running induction score scan...")
    tokens = generate_repeated_tokens(model, seq_len=SEQ_LEN)
    induction_res = compute_induction_scores(model, tokens, seq_len=SEQ_LEN)
    print("Top induction heads:", induction_res["top_5"])

    print("Running OV copying eigenvalue scan...")
    ov_res = compute_ov_copying_scores(model)
    print("Top OV copying heads:", ov_res["top_5"])
    print("L7H10 OV copying score:", ov_res["L7H10_score"])

    print("Running Direct Logit Attribution...")
    dla_res = compute_direct_logit_attribution(model)
    print("Top positive DLA heads:", dla_res["top_positive_5"])
    print("Top negative DLA heads:", dla_res["top_negative_5"])

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gpt2_metrics.json"

    data = {
        "model": MODEL_NAME,
        "architecture": {
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "d_model": model.cfg.d_model,
            "d_head": model.cfg.d_head,
            "d_vocab": model.cfg.d_vocab,
            "total_params": 117_000_000,
        },
        "induction_scan": induction_res,
        "ov_copying_eigenvalues": ov_res,
        "direct_logit_attribution": dla_res,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nAll benchmark metrics saved successfully to {out_file}")


if __name__ == "__main__":
    main()
