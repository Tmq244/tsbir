#!/usr/bin/env python3
"""Continuous-alpha fusion oracle for the frozen avg encoder.

For each caption-query we sweep alpha in a grid, build the fused query
q(alpha) = normalize(alpha*text + (1-alpha)*sketch), and rank it against the
5k image gallery.  The per-query best alpha gives the *label-leakage upper
bound* for any learned per-query gate.  Also reports the alpha* distribution
(a predictability hint) and compares human vs synthetic sketch.

This is an analysis script: it reads a checkpoint, encodes the test set once,
and prints metrics.  It does not train anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "code"))

from tsbir.data import read_jsonl  # noqa: E402
from tsbir.evaluate import (  # noqa: E402
    CaptionDataset,
    encode_visual,
    load_model,
)


def encode_captions(model, records, batch_size, workers, device):
    loader = DataLoader(
        CaptionDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    feats, record_idx = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for tokens, idx, _ in loader:
            f = F.normalize(model.encode_text(tokens.to(device)).float(), dim=-1)
            feats.append(f)
            record_idx.append(idx)
    return torch.cat(feats), torch.cat(record_idx)


def sweep_ranks(text, sketch_per_query, gallery, targets, alphas, device):
    """Return rank matrix [n_queries, n_alphas] (1-indexed, strict-greater convention)."""
    ranks = np.empty((text.shape[0], len(alphas)), dtype=np.int64)
    with torch.inference_mode():
        for j, a in enumerate(alphas):
            fused = F.normalize(a * text + (1.0 - a) * sketch_per_query, dim=-1)
            scores = fused @ gallery.t()              # [n, 5000]
            tgt = scores.gather(1, targets[:, None])
            r = (scores > tgt).sum(dim=1) + 1
            ranks[:, j] = r.cpu().numpy()
    return ranks


def report(name, ranks):
    r = np.asarray(ranks)
    print(f"  {name:30s} R@1={np.mean(r<=1):.4f}  R@5={np.mean(r<=5):.4f}  "
          f"R@10={np.mean(r<=10):.4f}  MRR={np.mean(1.0/r):.4f}")


def alpha_star_hist(ranks, alphas):
    best = ranks.argmin(axis=1)          # first alpha achieving min rank
    counts = np.bincount(best, minlength=len(alphas))
    return counts / counts.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("outputs/taskformer_coco_clip_10ep_gpu12_bs96/last_weights.pt"))
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifests/test.jsonl"))
    ap.add_argument("--alphas", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--visual-batch-size", type=int, default=256)
    ap.add_argument("--text-batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    device = torch.device("cuda", 0)
    records = read_jsonl(args.manifest)
    assert len(records) == 5000

    model = load_model(args.checkpoint, device, feature_fusion="auto")
    gallery = encode_visual(model, records, "image", 224, args.visual_batch_size, args.workers, device)
    text, rec_idx = encode_captions(model, records, args.text_batch_size, args.workers, device)
    rec_idx = rec_idx.to(device)
    targets = rec_idx.clone()
    sketch_human = encode_visual(model, records, "human_sketch", 224, args.visual_batch_size, args.workers, device, sketch=True)
    sketch_synth = encode_visual(model, records, "synthetic_sketch", 224, args.visual_batch_size, args.workers, device, sketch=True)
    sk_human_q = sketch_human[rec_idx]
    sk_synth_q = sketch_synth[rec_idx]

    print(f"\nalphas = {alphas}\n")
    for label, sk_q in (("HUMAN sketch (deploy dist.)", sk_human_q),
                        ("SYNTHETIC sketch (train dist.)", sk_synth_q)):
        ranks = sweep_ranks(text, sk_q, gallery, targets, alphas, device)
        print(f"===== {label} =====")
        ai = {round(a, 2): j for j, a in enumerate(alphas)}
        report(f"fixed α=0   (sketch-only)", ranks[:, ai[0.0]])
        report(f"fixed α=0.5 (avg mixed)",   ranks[:, ai[0.5]])
        report(f"fixed α=1   (text-only)",   ranks[:, ai[1.0]])
        oracle = ranks.min(axis=1)
        report(f"CONTINUOUS-α ORACLE",       oracle)
        print(f"    -> ΔR@1 over avg(0.5) = {np.mean(oracle<=1)-np.mean(ranks[:,ai[0.5]]<=1):+.4f}")
        h = alpha_star_hist(ranks, alphas)
        print("    α* (per-query argmin-rank) distribution:")
        for a, p in zip(alphas, h):
            bar = "#" * int(p * 50)
            print(f"      α={a:<3.1f}  {p*100:5.1f}%  {bar}")
        interior = sum(h[k] for k, a in enumerate(alphas) if 0.0 < a < 1.0)
        print(f"    α* at interior (0<α<1): {interior*100:.1f}%   at endpoints {{0,1}}: {(1-interior)*100:.1f}%")
        print()

    # human vs synthetic alpha* agreement (predictability across dist shift)
    rh = sweep_ranks(text, sk_human_q, gallery, targets, alphas, device)
    rs = sweep_ranks(text, sk_synth_q, gallery, targets, alphas, device)
    ast_h = rh.argmin(axis=1)
    ast_s = rs.argmin(axis=1)
    agree = np.mean(ast_h == ast_s)
    # tolerance: within one grid step
    near = np.mean(np.abs(ast_h - ast_s) <= 1)
    print("===== human vs synthetic α* agreement =====")
    print(f"  exact same α* bucket: {agree*100:.1f}%")
    print(f"  within ±1 grid step:  {near*100:.1f}%")


if __name__ == "__main__":
    main()
