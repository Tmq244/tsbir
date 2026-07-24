"""Single-GPU memory probe: peak memory for one real training step at batch 96,
bf16, with and without gradient checkpointing. Faithfully mirrors train.py's step.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:code python scripts/probe_grad_ckpt_mem.py
"""
from __future__ import annotations
import yaml, torch, contextlib
import tsbir.train as T
from tsbir.losses import AsymmetricLoss, distributed_contrastive_loss

CFG = "configs/train_coco_clip_10ep_gpu12.yaml"
BATCH = 96
STEPS = 3


def run(checkpointing: bool):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = yaml.safe_load(open(CFG))
    device = torch.device("cuda", 0)
    model = T.load_model(cfg, device)          # clip_pretrained init
    model.set_grad_checkpointing(checkpointing)  # override config
    model.train()
    optim = torch.optim.AdamW(T.parameter_groups(model, float(cfg["train"]["weight_decay"])), lr=1e-5)
    asl = AsymmetricLoss(4.0, 1.0, 0.05)

    def make_batch():
        return {
            "image": torch.randn(BATCH, 3, 224, 224, device=device),
            "sketch": torch.randn(BATCH, 3, 224, 224, device=device),
            "text_tokens": torch.randint(1, 49408, (BATCH, 77), device=device),
            "caption_tokens": torch.randint(1, 49408, (BATCH, 77), device=device),
            "labels": (torch.rand(BATCH, 90, device=device) > 0.9).float(),
        }

    for _ in range(STEPS):
        b = make_batch()
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img_f, txt_f, sk_f = model(b["image"], b["text_tokens"], b["sketch"], return_all_features=True)
            img_f, txt_f, sk_f = img_f.float(), txt_f.float(), sk_f.float()
            q = model.feature_fuse(txt_f, sk_f)
            emb = distributed_contrastive_loss(img_f, q, model.logit_scale.exp().clamp(max=100.0))
            cls = (asl(model.classify(img_f), b["labels"])
                   + asl(model.classify(sk_f), b["labels"])
                   + asl(model.classify(txt_f), b["labels"])) / 3.0
            ctx = ((img_f + sk_f) / 2.0).unsqueeze(1)
            dec_logits = model.decoder.net(b["caption_tokens"][:, :-1], context=ctx)
            dec = torch.nn.functional.cross_entropy(dec_logits.transpose(1, 2), b["caption_tokens"][:, 1:], ignore_index=0)
            loss = 100.0 * emb + 10.0 * cls + 1.0 * dec
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

    alloc = torch.cuda.max_memory_allocated() / 2**30
    reserv = torch.cuda.max_memory_reserved() / 2**30
    del model, optim, b
    torch.cuda.empty_cache()
    return alloc, reserv


for ckpt in (False, True):
    a, r = run(ckpt)
    print(f"gradient_checkpointing={ckpt!s:5}  batch={BATCH}  "
          f"peak_allocated={a:6.2f} GiB   peak_reserved={r:6.2f} GiB")
