from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from clip.model import CLIP  # noqa: E402
from tsbir.data import CaptionSketchDataset, DistributedUniqueImageBatchSampler  # noqa: E402
from tsbir.losses import AsymmetricLoss, distributed_contrastive_loss  # noqa: E402


def _invalid_checkpoint_keys(keys, allowed_prefixes):
    return [key for key in keys if not key.startswith(tuple(allowed_prefixes))]


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def set_seed(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config: dict, device: torch.device) -> CLIP:
    model_config_path = REPO_ROOT / "code" / "training" / "model_configs" / "ViT-B-16.json"
    with model_config_path.open() as handle:
        model_config = json.load(handle)
    model = CLIP(
        **model_config,
        weight_sharing=bool(config["model"]["weight_sharing"]),
        feature_fusion=config["model"]["feature_fusion"],
        num_class=int(config["model"]["num_classes"]),
        normalize_fused_query=bool(config["model"].get("normalize_fused_query", True)),
        gate_hidden_dim=int(config["model"].get("gate_hidden_dim", 256)),
        gate_alpha_init=float(config["model"].get("gate_alpha_init", 0.5)),
    )
    init_mode = config["model"].get("init", "official_taskformer")
    if init_mode == "official_taskformer":
        checkpoint = torch.load(
            config["model"]["checkpoint"],
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"]
        if next(iter(state_dict)).startswith("module."):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        # strict=False so a backbone checkpoint without gate.* params loads into
        # a gate-enabled model (gate keeps its constructor init); for an avg model
        # there are no missing keys, so behaviour is identical to strict=True.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing = ("gate.",) if config["model"]["feature_fusion"] == "gate" else ()
        invalid_missing = _invalid_checkpoint_keys(missing, allowed_missing)
        if int(os.environ.get("RANK", "0")) == 0 and (missing or unexpected):
            gate_missing = [k for k in missing if k.startswith("gate.")]
            print(f"[official_taskformer] loaded {len(state_dict)} tensors from "
                  f"{config['model']['checkpoint']}", flush=True)
            print(f"[official_taskformer] missing total={len(missing)} "
                  f"(gate.* expected under feature_fusion=gate = {len(gate_missing)}; "
                  f"backbone gaps must be 0 -> {len(invalid_missing)} {invalid_missing[:8]})", flush=True)
            print(f"[official_taskformer] unexpected = {len(unexpected)} "
                  f"{unexpected[:8]}", flush=True)
        if invalid_missing or unexpected:
            raise RuntimeError(
                "official TASK-former checkpoint is incompatible: "
                f"invalid missing keys={invalid_missing[:8]}, unexpected keys={unexpected[:8]}"
            )
    elif init_mode == "clip_pretrained":
        # Initialize the CLIP backbone from the original OpenAI ViT-B-16 weights
        # (not the TASK-former/TSBIR checkpoint). OpenAI's ViT-B-16.pt is a
        # TorchScript archive whose backbone keys match this codebase's CLIP
        # exactly (302 tensors, 0 shape mismatches); the extra TASK-former heads
        # (decoder.*, classification_fc_*.*) and the weight-shared visual2.* are
        # absent from the OpenAI checkpoint and keep their default (random) init.
        ckpt_path = config["model"]["checkpoint"]
        try:
            loaded = torch.jit.load(ckpt_path, map_location="cpu")
            state_dict = loaded.state_dict()
        except Exception:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint and not any(
                k.startswith(("visual.", "transformer.", "token_embedding", "positional_embedding"))
                for k in checkpoint
            ):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        state_dict = {key.removeprefix("module."): tensor.float() for key, tensor in state_dict.items()}
        # TorchScript CLIP archives expose these three scalar metadata entries;
        # they are not learnable model parameters.
        for metadata_key in ("input_resolution", "context_length", "vocab_size"):
            state_dict.pop(metadata_key, None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing = ("visual2.", "decoder.", "classification_fc_")
        if config["model"]["feature_fusion"] == "gate":
            allowed_missing += ("gate.",)
        missing_other = _invalid_checkpoint_keys(missing, allowed_missing)
        if int(os.environ.get("RANK", "0")) == 0:
            missing_extra = [k for k in missing if k not in set(missing_other)]
            print(f"[clip_pretrained] loaded {len(state_dict)} tensors from {ckpt_path}", flush=True)
            print(f"[clip_pretrained] missing total={len(missing)} "
                  f"(visual2*/decoder*/classification* expected = {len(missing_extra)}; "
                  f"backbone gaps must be 0 -> {len(missing_other)} {missing_other[:8]})", flush=True)
            print(f"[clip_pretrained] unexpected = {len(unexpected)} {unexpected[:8]}", flush=True)
        if missing_other or unexpected:
            raise RuntimeError(
                "OpenAI CLIP checkpoint is incompatible: "
                f"invalid missing keys={missing_other[:8]}, unexpected keys={unexpected[:8]}"
            )
    elif init_mode in ("random", "from_scratch", "scratch"):
        # Train from scratch: keep the CLIP constructor's initialize_parameters() init.
        pass
    else:
        raise ValueError(f"unsupported model.init: {init_mode}")
    model.set_grad_checkpointing(bool(config["train"].get("gradient_checkpointing", False)))
    return model.to(device)


def parameter_groups(model: torch.nn.Module, weight_decay: float, gate_lr=None):
    # Gate params get their own (higher) LR; gate and backbone parameters both
    # keep biases and one-dimensional parameters out of weight decay.
    decay, no_decay, gate_decay, gate_no_decay = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("gate."):
            target = gate_no_decay if parameter.ndim < 2 or name.endswith("bias") else gate_decay
            target.append(parameter)
        elif parameter.ndim < 2 or name.endswith("bias") or name == "logit_scale":
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"name": "backbone_no_decay", "params": no_decay, "weight_decay": 0.0},
        {"name": "backbone_decay", "params": decay, "weight_decay": weight_decay},
    ]
    gate_options = {"name": "gate", "weight_decay": weight_decay}
    if gate_lr is not None:
        gate_options["lr"] = gate_lr
    if gate_decay:
        groups.append({**gate_options, "params": gate_decay})
    if gate_no_decay:
        groups.append({**gate_options, "params": gate_no_decay, "weight_decay": 0.0})
    return groups


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, minimum_ratio: float):
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, schedule)


def save_checkpoint(path: Path, model: CLIP, optimizer, scheduler, epoch: int, global_step: int, config: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def save_weights(path: Path, model: CLIP, epoch: int, config: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_coco_ddp.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    set_seed(int(config["experiment"]["seed"]), rank)
    if world_size != int(config["train"]["expected_world_size"]):
        raise RuntimeError(f"config expects {config['train']['expected_world_size']} processes, got {world_size}")

    output_dir = Path(config["experiment"]["output_dir"])
    if args.smoke:
        output_dir = output_dir / "smoke"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, output_dir / "config.yaml")
    if world_size > 1:
        dist.barrier()

    dataset = CaptionSketchDataset(
        config["data"]["train_manifest"],
        image_size=int(config["data"]["image_size"]),
        training=True,
    )
    expected_pairs = int(config["data"]["train_pairs_per_epoch"])
    if len(dataset) != expected_pairs:
        raise RuntimeError(f"expected {expected_pairs} train pairs, found {len(dataset)}")
    batch_size = args.batch_size or int(config["train"]["batch_size_per_gpu"])
    sampler = DistributedUniqueImageBatchSampler(
        dataset,
        batch_size_per_rank=batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=int(config["experiment"]["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["data"]["workers_per_gpu"]),
        pin_memory=True,
        persistent_workers=True,
    )

    model = load_model(config, device)
    if world_size > 1:
        ddp = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=bool(config["distributed"]["broadcast_buffers"]),
            find_unused_parameters=bool(config["distributed"]["find_unused_parameters"]),
        )
        module = ddp.module
    else:
        ddp = model
        module = model
    gate_lr = float(config["train"].get("gate_learning_rate", config["train"]["learning_rate"]))
    optimizer = AdamW(
        parameter_groups(model, float(config["train"]["weight_decay"]), gate_lr),
        lr=float(config["train"]["learning_rate"]),
        betas=(float(config["train"]["beta1"]), float(config["train"]["beta2"])),
        eps=float(config["train"]["eps"]),
    )
    accumulation = int(config["train"]["gradient_accumulation_steps"])
    epochs = 1 if args.smoke else int(config["train"]["epochs"])
    updates_per_epoch = len(loader) // accumulation
    if args.smoke:
        updates_per_epoch = 2
    total_updates = updates_per_epoch * epochs
    scheduler = make_scheduler(
        optimizer,
        int(config["train"]["warmup_steps"]),
        total_updates,
        float(config["train"]["min_learning_rate"]) / float(config["train"]["learning_rate"]),
    )
    asymmetric_loss = AsymmetricLoss(
        float(config["loss"]["asl_gamma_negative"]),
        float(config["loss"]["asl_gamma_positive"]),
        float(config["loss"]["asl_clip"]),
    )

    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    csv_file = (output_dir / "train.csv").open("w", newline="") if rank == 0 else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(
            [
                "epoch", "step", "loss", "embedding", "classification", "decoder",
                "lr", "gate_lr", "alpha_mean", "alpha_std", "alpha_p10", "alpha_p50",
                "alpha_p90", "seconds",
            ]
        )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for epoch in range(epochs):
        ddp.train()
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if not args.smoke and batch_index >= updates_per_epoch * accumulation:
                break
            if args.smoke and global_step >= 2:
                break
            sync_step = (batch_index + 1) % accumulation == 0
            sync_context = contextlib.nullcontext() if sync_step or world_size == 1 else ddp.no_sync()
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                image_features, text_features, sketch_features = ddp(
                    batch["image"].to(device, non_blocking=True),
                    batch["text_tokens"].to(device, non_blocking=True),
                    batch["sketch"].to(device, non_blocking=True),
                    return_all_features=True,
                )
                image_features = image_features.float()
                text_features = text_features.float()
                sketch_features = sketch_features.float()
                query_features, fusion_alpha = module.feature_fuse(
                    text_features,
                    sketch_features,
                    return_alpha=True,
                )
                embedding = distributed_contrastive_loss(
                    image_features,
                    query_features,
                    module.logit_scale.exp().clamp(max=100.0),
                )
                labels = batch["labels"].to(device, non_blocking=True)
                classification = (
                    asymmetric_loss(module.classify(image_features), labels)
                    + asymmetric_loss(module.classify(sketch_features), labels)
                    + asymmetric_loss(module.classify(text_features), labels)
                ) / 3.0
                caption_tokens = batch["caption_tokens"].to(device, non_blocking=True)
                context = ((image_features + sketch_features) / 2.0).unsqueeze(1)
                decoder_logits = module.decoder.net(caption_tokens[:, :-1], context=context)
                decoder = F.cross_entropy(
                    decoder_logits.transpose(1, 2),
                    caption_tokens[:, 1:],
                    ignore_index=0,
                )
                loss = (
                    float(config["loss"]["contrastive_weight"]) * embedding
                    + float(config["loss"]["classification_weight"]) * classification
                    + float(config["loss"]["decoder_weight"]) * decoder
                )
                (loss / accumulation).backward()

            if not sync_step:
                continue
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), float(config["train"]["gradient_clip_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            module.logit_scale.data.clamp_(0, math.log(100.0))

            if rank == 0 and (global_step == 1 or global_step % 20 == 0):
                values = [float(x.detach()) for x in (loss, embedding, classification, decoder)]
                elapsed = time.time() - started
                lr = optimizer.param_groups[0]["lr"]
                gate_lr_value = next(
                    (group["lr"] for group in optimizer.param_groups if group.get("name") == "gate"),
                    lr,
                )
                alpha_values = fusion_alpha.detach().float().flatten()
                alpha_mean = float(alpha_values.mean())
                alpha_std = float(alpha_values.std(unbiased=False))
                alpha_p10, alpha_p50, alpha_p90 = (
                    float(value) for value in torch.quantile(
                        alpha_values,
                        torch.tensor([0.1, 0.5, 0.9], device=alpha_values.device),
                    )
                )
                print(
                    f"epoch={epoch + 1} step={global_step}/{total_updates} "
                    f"loss={values[0]:.4f} Le={values[1]:.4f} Lc={values[2]:.4f} "
                    f"Ld={values[3]:.4f} lr={lr:.3e} gate_lr={gate_lr_value:.3e} "
                    f"alpha={alpha_mean:.3f}+/-{alpha_std:.3f} "
                    f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                    flush=True,
                )
                csv_writer.writerow(
                    [
                        epoch + 1, global_step, *values, lr, gate_lr_value,
                        alpha_mean, alpha_std, alpha_p10, alpha_p50, alpha_p90, elapsed,
                    ]
                )
                csv_file.flush()
                for name, value in zip(("total", "embedding", "classification", "decoder"), values):
                    writer.add_scalar(f"loss/{name}", value, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/gate_lr", gate_lr_value, global_step)
                writer.add_scalar("fusion/alpha_mean", alpha_mean, global_step)
                writer.add_scalar("fusion/alpha_std", alpha_std, global_step)
                writer.add_scalar("fusion/alpha_p10", alpha_p10, global_step)
                writer.add_scalar("fusion/alpha_p50", alpha_p50, global_step)
                writer.add_scalar("fusion/alpha_p90", alpha_p90, global_step)

        if args.smoke:
            break
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            save_checkpoint(output_dir / "last.ckpt", model, optimizer, scheduler, epoch + 1, global_step, config)
            save_weights(output_dir / f"epoch_{epoch + 1:03d}_weights.pt", model, epoch + 1, config)
            save_weights(output_dir / "last_weights.pt", model, epoch + 1, config)
            print(f"epoch={epoch + 1} checkpoints saved", flush=True)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        if args.smoke:
            save_checkpoint(output_dir / "smoke.ckpt", model, optimizer, scheduler, 0, global_step, config)
        writer.close()
        csv_file.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
