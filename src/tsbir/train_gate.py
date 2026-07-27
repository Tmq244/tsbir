from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

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
from tsbir.losses import distributed_contrastive_loss  # noqa: E402
from tsbir.train import set_seed, setup_distributed  # noqa: E402


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, schedule)


def load_frozen_avg_with_gate(config: dict, device: torch.device) -> CLIP:
    source_path = Path(config["model"]["source_checkpoint"])
    checkpoint = torch.load(source_path, map_location="cpu", mmap=True, weights_only=False)
    state_dict = checkpoint["state_dict"]
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    source_fusion = checkpoint.get("config", {}).get("model", {}).get("feature_fusion")
    if source_fusion not in (None, "avg") or any(key.startswith("gate.") for key in state_dict):
        raise RuntimeError(f"gate-only stage requires an avg checkpoint, got {source_path}")

    model_config_path = REPO_ROOT / "code" / "training" / "model_configs" / "ViT-B-16.json"
    with model_config_path.open() as handle:
        model_config = json.load(handle)
    model = CLIP(
        **model_config,
        weight_sharing=bool(config["model"]["weight_sharing"]),
        feature_fusion="gate",
        num_class=int(config["model"]["num_classes"]),
        normalize_fused_query=True,
        gate_hidden_dim=int(config["model"].get("gate_hidden_dim", 256)),
        gate_alpha_init=float(config["model"].get("gate_alpha_init", 0.5)),
        gate_mixed_only=True,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected_missing = {name for name in model.state_dict() if name.startswith("gate.")}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            "avg checkpoint is incompatible with the gate model: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.gate.parameters():
        parameter.requires_grad = True
    model.to(device).eval()
    model.gate.train()
    return model


def gather_without_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def alpha_grid_teacher(
    text_features: torch.Tensor,
    sketch_features: torch.Tensor,
    all_images: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
    alpha_grid: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a soft continuous-alpha teacher from per-grid retrieval losses.

    Returns expected alpha, normalized confidence, hard best alpha, and the
    minimum per-sample teacher loss.  The target image is used only to build
    training supervision; it is never an input to the learned gate.
    """
    if temperature <= 0:
        raise ValueError(f"teacher temperature must be positive, got {temperature}")
    alpha = alpha_grid.view(1, -1, 1)
    candidates = F.normalize(
        alpha * text_features[:, None, :] + (1.0 - alpha) * sketch_features[:, None, :],
        dim=-1,
    )
    logits = scale * torch.einsum("bad,gd->bag", candidates, all_images)
    log_probabilities = F.log_softmax(logits, dim=-1)
    grid_losses = -log_probabilities.gather(
        2,
        targets[:, None, None].expand(-1, alpha_grid.numel(), 1),
    ).squeeze(-1)
    teacher_probabilities = F.softmax(-grid_losses / temperature, dim=-1)
    expected_alpha = (teacher_probabilities * alpha_grid[None, :]).sum(dim=-1)
    entropy = -(teacher_probabilities * teacher_probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    confidence = 1.0 - entropy / math.log(alpha_grid.numel())
    best_indices = grid_losses.argmin(dim=-1)
    best_alpha = alpha_grid[best_indices]
    best_loss = grid_losses.gather(1, best_indices[:, None]).squeeze(1)
    return expected_alpha, confidence.clamp(0.0, 1.0), best_alpha, best_loss


def gated_fusion(
    gate: torch.nn.Module,
    text_features: torch.Tensor,
    sketch_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=text_features.device.type, enabled=False):
        text = text_features.float()
        sketch = sketch_features.float()
        gate_input = torch.cat([text, sketch, text * sketch], dim=-1)
        alpha = torch.sigmoid(gate(gate_input).float())
    query = F.normalize(alpha * text_features + (1.0 - alpha) * sketch_features, dim=-1)
    return query, alpha


def gate_parameter_groups(gate: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in gate.named_parameters():
        target = no_decay if parameter.ndim < 2 or name.endswith("bias") else decay
        target.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_gate_checkpoint(
    path: Path,
    model: CLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    config: dict,
) -> None:
    atomic_torch_save(
        {
            "format_version": 1,
            "epoch": epoch,
            "global_step": global_step,
            "gate_state_dict": model.gate.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def save_full_weights(path: Path, model: CLIP, epoch: int, config: dict) -> None:
    atomic_torch_save(
        {
            "format_version": 2,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_coco_frozen_gate_grid_gpu45.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    set_seed(int(config["experiment"]["seed"]), rank)
    if world_size != int(config["train"]["expected_world_size"]):
        raise RuntimeError(
            f"config expects {config['train']['expected_world_size']} processes, got {world_size}"
        )

    output_dir = Path(config["experiment"]["output_dir"])
    if args.smoke:
        output_dir = output_dir / "smoke"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, output_dir / "config.yaml")
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    dataset = CaptionSketchDataset(
        config["data"]["train_manifest"],
        image_size=int(config["data"]["image_size"]),
        training=True,
        use_human_sketch=False,
        query_dropout=0.0,
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
    workers = int(config["data"]["workers_per_gpu"])
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    model = load_frozen_avg_with_gate(config, device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    if rank == 0:
        print(
            f"[model] source={config['model']['source_checkpoint']} "
            f"trainable_gate={trainable:,} frozen={frozen:,}",
            flush=True,
        )

    gate = model.gate
    if world_size > 1:
        gate_runner: torch.nn.Module = DistributedDataParallel(
            gate,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        gate_runner = gate

    learning_rate = float(config["train"]["learning_rate"])
    optimizer = AdamW(
        gate_parameter_groups(gate, float(config["train"]["weight_decay"])),
        lr=learning_rate,
        betas=(float(config["train"]["beta1"]), float(config["train"]["beta2"])),
        eps=float(config["train"]["eps"]),
    )
    epochs = 1 if args.smoke else int(config["train"]["epochs"])
    updates_per_epoch = 2 if args.smoke else len(loader)
    total_updates = epochs * updates_per_epoch
    scheduler = make_scheduler(
        optimizer,
        int(config["train"]["warmup_steps"]),
        total_updates,
        float(config["train"]["min_learning_rate"]) / learning_rate,
    )
    alpha_grid = torch.tensor(config["teacher"]["alpha_grid"], device=device, dtype=torch.float32)
    if alpha_grid.ndim != 1 or alpha_grid.numel() < 2:
        raise ValueError("teacher.alpha_grid must contain at least two values")
    if not torch.all(alpha_grid[1:] > alpha_grid[:-1]) or alpha_grid[0] < 0 or alpha_grid[-1] > 1:
        raise ValueError("teacher.alpha_grid must be strictly increasing within [0, 1]")
    teacher_temperature = float(config["teacher"]["temperature"])
    confidence_floor = float(config["teacher"].get("confidence_floor", 0.1))
    retrieval_weight = float(config["loss"]["retrieval_weight"])
    reliability_weight = float(config["loss"]["reliability_weight"])

    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    csv_file = (output_dir / "train.csv").open("w", newline="") if rank == 0 else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(
            [
                "epoch", "step", "loss", "retrieval", "reliability", "lr",
                "alpha_mean", "alpha_std", "target_mean", "target_std",
                "best_alpha_mean", "teacher_confidence", "teacher_best_loss", "seconds",
            ]
        )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index >= updates_per_epoch:
                break
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                image_features, text_features, sketch_features = model(
                    batch["image"].to(device, non_blocking=True),
                    batch["text_tokens"].to(device, non_blocking=True),
                    batch["sketch"].to(device, non_blocking=True),
                    return_all_features=True,
                )
                image_features = image_features.float()
                text_features = text_features.float()
                sketch_features = sketch_features.float()

            all_images = gather_without_grad(image_features)
            targets = rank * image_features.shape[0] + torch.arange(
                image_features.shape[0], device=device
            )
            scale = model.logit_scale.exp().clamp(max=100.0).detach()
            target_alpha, confidence, best_alpha, teacher_best_loss = alpha_grid_teacher(
                text_features,
                sketch_features,
                all_images,
                targets,
                scale,
                alpha_grid,
                teacher_temperature,
            )
            query_features, fusion_alpha = gated_fusion(
                gate_runner,
                text_features,
                sketch_features,
            )
            retrieval = distributed_contrastive_loss(
                image_features,
                query_features,
                scale,
            )
            reliability_per_sample = F.mse_loss(
                fusion_alpha.squeeze(-1),
                target_alpha,
                reduction="none",
            )
            reliability_weights = confidence_floor + confidence
            reliability = (reliability_weights * reliability_per_sample).sum() / reliability_weights.sum()
            loss = retrieval_weight * retrieval + reliability_weight * reliability

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), float(config["train"]["gradient_clip_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if rank == 0 and (
                global_step == 1
                or global_step % int(config["train"]["log_every_steps"]) == 0
                or global_step == total_updates
            ):
                elapsed = time.time() - started
                values = {
                    "loss": float(loss.detach()),
                    "retrieval": float(retrieval.detach()),
                    "reliability": float(reliability.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "alpha_mean": float(fusion_alpha.detach().mean()),
                    "alpha_std": float(fusion_alpha.detach().std(unbiased=False)),
                    "target_mean": float(target_alpha.mean()),
                    "target_std": float(target_alpha.std(unbiased=False)),
                    "best_alpha_mean": float(best_alpha.mean()),
                    "teacher_confidence": float(confidence.mean()),
                    "teacher_best_loss": float(teacher_best_loss.mean()),
                }
                print(
                    f"epoch={epoch + 1} step={global_step}/{total_updates} "
                    f"loss={values['loss']:.5f} Lret={values['retrieval']:.5f} "
                    f"Lrel={values['reliability']:.5f} lr={values['lr']:.3e} "
                    f"alpha={values['alpha_mean']:.3f}+/-{values['alpha_std']:.3f} "
                    f"target={values['target_mean']:.3f}+/-{values['target_std']:.3f} "
                    f"teacher_conf={values['teacher_confidence']:.3f} "
                    f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                    flush=True,
                )
                csv_writer.writerow(
                    [epoch + 1, global_step, *values.values(), elapsed]
                )
                csv_file.flush()
                for name, value in values.items():
                    writer.add_scalar(f"gate/{name}", value, global_step)

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        if rank == 0:
            gate_payload = {
                "format_version": 1,
                "epoch": epoch + 1,
                "gate_state_dict": model.gate.state_dict(),
                "config": config,
            }
            atomic_torch_save(gate_payload, output_dir / f"epoch_{epoch + 1:03d}_gate.pt")
            atomic_torch_save(gate_payload, output_dir / "last_gate.pt")
            save_gate_checkpoint(
                output_dir / "last.ckpt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                config,
            )
            if not args.smoke:
                save_full_weights(output_dir / "last_weights.pt", model, epoch + 1, config)
            print(f"epoch={epoch + 1} checkpoints saved", flush=True)
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

    if rank == 0:
        writer.close()
        csv_file.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
