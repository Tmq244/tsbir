from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

REPO_ROOT = Path(__file__).resolve().parents[2]
PHOTOSKETCH_ROOT = REPO_ROOT / "third_party" / "PhotoSketch"
sys.path.insert(0, str(PHOTOSKETCH_ROOT))

from models.networks import define_G  # noqa: E402


class PhotoDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = Compose(
            [
                Resize((256, 256), interpolation=Image.Resampling.BICUBIC),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        return self.transform(Image.open(path).convert("RGB")), path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_sketches"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/raw/photosketch_latest_net_G.pth"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    all_paths: list[Path] = []
    for directory in args.input:
        all_paths.extend(directory.glob("*.jpg"))
    all_paths = sorted(all_paths)
    if args.limit is not None:
        all_paths = all_paths[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [path for path in all_paths if not (args.output / path.name).is_file()]
    print(f"PhotoSketch: {len(all_paths)} inputs, {len(pending)} pending", flush=True)
    if not pending:
        return

    generator = define_G(3, 1, 64, "resnet_9blocks", "batch", False, "normal")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    generator.load_state_dict(state_dict, strict=True)
    generator = generator.cuda().eval()
    loader = DataLoader(
        PhotoDataset(pending),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    completed = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for images, filenames in loader:
            generated = generator(images.cuda(non_blocking=True)).float().cpu()
            generated = ((generated[:, 0] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
            for array, filename in zip(generated, filenames):
                Image.fromarray(np.asarray(array), mode="L").save(args.output / filename, quality=92)
            completed += len(filenames)
            if completed % 1024 < len(filenames):
                print(f"generated {completed}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
