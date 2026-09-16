"""Extract aligned 2048-D ImageNet Xception features for Twitter users."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from plench.data.twitter_ethnicity_2017 import FEATURE_DIM, load_twitter_ethnicity_bundle


class ManifestImages(Dataset):
    def __init__(self, bundle, transform) -> None:
        self.bundle = bundle
        self.transform = transform

    def __len__(self) -> int:
        return len(self.bundle.manifest)

    def __getitem__(self, index: int):
        raw_path = Path(str(self.bundle.manifest.iloc[index]["image_path"])).expanduser()
        if not raw_path.is_absolute():
            candidate = self.bundle.dataset_root / "raw" / raw_path
            raw_path = candidate if candidate.is_file() else self.bundle.dataset_root / raw_path
        if not raw_path.is_file():
            raise FileNotFoundError(f"DATA_MISSING: profile image {raw_path}")
        with Image.open(raw_path) as image:
            return self.transform(image.convert("RGB")), index


def _atomic_numpy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def extract(
    data_root: Path,
    *,
    model_name: str,
    batch_size: int,
    num_workers: int,
    device: str,
    overwrite: bool,
) -> None:
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as exc:
        raise ImportError(
            "Xception extraction requires timm. Install the project with the "
            "twitter optional dependencies: pip install -e '.[twitter]'"
        ) from exc

    bundle = load_twitter_ethnicity_bundle(data_root, require_features=False)
    feature_path = bundle.processed_dir / "xception_features.npy"
    ids_path = bundle.processed_dir / "instance_ids.npy"
    if (feature_path.exists() or ids_path.exists()) and not overwrite:
        raise FileExistsError(
            f"{feature_path} or {ids_path} exists; pass --overwrite to replace both"
        )
    actual_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    ).eval().to(actual_device)
    data_config = resolve_model_data_config(model)
    input_size = tuple(data_config.get("input_size", (3, 299, 299)))
    if input_size != (3, 299, 299):
        data_config = dict(data_config)
        data_config["input_size"] = (3, 299, 299)
    transform = create_transform(**data_config, is_training=False)
    loader = DataLoader(
        ManifestImages(bundle, transform),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=actual_device.type == "cuda",
    )
    output = np.empty((len(bundle.manifest), FEATURE_DIM), dtype=np.float32)
    seen = np.zeros(len(bundle.manifest), dtype=bool)
    with torch.no_grad():
        for images, indices in loader:
            features = model(images.to(actual_device)).detach().cpu().to(torch.float32).numpy()
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
                raise ValueError(
                    f"{model_name} returned {features.shape}; expected [N, {FEATURE_DIM}]"
                )
            rows = indices.numpy()
            output[rows] = features
            seen[rows] = True
    if not seen.all() or not np.isfinite(output).all():
        raise ValueError("feature extraction produced missing, NaN, or infinite rows")
    instance_ids = bundle.manifest["instance_id"].to_numpy(dtype=str)
    _atomic_numpy(feature_path, output)
    _atomic_numpy(ids_path, instance_ids)
    print(f"Wrote {feature_path}: shape={output.shape}, dtype={output.dtype}")
    print(f"Wrote {ids_path}; order exactly matches manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-name", default="legacy_xception")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract(
        args.data_root,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
