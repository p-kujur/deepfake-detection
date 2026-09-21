"""Train NPR-lite CNN second head (CIFAKE M3 or ProGAN-folder M3c)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from deepfake_detection.data.cifake import CIFAKEDataset, DEFAULT_CACHE_DIR, DEFAULT_HF_ID
from deepfake_detection.data.folder_dataset import RealFakeFolderDataset
from deepfake_detection.data.transforms import TrainAugment
from deepfake_detection.metrics import MetricsRecord, write_metrics
from deepfake_detection.models.npr_lite import (
    NPR_LITE_MODEL_ID,
    NPRLiteCNN,
    build_cnn_preprocess,
    save_npr_lite,
)
from deepfake_detection.models.unifd import resolve_device

logger = logging.getLogger(__name__)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_cnn_train_preprocess(
    image_size: int = 128,
    jpeg_prob: float = 0.5,
    blur_prob: float = 0.1,
) -> Callable[[Image.Image], torch.Tensor]:
    """ImageNet-norm CNN preprocess with optional Blur/JPEG train aug."""
    aug = TrainAugment(jpeg_prob=jpeg_prob, blur_prob=blur_prob)
    base = build_cnn_preprocess(image_size)

    def _fn(img: Image.Image):
        return base(aug(img))

    return _fn


@torch.inference_mode()
def eval_cnn(
    model: NPRLiteCNN,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    lats: list[float] = []
    for images, y in tqdm(loader, desc="cnn-eval", leave=False):
        images = images.to(device)
        t0 = time.perf_counter()
        probs = model.predict_proba(images).cpu().numpy()
        dt = (time.perf_counter() - t0) * 1000.0 / max(images.size(0), 1)
        lats.extend([dt] * images.size(0))
        probs_all.append(probs)
        labels_all.append(y.numpy())
    probs = np.concatenate(probs_all)
    y = np.concatenate(labels_all).astype(int)
    preds = (probs >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(y, preds)),
        "ap": float(average_precision_score(y, probs)),
        "auc": float(roc_auc_score(y, probs)),
        "n": float(len(y)),
        "latency_p50_ms": float(np.percentile(lats, 50)),
        "latency_p95_ms": float(np.percentile(lats, 95)),
    }


def train_one(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    image_size = int(model_cfg.get("image_size", 128))
    width = int(model_cfg.get("width", 32))
    epochs = int(train_cfg.get("epochs", 5))
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    threshold = float(model_cfg.get("threshold", 0.5))
    device = resolve_device(str(train_cfg.get("device", "auto")))
    use_aug = bool(train_cfg.get("aug", False))
    jpeg_prob = float(train_cfg.get("jpeg_prob", 0.5))
    blur_prob = float(train_cfg.get("blur_prob", 0.1))

    if use_aug:
        train_tf = build_cnn_train_preprocess(
            image_size, jpeg_prob=jpeg_prob, blur_prob=blur_prob
        )
    else:
        train_tf = build_cnn_preprocess(image_size)
    eval_tf = build_cnn_preprocess(image_size)

    data_name = str(data_cfg.get("name", "cifake")).lower()
    max_train = data_cfg.get("max_train")
    max_test = data_cfg.get("max_test")

    if data_name in ("progan_folder", "folder", "real_fake_folder"):
        train_root = Path(data_cfg["train_root"])
        test_root = Path(data_cfg.get("test_root") or data_cfg.get("val_root") or "")
        if not test_root:
            raise ValueError("progan_folder requires data.test_root (or val_root)")
        train_ds = RealFakeFolderDataset(
            train_root,
            transform=train_tf,
            max_per_class=int(max_train) if max_train else None,
            seed=seed,
        )
        test_ds = RealFakeFolderDataset(
            test_root,
            transform=eval_tf,
            max_per_class=int(max_test) if max_test else None,
            seed=seed,
        )
        dataset_name = "progan_holdout" if not (max_train or max_test) else "progan_holdout_subset"
        default_weights = "weights/npr_lite_progan.pth"
    else:
        train_ds = CIFAKEDataset(
            split="train",
            transform=train_tf,
            hf_id=data_cfg.get("hf_id", DEFAULT_HF_ID),
            cache_dir=data_cfg.get("cache_dir", str(DEFAULT_CACHE_DIR)),
            max_samples=int(max_train) if max_train else None,
            seed=seed,
        )
        test_ds = CIFAKEDataset(
            split="test",
            transform=eval_tf,
            hf_id=data_cfg.get("hf_id", DEFAULT_HF_ID),
            cache_dir=data_cfg.get("cache_dir", str(DEFAULT_CACHE_DIR)),
            max_samples=int(max_test) if max_test else None,
            seed=seed,
        )
        is_subset = max_train is not None or max_test is not None
        dataset_name = "cifake_subset" if is_subset else "cifake"
        default_weights = "weights/npr_lite_cifake.pth"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = NPRLiteCNN(width=width).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 1e-4))
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total, n_b = 0.0, 0
        for images, y in tqdm(train_loader, desc=f"cnn-epoch{epoch+1}", leave=False):
            images = images.to(device)
            yb = y.float().unsqueeze(1).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_b += 1
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, total / max(n_b, 1))

    metrics = eval_cnn(model, test_loader, device, threshold=threshold)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    cfg_hash = _config_hash(cfg)

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "artifacts/checkpoints"))
    ckpt_path = ckpt_dir / f"{run_id}_npr_lite.pth"
    weights_out = Path(train_cfg.get("weights_out", default_weights))
    meta = {
        "dataset": dataset_name,
        "data_name": data_name,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "metrics": metrics,
        "image_size": image_size,
        "width": width,
        "model_id": model_cfg.get("id", NPR_LITE_MODEL_ID),
        "aug": use_aug,
    }
    save_npr_lite(model, ckpt_path, meta=meta)
    save_npr_lite(model, weights_out, meta=meta)

    record = MetricsRecord(
        dataset=dataset_name,
        split="test",
        model_id=str(model_cfg.get("id", NPR_LITE_MODEL_ID)),
        acc=round(metrics["acc"], 6),
        ap=round(metrics["ap"], 6),
        auc=round(metrics["auc"], 6),
        n=int(metrics["n"]),
        threshold=threshold,
        latency_p50_ms=round(metrics["latency_p50_ms"], 3),
        latency_p95_ms=round(metrics["latency_p95_ms"], 3),
        aug={
            "jpeg": use_aug and jpeg_prob > 0,
            "blur": use_aug and blur_prob > 0,
            "jpeg_prob": jpeg_prob if use_aug else 0.0,
            "blur_prob": blur_prob if use_aug else 0.0,
            "resize": False,
        },
        notes=(
            f"NPR-lite CNN; data={dataset_name}; train={len(train_ds)} test={len(test_ds)}; "
            f"ckpt={ckpt_path.as_posix()}; aug={use_aug}"
        ),
        config_hash=cfg_hash,
    )
    out_dir = Path(cfg.get("metrics", {}).get("output_dir", "artifacts/runs"))
    metrics_path = write_metrics(record, run_id=run_id, output_dir=out_dir)
    (metrics_path.parent / "config.json").write_text(
        json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8"
    )

    result = {
        "run_id": run_id,
        "metrics_path": str(metrics_path),
        "checkpoint": str(ckpt_path),
        "weights_out": str(weights_out),
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train NPR-lite CNN (CIFAKE or ProGAN folders)")
    parser.add_argument("--config", default="configs/train_npr_lite_subset.yaml")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--aug", action="store_true", help="Enable Blur/JPEG train aug")
    args = parser.parse_args(argv)
    cfg = _load_yaml(args.config)
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    cfg.setdefault("model", {})
    if args.max_train is not None:
        cfg["data"]["max_train"] = args.max_train
    if args.max_test is not None:
        cfg["data"]["max_test"] = args.max_test
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.device:
        cfg["train"]["device"] = args.device
    if args.aug:
        cfg["train"]["aug"] = True
    train_one(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
