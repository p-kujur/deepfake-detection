"""Train UniFD-style linear probe on frozen CLIP features.

Supports:
  - CIFAKE (M2): ``data.name: cifake`` + HF cache
  - Folder packs (M3b ProGAN): ``data.name: progan_folder`` with train/test roots

Pipeline:
  1. Load train/test images (CIFAKE or real/fake folders).
  2. Extract frozen CLIP ViT-L/14 embeddings (cached under artifacts/).
  3. Fit ``nn.Linear(embed_dim, 1)`` with BCE-with-logits.
  4. Eval Acc / AP / AUC on the test holdout; write metrics.json.
  5. Save probe checkpoint under artifacts/ and weights/ (gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection import __model_id__
from deepfake_detection.data.cifake import CIFAKEDataset, DEFAULT_CACHE_DIR, DEFAULT_HF_ID
from deepfake_detection.data.folder_dataset import RealFakeFolderDataset
from deepfake_detection.data.transforms import build_clip_preprocess, build_train_preprocess
from deepfake_detection.metrics import MetricsRecord, write_metrics
from deepfake_detection.models.unifd import UniFDClipLinear, resolve_device

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


@torch.inference_mode()
def extract_features(
    model: UniFDClipLinear,
    loader: DataLoader,
    device: torch.device,
    desc: str = "features",
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Return (N, D) features, (N,) labels, per-batch wall latencies (ms)."""
    model.eval()
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    latencies: list[float] = []

    for images, y in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=False)
        t0 = time.perf_counter()
        # forward() already no-grads the backbone; head unused here
        with torch.no_grad():
            f = model._encode_image(images)
            if f.ndim > 2:
                f = f.flatten(1)
            f = f.float().cpu()
        latencies.append((time.perf_counter() - t0) * 1000.0 / max(images.size(0), 1))
        feats.append(f)
        labels.append(y.long())

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0), latencies


def train_linear_probe(
    feats: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> nn.Linear:
    """Fit a linear classifier on precomputed features."""
    device = device or torch.device("cpu")
    embed_dim = feats.shape[1]
    head = nn.Linear(embed_dim, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    n = feats.shape[0]
    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = feats[idx].to(device)
            yb = labels[idx].float().unsqueeze(1).to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        logger.info(
            "probe epoch %d/%d loss=%.4f",
            epoch + 1,
            epochs,
            total_loss / max(n_batches, 1),
        )
    head.eval()
    return head


@torch.inference_mode()
def eval_probe(
    head: nn.Linear,
    feats: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or torch.device("cpu")
    head.eval()
    logits = head(feats.to(device)).squeeze(-1).cpu()
    probs = torch.sigmoid(logits).numpy()
    y = labels.numpy().astype(int)
    preds = (probs >= threshold).astype(int)
    acc = float(accuracy_score(y, preds))
    try:
        ap = float(average_precision_score(y, probs))
    except ValueError:
        ap = float("nan")
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "ap": ap, "auc": auc, "n": int(len(y))}


def save_checkpoint(
    head: nn.Linear,
    path: Path,
    meta: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu()},
        "head": head.state_dict(),
        "meta": meta or {},
    }
    # Also store plain Linear keys at top level for UniFDClipLinear.load_probe_weights
    torch.save(
        {
            "weight": head.weight.detach().cpu().clone(),
            "bias": head.bias.detach().cpu().clone(),
            "meta": meta or {},
        },
        path,
    )
    logger.info("Saved probe checkpoint → %s", path)
    return path


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    metrics_cfg = cfg.get("metrics", {})

    hf_id = data_cfg.get("hf_id", DEFAULT_HF_ID)
    cache_dir = Path(data_cfg.get("cache_dir", str(DEFAULT_CACHE_DIR)))
    max_train = data_cfg.get("max_train")
    max_test = data_cfg.get("max_test")
    image_size = int(model_cfg.get("image_size", 224))
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    probe_batch = int(train_cfg.get("probe_batch_size", 512))
    threshold = float(model_cfg.get("threshold", 0.5))
    jpeg_prob = float(train_cfg.get("jpeg_prob", 0.0))
    blur_prob = float(train_cfg.get("blur_prob", 0.0))
    use_aug = bool(train_cfg.get("aug", False))

    device = resolve_device(str(train_cfg.get("device", "auto")))
    logger.info("device=%s", device)

    train_tf = (
        build_train_preprocess(image_size, jpeg_prob=jpeg_prob, blur_prob=blur_prob)
        if use_aug
        else build_clip_preprocess(image_size)
    )
    test_tf = build_clip_preprocess(image_size)

    data_name = str(data_cfg.get("name", "cifake")).lower()
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
            transform=test_tf,
            max_per_class=int(max_test) if max_test else None,
            seed=seed,
        )
    else:
        train_ds = CIFAKEDataset(
            split="train",
            transform=train_tf,
            hf_id=hf_id,
            cache_dir=cache_dir,
            max_samples=int(max_train) if max_train else None,
            seed=seed,
        )
        test_ds = CIFAKEDataset(
            split="test",
            transform=test_tf,
            hf_id=hf_id,
            cache_dir=cache_dir,
            max_samples=int(max_test) if max_test else None,
            seed=seed,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UniFDClipLinear(
        backbone=str(model_cfg.get("backbone", "ViT-L-14")),
        pretrained=str(model_cfg.get("pretrained", "openai")),
        backend=str(model_cfg.get("backend", "open_clip")),
    ).to(device)
    model.eval()

    # Feature cache key
    subset_tag = f"tr{len(train_ds)}_te{len(test_ds)}"
    cache_root = Path(train_cfg.get("feature_cache_dir", "artifacts/feature_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_prefix = "progan" if data_name in ("progan_folder", "folder", "real_fake_folder") else "cifake"
    feat_path = cache_root / f"{cache_prefix}_{subset_tag}_seed{seed}_sz{image_size}.pt"

    if feat_path.is_file() and not bool(train_cfg.get("force_reextract", False)) and not use_aug:
        logger.info("Loading cached features %s", feat_path)
        blob = torch.load(feat_path, map_location="cpu", weights_only=False)
        train_feats, train_y = blob["train_feats"], blob["train_y"]
        test_feats, test_y = blob["test_feats"], blob["test_y"]
        latencies = blob.get("latencies", [])
    else:
        logger.info("Extracting CLIP features (train n=%d)…", len(train_ds))
        train_feats, train_y, _ = extract_features(model, train_loader, device, desc="train-feat")
        logger.info("Extracting CLIP features (test n=%d)…", len(test_ds))
        test_feats, test_y, latencies = extract_features(
            model, test_loader, device, desc="test-feat"
        )
        if not use_aug:
            torch.save(
                {
                    "train_feats": train_feats,
                    "train_y": train_y,
                    "test_feats": test_feats,
                    "test_y": test_y,
                    "latencies": latencies,
                    "hf_id": hf_id,
                },
                feat_path,
            )
            logger.info("Cached features → %s", feat_path)

    head = train_linear_probe(
        train_feats,
        train_y,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=probe_batch,
        device=device,
    )
    metrics = eval_probe(head, test_feats, test_y, threshold=threshold, device=device)

    # Latency stats from per-image feature extract on test
    lat_p50 = float(np.percentile(latencies, 50)) if latencies else None
    lat_p95 = float(np.percentile(latencies, 95)) if latencies else None

    is_subset = max_train is not None or max_test is not None
    if data_name in ("progan_folder", "folder", "real_fake_folder"):
        dataset_name = "progan_holdout" if not is_subset else "progan_holdout_subset"
        ckpt_tag = "progan_probe"
        default_weights = "weights/progan_clip_vit_l14_linear.pth"
    else:
        dataset_name = "cifake_subset" if is_subset else "cifake"
        ckpt_tag = "cifake_probe"
        default_weights = "weights/cifake_clip_vit_l14_linear.pth"
    cfg_hash = _config_hash(cfg)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    out_dir = Path(metrics_cfg.get("output_dir", "artifacts/runs"))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "artifacts/checkpoints"))
    ckpt_path = ckpt_dir / f"{run_id}_{ckpt_tag}.pth"
    weights_copy = Path(train_cfg.get("weights_out", default_weights))

    meta = {
        "dataset": dataset_name,
        "data_name": data_name,
        "hf_id": hf_id if data_name.startswith("cifake") else None,
        "train_root": data_cfg.get("train_root"),
        "test_root": data_cfg.get("test_root"),
        "max_train": max_train,
        "max_test": max_test,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "metrics": metrics,
        "config_hash": cfg_hash,
        "model_id": model_cfg.get("id", __model_id__),
    }
    save_checkpoint(head, ckpt_path, meta=meta)
    save_checkpoint(head, weights_copy, meta=meta)

    # Also attach head onto model and save full-compatible path
    model.head.load_state_dict(
        {"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu()}
    )

    notes = (
        f"Linear probe on frozen CLIP; data={dataset_name}; "
        f"{'SUBSET' if is_subset else 'FULL'} "
        f"(train={len(train_ds)} test={len(test_ds)}); "
        f"ckpt={ckpt_path.as_posix()}; aug={use_aug}"
    )
    record = MetricsRecord(
        dataset=dataset_name,
        split="test",
        model_id=str(model_cfg.get("id", __model_id__)),
        acc=round(metrics["acc"], 6),
        ap=round(metrics["ap"], 6) if metrics["ap"] == metrics["ap"] else None,
        auc=round(metrics["auc"], 6) if metrics["auc"] == metrics["auc"] else None,
        n=int(metrics["n"]),
        threshold=threshold,
        latency_p50_ms=round(lat_p50, 3) if lat_p50 is not None else None,
        latency_p95_ms=round(lat_p95, 3) if lat_p95 is not None else None,
        aug={
            "jpeg": use_aug and jpeg_prob > 0,
            "blur": use_aug and blur_prob > 0,
            "jpeg_prob": jpeg_prob if use_aug else 0.0,
            "blur_prob": blur_prob if use_aug else 0.0,
            "resize": False,
        },
        notes=notes,
        config_hash=cfg_hash,
    )
    metrics_path = write_metrics(record, run_id=run_id, output_dir=out_dir)

    # Side-car config dump
    (metrics_path.parent / "config.json").write_text(
        json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (metrics_path.parent / "checkpoint_path.txt").write_text(
        ckpt_path.as_posix() + "\n", encoding="utf-8"
    )

    result = {
        "run_id": run_id,
        "metrics_path": str(metrics_path),
        "checkpoint": str(ckpt_path),
        "weights_copy": str(weights_copy),
        "metrics": metrics,
        "dataset": dataset_name,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "is_subset": is_subset,
        "config_hash": cfg_hash,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train CLIP linear probe (CIFAKE or ProGAN folders)")
    parser.add_argument(
        "--config",
        default="configs/train_cifake_subset.yaml",
        help="YAML config (subset default for CPU-friendly first run)",
    )
    parser.add_argument("--data-root", default=None, help="Override data.cache_dir")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--aug", action="store_true", help="Enable Blur/JPEG train aug")
    parser.add_argument("--force-reextract", action="store_true")
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    cfg.setdefault("model", {})
    cfg.setdefault("metrics", {})
    cfg["_config_path"] = args.config

    if args.data_root:
        cfg["data"]["cache_dir"] = args.data_root
    if args.max_train is not None:
        cfg["data"]["max_train"] = args.max_train
    if args.max_test is not None:
        cfg["data"]["max_test"] = args.max_test
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.device:
        cfg["train"]["device"] = args.device
    if args.aug:
        cfg["train"]["aug"] = True
    if args.force_reextract:
        cfg["train"]["force_reextract"] = True

    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
