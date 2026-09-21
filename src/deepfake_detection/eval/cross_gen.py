"""M3 cross-generator eval, score fusion, and JPEG/resize robustness.

Writes ``artifacts/runs/<id>/metrics.json`` per plan contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection.data.folder_dataset import RealFakeFolderDataset
from deepfake_detection.data.transforms import build_eval_preprocess
from deepfake_detection.metrics import MetricsRecord, write_metrics
from deepfake_detection.models.npr_lite import NPRLiteCNN, load_npr_lite
from deepfake_detection.models.unifd import load_detector, resolve_device

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _scores_to_metrics(
    probs: np.ndarray, labels: np.ndarray, threshold: float, latencies: list[float]
) -> dict[str, float]:
    y = labels.astype(int)
    preds = (probs >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(y, preds)),
        "ap": float(average_precision_score(y, probs)),
        "auc": float(roc_auc_score(y, probs)),
        "n": int(len(y)),
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else float("nan"),
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else float("nan"),
    }


@torch.inference_mode()
def score_model(model, loader, device, threshold: float = 0.5, desc: str = "eval"):
    model.eval()
    probs_all, labels_all, lats = [], [], []
    for images, y in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        t0 = time.perf_counter()
        probs = model.predict_proba(images).cpu().numpy()
        dt = (time.perf_counter() - t0) * 1000.0 / max(images.size(0), 1)
        lats.extend([dt] * images.size(0))
        probs_all.append(probs)
        labels_all.append(y.numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    return probs, labels, _scores_to_metrics(probs, labels, threshold, lats)


def fuse_probs(p_a: np.ndarray, p_b: np.ndarray, weight_a: float = 0.5) -> np.ndarray:
    w = float(weight_a)
    return w * p_a + (1.0 - w) * p_b


def _write_run(
    *,
    dataset: str,
    split: str,
    model_id: str,
    metrics: dict[str, float],
    threshold: float,
    aug: dict[str, Any],
    notes: str,
    config_path: str | None,
    tag: str,
) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8] + "-" + tag
    record = MetricsRecord(
        dataset=dataset,
        split=split,
        model_id=model_id,
        acc=round(metrics["acc"], 6),
        ap=round(metrics["ap"], 6),
        auc=round(metrics["auc"], 6),
        n=int(metrics["n"]),
        threshold=threshold,
        latency_p50_ms=round(metrics["latency_p50_ms"], 3)
        if metrics["latency_p50_ms"] == metrics["latency_p50_ms"]
        else None,
        latency_p95_ms=round(metrics["latency_p95_ms"], 3)
        if metrics["latency_p95_ms"] == metrics["latency_p95_ms"]
        else None,
        aug=aug,
        notes=notes,
        config_path=config_path,
    )
    path = write_metrics(record, run_id=run_id)
    print(
        f"[{tag}] {dataset} acc={metrics['acc']:.4f} ap={metrics['ap']:.4f} "
        f"auc={metrics['auc']:.4f} n={metrics['n']} → {path}"
    )
    return path


def run_pack(
    pack_name: str,
    pack_root: Path,
    unifd_model,
    device,
    *,
    npr_model: NPRLiteCNN | None,
    npr_image_size: int,
    clip_image_size: int,
    batch_size: int,
    threshold: float,
    fusion_weight: float,
    max_per_class: int | None,
    jpeg_quality: int | None,
    resize_short: int | None,
    config_path: str | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {"pack": pack_name, "root": str(pack_root)}

    clip_tf = build_eval_preprocess(
        image_size=clip_image_size,
        jpeg_quality=jpeg_quality,
        resize_short=resize_short,
        clip_norm=True,
    )
    ds_clip = RealFakeFolderDataset(
        pack_root, transform=clip_tf, max_per_class=max_per_class, seed=42
    )
    loader_clip = DataLoader(ds_clip, batch_size=batch_size, shuffle=False, num_workers=0)
    p_u, y, m_u = score_model(unifd_model, loader_clip, device, threshold=threshold, desc="unifd")

    aug = {
        "jpeg": jpeg_quality is not None,
        "jpeg_quality": jpeg_quality,
        "resize": resize_short is not None,
        "resize_short": resize_short,
    }
    tag_suffix = ""
    if jpeg_quality is not None:
        tag_suffix += f"-jpeg{jpeg_quality}"
    if resize_short is not None:
        tag_suffix += f"-rs{resize_short}"

    _write_run(
        dataset=f"crossgen_{pack_name}",
        split="test",
        model_id="unifd-clip-vit-l14-linear",
        metrics=m_u,
        threshold=threshold,
        aug=aug,
        notes=f"M3 cross-gen UniFD probe on {pack_name}; root={pack_root}",
        config_path=config_path,
        tag=f"cg-{pack_name}-unifd{tag_suffix}",
    )
    results["unifd"] = m_u

    if npr_model is not None:
        cnn_tf = build_eval_preprocess(
            image_size=npr_image_size,
            jpeg_quality=jpeg_quality,
            resize_short=resize_short,
            clip_norm=False,
        )
        ds_cnn = RealFakeFolderDataset(
            pack_root, transform=cnn_tf, max_per_class=max_per_class, seed=42
        )
        loader_cnn = DataLoader(ds_cnn, batch_size=batch_size, shuffle=False, num_workers=0)
        p_c, y_c, m_c = score_model(npr_model, loader_cnn, device, threshold=threshold, desc="npr")
        assert np.array_equal(y, y_c), "label mismatch between UniFD and CNN loaders"
        _write_run(
            dataset=f"crossgen_{pack_name}",
            split="test",
            model_id="npr-lite-cnn-v1",
            metrics=m_c,
            threshold=threshold,
            aug=aug,
            notes=f"M3 cross-gen NPR-lite on {pack_name}",
            config_path=config_path,
            tag=f"cg-{pack_name}-npr{tag_suffix}",
        )
        results["npr_lite"] = m_c

        p_f = fuse_probs(p_u, p_c, weight_a=fusion_weight)
        m_f = _scores_to_metrics(p_f, y, threshold, [])
        m_f["latency_p50_ms"] = m_u["latency_p50_ms"]
        m_f["latency_p95_ms"] = m_u["latency_p95_ms"]
        _write_run(
            dataset=f"crossgen_{pack_name}",
            split="test",
            model_id="fusion-unifd+npr-lite",
            metrics=m_f,
            threshold=threshold,
            aug={**aug, "fusion_weight_unifd": fusion_weight},
            notes=f"M3 mean-prob fusion UniFD×{fusion_weight} + NPR-lite×{1 - fusion_weight}",
            config_path=config_path,
            tag=f"cg-{pack_name}-fusion{tag_suffix}",
        )
        results["fusion"] = m_f

    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="M3 cross-gen + fusion + robustness eval")
    parser.add_argument("--config", default="configs/eval_crossgen.yaml")
    parser.add_argument("--probe-weights", default="weights/cifake_clip_vit_l14_linear.pth")
    parser.add_argument("--npr-weights", default="weights/npr_lite_cifake.pth")
    parser.add_argument("--data-root", default="datasets/crossgen")
    parser.add_argument("--packs", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--skip-npr", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--fusion-weight", type=float, default=0.5)
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    rob_cfg = cfg.get("robustness", {})
    threshold = float(model_cfg.get("threshold", 0.5))
    clip_size = int(model_cfg.get("image_size", 224))
    npr_size = int(model_cfg.get("npr_image_size", 128))

    packs = args.packs or data_cfg.get("packs") or ["progan_eval", "hemg_wild"]
    max_per_class = args.max_per_class if args.max_per_class is not None else data_cfg.get("max_per_class")
    data_root = Path(args.data_root)

    device = resolve_device(args.device)
    unifd, device = load_detector(
        probe_weights=args.probe_weights,
        backend=str(model_cfg.get("backend", "open_clip")),
        backbone=str(model_cfg.get("backbone", "ViT-L-14")),
        pretrained=str(model_cfg.get("pretrained", "openai")),
        device=str(device),
        require_weights=True,
    )

    npr_model = None
    if not args.skip_npr:
        npr_path = Path(args.npr_weights)
        if npr_path.is_file():
            npr_model = load_npr_lite(npr_path, device=device)
        else:
            logger.warning("NPR-lite weights missing at %s — UniFD-only eval", npr_path)

    summary: dict[str, Any] = {"packs": {}, "robustness": {}}
    for pack in packs:
        root = data_root / pack
        if not root.is_dir():
            logger.warning("Pack missing: %s — skip", root)
            continue
        summary["packs"][pack] = run_pack(
            pack,
            root,
            unifd,
            device,
            npr_model=npr_model,
            npr_image_size=npr_size,
            clip_image_size=clip_size,
            batch_size=args.batch_size,
            threshold=threshold,
            fusion_weight=args.fusion_weight,
            max_per_class=int(max_per_class) if max_per_class else None,
            jpeg_quality=None,
            resize_short=None,
            config_path=args.config,
        )

        if not args.skip_robustness:
            jpeg_q = int(rob_cfg.get("jpeg_quality", 70))
            rs = int(rob_cfg.get("resize_short", 128))
            summary["robustness"][f"{pack}_jpeg{jpeg_q}"] = run_pack(
                pack,
                root,
                unifd,
                device,
                npr_model=npr_model,
                npr_image_size=npr_size,
                clip_image_size=clip_size,
                batch_size=args.batch_size,
                threshold=threshold,
                fusion_weight=args.fusion_weight,
                max_per_class=int(max_per_class) if max_per_class else None,
                jpeg_quality=jpeg_q,
                resize_short=None,
                config_path=args.config,
            )
            summary["robustness"][f"{pack}_resize{rs}"] = run_pack(
                pack,
                root,
                unifd,
                device,
                npr_model=npr_model,
                npr_image_size=npr_size,
                clip_image_size=clip_size,
                batch_size=args.batch_size,
                threshold=threshold,
                fusion_weight=args.fusion_weight,
                max_per_class=int(max_per_class) if max_per_class else None,
                jpeg_quality=None,
                resize_short=rs,
                config_path=args.config,
            )

    out = Path("artifacts/runs") / (
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8] + "-m3-summary"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"Wrote M3 summary: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
