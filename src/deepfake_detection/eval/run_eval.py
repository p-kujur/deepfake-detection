"""Evaluation entry: Acc/AP/AUC on a CIFAKE split (or smoke stub).

Writes ``artifacts/runs/<id>/metrics.json`` per plan contract.
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection import __model_id__
from deepfake_detection.data.cifake import CIFAKEDataset, DEFAULT_CACHE_DIR
from deepfake_detection.data.transforms import build_clip_preprocess
from deepfake_detection.metrics import MetricsRecord, write_metrics
from deepfake_detection.models.unifd import load_detector, resolve_device

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@torch.inference_mode()
def eval_loader(model, loader, device, threshold: float = 0.5):
    model.eval()
    probs_all = []
    labels_all = []
    latencies = []
    for images, y in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        t0 = time.perf_counter()
        probs = model.predict_proba(images).cpu().numpy()
        latencies.extend(
            [(time.perf_counter() - t0) * 1000.0 / max(images.size(0), 1)] * images.size(0)
        )
        probs_all.append(probs)
        labels_all.append(y.numpy())
    probs = np.concatenate(probs_all)
    y = np.concatenate(labels_all).astype(int)
    preds = (probs >= threshold).astype(int)
    acc = float(accuracy_score(y, preds))
    ap = float(average_precision_score(y, probs))
    auc = float(roc_auc_score(y, probs))
    return {
        "acc": acc,
        "ap": ap,
        "auc": auc,
        "n": int(len(y)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Eval → artifacts/runs/<id>/metrics.json")
    parser.add_argument("--dataset", default="cifake", choices=["cifake", "smoke", "stub"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--probe-weights", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    model_id = args.model_id or model_cfg.get("id", __model_id__)
    threshold = float(model_cfg.get("threshold", 0.5))
    image_size = int(model_cfg.get("image_size", 224))
    probe = args.probe_weights or model_cfg.get("probe_weights")

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

    if args.dataset in ("stub", "smoke") and args.dataset == "stub":
        record = MetricsRecord(
            dataset=args.dataset,
            split=args.split,
            model_id=model_id,
            acc=None,
            ap=None,
            auc=None,
            n=0,
            threshold=threshold,
            latency_p50_ms=None,
            latency_p95_ms=None,
            aug={"jpeg": False, "resize": False},
            notes="Eval stub — no dataset run.",
            config_path=args.config,
        )
        path = write_metrics(record, run_id=run_id)
        print(f"Wrote stub metrics: {path}")
        return 0

    if args.dataset == "cifake":
        device = resolve_device(args.device)
        model, device = load_detector(
            probe_weights=probe,
            backend=str(model_cfg.get("backend", "open_clip")),
            backbone=str(model_cfg.get("backbone", "ViT-L-14")),
            pretrained=str(model_cfg.get("pretrained", "openai")),
            device=str(device),
            require_weights=bool(probe),
        )
        ds = CIFAKEDataset(
            split=args.split,
            transform=build_clip_preprocess(image_size),
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        m = eval_loader(model, loader, device, threshold=threshold)
        is_subset = args.max_samples is not None
        record = MetricsRecord(
            dataset="cifake_subset" if is_subset else "cifake",
            split=args.split,
            model_id=model_id,
            acc=round(m["acc"], 6),
            ap=round(m["ap"], 6),
            auc=round(m["auc"], 6),
            n=m["n"],
            threshold=threshold,
            latency_p50_ms=round(m["latency_p50_ms"], 3),
            latency_p95_ms=round(m["latency_p95_ms"], 3),
            aug={"jpeg": False, "resize": False},
            notes=f"CIFAKE eval; probe={probe}; subset={is_subset}",
            config_path=args.config,
        )
        path = write_metrics(record, run_id=run_id)
        print(f"Wrote metrics: {path}")
        print(
            f"acc={m['acc']:.4f} ap={m['ap']:.4f} auc={m['auc']:.4f} n={m['n']}"
        )
        return 0

    # smoke: leave as thin stub pointing at fixtures (covered by pytest)
    record = MetricsRecord(
        dataset="smoke",
        split=args.split,
        model_id=model_id,
        acc=None,
        ap=None,
        auc=None,
        n=0,
        threshold=threshold,
        latency_p50_ms=None,
        latency_p95_ms=None,
        aug={"jpeg": False, "resize": False},
        notes="Use pytest tests/test_smoke_infer.py for labeled smoke pack.",
        config_path=args.config,
    )
    path = write_metrics(record, run_id=run_id)
    print(f"Wrote smoke placeholder metrics: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
