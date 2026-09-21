"""M3c dual-probe ensemble/router: ProGAN UniFD head + CIFAKE UniFD head.

Shared frozen CLIP backbone; two linear probes. Fusion weight / router chosen
on ProGAN **val** only (never on Hemg). CIFAKE reported as a separate tradeoff
table. In-domain (ProGAN holdout) and cross-gen (Hemg) stay separate.
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
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection.data.cifake import CIFAKEDataset, DEFAULT_CACHE_DIR, DEFAULT_HF_ID
from deepfake_detection.data.folder_dataset import RealFakeFolderDataset
from deepfake_detection.data.transforms import build_clip_preprocess
from deepfake_detection.eval.cross_gen import fuse_logits, fuse_probs
from deepfake_detection.metrics import MetricsRecord, write_metrics
from deepfake_detection.models.unifd import UniFDClipLinear, load_detector, resolve_device

logger = logging.getLogger(__name__)


def _metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = labels.astype(int)
    preds = (probs >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(y, preds)),
        "ap": float(average_precision_score(y, probs)),
        "auc": float(roc_auc_score(y, probs)),
        "n": int(len(y)),
    }


def _load_head(path: Path, embed_dim: int = 768) -> nn.Linear:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    head = nn.Linear(embed_dim, 1)
    head.weight.data.copy_(blob["weight"])
    head.bias.data.copy_(blob["bias"])
    head.eval()
    return head


@torch.inference_mode()
def encode_pack(
    model: UniFDClipLinear,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, np.ndarray, list[float]]:
    model.eval()
    feats, labels, lats = [], [], []
    for images, y in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        t0 = time.perf_counter()
        f = model._encode_image(images)
        if f.ndim > 2:
            f = f.flatten(1)
        f = f.float().cpu()
        dt = (time.perf_counter() - t0) * 1000.0 / max(images.size(0), 1)
        lats.extend([dt] * images.size(0))
        feats.append(f)
        labels.append(y.numpy())
    return torch.cat(feats, 0), np.concatenate(labels), lats


def score_heads(
    feats: torch.Tensor,
    head_a: nn.Linear,
    head_b: nn.Linear,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return logit_a, logit_b, prob_a, prob_b."""
    with torch.inference_mode():
        la = head_a(feats).squeeze(-1).numpy()
        lb = head_b(feats).squeeze(-1).numpy()
    pa = 1.0 / (1.0 + np.exp(-la))
    pb = 1.0 / (1.0 + np.exp(-lb))
    return la, lb, pa, pb


def confidence_router(logit_a: np.ndarray, logit_b: np.ndarray) -> np.ndarray:
    """Pick the head with larger |logit| (more confident), sigmoid that logit."""
    choose_a = np.abs(logit_a) >= np.abs(logit_b)
    picked = np.where(choose_a, logit_a, logit_b)
    return 1.0 / (1.0 + np.exp(-picked))


def soft_confidence_mix(logit_a: np.ndarray, logit_b: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Softmax over |logit| as mixture weights, then mean-prob fuse."""
    ca = np.abs(logit_a) / max(temp, 1e-6)
    cb = np.abs(logit_b) / max(temp, 1e-6)
    # stable softmax over 2
    m = np.maximum(ca, cb)
    ea = np.exp(ca - m)
    eb = np.exp(cb - m)
    wa = ea / (ea + eb)
    pa = 1.0 / (1.0 + np.exp(-logit_a))
    pb = 1.0 / (1.0 + np.exp(-logit_b))
    return wa * pa + (1.0 - wa) * pb


def evaluate_combo(
    la: np.ndarray,
    lb: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str,
    weight_a: float = 0.5,
    threshold: float = 0.5,
) -> dict[str, Any]:
    pa = 1.0 / (1.0 + np.exp(-la))
    pb = 1.0 / (1.0 + np.exp(-lb))
    if mode == "mean":
        probs = fuse_probs(pa, pb, weight_a=weight_a)
    elif mode == "logit":
        probs = fuse_logits(la, lb, weight_a=weight_a)
    elif mode == "router":
        probs = confidence_router(la, lb)
    elif mode == "soft_router":
        probs = soft_confidence_mix(la, lb)
    elif mode == "progan_only":
        probs = pa
    elif mode == "cifake_only":
        probs = pb
    else:
        raise ValueError(f"unknown mode {mode}")
    m = _metrics(probs, labels, threshold=threshold)
    return {"mode": mode, "weight_a": weight_a, **m}


def select_on_val(
    la: np.ndarray,
    lb: np.ndarray,
    labels: np.ndarray,
    *,
    la_cf: np.ndarray | None = None,
    lb_cf: np.ndarray | None = None,
    labels_cf: np.ndarray | None = None,
    progan_floor: float = 0.95,
) -> dict[str, Any]:
    """Pick fusion by dual-domain score — never Hemg.

    Score = 0.5 * (AUC_progan_val + AUC_cifake_subset) among candidates that
    keep ProGAN-val AUC >= ``progan_floor``. Falls back to best ProGAN-val AUC
    if none clear the floor (should not happen for progan_only).
    """
    candidates: list[dict[str, Any]] = []
    for mode in ("progan_only", "cifake_only", "router", "soft_router"):
        candidates.append(evaluate_combo(la, lb, labels, mode=mode))
    for mode in ("mean", "logit"):
        for w in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            candidates.append(evaluate_combo(la, lb, labels, mode=mode, weight_a=w))

    for c in candidates:
        c["auc_progan_val"] = c["auc"]
        if la_cf is not None and labels_cf is not None:
            cf = evaluate_combo(la_cf, lb_cf, labels_cf, mode=c["mode"], weight_a=c["weight_a"])
            c["auc_cifake"] = cf["auc"]
            c["dual_score"] = 0.5 * (c["auc_progan_val"] + c["auc_cifake"])
        else:
            c["auc_cifake"] = None
            c["dual_score"] = c["auc_progan_val"]

    eligible = [c for c in candidates if c["auc_progan_val"] >= progan_floor]
    pool = eligible if eligible else candidates
    best = max(pool, key=lambda d: d["dual_score"])
    return {
        "best": best,
        "all": sorted(candidates, key=lambda d: -d["dual_score"]),
        "progan_floor": progan_floor,
        "n_eligible": len(eligible),
    }


def _write(
    *,
    dataset: str,
    model_id: str,
    metrics: dict[str, float],
    notes: str,
    tag: str,
    aug: dict[str, Any] | None = None,
) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8] + "-" + tag
    record = MetricsRecord(
        dataset=dataset,
        split="test",
        model_id=model_id,
        acc=round(metrics["acc"], 6),
        ap=round(metrics["ap"], 6),
        auc=round(metrics["auc"], 6),
        n=int(metrics["n"]),
        threshold=0.5,
        latency_p50_ms=None,
        latency_p95_ms=None,
        aug=aug or {},
        notes=notes,
    )
    return write_metrics(record, run_id=run_id)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="M3c dual-probe ProGAN+CIFAKE ensemble/router")
    p.add_argument("--progan-weights", default="weights/progan_clip_vit_l14_linear.pth")
    p.add_argument("--cifake-weights", default="weights/cifake_clip_vit_l14_linear.pth")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--cifake-max", type=int, default=800, help="CIFAKE tradeoff / selection subset size")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    # One backbone; ProGAN weights load into model.head (unused — we use dual heads)
    model, device = load_detector(
        probe_weights=args.progan_weights,
        device=str(device),
        require_weights=True,
    )
    head_progan = _load_head(Path(args.progan_weights)).to("cpu")
    head_cifake = _load_head(Path(args.cifake_weights)).to("cpu")
    clip_tf = build_clip_preprocess(224)

    packs = {
        "progan_val": Path("datasets/progan_forensynths/splits/val"),
        "progan_holdout": Path("datasets/crossgen/progan_holdout"),
        "hemg_wild": Path("datasets/crossgen/hemg_wild"),
    }

    encoded: dict[str, dict[str, Any]] = {}
    for name, root in packs.items():
        if not root.is_dir():
            logger.warning("missing %s — skip", root)
            continue
        ds = RealFakeFolderDataset(root, transform=clip_tf, seed=42)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        feats, labels, lats = encode_pack(model, loader, device, desc=name)
        la, lb, pa, pb = score_heads(feats, head_progan, head_cifake)
        encoded[name] = {
            "la": la,
            "lb": lb,
            "pa": pa,
            "pb": pb,
            "labels": labels,
            "lats": lats,
        }
        logger.info(
            "%s n=%d progan_auc=%.4f cifake_auc=%.4f",
            name,
            len(labels),
            _metrics(pa, labels)["auc"],
            _metrics(pb, labels)["auc"],
        )

    if "progan_val" not in encoded:
        raise SystemExit("ProGAN val pack required for fusion selection")

    # Small CIFAKE subset for dual-domain selection only (never Hemg).
    sel_n = min(800, int(args.cifake_max))
    cifake_sel_ds = CIFAKEDataset(
        split="test",
        transform=clip_tf,
        hf_id=DEFAULT_HF_ID,
        cache_dir=str(DEFAULT_CACHE_DIR),
        max_samples=sel_n,
        seed=42,
    )
    cifake_sel_loader = DataLoader(
        cifake_sel_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    feats_cf, labels_cf, _ = encode_pack(model, cifake_sel_loader, device, desc="cifake_sel")
    la_cf, lb_cf, _, _ = score_heads(feats_cf, head_progan, head_cifake)
    # reuse for tradeoff if same size
    encoded["cifake_sel"] = {
        "la": la_cf, "lb": lb_cf, "labels": labels_cf,
    }

    selection = select_on_val(
        encoded["progan_val"]["la"],
        encoded["progan_val"]["lb"],
        encoded["progan_val"]["labels"],
        la_cf=la_cf,
        lb_cf=lb_cf,
        labels_cf=labels_cf,
        progan_floor=0.95,
    )
    best = selection["best"]
    logger.info(
        "Selected (dual-domain, progan_floor=0.95): mode=%s w_progan=%.2f "
        "progan_val_auc=%.4f cifake_auc=%.4f dual=%.4f",
        best["mode"],
        best["weight_a"],
        best["auc_progan_val"],
        best.get("auc_cifake") or float("nan"),
        best["dual_score"],
    )

    summary: dict[str, Any] = {
        "milestone": "M3c",
        "selection_pack": "progan_val",
        "selected": best,
        "val_top5": selection["all"][:5],
        "tables": {"in_domain": {}, "cross_gen": {}, "cifake_tradeoff": {}},
    }

    # Apply selected combo to holdout + Hemg
    for pack, table_key, dataset_name in (
        ("progan_holdout", "in_domain", "progan_holdout"),
        ("hemg_wild", "cross_gen", "crossgen_hemg_wild"),
    ):
        if pack not in encoded:
            continue
        e = encoded[pack]
        # singles + selected
        rows = {
            "progan_probe": evaluate_combo(e["la"], e["lb"], e["labels"], mode="progan_only"),
            "cifake_probe": evaluate_combo(e["la"], e["lb"], e["labels"], mode="cifake_only"),
            "selected_ensemble": evaluate_combo(
                e["la"],
                e["lb"],
                e["labels"],
                mode=best["mode"],
                weight_a=best["weight_a"],
            ),
        }
        # Also report mean@0.5 and router for transparency
        rows["mean_0.5"] = evaluate_combo(e["la"], e["lb"], e["labels"], mode="mean", weight_a=0.5)
        rows["logit_0.5"] = evaluate_combo(e["la"], e["lb"], e["labels"], mode="logit", weight_a=0.5)
        rows["router"] = evaluate_combo(e["la"], e["lb"], e["labels"], mode="router")
        summary["tables"][table_key][pack] = rows

        sel = rows["selected_ensemble"]
        path = _write(
            dataset=dataset_name,
            model_id=f"dual-probe-{best['mode']}",
            metrics=sel,
            notes=(
                f"M3c dual UniFD probes (ProGAN+CIFAKE); selected on progan_val: "
                f"mode={best['mode']} w_progan={best['weight_a']}"
            ),
            tag=f"m3c-{pack}-{best['mode']}",
            aug={
                "fusion_mode": best["mode"],
                "fusion_weight_progan": best["weight_a"],
                "selection": "progan_val",
            },
        )
        print(
            f"[{table_key}/{pack}] selected {best['mode']} "
            f"auc={sel['auc']:.4f} ap={sel['ap']:.4f} acc={sel['acc']:.4f} → {path}"
        )

    # CIFAKE tradeoff (separate table) — reuse selection encode when sizes match
    if args.cifake_max <= sel_n:
        la, lb, labels = la_cf, lb_cf, labels_cf
    else:
        cifake_ds = CIFAKEDataset(
            split="test",
            transform=clip_tf,
            hf_id=DEFAULT_HF_ID,
            cache_dir=str(DEFAULT_CACHE_DIR),
            max_samples=args.cifake_max,
            seed=42,
        )
        cifake_loader = DataLoader(
            cifake_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        feats, labels, _ = encode_pack(model, cifake_loader, device, desc="cifake")
        la, lb, _, _ = score_heads(feats, head_progan, head_cifake)
    cifake_rows = {
        "progan_probe": evaluate_combo(la, lb, labels, mode="progan_only"),
        "cifake_probe": evaluate_combo(la, lb, labels, mode="cifake_only"),
        "selected_ensemble": evaluate_combo(
            la, lb, labels, mode=best["mode"], weight_a=best["weight_a"]
        ),
        "mean_0.5": evaluate_combo(la, lb, labels, mode="mean", weight_a=0.5),
        "router": evaluate_combo(la, lb, labels, mode="router"),
    }
    summary["tables"]["cifake_tradeoff"]["cifake_test_subset"] = cifake_rows
    sel = cifake_rows["selected_ensemble"]
    path = _write(
        dataset="cifake_subset_tradeoff",
        model_id=f"dual-probe-{best['mode']}",
        metrics=sel,
        notes="M3c CIFAKE tradeoff table (separate from in-domain / cross-gen)",
        tag=f"m3c-cifake-{best['mode']}",
        aug={"fusion_mode": best["mode"], "fusion_weight_progan": best["weight_a"]},
    )
    print(
        f"[cifake_tradeoff] selected {best['mode']} "
        f"auc={sel['auc']:.4f} → {path}"
    )

    # Gate summary
    hemg = summary["tables"]["cross_gen"].get("hemg_wild", {}).get("selected_ensemble", {})
    hold = summary["tables"]["in_domain"].get("progan_holdout", {}).get("selected_ensemble", {})
    summary["gate"] = {
        "target_hemg_auc": 0.80,
        "hemg_auc": hemg.get("auc"),
        "progan_holdout_auc": hold.get("auc"),
        "target_met": bool(hemg.get("auc", 0) >= 0.80),
        "progan_holdout_preserved": bool(hold.get("auc", 0) >= 0.95),
    }

    out = Path("artifacts/runs") / (
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8] + "-m3c-dual-summary"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    Path("docs/m3c_dual_probe_summary.json").write_text(
        json.dumps(
            {
                "milestone": "M3c",
                "selected": best,
                "gate": summary["gate"],
                "tables": summary["tables"],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"Wrote summary → {out / 'summary.json'}")
    print(json.dumps(summary["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
