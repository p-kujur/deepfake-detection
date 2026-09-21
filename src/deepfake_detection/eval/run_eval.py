"""Evaluation entry (stub): compute acc/ap/auc and write metrics.json."""

from __future__ import annotations

import argparse
import time
import uuid

from deepfake_detection.metrics import MetricsRecord, write_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval stub → artifacts/runs/<id>/metrics.json")
    parser.add_argument("--dataset", default="smoke")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-id", default="unifd-clip-vit-l14-linear")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    record = MetricsRecord(
        dataset=args.dataset,
        split=args.split,
        model_id=args.model_id,
        acc=None,
        ap=None,
        auc=None,
        n=0,
        threshold=0.5,
        latency_p50_ms=None,
        latency_p95_ms=None,
        aug={"jpeg": False, "resize": False},
        notes="M1 eval stub — no dataset run; placeholders only.",
        config_path=args.config,
    )
    path = write_metrics(record, run_id=run_id)
    print(f"Wrote stub metrics: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
