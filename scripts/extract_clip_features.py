#!/usr/bin/env python3
"""Extract frozen CLIP ViT-L/14 (OpenAI) features for probe training / eval (M5).

Sources:
  folder  : a pack with real/ and fake/ subfolders (ProGAN splits, Hemg, smoke)
  openfake: one or more OpenFake parquet shards (ComplexDataLab/OpenFake, CC-BY-NC-4.0)

Views (applied *before* CLIP preprocess, identically for real and fake):
  clean  : no change
  aug    : random downscale / JPEG(60-95) / light blur  (training view)
  social : longest side capped at 1024 + JPEG q=75     (robustness eval view)

Output: a torch .pt dict {feats (N,768) float16, labels (N,), meta: list[dict], view, source}.
"""

from __future__ import annotations

import argparse
import io
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageFile, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepfake_detection.data.transforms import build_clip_preprocess  # noqa: E402
from deepfake_detection.models.unifd import UniFDClipLinear  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract")
DRAFT_SHORT = 512
MAX_PIXELS_NON_JPEG = 40_000_000
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _cap_side(img: Image.Image, cap: int) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= cap:
        return img
    r = cap / s
    return img.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BICUBIC)


def apply_view(img: Image.Image, view: str, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    if view == "clean":
        return img
    if view == "social":
        return _jpeg(_cap_side(img, 1024), 75)
    if view == "aug":
        if rng.random() < 0.5:
            short = min(img.size)
            lo = max(0.4, 256 / max(short, 1))
            if lo < 1.0:
                f = rng.uniform(lo, 1.0)
                w, h = img.size
                img = img.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BICUBIC)
        if rng.random() < 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.5)))
        if rng.random() < 0.75:
            img = _jpeg(img, rng.randint(60, 95))
        return img
    raise ValueError(view)


def iter_folder(root: Path, max_per_class: int | None, seed: int):
    rng = random.Random(seed)
    for lab_name, lab in (("real", 0), ("fake", 1)):
        d = root / lab_name
        paths = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if max_per_class and len(paths) > max_per_class:
            paths = sorted(rng.sample(paths, max_per_class))
        for p in paths:
            yield (lambda p=p: p.read_bytes()), lab, {"src": str(root.name), "file": p.name, "model": f"{root.name}_{lab_name}"}


def iter_openfake(shards: list[Path], max_per_model: int, max_real_per_source: int, seed: int, exclude: set[str]):
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    # pass 1: pick rows using light columns
    picks: dict[Path, dict[int, list[int]]] = {}
    cands: dict[str, list[tuple[Path, int, int]]] = defaultdict(list)
    for sh in shards:
        pf = pq.ParquetFile(sh)
        off = 0
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["label", "model"])
            for i, (lab, mod) in enumerate(zip(t["label"].to_pylist(), t["model"].to_pylist())):
                key = f"{lab}::{mod}"
                if mod in exclude:
                    continue
                cands[key].append((sh, rg, i))
            off += t.num_rows
    chosen = []
    for key, rows in sorted(cands.items()):
        cap = max_real_per_source if key.startswith("real::") else max_per_model
        rng.shuffle(rows)
        chosen.extend(rows[:cap] if cap else rows)
    for sh, rg, i in chosen:
        picks.setdefault(sh, {}).setdefault(rg, []).append(i)
    tally = Counter()
    for sh, rgs in picks.items():
        for rg, idxs in rgs.items():
            for _ in idxs:
                tally["n"] += 1
    log.info("openfake: %d rows selected across %d models/sources", tally["n"], len(cands))
    # pass 2: stream image bytes for picked row groups only
    for sh, rgs in picks.items():
        pf = pq.ParquetFile(sh)
        for rg in sorted(rgs):
            t = pf.read_row_group(rg, columns=["image", "label", "model", "row_id", "type"])
            imgs = t["image"].to_pylist()
            labs = t["label"].to_pylist()
            mods = t["model"].to_pylist()
            rids = t["row_id"].to_pylist()
            typs = t["type"].to_pylist()
            for i in sorted(rgs[rg]):
                b = imgs[i]["bytes"]
                yield (lambda b=b: b), (1 if labs[i] == "fake" else 0), {
                    "src": sh.parent.name + "/" + sh.name,
                    "model": mods[i],
                    "type": typs[i],
                    "row_id": rids[i],
                }
            del t, imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["folder", "openfake"], required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--view", choices=["clean", "aug", "social"], default="clean")
    ap.add_argument("--views", nargs="*", default=None, help="extract several views in one decode pass")
    ap.add_argument("--out", required=True, help="output .pt (with --views: suffix _<view> added)")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--max-per-model", type=int, default=0)
    ap.add_argument("--max-real-per-source", type=int, default=0)
    ap.add_argument("--exclude-models", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    views = args.views or [args.view]
    model = UniFDClipLinear()
    model.eval()
    pre = build_clip_preprocess(224)
    rng = random.Random(args.seed)

    if args.source == "folder":
        gen = (x for root in args.inputs for x in iter_folder(Path(root), args.max_per_class, args.seed))
    else:
        gen = iter_openfake([Path(p) for p in args.inputs], args.max_per_model, args.max_real_per_source,
                            args.seed, set(args.exclude_models))

    out = Path(args.out)
    parts_dir = out.with_name(out.stem + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    done_parts = sorted(parts_dir.glob("part_*.pt"))
    n_consumed = 0  # items consumed from the (deterministic) generator, incl. skipped ones
    for pp in done_parts:
        n_consumed += torch.load(pp, weights_only=False)["consumed"]
    if done_parts:
        log.info("resume: %d parts, %d items already consumed", len(done_parts), n_consumed)

    def prep(item):
        get, lab, m = item
        t_item = time.time()
        try:
            img = Image.open(io.BytesIO(get()))
            if img.format != "JPEG" and img.size[0] * img.size[1] > MAX_PIXELS_NON_JPEG:
                log.warning("skip oversized %s %s %s", img.format, img.size, m)
                return None
            m = dict(m, fmt=img.format, w=img.size[0], h=img.size[1])
            w0, h0 = img.size
            if img.format == "JPEG" and min(w0, h0) > 2 * DRAFT_SHORT:
                # Huge photos (e.g. 24 MP Pexels): DCT-domain downscale so the short side
                # stays >= DRAFT_SHORT (>= 2x CLIP's 224). Speed only; negligible after Resize(224).
                k = DRAFT_SHORT / min(w0, h0)
                img.draft("RGB", (int(w0 * k) + 1, int(h0 * k) + 1))
                m["drafted"] = True
            img = img.convert("RGB")
            r = random.Random(hash((m.get("row_id") or m.get("file"), args.seed)))
            out_ = [pre(apply_view(img, v, r)) for v in views]
            if time.time() - t_item > 5:
                log.warning("slow item %.1fs %s", time.time() - t_item, m)
            return out_, lab, m
        except Exception as e:  # corrupt image
            log.warning("skip %s: %s", m, e)
            return None

    part = {"feats": {v: [] for v in views}, "labels": [], "meta": [], "consumed": 0}

    def save_part():
        if part["consumed"] == 0:
            return
        idx = len(list(parts_dir.glob("part_*.pt")))
        blob = {"feats": {v: (torch.cat(part["feats"][v]) if part["feats"][v] else torch.empty(0, 768).half()) for v in views},
                "labels": list(part["labels"]), "meta": list(part["meta"]), "consumed": part["consumed"]}
        tmp = parts_dir / f".tmp_{idx:05d}.pt"
        torch.save(blob, tmp)
        tmp.rename(parts_dir / f"part_{idx:05d}.pt")
        for v in views:
            part["feats"][v].clear()
        part["labels"].clear(); part["meta"].clear(); part["consumed"] = 0

    def flush(batch):
        outs = [o for o in map(prep, batch) if o is not None]
        part["consumed"] += len(batch)
        if not outs:
            return
        with torch.inference_mode():
            for vi, v in enumerate(views):
                x = torch.stack([o[0][vi] for o in outs])
                part["feats"][v].append(model._encode_image(x).float().half())
        part["labels"].extend(o[1] for o in outs)
        part["meta"].extend(o[2] for o in outs)

    t0 = time.time()
    batch = []
    n_new = 0
    for k, item in enumerate(gen):
        if k < n_consumed:
            continue
        batch.append(item)
        if len(batch) >= args.batch_size:
            flush(batch)
            n_new += len(batch)
            batch = []
            if part["consumed"] >= args.batch_size * 10:
                save_part()
                log.info("%d new imgs (+%d resumed)  %.2f img/s", n_new, n_consumed, n_new / (time.time() - t0))
    if batch:
        flush(batch)
    save_part()

    # merge parts -> final caches
    parts = [torch.load(pp, weights_only=False) for pp in sorted(parts_dir.glob("part_*.pt"))]
    labels = [l for p_ in parts for l in p_["labels"]]
    meta = [m for p_ in parts for m in p_["meta"]]
    for v in views:
        p = out if len(views) == 1 else out.with_name(f"{out.stem}_{v}{out.suffix}")
        torch.save({"feats": torch.cat([p_["feats"][v] for p_ in parts]), "labels": torch.tensor(labels), "meta": meta,
                    "view": v, "source": args.source, "inputs": args.inputs}, p)
        log.info("wrote %s n=%d (real=%d fake=%d) in %.0fs", p, len(labels), labels.count(0), labels.count(1), time.time() - t0)


if __name__ == "__main__":
    main()
