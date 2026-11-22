#!/usr/bin/env python3
"""M5: train a modern-generator linear probe on cached CLIP features and compare
against the shipped ProGAN probe (+ soft_router) on every eval pack.

Inputs are caches written by scripts/extract_clip_features.py under
artifacts/feature_cache/m5/. Output: weights/<name>.pth (Linear 768->1, raw CLIP
features, same format as the ProGAN probe) + docs/m5_modern_probe_summary.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FC = ROOT / "artifacts/feature_cache/m5"


def available(name):
    return (FC / name).exists() or _parts_dir(name).is_dir()


def _parts_dir(name):
    stem = Path(name).stem
    for v in ("_clean", "_social"):
        if stem.endswith(v):
            return FC / (stem[: -len(v)] + ".parts")
    return FC / (stem + ".parts")


def load(name):
    """Final cache, or (interim) merge of checkpoint parts written so far."""
    if (FC / name).exists():
        d = torch.load(FC / name, weights_only=False)
        return d["feats"].float().numpy(), d["labels"].numpy().astype(int), d["meta"]
    stem = Path(name).stem
    view = stem.rsplit("_", 1)[-1]
    parts = [torch.load(pp, weights_only=False) for pp in sorted(_parts_dir(name).glob("part_*.pt"))]
    feats = torch.cat([p["feats"][view] for p in parts]).float().numpy()
    labels = np.array([l for p in parts for l in p["labels"]], dtype=int)
    meta = [m for p in parts for m in p["meta"]]
    print(f"  (interim: {name} from {len(parts)} parts, n={len(labels)})")
    return feats, labels, meta


def head(path):
    w = torch.load(path, weights_only=False)
    return w["weight"].float().numpy().reshape(-1), float(w["bias"].float().reshape(-1)[0])


def sig(x):
    return 1 / (1 + np.exp(-x))


def soft_mix(la, lb):
    ca, cb = np.abs(la), np.abs(lb)
    m = np.maximum(ca, cb)
    wa = np.exp(ca - m) / (np.exp(ca - m) + np.exp(cb - m))
    return wa * sig(la) + (1 - wa) * sig(lb)


def metrics(y, p, thr=0.5):
    out = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    pred = (p >= thr).astype(int)
    out["acc"] = float(accuracy_score(y, pred))
    if (y == 0).any():
        out["real_acc"] = float((pred[y == 0] == 0).mean())
    if (y == 1).any():
        out["fake_acc"] = float((pred[y == 1] == 1).mean())
    if len(set(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    return out


def fit(X, y, C, groups=None):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit((X - mu) / sd, y)
    w = clf.coef_.reshape(-1) / sd
    b = float(clf.intercept_[0] - (clf.coef_.reshape(-1) * mu / sd).sum())
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-progan", action="store_true", help="add ProGAN train features")
    ap.add_argument("--C", type=float, default=None, help="skip group-CV, use this C")
    ap.add_argument("--out", default="weights/modern_clip_vit_l14_linear.pth")
    ap.add_argument("--md", default="artifacts/m5_results.md")
    ap.add_argument("--summary", default="docs/m5_modern_probe_summary.json")
    args = ap.parse_args()

    X, y, meta = load("of_train_aug.pt")
    groups = [m["model"] for m in meta]
    if args.with_progan and available("progan_train_aug.pt"):
        Xp, yp, mp = load("progan_train_aug.pt")
        X = np.concatenate([X, Xp]); y = np.concatenate([y, yp])
        groups += ["progan_" + ("fake" if l else "real") for l in yp]
    groups = np.array(groups)
    print(f"train n={len(y)} real={int((y==0).sum())} fake={int((y==1).sum())} gens={len(set(groups[y==1]))}")

    # Leave-generators-out CV to pick C (reals grouped by source, fakes by generator).
    if args.C is None:
        best = None
        # fakes: whole generators held out per fold; reals: random 5-way split
        rng = np.random.RandomState(0)
        gens = sorted(set(groups[y == 1]))
        rng.shuffle(gens)
        gfold = {g: i % 5 for i, g in enumerate(gens)}
        fold = np.where(y == 1, [gfold.get(g, 0) for g in groups], rng.randint(0, 5, len(y)))
        splits = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]
        for C in [0.0005, 0.002, 0.01, 0.05, 0.2]:
            aucs = []
            for tr, va in splits:
                if len(set(y[va])) < 2:
                    continue
                w, b = fit(X[tr], y[tr], C)
                aucs.append(roc_auc_score(y[va], X[va] @ w + b))
            s = float(np.mean(aucs)) if aucs else float("nan")
            print(f"  C={C}: leave-gen-out AUC {s:.4f} ({len(aucs)} folds)")
            if best is None or s > best[1]:
                best = (C, s)
        C = best[0]
    else:
        C = args.C
    w, b = fit(X, y, C)
    print("chosen C", C)

    heads = {
        "progan (old default)": head(ROOT / "weights/progan_clip_vit_l14_linear.pth"),
        "new": (w, b),
    }
    hc = head(ROOT / "weights/cifake_clip_vit_l14_linear.pth")

    evals = {
        "openfake_test_ood_clean": "of_test_clean.pt",
        "openfake_test_ood_social_jpeg75": "of_test_social.pt",
        "openfake_reddit_wild": "of_reddit_clean.pt",
        "hemg_clean": "hemg_clean.pt",
        "hemg_social_jpeg75": "hemg_social.pt",
        "progan_holdout": "progan_test_clean.pt",
        "smoke": "smoke_clean.pt",
    }
    res = {}
    per_gen = {}
    for name, f in evals.items():
        if not available(f):
            print("missing", f); continue
        Xe, ye, me = load(f)
        res[name] = {}
        for hn, (hw, hb) in heads.items():
            res[name][hn] = metrics(ye, sig(Xe @ hw + hb))
        pw, pb = heads["progan (old default)"]
        res[name]["soft_router (old)"] = metrics(ye, soft_mix(Xe @ pw + pb, Xe @ hc[0] + hc[1]))
        if name.startswith("openfake"):
            by = defaultdict(list)
            for i, m in enumerate(me):
                by[(m["model"], int(ye[i]))].append(i)
            per_gen[name] = {}
            for (mod, lab), idx in sorted(by.items()):
                idx = np.array(idx)
                per_gen[name][f"{'fake' if lab else 'real'}:{mod}"] = {
                    hn: float(((sig(Xe[idx] @ hw + hb) >= 0.5) == lab).mean()) for hn, (hw, hb) in heads.items()
                } | {"n": int(len(idx))}
    # CIFAKE tradeoff (existing M2 cache, test split)
    cf = ROOT / "artifacts/feature_cache/cifake_tr4000_te2000_seed42_sz224.pt"
    if cf.exists():
        d = torch.load(cf, weights_only=False)
        Xe, ye = d["test_feats"].float().numpy(), d["test_y"].numpy().astype(int)
        res["cifake_test"] = {hn: metrics(ye, sig(Xe @ hw + hb)) for hn, (hw, hb) in heads.items()}
    # Ezra's Grok Imagine image
    for f in ["ezra_grok_clean.pt"]:
        if (FC / f).exists():
            Xe, ye, _ = load(f)
            res["ezra_grok_imagine_image"] = {hn: {"p_fake": float(sig(Xe @ hw + hb)[0])} for hn, (hw, hb) in heads.items()}
            pw, pb = heads["progan (old default)"]
            res["ezra_grok_imagine_image"]["soft_router (old)"] = {"p_fake": float(soft_mix(Xe @ pw + pb, Xe @ hc[0] + hc[1])[0])}

    for name, r in res.items():
        print(f"\n== {name}")
        for hn, m in r.items():
            print(f"  {hn:22s} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    for name, g in per_gen.items():
        print(f"\n-- per-model acc@0.5 {name}")
        for k, v in g.items():
            print(f"  {k:40s} n={v['n']:4d} old={v['progan (old default)']:.3f} new={v['new']:.3f}")

    md = [f"# M5 modern probe results ({'interim' if not (FC / 'of_test_social.pt').exists() else 'final'})", "",
          f"train n={len(y)} (real={int((y==0).sum())}, fake={int((y==1).sum())}), with_progan={args.with_progan}, C={C}", "",
          "| eval set | head | n | AUC | AP | acc@0.5 | real acc (1-FPR) | fake acc (TPR) |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        for hn, m in r.items():
            if "p_fake" in m:
                md.append(f"| {name} | {hn} | 1 | | | | | p_fake={m['p_fake']:.4f} |")
                continue
            g = lambda k: f"{m[k]:.4f}" if k in m else ""
            md.append(f"| {name} | {hn} | {m['n']} | {g('auc')} | {g('ap')} | {g('acc')} | {g('real_acc')} | {g('fake_acc')} |")
    for name, gtab in per_gen.items():
        md += ["", f"## per-model acc@0.5 — {name}", "", "| model | n | old progan | new |", "|---|---|---|---|"]
        for k, v in gtab.items():
            md.append(f"| {k} | {v['n']} | {v['progan (old default)']:.3f} | {v['new']:.3f} |")
    mdp = ROOT / args.md
    mdp.write_text("\n".join(md) + "\n")
    print("wrote", mdp)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": torch.tensor(w, dtype=torch.float32).reshape(1, -1),
                "bias": torch.tensor([b], dtype=torch.float32),
                "meta": str({"dataset": "openfake_core_validation_shards_0-2" + ("+progan" if args.with_progan else ""),
                             "C": C, "n_train": int(len(y)), "aug": "downscale/jpeg60-95/blur"})}, out)
    s = ROOT / args.summary
    s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text(json.dumps({"C": C, "with_progan": args.with_progan, "n_train": int(len(y)),
                             "results": res, "per_generator_acc": per_gen}, indent=2))
    print("\nwrote", out, s)


if __name__ == "__main__":
    main()
