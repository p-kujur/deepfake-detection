"""Smoke tests for ProGAN folder splits + probe config wiring."""

from pathlib import Path

import yaml


def test_progan_train_config_exists():
    cfg_path = Path("configs/train_progan_subset.yaml")
    assert cfg_path.is_file()
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["data"]["name"] == "progan_folder"
    assert "train_root" in cfg["data"]
    assert "test_root" in cfg["data"]
    assert cfg["train"]["weights_out"].endswith("progan_clip_vit_l14_linear.pth")


def test_progan_splits_summary_committed():
    p = Path("configs/progan_splits_summary.json")
    assert p.is_file()
    import json

    slim = json.loads(p.read_text())
    assert slim["seed"] == 42
    assert slim["counts"]["test"]["n_real"] == 200
    assert slim["counts"]["test"]["n_fake"] == 200
    assert "Hemg" in slim["notes"]


def test_folder_dataset_import():
    from deepfake_detection.data import RealFakeFolderDataset

    assert RealFakeFolderDataset is not None


def test_npr_lite_progan_config_exists():
    cfg_path = Path("configs/train_npr_lite_progan.yaml")
    assert cfg_path.is_file()
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["data"]["name"] == "progan_folder"
    assert cfg["train"]["weights_out"].endswith("npr_lite_progan.pth")


def test_dual_probe_import():
    from deepfake_detection.eval import dual_probe

    assert hasattr(dual_probe, "confidence_router")
    assert hasattr(dual_probe, "select_on_val")
