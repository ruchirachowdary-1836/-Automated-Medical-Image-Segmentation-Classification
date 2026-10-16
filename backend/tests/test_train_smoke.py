import json

import torch

from neurovista.checkpoint import load_checkpoint
from neurovista.data import preprocess_dataset
from neurovista.synthetic import write_dataset
from neurovista.train import run_training


def _dataset(tmp_path, n):
    root = tmp_path / "ds"
    write_dataset(root / "raw", n, shape=(32, 32, 8))
    preprocess_dataset(root / "raw", root / "processed", seed=1, keep_empty=1.0)
    return root


def test_cpu_training_smoke_and_resume(tmp_path):
    root = _dataset(tmp_path, 6)
    run = tmp_path / "run"
    run.mkdir()
    cfg = {"processed_dir": str(root / "processed"), "epochs": 2, "batch_size": 8, "base": 8, "device": "cpu",
           "max_train_batches": 2, "synthetic": True, "patience": 5}
    (run / "config.json").write_text(json.dumps(cfg))
    assert run_training(run)["state"] == "completed"
    hist = [json.loads(x) for x in (run / "history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 and all(set(h["dice"]) == {"WT", "TC", "ET"} for h in hist)
    assert all(torch.isfinite(torch.tensor(h["train_loss"])) for h in hist)
    _, meta = load_checkpoint(run / "checkpoints" / "best.pt")
    assert meta["synthetic"] is True and meta["val_mean_dice"] >= 0
    cfg.update(epochs=3, resume=str(run / "checkpoints" / "last.pt"))
    (run / "config.json").write_text(json.dumps(cfg))
    run_training(run)
    assert len((run / "history.jsonl").read_text().splitlines()) == 3


def test_seeded_runs_are_reproducible(tmp_path):
    root = _dataset(tmp_path, 5)
    losses = []
    for i in range(2):
        run = tmp_path / f"r{i}"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({
            "processed_dir": str(root / "processed"), "epochs": 1, "batch_size": 8, "base": 8, "device": "cpu",
            "augment": False, "max_train_batches": 3, "seed": 5}))
        run_training(run)
        losses.append(json.loads((run / "history.jsonl").read_text().splitlines()[0])["train_loss"])
    assert losses[0] == losses[1]
