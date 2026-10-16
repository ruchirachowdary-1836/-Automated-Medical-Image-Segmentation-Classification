import gzip
import io
import time

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurovista.api.main import create_app
from neurovista.api.store import LAYERS
from neurovista.data import preprocess_dataset
from neurovista.synthetic import write_dataset


def _files(paths, **override):
    out = {}
    for k, p in paths.items():
        data = override.get(k, p.read_bytes())
        out[k] = (f"{k}.nii.gz", data, "application/gzip")
    return out


@pytest.fixture()
def client(settings, checkpoint):
    return TestClient(create_app(settings))


@pytest.fixture()
def no_model_client(settings):
    return TestClient(create_app(settings))


def test_health_ready_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is True
    info = client.get("/model/info").json()
    assert info["loaded"] and info["capabilities"]["mc_dropout"] and info["warning"]
    assert "Research prototype only" in info["disclaimer"]


def test_model_not_ready(no_model_client, scan_files):
    r = no_model_client.get("/ready")
    assert r.status_code == 503 and r.json()["ready"] is False and r.json()["reason"]
    assert no_model_client.get("/health").status_code == 200  # backend still starts
    assert no_model_client.get("/model/info").json()["loaded"] is False
    paths, _, _ = scan_files
    r = no_model_client.post("/predict", files=_files(paths))
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_not_ready"


def test_full_workflow(client, scan_files):
    paths, seg_path, _ = scan_files
    r = client.post("/predict", files=_files(paths), data={"tta": "true", "mc_samples": "2"})
    assert r.status_code == 200, r.text
    job = r.json()
    jid = job["id"]
    shape = job["viewer"]["shape"]
    assert job["status"] == "completed" and job["results"]["options"]["tta"] is True
    assert set(job["results"]["regions"]) == {"WT", "TC", "ET"}
    assert job["results"]["mean_uncertainty_wt"] is None or 0 <= job["results"]["mean_uncertainty_wt"] <= 1
    # viewer layers: correct size, uint8
    for name in LAYERS:
        raw = client.get(f"/jobs/{jid}/layer/{name}").content
        assert len(raw) == int(np.prod(shape))
    assert client.get(f"/jobs/{jid}/layer/nope").status_code == 404
    # downloads are valid artifacts
    mask = client.get(f"/jobs/{jid}/download/mask")
    img = nib.Nifti1Image.from_bytes(gzip.decompress(mask.content))
    assert img.shape == tuple(job["results"]["shape"])
    assert client.get(f"/jobs/{jid}/download/preview").content[:4] == b"\x89PNG"
    assert client.get(f"/jobs/{jid}/download/metrics").json()["id"] == jid
    assert client.get("/jobs").json()["jobs"][0]["id"] == jid
    # evaluation against ground truth
    ev = client.post("/evaluate", data={"job_id": jid},
                     files={"ground_truth": ("seg.nii.gz", seg_path.read_bytes(), "application/gzip")})
    assert ev.status_code == 200 and 0 <= ev.json()["regions"]["WT"]["dice"] <= 1
    bad = client.post("/evaluate", data={"job_id": jid}, files={"ground_truth": ("seg.nii.gz", b"junk", "x")})
    assert bad.status_code == 422
    assert client.delete(f"/jobs/{jid}").json()["deleted"]
    assert client.get(f"/jobs/{jid}").status_code == 404


def test_uploads_not_retained(client, scan_files, tmp_path, monkeypatch):
    import tempfile
    tmp = tmp_path / "uploadtmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    paths, _, _ = scan_files
    assert client.post("/predict", files=_files(paths)).status_code == 200
    assert list(tmp.iterdir()) == []  # raw uploads removed immediately; only derived result artifacts remain


@pytest.mark.parametrize("case", ["png", "missing", "corrupt", "empty", "mismatch", "bigfile"])
def test_invalid_uploads(client, scan_files, case):
    paths, _, (vols, _, affine) = scan_files
    files = _files(paths)
    if case == "png":
        files["t1"] = ("t1.png", b"\x89PNG\r\n", "image/png")
        status, code = 422, "unsupported_file_type"
    elif case == "missing":
        files.pop("flair")
        r = client.post("/predict", files=files)
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
        return
    elif case == "corrupt":
        files["t2"] = ("t2.nii.gz", b"not nifti at all", "application/gzip")
        status, code = 422, "unreadable_file"
    elif case == "empty":
        files["t2"] = ("t2.nii.gz", b"", "application/gzip")
        status, code = 422, "empty_file"
    elif case == "mismatch":
        buf = io.BytesIO()
        img = nib.Nifti1Image(vols["t1"][:32, :32], affine)
        files["t1"] = ("t1.nii.gz", gzip.compress(img.to_bytes()), "application/gzip")
        status, code = 422, "shape_mismatch"
    else:
        files["t1"] = ("t1.nii", b"\0" * (6 * 1024 * 1024), "application/octet-stream")
        status, code = 413, "file_too_large"
    r = client.post("/predict", files=files)
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["code"] == code and "Traceback" not in r.text


def test_bad_options(client, scan_files):
    paths, _, _ = scan_files
    r = client.post("/predict", files=_files(paths), data={"mc_samples": "1"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "bad_options"
    assert client.get("/jobs/not-a-uuid").status_code == 404


def test_settings_endpoint_does_not_leak_paths(client):
    body = client.get("/settings").text
    assert "retention_policy" in body and "var" not in client.get("/settings").json().get("device_preference", "")
    assert str(client.app.state.nv.settings.data_dir) not in body


def test_training_flow(client, settings):
    root = settings.datasets_dir / "tiny"
    write_dataset(root / "raw", 6, shape=(32, 32, 8))
    preprocess_dataset(root / "raw", root / "processed", seed=1, keep_empty=1.0)
    assert client.get("/training/defaults").json()["datasets"][0]["name"] == "tiny"
    bad = client.post("/training/runs", json={"dataset": "../etc", "epochs": 1})
    assert bad.status_code == 422
    r = client.post("/training/runs", json={"dataset": "tiny", "epochs": 2, "batch_size": 8, "base": 8,
                                           "device": "cpu", "max_train_batches": 2, "synthetic": True,
                                           "preset": "dice_ce"})
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    for _ in range(120):
        run = client.get(f"/training/runs/{rid}").json()
        if run["status"]["state"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert run["status"]["state"] == "completed", run
    assert len(run["history"]) == 2 and run["has_best"]
    exps = {e["id"]: e for e in client.get("/experiments").json()["experiments"]}
    assert exps["dice_ce"]["status"] == "trained_not_evaluated"
    assert exps["2d_vs_3d"]["status"] == "not_implemented" and exps["baseline_ce"]["status"] == "planned"
    act = client.post(f"/training/runs/{rid}/activate")
    assert act.status_code == 200 and act.json()["loaded"] and act.json()["meta"]["synthetic"] is True
    assert client.post("/training/runs/nope/cancel").status_code == 404


def test_training_disabled(settings):
    settings.training_enabled = False
    c = TestClient(create_app(settings))
    assert c.post("/training/runs", json={"dataset": "x"}).status_code in (403, 422)
