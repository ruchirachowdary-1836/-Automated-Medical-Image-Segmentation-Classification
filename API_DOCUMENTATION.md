# API

Base URL: `http://localhost:8000` (interactive docs at `/docs`). Errors share one shape:

```json
{"error": {"code": "shape_mismatch", "message": "...", "modality": "t1"}}
```

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness: `{status, version, uptime_s}` |
| `GET /ready` | `200 {ready:true}` or `503 {ready:false, reason}` when no valid checkpoint is loaded |
| `GET /model/info` | Checkpoint metadata, device, capabilities, confidence thresholds (+ synthetic-data warning) |
| `POST /model/reload` | Reload checkpoint (needs `NV_TRAINING_ENABLED=true`) |
| `GET /settings` | Safe runtime limits, retention text, dataset names (no paths/secrets) |
| `POST /predict` | multipart: `t1,t1ce,t2,flair` (`.nii`/`.nii.gz`), form `tta`, `mc_samples` (0 or 2–50), `min_component_voxels`. Returns the job (see below) |
| `GET /jobs`, `GET /jobs/{id}`, `DELETE /jobs/{id}` | Retained sessions |
| `GET /jobs/{id}/layer/{name}` | Raw `uint8` volume, RAS, shape from `viewer.shape`, index `(x*Y+y)*Z+z`. Names: `t1 t1ce t2 flair labels confidence uncertainty` |
| `GET /jobs/{id}/download/{mask\|preview\|metrics}` | NIfTI mask (BraTS labels 0/1/2/4, original affine), PNG preview, JSON |
| `POST /evaluate` | form `job_id` + file `ground_truth` (BraTS labels) → per-region Dice, IoU, sensitivity, specificity, HD95 (mm), volumes, volume error |
| `GET /training/defaults` | Defaults, experiment presets, prepared datasets |
| `POST /training/runs` | Start a run (`dataset` name under the data dir + hyper-parameters) |
| `GET /training/runs`, `GET /training/runs/{id}` | Status, `history`, log tail, finished evaluations |
| `POST /training/runs/{id}/cancel \| resume \| evaluate \| activate` | Control and checkpoint management |
| `GET /experiments` | The eight guide experiments with honest status (`planned`, `trained_not_evaluated`, `completed`, `not_implemented`) |
| `GET /dashboard` | Aggregate used by the dashboard |

## `POST /predict` response (abridged)

```json
{
  "id": "…", "status": "completed",
  "viewer": {"shape": [96,96,32], "spacing_mm": [1,1,2], "orientation": "RAS", "default_slice": 17},
  "results": {
    "regions": {"WT": {"voxels": 3881, "volume_ml": 7.76, "confidence": 0.91, "confidence_level": "high", "mean_uncertainty": 0.46}, "TC": {…}, "ET": {…}},
    "scan_confidence": 0.9, "scan_confidence_definition": "mean max-softmax probability over predicted whole-tumor voxels",
    "confidence_thresholds": {"high": 0.9, "medium": 0.7, "source": "configured-default (not calibrated)"},
    "inference_seconds": 3.4, "model": {"checkpoint": "best.pt", "run": "…", "synthetic": true},
    "warnings": []
  },
  "downloads": {"mask": "/jobs/…/download/mask", "preview": "…", "metrics": "…"}
}
```

When no tumor is predicted, region `confidence`/`mean_uncertainty` and `scan_confidence` are `null` and a warning is added (no division by zero).

## Error codes

`unsupported_file_type`, `empty_file`, `unreadable_file`, `bad_dimensions`, `dimensions_out_of_range`, `non_finite`, `empty_volume`, `bad_spacing`, `bad_affine`, `shape_mismatch`, `affine_mismatch`, `unsupported_orientation`, `missing_modality`, `invalid_request` (422); `file_too_large` (413); `model_not_ready` (503); `busy`, `rate_limited` (429); `timeout` (504); `out_of_memory` (507); `job_not_found`, `run_not_found` (404); `training_disabled`, `admin_disabled` (403); `internal_error` (500, no internals exposed).
