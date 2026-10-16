# Training guide

All commands run from `backend/` with the virtualenv active. Data lives under `NV_DATA_DIR` (default `./var`).

## 1. Prepare data
Place cases at `var/datasets/<name>/raw/<case>/<case>_{t1,t1ce,t2,flair,seg}.nii.gz` (see DATASET_GUIDE.md), then:

```bash
python -m neurovista.cli prepare --name <name> --seed 42
```

This normalises each volume (brain-mask z-score after 0.5–99.5 % clipping), remaps labels 4→3, crops 240²→192² when lossless, keeps every tumor slice plus ~10 % empty slices, writes `processed/*.npz`, and persists a **patient-wise** 70/15/15 split to `var/datasets/<name>/splits.json`. Slices of a patient never cross partitions (tested).

## 2. Train
From the UI (Training lab) or the API (`POST /training/runs`). Runs execute `python -m neurovista.train` as a subprocess and write `var/runs/<id>/{config.json,status.json,history.jsonl,train.log,checkpoints/}`. Training never starts on app startup.

Implemented: seeded Python/NumPy/PyTorch RNGs, per-worker augmentation RNG, AdamW, cosine or plateau schedule, gradient clipping, AMP (CUDA), best (`best.pt`, by validation mean Dice over WT/TC/ET) and `last.pt` checkpoints, resume (`POST …/resume`), early stopping, cancel flag, readable error on CUDA OOM. Validation Dice is accumulated over the whole validation set (not averaged per slice).

Experiment presets (guide §9.2): `baseline_ce`, `dice_ce`, `augmentation`, `attention_unet`, `flair_only`. `tta_postproc` and `mc_dropout_vs_error` are **evaluation-time** options on an existing run. `2d_vs_3d` is not implemented.

## 3. Evaluate
Training lab → select run → **Evaluate** (validation first). This runs `python -m neurovista.evaluate`, which predicts each held-out patient volume at original resolution and writes `eval_<label>.json`: per-patient and aggregate Dice/IoU/sensitivity/specificity/HD95 (mm, from header spacing)/volume error, voxel confusion matrix, ECE, Brier, reliability bins, uncertainty-vs-error AUROC. A validation evaluation also writes `calibration.json` (confidence thresholds); **never** derive thresholds or tune hyper-parameters on the test split — run test evaluation once, at the end.

## 4. Deploy a checkpoint
"Use for inference" copies `best.pt` (and `calibration.json` if present) to `var/checkpoints/` and reloads the model. Checkpoints stay out of Git (`*.pt` is ignored); host them on Releases / a model hub and mount them into `NV_DATA_DIR/checkpoints`.

## CPU smoke test (no dataset, no GPU)
```bash
python -m neurovista.cli synth --name synthetic --n 14
```
Then train 3–6 epochs with `base=16` from the UI. Mark the dataset as synthetic. This validates plumbing only.
