# Dataset guide

## Supported input
BraTS-style **3D NIfTI** (`.nii` / `.nii.gz`): four co-registered, skull-stripped modalities per case on one grid with a valid affine; axial slices along the third array axis. Ground truth uses labels 0, 1, 2, 4 (remapped internally to 0–3). 2D PNG/JPG, DICOM and single-channel inputs are **rejected** by design (the model needs four MRI channels); a single-image 2D mode is not implemented.

## Folder structure
```
var/datasets/<name>/
  raw/<case_id>/<case_id>_t1.nii.gz  _t1ce  _t2  _flair  _seg
  processed/            (created by `neurovista prepare`)
  splits.json           (patient-wise manifest: seed, fractions, ids)
```

## Candidate datasets (check each licence/terms yourself)
| Dataset | Notes |
|---|---|
| BraTS 2020/2021 (Synapse / Kaggle mirrors) | Main benchmark; label meanings changed in later editions — read the edition's description |
| Medical Segmentation Decathlon Task01 | BraTS-derived 484 volumes; different file layout/label order — convert to the folder structure above |
| Kaggle LGG, Figshare tumor slices | 2D / binary or classification tasks; **not** directly supported by this 4-channel pipeline |

Licences differ (research-only, attribution, no redistribution). Do not commit data or trained weights derived from restricted data. Dataset sizes are not asserted here; check the edition you download.

## Preparation
`python -m neurovista.cli prepare --name <name>`, then inspect `splits.json`. Splits are seeded and persisted; re-running with another seed produces a different manifest.
