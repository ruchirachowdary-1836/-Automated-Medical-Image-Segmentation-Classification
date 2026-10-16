# Model card — NeuroVista 2D U-Net

**Intended use:** education and research on multi-modal glioma segmentation. **Not** a medical device; not for diagnosis, prognosis, treatment planning or patient management. No regulatory approval or clinical validation is claimed.

## Architecture
2D slice-wise U-Net (four encoder levels + bottleneck, transposed-conv decoder, skip connections), `in_ch=4` (T1, T1ce, T2, FLAIR in that order), 4 output classes (0 background, 1 necrotic/non-enhancing core, 2 edema, 3 enhancing tumor; BraTS label 4 remapped to 3). Default base width 32 (≈7.8 M parameters), `Dropout2d` p=0.2 in the deepest blocks; optional attention gates. Optimiser AdamW (3e-4, wd 1e-4), Dice+CE loss, cosine schedule, gradient clipping, AMP on CUDA.

## Training data and measured performance
**No clinically meaningful model has been trained in this repository.** The only checkpoint ever produced here was trained for a pipeline smoke test on programmatically generated *synthetic phantoms* (`neurovista/synthetic.py`); checkpoints trained that way carry `synthetic: true`, and the UI/API show a warning on every result they produce. Validation Dice on those phantoms says nothing about real MRI.

Real performance must be measured by you: train on BraTS (see TRAINING_GUIDE.md), evaluate on the held-out **test** split once, and record the numbers here. Template (leave blank until measured):

| Split | n patients | Dice WT | Dice TC | Dice ET | HD95 WT (mm) | ECE |
|---|---|---|---|---|---|---|
| test | — | — | — | — | — | — |

## Confidence and uncertainty
Voxel confidence is the max softmax probability — a model score, not a probability of correctness. Region/scan definitions are in the Confidence lab and `inference.py`. High/Medium/Low cut-offs default to 0.90/0.70 (configurable) until a validation evaluation derives thresholds (smallest confidence whose accuracy on predicted-tumor voxels ≥ 95% / 85%); thresholds are chosen on validation only. Uncertainty = normalised predictive entropy, plus MC-dropout mutual information when enabled. Neither has been validated as a failure detector until you run the uncertainty-vs-error evaluation on real data.

## Limitations
- Expects skull-stripped, co-registered, BraTS-style axial volumes (warns heuristically if not). No resampling/registration is performed.
- Slice-wise 2D model: no through-plane context. 3D baseline is not implemented.
- Trained models inherit dataset bias (scanner, protocol, tumor type, population); behaviour on pediatric, post-treatment, metastasis or non-BraTS data is unknown.
- Predicted volumes depend on header voxel spacing being correct.
- Outputs can be confidently wrong; always require qualified review.

## Data licences
BraTS and related datasets have their own terms (see DATASET_GUIDE.md). Do not redistribute data or derived weights against those terms.
