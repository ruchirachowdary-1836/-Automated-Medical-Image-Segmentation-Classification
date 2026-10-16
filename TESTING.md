# Testing

```bash
cd backend  && pytest -q                 # 47 tests (unit, data integrity, API, training smoke)
cd backend  && ruff check neurovista tests
cd frontend && npm test                  # 17 tests (vitest + Testing Library)
cd frontend && npm run typecheck && npm run lint && npm run build
```

## Coverage of the requested areas
- **Unit:** U-Net shapes (incl. attention), losses, Dice/IoU/HD95 (spacing-aware), empty-mask conventions, label remap, z-score, grid crop/pad round-trip and image/mask alignment, confidence/volume calculations (spacing → mL), checkpoint round-trip and rejection of bad/mismatched checkpoints, output shape and affine.
- **Data integrity:** no patient appears in two splits; slices belong to their patient's split; modality order; train/inference preprocessing parity; missing/mismatched/corrupt/empty/NaN modalities; PNG and wrong-orientation rejection.
- **API:** health, readiness (503 without model), model info, full predict → layers → downloads → ground-truth evaluate → delete workflow, invalid type/missing/corrupt/empty/mismatched/oversized uploads, option validation, no leaked paths, uploads removed from the temp dir, training start/monitor/activate and experiment statuses.
- **Training:** CPU smoke run with checkpointing, resume, and seeded reproducibility.
- **Frontend:** upload validation, nifti header parsing, geometry/aspect/colour rendering, navigation, API failure states, results rendering, expiry message.

## Not covered
Real-data accuracy; GPU/AMP paths (no GPU available); Docker builds; browser E2E automation (the viewer was checked manually in a browser against a live backend).

Fixtures are tiny synthetic phantoms; they never support accuracy claims.
