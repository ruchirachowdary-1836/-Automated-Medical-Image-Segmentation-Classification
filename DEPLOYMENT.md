# Deployment

## Local
See the README quick start. Environment variables are listed in `.env.example`; the frontend reads `VITE_API_URL` at build time. If port 8000 or 5173 is taken, run on other ports and set `NV_CORS_ORIGINS` / `VITE_API_URL` accordingly.

## Docker
```bash
docker compose up --build        # frontend :8080, backend :8000, volume nv-data mounted at /data
```
> **Verification status:** `backend/Dockerfile`, `frontend/Dockerfile`, `frontend/nginx.conf` and `docker-compose.yml` have **not been built or run** (Docker was not available where this was developed). Treat them as unverified until you run the command above and check `docker compose ps` health.

Put datasets under `/data/datasets/<name>/raw` and checkpoints at `/data/checkpoints/best.pt` (volume or bind mount). The image is CPU-only; for GPU inference use a CUDA base image and CUDA PyTorch wheels, and `NV_DEVICE=cuda`.

## Hosting guidance
- Host the static frontend (Vercel, Netlify, Cloudflare Pages, S3+CDN) separately from the inference API.
- The API needs several GB of RAM for full 240×240×155 volumes (all layers are materialised) and tens of seconds per scan on CPU. Do **not** assume a free tier suffices; size memory first. Persistent storage is only needed for `/data` checkpoints/datasets.
- On any public deployment set `NV_TRAINING_ENABLED=false`, restrict `NV_CORS_ORIGINS` to your frontend origin, put the API behind TLS and authentication (none is built in), and keep the retention window short.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| UI says "API offline" | Backend not running or wrong `VITE_API_URL`; check CORS origin |
| "Model not loaded" | No `best.pt` in `NV_DATA_DIR/checkpoints`; train or mount one |
| `unsupported_orientation` | Third array axis is not superior–inferior; reorient to axial first |
| `shape_mismatch` / `affine_mismatch` | Modalities not co-registered on one grid |
| `429 busy` | One inference at a time; retry |
| Training "failed" | Open the run → Configuration & log |
