"""FastAPI application."""

import asyncio
import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .. import __version__
from ..checkpoint import CheckpointError, read_json
from ..config import CLASS_NAMES, MODALITIES, RESEARCH_DISCLAIMER, Settings, get_settings
from ..inference import InferenceOptions, Segmenter
from ..metrics import evaluate_case, nan_to_none
from ..model import count_parameters
from ..nifti_io import InputValidationError, has_nifti_extension, load_scan
from ..preprocess import PreprocessError, remap_labels
from .runs import PRESETS, RunManager
from .store import LAYERS, REGION_COLORS, JobStore

log = logging.getLogger("neurovista")
CHUNK = 1024 * 1024


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = JobStore(settings)
        self.runs = RunManager(settings)
        self.segmenter: Segmenter | None = None
        self.load_error: str | None = None
        self.gate = asyncio.Semaphore(1)  # one inference at a time (CPU/GPU memory safety)
        self.started = time.time()

    def load_model(self) -> None:
        self.segmenter, self.load_error = None, None
        path = self.settings.checkpoint_file
        if not path.exists():
            self.load_error = "No model checkpoint found. Train a model or mount one (see TRAINING_GUIDE.md)."
            return
        try:
            seg = Segmenter.from_checkpoint(path, self.settings)
            cal = read_json(path.parent / "calibration.json")
            if cal and cal.get("split") == "val":
                seg.meta["calibration_thresholds"] = {"high": cal["high"], "medium": cal["medium"]}
            self.segmenter = seg
        except (CheckpointError, RuntimeError) as e:
            self.load_error = str(e)
            log.warning("model load failed: %s", e)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    state = AppState(settings)
    state.load_model()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.store.cleanup()
        yield

    app = FastAPI(title="NeuroVista AI", version=__version__, lifespan=lifespan,
                  description=RESEARCH_DISCLAIMER)
    app.state.nv = state
    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])
    app.state.limiter = limiter
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["Content-Disposition"])

    def err(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra}})

    @app.exception_handler(InputValidationError)
    async def _iv(_: Request, e: InputValidationError):
        return err(422, e.code, str(e), modality=e.modality)

    @app.exception_handler(PreprocessError)
    async def _pp(_: Request, e: PreprocessError):
        return err(422, "preprocess_failed", str(e))

    @app.exception_handler(RequestValidationError)
    async def _rv(_: Request, e: RequestValidationError):
        return err(422, "invalid_request", "Request validation failed",
                   fields=[".".join(str(x) for x in d["loc"]) for d in e.errors()])

    @app.exception_handler(RateLimitExceeded)
    async def _rl(_: Request, e: RateLimitExceeded):
        return err(429, "rate_limited", "Too many requests; please slow down")

    @app.exception_handler(HTTPException)
    async def _http(_: Request, e: HTTPException):
        detail = e.detail if isinstance(e.detail, dict) else {"code": "http_error", "message": str(e.detail)}
        return JSONResponse(status_code=e.status_code, content={"error": detail})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        log.exception("unhandled error")
        return err(500, "internal_error", "An internal error occurred. See server logs.")

    def require_admin() -> None:
        """Model/run management is only available when training is enabled (single-user / trusted deployments)."""
        if not settings.training_enabled:
            raise HTTPException(403, {"code": "admin_disabled", "message": "Model management is disabled on this deployment"})

    # ---------------------------------------------------------------- health
    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - state.started, 1)}

    @app.get("/ready")
    def ready():
        ok = state.segmenter is not None
        body = {"ready": ok, "reason": None if ok else state.load_error}
        return JSONResponse(status_code=200 if ok else 503, content=body)

    def model_info() -> dict:
        s = state.segmenter
        base = {"disclaimer": RESEARCH_DISCLAIMER, "modalities": list(MODALITIES), "classes": list(CLASS_NAMES),
                "region_colors": REGION_COLORS, "app_version": __version__,
                "device": str(s.device) if s else ("cuda" if torch.cuda.is_available() else "cpu"),
                "cuda_available": torch.cuda.is_available(),
                "capabilities": {"tta": True, "mc_dropout": bool(s and s.has_dropout), "min_component_filter": True,
                                 "training": settings.training_enabled, "attention_unet": True}}
        if not s:
            return {**base, "loaded": False, "reason": state.load_error}
        return {**base, "loaded": True, "checkpoint": s.checkpoint_name, "params": count_parameters(s.model),
                "meta": {k: v for k, v in s.meta.items() if k != "calibration_thresholds"},
                "confidence_thresholds": s._thresholds(),
                "validated": False if s.meta.get("synthetic") else None,
                "warning": ("This checkpoint was trained on SYNTHETIC phantoms for a pipeline smoke test; "
                            "its predictions on real MRI are meaningless.") if s.meta.get("synthetic") else None}

    @app.get("/model/info")
    def get_model_info():
        return model_info()

    @app.post("/model/reload")
    def reload_model():
        require_admin()
        state.load_model()
        return model_info()

    @app.get("/settings")
    def public_settings():
        return {"max_upload_mb": settings.max_upload_mb, "max_inplane_dim": settings.max_inplane_dim,
                "max_depth": settings.max_depth, "artifact_ttl_minutes": settings.artifact_ttl_minutes,
                "retention_policy": ("Uploaded scans are processed from a temporary directory that is deleted "
                                     "immediately after inference. Result artifacts are kept for "
                                     f"{settings.artifact_ttl_minutes} minutes, then deleted."),
                "rate_limit_per_minute": settings.rate_limit_per_minute, "training_enabled": settings.training_enabled,
                "mc_dropout_samples": settings.mc_dropout_samples, "device_preference": settings.device,
                "confidence_defaults": {"high": settings.confidence_high, "medium": settings.confidence_medium},
                "datasets": state.runs.datasets(), "disclaimer": RESEARCH_DISCLAIMER}

    # ---------------------------------------------------------------- upload helpers
    async def save_upload(up: UploadFile, dest: Path, modality: str) -> None:
        name = up.filename or ""
        if not has_nifti_extension(name):
            raise InputValidationError(f"{modality}: '{name}' is not a .nii or .nii.gz file. "
                                       "PNG/JPG images cannot be used as four-channel MRI volumes.",
                                       "unsupported_file_type", modality)
        limit = settings.max_upload_mb * CHUNK
        size = 0
        with dest.open("wb") as f:
            while chunk := await up.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, {"code": "file_too_large", "modality": modality,
                                              "message": f"{modality}: file exceeds the {settings.max_upload_mb} MB limit"})
                f.write(chunk)
        if size == 0:
            raise InputValidationError(f"{modality}: file is empty", "empty_file", modality)

    # ---------------------------------------------------------------- predict
    @app.post("/predict")
    @limiter.limit(f"{max(settings.rate_limit_per_minute // 3, 1)}/minute")
    async def predict(request: Request, t1: UploadFile = File(...), t1ce: UploadFile = File(...),
                      t2: UploadFile = File(...), flair: UploadFile = File(...),
                      tta: bool = Form(False), mc_samples: int = Form(0),
                      min_component_voxels: int = Form(0), consent_retain: bool = Form(True)):
        seg = state.segmenter
        if seg is None:
            raise HTTPException(503, {"code": "model_not_ready", "message": state.load_error or "Model not loaded"})
        if not (0 <= mc_samples <= 50) or not (0 <= min_component_voxels <= 1_000_000):
            raise InputValidationError("mc_samples must be 0-50 and min_component_voxels 0-1,000,000", "bad_options")
        if mc_samples == 1:
            raise InputValidationError("mc_samples must be 0 (off) or at least 2", "bad_options")
        if mc_samples > 1 and not seg.has_dropout:
            raise InputValidationError("This checkpoint has no dropout layers, so MC-dropout is unavailable", "mc_unavailable")
        if state.gate.locked():
            raise HTTPException(429, {"code": "busy", "message": "Another inference is running; retry shortly"})
        async with state.gate:
            tmp = Path(tempfile.mkdtemp(prefix="nv_up_"))
            try:
                paths: dict[str, Path] = {}
                for name, up in (("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)):
                    suffix = ".nii.gz" if (up.filename or "").lower().endswith(".gz") else ".nii"
                    paths[name] = tmp / f"{name}{suffix}"
                    await save_upload(up, paths[name], name)
                opts = InferenceOptions(tta, mc_samples, min_component_voxels)

                def work():
                    scan = load_scan(paths, settings)
                    res = seg.predict(scan, opts)
                    return scan, res

                try:
                    scan, res = await asyncio.wait_for(run_in_threadpool(work), settings.inference_timeout_s)
                except TimeoutError:
                    raise HTTPException(504, {"code": "timeout", "message": "Inference exceeded the time limit"})
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        raise HTTPException(507, {"code": "out_of_memory",
                                                  "message": "Not enough memory to process this scan"})
                    raise
                if not consent_retain:
                    pass  # results are still needed to render; they expire per retention policy
                meta = state.store.save(scan, res, {"tta": tta, "mc_samples": mc_samples,
                                                    "min_component_voxels": min_component_voxels,
                                                    "files": {k: _safe_name(u.filename) for k, u in
                                                              (("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair))}})
                return meta
            finally:
                shutil.rmtree(tmp, ignore_errors=True)  # uploads are never retained

    def _safe_name(n: str | None) -> str:
        return Path(n or "").name[:120]

    # ---------------------------------------------------------------- jobs
    @app.get("/jobs")
    def list_jobs():
        return {"jobs": state.store.list()}

    def job_or_404(job_id: str) -> dict:
        try:
            return state.store.get(job_id)
        except KeyError:
            raise HTTPException(404, {"code": "job_not_found", "message": "Job not found or expired"})

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        return job_or_404(job_id)

    @app.delete("/jobs/{job_id}")
    def delete_job(job_id: str):
        job_or_404(job_id)
        state.store.delete(job_id)
        return {"deleted": True}

    @app.get("/jobs/{job_id}/layer/{name}")
    def get_layer(job_id: str, name: str):
        job_or_404(job_id)
        if name not in LAYERS:
            raise HTTPException(404, {"code": "layer_not_found", "message": "Unknown layer"})
        return FileResponse(state.store.dir(job_id) / "layers" / f"{name}.bin", media_type="application/octet-stream")

    @app.get("/jobs/{job_id}/download/{kind}")
    def download(job_id: str, kind: str):
        job_or_404(job_id)
        d = state.store.dir(job_id)
        if kind == "mask":
            return FileResponse(d / "mask.nii.gz", media_type="application/gzip", filename=f"segmentation_{job_id[:8]}.nii.gz")
        if kind == "preview":
            return FileResponse(d / "preview.png", media_type="image/png", filename=f"preview_{job_id[:8]}.png")
        if kind == "metrics":
            return FileResponse(d / "meta.json", media_type="application/json", filename=f"metrics_{job_id[:8]}.json")
        raise HTTPException(404, {"code": "unknown_download", "message": "Unknown download kind"})

    # ---------------------------------------------------------------- evaluate against ground truth
    @app.post("/evaluate")
    async def evaluate(job_id: str = Form(...), ground_truth: UploadFile = File(...)):
        job_or_404(job_id)
        tmp = Path(tempfile.mkdtemp(prefix="nv_gt_"))
        try:
            dest = tmp / ("gt.nii.gz" if (ground_truth.filename or "").lower().endswith(".gz") else "gt.nii")
            await save_upload(ground_truth, dest, "ground_truth")

            def work():
                img = nib.load(str(dest))
                gt = remap_labels(np.asarray(img.dataobj))
                zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
                pred = np.asarray(nib.load(str(state.store.dir(job_id) / "mask.nii.gz")).dataobj)
                pred = remap_labels(pred)
                if gt.shape != pred.shape:
                    raise InputValidationError(f"Ground truth shape {gt.shape} does not match prediction {pred.shape}",
                                               "shape_mismatch", "ground_truth")
                return nan_to_none(evaluate_case(pred, gt, zooms))  # type: ignore[arg-type]

            try:
                regions = await run_in_threadpool(work)
            except InputValidationError:
                raise
            except Exception as e:
                raise InputValidationError(f"Ground truth could not be read as a BraTS label volume ({type(e).__name__})",
                                           "unreadable_file", "ground_truth")
            return {"job_id": job_id, "regions": regions, "source": "user-supplied ground truth",
                    "note": "Measured against the supplied annotation; not a clinical validation."}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---------------------------------------------------------------- training & experiments
    class TrainRequest(BaseModel):
        dataset: str
        name: str = Field("run", max_length=64)
        preset: str | None = None
        model: str | None = None
        loss: str | None = None
        epochs: int | None = Field(None, ge=1, le=1000)
        batch_size: int | None = Field(None, ge=1, le=256)
        lr: float | None = Field(None, gt=0, lt=1)
        weight_decay: float | None = Field(None, ge=0, lt=1)
        base: int | None = Field(None, ge=4, le=128)
        dropout: float | None = Field(None, ge=0, lt=0.9)
        patience: int | None = Field(None, ge=1, le=200)
        seed: int | None = None
        device: str | None = None
        augment: bool | None = None
        amp: bool | None = None
        scheduler: str | None = None
        grad_clip: float | None = Field(None, gt=0)
        max_train_batches: int | None = Field(None, ge=1)
        modalities: list[str] | None = None
        synthetic: bool | None = None

    @app.get("/training/defaults")
    def training_defaults():
        from ..train import DEFAULTS
        d = {k: v for k, v in DEFAULTS.items() if k not in ("processed_dir", "resume")}
        return {"defaults": d, "presets": PRESETS_PUBLIC(), "datasets": state.runs.datasets(),
                "enabled": settings.training_enabled,
                "cuda_available": torch.cuda.is_available()}

    def PRESETS_PUBLIC():
        return [{k: v for k, v in p.items()} for p in PRESETS]

    @app.post("/training/runs")
    def start_run(req: TrainRequest):
        cfg = req.model_dump(exclude_none=True)
        if "modalities" in cfg:
            cfg["in_ch"] = len(cfg["modalities"])
        try:
            rid = state.runs.create(cfg)
        except PermissionError as e:
            raise HTTPException(403, {"code": "training_disabled", "message": str(e)})
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(422, {"code": "invalid_training_request", "message": str(e)})
        return state.runs.get(rid)

    @app.get("/training/runs")
    def list_runs():
        return {"runs": state.runs.list()}

    def run_or_404(rid: str) -> dict:
        try:
            return state.runs.get(rid)
        except KeyError:
            raise HTTPException(404, {"code": "run_not_found", "message": "Run not found"})

    @app.get("/training/runs/{rid}")
    def get_run(rid: str):
        return run_or_404(rid)

    @app.post("/training/runs/{rid}/cancel")
    def cancel_run(rid: str):
        run_or_404(rid)
        state.runs.cancel(rid)
        return {"cancel_requested": True}

    @app.post("/training/runs/{rid}/resume")
    def resume_run(rid: str):
        run_or_404(rid)
        try:
            state.runs.resume(rid)
        except FileNotFoundError as e:
            raise HTTPException(422, {"code": "cannot_resume", "message": str(e)})
        return state.runs.get(rid)

    class EvalRequest(BaseModel):
        split: str = "val"
        tta: bool = False
        mc_samples: int = Field(0, ge=0, le=50)
        min_component_voxels: int = Field(0, ge=0)
        label: str = "val"

    @app.post("/training/runs/{rid}/evaluate")
    def eval_run(rid: str, req: EvalRequest):
        run_or_404(rid)
        try:
            state.runs.evaluate(rid, req.split, req.tta, req.mc_samples, req.min_component_voxels, req.label)
        except (ValueError, KeyError) as e:
            raise HTTPException(422, {"code": "invalid_evaluation", "message": str(e)})
        return {"started": True}

    @app.post("/training/runs/{rid}/activate")
    def activate_run(rid: str):
        require_admin()
        run_or_404(rid)
        try:
            state.runs.activate(rid)
        except FileNotFoundError as e:
            raise HTTPException(422, {"code": "no_checkpoint", "message": str(e)})
        state.load_model()
        return model_info()

    @app.get("/experiments")
    def experiments():
        return {"experiments": state.runs.experiments(), "runs": state.runs.list()}

    @app.get("/dashboard")
    def dashboard():
        runs = state.runs.list()
        jobs = state.store.list()
        latest = runs[0] if runs else None
        return {"jobs_retained": len(jobs), "recent_jobs": jobs[:5], "model": model_info(), "runs": len(runs),
                "latest_run": state.runs.get(latest["id"]) if latest else None,
                "datasets": state.runs.datasets(), "ready": state.segmenter is not None}

    return app


app = create_app()
