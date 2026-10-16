import { Columns2, Crosshair, Layers, Maximize, Pause, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { MODALITIES, MODALITY_LABEL } from "../lib/types";
import {
  REGION_INFO, REGION_RGB, planeAspect, planeMax, planeSize, pixelToVoxel, renderPlane, voxelToPixel, ramp,
  type Plane, type RenderOpts, type Vec3, type ViewMode, type Volume,
} from "../lib/volume";
import { ErrorBox, Progress, Tabs } from "./ui";

const LAYER_NAMES = ["t1", "t1ce", "t2", "flair", "labels", "confidence", "uncertainty"];

export function useVolume(job: Job | null) {
  const [vol, setVol] = useState<Volume | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    if (!job) return;
    let alive = true;
    setVol(null);
    setError(null);
    setProgress(0);
    const layers: Record<string, Uint8Array> = {};
    let done = 0;
    const [X, Y, Z] = job.viewer.shape;
    Promise.all(
      LAYER_NAMES.map(async (n) => {
        const arr = await api.layer(job.id, n);
        if (arr.length !== X * Y * Z) throw new Error(`Layer ${n} has unexpected size`);
        layers[n] = arr;
        if (alive) setProgress(++done / LAYER_NAMES.length);
      }),
    )
      .then(() => alive && setVol({ shape: job.viewer.shape, spacing: job.viewer.spacing_mm, layers }))
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [job]);
  return { vol, error, progress };
}

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 600, h: 520 });
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}

interface PlaneViewProps {
  vol: Volume;
  plane: Plane;
  cur: Vec3;
  opts: RenderOpts;
  onPick: (v: Vec3) => void;
  onHover?: (v: Vec3 | null) => void;
  zoomable?: boolean;
  zoom?: { k: number; x: number; y: number };
  onZoom?: (z: { k: number; x: number; y: number }) => void;
  label?: string;
  height: number;
}

export function PlaneView({ vol, plane, cur, opts, onPick, onHover, zoomable, zoom = { k: 1, x: 0, y: 0 }, onZoom, label, height }: PlaneViewProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [box, size] = useSize<HTMLDivElement>();
  const drag = useRef<{ sx: number; sy: number; ox: number; oy: number; moved: boolean } | null>(null);
  const aspect = planeAspect(vol.shape, vol.spacing, plane);
  const { w, h } = planeSize(vol.shape, plane);
  // Fit the physical aspect ratio inside the container (no anatomical distortion).
  const fitW = Math.min(size.w, size.h * aspect);
  const fitH = fitW / aspect;

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const r = renderPlane(vol, plane, cur, opts);
    c.width = r.width;
    c.height = r.height;
    c.getContext("2d")?.putImageData(new ImageData(r.data, r.width, r.height), 0, 0);
  }, [vol, plane, cur, opts]);

  const locate = (e: React.MouseEvent) => {
    const rect = canvas.current!.getBoundingClientRect();
    const c = Math.min(w - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * w)));
    const r = Math.min(h - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * h)));
    return pixelToVoxel(vol.shape, plane, c, r, cur);
  };
  const px = voxelToPixel(vol.shape, plane, cur);

  return (
    <div className="flex flex-col gap-1.5">
      {label && <div className="text-xs font-medium text-mute px-1">{label}</div>}
      <div
        ref={box}
        style={{ height }}
        className="relative rounded-lg bg-black overflow-hidden grid place-items-center select-none"
        onWheel={(e) => {
          if (!zoomable || !onZoom) return;
          const k = Math.min(8, Math.max(1, zoom.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
          onZoom({ k, x: k === 1 ? 0 : zoom.x, y: k === 1 ? 0 : zoom.y });
        }}
        onMouseDown={(e) => {
          drag.current = { sx: e.clientX, sy: e.clientY, ox: zoom.x, oy: zoom.y, moved: false };
        }}
        onMouseMove={(e) => {
          const d = drag.current;
          if (d && zoomable && zoom.k > 1 && onZoom && (e.buttons & 1)) {
            const dx = e.clientX - d.sx, dy = e.clientY - d.sy;
            if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
            onZoom({ ...zoom, x: d.ox + dx, y: d.oy + dy });
          }
          onHover?.(locate(e));
        }}
        onMouseUp={(e) => {
          if (drag.current && !drag.current.moved) onPick(locate(e));
          drag.current = null;
        }}
        onMouseLeave={() => {
          drag.current = null;
          onHover?.(null);
        }}
      >
        <div style={{ width: fitW, height: fitH, transform: `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.k})`, transformOrigin: "center", cursor: zoom.k > 1 ? "grab" : "crosshair" }} className="relative">
          <canvas ref={canvas} className="mri w-full h-full block" aria-label={`${plane} slice`} />
          <div className="absolute inset-y-0 w-px bg-ai/70 pointer-events-none" style={{ left: `${((px.c + 0.5) / w) * 100}%` }} />
          <div className="absolute inset-x-0 h-px bg-ai/70 pointer-events-none" style={{ top: `${((px.r + 0.5) / h) * 100}%` }} />
        </div>
        <span className="absolute top-1.5 left-2 text-[10px] text-white/70 font-mono uppercase">{plane}</span>
        <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[10px] text-white/60 font-mono">{plane === "sagittal" ? "A" : "R"}</span>
        <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[10px] text-white/60 font-mono">{plane === "sagittal" ? "P" : "L"}</span>
      </div>
    </div>
  );
}

const MODES: { id: ViewMode; label: string }[] = [
  { id: "overlay", label: "Overlay" },
  { id: "original", label: "Original" },
  { id: "confidence", label: "Confidence" },
  { id: "uncertainty", label: "Uncertainty" },
];

export function Legend({ mode }: { mode: ViewMode }) {
  if (mode === "confidence" || mode === "uncertainty") {
    const name = mode === "confidence" ? "viridis" : "inferno";
    const grad = Array.from({ length: 6 }, (_, i) => `rgb(${ramp(name, i / 5).map(Math.round).join(",")})`).join(",");
    return (
      <div className="text-xs text-mute">
        <div className="mb-1 font-medium text-ink">{mode === "confidence" ? "Max softmax probability" : "Normalised predictive entropy"}</div>
        <div className="h-2.5 rounded-full" style={{ background: `linear-gradient(90deg, ${grad})` }} />
        <div className="flex justify-between mt-0.5 num"><span>0</span><span>1</span></div>
        <p className="mt-2">{mode === "confidence" ? "A model score, not a guarantee of correctness." : "Higher = the model is less decided between classes."}</p>
      </div>
    );
  }
  return (
    <ul className="space-y-1.5 text-xs">
      {REGION_INFO.map((r) => (
        <li key={r.id} className="flex items-start gap-2">
          <span className="mt-0.5 h-3 w-3 rounded-sm shrink-0" style={{ background: `rgb(${REGION_RGB[r.id].join(",")})` }} />
          <span><span className="font-medium">{r.label}</span><span className="text-mute block">{r.description}</span></span>
        </li>
      ))}
    </ul>
  );
}

export default function Viewer({ job, vol }: { job: Job; vol: Volume }) {
  const [plane, setPlane] = useState<Plane>("axial");
  const [cur, setCur] = useState<Vec3>(() => {
    const [X, Y] = vol.shape;
    return [Math.floor(X / 2), Math.floor(Y / 2), Math.min(job.viewer.default_slice, vol.shape[2] - 1)];
  });
  const [mode, setMode] = useState<ViewMode>("overlay");
  const [modality, setModality] = useState("flair");
  const [opacity, setOpacity] = useState(0.45);
  const [visible, setVisible] = useState<Record<number, boolean>>({ 1: true, 2: true, 3: true });
  const [boundaries, setBoundaries] = useState(false);
  const [mpr, setMpr] = useState(false);
  const [compare, setCompare] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [zoom, setZoom] = useState({ k: 1, x: 0, y: 0 });
  const [hover, setHover] = useState<Vec3 | null>(null);

  const idx = plane === "axial" ? cur[2] : plane === "coronal" ? cur[1] : cur[0];
  const max = planeMax(vol.shape, plane);
  const setIdx = useCallback(
    (n: number) => {
      const v = Math.min(max - 1, Math.max(0, n));
      setCur((c) => (plane === "axial" ? [c[0], c[1], v] : plane === "coronal" ? [c[0], v, c[2]] : [v, c[1], c[2]]));
    },
    [plane, max],
  );
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setCur((c) => {
      const k = plane === "axial" ? 2 : plane === "coronal" ? 1 : 0;
      const n = (c[k] + 1) % max;
      const out: Vec3 = [...c];
      out[k] = n;
      return out;
    }), 140);
    return () => clearInterval(t);
  }, [playing, plane, max]);

  const opts: RenderOpts = useMemo(() => ({ modality, mode, opacity, visible, boundaries }), [modality, mode, opacity, visible, boundaries]);
  const orig: RenderOpts = useMemo(() => ({ ...opts, mode: "original" }), [opts]);
  const reset = () => {
    setZoom({ k: 1, x: 0, y: 0 });
    setCur([Math.floor(vol.shape[0] / 2), Math.floor(vol.shape[1] / 2), job.viewer.default_slice]);
    setMode("overlay"); setOpacity(0.45); setVisible({ 1: true, 2: true, 3: true }); setCompare(false); setPlaying(false);
  };
  const info = hover ?? cur;
  const hi = (info[0] * vol.shape[1] + info[1]) * vol.shape[2] + info[2];
  const lab = vol.layers.labels[hi];
  const heatMode = mode === "confidence" || mode === "uncertainty";
  const H = 520;

  return (
    <div className="grid xl:grid-cols-[1fr_17rem] gap-4">
      <div
        className="space-y-3 outline-none"
        tabIndex={0}
        aria-label="MRI viewer. Use arrow keys to change slice."
        onKeyDown={(e) => {
          if (e.key === "ArrowUp" || e.key === "ArrowRight" || e.key === "PageUp") { e.preventDefault(); setIdx(idx + 1); }
          if (e.key === "ArrowDown" || e.key === "ArrowLeft" || e.key === "PageDown") { e.preventDefault(); setIdx(idx - 1); }
          if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
          if (e.key === "Home") reset();
        }}
      >
        <div className="flex flex-wrap items-center gap-2">
          <Tabs tabs={MODES} value={mode} onChange={setMode} />
          <select className="input !w-auto !h-8 text-xs" value={modality} onChange={(e) => setModality(e.target.value)} aria-label="Modality">
            {MODALITIES.map((m) => <option key={m} value={m}>{MODALITY_LABEL[m]}</option>)}
          </select>
          <Tabs tabs={[{ id: "axial", label: "Axial" }, { id: "coronal", label: "Coronal" }, { id: "sagittal", label: "Sagittal" }]} value={plane} onChange={(p) => { setPlane(p); setZoom({ k: 1, x: 0, y: 0 }); }} />
          <div className="ml-auto flex gap-1.5">
            <button className={`btn-sm ${mpr ? "btn-primary" : "btn-secondary"}`} onClick={() => setMpr((m) => !m)} aria-pressed={mpr}><Crosshair className="h-3.5 w-3.5" />Multi-planar</button>
            <button className={`btn-sm ${compare ? "btn-primary" : "btn-secondary"}`} onClick={() => setCompare((m) => !m)} aria-pressed={compare} disabled={mpr || heatMode}><Columns2 className="h-3.5 w-3.5" />Side by side</button>
            <button className="btn-secondary btn-sm" onClick={reset}><RotateCcw className="h-3.5 w-3.5" />Reset</button>
          </div>
        </div>

        {mpr ? (
          <div className="grid md:grid-cols-3 gap-2">
            {(["axial", "coronal", "sagittal"] as Plane[]).map((p) => (
              <PlaneView key={p} vol={vol} plane={p} cur={cur} opts={opts} height={300} onPick={setCur} onHover={setHover} label={p[0].toUpperCase() + p.slice(1)} />
            ))}
          </div>
        ) : compare && !heatMode ? (
          <div className="grid md:grid-cols-2 gap-2">
            <PlaneView vol={vol} plane={plane} cur={cur} opts={orig} height={H} onPick={setCur} onHover={setHover} label="Original" />
            <PlaneView vol={vol} plane={plane} cur={cur} opts={{ ...opts, mode: "overlay" }} height={H} onPick={setCur} onHover={setHover} label="Segmentation overlay" />
          </div>
        ) : (
          <PlaneView vol={vol} plane={plane} cur={cur} opts={opts} height={H} zoomable zoom={zoom} onZoom={setZoom} onPick={setCur} onHover={setHover} />
        )}

        <div className="flex items-center gap-3">
          <button className="btn-secondary btn-sm !px-2.5" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause slice animation" : "Play slice animation"}>
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>
          <input type="range" min={0} max={max - 1} value={idx} onChange={(e) => setIdx(+e.target.value)} className="flex-1 accent-[rgb(var(--brand))]" aria-label={`${plane} slice`} />
          <span className="num text-xs text-mute w-20 text-right">{idx + 1} / {max}</span>
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-mute num" aria-live="off">
          <span>Voxel {info.join(", ")}</span>
          <span>Grid {vol.shape.join(" × ")}</span>
          <span>Spacing {vol.spacing.map((s) => s.toFixed(2)).join(" × ")} mm</span>
          <span>Label {lab === 0 ? "background" : REGION_INFO[lab - 1]?.label}</span>
          <span>Conf {(vol.layers.confidence[hi] / 2.55).toFixed(0)}%</span>
          <span>Unc {(vol.layers.uncertainty[hi] / 2.55).toFixed(0)}%</span>
          <span>Zoom {zoom.k.toFixed(1)}×</span>
        </div>
      </div>

      <aside className="space-y-5">
        <div>
          <div className="h-sec mb-2 flex items-center gap-1.5"><Layers className="h-3.5 w-3.5" />Regions</div>
          <div className="space-y-1.5">
            {REGION_INFO.map((r) => (
              <label key={r.id} className="flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" checked={visible[r.id]} onChange={(e) => setVisible((v) => ({ ...v, [r.id]: e.target.checked }))} className="accent-[rgb(var(--brand))]" />
                <span className="h-3 w-3 rounded-sm" style={{ background: `rgb(${REGION_RGB[r.id].join(",")})` }} />
                {r.label.split(" /")[0]}
              </label>
            ))}
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input type="checkbox" checked={boundaries} onChange={(e) => setBoundaries(e.target.checked)} className="accent-[rgb(var(--brand))]" />
              Tumor boundaries
            </label>
          </div>
        </div>
        <div>
          <label className="label flex justify-between" htmlFor="op"><span>Overlay opacity</span><span className="num">{Math.round(opacity * 100)}%</span></label>
          <input id="op" type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} className="w-full accent-[rgb(var(--brand))]" />
        </div>
        <div>
          <div className="h-sec mb-2">Legend</div>
          <Legend mode={mode} />
        </div>
        <div className="text-xs text-mute flex items-start gap-1.5"><Maximize className="h-3.5 w-3.5 mt-0.5 shrink-0" />Scroll to zoom, drag to pan, click to move the crosshair. Keyboard: ↑/↓ slice, Space play, Home reset.</div>
      </aside>
    </div>
  );
}

export function ViewerLoader({ job }: { job: Job }) {
  const { vol, error, progress } = useVolume(job);
  if (error) return <ErrorBox message={`Could not load the volume: ${error}`} />;
  if (!vol)
    return (
      <div className="card card-pad space-y-3">
        <div className="text-sm text-mute">Loading volume layers…</div>
        <Progress value={progress} label="Loading volume" />
        <div className="skeleton h-[420px]" />
      </div>
    );
  return <Viewer job={job} vol={vol} />;
}
