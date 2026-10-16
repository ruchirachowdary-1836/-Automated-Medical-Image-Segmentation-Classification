import { CheckCircle2, Circle, FileUp, Info, Loader2, RefreshCw, Trash2, XCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api, predict } from "../lib/api";
import { fmtBytes } from "../lib/format";
import { parseNifti, type NiftiInfo } from "../lib/nifti";
import { MODALITIES, MODALITY_LABEL, type Modality } from "../lib/types";
import { crossCheck, validateFile } from "../lib/validate";
import { Badge, Card, ErrorBox, Progress, SectionTitle, Tooltip, useAsync, useToast } from "../components/ui";

interface Slot {
  file: File;
  error: string | null;
  info: NiftiInfo | null;
  parsing: boolean;
}

const HINT: Record<Modality, string> = {
  t1: "Native T1-weighted",
  t1ce: "Post-contrast T1 (gadolinium)",
  t2: "T2-weighted",
  flair: "T2-FLAIR",
};

function Preview({ info }: { info: NiftiInfo }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const p = info.preview;
    const c = ref.current;
    if (!p || !c) return;
    c.width = p.width;
    c.height = p.height;
    c.getContext("2d")?.putImageData(new ImageData(p.data, p.width, p.height), 0, 0);
  }, [info]);
  if (!info.preview) return null;
  return <canvas ref={ref} className="mri h-20 w-20 rounded-md bg-black object-contain" aria-label="Middle-slice preview" />;
}

function DropZone({ modality, slot, maxMb, onFile, onClear }: { modality: Modality; slot?: Slot; maxMb: number; onFile: (f: File) => void; onClear: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const status = !slot ? "empty" : slot.error ? "error" : slot.parsing ? "parsing" : "ready";
  return (
    <div
      className={`card p-4 transition-colors ${over ? "border-brand bg-brand/5" : ""} ${status === "error" ? "border-bad/50" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <div className="font-semibold">{MODALITY_LABEL[modality]} <span className="text-xs font-normal text-mute">required</span></div>
          <div className="text-xs text-mute">{HINT[modality]}</div>
        </div>
        {status === "ready" && <CheckCircle2 className="h-5 w-5 text-ok" />}
        {status === "error" && <XCircle className="h-5 w-5 text-bad" />}
        {status === "parsing" && <Loader2 className="h-5 w-5 animate-spin text-mute" />}
        {status === "empty" && <Circle className="h-5 w-5 text-line" />}
      </div>
      <input ref={input} type="file" accept=".nii,.gz,.nii.gz" className="sr-only" aria-label={`${MODALITY_LABEL[modality]} file`} onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      {!slot ? (
        <button type="button" onClick={() => input.current?.click()} className="w-full h-28 rounded-lg border border-dashed border-line grid place-items-center text-sm text-mute hover:border-brand hover:text-ink transition-colors">
          <span className="flex flex-col items-center gap-1"><FileUp className="h-5 w-5" />Drop .nii / .nii.gz or click to browse</span>
        </button>
      ) : (
        <div className="flex gap-3">
          {slot.info && <Preview info={slot.info} />}
          <div className="min-w-0 flex-1 text-sm">
            <div className="truncate font-medium" title={slot.file.name}>{slot.file.name}</div>
            <div className="text-xs text-mute num">{fmtBytes(slot.file.size)} (limit {maxMb} MB)</div>
            {slot.info && <div className="text-xs text-mute num mt-0.5">{slot.info.dims.join(" × ")} · {slot.info.spacing.map((s) => s.toFixed(2)).join(" × ")} mm</div>}
            {slot.error && <div className="text-xs text-bad mt-1">{slot.error}</div>}
            <div className="flex gap-1.5 mt-2">
              <button className="btn-secondary btn-sm" onClick={() => input.current?.click()}><RefreshCw className="h-3 w-3" />Replace</button>
              <button className="btn-ghost btn-sm" onClick={onClear}><Trash2 className="h-3 w-3" />Remove</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function UploadPage() {
  const nav = useNavigate();
  const toast = useToast();
  const { data: settings } = useAsync(() => api.settings(), []);
  const { data: model } = useAsync(() => api.modelInfo(), []);
  const maxMb = settings?.max_upload_mb ?? 200;
  const [slots, setSlots] = useState<Partial<Record<Modality, Slot>>>({});
  const [tta, setTta] = useState(false);
  const [mc, setMc] = useState(0);
  const [minComp, setMinComp] = useState(0);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState<null | { phase: "uploading" | "processing"; p: number }>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  const setFile = async (m: Modality, file: File) => {
    const err = validateFile(file, m, maxMb);
    setError(null);
    setSlots((s) => ({ ...s, [m]: { file, error: err, info: null, parsing: !err } }));
    if (err) return;
    try {
      const info = await parseNifti(file);
      setSlots((s) => ({ ...s, [m]: { file, error: null, info, parsing: false } }));
    } catch (e) {
      // A preview failure on an otherwise valid-looking file is non-fatal in browsers without gzip streams.
      const msg = (e as Error).message;
      const fatal = !/browser cannot preview/.test(msg);
      setSlots((s) => ({ ...s, [m]: { file, error: fatal ? `${MODALITY_LABEL[m]}: ${msg}` : null, info: null, parsing: false } }));
    }
  };

  const metas = Object.fromEntries(MODALITIES.filter((m) => slots[m]?.info).map((m) => [m, { dims: slots[m]!.info!.dims, spacing: slots[m]!.info!.spacing }]));
  const mismatch = crossCheck(metas);
  const allPresent = MODALITIES.every((m) => slots[m]);
  const noFileErrors = MODALITIES.every((m) => !slots[m]?.error);
  const parsing = MODALITIES.some((m) => slots[m]?.parsing);
  const modelReady = !!model?.loaded;
  const mcAvailable = !!model?.capabilities.mc_dropout;
  const checklist = [
    { ok: allPresent, text: "All four modalities selected (T1, T1ce, T2, FLAIR)" },
    { ok: allPresent && noFileErrors, text: "File type and size accepted" },
    { ok: allPresent && !mismatch && !parsing, text: mismatch ?? "Dimensions and voxel spacing match across modalities" },
    { ok: modelReady, text: modelReady ? "Model loaded and ready" : "Model not loaded — see Settings / Training lab" },
    { ok: consent, text: "Retention notice acknowledged" },
  ];
  const canRun = checklist.every((c) => c.ok) && !busy;

  const run = () => {
    if (!canRun) return;
    setError(null);
    setBusy({ phase: "uploading", p: 0 });
    const files = Object.fromEntries(MODALITIES.map((m) => [m, slots[m]!.file])) as Record<Modality, File>;
    const { promise, cancel } = predict(files, { tta, mcSamples: mc, minComponentVoxels: minComp }, (p, phase) => setBusy({ phase, p }));
    cancelRef.current = cancel;
    promise
      .then((job) => { toast("success", "Segmentation complete"); nav(`/app/results/${job.id}`); })
      .catch((e: ApiError) => { if (e.code !== "cancelled") setError(e); })
      .finally(() => { setBusy(null); cancelRef.current = null; });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="h-page">New segmentation</h1>
        <p className="text-mute text-sm mt-1 max-w-3xl">Upload one co-registered, skull-stripped 3D NIfTI volume per modality (BraTS-style). Single 2D images (PNG/JPG) are not accepted: the model needs all four MRI channels.</p>
      </div>
      <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {MODALITIES.map((m) => (
          <DropZone key={m} modality={m} slot={slots[m]} maxMb={maxMb} onFile={(f) => setFile(m, f)} onClear={() => setSlots((s) => { const n = { ...s }; delete n[m]; return n; })} />
        ))}
      </div>
      {mismatch && <ErrorBox message={mismatch} />}
      {error && (
        <ErrorBox message={error.modality ? `${error.message}` : error.message} onRetry={error.code === "network" || error.code === "timeout" || error.code === "busy" ? run : undefined} />
      )}
      <div className="grid lg:grid-cols-[1fr_22rem] gap-4">
        <Card>
          <SectionTitle>Inference options</SectionTitle>
          <div className="grid sm:grid-cols-3 gap-5 text-sm">
            <label className="flex items-start gap-2 cursor-pointer">
              <input type="checkbox" className="mt-1 accent-[rgb(var(--brand))]" checked={tta} onChange={(e) => setTta(e.target.checked)} />
              <span><span className="font-medium">Test-time augmentation</span><span className="block text-xs text-mute">Averages predictions over a horizontal flip. ~2× slower.</span></span>
            </label>
            <div>
              <label className="label" htmlFor="mc">MC-dropout samples</label>
              <Tooltip text={mcAvailable ? "0 disables. 2–50 stochastic passes add mutual-information uncertainty." : "This checkpoint has no dropout layers."}>
                <input id="mc" className="input" type="number" min={0} max={50} value={mc} disabled={!mcAvailable} onChange={(e) => setMc(Math.max(0, Math.min(50, +e.target.value || 0)))} />
              </Tooltip>
              {mc === 1 && <div className="text-xs text-bad mt-1">Use 0 (off) or at least 2.</div>}
            </div>
            <div>
              <label className="label" htmlFor="mcv">Min component size (voxels)</label>
              <input id="mcv" className="input" type="number" min={0} value={minComp} onChange={(e) => setMinComp(Math.max(0, +e.target.value || 0))} />
              <div className="text-xs text-mute mt-1">0 keeps every component. Removed components are reported.</div>
            </div>
          </div>
          <label className="flex items-start gap-2 mt-5 text-sm cursor-pointer">
            <input type="checkbox" className="mt-1 accent-[rgb(var(--brand))]" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            <span>I confirm these scans are de-identified and I am permitted to process them. {settings?.retention_policy ?? "Uploaded scans are deleted immediately after inference."}</span>
          </label>
        </Card>
        <Card>
          <SectionTitle>Pre-analysis checklist</SectionTitle>
          <ul className="space-y-2 text-sm">
            {checklist.map((c, i) => (
              <li key={i} className="flex items-start gap-2">
                {c.ok ? <CheckCircle2 className="h-4 w-4 text-ok mt-0.5 shrink-0" /> : <Circle className="h-4 w-4 text-mute mt-0.5 shrink-0" />}
                <span className={c.ok ? "" : "text-mute"}>{c.text}</span>
              </li>
            ))}
          </ul>
          {busy ? (
            <div className="mt-4 space-y-2">
              <Progress value={busy.phase === "uploading" ? busy.p : 1} label="Upload progress" />
              <div className="text-xs text-mute flex items-center gap-2">
                {busy.phase === "uploading" ? `Uploading… ${Math.round(busy.p * 100)}%` : <><Loader2 className="h-3.5 w-3.5 animate-spin" />Preprocessing and running the model…</>}
              </div>
              <button className="btn-secondary w-full" onClick={() => cancelRef.current?.()}>Cancel</button>
            </div>
          ) : (
            <button className="btn-primary w-full mt-4" disabled={!canRun} onClick={run}>Start segmentation</button>
          )}
          {model?.warning && <div className="mt-3 text-xs text-warn flex gap-1.5"><Info className="h-3.5 w-3.5 shrink-0 mt-0.5" />{model.warning}</div>}
          {!modelReady && model && <Badge tone="warn">{model.reason ?? "Model unavailable"}</Badge>}
        </Card>
      </div>
    </div>
  );
}
