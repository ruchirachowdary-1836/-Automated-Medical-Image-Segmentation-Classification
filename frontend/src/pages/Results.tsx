import { Download, FileJson, FileText, Image as ImageIcon, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ViewerLoader } from "../components/Viewer";
import { Badge, Card, ErrorBox, SectionTitle, Skeleton, Stat, Tooltip, useAsync, useToast } from "../components/ui";
import { api } from "../lib/api";
import { fmtDate, fmtDur, fmtMl, fmtNum, fmtPct, levelTone } from "../lib/format";
import type { GroundTruthEval } from "../lib/types";

const NAMES = { WT: "Whole tumor", TC: "Tumor core", ET: "Enhancing tumor" } as const;

function GroundTruthPanel({ jobId }: { jobId: string }) {
  const input = useRef<HTMLInputElement>(null);
  const toast = useToast();
  const [res, setRes] = useState<GroundTruthEval | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const go = async (f: File) => {
    setBusy(true); setErr(null);
    try { setRes(await api.evaluateGroundTruth(jobId, f)); toast("success", "Compared against ground truth"); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  return (
    <Card>
      <SectionTitle>Accuracy vs ground truth</SectionTitle>
      {!res ? (
        <>
          <p className="text-sm text-mute">Dice, IoU and HD95 are <b>only</b> measurable when a reference annotation exists. No accuracy is shown for this scan until you supply a BraTS-style label volume (values 0, 1, 2, 4) on the same grid.</p>
          <input ref={input} type="file" accept=".nii,.gz" className="sr-only" aria-label="Ground-truth label file" onChange={(e) => e.target.files?.[0] && go(e.target.files[0])} />
          <button className="btn-secondary mt-3" disabled={busy} onClick={() => input.current?.click()}><Upload className="h-4 w-4" />{busy ? "Evaluating…" : "Upload ground-truth labels"}</button>
          {err && <div className="mt-3"><ErrorBox message={err} /></div>}
        </>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-mute"><tr className="text-left"><th className="py-1 pr-4">Region</th><th>Dice</th><th>IoU</th><th>Sens.</th><th>Spec.</th><th>HD95 (mm)</th><th>Vol. error</th></tr></thead>
            <tbody className="num">
              {Object.entries(res.regions).map(([r, m]) => (
                <tr key={r} className="border-t border-line"><td className="py-1.5 pr-4 font-sans font-medium">{r}</td><td>{fmtNum(m.dice)}</td><td>{fmtNum(m.iou)}</td><td>{fmtNum(m.sensitivity)}</td><td>{fmtNum(m.specificity)}</td><td>{fmtNum(m.hd95_mm, 2)}</td><td>{fmtPct(m.volume_error)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-mute mt-2">{res.note} Empty-region convention: both empty → 1.0, one empty → 0.0; “—” means undefined.</p>
        </div>
      )}
    </Card>
  );
}

export default function Results() {
  const { id = "" } = useParams();
  const { data: job, error, loading } = useAsync(() => api.job(id), [id]);
  if (loading) return <div className="space-y-4"><Skeleton className="h-8 w-64" /><Skeleton className="h-[520px]" /></div>;
  if (error || !job)
    return (
      <div className="space-y-4">
        <ErrorBox message={error ?? "Job not found"} />
        <p className="text-sm text-mute">Results are retained only for a limited time and may have expired. <Link className="text-brand underline" to="/app/upload">Run a new segmentation</Link>.</p>
      </div>
    );
  const r = job.results;
  const m = r.model as Record<string, unknown>;
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="h-page">Segmentation result</h1>
          <p className="text-sm text-mute num mt-1">Job {job.id.slice(0, 8)} · {fmtDate(job.created)} · expires {fmtDate(job.expires)}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <a className="btn-secondary btn-sm" href={api.downloadUrl(job.downloads.mask)}><Download className="h-3.5 w-3.5" />Mask (NIfTI)</a>
          <a className="btn-secondary btn-sm" href={api.downloadUrl(job.downloads.preview)}><ImageIcon className="h-3.5 w-3.5" />Preview PNG</a>
          <a className="btn-secondary btn-sm" href={api.downloadUrl(job.downloads.metrics)}><FileJson className="h-3.5 w-3.5" />Metrics JSON</a>
        </div>
      </div>
      {m.synthetic === true && (
        <div role="alert" className="rounded-card border border-warn/40 bg-warn/10 p-3 text-sm">This result comes from a checkpoint trained on <b>synthetic phantoms</b> as a pipeline smoke test. It carries no information about real anatomy.</div>
      )}
      {r.warnings.map((w) => <div key={w} role="alert" className="rounded-card border border-warn/40 bg-warn/10 p-3 text-sm">{w}</div>)}

      <ViewerLoader job={job} />

      <div className="grid md:grid-cols-3 gap-4">
        {(["WT", "TC", "ET"] as const).map((k) => {
          const reg = r.regions[k];
          return (
            <Card key={k}>
              <div className="flex justify-between items-start">
                <div className="text-sm font-semibold">{NAMES[k]} <span className="text-mute font-normal">({k})</span></div>
                {reg.confidence_level && <Badge tone={levelTone(reg.confidence_level)}>{reg.confidence_level} confidence</Badge>}
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <Stat label="Predicted volume" value={fmtMl(reg.volume_ml)} hint={`${reg.voxels.toLocaleString()} voxels`} />
                <Stat label="Region confidence" value={fmtPct(reg.confidence, 1)} hint={reg.confidence == null ? "Not predicted" : "model score"} />
              </div>
              <div className="text-xs text-mute mt-3">Mean uncertainty: <span className="num">{fmtPct(reg.mean_uncertainty, 1)}</span></div>
            </Card>
          );
        })}
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card>
          <SectionTitle>Scan-level reliability</SectionTitle>
          <div className="grid grid-cols-3 gap-4">
            <Stat label="Scan confidence" value={fmtPct(r.scan_confidence)} hint={r.scan_confidence_level ?? "no tumor predicted"} />
            <Stat label="Mean uncertainty (WT)" value={fmtPct(r.mean_uncertainty_wt)} />
            <Stat label="Inference time" value={fmtDur(r.inference_seconds)} hint={r.device} />
          </div>
          <p className="text-xs text-mute mt-3">Definition: {r.scan_confidence_definition}. Levels use {r.confidence_thresholds.source} thresholds (high ≥ {r.confidence_thresholds.high.toFixed(2)}, medium ≥ {r.confidence_thresholds.medium.toFixed(2)}). Confidence is a model score, not a probability of correctness or clinical certainty.</p>
          <Link to={`/app/confidence?job=${job.id}`} className="text-sm text-brand font-medium mt-3 inline-block">Open in confidence lab →</Link>
        </Card>
        <Card>
          <SectionTitle>Provenance</SectionTitle>
          <dl className="grid grid-cols-[9rem_1fr] gap-y-1.5 text-sm">
            <dt className="text-mute">Checkpoint</dt><dd className="num">{r.model.checkpoint}</dd>
            <dt className="text-mute">Training run</dt><dd className="num">{String(m.run ?? "—")}</dd>
            <dt className="text-mute">Architecture</dt><dd>{String(m.model ?? "—")} · {typeof m.params === "number" ? `${(m.params / 1e6).toFixed(2)} M params` : ""}</dd>
            <dt className="text-mute">Trained on</dt><dd>{String(m.trained_on ?? "—")}</dd>
            <dt className="text-mute">Preprocessing</dt><dd>{r.preprocessing.normalisation}; order {r.preprocessing.modality_order.join(", ")}</dd>
            <dt className="text-mute">Options</dt><dd>TTA {r.options.tta ? "on" : "off"} · MC-dropout {r.options.mc_samples || "off"} · min component {r.options.min_component_voxels || "off"} ({r.options.components_removed} removed)</dd>
            <dt className="text-mute">Uncertainty</dt><dd>{r.uncertainty_method}</dd>
            <dt className="text-mute">Grid</dt><dd className="num">{r.shape.join("×")} · {r.spacing_mm.map((s) => s.toFixed(2)).join("×")} mm · {r.orientation}</dd>
          </dl>
        </Card>
      </div>

      <GroundTruthPanel jobId={job.id} />
      <Tooltip text="Outputs require qualified review"><span className="inline-flex items-center gap-1.5 text-xs text-mute"><FileText className="h-3.5 w-3.5" />Model output, not a validated measurement.</span></Tooltip>
    </div>
  );
}
