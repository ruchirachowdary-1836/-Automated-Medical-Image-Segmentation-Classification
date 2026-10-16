import { ShieldCheck } from "lucide-react";
import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { C, Histogram, ReliabilityDiagram } from "../components/charts";
import { useVolume } from "../components/Viewer";
import { Badge, Card, EmptyState, ErrorBox, SectionTitle, Skeleton, Stat, useAsync } from "../components/ui";
import { api } from "../lib/api";
import { fmtNum, fmtPct, levelTone } from "../lib/format";
import { EvalSchema } from "../lib/types";
import { tumorHistogram } from "../lib/volume";

function JobAnalysis({ id }: { id: string }) {
  const { data: job, error } = useAsync(() => api.job(id), [id]);
  const { vol, progress } = useVolume(job);
  const hist = useMemo(() => (vol ? { c: tumorHistogram(vol, "confidence"), u: tumorHistogram(vol, "uncertainty") } : null), [vol]);
  if (error) return <ErrorBox message={error} />;
  if (!job) return <Skeleton className="h-64" />;
  const r = job.results;
  const hasTumor = r.regions.WT.voxels > 0;
  return (
    <div className="space-y-4">
      <div className="grid md:grid-cols-4 gap-4">
        <Card><Stat label="Scan confidence" value={fmtPct(r.scan_confidence)} hint={r.scan_confidence_level ? <Badge tone={levelTone(r.scan_confidence_level)}>{r.scan_confidence_level}</Badge> : "no tumor predicted"} /></Card>
        <Card><Stat label="Mean entropy (WT)" value={fmtPct(r.mean_uncertainty_wt)} hint="normalised to [0,1]" /></Card>
        <Card><Stat label="Mean mutual info (WT)" value={r.mean_mutual_information_wt == null ? "—" : fmtNum(r.mean_mutual_information_wt, 4)} hint={r.options.mc_samples ? `MC-dropout ×${r.options.mc_samples}` : "enable MC-dropout when uploading"} /></Card>
        <Card><Stat label="TTA" value={r.options.tta ? "on" : "off"} hint="horizontal flip" /></Card>
      </div>
      <div className="grid lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-1">
          <SectionTitle>Region confidence</SectionTitle>
          <ul className="space-y-3">
            {(["WT", "TC", "ET"] as const).map((k) => {
              const g = r.regions[k];
              return (
                <li key={k}>
                  <div className="flex justify-between text-sm"><span className="font-medium">{k}</span><span className="num">{fmtPct(g.confidence)}</span></div>
                  <div className="h-2 rounded-full bg-line/70 mt-1"><div className="h-full rounded-full bg-brand" style={{ width: `${(g.confidence ?? 0) * 100}%` }} /></div>
                </li>
              );
            })}
          </ul>
          <p className="text-xs text-mute mt-4">Cut-offs: high ≥ {r.confidence_thresholds.high.toFixed(2)}, medium ≥ {r.confidence_thresholds.medium.toFixed(2)} — <b>{r.confidence_thresholds.source}</b>.</p>
        </Card>
        <Card className="lg:col-span-2">
          <SectionTitle>Distribution over predicted tumor voxels</SectionTitle>
          {!hasTumor ? <EmptyState title="No tumor voxels predicted">Distributions are only defined where the model predicts a tumor.</EmptyState> : !hist ? <Skeleton className="h-48" /> : (
            <div className="grid md:grid-cols-2 gap-4">
              <Histogram data={hist.c} title="Max-softmax confidence" color={C.brand} />
              <Histogram data={hist.u} title="Normalised predictive entropy" color={C.ai} />
            </div>
          )}
          {!vol && <p className="text-xs text-mute mt-2">Loading volume… {Math.round(progress * 100)}%</p>}
        </Card>
      </div>
      <Link to={`/app/results/${id}`} className="text-sm text-brand font-medium">View heatmaps in the viewer →</Link>
    </div>
  );
}

function Calibration() {
  const { data } = useAsync(async () => {
    const runs = (await api.runs()).runs.filter((r) => r.evals.length);
    const out: { run: string; label: string; ev: ReturnType<typeof EvalSchema.parse> }[] = [];
    for (const r of runs) {
      const full = await api.run(r.id);
      for (const [label, raw] of Object.entries(full.evals)) {
        const p = EvalSchema.safeParse(raw);
        if (p.success) out.push({ run: r.id, label, ev: p.data });
      }
    }
    return out;
  }, []);
  const first = data?.find((d) => d.ev.split === "val") ?? data?.[0];
  return (
    <Card>
      <SectionTitle>Calibration (from evaluations with ground truth)</SectionTitle>
      {!data ? <Skeleton className="h-52" /> : !first ? (
        <EmptyState icon={<ShieldCheck className="h-6 w-6" />} title="No calibration analysis available">Run an evaluation on a validation split from the Training lab. Calibration thresholds are chosen on validation data only, never on the test set.</EmptyState>
      ) : (
        <div className="grid lg:grid-cols-[1fr_20rem] gap-6">
          <ReliabilityDiagram bins={first.ev.calibration.bins} />
          <div className="space-y-4">
            <div className="text-xs text-mute">Run <span className="num">{first.run.slice(0, 15)}</span> · split <b>{first.ev.split}</b> · {first.ev.n_patients} patients · {first.ev.calibration.voxels_sampled.toLocaleString()} voxels</div>
            <Stat label="Expected calibration error (ECE)" value={fmtNum(first.ev.calibration.ece, 4)} hint="10 equal-width bins, brain voxels" />
            <Stat label="Brier score (top-1)" value={fmtNum(first.ev.calibration.brier_top1, 4)} />
            {first.ev.uncertainty_vs_error ? <Stat label="Uncertainty → error AUROC" value={fmtNum(first.ev.uncertainty_vs_error.auroc, 3)} hint="entropy as a detector of mis-classified voxels" /> : <p className="text-xs text-mute">Uncertainty-vs-error needs an evaluation containing errors.</p>}
            {first.ev.calibration.thresholds && <p className="text-xs text-mute">Derived cut-offs on predicted-tumor voxels: high ≥ {first.ev.calibration.thresholds.high.toFixed(3)}, medium ≥ {first.ev.calibration.thresholds.medium.toFixed(3)} (accuracy targets 95% / 85%).</p>}
          </div>
        </div>
      )}
    </Card>
  );
}

export default function Confidence() {
  const [params, setParams] = useSearchParams();
  const { data: jobs } = useAsync(() => api.jobs(), []);
  const id = params.get("job") ?? jobs?.jobs[0]?.id ?? null;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="h-page">Confidence & uncertainty laboratory</h1>
        <p className="text-sm text-mute mt-1 max-w-3xl">Confidence is the maximum softmax probability; uncertainty is predictive entropy (and MC-dropout mutual information). Both describe the model, not the patient: a high score does not guarantee a correct segmentation.</p>
      </div>
      <Card>
        <SectionTitle>Session</SectionTitle>
        {!jobs ? <Skeleton className="h-10" /> : jobs.jobs.length === 0 ? (
          <EmptyState title="No sessions to analyse">Run a segmentation first. <Link to="/app/upload" className="text-brand underline">Upload scans</Link>.</EmptyState>
        ) : (
          <select className="input max-w-md" value={id ?? ""} onChange={(e) => setParams({ job: e.target.value })} aria-label="Session">
            {jobs.jobs.map((j) => <option key={j.id} value={j.id}>{j.id.slice(0, 8)} · {new Date(j.created * 1000).toLocaleString()}</option>)}
          </select>
        )}
      </Card>
      {id && <JobAnalysis id={id} />}
      <Calibration />
      <Card>
        <SectionTitle>Definitions</SectionTitle>
        <ul className="text-sm text-mute space-y-1.5 list-disc pl-5">
          <li><b className="text-ink">Voxel confidence</b>: max softmax probability.</li>
          <li><b className="text-ink">Region confidence</b>: mean summed-class probability over voxels predicted inside WT/TC/ET.</li>
          <li><b className="text-ink">Scan confidence</b>: mean voxel confidence over predicted whole-tumor voxels.</li>
          <li><b className="text-ink">Uncertainty</b>: entropy of the averaged probabilities ÷ log 4; mutual information separates epistemic uncertainty when MC-dropout is on.</li>
          <li><b className="text-ink">ECE / Brier</b>: computed against ground truth on the validation split; test data is never used to choose thresholds.</li>
        </ul>
      </Card>
    </div>
  );
}
