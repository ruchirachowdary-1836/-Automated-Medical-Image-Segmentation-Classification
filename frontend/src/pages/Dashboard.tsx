import { ArrowRight, BarChart3, Database, FlaskConical, HardDrive, Upload } from "lucide-react";
import { Link } from "react-router-dom";
import { Badge, Card, EmptyState, ErrorBox, SectionTitle, Skeleton, Stat, useAsync } from "../components/ui";
import { api } from "../lib/api";
import { fmtDate, fmtMl, fmtNum, fmtPct } from "../lib/format";

export default function Dashboard() {
  const { data, error, loading, reload } = useAsync(() => api.dashboard(), [], 10000);
  if (loading) return <div className="space-y-4"><Skeleton className="h-8 w-56" /><div className="grid md:grid-cols-4 gap-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28" />)}</div></div>;
  if (error || !data)
    return (
      <div className="space-y-3">
        <h1 className="h-page">Dashboard</h1>
        <ErrorBox message={error ?? "No data"} onRetry={reload} />
        <p className="text-sm text-mute">Start the backend: <code className="num">uvicorn neurovista.api.main:app --port 8000</code></p>
      </div>
    );
  const m = data.model;
  const run = data.latest_run;
  const last = run?.history.at(-1);
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><h1 className="h-page">Dashboard</h1><p className="text-sm text-mute mt-1">Everything shown here is read from the backend; nothing is estimated or pre-filled.</p></div>
        <div className="flex gap-2">
          <Link to="/app/upload" className="btn-primary"><Upload className="h-4 w-4" />Upload MRI scans</Link>
          <Link to="/app/experiments" className="btn-secondary"><BarChart3 className="h-4 w-4" />Review experiments</Link>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <Card><Stat label="Retained inference jobs" value={data.jobs_retained} hint="Results are auto-deleted after the retention window" /></Card>
        <Card><Stat label="Model" value={m.loaded ? <Badge tone="ok">ready</Badge> : <Badge tone="warn">not loaded</Badge>} hint={m.loaded ? `${m.checkpoint} · ${m.device}` : (m.reason ?? "No checkpoint")} /></Card>
        <Card><Stat label="Training runs" value={data.runs} hint={data.runs ? "see Training lab" : "none yet"} /></Card>
        <Card><Stat label="Datasets" value={data.datasets.length} hint={data.datasets.length ? data.datasets.map((d) => d.name).join(", ") : "none configured"} /></Card>
      </div>

      {m.warning && <div role="alert" className="rounded-card border border-warn/40 bg-warn/10 p-3 text-sm">{m.warning}</div>}

      <div className="grid lg:grid-cols-2 gap-4">
        <Card>
          <SectionTitle action={<Link to="/app/history" className="text-xs text-brand font-medium">All sessions</Link>}>Recent segmentation sessions</SectionTitle>
          {data.recent_jobs.length === 0 ? (
            <EmptyState icon={<Upload className="h-6 w-6" />} title="No sessions yet" action={<Link to="/app/upload" className="btn-primary btn-sm">Upload a scan</Link>}>
              Results appear here after you run a segmentation. They are kept only temporarily.
            </EmptyState>
          ) : (
            <ul className="divide-y divide-line">
              {data.recent_jobs.map((j) => (
                <li key={j.id}>
                  <Link to={`/app/results/${j.id}`} className="flex items-center justify-between py-2.5 hover:bg-line/30 -mx-2 px-2 rounded-md">
                    <span><span className="num text-sm">{j.id.slice(0, 8)}</span><span className="block text-xs text-mute">{fmtDate(j.created)} · {j.model}</span></span>
                    <span className="text-right text-sm num">{fmtMl(j.wt_volume_ml)}<span className="block text-xs text-mute">WT · conf {fmtPct(j.scan_confidence, 0)}</span></span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card>
          <SectionTitle action={<Link to="/app/training" className="text-xs text-brand font-medium">Training lab</Link>}>Latest experiment</SectionTitle>
          {!run ? (
            <EmptyState icon={<FlaskConical className="h-6 w-6" />} title="The model has not been trained here">
              No training run exists in this workspace. Prepare a dataset (see the dataset guide) and start a run, or mount a checkpoint.
            </EmptyState>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center gap-2"><span className="font-medium">{String(run.config.name ?? run.id)}</span><Badge tone={run.status.state === "completed" ? "ok" : run.status.state === "failed" ? "bad" : "brand"}>{run.status.state}</Badge>{run.config.synthetic === true && <Badge tone="warn">synthetic data</Badge>}</div>
              <div className="grid grid-cols-3 gap-4">
                <Stat label="Epochs" value={`${run.history.length}/${run.status.total_epochs ?? "?"}`} />
                <Stat label="Best val Dice" value={fmtNum(run.status.best_mean_dice)} hint="mean WT/TC/ET (validation)" />
                <Stat label="Last val loss" value={fmtNum(last?.val_loss)} />
              </div>
              <Link to="/app/training" className="inline-flex items-center gap-1 text-sm text-brand font-medium">Open run <ArrowRight className="h-3.5 w-3.5" /></Link>
            </div>
          )}
        </Card>

        <Card>
          <SectionTitle>Model & checkpoint</SectionTitle>
          {m.loaded ? (
            <dl className="grid grid-cols-[8rem_1fr] gap-y-1.5 text-sm">
              <dt className="text-mute">Checkpoint</dt><dd className="num">{m.checkpoint}</dd>
              <dt className="text-mute">Parameters</dt><dd className="num">{m.params ? `${(m.params / 1e6).toFixed(2)} M` : "—"}</dd>
              <dt className="text-mute">Device</dt><dd>{m.device} {m.cuda_available ? "(CUDA available)" : "(CPU only)"}</dd>
              <dt className="text-mute">Validation Dice</dt><dd>{m.meta?.val_mean_dice != null ? `${fmtNum(Number(m.meta.val_mean_dice))} (mean of WT/TC/ET at best epoch)` : "not recorded"}</dd>
              <dt className="text-mute">Confidence cut-offs</dt><dd>{m.confidence_thresholds ? `${m.confidence_thresholds.source}` : "—"}</dd>
            </dl>
          ) : <EmptyState icon={<HardDrive className="h-6 w-6" />} title="No model checkpoint loaded">{m.reason}</EmptyState>}
        </Card>

        <Card>
          <SectionTitle>Datasets</SectionTitle>
          {data.datasets.length === 0 ? (
            <EmptyState icon={<Database className="h-6 w-6" />} title="No dataset found">Place BraTS-style cases under <code className="num">var/datasets/&lt;name&gt;/raw</code> and run <code className="num">neurovista prepare</code>.</EmptyState>
          ) : (
            <ul className="space-y-2 text-sm">
              {data.datasets.map((d) => (
                <li key={d.name} className="flex justify-between"><span className="font-medium">{d.name}</span>
                  <span className="text-mute num">{d.splits ? `train ${d.splits.train} · val ${d.splits.val} · test ${d.splits.test} patients` : d.has_raw ? "raw only — run prepare" : "empty"}</span></li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}
