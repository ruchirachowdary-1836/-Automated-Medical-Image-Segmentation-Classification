import { FlaskConical, Play, Square, RotateCw, CheckCheck, ClipboardCheck } from "lucide-react";
import { useState } from "react";
import { DiceChart, LossChart, LrChart } from "../components/charts";
import { Badge, Card, EmptyState, ErrorBox, Progress, SectionTitle, Skeleton, Stat, useAsync, useToast } from "../components/ui";
import { api } from "../lib/api";
import { fmtDate, fmtNum } from "../lib/format";

const STATE_TONE: Record<string, string> = { completed: "ok", running: "brand", failed: "bad", cancelled: "warn", queued: "neutral" };

function RunDetail({ id, onChanged }: { id: string; onChanged: () => void }) {
  const toast = useToast();
  const { data: run, error, reload } = useAsync(() => api.run(id), [id], 3000);
  const [evalSplit, setEvalSplit] = useState("val");
  if (error) return <ErrorBox message={error} />;
  if (!run) return <Skeleton className="h-72" />;
  const st = run.status;
  const running = st.state === "running" || st.state === "queued";
  const act = async (fn: () => Promise<unknown>, ok: string) => {
    try { await fn(); toast("success", ok); reload(); onChanged(); } catch (e) { toast("error", (e as Error).message); }
  };
  const cfg = run.config as Record<string, unknown>;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="font-semibold">{String(cfg.name ?? run.id)}</h3>
        <Badge tone={STATE_TONE[st.state ?? ""] ?? "neutral"}>{st.state}</Badge>
        {cfg.synthetic === true && <Badge tone="warn">synthetic data</Badge>}
        <span className="num text-xs text-mute">{run.id}</span>
        <div className="ml-auto flex flex-wrap gap-2">
          {running && <button className="btn-danger btn-sm" onClick={() => act(() => api.cancelRun(id), "Cancel requested")}><Square className="h-3.5 w-3.5" />Cancel</button>}
          {!running && st.state !== "completed" && <button className="btn-secondary btn-sm" onClick={() => act(() => api.resumeRun(id), "Resumed from last checkpoint")}><RotateCw className="h-3.5 w-3.5" />Resume</button>}
          {run.has_best && !running && <button className="btn-secondary btn-sm" onClick={() => act(() => api.activateRun(id), "Best checkpoint is now the serving model")}><CheckCheck className="h-3.5 w-3.5" />Use for inference</button>}
        </div>
      </div>
      {st.error && <ErrorBox message={st.error} />}
      <div className="grid sm:grid-cols-4 gap-4">
        <Stat label="Epoch" value={`${st.epoch ?? 0}/${st.total_epochs ?? "?"}`} />
        <Stat label="Best val Dice" value={fmtNum(st.best_mean_dice)} hint={st.best_epoch ? `epoch ${st.best_epoch}` : undefined} />
        <Stat label="Device" value={st.device ?? "—"} />
        <Stat label="Slices (train/val)" value={`${st.train_slices ?? "—"}/${st.val_slices ?? "—"}`} />
      </div>
      {st.total_epochs ? <Progress value={(st.epoch ?? 0) / st.total_epochs} label="Training progress" /> : null}
      {run.history.length === 0 ? <EmptyState title="No epochs finished yet">Curves appear after the first epoch completes.</EmptyState> : (
        <div className="grid lg:grid-cols-2 gap-6">
          <LossChart history={run.history} /><DiceChart history={run.history} /><LrChart history={run.history} />
          <div>
            <div className="text-xs font-medium text-mute mb-2">Best checkpoint</div>
            <p className="text-sm">{run.has_best ? <>Saved at epoch <b className="num">{st.best_epoch ?? "—"}</b> with validation mean Dice <b className="num">{fmtNum(st.best_mean_dice)}</b>.</> : "No checkpoint yet."}</p>
            {run.has_best && !running && (
              <div className="mt-3 flex flex-wrap gap-2 items-center">
                <select className="input !w-auto !h-8 text-xs" value={evalSplit} onChange={(e) => setEvalSplit(e.target.value)} aria-label="Evaluation split"><option value="val">validation</option><option value="test">test (report once)</option></select>
                <button className="btn-secondary btn-sm" onClick={() => act(() => api.evaluateRun(id, { split: evalSplit, tta: false, mc_samples: 0, min_component_voxels: 0, label: evalSplit }), "Evaluation started")}><ClipboardCheck className="h-3.5 w-3.5" />Evaluate</button>
                <button className="btn-secondary btn-sm" onClick={() => act(() => api.evaluateRun(id, { split: evalSplit, tta: true, mc_samples: 0, min_component_voxels: 20, label: `tta_postproc_${evalSplit}` }), "TTA + post-processing evaluation started")}>Evaluate + TTA</button>
                <button className="btn-secondary btn-sm" onClick={() => act(() => api.evaluateRun(id, { split: evalSplit, tta: false, mc_samples: 8, min_component_voxels: 0, label: `mc_dropout_vs_error_${evalSplit}` }), "MC-dropout evaluation started")}>Evaluate + MC-dropout</button>
              </div>
            )}
            {Object.keys(run.evals).length > 0 && <p className="text-xs text-mute mt-2">Finished evaluations: {Object.keys(run.evals).join(", ")}</p>}
          </div>
        </div>
      )}
      <details className="text-sm"><summary className="cursor-pointer text-mute">Configuration & log</summary>
        <pre className="mt-2 text-xs num bg-raised border border-line rounded-lg p-3 overflow-auto max-h-60">{JSON.stringify(cfg, null, 2)}</pre>
        <pre className="mt-2 text-xs num bg-raised border border-line rounded-lg p-3 overflow-auto max-h-48 whitespace-pre-wrap">{run.log || "(log is empty)"}</pre>
      </details>
    </div>
  );
}

export default function Training() {
  const toast = useToast();
  const { data: defs, error, reload: reloadDefs } = useAsync(() => api.trainingDefaults(), []);
  const { data: runs, reload: reloadRuns } = useAsync(() => api.runs(), [], 5000);
  const [sel, setSel] = useState<string | null>(null);
  const [form, setForm] = useState<Record<string, unknown>>({ dataset: "", preset: "dice_ce", name: "run", epochs: 60, batch_size: 16, lr: 0.0003, patience: 15, seed: 42, device: "auto", base: 32, scheduler: "cosine", loss: "dice_ce", model: "unet", synthetic: false });
  const [busy, setBusy] = useState(false);
  const set = (k: string, v: unknown) => setForm((f) => ({ ...f, [k]: v }));
  const applyPreset = (id: string) => {
    const p = defs?.presets.find((x) => x.id === id);
    setForm((f) => ({ ...f, preset: id || null, ...(p?.overrides ?? {}) }));
  };
  if (error) return <ErrorBox message={error} onRetry={reloadDefs} />;
  if (!defs) return <Skeleton className="h-96" />;
  const trainable = defs.presets.filter((p) => p.kind === "training");
  const start = async () => {
    setBusy(true);
    try {
      const run = await api.startRun(form);
      toast("success", "Training started");
      setSel(run.id); reloadRuns();
    } catch (e) { toast("error", (e as Error).message); } finally { setBusy(false); }
  };
  const num = (k: string, label: string, step?: string) => (
    <div><label className="label" htmlFor={k}>{label}</label><input id={k} className="input" type="number" step={step} value={String(form[k] ?? "")} onChange={(e) => set(k, e.target.value === "" ? "" : +e.target.value)} /></div>
  );
  const selected = sel ?? runs?.runs[0]?.id ?? null;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="h-page">Training laboratory</h1>
        <p className="text-sm text-mute mt-1 max-w-3xl">Training never starts automatically. Runs execute as a separate process, are seeded for reproducibility, checkpoint every epoch and can be cancelled or resumed.</p>
      </div>
      <Card>
        <SectionTitle>Configure a run</SectionTitle>
        {!defs.enabled ? <ErrorBox message="Training is disabled on this deployment (NV_TRAINING_ENABLED=false)." /> : defs.datasets.filter((d) => d.has_processed).length === 0 ? (
          <EmptyState icon={<FlaskConical className="h-6 w-6" />} title="No prepared dataset">
            Put BraTS-style cases in <code className="num">var/datasets/&lt;name&gt;/raw</code> and run <code className="num">python -m neurovista.cli prepare --name &lt;name&gt;</code>. For a pipeline smoke test only, <code className="num">python -m neurovista.cli synth</code> creates a synthetic phantom dataset. See TRAINING_GUIDE.md.
          </EmptyState>
        ) : (
          <>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div><label className="label" htmlFor="ds">Dataset</label>
                <select id="ds" className="input" value={String(form.dataset)} onChange={(e) => { const d = defs.datasets.find((x) => x.name === e.target.value); setForm((f) => ({ ...f, dataset: e.target.value, synthetic: /synth/i.test(e.target.value) || f.synthetic })); void d; }}>
                  <option value="">Select…</option>
                  {defs.datasets.filter((d) => d.has_processed).map((d) => <option key={d.name} value={d.name}>{d.name}{d.splits ? ` (${d.splits.train}/${d.splits.val}/${d.splits.test})` : ""}</option>)}
                </select></div>
              <div><label className="label" htmlFor="preset">Experiment preset</label>
                <select id="preset" className="input" value={String(form.preset ?? "")} onChange={(e) => applyPreset(e.target.value)}>
                  <option value="">Custom</option>{trainable.map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
                </select></div>
              <div><label className="label" htmlFor="nm">Run name</label><input id="nm" className="input" value={String(form.name)} onChange={(e) => set("name", e.target.value)} /></div>
              <div><label className="label" htmlFor="dev">Device</label>
                <select id="dev" className="input" value={String(form.device)} onChange={(e) => set("device", e.target.value)}><option value="auto">auto ({defs.cuda_available ? "CUDA found" : "CPU only"})</option><option value="cpu">cpu</option><option value="cuda" disabled={!defs.cuda_available}>cuda</option></select></div>
              {num("epochs", "Epochs")}{num("batch_size", "Batch size")}{num("lr", "Learning rate", "0.0001")}{num("patience", "Early-stop patience")}
              {num("seed", "Random seed")}{num("base", "Base channels")}
              <div><label className="label" htmlFor="loss">Loss</label><select id="loss" className="input" value={String(form.loss)} onChange={(e) => set("loss", e.target.value)}><option value="dice_ce">Dice + cross-entropy</option><option value="ce">Cross-entropy</option><option value="dice">Dice</option><option value="focal_tversky">Focal Tversky</option></select></div>
              <div><label className="label" htmlFor="sch">Scheduler</label><select id="sch" className="input" value={String(form.scheduler)} onChange={(e) => set("scheduler", e.target.value)}><option value="cosine">Cosine annealing</option><option value="plateau">ReduceLROnPlateau</option></select></div>
            </div>
            <p className="text-xs text-mute mt-3">Optimizer: AdamW (weight decay 1e-4) · gradient clipping 1.0 · mixed precision on CUDA · input modalities {(form.modalities as string[] | undefined)?.join(", ") ?? "t1, t1ce, t2, flair"}.</p>
            <label className="flex items-center gap-2 text-sm mt-3"><input type="checkbox" className="accent-[rgb(var(--brand))]" checked={form.synthetic === true} onChange={(e) => set("synthetic", e.target.checked)} />Dataset is synthetic (marks resulting checkpoints as smoke-test only)</label>
            <button className="btn-primary mt-4" disabled={busy || !form.dataset} onClick={start}><Play className="h-4 w-4" />Start training</button>
          </>
        )}
      </Card>
      <div className="grid xl:grid-cols-[18rem_1fr] gap-4">
        <Card>
          <SectionTitle>Runs</SectionTitle>
          {!runs || runs.runs.length === 0 ? <p className="text-sm text-mute">No runs yet.</p> : (
            <ul className="space-y-1">
              {runs.runs.map((r) => (
                <li key={r.id}><button onClick={() => setSel(r.id)} className={`w-full text-left rounded-lg px-3 py-2 text-sm ${selected === r.id ? "bg-brand/12" : "hover:bg-line/40"}`}>
                  <span className="font-medium">{r.name ?? r.id}</span> <Badge tone={STATE_TONE[r.state ?? ""] ?? "neutral"}>{r.state}</Badge>
                  <span className="block text-xs text-mute num">{r.epochs_done} epochs · best {fmtNum(r.best_mean_dice)} · {fmtDate(r.created)}</span></button></li>
              ))}
            </ul>
          )}
        </Card>
        <Card>{selected ? <RunDetail id={selected} onChanged={reloadRuns} /> : <EmptyState title="Select or start a run" />}</Card>
      </div>
    </div>
  );
}
