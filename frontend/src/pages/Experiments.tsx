import { Download } from "lucide-react";
import { useMemo, useState } from "react";
import { CaseStrip, ConfusionMatrix, DiceBars } from "../components/charts";
import { Badge, Card, EmptyState, ErrorBox, SectionTitle, Skeleton, Stat, useAsync } from "../components/ui";
import { api } from "../lib/api";
import { download, fmtNum, toCsv } from "../lib/format";
import { EvalSchema, type EvalResult } from "../lib/types";

const STATUS_TONE: Record<string, string> = { completed: "ok", running: "brand", planned: "neutral", trained_not_evaluated: "warn", not_implemented: "bad" };
const STATUS_LABEL: Record<string, string> = { completed: "completed", running: "running", planned: "planned (not run)", trained_not_evaluated: "trained, not evaluated", not_implemented: "not implemented" };

interface Row { run: string; name: string; label: string; ev: EvalResult }

export default function Experiments() {
  const { data, error, reload } = useAsync(async () => {
    const exp = await api.experiments();
    const rows: Row[] = [];
    for (const r of exp.runs.filter((x) => x.evals.length)) {
      const full = await api.run(r.id);
      for (const [label, raw] of Object.entries(full.evals)) {
        const p = EvalSchema.safeParse(raw);
        if (p.success) rows.push({ run: r.id, name: r.name ?? r.id, label, ev: p.data });
      }
    }
    return { exp, rows };
  }, []);
  const [pick, setPick] = useState<string | null>(null);
  const [region, setRegion] = useState("WT");
  const sel = useMemo(() => data?.rows.find((r) => `${r.run}/${r.label}` === pick) ?? data?.rows[0], [data, pick]);
  if (error) return <ErrorBox message={error} onRetry={reload} />;
  if (!data) return <Skeleton className="h-96" />;
  const bars = data.rows.map((r) => ({ name: `${r.name.slice(0, 14)}·${r.label}`, WT: r.ev.aggregate.WT.dice.mean, TC: r.ev.aggregate.TC.dice.mean, ET: r.ev.aggregate.ET.dice.mean }));
  const exportCsv = () => sel && download(`evaluation_${sel.run}_${sel.label}.csv`, toCsv(sel.ev.cases.flatMap((c) => Object.entries(c.regions).map(([r, m]) => ({ patient: c.patient, region: r, ...m })))), "text/csv");
  const exportJson = () => sel && download(`evaluation_${sel.run}_${sel.label}.json`, JSON.stringify(sel.ev, null, 2), "application/json");
  const worst = sel ? [...sel.ev.cases].sort((a, b) => (a.regions[region]?.dice ?? 0) - (b.regions[region]?.dice ?? 0)).slice(0, 5) : [];
  return (
    <div className="space-y-6">
      <div>
        <h1 className="h-page">Evaluation & experiments</h1>
        <p className="text-sm text-mute mt-1 max-w-3xl">Only evaluations that were actually executed appear in the results tables. Metrics are computed per patient volume (not averaged over slices); headline numbers should come from the held-out test split, used once.</p>
      </div>

      <Card>
        <SectionTitle>Experiment plan (from the build guide)</SectionTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-mute text-left"><tr><th className="py-1">#</th><th>Experiment</th><th>Status</th><th>Runs</th></tr></thead>
            <tbody>
              {data.exp.experiments.map((e, i) => (
                <tr key={e.id} className="border-t border-line align-top">
                  <td className="py-2 pr-3 num text-mute">{i + 1}</td>
                  <td className="pr-4"><div className="font-medium">{e.title}</div>{e.note && <div className="text-xs text-mute">{e.note}</div>}</td>
                  <td className="pr-4 whitespace-nowrap"><Badge tone={STATUS_TONE[e.status]}>{STATUS_LABEL[e.status] ?? e.status}</Badge></td>
                  <td className="num text-xs text-mute">{e.runs.length ? e.runs.map((r) => r.slice(0, 15)).join(", ") : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {data.rows.length === 0 ? (
        <EmptyState title="No completed evaluations">Train a model, then run an evaluation from the Training lab. Results will be compared here.</EmptyState>
      ) : (
        <>
          <Card>
            <SectionTitle>Run comparison</SectionTitle>
            <div className="grid lg:grid-cols-[1fr_1.2fr] gap-6">
              <DiceBars rows={bars} />
              <div className="overflow-x-auto">
                <table className="w-full text-sm num">
                  <thead className="text-xs text-mute text-left font-sans"><tr><th>Run · eval</th><th>Split</th><th>n</th><th>WT</th><th>TC</th><th>ET</th><th>HD95 WT</th><th>ECE</th></tr></thead>
                  <tbody>
                    {data.rows.map((r) => (
                      <tr key={`${r.run}/${r.label}`} className="border-t border-line cursor-pointer hover:bg-line/30" onClick={() => setPick(`${r.run}/${r.label}`)}>
                        <td className="py-1.5 font-sans">{r.name} · {r.label}</td><td>{r.ev.split}</td><td>{r.ev.n_patients}</td>
                        <td>{fmtNum(r.ev.aggregate.WT.dice.mean)}</td><td>{fmtNum(r.ev.aggregate.TC.dice.mean)}</td><td>{fmtNum(r.ev.aggregate.ET.dice.mean)}</td>
                        <td>{fmtNum(r.ev.aggregate.WT.hd95_mm.mean, 1)}</td><td>{fmtNum(r.ev.calibration.ece, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-xs text-mute mt-2 font-sans">Mean over patients. n is small when the evaluated split is small; do not over-interpret.</p>
              </div>
            </div>
          </Card>

          {sel && (
            <Card>
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <SectionTitle>Detail — {sel.name} · {sel.label} ({sel.ev.split})</SectionTitle>
                <div className="flex gap-2">
                  <select className="input !w-auto !h-8 text-xs" value={region} onChange={(e) => setRegion(e.target.value)} aria-label="Region"><option>WT</option><option>TC</option><option>ET</option></select>
                  <button className="btn-secondary btn-sm" onClick={exportCsv}><Download className="h-3.5 w-3.5" />CSV</button>
                  <button className="btn-secondary btn-sm" onClick={exportJson}><Download className="h-3.5 w-3.5" />JSON</button>
                </div>
              </div>
              <div className="grid sm:grid-cols-5 gap-4 mb-6">
                {(["dice", "iou", "sensitivity", "specificity", "hd95_mm"] as const).map((k) => (
                  <Stat key={k} label={`${region} ${k === "hd95_mm" ? "HD95 (mm)" : k}`} value={fmtNum(sel.ev.aggregate[region][k].mean, k === "hd95_mm" ? 1 : 3)} hint={`± ${fmtNum(sel.ev.aggregate[region][k].std, 3)} (n=${sel.ev.aggregate[region][k].n})`} />
                ))}
              </div>
              <div className="grid lg:grid-cols-2 gap-6">
                <CaseStrip cases={sel.ev.cases} region={region} />
                <ConfusionMatrix m={sel.ev.confusion_matrix} />
              </div>
              <div className="mt-6">
                <div className="text-xs font-medium text-mute mb-2">Worst cases by {region} Dice</div>
                <table className="w-full text-sm num max-w-xl">
                  <thead className="text-xs text-mute text-left font-sans"><tr><th>Patient</th><th>Dice</th><th>HD95</th><th>Vol. pred / GT (mL)</th></tr></thead>
                  <tbody>{worst.map((c) => <tr key={c.patient} className="border-t border-line"><td className="py-1">{c.patient}</td><td>{fmtNum(c.regions[region].dice)}</td><td>{fmtNum(c.regions[region].hd95_mm, 1)}</td><td>{fmtNum(c.regions[region].volume_pred_ml, 2)} / {fmtNum(c.regions[region].volume_gt_ml, 2)}</td></tr>)}</tbody>
                </table>
              </div>
              {sel.ev.uncertainty_vs_error && <p className="text-sm mt-5">Uncertainty vs error: AUROC <b className="num">{fmtNum(sel.ev.uncertainty_vs_error.auroc, 3)}</b>; mean entropy on correct voxels {fmtNum(sel.ev.uncertainty_vs_error.mean_uncertainty_correct, 3)} vs errors {fmtNum(sel.ev.uncertainty_vs_error.mean_uncertainty_error, 3)}.</p>}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
