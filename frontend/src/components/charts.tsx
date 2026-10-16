import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import type { EvalResult, HistoryEntry } from "../lib/types";

export const C = { brand: "rgb(13,148,136)", ai: "rgb(109,91,224)", warn: "rgb(220,150,20)", bad: "rgb(220,60,70)", blue: "rgb(59,130,246)", mute: "rgb(120,134,154)" };
const axis = { stroke: "rgb(120,134,154)", fontSize: 11, tickLine: false } as const;
const tip = { contentStyle: { background: "rgb(var(--surface))", border: "1px solid rgb(var(--line))", borderRadius: 8, fontSize: 12, color: "rgb(var(--ink))" } };

export function Frame({ title, children, h = 240 }: { title: string; children: React.ReactElement; h?: number }) {
  return (
    <figure>
      <figcaption className="text-xs font-medium text-mute mb-2">{title}</figcaption>
      <div style={{ height: h }}>
        <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
      </div>
    </figure>
  );
}

export function LossChart({ history }: { history: HistoryEntry[] }) {
  return (
    <Frame title="Training vs validation loss">
      <LineChart data={history} margin={{ left: -10, right: 8, top: 4 }}>
        <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 3" />
        <XAxis dataKey="epoch" {...axis} /><YAxis {...axis} /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="train_loss" name="train" stroke={C.brand} dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line type="monotone" dataKey="val_loss" name="validation" stroke={C.ai} dot={false} strokeWidth={2} isAnimationActive={false} />
      </LineChart>
    </Frame>
  );
}

export function DiceChart({ history }: { history: HistoryEntry[] }) {
  const data = history.map((h) => ({ epoch: h.epoch, ...h.dice }));
  return (
    <Frame title="Validation Dice per region">
      <LineChart data={data} margin={{ left: -10, right: 8, top: 4 }}>
        <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 3" />
        <XAxis dataKey="epoch" {...axis} /><YAxis domain={[0, 1]} {...axis} /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 12 }} />
        <Line dataKey="WT" stroke={C.brand} dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line dataKey="TC" stroke={C.blue} dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line dataKey="ET" stroke={C.bad} dot={false} strokeWidth={2} isAnimationActive={false} />
      </LineChart>
    </Frame>
  );
}

export function LrChart({ history }: { history: HistoryEntry[] }) {
  return (
    <Frame title="Learning rate" h={180}>
      <LineChart data={history} margin={{ left: 0, right: 8, top: 4 }}>
        <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 3" />
        <XAxis dataKey="epoch" {...axis} /><YAxis {...axis} tickFormatter={(v: number) => v.toExponential(0)} /><Tooltip {...tip} />
        <Line dataKey="lr" stroke={C.warn} dot={false} strokeWidth={2} isAnimationActive={false} />
      </LineChart>
    </Frame>
  );
}

export function Histogram({ data, title, color }: { data: number[]; title: string; color: string }) {
  const total = data.reduce((a, b) => a + b, 0) || 1;
  const rows = data.map((v, i) => ({ bin: `${(i / data.length).toFixed(2)}`, pct: (v / total) * 100 }));
  return (
    <Frame title={title} h={200}>
      <BarChart data={rows} margin={{ left: -10, right: 8, top: 4 }}>
        <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="bin" {...axis} interval={3} /><YAxis {...axis} unit="%" /><Tooltip {...tip} formatter={(v: number) => `${v.toFixed(1)}% of tumor voxels`} />
        <Bar dataKey="pct" fill={color} isAnimationActive={false} />
      </BarChart>
    </Frame>
  );
}

export function ReliabilityDiagram({ bins }: { bins: EvalResult["calibration"]["bins"] }) {
  const pts = bins.filter((b) => b.count > 0 && b.confidence != null && b.accuracy != null).map((b) => ({ conf: b.confidence as number, acc: b.accuracy as number, n: b.count }));
  return (
    <Frame title="Reliability diagram (validation voxels)" h={260}>
      <ScatterChart margin={{ left: -6, right: 12, top: 8, bottom: 4 }}>
        <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 3" />
        <XAxis type="number" dataKey="conf" domain={[0, 1]} name="confidence" {...axis} label={{ value: "mean confidence", position: "insideBottom", offset: -2, fontSize: 11, fill: C.mute }} />
        <YAxis type="number" dataKey="acc" domain={[0, 1]} name="accuracy" {...axis} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.mute} strokeDasharray="5 4" />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(3)} />
        <Scatter data={pts} fill={C.ai} line={{ stroke: C.ai }} isAnimationActive={false} />
      </ScatterChart>
    </Frame>
  );
}

export function DiceBars({ rows }: { rows: { name: string; WT: number | null; TC: number | null; ET: number | null }[] }) {
  return (
    <Frame title="Mean Dice by run (patient-level)" h={240}>
      <BarChart data={rows} margin={{ left: -10, right: 8, top: 4 }}>
        <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" {...axis} /><YAxis domain={[0, 1]} {...axis} /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="WT" fill={C.brand} isAnimationActive={false} /><Bar dataKey="TC" fill={C.blue} isAnimationActive={false} /><Bar dataKey="ET" fill={C.bad} isAnimationActive={false} />
      </BarChart>
    </Frame>
  );
}

export function CaseStrip({ cases, region }: { cases: EvalResult["cases"]; region: string }) {
  const data = cases.map((c, i) => ({ i, dice: c.regions[region]?.dice ?? 0, patient: c.patient }));
  return (
    <Frame title={`Per-patient ${region} Dice distribution`} h={200}>
      <BarChart data={[...data].sort((a, b) => a.dice - b.dice)} margin={{ left: -10, right: 8, top: 4 }}>
        <CartesianGrid stroke="rgb(var(--line))" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="patient" {...axis} hide={data.length > 12} /><YAxis domain={[0, 1]} {...axis} /><Tooltip {...tip} formatter={(v: number) => v.toFixed(3)} />
        <Bar dataKey="dice" fill={C.ai} isAnimationActive={false} />
      </BarChart>
    </Frame>
  );
}

export function ConfusionMatrix({ m }: { m: number[][] }) {
  const names = ["BG", "NCR/NET", "ED", "ET"];
  const rowSums = m.map((r) => r.reduce((a, b) => a + b, 0) || 1);
  return (
    <div className="overflow-x-auto">
      <div className="text-xs font-medium text-mute mb-2">Voxel confusion matrix (rows: ground truth, % of row)</div>
      <table className="text-xs num border-separate border-spacing-1">
        <thead><tr><th /> {names.map((n) => <th key={n} className="text-mute font-medium px-2">{n}</th>)}</tr></thead>
        <tbody>
          {m.map((row, i) => (
            <tr key={i}>
              <th className="text-mute font-medium text-right pr-2">{names[i]}</th>
              {row.map((v, j) => {
                const f = v / rowSums[i];
                return <td key={j} title={`${v.toLocaleString()} voxels`} className="rounded px-3 py-2 text-center" style={{ background: `rgb(109 91 224 / ${Math.min(0.08 + f * 0.85, 0.95)})`, color: f > 0.5 ? "#fff" : "inherit" }}>{(f * 100).toFixed(1)}%</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
