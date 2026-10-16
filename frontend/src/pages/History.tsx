import { Search, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Badge, Card, EmptyState, ErrorBox, Skeleton, useAsync, useToast } from "../components/ui";
import { api } from "../lib/api";
import { fmtDate, fmtMl, fmtPct } from "../lib/format";

export default function HistoryPage() {
  const toast = useToast();
  const { data, error, reload } = useAsync(() => api.jobs(), []);
  const [q, setQ] = useState("");
  const [model, setModel] = useState("all");
  const [since, setSince] = useState("");
  const jobs = useMemo(() => data?.jobs ?? [], [data]);
  const models = useMemo(() => Array.from(new Set(jobs.map((j) => j.model))), [jobs]);
  const filtered = jobs.filter((j) =>
    (!q || j.id.includes(q.toLowerCase())) && (model === "all" || j.model === model) && (!since || j.created * 1000 >= new Date(since).getTime()));
  const remove = async (id: string) => {
    try { await api.deleteJob(id); toast("success", "Session deleted"); reload(); } catch (e) { toast("error", (e as Error).message); }
  };
  return (
    <div className="space-y-6">
      <div>
        <h1 className="h-page">Research sessions</h1>
        <p className="text-sm text-mute mt-1 max-w-3xl">Sessions hold only derived results (masks, display layers, metrics) and are deleted automatically when the retention window ends. Original uploads are never stored.</p>
      </div>
      <Card>
        <div className="flex flex-wrap gap-3 mb-4">
          <div className="relative"><Search className="h-4 w-4 absolute left-3 top-3 text-mute" /><input className="input pl-9 !w-56" placeholder="Search by ID" aria-label="Search by ID" value={q} onChange={(e) => setQ(e.target.value)} /></div>
          <select className="input !w-auto" value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model filter"><option value="all">All models</option>{models.map((m) => <option key={m}>{m}</option>)}</select>
          <input className="input !w-auto" type="date" value={since} onChange={(e) => setSince(e.target.value)} aria-label="Since date" />
        </div>
        {error ? <ErrorBox message={error} onRetry={reload} /> : !data ? <Skeleton className="h-40" /> : filtered.length === 0 ? (
          <EmptyState title={jobs.length ? "No sessions match the filters" : "No sessions yet"} action={<Link to="/app/upload" className="btn-primary btn-sm">New segmentation</Link>}>Completed segmentations appear here until they expire.</EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-mute text-left"><tr><th className="py-1">Session</th><th>Created</th><th>Expires</th><th>Model</th><th>WT volume</th><th>Confidence</th><th>Options</th><th /></tr></thead>
              <tbody>
                {filtered.map((j) => (
                  <tr key={j.id} className="border-t border-line">
                    <td className="py-2"><Link to={`/app/results/${j.id}`} className="num text-brand font-medium">{j.id.slice(0, 8)}</Link> <Badge tone="ok">{j.status}</Badge></td>
                    <td className="text-mute">{fmtDate(j.created)}</td><td className="text-mute">{fmtDate(j.expires)}</td>
                    <td className="num text-xs">{j.model}{j.synthetic_model && <Badge tone="warn" title="Trained on synthetic phantoms">synthetic</Badge>}</td>
                    <td className="num">{fmtMl(j.wt_volume_ml)}</td><td className="num">{fmtPct(j.scan_confidence, 0)}</td>
                    <td className="text-xs text-mute">{j.options.tta ? "TTA " : ""}{j.options.mc_samples ? `MC×${j.options.mc_samples}` : ""}</td>
                    <td className="text-right"><button className="btn-ghost btn-sm" onClick={() => remove(j.id)} aria-label={`Delete session ${j.id.slice(0, 8)}`}><Trash2 className="h-3.5 w-3.5" /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
