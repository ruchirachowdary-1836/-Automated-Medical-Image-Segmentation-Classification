import { Moon, RefreshCw, Sun } from "lucide-react";
import { useTheme } from "../components/Layout";
import { Badge, Card, ErrorBox, SectionTitle, Skeleton, useAsync } from "../components/ui";
import { API_URL, api } from "../lib/api";

export default function SettingsPage() {
  const { theme, set } = useTheme();
  const { data, error, reload } = useAsync(async () => {
    const t0 = performance.now();
    const [health, ready, model, settings] = await Promise.all([api.health(), api.ready(), api.modelInfo(), api.settings()]);
    return { health, ready, model, settings, latency: Math.round(performance.now() - t0) };
  }, []);
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div><h1 className="h-page">Settings & system health</h1><p className="text-sm text-mute mt-1">Read-only runtime information. Secrets, filesystem paths and stack traces are never exposed here.</p></div>
        <button className="btn-secondary btn-sm" onClick={reload}><RefreshCw className="h-3.5 w-3.5" />Refresh</button>
      </div>
      {error ? <ErrorBox message={error} onRetry={reload} /> : !data ? <Skeleton className="h-64" /> : (
        <div className="grid lg:grid-cols-2 gap-4">
          <Card>
            <SectionTitle>Connectivity</SectionTitle>
            <dl className="grid grid-cols-[10rem_1fr] gap-y-2 text-sm">
              <dt className="text-mute">Backend URL</dt><dd className="num">{API_URL}</dd>
              <dt className="text-mute">API health</dt><dd><Badge tone="ok">{data.health.status}</Badge> <span className="text-mute num text-xs">v{data.health.version} · {data.latency} ms</span></dd>
              <dt className="text-mute">Inference ready</dt><dd>{data.ready.ready ? <Badge tone="ok">ready</Badge> : <Badge tone="warn">not ready</Badge>} {data.ready.reason && <span className="text-xs text-mute">{data.ready.reason}</span>}</dd>
            </dl>
          </Card>
          <Card>
            <SectionTitle>Model & compute</SectionTitle>
            <dl className="grid grid-cols-[10rem_1fr] gap-y-2 text-sm">
              <dt className="text-mute">Checkpoint</dt><dd className="num">{data.model.checkpoint ?? "none"}</dd>
              <dt className="text-mute">Device</dt><dd>{data.model.device} {data.model.cuda_available ? "· CUDA available" : "· no GPU detected"}</dd>
              <dt className="text-mute">Capabilities</dt><dd className="flex gap-1.5 flex-wrap">{Object.entries(data.model.capabilities).map(([k, v]) => <Badge key={k} tone={v ? "brand" : "neutral"}>{k}</Badge>)}</dd>
              <dt className="text-mute">Device preference</dt><dd>{data.settings.device_preference}</dd>
            </dl>
            {data.model.warning && <p className="text-xs text-warn mt-3">{data.model.warning}</p>}
          </Card>
          <Card>
            <SectionTitle>Limits & inference</SectionTitle>
            <dl className="grid grid-cols-[10rem_1fr] gap-y-2 text-sm">
              <dt className="text-mute">Max upload</dt><dd className="num">{data.settings.max_upload_mb} MB per file</dd>
              <dt className="text-mute">Max grid</dt><dd className="num">{data.settings.max_inplane_dim}² in-plane · {data.settings.max_depth} slices</dd>
              <dt className="text-mute">Rate limit</dt><dd className="num">{data.settings.rate_limit_per_minute} requests / min</dd>
              <dt className="text-mute">MC-dropout samples</dt><dd className="num">{data.settings.mc_dropout_samples} (default)</dd>
              <dt className="text-mute">Confidence cut-offs</dt><dd className="num">high ≥ {data.settings.confidence_defaults.high}, medium ≥ {data.settings.confidence_defaults.medium} (defaults; replaced by validation-derived values when available)</dd>
            </dl>
          </Card>
          <Card>
            <SectionTitle>Privacy & retention</SectionTitle>
            <p className="text-sm">{data.settings.retention_policy}</p>
            <p className="text-xs text-mute mt-2">Use de-identified data only. Sessions can be deleted at any time from the Sessions page.</p>
          </Card>
          <Card>
            <SectionTitle>Datasets</SectionTitle>
            {data.settings.datasets.length ? <ul className="text-sm space-y-1">{data.settings.datasets.map((d) => <li key={d.name}><b>{d.name}</b> <span className="text-mute text-xs">{d.has_processed ? "prepared" : d.has_raw ? "raw only" : "empty"}</span></li>)}</ul> : <p className="text-sm text-mute">None configured. Datasets live under the server&apos;s data directory (set <code className="num">NV_DATA_DIR</code>).</p>}
          </Card>
          <Card>
            <SectionTitle>Appearance</SectionTitle>
            <div className="flex gap-2">
              <button className={theme === "light" ? "btn-primary" : "btn-secondary"} onClick={() => set("light")} aria-pressed={theme === "light"}><Sun className="h-4 w-4" />Light</button>
              <button className={theme === "dark" ? "btn-primary" : "btn-secondary"} onClick={() => set("dark")} aria-pressed={theme === "dark"}><Moon className="h-4 w-4" />Dark</button>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
