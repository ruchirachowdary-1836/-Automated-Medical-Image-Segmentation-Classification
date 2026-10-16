import { AlertTriangle, CheckCircle2, Info, Loader2, X } from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { RESEARCH_DISCLAIMER } from "../lib/constants";

export function Card({ children, className = "", pad = true }: { children: ReactNode; className?: string; pad?: boolean }) {
  return <section className={`card ${pad ? "card-pad" : ""} ${className}`}>{children}</section>;
}

export function SectionTitle({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-3 gap-3">
      <h2 className="h-sec">{children}</h2>
      {action}
    </div>
  );
}

type Tone = "neutral" | "ok" | "warn" | "bad" | "brand" | "ai";
const TONES: Record<Tone, string> = {
  neutral: "bg-line/50 text-mute",
  ok: "bg-ok/12 text-ok",
  warn: "bg-warn/15 text-warn",
  bad: "bg-bad/12 text-bad",
  brand: "bg-brand/12 text-brand",
  ai: "bg-ai/12 text-ai",
};
export function Badge({ tone = "neutral", children, title }: { tone?: Tone | string; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${TONES[tone as Tone] ?? TONES.neutral}`}>
      {children}
    </span>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-mute font-medium">{label}</div>
      <div className={`num text-2xl font-semibold mt-1 truncate ${tone ?? ""}`}>{value}</div>
      {hint && <div className="text-xs text-mute mt-0.5">{hint}</div>}
    </div>
  );
}

export function Skeleton({ className = "h-5 w-full" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-mute text-sm" role="status">
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </span>
  );
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center text-center py-10 px-6 border border-dashed border-line rounded-card bg-raised/50">
      {icon && <div className="text-mute mb-3">{icon}</div>}
      <h3 className="font-semibold">{title}</h3>
      {children && <p className="text-sm text-mute mt-1 max-w-md">{children}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-card border border-bad/30 bg-bad/8 p-4 text-sm">
      <AlertTriangle className="h-5 w-5 text-bad shrink-0 mt-0.5" />
      <div className="flex-1">
        <div className="font-medium text-bad">Something went wrong</div>
        <div className="text-ink/80 mt-0.5">{message}</div>
      </div>
      {onRetry && (
        <button className="btn-secondary btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Tooltip({ text, children }: { text: string; children: ReactNode }) {
  return (
    <span className="relative group inline-flex">
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute z-30 bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-max max-w-[16rem] rounded-md bg-navy text-white text-xs px-2.5 py-1.5 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity"
      >
        {text}
      </span>
    </span>
  );
}

export function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <div role="note" className={`flex items-start gap-2 ${compact ? "text-[11px]" : "text-xs"} text-mute leading-snug`}>
      <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
      <span>{RESEARCH_DISCLAIMER}</span>
    </div>
  );
}

export function Progress({ value, label }: { value: number; label?: string }) {
  return (
    <div>
      <div className="h-2 rounded-full bg-line/70 overflow-hidden" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
        <div className="h-full bg-brand transition-all" style={{ width: `${Math.round(Math.min(Math.max(value, 0), 1) * 100)}%` }} />
      </div>
    </div>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: string }[]; value: T; onChange: (t: T) => void }) {
  return (
    <div role="tablist" className="inline-flex gap-1 p-1 rounded-lg bg-raised border border-line">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={value === t.id}
          onClick={() => onChange(t.id)}
          className={`px-3 h-8 rounded-md text-sm font-medium transition-colors ${value === t.id ? "bg-surface shadow-card text-ink" : "text-mute hover:text-ink"}`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

/* ---- toasts ---- */
interface Toast { id: number; kind: "success" | "error" | "info"; text: string }
const ToastCtx = createContext<(kind: Toast["kind"], text: string) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idRef = useRef(0);
  const push = useCallback((kind: Toast["kind"], text: string) => {
    const id = ++idRef.current;
    setToasts((t) => [...t, { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 6000);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-[min(22rem,calc(100vw-2rem))]" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className="card flex items-start gap-2.5 p-3 text-sm">
            {t.kind === "success" ? <CheckCircle2 className="h-4 w-4 text-ok mt-0.5" /> : t.kind === "error" ? <AlertTriangle className="h-4 w-4 text-bad mt-0.5" /> : <Info className="h-4 w-4 text-ai mt-0.5" />}
            <span className="flex-1">{t.text}</span>
            <button aria-label="Dismiss" onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}>
              <X className="h-4 w-4 text-mute" />
            </button>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

/* ---- async data hook ---- */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[], pollMs?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(fn, deps);
  useEffect(() => {
    let alive = true;
    run()
      .then((d) => alive && (setData(d), setError(null)))
      .catch((e: Error) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [run, tick]);
  useEffect(() => {
    if (!pollMs) return;
    const t = setInterval(() => setTick((x) => x + 1), pollMs);
    return () => clearInterval(t);
  }, [pollMs]);
  return { data, error, loading, reload: () => setTick((x) => x + 1) };
}
