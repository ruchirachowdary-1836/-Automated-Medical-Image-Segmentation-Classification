import { Activity, BarChart3, Brain, FlaskConical, History, LayoutDashboard, Menu, Moon, Settings, ShieldCheck, Sun, Upload, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { api } from "../lib/api";
import { Disclaimer, Tooltip, useAsync } from "./ui";

const NAV = [
  { to: "/app", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/app/upload", label: "New segmentation", icon: Upload },
  { to: "/app/confidence", label: "Confidence lab", icon: ShieldCheck },
  { to: "/app/training", label: "Training lab", icon: FlaskConical },
  { to: "/app/experiments", label: "Evaluation", icon: BarChart3 },
  { to: "/app/history", label: "Sessions", icon: History },
  { to: "/app/settings", label: "Settings", icon: Settings },
];

export function Logo({ className = "" }: { className?: string }) {
  return (
    <Link to="/" className={`flex items-center gap-2.5 font-semibold tracking-tight ${className}`} aria-label="NeuroVista AI home">
      <span className="grid place-items-center h-8 w-8 rounded-lg bg-navy text-brand">
        <Brain className="h-5 w-5" />
      </span>
      <span>
        NeuroVista <span className="text-ai">AI</span>
      </span>
    </Link>
  );
}

export function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => (document.documentElement.dataset.theme === "dark" ? "dark" : "light"));
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("nv-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  return { theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")), set: setTheme };
}

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button className="btn-ghost btn-sm !px-2" onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

function HealthPill() {
  const { data } = useAsync(async () => {
    const [h, r] = await Promise.allSettled([api.health(), api.ready()]);
    return { api: h.status === "fulfilled", model: r.status === "fulfilled" && r.value.ready, reason: r.status === "fulfilled" ? r.value.reason : "unreachable" };
  }, [], 15000);
  const apiOk = data?.api ?? false;
  const tone = !data ? "bg-line" : !apiOk ? "bg-bad" : data.model ? "bg-ok" : "bg-warn";
  const text = !data ? "Checking…" : !apiOk ? "API offline" : data.model ? "API & model ready" : "Model not loaded";
  return (
    <Tooltip text={!apiOk && data ? "Backend unreachable. Start it with uvicorn." : (data?.reason ?? "Backend connectivity and model readiness")}>
      <span className="inline-flex items-center gap-2 text-xs text-mute border border-line rounded-full px-3 h-8 bg-raised" role="status">
        <span className={`h-2 w-2 rounded-full ${tone}`} />
        {text}
      </span>
    </Tooltip>
  );
}

export default function Layout() {
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-full flex">
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 bg-surface border-r border-line flex flex-col transition-transform lg:translate-x-0 lg:static lg:shrink-0 ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="h-16 px-5 flex items-center justify-between border-b border-line">
          <Logo />
          <button className="lg:hidden" onClick={() => setOpen(false)} aria-label="Close menu">
            <X className="h-5 w-5" />
          </button>
        </div>
        <nav className="p-3 flex-1 space-y-0.5" aria-label="Primary">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 h-10 px-3 rounded-lg text-sm font-medium transition-colors ${isActive ? "bg-brand/12 text-brand" : "text-mute hover:text-ink hover:bg-line/40"}`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-line">
          <Disclaimer compact />
        </div>
      </aside>
      {open && <div className="fixed inset-0 bg-black/40 z-30 lg:hidden" onClick={() => setOpen(false)} />}
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-16 px-4 sm:px-8 border-b border-line bg-surface/80 backdrop-blur flex items-center gap-3 sticky top-0 z-20">
          <button className="lg:hidden btn-ghost !px-2" onClick={() => setOpen(true)} aria-label="Open menu">
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2 text-sm text-mute">
            <Activity className="h-4 w-4" /> Research workspace
          </div>
          <div className="ml-auto flex items-center gap-2">
            <HealthPill />
            <ThemeToggle />
          </div>
        </header>
        <div className="bg-warn/10 border-b border-warn/25 px-4 sm:px-8 py-2">
          <Disclaimer />
        </div>
        <main className="flex-1 px-4 sm:px-8 py-6 max-w-[1400px] w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
