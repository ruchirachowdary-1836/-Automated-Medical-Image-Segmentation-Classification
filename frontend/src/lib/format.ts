export const fmtMl = (v: number | null | undefined) => (v == null ? "—" : `${v.toFixed(v < 10 ? 2 : 1)} mL`);
export const fmtPct = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const fmtNum = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));
export const fmtBytes = (n: number) =>
  n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
export const fmtDate = (ts: number | null | undefined) => (ts ? new Date(ts * 1000).toLocaleString() : "—");
export const fmtDur = (s: number) => (s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

export function download(filename: string, content: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const esc = (v: unknown) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
}

export const levelTone = (l: string | null | undefined) =>
  l === "high" ? "ok" : l === "medium" ? "warn" : l === "low" ? "bad" : "neutral";
