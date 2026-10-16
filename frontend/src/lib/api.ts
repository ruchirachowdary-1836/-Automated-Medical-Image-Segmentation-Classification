import { z } from "zod";
import {
  DashboardSchema,
  EvalSchema,
  ExperimentSchema,
  GroundTruthEvalSchema,
  JobSchema,
  JobSummarySchema,
  ModelInfoSchema,
  RunSchema,
  RunSummarySchema,
  SettingsSchema,
  TrainingDefaultsSchema,
  type Modality,
} from "./types";

export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";
const TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  constructor(
    message: string,
    public code: string,
    public status: number,
    public modality?: string,
  ) {
    super(message);
  }
}

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const e = body?.error ?? {};
    return new ApiError(e.message ?? `Request failed (${res.status})`, e.code ?? "http_error", res.status, e.modality);
  } catch {
    return new ApiError(`Request failed (${res.status})`, "http_error", res.status);
  }
}

async function request<T>(path: string, schema: z.ZodType<T>, init: RequestInit & { timeout?: number } = {}): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), init.timeout ?? TIMEOUT_MS);
  try {
    const res = await fetch(`${API_URL}${path}`, { ...init, signal: init.signal ?? ctrl.signal });
    if (!res.ok) throw await parseError(res);
    const json = await res.json();
    const parsed = schema.safeParse(json);
    if (!parsed.success) throw new ApiError("The server returned an unexpected response format.", "bad_response", res.status);
    return parsed.data;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if (e instanceof DOMException && e.name === "AbortError") throw new ApiError("The request timed out.", "timeout", 0);
    throw new ApiError("Cannot reach the NeuroVista backend. Is it running?", "network", 0);
  } finally {
    clearTimeout(timer);
  }
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request("/health", z.object({ status: z.string(), version: z.string(), uptime_s: z.number() }), { timeout: 5000 }),
  ready: async () => {
    try {
      const res = await fetch(`${API_URL}/ready`);
      return (await res.json()) as { ready: boolean; reason: string | null };
    } catch {
      return { ready: false, reason: "Backend unreachable" };
    }
  },
  modelInfo: () => request("/model/info", ModelInfoSchema),
  settings: () => request("/settings", SettingsSchema),
  dashboard: () => request("/dashboard", DashboardSchema),
  jobs: () => request("/jobs", z.object({ jobs: z.array(JobSummarySchema) })),
  job: (id: string) => request(`/jobs/${id}`, JobSchema),
  deleteJob: (id: string) => request(`/jobs/${id}`, z.object({ deleted: z.boolean() }), { method: "DELETE" }),
  experiments: () => request("/experiments", z.object({ experiments: z.array(ExperimentSchema), runs: z.array(RunSummarySchema) })),
  trainingDefaults: () => request("/training/defaults", TrainingDefaultsSchema),
  runs: () => request("/training/runs", z.object({ runs: z.array(RunSummarySchema) })),
  run: (id: string) => request(`/training/runs/${id}`, RunSchema),
  startRun: (cfg: Record<string, unknown>) => request("/training/runs", RunSchema, json(cfg)),
  cancelRun: (id: string) => request(`/training/runs/${id}/cancel`, z.object({ cancel_requested: z.boolean() }), { method: "POST" }),
  resumeRun: (id: string) => request(`/training/runs/${id}/resume`, RunSchema, { method: "POST" }),
  activateRun: (id: string) => request(`/training/runs/${id}/activate`, ModelInfoSchema, { method: "POST" }),
  evaluateRun: (id: string, body: { split: string; tta: boolean; mc_samples: number; min_component_voxels: number; label: string }) =>
    request(`/training/runs/${id}/evaluate`, z.object({ started: z.boolean() }), json(body)),
  evalResult: (r: unknown) => EvalSchema.safeParse(r),
  evaluateGroundTruth: (jobId: string, file: File) => {
    const fd = new FormData();
    fd.append("job_id", jobId);
    fd.append("ground_truth", file);
    return request("/evaluate", GroundTruthEvalSchema, { method: "POST", body: fd, timeout: 120_000 });
  },
  async layer(jobId: string, name: string): Promise<Uint8Array> {
    const res = await fetch(`${API_URL}/jobs/${jobId}/layer/${name}`);
    if (!res.ok) throw await parseError(res);
    return new Uint8Array(await res.arrayBuffer());
  },
  downloadUrl: (path: string) => `${API_URL}${path}`,
};

export interface PredictOptions {
  tta: boolean;
  mcSamples: number;
  minComponentVoxels: number;
}

/** Multipart upload via XHR so we get real upload progress and can cancel. */
export function predict(
  files: Record<Modality, File>,
  opts: PredictOptions,
  onProgress: (fraction: number, phase: "uploading" | "processing") => void,
): { promise: Promise<z.infer<typeof JobSchema>>; cancel: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<z.infer<typeof JobSchema>>((resolve, reject) => {
    const fd = new FormData();
    (Object.keys(files) as Modality[]).forEach((m) => fd.append(m, files[m], files[m].name));
    fd.append("tta", String(opts.tta));
    fd.append("mc_samples", String(opts.mcSamples));
    fd.append("min_component_voxels", String(opts.minComponentVoxels));
    xhr.open("POST", `${API_URL}/predict`);
    xhr.timeout = 15 * 60_000;
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total, "uploading");
    xhr.upload.onload = () => onProgress(1, "processing");
    xhr.onerror = () => reject(new ApiError("Cannot reach the NeuroVista backend. Is it running?", "network", 0));
    xhr.ontimeout = () => reject(new ApiError("The request timed out.", "timeout", 0));
    xhr.onabort = () => reject(new ApiError("Upload cancelled.", "cancelled", 0));
    xhr.onload = () => {
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        return reject(new ApiError(`Request failed (${xhr.status})`, "http_error", xhr.status));
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        const parsed = JobSchema.safeParse(body);
        return parsed.success
          ? resolve(parsed.data)
          : reject(new ApiError("The server returned an unexpected response format.", "bad_response", xhr.status));
      }
      const e = (body as { error?: { message?: string; code?: string; modality?: string } })?.error ?? {};
      reject(new ApiError(e.message ?? `Request failed (${xhr.status})`, e.code ?? "http_error", xhr.status, e.modality));
    };
    xhr.send(fd);
  });
  return { promise, cancel: () => xhr.abort() };
}
