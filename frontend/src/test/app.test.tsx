import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { ApiError, api } from "../lib/api";

const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

const JOB = {
  id: "11111111-1111-1111-1111-111111111111", status: "completed", created: 1_700_000_000, expires: 1_700_003_600,
  viewer: { shape: [4, 4, 2], spacing_mm: [1, 1, 2], layers: ["t1"], default_slice: 1, region_colors: {} },
  request: { tta: false, mc_samples: 0, min_component_voxels: 0 },
  results: {
    regions: {
      WT: { voxels: 10, volume_ml: 0.02, confidence: 0.8, confidence_level: "medium", mean_uncertainty: 0.3 },
      TC: { voxels: 5, volume_ml: 0.01, confidence: 0.7, confidence_level: "low", mean_uncertainty: 0.4 },
      ET: { voxels: 0, volume_ml: 0, confidence: null, confidence_level: null, mean_uncertainty: null },
    },
    classes: {}, scan_confidence: 0.8, scan_confidence_level: "medium", scan_confidence_definition: "mean max-softmax",
    confidence_thresholds: { high: 0.9, medium: 0.7, source: "configured-default (not calibrated)" },
    mean_uncertainty_wt: 0.3, mean_mutual_information_wt: null, uncertainty_method: "predictive entropy", inference_seconds: 1.2, device: "cpu",
    options: { tta: false, mc_samples: 0, min_component_voxels: 0, components_removed: 0 },
    preprocessing: { normalisation: "z-score", modality_order: ["t1", "t1ce", "t2", "flair"] },
    spacing_mm: [1, 1, 2], shape: [4, 4, 2], orientation: "LPS", model: { checkpoint: "best.pt", synthetic: true }, warnings: [],
  },
  downloads: { mask: "/jobs/x/download/mask", preview: "/jobs/x/download/preview", metrics: "/jobs/x/download/metrics" },
};

function mockFetch(routes: Record<string, () => Promise<Response>>) {
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    for (const [k, v] of Object.entries(routes)) if (path.startsWith(k)) return v();
    return json({ error: { code: "nope", message: "not mocked" } }, 404);
  }));
}
beforeEach(() => localStorage.clear());
afterEach(() => vi.unstubAllGlobals());

const at = (path: string) => render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);

describe("navigation", () => {
  it("renders the landing page with disclaimer and a launch link", () => {
    mockFetch({});
    at("/");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/Segment glioma/i);
    expect(screen.getAllByRole("link", { name: /launch workspace/i })[0]).toHaveAttribute("href", "/app");
    expect(screen.getByText(/Research prototype only/)).toBeInTheDocument();
  });
  it("shows every workspace page in the sidebar", async () => {
    mockFetch({});
    at("/app/settings");
    for (const name of ["Dashboard", "New segmentation", "Confidence lab", "Training lab", "Evaluation", "Sessions", "Settings"]) {
      expect(screen.getByRole("link", { name })).toBeInTheDocument();
    }
  });
});

describe("api failure handling", () => {
  it("dashboard shows an error and a retry when the backend is down", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("fail"))));
    at("/app");
    expect(await screen.findByText(/Cannot reach the NeuroVista backend/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });
  it("maps structured errors and rejects malformed payloads", async () => {
    mockFetch({ "/model/info": () => json({ error: { code: "model_not_ready", message: "No checkpoint" } }, 503), "/settings": () => json({ nope: 1 }) });
    await expect(api.modelInfo()).rejects.toMatchObject({ code: "model_not_ready", status: 503 });
    await expect(api.settings()).rejects.toMatchObject({ code: "bad_response" });
    expect(new ApiError("x", "c", 1)).toBeInstanceOf(Error);
  });
});

describe("upload page", () => {
  it("rejects a PNG and keeps start disabled", async () => {
    mockFetch({
      "/settings": () => json({ max_upload_mb: 200, max_inplane_dim: 320, max_depth: 400, artifact_ttl_minutes: 60, retention_policy: "Deleted immediately.", rate_limit_per_minute: 30, training_enabled: true, mc_dropout_samples: 8, device_preference: "cpu", confidence_defaults: { high: 0.9, medium: 0.7 }, datasets: [], disclaimer: "x" }),
      "/model/info": () => json({ loaded: false, reason: "No checkpoint", disclaimer: "x", device: "cpu", cuda_available: false, app_version: "0", modalities: [], capabilities: { tta: true, mc_dropout: false, training: true } }),
    });
    at("/app/upload");
    const input = screen.getByLabelText("T1 file", { selector: "input" });
    await userEvent.upload(input, new File(["x"], "brain.png", { type: "image/png" }), { applyAccept: false });
    expect(await screen.findAllByText(/2D images and DICOM cannot be used/)).not.toHaveLength(0);
    expect(screen.getByRole("button", { name: /start segmentation/i })).toBeDisabled();
  });
});

describe("results page", () => {
  it("renders backend results, region volumes, provenance and download links", async () => {
    mockFetch({ [`/jobs/${JOB.id}/layer`]: () => Promise.resolve(new Response(new Uint8Array(32))), [`/jobs/${JOB.id}`]: () => json(JOB) });
    at(`/app/results/${JOB.id}`);
    expect(await screen.findByText("Segmentation result")).toBeInTheDocument();
    expect(screen.getByText(/Whole tumor/)).toBeInTheDocument();
    expect(screen.getByText(/synthetic phantoms/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Mask \(NIfTI\)/ })).toHaveAttribute("href", expect.stringContaining("/jobs/x/download/mask"));
    expect(screen.getByText(/No accuracy is shown|only.*measurable/i)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByLabelText(/MRI viewer/)).toBeInTheDocument());
  });
  it("shows an expiry message for unknown jobs", async () => {
    mockFetch({ "/jobs/": () => json({ error: { code: "job_not_found", message: "Job not found or expired" } }, 404) });
    at("/app/results/22222222-2222-2222-2222-222222222222");
    expect(await screen.findByText(/Job not found or expired/)).toBeInTheDocument();
  });
});
