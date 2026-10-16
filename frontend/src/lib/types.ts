import { z } from "zod";

const num = z.number().nullable();

export const RegionSchema = z.object({
  voxels: z.number(),
  volume_ml: z.number(),
  confidence: num,
  confidence_level: z.string().nullable(),
  mean_uncertainty: num,
});

export const JobSchema = z.object({
  id: z.string(),
  status: z.string(),
  created: z.number(),
  expires: z.number().nullable(),
  viewer: z.object({
    shape: z.tuple([z.number(), z.number(), z.number()]),
    spacing_mm: z.tuple([z.number(), z.number(), z.number()]),
    layers: z.array(z.string()),
    default_slice: z.number(),
    region_colors: z.record(z.string()),
  }),
  request: z.object({ tta: z.boolean(), mc_samples: z.number(), min_component_voxels: z.number() }).passthrough(),
  results: z.object({
    regions: z.object({ WT: RegionSchema, TC: RegionSchema, ET: RegionSchema }),
    classes: z.record(z.object({ voxels: z.number(), volume_ml: z.number() })),
    scan_confidence: num,
    scan_confidence_level: z.string().nullable(),
    scan_confidence_definition: z.string(),
    confidence_thresholds: z.object({ high: z.number(), medium: z.number(), source: z.string() }),
    mean_uncertainty_wt: num,
    mean_mutual_information_wt: num,
    uncertainty_method: z.string(),
    inference_seconds: z.number(),
    device: z.string(),
    options: z.object({ tta: z.boolean(), mc_samples: z.number(), min_component_voxels: z.number(), components_removed: z.number() }),
    preprocessing: z.object({ normalisation: z.string(), modality_order: z.array(z.string()) }).passthrough(),
    spacing_mm: z.array(z.number()),
    shape: z.array(z.number()),
    orientation: z.string(),
    model: z.object({ checkpoint: z.string() }).passthrough(),
    warnings: z.array(z.string()),
  }),
  downloads: z.object({ mask: z.string(), preview: z.string(), metrics: z.string() }),
});
export type Job = z.infer<typeof JobSchema>;

export const JobSummarySchema = z.object({
  id: z.string(),
  created: z.number(),
  expires: z.number().nullable(),
  status: z.string(),
  model: z.string(),
  wt_volume_ml: z.number(),
  scan_confidence: num,
  shape: z.array(z.number()),
  synthetic_model: z.boolean(),
  options: z.object({ tta: z.boolean(), mc_samples: z.number() }).passthrough(),
});
export type JobSummary = z.infer<typeof JobSummarySchema>;

export const ModelInfoSchema = z
  .object({
    loaded: z.boolean(),
    reason: z.string().nullable().optional(),
    disclaimer: z.string(),
    device: z.string(),
    cuda_available: z.boolean(),
    app_version: z.string(),
    modalities: z.array(z.string()),
    capabilities: z.object({ tta: z.boolean(), mc_dropout: z.boolean(), training: z.boolean() }).passthrough(),
    checkpoint: z.string().optional(),
    params: z.number().optional(),
    meta: z.record(z.unknown()).optional(),
    confidence_thresholds: z.object({ high: z.number(), medium: z.number(), source: z.string() }).optional(),
    warning: z.string().nullable().optional(),
  })
  .passthrough();
export type ModelInfo = z.infer<typeof ModelInfoSchema>;

export const DatasetSchema = z.object({
  name: z.string(),
  has_raw: z.boolean(),
  has_processed: z.boolean(),
  splits: z.record(z.number()).nullable(),
  seed: z.number().nullable(),
});
export type Dataset = z.infer<typeof DatasetSchema>;

export const SettingsSchema = z
  .object({
    max_upload_mb: z.number(),
    max_inplane_dim: z.number(),
    max_depth: z.number(),
    artifact_ttl_minutes: z.number(),
    retention_policy: z.string(),
    rate_limit_per_minute: z.number(),
    training_enabled: z.boolean(),
    mc_dropout_samples: z.number(),
    device_preference: z.string(),
    confidence_defaults: z.object({ high: z.number(), medium: z.number() }),
    datasets: z.array(DatasetSchema),
    disclaimer: z.string(),
  })
  .passthrough();
export type AppSettings = z.infer<typeof SettingsSchema>;

export const HistoryEntrySchema = z.object({
  epoch: z.number(),
  train_loss: z.number(),
  val_loss: z.number(),
  dice: z.record(z.number()),
  mean_dice: z.number(),
  lr: z.number(),
  seconds: z.number(),
});
export type HistoryEntry = z.infer<typeof HistoryEntrySchema>;

const Agg = z.object({ mean: num, std: num, n: z.number() });
export const EvalSchema = z
  .object({
    n_patients: z.number(),
    split: z.string().optional(),
    cases: z.array(
      z.object({
        patient: z.string(),
        regions: z.record(z.record(z.number().nullable())),
        inference_seconds: z.number(),
      }),
    ),
    aggregate: z.record(z.record(Agg)),
    confusion_matrix: z.array(z.array(z.number())),
    calibration: z
      .object({
        ece: num,
        bins: z.array(z.object({ lo: z.number(), hi: z.number(), count: z.number(), confidence: num, accuracy: num })),
        brier_top1: num,
        voxels_sampled: z.number(),
        thresholds: z.object({ high: z.number(), medium: z.number() }).passthrough().nullable(),
      })
      .passthrough(),
    uncertainty_vs_error: z
      .object({ auroc: z.number(), mean_uncertainty_correct: z.number(), mean_uncertainty_error: z.number(), n_error_voxels: z.number() })
      .nullable(),
    options: z.object({ tta: z.boolean(), mc_samples: z.number(), min_component_voxels: z.number() }),
  })
  .passthrough();
export type EvalResult = z.infer<typeof EvalSchema>;

export const RunSchema = z.object({
  id: z.string(),
  config: z.record(z.unknown()),
  status: z
    .object({
      state: z.string().optional(),
      epoch: z.number().optional(),
      total_epochs: z.number().optional(),
      best_mean_dice: z.number().optional(),
      best_epoch: z.number().optional(),
      device: z.string().optional(),
      error: z.string().optional(),
      params: z.number().optional(),
      train_slices: z.number().optional(),
      val_slices: z.number().optional(),
    })
    .passthrough(),
  history: z.array(HistoryEntrySchema),
  log: z.string(),
  evals: z.record(z.unknown()),
  has_best: z.boolean(),
});
export type Run = z.infer<typeof RunSchema>;

export const RunSummarySchema = z.object({
  id: z.string(),
  name: z.string().nullable(),
  preset: z.string().nullable(),
  model: z.string().nullable(),
  loss: z.string().nullable(),
  state: z.string().nullable(),
  epochs_done: z.number(),
  best_mean_dice: z.number().nullable().optional(),
  evals: z.array(z.string()),
  synthetic: z.boolean(),
  created: z.number().nullable().optional(),
});
export type RunSummary = z.infer<typeof RunSummarySchema>;

export const ExperimentSchema = z.object({
  id: z.string(),
  title: z.string(),
  kind: z.string(),
  note: z.string().optional(),
  status: z.string(),
  runs: z.array(z.string()),
});
export type Experiment = z.infer<typeof ExperimentSchema>;

export const PresetSchema = z.object({ id: z.string(), title: z.string(), kind: z.string(), overrides: z.record(z.unknown()).optional() });

export const TrainingDefaultsSchema = z.object({
  defaults: z.record(z.unknown()),
  presets: z.array(PresetSchema),
  datasets: z.array(DatasetSchema),
  enabled: z.boolean(),
  cuda_available: z.boolean(),
});
export type TrainingDefaults = z.infer<typeof TrainingDefaultsSchema>;

export const DashboardSchema = z.object({
  jobs_retained: z.number(),
  recent_jobs: z.array(JobSummarySchema),
  model: ModelInfoSchema,
  runs: z.number(),
  latest_run: RunSchema.nullable(),
  datasets: z.array(DatasetSchema),
  ready: z.boolean(),
});
export type Dashboard = z.infer<typeof DashboardSchema>;

export const GroundTruthEvalSchema = z.object({
  job_id: z.string(),
  regions: z.record(z.record(z.number().nullable())),
  source: z.string(),
  note: z.string(),
});
export type GroundTruthEval = z.infer<typeof GroundTruthEvalSchema>;

export const MODALITIES = ["t1", "t1ce", "t2", "flair"] as const;
export type Modality = (typeof MODALITIES)[number];
export const MODALITY_LABEL: Record<Modality, string> = { t1: "T1", t1ce: "T1ce", t2: "T2", flair: "FLAIR" };
