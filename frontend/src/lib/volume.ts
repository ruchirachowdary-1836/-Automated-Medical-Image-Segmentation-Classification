/**
 * Volume geometry + slice rendering. Layers arrive from the API in RAS order, shape [X, Y, Z], C-order:
 * index = (x * Y + y) * Z + z. Display follows radiological convention: patient-right on screen-left,
 * anterior up (axial/coronal) or screen-left (sagittal), superior up.
 */
export type Plane = "axial" | "coronal" | "sagittal";
export type Vec3 = [number, number, number];
export type ViewMode = "overlay" | "original" | "confidence" | "uncertainty" | "mask";

export interface Volume {
  shape: Vec3;
  spacing: Vec3;
  layers: Record<string, Uint8Array>;
}

export const REGION_RGB: Record<number, Vec3> = { 1: [59, 130, 246], 2: [245, 197, 66], 3: [239, 68, 68] };
export const REGION_INFO: { id: 1 | 2 | 3; key: string; label: string; description: string }[] = [
  { id: 1, key: "necrotic_non_enhancing_core", label: "Necrotic / non-enhancing core", description: "BraTS label 1 (NCR/NET)" },
  { id: 2, key: "edema", label: "Peritumoral edema", description: "BraTS label 2 (ED)" },
  { id: 3, key: "enhancing_tumor", label: "Enhancing tumor", description: "BraTS label 4, remapped to class 3 (ET)" },
];

export function planeSize(shape: Vec3, plane: Plane): { w: number; h: number } {
  const [X, Y, Z] = shape;
  return plane === "axial" ? { w: X, h: Y } : plane === "coronal" ? { w: X, h: Z } : { w: Y, h: Z };
}

/** Physical aspect ratio (width / height) so anatomy is never stretched. */
export function planeAspect(shape: Vec3, spacing: Vec3, plane: Plane): number {
  const { w, h } = planeSize(shape, plane);
  const [sx, sy, sz] = spacing;
  const [pw, ph] = plane === "axial" ? [sx, sy] : plane === "coronal" ? [sx, sz] : [sy, sz];
  return (w * pw) / (h * ph);
}

export function planeMax(shape: Vec3, plane: Plane): number {
  return plane === "axial" ? shape[2] : plane === "coronal" ? shape[1] : shape[0];
}

/** Pixel (c, r) in a plane image -> voxel [x, y, z] for the current crosshair. */
export function pixelToVoxel(shape: Vec3, plane: Plane, c: number, r: number, cur: Vec3): Vec3 {
  const [X, Y, Z] = shape;
  if (plane === "axial") return [X - 1 - c, Y - 1 - r, cur[2]];
  if (plane === "coronal") return [X - 1 - c, cur[1], Z - 1 - r];
  return [cur[0], Y - 1 - c, Z - 1 - r];
}

/** Voxel -> pixel in a plane image. */
export function voxelToPixel(shape: Vec3, plane: Plane, v: Vec3): { c: number; r: number } {
  const [X, Y, Z] = shape;
  if (plane === "axial") return { c: X - 1 - v[0], r: Y - 1 - v[1] };
  if (plane === "coronal") return { c: X - 1 - v[0], r: Z - 1 - v[2] };
  return { c: Y - 1 - v[1], r: Z - 1 - v[2] };
}

const STOPS: Record<"viridis" | "inferno", Vec3[]> = {
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  inferno: [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]],
};

export function ramp(name: "viridis" | "inferno", t: number): Vec3 {
  const s = STOPS[name];
  const x = Math.min(Math.max(t, 0), 1) * (s.length - 1);
  const i = Math.min(Math.floor(x), s.length - 2);
  const f = x - i;
  return [0, 1, 2].map((k) => s[i][k] + (s[i + 1][k] - s[i][k]) * f) as Vec3;
}

export interface RenderOpts {
  modality: string;
  mode: ViewMode;
  opacity: number; // 0..1
  visible: Record<number, boolean>;
  boundaries: boolean;
}

export interface Raster {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export function renderPlane(vol: Volume, plane: Plane, cur: Vec3, o: RenderOpts): Raster {
  const { w, h } = planeSize(vol.shape, plane);
  const [, Y, Z] = vol.shape;
  const base = vol.layers[o.modality];
  const labels = vol.layers.labels;
  const conf = vol.layers.confidence;
  const unc = vol.layers.uncertainty;
  const data = new Uint8ClampedArray(w * h * 4);
  const at = (x: number, y: number, z: number) => (x * Y + y) * Z + z;
  const label = (x: number, y: number, z: number) => (x < 0 || y < 0 || z < 0 || x >= vol.shape[0] || y >= Y || z >= Z ? 0 : labels[at(x, y, z)]);
  const a = o.opacity;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const [x, y, z] = pixelToVoxel(vol.shape, plane, c, r, cur);
      const i = at(x, y, z);
      const g = o.mode === "mask" ? 0 : base ? base[i] : 0;
      let R = g, G = g, B = g;
      const lab = labels[i];
      if (o.mode === "overlay" || o.mode === "mask") {
        if (lab > 0 && o.visible[lab]) {
          const col = REGION_RGB[lab];
          R = R * (1 - a) + col[0] * a;
          G = G * (1 - a) + col[1] * a;
          B = B * (1 - a) + col[2] * a;
        }
        if (o.boundaries && lab > 0 && o.visible[lab]) {
          const edge =
            plane === "axial"
              ? label(x - 1, y, z) !== lab || label(x + 1, y, z) !== lab || label(x, y - 1, z) !== lab || label(x, y + 1, z) !== lab
              : plane === "coronal"
                ? label(x - 1, y, z) !== lab || label(x + 1, y, z) !== lab || label(x, y, z - 1) !== lab || label(x, y, z + 1) !== lab
                : label(x, y - 1, z) !== lab || label(x, y + 1, z) !== lab || label(x, y, z - 1) !== lab || label(x, y, z + 1) !== lab;
          if (edge) {
            const col = REGION_RGB[lab];
            R = col[0]; G = col[1]; B = col[2];
          }
        }
      } else if (o.mode === "confidence" || o.mode === "uncertainty") {
        const src = o.mode === "confidence" ? conf : unc;
        const v = src ? src[i] : 0;
        if (conf && conf[i] > 0) {
          const col = ramp(o.mode === "confidence" ? "viridis" : "inferno", v / 255);
          R = R * (1 - a) + col[0] * a;
          G = G * (1 - a) + col[1] * a;
          B = B * (1 - a) + col[2] * a;
        }
      }
      const p = (r * w + c) * 4;
      data[p] = R; data[p + 1] = G; data[p + 2] = B; data[p + 3] = 255;
    }
  }
  return { width: w, height: h, data };
}

/** Histogram (0..255 bins collapsed to `bins`) of a layer restricted to predicted-tumor voxels. */
export function tumorHistogram(vol: Volume, layer: "confidence" | "uncertainty", bins = 20): number[] {
  const out = new Array<number>(bins).fill(0);
  const labels = vol.layers.labels;
  const src = vol.layers[layer];
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] > 0) out[Math.min(bins - 1, Math.floor((src[i] / 256) * bins))]++;
  }
  return out;
}

export function sliceWithTumor(vol: Volume): number {
  const [X, Y, Z] = vol.shape;
  let best = Math.floor(Z / 2), bestN = 0;
  for (let z = 0; z < Z; z++) {
    let n = 0;
    for (let x = 0; x < X; x++) for (let y = 0; y < Y; y++) if (vol.layers.labels[(x * Y + y) * Z + z] > 0) n++;
    if (n > bestN) { bestN = n; best = z; }
  }
  return best;
}
