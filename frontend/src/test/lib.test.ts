import { describe, expect, it } from "vitest";
import { parseNiftiBuffer } from "../lib/nifti";
import { crossCheck, validateFile } from "../lib/validate";
import { pixelToVoxel, planeAspect, renderPlane, tumorHistogram, voxelToPixel, type Volume } from "../lib/volume";

const file = (name: string, size = 10) => new File([new Uint8Array(size)], name);

describe("upload validation", () => {
  it("accepts .nii and .nii.gz", () => {
    expect(validateFile(file("a.nii.gz"), "t1", 200)).toBeNull();
    expect(validateFile(file("a.nii"), "t2", 200)).toBeNull();
  });
  it("rejects 2D images with an explanation", () => {
    expect(validateFile(file("scan.png"), "flair", 200)).toMatch(/2D images/);
    expect(validateFile(file("scan.jpg"), "t1", 200)).toMatch(/NIfTI/);
  });
  it("rejects wrong extension, empty and oversized files", () => {
    expect(validateFile(file("a.txt"), "t1", 200)).toMatch(/\.nii/);
    expect(validateFile(new File([], "a.nii"), "t1", 200)).toMatch(/empty/);
    expect(validateFile(file("a.nii", 3 * 1024 * 1024), "t1", 2)).toMatch(/limit/);
  });
  it("detects mismatched grids", () => {
    const a = { dims: [64, 64, 20], spacing: [1, 1, 2] } as never;
    const b = { dims: [64, 64, 21], spacing: [1, 1, 2] } as never;
    expect(crossCheck({ t1: a, t2: a })).toBeNull();
    expect(crossCheck({ t1: a, t2: b })).toMatch(/co-registered/);
  });
});

function makeNifti(dims: [number, number, number], fill: (x: number, y: number, z: number) => number) {
  const [X, Y, Z] = dims;
  const buf = new ArrayBuffer(352 + X * Y * Z * 2);
  const v = new DataView(buf);
  v.setInt32(0, 348, true);
  v.setInt16(40, 3, true);
  dims.forEach((d, i) => v.setInt16(42 + 2 * i, d, true));
  v.setInt16(70, 4, true); // int16
  [1.5, 1.5, 3].forEach((s, i) => v.setFloat32(80 + 4 * i, s, true));
  v.setFloat32(108, 352, true);
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) v.setInt16(352 + (x + y * X + z * X * Y) * 2, fill(x, y, z), true);
  return buf;
}

describe("nifti header", () => {
  it("reads dims, spacing and a mid-slice preview", () => {
    const info = parseNiftiBuffer(makeNifti([8, 6, 4], (x) => x * 10 + 1));
    expect(info.dims).toEqual([8, 6, 4]);
    expect(info.spacing).toEqual([1.5, 1.5, 3]);
    expect(info.preview?.width).toBe(8);
  });
  it("rejects non-NIfTI data", () => {
    expect(() => parseNiftiBuffer(new ArrayBuffer(400))).toThrow(/NIfTI/);
    expect(() => parseNiftiBuffer(new ArrayBuffer(10))).toThrow(/small/);
  });
});

function vol(): Volume {
  const shape: [number, number, number] = [4, 5, 3];
  const n = 60;
  const labels = new Uint8Array(n);
  const base = new Uint8Array(n).fill(100);
  const conf = new Uint8Array(n).fill(200);
  const idx = (x: number, y: number, z: number) => (x * 5 + y) * 3 + z;
  labels[idx(1, 2, 1)] = 3;
  return { shape, spacing: [1, 1, 2], layers: { flair: base, t1: base, t1ce: base, t2: base, labels, confidence: conf, uncertainty: new Uint8Array(n).fill(50) } };
}

describe("viewer geometry and rendering", () => {
  it("maps pixels <-> voxels consistently in every plane", () => {
    const v = vol();
    for (const p of ["axial", "coronal", "sagittal"] as const) {
      const cur: [number, number, number] = [1, 2, 1];
      const px = voxelToPixel(v.shape, p, cur);
      expect(pixelToVoxel(v.shape, p, px.c, px.r, cur)).toEqual(cur);
    }
  });
  it("preserves physical aspect ratio using voxel spacing", () => {
    const v = vol();
    expect(planeAspect(v.shape, v.spacing, "axial")).toBeCloseTo(4 / 5);
    expect(planeAspect(v.shape, v.spacing, "coronal")).toBeCloseTo(4 / (3 * 2));
  });
  it("colours the enhancing-tumor voxel red and honours region visibility", () => {
    const v = vol();
    const o = { modality: "flair", mode: "overlay" as const, opacity: 1, visible: { 1: true, 2: true, 3: true }, boundaries: false };
    const r = renderPlane(v, "axial", [1, 2, 1], o);
    const p = voxelToPixel(v.shape, "axial", [1, 2, 1]);
    const i = (p.r * r.width + p.c) * 4;
    expect([r.data[i], r.data[i + 1], r.data[i + 2]]).toEqual([239, 68, 68]);
    const hidden = renderPlane(v, "axial", [1, 2, 1], { ...o, visible: { 1: true, 2: true, 3: false } });
    expect([hidden.data[i], hidden.data[i + 1], hidden.data[i + 2]]).toEqual([100, 100, 100]);
  });
  it("computes histograms only over predicted tumor voxels", () => {
    const h = tumorHistogram(vol(), "confidence", 10);
    expect(h.reduce((a, b) => a + b, 0)).toBe(1);
    expect(h[7]).toBe(1);
  });
});
