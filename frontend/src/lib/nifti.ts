/** Minimal client-side NIfTI-1 header reader + mid-slice preview, used only for upload validation. */

export interface NiftiInfo {
  dims: [number, number, number];
  spacing: [number, number, number];
  datatype: number;
  preview: { width: number; height: number; data: Uint8ClampedArray } | null;
}

const READERS: Record<number, [number, (v: DataView, o: number, le: boolean) => number]> = {
  2: [1, (v, o) => v.getUint8(o)],
  4: [2, (v, o, le) => v.getInt16(o, le)],
  8: [4, (v, o, le) => v.getInt32(o, le)],
  16: [4, (v, o, le) => v.getFloat32(o, le)],
  64: [8, (v, o, le) => v.getFloat64(o, le)],
  256: [1, (v, o) => v.getInt8(o)],
  512: [2, (v, o, le) => v.getUint16(o, le)],
  768: [4, (v, o, le) => v.getUint32(o, le)],
};

export class NiftiParseError extends Error {}

async function readAll(file: File): Promise<ArrayBuffer> {
  const gz = file.name.toLowerCase().endsWith(".gz");
  if (!gz) return file.arrayBuffer();
  if (typeof DecompressionStream === "undefined") throw new NiftiParseError("This browser cannot preview .nii.gz files");
  const stream = file.stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export function parseNiftiBuffer(buf: ArrayBuffer, withPreview = true): NiftiInfo {
  if (buf.byteLength < 348) throw new NiftiParseError("File is too small to be a NIfTI volume");
  const v = new DataView(buf);
  let le = true;
  if (v.getInt32(0, true) !== 348) {
    if (v.getInt32(0, false) !== 348) throw new NiftiParseError("Not a NIfTI-1 file (invalid header)");
    le = false;
  }
  const ndim = v.getInt16(40, le);
  const d = [1, 2, 3, 4].map((i) => v.getInt16(40 + 2 * i, le));
  if (ndim < 3 || ndim > 4 || (ndim === 4 && d[3] !== 1)) throw new NiftiParseError(`Expected a 3D volume, found ${ndim}D`);
  const dims: [number, number, number] = [d[0], d[1], d[2]];
  if (dims.some((x) => x < 1)) throw new NiftiParseError("Invalid dimensions in header");
  const spacing: [number, number, number] = [1, 2, 3].map((i) => Math.abs(v.getFloat32(76 + 4 * i, le))) as [number, number, number];
  const datatype = v.getInt16(70, le);
  const offset = Math.max(Math.round(v.getFloat32(108, le)), 348);
  const rd = READERS[datatype];
  let preview: NiftiInfo["preview"] = null;
  if (withPreview && rd) {
    const [size, read] = rd;
    const [X, Y, Z] = dims;
    if (offset + X * Y * Z * size <= buf.byteLength) {
      const z = Math.floor(Z / 2);
      const vals = new Float64Array(X * Y);
      let lo = Infinity;
      let hi = -Infinity;
      for (let y = 0; y < Y; y++)
        for (let x = 0; x < X; x++) {
          const val = read(v, offset + (x + y * X + z * X * Y) * size, le);
          vals[x + y * X] = val;
          if (val > 0) {
            lo = Math.min(lo, val);
            hi = Math.max(hi, val);
          }
        }
      const data = new Uint8ClampedArray(X * Y * 4);
      const range = hi > lo ? hi - lo : 1;
      for (let y = 0; y < Y; y++)
        for (let x = 0; x < X; x++) {
          const val = vals[x + y * X];
          const g = val > 0 ? ((val - lo) / range) * 255 : 0;
          const o = ((Y - 1 - y) * X + x) * 4; // flip so anterior is up for the common x-fastest layout
          data[o] = data[o + 1] = data[o + 2] = g;
          data[o + 3] = 255;
        }
      preview = { width: X, height: Y, data };
    }
  }
  return { dims, spacing, datatype, preview };
}

export async function parseNifti(file: File): Promise<NiftiInfo> {
  return parseNiftiBuffer(await readAll(file));
}
