import { MODALITY_LABEL, type Modality } from "./types";

export const isNifti = (name: string) => /\.nii(\.gz)?$/i.test(name);

/** Client-side checks run before upload. The server repeats and extends them. */
export function validateFile(file: File, modality: Modality, maxMb: number): string | null {
  if (!isNifti(file.name)) {
    return /\.(png|jpe?g|tiff?|bmp|dcm)$/i.test(file.name)
      ? `${MODALITY_LABEL[modality]}: 2D images and DICOM cannot be used as a four-channel MRI volume. Provide a 3D NIfTI (.nii / .nii.gz) per modality.`
      : `${MODALITY_LABEL[modality]}: expected a .nii or .nii.gz file.`;
  }
  if (file.size === 0) return `${MODALITY_LABEL[modality]}: file is empty.`;
  if (file.size > maxMb * 1024 * 1024) return `${MODALITY_LABEL[modality]}: file exceeds the ${maxMb} MB limit.`;
  return null;
}

export interface FileMeta {
  dims: [number, number, number];
  spacing: [number, number, number];
}

export function crossCheck(meta: Partial<Record<Modality, FileMeta>>): string | null {
  const entries = Object.entries(meta) as [Modality, FileMeta][];
  if (entries.length < 2) return null;
  const [m0, ref] = entries[0];
  for (const [m, v] of entries.slice(1)) {
    if (v.dims.join() !== ref.dims.join()) {
      return `${MODALITY_LABEL[m]} is ${v.dims.join("×")} but ${MODALITY_LABEL[m0]} is ${ref.dims.join("×")}. All modalities must share one co-registered grid.`;
    }
    if (v.spacing.some((s, i) => Math.abs(s - ref.spacing[i]) > 1e-3)) {
      return `${MODALITY_LABEL[m]} voxel spacing differs from ${MODALITY_LABEL[m0]}.`;
    }
  }
  return null;
}
