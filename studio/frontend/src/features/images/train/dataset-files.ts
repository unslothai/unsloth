// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirrors _DIFFUSION_DATASET_IMAGE_EXTS, _CLIP_EXTS and _TEXT_EXTS in backend/routes/training.py.
export const DATASET_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"];
export const DATASET_CLIP_EXTS = [".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"];
export const DATASET_MEDIA_EXTS = [...DATASET_IMAGE_EXTS, ...DATASET_CLIP_EXTS];
export const DATASET_TEXT_EXTS = [".txt", ".caption", ".jsonl"];
export const DATASET_FILE_ACCEPT = [...DATASET_MEDIA_EXTS, ...DATASET_TEXT_EXTS].join(",");

const ACCEPTED = new Set([...DATASET_MEDIA_EXTS, ...DATASET_TEXT_EXTS]);

const METADATA_SCAN_BYTES = 1024 * 1024;

// starlette caps a multipart body at 1000 file parts, so send in slices.
export const DATASET_UPLOAD_CHUNK = 500;

/** Casefold-equal names stay in one request (the backend only compares within one) and go first. */
export function chunkDatasetUpload(files: File[], maxBytes: number): File[][] {
  const groups = new Map<string, File[]>();
  for (const file of files) {
    const key = destinationName(file).toLowerCase();
    const group = groups.get(key);
    if (group) group.push(file);
    else groups.set(key, [file]);
  }
  // Case-variant sets go first so the backend refusal lands before anything is committed.
  const all = [...groups.values()];
  const chunks: File[][] = [];
  let current: File[] = [];
  let bytes = 0;
  for (const group of [...all.filter((g) => g.length > 1), ...all.filter((g) => g.length === 1)]) {
    const groupBytes = group.reduce((sum, f) => sum + f.size, 0);
    const overCount = current.length + group.length > DATASET_UPLOAD_CHUNK;
    const overBytes = current.length > 0 && bytes + groupBytes > maxBytes;
    if (current.length > 0 && (overCount || overBytes)) {
      chunks.push(current);
      current = [];
      bytes = 0;
    }
    current.push(...group);
    bytes += groupBytes;
  }
  if (current.length > 0) chunks.push(current);
  return chunks;
}

/** Uploads accumulate, so a 413'd chunk leaves earlier chunks on disk. */
export function oversizedChunk(chunks: File[][], maxBytes: number): string | null {
  for (const chunk of chunks) {
    if (chunk.reduce((sum, f) => sum + f.size, 0) > maxBytes) return destinationName(chunk[0]);
  }
  return null;
}

/** A folder pick flattens the tree, so rows keyed on subfolder paths match no caption. */
export async function metadataKeyedOnSubfolders(files: File[]): Promise<string | null> {
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".jsonl")) continue;
    let text: string;
    try {
      text = await file.slice(0, METADATA_SCAN_BYTES).text();
    } catch {
      continue;
    }
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let row: unknown;
      try {
        row = JSON.parse(trimmed);
      } catch {
        continue;
      }
      if (!row || typeof row !== "object") continue;
      const record = row as Record<string, unknown>;
      const key = record.file_name || record.image || record.file;
      if (typeof key === "string" && (key.includes("/") || key.includes("\\"))) {
        return file.name;
      }
    }
  }
  return null;
}

// Matches Python str.strip() (str.isspace() over the BMP), which differs from trim(). NUL is
// stripped separately.
const PY_SPACE_CLASS =
  "\\t\\n\\v\\f\\r\\u001c-\\u001f\\u0020\\u0085\\u00a0\\u1680" +
  "\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_STRIP = new RegExp(`^[${PY_SPACE_CLASS}]+|[${PY_SPACE_CLASS}]+$`, "g");

/** Matches the normalisation in training.py. */
export function destinationName(file: File): string {
  const base = file.name.replace(/\\/g, "/").split("/").pop() ?? "";
  // biome-ignore lint/suspicious/noControlCharactersInRegex: the backend strips nulls here too
  return base.replace(PY_STRIP, "").replace(/\0/g, "");
}

// Mirrors Path(name).suffix.lower(): ".png" has no suffix.
function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 1 ? "" : name.slice(dot).toLowerCase();
}

function displayPath(file: File): string {
  return (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
}

function isHidden(name: string): boolean {
  return name.startsWith(".");
}

// A dataset folder holds a .thumbs cache whose jpegs would re-upload as training images.
function inHiddenPath(file: File): boolean {
  const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  return relative ? relative.split("/").slice(1).some(isHidden) : false;
}

/** Mirrors `_shares_sidecar`: only an extension-case pair of one spelling is exempt. */
function sharesSidecar(other: string, name: string): boolean {
  const otherExt = extensionOf(other);
  if (other === name || !DATASET_MEDIA_EXTS.includes(otherExt)) return false;
  const otherStem = other.slice(0, other.length - otherExt.length);
  const stem = name.slice(0, name.length - extensionOf(name).length);
  if (otherStem.toLowerCase() !== stem.toLowerCase()) return false;
  return otherStem === stem || other.toLowerCase() !== name.toLowerCase();
}

export interface DatasetCollision {
  kind: "name" | "stem";
  first: string;
  second: string;
}

export interface DatasetFileSelection {
  files: File[];
  imageCount: number;
  clipCount: number;
  captionCount: number;
  skipped: number;
  collisions: DatasetCollision[];
}

export function selectDatasetFiles(input: File[]): DatasetFileSelection {
  const files: File[] = [];
  const collisions: DatasetCollision[] = [];
  const seen = new Map<string, string>();
  const mediaStems = new Map<string, Array<{ name: string; path: string }>>();
  let imageCount = 0;
  let clipCount = 0;
  let captionCount = 0;
  let skipped = 0;

  for (const file of input) {
    if (inHiddenPath(file)) continue;
    const dest = destinationName(file);
    const ext = extensionOf(dest);
    if (!dest || dest.includes("..") || !ACCEPTED.has(ext)) {
      skipped += 1;
      continue;
    }
    // Matched exactly: only the backend knows whether the filesystem folds case.
    const path = displayPath(file);
    const previous = seen.get(dest);
    if (previous !== undefined) {
      collisions.push({ kind: "name", first: previous, second: path });
      continue;
    }
    const isImage = DATASET_IMAGE_EXTS.includes(ext);
    const isClip = DATASET_CLIP_EXTS.includes(ext);
    // Every accepted variant is compared, since the exemption is not transitive.
    if (isImage || isClip) {
      const key = dest.slice(0, dest.length - ext.length).toLowerCase();
      const variants = mediaStems.get(key) ?? [];
      const clash = variants.find((v) => sharesSidecar(v.name, dest));
      if (clash !== undefined) {
        collisions.push({ kind: "stem", first: clash.path, second: path });
        continue;
      }
      variants.push({ name: dest, path });
      mediaStems.set(key, variants);
    }
    seen.set(dest, path);
    files.push(file);
    if (isImage) imageCount += 1;
    else if (isClip) clipCount += 1;
    else captionCount += 1;
  }

  return { files, imageCount, clipCount, captionCount, skipped, collisions };
}

/** On a chunked top-up the backend 400 lands after earlier slices were already written. */
export function existingStemClash(files: File[], existing: string[]): DatasetCollision | null {
  for (const file of files) {
    const dest = destinationName(file);
    if (!DATASET_MEDIA_EXTS.includes(extensionOf(dest))) continue;
    const clash = existing.find((name) => sharesSidecar(name, dest));
    if (clash !== undefined) return { kind: "stem", first: clash, second: displayPath(file) };
  }
  return null;
}

export function existingDatasetName(name: string, datasets: { name: string }[]): string | null {
  const folded = name.trim().toLowerCase();
  return datasets.find((d) => d.name.toLowerCase() === folded)?.name ?? null;
}

export function datasetNamesForCreation(info: {
  datasets: { name: string }[];
  dataset_names?: string[];
} | null): { name: string }[] {
  return info?.dataset_names?.map((name) => ({ name })) ?? info?.datasets ?? [];
}

// exact spelling: the upload sends the typed name, and a case variant is a separate folder on a
// case-sensitive filesystem.
export function isDatasetContinuation(name: string, continuationName: string | null): boolean {
  return continuationName !== null && name.trim() === continuationName;
}

export function freeDatasetName(datasets: { name: string }[]): string {
  let name = "my-images";
  for (let i = 2; existingDatasetName(name, datasets); i += 1) {
    name = `my-images-${i}`;
  }
  return name;
}

// readEntries yields at most 100 entries per call and ends with an empty batch.
function readAllEntries(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((resolve, reject) => {
    const all: FileSystemEntry[] = [];
    const next = () =>
      reader.readEntries((batch) => {
        if (batch.length === 0) {
          resolve(all);
          return;
        }
        all.push(...batch);
        next();
      }, reject);
    next();
  });
}

function fileOf(entry: FileSystemFileEntry): Promise<File> {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

async function walkEntry(entry: FileSystemEntry, out: File[], prefix = ""): Promise<void> {
  const path = prefix + entry.name;
  if (entry.isFile) {
    const file = await fileOf(entry as FileSystemFileEntry);
    Object.defineProperty(file, "webkitRelativePath", { value: path, configurable: true });
    out.push(file);
    return;
  }
  if (!entry.isDirectory) return;
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  for (const child of await readAllEntries(reader)) {
    if (isHidden(child.name)) continue;
    await walkEntry(child, out, `${path}/`);
  }
}

export async function filesFromDataTransfer(transfer: DataTransfer): Promise<File[]> {
  const entries: FileSystemEntry[] = [];
  let fileItems = 0;
  for (const item of Array.from(transfer.items ?? [])) {
    if (item.kind !== "file") continue;
    fileItems += 1;
    const entry = item.webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }
  // Must run synchronously: the drag data store is protected once drop yields.
  if (entries.length === 0) return Array.from(transfer.files ?? []);
  if (entries.length < fileItems) {
    throw new Error("Some dropped items could not be read.");
  }
  const out: File[] = [];
  // No partial result: a half-walked folder would upload as if it were the whole dataset.
  for (const entry of entries) await walkEntry(entry, out);
  return out;
}
