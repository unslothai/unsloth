// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// mirrors _DIFFUSION_DATASET_IMAGE_EXTS and _DIFFUSION_DATASET_TEXT_EXTS in backend/routes/training.py.
export const DATASET_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"];
export const DATASET_TEXT_EXTS = [".txt", ".caption", ".jsonl"];
export const DATASET_FILE_ACCEPT = [...DATASET_IMAGE_EXTS, ...DATASET_TEXT_EXTS].join(",");

const ACCEPTED = new Set([...DATASET_IMAGE_EXTS, ...DATASET_TEXT_EXTS]);

// a caption jsonl runs to kilobytes, so scan a prefix rather than decoding the whole upload.
const METADATA_SCAN_BYTES = 1024 * 1024;

// starlette caps a multipart body at 1000 file parts and fastapi does not raise it, so a larger
// selection is sent in slices; the endpoint accumulates repeat uploads into the same folder.
export const DATASET_UPLOAD_CHUNK = 500;

/** slice `files` so no request exceeds the part cap or `maxBytes`, keeping casefold-equal
 *  names together: the backend can only compare names inside one request, and splitting a
 *  group is exactly what lets the second request overwrite the first. Such a group ships
 *  whole even when it is over `maxBytes`; `oversizedChunk` catches that before any upload. */
export function chunkDatasetUpload(files: File[], maxBytes: number): File[][] {
  const groups = new Map<string, File[]>();
  for (const file of files) {
    const key = destinationName(file).toLowerCase();
    const group = groups.get(key);
    if (group) group.push(file);
    else groups.set(key, [file]);
  }
  const chunks: File[][] = [];
  let current: File[] = [];
  let bytes = 0;
  for (const group of groups.values()) {
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

/** the destination name in the first chunk no split can fit under `maxBytes`, or null.
 *
 *  uploads accumulate, so a chunk the endpoint 413s leaves every chunk before it on disk. */
export function oversizedChunk(chunks: File[][], maxBytes: number): string | null {
  for (const chunk of chunks) {
    if (chunk.reduce((sum, f) => sum + f.size, 0) > maxBytes) return destinationName(chunk[0]);
  }
  return null;
}

/** the first metadata file whose captions are keyed on a subfolder path, or null. a folder pick
 *  flattens the tree, so rows keyed "images/001.png" silently resolve to no caption at all. */
export async function metadataKeyedOnSubfolders(files: File[]): Promise<string | null> {
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".jsonl")) continue;
    let text: string;
    try {
      text = await file.slice(0, METADATA_SCAN_BYTES).text();
    } catch {
      continue; // unreadable here is the backend's problem, not a reason to block the upload
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

/** the name the upload stores this file under, matching the normalisation in training.py. */
export function destinationName(file: File): string {
  const base = file.name.replace(/\\/g, "/").split("/").pop() ?? "";
  // biome-ignore lint/suspicious/noControlCharactersInRegex: the backend strips nulls here too
  return base.trim().replace(/\0/g, "");
}

// mirrors Path(name).suffix.lower(): a leading dot starts the stem, so ".png" has no suffix.
function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 1 ? "" : name.slice(dot).toLowerCase();
}

/** where a picked file sat: the folder-relative path when a folder pick supplied one. */
function displayPath(file: File): string {
  return (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
}

function isHidden(name: string): boolean {
  return name.startsWith(".");
}

// a dataset folder holds a .thumbs cache whose jpegs would re-upload as training images.
function inHiddenPath(file: File): boolean {
  const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  // a name typed into a dialog, and segment 0 (the picked folder itself), are the user's choice.
  return relative ? relative.split("/").slice(1).some(isHidden) : false;
}

/** two picked paths the flat dataset folder cannot hold apart; `kind` picks the wording. */
export interface DatasetCollision {
  kind: "name" | "stem";
  first: string;
  second: string;
}

export interface DatasetFileSelection {
  /** the files to upload, in picked order. */
  files: File[];
  imageCount: number;
  captionCount: number;
  /** files left out because the endpoint does not accept their extension. */
  skipped: number;
  collisions: DatasetCollision[];
}

/** filters a picked or dropped batch down to what the dataset endpoint accepts. */
export function selectDatasetFiles(input: File[]): DatasetFileSelection {
  const files: File[] = [];
  const collisions: DatasetCollision[] = [];
  const seen = new Map<string, string>();
  const imageStems = new Map<string, Array<{ name: string; stem: string; path: string }>>();
  let imageCount = 0;
  let captionCount = 0;
  let skipped = 0;

  for (const file of input) {
    if (inHiddenPath(file)) continue;
    // keyed on the stored name, so " cat.png" and "cat.png" are seen as one destination
    const dest = destinationName(file);
    const ext = extensionOf(dest);
    if (!dest || dest.includes("..") || !ACCEPTED.has(ext)) {
      skipped += 1;
      continue;
    }
    // a dataset folder is flat, so two tree paths sharing a basename are one destination.
    // matched exactly: only the backend knows whether the dataset filesystem folds case.
    const path = displayPath(file);
    const previous = seen.get(dest);
    if (previous !== undefined) {
      collisions.push({ kind: "name", first: previous, second: path });
      continue;
    }
    const isImage = DATASET_IMAGE_EXTS.includes(ext);
    // two images sharing a stem resolve to one <stem>.txt caption, which the backend refuses.
    // _shares_sidecar clashes on an exact stem match or a differing casefolded name, so only an
    // extension-case pair of one stem spelling is exempt.
    const stem = isImage ? dest.slice(0, dest.length - ext.length) : null;
    if (stem !== null) {
      const key = stem.toLowerCase();
      const variants = imageStems.get(key) ?? [];
      const clash = variants.find(
        (v) => v.stem === stem || v.name.toLowerCase() !== dest.toLowerCase(),
      );
      if (clash !== undefined) {
        collisions.push({ kind: "stem", first: clash.path, second: path });
        continue;
      }
      variants.push({ name: dest, stem, path });
      imageStems.set(key, variants);
    }
    seen.set(dest, path);
    files.push(file);
    if (isImage) imageCount += 1;
    else captionCount += 1;
  }

  return { files, imageCount, captionCount, skipped, collisions };
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
    // a dropped File has no webkitRelativePath, so stand one in and messages can name the tree.
    Object.defineProperty(file, "webkitRelativePath", { value: path, configurable: true });
    out.push(file);
    return;
  }
  if (!entry.isDirectory) return;
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  for (const child of await readAllEntries(reader)) {
    // a dropped entry carries no relative path, so nested hidden names are filtered here.
    if (isHidden(child.name)) continue;
    await walkEntry(child, out, `${path}/`);
  }
}

/** every file in a drop, descending into any dropped folders; throws if the walk cannot finish. */
export async function filesFromDataTransfer(transfer: DataTransfer): Promise<File[]> {
  const entries: FileSystemEntry[] = [];
  let fileItems = 0;
  for (const item of Array.from(transfer.items ?? [])) {
    if (item.kind !== "file") continue;
    fileItems += 1;
    const entry = item.webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }
  // still synchronous here, so the drag data store is readable; it is protected once drop yields.
  if (entries.length === 0) return Array.from(transfer.files ?? []);
  if (entries.length < fileItems) {
    // some items resolved and some did not, so the walk would quietly upload part of the drop.
    throw new Error("Some dropped items could not be read.");
  }
  const out: File[] = [];
  // no partial result: a half-walked folder would upload as if it were the whole dataset.
  for (const entry of entries) await walkEntry(entry, out);
  return out;
}
