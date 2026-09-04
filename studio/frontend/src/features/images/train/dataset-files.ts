// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirrors _DIFFUSION_DATASET_IMAGE_EXTS, _CLIP_EXTS and _TEXT_EXTS in
// backend/routes/training.py.
export const DATASET_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"];
// video containers, for the families that train from clips rather than stills.
export const DATASET_CLIP_EXTS = [".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"];
// the trainable unit, whichever kind: everything that pairs with a <stem>.txt caption.
export const DATASET_MEDIA_EXTS = [...DATASET_IMAGE_EXTS, ...DATASET_CLIP_EXTS];
export const DATASET_TEXT_EXTS = [".txt", ".caption", ".jsonl"];
export const DATASET_FILE_ACCEPT = [...DATASET_MEDIA_EXTS, ...DATASET_TEXT_EXTS].join(",");

const ACCEPTED = new Set([...DATASET_MEDIA_EXTS, ...DATASET_TEXT_EXTS]);

// a caption jsonl runs to kilobytes, so scan a prefix rather than decoding the whole upload.
const METADATA_SCAN_BYTES = 1024 * 1024;

// starlette caps a multipart body at 1000 file parts and fastapi does not raise it, so a
// larger selection is sent in slices; the endpoint accumulates uploads into one folder.
export const DATASET_UPLOAD_CHUNK = 500;

/** Slice `files` so no request exceeds the part cap or `maxBytes`, keeping casefold-equal names
 *  together and sending those groups first: the backend can only compare names inside one
 *  request, and splitting a group lets the second request overwrite the first. Such a group
 *  ships whole even over `maxBytes`; `oversizedChunk` catches that. */
export function chunkDatasetUpload(files: File[], maxBytes: number): File[][] {
  const groups = new Map<string, File[]>();
  for (const file of files) {
    const key = destinationName(file).toLowerCase();
    const group = groups.get(key);
    if (group) group.push(file);
    else groups.set(key, [file]);
  }
  // A group of more than one is a case-variant set, which the backend refuses outright on a
  // case-insensitive dataset folder. Send those first so the refusal lands while nothing is
  // committed, rather than behind slices already written.
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

/** The destination name in the first chunk no split can fit under `maxBytes`, or null. Uploads
 *  accumulate, so a chunk the endpoint 413s leaves every chunk before it on disk. */
export function oversizedChunk(chunks: File[][], maxBytes: number): string | null {
  for (const chunk of chunks) {
    if (chunk.reduce((sum, f) => sum + f.size, 0) > maxBytes) return destinationName(chunk[0]);
  }
  return null;
}

/** The first metadata file whose captions are keyed on a subfolder path, or null. A folder pick
 *  flattens the tree, so rows keyed "images/001.png" silently resolve to no caption. */
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

// str.strip() is not trim(): Python keeps U+FEFF and strips U+001C-001F and U+0085. trim()
// read a name ending in U+FEFF as an accepted "cat.png", but the multipart part carries the
// name whole, so the endpoint refused the slice. This class is exactly str.isspace() over
// the BMP; NUL is stripped separately, as it is there.
const PY_SPACE_CLASS =
  "\\t\\n\\v\\f\\r\\u001c-\\u001f\\u0020\\u0085\\u00a0\\u1680" +
  "\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_STRIP = new RegExp(`^[${PY_SPACE_CLASS}]+|[${PY_SPACE_CLASS}]+$`, "g");

/** the name the upload stores this file under, matching the normalisation in training.py. */
export function destinationName(file: File): string {
  const base = file.name.replace(/\\/g, "/").split("/").pop() ?? "";
  // biome-ignore lint/suspicious/noControlCharactersInRegex: the backend strips nulls here too
  return base.replace(PY_STRIP, "").replace(/\0/g, "");
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

/** Whether two item names resolve to one `<stem>.txt` sidecar, as `_shares_sidecar` does: the
 *  stems clash on an exact match, so only an extension-case pair of one spelling is exempt.
 *  Images and clips are compared together, since the sidecar keys on the stem alone. */
function sharesSidecar(other: string, name: string): boolean {
  const otherExt = extensionOf(other);
  if (other === name || !DATASET_MEDIA_EXTS.includes(otherExt)) return false;
  const otherStem = other.slice(0, other.length - otherExt.length);
  const stem = name.slice(0, name.length - extensionOf(name).length);
  if (otherStem.toLowerCase() !== stem.toLowerCase()) return false;
  return otherStem === stem || other.toLowerCase() !== name.toLowerCase();
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
  clipCount: number;
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
  const mediaStems = new Map<string, Array<{ name: string; path: string }>>();
  let imageCount = 0;
  let clipCount = 0;
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
    // A dataset folder is flat, so two tree paths sharing a basename are one destination. Matched
    // exactly: only the backend knows whether the dataset filesystem folds case.
    const path = displayPath(file);
    const previous = seen.get(dest);
    if (previous !== undefined) {
      collisions.push({ kind: "name", first: previous, second: path });
      continue;
    }
    const isImage = DATASET_IMAGE_EXTS.includes(ext);
    const isClip = DATASET_CLIP_EXTS.includes(ext);
    // Two items sharing a stem resolve to one <stem>.txt caption, which the backend refuses. Every
    // accepted variant is kept and compared, since the exemption is not transitive.
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

/** The first selected item the dataset folder already holds under a sidecar-sharing name, or
 *  null. The backend compares each upload against the folder, so on a chunked top-up that 400
 *  lands with the slices before it already written. `existing` is the folder's item names. */
export function existingStemClash(files: File[], existing: string[]): DatasetCollision | null {
  for (const file of files) {
    const dest = destinationName(file);
    if (!DATASET_MEDIA_EXTS.includes(extensionOf(dest))) continue;
    const clash = existing.find((name) => sharesSidecar(name, dest));
    if (clash !== undefined) return { kind: "stem", first: clash, second: displayPath(file) };
  }
  return null;
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
