


export const TRAINING_DATASET_UPLOAD_EXTENSIONS = [
  ".csv",
  ".jsonl",
  ".json",
  ".parquet",
] as const;
export const TRAINING_DATASET_UPLOAD_ACCEPT =
  TRAINING_DATASET_UPLOAD_EXTENSIONS.join(",");

export const TRAINING_DOCUMENT_REDIRECT_EXTENSIONS = [
  ".pdf",
  ".docx",
  ".txt",
  ".md",
] as const;

const DATASET_EXTENSION_SET = new Set<string>(
  TRAINING_DATASET_UPLOAD_EXTENSIONS,
);
const DOCUMENT_EXTENSION_SET = new Set<string>(
  TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
);
const NATIVE_PATH_SEPARATOR_RE = /[\\/]/;

export type NativeTrainingDatasetDrop =
  | { kind: "dataset"; path: string; filename: string }
  | { kind: "document"; filename: string }
  | { kind: "multiple" }
  | { kind: "unsupported" };

function extensionOf(path: string): string {
  const filename = path.split(NATIVE_PATH_SEPARATOR_RE).pop()?.trim() ?? "";
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

export function isTrainingDatasetUploadPath(path: string): boolean {
  return DATASET_EXTENSION_SET.has(extensionOf(path));
}

export function nativePathFilename(path: string): string {
  const filename = path.split(NATIVE_PATH_SEPARATOR_RE).pop()?.trim() ?? "";
  const sanitized = Array.from(filename, (character) => {
    const code = character.charCodeAt(0);
    return code <= 31 || code === 127 ? " " : character;
  })
    .slice(0, 160)
    .join("")
    .trim();
  return sanitized || "dataset";
}

export function classifyNativeTrainingDatasetDrop(
  paths: readonly string[],
): NativeTrainingDatasetDrop {
  if (paths.length !== 1) {
    return paths.length > 1 ? { kind: "multiple" } : { kind: "unsupported" };
  }
  const path = paths[0];
  if (path === undefined) {
    return { kind: "unsupported" };
  }
  const filename = nativePathFilename(path);
  const extension = extensionOf(path);
  if (DATASET_EXTENSION_SET.has(extension)) {
    return { kind: "dataset", path, filename };
  }
  if (DOCUMENT_EXTENSION_SET.has(extension)) {
    return { kind: "document", filename };
  }
  return { kind: "unsupported" };
}

export function nativeDropPositionHitsBounds(
  position: { x: number; y: number },
  scaleFactor: number,
  bounds: { left: number; right: number; top: number; bottom: number },
): boolean {
  const scale =
    Number.isFinite(scaleFactor) && scaleFactor > 0 ? scaleFactor : 1;
  const x = position.x / scale;
  const y = position.y / scale;
  return (
    x >= bounds.left &&
    x <= bounds.right &&
    y >= bounds.top &&
    y <= bounds.bottom
  );
}
