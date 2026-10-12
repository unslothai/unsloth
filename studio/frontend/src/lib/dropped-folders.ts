// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Browser folder drops and picks, flattened to files (the desktop links folders instead).

export const MAX_FOLDER_FILES = 2000;

const SKIPPED_DIRS = new Set([
  "node_modules",
  "bower_components",
  "venv",
  "__pycache__",
]);

function skippedDir(name: string): boolean {
  return name.startsWith(".") || SKIPPED_DIRS.has(name.toLowerCase());
}

// Mirrors _is_ignored_scan_file in folder_sync.py: secrets and lockfiles.
function skippedFile(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    lower.startsWith(".") ||
    lower.endsWith(".env") ||
    lower.endsWith(".lock") ||
    lower.endsWith(".lockb") ||
    lower.endsWith(".tfstate") ||
    lower === "package-lock.json" ||
    lower === "npm-shrinkwrap.json" ||
    lower === "pnpm-lock.yaml" ||
    lower === "pnpm-lock.yml"
  );
}

export interface FolderFiles {
  files: File[];
  /** Files left out past MAX_FOLDER_FILES; a dropped folder stops at the first, so reports 1. */
  truncated: number;
  hadFolder: boolean;
}

function readBatch(
  reader: FileSystemDirectoryReader,
): Promise<FileSystemEntry[]> {
  return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
}

function entryFile(entry: FileSystemFileEntry): Promise<File> {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

/** Call inside the drop handler: the browser empties `items` once the event returns. */
export function filesFromDrop(
  dataTransfer: DataTransfer,
  /** Inside a folder, files this rejects are left out before the cap counts them. */
  keep: (name: string) => boolean = () => true,
): Promise<FolderFiles> {
  const entries = Array.from(dataTransfer.items ?? [])
    .filter((item) => item.kind === "file")
    .map((item) => item.webkitGetAsEntry?.() ?? null);
  const plain = Array.from(dataTransfer.files ?? []);
  if (!entries.some((entry) => entry?.isDirectory)) {
    return Promise.resolve({ files: plain, truncated: 0, hadFolder: false });
  }
  return walkEntries(
    entries.filter((entry): entry is FileSystemEntry => entry !== null),
    keep,
  );
}

async function walkEntries(
  roots: FileSystemEntry[],
  keep: (name: string) => boolean,
): Promise<FolderFiles> {
  const files: File[] = [];
  let truncated = 0;
  const pending = [...roots];
  while (pending.length > 0) {
    const entry = pending.shift() as FileSystemEntry;
    if (entry.isFile) {
      const root = roots.indexOf(entry) >= 0;
      if (!root && (skippedFile(entry.name) || !keep(entry.name))) continue;
      if (files.length >= MAX_FOLDER_FILES) {
        truncated = 1;
        break;
      }
      try {
        const file = await entryFile(entry as FileSystemFileEntry);
        // The upload refuses empty files.
        if (file.size > 0 || root) files.push(file);
      } catch {
      }
      continue;
    }
    if (
      !entry.isDirectory ||
      (roots.indexOf(entry) < 0 && skippedDir(entry.name))
    ) {
      continue;
    }
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    // readEntries returns ~100 entries per call; an empty batch ends the folder.
    for (;;) {
      let batch: FileSystemEntry[];
      try {
        batch = await readBatch(reader);
      } catch {
        break;
      }
      if (batch.length === 0) break;
      pending.push(...batch);
    }
  }
  return { files, truncated, hadFolder: true };
}

export function filesFromFolderInput(
  list: FileList | File[],
  /** Files this rejects are left out before the cap counts them. */
  keep: (name: string) => boolean = () => true,
): FolderFiles {
  const all = Array.from(list).filter((file) => {
    const segments = (file.webkitRelativePath || file.name).split("/");
    // Segment 0 is the chosen folder itself.
    return (
      file.size > 0 &&
      !segments.slice(1, -1).some(skippedDir) &&
      !skippedFile(file.name) &&
      keep(file.name)
    );
  });
  const files = all.slice(0, MAX_FOLDER_FILES);
  return { files, truncated: all.length - files.length, hadFolder: true };
}

/** Call from a click handler (needs user activation). */
export function openFolderPicker(
  onFiles: (folder: FolderFiles) => void,
  keep?: (name: string) => boolean,
): void {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.webkitdirectory = true;
  input.hidden = true;

  document.body.appendChild(input);
  input.onchange = () => {
    const files = input.files;
    if (files && files.length > 0) {
      onFiles(filesFromFolderInput(files, keep));
    }
    document.body.removeChild(input);
  };
  input.oncancel = () => {
    if (!input.files || input.files.length === 0) {
      document.body.removeChild(input);
    }
  };
  input.click();
}
