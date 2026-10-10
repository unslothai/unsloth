// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Folders dropped or picked in the browser, flattened to their files. The desktop app links a
// folder instead; the browser has no lasting handle to one, so it can only copy it in once.

/** A folder upload past this many files is cut short with a warning. */
export const MAX_FOLDER_FILES = 2000;

// Same spirit as the linked-folder scan: dependency, VCS and cache trees, and hidden folders.
const SKIPPED_DIRS = new Set([
  "node_modules",
  "bower_components",
  "venv",
  "__pycache__",
]);

function skippedDir(name: string): boolean {
  return name.startsWith(".") || SKIPPED_DIRS.has(name.toLowerCase());
}

// .env files and Terraform state can hold plaintext secrets; lockfiles are noise. Mirrors
// _is_ignored_scan_file in the backend's folder_sync.py.
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
    lower === "pnpm-lock.yaml"
  );
}

export interface FolderFiles {
  files: File[];
  /** Files left out past MAX_FOLDER_FILES. */
  truncated: number;
  /** Whether any of the input was a folder. */
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

/**
 * The files in a drop, walking any folders in it. Must be called inside the drop handler: the
 * browser empties `items` once the event returns, so the entries are taken synchronously here.
 */
export function filesFromDrop(
  dataTransfer: DataTransfer,
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
  );
}

async function walkEntries(roots: FileSystemEntry[]): Promise<FolderFiles> {
  const files: File[] = [];
  let truncated = 0;
  const pending = [...roots];
  while (pending.length > 0) {
    const entry = pending.shift() as FileSystemEntry;
    if (entry.isFile) {
      // A file dropped on its own is the user's explicit choice; inside a folder, skip secrets.
      if (roots.indexOf(entry) < 0 && skippedFile(entry.name)) continue;
      if (files.length >= MAX_FOLDER_FILES) {
        truncated += 1;
        continue;
      }
      try {
        const file = await entryFile(entry as FileSystemFileEntry);
        // An empty file has nothing to index and the upload refuses it (__init__.py and the like).
        if (file.size > 0 || roots.indexOf(entry) >= 0) files.push(file);
      } catch {
        // Unreadable (permissions, vanished mid-walk): leave it out.
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
    // readEntries hands back at most ~100 entries per call; an empty batch ends the folder.
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

/** Files from a `webkitdirectory` input, minus hidden and dependency folders and empty files. */
export function filesFromFolderInput(list: FileList | File[]): FolderFiles {
  const all = Array.from(list).filter((file) => {
    const segments = (file.webkitRelativePath || file.name).split("/");
    // The first segment is the folder the user chose, whatever it is called.
    return (
      file.size > 0 &&
      !segments.slice(1, -1).some(skippedDir) &&
      !skippedFile(file.name)
    );
  });
  const files = all.slice(0, MAX_FOLDER_FILES);
  return { files, truncated: all.length - files.length, hadFolder: true };
}

/** Opens the folder chooser. Call from a click handler (needs user activation). */
export function openFolderPicker(onFiles: (folder: FolderFiles) => void): void {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.webkitdirectory = true;
  input.hidden = true;

  document.body.appendChild(input);
  input.onchange = () => {
    const files = input.files;
    if (files && files.length > 0) {
      onFiles(filesFromFolderInput(files));
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
