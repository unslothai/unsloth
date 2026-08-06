


import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { File02Icon, FolderAddIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { XIcon } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import {
  invalidateProjectSources,
  uploadProjectDocument,
} from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT } from "../types/rag";
import { resolveVisionOverrides } from "./vision-overrides";

/** A file picked before the project exists, held until create commits. */
export interface StagedSource {
  id: string;
  file: File;
}

// Client-side dedup key; backend dedups authoritatively by content hash.
function fileSignature(file: File): string {
  return `${file.name}|${file.size}|${file.lastModified}`;
}

function formatSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const shown =
    value >= 10 || unit === 0
      ? String(Math.round(value))
      : value.toFixed(1).replace(/\.0$/, "");
  return `${shown} ${units[unit]}`;
}

const ACCEPTED_EXTS = new Set(
  RAG_UPLOAD_ACCEPT.split(",").map((ext) => ext.trim().toLowerCase()),
);

// `accept` only filters the picker, so a drop can carry anything. A folder
// arrives as an extension-less entry, which this rejects along with the types
// the backend would 400 on.
function isSupported(file: File): boolean {
  const dot = file.name.lastIndexOf(".");
  if (dot <= 0) return false;
  return ACCEPTED_EXTS.has(file.name.slice(dot).toLowerCase());
}

/** Merge a selection into the staged list. Returns the names it would not take,
 * so the caller can say so once instead of dropping them silently. */
function addStagedSources(
  staged: StagedSource[],
  incoming: FileList | File[],
): { next: StagedSource[]; unsupported: string[]; duplicates: string[] } {
  const seen = new Set(staged.map((entry) => fileSignature(entry.file)));
  const next = [...staged];
  const unsupported: string[] = [];
  const duplicates: string[] = [];
  for (const file of Array.from(incoming)) {
    if (!isSupported(file)) {
      unsupported.push(file.name);
      continue;
    }
    const signature = fileSignature(file);
    if (seen.has(signature)) {
      duplicates.push(file.name);
      continue;
    }
    seen.add(signature);
    next.push({
      id: `staged_${Math.random().toString(36).slice(2)}`,
      file,
    });
  }
  return { next, unsupported, duplicates };
}

// Projects created with staged files, so the landing can open on Sources.
const projectsWithPendingSources = new Set<string>();

function markProjectSourcesPending(projectId: string): void {
  projectsWithPendingSources.add(projectId);
}

/** Whether this project was just created with staged sources. Read-only, so it
 * is safe in a render pass that React may replay. */
export function hasProjectSourcesPending(projectId: string): boolean {
  return projectsWithPendingSources.has(projectId);
}

/** Drop the marker once the landing has committed. */
export function consumeProjectSourcesPending(projectId: string): void {
  projectsWithPendingSources.delete(projectId);
}

/** Upload staged files to a new project. Indexing runs in the background; a
 * per-file failure toasts and never blocks project creation. */
export async function uploadStagedSources(
  projectId: string,
  staged: StagedSource[],
): Promise<void> {
  if (staged.length === 0) return;
  invalidateProjectSources(projectId);
  markProjectSourcesPending(projectId);
  const { ocr, caption } = resolveVisionOverrides();
  const documentIds = new Set<string>();
  const merged: string[] = [];
  for (const { file } of staged) {
    try {
      const result = await uploadProjectDocument(projectId, file, ocr, caption);
      // Same bytes under another name: the backend hashes content, so this is
      // the document already uploaded. Say so rather than imply a new source.
      if (documentIds.has(result.documentId)) merged.push(file.name);
      else documentIds.add(result.documentId);
    } catch (error) {
      toast.error(`Couldn't upload ${file.name}`, {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }
  if (merged.length > 0) {
    toast.info(
      merged.length === 1
        ? `${merged[0]} matched a file already added`
        : `${merged.length} files matched files already added`,
      { description: "Identical contents are stored once." },
    );
  }
  invalidateProjectSources(projectId);
}

/** Create-project drop area: stages files until the project exists. */
export function ProjectSourceDropzone({
  staged,
  onChange,
  disabled = false,
}: {
  staged: StagedSource[];
  onChange: (next: StagedSource[]) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  // Count enter/leave pairs: children fire dragleave on the parent.
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const { next, unsupported, duplicates } = addStagedSources(staged, files);
      if (next.length !== staged.length) onChange(next);
      if (unsupported.length > 0) {
        toast.info(
          unsupported.length === 1
            ? `Can't add ${unsupported[0]}`
            : `Can't add ${unsupported.length} files`,
          { description: `Supported types: ${RAG_UPLOAD_ACCEPT}` },
        );
      }
      // Name, size and mtime can in principle match for two different files, so
      // never drop one without saying so.
      if (duplicates.length > 0) {
        toast.info(
          duplicates.length === 1
            ? `${duplicates[0]} is already added`
            : `${duplicates.length} files were already added`,
        );
      }
    },
    [staged, onChange],
  );

  const endDrag = useCallback(() => {
    dragDepth.current = 0;
    setDragging(false);
  }, []);

  return (
    <div className="space-y-2.5">
      <p className="text-ui-15 font-medium text-foreground">Sources</p>
      {/* Panel is the drop target; the inner button owns the click so staged
          rows can carry their own remove buttons. */}
      <div
        // preventDefault runs even while disabled: nothing else on the page
        // cancels a file drop, so the browser would navigate to the file and
        // kill the uploads in flight.
        onDragEnter={(e) => {
          e.preventDefault();
          if (disabled) return;
          dragDepth.current += 1;
          setDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (disabled) return;
          e.dataTransfer.dropEffect = "copy";
        }}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          if (disabled) return;
          endDrag();
          addFiles(Array.from(e.dataTransfer.files ?? []));
        }}
        className={cn(
          "rounded-[22px] border border-border transition-colors dark:border-white/10",
          dragging && "border-primary/60 bg-primary/5",
          disabled && "opacity-60",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          multiple={true}
          accept={RAG_UPLOAD_ACCEPT}
          className="hidden"
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            e.target.value = "";
            addFiles(files);
          }}
        />
        {staged.length === 0 ? (
          <button
            type="button"
            aria-label="Add sources"
            disabled={disabled}
            onClick={() => inputRef.current?.click()}
            className="flex w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-[22px] px-6 py-12 text-center transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <HugeiconsIcon
              icon={FolderAddIcon}
              strokeWidth={1.75}
              className="size-6 text-muted-foreground"
            />
            <span className="text-sm text-muted-foreground">
              Add files every chat in this project can read
            </span>
          </button>
        ) : (
          <div className="flex flex-col gap-1 p-2">
            <ul className="max-h-52 space-y-0.5 overflow-y-auto">
              {staged.map((entry) => (
                <li
                  key={entry.id}
                  className="flex items-center gap-2.5 rounded-[10px] px-2.5 py-2 hover:bg-muted/50"
                >
                  <HugeiconsIcon
                    icon={File02Icon}
                    strokeWidth={1.75}
                    className="size-4 shrink-0 text-muted-foreground"
                  />
                  <span
                    className="min-w-0 flex-1 truncate text-ui-14 text-foreground"
                    title={entry.file.name}
                  >
                    {entry.file.name}
                  </span>
                  <span className="shrink-0 text-ui-11 text-muted-foreground">
                    {formatSize(entry.file.size)}
                  </span>
                  <button
                    type="button"
                    aria-label={`Remove ${entry.file.name}`}
                    disabled={disabled}
                    onClick={() =>
                      onChange(staged.filter((row) => row.id !== entry.id))
                    }
                    className="shrink-0 rounded-full text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
                  >
                    <XIcon className="size-3.5" />
                  </button>
                </li>
              ))}
            </ul>
            <button
              type="button"
              disabled={disabled}
              onClick={() => inputRef.current?.click()}
              className="flex items-center justify-center gap-2 rounded-[10px] py-2 text-ui-13 font-medium text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <HugeiconsIcon
                icon={FolderAddIcon}
                strokeWidth={1.75}
                className="size-4"
              />
              Add files
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
