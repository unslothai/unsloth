// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  consumeNativePathToken,
  registerNativeAttachmentPath,
  useNativeDropTarget,
} from "@/features/native-intents";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { File02Icon, FolderAddIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { XIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  invalidateProjectSources,
  uploadProjectDocument,
} from "../api/rag-api";
import { RAG_UPLOAD_ACCEPT } from "../types/rag";
import {
  addStagedSources,
  EXPIRY_GRACE_MS,
  isExpired,
  nativeExpiryMs,
  type StagedSource,
  stagedFromFile,
  stagedFromIntent,
} from "./staged-source";
import { resolveVisionOverrides } from "./vision-overrides";

export type { StagedSource };

function nativeFileName(path: string): string {
  const segments = path.split(/[\\/]/);
  return segments[segments.length - 1] || path;
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
function isSupported(name: string): boolean {
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return false;
  return ACCEPTED_EXTS.has(name.slice(dot).toLowerCase());
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
  for (const entry of staged) {
    try {
      if (isExpired(entry, Date.now())) {
        throw new Error("The drop expired. Add it again from the project.");
      }
      // Leases are short-lived, so mint one per file as its turn comes up.
      const source =
        entry.upload instanceof File
          ? entry.upload
          : {
              nativePathLease: (
                await consumeNativePathToken(entry.upload.nativeToken, "attach")
              ).nativePathLease,
            };
      const result = await uploadProjectDocument(projectId, source, ocr, caption);
      // Same bytes under another name: the backend hashes content, so this is
      // the document already uploaded. Say so rather than imply a new source.
      if (documentIds.has(result.documentId)) merged.push(entry.name);
      else documentIds.add(result.documentId);
    } catch (error) {
      toast.error(`Couldn't upload ${entry.name}`, {
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
  onPendingChange,
}: {
  staged: StagedSource[];
  onChange: (next: StagedSource[]) => void;
  disabled?: boolean;
  /** Native drops register asynchronously and reach `onChange` only once they
   * settle. Create must wait, or it commits without the files just dropped. */
  onPendingChange?: (pending: boolean) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  // Count enter/leave pairs: children fire dragleave on the parent.
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);
  // Registering a native drop is async, so the props captured when it started
  // are stale by the time it resolves. Merge against these instead.
  const stagedRef = useRef(staged);
  stagedRef.current = staged;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  // Radix unmounts the dialog content on close, so a cancel takes this
  // component down before the reset below reaches it. Set on setup, or
  // StrictMode's replayed cleanup would leave it false forever.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  // The dialog can stay mounted across close, so an unmount flag alone cannot
  // see a cancel. Any array this component did not hand off is an external
  // reset, and a drop still registering is no longer wanted. Identity, not
  // length: `reset()` swaps in a fresh array even when it was already empty.
  const generation = useRef(0);
  const handedOff = useRef<StagedSource[] | null>(null);
  useEffect(() => {
    if (staged === handedOff.current) return;
    handedOff.current = staged;
    generation.current += 1;
  }, [staged]);

  /** Hand a list to the owner without reading it back as an external reset. */
  const commit = useCallback((next: StagedSource[]) => {
    // The ref too, not just on the next render: two drops settling in one tick
    // would both merge against the old list and the second would lose the first.
    stagedRef.current = next;
    handedOff.current = next;
    onChangeRef.current(next);
  }, []);

  const onPendingChangeRef = useRef(onPendingChange);
  onPendingChangeRef.current = onPendingChange;
  const pending = useRef(0);
  const addPending = useCallback((delta: number) => {
    pending.current += delta;
    // A drop from a previous mount must not answer for the live dropzone: its
    // "done" would re-enable Create while the current drop is still pending.
    if (!mounted.current) return;
    onPendingChangeRef.current?.(pending.current > 0);
  }, []);

  // Prune desktop drops whose token TTL has lapsed, so Create never commits a
  // project against sources the native layer has already pruned.
  useEffect(() => {
    const expiries = staged
      .map(nativeExpiryMs)
      .filter((value): value is number => value !== null);
    if (expiries.length === 0) return;
    const timer = setTimeout(
      () => {
        const current = stagedRef.current;
        const kept = current.filter((entry) => !isExpired(entry, Date.now()));
        if (kept.length === current.length) return;
        commit(kept);
        const dropped = current.length - kept.length;
        toast.info(
          dropped === 1
            ? "A dropped file expired"
            : `${dropped} dropped files expired`,
          { description: "Drag them in again to add them." },
        );
      },
      Math.max(0, Math.min(...expiries) - EXPIRY_GRACE_MS - Date.now()),
    );
    return () => clearTimeout(timer);
  }, [staged, commit]);

  const addSources = useCallback(
    (incoming: StagedSource[], unsupported: string[]) => {
      const current = stagedRef.current;
      const { next, duplicates } = addStagedSources(current, incoming);
      if (next.length !== current.length) commit(next);
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
    [commit],
  );

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const incoming = Array.from(files);
      addSources(
        incoming.filter((file) => isSupported(file.name)).map(stagedFromFile),
        incoming.filter((file) => !isSupported(file.name)).map((file) => file.name),
      );
    },
    [addSources],
  );

  const addNativePaths = useCallback(
    async (paths: string[]) => {
      const claimed = generation.current;
      const supported = paths.filter((path) => isSupported(nativeFileName(path)));
      const unsupported = paths
        .filter((path) => !isSupported(nativeFileName(path)))
        .map(nativeFileName);
      // Per path, so one rejected file does not discard the rest of the drop.
      addPending(1);
      const settled = await Promise.allSettled(
        supported.map(registerNativeAttachmentPath),
      ).finally(() => addPending(-1));
      // Cleared or closed while registering: let the tokens lapse rather than
      // refill a draft the next dialog would open on.
      if (!mounted.current || claimed !== generation.current) return;
      const staged = settled.flatMap((result) =>
        result.status === "fulfilled" ? [stagedFromIntent(result.value)] : [],
      );
      addSources(staged, unsupported);
      const failed = settled.length - staged.length;
      if (failed > 0) {
        toast.error(
          failed === 1 ? "Couldn't add a dropped file" : `Couldn't add ${failed} dropped files`,
        );
      }
    },
    [addSources, addPending],
  );

  // Stay claimed while disabled: unregistering hands the drop to the chat-wide
  // handler, which would attach it to the chat behind the dialog.
  const nativeDropRef = useNativeDropTarget({
    onDrop: (paths) => {
      if (disabled) return;
      void addNativePaths(paths);
    },
    onDragOver: (over) => setDragging(over && !disabled),
  });

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
        ref={nativeDropRef}
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
            <ul className="overlay-scrollbar-gutter max-h-52 space-y-0.5 overflow-y-auto">
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
                    title={entry.name}
                  >
                    {entry.name}
                  </span>
                  <span className="shrink-0 text-ui-11 text-muted-foreground">
                    {formatSize(entry.size)}
                  </span>
                  <button
                    type="button"
                    aria-label={`Remove ${entry.name}`}
                    disabled={disabled}
                    onClick={() =>
                      commit(staged.filter((row) => row.id !== entry.id))
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
