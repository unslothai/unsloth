import { useCallback, useEffect, useRef } from "react";
import { consumeNativePathToken, useNativeFileDrop } from "@/features/native-intents";
import {
  CloudUploadIcon,
  Cancel01Icon,
  CheckmarkCircle02Icon,
  Alert02Icon,
} from "@hugeicons/core-free-icons";
import { Spinner } from "@/components/ui/spinner";
import { HugeiconsIcon } from "@hugeicons/react";
import { uploadUnstructuredFile, removeUnstructuredFile } from "../../api";
import {
  UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES,
  UNSTRUCTURED_RECIPE_UPLOAD_MAX_LABEL,
  UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_BYTES,
  UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_LABEL,
} from "./upload-limits";

const ACCEPTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".md"];

/** One queued upload: browser bytes, or a desktop drop the backend redeems. */
type UploadCandidate = {
  name: string;
  size: number;
  source: File | { token: string };
};

type FileEntry = {
  id: string;
  name: string;
  size: number;
  status: "uploading" | "ok" | "error";
  error?: string;
  abortController?: AbortController;
};

type UnstructuredDropZoneProps = {
  blockId: string;
  files: FileEntry[];
  onFilesChange: (
    files: FileEntry[] | ((prev: FileEntry[]) => FileEntry[]),
  ) => void;
  disabled?: boolean;
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isValidExtension(name: string): boolean {
  const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
  return ACCEPTED_EXTENSIONS.includes(ext);
}

export function UnstructuredDropZone({
  blockId,
  files,
  onFilesChange,
  disabled,
}: UnstructuredDropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const filesRef = useRef(files);
  const blockIdRef = useRef(blockId);
  const mountedRef = useRef(true);

  useEffect(() => {
    filesRef.current = files;
    blockIdRef.current = blockId;
  }, [files, blockId]);
  useEffect(() => () => {
    mountedRef.current = false;
  }, []);

  const totalSize = files.reduce((sum, f) => sum + f.size, 0);

  const uploadCandidates = useCallback(
    async (candidates: UploadCandidate[]) => {
      const valid = candidates.filter((f) => {
        if (!isValidExtension(f.name)) return false;
        if (f.size > UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES) return false;
        return true;
      });

      if (valid.length === 0) return;

      const addedSize = valid.reduce((s, f) => s + f.size, 0);
      const currentTotal = filesRef.current.reduce((sum, f) => sum + f.size, 0);
      if (currentTotal + addedSize > UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_BYTES)
        return;

      const entries: FileEntry[] = valid.map((f) => ({
        id: "",
        name: f.name,
        size: f.size,
        status: "uploading" as const,
        abortController: new AbortController(),
      }));

      onFilesChange((prev) => [...prev, ...entries]);

      for (let i = 0; i < valid.length; i++) {
        const candidate = valid[i];
        const entry = entries[i];
        let updatedId = "";
        let updatedStatus: FileEntry["status"] = "error";
        let updatedError: string | undefined;
        try {
          // Leases are short-lived, so mint one per upload, not at drop time.
          const source =
            candidate.source instanceof File
              ? candidate.source
              : {
                  nativePathLease: (
                    await consumeNativePathToken(candidate.source.token, "attach")
                  ).nativePathLease,
                  name: candidate.name,
                  size: candidate.size,
                };
          const result = await uploadUnstructuredFile(
            source,
            blockId,
            entry.abortController?.signal,
          );
          updatedId = result.file_id;
          updatedStatus = result.status === "ok" ? "ok" : "error";
          updatedError = result.error;
        } catch (e) {
          if (e instanceof DOMException && e.name === "AbortError") {
            updatedError = "Cancelled";
          } else {
            updatedError = e instanceof Error ? e.message : "Upload failed";
          }
        }
        onFilesChange((prev) =>
          prev.map((f) =>
            f === entry
              ? {
                  ...f,
                  id: updatedId,
                  status: updatedStatus,
                  error: updatedError,
                }
              : f,
          ),
        );
      }
    },
    [blockId, onFilesChange],
  );

  const handleFiles = useCallback(
    (newFiles: File[]) =>
      uploadCandidates(
        newFiles.map((file) => ({
          name: file.name,
          size: file.size,
          source: file,
        })),
      ),
    [uploadCandidates],
  );

  const deletedIdsRef = useRef(new Set<string>());
  const handleRemove = useCallback(
    (index: number) => {
      const entry = filesRef.current[index];
      if (!entry) return;
      if (entry.status === "uploading" && entry.abortController) {
        entry.abortController.abort();
      }
      const needsServerRemove =
        entry.id &&
        entry.status === "ok" &&
        !deletedIdsRef.current.has(entry.id);
      onFilesChange((prev) => prev.filter((_, i) => i !== index));
      if (!needsServerRemove) return;
      deletedIdsRef.current.add(entry.id);
      removeUnstructuredFile(blockId, entry.id).catch(() => {
        // Skip if the drop zone unmounted or its block changed: the id no
        // longer belongs here and restoring would leak it into another block.
        if (!mountedRef.current || blockIdRef.current !== blockId) return;
        // Still exists server-side (counts toward quota); restore it at its
        // original position.
        deletedIdsRef.current.delete(entry.id);
        onFilesChange((prev) => {
          const next = [...prev];
          next.splice(Math.min(index, next.length), 0, {
            id: entry.id,
            name: entry.name,
            size: entry.size,
            status: "ok",
            error: "Remove failed — try again",
          });
          return next;
        });
      });
    },
    [blockId, onFilesChange],
  );

  // Tauri suppresses webview drop events, so the plain `onDrop` this zone
  // carried was dead on desktop (#9036).
  const { ref: dropRef, dragging: isDragOver, dragHandlers } = useNativeFileDrop({
    onFiles: handleFiles,
    // A seed corpus can run to hundreds of MB, so the backend redeems the
    // signed path itself rather than routing bytes through the webview.
    onNativeIntents: (intents) =>
      uploadCandidates(
        intents.map((intent) => ({
          name: intent.path.displayLabel,
          size: intent.path.sizeBytes ?? 0,
          source: { token: intent.path.token },
        })),
      ),
    accept: ACCEPTED_EXTENSIONS.join(","),
    disabled,
    disabledReason: "This block is busy. Try the drop again in a moment.",
  });

  const handleClick = useCallback(() => {
    if (!disabled) inputRef.current?.click();
  }, [disabled]);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = Array.from(e.target.files || []);
      handleFiles(selected);
      e.target.value = "";
    },
    [handleFiles],
  );

  const successFiles = files.filter((f) => f.status === "ok");

  return (
    <div className="space-y-2">
      <div
        // Stays hit-testable while disabled: pointer-events-none hides it from
        // elementFromPoint, so a native drop misses this target and falls
        // through to the window instead of saying the block is busy.
        className={`nodrag flex flex-col items-center justify-center rounded-md border-2 border-dashed px-4 py-6 text-center transition-colors ${
          isDragOver
            ? "border-ring-strong bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50"
        } ${disabled ? "cursor-default opacity-50" : "cursor-pointer"}`}
        ref={dropRef}
        {...dragHandlers}
        onClick={handleClick}
      >
        <HugeiconsIcon
          icon={CloudUploadIcon}
          className="text-muted-foreground mb-2 size-8"
        />
        <p className="text-muted-foreground text-sm">
          Drop files here or click to browse
        </p>
        <p className="text-muted-foreground/60 mt-1 text-xs">
          PDF, DOCX, TXT, MD - up to {UNSTRUCTURED_RECIPE_UPLOAD_MAX_LABEL}{" "}
          each, {UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_LABEL} total
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        multiple
        className="hidden"
        onChange={handleInputChange}
      />

      {files.length > 0 && (
        <div className="space-y-1">
          {files.map((entry, i) => (
            <div
              key={`${entry.name}-${i}`}
              className="flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm"
            >
              {entry.status === "uploading" && (
                <Spinner className="text-muted-foreground size-4" />
              )}
              {entry.status === "ok" && (
                <HugeiconsIcon
                  icon={CheckmarkCircle02Icon}
                  className="size-4 text-green-500"
                />
              )}
              {entry.status === "error" && (
                <HugeiconsIcon
                  icon={Alert02Icon}
                  className="size-4 text-red-500"
                />
              )}
              <span className="flex-1 truncate">{entry.name}</span>
              <span className="text-muted-foreground text-xs">
                {formatSize(entry.size)}
              </span>
              {entry.error && (
                <span className="text-xs text-red-500">{entry.error}</span>
              )}
              <button
                type="button"
                className="ml-auto inline-flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition hover:bg-destructive/10 hover:text-destructive"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemove(i);
                }}
              >
                <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
              </button>
            </div>
          ))}
          <div className="text-muted-foreground flex justify-between px-1 text-xs">
            <span>
              {successFiles.length} file{successFiles.length !== 1 ? "s" : ""}{" "}
              uploaded
            </span>
            <span>
              {formatSize(totalSize)} /{" "}
              {UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_LABEL}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export type { FileEntry };
