import { useCallback, useRef, useState } from "react";
import { CloudUploadIcon, Cancel01Icon, Loading03Icon, CheckmarkCircle02Icon, Alert02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { uploadUnstructuredFile, removeUnstructuredFile } from "../../api";

const ACCEPTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".md"];
const MAX_FILE_SIZE = 50 * 1024 * 1024;
const MAX_TOTAL_SIZE = 100 * 1024 * 1024;

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
  onFilesChange: (files: FileEntry[] | ((prev: FileEntry[]) => FileEntry[])) => void;
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
  filesRef.current = files;
  const [isDragOver, setIsDragOver] = useState(false);

  const totalSize = files.reduce((sum, f) => sum + f.size, 0);

  const handleFiles = useCallback(
    async (newFiles: File[]) => {
      const valid = newFiles.filter((f) => {
        if (!isValidExtension(f.name)) return false;
        if (f.size > MAX_FILE_SIZE) return false;
        return true;
      });

      if (valid.length === 0) return;

      const addedSize = valid.reduce((s, f) => s + f.size, 0);
      const currentTotal = filesRef.current.reduce((sum, f) => sum + f.size, 0);
      if (currentTotal + addedSize > MAX_TOTAL_SIZE) return;

      const entries: FileEntry[] = valid.map((f) => ({
        id: "",
        name: f.name,
        size: f.size,
        status: "uploading" as const,
        abortController: new AbortController(),
      }));

      onFilesChange((prev) => [...prev, ...entries]);

      for (let i = 0; i < valid.length; i++) {
        const file = valid[i];
        const entry = entries[i];
        let updatedId = "";
        let updatedStatus: FileEntry["status"] = "error";
        let updatedError: string | undefined;
        try {
          const existingIds = filesRef.current.filter((f) => f.id).map((f) => f.id);
          const result = await uploadUnstructuredFile(
            file,
            blockId,
            entry.abortController?.signal,
            existingIds,
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
              ? { ...f, id: updatedId, status: updatedStatus, error: updatedError }
              : f,
          ),
        );
      }

    },
    [blockId, onFilesChange],
  );

  const deletedIdsRef = useRef(new Set<string>());
  const handleRemove = useCallback(
    (index: number) => {
      const entry = filesRef.current[index];
      if (!entry) return;
      if (entry.status === "uploading" && entry.abortController) {
        entry.abortController.abort();
      }
      if (entry.id && entry.status === "ok" && !deletedIdsRef.current.has(entry.id)) {
        deletedIdsRef.current.add(entry.id);
        void removeUnstructuredFile(blockId, entry.id).catch(() => {});
      }
      onFilesChange((prev) => prev.filter((_, i) => i !== index));
    },
    [blockId, onFilesChange],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      if (disabled) return;
      const dropped = Array.from(e.dataTransfer.files);
      handleFiles(dropped);
    },
    [disabled, handleFiles],
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!disabled) setIsDragOver(true);
    },
    [disabled],
  );

  const handleDragLeave = useCallback(() => setIsDragOver(false), []);

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
        className={`nodrag flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed px-4 py-6 text-center transition-colors ${
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50"
        } ${disabled ? "pointer-events-none opacity-50" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <HugeiconsIcon icon={CloudUploadIcon} className="text-muted-foreground mb-2 size-8" />
        <p className="text-muted-foreground text-sm">
          Drop files here or click to browse
        </p>
        <p className="text-muted-foreground/60 mt-1 text-xs">
          PDF, DOCX, TXT, MD - up to 50MB each, 100MB total
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
                <HugeiconsIcon icon={Loading03Icon} className="text-muted-foreground size-4 animate-spin" />
              )}
              {entry.status === "ok" && (
                <HugeiconsIcon icon={CheckmarkCircle02Icon} className="size-4 text-green-500" />
              )}
              {entry.status === "error" && (
                <HugeiconsIcon icon={Alert02Icon} className="size-4 text-red-500" />
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
            <span>{successFiles.length} file{successFiles.length !== 1 ? "s" : ""} uploaded</span>
            <span>{formatSize(totalSize)} / 100MB</span>
          </div>
        </div>
      )}
    </div>
  );
}

export type { FileEntry };
