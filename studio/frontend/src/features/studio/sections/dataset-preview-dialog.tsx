import type { ColumnDef } from "@tanstack/react-table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DataTable } from "@/components/ui/data-table";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { Database02Icon, AlertCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useState } from "react";

// ---------------------------------------------------------------------------
// Types (matches CheckFormatResponse from backend)
// ---------------------------------------------------------------------------

type CheckFormatResponse = {
  requires_manual_mapping: boolean;
  detected_format: string;
  columns: string[];
  suggested_mapping?: Record<string, string> | null;
  detected_image_column?: string | null;
  detected_text_column?: string | null;
  preview_samples?: Record<string, unknown>[] | null;
  total_rows?: number | null;
};

type PreviewImagePayload = {
  type: "image";
  mime?: string;
  width?: number;
  height?: number;
  data?: string;
};

type DatasetPreviewDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetName: string | null;
  hfToken: string | null;
};

// ---------------------------------------------------------------------------
// API -- uses existing /check-format endpoint
// ---------------------------------------------------------------------------

async function fetchCheckFormat(
  datasetName: string,
  hfToken: string | null,
): Promise<CheckFormatResponse> {
  const res = await fetch("/api/datasets/check-format", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_name: datasetName,
      hf_token: hfToken || undefined,
      split: "train",
    }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DatasetPreviewDialog({
  open,
  onOpenChange,
  datasetName,
  hfToken,
}: DatasetPreviewDialogProps) {
  const [data, setData] = useState<CheckFormatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !datasetName) {
      setData(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchCheckFormat(datasetName, hfToken)
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || "Failed to load preview");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, datasetName, hfToken]);

  const rows = data?.preview_samples ?? [];
  const columns = data?.columns ?? [];

  // Determine source label
  const sourceLabel = useMemo(() => {
    if (!datasetName) return "";
    if (datasetName.includes("/")) return `Hugging Face (${datasetName})`;
    return `Local Files (${datasetName})`;
  }, [datasetName]);

  // Build TanStack Table columns from the column names
  const tableColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    if (!columns.length) return [];
    return columns.map((colName) => ({
      accessorKey: colName,
      header: () => (
        <span className="font-heading text-[13px] font-semibold tracking-tight text-foreground">
          {colName}
        </span>
      ),
      cell: ({ getValue }: { getValue: () => unknown }) => {
        const value = getValue();
        const images = collectPreviewImages(value);
        if (images.length > 0) {
          return (
            <div className="flex flex-wrap gap-2">
              {images.slice(0, 4).map((image, index) => {
                const mime = image.mime || "image/jpeg";
                const src = image.data ? `data:${mime};base64,${image.data}` : "";
                const width = image.width ?? 128;
                const height = image.height ?? 128;
                return (
                  <img
                    key={`${colName}-img-${index}`}
                    src={src}
                    alt={`preview-${index}`}
                    className="h-16 w-auto max-w-40 rounded-md border object-contain bg-muted"
                    width={width}
                    height={height}
                    loading="lazy"
                  />
                );
              })}
              {images.length > 4 && (
                <span className="text-xs text-muted-foreground self-end">
                  +{images.length - 4} more
                </span>
              )}
            </div>
          );
        }

        const text = formatCell(value);
        if (!text) {
          return (
            <span className="text-muted-foreground/40 italic text-[13px]">
              --
            </span>
          );
        }
        const full =
          typeof value === "string" ? value : JSON.stringify(value);
        return (
          <p
            className="text-[13px] leading-relaxed line-clamp-6"
            title={full}
          >
            {text}
          </p>
        );
      },
    }));
  }, [columns]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-5xl w-[90vw] max-h-[88vh] flex flex-col gap-0 p-0 overflow-hidden rounded-3xl corner-squircle"
        showCloseButton={true}
      >
        {/* Header */}
        <DialogHeader className="px-6 pt-5 pb-4 shrink-0">
          <div className="flex items-center gap-3 pr-10">
            <div className="rounded-xl corner-squircle p-2 ring-1 ring-indigo-200 bg-indigo-50 text-indigo-600 dark:ring-indigo-800 dark:bg-indigo-950 dark:text-indigo-400 shrink-0">
              <HugeiconsIcon icon={Database02Icon} className="size-4" />
            </div>
            <DialogTitle className="font-heading text-lg font-semibold tracking-tight">
              Dataset Preview
            </DialogTitle>
          </div>
        </DialogHeader>

        {/* Body */}
        <div className="flex flex-col min-h-0 flex-1 overflow-hidden px-6 pb-6">
          {/* Loading */}
          {loading && (
            <div className="py-24 flex flex-col items-center justify-center gap-3">
              <div className="rounded-2xl corner-squircle bg-primary/5 p-4">
                <Spinner className="size-5 text-primary" />
              </div>
              <p className="text-sm text-muted-foreground font-medium">
                Loading preview...
              </p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="py-20 flex flex-col items-center justify-center gap-3">
              <div className="rounded-2xl corner-squircle bg-destructive/10 p-3">
                <HugeiconsIcon
                  icon={AlertCircleIcon}
                  className="size-5 text-destructive"
                />
              </div>
              <div className="text-center space-y-1">
                <p className="text-sm font-medium text-destructive">{error}</p>
                <p className="text-xs text-muted-foreground">
                  Make sure the backend is running on port 8000.
                </p>
              </div>
            </div>
          )}

          {/* Content */}
          {!loading && !error && data && (
            <>
              {/* Metadata card */}
              <div className="rounded-xl corner-squircle ring-1 ring-border/60 bg-muted/30 px-5 py-4 mb-4 space-y-2">
                <MetaRow label="Source" value={sourceLabel} />
                <MetaRow
                  label="Format"
                  value={data.detected_format || "--"}
                />
                <MetaRow
                  label="Total Rows"
                  value={
                    data.total_rows != null
                      ? data.total_rows.toLocaleString()
                      : "--"
                  }
                />
                <MetaRow
                  label="Columns"
                  value={
                    <span className="flex items-center gap-1.5 flex-wrap">
                      {columns.map((col) => (
                        <Badge
                          key={col}
                          variant="outline"
                          className="text-[11px] font-mono h-5"
                        >
                          {col}
                        </Badge>
                      ))}
                    </span>
                  }
                />
              </div>

              {/* Data table */}
              <div className="flex-1 min-h-0 rounded-xl corner-squircle ring-1 ring-border/60 overflow-auto">
                <DataTable columns={tableColumns} data={rows} />
              </div>

              {/* Footer */}
              <p className="text-[11px] text-muted-foreground/60 mt-3 text-center tabular-nums">
                Showing {rows.length}
                {data.total_rows != null &&
                  ` of ${data.total_rows.toLocaleString()}`}{" "}
                rows
              </p>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Metadata row
// ---------------------------------------------------------------------------

function MetaRow({
  label,
  value,
}: {
  label: string;
  value: ReactNode;
}) {
  return (
    <div className="flex items-baseline gap-3 text-sm">
      <span className="text-muted-foreground font-medium text-xs w-24 shrink-0">
        {label}:
      </span>
      <span className="text-foreground text-[13px] min-w-0">{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatCell(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean")
    return String(value);
  if (Array.isArray(value) || typeof value === "object")
    return JSON.stringify(value).slice(0, 500);
  return String(value);
}

function isPreviewImagePayload(value: unknown): value is PreviewImagePayload {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return (
    record.type === "image" &&
    typeof record.data === "string" &&
    record.data.length > 0
  );
}

function collectPreviewImages(value: unknown): PreviewImagePayload[] {
  const images: PreviewImagePayload[] = [];
  const stack: unknown[] = [value];
  let steps = 0;

  while (stack.length > 0 && steps < 200) {
    steps += 1;
    const current = stack.pop();
    if (isPreviewImagePayload(current)) {
      images.push(current);
      continue;
    }

    if (Array.isArray(current)) {
      for (const item of current) stack.push(item);
      continue;
    }

    if (current && typeof current === "object") {
      for (const nested of Object.values(current as Record<string, unknown>)) {
        stack.push(nested);
      }
    }
  }

  return images;
}
