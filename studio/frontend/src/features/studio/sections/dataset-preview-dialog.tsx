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
import { useTrainingActions, useTrainingConfigStore } from "@/features/training";
import { checkDatasetFormat } from "@/features/training/api/datasets-api";
import type { CheckFormatResponse } from "@/features/training/types/datasets";
import { collectPreviewImages, formatCell } from "./dataset-preview-dialog-utils";
import {
  DatasetMappingCard,
  DatasetMappingFooter,
  HeaderPick,
  deriveDefaultMapping,
} from "./dataset-preview-dialog-mapping";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type DatasetPreviewDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetName: string | null;
  hfToken: string | null;
  datasetSubset?: string | null;
  datasetSplit?: string | null;
  mode?: "preview" | "mapping";
  initialData?: CheckFormatResponse | null;
  isVlm?: boolean;
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DatasetPreviewDialog({
  open,
  onOpenChange,
  datasetName,
  hfToken,
  datasetSubset,
  datasetSplit,
  mode = "preview",
  initialData,
  isVlm = false,
}: DatasetPreviewDialogProps) {
  const [data, setData] = useState<CheckFormatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const manualMapping = useTrainingConfigStore((s) => s.datasetManualMapping);
  const setManualMapping = useTrainingConfigStore((s) => s.setDatasetManualMapping);
  const { isStarting, startError, startTrainingRun } = useTrainingActions();

  const mappingEnabled = !!data?.requires_manual_mapping;
  const showMappingFooter = mode === "mapping" && mappingEnabled;
  const mappingOk = !!manualMapping.input && !!manualMapping.output;
  const leftLabel = isVlm ? "Image" : "Input";
  const rightLabel = isVlm ? "Text" : "Output";

  useEffect(() => {
    if (!manualMapping.input || !manualMapping.output) return;
    if (manualMapping.input !== manualMapping.output) return;
    setManualMapping({ input: manualMapping.input, output: null });
  }, [manualMapping.input, manualMapping.output, setManualMapping]);

  useEffect(() => {
    if (!open || !datasetName) {
      setData(null);
      setError(null);
      return;
    }
    if (initialData) {
      setData(initialData);
      setError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    checkDatasetFormat({
      datasetName,
      hfToken,
      subset: datasetSubset,
      split: datasetSplit,
      isVlm,
    })
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
  }, [open, datasetName, hfToken, datasetSubset, datasetSplit, isVlm, initialData]);

  useEffect(() => {
    if (!open || !datasetName || !data?.requires_manual_mapping) return;
    if (manualMapping.input || manualMapping.output) return;
    const derived = deriveDefaultMapping(data, isVlm);
    if (!derived.input && !derived.output) return;
    setManualMapping(derived);
  }, [
    open,
    datasetName,
    data?.requires_manual_mapping,
    isVlm,
    manualMapping.input,
    manualMapping.output,
    setManualMapping,
    data,
  ]);

  const rows = data?.preview_samples ?? [];
  const columns = data?.columns ?? [];

  // Determine source label
  const sourceLabel = useMemo(() => {
    if (!datasetName) return "";
    if (datasetName.includes("/")) {
      let label = `Hugging Face (${datasetName}`;
      if (datasetSubset) label += ` / ${datasetSubset}`;
      if (datasetSplit) label += ` / ${datasetSplit}`;
      label += ")";
      return label;
    }
    return `Local Files (${datasetName})`;
  }, [datasetName, datasetSubset, datasetSplit]);

  // Build TanStack Table columns from the column names
  const tableColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    if (!columns.length) return [];
    return columns.map((colName) => ({
      accessorKey: colName,
      header: () => (
        <div className="flex flex-col gap-2">
          <span className="font-heading text-[13px] font-semibold tracking-tight text-foreground">
            {colName}
          </span>
          {mappingEnabled && (
            <div className="flex items-center gap-3">
              {(manualMapping.output == null || manualMapping.output !== colName) &&
                (manualMapping.input == null || manualMapping.input === colName) && (
                  <HeaderPick
                    label={leftLabel}
                    checked={manualMapping.input === colName}
                    onCheckedChange={(checked) => {
                      setManualMapping({
                        input: checked ? colName : null,
                        output:
                          checked && manualMapping.output === colName
                            ? null
                            : manualMapping.output,
                      });
                    }}
                  />
                )}
              {(manualMapping.input == null || manualMapping.input !== colName) &&
                (manualMapping.output == null || manualMapping.output === colName) && (
                  <HeaderPick
                    label={rightLabel}
                    checked={manualMapping.output === colName}
                    onCheckedChange={(checked) => {
                      setManualMapping({
                        input:
                          checked && manualMapping.input === colName
                            ? null
                            : manualMapping.input,
                        output: checked ? colName : null,
                      });
                    }}
                  />
                )}
            </div>
          )}
        </div>
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
  }, [
    columns,
    manualMapping.input,
    manualMapping.output,
    setManualMapping,
    mappingEnabled,
    leftLabel,
    rightLabel,
  ]);

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

              {mappingEnabled && (
                <DatasetMappingCard
                  leftLabel={leftLabel}
                  rightLabel={rightLabel}
                  mappingOk={mappingOk}
                  input={manualMapping.input}
                  output={manualMapping.output}
                />
              )}

              {/* Data table */}
              <div className="flex-1 min-h-0 rounded-xl corner-squircle ring-1 ring-border/60 overflow-auto">
                <DataTable columns={tableColumns} data={rows} />
              </div>

              {/* Footer */}
              <div className="mt-3">
                <p className="text-[11px] text-muted-foreground/60 text-center tabular-nums">
                  Showing {rows.length}
                  {data.total_rows != null &&
                    ` of ${data.total_rows.toLocaleString()}`}{" "}
                  rows
                </p>

                {mode === "preview" && mappingEnabled && (
                  <p className="mt-2 text-[11px] text-muted-foreground/70 text-center">
                    Mapping is saved automatically. You can start training anytime.
                  </p>
                )}

                {showMappingFooter && (
                  <DatasetMappingFooter
                    leftLabel={leftLabel}
                    rightLabel={rightLabel}
                    mappingOk={mappingOk}
                    isStarting={isStarting}
                    startError={startError}
                    onCancel={() => onOpenChange(false)}
                    onStartTraining={async () => {
                      const ok = await startTrainingRun();
                      if (ok) onOpenChange(false);
                    }}
                  />
                )}
              </div>
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

// mapping UI extracted to ./dataset-preview-dialog-mapping.tsx
