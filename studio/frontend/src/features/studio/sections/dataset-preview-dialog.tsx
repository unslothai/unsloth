// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { aiAssistMapping } from "@/features/training/api/datasets-api";
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
import { useTrainingActions, useTrainingConfigStore } from "@/features/training";
import { checkDatasetFormat } from "@/features/training/api/datasets-api";
import type { CheckFormatResponse } from "@/features/training/types/datasets";
import { Database02Icon, AlertCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useShallow } from "zustand/react/shallow";
import { collectPreviewImages, formatCell } from "./dataset-preview-dialog-utils";
import {
  DatasetMappingCard,
  DatasetMappingFooter,
  HeaderRolePicker,
  deriveDefaultMapping,
  getAvailableRoles,
  isMappingComplete,
  remapRolesForFormat,
} from "./dataset-preview-dialog-mapping";

/** Chatml → format-specific role remap (only for formats that differ from chatml). */
const ROLE_REMAP: Record<string, Record<string, string>> = {
  alpaca: { user: "instruction", system: "input", assistant: "output" },
  sharegpt: { user: "human", assistant: "gpt", system: "system" },
};

type DatasetPreviewDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetName: string | null;
  datasetSource?: "huggingface" | "upload";
  hfToken: string | null;
  datasetSubset?: string | null;
  datasetSplit?: string | null;
  mode?: "preview" | "mapping";
  initialData?: CheckFormatResponse | null;
  isVlm?: boolean;
};

export function DatasetPreviewDialog({
  open,
  onOpenChange,
  datasetName,
  datasetSource,
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

  const {
    manualMapping, setManualMapping, datasetFormat,
    setDatasetAdvisorFields, datasetAdvisorNotification,
    datasetSystemPrompt,
    selectedModel,
    modelType,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      manualMapping: s.datasetManualMapping,
      setManualMapping: s.setDatasetManualMapping,
      datasetFormat: s.datasetFormat,
      setDatasetAdvisorFields: s.setDatasetAdvisorFields,
      datasetAdvisorNotification: s.datasetAdvisorNotification,
      datasetSystemPrompt: s.datasetSystemPrompt,
      selectedModel: s.selectedModel,
      modelType: s.modelType,
    })),
  );
  const { isStarting, startError, startTrainingRun } = useTrainingActions();

  // If the backend reports image data, treat as VLM even if the prop
  // hasn't caught up yet (isDatasetImage may still be null in the store).
  const effectiveIsAudio = !!data?.is_audio;
  const effectiveIsVlm = isVlm || !!data?.is_image;

  const hasHeuristicMapping = !data?.requires_manual_mapping && !!data?.suggested_mapping;
  const mappingEnabled = !!data?.requires_manual_mapping || hasHeuristicMapping;
  const showMappingFooter = mode === "mapping" && mappingEnabled;
  const mappingOk = isMappingComplete(manualMapping, effectiveIsVlm, datasetFormat, effectiveIsAudio);
  const availableRoles = getAvailableRoles(effectiveIsVlm, datasetFormat, effectiveIsAudio);
  const isHfDataset = datasetSource === "huggingface";

  // ── AI Assist ──────────────────────────────────────────────────────
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);

  const handleAiAssist = useCallback(async () => {
    if (!data?.columns || !data?.preview_samples) return;
    setIsAiLoading(true);
    setAiError(null);

    try {
      const result = await aiAssistMapping({
        columns: data.columns,
        samples: data.preview_samples,
        datasetName: datasetName,
        hfToken: hfToken,
        modelName: selectedModel,
        modelType: modelType,
      });

      if (result.success && result.suggested_mapping) {
        // Remap from chatml roles (user/assistant/system) to format-specific roles
        const table = ROLE_REMAP[datasetFormat];
        const mapped: Record<string, string> = {};
        for (const [col, role] of Object.entries(result.suggested_mapping)) {
          mapped[col] = table ? (table[role] ?? role) : role;
        }
        setManualMapping(mapped);

        // Store conversion advisor fields (system prompt, label mapping, notification)
        if (result.system_prompt || result.label_mapping || result.user_notification) {
          setDatasetAdvisorFields({
            systemPrompt: result.system_prompt ?? undefined,
            labelMapping: result.label_mapping ?? undefined,
            notification: result.user_notification ?? null,
          });
        }
      } else {
        setAiError(result.warning || "AI could not determine column roles.");
      }
    } catch (err) {
      setAiError(err instanceof Error ? err.message : "AI assist failed.");
    } finally {
      setIsAiLoading(false);
    }
  }, [data, datasetFormat, datasetName, hfToken, setManualMapping, setDatasetAdvisorFields, selectedModel, modelType]);

  // When format changes, remap existing mapping roles to the new format's role names
  const prevFormatRef = useRef(datasetFormat);
  useEffect(() => {
    const prev = prevFormatRef.current;
    prevFormatRef.current = datasetFormat;
    if (prev === datasetFormat) return;
    if (Object.keys(manualMapping).length === 0) return;
    setManualMapping(remapRolesForFormat(manualMapping, datasetFormat));
  }, [datasetFormat]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle role change for a column
  const handleRoleChange = useCallback(
    (colName: string, role: string | undefined) => {
      const next = { ...manualMapping };
      // Remove this column's previous role
      delete next[colName];
      if (role) {
        // Remove any other column that had this role (each role can only be assigned once)
        for (const [col, r] of Object.entries(next)) {
          if (r === role) delete next[col];
        }
        next[colName] = role;
      }
      setManualMapping(next);
    },
    [manualMapping, setManualMapping],
  );

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

  // Pre-fill mapping from suggested_mapping when data arrives
  useEffect(() => {
    if (!open || !datasetName) return;
    if (!data?.requires_manual_mapping && !data?.suggested_mapping) return;
    // Don't overwrite if mapping already has entries
    if (Object.keys(manualMapping).length > 0) return;
    const derived = deriveDefaultMapping(data, effectiveIsVlm, datasetFormat, effectiveIsAudio);
    if (Object.keys(derived).length === 0) return;
    setManualMapping(derived);
  }, [open, datasetName, data, effectiveIsVlm, datasetFormat, effectiveIsAudio, manualMapping, setManualMapping]);

  const rows = data?.preview_samples ?? [];
  const columns = data?.columns ?? [];

  // Determine source label
  const sourceLabel = useMemo(() => {
    if (!datasetName) return "";
    if (datasetSource === "huggingface") {
      let label = `Hugging Face (${datasetName}`;
      if (datasetSubset) label += ` / ${datasetSubset}`;
      if (datasetSplit) label += ` / ${datasetSplit}`;
      label += ")";
      return label;
    }
    return `Local Files (${datasetName})`;
  }, [datasetName, datasetSource, datasetSubset, datasetSplit]);

  // Build TanStack Table columns from the column names
  const tableColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    if (!columns.length) return [];

    const dataCols: ColumnDef<Record<string, unknown>>[] = columns.map((colName) => ({
      accessorKey: colName,
      header: () => (
        <div className="flex flex-col gap-2">
          <span className="font-heading text-[13px] font-semibold tracking-tight text-foreground">
            {colName}
          </span>
          {mappingEnabled && (
            <HeaderRolePicker
              currentRole={manualMapping[colName]}
              onRoleChange={(role) => handleRoleChange(colName, role)}
              availableRoles={availableRoles}
            />
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
        const full = typeof value === "string" ? value : JSON.stringify(value);
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

    // Prepend generated system prompt column when advisor is active
    if (datasetSystemPrompt) {
      dataCols.unshift({
        id: "__system_generated",
        header: () => (
          <div className="flex flex-col gap-2">
            <span className="font-heading text-[13px] font-semibold tracking-tight text-foreground">
              System <span className="text-muted-foreground font-normal">(generated)</span>
            </span>
            {mappingEnabled && (
              <Badge variant="outline" className="h-6 w-fit text-[10px] px-2 py-0 border-dashed text-muted-foreground">
                System
              </Badge>
            )}
          </div>
        ),
        cell: () => (
          <p
            className="text-[13px] leading-relaxed line-clamp-6 text-muted-foreground italic"
            title={datasetSystemPrompt}
          >
            {datasetSystemPrompt}
          </p>
        ),
      });
    }

    return dataCols;
  }, [
    columns,
    manualMapping,
    handleRoleChange,
    mappingEnabled,
    availableRoles,
    datasetSystemPrompt,
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
        <div className="flex flex-col min-h-0 flex-1 overflow-auto px-6 pb-6">
          {/* Loading */}
          {loading && (
            <div className="py-24 flex flex-col items-center justify-center gap-3">
              <div className="rounded-2xl corner-squircle bg-primary/5 p-4">
                <Spinner className="size-5 text-primary" />
              </div>
              <p className="text-sm text-muted-foreground font-medium">
                {isHfDataset ? "Fetching dataset preview from Hugging Face..." : "Loading preview..."}
              </p>
              {isHfDataset && (
                <p className="text-xs text-muted-foreground/60">
                  This may take a moment for large datasets
                </p>
              )}
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
                  Make sure the backend is running and reachable.
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

              {data.warning && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400 mb-4 flex items-start gap-2.5">
                  <HugeiconsIcon icon={AlertCircleIcon} className="size-4 shrink-0 mt-0.5" />
                  <span>{data.warning}</span>
                </div>
              )}

              {mappingEnabled && (
                <DatasetMappingCard
                  mapping={manualMapping}
                  mappingOk={mappingOk}
                  autoDetected={hasHeuristicMapping}
                  isVlm={effectiveIsVlm}
                  isAudio={effectiveIsAudio}
                  format={datasetFormat}
                  onAiAssist={handleAiAssist}
                  isAiLoading={isAiLoading}
                  aiError={aiError}
                  advisorNotification={datasetAdvisorNotification}
                  advisorSystemPrompt={datasetSystemPrompt || undefined}
                />
              )}

              {/* Data table */}
              <div className="flex-1 min-h-[250px] rounded-xl corner-squircle ring-1 ring-border/60 overflow-auto">
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
