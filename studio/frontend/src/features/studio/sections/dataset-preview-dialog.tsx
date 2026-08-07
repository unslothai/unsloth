// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { DataTable } from "@/components/ui/data-table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import {
  type CheckFormatResponse,
  DatasetFormatError,
  aiAssistMapping,
  checkDatasetFormat,
  clearDeletedDataset,
  isRawTextDatasetFormat,
  useTrainingActions,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import type { DatasetSource } from "@/types/training";
import {
  AlertCircleIcon,
  CheckmarkCircle02Icon,
  Database02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ColumnDef } from "@tanstack/react-table";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import {
  DatasetMappingCard,
  DatasetMappingFooter,
  HeaderRolePicker,
} from "./dataset-preview-dialog-mapping";
import {
  deriveDefaultMapping,
  getAvailableRoles,
  isMappingComplete,
  remapRolesForFormat,
} from "./dataset-preview-dialog-mapping-utils";
import {
  collectPreviewImages,
  formatCell,
} from "./dataset-preview-dialog-utils";

/** Chatml → format-specific role remap (only for formats that differ from chatml). */
const ROLE_REMAP: Record<string, Record<string, string>> = {
  alpaca: { user: "instruction", system: "input", assistant: "output" },
  sharegpt: { user: "human", assistant: "gpt", system: "system" },
};

const EMPTY_PREVIEW_ROWS: Record<string, unknown>[] = [];
const EMPTY_PREVIEW_COLUMNS: string[] = [];

type DatasetPreviewResult = {
  requestKey: symbol;
  data: CheckFormatResponse | null;
  error: string | null;
};

type DatasetPreviewRequest = {
  requestKey: symbol;
  datasetName: string;
  hfToken: string | null;
  subset: string | null | undefined;
  split: string | null | undefined;
  isVlm: boolean;
  preferLocalCache: boolean;
  localPath: string | null;
};

type DatasetPreviewDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetName: string | null;
  datasetSource?: DatasetSource;
  hfToken: string | null;
  datasetKnownCached?: boolean;
  datasetLocalPath?: string | null;
  datasetStreaming?: boolean;
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
  datasetKnownCached = false,
  datasetLocalPath,
  datasetStreaming = false,
  datasetSubset,
  datasetSplit,
  mode = "preview",
  initialData,
  isVlm = false,
}: DatasetPreviewDialogProps) {
  const t = useT();
  const previewRequest = useMemo<DatasetPreviewRequest | null>(() => {
    if (!open) {
      return null;
    }
    if (!datasetName) {
      return null;
    }
    if (initialData) {
      return null;
    }

    return {
      requestKey: Symbol("dataset-preview-request"),
      datasetName,
      hfToken,
      subset: datasetSubset,
      split: datasetSplit,
      isVlm,
      preferLocalCache:
        datasetSource === "huggingface" &&
        datasetKnownCached &&
        !datasetStreaming,
      localPath:
        datasetSource === "huggingface" && !datasetStreaming
          ? (datasetLocalPath ?? null)
          : null,
    };
  }, [
    open,
    datasetName,
    datasetSource,
    hfToken,
    datasetKnownCached,
    datasetLocalPath,
    datasetStreaming,
    datasetSubset,
    datasetSplit,
    isVlm,
    initialData,
  ]);
  const requestKey = previewRequest?.requestKey ?? null;
  const [previewResult, setPreviewResult] =
    useState<DatasetPreviewResult | null>(null);
  const matchingResult =
    requestKey !== null && previewResult?.requestKey === requestKey
      ? previewResult
      : null;
  const data = initialData ?? matchingResult?.data ?? null;
  const error = matchingResult?.error ?? null;
  const loading = requestKey !== null && matchingResult === null;

  const {
    manualMapping,
    setManualMapping,
    datasetFormat,
    setDatasetAdvisorFields,
    datasetAdvisorNotification,
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
  const { startError, startBlocked, stopRequested, startTrainingRun } =
    useTrainingActions();

  // Treat backend-reported image data as VLM even if the prop hasn't caught up.
  const effectiveIsAudio = !!data?.is_audio;
  const effectiveIsVlm = isVlm || !!data?.is_image;

  const isRawFormat = isRawTextDatasetFormat(datasetFormat);
  const hasHeuristicMapping =
    !data?.requires_manual_mapping && !!data?.suggested_mapping;
  const mappingEnabled =
    !isRawFormat && (!!data?.requires_manual_mapping || hasHeuristicMapping);
  const showMappingFooter = mode === "mapping" && mappingEnabled;
  const mappingOk =
    isRawFormat ||
    isMappingComplete(
      manualMapping,
      effectiveIsVlm,
      datasetFormat,
      effectiveIsAudio,
    );
  const availableRoles = getAvailableRoles(
    effectiveIsVlm,
    datasetFormat,
    effectiveIsAudio,
  );
  const isHfDataset = datasetSource === "huggingface";
  const readyForTraining =
    !(isRawFormat || mappingEnabled) &&
    !data?.requires_manual_mapping &&
    !!data?.detected_format &&
    data.detected_format !== "unknown";
  const readyDetail =
    data?.chat_column && data.detected_format === "chatml"
      ? `Detected ChatML conversation column: ${data.chat_column}`
      : data?.detected_format
        ? `Detected ${data.detected_format} format. No manual column mapping needed.`
        : null;

  // ── AI Assist ──────────────────────────────────────────────────────
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);
  const aiAssistControllerRef = useRef<AbortController | null>(null);

  const cancelAiAssist = useCallback(() => {
    aiAssistControllerRef.current?.abort();
    aiAssistControllerRef.current = null;
    setIsAiLoading(false);
    setAiError(null);
  }, []);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen) {
        setPreviewResult(null);
        cancelAiAssist();
      }
      onOpenChange(nextOpen);
    },
    [cancelAiAssist, onOpenChange],
  );

  const handleAiAssist = useCallback(async () => {
    if (!data?.columns || !data?.preview_samples) return;
    aiAssistControllerRef.current?.abort();
    const controller = new AbortController();
    aiAssistControllerRef.current = controller;
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
        signal: controller.signal,
      });

      if (
        controller.signal.aborted ||
        aiAssistControllerRef.current !== controller
      ) {
        return;
      }

      if (result.success && result.suggested_mapping) {
        // Remap chatml roles to format-specific roles
        const table = ROLE_REMAP[datasetFormat];
        const mapped: Record<string, string> = {};
        for (const [col, role] of Object.entries(result.suggested_mapping)) {
          mapped[col] = table ? (table[role] ?? role) : role;
        }
        setManualMapping(mapped);

        // Store conversion advisor fields (system prompt, label mapping, notification)
        if (
          result.system_prompt ||
          result.label_mapping ||
          result.user_notification
        ) {
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
      if (controller.signal.aborted) {
        return;
      }
      setAiError(err instanceof Error ? err.message : "AI assist failed.");
    } finally {
      if (aiAssistControllerRef.current === controller) {
        aiAssistControllerRef.current = null;
        setIsAiLoading(false);
      }
    }
  }, [
    data,
    datasetFormat,
    datasetName,
    hfToken,
    setManualMapping,
    setDatasetAdvisorFields,
    selectedModel,
    modelType,
  ]);

  useEffect(() => {
    aiAssistControllerRef.current?.abort();
  }, [data, datasetFormat, datasetName, hfToken, modelType, selectedModel]);

  useEffect(
    () => () => {
      aiAssistControllerRef.current?.abort();
    },
    [],
  );

  // When format changes, remap existing mapping roles to the new format's role names
  const prevFormatRef = useRef(datasetFormat);
  useEffect(() => {
    const prev = prevFormatRef.current;
    prevFormatRef.current = datasetFormat;
    if (prev === datasetFormat) return;
    if (Object.keys(manualMapping).length === 0) return;
    setManualMapping(remapRolesForFormat(manualMapping, datasetFormat));
  }, [datasetFormat, manualMapping, setManualMapping]);

  const handleRoleChange = useCallback(
    (colName: string, role: string | undefined) => {
      const next = { ...manualMapping };
      delete next[colName];
      if (role) {
        // Each role maps to one column, so drop any other column holding it
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
    if (!previewRequest) {
      return;
    }

    const controller = new AbortController();

    checkDatasetFormat({
      datasetName: previewRequest.datasetName,
      hfToken: previewRequest.hfToken,
      subset: previewRequest.subset,
      split: previewRequest.split,
      isVlm: previewRequest.isVlm,
      preferLocalCache: previewRequest.preferLocalCache,
      localPath: previewRequest.localPath,
      signal: controller.signal,
    })
      .then((res) => {
        if (!controller.signal.aborted) {
          setPreviewResult({
            requestKey: previewRequest.requestKey,
            data: res,
            error: null,
          });
        }
      })
      .catch((err) => {
        if (controller.signal.aborted) {
          return;
        }
        if (err instanceof DatasetFormatError && err.status === 404) {
          clearDeletedDataset(previewRequest.datasetName);
        }
        setPreviewResult({
          requestKey: previewRequest.requestKey,
          data: null,
          error: err instanceof Error ? err.message : "Failed to load preview",
        });
      });

    return () => {
      controller.abort();
    };
  }, [previewRequest]);

  // Pre-fill mapping from suggested_mapping when data arrives (never overwriting existing entries).
  useEffect(() => {
    if (!open || !datasetName) return;
    if (!data?.requires_manual_mapping && !data?.suggested_mapping) return;
    if (Object.keys(manualMapping).length > 0) return;
    const derived = deriveDefaultMapping(
      data,
      effectiveIsVlm,
      datasetFormat,
      effectiveIsAudio,
    );
    if (Object.keys(derived).length === 0) return;
    setManualMapping(derived);
  }, [
    open,
    datasetName,
    data,
    effectiveIsVlm,
    datasetFormat,
    effectiveIsAudio,
    manualMapping,
    setManualMapping,
  ]);

  const rows = data?.preview_samples ?? EMPTY_PREVIEW_ROWS;
  const columns = data?.columns ?? EMPTY_PREVIEW_COLUMNS;

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

  const tableColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    if (!columns.length) return [];

    const dataCols: ColumnDef<Record<string, unknown>>[] = columns.map(
      (colName) => ({
        accessorKey: colName,
        header: () => (
          <div className="flex flex-col gap-2">
            <span className="font-heading text-ui-13 font-semibold tracking-tight text-foreground">
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
                {images.slice(0, 4).map(({ image, sourcePath }, index) => {
                  const mime = image.mime || "image/jpeg";
                  const src = image.data
                    ? `data:${mime};base64,${image.data}`
                    : "";
                  const width = image.width ?? 128;
                  const height = image.height ?? 128;
                  return (
                    <img
                      key={`${colName}:${sourcePath}`}
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
              <span className="text-muted-foreground/40 italic text-ui-13">
                --
              </span>
            );
          }
          const full =
            typeof value === "string" ? value : JSON.stringify(value);
          return (
            <p className="text-ui-13 leading-relaxed line-clamp-6" title={full}>
              {text}
            </p>
          );
        },
      }),
    );

    // Prepend generated system prompt column when advisor is active
    if (datasetSystemPrompt) {
      dataCols.unshift({
        id: "__system_generated",
        header: () => (
          <div className="flex flex-col gap-2">
            <span className="font-heading text-ui-13 font-semibold tracking-tight text-foreground">
              System{" "}
              <span className="text-muted-foreground font-normal">
                (generated)
              </span>
            </span>
            {mappingEnabled && (
              <Badge
                variant="outline"
                className="h-6 w-fit text-ui-10 px-2 py-0 border-dashed text-muted-foreground"
              >
                System
              </Badge>
            )}
          </div>
        ),
        cell: () => (
          <p
            className="text-ui-13 leading-relaxed line-clamp-6 text-muted-foreground italic"
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
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        className="sm:max-w-5xl w-[90vw] max-h-[88dvh] flex flex-col gap-0 p-0 overflow-hidden rounded-3xl corner-squircle"
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
                {isHfDataset
                  ? t("studio.dataset.previewLoadingHuggingFace")
                  : t("studio.dataset.previewLoading")}
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
                  value={
                    isRawFormat ? "Raw Text" : data.detected_format || "--"
                  }
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
                          className="text-ui-11 font-mono h-5"
                        >
                          {col}
                        </Badge>
                      ))}
                    </span>
                  }
                />
              </div>

              {readyForTraining && (
                <div className="mb-4 flex items-start gap-2.5 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/70 dark:text-emerald-300">
                  <HugeiconsIcon
                    icon={CheckmarkCircle02Icon}
                    className="mt-0.5 size-4 shrink-0"
                  />
                  <div className="space-y-0.5">
                    <p className="font-medium">Ready for training</p>
                    {readyDetail && <p>{readyDetail}</p>}
                  </div>
                </div>
              )}

              {data.warning && !isRawFormat && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400 mb-4 flex items-start gap-2.5">
                  <HugeiconsIcon
                    icon={AlertCircleIcon}
                    className="size-4 shrink-0 mt-0.5"
                  />
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
                <p className="text-ui-11 text-muted-foreground/60 text-center tabular-nums">
                  Showing {rows.length}
                  {data.total_rows != null &&
                    ` of ${data.total_rows.toLocaleString()}`}{" "}
                  rows
                </p>

                {mode === "preview" && mappingEnabled && (
                  <p className="mt-2 text-ui-11 text-muted-foreground/70 text-center">
                    Mapping is saved automatically. You can start training
                    anytime.
                  </p>
                )}

                {showMappingFooter && (
                  <DatasetMappingFooter
                    mappingOk={mappingOk}
                    startBlocked={startBlocked}
                    stopRequested={stopRequested}
                    startError={startError}
                    onCancel={() => handleOpenChange(false)}
                    onStartTraining={async () => {
                      const ok = await startTrainingRun();
                      if (ok) handleOpenChange(false);
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
      <span className="text-foreground text-ui-13 min-w-0">{value}</span>
    </div>
  );
}
