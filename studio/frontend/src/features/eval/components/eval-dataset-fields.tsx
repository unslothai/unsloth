// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Button } from "@/components/ui/button";
import { InputGroupAddon } from "@/components/ui/input-group";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  useDebouncedValue,
  useHfDatasetSearch,
  useHfDatasetSplits,
  useInfiniteScroll,
} from "@/hooks";
import { uploadTrainingDataset } from "@/features/training";
import {
  checkDatasetFormat,
  listLocalDatasets,
} from "@/features/training/api/datasets-api";
import type { LocalDatasetInfo } from "@/features/training/types/datasets";
import {
  CloudUploadIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface EvalDatasetValue {
  isLocal: boolean;
  name: string;   // HF repo id (when !isLocal)
  path: string;   // local path / uploaded stored path (when isLocal)
  split: string;
  subset: string; // "" = none
  inputColumn: string;
  referenceColumn: string;
}

// ── Component ──────────────────────────────────────────────────────────────────

export function EvalDatasetFields({
  hfToken,
  value,
  onChange,
}: {
  hfToken: string;
  value: EvalDatasetValue;
  onChange: (next: EvalDatasetValue) => void;
}): React.JSX.Element {
  const update = (partial: Partial<EvalDatasetValue>) =>
    onChange({ ...value, ...partial });

  // ── Internal state ──────────────────────────────────────────────────────────
  const [searchQuery, setSearchQuery] = useState("");
  const [detectedColumns, setDetectedColumns] = useState<string[]>([]);
  const [previewSample, setPreviewSample] = useState<Record<string, unknown> | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [localDatasets, setLocalDatasets] = useState<LocalDatasetInfo[]>([]);
  const [isLoadingLocal, setIsLoadingLocal] = useState(true);
  const [localDatasetsError, setLocalDatasetsError] = useState<string | null>(
    null,
  );

  const debouncedQuery = useDebouncedValue(searchQuery);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const comboboxAnchorRef = useRef<HTMLDivElement>(null);

  // ── HF dataset search ───────────────────────────────────────────────────────
  const {
    results,
    isLoading,
    isLoadingMore,
    fetchMore,
    error: hfSearchError,
  } = useHfDatasetSearch(debouncedQuery, {
    accessToken: hfToken || undefined,
    enabled: !value.isLocal,
  });

  const resultIds = useMemo(() => {
    const ids = results.map((r) => r.id);
    if (value.name && !ids.includes(value.name)) {
      ids.push(value.name);
    }
    return ids;
  }, [results, value.name]);

  const hfResultById = useMemo(() => {
    const map = new Map(results.map((r) => [r.id, r]));
    return map;
  }, [results]);

  const { scrollRef, sentinelRef } = useInfiniteScroll(fetchMore, results.length);

  // ── Subset + split ──────────────────────────────────────────────────────────
  const {
    subsets,
    splits,
    isLoading: splitsLoading,
  } = useHfDatasetSplits(value.isLocal ? null : value.name || null, value.subset || null, {
    accessToken: hfToken || undefined,
  });

  // Auto-correct split when splits list changes
  useEffect(() => {
    if (!splits.length) return;
    if (splits.includes(value.split)) return;
    update({ split: splits.includes("train") ? "train" : splits[0] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splits]);

  // ── File upload ─────────────────────────────────────────────────────────────
  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    try {
      const uploaded = await uploadTrainingDataset(file);
      update({ path: uploaded.stored_path });
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsUploading(false);
    }
  }

  // ── Column detection ────────────────────────────────────────────────────────
  async function detectColumns() {
    const ref = value.isLocal ? value.path : value.name;
    if (!ref.trim()) {
      setDetectError("Pick or enter a dataset first.");
      return;
    }
    setDetecting(true);
    setDetectError(null);
    try {
      const res = await checkDatasetFormat({
        datasetName: ref.trim(),
        hfToken: hfToken || null,
        subset: value.subset || null,
        split: value.split || "train",
      });
      const cols = res.columns ?? [];
      setDetectedColumns(cols);
      setPreviewSample(res.preview_samples?.[0] ?? null);
      if (cols.length) {
        // Reconcile the mapping against the dataset's actual columns so a stale
        // selection from a previous dataset (e.g. "html"/"json") doesn't linger
        // as a phantom option that doesn't exist here.
        const next: Partial<EvalDatasetValue> = {};
        if (!value.inputColumn || !cols.includes(value.inputColumn)) {
          next.inputColumn = cols[0];
        }
        if (!value.referenceColumn || !cols.includes(value.referenceColumn)) {
          next.referenceColumn = cols[1] ?? cols[0];
        }
        if (Object.keys(next).length) update(next);
      }
    } catch (err) {
      setDetectError(err instanceof Error ? err.message : String(err));
    } finally {
      setDetecting(false);
    }
  }

  // Auto-detect columns shortly after the dataset / split / subset changes, so
  // the Input/Output dropdowns populate without a manual click.
  useEffect(() => {
    const ref = (value.isLocal ? value.path : value.name).trim();
    if (!ref) {
      setDetectedColumns([]);
      setPreviewSample(null);
      setDetectError(null);
      return;
    }
    const id = setTimeout(() => {
      void detectColumns();
    }, 500);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value.isLocal, value.path, value.name, value.split, value.subset, hfToken]);

  // Load locally-available datasets (Data Recipe outputs) on mount so the
  // user can pick from a list — mirrors the training view's flow.
  useEffect(() => {
    let cancelled = false;
    setIsLoadingLocal(true);
    listLocalDatasets()
      .then((r) => {
        if (!cancelled) setLocalDatasets(r.datasets);
      })
      .catch((e) => {
        if (!cancelled)
          setLocalDatasetsError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setIsLoadingLocal(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Local options shown in the Select = recipe outputs, plus a synthetic
  // "Uploaded:" entry when the current path was uploaded (and isn't in the
  // recipe list) so the Select still displays it.
  const localOptions = useMemo(() => {
    const opts = localDatasets.slice();
    if (
      value.isLocal &&
      value.path &&
      !opts.find((d) => d.path === value.path)
    ) {
      opts.unshift({
        id: "_uploaded_",
        label: `Uploaded: ${value.path.split("/").pop() || value.path}`,
        path: value.path,
        rows: null,
        updated_at: null,
        metadata: null,
      });
    }
    return opts;
  }, [localDatasets, value.isLocal, value.path]);

  // Detected columns plus the current value (so a persisted/typed selection
  // still shows while detection is pending or if it isn't in the dataset).
  const columnOptions = (current: string) =>
    current && !detectedColumns.includes(current)
      ? [current, ...detectedColumns]
      : detectedColumns;

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-3">
      {/* 1. Source toggle */}
      <Tabs
        value={value.isLocal ? "local" : "hf"}
        onValueChange={(v) =>
          update({ isLocal: v === "local", name: "", path: "" })
        }
      >
        <TabsList>
          <TabsTrigger value="hf">Hugging Face</TabsTrigger>
          <TabsTrigger value="local">Local</TabsTrigger>
        </TabsList>
      </Tabs>

      {/* 2a. HF mode */}
      {!value.isLocal && (
        <div className="flex flex-col gap-3">
          {/* HF dataset search combobox */}
          <div className="flex flex-col gap-1.5">
            <Label className="text-xs font-medium text-muted-foreground">
              Dataset
            </Label>
            <div ref={comboboxAnchorRef} className="min-w-0">
              <Combobox
                items={resultIds}
                filteredItems={resultIds}
                filter={null}
                value={value.name || null}
                onValueChange={(id) =>
                  update({ name: id ?? "", subset: "", split: "train" })
                }
                onInputValueChange={setSearchQuery}
                itemToStringValue={(id) => id}
                autoHighlight={true}
              >
                <ComboboxInput
                  placeholder="Search Hugging Face datasets..."
                  className="w-full leading-5"
                >
                  <InputGroupAddon>
                    <HugeiconsIcon icon={Search01Icon} className="size-4" />
                  </InputGroupAddon>
                </ComboboxInput>
                <ComboboxContent anchor={comboboxAnchorRef}>
                  {isLoading ? (
                    <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
                      <Spinner className="size-4" /> Searching…
                    </div>
                  ) : (
                    <ComboboxEmpty>No datasets found</ComboboxEmpty>
                  )}
                  {hfSearchError && (
                    <p className="px-3 py-2 text-xs text-destructive">
                      {hfSearchError}
                    </p>
                  )}
                  <div
                    ref={scrollRef}
                    className="max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]"
                  >
                    <ComboboxList className="p-1 !max-h-none !overflow-visible">
                      {(id: string) => {
                        const meta = hfResultById.get(id);
                        return (
                          <ComboboxItem key={id} value={id} className="gap-2">
                            <span className="block min-w-0 flex-1 truncate">
                              {id}
                            </span>
                            {meta?.totalExamples != null && (
                              <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                                {meta.totalExamples.toLocaleString()} rows
                              </span>
                            )}
                          </ComboboxItem>
                        );
                      }}
                    </ComboboxList>
                    <div ref={sentinelRef} className="h-px" />
                    {isLoadingMore && (
                      <div className="flex items-center justify-center py-2">
                        <Spinner className="size-3.5 text-muted-foreground" />
                      </div>
                    )}
                  </div>
                </ComboboxContent>
              </Combobox>
            </div>
          </div>

          {/* Subset + Split selectors */}
          <div className="grid grid-cols-2 gap-3">
            {subsets.length > 1 && (
              <div className="flex flex-col gap-1.5">
                <Label className="text-xs font-medium text-muted-foreground">
                  Subset
                </Label>
                <Select
                  value={value.subset}
                  onValueChange={(s) => update({ subset: s, split: "train" })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select subset" />
                  </SelectTrigger>
                  <SelectContent>
                    {subsets.map((s) => (
                      <SelectItem key={s} value={s}>
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="flex flex-col gap-1.5">
              <Label className="text-xs font-medium text-muted-foreground">
                Split
              </Label>
              <Select
                value={value.split}
                onValueChange={(s) => update({ split: s })}
                disabled={splitsLoading || !value.name}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder={splitsLoading ? "Loading…" : "Select split"} />
                </SelectTrigger>
                <SelectContent>
                  {splits.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      )}

      {/* 2b. Local mode */}
      {value.isLocal && (
        <div className="flex flex-col gap-3">
          {/* Local datasets list (Data Recipe outputs + any uploaded file) */}
          <div className="flex flex-col gap-1.5">
            <Label className="text-xs font-medium text-muted-foreground">
              Local dataset
            </Label>
            <Select
              value={value.path || undefined}
              onValueChange={(p) => update({ path: p })}
              disabled={isLoadingLocal && localOptions.length === 0}
            >
              <SelectTrigger className="w-full">
                <SelectValue
                  placeholder={
                    isLoadingLocal
                      ? "Loading local datasets…"
                      : localOptions.length === 0
                        ? "No local datasets — upload a file below"
                        : "Select a local dataset…"
                  }
                />
              </SelectTrigger>
              <SelectContent>
                {localOptions.map((d) => (
                  <SelectItem key={d.path} value={d.path}>
                    {d.label}
                    {d.rows != null
                      ? ` · ${d.rows.toLocaleString()} rows`
                      : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {localDatasetsError && (
              <p className="text-xs text-destructive">{localDatasetsError}</p>
            )}
          </div>

          {/* Upload — always available; uploaded files appear in the Select. */}
          <button
            type="button"
            className="flex w-full cursor-pointer items-center gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-3.5 py-3 text-left transition-colors hover:border-indigo-500/50 hover:bg-indigo-500/5 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={isUploading}
            onClick={() => fileInputRef.current?.click()}
          >
            {isUploading ? (
              <Spinner className="size-4 shrink-0 text-indigo-500" />
            ) : (
              <HugeiconsIcon
                icon={CloudUploadIcon}
                className="pointer-events-none size-4 shrink-0 text-indigo-500"
              />
            )}
            <span className="pointer-events-none min-w-0">
              <span className="block text-xs font-medium text-foreground">
                {isUploading ? "Uploading…" : "Or upload a file"}
              </span>
              <span className="mt-0.5 block truncate text-[10px] text-muted-foreground">
                CSV, JSONL, JSON, Parquet
              </span>
            </span>
          </button>

          {uploadError && (
            <p className="text-xs text-destructive">{uploadError}</p>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.jsonl,.json,.parquet"
            className="hidden"
            onChange={(e) => {
              void handleFileChange(e);
            }}
          />
        </div>
      )}

      {/* 3. Column mapping */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <Label className="text-xs font-medium text-muted-foreground">
            Column mapping
          </Label>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            disabled={detecting}
            onClick={() => {
              void detectColumns();
            }}
          >
            {detecting ? (
              <>
                <Spinner className="size-3.5" />
                Detecting…
              </>
            ) : (
              "Refresh columns"
            )}
          </Button>
        </div>

        {detectError && (
          <p className="text-xs text-destructive">{detectError}</p>
        )}

        <div className="grid grid-cols-2 gap-3">
          {/* Input column */}
          <div className="flex flex-col gap-1.5">
            <Label className="text-xs font-medium text-muted-foreground">
              Input column
            </Label>
            <Select
              value={value.inputColumn || undefined}
              onValueChange={(col) => update({ inputColumn: col })}
              disabled={detectedColumns.length === 0 && !value.inputColumn}
            >
              <SelectTrigger className="w-full">
                <SelectValue
                  placeholder={detecting ? "Detecting…" : "Select column"}
                />
              </SelectTrigger>
              <SelectContent>
                {columnOptions(value.inputColumn).map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Output column */}
          <div className="flex flex-col gap-1.5">
            <Label className="text-xs font-medium text-muted-foreground">
              Output column
            </Label>
            <Select
              value={value.referenceColumn || undefined}
              onValueChange={(col) => update({ referenceColumn: col })}
              disabled={detectedColumns.length === 0 && !value.referenceColumn}
            >
              <SelectTrigger className="w-full">
                <SelectValue
                  placeholder={detecting ? "Detecting…" : "Select column"}
                />
              </SelectTrigger>
              <SelectContent>
                {columnOptions(value.referenceColumn).map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Preview sample */}
      {previewSample && (
        <pre className="max-h-40 overflow-auto rounded-md bg-muted p-2 font-mono text-xs">
          {JSON.stringify(previewSample, null, 2)}
        </pre>
      )}
    </div>
  );
}
