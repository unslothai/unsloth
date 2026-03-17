// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import mammoth from "mammoth";
import { type ReactElement, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { extractText, getDocumentProxy } from "unpdf";
import { cn } from "@/lib/utils";
import { inspectSeedDataset, inspectSeedUpload } from "../../api";
import { resolveImagePreview } from "../../utils/image-preview";
import type {
  SeedConfig,
  SeedSamplingStrategy,
  SeedSelectionType,
} from "../../types";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { HfDatasetCombobox } from "../../components/shared/hf-dataset-combobox";
import { FieldLabel } from "../shared/field-label";

const SAMPLING_OPTIONS: Array<{ value: SeedSamplingStrategy; label: string }> = [
  { value: "ordered", label: "Ordered" },
  { value: "shuffle", label: "Shuffle" },
];

const SELECTION_OPTIONS: Array<{ value: SeedSelectionType; label: string }> = [
  { value: "none", label: "None" },
  { value: "index_range", label: "Index range" },
  { value: "partition_block", label: "Partition block" },
];

const LOCAL_ACCEPT = ".csv,.json,.jsonl";
const UNSTRUCTURED_ACCEPT = ".txt,.pdf,.docx";
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const DEFAULT_CHUNK_SIZE = 1200;
const DEFAULT_CHUNK_OVERLAP = 200;
const MAX_CHUNK_SIZE = 20000;
const PREVIEW_TRUNCATE_AT = 320;

type SeedDialogProps = {
  config: SeedConfig;
  onUpdate: (patch: Partial<SeedConfig>) => void;
  open: boolean;
};

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}

function stringifyCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function isExpandablePreviewValue(value: string): boolean {
  return value.length > PREVIEW_TRUNCATE_AT;
}

function truncatePreviewValue(value: string): string {
  if (!isExpandablePreviewValue(value)) {
    return value;
  }
  return `${value.slice(0, PREVIEW_TRUNCATE_AT)}…`;
}

function getPreviewEmptyStateCopy(mode: SeedConfig["seed_source_type"]): {
  title: string;
  description: string;
} {
  if (mode === "local") {
    return {
      title: "No local preview yet",
      description: "Choose a CSV/JSON/JSONL file, then click Load to fetch 10 rows.",
    };
  }
  if (mode === "unstructured") {
    return {
      title: "No chunk preview yet",
      description:
        "Choose a TXT/PDF/DOCX file, then click Load to extract + preview chunk_text rows.",
    };
  }
  return {
    title: "No dataset preview yet",
    description: "Pick a Hugging Face dataset and click Load to fetch 10 sample rows.",
  };
}

function parseChunkNumber(
  value: string | undefined,
  fallback: number,
  min: number,
  max: number,
): number {
  const raw = value?.trim();
  if (!raw) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return fallback;
  const int = Math.floor(parsed);
  if (int < min) return min;
  if (int > max) return max;
  return int;
}

function resolveChunking(config: SeedConfig): {
  chunkSize: number;
  chunkOverlap: number;
} {
  const chunkSize = parseChunkNumber(
    config.unstructured_chunk_size,
    DEFAULT_CHUNK_SIZE,
    1,
    MAX_CHUNK_SIZE,
  );
  const chunkOverlap = parseChunkNumber(
    config.unstructured_chunk_overlap,
    DEFAULT_CHUNK_OVERLAP,
    0,
    Math.max(0, chunkSize - 1),
  );
  return { chunkSize, chunkOverlap };
}

async function fileToBase64Payload(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const value = String(reader.result ?? "");
      const parts = value.split(",");
      resolve(parts.length > 1 ? parts[1] : value);
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

async function extractUnstructuredText(file: File): Promise<string> {
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".txt")) {
    return file.text();
  }
  if (lower.endsWith(".pdf")) {
    const buffer = new Uint8Array(await file.arrayBuffer());
    const pdf = await getDocumentProxy(buffer);
    const { text } = await extractText(pdf, { mergePages: true });
    return text;
  }
  if (lower.endsWith(".docx")) {
    const arrayBuffer = await file.arrayBuffer();
    const { value } = await mammoth.extractRawText({ arrayBuffer });
    return value;
  }
  throw new Error("Unsupported unstructured file type");
}

async function toUnstructuredUploadFile(file: File): Promise<File> {
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".txt") || lower.endsWith(".md")) {
    return file;
  }

  const text = (await extractUnstructuredText(file)).trim();
  if (!text) {
    throw new Error("No text found in file.");
  }
  const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const stem = file.name.replace(/\.(pdf|docx)$/i, "") || "unstructured_seed";
  return new File([normalized], `${stem}.txt`, {
    type: "text/plain",
  });
}

export function SeedDialog({ config, onUpdate, open }: SeedDialogProps): ReactElement {
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [isInspecting, setIsInspecting] = useState(false);
  const advancedOpen = config.advancedOpen === true;
  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[]>([]);
  const [expandedPreviewRows, setExpandedPreviewRows] = useState<Record<number, boolean>>({});
  const [localFile, setLocalFile] = useState<File | null>(null);
  const [unstructuredFile, setUnstructuredFile] = useState<File | null>(null);

  const mode = config.seed_source_type ?? "hf";
  const previewEmpty = getPreviewEmptyStateCopy(mode);

  useEffect(() => {
    setInspectError(null);
    setLocalFile(null);
    setUnstructuredFile(null);
  }, [mode]);

  useEffect(() => {
    setPreviewRows(config.seed_preview_rows ?? []);
    setExpandedPreviewRows({});
  }, [config.seed_preview_rows]);

  const samplingId = `${config.id}-sampling`;
  const selectionId = `${config.id}-selection`;
  const tokenId = `${config.id}-hf-token`;
  const datasetId = `${config.id}-hf-dataset`;
  const chunkSizeId = `${config.id}-chunk-size`;
  const chunkOverlapId = `${config.id}-chunk-overlap`;
  const [lastLoadedKey, setLastLoadedKey] = useState<string | null>(null);
  const wasOpenRef = useRef(open);

  const getCurrentLoadKey = useCallback((): string | null => {
    if (mode === "hf") {
      const dataset = config.hf_repo_id.trim();
      if (!dataset) return null;
      const token = config.hf_token?.trim() ?? "";
      return `hf:${dataset}|${token}`;
    }
    if (mode === "local") {
      if (!localFile) return null;
      return `local:${localFile.name}|${localFile.size}|${localFile.lastModified}`;
    }
    if (!unstructuredFile) return null;
    const { chunkSize, chunkOverlap } = resolveChunking(config);
    return `unstructured:${unstructuredFile.name}|${unstructuredFile.size}|${unstructuredFile.lastModified}|${chunkSize}|${chunkOverlap}`;
  }, [
    config,
    localFile,
    mode,
    unstructuredFile,
  ]);

  const loadSeedMetadata = useCallback(async (opts?: { silent?: boolean }): Promise<boolean> => {
    const loadKey = getCurrentLoadKey();
    if (!opts?.silent) {
      setInspectError(null);
    }
    setIsInspecting(true);
    try {
      if (mode === "hf") {
        const datasetName = config.hf_repo_id.trim();
        if (!datasetName) {
          throw new Error("Dataset repo is required.");
        }
        const response = await inspectSeedDataset({
          dataset_name: datasetName,
          hf_token: config.hf_token?.trim() || undefined,
          split: config.hf_split?.trim() || undefined,
          subset: config.hf_subset?.trim() || undefined,
          preview_size: 10,
        });
        onUpdate({
          hf_path: response.resolved_path,
          seed_columns: response.columns,
          seed_drop_columns: (config.seed_drop_columns ?? []).filter((name) =>
            response.columns.includes(name),
          ),
          seed_preview_rows: response.preview_rows ?? [],
          hf_split: response.split ?? "",
          hf_subset: response.subset ?? "",
          local_file_name: "",
          unstructured_file_name: "",
        });
        setPreviewRows(response.preview_rows ?? []);
        setLastLoadedKey(loadKey);
        return true;
      }

      if (mode === "local") {
        if (!localFile) {
          throw new Error("Select a local CSV/JSON/JSONL file first.");
        }
        if (localFile.size > MAX_UPLOAD_BYTES) {
          throw new Error("File too large (max 50MB).");
        }
        const payload = await fileToBase64Payload(localFile);
        const response = await inspectSeedUpload({
          filename: localFile.name,
          content_base64: payload,
          preview_size: 10,
        });
        onUpdate({
          hf_path: response.resolved_path,
          seed_columns: response.columns,
          seed_drop_columns: (config.seed_drop_columns ?? []).filter((name) =>
            response.columns.includes(name),
          ),
          seed_preview_rows: response.preview_rows ?? [],
          hf_repo_id: "",
          hf_subset: "",
          hf_split: "",
          local_file_name: localFile.name,
          unstructured_file_name: "",
        });
        setPreviewRows(response.preview_rows ?? []);
        setLastLoadedKey(loadKey);
        return true;
      }

      if (!unstructuredFile) {
        throw new Error("Select a PDF/DOCX/TXT file first.");
      }
      if (unstructuredFile.size > MAX_UPLOAD_BYTES) {
        throw new Error("File too large (max 50MB).");
      }

      const { chunkSize, chunkOverlap } = resolveChunking(config);
      const uploadFile = await toUnstructuredUploadFile(unstructuredFile);
      if (uploadFile.size > MAX_UPLOAD_BYTES) {
        throw new Error("Processed text is too large (max 50MB).");
      }
      const payload = await fileToBase64Payload(uploadFile);
      const response = await inspectSeedUpload({
        filename: uploadFile.name,
        content_base64: payload,
        preview_size: 10,
        seed_source_type: "unstructured",
        unstructured_chunk_size: chunkSize,
        unstructured_chunk_overlap: chunkOverlap,
      });
      onUpdate({
        hf_path: response.resolved_path,
        seed_columns: response.columns,
        seed_drop_columns: (config.seed_drop_columns ?? []).filter((name) =>
          response.columns.includes(name),
        ),
        seed_preview_rows: response.preview_rows ?? [],
        hf_repo_id: "",
        hf_subset: "",
        hf_split: "",
        local_file_name: "",
        unstructured_file_name: unstructuredFile.name,
      });
      setPreviewRows(response.preview_rows ?? []);
      setLastLoadedKey(loadKey);
      return true;
    } catch (error) {
      if (!opts?.silent) {
        setInspectError(getErrorMessage(error, "Failed to load seed metadata."));
      }
      setPreviewRows([]);
      return false;
    } finally {
      setIsInspecting(false);
    }
  }, [
    config,
    getCurrentLoadKey,
    localFile,
    mode,
    onUpdate,
    unstructuredFile,
  ]);

  useEffect(() => {
    const wasOpen = wasOpenRef.current;
    wasOpenRef.current = open;
    if (!wasOpen || open || isInspecting) {
      return;
    }
    const key = getCurrentLoadKey();
    if (!key || key === lastLoadedKey) {
      return;
    }
    void loadSeedMetadata({ silent: true });
  }, [getCurrentLoadKey, isInspecting, lastLoadedKey, loadSeedMetadata, open]);

  const previewColumns = useMemo(() => {
    const loadedColumns = config.seed_columns ?? [];
    if (loadedColumns.length > 0) return loadedColumns;
    if (previewRows[0]) return Object.keys(previewRows[0]);
    return [];
  }, [config.seed_columns, previewRows]);
  const selectedSeedDropColumns = useMemo(
    () => (config.seed_drop_columns ?? []).filter((name) => name.trim().length > 0),
    [config.seed_drop_columns],
  );
  const selectedSeedDropSet = useMemo(
    () => new Set(selectedSeedDropColumns),
    [selectedSeedDropColumns],
  );
  const rowHasExpandableText = useCallback(
    (row: Record<string, unknown>): boolean =>
      previewColumns.some((columnName) => {
        if (resolveImagePreview(row[columnName])) {
          return false;
        }
        return isExpandablePreviewValue(stringifyCell(row[columnName]));
      }),
    [previewColumns],
  );

  return (
    <Tabs defaultValue="config" className="w-full min-w-0">
      <TabsList className="w-full">
        <TabsTrigger value="config">Config</TabsTrigger>
        <TabsTrigger value="preview">Preview</TabsTrigger>
      </TabsList>

      <TabsContent value="config" className="min-w-0 pt-3">
        <div className="space-y-4">
          {mode === "hf" && (
            <>
              <div className="grid gap-2">
                <FieldLabel
                  label="Dataset"
                  htmlFor={datasetId}
                  hint="Hugging Face dataset repo id (org/repo)."
                />
                <div className="flex items-center gap-2">
                  <HfDatasetCombobox
                    inputId={datasetId}
                    className="flex-1"
                    value={config.hf_repo_id}
                    accessToken={config.hf_token?.trim() || undefined}
                    placeholder="org/repo"
                    onValueChange={(nextValue) =>
                      onUpdate({
                        hf_repo_id: nextValue,
                        hf_subset: "",
                        hf_split: "",
                        hf_path: "",
                        seed_columns: [],
                        seed_drop_columns: [],
                        seed_preview_rows: [],
                      })
                    }
                  />
                  <Button
                    type="button"
                    variant="outline"
                    className="nodrag shrink-0"
                    onClick={() => void loadSeedMetadata()}
                    disabled={isInspecting || !config.hf_repo_id.trim()}
                  >
                    {isInspecting ? "Loading..." : "Load"}
                  </Button>
                </div>
              </div>

              <div className="grid gap-2">
                <FieldLabel
                  label="HF token (optional)"
                  htmlFor={tokenId}
                  hint="Only needed for private/gated datasets."
                />
                <Input
                  id={tokenId}
                  className="nodrag"
                  placeholder="hf_..."
                  value={config.hf_token ?? ""}
                  onChange={(event) => onUpdate({ hf_token: event.target.value })}
                />
              </div>

            </>
          )}

          {mode === "local" && (
            <div className="grid gap-2">
              <FieldLabel
                label="Structured file"
                hint="Upload CSV, JSON, or JSONL seed file."
              />
              <div className="flex items-center gap-2">
                <Input
                  className="nodrag flex-1"
                  type="file"
                  accept={LOCAL_ACCEPT}
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null;
                    setLocalFile(file);
                    onUpdate({
                      hf_path: "",
                      seed_columns: [],
                      seed_drop_columns: [],
                      seed_preview_rows: [],
                      local_file_name: file?.name ?? "",
                    });
                  }}
                />
                <Button
                  type="button"
                  variant="outline"
                  className="nodrag shrink-0"
                  onClick={() => void loadSeedMetadata()}
                  disabled={isInspecting || !localFile}
                >
                  {isInspecting ? "Loading..." : "Load"}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Upload-only. Max 50MB.
              </p>
              {(localFile?.name || config.local_file_name?.trim()) && (
                <p className="text-xs text-muted-foreground">
                  Selected: {localFile?.name ?? config.local_file_name?.trim()}
                </p>
              )}
            </div>
          )}

          {mode === "unstructured" && (
            <div className="grid gap-2">
              <FieldLabel
                label="Unstructured file"
                hint="Upload PDF, DOCX, or TXT. We chunk text into seed rows."
              />
              <div className="flex items-center gap-2">
                <Input
                  className="nodrag flex-1"
                  type="file"
                  accept={UNSTRUCTURED_ACCEPT}
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null;
                    setUnstructuredFile(file);
                    onUpdate({
                      hf_path: "",
                      seed_columns: [],
                      seed_drop_columns: [],
                      seed_preview_rows: [],
                      unstructured_file_name: file?.name ?? "",
                    });
                  }}
                />
                <Button
                  type="button"
                  variant="outline"
                  className="nodrag shrink-0"
                  onClick={() => void loadSeedMetadata()}
                  disabled={isInspecting || !unstructuredFile}
                >
                  {isInspecting ? "Loading..." : "Load"}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                File is converted to text, then chunked server-side into chunk_text rows. Max 50MB.
              </p>
              {(unstructuredFile?.name ||
                config.unstructured_file_name?.trim()) && (
                <p className="text-xs text-muted-foreground">
                  Selected:{" "}
                  {unstructuredFile?.name ?? config.unstructured_file_name?.trim()}
                </p>
              )}
            </div>
          )}

          {inspectError && <p className="text-xs text-red-600">{inspectError}</p>}

          {mode !== "unstructured" && (
            <div className="space-y-2 rounded-xl corner-squircle border border-border/60 p-3">
              <FieldLabel
                label="Drop specific seed columns"
                hint="Dropped columns stay usable in prompts/expressions but are omitted from final dataset."
              />
              {previewColumns.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  Load columns to select which seed fields to drop.
                </p>
              ) : (
                <div className="grid gap-2 sm:grid-cols-2">
                  {previewColumns.map((columnName) => {
                    const checked = selectedSeedDropSet.has(columnName);
                    return (
                      <label
                        key={columnName}
                        className="flex cursor-pointer items-center gap-2 rounded-md border border-border/60 px-2 py-1.5 text-xs"
                      >
                        <Checkbox
                          checked={checked}
                          onCheckedChange={(value) => {
                            const isChecked = value === true;
                            const next = isChecked
                              ? Array.from(new Set([...selectedSeedDropColumns, columnName]))
                              : selectedSeedDropColumns.filter((name) => name !== columnName);
                            onUpdate({ seed_drop_columns: next });
                          }}
                        />
                        <span className="truncate">{columnName}</span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          <Collapsible
            open={advancedOpen}
            onOpenChange={(openState) => onUpdate({ advancedOpen: openState })}
          >
            <CollapsibleTrigger asChild={true}>
              <CollapsibleSectionTriggerButton
                label="Advanced source options"
                open={advancedOpen}
              />
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2 space-y-3">
              <div className="grid gap-2">
                <FieldLabel
                  label="Sampling strategy"
                  htmlFor={samplingId}
                  hint="Ordered keeps row order. Shuffle randomizes sampled rows."
                />
                <Select
                  value={config.sampling_strategy}
                  onValueChange={(value) =>
                    onUpdate({ sampling_strategy: value as SeedSamplingStrategy })
                  }
                >
                  <SelectTrigger className="nodrag w-full" id={samplingId}>
                    <SelectValue placeholder="Select sampling" />
                  </SelectTrigger>
                  <SelectContent>
                    {SAMPLING_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid gap-2">
                <FieldLabel
                  label="Selection strategy"
                  htmlFor={selectionId}
                  hint="Select all, a row range, or partition block."
                />
                <Select
                  value={config.selection_type}
                  onValueChange={(value) =>
                    onUpdate({ selection_type: value as SeedSelectionType })
                  }
                >
                  <SelectTrigger className="nodrag w-full" id={selectionId}>
                    <SelectValue placeholder="Select selection" />
                  </SelectTrigger>
                  <SelectContent>
                    {SELECTION_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {mode === "unstructured" && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="grid gap-2">
                    <FieldLabel
                      label="Chunk size"
                      htmlFor={chunkSizeId}
                      hint="Characters per chunk."
                    />
                    <Input
                      id={chunkSizeId}
                      className="nodrag"
                      inputMode="numeric"
                      value={config.unstructured_chunk_size ?? String(DEFAULT_CHUNK_SIZE)}
                      onChange={(event) =>
                        onUpdate({ unstructured_chunk_size: event.target.value })
                      }
                    />
                  </div>
                  <div className="grid gap-2">
                    <FieldLabel
                      label="Chunk overlap"
                      htmlFor={chunkOverlapId}
                      hint="Shared chars between adjacent chunks."
                    />
                    <Input
                      id={chunkOverlapId}
                      className="nodrag"
                      inputMode="numeric"
                      value={
                        config.unstructured_chunk_overlap ??
                        String(DEFAULT_CHUNK_OVERLAP)
                      }
                      onChange={(event) =>
                        onUpdate({ unstructured_chunk_overlap: event.target.value })
                      }
                    />
                  </div>
                </div>
              )}

              {config.selection_type === "index_range" && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="grid gap-2">
                    <FieldLabel label="Start" hint="Inclusive start row index for index_range." />
                    <Input
                      className="nodrag"
                      inputMode="numeric"
                      value={config.selection_start ?? ""}
                      onChange={(event) => onUpdate({ selection_start: event.target.value })}
                    />
                  </div>
                  <div className="grid gap-2">
                    <FieldLabel label="End" hint="Inclusive end row index for index_range." />
                    <Input
                      className="nodrag"
                      inputMode="numeric"
                      value={config.selection_end ?? ""}
                      onChange={(event) => onUpdate({ selection_end: event.target.value })}
                    />
                  </div>
                </div>
              )}

              {config.selection_type === "partition_block" && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="grid gap-2">
                    <FieldLabel label="Index" hint="Partition index to load." />
                    <Input
                      className="nodrag"
                      inputMode="numeric"
                      value={config.selection_index ?? ""}
                      onChange={(event) => onUpdate({ selection_index: event.target.value })}
                    />
                  </div>
                  <div className="grid gap-2">
                    <FieldLabel label="Partitions" hint="Total number of partitions." />
                    <Input
                      className="nodrag"
                      inputMode="numeric"
                      value={config.selection_num_partitions ?? ""}
                      onChange={(event) =>
                        onUpdate({ selection_num_partitions: event.target.value })
                      }
                    />
                  </div>
                </div>
              )}
            </CollapsibleContent>
          </Collapsible>
        </div>
      </TabsContent>

      <TabsContent value="preview" className="min-w-0 pt-3">
        <div className="space-y-4">
          {previewRows.length === 0 ? (
            <div className="flex w-full items-center justify-center">
              <Empty className="max-w-lg">
                <EmptyHeader>
                  <EmptyTitle>{previewEmpty.title}</EmptyTitle>
                  <EmptyDescription>
                    {previewEmpty.description}
                  </EmptyDescription>
                </EmptyHeader>
                <EmptyContent className="text-xs text-muted-foreground">
                  Preview appears here after loading source metadata.
                </EmptyContent>
              </Empty>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">
                Loaded columns: {previewColumns.join(", ") || "None"}
              </div>
              <div className="max-h-[360px] overflow-y-auto overflow-x-hidden rounded-xl corner-squircle border border-border/60">
                <Table className="corner-squircle min-w-max">
                  <TableHeader>
                    <TableRow>
                      {previewColumns.map((col) => (
                        <TableHead key={col} className="whitespace-nowrap">
                          {col}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {previewRows.map((row, rowIdx) => (
                      <TableRow
                        key={`row-${rowIdx}`}
                        className={cn(
                          rowHasExpandableText(row) && "cursor-pointer hover:bg-primary/[0.06]",
                          expandedPreviewRows[rowIdx] && "bg-primary/[0.05]",
                        )}
                        onClick={() => {
                          const canExpand = rowHasExpandableText(row);
                          if (!canExpand) {
                            return;
                          }
                          setExpandedPreviewRows((current) => ({
                            ...current,
                            [rowIdx]: !current[rowIdx],
                          }));
                        }}
                      >
                        {previewColumns.map((col) => (
                          <TableCell
                            key={`${rowIdx}-${col}`}
                            className="max-w-[260px] whitespace-pre-wrap break-words text-xs"
                          >
                            {(() => {
                              const imagePreview = resolveImagePreview(row[col]);
                              if (imagePreview?.kind === "ready") {
                                return (
                                  <img
                                    src={imagePreview.src}
                                    alt={`${col} preview`}
                                    loading="lazy"
                                    className="h-20 w-auto max-w-[220px] rounded-md border border-border/60 bg-muted/20 object-contain"
                                  />
                                );
                              }
                              if (imagePreview?.kind === "too_large") {
                                return "Image too large to preview";
                              }
                              const value = stringifyCell(row[col]);
                              const rowHasExpandableCell = rowHasExpandableText(row);
                              const rowExpanded = Boolean(expandedPreviewRows[rowIdx]);
                              return rowHasExpandableCell && !rowExpanded
                                ? truncatePreviewValue(value)
                                : value;
                            })()}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}
        </div>
      </TabsContent>
    </Tabs>
  );
}
