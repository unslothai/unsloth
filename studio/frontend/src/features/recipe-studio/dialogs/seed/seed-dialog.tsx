import { Button } from "@/components/ui/button";
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
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import mammoth from "mammoth";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import { extractText, getDocumentProxy } from "unpdf";
import { inspectSeedDataset, inspectSeedUpload } from "../../api";
import type {
  SeedConfig,
  SeedSamplingStrategy,
  SeedSelectionType,
} from "../../types";
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
const CHUNK_SIZE = 1200;
const CHUNK_OVERLAP = 200;
const UNSTRUCTURED_SPLITTER = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

type SeedDialogProps = {
  config: SeedConfig;
  onUpdate: (patch: Partial<SeedConfig>) => void;
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

async function chunkText(input: string): Promise<string[]> {
  const text = input.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  if (!text) return [];

  const chunks = await UNSTRUCTURED_SPLITTER.splitText(text);
  return chunks.map((chunk) => chunk.trim()).filter(Boolean);
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

export function SeedDialog({ config, onUpdate }: SeedDialogProps): ReactElement {
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [isInspecting, setIsInspecting] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[]>([]);
  const [localFile, setLocalFile] = useState<File | null>(null);
  const [unstructuredFile, setUnstructuredFile] = useState<File | null>(null);

  const mode = config.seed_source_type ?? "hf";

  useEffect(() => {
    setInspectError(null);
    setPreviewRows([]);
    setLocalFile(null);
    setUnstructuredFile(null);
  }, [mode]);

  const samplingId = `${config.id}-sampling`;
  const selectionId = `${config.id}-selection`;
  const tokenId = `${config.id}-hf-token`;
  const subsetId = `${config.id}-hf-subset`;
  const splitId = `${config.id}-hf-split`;
  const datasetId = `${config.id}-hf-dataset`;

  async function loadSeedMetadata(): Promise<void> {
    setInspectError(null);
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
          subset: config.hf_subset || undefined,
          split: config.hf_split || "train",
          preview_size: 10,
        });
        onUpdate({
          hf_path: response.resolved_path,
          seed_columns: response.columns,
          hf_split: response.split ?? config.hf_split ?? "",
          hf_subset: response.subset ?? config.hf_subset ?? "",
          local_file_name: "",
          unstructured_file_name: "",
        });
        setPreviewRows(response.preview_rows ?? []);
        return;
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
          hf_repo_id: "",
          hf_subset: "",
          hf_split: "",
          local_file_name: localFile.name,
          unstructured_file_name: "",
        });
        setPreviewRows(response.preview_rows ?? []);
        return;
      }

      if (!unstructuredFile) {
        throw new Error("Select a PDF/DOCX/TXT file first.");
      }
      if (unstructuredFile.size > MAX_UPLOAD_BYTES) {
        throw new Error("File too large (max 50MB).");
      }

      const text = await extractUnstructuredText(unstructuredFile);
      const chunks = await chunkText(text);
      if (chunks.length === 0) {
        throw new Error("No text found in file.");
      }
      const jsonl = chunks
        .map((chunk) => JSON.stringify({ chunk_text: chunk }))
        .join("\n");
      const stem =
        unstructuredFile.name.replace(/\.(pdf|docx|txt)$/i, "") ||
        "unstructured_seed";
      const jsonlFile = new File([jsonl], `${stem}.jsonl`, {
        type: "application/json",
      });

      const payload = await fileToBase64Payload(jsonlFile);
      const response = await inspectSeedUpload({
        filename: jsonlFile.name,
        content_base64: payload,
        preview_size: 10,
      });
      onUpdate({
        hf_path: response.resolved_path,
        seed_columns: response.columns,
        hf_repo_id: "",
        hf_subset: "",
        hf_split: "",
        local_file_name: "",
        unstructured_file_name: unstructuredFile.name,
      });
      setPreviewRows(response.preview_rows ?? []);
    } catch (error) {
      setInspectError(getErrorMessage(error, "Failed to load seed metadata."));
      setPreviewRows([]);
    } finally {
      setIsInspecting(false);
    }
  }

  const previewColumns = useMemo(() => {
    const loadedColumns = config.seed_columns ?? [];
    if (loadedColumns.length > 0) return loadedColumns;
    if (previewRows[0]) return Object.keys(previewRows[0]);
    return [];
  }, [config.seed_columns, previewRows]);

  const canLoad =
    mode === "hf"
      ? Boolean(config.hf_repo_id.trim())
      : mode === "local"
        ? Boolean(localFile)
        : Boolean(unstructuredFile);

  return (
    <Tabs defaultValue="config" className="w-full">
      <TabsList className="w-full">
        <TabsTrigger value="config">Config</TabsTrigger>
        <TabsTrigger value="preview">Preview</TabsTrigger>
      </TabsList>

      <TabsContent value="config" className="pt-3">
        <div className="space-y-4">
          {mode === "hf" && (
            <>
              <div className="grid gap-2">
                <FieldLabel
                  label="Dataset"
                  htmlFor={datasetId}
                  hint="Hugging Face dataset repo id (org/repo)."
                />
                <Input
                  id={datasetId}
                  className="nodrag"
                  placeholder="org/repo"
                  value={config.hf_repo_id}
                  onChange={(event) =>
                    onUpdate({
                      hf_repo_id: event.target.value,
                      hf_path: "",
                      seed_columns: [],
                    })
                  }
                />
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

              <div className="grid grid-cols-2 gap-3">
                <div className="grid gap-2">
                  <FieldLabel
                    label="Subset (optional)"
                    htmlFor={subsetId}
                    hint="Dataset config/subset name."
                  />
                  <Input
                    id={subsetId}
                    className="nodrag"
                    placeholder="default"
                    value={config.hf_subset ?? ""}
                    onChange={(event) => onUpdate({ hf_subset: event.target.value })}
                  />
                </div>
                <div className="grid gap-2">
                  <FieldLabel
                    label="Split"
                    htmlFor={splitId}
                    hint="Split to inspect (default train)."
                  />
                  <Input
                    id={splitId}
                    className="nodrag"
                    placeholder="train"
                    value={config.hf_split ?? ""}
                    onChange={(event) => onUpdate({ hf_split: event.target.value })}
                  />
                </div>
              </div>
            </>
          )}

          {mode === "local" && (
            <div className="grid gap-2">
              <FieldLabel
                label="Local file"
                hint="Upload CSV, JSON, or JSONL seed file."
              />
              <Input
                className="nodrag"
                type="file"
                accept={LOCAL_ACCEPT}
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setLocalFile(file);
                  onUpdate({ hf_path: "", seed_columns: [] });
                }}
              />
              <p className="text-xs text-muted-foreground">
                Upload-only. Max 50MB.
              </p>
              {localFile && (
                <p className="text-xs text-muted-foreground">Selected: {localFile.name}</p>
              )}
            </div>
          )}

          {mode === "unstructured" && (
            <div className="grid gap-2">
              <FieldLabel
                label="Unstructured file"
                hint="Upload PDF, DOCX, or TXT. We chunk text into seed rows."
              />
              <Input
                className="nodrag"
                type="file"
                accept={UNSTRUCTURED_ACCEPT}
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setUnstructuredFile(file);
                  onUpdate({ hf_path: "", seed_columns: [] });
                }}
              />
              <p className="text-xs text-muted-foreground">
                Chunking uses chunk_text only. Max 50MB.
              </p>
              {unstructuredFile && (
                <p className="text-xs text-muted-foreground">
                  Selected: {unstructuredFile.name}
                </p>
              )}
            </div>
          )}

          <Button
            type="button"
            variant="outline"
            className="nodrag"
            onClick={() => void loadSeedMetadata()}
            disabled={isInspecting || !canLoad}
          >
            {isInspecting ? "Loading..." : "Load columns + 10 rows"}
          </Button>

          {inspectError && <p className="text-xs text-red-600">{inspectError}</p>}

          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger asChild={true}>
              <button
                type="button"
                className="flex w-full items-center justify-between text-left text-xs text-muted-foreground"
              >
                <span className="font-semibold uppercase">Advanced</span>
                <span>{advancedOpen ? "Hide" : "Show"}</span>
              </button>
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

      <TabsContent value="preview" className="pt-3">
        <div className="space-y-4">
          {previewRows.length === 0 ? (
            <div className="flex w-full items-center justify-center">
              <Empty className="max-w-lg">
                <EmptyHeader>
                  <EmptyTitle>Seed preview</EmptyTitle>
                  <EmptyDescription>
                    Click load to fetch 10 rows from the selected seed source.
                  </EmptyDescription>
                </EmptyHeader>
                <EmptyContent>
                  <Button
                    type="button"
                    variant="outline"
                    className="nodrag"
                    onClick={() => void loadSeedMetadata()}
                    disabled={isInspecting || !canLoad}
                  >
                    {isInspecting ? "Loading..." : "Load 10 rows"}
                  </Button>
                </EmptyContent>
              </Empty>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">
                Loaded columns: {previewColumns.join(", ") || "None"}
              </div>
              <div className="max-h-[360px] overflow-auto rounded-xl border border-border/60">
                <Table>
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
                      <TableRow key={`row-${rowIdx}`}>
                        {previewColumns.map((col) => (
                          <TableCell
                            key={`${rowIdx}-${col}`}
                            className="max-w-[260px] whitespace-pre-wrap break-words text-xs"
                          >
                            {stringifyCell(row[col])}
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
