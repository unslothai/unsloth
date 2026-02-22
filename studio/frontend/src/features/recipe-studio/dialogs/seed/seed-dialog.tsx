import { Button } from "@/components/ui/button";
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
import { type ReactElement, useMemo, useState } from "react";
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

type SeedDialogProps = {
  config: SeedConfig;
  onUpdate: (patch: Partial<SeedConfig>) => void;
};

function parseHfDatasetRepoId(input: string): string | null {
  const raw = input.trim();
  if (!raw) return null;
  if (!raw.includes("://") && raw.split("/").length === 2) {
    return raw;
  }
  try {
    const url = new URL(raw);
    const parts = url.pathname.split("/").filter(Boolean);
    const datasetsIdx = parts.indexOf("datasets");
    if (datasetsIdx === -1) return null;
    const org = parts[datasetsIdx + 1];
    const repo = parts[datasetsIdx + 2];
    if (!org || !repo) return null;
    return `${org}/${repo}`;
  } catch {
    return null;
  }
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

export function SeedDialog({ config, onUpdate }: SeedDialogProps): ReactElement {
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[]>([]);

  const repoId = useMemo(
    () => parseHfDatasetRepoId(config.hf_url ?? ""),
    [config.hf_url],
  );

  const pathId = `${config.id}-hf-path`;
  const urlId = `${config.id}-hf-url`;
  const tokenId = `${config.id}-hf-token`;
  const splitId = `${config.id}-hf-split`;
  const samplingId = `${config.id}-sampling`;
  const selectionId = `${config.id}-selection`;

  async function onInspect(): Promise<void> {
    setInspectError("Seed inspect disabled (backend /seed/inspect removed).");
  }

  async function onPreview(): Promise<void> {
    setPreviewError("Seed preview disabled (backend /seed/preview removed).");
    setPreviewRows([]);
  }

  const previewColumns = useMemo(() => {
    const cols = config.seed_columns ?? [];
    if (cols.length > 0) return cols;
    if (previewRows[0]) return Object.keys(previewRows[0]);
    return [];
  }, [config.seed_columns, previewRows]);

  return (
    <Tabs defaultValue="config" className="w-full">
      <TabsList className="w-full">
        <TabsTrigger value="config">Config</TabsTrigger>
        <TabsTrigger value="preview">Preview</TabsTrigger>
      </TabsList>

      <TabsContent value="config" className="pt-3">
        <div className="space-y-4">
          <div className="grid gap-2">
            <FieldLabel
              label="HF dataset URL"
              htmlFor={urlId}
              hint="Dataset URL or org/repo used to bootstrap seed columns."
            />
            <div className="flex items-center gap-2">
              <Input
                id={urlId}
                className="nodrag flex-1"
                placeholder="https://huggingface.co/datasets/org/repo"
                value={config.hf_url ?? ""}
                onChange={(e) => onUpdate({ hf_url: e.target.value })}
              />
              <Button
                type="button"
                variant="outline"
                className="nodrag"
                onClick={() => void onInspect()}
                disabled
              >
                Load (disabled)
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Repo: {repoId ?? "-"}
            </p>
          </div>
          {inspectError && (
            <p className="text-xs text-red-600">{inspectError}</p>
          )}

          {(config.seed_splits?.length ?? 0) > 0 && (
            <div className="grid gap-2">
              <FieldLabel
                label="Split"
                htmlFor={splitId}
                hint="Dataset split to sample from (train/validation/test)."
              />
              <Select
                value={config.hf_split ?? ""}
                onValueChange={(value) => {
                  const nextPath = config.seed_globs_by_split?.[value] ?? "";
                  onUpdate({ hf_split: value, hf_path: nextPath || config.hf_path });
                }}
              >
                <SelectTrigger className="nodrag w-full" id={splitId}>
                  <SelectValue placeholder="Select split" />
                </SelectTrigger>
                <SelectContent>
                  {(config.seed_splits ?? []).map((value) => (
                    <SelectItem key={value} value={value}>
                      {value}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="grid gap-2">
            <FieldLabel
              label="HF path (auto)"
              htmlFor={pathId}
              hint="Resolved dataset file path/pattern."
            />
            <Input
              id={pathId}
              className="nodrag"
              placeholder="datasets/org/repo/data/train-*.parquet"
              value={config.hf_path}
              onChange={(e) => onUpdate({ hf_path: e.target.value })}
            />
          </div>

          <div className="grid gap-2">
            <FieldLabel
              label="HF token (optional)"
              htmlFor={tokenId}
              hint="Optional private dataset access token."
            />
            <Input
              id={tokenId}
              className="nodrag"
              placeholder="hf_..."
              value={config.hf_token ?? ""}
              onChange={(e) => onUpdate({ hf_token: e.target.value })}
            />
          </div>

          <div className="grid gap-2">
            <FieldLabel
              label="Sampling strategy"
              htmlFor={samplingId}
              hint="Ordered keeps row order. shuffle randomizes sampled rows."
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
                {SAMPLING_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
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
                {SELECTION_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {config.selection_type === "index_range" && (
            <div className="grid grid-cols-2 gap-3">
              <div className="grid gap-2">
                <FieldLabel
                  label="Start"
                  hint="Inclusive start row index for index_range."
                />
                <Input
                  className="nodrag"
                  inputMode="numeric"
                  value={config.selection_start ?? ""}
                  onChange={(e) => onUpdate({ selection_start: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <FieldLabel
                  label="End"
                  hint="Inclusive end row index for index_range."
                />
                <Input
                  className="nodrag"
                  inputMode="numeric"
                  value={config.selection_end ?? ""}
                  onChange={(e) => onUpdate({ selection_end: e.target.value })}
                />
              </div>
            </div>
          )}

          {config.selection_type === "partition_block" && (
            <div className="grid grid-cols-2 gap-3">
              <div className="grid gap-2">
                <FieldLabel
                  label="Index"
                  hint="Partition index to load."
                />
                <Input
                  className="nodrag"
                  inputMode="numeric"
                  value={config.selection_index ?? ""}
                  onChange={(e) => onUpdate({ selection_index: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <FieldLabel
                  label="Partitions"
                  hint="Total number of partitions."
                />
                <Input
                  className="nodrag"
                  inputMode="numeric"
                  value={config.selection_num_partitions ?? ""}
                  onChange={(e) =>
                    onUpdate({ selection_num_partitions: e.target.value })
                  }
                />
              </div>
            </div>
          )}

          <p className="text-xs text-muted-foreground">
            Seed columns auto-add. Reference by name (ex: {"{{ rubrics }}"}).
          </p>
        </div>
      </TabsContent>

      <TabsContent value="preview" className="pt-3">
        <div className="space-y-4">
          {previewRows.length === 0 ? (
            <div className="flex w-full items-center justify-center">
              <Empty className="max-w-lg">
                <EmptyHeader>
                  <EmptyTitle>Preview samples</EmptyTitle>
                  <EmptyDescription>
                    Load 10 rows to see columns and sample values.
                  </EmptyDescription>
                </EmptyHeader>
                <EmptyContent>
                  <Button
                    type="button"
                    variant="outline"
                    className="nodrag"
                    onClick={() => void onPreview()}
                    disabled
                  >
                    Load 10 rows (disabled)
                  </Button>
                </EmptyContent>
              </Empty>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <Button
                  type="button"
                  variant="outline"
                  className="nodrag"
                  onClick={() => void onPreview()}
                  disabled
                >
                  Reload 10 rows (disabled)
                </Button>
              </div>
              <Table className="border border-border/60 rounded-xl">
                <TableHeader>
                  <TableRow>
                    {previewColumns.map((col) => (
                      <TableHead key={col} className="max-w-[260px]">
                        {col}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {previewRows.map((row, idx) => (
                    <TableRow key={idx}>
                      {previewColumns.map((col) => (
                        <TableCell key={col} className="max-w-[260px]">
                          <div className="truncate">{stringifyCell(row[col])}</div>
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
          {previewError && <p className="text-xs text-red-600">{previewError}</p>}
        </div>
      </TabsContent>
    </Tabs>
  );
}
