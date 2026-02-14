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
import { Spinner } from "@/components/ui/spinner";
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
import { inspectSeedDataset, previewSeedDataset } from "../../api";
import type {
  SeedConfig,
  SeedSamplingStrategy,
  SeedSelectionType,
} from "../../types";

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

function parseOptionalInt(value: string | undefined): number | null {
  const trimmed = value?.trim();
  if (!trimmed) return null;
  const num = Number(trimmed);
  return Number.isFinite(num) ? num : null;
}

export function SeedDialog({ config, onUpdate }: SeedDialogProps): ReactElement {
  const [inspectLoading, setInspectLoading] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
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
    setInspectError(null);
    const repo_id = repoId;
    if (!repo_id) {
      setInspectError("Invalid HF dataset URL (need /datasets/org/repo).");
      return;
    }
    setInspectLoading(true);
    try {
      const res = await inspectSeedDataset({
        // biome-ignore lint/style/useNamingConvention: api schema
        repo_id,
        // biome-ignore lint/style/useNamingConvention: api schema
        hf_token: config.hf_token?.trim() || null,
        split: config.hf_split?.trim() || null,
      });
      const splits = res.splits ?? [];
      const globs = res.globs_by_split ?? {};
      let nextSplit = "";
      if (config.hf_split && splits.includes(config.hf_split)) {
        nextSplit = config.hf_split;
      } else if (splits[0]) {
        nextSplit = splits[0];
      }

      const nextPath = nextSplit ? (globs[nextSplit] ?? "") : "";
      onUpdate({
        hf_repo_id: res.repo_id,
        seed_splits: splits,
        seed_globs_by_split: globs,
        seed_columns: res.columns ?? [],
        hf_split: nextSplit,
        hf_path: nextPath || config.hf_path,
      });
    } catch (err) {
      setInspectError(err instanceof Error ? err.message : "Inspect failed.");
    } finally {
      setInspectLoading(false);
    }
  }

  async function onPreview(): Promise<void> {
    setPreviewError(null);
    const hf_path = config.hf_path.trim();
    if (!hf_path) {
      setPreviewError("HF path missing (Load first).");
      return;
    }
    setPreviewLoading(true);
    try {
      const res = await previewSeedDataset({
        // biome-ignore lint/style/useNamingConvention: api schema
        hf_path,
        // biome-ignore lint/style/useNamingConvention: api schema
        hf_token: config.hf_token?.trim() || null,
        // biome-ignore lint/style/useNamingConvention: api schema
        sampling_strategy: config.sampling_strategy,
        // biome-ignore lint/style/useNamingConvention: api schema
        selection_type: config.selection_type,
        // biome-ignore lint/style/useNamingConvention: api schema
        selection_start: parseOptionalInt(config.selection_start),
        // biome-ignore lint/style/useNamingConvention: api schema
        selection_end: parseOptionalInt(config.selection_end),
        // biome-ignore lint/style/useNamingConvention: api schema
        selection_index: parseOptionalInt(config.selection_index),
        // biome-ignore lint/style/useNamingConvention: api schema
        selection_num_partitions: parseOptionalInt(config.selection_num_partitions),
        limit: 10,
      });
      const rows = res.rows ?? [];
      setPreviewRows(rows);
      if ((config.seed_columns?.length ?? 0) === 0 && rows[0]) {
        onUpdate({ seed_columns: Object.keys(rows[0]) });
      }
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : "Preview failed.");
    } finally {
      setPreviewLoading(false);
    }
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
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={urlId}
            >
              HF dataset URL
            </label>
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
                disabled={inspectLoading}
              >
                {inspectLoading ? "Loading..." : "Load"}
              </Button>
              {inspectLoading && <Spinner />}
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
              <label
                className="text-xs font-semibold uppercase text-muted-foreground"
                htmlFor={splitId}
              >
                Split
              </label>
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
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={pathId}
            >
              HF path (auto)
            </label>
            <Input
              id={pathId}
              className="nodrag"
              placeholder="datasets/org/repo/data/train-*.parquet"
              value={config.hf_path}
              onChange={(e) => onUpdate({ hf_path: e.target.value })}
            />
          </div>

          <div className="grid gap-2">
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={tokenId}
            >
              HF token (optional)
            </label>
            <Input
              id={tokenId}
              className="nodrag"
              placeholder="hf_..."
              value={config.hf_token ?? ""}
              onChange={(e) => onUpdate({ hf_token: e.target.value })}
            />
          </div>

          <div className="grid gap-2">
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={samplingId}
            >
              Sampling strategy
            </label>
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
            <label
              className="text-xs font-semibold uppercase text-muted-foreground"
              htmlFor={selectionId}
            >
              Selection strategy
            </label>
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
                <label className="text-xs font-semibold uppercase text-muted-foreground">
                  Start
                </label>
                <Input
                  className="nodrag"
                  inputMode="numeric"
                  value={config.selection_start ?? ""}
                  onChange={(e) => onUpdate({ selection_start: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <label className="text-xs font-semibold uppercase text-muted-foreground">
                  End
                </label>
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
                <label className="text-xs font-semibold uppercase text-muted-foreground">
                  Index
                </label>
                <Input
                  className="nodrag"
                  inputMode="numeric"
                  value={config.selection_index ?? ""}
                  onChange={(e) => onUpdate({ selection_index: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <label className="text-xs font-semibold uppercase text-muted-foreground">
                  Partitions
                </label>
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
                    disabled={previewLoading}
                  >
                    {previewLoading ? "Loading..." : "Load 10 rows"}
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
                  disabled={previewLoading}
                >
                  {previewLoading ? "Loading..." : "Reload 10 rows"}
                </Button>
                {previewLoading && <Spinner />}
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
