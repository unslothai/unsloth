import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
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
import { InputGroupAddon } from "@/components/ui/input-group";
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
import {
  useDebouncedValue,
  useHfDatasetSearch,
  useHfDatasetSplits,
  useHfTokenValidation,
  useInfiniteScroll,
} from "@/hooks";
import { formatCompact } from "@/lib/utils";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import { inspectSeedDataset } from "../../api";
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
  const [inputValue, setInputValue] = useState("");
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [isInspecting, setIsInspecting] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[]>([]);

  const selectingRef = useRef(false);
  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const debouncedQuery = useDebouncedValue(inputValue);

  const hfToken = config.hf_token?.trim() ?? "";
  const dataset = config.hf_repo_id.trim();

  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
    error: hfSearchError,
  } = useHfDatasetSearch(debouncedQuery, {
    accessToken: hfToken || undefined,
  });

  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  const { scrollRef, sentinelRef } = useInfiniteScroll(fetchMore, hfResults.length);

  const {
    subsets,
    splits,
    hasMultipleSubsets,
    hasMultipleSplits,
    isLoading: splitsLoading,
    error: splitsError,
  } = useHfDatasetSplits(dataset || null, config.hf_subset || null, {
    accessToken: hfToken || undefined,
  });

  useEffect(() => {
    if (subsets.length === 1 && config.hf_subset !== subsets[0]) {
      onUpdate({ hf_subset: subsets[0] });
    }
  }, [subsets, config.hf_subset, onUpdate]);

  useEffect(() => {
    if (splits.length === 0) return;
    if (hasMultipleSubsets && !config.hf_subset) return;

    if (splits.length === 1 && config.hf_split !== splits[0]) {
      onUpdate({ hf_split: splits[0] });
      return;
    }
    if (!config.hf_split && splits.includes("train")) {
      onUpdate({ hf_split: "train" });
      return;
    }
    if (!config.hf_split) {
      onUpdate({ hf_split: splits[0] });
    }
  }, [
    splits,
    hasMultipleSubsets,
    config.hf_subset,
    config.hf_split,
    onUpdate,
  ]);

  const resultIds = useMemo(() => {
    const ids = hfResults.map((result) => result.id);
    if (dataset && !ids.includes(dataset)) {
      ids.push(dataset);
    }
    return ids;
  }, [hfResults, dataset]);

  const samplingId = `${config.id}-sampling`;
  const selectionId = `${config.id}-selection`;
  const tokenId = `${config.id}-hf-token`;
  const subsetId = `${config.id}-hf-subset`;
  const splitId = `${config.id}-hf-split`;

  function handleDatasetSelect(id: string | null): void {
    selectingRef.current = true;
    const value = id ?? "";
    onUpdate({
      hf_repo_id: value,
      hf_subset: "",
      hf_split: "",
      hf_path: "",
      seed_columns: [],
    });
    setInspectError(null);
    setPreviewRows([]);
  }

  function handleInputChange(value: string): void {
    if (selectingRef.current) {
      selectingRef.current = false;
      return;
    }
    setInputValue(value);
  }

  async function loadSeedMetadata(): Promise<void> {
    const datasetName = config.hf_repo_id.trim();
    if (!datasetName) {
      setInspectError("Select a dataset first.");
      return;
    }

    setInspectError(null);
    setIsInspecting(true);
    try {
      const response = await inspectSeedDataset({
        dataset_name: datasetName,
        hf_token: hfToken || undefined,
        subset: config.hf_subset || undefined,
        split: config.hf_split || "train",
        preview_size: 10,
      });
      onUpdate({
        hf_path: response.resolved_path,
        seed_columns: response.columns,
        hf_split: response.split ?? config.hf_split ?? "",
        hf_subset: response.subset ?? config.hf_subset ?? "",
      });
      setPreviewRows(response.preview_rows ?? []);
    } catch (error) {
      setInspectError(error instanceof Error ? error.message : "Failed to load seed metadata.");
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
              label="Dataset"
              hint="Search and select a Hugging Face dataset repo (org/repo)."
            />
            <div
              ref={comboboxAnchorRef}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                if (!(event.target instanceof HTMLInputElement)) return;
                event.preventDefault();
                if (hfResults.length > 0) {
                  handleDatasetSelect(hfResults[0].id);
                } else {
                  const text = event.target.value.trim();
                  if (text) handleDatasetSelect(text);
                }
              }}
            >
              <Combobox
                items={resultIds}
                filteredItems={resultIds}
                filter={null}
                value={dataset || null}
                onValueChange={handleDatasetSelect}
                onInputValueChange={handleInputChange}
                itemToStringValue={(id) => id}
                autoHighlight={true}
              >
                <ComboboxInput placeholder="Search datasets..." className="nodrag w-full">
                  <InputGroupAddon>
                    <HugeiconsIcon icon={Search01Icon} className="size-4" />
                  </InputGroupAddon>
                </ComboboxInput>
                <ComboboxContent anchor={comboboxAnchorRef}>
                  {isLoading ? (
                    <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
                      <Spinner className="size-4" /> Searching...
                    </div>
                  ) : (
                    <ComboboxEmpty>No datasets found</ComboboxEmpty>
                  )}
                  <div
                    ref={scrollRef}
                    className="max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]"
                  >
                    <ComboboxList className="p-1 !max-h-none !overflow-visible">
                      {(id: string) => {
                        const result = hfResults.find((entry) => entry.id === id);
                        let detail: string | null = null;
                        if (result?.totalExamples) {
                          detail = `${formatCompact(result.totalExamples)} rows`;
                        } else if (result?.sizeCategory) {
                          detail = result.sizeCategory;
                        } else if (result?.downloads != null) {
                          detail = `↓${formatCompact(result.downloads)}`;
                        }
                        return (
                          <ComboboxItem key={id} value={id} className="justify-between">
                            <span className="min-w-0 flex-1 truncate">{id}</span>
                            {detail && (
                              <span className="shrink-0 text-[10px] text-muted-foreground">{detail}</span>
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

          <div className="grid gap-2">
            <FieldLabel
              label="HF token (optional)"
              htmlFor={tokenId}
              hint="Only needed for private or gated datasets."
            />
            <Input
              id={tokenId}
              className="nodrag"
              placeholder="hf_..."
              value={config.hf_token ?? ""}
              onChange={(event) => onUpdate({ hf_token: event.target.value })}
            />
            {(tokenValidationError ?? hfSearchError) && (
              <p className="text-xs text-destructive">{tokenValidationError ?? hfSearchError}</p>
            )}
            {isCheckingToken && (
              <p className="text-xs text-muted-foreground">Checking token…</p>
            )}
          </div>

          {splitsLoading && dataset ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner className="size-3.5" /> Loading subsets and splits...
            </div>
          ) : null}

          {splitsError ? (
            <p className="text-xs text-amber-700 dark:text-amber-400">
              Could not fetch dataset splits: {splitsError}
            </p>
          ) : null}

          {hasMultipleSubsets && (
            <div className="grid gap-2">
              <FieldLabel
                label="Subset"
                htmlFor={subsetId}
                hint="Pick a dataset subset/config when multiple are available."
              />
              <Select
                value={config.hf_subset ?? ""}
                onValueChange={(value) => onUpdate({ hf_subset: value || "", hf_split: "" })}
              >
                <SelectTrigger className="nodrag w-full" id={subsetId}>
                  <SelectValue placeholder="Select subset" />
                </SelectTrigger>
                <SelectContent>
                  {subsets.map((subset) => (
                    <SelectItem key={subset} value={subset}>
                      {subset}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {hasMultipleSplits && (
            <div className="grid gap-2">
              <FieldLabel
                label="Split"
                htmlFor={splitId}
                hint="Pick split used for preview sampling."
              />
              <Select
                value={config.hf_split ?? ""}
                onValueChange={(value) => onUpdate({ hf_split: value || "" })}
              >
                <SelectTrigger className="nodrag w-full" id={splitId}>
                  <SelectValue placeholder="Select split" />
                </SelectTrigger>
                <SelectContent>
                  {splits.map((split) => (
                    <SelectItem key={split} value={split}>
                      {split}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

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
                    Click load to fetch 10 rows from the selected dataset.
                  </EmptyDescription>
                </EmptyHeader>
                <EmptyContent>
                  <Button
                    type="button"
                    variant="outline"
                    className="nodrag"
                    onClick={() => void loadSeedMetadata()}
                    disabled={isInspecting || !dataset}
                  >
                    {isInspecting ? "Loading..." : "Load 10 rows"}
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
                  onClick={() => void loadSeedMetadata()}
                  disabled={isInspecting || !dataset}
                >
                  {isInspecting ? "Loading..." : "Reload 10 rows"}
                </Button>
              </div>
              <Table className="rounded-xl border border-border/60">
                <TableHeader>
                  <TableRow>
                    {previewColumns.map((column) => (
                      <TableHead key={column} className="max-w-[260px]">
                        {column}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {previewRows.map((row, index) => (
                    <TableRow key={index}>
                      {previewColumns.map((column) => (
                        <TableCell key={column} className="max-w-[260px]">
                          <div className="truncate">{stringifyCell(row[column])}</div>
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
          {inspectError && <p className="text-xs text-red-600">{inspectError}</p>}
        </div>
      </TabsContent>
    </Tabs>
  );
}
