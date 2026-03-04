import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
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
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  useDebouncedValue,
  useHfDatasetSearch,
  useHfTokenValidation,
  useInfiniteScroll,
} from "@/hooks";
import {
  HfDatasetSubsetSplitSelectors,
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
} from "@/features/training";
import {
  ArrowDown01Icon,
  CloudUploadIcon,
  Database02Icon,
  FileAttachmentIcon,
  InformationCircleIcon,
  Search01Icon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

function isLikelyLocalDatasetRef(value: string) {
  return (
    value.startsWith("/") ||
    value.startsWith("./") ||
    value.startsWith("../") ||
    value.includes("\\") ||
    /\.(jsonl|json|csv|parquet)$/i.test(value)
  );
}

export function DatasetSection() {
  const {
    dataset,
    setDataset,
    datasetFormat,
    setDatasetFormat,
    datasetSubset,
    setDatasetSubset,
    datasetSplit,
    setDatasetSplit,
    datasetEvalSplit,
    setDatasetEvalSplit,
    hfToken,
    modelType,
    datasetSliceStart,
    setDatasetSliceStart,
    datasetSliceEnd,
    setDatasetSliceEnd,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      dataset: s.dataset,
      setDataset: s.setDataset,
      datasetFormat: s.datasetFormat,
      setDatasetFormat: s.setDatasetFormat,
      datasetSubset: s.datasetSubset,
      setDatasetSubset: s.setDatasetSubset,
      datasetSplit: s.datasetSplit,
      setDatasetSplit: s.setDatasetSplit,
      datasetEvalSplit: s.datasetEvalSplit,
      setDatasetEvalSplit: s.setDatasetEvalSplit,
      hfToken: s.hfToken,
      modelType: s.modelType,
      datasetSliceStart: s.datasetSliceStart,
      setDatasetSliceStart: s.setDatasetSliceStart,
      datasetSliceEnd: s.datasetSliceEnd,
      setDatasetSliceEnd: s.setDatasetSliceEnd,
    })),
  );

  const [inputValue, setInputValue] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const openPreview = useDatasetPreviewDialogStore((s) => s.openPreview);
  const selectingRef = useRef(false);
  const debouncedQuery = useDebouncedValue(inputValue);

  function handleDatasetSelect(id: string | null) {
    selectingRef.current = true;
    setDataset(id);
  }

  function handleInputChange(val: string) {
    if (selectingRef.current) {
      selectingRef.current = false;
      return;
    }
    setInputValue(val);
  }
  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
    error: hfSearchError,
  } = useHfDatasetSearch(debouncedQuery, {
    modelType,
    accessToken: hfToken || undefined,
  });

  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  const resultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    if (dataset && !ids.includes(dataset)) {
      ids.push(dataset);
    }
    return ids;
  }, [hfResults, dataset]);

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

  return (
    <div data-tour="studio-dataset" className="col-span-1 xl:col-span-4">
      <SectionCard
        icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
        title="Dataset"
        description="Select or upload training data"
        accent="indigo"
        className="md:min-h-[470px] dark:shadow-border"
      >
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              Load from Hub
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-foreground/70 hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  Search Hugging Face datasets or enter a path like
                  'username/dataset-name'.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    Read more
                  </a>
                </TooltipContent>
              </Tooltip>
            </span>
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
                value={dataset}
                onValueChange={handleDatasetSelect}
                onInputValueChange={handleInputChange}
                itemToStringValue={(id) => id}
                autoHighlight={true}
              >
                <ComboboxInput
                  placeholder="Search datasets..."
                  className="w-full"
                >
                  <InputGroupAddon>
                    <HugeiconsIcon icon={Search01Icon} className="size-4" />
                  </InputGroupAddon>
                </ComboboxInput>
                <ComboboxContent anchor={comboboxAnchorRef}>
                  {isLoading ? (
                    <div className="flex items-center justify-center py-4 gap-2 text-xs text-muted-foreground">
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
                        return (
                          <ComboboxItem key={id} value={id} className="gap-2">
                            <Tooltip>
                              <TooltipTrigger asChild={true}>
                                <span className="block min-w-0 flex-1 truncate">
                                  {id}
                                </span>
                              </TooltipTrigger>
                              <TooltipContent
                                side="left"
                                className="max-w-xs break-all"
                              >
                                {id}
                              </TooltipContent>
                            </Tooltip>
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
            {(tokenValidationError ?? hfSearchError) && (
              <p className="text-xs text-destructive">
                {tokenValidationError ?? hfSearchError}
                {" — "}
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  Get or update token
                </a>
              </p>
            )}
            {isCheckingToken && (
              <p className="text-xs text-muted-foreground">Checking token…</p>
            )}
          </div>

        <HfDatasetSubsetSplitSelectors
          variant="studio"
          enabled={!!dataset && !isLikelyLocalDatasetRef(dataset)}
          datasetName={dataset}
          accessToken={hfToken || undefined}
          datasetSubset={datasetSubset}
          setDatasetSubset={setDatasetSubset}
          datasetSplit={datasetSplit}
          setDatasetSplit={setDatasetSplit}
          datasetEvalSplit={datasetEvalSplit}
          setDatasetEvalSplit={setDatasetEvalSplit}
          datasetSliceStart={datasetSliceStart}
          setDatasetSliceStart={setDatasetSliceStart}
          datasetSliceEnd={datasetSliceEnd}
          setDatasetSliceEnd={setDatasetSliceEnd}
        />

        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              className={`size-3.5 transition-transform ${advancedOpen ? "rotate-180" : ""}`}
            />
            Advanced
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3">
            <div className="flex flex-col gap-2">
              <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                Target Format
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      className="text-foreground/70 hover:text-foreground"
                    >
                      <HugeiconsIcon
                        icon={InformationCircleIcon}
                        className="size-3"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    Format of your training data. Auto-detect works for most
                    datasets.{" "}
                    <a
                      href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline"
                    >
                      Read more
                    </a>
                  </TooltipContent>
                </Tooltip>
              </span>
              <Select
                value={datasetFormat}
                onValueChange={(v) =>
                  setDatasetFormat(v as typeof datasetFormat)
                }
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto</SelectItem>
                  <SelectItem value="alpaca">Alpaca</SelectItem>
                  <SelectItem value="chatml">ChatML</SelectItem>
                  <SelectItem value="sharegpt">ShareGPT</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {dataset ? (
          <div className="flex items-center gap-3 rounded-lg border bg-muted/40 px-3.5 py-3">
            <div className="rounded-md bg-indigo-500/10 p-1.5">
              <HugeiconsIcon
                icon={FileAttachmentIcon}
                className="size-4 text-indigo-500"
              />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-mono text-sm font-medium truncate">
                {dataset}
              </p>
              <p className="text-[10px] text-muted-foreground">
                Hugging Face Dataset
                {datasetSubset && ` / ${datasetSubset}`}
                {datasetSplit && ` / ${datasetSplit}`}
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-3 rounded-lg border border-dashed bg-muted/20 px-3.5 py-3">
            <HugeiconsIcon
              icon={Database02Icon}
              className="size-4 text-muted-foreground/40"
            />
            <span className="text-xs text-muted-foreground">
              No dataset selected
            </span>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2">
          <Button
            variant="outline"
            size="sm"
            className="cursor-pointer gap-1.5"
          >
            <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
            Upload
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="cursor-pointer gap-1.5"
            disabled={!dataset}
            onClick={() => openPreview()}
          >
            <HugeiconsIcon icon={ViewIcon} className="size-3.5" />
            View dataset
          </Button>
        </div>
      </div>
      </SectionCard>
    </div>
  );
}
