import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
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
  useInfiniteScroll,
} from "@/hooks";
import { formatCompact } from "@/lib/utils";
import { useWizardStore } from "@/stores/training";
import {
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

export function DatasetSection() {
  const { dataset, setDataset, datasetFormat, setDatasetFormat, hfToken } =
    useWizardStore(
      useShallow((s) => ({
        dataset: s.dataset,
        setDataset: s.setDataset,
        datasetFormat: s.datasetFormat,
        setDatasetFormat: s.setDatasetFormat,
        hfToken: s.hfToken,
      })),
    );

  const [inputValue, setInputValue] = useState("");
  const selectingRef = useRef(false);
  const debouncedQuery = useDebouncedValue(inputValue);
  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
  } = useHfDatasetSearch(debouncedQuery, {
    accessToken: hfToken || undefined,
  });

  const resultIds = useMemo(() => hfResults.map((r) => r.id), [hfResults]);

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
      title="Dataset"
      description="Select or upload training data"
      accent="indigo"
      className="lg:col-span-4 min-h-[450px]"
    >
      <div className="flex flex-col gap-4">
        {/* Load from Hub */}
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
          <div ref={comboboxAnchorRef}>
            <Combobox
              items={resultIds}
              filteredItems={resultIds}
              filter={null}
              value={dataset}
              onValueChange={(id) => { selectingRef.current = true; setDataset(id); }}
              onInputValueChange={(val) => { if (selectingRef.current) { selectingRef.current = false; return; } setInputValue(val); }}
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
                      const r = hfResults.find((r) => r.id === id);
                      const detail = r?.totalExamples
                        ? `${formatCompact(r.totalExamples)} rows`
                        : (r?.sizeCategory ?? null);
                      return (
                        <ComboboxItem
                          key={id}
                          value={id}
                          className="justify-between"
                        >
                          <Tooltip>
                            <TooltipTrigger asChild={true}>
                              <span className="min-w-0 flex-1 truncate">
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
                          {detail ? (
                            <span className="text-[10px] text-muted-foreground shrink-0">
                              {detail}
                            </span>
                          ) : r?.downloads != null ? (
                            <span className="text-[10px] text-muted-foreground shrink-0">
                              ↓{formatCompact(r.downloads)}
                            </span>
                          ) : null}
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

        {/* Format */}
        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            Dataset Format
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
            onValueChange={(v) => setDatasetFormat(v as typeof datasetFormat)}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto-detect</SelectItem>
              <SelectItem value="alpaca">Alpaca</SelectItem>
              <SelectItem value="chatml">ChatML</SelectItem>
              <SelectItem value="sharegpt">ShareGPT</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Active dataset display */}
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

        {/* Action buttons */}
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
            className="cursor-pointer gap-1.5 text-muted-foreground"
          >
            <HugeiconsIcon icon={ViewIcon} className="size-3.5" />
            Preview
          </Button>
        </div>
      </div>
    </SectionCard>
  );
}
