import { Badge } from "@/components/ui/badge";
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
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
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
import { DATASETS } from "@/config/training";
import {
  useDebouncedValue,
  useHfDatasetSearch,
  useInfiniteScroll,
} from "@/hooks";
import { cn, formatCompact } from "@/lib/utils";
import { useWizardStore } from "@/stores/training";
import type { DatasetFormat } from "@/types/training";
import {
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
  Upload04Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

const FORMAT_OPTIONS: { value: DatasetFormat; label: string }[] = [
  { value: "auto", label: "Auto Detect" },
  { value: "alpaca", label: "Alpaca" },
  { value: "chatml", label: "ChatML" },
  { value: "sharegpt", label: "ShareGPT" },
];

export function DatasetStep() {
  const {
    hfToken,
    setHfToken,
    datasetSource,
    setDatasetSource,
    datasetFormat,
    setDatasetFormat,
    dataset,
    setDataset,
    uploadedFile,
    setUploadedFile,
  } = useWizardStore(
    useShallow((s) => ({
      hfToken: s.hfToken,
      setHfToken: s.setHfToken,
      datasetSource: s.datasetSource,
      setDatasetSource: s.setDatasetSource,
      datasetFormat: s.datasetFormat,
      setDatasetFormat: s.setDatasetFormat,
      dataset: s.dataset,
      setDataset: s.setDataset,
      uploadedFile: s.uploadedFile,
      setUploadedFile: s.setUploadedFile,
    })),
  );

  const [inputValue, setInputValue] = useState("");
  const debouncedQuery = useDebouncedValue(inputValue);
  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    hasMore,
    fetchMore,
  } = useHfDatasetSearch(debouncedQuery, {
    accessToken: hfToken || undefined,
  });

  const curatedDatasets = useMemo(
    () =>
      [...DATASETS].sort(
        (a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0),
      ),
    [],
  );

  const datasetMap = useMemo(() => {
    const map = new Map<
      string,
      {
        label: string;
        description?: string;
        size?: string;
        totalExamples?: number;
        sizeCategory?: string;
        downloads?: number;
        recommended?: boolean;
      }
    >();
    for (const d of curatedDatasets) {
      map.set(d.id, {
        label: d.name,
        description: d.description,
        size: d.size,
        recommended: d.recommended,
      });
    }
    for (const r of hfResults) {
      if (!map.has(r.id)) {
        map.set(r.id, {
          label: r.id,
          downloads: r.downloads,
          totalExamples: r.totalExamples,
          sizeCategory: r.sizeCategory,
        });
      }
    }
    return map;
  }, [curatedDatasets, hfResults]);

  const displayIds = useMemo(() => {
    if (!debouncedQuery.trim()) {
      return curatedDatasets.map((d) => d.id);
    }
    const q = debouncedQuery.toLowerCase();
    const curatedIds = curatedDatasets
      .filter(
        (d) =>
          d.name.toLowerCase().includes(q) || d.id.toLowerCase().includes(q),
      )
      .map((d) => d.id);
    const liveIds = hfResults
      .map((r) => r.id)
      .filter((id) => !curatedIds.includes(id));
    return [...curatedIds, ...liveIds];
  }, [debouncedQuery, curatedDatasets, hfResults]);

  const allIds = useMemo(
    () => [
      ...new Set([
        ...curatedDatasets.map((d) => d.id),
        ...hfResults.map((r) => r.id),
      ]),
    ],
    [curatedDatasets, hfResults],
  );

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(fetchMore);

  const handleFileUpload = () => {
    setUploadedFile("my_dataset.jsonl");
  };

  return (
    <FieldGroup>
      <Field>
        <FieldLabel>Source</FieldLabel>
        <div className="flex gap-2">
          <Button
            variant={datasetSource === "huggingface" ? "dark" : "outline"}
            onClick={() => setDatasetSource("huggingface")}
            className="flex-1"
          >
            <img
              src="/huggingface.svg"
              alt=""
              className="size-4 invert"
              data-icon="inline-start"
            />
            Hugging Face
          </Button>
          <Button
            variant={datasetSource === "upload" ? "dark" : "outline"}
            onClick={() => setDatasetSource("upload")}
            className="flex-1"
          >
            <HugeiconsIcon icon={Upload04Icon} data-icon="inline-start" />
            Upload
          </Button>
        </div>
      </Field>

      {datasetSource === "huggingface" ? (
        <>
          <Field>
            <FieldLabel>
              Hugging Face Token{" "}
              <span className="text-muted-foreground font-normal">
                (Optional)
              </span>
            </FieldLabel>
            <FieldDescription>
              Required for gated or private datasets.{" "}
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Get token
              </a>
            </FieldDescription>
            <InputGroup>
              <InputGroupAddon>
                <HugeiconsIcon icon={Key01Icon} className="size-4" />
              </InputGroupAddon>
              <InputGroupInput
                type="password"
                placeholder="hf_..."
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
              />
            </InputGroup>
          </Field>

          <Field>
            <FieldLabel>Search datasets</FieldLabel>
            <div ref={comboboxAnchorRef}>
              <Combobox
                items={allIds}
                filteredItems={displayIds}
                filter={null}
                value={dataset}
                onValueChange={(id) => setDataset(id)}
                onInputValueChange={(val) => setInputValue(val)}
                itemToStringValue={(id) => datasetMap.get(id)?.label ?? id}
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
                      <Spinner className="size-4" /> Searching…
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
                        const meta = datasetMap.get(id);
                        const label = meta?.label ?? id;
                        const rowLabel =
                          meta?.size ??
                          (meta?.totalExamples
                            ? `${formatCompact(meta.totalExamples)} rows`
                            : null);
                        return (
                          <ComboboxItem
                            key={id}
                            value={id}
                            className="justify-between"
                          >
                            <Tooltip>
                              <TooltipTrigger asChild={true}>
                                <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                                  <span className="truncate">{label}</span>
                                  {meta?.description && (
                                    <span className="text-xs text-muted-foreground truncate">
                                      {meta.description}
                                    </span>
                                  )}
                                </div>
                              </TooltipTrigger>
                              <TooltipContent
                                side="left"
                                className="max-w-xs break-all"
                              >
                                {label}
                              </TooltipContent>
                            </Tooltip>
                            {rowLabel ? (
                              <Badge variant="outline" className="shrink-0">
                                {rowLabel}
                              </Badge>
                            ) : meta?.sizeCategory ? (
                              <span className="text-[10px] text-muted-foreground shrink-0">
                                {meta.sizeCategory}
                              </span>
                            ) : meta?.downloads != null ? (
                              <span className="text-[10px] text-muted-foreground shrink-0">
                                ↓{formatCompact(meta.downloads)}
                              </span>
                            ) : null}
                          </ComboboxItem>
                        );
                      }}
                    </ComboboxList>
                    {hasMore && <div ref={sentinelRef} className="h-px" />}
                    {isLoadingMore && (
                      <div className="flex items-center justify-center py-2">
                        <Spinner className="size-3.5 text-muted-foreground" />
                      </div>
                    )}
                  </div>
                </ComboboxContent>
              </Combobox>
            </div>
          </Field>
        </>
      ) : (
        <>
          <Field>
            <FieldLabel>Upload Dataset</FieldLabel>
            <FieldDescription>
              Supports JSONL, JSON, CSV formats
            </FieldDescription>
            <button
              type="button"
              className={cn(
                "border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer hover:border-primary/50 hover:bg-muted/50",
                uploadedFile && "border-primary/50 bg-primary/5",
              )}
              onClick={handleFileUpload}
            >
              {uploadedFile ? (
                <div className="flex flex-col items-center gap-2">
                  <Badge variant="secondary" className="text-sm">
                    {uploadedFile}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    Click to replace
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <HugeiconsIcon
                    icon={Upload04Icon}
                    className="size-8 text-muted-foreground"
                  />
                  <span className="text-sm text-muted-foreground">
                    Click to upload or drag and drop
                  </span>
                </div>
              )}
            </button>
          </Field>
        </>
      )}

      <Field>
        <div className="flex items-center justify-between">
          <FieldLabel className="flex items-center gap-1.5">
            Format
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  className="text-muted-foreground/50 hover:text-muted-foreground"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                Auto will try to identify and convert your dataset to a
                supported format.{" "}
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
          </FieldLabel>
          <Select
            value={datasetFormat}
            onValueChange={(v) => setDatasetFormat(v as DatasetFormat)}
          >
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {FORMAT_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </Field>
    </FieldGroup>
  );
}
