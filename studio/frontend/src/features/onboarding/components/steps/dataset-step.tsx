// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import {
  useDebouncedValue,
  useHfDatasetSearch,
  useHfTokenValidation,
  useInfiniteScroll,
} from "@/hooks";
import { cn } from "@/lib/utils";
import {
  HfDatasetSubsetSplitSelectors,
  useTrainingConfigStore,
} from "@/features/training";
import type { DatasetFormat } from "@/types/training";
import {
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
  SparklesIcon,
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
    selectHfDataset,
    selectLocalDataset,
    datasetFormat,
    setDatasetFormat,
    dataset,
    setDataset,
    datasetSubset,
    setDatasetSubset,
    datasetSplit,
    setDatasetSplit,
    datasetEvalSplit,
    setDatasetEvalSplit,
    uploadedFile,
    setUploadedFile,
    modelType,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      hfToken: s.hfToken,
      setHfToken: s.setHfToken,
      datasetSource: s.datasetSource,
      selectHfDataset: s.selectHfDataset,
      selectLocalDataset: s.selectLocalDataset,
      datasetFormat: s.datasetFormat,
      setDatasetFormat: s.setDatasetFormat,
      dataset: s.dataset,
      setDataset: s.setDataset,
      datasetSubset: s.datasetSubset,
      setDatasetSubset: s.setDatasetSubset,
      datasetSplit: s.datasetSplit,
      setDatasetSplit: s.setDatasetSplit,
      datasetEvalSplit: s.datasetEvalSplit,
      setDatasetEvalSplit: s.setDatasetEvalSplit,
      uploadedFile: s.uploadedFile,
      setUploadedFile: s.setUploadedFile,
      modelType: s.modelType,
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
    error: hfSearchError,
  } = useHfDatasetSearch(debouncedQuery, {
    modelType,
    accessToken: hfToken || undefined,
  });

  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  const resultIds = useMemo(() => hfResults.map((r) => r.id), [hfResults]);

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

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
            onClick={() =>
              selectHfDataset(datasetSource === "huggingface" ? dataset : null)
            }
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
            onClick={() =>
              selectLocalDataset(
                datasetSource === "upload" ? uploadedFile : null,
              )
            }
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
                autoComplete="new-password"
                name="hf-token"
                placeholder="hf_..."
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
              />
            </InputGroup>
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
          </Field>

          <Field>
            <FieldLabel>Search datasets</FieldLabel>
            <div ref={comboboxAnchorRef}>
              <Combobox
                items={resultIds}
                filteredItems={resultIds}
                filter={null}
                value={dataset}
                onValueChange={(id) => {
                  selectingRef.current = true;
                  setDataset(id);
                }}
                onInputValueChange={(val) => {
                  if (selectingRef.current) {
                    selectingRef.current = false;
                    return;
                  }
                  setInputValue(val);
                }}
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
          </Field>

          <HfDatasetSubsetSplitSelectors
            variant="wizard"
            enabled={datasetSource === "huggingface"}
            datasetName={dataset}
            accessToken={hfToken || undefined}
            datasetSubset={datasetSubset}
            setDatasetSubset={setDatasetSubset}
            datasetSplit={datasetSplit}
            setDatasetSplit={setDatasetSplit}
            datasetEvalSplit={datasetEvalSplit}
            setDatasetEvalSplit={setDatasetEvalSplit}
          />
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
                  {opt.value === "auto" && (
                    <HugeiconsIcon
                      icon={SparklesIcon}
                      className="mr-1.5 inline size-3.5 align-text-bottom"
                    />
                  )}
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
