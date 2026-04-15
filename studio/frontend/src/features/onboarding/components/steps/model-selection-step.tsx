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
import { MODEL_TYPE_TO_HF_TASK, PRIORITY_TRAINING_MODELS, applyPriorityOrdering } from "@/config/training";
import {
  useDebouncedValue,
  useGpuInfo,
  useHfModelSearch,
  useHfTokenValidation,
  useInfiniteScroll,
} from "@/hooks";
import { formatCompact } from "@/lib/utils";
import {
  type TrainingMethod as VramTrainingMethod,
  type VramFitStatus,
  buildModelVramMap,
} from "@/lib/vram";
import { useTrainingConfigStore } from "@/features/training";
import type { TrainingMethod } from "@/types/training";
import {
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

/** Extract param count label from model name (e.g. "Qwen3-0.6B" -> "0.6B"). */
function extractParamLabel(id: string): string | null {
  const name = id.split("/").pop() ?? id;
  const match = name.match(/(?:^|[-_])(\d+(?:\.\d+)?)[Bb](?:[-_]|$)/);
  return match ? `${match[1]}B` : null;
}

export function ModelSelectionStep() {
  const gpu = useGpuInfo();
  const {
    modelType,
    selectedModel,
    setSelectedModel,
    ensureModelDefaultsLoaded,
    trainingMethod,
    setTrainingMethod,
    hfToken,
    setHfToken,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      modelType: s.modelType,
      selectedModel: s.selectedModel,
      setSelectedModel: s.setSelectedModel,
      ensureModelDefaultsLoaded: s.ensureModelDefaultsLoaded,
      trainingMethod: s.trainingMethod,
      setTrainingMethod: s.setTrainingMethod,
      hfToken: s.hfToken,
      setHfToken: s.setHfToken,
    })),
  );

  const [inputValue, setInputValue] = useState("");
  const selectingRef = useRef(false);
  const debouncedQuery = useDebouncedValue(inputValue);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);
  const task = modelType ? MODEL_TYPE_TO_HF_TASK[modelType] : undefined;
  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
    error: hfSearchError,
  } = useHfModelSearch(debouncedQuery, {
    task,
    accessToken: debouncedHfToken || undefined,
    excludeGguf: true,
    priorityIds: PRIORITY_TRAINING_MODELS,
  });

  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  const resultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    return applyPriorityOrdering(ids);
  }, [hfResults]);

  // Match Studio behavior: only show exception signals (OOM/TIGHT) in training flows.
  const vramMap = useMemo(() => {
    const fitMap = buildModelVramMap(
      hfResults,
      trainingMethod as VramTrainingMethod,
      gpu,
    );
    const map = new Map<string, { status: VramFitStatus | null; detail: string | null }>();
    for (const r of hfResults) {
      const fit = fitMap.get(r.id);
      map.set(r.id, {
        status: fit?.status ?? null,
        detail: r.totalParams ? formatCompact(r.totalParams) : extractParamLabel(r.id),
      });
    }
    return map;
  }, [hfResults, gpu, trainingMethod]);

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

  useEffect(() => {
    ensureModelDefaultsLoaded();
  }, [selectedModel, ensureModelDefaultsLoaded]);

  return (
    <FieldGroup>
      <Field>
        <FieldLabel>
          Hugging Face Token{" "}
          <span className="text-muted-foreground font-normal">(Optional)</span>
        </FieldLabel>
        <FieldDescription>
          Required for gated or private models.{" "}
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
        <FieldLabel className="flex items-center gap-1.5">
          Search models
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
            <TooltipContent>
              Search Hugging Face models or pick from our recommended list.{" "}
              <a
                href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline"
              >
                Read more
              </a>
            </TooltipContent>
          </Tooltip>
        </FieldLabel>
        <div ref={comboboxAnchorRef}>
          <Combobox
            items={resultIds}
            filteredItems={resultIds}
            filter={null}
            value={selectedModel}
            onValueChange={(id) => {
              selectingRef.current = true;
              setSelectedModel(id);
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
            <ComboboxInput placeholder="Search models..." className="w-full">
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
                <ComboboxEmpty>No models found</ComboboxEmpty>
              )}
              <div
                ref={scrollRef}
                className="max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]"
              >
                <ComboboxList className="p-1 !max-h-none !overflow-visible">
                  {(id: string) => {
                    const entry = vramMap.get(id);
                    const sizeLabel = entry?.detail ?? null;
                    const fitStatus = entry?.status ?? null;
                    const exceeds = fitStatus === "exceeds";
                    return (
                      <ComboboxItem
                        key={id}
                        value={id}
                        className="justify-between"
                      >
                        <Tooltip>
                          <TooltipTrigger asChild={true}>
                            <span
                              className={`min-w-0 flex-1 truncate ${exceeds ? "!text-gray-500 dark:!text-gray-400" : ""}`}
                            >
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
                        <span className="flex items-center gap-1.5 shrink-0">
                          {fitStatus === "exceeds" && (
                            <span className="text-[9px] font-medium !text-red-700 !bg-red-50 dark:!text-red-400 dark:!bg-red-950 px-1.5 py-0.5 rounded">
                              OOM
                            </span>
                          )}
                          {fitStatus === "tight" && (
                            <span className="text-[9px] font-medium !text-amber-400">
                              TIGHT
                            </span>
                          )}
                          {sizeLabel ? (
                            <span className="text-xs text-muted-foreground">
                              {sizeLabel}
                            </span>
                          ) : null}
                        </span>
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

      {selectedModel && (
        <Field>
          <div className="flex items-center justify-between">
            <div>
              <FieldLabel className="flex items-center gap-1.5">
                Training method
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
                    QLoRA uses 4-bit quantization for lowest VRAM. LoRA uses
                    16-bit for better quality. Full fine-tune updates all
                    weights.{" "}
                    <a
                      href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline"
                    >
                      Read more
                    </a>
                  </TooltipContent>
                </Tooltip>
              </FieldLabel>
              <FieldDescription>
                Choose how to fine-tune {selectedModel}
              </FieldDescription>
            </div>
            <Select
              value={trainingMethod}
              onValueChange={(v) => setTrainingMethod(v as TrainingMethod)}
            >
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="qlora">QLoRA (4-bit)</SelectItem>
                <SelectItem value="lora">LoRA (16-bit)</SelectItem>
                <SelectItem value="full">Full Fine-tune</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </Field>
      )}
    </FieldGroup>
  );
}
