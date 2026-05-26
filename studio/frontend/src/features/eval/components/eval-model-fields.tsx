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
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type LocalModelInfo,
  listLocalModels,
} from "@/features/training/api/models-api";
import {
  useDebouncedValue,
  useHfModelSearch,
  useHfTokenValidation,
  useInfiniteScroll,
} from "@/hooks";
import {
  FolderSearchIcon,
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";

const DARK_COMBOBOX_CONTENT =
  "bg-foreground text-background shadow-xl border-background/10 dark:[--accent:rgba(2,6,23,0.08)] dark:[--accent-foreground:rgb(2,6,23)] dark:[&_[data-slot=combobox-item]]:text-slate-900 dark:[&_.text-muted-foreground]:text-slate-500";

/** Extract param count label from model name (e.g. "Qwen3-0.6B" -> "0.6B"). */
function extractParamLabel(id: string): string | null {
  const name = id.split("/").pop() ?? id;
  const match = name.match(/(?:^|[-_])(\d+(?:\.\d+)?)[Bb](?:[-_]|$)/);
  return match ? `${match[1]}B` : null;
}

export function EvalModelFields({
  modelIdentifier,
  onModelChange,
  hfToken,
  onHfTokenChange,
}: {
  modelIdentifier: string;
  onModelChange: (id: string) => void;
  hfToken: string;
  onHfTokenChange: (token: string) => void;
}) {
  const [inputValue, setInputValue] = useState("");
  const [localModelInput, setLocalModelInput] = useState("");
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [isLoadingLocalModels, setIsLoadingLocalModels] = useState(true);
  const [localModelsError, setLocalModelsError] = useState<string | null>(null);
  const selectingRef = useRef(false);

  const debouncedQuery = useDebouncedValue(inputValue);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);

  function handleModelSelect(id: string | null) {
    selectingRef.current = true;
    onModelChange(id ?? "");
  }

  function handleInputChange(val: string) {
    if (selectingRef.current) {
      selectingRef.current = false;
      return;
    }
    setInputValue(val);
  }

  function applyLocalModel(value: string) {
    const next = value.trim();
    if (!next) return;
    onModelChange(next);
  }

  useEffect(() => {
    const controller = new AbortController();
    void listLocalModels(controller.signal)
      .then((models) => {
        if (controller.signal.aborted) return;
        setLocalModels(models);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setLocalModelsError(
          error instanceof Error
            ? error.message
            : "Failed to load local models",
        );
      })
      .finally(() => {
        if (controller.signal.aborted) return;
        setIsLoadingLocalModels(false);
      });
    return () => controller.abort();
  }, []);

  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
    error: hfSearchError,
  } = useHfModelSearch(debouncedQuery, {
    accessToken: debouncedHfToken || undefined,
    excludeGguf: true,
  });

  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  // Filter out GGUF models — they can't be used for eval loading either
  const evalableLocalModels = useMemo(
    () =>
      localModels.filter((m) => {
        if (m.source === "lmstudio") return false;
        if (m.path.endsWith(".gguf")) return false;
        if (m.id.toLowerCase().includes("-gguf")) return false;
        return true;
      }),
    [localModels],
  );

  const localMetaById = useMemo(() => {
    const map = new Map<string, LocalModelInfo>();
    for (const model of evalableLocalModels) map.set(model.id, model);
    return map;
  }, [evalableLocalModels]);

  const localResultIds = useMemo(() => {
    const ids = evalableLocalModels.map((model) => model.id);
    const manual = localModelInput.trim();
    if (manual && !ids.includes(manual)) {
      ids.unshift(manual);
    }
    return ids;
  }, [localModelInput, evalableLocalModels]);

  const localFilteredIds = useMemo(() => {
    const q = localModelInput.trim().toLowerCase();
    if (!q) return localResultIds;
    return localResultIds.filter((id) => {
      const meta = localMetaById.get(id);
      if (id.toLowerCase().includes(q)) return true;
      if (meta?.display_name.toLowerCase().includes(q)) return true;
      if (meta?.path.toLowerCase().includes(q)) return true;
      return false;
    });
  }, [localMetaById, localModelInput, localResultIds]);

  const resultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    if (modelIdentifier && !ids.includes(modelIdentifier)) {
      ids.push(modelIdentifier);
    }
    return ids;
  }, [hfResults, modelIdentifier]);

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const localComboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

  return (
    <div className="grid min-w-0 gap-4 md:grid-cols-2">
      {/* Local Model */}
      <div className="flex min-w-0 flex-col gap-2">
        <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          Local Model
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
              Path to a locally downloaded model or a custom HF repo.
            </TooltipContent>
          </Tooltip>
        </span>
        <div ref={localComboboxAnchorRef} className="min-w-0">
          <Combobox
            items={localResultIds}
            filteredItems={localFilteredIds}
            filter={null}
            value={localModelInput || null}
            onValueChange={(id) => {
              const next = id ?? "";
              setLocalModelInput(next);
              if (next) onModelChange(next);
            }}
            onInputValueChange={setLocalModelInput}
            itemToStringValue={(id) => id}
            autoHighlight={true}
          >
            <ComboboxInput
              placeholder={
                isLoadingLocalModels
                  ? "Scanning local and cached models..."
                  : "./models/my-model"
              }
              className="w-full bg-foreground text-background [&_input]:text-background [&_input]:placeholder:text-background/40 [&_svg]:text-background/50 hover:bg-foreground/90"
              onBlur={() => applyLocalModel(localModelInput)}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                event.preventDefault();
                applyLocalModel(localModelInput);
              }}
            >
              <InputGroupAddon>
                <HugeiconsIcon icon={FolderSearchIcon} className="size-4" />
              </InputGroupAddon>
            </ComboboxInput>
            <ComboboxContent
              anchor={localComboboxAnchorRef}
              className={DARK_COMBOBOX_CONTENT}
            >
              {isLoadingLocalModels ? (
                <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
                  <Spinner className="size-4" /> Scanning...
                </div>
              ) : localModelsError ? (
                <div className="px-3 py-2 text-xs text-red-500">
                  {localModelsError}
                </div>
              ) : (
                <ComboboxEmpty>No local models found</ComboboxEmpty>
              )}
              <ComboboxList className="p-1">
                {(id: string) => {
                  const model = localMetaById.get(id);
                  const source =
                    model?.source === "hf_cache"
                      ? "HF cache"
                      : model?.source === "lmstudio"
                        ? "LM Studio"
                        : model?.source === "custom"
                          ? "Custom Folders"
                          : "Local dir";
                  return (
                    <ComboboxItem key={id} value={id} className="gap-2">
                      <Tooltip>
                        <TooltipTrigger asChild={true}>
                          <span className="block min-w-0 flex-1 truncate">
                            {model?.display_name ?? id}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent
                          side="left"
                          className="max-w-xs break-all"
                        >
                          {model?.path ?? id}
                        </TooltipContent>
                      </Tooltip>
                      <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                        {source}
                      </span>
                    </ComboboxItem>
                  );
                }}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>
        {isLoadingLocalModels ? (
          <p className="text-[10px] text-muted-foreground">
            Scanning local models...
          </p>
        ) : localModelsError ? (
          <p className="text-[10px] text-red-500">{localModelsError}</p>
        ) : (
          <p className="text-[10px] text-muted-foreground">
            {evalableLocalModels.length > 0
              ? `${evalableLocalModels.length} local/cached models found`
              : "No local models found. Enter path manually."}
          </p>
        )}
      </div>

      {/* Hugging Face Model */}
      <div className="flex min-w-0 flex-col gap-2">
        <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          Hugging Face Model
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
              Search Hugging Face for any model to evaluate.
            </TooltipContent>
          </Tooltip>
        </span>
        <div
          ref={comboboxAnchorRef}
          className="min-w-0"
          onKeyDown={(event) => {
            if (event.key !== "Enter") return;
            if (!(event.target instanceof HTMLInputElement)) return;
            event.preventDefault();
            if (hfResults.length > 0) {
              handleModelSelect(hfResults[0].id);
            } else {
              const text = event.target.value.trim();
              if (text) handleModelSelect(text);
            }
          }}
        >
          <Combobox
            items={resultIds}
            filteredItems={resultIds}
            filter={null}
            value={modelIdentifier}
            onValueChange={handleModelSelect}
            onInputValueChange={handleInputChange}
            itemToStringValue={(id) => id}
            autoHighlight={true}
          >
            <ComboboxInput
              placeholder="Search models..."
              className="w-full leading-5"
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
                <ComboboxEmpty>No models found</ComboboxEmpty>
              )}
              <div
                ref={scrollRef}
                className="max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]"
              >
                <ComboboxList className="p-1 !max-h-none !overflow-visible">
                  {(id: string) => {
                    const paramLabel = extractParamLabel(id);
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
                        {paramLabel && (
                          <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                            {paramLabel}
                          </span>
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

      {/* Hugging Face Token (Optional) — spans full width on md+ */}
      <div className="flex min-w-0 flex-col gap-2 md:col-span-2">
        <span className="text-xs font-medium text-muted-foreground">
          Hugging Face Token (Optional)
        </span>
        <InputGroup>
          <InputGroupAddon>
            <HugeiconsIcon icon={Key01Icon} className="size-4" />
          </InputGroupAddon>
          <InputGroupInput
            type="password"
            autoComplete="new-password"
            name="hf-token-eval"
            placeholder="hf_..."
            value={hfToken}
            onChange={(e) => onHfTokenChange(e.target.value)}
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
        <p className="text-[10px] text-muted-foreground">
          Used to search and preview gated repos. Not sent with the eval run.
        </p>
      </div>
    </div>
  );
}
