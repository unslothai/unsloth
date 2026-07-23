// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import {
  type GgufVariantDetail,
  type LocalModelInfo,
  listGgufVariants,
  listLocalModels,
} from "@/features/chat";
import { cn } from "@/lib/utils";
import { Link } from "@tanstack/react-router";
import { ChevronDownIcon, ChevronRightIcon, RefreshCwIcon } from "lucide-react";
import {
  type ComponentPropsWithoutRef,
  type ReactElement,
  forwardRef,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

const GGUF_SUFFIX_PATTERN = /-GGUF(?:$|-)/i;

type LocalRecipeModelSelectorProps = {
  value: string;
  ggufVariant?: string | null;
  onChange: (modelId: string, ggufVariant?: string | null) => void;
  inputId?: string;
  disabled?: boolean;
  compact?: boolean;
  className?: string;
};

function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[\s_.-]/g, "");
}

function hasGgufSuffix(value: string | null | undefined): boolean {
  return GGUF_SUFFIX_PATTERN.test(value ?? "");
}

function getModelLabel(model: LocalModelInfo): string {
  return model.model_id?.trim() || model.display_name || model.id;
}

function isDirectGguf(model: LocalModelInfo): boolean {
  return model.path.toLowerCase().endsWith(".gguf");
}

function isExpandableGguf(model: LocalModelInfo): boolean {
  return (
    !isDirectGguf(model) &&
    (hasGgufSuffix(model.id) ||
      hasGgufSuffix(model.display_name) ||
      hasGgufSuffix(model.model_id))
  );
}

function sourceLabel(model: LocalModelInfo): string {
  switch (model.source) {
    case "models_dir":
      return "Models";
    case "hf_cache":
      return "HF cache";
    case "lmstudio":
      return "LM Studio";
    case "custom":
      return "Custom folder";
    default:
      return "Local";
  }
}

type SelectedModelSummary = {
  label: string;
  source: string;
  isGguf: boolean;
};

function getSelectedModelSummary(
  value: string,
  selectedModel: LocalModelInfo | null,
  ggufVariant?: string | null,
): SelectedModelSummary {
  if (!selectedModel) {
    return {
      label: value,
      source: "Local model",
      isGguf: Boolean(ggufVariant),
    };
  }

  return {
    label: getModelLabel(selectedModel),
    source: sourceLabel(selectedModel),
    isGguf: isDirectGguf(selectedModel) || isExpandableGguf(selectedModel),
  };
}

function LocalGgufVariantList({
  repoId,
  selectedVariant,
  onSelect,
}: {
  repoId: string;
  selectedVariant?: string | null;
  onSelect: (variant: string) => void;
}): ReactElement {
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  const [defaultVariant, setDefaultVariant] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listGgufVariants(repoId)
      .then((response) => {
        if (cancelled) {
          return;
        }
        setVariants(response.variants);
        setDefaultVariant(response.default_variant);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setError(
          err instanceof Error ? err.message : "Failed to load variants.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [repoId]);

  const sortedVariants = useMemo(() => {
    if (!variants) {
      return null;
    }
    return [...variants].sort((a, b) => {
      if (a.quant === defaultVariant) {
        return -1;
      }
      if (b.quant === defaultVariant) {
        return 1;
      }
      if (a.downloaded !== b.downloaded) {
        return a.downloaded ? -1 : 1;
      }
      return a.quant.localeCompare(b.quant);
    });
  }, [defaultVariant, variants]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 text-xs text-muted-foreground">
        <Spinner className="size-3" />
        Loading quantizations...
      </div>
    );
  }

  if (error) {
    return <div className="px-4 py-2 text-xs text-destructive">{error}</div>;
  }

  if (!sortedVariants || sortedVariants.length === 0) {
    return (
      <div className="px-4 py-2 text-xs text-muted-foreground">
        No GGUF quantizations found for this model.
      </div>
    );
  }

  return (
    <div className="ml-6 mt-1 rounded-lg bg-muted/25 p-1.5">
      <div className="mb-1 px-2 text-ui-10 font-medium uppercase tracking-wide text-muted-foreground">
        Quantization
      </div>
      <div className="space-y-0.5">
        {sortedVariants.map((variant) => {
          const selected = selectedVariant === variant.quant;
          return (
            <button
              key={variant.filename}
              type="button"
              onClick={() => onSelect(variant.quant)}
              className={cn(
                "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors hover:bg-muted/60",
                selected && "bg-background text-foreground shadow-sm",
              )}
            >
              <span className="min-w-0 flex-1 truncate font-mono">
                {variant.quant}
              </span>
              {variant.quant === defaultVariant ? (
                <Badge variant="secondary" className="h-4 px-1.5 text-ui-10">
                  recommended
                </Badge>
              ) : null}
              {variant.downloaded ? (
                <Badge variant="outline" className="h-4 px-1.5 text-ui-10">
                  ready
                </Badge>
              ) : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}

type SelectorTriggerProps = ComponentPropsWithoutRef<"button"> & {
  value: string;
  selectedModel: LocalModelInfo | null;
  ggufVariant?: string | null;
  inputId?: string;
  disabled: boolean;
  compact: boolean;
  className?: string;
};

const SelectorTrigger = forwardRef<HTMLButtonElement, SelectorTriggerProps>(
  function SelectorTrigger(
    {
      value,
      selectedModel,
      ggufVariant,
      inputId,
      disabled,
      compact,
      className,
      ...triggerProps
    },
    ref,
  ): ReactElement {
    const selected = getSelectedModelSummary(value, selectedModel, ggufVariant);

    return (
      <button
        {...triggerProps}
        ref={ref}
        id={inputId}
        type="button"
        disabled={disabled}
        className={cn(
          "nodrag flex w-full min-w-0 items-center gap-2 rounded-xl border border-border/70 bg-background px-3 text-left transition-colors hover:bg-muted/40 disabled:pointer-events-none disabled:opacity-60",
          compact ? "min-h-8 py-1.5 text-xs" : "min-h-10 py-2 text-sm",
          className,
        )}
      >
        <span className="min-w-0 flex-1">
          <span
            className={cn(
              "block truncate font-medium",
              !selected.label && "text-muted-foreground",
            )}
          >
            {selected.label || "Choose a local model"}
          </span>
          {compact ? null : (
            <span className="mt-0.5 flex min-w-0 items-center gap-1.5 text-ui-11 text-muted-foreground">
              <span className="truncate">
                {selected.label
                  ? selected.source
                  : "Select from local and cached models"}
              </span>
              {selected.isGguf ? <span>GGUF</span> : null}
              {ggufVariant ? (
                <span className="truncate font-mono">{ggufVariant}</span>
              ) : null}
            </span>
          )}
        </span>
        {compact && ggufVariant ? (
          <Badge
            variant="secondary"
            className="h-4 px-1.5 font-mono text-ui-10"
          >
            {ggufVariant}
          </Badge>
        ) : null}
        <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground" />
      </button>
    );
  },
);

function LocalModelRow({
  model,
  selected,
  expanded,
  probing,
  ggufVariant,
  onSelectModel,
  onSelectVariant,
}: {
  model: LocalModelInfo;
  selected: boolean;
  expanded: boolean;
  probing: boolean;
  ggufVariant?: string | null;
  onSelectModel: (model: LocalModelInfo) => void;
  onSelectVariant: (modelId: string, variant: string) => void;
}): ReactElement {
  const expandable = isExpandableGguf(model);
  const directGguf = isDirectGguf(model);

  return (
    <div>
      <button
        type="button"
        disabled={probing}
        onClick={() => onSelectModel(model)}
        className={cn(
          "flex w-full items-center gap-2 rounded-lg px-2.5 py-2.5 text-left text-sm transition-colors hover:bg-muted/50",
          selected && "bg-muted/70 text-foreground ring-1 ring-border/70",
        )}
      >
        {expandable ? (
          expanded ? (
            <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
          )
        ) : (
          <span className="size-3.5 shrink-0" />
        )}
        <span className="min-w-0 flex-1">
          <span className="block truncate font-medium">
            {getModelLabel(model)}
          </span>
          <span className="mt-0.5 block truncate text-ui-11 text-muted-foreground">
            {model.id}
          </span>
        </span>
        <span className="flex shrink-0 items-center gap-1">
          {probing ? (
            <Spinner className="size-3 text-muted-foreground" />
          ) : null}
          {expandable || directGguf ? (
            <Badge variant="secondary" className="h-4 px-1.5 text-ui-10">
              GGUF
            </Badge>
          ) : null}
          <Badge variant="outline" className="h-4 px-1.5 text-ui-10">
            {sourceLabel(model)}
          </Badge>
        </span>
      </button>
      {expanded ? (
        <LocalGgufVariantList
          repoId={model.id}
          selectedVariant={selected ? ggufVariant : null}
          onSelect={(variant) => onSelectVariant(model.id, variant)}
        />
      ) : null}
    </div>
  );
}

function LocalModelResults({
  loading,
  error,
  models,
  value,
  ggufVariant,
  expandedModelId,
  probingVariantModelId,
  onRefresh,
  onSelectModel,
  onSelectVariant,
}: {
  loading: boolean;
  error: string | null;
  models: LocalModelInfo[];
  value: string;
  ggufVariant?: string | null;
  expandedModelId: string | null;
  probingVariantModelId: string | null;
  onRefresh: () => void;
  onSelectModel: (model: LocalModelInfo) => void;
  onSelectVariant: (modelId: string, variant: string) => void;
}): ReactElement {
  if (loading) {
    return (
      <div className="flex items-center gap-2 px-3 py-3 text-xs text-muted-foreground">
        <Spinner className="size-3" />
        Scanning local models...
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-2 px-3 py-3 text-xs">
        <p className="text-destructive">{error}</p>
        <Button type="button" variant="outline" size="xs" onClick={onRefresh}>
          Try again
        </Button>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="space-y-2 px-3 py-3 text-xs text-muted-foreground">
        <p className="font-medium text-foreground">No local models found.</p>
        <p>
          Download a model or add a scan folder from Chat, then refresh this
          list.
        </p>
        <Link
          to="/chat"
          className="inline-flex font-medium text-primary underline-offset-4 hover:underline"
        >
          Open Chat model picker
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {models.map((model) => (
        <LocalModelRow
          key={model.id}
          model={model}
          selected={model.id === value}
          expanded={expandedModelId === model.id}
          probing={probingVariantModelId === model.id}
          ggufVariant={ggufVariant}
          onSelectModel={onSelectModel}
          onSelectVariant={onSelectVariant}
        />
      ))}
    </div>
  );
}

export function LocalRecipeModelSelector({
  value,
  ggufVariant,
  onChange,
  inputId,
  disabled = false,
  compact = false,
  className,
}: LocalRecipeModelSelectorProps): ReactElement {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [models, setModels] = useState<LocalModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);
  const [probingVariantModelId, setProbingVariantModelId] = useState<
    string | null
  >(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const requestModelRefresh = useCallback(() => {
    setLoading(true);
    setError(null);
    setRefreshKey((key) => key + 1);
  }, []);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      setOpen(nextOpen);
      if (nextOpen) {
        requestModelRefresh();
      }
    },
    [requestModelRefresh],
  );

  useEffect(() => {
    if (!open || refreshKey < 0) {
      return;
    }
    let cancelled = false;
    listLocalModels()
      .then((response) => {
        if (cancelled) {
          return;
        }
        setModels(response.models);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setError(
          err instanceof Error ? err.message : "Failed to list local models.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [open, refreshKey]);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === value) ?? null,
    [models, value],
  );

  const filteredModels = useMemo(() => {
    const needle = normalizeForSearch(query.trim());
    if (!needle) {
      return models;
    }
    return models.filter((model) => {
      const haystack = normalizeForSearch(
        `${model.id} ${model.display_name} ${model.model_id ?? ""} ${model.path}`,
      );
      return haystack.includes(needle);
    });
  }, [models, query]);

  const selectModel = useCallback(
    async (model: LocalModelInfo) => {
      if (isExpandableGguf(model)) {
        setExpandedModelId((current) =>
          current === model.id ? null : model.id,
        );
        return;
      }
      if (!isDirectGguf(model)) {
        setProbingVariantModelId(model.id);
        try {
          const response = await listGgufVariants(model.id);
          if (response.variants.length > 0) {
            setExpandedModelId(model.id);
            return;
          }
        } catch {
          // Non-GGUF local models commonly have no variant endpoint;
          // fall through to regular selection so they stay choosable.
        } finally {
          setProbingVariantModelId(null);
        }
      }
      onChange(model.id, null);
      setOpen(false);
    },
    [onChange],
  );

  const selectVariant = useCallback(
    (modelId: string, variant: string) => {
      onChange(modelId, variant);
      setOpen(false);
    },
    [onChange],
  );

  return (
    // modal keeps the list wheel-scrollable inside dialog scroll locks
    <Popover open={open} onOpenChange={handleOpenChange} modal={true}>
      <PopoverTrigger asChild={true}>
        <SelectorTrigger
          value={value}
          selectedModel={selectedModel}
          ggufVariant={ggufVariant}
          inputId={inputId}
          disabled={disabled}
          compact={compact}
          className={className}
        />
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={0}
        className="menu-soft-surface nodrag nowheel gap-0 overflow-hidden p-0"
        style={{
          width:
            "min(max(var(--radix-popover-trigger-width), 34rem), calc(100vw - 1rem))",
        }}
      >
        <div className="flex flex-col">
          <div className="border-b border-border/60 p-2.5">
            <div className="flex items-center gap-2">
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Filter local models"
                className="h-8 flex-1"
                autoFocus={true}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={requestModelRefresh}
                aria-label="Refresh local models"
              >
                <RefreshCwIcon className="size-3.5" />
              </Button>
            </div>
          </div>

          <div
            className="nowheel max-h-[min(24rem,calc(100vh-12rem))] overflow-y-auto overscroll-contain p-1.5"
            onWheelCapture={(event) => event.stopPropagation()}
          >
            <LocalModelResults
              loading={loading}
              error={error}
              models={filteredModels}
              value={value}
              ggufVariant={ggufVariant}
              expandedModelId={expandedModelId}
              probingVariantModelId={probingVariantModelId}
              onRefresh={requestModelRefresh}
              onSelectModel={selectModel}
              onSelectVariant={selectVariant}
            />
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
