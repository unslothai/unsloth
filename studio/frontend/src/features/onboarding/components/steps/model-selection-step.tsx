import { Badge } from "@/components/ui/badge";
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
import { MODELS, MODEL_TYPE_TO_HF_TASK } from "@/config/training";
import {
  useDebouncedValue,
  useHfModelSearch,
  useInfiniteScroll,
} from "@/hooks";
import { formatCompact } from "@/lib/utils";
import { useWizardStore } from "@/stores/training";
import type { TrainingMethod } from "@/types/training";
import {
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

export function ModelSelectionStep() {
  const {
    modelType,
    selectedModel,
    setSelectedModel,
    trainingMethod,
    setTrainingMethod,
    hfToken,
    setHfToken,
  } = useWizardStore(
    useShallow((s) => ({
      modelType: s.modelType,
      selectedModel: s.selectedModel,
      setSelectedModel: s.setSelectedModel,
      trainingMethod: s.trainingMethod,
      setTrainingMethod: s.setTrainingMethod,
      hfToken: s.hfToken,
      setHfToken: s.setHfToken,
    })),
  );

  const [inputValue, setInputValue] = useState("");
  const debouncedQuery = useDebouncedValue(inputValue);
  const task = modelType ? MODEL_TYPE_TO_HF_TASK[modelType] : undefined;
  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    hasMore,
    fetchMore,
  } = useHfModelSearch(debouncedQuery, {
    task,
    accessToken: hfToken || undefined,
  });

  const curatedModels = useMemo(() => {
    if (!modelType) {
      return [];
    }
    return MODELS.filter((m) => m.type === modelType).sort(
      (a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0),
    );
  }, [modelType]);

  const modelMap = useMemo(() => {
    const map = new Map<
      string,
      {
        label: string;
        params?: string;
        totalParams?: number;
        downloads?: number;
        recommended?: boolean;
      }
    >();
    for (const m of curatedModels) {
      map.set(m.hfRepo ?? m.id, {
        label: m.name,
        params: m.params,
        recommended: m.recommended,
      });
    }
    for (const r of hfResults) {
      if (!map.has(r.id)) {
        map.set(r.id, {
          label: r.id,
          downloads: r.downloads,
          totalParams: r.totalParams,
        });
      }
    }
    return map;
  }, [curatedModels, hfResults]);

  const displayIds = useMemo(() => {
    if (!debouncedQuery.trim()) {
      return curatedModels.map((m) => m.hfRepo ?? m.id);
    }
    const q = debouncedQuery.toLowerCase();
    const curatedIds = curatedModels
      .filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          m.id.toLowerCase().includes(q) ||
          m.hfRepo?.toLowerCase().includes(q),
      )
      .map((m) => m.hfRepo ?? m.id);
    const liveIds = hfResults
      .map((r) => r.id)
      .filter((id) => !curatedIds.includes(id));
    return [...curatedIds, ...liveIds];
  }, [debouncedQuery, curatedModels, hfResults]);

  const allIds = useMemo(
    () => [
      ...new Set([
        ...curatedModels.map((m) => m.hfRepo ?? m.id),
        ...hfResults.map((r) => r.id),
      ]),
    ],
    [curatedModels, hfResults],
  );

  const selectedModelData = MODELS.find(
    (m) => m.id === selectedModel || m.hfRepo === selectedModel,
  );
  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(fetchMore);

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
            placeholder="hf_..."
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
          />
        </InputGroup>
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
            items={allIds}
            filteredItems={displayIds}
            filter={null}
            value={selectedModel}
            onValueChange={(id) => setSelectedModel(id)}
            onInputValueChange={(val) => setInputValue(val)}
            itemToStringValue={(id) => modelMap.get(id)?.label ?? id}
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
                    const meta = modelMap.get(id);
                    const label = meta?.label ?? id;
                    const sizeLabel =
                      meta?.params ??
                      (meta?.totalParams
                        ? formatCompact(meta.totalParams)
                        : null);
                    return (
                      <ComboboxItem
                        key={id}
                        value={id}
                        className="justify-between"
                      >
                        <Tooltip>
                          <TooltipTrigger asChild={true}>
                            <span className="min-w-0 flex-1 truncate">
                              {label}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent
                            side="left"
                            className="max-w-xs break-all"
                          >
                            {label}
                          </TooltipContent>
                        </Tooltip>
                        <span className="flex items-center gap-1.5 shrink-0">
                          {meta?.recommended && (
                            <Badge
                              variant="outline"
                              className="text-[10px] px-1.5 py-0 text-emerald-600 border-emerald-200 dark:border-emerald-800 dark:text-emerald-400"
                            >
                              Recommended
                            </Badge>
                          )}
                          {sizeLabel ? (
                            <Badge variant="outline">{sizeLabel}</Badge>
                          ) : meta?.downloads != null ? (
                            <span className="text-[10px] text-muted-foreground">
                              ↓{formatCompact(meta.downloads)}
                            </span>
                          ) : null}
                        </span>
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

      {(selectedModelData || selectedModel) && (
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
                Choose how to fine-tune{" "}
                {selectedModelData?.name ?? selectedModel}
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
