import { SectionCard } from "@/components/section-card";
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
import { MODEL_TYPE_TO_HF_TASK, MODELS } from "@/config/training";
import { useDebouncedValue, useHfModelSearch, useInfiniteScroll } from "@/hooks";
import { formatCompact } from "@/lib/utils";
import { useWizardStore } from "@/stores/training";
import type { TrainingMethod } from "@/types/training";
import {
  ChipIcon,
  FolderSearchIcon,
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

const METHOD_DOTS: Record<string, string> = {
  qlora: "bg-emerald-400",
  lora: "bg-blue-400",
  full: "bg-amber-400",
};

const DARK_TRIGGER =
  "w-full bg-foreground text-background hover:bg-foreground/85 dark:bg-foreground dark:text-background [&_svg]:text-background/50";
const DARK_CONTENT =
  "bg-foreground text-background shadow-xl border-background/10 [--accent:rgba(255,255,255,0.1)] [--accent-foreground:white] [&_[data-slot=select-item]]:text-white/70 [&_[data-slot=select-scroll-up-button]]:bg-foreground [&_[data-slot=select-scroll-down-button]]:bg-foreground";

export function ModelSection() {
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
  const { results: hfResults, isLoading, isLoadingMore, hasMore, fetchMore } = useHfModelSearch(debouncedQuery, {
    task,
    accessToken: hfToken || undefined,
  });

  const curatedModels = useMemo(() => {
    if (!modelType) return MODELS;
    return MODELS.filter((m) => m.type === modelType).sort(
      (a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0),
    );
  }, [modelType]);

  const modelMap = useMemo(() => {
    const map = new Map<string, { label: string; params?: string; totalParams?: number; downloads?: number; recommended?: boolean }>();
    for (const m of curatedModels) {
      map.set(m.hfRepo ?? m.id, { label: m.name, params: m.params, recommended: m.recommended });
    }
    for (const r of hfResults) {
      if (!map.has(r.id)) {
        map.set(r.id, { label: r.id, downloads: r.downloads, totalParams: r.totalParams });
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
      .filter((m) => m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q) || m.hfRepo?.toLowerCase().includes(q))
      .map((m) => m.hfRepo ?? m.id);
    const liveIds = hfResults.map((r) => r.id).filter((id) => !curatedIds.includes(id));
    return [...curatedIds, ...liveIds];
  }, [debouncedQuery, curatedModels, hfResults]);

  const allIds = useMemo(
    () => [...new Set([...curatedModels.map((m) => m.hfRepo ?? m.id), ...hfResults.map((r) => r.id)])],
    [curatedModels, hfResults],
  );

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(fetchMore);

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChipIcon} className="size-5" />}
      title="Model"
      description="Select base model and training method"
      accent="emerald"
      featured={true}
      badge="2x Faster Training"
      className="col-span-12 shadow-border ring-1 ring-border"
    >
      <div className="grid gap-4 lg:grid-cols-4">
        {/* Local Model */}
        <div className="flex flex-col gap-2">
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
          <InputGroup className="bg-foreground text-background [&_input]:text-background [&_input]:placeholder:text-background/40 [&_svg]:text-background/50 hover:bg-foreground/90">
            <InputGroupAddon>
              <HugeiconsIcon icon={FolderSearchIcon} className="size-4" />
            </InputGroupAddon>
            <InputGroupInput
              placeholder="./models/my-model"
              value={
                selectedModel
                  ? (MODELS.find((m) => m.id === selectedModel || m.hfRepo === selectedModel)?.hfRepo ?? selectedModel)
                  : ""
              }
              onChange={(e) => setSelectedModel(e.target.value || null)}
            />
          </InputGroup>
        </div>

        {/* Base Model Search */}
        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            Base Model
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
          </span>
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
                <div ref={scrollRef} className="max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]">
                  <ComboboxList className="p-1 !max-h-none !overflow-visible">
                    {(id: string) => {
                      const meta = modelMap.get(id);
                      const label = meta?.label ?? id;
                      const sizeLabel = meta?.params ?? (meta?.totalParams ? formatCompact(meta.totalParams) : null);
                      return (
                        <ComboboxItem key={id} value={id} className="justify-between">
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="min-w-0 flex-1 truncate">{label}</span>
                            </TooltipTrigger>
                            <TooltipContent side="left" className="max-w-xs break-all">
                              {label}
                            </TooltipContent>
                          </Tooltip>
                          {sizeLabel ? (
                            <span className="text-xs text-muted-foreground shrink-0">
                              {sizeLabel}
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
        </div>

        {/* Training Method */}
        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            Method
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
              <TooltipContent className="max-w-xs">
                QLoRA uses 4-bit quantization for lowest VRAM. LoRA uses 16-bit.
                Full updates all weights.{" "}
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
          </span>
          <Select
            value={trainingMethod}
            onValueChange={(v) => setTrainingMethod(v as TrainingMethod)}
          >
            <SelectTrigger className={DARK_TRIGGER}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent
              position="popper"
              className={`${DARK_CONTENT} w-[var(--radix-select-trigger-width)]`}
            >
              <SelectItem value="qlora">
                <span className="flex items-center gap-2">
                  <span
                    className={`size-2 shrink-0 rounded-full ${METHOD_DOTS.qlora}`}
                  />
                  QLoRA (4-bit)
                </span>
              </SelectItem>
              <SelectItem value="lora">
                <span className="flex items-center gap-2">
                  <span
                    className={`size-2 shrink-0 rounded-full ${METHOD_DOTS.lora}`}
                  />
                  LoRA (16-bit)
                </span>
              </SelectItem>
              <SelectItem value="full">
                <span className="flex items-center gap-2">
                  <span
                    className={`size-2 shrink-0 rounded-full ${METHOD_DOTS.full}`}
                  />
                  Full Fine-tune
                </span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* HF Token */}
        <div className="flex flex-col gap-2">
          <span className="text-xs font-medium text-muted-foreground">
            Hugging Face Token (Optional)
          </span>
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
        </div>
      </div>
    </SectionCard>
  );
}
