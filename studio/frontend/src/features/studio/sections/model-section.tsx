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
import { MODEL_TYPE_TO_HF_TASK } from "@/config/training";
import {
  useDebouncedValue,
  useHfModelSearch,
  useInfiniteScroll,
} from "@/hooks";
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
    useShallow(({
      modelType, selectedModel, setSelectedModel,
      trainingMethod, setTrainingMethod, hfToken, setHfToken,
    }) => ({
      modelType, selectedModel, setSelectedModel,
      trainingMethod, setTrainingMethod, hfToken, setHfToken,
    })),
  );

  const [inputValue, setInputValue] = useState("");
  const selectingRef = useRef(false);
  const debouncedQuery = useDebouncedValue(inputValue);

  function handleModelSelect(id: string | null) {
    selectingRef.current = true;
    setSelectedModel(id);
  }

  function handleInputChange(val: string) {
    if (selectingRef.current) {
      selectingRef.current = false;
      return;
    }
    setInputValue(val);
  }
  const task = modelType ? MODEL_TYPE_TO_HF_TASK[modelType] : undefined;
  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
  } = useHfModelSearch(debouncedQuery, {
    task,
    accessToken: hfToken || undefined,
  });

  const resultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    if (selectedModel && !ids.includes(selectedModel)) {
      ids.unshift(selectedModel);
    }
    return ids;
  }, [hfResults, selectedModel]);

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

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
            />
          </InputGroup>
        </div>

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
              items={resultIds}
              filteredItems={resultIds}
              filter={null}
              value={selectedModel}
              onValueChange={handleModelSelect}
              onInputValueChange={handleInputChange}
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
                      const r = hfResults.find((m) => m.id === id);
                      const detail = r?.totalParams
                        ? formatCompact(r.totalParams)
                        : r?.downloads != null
                          ? `↓${formatCompact(r.downloads)}`
                          : null;
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
                          {detail && (
                            <span className="text-[10px] text-muted-foreground shrink-0">
                              {detail}
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
