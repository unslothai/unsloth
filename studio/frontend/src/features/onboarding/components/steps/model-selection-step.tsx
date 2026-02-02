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
  const selectingRef = useRef(false);
  const debouncedQuery = useDebouncedValue(inputValue);
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

  const resultIds = useMemo(() => hfResults.map((r) => r.id), [hfResults]);

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

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
            items={resultIds}
            filteredItems={resultIds}
            filter={null}
            value={selectedModel}
            onValueChange={(id) => { selectingRef.current = true; setSelectedModel(id); }}
            onInputValueChange={(val) => { if (selectingRef.current) { selectingRef.current = false; return; } setInputValue(val); }}
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
                    const r = hfResults.find((r) => r.id === id);
                    const sizeLabel = r?.totalParams
                      ? formatCompact(r.totalParams)
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
                        {sizeLabel ? (
                          <span className="text-xs text-muted-foreground shrink-0">
                            {sizeLabel}
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
