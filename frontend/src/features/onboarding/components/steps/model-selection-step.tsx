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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { MODELS } from "@/config/training";
import { useWizardStore } from "@/stores/training";
import type { TrainingMethod } from "@/types/training";
import {
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useRef } from "react";
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

  const filteredModels = useMemo(() => {
    if (!modelType) {
      return [];
    }
    // Sort recommended first
    return MODELS.filter((m) => m.type === modelType).sort(
      (a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0),
    );
  }, [modelType]);

  const selectedModelData = MODELS.find((m) => m.id === selectedModel);
  const comboboxAnchorRef = useRef<HTMLDivElement>(null);

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
              Search from our curated list of optimized models.{" "}
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
            items={filteredModels.map((m) => m.name)}
            value={selectedModelData?.name ?? null}
            onValueChange={(name) => {
              const model = filteredModels.find((m) => m.name === name);
              if (model) {
                setSelectedModel(model.id);
              }
            }}
            autoHighlight={true}
          >
            <ComboboxInput placeholder="Search by name..." className="w-full">
              <InputGroupAddon>
                <HugeiconsIcon icon={Search01Icon} className="size-4" />
              </InputGroupAddon>
            </ComboboxInput>
            <ComboboxContent anchor={comboboxAnchorRef}>
              <ComboboxEmpty>No models found</ComboboxEmpty>
              <ComboboxList className="p-1">
                {(name: string) => {
                  const model = filteredModels.find((m) => m.name === name);
                  return (
                    <ComboboxItem key={name} value={name}>
                      <span className="flex-1">{name}</span>
                      {model && (
                        <Badge variant="outline" className="ml-auto">
                          {model.params}
                        </Badge>
                      )}
                    </ComboboxItem>
                  );
                }}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>
      </Field>

      {selectedModelData && (
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
                Choose how to fine-tune {selectedModelData.name}
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
