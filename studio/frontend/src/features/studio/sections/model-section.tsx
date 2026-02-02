import { SectionCard } from "@/components/section-card";
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
  ChipIcon,
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";

const HF_REPO_MAP: Record<string, string> = {
  "llava-1.6-7b": "unsloth/llava-v1.6-mistral-7b",
  "llava-1.6-13b": "unsloth/llava-v1.6-vicuna-13b",
  "qwen-vl-7b": "Qwen/Qwen-VL-Chat",
  bark: "suno/bark",
  "xtts-v2": "coqui/XTTS-v2",
  "gemma-3-27b": "unsloth/gemma-3-27b",
  "llama-3.1-8b": "unsloth/Llama-3.1-8B",
  "mistral-7b": "unsloth/mistral-7b-v0.3",
  "phi-4": "unsloth/phi-4",
  "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
};

const DOT_COLORS = [
  "bg-amber-400",
  "bg-blue-400",
  "bg-emerald-400",
  "bg-rose-400",
  "bg-violet-400",
  "bg-cyan-400",
  "bg-orange-400",
  "bg-pink-400",
  "bg-teal-400",
  "bg-indigo-400",
  "bg-lime-400",
  "bg-fuchsia-400",
  "bg-sky-400",
  "bg-red-400",
  "bg-yellow-400",
  "bg-purple-400",
];

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

  const filteredModels = useMemo(() => {
    if (!modelType) {
      return MODELS;
    }
    return MODELS.filter((m) => m.type === modelType).sort(
      (a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0),
    );
  }, [modelType]);

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
        {/* Base Model */}
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
                Search from curated optimized models.{" "}
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
          <Select value={selectedModel ?? ""} onValueChange={setSelectedModel}>
            <SelectTrigger className={DARK_TRIGGER}>
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent
              position="popper"
              className={`${DARK_CONTENT} max-h-64 overflow-y-auto w-[var(--radix-select-trigger-width)]`}
            >
              {filteredModels.map((m, i) => (
                <SelectItem key={m.id} value={m.id}>
                  <span className="flex items-center gap-2">
                    <span
                      className={`size-2 shrink-0 rounded-full ${DOT_COLORS[i % DOT_COLORS.length]}`}
                    />
                    {m.name}
                    <span className="text-background/40 ml-auto text-xs">
                      {m.params}
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* HF Repo */}
        <div className="flex flex-col gap-2">
          <span className="text-xs font-medium text-muted-foreground">
            Hugging Face Repo
          </span>
          <InputGroup>
            <InputGroupAddon>
              <HugeiconsIcon icon={Search01Icon} className="size-4" />
            </InputGroupAddon>
            <InputGroupInput
              placeholder="unsloth/gemma-3-27b"
              value={
                selectedModel
                  ? (HF_REPO_MAP[selectedModel] ?? selectedModel)
                  : ""
              }
              onChange={(e) => setSelectedModel(e.target.value || null)}
            />
          </InputGroup>
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
