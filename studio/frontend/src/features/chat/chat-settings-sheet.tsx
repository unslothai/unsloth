import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  ArrowDown01Icon,
  Delete02Icon,
  EngineIcon,
  FloppyDiskIcon,
  PencilEdit01Icon,
  Settings02Icon,
  SlidersHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";
import { useState } from "react";

export interface InferenceParams {
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  maxTokens: number;
  systemPrompt: string;
  inferenceEngine: string;
  checkpoint: string;
}

export const defaultInferenceParams: InferenceParams = {
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  repetitionPenalty: 1.1,
  maxTokens: 512,
  systemPrompt: "",
  inferenceEngine: "unsloth",
  checkpoint: "outputs/llama-3.1-8b-instruct-lora",
};

export interface Preset {
  name: string;
  params: InferenceParams;
}

const BUILTIN_PRESETS: Preset[] = [
  { name: "Default", params: { ...defaultInferenceParams } },
  {
    name: "Creative",
    params: {
      ...defaultInferenceParams,
      temperature: 1.2,
      topP: 0.95,
      topK: 80,
      repetitionPenalty: 1.05,
    },
  },
  {
    name: "Precise",
    params: {
      ...defaultInferenceParams,
      temperature: 0.2,
      topP: 0.7,
      topK: 20,
      repetitionPenalty: 1.2,
    },
  },
];

const ENGINE_OPTIONS = [
  { value: "unsloth", label: "Unsloth" },
  { value: "llama-cpp", label: "llama.cpp (GGUF)" },
];

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">{label}</span>
        <span className="text-xs tabular-nums text-muted-foreground">
          {value}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
      />
    </div>
  );
}

function CollapsibleSection({
  icon,
  label,
  children,
  defaultOpen = false,
}: {
  icon: Parameters<typeof HugeiconsIcon>[0]["icon"];
  label: string;
  children?: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center corner-squircle gap-2.5 rounded-md px-2 py-2 text-sm transition-colors hover:bg-accent"
      >
        <HugeiconsIcon icon={icon} className="size-4 text-muted-foreground" />
        <span className="flex-1 text-left font-medium">{label}</span>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.15 }}
        >
          <HugeiconsIcon
            icon={ArrowDown01Icon}
            className="size-3.5 text-muted-foreground"
          />
        </motion.div>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-2 pb-3 pt-1">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface ChatSettingsPanelProps {
  open: boolean;
  params: InferenceParams;
  onParamsChange: (params: InferenceParams) => void;
}

export function ChatSettingsPanel({
  open,
  params,
  onParamsChange,
}: ChatSettingsPanelProps) {
  const [presets, setPresets] = useState<Preset[]>(BUILTIN_PRESETS);
  const [activePreset, setActivePreset] = useState("Default");

  function set<K extends keyof InferenceParams>(key: K) {
    return (v: InferenceParams[K]) => onParamsChange({ ...params, [key]: v });
  }

  function applyPreset(name: string) {
    const p = presets.find((pr) => pr.name === name);
    if (p) {
      onParamsChange({ ...p.params, systemPrompt: params.systemPrompt });
      setActivePreset(name);
    }
  }

  function savePreset() {
    const name = prompt("Preset name:");
    if (!name?.trim()) {
      return;
    }
    const trimmed = name.trim();
    setPresets((prev) => [
      ...prev.filter((p) => p.name !== trimmed),
      { name: trimmed, params: { ...params } },
    ]);
    setActivePreset(trimmed);
  }

  function deletePreset(name: string) {
    if (BUILTIN_PRESETS.some((p) => p.name === name)) {
      return;
    }
    setPresets((prev) => prev.filter((p) => p.name !== name));
    if (activePreset === name) {
      setActivePreset("Default");
    }
  }

  return (
    <aside
      className={`shrink-0 h-full overflow-hidden bg-sidebar rounded-2xl corner-squircle transition-[width] duration-200 ease-linear ${open ? "w-[17rem] border-sidebar-border" : "w-0"}`}
    >
      <div className="flex h-full w-[17rem] flex-col">
          <div className="flex items-center gap-2 px-3 py-2">
            <HugeiconsIcon
              icon={PencilEdit01Icon}
              className="size-3.5 text-muted-foreground"
            />
            <span className="flex-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Advanced Configuration
            </span>
          </div>

          <div className="flex-1 overflow-y-auto px-1.5">
            <div className="px-2 pb-3">
              <div className="flex items-center gap-2">
                <Select value={activePreset} onValueChange={applyPreset}>
                  <SelectTrigger className="h-8 flex-1 corner-squircle text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {presets.map((p) => (
                      <SelectItem key={p.name} value={p.name}>
                        <div className="flex w-full items-center justify-between gap-2">
                          <span>{p.name}</span>
                          {!BUILTIN_PRESETS.some(
                            (bp) => bp.name === p.name,
                          ) && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                deletePreset(p.name);
                              }}
                              className="text-muted-foreground hover:text-destructive"
                            >
                              <HugeiconsIcon
                                icon={Delete02Icon}
                                className="size-3"
                              />
                            </button>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <button
                  type="button"
                  onClick={savePreset}
                  className="flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-xs text-muted-foreground transition-colors hover:bg-accent"
                  title="Save preset"
                >
                  <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
                  Save
                </button>
              </div>
            </div>

            <div className="px-2 pb-4">
              <label
                htmlFor="system-prompt"
                className="mb-1.5 block text-xs font-medium"
              >
                System Prompt
              </label>
              <Textarea
                id="system-prompt"
                value={params.systemPrompt}
                onChange={(e) => set("systemPrompt")(e.target.value)}
                placeholder="You are a helpful assistant..."
                className="min-h-20 text-xs corner-squircle"
                rows={3}
              />
            </div>

            <CollapsibleSection
              icon={EngineIcon}
              label="Inference Engine"
              defaultOpen={true}
            >
              <div>
                <span className="mb-1 block text-[11px] text-muted-foreground">
                  Backend
                </span>
                <Select
                  value={params.inferenceEngine}
                  onValueChange={set("inferenceEngine")}
                >
                  <SelectTrigger className="h-8 w-full text-xs corner-squircle">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {ENGINE_OPTIONS.map((o) => (
                      <SelectItem key={o.value} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CollapsibleSection>

            <CollapsibleSection
              icon={SlidersHorizontalIcon}
              label="Sampling"
              defaultOpen={true}
            >
              <div className="flex flex-col gap-5">
                <ParamSlider
                  label="Temperature"
                  value={params.temperature}
                  min={0}
                  max={2}
                  step={0.1}
                  onChange={set("temperature")}
                />
                <ParamSlider
                  label="Top P"
                  value={params.topP}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={set("topP")}
                />
                <ParamSlider
                  label="Top K"
                  value={params.topK}
                  min={0}
                  max={100}
                  step={1}
                  onChange={set("topK")}
                />
                <ParamSlider
                  label="Repetition Penalty"
                  value={params.repetitionPenalty}
                  min={1}
                  max={2}
                  step={0.05}
                  onChange={set("repetitionPenalty")}
                />
                <ParamSlider
                  label="Max Tokens"
                  value={params.maxTokens}
                  min={64}
                  max={4096}
                  step={64}
                  onChange={set("maxTokens")}
                />
              </div>
            </CollapsibleSection>

            <CollapsibleSection icon={Settings02Icon} label="Settings">
              <p className="text-xs text-muted-foreground">
                No additional settings yet.
              </p>
            </CollapsibleSection>
          </div>
      </div>
    </aside>
  );
}
