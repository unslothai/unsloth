/* eslint-disable react-refresh/only-export-components */

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import {
  Logout01Icon,
  Settings04Icon,
  SidebarLeft01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export interface ModelOption {
  value: string;
  label: string;
}

// TODO: fetch from GET /api/loras at runtime
export const MOCK_CHECKPOINTS: ModelOption[] = [
  {
    value: "outputs/llama-3.1-8b-instruct-lora",
    label: "meta-llama/Llama-3.1-8B-Instruct — LoRA v1",
  },
  {
    value: "outputs/qwen2.5-7b-lora",
    label: "Qwen/Qwen2.5-7B-Instruct — LoRA v2",
  },
  {
    value: "outputs/mistral-7b-v0.3-lora",
    label: "mistralai/Mistral-7B-Instruct-v0.3 — LoRA v1",
  },
];

export const MOCK_GGUFS: ModelOption[] = [
  {
    value: "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    label: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
  },
  {
    value: "models/Qwen2.5-7B-Instruct-Q5_K_M.gguf",
    label: "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
  },
  {
    value: "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    label: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
  },
];

export const DEFAULT_CHECKPOINT = MOCK_CHECKPOINTS[0].value;

interface ChatTopBarProps {
  checkpoint: string;
  inferenceEngine: string;
  onCheckpointChange: (value: string) => void;
  onEject: () => void;
  sidebarOpen: boolean;
  onSidebarToggle: () => void;
  settingsOpen: boolean;
  onSettingsToggle: () => void;
}

export function ChatTopBar({
  checkpoint,
  inferenceEngine,
  onCheckpointChange,
  onEject,
  sidebarOpen,
  onSidebarToggle,
  settingsOpen,
  onSettingsToggle,
}: ChatTopBarProps) {
  const isLoaded = checkpoint !== "";
  const items = inferenceEngine === "llama-cpp" ? MOCK_GGUFS : MOCK_CHECKPOINTS;

  return (
    <div className="flex h-11 shrink-0 items-center gap-2 px-3">
      {/* sidebar toggle */}
      <button
        type="button"
        onClick={onSidebarToggle}
        className={cn(
          "flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent",
          sidebarOpen && "bg-accent text-foreground",
        )}
        title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
      >
        <HugeiconsIcon icon={SidebarLeft01Icon} className="size-4" />
      </button>

      {/* center group: settings + model selector + eject */}
      <div className="flex flex-1 items-center justify-center gap-1.5">
        <button
          type="button"
          onClick={onSettingsToggle}
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent",
            settingsOpen && "bg-accent text-foreground",
          )}
          title="Inference settings"
        >
          <HugeiconsIcon icon={Settings04Icon} className="size-4" />
        </button>

        <Select value={checkpoint} onValueChange={onCheckpointChange}>
          <SelectTrigger className="h-8 min-w-[320px] text-xs">
            {isLoaded && (
              <span className="size-2 shrink-0 rounded-full bg-emerald-500" />
            )}
            <SelectValue placeholder="Select a model…" />
          </SelectTrigger>
          <SelectContent position="popper" className="min-w-[340px]">
            <SelectGroup>
              <SelectLabel>
                {inferenceEngine === "llama-cpp"
                  ? "GGUF Models"
                  : "LoRA Checkpoints"}
              </SelectLabel>
              {items.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>

        <button
          type="button"
          onClick={onEject}
          disabled={!isLoaded}
          className={cn(
            "flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-xs transition-colors",
            isLoaded
              ? "text-muted-foreground hover:bg-accent hover:text-foreground"
              : "cursor-default border-transparent text-muted-foreground/30",
          )}
        >
          <HugeiconsIcon icon={Logout01Icon} className="size-3.5" />
          Eject
        </button>
      </div>
    </div>
  );
}
