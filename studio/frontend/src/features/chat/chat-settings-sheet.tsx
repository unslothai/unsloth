// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
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
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useIsMobile } from "@/hooks/use-mobile";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  CodeIcon,
  Delete02Icon,
  FloppyDiskIcon,
  Settings02Icon,
  Settings05Icon,
  SlidersHorizontalIcon,
  Wrench01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Tooltip,
  TooltipContent,
} from "@/components/ui/tooltip";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import {
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "./types/runtime";

export const defaultInferenceParams = DEFAULT_INFERENCE_PARAMS;
export type { InferenceParams } from "./types/runtime";

export interface Preset {
  name: string;
  params: InferenceParams;
}

interface LegacySystemPromptTemplate {
  name: string;
  content: string;
}

const BUILTIN_PRESETS: Preset[] = [
  { name: "Default", params: { ...defaultInferenceParams } },
  {
    name: "Creative",
    params: {
      ...defaultInferenceParams,
      temperature: 1.5,
      topP: 1.0,
      topK: 0,
      minP: 0.1,
      repetitionPenalty: 1.0,
    },
  },
  {
    name: "Precise",
    params: {
      ...defaultInferenceParams,
      temperature: 0.1,
      topP: 0.95,
      topK: 80,
      minP: 0.01,
      repetitionPenalty: 1.0,
    },
  },
];

const CHAT_PRESETS_KEY = "unsloth_chat_custom_presets";
const CHAT_ACTIVE_PRESET_KEY = "unsloth_chat_active_preset";
const LEGACY_CHAT_SYSTEM_PROMPTS_KEY = "unsloth_chat_system_prompts";
const LEGACY_CHAT_SYSTEM_PROMPTS_MIGRATED_KEY =
  "unsloth_chat_system_prompts_migrated";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function getUniquePresetName(baseName: string, usedNames: Set<string>): string {
  const normalizedBase = baseName.trim() || "Imported Prompt";
  let nextName = normalizedBase;
  let suffix = 2;
  while (usedNames.has(nextName)) {
    nextName = `${normalizedBase} ${suffix}`;
    suffix += 1;
  }
  usedNames.add(nextName);
  return nextName;
}

function migrateLegacySystemPromptTemplates(presets: Preset[]): Preset[] {
  if (!canUseStorage()) return presets;
  try {
    const raw = localStorage.getItem(LEGACY_CHAT_SYSTEM_PROMPTS_KEY);
    if (!raw) return presets;
    if (localStorage.getItem(LEGACY_CHAT_SYSTEM_PROMPTS_MIGRATED_KEY) === raw) {
      return presets;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw) as unknown;
    } catch {
      localStorage.removeItem(LEGACY_CHAT_SYSTEM_PROMPTS_KEY);
      localStorage.setItem(LEGACY_CHAT_SYSTEM_PROMPTS_MIGRATED_KEY, raw);
      return presets;
    }
    if (!Array.isArray(parsed)) {
      localStorage.removeItem(LEGACY_CHAT_SYSTEM_PROMPTS_KEY);
      localStorage.setItem(LEGACY_CHAT_SYSTEM_PROMPTS_MIGRATED_KEY, raw);
      return presets;
    }
    const usedNames = new Set([
      ...BUILTIN_PRESETS.map((preset) => preset.name),
      ...presets.map((preset) => preset.name),
    ]);
    const seenImportedConfigKeys = new Set(
      [...BUILTIN_PRESETS, ...presets].map((preset) =>
        JSON.stringify({
          temperature: preset.params.temperature,
          topP: preset.params.topP,
          topK: preset.params.topK,
          minP: preset.params.minP,
          repetitionPenalty: preset.params.repetitionPenalty,
          presencePenalty: preset.params.presencePenalty,
          maxSeqLength: preset.params.maxSeqLength,
          maxTokens: preset.params.maxTokens,
          systemPrompt: preset.params.systemPrompt,
          trustRemoteCode: preset.params.trustRemoteCode ?? false,
        }),
      ),
    );
    const importedPresets = parsed
      .filter((item): item is LegacySystemPromptTemplate => {
        if (!item || typeof item !== "object") return false;
        const maybe = item as Partial<LegacySystemPromptTemplate>;
        return (
          typeof maybe.name === "string" && typeof maybe.content === "string"
        );
      })
      .map((template) => ({
        template,
        importedParams: {
          ...defaultInferenceParams,
          systemPrompt: template.content,
        },
      }))
      .filter(({ importedParams }) => {
        const configKey = JSON.stringify({
          temperature: importedParams.temperature,
          topP: importedParams.topP,
          topK: importedParams.topK,
          minP: importedParams.minP,
          repetitionPenalty: importedParams.repetitionPenalty,
          presencePenalty: importedParams.presencePenalty,
          maxSeqLength: importedParams.maxSeqLength,
          maxTokens: importedParams.maxTokens,
          systemPrompt: importedParams.systemPrompt,
          trustRemoteCode: importedParams.trustRemoteCode ?? false,
        });
        if (seenImportedConfigKeys.has(configKey)) return false;
        seenImportedConfigKeys.add(configKey);
        return true;
      })
      .map(({ template, importedParams }) => ({
        name: getUniquePresetName(`${template.name} Prompt`, usedNames),
        params: importedParams,
      }));
    if (importedPresets.length === 0) {
      localStorage.removeItem(LEGACY_CHAT_SYSTEM_PROMPTS_KEY);
      localStorage.setItem(LEGACY_CHAT_SYSTEM_PROMPTS_MIGRATED_KEY, raw);
      return presets;
    }
    const mergedPresets = [...presets, ...importedPresets];
    localStorage.setItem(CHAT_PRESETS_KEY, JSON.stringify(mergedPresets));
    try {
      localStorage.setItem(LEGACY_CHAT_SYSTEM_PROMPTS_MIGRATED_KEY, raw);
      localStorage.removeItem(LEGACY_CHAT_SYSTEM_PROMPTS_KEY);
    } catch {
      // ignore cleanup failure after successful import write
    }
    return mergedPresets;
  } catch {
    return presets;
  }
}

function loadSavedCustomPresets(): Preset[] {
  if (!canUseStorage()) return [];
  try {
    const raw = localStorage.getItem(CHAT_PRESETS_KEY);
    if (!raw) {
      return migrateLegacySystemPromptTemplates([]);
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return migrateLegacySystemPromptTemplates([]);
    }
    const presets = parsed
      .filter((item): item is Preset => {
        if (!item || typeof item !== "object") return false;
        const maybe = item as Partial<Preset>;
        return typeof maybe.name === "string" && !!maybe.params;
      })
      .map((preset) => ({
        name: preset.name.trim(),
        params: {
          ...defaultInferenceParams,
          ...preset.params,
        },
      }))
      .filter((preset) => preset.name.length > 0);
    return migrateLegacySystemPromptTemplates(presets);
  } catch {
    return migrateLegacySystemPromptTemplates([]);
  }
}

function loadSavedActivePreset(): string {
  if (!canUseStorage()) return "Default";
  try {
    return localStorage.getItem(CHAT_ACTIVE_PRESET_KEY) ?? "Default";
  } catch {
    return "Default";
  }
}

type PresetSaveMode =
  | "disabled"
  | "overwrite-active"
  | "overwrite-other"
  | "create";

interface PresetSaveState {
  mode: PresetSaveMode;
  canSubmit: boolean;
  isSaveReady: boolean;
  buttonLabel: string;
  title: string;
}

function isSamePresetConfig(a: InferenceParams, b: InferenceParams): boolean {
  return (
    a.temperature === b.temperature &&
    a.topP === b.topP &&
    a.topK === b.topK &&
    a.minP === b.minP &&
    a.repetitionPenalty === b.repetitionPenalty &&
    a.presencePenalty === b.presencePenalty &&
    a.maxSeqLength === b.maxSeqLength &&
    a.maxTokens === b.maxTokens &&
    a.systemPrompt === b.systemPrompt &&
    (a.trustRemoteCode ?? false) === (b.trustRemoteCode ?? false)
  );
}

function getPresetSaveState({
  rawName,
  activePreset,
  presets,
  activePresetDirty,
}: {
  rawName: string;
  activePreset: string;
  presets: Preset[];
  activePresetDirty: boolean;
}): PresetSaveState {
  const trimmedName = rawName.trim();
  if (!trimmedName) {
    return {
      mode: "disabled",
      canSubmit: false,
      isSaveReady: false,
      buttonLabel: "Save",
      title: "Enter a preset name",
    };
  }

  const matchingPreset = presets.find((preset) => preset.name === trimmedName);
  if (matchingPreset) {
    const isActiveMatch = matchingPreset.name === activePreset;
    return {
      mode: isActiveMatch ? "overwrite-active" : "overwrite-other",
      canSubmit: !isActiveMatch || activePresetDirty,
      isSaveReady: !isActiveMatch || activePresetDirty,
      buttonLabel: isActiveMatch && !activePresetDirty ? "Saved" : "Overwrite",
      title: isActiveMatch
        ? activePresetDirty
          ? "Save current settings to this preset"
          : "No unsaved changes"
        : `Overwrite preset "${trimmedName}"`,
    };
  }

  return {
    mode: "create",
    canSubmit: true,
    isSaveReady: true,
    buttonLabel: "Save as New",
    title: `Save current settings as "${trimmedName}"`,
  };
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  displayValue?: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">{label}</span>
        <span className="text-xs tabular-nums text-muted-foreground">
          {displayValue ?? value}
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

const COLLAPSIBLE_STATE_KEY = "unsloth_chat_collapsible_state";

function loadCollapsibleState(): Record<string, boolean> {
  if (!canUseStorage()) return {};
  try {
    const raw = localStorage.getItem(COLLAPSIBLE_STATE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, boolean] => typeof entry[1] === "boolean",
      ),
    );
  } catch (error) {
    console.warn("Failed to load collapsible state from localStorage:", error);
    return {};
  }
}

function saveCollapsibleOpen(label: string, open: boolean) {
  if (!canUseStorage()) return;
  try {
    const state = loadCollapsibleState();
    state[label] = open;
    localStorage.setItem(COLLAPSIBLE_STATE_KEY, JSON.stringify(state));
  } catch {
    // ignore
  }
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
  const [open, setOpen] = useState(() => {
    const saved = loadCollapsibleState();
    return Object.hasOwn(saved, label) ? saved[label] : defaultOpen;
  });

  return (
    <div>
      <button
        type="button"
        onClick={() => {
          const next = !open;
          setOpen(next);
          saveCollapsibleOpen(label, next);
        }}
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
  onOpenChange?: (open: boolean) => void;
  params: InferenceParams;
  onParamsChange: (params: InferenceParams) => void;
  onReloadModel?: () => void;
}

export function ChatSettingsPanel({
  open,
  onOpenChange,
  params,
  onParamsChange,
  onReloadModel,
}: ChatSettingsPanelProps) {
  const isMobile = useIsMobile();
  const isGguf = useChatRuntimeStore((s) => s.activeGgufVariant) != null;
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const setSpeculativeType = useChatRuntimeStore((s) => s.setSpeculativeType);
  const loadedSpeculativeType = useChatRuntimeStore(
    (s) => s.loadedSpeculativeType,
  );
  const currentModels = useChatRuntimeStore((s) => s.models);
  const modelRequiresTrustRemoteCode = useChatRuntimeStore(
    (s) => s.modelRequiresTrustRemoteCode,
  );
  const currentCheckpoint = params.checkpoint;
  const currentModelIsVision =
    currentModels.find((m) => m.id === currentCheckpoint)?.isVision ?? false;
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const ggufMaxContextLength = useChatRuntimeStore(
    (s) => s.ggufMaxContextLength,
  );
  const ggufNativeContextLength = useChatRuntimeStore(
    (s) => s.ggufNativeContextLength,
  );
  const kvCacheDtype = useChatRuntimeStore((s) => s.kvCacheDtype);
  const setKvCacheDtype = useChatRuntimeStore((s) => s.setKvCacheDtype);
  const loadedKvCacheDtype = useChatRuntimeStore((s) => s.loadedKvCacheDtype);
  const customContextLength = useChatRuntimeStore((s) => s.customContextLength);
  const setCustomContextLength = useChatRuntimeStore(
    (s) => s.setCustomContextLength,
  );

  const ctxDisplayValue = customContextLength ?? ggufContextLength ?? "";
  const ctxMaxValue = ggufNativeContextLength ?? ggufContextLength ?? null;
  const kvDirty = kvCacheDtype !== loadedKvCacheDtype;
  const ctxDirty = customContextLength !== null;
  const specDirty = speculativeType !== loadedSpeculativeType;
  const modelSettingsDirty = kvDirty || ctxDirty || specDirty;
  const [customPresets, setCustomPresets] = useState<Preset[]>(() =>
    loadSavedCustomPresets(),
  );
  const [activePreset, setActivePreset] = useState(() =>
    loadSavedActivePreset(),
  );
  const [presetNameInput, setPresetNameInput] = useState(() =>
    loadSavedActivePreset(),
  );
  const presetControlRowRef = useRef<HTMLDivElement>(null);
  const [presetMenuWidthPx, setPresetMenuWidthPx] = useState<
    number | undefined
  >(undefined);
  const [systemPromptEditorOpen, setSystemPromptEditorOpen] = useState(false);
  const [systemPromptDraft, setSystemPromptDraft] = useState("");
  const presets = useMemo(() => {
    const overrides = new Set(customPresets.map((preset) => preset.name));
    return [
      ...BUILTIN_PRESETS.filter((preset) => !overrides.has(preset.name)),
      ...customPresets,
    ];
  }, [customPresets]);
  const activePresetDefinition = useMemo(
    () => presets.find((preset) => preset.name === activePreset) ?? null,
    [activePreset, presets],
  );
  const activeCustomPreset = useMemo(
    () => customPresets.find((preset) => preset.name === activePreset) ?? null,
    [activePreset, customPresets],
  );
  const activeBuiltinPreset = useMemo(
    () =>
      BUILTIN_PRESETS.find((preset) => preset.name === activePreset) ?? null,
    [activePreset],
  );
  const activePresetDirty = useMemo(
    () =>
      activePresetDefinition == null
        ? false
        : !isSamePresetConfig(activePresetDefinition.params, params),
    [activePresetDefinition, params],
  );
  const presetSaveState = useMemo(
    () =>
      getPresetSaveState({
        rawName: presetNameInput,
        activePreset,
        presets,
        activePresetDirty,
      }),
    [activePreset, activePresetDirty, presetNameInput, presets],
  );
  const systemPromptEditorDirty = systemPromptDraft !== params.systemPrompt;
  const trustRemoteCodeMissing =
    Boolean(currentCheckpoint) &&
    modelRequiresTrustRemoteCode &&
    !(params.trustRemoteCode ?? false);

  function set<K extends keyof InferenceParams>(key: K) {
    return (v: InferenceParams[K]) => onParamsChange({ ...params, [key]: v });
  }

  function applyPreset(name: string) {
    const p = presets.find((pr) => pr.name === name);
    if (p) {
      if (
        modelRequiresTrustRemoteCode &&
        !(p.params.trustRemoteCode ?? false)
      ) {
        toast.warning("This configuration turns custom code off", {
          description:
            "The current model needs custom code enabled to load. Keep it on for this model.",
        });
        return;
      }
      onParamsChange({
        ...p.params,
        checkpoint: params.checkpoint,
      });
      setActivePreset(name);
      if (canUseStorage()) {
        try {
          localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, name);
        } catch {
          // ignore
        }
      }
    }
  }

  function savePresetWithName(rawName: string) {
    const trimmed = rawName.trim();
    if (!trimmed) {
      toast.error("Enter a preset name");
      return;
    }
    setCustomPresets((prev) => {
      const next = prev.filter((p) => p.name !== trimmed);
      const merged = [...next, { name: trimmed, params: { ...params } }];
      if (canUseStorage()) {
        try {
          localStorage.setItem(CHAT_PRESETS_KEY, JSON.stringify(merged));
        } catch {
          // ignore
        }
      }
      return merged;
    });
    if (canUseStorage()) {
      try {
        localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, trimmed);
      } catch {
        // ignore
      }
    }
    setActivePreset(trimmed);
    setPresetNameInput(trimmed);
  }

  function deletePreset(name: string) {
    const hasCustomPreset = customPresets.some(
      (preset) => preset.name === name,
    );
    if (!hasCustomPreset) {
      return;
    }
    const builtinPreset = BUILTIN_PRESETS.find((preset) => preset.name === name);
    const fallbackPreset =
      builtinPreset ??
      BUILTIN_PRESETS.find((preset) => preset.name === "Default") ??
      null;
    if (
      activePreset === name &&
      fallbackPreset &&
      modelRequiresTrustRemoteCode &&
      !(fallbackPreset.params.trustRemoteCode ?? false)
    ) {
      toast.warning("Reset would turn custom code off", {
        description:
          "The current model needs custom code enabled to load. Keep it on for this model.",
      });
      return;
    }
    setCustomPresets((prev) => {
      const next = prev.filter((preset) => preset.name !== name);
      if (canUseStorage()) {
        try {
          localStorage.setItem(CHAT_PRESETS_KEY, JSON.stringify(next));
        } catch {
          // ignore
        }
      }
      return next;
    });
    if (activePreset === name) {
      if (fallbackPreset) {
        onParamsChange({
          ...fallbackPreset.params,
          checkpoint: params.checkpoint,
        });
        setActivePreset(fallbackPreset.name);
        if (canUseStorage()) {
          try {
            localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, fallbackPreset.name);
          } catch {
            // ignore
          }
        }
      }
    }
  }

  function openSystemPromptEditor() {
    setSystemPromptDraft(params.systemPrompt);
    setSystemPromptEditorOpen(true);
  }

  function saveSystemPromptEditor() {
    set("systemPrompt")(systemPromptDraft);
    setSystemPromptEditorOpen(false);
  }

  useEffect(() => {
    if (presets.some((preset) => preset.name === activePreset)) return;
    setActivePreset("Default");
    if (canUseStorage()) {
      try {
        localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, "Default");
      } catch {
        // ignore
      }
    }
  }, [activePreset, presets]);

  useEffect(() => {
    setPresetNameInput(activePreset);
  }, [activePreset]);

  useEffect(() => {
    if (!open) {
      setSystemPromptEditorOpen(false);
    }
  }, [open]);

  useLayoutEffect(() => {
    const el = presetControlRowRef.current;
    if (!el || !open) return;
    const measure = () => {
      setPresetMenuWidthPx(el.getBoundingClientRect().width);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [open]);

  const settingsContent = (
    <>
      <div className="aui-thread-viewport relative h-full overflow-y-auto">
      <div className="sticky top-0 z-10 flex h-[48px] items-start gap-2 pl-2 pr-2 pt-[11px] backdrop-blur">
        {isMobile ? (
          <span className="flex h-[34px] flex-1 items-center pl-1 text-base font-semibold tracking-tight">
            Configuration
          </span>
        ) : (
          <>
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={() => onOpenChange?.(false)}
                  className="flex h-[34px] w-[34px] items-center justify-center rounded-[8px] text-[#383835] dark:text-[#c7c7c4] transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Close configuration"
                >
                  <HugeiconsIcon icon={Settings05Icon} className="size-5" />
                </button>
              </TooltipPrimitive.Trigger>
              <TooltipContent side="bottom" sideOffset={6}>
                Close configuration
              </TooltipContent>
            </Tooltip>
            <span className="flex h-[34px] flex-1 items-center text-base font-semibold tracking-tight">
              Configuration
            </span>
          </>
        )}
      </div>

      <div className="px-1.5">
        {/* mt-4 matches the Playground sidebar gap (SidebarHeader py-3 + SidebarGroup pt-1) */}
        <div className="mt-4 px-2 pb-3">
          <div className="space-y-1.5">
            <div ref={presetControlRowRef} className="w-full min-w-0">
              <DropdownMenu>
                <InputGroup className="!h-8 min-h-8 min-w-0 items-stretch gap-0 rounded-2xl pr-0 focus-within:border-input focus-within:ring-0 focus-within:shadow-none has-[[data-slot=input-group-control]:focus-visible]:border-input has-[[data-slot=input-group-control]:focus-visible]:ring-0 has-[[data-slot=input-group-control]:focus-visible]:shadow-none">
                  <InputGroupInput
                    id="inference-preset-name"
                    value={presetNameInput}
                    onChange={(e) => setPresetNameInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && presetSaveState.canSubmit) {
                        e.preventDefault();
                        savePresetWithName(presetNameInput);
                      }
                    }}
                    placeholder="Preset name"
                    maxLength={80}
                    autoComplete="off"
                    className={cn(
                      "!h-8 min-h-0 min-w-0 self-stretch !pl-2.5 !pr-2 pt-1 pb-1 text-sm leading-10 md:text-sm",
                      presetSaveState.isSaveReady &&
                        "text-foreground placeholder:text-primary/45",
                    )}
                    aria-label="Inference preset name"
                  />
                  <InputGroupAddon
                    align="inline-end"
                    className="min-h-0 shrink-0 gap-0 self-stretch border-0 py-0 pl-0 !pr-0 has-[>button]:mr-0"
                  >
                    <DropdownMenuTrigger asChild={true}>
                      <InputGroupButton
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        className="!h-8 min-h-8 !w-7 min-w-7 shrink-0 rounded-none rounded-r-2xl border-l border-border px-0 text-muted-foreground transition-colors hover:bg-primary/15 hover:text-primary data-[state=open]:bg-primary/20 data-[state=open]:text-primary"
                        title="Choose a preset"
                        aria-label="Open preset list"
                      >
                        <HugeiconsIcon
                          icon={ArrowDown01Icon}
                          className="size-3.5"
                          strokeWidth={2}
                        />
                      </InputGroupButton>
                    </DropdownMenuTrigger>
                  </InputGroupAddon>
                </InputGroup>
                <DropdownMenuContent
                  align="end"
                  className="min-w-40 max-w-none"
                  style={
                    presetMenuWidthPx != null
                      ? {
                          width: presetMenuWidthPx,
                          minWidth: presetMenuWidthPx,
                        }
                      : undefined
                  }
                >
                  {presets.map((p) => (
                    <DropdownMenuItem
                      key={p.name}
                      onSelect={() => applyPreset(p.name)}
                    >
                      {p.name}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <Button
                type="button"
                onClick={() => savePresetWithName(presetNameInput)}
                disabled={!presetSaveState.canSubmit}
                variant={presetSaveState.isSaveReady ? "default" : "outline"}
                size="sm"
                className={cn(
                  "h-8 w-full text-xs",
                  presetSaveState.isSaveReady &&
                    "bg-primary/92 text-primary-foreground hover:bg-primary",
                )}
                title={presetSaveState.title}
                aria-label={presetSaveState.title}
              >
                <span className="inline-flex shrink-0 items-center pr-1.5">
                  <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
                </span>
                {presetSaveState.buttonLabel}
              </Button>
              <Button
                type="button"
                onClick={() => deletePreset(activePreset)}
                disabled={!activeCustomPreset}
                variant="outline"
                size="sm"
                className="h-8 w-full text-xs text-muted-foreground"
                title={
                  activeCustomPreset
                    ? activeBuiltinPreset
                      ? "Reset selected preset to built-in defaults"
                      : "Delete selected preset"
                    : "No saved override to delete"
                }
              >
                <span className="inline-flex shrink-0 items-center pr-1.5">
                  <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                </span>
                Delete
              </Button>
            </div>
          </div>
        </div>

        <div className="px-2 pb-4">
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <label
              htmlFor="system-prompt"
              className="block text-xs font-medium"
            >
              System Prompt
            </label>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-6 px-2 text-[11px]"
              onClick={openSystemPromptEditor}
              title="Open the full system prompt editor"
            >
              Edit
            </Button>
          </div>
          <Textarea
            id="system-prompt"
            value={params.systemPrompt}
            onChange={(e) => set("systemPrompt")(e.target.value)}
            placeholder="You are a helpful assistant..."
            className="min-h-20 max-h-48 overflow-y-auto text-xs corner-squircle focus-visible:ring-[1px]"
            rows={3}
          />
        </div>

        <CollapsibleSection
          icon={Settings02Icon}
          label="Model"
          defaultOpen={true}
        >
          <div className="flex flex-col gap-3 py-1">
            {isGguf && (
              <>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium">Context Length</span>
                    <Input
                      type="number"
                      value={
                        typeof ctxDisplayValue === "number"
                          ? ctxDisplayValue
                          : (ggufContextLength ?? "")
                      }
                      placeholder="..."
                      min={128}
                      max={ctxMaxValue ?? undefined}
                      step={1024}
                      className="h-6 w-[100px] text-right text-xs tabular-nums"
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === "") {
                          setCustomContextLength(null);
                          return;
                        }
                        const v = Number.parseInt(raw, 10);
                        if (!Number.isNaN(v) && v >= 0) {
                          const maxCtx =
                            ctxMaxValue ?? Number.POSITIVE_INFINITY;
                          const clamped = Math.min(v, maxCtx);
                          setCustomContextLength(
                            clamped === (ggufContextLength ?? 0)
                              ? null
                              : clamped,
                          );
                        }
                      }}
                    />
                  </div>
                  <Slider
                    min={1024}
                    max={ctxMaxValue ?? 4096}
                    step={1024}
                    value={[
                      Math.min(
                        typeof ctxDisplayValue === "number"
                          ? ctxDisplayValue
                          : (ggufContextLength ?? 4096),
                        ctxMaxValue ?? 4096,
                      ),
                    ]}
                    onValueChange={([v]) => {
                      setCustomContextLength(
                        v === (ggufContextLength ?? 0) ? null : v,
                      );
                    }}
                  />
                  {ggufMaxContextLength != null &&
                    typeof ctxDisplayValue === "number" &&
                    ctxDisplayValue > ggufMaxContextLength && (
                      <p className="text-[11px] text-amber-500">
                        Exceeds estimated VRAM capacity (
                        {ggufMaxContextLength.toLocaleString()} tokens). The
                        model may use system RAM.
                      </p>
                    )}
                </div>
                <div className="grid grid-cols-[minmax(0,1fr)_65px] items-center gap-x-3">
                  <div className="min-w-0">
                    <div className="text-xs font-medium">KV Cache Dtype</div>
                    <div className="text-[11px] text-muted-foreground">
                      Quantize KV cache to reduce VRAM.
                    </div>
                  </div>
                  <div className="w-full min-w-0">
                    <Select
                      value={kvCacheDtype ?? "f16"}
                      onValueChange={(v) => {
                        setKvCacheDtype(v === "f16" ? null : v);
                      }}
                    >
                      <SelectTrigger className="grid h-7 w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 px-2 py-0 text-xs [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="f16">f16</SelectItem>
                        <SelectItem value="bf16">bf16</SelectItem>
                        <SelectItem value="q8_0">q8_0</SelectItem>
                        <SelectItem value="q5_1">q5_1</SelectItem>
                        <SelectItem value="q4_1">q4_1</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                {!currentModelIsVision && (
                  <div className="grid grid-cols-[minmax(0,1fr)_65px] items-center gap-x-3">
                    <div className="min-w-0">
                      <div className="text-xs font-medium">
                        Speculative Decoding
                      </div>
                      <div className="text-[11px] text-muted-foreground">
                        Speed up generation with no VRAM cost.
                      </div>
                    </div>
                    <div className="w-full min-w-0">
                      <Select
                        value={speculativeType ?? "off"}
                        onValueChange={(v) => {
                          setSpeculativeType(v === "off" ? null : v);
                        }}
                      >
                        <SelectTrigger className="grid h-7 w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 px-2 py-0 text-xs [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="default">On</SelectItem>
                          <SelectItem value="off">Off</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                )}
                {modelSettingsDirty && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    <button
                      type="button"
                      onClick={() => onReloadModel?.()}
                      className="rounded-md bg-primary px-2.5 py-1 text-[11px] font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                    >
                      Apply
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setCustomContextLength(null);
                        setKvCacheDtype(loadedKvCacheDtype);
                        setSpeculativeType(loadedSpeculativeType);
                      }}
                      className="rounded-md border px-2.5 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-accent"
                    >
                      Reset
                    </button>
                  </div>
                )}
              </>
            )}
            {!isGguf && params.checkpoint && (
              <>
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-xs font-medium">Enable custom code</div>
                    <div className="text-[11px] text-muted-foreground">
                      Allow models with custom code (e.g. Nemotron). Only
                      enable if sure.
                    </div>
                  </div>
                  <Switch
                    checked={params.trustRemoteCode ?? false}
                    onCheckedChange={set("trustRemoteCode")}
                  />
                </div>
                {trustRemoteCodeMissing && (
                  <Alert className="border-amber-200/70 bg-amber-50/70 px-3 py-2 text-amber-950 dark:border-amber-900/70 dark:bg-amber-950/35 dark:text-amber-100">
                    <AlertTitle className="text-[11px] font-medium">
                      Keep custom code enabled for this model
                    </AlertTitle>
                    <AlertDescription className="text-[11px] text-amber-800 dark:text-amber-200">
                      This model requires custom code to load. You can edit the
                      toggle, but loading will stay blocked until it is turned
                      back on.
                    </AlertDescription>
                  </Alert>
                )}
              </>
            )}
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
              displayValue={params.topP === 1 ? "Off" : undefined}
            />
            <ParamSlider
              label="Top K"
              value={params.topK}
              min={0}
              max={100}
              step={1}
              onChange={set("topK")}
              displayValue={params.topK === 0 ? "Off" : undefined}
            />
            <ParamSlider
              label="Min P"
              value={params.minP}
              min={0}
              max={1}
              step={0.01}
              onChange={set("minP")}
            />
            <ParamSlider
              label="Repetition Penalty"
              value={params.repetitionPenalty}
              min={1}
              max={2}
              step={0.05}
              onChange={set("repetitionPenalty")}
              displayValue={params.repetitionPenalty === 1 ? "Off" : undefined}
            />
            <ParamSlider
              label="Presence Penalty"
              value={params.presencePenalty}
              min={0}
              max={2}
              step={0.1}
              onChange={set("presencePenalty")}
              displayValue={params.presencePenalty === 0 ? "Off" : undefined}
            />
            {!isGguf && (
              <ParamSlider
                label="Max Seq Length"
                value={params.maxSeqLength}
                min={128}
                max={32768}
                step={128}
                onChange={set("maxSeqLength")}
              />
            )}
            <ParamSlider
              label="Max Tokens"
              value={params.maxTokens}
              min={64}
              max={isGguf && ggufContextLength ? ggufContextLength : 32768}
              step={64}
              onChange={set("maxTokens")}
              displayValue={
                isGguf &&
                ggufContextLength &&
                params.maxTokens >= ggufContextLength
                  ? "Max"
                  : undefined
              }
            />
          </div>
        </CollapsibleSection>

        <CollapsibleSection icon={Wrench01Icon} label="Tools">
          <div className="flex flex-col gap-3 py-1">
            <AutoHealToolCallsToggle />
            <MaxToolCallsSlider />
            <ToolCallTimeoutSlider />
          </div>
        </CollapsibleSection>

        <ChatTemplateSection onReloadModel={onReloadModel} />
      </div>
      </div>
      <Dialog
        open={systemPromptEditorOpen}
        onOpenChange={(nextOpen) => {
          setSystemPromptEditorOpen(nextOpen);
        }}
      >
        <DialogContent
          className="corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-3xl"
          overlayClassName="bg-background/35 supports-backdrop-filter:backdrop-blur-[1px]"
        >
          <DialogHeader>
            <DialogTitle>Edit System Prompt</DialogTitle>
            <DialogDescription>
              This prompt is part of the current configuration and saves with
              the preset.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <div className="space-y-0.5 px-0.5">
              <div className="text-[11px] font-medium">Prompt editor</div>
              <p className="text-[11px] text-muted-foreground">
                Use this for longer edits. Save writes back to the active
                configuration only.
              </p>
            </div>
            <Textarea
              value={systemPromptDraft}
              onChange={(event) => setSystemPromptDraft(event.target.value)}
              placeholder="You are a helpful assistant..."
              fieldSizing="fixed"
              className="min-h-[24rem] max-h-[50vh] overflow-y-auto text-sm leading-6 corner-squircle"
              rows={14}
            />
          </div>
          <DialogFooter className="flex-wrap gap-2 sm:justify-between">
            <Button
              type="button"
              variant="ghost"
              onClick={() => {
                setSystemPromptDraft(params.systemPrompt);
                setSystemPromptEditorOpen(false);
              }}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={saveSystemPromptEditor}
              disabled={!systemPromptEditorDirty}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );

  if (isMobile) {
    return (
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent side="right" className="w-[18rem] p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Configuration</SheetTitle>
            <SheetDescription>Chat inference settings</SheetDescription>
          </SheetHeader>
          <div className="flex h-full flex-col">{settingsContent}</div>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <aside
      className={`relative z-50 shrink-0 h-full overflow-hidden bg-muted/70 ${open ? "w-[17rem]" : "w-0"}`}
    >
      <div className="h-full w-[17rem]">{settingsContent}</div>
    </aside>
  );
}

function MaxToolCallsSlider() {
  const maxToolCalls = useChatRuntimeStore((s) => s.maxToolCallsPerMessage);
  const setMaxToolCalls = useChatRuntimeStore(
    (s) => s.setMaxToolCallsPerMessage,
  );

  // Slider range 0-41; 41 maps to 9999 ("Max")
  const sliderValue = maxToolCalls >= 9999 ? 41 : Math.min(maxToolCalls, 40);

  return (
    <ParamSlider
      label="Max Tool Calls Per Message"
      value={sliderValue}
      min={0}
      max={41}
      step={1}
      onChange={(v) => setMaxToolCalls(v >= 41 ? 9999 : v)}
      displayValue={
        sliderValue >= 41 ? "Max" : sliderValue === 0 ? "Off" : undefined
      }
    />
  );
}

function ToolCallTimeoutSlider() {
  const timeout = useChatRuntimeStore((s) => s.toolCallTimeout);
  const setTimeout_ = useChatRuntimeStore((s) => s.setToolCallTimeout);

  // Slider 1-31; 31 maps to 9999 ("Max")
  const sliderValue = timeout >= 9999 ? 31 : Math.min(Math.max(timeout, 1), 30);

  const displayValue =
    sliderValue >= 31
      ? "Max"
      : sliderValue === 1
        ? "1 minute"
        : `${sliderValue} minutes`;

  return (
    <ParamSlider
      label="Max Tool Call Duration"
      value={sliderValue}
      min={1}
      max={31}
      step={1}
      onChange={(v) => setTimeout_(v >= 31 ? 9999 : v)}
      displayValue={displayValue}
    />
  );
}

function AutoHealToolCallsToggle() {
  const autoHealToolCalls = useChatRuntimeStore((s) => s.autoHealToolCalls);
  const setAutoHealToolCalls = useChatRuntimeStore(
    (s) => s.setAutoHealToolCalls,
  );

  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <div className="text-xs font-medium">Auto Heal Tool Calls 🦥</div>
        <div className="text-[11px] text-muted-foreground">
          Fix malformed tool calls from the model automatically.
        </div>
      </div>
      <Switch
        checked={autoHealToolCalls}
        onCheckedChange={setAutoHealToolCalls}
      />
    </div>
  );
}

function ChatTemplateSection({
  onReloadModel,
}: {
  onReloadModel?: () => void;
}) {
  const defaultTemplate = useChatRuntimeStore((s) => s.defaultChatTemplate);
  const override = useChatRuntimeStore((s) => s.chatTemplateOverride);
  const setOverride = useChatRuntimeStore((s) => s.setChatTemplateOverride);

  if (!defaultTemplate) return null;

  const displayValue = override ?? defaultTemplate;
  const isModified = override !== null;

  return (
    <CollapsibleSection icon={CodeIcon} label="Chat Template">
      <div className="flex flex-col gap-2 py-1">
        <Textarea
          value={displayValue}
          onChange={(e) => setOverride(e.target.value)}
          className="min-h-32 max-h-64 overflow-y-auto font-mono text-[10px] leading-relaxed md:text-[10px] corner-squircle"
          rows={6}
          spellCheck={false}
        />
        <div className="flex flex-wrap gap-1.5">
          {isModified && (
            <>
              <button
                type="button"
                onClick={() => {
                  onReloadModel?.();
                }}
                className="rounded-md bg-primary px-2.5 py-1 text-[11px] font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                Apply & Reload
              </button>
              <button
                type="button"
                onClick={() => setOverride(null)}
                className="rounded-md border px-2.5 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-accent"
              >
                Revert changes
              </button>
            </>
          )}
        </div>
      </div>
    </CollapsibleSection>
  );
}
