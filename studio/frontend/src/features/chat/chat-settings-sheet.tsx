// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/components/assistant-ui/model-selector";
import { HubModelPicker } from "@/components/assistant-ui/model-selector/pickers";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
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
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useIsMobile } from "@/hooks/use-mobile";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowTurnBackwardIcon,
  CodeIcon,
  Delete02Icon,
  File01Icon,
  FloppyDiskIcon,
  InformationCircleIcon,
  LayoutAlignRightIcon,
  Logout01Icon,
  Settings02Icon,
  Settings05Icon,
  SlidersHorizontalIcon,
  Wrench01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { ChevronDown } from "lucide-react";
import {
  Fragment,
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { getCachedDocumentSupport } from "./api/chat-api";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import type { DocumentSupport } from "./types";
import {
  applyPresetParams,
  BUILTIN_PRESET_NAMES,
  BUILTIN_PRESETS,
  defaultInferenceParams,
  getBuiltinVariantName,
  getOrderedPresets,
  getPresetOwnedConfigKey,
  getPresetSaveState,
  getPresetSource,
  getUniquePresetName,
  isSamePresetConfig,
  normalizeCustomPresets,
  toPresetParams,
  type Preset,
} from "./presets/preset-policy";
import {
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "./types/runtime";
import {
  OCR_MODEL_PRESETS,
  resolveOcrModelTarget,
} from "./utils/ocr-model-presets";

export { defaultInferenceParams, type Preset } from "./presets/preset-policy";
export type { InferenceParams } from "./types/runtime";

interface LegacySystemPromptTemplate {
  name: string;
  content: string;
}

const CHAT_PRESETS_KEY = "unsloth_chat_custom_presets";
const CHAT_ACTIVE_PRESET_KEY = "unsloth_chat_active_preset";
const LEGACY_CHAT_SYSTEM_PROMPTS_KEY = "unsloth_chat_system_prompts";
const LEGACY_CHAT_SYSTEM_PROMPTS_MIGRATED_KEY =
  "unsloth_chat_system_prompts_migrated";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function saveCustomPresets(presets: Preset[]): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(CHAT_PRESETS_KEY, JSON.stringify(presets));
  } catch {
    // ignore
  }
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
        getPresetOwnedConfigKey(preset.params),
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
        const configKey = getPresetOwnedConfigKey(importedParams);
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
    const mergedPresets = normalizeCustomPresets([
      ...presets,
      ...importedPresets,
    ]);
    saveCustomPresets(mergedPresets);
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
    const normalized = normalizeCustomPresets(presets);
    if (JSON.stringify(normalized) !== JSON.stringify(presets)) {
      saveCustomPresets(normalized);
    }
    return migrateLegacySystemPromptTemplates(normalized);
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

function InfoHint({ children }: { children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label="More info"
          className="inline-flex size-4 shrink-0 cursor-help items-center justify-center rounded-full text-muted-foreground/70 transition-colors hover:text-[#383835] dark:hover:text-[#e8e8e8] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="left"
        sideOffset={8}
        className="tooltip-compact [&_span>svg]:hidden! duration-0 max-w-64"
      >
        {children}
      </TooltipContent>
    </Tooltip>
  );
}

/**
 * Editable numeric value display.
 *
 * Renders as a single <input> that *looks* like text by default —
 * transparent background, no border, no ring — and only shows a faint
 * surface tint on hover/focus to signal editability. When unfocused,
 * the input shows the formatted display string (`displayValue ?? value`,
 * so labels like "Off" / "Max" still render); on focus, it switches to
 * the raw numeric value, selects it, and accepts free text input.
 * Commit happens on blur or Enter; Escape reverts. The clamp-to-range
 * happens on commit so users can type intermediate values without the
 * input fighting them mid-keystroke. Single component shared by every
 * slider value and the Context Length input so the click-to-edit
 * affordance is consistent across the panel.
 */
function snapToStep(
  value: number,
  step: number,
  min?: number,
  max?: number,
): number {
  const lo = min ?? Number.NEGATIVE_INFINITY;
  const hi = max ?? Number.POSITIVE_INFINITY;
  const clamped = Math.min(Math.max(value, lo), hi);
  const stepStr = String(step);
  const decimals = stepStr.includes(".") ? stepStr.split(".")[1].length : 0;
  const base = Number.isFinite(lo) ? lo : 0;
  const snapped = base + Math.round((clamped - base) / step) * step;
  const reclamped = Math.min(Math.max(snapped, lo), hi);
  return Number(reclamped.toFixed(decimals));
}

function NumericValueInput({
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
  className,
  ariaLabel,
  size: sizeAttr,
}: {
  value: number;
  min?: number;
  max?: number;
  step: number;
  onChange: (v: number) => void;
  displayValue?: string;
  className?: string;
  ariaLabel?: string;
  size?: number;
}) {
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState("");
  const cancelBlurCommitRef = useRef(false);

  const commit = (raw: string) => {
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) {
      return;
    }
    const final = snapToStep(parsed, step, min, max);
    if (final !== value) {
      onChange(final);
    }
  };

  return (
    <input
      type="text"
      inputMode="decimal"
      size={sizeAttr}
      value={focused ? draft : (displayValue ?? String(value))}
      aria-label={ariaLabel}
      onFocus={(e) => {
        cancelBlurCommitRef.current = false;
        setDraft(String(value));
        setFocused(true);
        // Defer the select() so it runs after the value swap above.
        const target = e.currentTarget;
        requestAnimationFrame(() => target.select());
      }}
      onBlur={() => {
        if (cancelBlurCommitRef.current) {
          cancelBlurCommitRef.current = false;
        } else {
          commit(draft);
        }
        setFocused(false);
      }}
      onChange={(e) => setDraft(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.currentTarget.blur();
        } else if (e.key === "Escape") {
          cancelBlurCommitRef.current = true;
          setDraft(String(value));
          e.currentTarget.blur();
        }
      }}
      className={cn("panel-number-input", className)}
    />
  );
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
  info,
  valueSize,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  displayValue?: string;
  info?: ReactNode;
  valueSize?: number;
}) {
  return (
    <div className="space-y-3.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            {label}
          </span>
          {info && <InfoHint>{info}</InfoHint>}
        </div>
        <NumericValueInput
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          displayValue={displayValue}
          ariaLabel={label}
          size={valueSize ?? 6}
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(snapToStep(v, step, min, max))}
        className="panel-slider"
      />
    </div>
  );
}

function normalizeNonNegativeInteger(value: number): number {
  return Math.max(0, Math.round(value));
}

function parseNonNegativeIntegerInputValue(
  raw: string,
  fallback: number,
): number {
  if (raw.trim() === "") return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed)
    ? fallback
    : normalizeNonNegativeInteger(parsed);
}

const DOC_EXTRACT_SLIDER_MAXES = {
  maxFigures: 1000,
  maxVisualPayloads: 10,
  tokenBudget: 32000,
  extractConcurrency: 8,
} as const;

function InlineNumberInput({
  value,
  onCommit,
  disabled,
  ariaLabel,
}: {
  value: number;
  onCommit: (value: number) => void;
  disabled?: boolean;
  ariaLabel: string;
}) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  const commitDraft = useCallback(() => {
    const next = parseNonNegativeIntegerInputValue(draft, value);
    setDraft(String(next));
    onCommit(next);
  }, [draft, onCommit, value]);

  return (
    <Input
      type="number"
      min={0}
      step={1}
      inputMode="numeric"
      value={draft}
      onFocus={(event) => event.currentTarget.select()}
      onChange={(event) => setDraft(event.currentTarget.value)}
      onBlur={commitDraft}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
      disabled={disabled}
      aria-label={ariaLabel}
      className="h-5 w-[3.75rem] rounded border border-border/50 bg-transparent px-1.5 py-0 text-right !text-xs leading-none tabular-nums text-muted-foreground shadow-none transition-colors [appearance:textfield] hover:border-border focus-visible:border-primary focus-visible:ring-0 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50 md:!text-xs [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
    />
  );
}

function DocumentNumberSliderRow({
  label,
  tooltip,
  value,
  sliderMax,
  sliderMin = 0,
  step = 1,
  disabled,
  valueAriaLabel,
  onValueChange,
}: {
  label: string;
  tooltip: string;
  value: number;
  sliderMax: number;
  sliderMin?: number;
  step?: number;
  disabled?: boolean;
  valueAriaLabel: string;
  onValueChange: (value: number) => void;
}) {
  const effectiveMax = Math.max(1, sliderMax);
  const effectiveMin = Math.max(0, Math.min(sliderMin, effectiveMax));
  const sliderValue = Math.min(Math.max(value, effectiveMin), effectiveMax);

  return (
    <div className="space-y-2 py-2">
      <div className="flex items-center justify-between gap-3">
        <span className="flex min-w-0 flex-wrap items-center gap-1.5 text-xs font-medium">
          {label}
          <SettingInfoTooltip content={tooltip} />
        </span>
        <InlineNumberInput
          value={value}
          onCommit={onValueChange}
          disabled={disabled}
          ariaLabel={valueAriaLabel}
        />
      </div>
      <Slider
        min={effectiveMin}
        max={effectiveMax}
        step={step}
        value={[sliderValue]}
        onValueChange={([next]) => onValueChange(next ?? value)}
        disabled={disabled}
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
  label,
  children,
  defaultOpen = false,
  first = false,
}: {
  label: string;
  children?: ReactNode;
  defaultOpen?: boolean;
  first?: boolean;
}) {
  const [open, setOpen] = useState(() => {
    const saved = loadCollapsibleState();
    return Object.hasOwn(saved, label) ? saved[label] : defaultOpen;
  });

  return (
    <div
      className={cn(
        !first &&
          "border-t border-black/[0.13] dark:border-white/[0.09]",
      )}
    >
      <button
        type="button"
        onClick={() => {
          const next = !open;
          setOpen(next);
          saveCollapsibleOpen(label, next);
        }}
        className={cn(
          "flex w-full cursor-pointer items-center justify-between text-[12px] font-medium normal-case tracking-[0.04em] text-nav-fg-muted transition-colors hover:text-nav-fg focus-visible:outline-none focus-visible:ring-0",
          first ? "pt-4 pb-5" : "py-5",
        )}
      >
        <span className="leading-none">{label}</span>
        <span className="flex shrink-0 items-center leading-none">
          <ChevronDown
            className={cn("size-3.5", open ? "rotate-0" : "-rotate-90")}
          />
        </span>
      </button>
      {open && <div className="pb-7">{children}</div>}
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
  const hasModelContent = isGguf || Boolean(params.checkpoint);
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const setSpeculativeType = useChatRuntimeStore((s) => s.setSpeculativeType);
  const loadedSpeculativeType = useChatRuntimeStore(
    (s) => s.loadedSpeculativeType,
  );
  const modelRequiresTrustRemoteCode = useChatRuntimeStore(
    (s) => s.modelRequiresTrustRemoteCode,
  );
  const currentCheckpoint = params.checkpoint;
  const currentModelIsMultimodal = useChatRuntimeStore((s) => {
    if (s.loadedIsMultimodal) return true;
    const m = s.models.find((m) => m.id === currentCheckpoint);
    return (
      Boolean(m?.isVision) ||
      Boolean(m?.isAudio) ||
      Boolean(m?.hasAudioInput) ||
      m?.audioType === "audio_vlm"
    );
  });
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
  const setActivePresetSource = useChatRuntimeStore(
    (s) => s.setActivePresetSource,
  );
  const activePresetSource = useChatRuntimeStore((s) => s.activePresetSource);

  const ctxDisplayValue = customContextLength ?? ggufContextLength ?? "";
  const ctxMaxValue = ggufNativeContextLength ?? ggufContextLength ?? null;
  const kvDirty = kvCacheDtype !== loadedKvCacheDtype;
  const ctxDirty = customContextLength !== null;
  const specDirty = speculativeType !== loadedSpeculativeType;
  const modelSettingsDirty = kvDirty || ctxDirty || specDirty;
  const chatTemplateOverride = useChatRuntimeStore(
    (s) => s.chatTemplateOverride,
  );
  const loadedChatTemplateOverride = useChatRuntimeStore(
    (s) => s.loadedChatTemplateOverride,
  );
  const setChatTemplateOverride = useChatRuntimeStore(
    (s) => s.setChatTemplateOverride,
  );
  const templateDirty = chatTemplateOverride !== loadedChatTemplateOverride;
  const [customPresets, setCustomPresets] = useState<Preset[]>(() =>
    loadSavedCustomPresets(),
  );
  const [activePreset, setActivePreset] = useState(() => {
    const saved = loadSavedActivePreset();
    const available = new Set([
      ...BUILTIN_PRESETS.map((preset) => preset.name),
      ...customPresets.map((preset) => preset.name),
    ]);
    return available.has(saved) ? saved : "Default";
  });
  const [presetNameInput, setPresetNameInput] = useState(() => activePreset);
  const presetControlRowRef = useRef<HTMLDivElement>(null);
  const [presetMenuWidthPx, setPresetMenuWidthPx] = useState<
    number | undefined
  >(undefined);
  const [systemPromptEditorOpen, setSystemPromptEditorOpen] = useState(false);
  const [systemPromptDraft, setSystemPromptDraft] = useState("");
  const [activePresetBaseline, setActivePresetBaseline] = useState(params);
  const presets = useMemo(() => {
    return getOrderedPresets(customPresets);
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
  const hasUnsavedPresetChanges = useMemo(
    () => {
      if (activePresetDefinition == null) {
        return false;
      }
      if (activePresetDefinition.name === "Default") {
        return activePresetSource === "modified";
      }
      return !isSamePresetConfig(activePresetDefinition.params, params);
    },
    [activePresetDefinition, activePresetSource, params],
  );
  const presetSaveState = useMemo(
    () =>
      getPresetSaveState({
        rawName: presetNameInput,
        activePreset,
        presets,
        hasUnsavedPresetChanges,
      }),
    [activePreset, hasUnsavedPresetChanges, presetNameInput, presets],
  );
  const systemPromptEditorDirty = systemPromptDraft !== params.systemPrompt;
  const trustRemoteCodeMissing =
    Boolean(currentCheckpoint) &&
    modelRequiresTrustRemoteCode &&
    !(params.trustRemoteCode ?? false);

  function set<K extends keyof InferenceParams>(key: K) {
    return (v: InferenceParams[K]) => {
      const nextParams = { ...params, [key]: v };
      const nextSource = isSamePresetConfig(activePresetBaseline, nextParams)
        ? getPresetSource(activePreset)
        : "modified";
      setActivePresetSource(nextSource);
      onParamsChange(nextParams);
    };
  }

  function applyPreset(name: string) {
    const p = presets.find((pr) => pr.name === name);
    if (p) {
      onParamsChange({
        ...applyPresetParams(params, p.params),
      });
      setActivePreset(name);
      setActivePresetSource(getPresetSource(name));
      setPresetNameInput(name);
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
    const usedNames = new Set([
      ...BUILTIN_PRESET_NAMES,
      ...customPresets.map((preset) => preset.name),
    ]);
    const saveName = BUILTIN_PRESET_NAMES.has(trimmed)
      ? getBuiltinVariantName(trimmed, usedNames)
      : trimmed;
    setCustomPresets((prev) => {
      const next = prev.filter((p) => p.name !== saveName);
      const merged = [
        ...next,
        { name: saveName, params: toPresetParams(params) },
      ];
      saveCustomPresets(merged);
      return merged;
    });
    if (canUseStorage()) {
      try {
        localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, saveName);
      } catch {
        // ignore
      }
    }
    setActivePreset(saveName);
    setActivePresetSource("custom");
    setPresetNameInput(saveName);
  }

  function deletePreset(name: string) {
    const hasCustomPreset = customPresets.some(
      (preset) => preset.name === name,
    );
    if (!hasCustomPreset) {
      return;
    }
    const builtinPreset = BUILTIN_PRESETS.find(
      (preset) => preset.name === name,
    );
    const fallbackPreset =
      BUILTIN_PRESETS.find((preset) => preset.name === "Default") ??
      null;
    setCustomPresets((prev) => {
      const next = prev.filter((preset) => preset.name !== name);
      saveCustomPresets(next);
      return next;
    });
    if (activePreset === name) {
      if (fallbackPreset) {
        onParamsChange({
          ...applyPresetParams(params, fallbackPreset.params),
        });
        setActivePreset(fallbackPreset.name);
        setActivePresetSource("builtin-default");
        setPresetNameInput(fallbackPreset.name);
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
    if (activePresetSource !== "modified") {
      setActivePresetBaseline(params);
    }
  }, [activePresetSource, params]);

  useEffect(() => {
    if (presets.some((preset) => preset.name === activePreset)) {
      const expectedSource = getPresetSource(activePreset);
      if (
        activePresetSource !== "modified" &&
        activePresetSource !== expectedSource
      ) {
        setActivePresetSource(expectedSource);
      }
      return;
    }
    setActivePreset("Default");
    setActivePresetSource("builtin-default");
    if (canUseStorage()) {
      try {
        localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, "Default");
      } catch {
        // ignore
      }
    }
  }, [
    activePreset,
    activePresetSource,
    presets,
    setActivePresetSource,
  ]);

  useEffect(() => {
    setPresetNameInput(activePreset);
  }, [activePreset]);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen) {
        setSystemPromptEditorOpen(false);
      }
      onOpenChange?.(nextOpen);
    },
    [onOpenChange],
  );

  const settingsContent = (
    <>
      <div className="aui-thread-viewport relative h-full overflow-y-auto">
      <div className="sticky top-0 z-10 flex h-[48px] items-start gap-2 bg-panel-surface pl-[18px] pr-[14px] pt-[11px]">
        {isMobile ? (
          <span className="flex h-[34px] flex-1 items-center text-[15px] font-semibold tracking-[-0.01em] dark:tracking-[0.015em] text-nav-fg">
            Configuration
          </span>
        ) : (
          <>
            <span className="flex h-[34px] flex-1 items-center text-[15px] font-semibold tracking-[-0.01em] dark:tracking-[0.015em] text-nav-fg">
              Configuration
            </span>
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={() => onOpenChange?.(false)}
                  className="flex h-[34px] w-[34px] items-center justify-center rounded-[12px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Close configuration"
                >
                  <HugeiconsIcon
                    icon={LayoutAlignRightIcon}
                    strokeWidth={1.75}
                    className="size-icon"
                  />
                </button>
              </TooltipPrimitive.Trigger>
              <TooltipContent
                side="bottom"
                sideOffset={6}
                className="tooltip-compact"
              >
                Close configuration
              </TooltipContent>
            </Tooltip>
          </>
        )}
      </div>

      <div className="px-[18px] pt-3">
        {hasModelContent && (
        <CollapsibleSection label="Model" defaultOpen={true} first>
          <div className="flex flex-col gap-4 pt-1">
            {isGguf && (
              <>
                <div className="space-y-3.5">
                  <div className="flex items-center justify-between gap-3">
                    <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                      Context Length
                    </span>
                    <NumericValueInput
                      value={
                        typeof ctxDisplayValue === "number"
                          ? ctxDisplayValue
                          : (ggufContextLength ?? 0)
                      }
                      min={128}
                      max={ctxMaxValue ?? undefined}
                      step={1}
                      onChange={(v) => {
                        setCustomContextLength(
                          v === (ggufContextLength ?? 0) ? null : v,
                        );
                      }}
                      ariaLabel="Context Length"
                      size={8}
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
                      const snapped = Math.round(v);
                      setCustomContextLength(
                        snapped === (ggufContextLength ?? 0) ? null : snapped,
                      );
                    }}
                    className="panel-slider"
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
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                      KV Cache Dtype
                    </span>
                    <InfoHint>
                      Lower KV cache precision to save VRAM at the cost of some
                      quality. f16/bf16 are full precision; q8_0/q5_1/q4_1 are
                      quantized.
                    </InfoHint>
                  </div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    <Select
                      value={kvCacheDtype ?? "f16"}
                      onValueChange={(v) => {
                        setKvCacheDtype(v === "f16" ? null : v);
                      }}
                    >
                      <SelectTrigger
                        animateRadius={false}
                        icon={ArrowDown01Icon}
                        iconClassName="size-3.5"
                        className="grid h-7 w-[60px] min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-[10px] border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.07] px-2 py-0 text-[13px]! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0"
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
                        <SelectItem value="f16">f16</SelectItem>
                        <SelectItem value="bf16">bf16</SelectItem>
                        <SelectItem value="q8_0">q8_0</SelectItem>
                        <SelectItem value="q5_1">q5_1</SelectItem>
                        <SelectItem value="q4_1">q4_1</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                {!currentModelIsMultimodal && (
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                        Speculative Decoding
                      </span>
                      <InfoHint>
                        N-gram speculation; faster generation with negligible
                        VRAM overhead. Text-only models.
                      </InfoHint>
                    </div>
                    <Switch
                      className="panel-switch shrink-0"
                      checked={speculativeType != null}
                      onCheckedChange={(checked) => {
                        setSpeculativeType(checked ? "default" : null);
                      }}
                    />
                  </div>
                )}
              </>
            )}
            {!isGguf && params.checkpoint && (
              <>
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                      Enable custom code
                    </span>
                    <InfoHint>
                      Run custom Python from the model repo (e.g. Nemotron).
                      Only enable for trusted sources.
                    </InfoHint>
                  </div>
                  <Switch
                    className="panel-switch shrink-0"
                    checked={params.trustRemoteCode ?? false}
                    onCheckedChange={set("trustRemoteCode")}
                  />
                </div>
                {trustRemoteCodeMissing && (
                  <Alert className="rounded-[14px] border-amber-200/70 bg-amber-50/70 px-3 py-2 text-amber-950 dark:border-amber-900/70 dark:bg-amber-950/35 dark:text-amber-100">
                    <AlertTitle className="text-[12px] font-medium">
                      Keep custom code enabled for this model
                    </AlertTitle>
                    <AlertDescription className="text-[11.5px] leading-[1.45] text-amber-800 dark:text-amber-200">
                      This model requires custom code to load. You can edit the
                      toggle, but loading will stay blocked until it is turned
                      back on.
                    </AlertDescription>
                  </Alert>
                )}
              </>
            )}
            <ChatTemplateFields />
            {(modelSettingsDirty || templateDirty) && (
              <div className="flex flex-wrap gap-1.5 pt-1">
                <Button
                  type="button"
                  onClick={() => onReloadModel?.()}
                  size="sm"
                  className="h-7 px-3 text-[12px] font-medium tracking-nav bg-primary/92 text-primary-foreground hover:bg-primary"
                >
                  Apply
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setCustomContextLength(null);
                    setKvCacheDtype(loadedKvCacheDtype);
                    setSpeculativeType(loadedSpeculativeType);
                    setChatTemplateOverride(loadedChatTemplateOverride);
                  }}
                  className="h-7 px-3 text-[12px] font-medium tracking-nav text-muted-foreground"
                >
                  Reset
                </Button>
              </div>
            )}
          </div>
        </CollapsibleSection>
        )}

        <CollapsibleSection
          label="Preset"
          defaultOpen={true}
          first={!hasModelContent}
        >
          <div className="flex flex-col gap-3 pt-1">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <div
                  className="w-full min-w-0 cursor-pointer outline-none focus-visible:outline-none"
                  aria-label="Open preset list"
                >
                  <InputGroup className="panel-input-group">
                    <InputGroupInput
                      id="inference-preset-name"
                      value={presetNameInput}
                      onChange={(e) => setPresetNameInput(e.target.value)}
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && presetSaveState.canSubmit) {
                          e.preventDefault();
                          savePresetWithName(presetNameInput);
                        }
                        e.stopPropagation();
                      }}
                      placeholder="Preset name"
                      maxLength={80}
                      autoComplete="off"
                      className={cn(
                        "!h-9 min-h-0 min-w-0 self-stretch !pl-3.5 !pr-2 py-0 text-[13px] font-medium leading-9 text-nav-fg md:text-[13px]",
                        presetSaveState.isSaveReady &&
                          "placeholder:text-primary/50",
                      )}
                      aria-label="Inference preset name"
                    />
                    <InputGroupAddon
                      align="inline-end"
                      className="min-h-0 shrink-0 gap-0 self-stretch border-0 py-0 pl-0 !pr-1 has-[>button]:mr-0"
                    >
                      <span
                        className="!h-7 min-h-7 !w-7 min-w-7 shrink-0 self-center inline-flex items-center justify-center rounded-full border-0 px-0 text-[#a0a097] dark:text-nav-fg pointer-events-none"
                        aria-hidden="true"
                      >
                        <HugeiconsIcon
                          icon={ArrowDown01Icon}
                          className="size-3.5"
                          strokeWidth={2}
                        />
                      </span>
                    </InputGroupAddon>
                  </InputGroup>
                </div>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="start"
                sideOffset={6}
                className="menu-soft-surface ring-0 border-0 rounded-lg p-1.5"
              >
                {presets.map((p, index) => (
                  <Fragment key={p.name}>
                    <DropdownMenuItem
                      onSelect={() => applyPreset(p.name)}
                      className="flex min-h-9 items-center px-3 py-0 text-[13px] font-medium leading-[1.4] tracking-nav"
                    >
                      {p.name}
                    </DropdownMenuItem>
                    {index === BUILTIN_PRESETS.length - 1 &&
                      presets.length > BUILTIN_PRESETS.length && (
                        <DropdownMenuSeparator className="mx-3 my-1.5 h-px bg-black/8 dark:bg-white/8" />
                      )}
                  </Fragment>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <div className="grid grid-cols-2 gap-3">
              <Button
                type="button"
                onClick={() => savePresetWithName(presetNameInput)}
                disabled={!presetSaveState.canSubmit}
                variant={presetSaveState.isSaveReady ? "default" : "outline"}
                size="sm"
                className={cn(
                  "h-9 w-full rounded-[10px] text-[13px] font-medium tracking-nav",
                  presetSaveState.isSaveReady &&
                    "bg-primary text-primary-foreground hover:bg-primary/90",
                )}
                title={presetSaveState.title}
                aria-label={presetSaveState.title}
              >
                {presetSaveState.buttonLabel}
              </Button>
              <Button
                type="button"
                onClick={() => deletePreset(activePreset)}
                disabled={!activeCustomPreset}
                variant="outline"
                size="sm"
                className="h-9 w-full rounded-[10px] text-[13px] font-medium tracking-nav text-muted-foreground"
                title={
                  activeCustomPreset
                    ? activeBuiltinPreset
                      ? "Reset selected preset to built-in defaults"
                      : "Delete selected preset"
                    : "No saved override to delete"
                }
              >
                Delete
              </Button>
            </div>
          </div>
        </CollapsibleSection>

        <CollapsibleSection label="System Prompt" defaultOpen={true}>
          <button
            type="button"
            onClick={openSystemPromptEditor}
            aria-label="Edit system prompt"
            className={cn(
              "panel-text-surface mt-1 flex w-full h-20 overflow-hidden cursor-pointer items-start px-3.5 py-2.5 text-left text-[13px] font-medium leading-relaxed corner-squircle focus-visible:outline-none focus-visible:border-ring focus-visible:ring-[1px] focus-visible:ring-ring/40",
              params.systemPrompt
                ? "text-nav-fg"
                : "text-muted-foreground",
            )}
          >
            <span className="block line-clamp-3 whitespace-pre-wrap break-words">
              {params.systemPrompt ||
                "Example: You are a helpful assistant..."}
            </span>
          </button>
        </CollapsibleSection>

        <CollapsibleSection label="Sampling" defaultOpen={true}>
          <div className="flex flex-col gap-5 pt-1">
            <ParamSlider
              label="Temperature"
              value={params.temperature}
              min={0}
              max={2}
              step={0.01}
              onChange={set("temperature")}
              info="Controls randomness. Lower values make output focused and deterministic; higher values increase variety and creativity."
            />
            <ParamSlider
              label="Top P"
              value={params.topP}
              min={0}
              max={1}
              step={0.05}
              onChange={set("topP")}
              displayValue={params.topP === 1 ? "Off" : undefined}
              info="Nucleus sampling. Restricts choices to the smallest set of tokens whose cumulative probability reaches this threshold. 1.0 = off."
            />
            <ParamSlider
              label="Top K"
              value={params.topK}
              min={0}
              max={100}
              step={1}
              onChange={set("topK")}
              displayValue={params.topK === 0 ? "Off" : undefined}
              info="Limits sampling to the K most likely tokens at each step. 0 = off."
            />
            <ParamSlider
              label="Min P"
              value={params.minP}
              min={0}
              max={1}
              step={0.01}
              onChange={set("minP")}
              info="Drops tokens whose probability is below this fraction of the top token's probability. Filters unlikely candidates."
            />
            <ParamSlider
              label="Repetition Penalty"
              value={params.repetitionPenalty}
              min={1}
              max={2}
              step={0.05}
              onChange={set("repetitionPenalty")}
              displayValue={params.repetitionPenalty === 1 ? "Off" : undefined}
              info="Down-weights tokens that have already appeared, reducing repetition. 1.0 = off; higher values penalize more strongly."
            />
            <ParamSlider
              label="Presence Penalty"
              value={params.presencePenalty}
              min={0}
              max={2}
              step={0.1}
              onChange={set("presencePenalty")}
              displayValue={params.presencePenalty === 0 ? "Off" : undefined}
              info="Penalizes any token that has already appeared at least once, encouraging the model to introduce new topics. 0 = off."
            />
            {!isGguf && (
              <ParamSlider
                label="Max Seq Length"
                value={params.maxSeqLength}
                min={128}
                max={32768}
                step={128}
                onChange={set("maxSeqLength")}
                info="Maximum context window size in tokens — input prompt plus generated output combined. Capped by the model's trained limit."
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
              info="Maximum number of tokens to generate per response. Generation stops at this limit or when the model emits an end-of-sequence token."
            />
          </div>
        </CollapsibleSection>

        <CollapsibleSection label="Tools">
          <div className="flex flex-col gap-5 pt-1">
            <AutoHealToolCallsToggle />
            <MaxToolCallsSlider />
            <ToolCallTimeoutSlider />
          </div>
        </CollapsibleSection>

        <DocumentExtractionSection />
      </div>
      </div>
      <Dialog
        open={open && systemPromptEditorOpen}
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
              className="min-h-[24rem] max-h-[50vh] overflow-y-auto text-sm leading-6 corner-squircle focus-visible:border-input focus-visible:ring-0"
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
      <Sheet open={open} onOpenChange={handleOpenChange}>
        <SheetContent side="right" className="w-[18rem] p-0 font-heading">

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
      className={`relative z-50 shrink-0 h-full overflow-hidden bg-panel-surface text-panel-surface-fg font-heading ${open ? "w-[17rem] border-l border-sidebar-border" : "w-0"}`}
    >
      <div className="h-full w-full">{settingsContent}</div>
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
      info="Cap on tool/function calls the model may invoke within a single response. 0 disables tool use; Max removes the cap."
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
      valueSize={10}
      info="Per-call wall-clock limit. Long-running tool executions are terminated when this elapses; the model continues with what completed."
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
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
          Auto-Healing Tool Calls
        </span>
        <InfoHint>
          Unsloth auto-fixes broken tool calls so inference output is never
          broken.
        </InfoHint>
      </div>
      <Switch
        className="panel-switch"
        checked={autoHealToolCalls}
        onCheckedChange={setAutoHealToolCalls}
      />
    </div>
  );
}

type DocExtractMode = "off" | "text" | "images" | "scanned";

const DOC_EXTRACT_MODES: ReadonlyArray<{
  value: DocExtractMode;
  label: string;
}> = [
  { value: "off", label: "Off" },
  { value: "text", label: "Text" },
  { value: "images", label: "Images" },
  { value: "scanned", label: "Scanned" },
];

function getDocExtractModeHelp(mode: DocExtractMode, hasVlm: boolean): string {
  switch (mode) {
    case "off":
      return "Extraction disabled. Uploaded documents are skipped.";
    case "text":
      return "Extract text only. Best for born-digital PDFs and Office files.";
    case "images":
      return hasVlm
        ? "Extract text plus figures as image inputs for the vision model."
        : "Text with figure/page citations. Load a vision model to include images.";
    case "scanned":
      return hasVlm
        ? "Render pages as images for OCR. Use for scanned or image-only PDFs."
        : "Renders pages as images. Load a vision model for OCR.";
  }
}

function getDocExtractModePreset(
  mode: DocExtractMode,
  hasVlm: boolean,
): Record<string, unknown> {
  switch (mode) {
    case "off":
      return { enabled: false };
    case "text":
      return {
        enabled: true,
        useVlmOcr: false,
        describeImages: false,
        maxFigures: 0,
        maxVisualPayloads: 0,
      };
    case "images":
      return {
        enabled: true,
        useVlmOcr: false,
        describeImages: hasVlm,
        maxFigures: 20,
        maxVisualPayloads: hasVlm ? 3 : 0,
      };
    case "scanned":
      return {
        enabled: true,
        useVlmOcr: true,
        describeImages: hasVlm,
        maxFigures: 20,
        maxVisualPayloads: hasVlm ? 3 : 0,
      };
  }
}

function deriveDocExtractMode(docExtract: {
  enabled: boolean;
  useVlmOcr: boolean;
  describeImages: boolean;
  maxFigures: number;
  maxVisualPayloads: number;
}): DocExtractMode {
  if (!docExtract.enabled) return "off";
  if (docExtract.useVlmOcr) return "scanned";
  if (
    docExtract.maxFigures > 0 ||
    docExtract.describeImages ||
    docExtract.maxVisualPayloads > 0
  ) {
    return "images";
  }
  return "text";
}

function SettingInfoTooltip({ content }: { content: string }) {
  return (
    <Tooltip>
      <TooltipPrimitive.Trigger asChild={true}>
        <button
          type="button"
          aria-label="More info"
          className="inline-flex size-3.5 items-center justify-center rounded-sm text-muted-foreground/70 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            className="size-3.5"
            strokeWidth={2}
          />
        </button>
      </TooltipPrimitive.Trigger>
      <TooltipContent
        side="top"
        sideOffset={6}
        className="max-w-[240px] text-[11px] leading-relaxed"
      >
        {content}
      </TooltipContent>
    </Tooltip>
  );
}

function DocumentExtractionSection() {
  const docExtract = useChatRuntimeStore((s) => s.docExtract);
  const setDocExtract = useChatRuntimeStore((s) => s.setDocExtract);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const trustRemoteCode = useChatRuntimeStore(
    (s) => s.params.trustRemoteCode ?? false,
  );
  const ocrPhase = useChatRuntimeStore((s) => s.ocrPhase);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const allModels = useChatRuntimeStore((s) => s.models);
  const [ocrPickerOpen, setOcrPickerOpen] = useState(false);
  const reducedMotion = useReducedMotion();

  const [support, setSupport] = useState<DocumentSupport | null>(null);
  const [probing, setProbing] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const runProbe = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setProbing(true);
    void getCachedDocumentSupport(ctrl.signal)
      .then((result) => {
        if (!ctrl.signal.aborted) setSupport(result);
      })
      .catch(() => {
        if (!ctrl.signal.aborted) setSupport(null);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setProbing(false);
      });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setProbing(true);
    void getCachedDocumentSupport(ctrl.signal)
      .then((result) => {
        if (!cancelled) setSupport(result);
      })
      .catch(() => {
        if (!cancelled) setSupport(null);
      })
      .finally(() => {
        if (!cancelled) setProbing(false);
      });
    return () => {
      cancelled = true;
      ctrl.abort();
    };
  }, [checkpoint]);

  const extractorReady = support?.extraction_available ?? false;
  const unavailableFormatCount = Object.keys(
    support?.unavailable_formats ?? {},
  ).length;
  const extractorLimited = extractorReady && unavailableFormatCount > 0;
  const vlm = support?.vlm;
  const hasVlm = vlm?.is_vlm ?? false;
  const ocrTarget = resolveOcrModelTarget(docExtract);
  const ocrSelected = ocrTarget !== null;
  const ocrModelId = ocrTarget?.modelId ?? "";
  const defaultOcrLabel = hasVlm ? vlm?.model_name || "Loaded VLM" : "None";
  const selectedOcrLabel =
    ocrTarget?.label ??
    (docExtract.ocrModel === "default"
      ? `Default: ${defaultOcrLabel}`
      : "None");
  const defaultOcrSelected = docExtract.ocrModel === "default";
  const noneOcrSelected = docExtract.ocrModel === "none";
  const defaultUsesLoadedVlm = defaultOcrSelected && hasVlm;
  const visionAvailableForExtraction = hasVlm || ocrSelected;
  // Scanned mode is normally gated on a vision-capable chat model, but a
  // selected dedicated OCR model satisfies that requirement at extract time.
  const ocrControlsDisabled = modelLoading || ocrPhase !== "idle";
  const trcMissing =
    ocrSelected &&
    (ocrTarget?.requiresTrustRemoteCode ?? false) &&
    !trustRemoteCode;
  const visionReadyForExtraction =
    visionAvailableForExtraction && !trcMissing;
  const canScan = extractorReady && visionReadyForExtraction;
  const activeMode = deriveDocExtractMode(docExtract);

  // OCR-picker model list: 3 OCR presets pinned at top + the user's
  // vision-capable downloaded models filtered in below.
  const ocrPickerModels = useMemo<ModelOption[]>(() => {
    const presetIds = new Set(OCR_MODEL_PRESETS.map((p) => p.modelId));
    const presetEntries: ModelOption[] = OCR_MODEL_PRESETS.map((preset) => ({
      id: preset.modelId,
      name: preset.label,
      description: "OCR preset",
    }));
    const userEntries: ModelOption[] = allModels
      .filter((m) => m.isVision && !presetIds.has(m.id))
      .map((m) => ({
        id: m.id,
        name: m.name,
        description: m.description,
        isGguf: m.isGguf,
      }));
    return [...presetEntries, ...userEntries];
  }, [allModels]);

  const handleOcrSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      const matchedPreset = OCR_MODEL_PRESETS.find((p) => p.modelId === id);
      if (matchedPreset) {
        setDocExtract({
          ocrModel: matchedPreset.id,
          customOcrModelId: "",
          customOcrGgufVariant: null,
        });
      } else {
        setDocExtract({
          ocrModel: "custom",
          customOcrModelId: id,
          customOcrGgufVariant: meta.ggufVariant ?? null,
        });
      }
      setOcrPickerOpen(false);
    },
    [setDocExtract],
  );

  const handleOcrDefault = useCallback(() => {
    setDocExtract({
      ocrModel: "default",
      customOcrModelId: "",
      customOcrGgufVariant: null,
    });
    setOcrPickerOpen(false);
  }, [setDocExtract]);

  const handleOcrNone = useCallback(() => {
    setDocExtract({
      ocrModel: "none",
      customOcrModelId: "",
      customOcrGgufVariant: null,
    });
    setOcrPickerOpen(false);
  }, [setDocExtract]);
  const setVisualPayloadLimit = (value: number): void => {
    const next = normalizeNonNegativeInteger(value);
    setDocExtract({
      maxVisualPayloads: next,
    });
  };
  const setFigureReferenceLimit = (value: number): void => {
    const next = normalizeNonNegativeInteger(value);
    setDocExtract({
      maxFigures: next,
    });
  };
  const setTokenBudget = (value: number): void => {
    const next = normalizeNonNegativeInteger(value);
    setDocExtract({
      tokenBudget: next,
    });
  };
  const setExtractConcurrency = (value: number): void => {
    const next = Math.max(
      1,
      Math.min(
        DOC_EXTRACT_SLIDER_MAXES.extractConcurrency,
        normalizeNonNegativeInteger(value),
      ),
    );
    setDocExtract({
      extractConcurrency: next,
    });
  };

  function applyMode(mode: DocExtractMode) {
    // OCR selection grants vision capability for the extraction window, so
    // describe-images and visual-payload defaults should match the
    // "VLM available" branch even if no VLM is loaded right now.
    setDocExtract(getDocExtractModePreset(mode, visionReadyForExtraction));
  }

  const statusLabel = probing
    ? "Checking"
    : extractorLimited
      ? "Limited"
    : extractorReady
      ? "Ready"
      : "Unavailable";
  const vlmLabel = probing
    ? "Checking vision model"
    : hasVlm
      ? vlm?.model_name || "Vision model"
      : "No vision model";
  const modeHelp = canScan
    ? getDocExtractModeHelp(activeMode, visionReadyForExtraction)
    : getDocExtractModeHelp(activeMode, hasVlm);
  const canCaption = visionReadyForExtraction && docExtract.maxFigures > 0;

  return (
    <CollapsibleSection label="Document extraction">
      <div className="flex flex-col gap-3 py-1">
        {!extractorReady && !probing && (
          <Alert className="border-amber-200/70 bg-amber-50/70 px-3 py-2 text-amber-950 dark:border-amber-900/70 dark:bg-amber-950/35 dark:text-amber-100">
            <AlertTitle className="text-[11px] font-medium">
              Document extraction unavailable
            </AlertTitle>
            <AlertDescription className="text-[11px] text-amber-800 dark:text-amber-200">
              Re-run Studio setup to install the server-side parser
              dependencies.
            </AlertDescription>
          </Alert>
        )}

        {/* Compact status pill */}
        <div className="flex items-center justify-between gap-2 rounded-md border bg-muted/30 px-2.5 py-1.5 text-[11px]">
          <div className="flex min-w-0 items-center gap-1.5">
            <span
              className={cn(
                "size-1.5 shrink-0 rounded-full",
                extractorReady ? "bg-emerald-500" : "bg-amber-500",
              )}
              aria-hidden="true"
            />
            <span className="font-medium">{statusLabel}</span>
            <span className="text-muted-foreground">·</span>
            <span className="truncate text-muted-foreground">{vlmLabel}</span>
          </div>
          {!extractorReady && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-5 shrink-0 px-1.5 text-[11px]"
              onClick={runProbe}
              disabled={probing}
              aria-label="Retry capability probe"
            >
              {probing ? <Spinner className="size-3" /> : "Retry"}
            </Button>
          )}
        </div>

        {/* OCR model — Default follows the loaded VLM when available; explicit
            preset/custom choices temporarily load a dedicated OCR model. */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-1.5 text-xs font-medium">
              OCR model
              <SettingInfoTooltip content="Default uses the currently loaded vision model when available. Pick a dedicated OCR model to load it only for extraction, then restore your chat model." />
            </span>
            {ocrPhase !== "idle" && (
              <span
                className="text-[11px] text-muted-foreground tabular-nums"
                aria-live="polite"
              >
                {ocrPhase === "validating" && "Validating…"}
                {ocrPhase === "unloading" && "Unloading chat model…"}
                {ocrPhase === "loading_ocr" &&
                  `Loading ${ocrTarget?.label ?? "OCR model"}…`}
                {ocrPhase === "extracting" && "Extracting…"}
                {ocrPhase === "restoring" && "Restoring chat model…"}
                {ocrPhase === "error" && "Error"}
              </span>
            )}
          </div>
          <Popover open={ocrPickerOpen} onOpenChange={setOcrPickerOpen}>
            <PopoverTrigger asChild={true}>
              <button
                type="button"
                disabled={ocrControlsDisabled}
                aria-describedby="ocr-model-help"
                aria-haspopup="dialog"
                className="flex h-9 w-full items-center gap-2 rounded-md border border-input bg-transparent px-2.5 text-xs transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
              >
                {(ocrSelected || defaultUsesLoadedVlm) && (
                  <span
                    className="size-1.5 shrink-0 rounded-full bg-emerald-500"
                    aria-hidden="true"
                  />
                )}
                <span className="flex-1 truncate text-left font-medium">
                  {selectedOcrLabel}
                </span>
                {ocrTarget?.requiresTrustRemoteCode && (
                  <span className="shrink-0 rounded bg-amber-500/15 px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                    TRC
                  </span>
                )}
                <HugeiconsIcon
                  icon={ArrowDown01Icon}
                  className="size-3.5 shrink-0 text-muted-foreground"
                />
              </button>
            </PopoverTrigger>
            <PopoverContent
              side="bottom"
              align="start"
              sideOffset={4}
              collisionPadding={8}
              className="flex w-[min(16rem,calc(100vw-1rem))] max-w-[calc(100vw-1rem)] flex-col gap-0 p-1.5"
              style={{
                maxHeight: "var(--radix-popover-content-available-height)",
              }}
            >
              <div className="min-h-0 flex-1 overflow-y-auto">
                <div className="mb-1 border-b border-border/70 pb-1">
                  <button
                    type="button"
                    onClick={handleOcrDefault}
                    className={cn(
                      "flex w-full items-center gap-2 rounded-[6px] px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035]",
                      defaultOcrSelected && "bg-[#ececec] dark:bg-[#2e3035]",
                    )}
                  >
                    <span
                      className={cn(
                        "size-1.5 shrink-0 rounded-full",
                        defaultOcrSelected
                          ? "bg-emerald-500"
                          : "bg-muted-foreground/25",
                      )}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 flex-1 truncate">Default</span>
                    <span className="shrink-0 truncate text-[10px] text-muted-foreground">
                      {defaultOcrLabel}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={handleOcrNone}
                    className={cn(
                      "flex w-full items-center gap-2 rounded-[6px] px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035]",
                      noneOcrSelected && "bg-[#ececec] dark:bg-[#2e3035]",
                    )}
                  >
                    <span
                      className={cn(
                        "size-1.5 shrink-0 rounded-full",
                        noneOcrSelected
                          ? "bg-emerald-500"
                          : "bg-muted-foreground/25",
                      )}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 flex-1 truncate">None</span>
                    <span className="shrink-0 text-[10px] text-muted-foreground">
                      No override
                    </span>
                  </button>
                </div>
                <HubModelPicker
                  models={ocrPickerModels}
                  value={ocrModelId}
                  onSelect={handleOcrSelect}
                />
              </div>
              {!defaultOcrSelected && (
                <div className="mt-2 shrink-0 border-t border-border/70 pt-2">
                  <button
                    type="button"
                    onClick={handleOcrDefault}
                    className="flex w-full items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                    title="Return OCR model selection to Default"
                  >
                    <HugeiconsIcon icon={Logout01Icon} className="size-3.5" />
                    Return to default
                  </button>
                </div>
              )}
            </PopoverContent>
          </Popover>
          <p
            id="ocr-model-help"
            className="text-[11px] leading-relaxed text-muted-foreground"
          >
            {ocrSelected
              ? `Scanned PDFs use ${ocrTarget?.label} for OCR/captions, then return to your chat model.`
              : defaultOcrSelected
                ? hasVlm
                  ? `Default uses ${defaultOcrLabel} for OCR/captions.`
                  : "Default resolves to None until a vision model is loaded."
                : "No dedicated OCR model is selected."}
          </p>
          {trcMissing && (
            <p className="text-[11px] text-amber-500">
              {ocrTarget?.label} requires <em>Enable custom code</em>. Turn it
              on under Inference settings before scanning.
            </p>
          )}
        </div>

        {/* Mode segmented — matches theme-segmented idiom */}
        <div>
          <div className="mb-1.5 text-xs font-medium">Mode</div>
          <div
            className="grid grid-cols-4 items-center rounded-md border border-border bg-muted/30 p-0.5"
            role="radiogroup"
            aria-label="Document extraction mode"
          >
            {DOC_EXTRACT_MODES.map((opt) => {
              const active = activeMode === opt.value;
              const disabled =
                (!extractorReady && opt.value !== "off") ||
                (opt.value === "scanned" && !canScan);
              return (
                <button
                  key={opt.value}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  disabled={disabled}
                  onClick={() => applyMode(opt.value)}
                  className={cn(
                    "relative flex h-7 items-center justify-center rounded px-1 text-[11px] font-medium transition-colors",
                    active
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                    disabled && "cursor-not-allowed opacity-50",
                  )}
                >
                  {active && (
                    <motion.span
                      layoutId="doc-extract-mode-pill"
                      className="absolute inset-0 rounded bg-background shadow-border"
                      transition={
                        reducedMotion
                          ? { duration: 0 }
                          : {
                              type: "spring",
                              stiffness: 500,
                              damping: 35,
                              mass: 0.5,
                            }
                      }
                    />
                  )}
                  <span className="relative z-10">{opt.label}</span>
                </button>
              );
            })}
          </div>
          <p className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground">
            {modeHelp}
          </p>
        </div>

        {/* Advanced disclosure */}
        {docExtract.enabled && (
          <div className="flex flex-col">
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="flex items-center gap-1 self-start rounded px-1 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground"
              aria-expanded={showAdvanced}
            >
              <motion.span
                animate={{ rotate: showAdvanced ? 180 : 0 }}
                transition={{ duration: 0.15 }}
                className="inline-flex"
              >
                <HugeiconsIcon icon={ArrowDown01Icon} className="size-3" />
              </motion.span>
              Advanced
            </button>
            <AnimatePresence initial={false}>
              {showAdvanced && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2, ease: "easeInOut" }}
                  className="overflow-hidden"
                >
                  <div className="flex flex-col gap-4 pt-2">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-xs font-medium">
                          Caption images
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                          {hasVlm
                            ? "Describe attached figures with the vision model."
                            : ocrSelected
                              ? `Describe attached figures with ${ocrTarget?.label} during extraction.`
                              : defaultOcrSelected
                                ? "Default will enable this when a vision model is loaded."
                                : "Load a vision model or pick an OCR model to enable."}
                        </div>
                      </div>
                      <Switch
                        aria-label="Caption images"
                        checked={docExtract.describeImages && canCaption}
                        onCheckedChange={(v) =>
                          setDocExtract({ describeImages: !!v })
                        }
                        disabled={!canCaption}
                      />
                    </div>

                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-xs font-medium">
                          Render pages as images
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                          OCR scanned PDFs. Leave off for born-digital docs.
                        </div>
                      </div>
                      <Switch
                        aria-label="Render pages as images"
                        checked={docExtract.useVlmOcr}
                        onCheckedChange={(v) =>
                          setDocExtract({ useVlmOcr: !!v })
                        }
                        disabled={!extractorReady || trcMissing}
                      />
                    </div>

                    <DocumentNumberSliderRow
                      label="Figure/page citations"
                      tooltip="How many figure and page references to include in the extracted text, e.g. [Figure 3] or [Page 7]. Set to 0 to disable citations and image inputs."
                      value={docExtract.maxFigures}
                      sliderMax={DOC_EXTRACT_SLIDER_MAXES.maxFigures}
                      onValueChange={setFigureReferenceLimit}
                      disabled={!extractorReady}
                      valueAriaLabel="Figure and page citation limit"
                    />

                    <div className="space-y-1">
                      <DocumentNumberSliderRow
                        label="Image inputs"
                        tooltip="How many figure or page images to attach or caption for each document. Set to 0 to keep visual references text-only."
                        value={docExtract.maxVisualPayloads}
                        sliderMax={DOC_EXTRACT_SLIDER_MAXES.maxVisualPayloads}
                        onValueChange={setVisualPayloadLimit}
                        disabled={!extractorReady}
                        valueAriaLabel="Image input limit"
                      />
                      {!visionReadyForExtraction && (
                        <p className="text-[11px] leading-relaxed text-muted-foreground">
                          Load a vision model or pick an OCR model to attach
                          images.
                        </p>
                      )}
                    </div>

                    <DocumentNumberSliderRow
                      label="Token budget"
                      tooltip="Cap on extracted text tokens sent to the model per document. Lower values trim long PDFs; raise for more context at higher cost."
                      value={docExtract.tokenBudget}
                      sliderMax={DOC_EXTRACT_SLIDER_MAXES.tokenBudget}
                      step={500}
                      onValueChange={setTokenBudget}
                      disabled={!extractorReady}
                      valueAriaLabel="Document extraction token budget"
                    />

                    <DocumentNumberSliderRow
                      label="Parallel extractions"
                      tooltip="Maximum number of documents extracted in parallel. Extra files queue client-side. Must be ≤ the backend's UNSLOTH_STUDIO_EXTRACT_CONCURRENCY (default 2) to avoid 503 busy responses."
                      value={docExtract.extractConcurrency}
                      sliderMax={DOC_EXTRACT_SLIDER_MAXES.extractConcurrency}
                      sliderMin={1}
                      step={1}
                      onValueChange={setExtractConcurrency}
                      disabled={!extractorReady}
                      valueAriaLabel="Parallel document extractions limit"
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </CollapsibleSection>
  );
}

function ChatTemplateFields() {
  const defaultTemplate = useChatRuntimeStore((s) => s.defaultChatTemplate);
  const override = useChatRuntimeStore((s) => s.chatTemplateOverride);
  const setOverride = useChatRuntimeStore((s) => s.setChatTemplateOverride);
  const [editorOpen, setEditorOpen] = useState(false);
  const [draft, setDraft] = useState("");

  if (!defaultTemplate) return null;

  const displayValue = override ?? defaultTemplate;
  const isModified = override !== null;
  const draftDirty = draft !== displayValue;

  const openEditor = () => {
    setDraft(displayValue);
    setEditorOpen(true);
  };
  const saveEditor = () => {
    setOverride(
      draft.trim().length === 0 || draft === defaultTemplate ? null : draft,
    );
    setEditorOpen(false);
  };

  return (
    <>
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] font-medium tracking-nav text-nav-fg">
            Chat Template
          </span>
          {isModified && (
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={() => setOverride(null)}
                  className="nav-icon-btn text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
                  aria-label="Revert chat template"
                >
                  <HugeiconsIcon
                    icon={ArrowTurnBackwardIcon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                </button>
              </TooltipPrimitive.Trigger>
              <TooltipContent
                side="top"
                sideOffset={6}
                className="tooltip-compact"
              >
                Revert changes
              </TooltipContent>
            </Tooltip>
          )}
        </div>
        <button
          type="button"
          onClick={openEditor}
          aria-label="Edit chat template"
          className="panel-text-surface mt-1 flex w-full h-20 overflow-hidden cursor-pointer items-start px-3.5 py-2.5 text-left text-[13px] font-medium leading-relaxed text-nav-fg corner-squircle focus-visible:outline-none focus-visible:border-ring focus-visible:ring-[1px] focus-visible:ring-ring/40"
        >
          <span className="block line-clamp-3 whitespace-pre-wrap break-words">
            {displayValue}
          </span>
        </button>
      </div>
      <Dialog open={editorOpen} onOpenChange={setEditorOpen}>
        <DialogContent
          className="corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-3xl"
          overlayClassName="bg-background/35 supports-backdrop-filter:backdrop-blur-[1px]"
        >
          <DialogHeader>
            <DialogTitle>Edit Chat Template</DialogTitle>
            <DialogDescription>
              Override the model's chat template. The change applies on the
              next model reload.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <div className="space-y-0.5 px-0.5">
              <div className="text-[11px] font-medium">Template editor</div>
              <p className="text-[11px] text-muted-foreground">
                Jinja syntax. Save matching the default clears the override.
              </p>
            </div>
            <Textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              fieldSizing="fixed"
              className="min-h-[24rem] max-h-[50vh] overflow-y-auto font-mono text-xs leading-5 corner-squircle focus-visible:border-input focus-visible:ring-0"
              rows={14}
              spellCheck={false}
            />
          </div>
          <DialogFooter className="flex-wrap gap-2 sm:justify-between">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setEditorOpen(false)}
            >
              Cancel
            </Button>
            <Button type="button" onClick={saveEditor} disabled={!draftDirty}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
