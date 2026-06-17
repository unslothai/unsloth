// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGpuDevices } from "@/hooks/use-gpu-info";
import { useIsMobile } from "@/hooks/use-mobile";
import { useLlamaUpdateCheck } from "@/hooks/use-llama-update-check";
import { cn } from "@/lib/utils";
import {
  ArrowTurnBackwardIcon,
  Edit03Icon,
  InformationCircleIcon,
  LayoutAlignRightIcon,
} from "@hugeicons/core-free-icons";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown, ExternalLink } from "lucide-react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { Fragment, type ReactNode } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { OpenAICodeExecSection } from "./components/openai-code-exec-section";
import {
  type ExternalProviderConfig,
  getExternalProviderApiKey,
  parseExternalModelId,
  supportsProviderPromptCaching,
  supportsProviderPromptCacheTtl,
} from "./external-providers";
import {
  BUILTIN_PRESETS,
  BUILTIN_PRESET_NAMES,
  applyPresetParams,
  getBuiltinVariantName,
  getOrderedPresets,
  getPresetSaveState,
  getPresetSource,
  isSamePresetConfig,
  toPresetParams,
} from "./presets/preset-policy";
import {
  type ProviderCapabilities,
  getExternalMaxOutputTokens,
  getExternalMinOutputTokens,
  providerSupportsBuiltinCodeExecution,
  providerSupportsFastMode,
} from "./provider-capabilities";
import {
  GPU_LAYERS_ALL,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import { RetrievalSettingsSection } from "@/features/rag/components/retrieval-settings-section";
import type { InferenceParams } from "./types/runtime";

export { defaultInferenceParams, type Preset } from "./presets/preset-policy";
export type { InferenceParams } from "./types/runtime";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function InfoHint({ children }: { children: ReactNode }) {
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
 * Editable numeric value display, shared by every slider value and the Context
 * Length input. An <input> that looks like text (shows `displayValue ?? value`,
 * so "Off"/"Max" labels render) until focus, when it swaps to the raw number,
 * selects it, and accepts free text. Commits on blur/Enter, reverts on Escape.
 * Clamping happens on commit so typing intermediate values isn't fought.
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

  const displayed = focused ? draft : (displayValue ?? String(value));

  return (
    <input
      type="text"
      inputMode="decimal"
      size={sizeAttr}
      /* Fixed 4ch pill; grows only when a longer value would clip. */
      style={{ width: `calc(${Math.max(displayed.length, 4)}ch + 18px)` }}
      value={displayed}
      aria-label={ariaLabel}
      onFocus={(e) => {
        cancelBlurCommitRef.current = false;
        setDraft(String(value));
        setFocused(true);
        // Defer select() so it runs after the value swap above.
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
          size={valueSize ?? 4}
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
  labelHref,
  headerAction,
  onLabelClick,
  children,
  defaultOpen = false,
  first = false,
}: {
  label: string;
  /**
   * When set, the label becomes an external link (e.g. the feature's GitHub PR)
   * instead of part of the toggle. The chevron still toggles, so link and button
   * are siblings rather than an <a> nested in a <button> (invalid HTML).
   */
  labelHref?: string;
  /**
   * Optional control rendered before the chevron (e.g. an edit icon). The
   * label and chevron become sibling toggles so the action is not a button
   * nested in a button.
   */
  headerAction?: ReactNode;
  /** When set, clicking the label runs this instead of toggling collapse. */
  onLabelClick?: () => void;
  children?: ReactNode;
  defaultOpen?: boolean;
  first?: boolean;
}) {
  const [open, setOpen] = useState(() => {
    const saved = loadCollapsibleState();
    return Object.hasOwn(saved, label) ? saved[label] : defaultOpen;
  });

  const toggle = () => {
    const next = !open;
    setOpen(next);
    saveCollapsibleOpen(label, next);
  };

  const headerClasses = cn(
    "flex w-full items-center justify-between text-[12px] font-medium normal-case tracking-[0.04em] text-nav-fg-muted transition-colors focus-visible:outline-none focus-visible:ring-0",
    first ? "pt-4 pb-5" : "py-5",
  );

  return (
    <div
      className={cn(
        !first &&
          "border-t border-black/[0.13] dark:border-white/[0.09]",
      )}
    >
      {labelHref ? (
        <div className={headerClasses}>
          <a
            href={labelHref}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex cursor-pointer items-center gap-1 leading-none transition-colors hover:text-nav-fg"
          >
            <span>{label}</span>
            <ExternalLink className="size-3" />
          </a>
          <button
            type="button"
            onClick={toggle}
            aria-label={open ? `Collapse ${label}` : `Expand ${label}`}
            className="flex shrink-0 cursor-pointer items-center leading-none transition-colors hover:text-nav-fg"
          >
            <ChevronDown
              className={cn("size-3.5", open ? "rotate-0" : "-rotate-90")}
            />
          </button>
        </div>
      ) : headerAction ? (
        <div className={headerClasses}>
          <button
            type="button"
            onClick={onLabelClick ?? toggle}
            className="flex min-w-0 flex-1 cursor-pointer items-center text-left leading-none transition-colors hover:text-nav-fg"
          >
            <span className="leading-none">{label}</span>
          </button>
          <span className="flex shrink-0 items-center gap-1">
            {headerAction}
            <button
              type="button"
              onClick={toggle}
              aria-label={open ? `Collapse ${label}` : `Expand ${label}`}
              className="flex shrink-0 cursor-pointer items-center leading-none transition-colors hover:text-nav-fg"
            >
              <ChevronDown
                className={cn("size-3.5", open ? "rotate-0" : "-rotate-90")}
              />
            </button>
          </span>
        </div>
      ) : (
        <button
          type="button"
          onClick={toggle}
          className={cn("cursor-pointer hover:text-nav-fg", headerClasses)}
        >
          <span className="leading-none">{label}</span>
          <span className="flex shrink-0 items-center leading-none">
            <ChevronDown
              className={cn("size-3.5", open ? "rotate-0" : "-rotate-90")}
            />
          </span>
        </button>
      )}
      {open && <div className="pb-7">{children}</div>}
    </div>
  );
}

interface ChatSettingsPanelProps {
  open: boolean;
  onOpenChange?: (open: boolean) => void;
  params: InferenceParams;
  onParamsChange: (params: InferenceParams) => void;
  isExternalModel?: boolean;
  /**
   * Sampling-param capabilities for the active external provider, or `null` for
   * local models (every knob rendered). Drives per-param sampling visibility.
   */
  providerCapabilities?: ProviderCapabilities | null;
  activeExternalProvider?: ExternalProviderConfig | null;
  onExternalProviderChange?: (provider: ExternalProviderConfig) => void;
  /**
   * Backend provider type for the active external model (e.g. "kimi",
   * "anthropic", "openai"), or `null` for local models. Drives the per-provider
   * Max Tokens floor in the slider.
   */
  externalProviderType?: string | null;
  onReloadModel?: () => void;
}

export function ChatSettingsPanel({
  open,
  onOpenChange,
  params,
  onParamsChange,
  isExternalModel = false,
  providerCapabilities = null,
  activeExternalProvider = null,
  onExternalProviderChange,
  externalProviderType = null,
  onReloadModel,
}: ChatSettingsPanelProps) {
  // Local models show every knob; providerCapabilities is only consulted when
  // isExternalModel. Unknown providers fall back to the OpenAI-compat shape via
  // getProviderCapabilities, so these flags never undercount support.
  const showTemperature =
    !isExternalModel || Boolean(providerCapabilities?.temperature);
  const showTopP = !isExternalModel || Boolean(providerCapabilities?.topP);
  const showTopK = !isExternalModel || Boolean(providerCapabilities?.topK);
  const showMinP = !isExternalModel || Boolean(providerCapabilities?.minP);
  const showRepetitionPenalty =
    !isExternalModel || Boolean(providerCapabilities?.repetitionPenalty);
  const showPresencePenalty =
    !isExternalModel || Boolean(providerCapabilities?.presencePenalty);
  const isMobile = useIsMobile();
  const isGguf = useChatRuntimeStore((s) => s.activeGgufVariant) != null;
  const hasModelContent =
    !isExternalModel && (isGguf || Boolean(params.checkpoint));
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const setSpeculativeType = useChatRuntimeStore((s) => s.setSpeculativeType);
  const loadedSpeculativeType = useChatRuntimeStore(
    (s) => s.loadedSpeculativeType,
  );
  const specFallbackReason = useChatRuntimeStore((s) => s.specFallbackReason);
  // "binary_no_mtp" / "binary_outdated" mean a newer prebuilt would re-enable
  // MTP; "runtime_error" means the current build cannot run it (no update push).
  const mtpUpdatable =
    specFallbackReason === "binary_no_mtp" ||
    specFallbackReason === "binary_outdated";
  const {
    status: llamaUpdateStatus,
    applying: llamaUpdating,
    apply: applyLlamaUpdate,
  } = useLlamaUpdateCheck({ enabled: mtpUpdatable });
  const handleMtpUpdate = useCallback(async () => {
    const result = await applyLlamaUpdate();
    if (result.ok) {
      toast.success(
        `llama.cpp updated to ${result.tag ?? "the latest build"}. Reload your model to enable MTP.`,
      );
    } else {
      toast.error(`llama.cpp update failed: ${result.error ?? "unknown error"}`);
    }
  }, [applyLlamaUpdate]);
  const specDraftNMax = useChatRuntimeStore((s) => s.specDraftNMax);
  const setSpecDraftNMax = useChatRuntimeStore((s) => s.setSpecDraftNMax);
  const loadedSpecDraftNMax = useChatRuntimeStore(
    (s) => s.loadedSpecDraftNMax,
  );
  const modelRequiresTrustRemoteCode = useChatRuntimeStore(
    (s) => s.modelRequiresTrustRemoteCode,
  );
  const currentCheckpoint = params.checkpoint;
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
  const tensorParallel = useChatRuntimeStore((s) => s.tensorParallel);
  const setTensorParallel = useChatRuntimeStore((s) => s.setTensorParallel);
  const loadedTensorParallel = useChatRuntimeStore(
    (s) => s.loadedTensorParallel,
  );
  const gpuMemoryMode = useChatRuntimeStore((s) => s.gpuMemoryMode);
  const setGpuMemoryMode = useChatRuntimeStore((s) => s.setGpuMemoryMode);
  const loadedGpuMemoryMode = useChatRuntimeStore((s) => s.loadedGpuMemoryMode);
  const gpuLayers = useChatRuntimeStore((s) => s.gpuLayers);
  const setGpuLayers = useChatRuntimeStore((s) => s.setGpuLayers);
  const loadedGpuLayers = useChatRuntimeStore((s) => s.loadedGpuLayers);
  const cpuMoe = useChatRuntimeStore((s) => s.cpuMoe);
  const setCpuMoe = useChatRuntimeStore((s) => s.setCpuMoe);
  const loadedCpuMoe = useChatRuntimeStore((s) => s.loadedCpuMoe);
  const ggufLayerCount = useChatRuntimeStore((s) => s.ggufLayerCount);
  const selectedGpuIds = useChatRuntimeStore((s) => s.selectedGpuIds);
  const setSelectedGpuIds = useChatRuntimeStore((s) => s.setSelectedGpuIds);
  const loadedGpuIds = useChatRuntimeStore((s) => s.loadedGpuIds);
  const gpuDevices = useGpuDevices();
  const customContextLength = useChatRuntimeStore((s) => s.customContextLength);
  const setCustomContextLength = useChatRuntimeStore(
    (s) => s.setCustomContextLength,
  );
  const setActivePresetSource = useChatRuntimeStore(
    (s) => s.setActivePresetSource,
  );
  const activePresetSource = useChatRuntimeStore((s) => s.activePresetSource);
  const customPresets = useChatRuntimeStore((s) => s.customPresets);
  const setCustomPresets = useChatRuntimeStore((s) => s.setCustomPresets);
  const activePreset = useChatRuntimeStore((s) => s.activePreset);
  const setActivePreset = useChatRuntimeStore((s) => s.setActivePreset);
  const settingsHydrated = useChatRuntimeStore((s) => s.settingsHydrated);

  const ctxDisplayValue = customContextLength ?? ggufContextLength ?? "";
  const ctxMaxValue = ggufNativeContextLength ?? ggufContextLength ?? null;
  const kvDirty = kvCacheDtype !== loadedKvCacheDtype;
  const ctxDirty = customContextLength !== null;
  const specDirty = speculativeType !== loadedSpeculativeType;
  const specDraftDirty = specDraftNMax !== loadedSpecDraftNMax;
  const tpDirty = tensorParallel !== (loadedTensorParallel ?? false);
  const gpuDirty = gpuMemoryMode !== (loadedGpuMemoryMode ?? "auto");
  const isAutoFit = gpuMemoryMode === "fit";
  const isManual = gpuMemoryMode === "manual";
  // fit and manual both bypass Unsloth's tensor-parallel planning, so the
  // Tensor Parallelism toggle is disabled outside Automatic.
  const tpDisabled = gpuMemoryMode !== "auto";
  // Manual gpu-layers ceiling = model layer count (else a safe fallback).
  const gpuLayersMax = ggufLayerCount ?? 256;
  const manualDirty =
    isManual &&
    (gpuLayers !== loadedGpuLayers || cpuMoe !== (loadedCpuMoe ?? false));
  // GPU picker: only meaningful on multi-GPU, and only when the reported
  // indices are physical (relative ordinals from a parent CUDA_VISIBLE_DEVICES
  // mask can't be mapped back to pin a device). null = use all (auto).
  const showGpuPicker =
    isGguf &&
    gpuDevices.length > 1 &&
    gpuDevices.every((d) => d.physicalIndex);
  const isGpuChecked = (index: number) =>
    selectedGpuIds === null || selectedGpuIds.includes(index);
  const toggleGpu = (index: number) => {
    const all = gpuDevices.map((d) => d.index);
    const current = selectedGpuIds ?? all;
    const next = current.includes(index)
      ? current.filter((i) => i !== index)
      : [...current, index].sort((a, b) => a - b);
    if (next.length === 0) return; // keep at least one GPU selected
    setSelectedGpuIds(next.length === all.length ? null : next);
  };
  const gpuIdsKey = (ids: number[] | null) => (ids === null ? "auto" : ids.join(","));
  const gpuIdsDirty = gpuIdsKey(selectedGpuIds) !== gpuIdsKey(loadedGpuIds);
  // Auto-fit context: null / <= 0 means "Auto" (let --fit size it); a positive
  // value pins it (--fit optimizes gpu-layers around it). After a fit load,
  // surface the length --fit actually chose.
  const fitCtxAuto = isAutoFit && (customContextLength ?? 0) <= 0;
  const fitResolvedCtx =
    fitCtxAuto && loadedGpuMemoryMode === "fit" ? ggufContextLength : null;
  const modelSettingsDirty =
    kvDirty ||
    ctxDirty ||
    specDirty ||
    specDraftDirty ||
    tpDirty ||
    gpuDirty ||
    manualDirty ||
    gpuIdsDirty;
  const loadedChatTemplateOverride = useChatRuntimeStore(
    (s) => s.loadedChatTemplateOverride,
  );
  const setChatTemplateOverride = useChatRuntimeStore(
    (s) => s.setChatTemplateOverride,
  );
  const [presetNameInput, setPresetNameInput] = useState(activePreset);
  const [systemPromptEditorOpen, setSystemPromptEditorOpen] = useState(false);
  const [systemPromptDraft, setSystemPromptDraft] = useState("");
  // When the prompt overflows the inline box, clicking opens the popup editor.
  const systemPromptBoxRef = useRef<HTMLTextAreaElement>(null);
  const [systemPromptOverflows, setSystemPromptOverflows] = useState(false);
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
  const showPromptCacheTtlControl = Boolean(
    activeExternalProvider &&
      supportsProviderPromptCacheTtl(activeExternalProvider.providerType),
  );
  const showPromptCachingControl =
    activeExternalProvider != null &&
    supportsProviderPromptCaching(activeExternalProvider.providerType);
  const promptCachingEnabled =
    activeExternalProvider?.enablePromptCaching !== false;
  const externalSelection = currentCheckpoint
    ? parseExternalModelId(currentCheckpoint)
    : null;
  const showOpenAICodeExecSection =
    activeExternalProvider != null &&
    providerSupportsBuiltinCodeExecution(
      activeExternalProvider.providerType,
      externalSelection?.modelId,
      activeExternalProvider.baseUrl,
    ) &&
    activeExternalProvider.providerType === "openai";
  const showFastModeControl =
    activeExternalProvider != null &&
    providerSupportsFastMode(
      activeExternalProvider.providerType,
      externalSelection?.modelId,
    );
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const openAiApiKeyForSection = activeExternalProvider
    ? getExternalProviderApiKey(activeExternalProvider.id) || null
    : null;

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
    if (!settingsHydrated) {
      return;
    }
    const p = presets.find((pr) => pr.name === name);
    if (p) {
      onParamsChange({
        ...applyPresetParams(params, p.params),
      });
      setActivePreset(name);
      setActivePresetSource(getPresetSource(name));
    }
  }

  function savePresetWithName(rawName: string) {
    if (!settingsHydrated) {
      return;
    }
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
    const next = customPresets.filter((p) => p.name !== saveName);
    const merged = [
      ...next,
      { name: saveName, params: toPresetParams(params) },
    ];
    setCustomPresets(merged);
    setActivePreset(saveName);
    setActivePresetSource("custom");
    setPresetNameInput(saveName);
  }

  function deletePreset(name: string) {
    if (!settingsHydrated) {
      return;
    }
    const hasCustomPreset = customPresets.some(
      (preset) => preset.name === name,
    );
    if (!hasCustomPreset) {
      return;
    }
    const fallbackPreset =
      BUILTIN_PRESETS.find((preset) => preset.name === "Default") ??
      null;
    const next = customPresets.filter((preset) => preset.name !== name);
    setCustomPresets(next);
    if (activePreset === name) {
      if (fallbackPreset) {
        onParamsChange({
          ...applyPresetParams(params, fallbackPreset.params),
        });
        setActivePreset(fallbackPreset.name);
        setActivePresetSource("builtin-default");
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
    if (!settingsHydrated) {
      return;
    }
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
  }, [
    activePreset,
    activePresetSource,
    presets,
    setActivePreset,
    setActivePresetSource,
    settingsHydrated,
  ]);

  useEffect(() => {
    setPresetNameInput(activePreset);
  }, [activePreset]);

  useEffect(() => {
    if (!open) {
      setSystemPromptEditorOpen(false);
    }
  }, [open]);

  useEffect(() => {
    const el = systemPromptBoxRef.current;
    setSystemPromptOverflows(
      params.systemPrompt.length > 0 &&
        el != null &&
        el.clientHeight > 0 &&
        el.scrollHeight > el.clientHeight + 1,
    );
  }, [params.systemPrompt, open]);

  const settingsScrollRef = useRef<HTMLDivElement>(null);

  const settingsContent = (
    <>
      <div className="flex h-full min-h-0 flex-col">
      {/* Header is outside the scroll area so the scrollbar never shifts the close button. */}
      <div className="flex h-[48px] shrink-0 items-start gap-2 bg-panel-surface pl-[18px] pr-[16px] pt-[11px]">
        {isMobile ? (
          <span className="flex h-[34px] flex-1 items-center text-[16px] font-semibold tracking-[0em] dark:tracking-[0.015em] text-nav-fg">
            Run settings
          </span>
        ) : (
          <>
            <span className="flex h-[34px] flex-1 items-center text-[16px] font-semibold tracking-[0em] dark:tracking-[0.015em] text-nav-fg">
              Run settings
            </span>
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={() => onOpenChange?.(false)}
                  className="flex h-[34px] w-[34px] cursor-pointer items-center justify-center rounded-full text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Close run settings"
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
                Close run settings
              </TooltipContent>
            </Tooltip>
          </>
        )}
      </div>

      <div
        ref={settingsScrollRef}
        className="run-settings-scroll relative min-h-0 flex-1 overflow-y-auto"
      >
      <div className="px-[18px] pt-3">
        {hasModelContent && (
        <CollapsibleSection label="Model" defaultOpen={true} first>
          <div className="flex flex-col gap-4 pt-1">
            {isGguf && (
              <>
                {isAutoFit ? (
                  <div className="space-y-3.5">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex min-w-0 items-center gap-1.5">
                        <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                          Context Length
                        </span>
                        <InfoHint>
                          Auto: llama.cpp's --fit sizes the context to fit VRAM.
                          Set a length to pin it instead -- --fit then optimizes
                          GPU layer offload around it. The length --fit chose
                          shows here after loading.
                        </InfoHint>
                      </div>
                      <NumericValueInput
                        value={fitCtxAuto ? 0 : (customContextLength ?? 0)}
                        displayValue={fitCtxAuto ? "Auto" : undefined}
                        min={0}
                        max={ctxMaxValue ?? undefined}
                        step={1}
                        onChange={(v) => {
                          setCustomContextLength(v > 0 ? v : null);
                        }}
                        ariaLabel="Context Length"
                        size={8}
                      />
                    </div>
                    <Slider
                      min={0}
                      max={ctxMaxValue ?? 4096}
                      step={1024}
                      value={[
                        fitCtxAuto
                          ? 0
                          : Math.min(
                              customContextLength ?? 0,
                              ctxMaxValue ?? 4096,
                            ),
                      ]}
                      onValueChange={([v]) => {
                        // Far-left snaps to Auto; otherwise to the nearest 1024.
                        if (v < 512) {
                          setCustomContextLength(null);
                        } else {
                          setCustomContextLength(Math.round(v / 1024) * 1024);
                        }
                      }}
                      className="panel-slider"
                    />
                    {fitResolvedCtx != null && (
                      <p className="text-[11px] text-nav-fg/40">
                        llama.cpp loaded {fitResolvedCtx.toLocaleString()} tokens.
                      </p>
                    )}
                  </div>
                ) : (
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
                )}
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
                        icon={ChevronDownStandardIcon}
                        iconClassName="size-3.5"
                        className="grid h-7 w-[64px] min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1] pl-3 pr-2 py-0 text-[13px]! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0"
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
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                      Speculative Decoding
                    </span>
                    <InfoHint>
                      Faster generation with 0% accuracy hit. Auto picks
                      MTP / ngram-mod based on the model and platform.
                      Pick MTP, Ngram, or MTP+Ngram to force a specific
                      strategy on both GPU and CPU.
                    </InfoHint>
                  </div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    <Select
                      value={speculativeType ?? "auto"}
                      onValueChange={(v) => {
                        setSpeculativeType(v);
                        if (v !== "mtp" && v !== "mtp+ngram") {
                          setSpecDraftNMax(null);
                        }
                      }}
                    >
                      <SelectTrigger
                        animateRadius={false}
                        icon={ChevronDownStandardIcon}
                        iconClassName="size-3.5"
                        className="grid h-7 w-[124px] min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1] pl-3 pr-2 py-0 text-[13px]! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0"
                        data-test-id="speculative-type-select"
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
                        <SelectItem value="auto">Auto</SelectItem>
                        <SelectItem value="mtp">MTP</SelectItem>
                        <SelectItem value="ngram">Ngram</SelectItem>
                        <SelectItem value="mtp+ngram">MTP+Ngram</SelectItem>
                        <SelectItem value="off">Off</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                {specFallbackReason &&
                  (speculativeType === "auto" ||
                    speculativeType === "mtp" ||
                    speculativeType === "mtp+ngram") && (
                    <div className="rounded-lg bg-amber-500/[0.08] px-3 py-2 text-[12px] leading-[1.4] text-nav-fg/80">
                      <p>
                        {specFallbackReason === "runtime_error"
                          ? "MTP could not start for this model on the installed llama.cpp build, so it is running without speculative decoding."
                          : "MTP is not available in the installed llama.cpp build, so this model is running without it." +
                            (llamaUpdateStatus?.update_available
                              ? " Update llama.cpp to enable it."
                              : "")}
                      </p>
                      {mtpUpdatable && llamaUpdateStatus?.update_available && (
                        <Button
                          size="sm"
                          className="corner-squircle mt-2 h-7 text-[12px]"
                          onClick={handleMtpUpdate}
                          disabled={llamaUpdating}
                          data-test-id="mtp-update-button"
                        >
                          {llamaUpdating ? "Updating..." : "Update llama.cpp"}
                        </Button>
                      )}
                    </div>
                  )}
                {(speculativeType === "mtp" ||
                  speculativeType === "mtp+ngram") && (
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                        Draft Tokens
                      </span>
                      <InfoHint>
                        Max MTP draft tokens per step
                        (--spec-draft-n-max). Lower = less wasted
                        draft decode; higher = bigger speedup when
                        acceptance stays high. Default: 2 on GPU,
                        3 on CPU/Mac.
                      </InfoHint>
                    </div>
                    <input
                      type="number"
                      min={1}
                      max={16}
                      step={1}
                      value={specDraftNMax ?? ""}
                      placeholder="auto"
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === "") {
                          setSpecDraftNMax(null);
                          return;
                        }
                        const parsed = Number.parseInt(raw, 10);
                        if (Number.isFinite(parsed)) {
                          const clamped = Math.max(1, Math.min(16, parsed));
                          setSpecDraftNMax(clamped);
                        }
                      }}
                      data-test-id="spec-draft-n-max-input"
                      aria-label="Speculative decoding draft tokens"
                      className="h-7 w-[76px] rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1] pl-3 pr-2 py-0 text-[13px] font-medium text-nav-fg outline-none focus-visible:ring-0"
                    />
                  </div>
                )}
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                      Tensor Parallelism
                    </span>
                    <InfoHint>
                      No effect on a single GPU. On multi-GPU setups, improves
                      tokens/sec during generation when using dense models. MoE
                      models don't benefit and can be much slower.
                    </InfoHint>
                  </div>
                  <Switch
                    className="panel-switch shrink-0"
                    checked={tensorParallel}
                    onCheckedChange={setTensorParallel}
                    disabled={tpDisabled}
                    data-test-id="tensor-parallel-switch"
                  />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                      GPU Memory
                    </span>
                    <InfoHint>
                      <div className="flex flex-col gap-1.5">
                        <div>
                          <span className="font-medium">Default:</span> Unsloth
                          fits the model and context to your GPUs.
                        </div>
                        <div>
                          <span className="font-medium">llama.cpp --fit:</span>{" "}
                          llama.cpp sizes context and offloads overflow
                          (including MoE experts) to RAM.
                        </div>
                        <div>
                          <span className="font-medium">Manual:</span> pin GPU
                          layers and MoE offload yourself.
                        </div>
                        <div>fit and Manual turn off Tensor Parallelism.</div>
                      </div>
                    </InfoHint>
                  </div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    <Select
                      value={gpuMemoryMode}
                      onValueChange={(v) => {
                        const mode = v as "auto" | "fit" | "manual";
                        setGpuMemoryMode(mode);
                        // fit/manual bypass Unsloth's tensor-parallel planning.
                        if (mode !== "auto") setTensorParallel(false);
                      }}
                    >
                      <SelectTrigger
                        animateRadius={false}
                        icon={ChevronDownStandardIcon}
                        iconClassName="size-3.5"
                        className="grid h-7 w-[136px] min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1] pl-3 pr-2 py-0 text-[13px]! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0"
                        data-test-id="gpu-memory-mode-select"
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
                        <SelectItem value="auto">Default</SelectItem>
                        <SelectItem value="fit">llama.cpp --fit</SelectItem>
                        <SelectItem value="manual">Manual</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                {isManual && (
                  <>
                    <ParamSlider
                      label="GPU Layers"
                      value={Math.min(gpuLayers, gpuLayersMax)}
                      min={0}
                      max={gpuLayersMax}
                      step={1}
                      onChange={setGpuLayers}
                      valueSize={6}
                      info={
                        <>
                          Layers to keep on the GPU (--gpu-layers); the rest
                          run on CPU. At the maximum, the whole model is on the
                          GPU.
                        </>
                      }
                    />
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex min-w-0 items-center gap-1.5">
                        <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                          Keep MoE on CPU
                        </span>
                        <InfoHint>
                          Move MoE expert tensors to the CPU (--cpu-moe). Saves
                          VRAM on MoE models at some speed cost; no effect on
                          dense models.
                        </InfoHint>
                      </div>
                      <Switch
                        className="panel-switch shrink-0"
                        checked={cpuMoe}
                        onCheckedChange={setCpuMoe}
                        data-test-id="cpu-moe-switch"
                      />
                    </div>
                  </>
                )}
                {showGpuPicker && (
                  <div className="space-y-2">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                        GPUs
                      </span>
                      <InfoHint>
                        Which GPUs this model may use. Unchecked GPUs are hidden
                        from llama.cpp (CUDA_VISIBLE_DEVICES, or
                        HIP_VISIBLE_DEVICES on ROCm). Leave all checked to use
                        every GPU.
                      </InfoHint>
                    </div>
                    <div className="flex flex-col gap-2">
                      {gpuDevices.map((d) => (
                        <div
                          key={d.index}
                          className="flex items-center justify-between gap-3"
                        >
                          <span className="min-w-0 truncate text-[12px] text-nav-fg/80">
                            GPU {d.index}: {d.name}
                            {d.memoryTotalGb
                              ? ` · ${Math.round(d.memoryTotalGb)} GB`
                              : ""}
                          </span>
                          <Switch
                            className="panel-switch shrink-0"
                            checked={isGpuChecked(d.index)}
                            onCheckedChange={() => toggleGpu(d.index)}
                            data-test-id={`gpu-pick-${d.index}`}
                          />
                        </div>
                      ))}
                    </div>
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
            {/* Apply/Reset belongs to the model-reload settings above (context
                length, KV cache, speculative decoding). Render it here, before
                the Chat Template row, so it never reads as attached to Chat
                Template (which is edited via its own dialog). */}
            {modelSettingsDirty && (
              <div className="flex flex-wrap gap-1.5">
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
                    setSpecDraftNMax(loadedSpecDraftNMax);
                    setTensorParallel(loadedTensorParallel ?? false);
                    setGpuMemoryMode(loadedGpuMemoryMode ?? "auto");
                    setGpuLayers(loadedGpuLayers ?? GPU_LAYERS_ALL);
                    setCpuMoe(loadedCpuMoe ?? false);
                    setSelectedGpuIds(loadedGpuIds);
                    setChatTemplateOverride(loadedChatTemplateOverride);
                  }}
                  className="h-7 px-3 text-[12px] font-medium tracking-nav text-muted-foreground"
                >
                  Reset
                </Button>
              </div>
            )}
            <ChatTemplateFields />
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
                        if (
                          e.key === "Enter" &&
                          settingsHydrated &&
                          presetSaveState.canSubmit
                        ) {
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
                      className="min-h-0 shrink-0 gap-0 self-stretch border-0 py-0 pl-0 !pr-1 has-[>button]:mr-0 !cursor-pointer"
                    >
                      <span
                        className="!h-7 min-h-7 !w-7 min-w-7 shrink-0 self-center inline-flex items-center justify-center rounded-full border-0 px-0 text-[#a0a097] dark:text-nav-fg pointer-events-none"
                        aria-hidden="true"
                      >
                        <HugeiconsIcon
                          icon={ChevronDownStandardIcon}
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
                sideOffset={0}
                className="menu-soft-surface ring-0 border-0 rounded-lg p-1.5"
              >
                {presets.map((p, index) => (
                  <Fragment key={p.name}>
                    <DropdownMenuItem
                      disabled={!settingsHydrated}
                      onSelect={(event) => {
                        if (!settingsHydrated) {
                          event.preventDefault();
                          return;
                        }
                        applyPreset(p.name);
                      }}
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
                disabled={!(settingsHydrated && presetSaveState.canSubmit)}
                variant={presetSaveState.isSaveReady ? "default" : "outline"}
                size="sm"
                className={cn(
                  "h-9 w-full rounded-full text-[13px] font-medium tracking-nav",
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
                disabled={!(settingsHydrated && activeCustomPreset)}
                variant="outline"
                size="sm"
                className="h-9 w-full rounded-full text-[13px] font-medium tracking-nav text-muted-foreground"
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

        {showPromptCachingControl && activeExternalProvider ? (
          <CollapsibleSection label="Provider" defaultOpen={true}>
            <div className="flex items-center justify-between gap-3 pt-1">
              <div className="flex min-w-0 items-center gap-1.5">
                <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                  Prompt caching
                </span>
                <InfoHint>
                  Reuse compatible prompt prefixes for lower latency and cost.
                </InfoHint>
              </div>
              <Switch
                className="panel-switch shrink-0"
                checked={promptCachingEnabled}
                onCheckedChange={(checked) => {
                  onExternalProviderChange?.({
                    ...activeExternalProvider,
                    enablePromptCaching: checked,
                  });
                }}
                aria-label="Enable prompt caching"
              />
            </div>
            {showPromptCacheTtlControl && promptCachingEnabled ? (
              <div className="flex items-center justify-between gap-3 pt-3">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                    Cache TTL
                  </span>
                  <InfoHint>
                    Anthropic exposes a 5 minute and a 1 hour ephemeral
                    cache pool. The 1 hour pool costs 2x base input on
                    write vs 1.25x for 5 minute, but reads stay 0.1x for
                    both, so a single read landing more than 5 minutes
                    after the write pays off the premium.
                  </InfoHint>
                </div>
                <Select
                  value={activeExternalProvider.promptCacheTtl ?? "5m"}
                  onValueChange={(value) => {
                    if (value !== "5m" && value !== "1h") return;
                    onExternalProviderChange?.({
                      ...activeExternalProvider,
                      promptCacheTtl: value,
                    });
                  }}
                >
                  <SelectTrigger
                    className="panel-select-trigger h-8 w-[124px] shrink-0"
                    aria-label="Prompt cache TTL"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="5m">5 minutes</SelectItem>
                    <SelectItem value="1h">1 hour</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            ) : null}
            {showFastModeControl ? (
              <div className="flex items-center justify-between gap-3 pt-3">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                    Fast mode
                  </span>
                  <InfoHint>
                    Beta. Up to 2.5x higher output tokens per second on
                    Claude Opus 4.6 and 4.7 at 6x standard Opus pricing.
                    Switching between fast and standard invalidates the
                    prompt cache and is incompatible with the Priority
                    service tier.
                  </InfoHint>
                </div>
                <Switch
                  className="panel-switch shrink-0"
                  checked={Boolean(params.fastMode)}
                  onCheckedChange={set("fastMode")}
                  aria-label="Fast mode"
                />
              </div>
            ) : null}
          </CollapsibleSection>
        ) : null}

        {showOpenAICodeExecSection && activeExternalProvider ? (
          <CollapsibleSection label="Code Execution" defaultOpen={false}>
            <OpenAICodeExecSection
              provider={activeExternalProvider}
              apiKey={openAiApiKeyForSection}
              activeThreadId={activeThreadId}
              onProviderChange={(p) => onExternalProviderChange?.(p)}
            />
          </CollapsibleSection>
        ) : null}

        <CollapsibleSection
          label="System Prompt"
          defaultOpen={true}
          onLabelClick={openSystemPromptEditor}
          headerAction={
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={openSystemPromptEditor}
                  className="nav-icon-btn text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
                  aria-label="Edit system prompt"
                >
                  <HugeiconsIcon
                    icon={Edit03Icon}
                    strokeWidth={1.75}
                    className="size-3"
                  />
                </button>
              </TooltipPrimitive.Trigger>
              <TooltipContent
                side="top"
                sideOffset={6}
                className="tooltip-compact"
              >
                Edit prompt
              </TooltipContent>
            </Tooltip>
          }
        >
          {/* Rounded wrapper clips overflowing text and the scrollbar. */}
          <div
            className={cn(
              "panel-text-surface -mt-1 h-20 w-full overflow-hidden corner-squircle",
              systemPromptOverflows && "cursor-pointer",
            )}
          >
            <textarea
              ref={systemPromptBoxRef}
              value={params.systemPrompt}
              onChange={(e) => set("systemPrompt")(e.target.value)}
              onMouseDown={(e) => {
                // Overflowing prompt: click opens the popup editor instead.
                // While focused, clicks still move the caret normally.
                if (
                  systemPromptOverflows &&
                  document.activeElement !== e.currentTarget
                ) {
                  e.preventDefault();
                  openSystemPromptEditor();
                }
              }}
              placeholder="Example: You are a helpful assistant..."
              aria-label="System prompt"
              className={cn(
                "block size-full resize-none bg-transparent px-3.5 py-2.5 text-left text-[13px] font-medium leading-relaxed text-nav-fg outline-none placeholder:text-muted-foreground",
                systemPromptOverflows && "cursor-pointer",
              )}
            />
          </div>
        </CollapsibleSection>

        <CollapsibleSection label="Sampling" defaultOpen={true}>
          <div className="flex flex-col gap-5 pt-1">
            {showTemperature ? (
              <ParamSlider
                label="Temperature"
                value={params.temperature}
                min={0}
                max={2}
                step={0.01}
                onChange={set("temperature")}
                info="Controls randomness. Lower values make output focused and deterministic; higher values increase variety and creativity."
              />
            ) : null}
            {showTopP ? (
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
            ) : null}
            {showTopK ? (
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
            ) : null}
            {showMinP ? (
              <ParamSlider
                label="Min P"
                value={params.minP}
                min={0}
                max={1}
                step={0.01}
                onChange={set("minP")}
                info="Drops tokens whose probability is below this fraction of the top token's probability. Filters unlikely candidates."
              />
            ) : null}
            {showRepetitionPenalty ? (
              <ParamSlider
                label="Repetition Penalty"
                value={params.repetitionPenalty}
                min={1}
                max={2}
                step={0.05}
                onChange={set("repetitionPenalty")}
                displayValue={
                  params.repetitionPenalty === 1 ? "Off" : undefined
                }
                info="Down-weights tokens that have already appeared, reducing repetition. 1.0 = off; higher values penalize more strongly."
              />
            ) : null}
            {showPresencePenalty ? (
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
            ) : null}
            {!isExternalModel && !isGguf && (
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
              min={
                isExternalModel
                  ? getExternalMinOutputTokens(externalProviderType)
                  : 64
              }
              max={
                isExternalModel
                  ? getExternalMaxOutputTokens(
                      externalProviderType,
                      externalSelection?.modelId,
                    )
                  : isGguf && ggufContextLength
                    ? ggufContextLength
                    : 32768
              }
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

        {!isExternalModel ? (
          <CollapsibleSection label="Tools">
            <div className="flex flex-col gap-5 pt-1">
              <AutoHealToolCallsToggle />
              <ConfirmToolCallsToggle />
              <BypassPermissionsToggle />
              <MaxToolCallsSlider />
              <ToolCallTimeoutSlider />
            </div>
          </CollapsibleSection>
        ) : null}

        {!isExternalModel ? (
          <CollapsibleSection label="Retrieval">
            <RetrievalSettingsSection />
          </CollapsibleSection>
        ) : null}
      </div>
      </div>
      </div>
      <Dialog
        open={systemPromptEditorOpen}
        onOpenChange={(nextOpen) => {
          setSystemPromptEditorOpen(nextOpen);
        }}
      >
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-3xl">
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
              className="min-h-[24rem] max-h-[50vh] overflow-y-auto border-0 text-sm leading-6 corner-squircle focus-visible:ring-0"
              rows={14}
            />
          </div>
          <DialogFooter className="flex-wrap gap-2 sm:justify-between">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setSystemPromptDraft("")}
              disabled={systemPromptDraft.length === 0}
              className="text-muted-foreground"
            >
              Reset
            </Button>
            <div className="flex flex-wrap gap-2">
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
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );

  if (isMobile) {
    return (
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent side="right" className="w-[18rem] p-0 font-heading">
          <SheetHeader className="sr-only">
            <SheetTitle>Run settings</SheetTitle>
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

function ConfirmToolCallsToggle() {
  const confirmToolCalls = useChatRuntimeStore((s) => s.confirmToolCalls);
  const setConfirmToolCalls = useChatRuntimeStore((s) => s.setConfirmToolCalls);
  const bypassPermissions = useChatRuntimeStore((s) => s.bypassPermissions);

  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex min-w-0 flex-col gap-0.5">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            Confirm tool calls
          </span>
          <InfoHint>
            When on, local Studio tool calls pause for your approval before they
            run. Provider-hosted tools are not gated here.
          </InfoHint>
        </div>
        {bypassPermissions ? (
          <span className="text-[11px] text-muted-foreground">
            Overridden by Bypass Permissions
          </span>
        ) : null}
      </div>
      <Switch
        className="panel-switch"
        checked={confirmToolCalls && !bypassPermissions}
        onCheckedChange={setConfirmToolCalls}
        disabled={bypassPermissions}
      />
    </div>
  );
}

function BypassPermissionsToggle() {
  const bypassPermissions = useChatRuntimeStore((s) => s.bypassPermissions);
  const setBypassPermissions = useChatRuntimeStore(
    (s) => s.setBypassPermissions,
  );
  const [dialogOpen, setDialogOpen] = useState(false);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            Bypass Permissions
          </span>
          <InfoHint>
            Dangerous. Runs every tool call with no confirmation and disables
            the python/terminal sandbox. Environment secrets are stripped, but
            code can still read files and credentials on your machine.
          </InfoHint>
        </div>
        <Switch
          className="panel-switch"
          checked={bypassPermissions}
          onCheckedChange={(next) => {
            if (next) setDialogOpen(true);
            else setBypassPermissions(false);
          }}
        />
      </div>
      {bypassPermissions ? (
        <span className="text-[11px] text-destructive">
          Tool calls run with no confirmation and no sandbox.
        </span>
      ) : null}
      <AlertDialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Enable Bypass Permissions?</AlertDialogTitle>
            <AlertDialogDescription>
              Bypass Permissions is dangerous since the AI model might delete,
              corrupt your machine, and or cause real world damage to you or the
              world - only accept if you are certain
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              className="!bg-destructive !text-destructive-foreground hover:!bg-destructive/90"
              onClick={() => {
                setBypassPermissions(true);
                setDialogOpen(false);
              }}
            >
              I understand
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
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
    const cleared = draft.trim().length === 0 || draft === defaultTemplate;
    setOverride(cleared ? null : draft);
    setEditorOpen(false);
    toast.success(
      cleared
        ? "Chat template reset to default. It applies on the next model reload."
        : "Chat template saved. It applies on the next model reload.",
    );
  };

  return (
    <>
      <div className="-mb-1.5 flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={openEditor}
          className="cursor-pointer text-left text-[13px] font-medium tracking-nav text-nav-fg"
        >
          Chat Template
        </button>
        <div className="flex items-center gap-1">
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
          <Tooltip>
            <TooltipPrimitive.Trigger asChild>
              <button
                type="button"
                onClick={openEditor}
                className="nav-icon-btn text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
                aria-label="Edit chat template"
              >
                <HugeiconsIcon
                  icon={Edit03Icon}
                  strokeWidth={1.75}
                  className="size-3"
                />
              </button>
            </TooltipPrimitive.Trigger>
            <TooltipContent
              side="top"
              sideOffset={6}
              className="tooltip-compact"
            >
              Edit template
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      <Dialog open={editorOpen} onOpenChange={setEditorOpen}>
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-3xl">
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
              className="min-h-[24rem] max-h-[50vh] overflow-y-auto border-0 font-mono text-xs leading-5 corner-squircle focus-visible:ring-0"
              rows={14}
              spellCheck={false}
            />
          </div>
          <DialogFooter className="flex-wrap gap-2 sm:justify-between">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setDraft(defaultTemplate)}
              disabled={draft === defaultTemplate}
              className="text-muted-foreground"
            >
              Reset
            </Button>
            <div className="flex flex-wrap gap-2">
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
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
