// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/components/assistant-ui/model-selector";
import { HubModelPicker } from "@/components/assistant-ui/model-selector/pickers";
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
import { InfoHint } from "@/components/ui/info-hint";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { useIsMobile } from "@/hooks/use-mobile";
import { useLlamaUpdateCheck } from "@/hooks/use-llama-update-check";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowTurnBackwardIcon,
  CodeIcon,
  Delete02Icon,
  Edit03Icon,
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
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { ChevronDown, ExternalLink } from "lucide-react";
import { Fragment, type ReactNode } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { getCachedDocumentSupport } from "./api/chat-api";
import { OpenAICodeExecSection } from "./components/openai-code-exec-section";
import {
  type ExternalProviderConfig,
  getExternalProviderApiKey,
  parseExternalModelId,
  supportsProviderPromptCaching,
  supportsProviderPromptCacheTtl,
} from "./external-providers";
import {
  type DocExtractSettings,
  isPendingGguf,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import type { DocumentSupport } from "./types";
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
import { RetrievalSettingsSection } from "@/features/rag/components/retrieval-settings-section";
import {
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "./types/runtime";
import {
  OCR_MODEL_PRESETS,
  resolveOcrModelTarget,
} from "./utils/ocr-model-presets";
import { ensureOcrModelRemoteCodeApproved } from "./utils/ocr-model-orchestrator";

export { defaultInferenceParams, type Preset } from "./presets/preset-policy";
export type { InferenceParams } from "./types/runtime";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
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
  /** Loads the staged `pendingSelection` (deferred "Load on selection" flow). */
  onLoadPendingModel?: () => void;
  /** Download progress (0–1) for a staged GGUF being fetched, or null when idle. */
  stagedDownloadFraction?: number | null;
  /** Cancels the in-flight staged download (paired with abandoning the stage). */
  onCancelStagedDownload?: () => void;
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
  onLoadPendingModel,
  stagedDownloadFraction,
  onCancelStagedDownload,
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
  const pendingSelection = useChatRuntimeStore((s) => s.pendingSelection);
  const abandonStagedModel = useChatRuntimeStore((s) => s.abandonStagedModel);
  const resetModelSettingsToLoaded = useChatRuntimeStore(
    (s) => s.resetModelSettingsToLoaded,
  );
  // A staged GGUF pick (deferred load) shows the GGUF load knobs so they can be
  // set before the single load.
  const pendingIsGguf = isPendingGguf(pendingSelection);
  // Short, human-readable name for the staged pick (HF ids carry an org prefix;
  // native picks are already a display label). Drives the "staged, not loaded"
  // callout so it's obvious the selection hasn't loaded yet.
  const stagedLabel = (() => {
    const id = pendingSelection?.id ?? "";
    const slash = id.lastIndexOf("/");
    const base = slash >= 0 ? id.slice(slash + 1) : id;
    return base || id;
  })();
  const isLoadedGguf =
    useChatRuntimeStore((s) => s.activeGgufVariant) != null;
  const isGguf = isLoadedGguf || pendingIsGguf;
  // A staged pick is always a local GGUF, so show its Model section (and the
  // Load button) even when the currently active model is external.
  const hasModelContent =
    pendingSelection != null ||
    (!isExternalModel && (isGguf || Boolean(params.checkpoint)));
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

  // A staged (not-yet-loaded) GGUF carries its own header context length on
  // pendingSelection, so the slider can use the staged model's real ceiling
  // without reading the loaded model's `ggufContextLength`.
  const stagedContextLength = pendingSelection?.contextLength ?? null;
  // While staging, the sheet reflects the STAGED model, so its header context
  // takes precedence over the loaded model's (which may differ or be larger).
  const baseContext = pendingIsGguf ? stagedContextLength : ggufContextLength;
  const baseNativeContext = pendingIsGguf
    ? stagedContextLength
    : ggufNativeContextLength;
  // Context controls render once we actually have a ceiling: for a staged GGUF,
  // once its header metadata arrives (post-download); otherwise post-load.
  const showContextControl = pendingIsGguf
    ? stagedContextLength != null
    : isLoadedGguf;
  const stagedDownloading =
    stagedDownloadFraction != null && stagedDownloadFraction < 1;
  const ctxDisplayValue = customContextLength ?? baseContext ?? "";
  const ctxMaxValue = baseNativeContext ?? baseContext ?? null;
  const kvDirty = kvCacheDtype !== loadedKvCacheDtype;
  const ctxDirty = customContextLength !== null;
  const specDirty = speculativeType !== loadedSpeculativeType;
  const specDraftDirty = specDraftNMax !== loadedSpecDraftNMax;
  const tpDirty = tensorParallel !== (loadedTensorParallel ?? false);
  const modelSettingsDirty =
    kvDirty || ctxDirty || specDirty || specDraftDirty || tpDirty;
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
  const [presetNameInput, setPresetNameInput] = useState(() => activePreset);
  const presetControlRowRef = useRef<HTMLDivElement>(null);
  const [presetMenuWidthPx, setPresetMenuWidthPx] = useState<
    number | undefined
  >(undefined);
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
      setPresetNameInput(name);
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
    const builtinPreset = BUILTIN_PRESETS.find(
      (preset) => preset.name === name,
    );
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
        setPresetNameInput(fallbackPreset.name);
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

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen) {
        setSystemPromptEditorOpen(false);
      }
      onOpenChange?.(nextOpen);
    },
    [onOpenChange],
  );

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
            {pendingSelection && (
              <Alert className="rounded-[14px] border-primary/30 bg-primary/5 px-3 py-2">
                <AlertTitle className="text-[12px] font-medium">
                  {stagedLabel} is staged, not loaded yet
                </AlertTitle>
                <AlertDescription className="text-[11.5px] leading-[1.45] text-muted-foreground">
                  Set the options below, then choose Load model to load it.
                </AlertDescription>
              </Alert>
            )}
            {isGguf && (
              <>
                {showContextControl && (
                <div className="space-y-3.5">
                  <div className="flex items-center justify-between gap-3">
                    <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                      Context Length
                    </span>
                    <NumericValueInput
                      value={
                        typeof ctxDisplayValue === "number"
                          ? ctxDisplayValue
                          : (baseContext ?? 0)
                      }
                      min={128}
                      max={ctxMaxValue ?? undefined}
                      step={1}
                      onChange={(v) => {
                        setCustomContextLength(
                          v === (baseContext ?? 0) ? null : v,
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
                          : (baseContext ?? 4096),
                        ctxMaxValue ?? 4096,
                      ),
                    ]}
                    onValueChange={([v]) => {
                      const snapped = Math.round(v);
                      setCustomContextLength(
                        snapped === (baseContext ?? 0) ? null : snapped,
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
                {isGguf && (
                  <>
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
                        {specFallbackReason === "mla_mtp_disabled"
                          ? "MTP is disabled by default for this model architecture because it currently runs slower than standard decoding. Select MTP above to force it."
                          : specFallbackReason === "runtime_error"
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
                  </>
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
                    data-test-id="tensor-parallel-switch"
                  />
                </div>
              </>
            )}
            {/* No persistent "enable custom code" toggle: it is consented per model
                via the load-time review dialog. */}
            {/* Apply/Reset belongs to the model-reload settings above (context
                length, KV cache, speculative decoding). Render it here, before
                the Chat Template row, so it never reads as attached to Chat
                Template (which is edited via its own dialog). When a model is
                staged (deferred load), Load/Cancel takes its place: there's
                nothing loaded to "apply" against yet. */}
            {pendingSelection ? (
              <div className="flex flex-col gap-2">
                {stagedDownloading && (
                  <p className="text-[11px] text-muted-foreground">
                    Downloading…{" "}
                    {Math.round((stagedDownloadFraction ?? 0) * 100)}%
                  </p>
                )}
                <div className="flex flex-wrap gap-1.5">
                  <Button
                    type="button"
                    onClick={() => onLoadPendingModel?.()}
                    disabled={stagedDownloading}
                    size="sm"
                    className="h-7 px-3 text-[12px] font-medium tracking-nav bg-primary/92 text-primary-foreground hover:bg-primary"
                  >
                    Load model
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      // Cancel abandons the stage; if a download is mid-flight,
                      // stop it too rather than leaving it running headless.
                      if (stagedDownloading) onCancelStagedDownload?.();
                      abandonStagedModel();
                    }}
                    className="h-7 px-3 text-[12px] font-medium tracking-nav text-muted-foreground"
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            ) : modelSettingsDirty ? (
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
                  onClick={() => resetModelSettingsToLoaded()}
                  className="h-7 px-3 text-[12px] font-medium tracking-nav text-muted-foreground"
                >
                  Reset
                </Button>
              </div>
            ) : null}
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
                // A staged GGUF caps to its own context even over an active
                // external model (the staged model is what will load).
                !pendingIsGguf && isExternalModel
                  ? getExternalMaxOutputTokens(
                      externalProviderType,
                      externalSelection?.modelId,
                    )
                  : isGguf && baseContext
                    ? baseContext
                    : 32768
              }
              step={64}
              onChange={set("maxTokens")}
              displayValue={
                isGguf && baseContext && params.maxTokens >= baseContext
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

        <DocumentExtractionSection />
      </div>
      </div>
      </div>
      <Dialog
        open={open && systemPromptEditorOpen}
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
      <Sheet open={open} onOpenChange={handleOpenChange}>
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
  const ocrPhase = useChatRuntimeStore((s) => s.ocrPhase);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const allModels = useChatRuntimeStore((s) => s.models);
  const [ocrPickerOpen, setOcrPickerOpen] = useState(false);
  const reducedMotion = useReducedMotion();

  const [support, setSupport] = useState<DocumentSupport | null>(null);
  const [probing, setProbing] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // Debounced, supersedable custom-code consent check for OCR selection.
  const [ocrChecking, setOcrChecking] = useState(false);
  const ocrConsentGen = useRef(0);
  const ocrConsentTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (ocrConsentTimer.current) clearTimeout(ocrConsentTimer.current);
    },
    [],
  );

  const probeSupport = useCallback(() => {
    abortRef.current?.abort();
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
    return ctrl;
  }, []);

  const runProbe = useCallback(() => {
    probeSupport();
  }, [probeSupport]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    const ctrl = probeSupport();
    return () => ctrl.abort();
  }, [checkpoint, probeSupport]);

  const extractorReady = support?.extraction_available ?? false;
  const unavailableFormatCount = Object.keys(
    support?.unavailable_formats ?? {},
  ).length;
  const extractorLimited = extractorReady && unavailableFormatCount > 0;
  const backendExtractConcurrencyLimit = Math.max(
    1,
    Math.min(
      DOC_EXTRACT_SLIDER_MAXES.extractConcurrency,
      support?.max_extract_concurrency ??
        DOC_EXTRACT_SLIDER_MAXES.extractConcurrency,
    ),
  );
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
  // Custom-code OCR models load like any other model (consent is gathered at
  // load time via the review dialog), so selecting one is enough to scan.
  const visionReadyForExtraction = visionAvailableForExtraction;
  const canScan = extractorReady && visionReadyForExtraction;
  const activeMode = deriveDocExtractMode(docExtract);

  // OCR-picker model list: the 4 OCR presets pinned at top + the user's
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

  // Pin these to the top of the OCR picker's Recommended list, tagged "OCR".
  const ocrPresetIdSet = useMemo(
    () => new Set(OCR_MODEL_PRESETS.map((p) => p.modelId)),
    [],
  );

  const handleOcrSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      const matchedPreset = OCR_MODEL_PRESETS.find((p) => p.modelId === id);
      const nextSettings: Partial<DocExtractSettings> = matchedPreset
        ? {
            ocrModel: matchedPreset.id,
            customOcrModelId: "",
            customOcrGgufVariant: null,
          }
        : {
            ocrModel: "custom",
            customOcrModelId: id,
            customOcrGgufVariant: meta.ggufVariant ?? null,
          };
      const prevSettings: Partial<DocExtractSettings> = {
        ocrModel: docExtract.ocrModel,
        customOcrModelId: docExtract.customOcrModelId,
        customOcrGgufVariant: docExtract.customOcrGgufVariant,
      };
      // Apply and close immediately so the picker never blocks on the network
      // custom-code check.
      setDocExtract(nextSettings);
      setOcrPickerOpen(false);

      // Debounce the consent check so clicking through models does not queue a
      // security scan per click; only the settled selection is verified. The
      // review dialog still appears up front (before any load), and declining
      // reverts to the previous model.
      const gen = ++ocrConsentGen.current;
      if (ocrConsentTimer.current) clearTimeout(ocrConsentTimer.current);
      ocrConsentTimer.current = setTimeout(() => {
        void (async () => {
          setOcrChecking(true);
          let approved = true;
          try {
            approved = await ensureOcrModelRemoteCodeApproved({
              ...docExtract,
              ...nextSettings,
            });
          } finally {
            if (gen === ocrConsentGen.current) setOcrChecking(false);
          }
          // Ignore a stale result if a newer selection superseded this one.
          if (gen === ocrConsentGen.current && !approved) {
            setDocExtract(prevSettings);
          }
        })();
      }, 300);
    },
    [docExtract, setDocExtract],
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
        backendExtractConcurrencyLimit,
        normalizeNonNegativeInteger(value),
      ),
    );
    setDocExtract({
      extractConcurrency: next,
    });
  };

  useEffect(() => {
    if (docExtract.extractConcurrency > backendExtractConcurrencyLimit) {
      setDocExtract({ extractConcurrency: backendExtractConcurrencyLimit });
    }
  }, [
    backendExtractConcurrencyLimit,
    docExtract.extractConcurrency,
    setDocExtract,
  ]);

  function applyMode(mode: DocExtractMode) {
    // An OCR selection grants vision capability at extract time, so defaults
    // match the "VLM available" branch even with no VLM loaded.
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
                  visionOnly
                  ocrPresetIds={ocrPresetIdSet}
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
          {ocrChecking && (
            <p className="text-[11px] text-muted-foreground">
              Verifying custom code…
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
                        disabled={!extractorReady}
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
                      tooltip="Maximum number of documents extracted in parallel. Extra files queue client-side and this value is capped to the backend worker limit."
                      value={docExtract.extractConcurrency}
                      sliderMax={backendExtractConcurrencyLimit}
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
            Overridden by Bypass permissions
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
            Bypass permissions
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
        <span className="text-[11px] text-bypass">
          Tool calls run with no confirmation and no sandbox.
        </span>
      ) : null}
      <AlertDialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Enable Bypass permissions?</AlertDialogTitle>
            <AlertDialogDescription>
              Bypass permissions is dangerous since the AI model might delete,
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
