// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { InfoHint } from "@/components/ui/info-hint";
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
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { NumericValueInput, snapToStep } from "@/features/model-picker";
import { RetrievalSettingsSection } from "@/features/rag";
import { useLlamaUpdateCheck } from "@/hooks/use-llama-update-check";
import { useIsMobile } from "@/hooks/use-mobile";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Edit03Icon, LayoutAlignRightIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Braces, ChevronDown, ExternalLink } from "lucide-react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { Fragment, type ReactNode } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { OpenAICodeExecSection } from "./components/openai-code-exec-section";
import { PermissionModeDropdown } from "./permission-mode-select";
import { resyncInferenceStatusAfterServerModelChange } from "./hooks/use-chat-model-runtime";
import {
  type ExternalProviderConfig,
  getExternalProviderApiKey,
  parseExternalModelId,
  supportsProviderPromptCacheTtl,
  supportsProviderPromptCaching,
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
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import type { InferenceParams } from "./types/runtime";

export { defaultInferenceParams, type Preset } from "./presets/preset-policy";
export type { InferenceParams } from "./types/runtime";

const PROMPT_VARIABLE_PATTERN = /{{\s*[a-zA-Z_$][a-zA-Z0-9_$.-]*\s*}}/;

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function getPromptVariablesError(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return null;
    }
  } catch {
    return 'Use valid JSON, for example { "env": "staging" }.';
  }
  return "Variables must be a JSON object.";
}

function hasPromptVariableSyntax(prompt: string): boolean {
  return PROMPT_VARIABLE_PATTERN.test(prompt);
}

export function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
  info,
  valueSize,
  disabled,
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
  disabled?: boolean;
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
          className="panel-number-input"
          disabled={disabled}
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(snapToStep(v, step, min, max))}
        className="panel-slider"
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
        !first && "border-t border-black/[0.13] dark:border-white/[0.09]",
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
  modelConfig?: ReactNode;
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
}

export function ChatSettingsPanel({
  open,
  onOpenChange,
  params,
  onParamsChange,
  modelConfig = null,
  isExternalModel = false,
  providerCapabilities = null,
  activeExternalProvider = null,
  onExternalProviderChange,
  externalProviderType = null,
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
  const isLoadedGguf = useChatRuntimeStore((s) => s.activeGgufVariant) != null;
  const currentCheckpoint = params.checkpoint;
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  // Direct-file / custom-folder GGUFs load without a variant label but still
  // report a GGUF context, so detect them via the context and the checkpoint
  // suffix too (mirrors the chat page's activeModelIsGguf). Otherwise Max Tokens
  // would fall back to params.maxSeqLength instead of the loaded GGUF context.
  const isGguf =
    isLoadedGguf ||
    ggufContextLength != null ||
    (currentCheckpoint?.toLowerCase().endsWith(".gguf") ?? false);
  const ggufMaxContextLength = useChatRuntimeStore(
    (s) => s.ggufMaxContextLength,
  );
  const customContextLength = useChatRuntimeStore((s) => s.customContextLength);
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const specFallbackReason = useChatRuntimeStore((s) => s.specFallbackReason);
  const mtpUpdatable =
    specFallbackReason === "binary_no_mtp" ||
    specFallbackReason === "binary_outdated";
  const {
    status: llamaUpdateStatus,
    applying: llamaUpdating,
    apply: applyLlamaUpdate,
  } = useLlamaUpdateCheck({
    enabled: mtpUpdatable,
    onReloadRequired: resyncInferenceStatusAfterServerModelChange,
  });
  const handleMtpUpdate = useCallback(async () => {
    const result = await applyLlamaUpdate();
    if (result.ok) {
      const reloadHint = result.reloadRequired
        ? " Reload your model to enable MTP."
        : "";
      toast.success(
        `llama.cpp updated to ${result.tag ?? "the latest build"}.${reloadHint}`,
      );
    } else {
      toast.error(
        `llama.cpp update failed: ${result.error ?? "unknown error"}`,
      );
    }
  }, [applyLlamaUpdate]);
  const loadedEffectiveContext = customContextLength ?? ggufContextLength;
  const showSpecFallback =
    !isExternalModel &&
    isGguf &&
    specFallbackReason != null &&
    (speculativeType === "auto" ||
      speculativeType === "mtp" ||
      speculativeType === "mtp+ngram");
  const showContextVramWarning =
    !isExternalModel &&
    isGguf &&
    ggufMaxContextLength != null &&
    loadedEffectiveContext != null &&
    loadedEffectiveContext > ggufMaxContextLength;
  const showLoadedDiagnostics = showSpecFallback || showContextVramWarning;
  const hasModelContent = showLoadedDiagnostics;
  const setActivePresetSource = useChatRuntimeStore(
    (s) => s.setActivePresetSource,
  );
  const activePresetSource = useChatRuntimeStore((s) => s.activePresetSource);
  const customPresets = useChatRuntimeStore((s) => s.customPresets);
  const setCustomPresets = useChatRuntimeStore((s) => s.setCustomPresets);
  const activePreset = useChatRuntimeStore((s) => s.activePreset);
  const setActivePreset = useChatRuntimeStore((s) => s.setActivePreset);
  const settingsHydrated = useChatRuntimeStore((s) => s.settingsHydrated);

  const baseContext = ggufContextLength;
  const [presetNameInput, setPresetNameInput] = useState(activePreset);
  const [systemPromptEditorOpen, setSystemPromptEditorOpen] = useState(false);
  const [systemPromptDraft, setSystemPromptDraft] = useState("");
  const [systemVariablesDraft, setSystemVariablesDraft] = useState("");
  const [systemVariablesOpen, setSystemVariablesOpen] = useState(false);
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
  const hasUnsavedPresetChanges = useMemo(() => {
      if (activePresetDefinition == null) {
        return false;
      }
      if (activePresetDefinition.name === "Default") {
        return activePresetSource === "modified";
      }
      return !isSamePresetConfig(activePresetDefinition.params, params);
  }, [activePresetDefinition, activePresetSource, params]);
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
  const systemVariablesError = getPromptVariablesError(systemVariablesDraft);
  const currentSystemPrompt = params.systemPrompt ?? "";
  const currentSystemVariables = params.systemVariables ?? "";
  const systemPromptEditorDirty =
    systemPromptDraft !== currentSystemPrompt ||
    systemVariablesDraft !== currentSystemVariables;
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
  const maxTokensMax = isExternalModel
      ? getExternalMaxOutputTokens(
          externalProviderType,
          externalSelection?.modelId,
        )
      : isGguf && baseContext
        ? baseContext
        : Math.max(64, params.maxSeqLength);
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
      BUILTIN_PRESETS.find((preset) => preset.name === "Default") ?? null;
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
    setSystemPromptDraft(currentSystemPrompt);
    setSystemVariablesDraft(currentSystemVariables);
    setSystemVariablesOpen(
      currentSystemVariables.trim().length > 0 ||
        hasPromptVariableSyntax(currentSystemPrompt),
    );
    setSystemPromptEditorOpen(true);
  }

  function saveSystemPromptEditor() {
    if (systemVariablesError) {
      toast.error("Fix prompt variables before saving", {
        description: systemVariablesError,
      });
      return;
    }
    const nextParams = {
      ...params,
      systemPrompt: systemPromptDraft,
      systemVariables: systemVariablesDraft.trim(),
    };
    const nextSource = isSamePresetConfig(activePresetBaseline, nextParams)
      ? getPresetSource(activePreset)
      : "modified";
    setActivePresetSource(nextSource);
    onParamsChange(nextParams);
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
      currentSystemPrompt.length > 0 &&
        el != null &&
        el.clientHeight > 0 &&
        el.scrollHeight > el.clientHeight + 1,
    );
  }, [currentSystemPrompt, open]);

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
                <TooltipPrimitive.Trigger asChild={true}>
                <button
                  type="button"
                  onClick={() => onOpenChange?.(false)}
                  className="flex h-[34px] w-[34px] cursor-pointer items-center justify-center rounded-full text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
        {(hasModelContent || modelConfig) && (
              <CollapsibleSection label="Model" defaultOpen={true} first={true}>
            <div className="flex flex-col gap-3 pt-1">
              {modelConfig}
              {showSpecFallback && (
                <div className="rounded-lg bg-amber-500/[0.08] px-3 py-2 text-[12px] leading-[1.4] text-nav-fg/80">
                  <p>
                    {specFallbackReason === "mla_mtp_disabled"
                      ? "MTP is disabled by default for this model architecture because it currently runs slower than standard decoding. Choose MTP in the model picker to force it."
                      : specFallbackReason === "runtime_error"
                        ? "MTP could not start for this model on the installed llama.cpp build, so it is running without speculative decoding."
                        : specFallbackReason === "drafter_not_found"
                          ? "This model supports MTP, but its drafter file could not be downloaded, so MTP is off and it falls back to n-gram speculative decoding where the llama.cpp build supports it. Check your network connection or Hugging Face access, then reload the model to retry the drafter."
                          : `MTP is not available in the installed llama.cpp build, so this model is running without it.${
                              llamaUpdateStatus?.update_available
                                ? " Update llama.cpp to enable it."
                                : ""
                            }`}
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
              {showContextVramWarning && (
                <p className="text-[11px] text-amber-500">
                  Context length exceeds the estimated VRAM capacity (
                      {ggufMaxContextLength?.toLocaleString()} tokens). The
                      model may use system RAM.
                </p>
              )}
            </div>
          </CollapsibleSection>
        )}

        <CollapsibleSection
          label="Preset"
          defaultOpen={true}
          first={!hasModelContent && !modelConfig}
        >
          <div className="flex flex-col gap-3 pt-1">
            <DropdownMenu>
                  <DropdownMenuTrigger asChild={true}>
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
                    variant={
                      presetSaveState.isSaveReady ? "default" : "outline"
                    }
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
                      Reuse compatible prompt prefixes for lower latency and
                      cost.
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
                        cache pool. The 1 hour pool costs 2x base input on write
                        vs 1.25x for 5 minute, but reads stay 0.1x for both, so
                        a single read landing more than 5 minutes after the
                        write pays off the premium.
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
                  <TooltipPrimitive.Trigger asChild={true}>
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
              value={currentSystemPrompt}
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
                    displayValue={
                      params.presencePenalty === 0 ? "Off" : undefined
                    }
                info="Penalizes any token that has already appeared at least once, encouraging the model to introduce new topics. 0 = off."
              />
            ) : null}
            <ParamSlider
              label="Max Tokens"
              value={params.maxTokens}
              min={
                isExternalModel
                  ? getExternalMinOutputTokens(externalProviderType)
                  : 64
              }
              max={maxTokensMax}
              step={64}
              onChange={set("maxTokens")}
              displayValue={
                isGguf && baseContext && params.maxTokens >= baseContext
                  ? "Max"
                  : !isExternalModel &&
                      !isGguf &&
                      params.maxTokens >= maxTokensMax
                    ? "Max"
                    : undefined
              }
              info="Maximum number of tokens to generate per response. Generation stops at this limit or when the model emits an end-of-sequence token."
            />
          </div>
        </CollapsibleSection>

            {isExternalModel ? null : (
          <CollapsibleSection label="Tools">
            <div className="flex flex-col gap-5 pt-1">
              <AutoHealToolCallsToggle />
              <NudgeToolCallsToggle />
              <ConfirmToolCallsToggle />
              <BypassPermissionsToggle />
              <MaxToolCallsSlider />
              <ToolCallTimeoutSlider />
            </div>
          </CollapsibleSection>
            )}

            {isExternalModel ? null : (
          <CollapsibleSection label="Retrieval">
            <RetrievalSettingsSection />
          </CollapsibleSection>
            )}
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
          <div className="space-y-3">
            <div className="space-y-0.5 px-0.5">
              <div className="flex items-center justify-between gap-3">
                <div className="text-[11px] font-medium">Prompt editor</div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setSystemVariablesOpen((open) => !open)}
                  className="h-7 gap-1.5 rounded-full px-2.5 text-[11px] text-muted-foreground"
                  aria-expanded={systemVariablesOpen}
                >
                  <Braces className="size-3.5" />
                  Variables
                  <ChevronDown
                    className={cn(
                      "size-3.5 transition-transform",
                      systemVariablesOpen && "rotate-180",
                    )}
                  />
                </Button>
              </div>
              <p className="text-[11px] text-muted-foreground">
                Use this for longer edits. Save writes back to the active
                configuration only. Insert variables with {"{{ env }}"}.
              </p>
            </div>
            {systemVariablesOpen ? (
              <div className="space-y-2 px-0.5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="space-y-0.5">
                    <div className="text-[11px] font-medium">
                      Prompt variables
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                      Define values as JSON below, then use each key in your
                      prompt, like {"{{ env }}"}.
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span className="text-[10px] text-muted-foreground">
                      Built-in, fill in automatically
                    </span>
                    <div className="flex flex-wrap justify-end gap-1">
                      {["{{$date}}", "{{$time}}", "{{$now}}"].map((token) => (
                        <span
                          key={token}
                          title={`${token} is replaced automatically when you send`}
                          className="rounded-full bg-muted px-2 py-0.5 font-mono text-[10px] text-muted-foreground"
                        >
                          {token}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                <Textarea
                  value={systemVariablesDraft}
                  onChange={(event) =>
                    setSystemVariablesDraft(event.target.value)
                  }
                  placeholder='{ "env": "staging", "version": "v2.3.1" }'
                  fieldSizing="fixed"
                  className={cn(
                    "min-h-24 border-0 font-mono text-xs leading-5 corner-squircle focus-visible:ring-0",
                    systemVariablesError &&
                      "ring-1 ring-destructive focus-visible:ring-destructive",
                  )}
                  rows={5}
                  aria-label="Prompt variables JSON"
                  aria-invalid={Boolean(systemVariablesError)}
                />
                {systemVariablesError ? (
                  <p className="px-1 text-[11px] text-destructive">
                    {systemVariablesError}
                  </p>
                ) : (
                  <p className="px-1 text-[11px] text-muted-foreground">
                    Names you don&apos;t define are left unchanged, so a stray
                    {" {{ typo }} "}stays visible in the prompt.
                  </p>
                )}
              </div>
            ) : null}
            <Textarea
              value={systemPromptDraft}
              onChange={(event) => setSystemPromptDraft(event.target.value)}
              placeholder="You are a helpful assistant..."
              fieldSizing="fixed"
              className="min-h-[20rem] max-h-[48vh] overflow-y-auto border-0 text-sm leading-6 corner-squircle focus-visible:ring-0"
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
                  setSystemPromptDraft(currentSystemPrompt);
                  setSystemVariablesDraft(currentSystemVariables);
                  setSystemPromptEditorOpen(false);
                }}
              >
                Cancel
              </Button>
              <Button
                type="button"
                onClick={saveSystemPromptEditor}
                disabled={
                  !systemPromptEditorDirty || Boolean(systemVariablesError)
                }
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
          <div data-tour="chat-settings" className="flex h-full flex-col">
            {settingsContent}
          </div>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <aside
      data-tour="chat-settings"
      className={cn(
        "relative z-50 shrink-0 overflow-hidden bg-panel-surface text-panel-surface-fg font-heading",
        open ? "w-[17rem] border-l border-sidebar-border" : "w-0",
      )}
      style={{
        height: "calc(100% - var(--studio-custom-titlebar-height, 0px))",
        marginTop: "var(--studio-custom-titlebar-height, 0px)",
      }}
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

function NudgeToolCallsToggle() {
  const nudgeToolCalls = useChatRuntimeStore((s) => s.nudgeToolCalls);
  const setNudgeToolCalls = useChatRuntimeStore((s) => s.setNudgeToolCalls);

  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
          Nudge Tool Calls
        </span>
        <InfoHint>
          When a tool call cannot be repaired, re-ask the model once so the
          intended tool still runs. API requests stay opt-in.
        </InfoHint>
      </div>
      <Switch
        className="panel-switch"
        checked={nudgeToolCalls}
        onCheckedChange={setNudgeToolCalls}
      />
    </div>
  );
}

function ConfirmToolCallsToggle() {
  const setConfirmToolCalls = useChatRuntimeStore((s) => s.setConfirmToolCalls);
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);

  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex min-w-0 flex-col gap-0.5">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            Confirm tool calls
          </span>
          <InfoHint>
            When on, every local Unsloth tool call pauses for your approval
            before it runs (the "Ask for approval" level). When off, tool calls
            run without prompts inside the sandbox (the "Off" level).
            Provider-hosted tools are not gated here.
          </InfoHint>
        </div>
        {permissionMode === "full" ? (
          <span className="text-[11px] text-muted-foreground">
            Overridden by Full access (Bypass permissions)
          </span>
        ) : null}
      </div>
      <Switch
        className="panel-switch"
        checked={permissionMode === "ask"}
        onCheckedChange={setConfirmToolCalls}
        disabled={permissionMode === "full"}
      />
    </div>
  );
}

function BypassPermissionsToggle() {
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="whitespace-nowrap text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
          Bypass permissions
        </span>
        <InfoHint>
          How Unsloth approves tool calls before they run. Full access is
          dangerous: it disables confirmations and the code sandbox.
        </InfoHint>
      </div>
      {/* Full width, styled like the panel selects/preset input. */}
      <PermissionModeDropdown triggerClassName="h-9 w-full justify-between rounded-full border-0 bg-[var(--panel-input-surface)] px-3.5 text-[13px] font-medium text-nav-fg shadow-none hover:bg-[var(--panel-input-surface)]" />
      {permissionMode === "full" ? (
        <span className="text-[11px] text-bypass">
          Tool calls run with no confirmation and no sandbox.
        </span>
      ) : null}
    </div>
  );
}
