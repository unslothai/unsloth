// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import {
  type KBMode,
  type ChunkingStrategy as RagChunkingStrategy,
  precacheRagReranker,
} from "@/features/rag/api/rag-api";
import { DocumentRow } from "@/features/rag/components/document-row";
import { KBCreateDialog } from "@/features/rag/components/kb-create-dialog";
import { PreviewPanel } from "@/features/rag/components/preview-panel";
import { useThreadDocuments } from "@/features/rag/hooks/use-kb-documents";
import { useKnowledgeBases } from "@/features/rag/hooks/use-knowledge-bases";
import { useResizablePanelWidth } from "@/features/rag/hooks/use-resizable-width";
import { usePreviewStore } from "@/features/rag/stores/preview-store";
import { useRagStore } from "@/features/rag/stores/rag-store";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useAui } from "@assistant-ui/react";
import {
  ArrowDown01Icon,
  ArrowTurnBackwardIcon,
  InformationCircleIcon,
  LayoutAlignRightIcon,
} from "@hugeicons/core-free-icons";
import { Add01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown } from "lucide-react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { type CSSProperties, Fragment, type ReactNode } from "react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import type { RagMode, RagSource } from "./api/chat-settings-api";
import { OpenAICodeExecSection } from "./components/openai-code-exec-section";
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
  EXTERNAL_MAX_OUTPUT_TOKENS,
  type ProviderCapabilities,
  getExternalMaxOutputTokens,
  getExternalMinOutputTokens,
  providerSupportsBuiltinCodeExecution,
  providerSupportsFastMode,
} from "./provider-capabilities";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import { ChatMcpServersDialog } from "./chat-mcp-servers-dialog";
import { listMcpServers } from "./api/mcp-servers-api";
import type { InferenceParams } from "./types/runtime";

function ragSourceLabel(
  source: RagSource,
  kbs: { id: string; name: string }[],
): string {
  if (source.kind === "off") return "Off";
  if (source.kind === "thread") return "This thread's documents";
  const kb = kbs.find((k) => k.id === source.kbId);
  return kb ? kb.name : "Knowledge base (missing)";
}

export { defaultInferenceParams, type Preset } from "./presets/preset-policy";
export type { InferenceParams } from "./types/runtime";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function InfoHint({ children }: { children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
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

const COLLAPSIBLE_STATE_KEY = "unsloth_chat_collapsible_state";
const RIGHT_SLOT_SHEET_BREAKPOINT = 1024;
const RIGHT_SLOT_SHEET_QUERY =
  `(max-width: ${RIGHT_SLOT_SHEET_BREAKPOINT - 1}px)`;

function getRightSlotSheetSnapshot(): boolean {
  if (
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return false;
  }
  return window.matchMedia(RIGHT_SLOT_SHEET_QUERY).matches;
}

function noopRightSlotSheetSubscription(): void {
  // No browser media-query subscription is needed during SSR/tests.
}

function subscribeRightSlotSheet(callback: () => void): () => void {
  if (
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return noopRightSlotSheetSubscription;
  }
  const mql = window.matchMedia(RIGHT_SLOT_SHEET_QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function useRightSlotSheet(): boolean {
  return useSyncExternalStore(
    subscribeRightSlotSheet,
    getRightSlotSheetSnapshot,
    () => false,
  );
}

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
        !first && "border-t border-black/[0.13] dark:border-white/[0.09]",
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
  isExternalModel?: boolean;
  /**
   * Sampling-param capability set for the active external provider, or `null`
   * for local models (in which case every knob is rendered). Drives the
   * per-param visibility in the sampling section.
   */
  providerCapabilities?: ProviderCapabilities | null;
  activeExternalProvider?: ExternalProviderConfig | null;
  onExternalProviderChange?: (provider: ExternalProviderConfig) => void;
  /**
   * Backend provider type for the active external model (e.g. "kimi",
   * "anthropic", "openai"), or `null` for local models. Drives the
   * per-provider Max Tokens floor in the slider.
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
  // For non-external (local) models we show every knob — providerCapabilities
  // is only consulted when `isExternalModel` is true. An external model with an
  // unknown provider falls back to the OpenAI-compat shape via
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
  const rightSlotUsesSheet = useRightSlotSheet();
  const isGguf = useChatRuntimeStore((s) => s.activeGgufVariant) != null;
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const setRagSource = useChatRuntimeStore((s) => s.setRagSource);
  const ragMode = useChatRuntimeStore((s) => s.ragMode);
  const setRagMode = useChatRuntimeStore((s) => s.setRagMode);
  const ragToolEnabled = useChatRuntimeStore((s) => s.ragToolEnabled);
  const enableRerank = useChatRuntimeStore((s) => s.enableRerank);
  const setEnableRerank = useChatRuntimeStore((s) => s.setEnableRerank);
  const ragTopK = useChatRuntimeStore((s) => s.ragTopK);
  const setRagTopK = useChatRuntimeStore((s) => s.setRagTopK);
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const { knowledgeBases, deleteKB } = useKnowledgeBases();
  const { documents: threadDocs, remove: removeThreadDoc } = useThreadDocuments(
    ragSource.kind === "thread" ? activeThreadId : null,
  );
  const clearThreadIndex = useRagStore((s) => s.clearThreadIndex);
  const reingestThread = useRagStore((s) => s.reingestThread);
  const threadSettings = useRagStore((s) =>
    activeThreadId ? s.threadSettings[activeThreadId] : undefined,
  );
  const loadThreadSettings = useRagStore((s) => s.loadThreadSettings);
  const updateThreadSettings = useRagStore((s) => s.updateThreadSettings);
  const ragDefaults = useRagStore((s) => s.defaults);

  // Load thread RAG settings once per threadId. Ref-guarded; keep
  // `threadSettings` out of deps to avoid post-load update loops.
  const threadSettingsLoadedRef = useRef<string | null>(null);
  useEffect(() => {
    if (
      ragSource.kind === "thread" &&
      activeThreadId &&
      threadSettingsLoadedRef.current !== activeThreadId
    ) {
      threadSettingsLoadedRef.current = activeThreadId;
      void loadThreadSettings(activeThreadId);
    }
  }, [ragSource.kind, activeThreadId, loadThreadSettings]);

  const effectiveThreadChunking: RagChunkingStrategy =
    threadSettings?.chunking_strategy ??
    ragDefaults?.chunking_strategy ??
    "standard";
  const effectiveThreadMode: KBMode =
    threadSettings?.mode ?? ragDefaults?.mode ?? "text";

  const aui = useAui();
  // Brand-new chat has no backend thread yet — initialize the local
  // assistant-ui thread to mint a remoteId so per-thread RAG settings
  // can be saved before the user sends a message or attaches a file.
  const ensureThreadId = async (): Promise<string | null> => {
    const stored = useChatRuntimeStore.getState().activeThreadId;
    if (stored) return stored;
    try {
      const runtime = aui.threads().__internal_getAssistantRuntime?.();
      if (!runtime) return null;
      const localId = runtime.threads.getState().mainThreadId;
      if (!localId) return null;
      const { remoteId } = await runtime.threads
        .getItemById(localId)
        .initialize();
      useChatRuntimeStore.getState().setActiveThreadId(remoteId);
      return remoteId;
    } catch {
      return null;
    }
  };

  const applyThreadSettingChange = (patch: {
    chunking_strategy?: RagChunkingStrategy;
    mode?: KBMode;
  }) => {
    void (async () => {
      const threadId = await ensureThreadId();
      if (!threadId) return;
      if (threadDocs.length === 0) {
        void updateThreadSettings(threadId, patch);
        return;
      }
      const ok = window.confirm(
        `Re-index ${threadDocs.length} document${threadDocs.length === 1 ? "" : "s"} ` +
          `with the new settings? Existing chunks will be deleted and rebuilt.`,
      );
      if (ok) {
        void reingestThread(threadId, patch);
      } else {
        // User declined: refresh so the select snaps back.
        void loadThreadSettings(threadId);
      }
    })();
  };
  const [kbCreateOpen, setKbCreateOpen] = useState(false);
  const ragEnabled = ragSource.kind !== "off";
  const activeKbId = ragSource.kind === "kb" ? ragSource.kbId : null;
  const hasModelContent =
    !isExternalModel && (isGguf || Boolean(params.checkpoint));
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const setSpeculativeType = useChatRuntimeStore((s) => s.setSpeculativeType);
  const loadedSpeculativeType = useChatRuntimeStore(
    (s) => s.loadedSpeculativeType,
  );
  const specDraftNMax = useChatRuntimeStore((s) => s.specDraftNMax);
  const setSpecDraftNMax = useChatRuntimeStore((s) => s.setSpecDraftNMax);
  const loadedSpecDraftNMax = useChatRuntimeStore((s) => s.loadedSpecDraftNMax);
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
  const modelSettingsDirty = kvDirty || ctxDirty || specDirty || specDraftDirty;
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
  const [presetNameInput, setPresetNameInput] = useState(activePreset);
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

  const openPreview = usePreviewStore((s) => s.open);
  const previewTarget = usePreviewStore((s) => s.target);
  const previewStatus = usePreviewStore((s) => s.status);

  const settingsContent = (
    <>
      <div className="aui-thread-viewport relative h-full overflow-y-auto">
        <div className="sticky top-0 z-10 flex h-[48px] items-start gap-2 bg-panel-surface pl-[18px] pr-[14px] pt-[11px]">
          {rightSlotUsesSheet ? (
            <span className="flex h-[34px] flex-1 items-center text-[15px] font-semibold tracking-[-0.01em] dark:tracking-[0.015em] text-nav-fg">
              Configuration
            </span>
          ) : (
            <>
              <span className="flex h-[34px] flex-1 items-center text-[15px] font-semibold tracking-[-0.01em] dark:tracking-[0.015em] text-nav-fg">
                Configuration
              </span>
              <Tooltip>
                <TooltipPrimitive.Trigger asChild={true}>
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
            <CollapsibleSection label="Model" defaultOpen={true} first={true}>
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
                            snapped === (ggufContextLength ?? 0)
                              ? null
                              : snapped,
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
                          Lower KV cache precision to save VRAM at the cost of
                          some quality. f16/bf16 are full precision;
                          q8_0/q5_1/q4_1 are quantized.
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
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex min-w-0 items-center gap-1.5">
                        <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                          Speculative Decoding
                        </span>
                        <InfoHint>
                          Faster generation with 0% accuracy hit. Auto picks MTP
                          / ngram-mod based on the model and platform. Pick MTP,
                          Ngram, or MTP+Ngram to force a specific strategy on
                          both GPU and CPU.
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
                            icon={ArrowDown01Icon}
                            iconClassName="size-3.5"
                            className="grid h-7 w-[120px] min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-[10px] border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.07] px-2 py-0 text-[13px]! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0"
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
                    {(speculativeType === "mtp" ||
                      speculativeType === "mtp+ngram") && (
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex min-w-0 items-center gap-1.5">
                          <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
                            Draft Tokens
                          </span>
                          <InfoHint>
                            Max MTP draft tokens per step (--spec-draft-n-max).
                            Lower = less wasted draft decode; higher = bigger
                            speedup when acceptance stays high. Default: 2 on
                            GPU, 3 on CPU/Mac.
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
                          className="h-7 w-[72px] rounded-[10px] border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.07] px-2 py-0 text-[13px] font-medium text-nav-fg outline-none focus-visible:ring-0"
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
                          This model requires custom code to load. You can edit
                          the toggle, but loading will stay blocked until it is
                          turned back on.
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
                        setSpecDraftNMax(loadedSpecDraftNMax);
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
                  disabled={!(settingsHydrated && activeCustomPreset)}
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
                      Anthropic exposes a 5 minute and a 1 hour ephemeral cache
                      pool. The 1 hour pool costs 2x base input on write vs
                      1.25x for 5 minute, but reads stay 0.1x for both, so a
                      single read landing more than 5 minutes after the write
                      pays off the premium.
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

          {ragToolEnabled ? (
            <CollapsibleSection label="Retrieval" defaultOpen={true}>
              <div className="flex flex-col gap-3 pt-1">
                <div className="flex flex-col gap-1.5">
                  <label className="text-[12px] font-medium text-muted-foreground">
                    Knowledge base
                  </label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild={true}>
                      <Button
                        variant="outline"
                        className="w-full justify-between"
                      >
                        <span className="truncate">
                          {ragSourceLabel(ragSource, knowledgeBases)}
                        </span>
                        <ChevronDown className="size-4 opacity-60" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                      align="start"
                      className="w-[var(--radix-dropdown-menu-trigger-width)]"
                    >
                      <DropdownMenuItem
                        onSelect={() => setRagSource({ kind: "thread" })}
                      >
                        This thread's documents
                      </DropdownMenuItem>
                      {knowledgeBases.length > 0 ? (
                        <DropdownMenuSeparator />
                      ) : null}
                      {knowledgeBases.map((kb) => {
                        const isActive = kb.id === activeKbId;
                        const isLate = kb.chunking_strategy === "late";
                        const isMultimodal = kb.mode === "multimodal";
                        return (
                          <DropdownMenuItem
                            key={kb.id}
                            className={cn(
                              "flex items-center justify-between gap-2",
                              isActive && "bg-accent text-accent-foreground",
                            )}
                            onSelect={() =>
                              setRagSource({ kind: "kb", kbId: kb.id })
                            }
                          >
                            <span className="flex min-w-0 items-center gap-1.5 truncate">
                              <span className="truncate">{kb.name}</span>
                              {isLate ? (
                                <span
                                  className="rounded-sm bg-amber-500/15 px-1 text-[10px] font-medium text-amber-700 dark:text-amber-300"
                                  title="Late chunking enabled"
                                >
                                  ⚡ Late
                                </span>
                              ) : null}
                              {isMultimodal ? (
                                <span
                                  className="rounded-sm bg-violet-500/15 px-1 text-[10px] font-medium text-violet-700 dark:text-violet-300"
                                  title="Multimodal mode — text + image embeddings"
                                >
                                  🖼️ MM
                                </span>
                              ) : null}
                            </span>
                            <button
                              type="button"
                              aria-label={`Delete ${kb.name}`}
                              className="ml-2 inline-flex size-5 items-center justify-center rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                              onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                if (
                                  window.confirm(
                                    `Delete "${kb.name}" and all its documents?`,
                                  )
                                ) {
                                  void deleteKB(kb.id).then(() => {
                                    if (isActive) {
                                      setRagSource({ kind: "thread" });
                                    }
                                  });
                                }
                              }}
                            >
                              <HugeiconsIcon icon={Delete02Icon} size={12} />
                            </button>
                          </DropdownMenuItem>
                        );
                      })}
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        onSelect={() => setKbCreateOpen(true)}
                        className="text-muted-foreground"
                      >
                        <HugeiconsIcon
                          icon={Add01Icon}
                          size={12}
                          className="mr-2"
                        />
                        Create knowledge base…
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <p className="text-[11px] text-muted-foreground">
                    Each message retrieves matching context from the selected
                    source before sending.
                  </p>
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-[12px] font-medium text-muted-foreground">
                    Search mode
                  </label>
                  <Select
                    value={ragMode}
                    onValueChange={(v) => setRagMode(v as RagMode)}
                    disabled={!ragEnabled}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="hybrid">
                        Hybrid (BM25 + semantic)
                      </SelectItem>
                      <SelectItem value="dense">Semantic only</SelectItem>
                      <SelectItem value="bm25">BM25 (lexical) only</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-[11px] text-muted-foreground">
                    Hybrid blends keyword (BM25) with vector similarity — best
                    default. Semantic-only ignores exact terms; BM25-only
                    ignores meaning.
                  </p>
                </div>
                <KBCreateDialog
                  open={kbCreateOpen}
                  onOpenChange={setKbCreateOpen}
                  onCreated={(kb) => setRagSource({ kind: "kb", kbId: kb.id })}
                />
                {ragSource.kind === "thread" ? (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex flex-col gap-1">
                        <label className="text-[11px] font-medium text-muted-foreground">
                          Mode
                        </label>
                        <Select
                          value={effectiveThreadMode}
                          onValueChange={(v) => {
                            const next = v as KBMode;
                            if (next === effectiveThreadMode) return;
                            applyThreadSettingChange({ mode: next });
                          }}
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="text">Text only</SelectItem>
                            <SelectItem
                              value="multimodal"
                              disabled={effectiveThreadChunking === "late"}
                              title={
                                effectiveThreadChunking === "late"
                                  ? "Multimodal cannot be combined with late chunking"
                                  : undefined
                              }
                            >
                              Multimodal
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="flex flex-col gap-1">
                        <label className="text-[11px] font-medium text-muted-foreground">
                          Chunking
                        </label>
                        <Select
                          value={effectiveThreadChunking}
                          onValueChange={(v) => {
                            const next = v as RagChunkingStrategy;
                            if (next === effectiveThreadChunking) return;
                            applyThreadSettingChange({
                              chunking_strategy: next,
                            });
                          }}
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="standard">Standard</SelectItem>
                            <SelectItem
                              value="late"
                              disabled={effectiveThreadMode === "multimodal"}
                              title={
                                effectiveThreadMode === "multimodal"
                                  ? "Late chunking cannot be combined with multimodal mode"
                                  : undefined
                              }
                            >
                              Late
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                      Changing either setting will re-index this thread's
                      existing documents.
                    </p>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-[12px] font-medium text-muted-foreground">
                        Documents in this thread
                      </label>
                      {threadDocs.length === 0 ? (
                        <p className="text-[11px] text-muted-foreground italic">
                          Attach a file using the + button in the composer to
                          add documents to this thread.
                        </p>
                      ) : (
                        <>
                          <div className="flex flex-col gap-1.5">
                            {threadDocs.map((doc) => (
                              <DocumentRow
                                key={doc.id}
                                doc={doc}
                                onPreview={
                                  doc.status === "completed"
                                    ? () => {
                                        void openPreview({
                                          documentId: doc.id,
                                        });
                                      }
                                    : undefined
                                }
                                onDelete={() => {
                                  void removeThreadDoc(doc.id);
                                }}
                              />
                            ))}
                          </div>
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                if (!activeThreadId) return;
                                if (
                                  window.confirm(
                                    `Re-index all ${threadDocs.length} document${threadDocs.length === 1 ? "" : "s"}? Existing chunks will be deleted and rebuilt; search will be unavailable until ingestion finishes.`,
                                  )
                                ) {
                                  void reingestThread(activeThreadId);
                                }
                              }}
                            >
                              Re-index
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="text-destructive hover:text-destructive hover:bg-destructive/10"
                              onClick={() => {
                                if (!activeThreadId) return;
                                if (
                                  window.confirm(
                                    `Delete all ${threadDocs.length} document${threadDocs.length === 1 ? "" : "s"} from this thread? This cannot be undone.`,
                                  )
                                ) {
                                  void clearThreadIndex(activeThreadId);
                                }
                              }}
                            >
                              Clear thread index
                            </Button>
                          </div>
                        </>
                      )}
                    </div>
                  </>
                ) : null}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-[12px] font-medium text-muted-foreground">
                      RAG top K
                    </label>
                    <span className="text-[12px] tabular-nums text-muted-foreground">
                      {ragTopK}
                    </span>
                  </div>
                  <Slider
                    value={[ragTopK]}
                    min={1}
                    max={20}
                    step={1}
                    onValueChange={([v]) => v != null && setRagTopK(v)}
                    disabled={!ragEnabled}
                  />
                  <p className="text-[11px] text-muted-foreground">
                    Number of retrieved chunks passed to the model as context
                    (distinct from the sampling Top K below). Higher = more
                    grounding, more tokens.
                  </p>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="text-[13px] font-medium">
                      Use reranker
                    </span>
                    <span className="text-[11px] text-muted-foreground">
                      Slower; uses GPU. Improves quality for fact-heavy
                      questions.
                    </span>
                  </div>
                  <Switch
                    checked={enableRerank}
                    onCheckedChange={(next) => {
                      setEnableRerank(next);
                      if (!next) return;
                      // First flip-on may have to download ~1.1 GB; the
                      // toast covers the latency so the user doesn't think
                      // the next query is hung waiting on the reranker.
                      const toastId = toast.loading(
                        "Preparing reranker (one-time download)…",
                      );
                      void precacheRagReranker()
                        .then((res) => {
                          if (res.ok) {
                            toast.success("Reranker ready", { id: toastId });
                          } else {
                            toast.error(
                              `Reranker download failed: ${res.error ?? "unknown"}`,
                              { id: toastId },
                            );
                            setEnableRerank(false);
                          }
                        })
                        .catch((err: unknown) => {
                          toast.error(
                            `Reranker download failed: ${
                              err instanceof Error ? err.message : String(err)
                            }`,
                            { id: toastId },
                          );
                          setEnableRerank(false);
                        });
                    }}
                    disabled={!ragEnabled}
                  />
                </div>
              </div>
            </CollapsibleSection>
          ) : null}

          <CollapsibleSection label="System Prompt" defaultOpen={true}>
            <button
              type="button"
              onClick={openSystemPromptEditor}
              aria-label="Edit system prompt"
              className={cn(
                "panel-text-surface mt-1 flex w-full h-20 overflow-hidden cursor-pointer items-start px-3.5 py-2.5 text-left text-[13px] font-medium leading-relaxed corner-squircle focus-visible:outline-none focus-visible:border-ring focus-visible:ring-[1px] focus-visible:ring-ring/40",
                params.systemPrompt ? "text-nav-fg" : "text-muted-foreground",
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
                    ? EXTERNAL_MAX_OUTPUT_TOKENS
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

          {isExternalModel ? null : (
            <CollapsibleSection label="Tools">
              <div className="flex flex-col gap-5 pt-1">
                <AutoHealToolCallsToggle />
                <MaxToolCallsSlider />
                <ToolCallTimeoutSlider />
              </div>
            </CollapsibleSection>
          )}

          {!isExternalModel ? (
            <CollapsibleSection label="MCP Servers">
              <McpServersSection />
            </CollapsibleSection>
          ) : null}
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

  // Right-slot routing (decision Q6 / Risk #11): the same slot hosts
  // EITHER the inference settings panel OR the document-preview panel,
  // never both. Preview takes precedence when a target is open. The
  // slot widens for preview because PDFs need more real estate than
  // a 17rem settings strip.
  const previewActive =
    previewTarget !== null ||
    previewStatus === "loading" ||
    previewStatus === "error";
  const slotOpen = open || previewActive;
  const slotShowsPreview = previewActive;
  const slotBody = slotShowsPreview ? (
    <PreviewPanel open={slotOpen} />
  ) : (
    settingsContent
  );

  if (rightSlotUsesSheet) {
    return (
      <Sheet
        open={slotOpen}
        onOpenChange={(next) => {
          if (next) {
            onOpenChange?.(true);
          } else {
            // Closing the sheet from outside closes BOTH the preview
            // and the settings — single right slot.
            usePreviewStore.getState().close();
            onOpenChange?.(false);
          }
        }}
      >
        <SheetContent
          side="right"
          className={cn(
            "p-0 font-heading",
            slotShowsPreview ? "w-[22rem]" : "w-[18rem]",
          )}
        >
          <SheetHeader className="sr-only">
            <SheetTitle>
              {slotShowsPreview ? "Document preview" : "Configuration"}
            </SheetTitle>
            <SheetDescription>
              {slotShowsPreview
                ? "Preview of the cited document"
                : "Chat inference settings"}
            </SheetDescription>
          </SheetHeader>
          <div className="flex h-full flex-col">
            {slotShowsPreview ? (
              <PreviewPanel open={slotOpen} disableDrawer={true} />
            ) : (
              settingsContent
            )}
          </div>
        </SheetContent>
      </Sheet>
    );
  }

  const {
    width: previewWidth,
    isResizing: previewResizing,
    startResize: startPreviewResize,
    adjustWidth: adjustPreviewWidth,
    resetWidth: resetPreviewWidth,
  } = useResizablePanelWidth({
    storageKey: "unsloth:chat-preview-panel-width",
    defaultWidth: 640,
    minWidth: 360,
    maxWidthFraction: 0.8,
    enabled: slotOpen && slotShowsPreview,
  });

  return (
    <aside
      className={cn(
        "relative z-50 shrink-0 h-full overflow-hidden bg-panel-surface text-panel-surface-fg font-heading",
        !previewResizing && "transition-[width] duration-200 ease-out",
        slotOpen
          ? slotShowsPreview
            ? "w-0 lg:w-[var(--preview-w)] lg:border-l lg:border-sidebar-border max-lg:border-0"
            : "w-0 lg:w-[17rem] lg:border-l lg:border-sidebar-border max-lg:border-0"
          : "w-0",
      )}
      style={
        slotShowsPreview
          ? ({ "--preview-w": `${previewWidth}px` } as CSSProperties)
          : undefined
      }
    >
      {slotShowsPreview ? (
        <button
          type="button"
          aria-label="Resize preview panel — drag, or use arrow keys"
          aria-orientation="vertical"
          onPointerDown={startPreviewResize}
          onDoubleClick={resetPreviewWidth}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft") {
              event.preventDefault();
              adjustPreviewWidth(24);
            } else if (event.key === "ArrowRight") {
              event.preventDefault();
              adjustPreviewWidth(-24);
            } else if (event.key === "Home") {
              event.preventDefault();
              resetPreviewWidth();
            }
          }}
          className={cn(
            "group absolute inset-y-0 left-0 z-20 hidden w-1.5 cursor-col-resize touch-none select-none border-0 bg-transparent p-0 outline-none lg:block",
            "before:absolute before:inset-y-0 before:left-1/2 before:-translate-x-1/2 before:w-px before:bg-transparent before:transition-colors",
            "hover:before:bg-primary/40 focus-visible:before:bg-primary/60",
            previewResizing && "before:bg-primary/60",
          )}
        >
          <span
            aria-hidden="true"
            className={cn(
              "absolute left-1/2 top-1/2 h-10 w-[3px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-border/70 opacity-0 transition-opacity",
              "group-hover:opacity-100 group-focus-visible:opacity-100",
              previewResizing && "opacity-100",
            )}
          />
        </button>
      ) : null}
      <div
        className={cn(
          "h-full w-full max-lg:invisible max-lg:pointer-events-none lg:visible lg:pointer-events-auto",
          slotShowsPreview && "max-lg:hidden",
        )}
      >
        {slotBody}
      </div>
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

function McpServersSection() {
  const mcpEnabledForChat = useChatRuntimeStore((s) => s.mcpEnabledForChat);
  const setMcpEnabledForChat = useChatRuntimeStore(
    (s) => s.setMcpEnabledForChat,
  );
  const [enabledServerCount, setEnabledServerCount] = useState<number | null>(
    null,
  );
  const [dialogOpen, setDialogOpen] = useState(false);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    listMcpServers()
      .then((rows) => {
        if (cancelled) return;
        setEnabledServerCount(rows.filter((row) => row.is_enabled).length);
      })
      .catch(() => {
        if (!cancelled) setEnabledServerCount(0);
      });
    return () => {
      cancelled = true;
    };
  }, [refreshTick]);

  return (
    <div className="flex flex-col gap-3 pt-1">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            Use MCP Servers
          </span>
          <InfoHint>
            When on, every server marked enabled in the manage dialog is
            attached to this chat's tool list.
          </InfoHint>
        </div>
        <Switch
          className="panel-switch"
          checked={mcpEnabledForChat}
          onCheckedChange={setMcpEnabledForChat}
          disabled={enabledServerCount === 0 && !mcpEnabledForChat}
        />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-muted-foreground">
          {enabledServerCount === null
            ? "Loading…"
            : enabledServerCount === 0
              ? "No servers configured"
              : `${enabledServerCount} server${enabledServerCount === 1 ? "" : "s"} enabled`}
        </span>
        <Button variant="ghost" size="sm" onClick={() => setDialogOpen(true)}>
          Manage…
        </Button>
      </div>
      <ChatMcpServersDialog
        open={dialogOpen}
        onOpenChange={(next) => {
          setDialogOpen(next);
          if (!next) setRefreshTick((tick) => tick + 1);
        }}
      />
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
              <TooltipPrimitive.Trigger asChild={true}>
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
              Override the model's chat template. The change applies on the next
              model reload.
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
