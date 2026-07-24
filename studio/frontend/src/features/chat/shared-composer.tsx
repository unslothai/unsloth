// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  thinkEffortAriaLabel,
  thinkToggleAriaLabel,
} from "@/components/assistant-ui/think-aria-label";
import { Button } from "@/components/ui/button";
import { BulbIcon } from "@/lib/bulb-icon";
import { MicIcon } from "@/lib/mic-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { applyQwenThinkingParams } from "@/features/chat/utils/qwen-params";
import {
  StudioDictationAdapter,
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
} from "@/features/chat/adapters/studio-dictation-adapter";
import type { StudioDictationSession } from "@/features/chat/adapters/studio-web-speech-dictation-adapter";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE, fileToBase64 } from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { isDownloadCancelled } from "@/lib/native-files";
import { isMultimodalResponse } from "./types/api";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import { useAui } from "@assistant-ui/react";
import {
  ArrowUpIcon,
  Columns2Icon,
  GlobeIcon,
  HeadphonesIcon,
  MoreHorizontalIcon,
  PlusIcon,
  SquareIcon,
  XIcon,
} from "lucide-react";
import {
  AttachmentIcon,
  Bookmark02Icon,
  CodeIcon,
  Download01Icon,
  FileDatabaseIcon,
  Folder01Icon,
  FolderAddIcon,
  Image03Icon,
  McpServerIcon,
  PencilRulerIcon,
} from "@hugeicons/core-free-icons";
import { useNavigate } from "@tanstack/react-router";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "@/lib/toast";
import {
  PromptStorageDialog,
  exportConversationShareGPT,
  exportConversationRawJsonl,
  exportConversationCsv,
} from "./prompt-storage/prompt-storage-dialog";
import { listPromptEntries, type PromptEntry } from "./api/prompts-api";
import { McpComposerButton } from "./mcp-composer-button";
import { BypassPermissionsMenuItem } from "./bypass-permissions-menu-item";
import { PermissionModeComposerPill } from "./permission-mode-select";
import { reasoningCapsFromLoad } from "./lib/apply-inference-status-to-store";
import { KnowledgeBaseComposerButton } from "@/features/rag/components/knowledge-base-composer-button";
import { NewProjectDialog } from "./components/new-project-dialog";
import { useChatProjects } from "./hooks/use-chat-projects";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import {
  DEFAULT_MAX_SEQ_LENGTH,
  normalizeMaxSeqLength,
  resolveInitialConfig,
  type PerModelConfig,
} from "@/features/model-picker";
import {
  confirmTransformersUpgradeIfNeeded,
  useTransformersUpgradeDialogStore,
} from "@/features/transformers-upgrade";
import { loadModel, validateModel } from "./api/chat-api";
import { resolveFitMaxSeqLength, resolveManualAutoCtxPin } from "./presets/preset-policy";
import { ensureGpuDeviceCache } from "@/hooks/use-gpu-info";
import {
  parseExternalModelId,
  providerTypeSupportsVision,
} from "./external-providers";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import { useIsMobile } from "@/hooks/use-mobile";
import {
  PLUS_MENU_ORDER,
  type PlusMenuItemId,
  usePlusMenuPrefsStore,
} from "./stores/plus-menu-prefs-store";
import {
  loadedGpuMemoryFields,
  type ReasoningEffort,
  reconcilePersistedGpuIds,
  resolveLoadedSpeculativeSettings,
  persistGpuMemoryModeOnLoad,
  resolveSpeculativeSettingsForLoad,
  saveSpeculativeType,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import {
  getExternalReasoningCapabilities,
  providerSupportsBuiltinCodeExecution,
  providerSupportsBuiltinImageGeneration,
  providerSupportsBuiltinWebFetch,
} from "./provider-capabilities";
import {
  type CompositionEvent,
  type FC,
  type KeyboardEvent,
  type MutableRefObject,
  type ReactElement,
  type ReactNode,
  Fragment,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

export type CompareMessagePart =
  | { type: "text"; text: string }
  | { type: "image"; image: string }
  | { type: "audio"; audio: string };

export interface CompareHandle {
  append: (content: CompareMessagePart[]) => void;
  /** Append a user message without triggering generation. */
  appendMessage: (content: CompareMessagePart[]) => void;
  /** Trigger generation on the current thread (after appendMessage). */
  startRun: () => void;
  cancel: () => void;
  isRunning: () => boolean;
  /** Returns a promise that resolves when the current or next run finishes. */
  waitForRunEnd: () => Promise<void>;
}

const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp,image/gif";
const MAX_IMAGE_SIZE = 20 * 1024 * 1024;

// Inlined to avoid a new icon dep. Kept in sync with the main composer.
const ArrowDownStandardIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    strokeLinecap="round"
    strokeLinejoin="round"
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden={true}
  >
    <path d="M5.99977 9.00005L11.9998 15L17.9998 9" />
  </svg>
);

function isNativeComposing(event: Event) {
  return "isComposing" in event && (event as InputEvent).isComposing === true;
}

// Mirrors the threshold in thread.tsx. Chrome on Windows-over-WSL (#5546)
// never fires `compositionend` after IME commit, so the compose flag would
// otherwise stay true forever.
const IME_STUCK_TIMEOUT_MS = 2500;

function fileToBase64DataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read image file"));
    reader.readAsDataURL(file);
  });
}

function formatReasoningEffortLabel(
  level: ReasoningEffort,
  modelId?: string,
): string {
  if (level === "max") return "Max";
  if (level === "xhigh") {
    const normalized = modelId?.trim().toLowerCase() ?? "";
    if (
      normalized.startsWith("claude-opus-4-6") ||
      normalized.startsWith("claude-sonnet-4-6")
    ) {
      return "Max";
    }
    return "Extra High";
  }
  return level.charAt(0).toUpperCase() + level.slice(1);
}

function formatReasoningDisabledLabel(
  supportsReasoningOff: boolean,
  isExternalOpenAIReasoning: boolean,
  modelId?: string,
): string {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  // Magistral keeps the "none" wire value, but UX presents this floor as
  // "Medium" rather than a disabled-state label.
  if (normalized.includes("magistral-medium-latest")) return "Medium";
  return supportsReasoningOff && isExternalOpenAIReasoning ? "None" : "Off";
}

function useDictation(
  setText: (value: string | ((prev: string) => string)) => void,
) {
  // Re-render support state when the user switches recognition engines.
  const dictationEngine = useVoiceSettingsStore((s) => s.dictationEngine);
  const [isDictating, setIsDictating] = useState(false);
  // True while a stopped recording's final audio is still transcribing; a
  // second click then cancels the pending transcription instead of re-stopping.
  const [isFinalizing, setIsFinalizing] = useState(false);
  const sessionRef = useRef<StudioDictationSession | null>(null);
  const startingRef = useRef(false);
  const finalizingRef = useRef(false);

  const start = useCallback(async () => {
    if (startingRef.current || sessionRef.current) return;
    // Unsupported engine (e.g. Firefox): explain and steer to the local model.
    if (!isStudioDictationAvailable()) {
      notifyStudioDictationUnavailable();
      return;
    }
    startingRef.current = true;

    let session: StudioDictationSession;
    try {
      // Routes to the engine chosen in Voice settings (browser or STT model),
      // honoring the selected microphone, language, and dictionary. Compare
      // feeds two panes, so recent dictations must not link the unrelated
      // single-chat active thread.
      session = new StudioDictationAdapter({ chatId: null }).listen();
    } catch {
      startingRef.current = false;
      notifyStudioDictationUnavailable();
      return;
    }
    sessionRef.current = session;
    setIsDictating(true);

    // Append final transcripts; the adapter has already applied the dictionary
    // and records the session in Recent dictations.
    session.onSpeech((result) => {
      if (!result.isFinal) return;
      const transcript = result.transcript?.trim() ?? "";
      if (transcript) {
        setText((prev) => (prev ? `${prev} ${transcript}` : transcript));
      }
    });
    session.onEnd?.(() => {
      if (sessionRef.current === session) sessionRef.current = null;
      finalizingRef.current = false;
      setIsFinalizing(false);
      setIsDictating(false);
    });
    startingRef.current = false;
  }, [setText]);

  const stop = useCallback(() => {
    const session = sessionRef.current;
    if (!session) return;
    // A second click while the final segment is transcribing discards the
    // pending transcription instead of leaving the pane stuck until timeout.
    if (finalizingRef.current) {
      session.cancel();
      if (sessionRef.current === session) sessionRef.current = null;
      finalizingRef.current = false;
      setIsFinalizing(false);
      setIsDictating(false);
      return;
    }
    finalizingRef.current = true;
    setIsFinalizing(true);
    // Keep the session and dictation state alive while its final audio segment
    // is transcribed. onEnd clears both after the transcript callbacks run.
    void session.stop().catch((error) => {
      console.error("Could not stop dictation:", error);
      session.cancel();
      if (sessionRef.current === session) sessionRef.current = null;
      finalizingRef.current = false;
      setIsFinalizing(false);
      setIsDictating(false);
    });
  }, []);

  useEffect(() => {
    return () => {
      sessionRef.current?.cancel();
      sessionRef.current = null;
    };
  }, []);

  const supported = StudioDictationAdapter.isSupported(dictationEngine);

  return { isDictating, isFinalizing, start, stop, supported };
}

export type CompareHandles = MutableRefObject<Record<string, CompareHandle>>;

const CompareHandlesContext = createContext<CompareHandles | null>(null);

export function CompareHandlesProvider({
  handlesRef,
  children,
}: {
  handlesRef: CompareHandles;
  children: ReactNode;
}): ReactElement {
  return (
    <CompareHandlesContext.Provider value={handlesRef}>
      {children}
    </CompareHandlesContext.Provider>
  );
}

export function RegisterCompareHandle({
  name,
}: {
  name: string;
}): ReactElement | null {
  const handlesRef = useContext(CompareHandlesContext);
  const aui = useAui();

  useEffect(() => {
    if (!handlesRef) {
      return;
    }
    const currentHandles = handlesRef.current;
    currentHandles[name] = {
      // fixes occasional reorder on reload.
      append: (content) =>
        aui
          .thread()
          .append({ role: "user", content, createdAt: new Date() } as never),
      appendMessage: (content) =>
        aui
          .thread()
          .append({
            role: "user",
            content,
            createdAt: new Date(),
            startRun: false,
          } as never),
      startRun: () => {
        const msgs = aui.thread().getState().messages;
        const lastId = msgs.length > 0 ? msgs[msgs.length - 1].id : null;
        aui.thread().startRun({ parentId: lastId });
      },
      cancel: () => aui.thread().cancelRun(),
      isRunning: () => aui.thread().getState().isRunning,
      waitForRunEnd: () =>
        new Promise<void>((resolve) => {
          let wasRunning = false;
          const unsub = useChatRuntimeStore.subscribe((state) => {
            const anyRunning = Object.keys(state.runningByThreadId).length > 0;
            if (anyRunning) wasRunning = true;
            if (wasRunning && !anyRunning) {
              unsub();
              resolve();
            }
          });
        }),
    };
    return () => {
      delete currentHandles[name];
    };
  }, [handlesRef, name, aui]);

  return null;
}

type PendingImage = { id: string; file: File };

function PendingImageThumb({
  file,
  onRemove,
}: {
  file: File;
  onRemove: () => void;
}): ReactElement {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  if (!src)
    return <div className="size-14 animate-pulse rounded-[14px] bg-muted" />;
  return (
    <div className="relative size-14 shrink-0 overflow-hidden rounded-[14px] border border-foreground/20 bg-muted">
      <img src={src} alt={file.name} className="h-full w-full object-cover" />
      <button
        type="button"
        onClick={onRemove}
        className="absolute top-1 right-1 flex size-5 items-center justify-center rounded-full bg-white text-muted-foreground shadow-sm hover:bg-destructive hover:text-destructive-foreground"
        aria-label="Remove attachment"
      >
        <XIcon className="size-3" />
      </button>
    </div>
  );
}

type CompareModelSelection = {
  id: string;
  isLora: boolean;
  ggufVariant?: string;
  config?: PerModelConfig;
};

function cleanCompareChatTemplate(
  value: string | null | undefined,
): string | null {
  return value?.trim() ? value : null;
}

function resolveCompareSpecDraftNMax(
  speculativeType: string | null,
  value: number | null,
): number | null {
  return speculativeType === "mtp" || speculativeType === "mtp+ngram"
    ? value
    : null;
}

// Tool icon plus an X overlay CSS reveals on hover when the pill is active.
function PillGlyph({ children }: { children: ReactNode }) {
  return (
    <span className="composer-pill-glyph">
      {children}
      <XIcon className="composer-pill-x" />
    </span>
  );
}

export function SharedComposer({
  handlesRef,
  model1,
  model2,
  onExitCompare,
  model1ThreadId,
  model2ThreadId,
}: {
  handlesRef: CompareHandles;
  model1?: CompareModelSelection;
  model2?: CompareModelSelection;
  onExitCompare?: () => void;
  model1ThreadId?: string;
  model2ThreadId?: string;
}): ReactElement {
  const navigate = useNavigate();
  // Exit compare: parent's restore handler, or fresh chat if opened by URL.
  const handleExitCompare = useCallback(() => {
    if (onExitCompare) {
      onExitCompare();
      return;
    }
    navigate({ to: "/chat" });
  }, [navigate, onExitCompare]);
  const [text, setText] = useState("");
  const [running, setRunning] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [pendingImages, setPendingImages] = useState<PendingImage[]>([]);
  const [pendingAudio, setPendingAudio] = useState<{
    name: string;
    base64: string;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [isComposing, setIsComposing] = useState(false);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [promptStorageOpen, setPromptStorageOpen] = useState(false);
  const [recentPrompts, setRecentPrompts] = useState<PromptEntry[]>([]);
  const refreshRecentPrompts = useCallback(async () => {
    try {
      const rows = await listPromptEntries();
      const byRecent = [...rows].sort((a, b) => b.updatedAt - a.updatedAt);
      // Pinned prompts take over the submenu; fall back to the 3 most recent.
      const pinnedIds = usePlusMenuPrefsStore.getState().pinnedPromptIds;
      const pinned = byRecent.filter((p) => pinnedIds.includes(p.id));
      setRecentPrompts(pinned.length > 0 ? pinned : byRecent.slice(0, 3));
    } catch {
    }
  }, []);
  const plusPins = usePlusMenuPrefsStore((s) => s.pins);
  const [isQueueRunning, setIsQueueRunning] = useState(false);
  const [queueProgress, setQueueProgress] = useState({ current: 0, total: 0 });
  const queueRef = useRef<string[]>([]);
  const queueIndexRef = useRef(0);
  const isQueueRunningRef = useRef(false);
  const prevRunningRef = useRef(false);
  const prevComparingRef = useRef(false);
  const compareStepSucceededRef = useRef(false);
  const sendRef = useRef<(() => void) | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composingRef = useRef(false);
  const stuckImeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  const activeModel = useChatRuntimeStore((s) => {
    const checkpoint = s.params.checkpoint;
    return s.models.find((m) => m.id === checkpoint);
  });
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const externalProvidersAll = useExternalProvidersStore((s) => s.providers);
  const externalProviders = connectionsEnabled ? externalProvidersAll : [];
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const lastModelLoadError = useChatRuntimeStore((s) => s.lastModelLoadError);
  const loadedIsMultimodal = useChatRuntimeStore((s) => s.loadedIsMultimodal);
  const supportsReasoning = useChatRuntimeStore((s) => s.supportsReasoning);
  const reasoningAlwaysOn = useChatRuntimeStore((s) => s.reasoningAlwaysOn);
  const reasoningEnabled = useChatRuntimeStore((s) => s.reasoningEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const reasoningStyle = useChatRuntimeStore((s) => s.reasoningStyle);
  const reasoningEffort = useChatRuntimeStore((s) => s.reasoningEffort);
  const supportsReasoningOff = useChatRuntimeStore(
    (s) => s.supportsReasoningOff,
  );
  const reasoningEffortLevels = useChatRuntimeStore(
    (s) => s.reasoningEffortLevels,
  );
  const setReasoningEffort = useChatRuntimeStore((s) => s.setReasoningEffort);
  const supportsPreserveThinking = useChatRuntimeStore(
    (s) => s.supportsPreserveThinking,
  );
  const preserveThinking = useChatRuntimeStore((s) => s.preserveThinking);
  const setPreserveThinking = useChatRuntimeStore((s) => s.setPreserveThinking);
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const supportsBuiltinWebSearch = useChatRuntimeStore(
    (s) => s.supportsBuiltinWebSearch,
  );
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const setCodeToolsEnabled = useChatRuntimeStore((s) => s.setCodeToolsEnabled);
  const imageToolsEnabled = useChatRuntimeStore((s) => s.imageToolsEnabled);
  const setImageToolsEnabled = useChatRuntimeStore(
    (s) => s.setImageToolsEnabled,
  );
  const artifactsEnabled = useChatRuntimeStore((s) => s.artifactsEnabled);
  const setArtifactsEnabled = useChatRuntimeStore((s) => s.setArtifactsEnabled);
  const showCanvasMenuItem = useChatRuntimeStore((s) => s.showCanvasMenuItem);
  const mcpEnabledForChat = useChatRuntimeStore((s) => s.mcpEnabledForChat);
  const setMcpEnabledForChat = useChatRuntimeStore(
    (s) => s.setMcpEnabledForChat,
  );
  // Three most recently updated projects for the quick-access submenu
  const { projects } = useChatProjects();
  const recentProjects = [...projects]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 3);
  const openProject = (projectId: string) => {
    useChatRuntimeStore.getState().setActiveProjectId(projectId);
    navigate({ to: "/chat", search: { project: projectId } });
  };
  const webFetchToolsEnabled = useChatRuntimeStore(
    (s) => s.webFetchToolsEnabled,
  );
  const setWebFetchToolsEnabled = useChatRuntimeStore(
    (s) => s.setWebFetchToolsEnabled,
  );
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const setRagEnabled = useChatRuntimeStore((s) => s.setRagEnabled);
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  // Empty until a compare run; gates Export chat off.
  const exportThreadIds = [model1ThreadId, model2ThreadId, activeThreadId].filter(
    (id): id is string => Boolean(id),
  );
  const lastOpenRouterChosenModel = useChatRuntimeStore(
    (s) => s.lastOpenRouterChosenModel,
  );
  const externalSelection = parseExternalModelId(checkpoint);
  const isExternalModel = externalSelection !== null;
  const selectedExternalProvider =
    externalSelection != null
      ? externalProviders.find((p) => p.id === externalSelection.providerId)
      : undefined;
  const imageUnavailableReason = getImageInputUnavailableReason({
    activeModel,
    isExternalModel,
    externalSupportsVision: providerTypeSupportsVision(
      selectedExternalProvider?.providerType,
    ),
    externalModelLabel: externalSelection?.modelId ?? null,
    loadedIsMultimodal,
    modelLoaded,
    loadError: lastModelLoadError,
  });
  const isCompareMode = Boolean(model1?.id || model2?.id);
  // Attach-time gate. Compare mode defers to send: the catalog can lag a
  // model's real capabilities (e.g. a GGUF whose mmproj arrives after the
  // snapshot), and models[] only syncs after ensureModelLoaded at send time.
  // Single mode uses the loaded model's runtime capability.
  const attachUnavailableReason = isCompareMode ? null : imageUnavailableReason;
  const effectiveExternalModelId =
    selectedExternalProvider?.providerType === "openrouter" &&
    externalSelection?.modelId === "openrouter/free" &&
    lastOpenRouterChosenModel
      ? lastOpenRouterChosenModel
      : externalSelection?.modelId;
  const externalReasoningCaps =
    externalSelection != null
      ? getExternalReasoningCapabilities(
          selectedExternalProvider?.providerType,
          effectiveExternalModelId,
          {
            isReasoningProvider:
              selectedExternalProvider?.isReasoningModel === true,
            baseUrl: selectedExternalProvider?.baseUrl ?? null,
          },
        )
      : null;
  const isExternalOpenAIReasoning =
    externalReasoningCaps?.supportsReasoning === true &&
    externalReasoningCaps.reasoningStyle === "reasoning_effort";
  const effectiveReasoningStyle =
    externalReasoningCaps?.reasoningStyle ?? reasoningStyle;
  const effectiveReasoningAlwaysOn =
    externalReasoningCaps?.reasoningAlwaysOn ?? reasoningAlwaysOn;
  const effectiveSupportsReasoningOff =
    externalReasoningCaps?.supportsReasoningOff ?? supportsReasoningOff;
  const effectiveReasoningEffortLevels =
    externalReasoningCaps?.reasoningEffortLevels ?? reasoningEffortLevels;
  const effectiveSupportsReasoning =
    externalReasoningCaps?.supportsReasoning ?? supportsReasoning;
  const reasoningLockedOn =
    effectiveSupportsReasoning &&
    (effectiveReasoningAlwaysOn || !effectiveSupportsReasoningOff);
  // Kimi's $web_search builtin mandates thinking=disabled
  // (https://platform.kimi.ai/docs/guide/use-web-search). Both pills stay
  // clickable, but turning one on flips the other off; the click handlers
  // below enforce this so the visible state matches what the backend sends.
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";
  const effectiveReasoningEnabled = reasoningLockedOn ? true : reasoningEnabled;
  const effectiveReasoningVisualEnabled =
    effectiveReasoningEnabled && reasoningEffort !== "none";
  const reasoningDisabled = !modelLoaded || !effectiveSupportsReasoning;
  const showReasoningControl =
    effectiveSupportsReasoning || effectiveReasoningAlwaysOn;
  // enable_thinking_effort (GLM-5.2: high|max + disable) reuses the effort
  // dropdown; it just also carries an Off row via supportsReasoningOff.
  const isEffort =
    effectiveReasoningStyle === "reasoning_effort" ||
    effectiveReasoningStyle === "enable_thinking_effort";
  // GLM-5.2's effort menu (Off, high, max) has short rows, so it can sit a
  // touch skinnier. Skip the narrower floor when a Preserve thinking row is
  // present, since that longer label needs the wider width to stay one line.
  const narrowEffortMenu =
    effectiveReasoningStyle === "enable_thinking_effort" &&
    !supportsPreserveThinking;
  const thinkingActiveLook = isEffort
    ? reasoningLockedOn || (effectiveReasoningVisualEnabled && !reasoningDisabled)
    : reasoningLockedOn || (effectiveReasoningEnabled && !reasoningDisabled);
  // Two-pill gating: Search lights up on a local tool runtime (supportsTools:
  // Code/python + local web_search) OR a provider-run server-side web_search
  // (supportsBuiltinWebSearch: OpenAI/Anthropic/OpenRouter/Kimi). Code lights
  // up on the local runtime OR Anthropic with a model accepting the
  // server-side code_execution_20250825 tool (see
  // providerSupportsBuiltinCodeExecution). Anthropic is the only external
  // provider shipping a code-execution tool today.
  const supportsBuiltinCodeExecution = providerSupportsBuiltinCodeExecution(
    selectedExternalProvider?.providerType,
    effectiveExternalModelId,
    selectedExternalProvider?.baseUrl,
  );
  const supportsBuiltinImageGeneration = providerSupportsBuiltinImageGeneration(
    selectedExternalProvider?.providerType,
    effectiveExternalModelId,
    selectedExternalProvider?.baseUrl,
  );
  const supportsBuiltinWebFetch = providerSupportsBuiltinWebFetch(
    selectedExternalProvider?.providerType,
  );
  // Gemini rejects codeExecution alongside image modalities. Search is
  // blocked on older Gemini image ids but allowed on Gemini 3 image models
  // (supportsBuiltinWebSearch encodes the per-model allowance), so we only
  // disable Code unconditionally in Gemini image mode.
  const isExternalGemini = selectedExternalProvider?.providerType === "gemini";
  const imageDisabled = !modelLoaded || !supportsBuiltinImageGeneration;
  const imageModeDisablesCode =
    isExternalGemini && imageToolsEnabled && !imageDisabled;
  // Image-tier Gemini models always reject codeExecution and reject
  // web_search on older ids (Gemini 3.x Pro/Flash allow it, encoded in
  // supportsBuiltinWebSearch). Don't let local `supportsTools` re-enable a
  // pill the Gemini backend silently drops: detect image-tier Gemini and
  // gate strictly on provider builtin support.
  const isGeminiImageTier =
    isExternalGemini && supportsBuiltinImageGeneration;
  // Disable only when a loaded model lacks the capability; with no model the
  // tool can still be pre-selected, matching the + menu.
  const searchDisabled =
    modelLoaded &&
    (isGeminiImageTier
      ? !supportsBuiltinWebSearch
      : !(supportsTools || supportsBuiltinWebSearch));
  const codeDisabled =
    (modelLoaded &&
      (isGeminiImageTier
        ? true
        : !(supportsTools || supportsBuiltinCodeExecution))) ||
    imageModeDisablesCode;
  // Images pill lights only on OpenAI cloud Responses-API models and the
  // Gemini Nano Banana family. No local tool runtime fallback.
  const showImagePill = supportsBuiltinImageGeneration;
  // Fetch pill: Anthropic-only (web_fetch_20250910 / web_fetch_20260209).
  const webFetchDisabled = !modelLoaded || !supportsBuiltinWebFetch;
  const showWebFetchPill = supportsBuiltinWebFetch;
  // Docs (RAG) is local-only: search_knowledge_base needs the local runtime.
  // Disable only when a loaded model can't run it; with no model the toggle
  // can still be pre-selected, matching Web search/Code/MCP.
  const ragDisabled = modelLoaded && (isExternalModel || !supportsTools);
  const showRagPill = !isExternalModel;
  // Above 4 pills, collapse to icons only. Compare, Search, Code, and
  // permissions always show; the rest are conditional. Narrow viewports
  // collapse too: the labelled row is wider than a phone-width composer.
  const isMobile = useIsMobile();
  const pillCount =
    4 +
    (showImagePill ? 1 : 0) +
    (showRagPill && ragEnabled ? 1 : 0) +
    (showWebFetchPill ? 1 : 0) +
    (artifactsEnabled ? 1 : 0) +
    (mcpEnabledForChat ? 1 : 0);
  const pillsCompact = isMobile || pillCount > 4;
  // Backwards-compatible alias for call sites still referencing
  // `toolsDisabled` (rare; both pills used it before).
  const toolsDisabled = codeDisabled;
  const setPendingAudioStore = useChatRuntimeStore((s) => s.setPendingAudio);
  const clearPendingAudioStore = useChatRuntimeStore(
    (s) => s.clearPendingAudio,
  );

  const {
    isDictating,
    isFinalizing: isDictationFinalizing,
    start: startDictation,
    stop: stopDictation,
  } = useDictation(setText);

  useEffect(() => {
    const id = setInterval(() => {
      const handles = handlesRef.current;
      const any = Object.values(handles).some((h) => h.isRunning());
      setRunning(any);
    }, 200);
    return () => clearInterval(id);
  }, [handlesRef]);

  function advanceQueue() {
    const nextIndex = queueIndexRef.current + 1;
    if (nextIndex >= queueRef.current.length) {
      isQueueRunningRef.current = false;
      setIsQueueRunning(false);
      queueRef.current = [];
      queueIndexRef.current = 0;
      setQueueProgress({ current: 0, total: 0 });
      toast.success("Prompt queue complete");
      return;
    }
    queueIndexRef.current = nextIndex;
    setQueueProgress({ current: nextIndex + 1, total: queueRef.current.length });
    const next = queueRef.current[nextIndex];
    toast(`Prompt ${nextIndex + 1} / ${queueRef.current.length}`, {
      description: next.length > 80 ? next.slice(0, 80) + "…" : next,
    });
    setText(next);
    setTimeout(() => { sendRef.current?.(); }, 100);
  }

  // Compare mode: advance the queue on cycle end, but stop on a failed step so we
  // don't burn prompts on incomplete results.
  useEffect(() => {
    const wasComparing = prevComparingRef.current;
    prevComparingRef.current = comparing;
    if (!isQueueRunningRef.current || !wasComparing || comparing) return;
    if (!compareStepSucceededRef.current) {
      isQueueRunningRef.current = false;
      setIsQueueRunning(false);
      queueRef.current = [];
      queueIndexRef.current = 0;
      setQueueProgress({ current: 0, total: 0 });
      toast.error("Prompt queue stopped", {
        description: "A compare step failed; remaining prompts were not sent.",
      });
      return;
    }
    prevRunningRef.current = false;
    advanceQueue();
  }, [comparing]);

  useEffect(() => {
    const wasRunning = prevRunningRef.current;
    prevRunningRef.current = running;
    if (!isQueueRunningRef.current || !wasRunning || running || comparing) return;
    advanceQueue();
  }, [running, comparing]);

  // Auto-expand textarea up to 6 rows, then scroll (matches regular chat composer).
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    const styles = window.getComputedStyle(ta);
    const lineHeight = parseFloat(styles.lineHeight) || 20;
    const paddingY =
      parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
    const borderY =
      parseFloat(styles.borderTopWidth) + parseFloat(styles.borderBottomWidth);
    const maxHeight = lineHeight * 6 + paddingY + borderY;
    const next = Math.min(ta.scrollHeight, maxHeight);
    ta.style.height = `${next}px`;
    ta.style.overflowY = ta.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [text]);

  const addFiles = useCallback(
    (files: FileList | null) => {
      if (!files?.length) return;
      const next: PendingImage[] = [];
      let droppedImageForUnavailable = false;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file) continue;
        // Handle audio files
        if (file.type.match(/^audio\//i) && file.size <= MAX_AUDIO_SIZE) {
          fileToBase64(file).then((base64) => {
            setPendingAudio({ name: file.name, base64 });
            setPendingAudioStore(base64, file.name);
          });
          continue;
        }
        // Handle image files
        if (!file.type.match(/^image\/(jpeg|png|webp|gif)$/i)) continue;
        if (file.size > MAX_IMAGE_SIZE) continue;
        if (attachUnavailableReason) {
          droppedImageForUnavailable = true;
          continue;
        }
        next.push({ id: crypto.randomUUID(), file });
      }
      if (droppedImageForUnavailable && attachUnavailableReason) {
        toast.error(attachUnavailableReason);
      }
      setPendingImages((prev) => [...prev, ...next]);
    },
    [setPendingAudioStore, attachUnavailableReason],
  );

  const removePendingImage = useCallback((id: string) => {
    setPendingImages((prev) => prev.filter((p) => p.id !== id));
  }, []);

  function clearStuckImeTimer() {
    if (stuckImeTimerRef.current) {
      clearTimeout(stuckImeTimerRef.current);
      stuckImeTimerRef.current = null;
    }
  }

  function setCompositionState(next: boolean) {
    composingRef.current = next;
    setIsComposing(next);
    clearStuckImeTimer();
    if (next) {
      stuckImeTimerRef.current = setTimeout(() => {
        stuckImeTimerRef.current = null;
        composingRef.current = false;
        setIsComposing(false);
      }, IME_STUCK_TIMEOUT_MS);
    }
  }

  function refreshStuckImeTimer() {
    if (!composingRef.current) {
      return;
    }
    clearStuckImeTimer();
    stuckImeTimerRef.current = setTimeout(() => {
      stuckImeTimerRef.current = null;
      composingRef.current = false;
      setIsComposing(false);
    }, IME_STUCK_TIMEOUT_MS);
  }

  useEffect(() => () => clearStuckImeTimer(), []);

  async function send() {
    if (composingRef.current) return;
    const msg = text.trim();
    if (!msg && pendingImages.length === 0 && !pendingAudio) return;

    const hasCompareHandles = Boolean(
      handlesRef.current["model1"] || handlesRef.current["model2"],
    );
    const isGeneralizedCompare =
      hasCompareHandles && Boolean(model1?.id && model2?.id);

    // Generalized compare requires both panes to have a model. A half-
    // selected send either races to an empty bubble with bogus tok/s (#5569)
    // or leaves the empty pane with a dangling prompt. hasCompareHandles is
    // true only in GeneralCompareContent, so LoraCompare and single-pane
    // chats are unaffected.
    if (hasCompareHandles && !isGeneralizedCompare) {
      toast.error("Pick a model in each pane to compare", {
        description:
          "Use the model dropdown above each pane, then send your prompt.",
      });
      return;
    }

    if (
      pendingImages.length > 0 &&
      !isGeneralizedCompare &&
      imageUnavailableReason
    ) {
      // Single mode: the loaded model's runtime capability is known here.
      // Compare mode defers: each ensureModelLoaded sets loadedIsMultimodal
      // for its side, and the chat-adapter's pre-stream gate runs per-side
      // against that fresh state.
      toast.error(imageUnavailableReason);
      return;
    }

    const content: CompareMessagePart[] = [];
    for (const { file } of pendingImages) {
      try {
        const image = await fileToBase64DataURL(file);
        content.push({ type: "image", image });
      } catch {
        // skip failed image
      }
    }
    if (pendingAudio) {
      content.push({ type: "audio", audio: pendingAudio.base64 });
    }
    if (msg) {
      content.push({ type: "text", text: msg });
    }
    if (content.length === 0) return;

    setText("");
    setPendingImages([]);
    setPendingAudio(null);
    clearPendingAudioStore();
    textareaRef.current?.focus();

    // Generalized compare: load each model before dispatching to its side
    if (isGeneralizedCompare) {
      const store = useChatRuntimeStore.getState();
      const trustRemoteCode = store.params.trustRemoteCode ?? false;
      const fallbackTensorParallel = store.tensorParallel;
      const specSettings = resolveSpeculativeSettingsForLoad({
        usePersistedPreference: true,
      });
      let loadedFromConfig = false;

      function modelDisplayName(id: string): string {
        const parts = id.split("/");
        return parts[parts.length - 1] || id;
      }

      // Warm the device cache before the snapshot below reconciles the GPU
      // pick: on a cold cache the reconcile passes a stale pick through.
      if (store.selectedGpuIds != null) {
        await ensureGpuDeviceCache();
      }
      // The GPU/offload knobs both compare loads must use, snapshotted at Send.
      // ensureModelLoaded runs sequentially and the first load's response echo
      // (loadedGpuMemoryFields) rewrites the live store -- a non-GGUF or Auto
      // first model resets gpuLayers/nCpuMoe/split/pick to defaults -- so
      // reading the store per load would hand model 2 the first model's echoed
      // defaults instead of the settings the user pressed Send with.
      const compareLoadKnobs = {
        gpuMemoryMode: store.gpuMemoryMode,
        gpuLayers: store.gpuLayers,
        nCpuMoe: store.nCpuMoe,
        splitRatio: store.splitRatio,
        // Reconcile the pick against the GPUs present now, like the model-switch
        // path: an early remember-restore can hold a stale cross-host pick that
        // /load would reject (the device cache is populated by send time).
        selectedGpuIds: reconcilePersistedGpuIds(store.selectedGpuIds),
        customContextLength: store.customContextLength,
      };
      // Set when an accepted transformers install unloaded the active model
      // server-side; a later failure must then clear the stale checkpoint.
      let upgradeUnloadedActive = false;
      // Helper: load a model and update store checkpoint
      async function ensureModelLoaded(
        sel: CompareModelSelection,
      ): Promise<string> {
        const currentStore = useChatRuntimeStore.getState();
        const config = sel.config ?? null;
        // This pane's effective config: an explicit selection config, else the
        // remembered store config for this model/quant (never the other pane's).
        // No saved config resolves to all-null defaults, so settings below fall
        // through to their session default.
        const resolved = config
          ? { config, remembered: true }
          : resolveInitialConfig(sel.id, sel.ggufVariant ?? null);
        const ownConfig = resolved.config;
        const ownRemembered = resolved.remembered;
        // Mirror single-view resolveLoadMaxSeqLength: a GGUF pane with no explicit
        // context loads at native (0 -> n_ctx_train), not the session maxSeqLength,
        // which would silently shrink the shown context.
        const isGgufLoad =
          (sel.ggufVariant ?? null) != null ||
          sel.id.toLowerCase().endsWith(".gguf");
        // A non-GGUF pane with no saved maxSeqLength falls back to the app default,
        // not the active model's shared runtime snapshot: else comparing a saved
        // 128K model against an unconfigured one loads the latter at 128K and OOMs.
        const effectiveMaxSeqLength =
          ownConfig.customContextLength ??
          normalizeMaxSeqLength(ownConfig.maxSeqLength) ??
          (isGgufLoad ? 0 : DEFAULT_MAX_SEQ_LENGTH);
        const effectiveChatTemplateOverride = cleanCompareChatTemplate(
          ownConfig.chatTemplateOverride,
        );
        const effectiveSpeculativeType =
          ownConfig.speculativeType ?? specSettings.speculativeType;
        const effectiveSpecDraftNMax = ownRemembered
          ? resolveCompareSpecDraftNMax(
              effectiveSpeculativeType,
              ownConfig.specDraftNMax,
            )
          : specSettings.specDraftNMax;
        const effectiveTensorParallel = ownRemembered
          ? ownConfig.tensorParallel
          : fallbackTensorParallel;
        if (ownConfig.selectedGpuIds != null) {
          await ensureGpuDeviceCache();
        }
        const effectiveGpuMemoryMode =
          ownConfig.gpuMemoryMode ?? compareLoadKnobs.gpuMemoryMode;
        const effectiveGpuLayers =
          ownConfig.gpuLayers ?? compareLoadKnobs.gpuLayers;
        const effectiveNCpuMoe =
          ownConfig.nCpuMoe ?? compareLoadKnobs.nCpuMoe;
        const effectiveSelectedGpuIds =
          ownConfig.selectedGpuIds !== undefined
            ? reconcilePersistedGpuIds(
                ownConfig.selectedGpuIds,
                ownConfig.selectedGpuIndexKind ?? null,
              )
            : compareLoadKnobs.selectedGpuIds;
        const effectiveMemoryMode = ownConfig.ggufMemoryMode ?? null;
        // A pane's context comes from its own config only: a saved pin, or null
        // (Auto/native). It must not inherit the active model's shared snapshot --
        // resolveFitMaxSeqLength would treat that as a pin and load this pane at
        // the other model's context (changing VRAM/results or OOMing).
        const effectiveCustomContextLength = ownConfig.customContextLength;
        let loadTrustRemoteCode = trustRemoteCode;
        let approvedRemoteCodeFingerprint: string | null = null;
        const isAlreadyActive =
          currentStore.params.checkpoint === sel.id &&
          (currentStore.activeGgufVariant ?? null) ===
            (sel.ggufVariant ?? null);
        if (isAlreadyActive && !config && !loadedFromConfig) {
          return "ready";
        }
        const targetIsGguf =
          sel.id.toLowerCase().endsWith(".gguf") || sel.ggufVariant != null;
        // Size validation exactly as the load below, so the training-guard
        // preflight checks the footprint that actually loads (under Manual + Auto
        // layers the load sends 0 / the pinned context, not raw maxSeqLength).
        const compareMaxSeqLength = resolveFitMaxSeqLength(
          targetIsGguf,
          effectiveGpuMemoryMode,
          effectiveGpuLayers,
          // Prefer this pane's own saved context pin over the shared snapshot,
          // falling back to its per-pane effective context (GGUF with no saved
          // context loads at native, not the session maxSeqLength).
          effectiveCustomContextLength,
          effectiveMaxSeqLength,
        );
        const validation = await validateModel({
          model_path: sel.id,
          hf_token: currentStore.hfToken || null,
          max_seq_length: compareMaxSeqLength,
          load_in_4bit: true,
          is_lora: sel.isLora,
          gguf_variant: sel.ggufVariant ?? null,
          trust_remote_code: loadTrustRemoteCode,
          chat_template_override: effectiveChatTemplateOverride,
          // Scope the validate to the picked GPUs. GGUF-only, like the load
          // below: a non-GGUF target must not inherit a hidden GGUF GPU pick.
          ...(targetIsGguf
            ? {
                gpu_ids: effectiveSelectedGpuIds ?? undefined,
                gpu_memory_mode: effectiveGpuMemoryMode,
                gguf_memory_mode: effectiveMemoryMode,
              }
            : {}),
        });
        // Upgrade dialog first (mirrors the primary load path).
        if (validation.requires_transformers_upgrade) {
          const upgraded = await confirmTransformersUpgradeIfNeeded({
            modelName: sel.id,
            upgrade: validation.transformers_upgrade,
            // No installable release: custom-code models may fall back to the trust_remote_code gate below.
            trustRemoteCodeFallback: validation.requires_trust_remote_code,
          });
          // The install unloads the active model before the swap (even when the
          // swap then fails); if a later gate cancels or the load fails, the UI
          // must stop pointing at that unloaded model.
          if (
            useTransformersUpgradeDialogStore
              .getState()
              .consumeServerUnloadedChat()
            && currentStore.params.checkpoint
          ) {
            upgradeUnloadedActive = true;
          }
          if (!upgraded) {
            throw new Error(
              `${modelDisplayName(sel.id)} needs a newer transformers release to load.`,
            );
          }
        }
        if (
          validation.requires_trust_remote_code ||
          validation.requires_security_review
        ) {
          const approved = await confirmRemoteCodeIfNeeded({
            modelName: sel.id,
            hfToken: currentStore.hfToken || null,
            requiresTrustRemoteCode: true,
            onApprove: (fp) => {
              loadTrustRemoteCode = true;
              approvedRemoteCodeFingerprint = fp;
            },
          });
          if (!approved) {
            throw new Error(
              `${modelDisplayName(sel.id)} needs custom code approval to load.`,
            );
          }
        }
        const resp = await loadModel({
          model_path: sel.id,
          hf_token: useChatRuntimeStore.getState().hfToken || null,
          max_seq_length: compareMaxSeqLength,
          load_in_4bit: true,
          is_lora: sel.isLora,
          gguf_variant: sel.ggufVariant ?? null,
          trust_remote_code: loadTrustRemoteCode,
          approved_remote_code_fingerprint: approvedRemoteCodeFingerprint,
          chat_template_override: effectiveChatTemplateOverride,
          cache_type_kv: ownConfig.kvCacheDtype ?? null,
          speculative_type: effectiveSpeculativeType,
          spec_draft_n_max: effectiveSpecDraftNMax,
          tensor_parallel: effectiveTensorParallel,
          ...(targetIsGguf
            ? {
                gpu_memory_mode: effectiveGpuMemoryMode,
                gpu_layers: effectiveGpuLayers,
                n_cpu_moe: effectiveNCpuMoe,
                tensor_split: compareLoadKnobs.splitRatio ?? undefined,
                gpu_ids: effectiveSelectedGpuIds ?? undefined,
                gguf_memory_mode: effectiveMemoryMode,
              }
            : {}),
        });
        // Keep a compare pane's per-model speculative choice load-local: persist
        // the global preference only when it came from global settings.
        if (ownConfig.speculativeType == null) {
          saveSpeculativeType(effectiveSpeculativeType);
        }
        // Persist the GPU Memory mode on a non-diffusion GGUF compare-load too,
        // so an applied manual choice survives a restart.
        persistGpuMemoryModeOnLoad(resp, effectiveGpuMemoryMode);
        upgradeUnloadedActive = false;
        const store = useChatRuntimeStore.getState();
        store.setCheckpoint(
          resp.model,
          resp.is_gguf ? (sel.ggufVariant ?? undefined) : null,
        );
        store.setModelRequiresTrustRemoteCode(
          resp.requires_trust_remote_code ?? false,
        );
        // Keep an explicit Manual+Auto context pin the load just applied (so a
        // later Apply/Reset doesn't silently revert the model to auto-fit
        // sizing), mirroring the interactive path's keepCustomCtx. Non-GGUF
        // compare loads don't send the pin, so their baseline clears.
        const keepCustomCtx = targetIsGguf
          ? resolveManualAutoCtxPin(
              effectiveGpuMemoryMode,
              effectiveGpuLayers,
              effectiveCustomContextLength,
            )
          : null;
        useChatRuntimeStore.setState({
          supportsReasoning: resp.supports_reasoning ?? false,
          reasoningAlwaysOn: resp.reasoning_always_on ?? false,
          ...reasoningCapsFromLoad(resp),
          supportsPreserveThinking: resp.supports_preserve_thinking ?? false,
          supportsTools: resp.supports_tools ?? false,
          kvCacheDtype: resp.cache_type_kv ?? null,
          loadedKvCacheDtype: resp.cache_type_kv ?? null,
          tensorParallel: resp.tensor_parallel ?? false,
          loadedTensorParallel: resp.tensor_parallel ?? false,
          defaultChatTemplate: resp.chat_template ?? null,
          chatTemplateOverride: effectiveChatTemplateOverride,
          loadedChatTemplateOverride: effectiveChatTemplateOverride,
          // The context baseline this pane loaded with (see keepCustomCtx above),
          // so a later Apply/Reset can't silently revert a Manual+Auto pin.
          loadedCustomContextLength: keepCustomCtx,
          // Adopt the load response's GPU-memory fields (mode/layers/MoE/split/pick
          // plus loaded baselines) so the GPU controls round-trip. (gguf context,
          // customContextLength and native-path token/expiry clear in the tail below.)
          ...loadedGpuMemoryFields(resp),
          // Drives the GPU Memory controls' diffusion gate; set alongside the
          // GPU fields on every load path so the gate can't read stale.
          loadedIsDiffusion: resp.is_diffusion ?? false,
          loadedIsMultimodal: isMultimodalResponse(resp),
          // Record the context this pane loaded with (like the single-model path)
          // so when it becomes the active model, the UI and later reload/save use
          // its context, not the previous/default one.
          customContextLength: isGgufLoad
            ? (ownConfig.customContextLength ?? keepCustomCtx)
            : null,
          ggufContextLength: resp.is_gguf ? (resp.context_length ?? null) : null,
          ggufNativeContextLength: resp.is_gguf
            ? (resp.native_context_length ?? null)
            : null,
          ggufMaxContextLength: resp.is_gguf
            ? (resp.max_context_length ?? null)
            : null,
          // Compare selections load by repo/variant, never from the file picker,
          // so they carry no native lease. Clear any prior picked file's
          // token/expiry so the reload path never sends a stale lease.
          activeNativePathToken: null,
          activeNativePathExpiresAtMs: null,
          ...resolveLoadedSpeculativeSettings(resp),
        });
        if (!isGgufLoad) {
          // Non-GGUF panes carry their context in params.maxSeqLength.
          store.setParams({
            ...useChatRuntimeStore.getState().params,
            maxSeqLength: effectiveMaxSeqLength,
          });
        }
        loadedFromConfig = config != null;
        // Sync the models[] entry with the load response so attach/send gates
        // read fresh capabilities. /api/models/list can lag a model's actual
        // state (e.g. a GGUF whose mmproj arrived after the snapshot).
        const currentModels = useChatRuntimeStore.getState().models;
        const idx = currentModels.findIndex((m) => m.id === sel.id);
        const synced = {
          isVision: Boolean(resp.is_vision),
          isGguf: Boolean(resp.is_gguf),
          isAudio: Boolean(resp.is_audio),
          audioType: resp.audio_type ?? null,
          hasAudioInput: Boolean(resp.has_audio_input),
        };
        if (idx === -1) {
          store.setModels([
            ...currentModels,
            {
              id: sel.id,
              name: resp.display_name ?? sel.id,
              isLora: sel.isLora,
              ...synced,
            },
          ]);
        } else {
          const next = [...currentModels];
          next[idx] = { ...next[idx], ...synced };
          store.setModels(next);
        }
        return resp.status;
      }

      const handle1 = handlesRef.current["model1"];
      const handle2 = handlesRef.current["model2"];

      // Show user messages immediately on both sides
      if (handle1) handle1.appendMessage(content);
      if (handle2) handle2.appendMessage(content);

      const name1 = model1?.id ? modelDisplayName(model1.id) : "";
      const name2 = model2?.id ? modelDisplayName(model2.id) : "";
      const toastId = toast("Comparing models…", { duration: Infinity });

      setComparing(true);
      try {
        // Side 1: load → generate → wait
        if (handle1 && model1?.id) {
          toast("Loading Model 1…", {
            id: toastId,
            description: name1,
            duration: Infinity,
          });
          const status1 = await ensureModelLoaded(model1);
          toast("Generating with Model 1…", {
            id: toastId,
            description: `${name1} (${status1})`,
            duration: Infinity,
          });
          const done = handle1.waitForRunEnd();
          handle1.startRun();
          await done;
        }

        // Side 2: load → generate → wait
        if (handle2 && model2?.id) {
          const needsLoad =
            model2.id.toLowerCase() !== (model1?.id || "").toLowerCase() ||
            (model2.ggufVariant ?? "") !== (model1?.ggufVariant ?? "");
          if (needsLoad) {
            toast("Loading Model 2…", {
              id: toastId,
              description: name2,
              duration: Infinity,
            });
          }
          const status2 = await ensureModelLoaded(model2);
          toast("Generating with Model 2…", {
            id: toastId,
            description: `${name2} (${status2})`,
            duration: Infinity,
          });
          const done = handle2.waitForRunEnd();
          handle2.startRun();
          await done;
        }

        compareStepSucceededRef.current = true;
        toast.success("Compare complete", { id: toastId, duration: 2000 });
      } catch (err) {
        compareStepSucceededRef.current = false;
        // The install already unloaded the previously active model; drop the
        // checkpoint so the UI does not keep pointing at an unloaded model.
        if (upgradeUnloadedActive) {
          useChatRuntimeStore.getState().clearCheckpoint();
        }
        toast.error("Compare failed", {
          id: toastId,
          description: err instanceof Error ? err.message : "Unknown error",
          duration: 4000,
        });
      } finally {
        setComparing(false);
      }
    } else {
      // Original behavior: fire all handles simultaneously
      for (const handle of Object.values(handlesRef.current)) {
        handle.append(content);
      }
    }
  }
  sendRef.current = send;

  function stop() {
    if (isDictating) stopDictation();
    for (const handle of Object.values(handlesRef.current)) {
      handle.cancel();
    }
  }

  const busy = running || comparing;

  function onKeyDown(e: KeyboardEvent) {
    // IME composition (JP/CN/KR): Enter commits the candidate, don't hijack it
    // (#5318). Re-pin composingRef in case the stuck watchdog (#5546) cleared
    // it during a long candidate-window pause, so a follow-up click-Send won't
    // submit preedit text. Re-arm the watchdog on the same path; without it the
    // WSL+Chrome no-compositionend case pins composingRef forever after an IME
    // keypress and re-locks Send.
    if (e.nativeEvent.isComposing || e.keyCode === 229) {
      composingRef.current = true;
      refreshStuckImeTimer();
      return;
    }
    // Non-IME key while composingRef is stuck; mirrors the fix in thread.tsx.
    // On macOS, switching input methods without composing can leave composingRef
    // pinned; clear it immediately on the first non-IME keystroke.
    if (composingRef.current) {
      // Candidate-confirming Enter can arrive as non-composing; keep it gated.
      if (e.key === "Enter") {
        if (!e.shiftKey) {
          e.preventDefault();
        }
        refreshStuckImeTimer();
        return;
      }
      setCompositionState(false);
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!busy && !isDictating) {
        send();
      }
    }
  }

  const canSend =
    (text.trim().length > 0 ||
      pendingImages.length > 0 ||
      pendingAudio !== null) &&
    !busy &&
    !isComposing &&
    !isDictating;

  // Adjustable "+" menu items, keyed by id. Pinned ones render at the top
  // level; the rest fall into the "More" overflow submenu. Core items (photos,
  // web search, code) and "More" itself live outside this map.
  const plusMenuNodes: Record<PlusMenuItemId, ReactNode> = {
    chatWithFiles: (
      <DropdownMenuItem
        disabled={ragDisabled}
        className={
          ragEnabled && !ragDisabled ? "text-primary font-medium" : undefined
        }
        onSelect={() => setRagEnabled(!ragEnabled)}
      >
        <HugeiconsIcon icon={FileDatabaseIcon} strokeWidth={2} />
        Chat with Files
        {ragEnabled && !ragDisabled ? (
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
        ) : null}
      </DropdownMenuItem>
    ),
    mcp: (
      <DropdownMenuItem
        disabled={!supportsTools}
        className={mcpEnabledForChat ? "text-primary font-medium" : undefined}
        onSelect={() => setMcpEnabledForChat(!mcpEnabledForChat)}
      >
        <HugeiconsIcon icon={McpServerIcon} strokeWidth={2} />
        MCP
        {mcpEnabledForChat ? (
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
        ) : null}
      </DropdownMenuItem>
    ),
    savedPrompts: (
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>
          <HugeiconsIcon icon={Bookmark02Icon} strokeWidth={2} />
          Saved prompts
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent
          collisionPadding={16}
          className="unsloth-plus-menu w-[208px]"
        >
          {recentPrompts.map((p) => (
            <DropdownMenuItem
              key={p.id}
              onSelect={() => {
                setText(p.text);
                requestAnimationFrame(() => textareaRef.current?.focus());
              }}
            >
              <span className="truncate">{p.name}</span>
            </DropdownMenuItem>
          ))}
          {recentPrompts.length > 0 ? <DropdownMenuSeparator /> : null}
          <DropdownMenuItem onSelect={() => setPromptStorageOpen(true)}>
            All saved prompts…
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    ),
    compareChat: (
      // Always active: this menu only renders in compare mode. Click exits.
      <DropdownMenuItem
        className="text-primary font-medium"
        onSelect={handleExitCompare}
      >
        <Columns2Icon />
        Compare chat
        <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
      </DropdownMenuItem>
    ),
    exportChat: (
      <DropdownMenuSub>
        <DropdownMenuSubTrigger disabled={exportThreadIds.length === 0}>
          <HugeiconsIcon icon={Download01Icon} strokeWidth={2} />
          Export chat
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent
          collisionPadding={16}
          className="unsloth-plus-menu w-[208px]"
        >
          {[
            { label: "Raw JSONL", fn: exportConversationRawJsonl },
            { label: "CSV", fn: exportConversationCsv },
            { label: "ShareGPT JSONL", fn: exportConversationShareGPT },
          ].map(({ label, fn }) => (
            <DropdownMenuItem
              key={label}
              disabled={exportThreadIds.length === 0}
              onSelect={() => {
                if (!exportThreadIds.length) {
                  toast.error("No conversation to export yet.");
                  return;
                }
                (async () => {
                  for (const id of exportThreadIds) {
                    await fn(id);
                  }
                })().catch((error) => {
                  if (!isDownloadCancelled(error)) toast.error("Export failed.");
                });
              }}
            >
              {label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    ),
    // Hidden by default; enabled from Settings > Chat > Canvas.
    canvas: showCanvasMenuItem ? (
      <DropdownMenuItem
        className={artifactsEnabled ? "text-primary font-medium" : undefined}
        onSelect={() => setArtifactsEnabled(!artifactsEnabled)}
      >
        <HugeiconsIcon icon={PencilRulerIcon} strokeWidth={2} />
        Canvas
        {artifactsEnabled ? (
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
        ) : null}
      </DropdownMenuItem>
    ) : null,
    bypassPermissions: <BypassPermissionsMenuItem />,
    projects: (
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>
          <HugeiconsIcon icon={Folder01Icon} strokeWidth={2} />
          Projects
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent className="unsloth-plus-menu w-[232px]">
          <DropdownMenuItem onSelect={() => setNewProjectOpen(true)}>
            <HugeiconsIcon icon={FolderAddIcon} strokeWidth={2} />
            New project
          </DropdownMenuItem>
          <DropdownMenuLabel>Recents</DropdownMenuLabel>
          {recentProjects.length > 0 ? (
            recentProjects.map((project) => (
              <DropdownMenuItem
                key={project.id}
                onSelect={() => openProject(project.id)}
              >
                <HugeiconsIcon icon={Folder01Icon} strokeWidth={2} />
                <span className="truncate">{project.name}</span>
              </DropdownMenuItem>
            ))
          ) : (
            <DropdownMenuItem disabled={true}>
              No recent projects
            </DropdownMenuItem>
          )}
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    ),
  };
  const pinnedPlusItems = PLUS_MENU_ORDER.filter((id) => plusPins[id]);
  const overflowPlusItems = PLUS_MENU_ORDER.filter((id) => !plusPins[id]);

  return (
    <div
      className="chat-composer-surface"
      onDragOver={(e) => {
        if (isTauri) return;
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        // Phase 1 native model drops own Tauri local-path drops. Restore
        // browser attachment drops in Tauri once Phase 1d adds token bridging.
        if (isTauri) return;
        e.preventDefault();
        setDragging(false);
        addFiles(e.dataTransfer.files);
      }}
    >
      <PromptStorageDialog
        open={promptStorageOpen}
        onOpenChange={setPromptStorageOpen}
        onUse={(t) => {
          setText(t);
          requestAnimationFrame(() => textareaRef.current?.focus());
        }}
        onRunList={(items) => {
          const filtered = items.filter((p) => p.trim());
          if (!filtered.length) return;
          const hasCompareHandles = Boolean(
            handlesRef.current["model1"] || handlesRef.current["model2"],
          );
          const isGeneralizedCompare =
            hasCompareHandles && Boolean(model1?.id && model2?.id);
          if (hasCompareHandles && !isGeneralizedCompare) {
            toast.error("Pick a model in each pane to compare", {
              description:
                "Use the model dropdown above each pane, then send your prompt.",
            });
            return;
          }
          setPromptStorageOpen(false);
          queueRef.current = filtered;
          queueIndexRef.current = 0;
          isQueueRunningRef.current = true;
          setIsQueueRunning(true);
          setQueueProgress({ current: 1, total: filtered.length });
          toast(`Prompt 1 / ${filtered.length}`, {
            description: filtered[0].length > 80 ? filtered[0].slice(0, 80) + "…" : filtered[0],
          });
          setText(filtered[0]);
          setTimeout(() => { sendRef.current?.(); }, 100);
        }}
      />
      {/* Gemini-style drop affordance, mirrored from the single composer. */}
      <div
        className={`pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center gap-1 overflow-hidden rounded-[32px] bg-background/90 backdrop-blur-sm transition-opacity duration-150 dark:bg-card/90 ${dragging ? "opacity-100" : "opacity-0"}`}
      >
        <HugeiconsIcon
          icon={AttachmentIcon}
          strokeWidth={2}
          className="size-6 text-primary"
        />
        <span className="text-sm font-medium text-primary">Drop files here</span>
      </div>
      {(pendingImages.length > 0 || pendingAudio) && (
        <div className="mb-2 flex w-full flex-row flex-wrap items-center gap-2 px-1.5 pt-0.5 pb-1">
          {pendingImages.map(({ id, file }) => (
            <PendingImageThumb
              key={id}
              file={file}
              onRemove={() => removePendingImage(id)}
            />
          ))}
          {pendingAudio && (
            <div className="flex items-center gap-2 rounded-lg border border-foreground/20 bg-muted px-3 py-1.5 text-xs">
              <HeadphonesIcon className="size-3.5 text-muted-foreground" />
              <span className="max-w-48 truncate">{pendingAudio.name}</span>
              <button
                type="button"
                onClick={() => {
                  setPendingAudio(null);
                  clearPendingAudioStore();
                }}
                className="flex size-4 items-center justify-center rounded-full hover:bg-destructive hover:text-destructive-foreground"
                aria-label="Remove audio"
              >
                <XIcon className="size-3" />
              </button>
            </div>
          )}
        </div>
      )}
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => {
          // ALWAYS mirror the DOM value into React state, even during IME
          // composition: the controlled `value` must match the DOM at all
          // times, else an unrelated parent re-render reconciles the textarea
          // back to the stored value mid-composition, wiping the IME preedit
          // AND prior committed text (e.g. Tab-cycling candidates erases
          // earlier words). #5318.
          setCompositionState(isNativeComposing(e.nativeEvent));
          setText(e.target.value);
        }}
        onCompositionStart={() => {
          setCompositionState(true);
        }}
        onCompositionUpdate={() => {
          refreshStuckImeTimer();
        }}
        onCompositionEnd={(e: CompositionEvent<HTMLTextAreaElement>) => {
          setCompositionState(false);
          setText(e.currentTarget.value);
        }}
        onKeyDown={onKeyDown}
        onBlur={() => {
          // Mac: switching input methods can fire compositionstart without a
          // matching compositionend, leaving composingRef pinned. The OS always
          // commits or cancels composition before the element loses focus.
          setCompositionState(false);
        }}
        placeholder="Send to both models..."
        className="composer-input"
        rows={1}
        // dir="auto" detects RTL (Arabic/Hebrew/Persian/Urdu) from the first
        // strong character; no effect on LTR scripts.
        dir="auto"
      />
      <div className="composer-action-wrapper">
        <div
          className="flex min-w-0 flex-wrap items-center gap-0.5"
          data-pill-compact={pillsCompact ? "true" : undefined}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={IMAGE_ACCEPT}
            multiple
            className="hidden"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={audioInputRef}
            type="file"
            accept={AUDIO_ACCEPT}
            className="hidden"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <NewProjectDialog
            open={newProjectOpen}
            onOpenChange={setNewProjectOpen}
          />
          {/* Same + menu as single-chat (ComposerToolsMenu), wired to the
              compare composer's own file/audio inputs and tools. */}
          <DropdownMenu
            onOpenChange={(open) => {
              if (open) void refreshRecentPrompts();
            }}
          >
            <DropdownMenuTrigger asChild={true}>
              <button
                type="button"
                aria-label="Tools and attachments"
                className="unsloth-composer-plus"
              >
                <PlusIcon className="size-[22px] stroke-[1.75px]" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              side="top"
              align="start"
              sideOffset={0}
              avoidCollisions={true}
              className="unsloth-plus-menu w-[244px]"
              onCloseAutoFocus={(event) => event.preventDefault()}
            >
              <DropdownMenuItem onSelect={() => fileInputRef.current?.click()}>
                <HugeiconsIcon icon={AttachmentIcon} strokeWidth={2} />
                Add photos &amp; files
              </DropdownMenuItem>
              {activeModel?.hasAudioInput && (
                <DropdownMenuItem
                  onSelect={() => audioInputRef.current?.click()}
                >
                  <HeadphonesIcon />
                  Upload audio
                </DropdownMenuItem>
              )}
              <DropdownMenuItem
                disabled={searchDisabled}
                className={
                  toolsEnabled && !searchDisabled
                    ? "text-primary font-medium"
                    : undefined
                }
                onSelect={() => {
                  const next = !toolsEnabled;
                  setToolsEnabled(next);
                  // Mirror the Search pill: Kimi forbids search + thinking together.
                  if (isKimiExternal) {
                    setReasoningEnabled(!next, { persist: false });
                    applyQwenThinkingParams(!next);
                  }
                }}
              >
                <GlobeIcon />
                Web search
                {toolsEnabled && !searchDisabled ? (
                  <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                    className="ml-auto"
                  />
                ) : null}
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={codeDisabled}
                className={
                  codeToolsEnabled && !codeDisabled
                    ? "text-primary font-medium"
                    : undefined
                }
                onSelect={() => setCodeToolsEnabled(!codeToolsEnabled)}
              >
                {/* Scale, not width: an oversized box pushed the label out of
                    line. */}
                <HugeiconsIcon
                  icon={CodeIcon}
                  strokeWidth={2}
                  className="scale-[1.12]"
                />
                Code
                {codeToolsEnabled && !codeDisabled ? (
                  <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                    className="ml-auto"
                  />
                ) : null}
              </DropdownMenuItem>
              {showImagePill && (
                <DropdownMenuItem
                  disabled={imageDisabled}
                  className={
                    imageToolsEnabled && !imageDisabled
                      ? "text-primary font-medium"
                      : undefined
                  }
                  onSelect={() => setImageToolsEnabled(!imageToolsEnabled)}
                >
                  <HugeiconsIcon icon={Image03Icon} strokeWidth={2} />
                  Images
                  {imageToolsEnabled && !imageDisabled ? (
                    <HugeiconsIcon
                      icon={Tick02Icon}
                      strokeWidth={2}
                      className="ml-auto"
                    />
                  ) : null}
                </DropdownMenuItem>
              )}
              <DropdownMenuSeparator />
              {pinnedPlusItems.map((id) => (
                <Fragment key={id}>{plusMenuNodes[id]}</Fragment>
              ))}
              {overflowPlusItems.length > 0 ? (
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger>
                    <MoreHorizontalIcon className="size-4" />
                    More
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent className="unsloth-plus-menu w-[248px]">
                    {overflowPlusItems.map((id) => (
                      <Fragment key={id}>{plusMenuNodes[id]}</Fragment>
                    ))}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              ) : null}
            </DropdownMenuContent>
          </DropdownMenu>
          {/* Active in compare mode; sits first. Click to exit back to single chat. */}
          <button
            type="button"
            onClick={handleExitCompare}
            className="composer-pill-btn"
            data-active="true"
            data-keep-label="true"
            aria-label="Exit compare chat"
          >
            <PillGlyph>
              <Columns2Icon className="size-[14px]" />
            </PillGlyph>
            <span>Compare</span>
          </button>
          {/* Permission-level pill sits immediately after Compare and ahead
              of every other tool pill (Search, Code, ...) so the Full access
              danger state reads first; only Compare outranks it. */}
          <PermissionModeComposerPill side="top" />
          <button
            type="button"
            disabled={searchDisabled}
            onClick={() => {
              const next = !toolsEnabled;
              setToolsEnabled(next);
              // Kimi's $web_search builtin requires thinking=disabled
              // (https://platform.kimi.ai/docs/guide/use-web-search): toggle
              // the Think pill off when Search is on, mirroring the backend.
              if (isKimiExternal) {
                setReasoningEnabled(!next, { persist: false });
                applyQwenThinkingParams(!next);
              }
            }}
            className="composer-pill-btn"
            data-pill-label="Search"
            data-active={toolsEnabled && !searchDisabled ? "true" : "false"}
            aria-label={
              toolsEnabled ? "Disable web search" : "Enable web search"
            }
          >
            <PillGlyph>
              <GlobeIcon className="size-[15px]" />
            </PillGlyph>
            <span>Search</span>
          </button>
          <button
            type="button"
            disabled={codeDisabled}
            onClick={() => setCodeToolsEnabled(!codeToolsEnabled)}
            className="composer-pill-btn"
            data-pill-label="Code"
            data-active={codeToolsEnabled && !codeDisabled ? "true" : "false"}
            aria-label={
              codeToolsEnabled
                ? "Disable code execution"
                : "Enable code execution"
            }
          >
            <PillGlyph>
              <HugeiconsIcon
                icon={CodeIcon}
                className="size-[18.5px]"
                strokeWidth={2}
              />
            </PillGlyph>
            <span>Code</span>
          </button>
          {showImagePill && (
            <button
              type="button"
              disabled={imageDisabled}
              onClick={() => setImageToolsEnabled(!imageToolsEnabled)}
              className="composer-pill-btn"
              data-pill-label="Images"
              data-active={
                imageToolsEnabled && !imageDisabled ? "true" : "false"
              }
              aria-label={
                imageToolsEnabled
                  ? "Disable image generation"
                  : "Enable image generation"
              }
            >
              <PillGlyph>
                <HugeiconsIcon
                  icon={Image03Icon}
                  className="size-3.5"
                  strokeWidth={2}
                />
              </PillGlyph>
              <span>Images</span>
            </button>
          )}
          {showRagPill && <KnowledgeBaseComposerButton side="top" />}
          {showWebFetchPill && (
            <button
              type="button"
              disabled={webFetchDisabled}
              onClick={() => setWebFetchToolsEnabled(!webFetchToolsEnabled)}
              className="composer-pill-btn"
              data-pill-label="Fetch"
              data-active={
                webFetchToolsEnabled && !webFetchDisabled ? "true" : "false"
              }
              aria-label={
                webFetchToolsEnabled ? "Disable URL fetch" : "Enable URL fetch"
              }
            >
              <PillGlyph>
                <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
              </PillGlyph>
              <span>Fetch</span>
            </button>
          )}
          {artifactsEnabled ? (
            <button
              type="button"
              onClick={() => setArtifactsEnabled(false)}
              className="composer-pill-btn"
              data-pill-label="Canvas"
              data-active="true"
              aria-label="Disable canvas"
            >
              <PillGlyph>
                <HugeiconsIcon
                  icon={PencilRulerIcon}
                  className="size-[15.5px]"
                  strokeWidth={2}
                />
              </PillGlyph>
              <span>Canvas</span>
            </button>
          ) : null}
          {mcpEnabledForChat ? <McpComposerButton side="top" /> : null}
        </div>
        {/* mr-0.5 matches the send button inset from the edge in normal chat;
            gap-1.5 matches its control spacing. */}
        <div className="ml-auto mr-0.5 flex items-center gap-1.5">
          {showReasoningControl ? (
            isEffort || supportsPreserveThinking ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <button
                    type="button"
                    disabled={reasoningDisabled}
                    className="unsloth-thinking-pill"
                    data-pill-label="Thinking settings"
                    data-active={thinkingActiveLook ? "true" : "false"}
                    aria-label={thinkEffortAriaLabel({
                      modelLoaded,
                      reasoningDisabled,
                      reasoningEffort,
                    })}
                  >
                    <BulbIcon className="size-[15.5px]" />
                    {thinkingActiveLook ? (
                      <span className="unsloth-thinking-label">
                        {isEffort
                          ? `Thinking · ${formatReasoningEffortLabel(
                              reasoningEffort,
                              externalSelection?.modelId,
                            )}`
                          : "Thinking"}
                      </span>
                    ) : null}
                    <ArrowDownStandardIcon className="unsloth-thinking-caret size-[15px]" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  side="top"
                  align="end"
                  className={cn(
                    "unsloth-plus-menu",
                    narrowEffortMenu ? "min-w-40" : "min-w-44",
                  )}
                >
                  {isEffort ? (
                    <>
                      {effectiveSupportsReasoningOff && (
                        <DropdownMenuItem
                          onSelect={() => {
                            setReasoningEnabled(false);
                            applyQwenThinkingParams(false);
                            // Preserve thinking needs thinking on, so turn it off too.
                            setPreserveThinking(false);
                          }}
                        >
                          <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                            className={cn(
                              "unsloth-tick size-4",
                              effectiveReasoningVisualEnabled && "opacity-0",
                            )}
                          />
                          {formatReasoningDisabledLabel(
                            effectiveSupportsReasoningOff,
                            isExternalOpenAIReasoning,
                            checkpoint,
                          )}
                        </DropdownMenuItem>
                      )}
                      {effectiveReasoningEffortLevels
                        .filter((level) => level !== "none")
                        .map((level) => (
                          <DropdownMenuItem
                            key={level}
                            onSelect={() => {
                              setReasoningEffort(level);
                              setReasoningEnabled(true);
                              applyQwenThinkingParams(true);
                              // Mutual exclusion: turning thinking on for a
                              // Kimi model forces the web_search builtin off.
                              if (isKimiExternal && toolsEnabled) {
                                setToolsEnabled(false, { persist: false });
                              }
                            }}
                          >
                            <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                              className={cn(
                                "unsloth-tick size-4",
                                !(
                                  effectiveReasoningVisualEnabled &&
                                  reasoningEffort === level
                                ) && "opacity-0",
                              )}
                            />
                            {formatReasoningEffortLabel(
                              level,
                              externalSelection?.modelId,
                            )}
                          </DropdownMenuItem>
                        ))}
                    </>
                  ) : (
                    effectiveSupportsReasoningOff &&
                    !reasoningLockedOn && (
                      <DropdownMenuItem
                        onSelect={() => {
                          const next = !reasoningEnabled;
                          setReasoningEnabled(next);
                          applyQwenThinkingParams(next);
                          // Preserve thinking cannot run without thinking.
                          if (!next) setPreserveThinking(false);
                          if (isKimiExternal && next && toolsEnabled) {
                            setToolsEnabled(false, { persist: false });
                          }
                        }}
                      >
                        <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                          className={cn(
                            "unsloth-tick size-4",
                            !effectiveReasoningEnabled && "opacity-0",
                          )}
                        />
                        Thinking
                      </DropdownMenuItem>
                    )
                  )}
                  {supportsPreserveThinking && (
                    <DropdownMenuItem
                      disabled={!modelLoaded}
                      onSelect={(e) => {
                        e.preventDefault();
                        const next = !preserveThinking;
                        setPreserveThinking(next);
                        // Preserve thinking requires thinking on.
                        if (next) {
                          setReasoningEnabled(true);
                          applyQwenThinkingParams(true);
                        }
                      }}
                    >
                      <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                        className={cn(
                          "unsloth-tick size-4",
                          !preserveThinking && "opacity-0",
                        )}
                      />
                      Preserve thinking
                    </DropdownMenuItem>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <button
                type="button"
                disabled={reasoningDisabled || reasoningLockedOn}
                aria-disabled={reasoningDisabled || reasoningLockedOn}
                title={
                  reasoningLockedOn
                    ? "This model requires reasoning to stay on."
                    : undefined
                }
                onClick={() => {
                  if (reasoningLockedOn) return;
                  const next = !reasoningEnabled;
                  setReasoningEnabled(next);
                  applyQwenThinkingParams(next);
                  // Mutual exclusion: Kimi's $web_search builtin requires
                  // thinking off, so turning thinking on flips Search off.
                  if (isKimiExternal && next && toolsEnabled) {
                    setToolsEnabled(false, { persist: false });
                  }
                }}
                className="unsloth-thinking-pill"
                data-pill-label="Thinking"
                data-active={thinkingActiveLook ? "true" : "false"}
                aria-label={thinkToggleAriaLabel({
                  reasoningLockedOn,
                  modelLoaded,
                  reasoningDisabled,
                  effectiveReasoningEnabled,
                })}
              >
                <PillGlyph>
                  <BulbIcon className="size-[15.5px]" />
                </PillGlyph>
                {thinkingActiveLook ? (
                  <span className="unsloth-thinking-label">Thinking</span>
                ) : null}
              </button>
            )
          ) : null}
          {
            <>
              {!isDictating ? (
                <TooltipIconButton
                  tooltip="Dictate"
                  side="bottom"
                  variant="ghost"
                  size="icon"
                  className="size-8 rounded-full text-muted-foreground"
                  onClick={startDictation}
                  aria-label="Dictate"
                >
                  <MicIcon className="size-4" />
                </TooltipIconButton>
              ) : (
                <TooltipIconButton
                  tooltip={
                    isDictationFinalizing
                      ? "Cancel transcription"
                      : "Stop dictation"
                  }
                  side="bottom"
                  variant="ghost"
                  size="icon"
                  className="size-8 rounded-full text-destructive"
                  onClick={stopDictation}
                  aria-label={
                    isDictationFinalizing
                      ? "Cancel transcription"
                      : "Stop dictation"
                  }
                >
                  <SquareIcon className="size-3 animate-pulse fill-current" />
                </TooltipIconButton>
              )}
            </>
          }
          {isQueueRunning ? (
            <button
              type="button"
              onClick={() => {
                isQueueRunningRef.current = false;
                setIsQueueRunning(false);
                queueRef.current = [];
                queueIndexRef.current = 0;
                setQueueProgress({ current: 0, total: 0 });
                stop();
              }}
              aria-label="Stop prompt queue"
              className="ml-1.5 flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/60 px-2.5 py-1 text-xs font-semibold text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <SquareIcon className="size-2.5 shrink-0 fill-current" />
              <span className="tabular-nums">
                Stop queue {queueProgress.current}/{queueProgress.total}
              </span>
            </button>
          ) : busy ? (
            <Button
              type="button"
              variant="default"
              size="icon"
              className="ml-1.5 size-8 rounded-full"
              onClick={stop}
            >
              <SquareIcon className="size-3 fill-current" />
            </Button>
          ) : (
            <TooltipIconButton
              tooltip="Send message"
              side="bottom"
              variant="default"
              size="icon"
              className="ml-1.5 size-8 rounded-full"
              onClick={send}
              disabled={!canSend}
              aria-label="Send message"
            >
              <ArrowUpIcon className="size-[22px] stroke-2" />
            </TooltipIconButton>
          )}
        </div>
      </div>
    </div>
  );
}
