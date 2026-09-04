// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { mlxRuntimeStateFrom } from "./lib/mlx-runtime-state";
import {
  clearedServerTuningState,
  committedServerTuningState,
  serverTuningLoadPayload,
} from "./lib/server-tuning-fields";
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
import { FIND_SKIP_ATTRIBUTE } from "@/features/find-in-page";
import { DRAFT_N_MAX_SPEC_TYPES } from "@/lib/speculative-modes";
import {
  StudioDictationAdapter,
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
} from "@/features/chat/adapters/studio-dictation-adapter";
import type { StudioDictationSession } from "@/features/chat/adapters/studio-web-speech-dictation-adapter";
import {
  COMPOSER_INPUT_SELECTOR,
  isSurfaceInForeground,
  useShortcut,
} from "@/features/settings";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import {
  AUDIO_PICKER_ACCEPT,
  isAudioAttachmentFile,
  fileToBase64,
  getAudioSizeError,
} from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { classifiedAttachmentFiles, isVideoFile } from "@/lib/video-utils";
import { isDownloadCancelled } from "@/lib/native-files";
import { isMultimodalResponse } from "./types/api";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import { CONVERSATION_MARKDOWN_LABEL } from "./utils/conversation-markdown";
import { pasteClipboardFiles } from "./utils/clipboard-files";
import { confirmStopRunningChatsIfNeeded } from "./utils/confirm-stop-running-chats";
import { requestLocalPromptQueueStop } from "./utils/prompt-queue-boundary";
import {
  cancelPreStreamRunReservations,
  releasePreStreamRunReservation,
  reservePreStreamRun,
} from "./utils/pre-stream-run-reservation";
import type { ModelLifecycleLease } from "./utils/model-lifecycle-gate";
import { useAui } from "@assistant-ui/react";
import {
  ArrowUpIcon,
  ChevronDownIcon,
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
import { useChatActive } from "./runtime-provider";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "@/lib/toast";
import {
  PromptStorageDialog,
  exportConversationShareGPT,
  exportConversationRawJsonl,
  exportConversationMessagesJsonl,
  exportConversationCsv,
  exportConversationMarkdown,
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
  DEFAULT_PER_MODEL_CONFIG,
  isServedByMlx,
  savedContextPin,
  resolveInitialConfig,
  type PerModelConfig,
  loadedContextFields,
} from "@/features/model-picker";
import { loadManagedLlamaFlags } from "@/features/model-picker/api/llama-flags";
import { fetchLoadExtraArgs } from "@/features/model-picker/api/model-overrides";
import { sanitizeStoredExtraArgs } from "@/features/model-picker/model-config/llama-extra-args";
import { usePlatformStore } from "@/config/env";
import {
  confirmTransformersUpgradeIfNeeded,
  useTransformersUpgradeDialogStore,
} from "@/features/transformers-upgrade";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import {
  fetchGgufStagedMetadata,
  loadModel,
  validateModel,
} from "./api/chat-api";
import {
  loadedContextForParams,
  resolveExplicitCtxPin,
  resolveFitMaxSeqLength,
  retainedContextPin,
  unpinnedLoadContext,
  replayMaxTokensCap,
} from "./presets/preset-policy";
import { ensureGpuDeviceCache } from "@/hooks/use-gpu-info";
import {
  parseExternalModelId,
  providerModelSupportsVision,

  providerModelSupportsStudioTools,
} from "./external-providers";
import { compareModelDisplayName } from "./lib/external-model-label";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import { useComposerPillFit } from "@/hooks/use-composer-pill-fit";
import { useIsMobile } from "@/hooks/use-mobile";
import {
  PLUS_MENU_ORDER,
  type PlusMenuItemId,
  usePlusMenuPrefsStore,
} from "./stores/plus-menu-prefs-store";
import {
  resolveComparePlacement,
  shouldPinDiffusionPlacement,
} from "./lib/gpu-placement";
import {
  loadedGpuMemoryFields,
  type ReasoningEffort,
  reconcilePersistedGpuIds,
  resolveLoadedSpeculativeSettings,
  resolvePreserveThinkingOnLoad,
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
  type ClipboardEvent,
  type DragEvent as ReactDragEvent,
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
  | { type: "audio"; audio: string; name: string };

export interface CompareHandle {
  append: (content: CompareMessagePart[]) => void;
  /** Append a user message without triggering generation. */
  appendMessage: (content: CompareMessagePart[]) => void;
  /** Trigger generation on the current thread (after appendMessage). */
  startRun: () => void;
  cancel: () => void;
  isRunning: () => boolean;
  threadIds: () => string[];
  /** Returns a promise that resolves when the current or next run finishes. */
  waitForRunEnd: () => Promise<void>;
}

const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp,image/gif";
const MAX_IMAGE_SIZE = 20 * 1024 * 1024;

// Inlined to avoid a new icon dep. Kept in sync with the main composer.
function isNativeComposing(event: Event) {
  return "isComposing" in event && (event as InputEvent).isComposing === true;
}

// Mirrors the threshold in thread.tsx. Chrome on Windows-over-WSL (#5546) never fires
// `compositionend` after IME commit, so the compose flag would otherwise stay true forever.
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
  // Magistral keeps the "none" wire value, but UX presents this floor as "Medium" rather than a disabled-state label.
  if (normalized.includes("magistral-medium-latest")) return "Medium";
  return supportsReasoningOff && isExternalOpenAIReasoning ? "None" : "Off";
}

function useDictation(
  setText: (value: string | ((prev: string) => string)) => void,
) {
  // Re-render support state when the user switches recognition engines.
  const dictationEngine = useVoiceSettingsStore((s) => s.dictationEngine);
  const [isDictating, setIsDictating] = useState(false);
  // True while a stopped recording's final audio is still transcribing; a second click then cancels
  // the pending transcription instead of re-stopping.
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
      // Routes to the engine chosen in Voice settings, honoring the selected microphone, language and
      // dictionary. Compare feeds two panes, so recent dictations must not link the unrelated
      // single-chat active thread.
      session = new StudioDictationAdapter({ chatId: null }).listen();
    } catch {
      startingRef.current = false;
      notifyStudioDictationUnavailable();
      return;
    }
    sessionRef.current = session;
    setIsDictating(true);

    // Append final transcripts; the adapter has already applied the dictionary and recorded the session.
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
    // A second click while the final segment is transcribing discards the pending transcription
    // instead of leaving the pane stuck until timeout.
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
    // Keep the session and dictation state alive while its final audio segment is transcribed; onEnd
    // clears both after the transcript callbacks run.
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
    const getThreadIds = () => {
      const itemState = aui.threadListItem().getState();
      return Array.from(
        new Set(
          [itemState.id, itemState.remoteId].filter(
            (id): id is string => Boolean(id),
          ),
        ),
      );
    };
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
      threadIds: getThreadIds,
      waitForRunEnd: () =>
        new Promise<void>((resolve, reject) => {
          const runtime =
            aui.threads().__internal_getAssistantRuntime?.();
          const threadIds = getThreadIds();
          let thread = null;
          for (const threadId of threadIds) {
            try {
              thread = runtime?.threads.getById(threadId) ?? null;
              if (thread) break;
            } catch {
              // Thread hydration can retire an alias; try the next one.
            }
          }
          if (!thread) {
            reject(new Error("Comparison thread is unavailable"));
            return;
          }
          let wasRunning = thread.getState().isRunning;
          let unsubscribe = () => {};
          unsubscribe = thread.subscribe(() => {
            const isRunning = thread.getState().isRunning;
            if (isRunning) wasRunning = true;
            if (wasRunning && !isRunning) {
              unsubscribe();
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
    <div
      data-reload-snapshot-sensitive
      className="relative size-14 shrink-0 overflow-hidden rounded-[14px] border border-foreground/20 bg-muted"
    >
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
  isDiffusion?: boolean;
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
  return speculativeType != null && DRAFT_N_MAX_SPEC_TYPES.has(speculativeType)
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

/** True when a drop reached this composer from a portaled child, such as a dialog and its
 *  overlay, which React routes through here but which owns it. */
function isPortaledDrop(event: ReactDragEvent): boolean {
  const target = event.target as Element | null;
  return !target?.closest?.(".chat-composer-surface");
}

export function SharedComposer({
  handlesRef,
  model1,
  model2,
  onExitCompare,
  model1ThreadId,
  model2ThreadId,
  sendUnavailableReason,
  requireStableCheckpoint = false,
}: {
  handlesRef: CompareHandles;
  model1?: CompareModelSelection;
  model2?: CompareModelSelection;
  onExitCompare?: () => void;
  model1ThreadId?: string;
  model2ThreadId?: string;
  sendUnavailableReason?: string;
  requireStableCheckpoint?: boolean;
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
    contentType: string;
  } | null>(null);
  const textRef = useRef(text);
  const pendingImagesRef = useRef(pendingImages);
  const pendingAudioRef = useRef(pendingAudio);
  useEffect(() => {
    textRef.current = text;
    pendingImagesRef.current = pendingImages;
    pendingAudioRef.current = pendingAudio;
  }, [text, pendingImages, pendingAudio]);
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
  const loadedVisionDisabledByUser = useChatRuntimeStore(
    (s) => s.loadedVisionDisabledByUser,
  );
  const mmprojFallbackReason = useChatRuntimeStore(
    (s) => s.mmprojFallbackReason,
  );
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
    externalSupportsVision: providerModelSupportsVision(
      selectedExternalProvider?.providerType,
      externalSelection?.modelId,
    ),
    externalModelLabel: externalSelection?.modelId ?? null,
    loadedIsMultimodal,
    modelLoaded,
    loadError: lastModelLoadError,
    visionDisabledByUser: loadedVisionDisabledByUser,
    mmprojFallbackReason,
  });
  const isCompareMode = Boolean(model1?.id || model2?.id);
  // Attach-time gate. Compare mode defers to send: the catalog can lag a model's real capabilities,
  // and models[] only syncs after ensureModelLoaded at send time. Single mode uses the loaded
  // model's runtime capability.
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
  // Kimi's $web_search builtin mandates thinking=disabled. Both pills stay clickable, but turning
  // one on flips the other off, so the visible state matches what the backend sends.
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";
  const effectiveReasoningEnabled = reasoningLockedOn ? true : reasoningEnabled;
  const effectiveReasoningVisualEnabled =
    effectiveReasoningEnabled && reasoningEffort !== "none";
  const reasoningDisabled = !modelLoaded || !effectiveSupportsReasoning;
  const showReasoningControl =
    effectiveSupportsReasoning || effectiveReasoningAlwaysOn;
  // enable_thinking_effort (GLM-5.2: high|max + disable) reuses the effort dropdown; it just also
  // carries an Off row via supportsReasoningOff.
  const isEffort =
    effectiveReasoningStyle === "reasoning_effort" ||
    effectiveReasoningStyle === "enable_thinking_effort";
  // GLM-5.2's effort menu has short rows, so it can sit skinnier. Skip the narrower floor when a
  // Preserve thinking row is present, since that label needs the wider width.
  const narrowEffortMenu =
    effectiveReasoningStyle === "enable_thinking_effort" &&
    !supportsPreserveThinking;
  const thinkingActiveLook = isEffort
    ? reasoningLockedOn || (effectiveReasoningVisualEnabled && !reasoningDisabled)
    : reasoningLockedOn || (effectiveReasoningEnabled && !reasoningDisabled);
  // Two-pill gating: Search lights up on a local tool runtime OR a provider-run server-side
  // web_search; Code on the local runtime OR Anthropic with a model accepting
  // code_execution_20250825, the only external code-execution tool today.
  // Search: supportsTools (Code/python plus local web_search) OR supportsBuiltinWebSearch
  // (OpenAI/Anthropic/OpenRouter/Kimi). Code: the local runtime OR Anthropic with a model taking
  // code_execution_20250825, per providerSupportsBuiltinCodeExecution. Anthropic is the only
  // external provider shipping a code-execution tool today.
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
  // Gemini rejects codeExecution alongside image modalities. Search is blocked on older Gemini
  // image ids but allowed on Gemini 3 image models, so only Code is disabled unconditionally
  // in Gemini image mode.
  const isExternalGemini = selectedExternalProvider?.providerType === "gemini";
  const imageDisabled = !modelLoaded || !supportsBuiltinImageGeneration;
  const imageModeDisablesCode =
    isExternalGemini && imageToolsEnabled && !imageDisabled;
  // Image-tier Gemini models always reject codeExecution and reject web_search on older ids, so do
  // not let local `supportsTools` re-enable a pill the Gemini backend silently drops: gate
  // strictly on provider builtin support.
  const isGeminiImageTier =
    isExternalGemini && supportsBuiltinImageGeneration;
  // Disable only when a loaded model lacks the capability; with no model the tool can still be
  // pre-selected, matching the + menu.
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
  // Images pill lights only on OpenAI cloud Responses-API models and the Gemini Nano Banana
  // family. No local tool runtime fallback.
  const showImagePill = supportsBuiltinImageGeneration;
  // Fetch pill: Anthropic-only (web_fetch_20250910 / web_fetch_20260209).
  const webFetchDisabled = !modelLoaded || !supportsBuiltinWebFetch;
  const showWebFetchPill = supportsBuiltinWebFetch;
  const externalUsesStudioTools =
    providerModelSupportsStudioTools(
      selectedExternalProvider?.providerType,
      externalSelection?.modelId,
    ) === true;
  const ragDisabled =
    modelLoaded && ((!externalUsesStudioTools && isExternalModel) || !supportsTools);
  const showRagPill = !isExternalModel || externalUsesStudioTools;
  // Above 4 pills, collapse to icons only. Compare, Search, Code and permissions always show.
  // Narrow viewports collapse too: the labelled row is wider than a phone-width composer.
  const isMobile = useIsMobile();
  const pillCount =
    4 +
    (showImagePill ? 1 : 0) +
    (showRagPill && ragEnabled ? 1 : 0) +
    (showWebFetchPill ? 1 : 0) +
    (artifactsEnabled ? 1 : 0) +
    (mcpEnabledForChat ? 1 : 0);
  // Under the count threshold the row still overflows on long labels, wrapping onto a second line
  // inside the action bar, so measuring collapses just enough to keep it beside send.
  const { pillRowRef, pillCompact } = useComposerPillFit(
    isMobile || pillCount > 4,
  );
  // Backwards-compatible alias for call sites still referencing `toolsDisabled`.
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

  function resetPromptQueue() {
    if (!isQueueRunningRef.current && queueRef.current.length === 0) {
      return;
    }
    isQueueRunningRef.current = false;
    setIsQueueRunning(false);
    queueRef.current = [];
    queueIndexRef.current = 0;
    setQueueProgress({ current: 0, total: 0 });
  }

  function advanceQueue() {
    const nextIndex = queueIndexRef.current + 1;
    if (nextIndex >= queueRef.current.length) {
      resetPromptQueue();
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

  // Compare mode: advance the queue on cycle end, but stop on a failed step so prompts are not
  // burned on incomplete results.
  useEffect(() => {
    const wasComparing = prevComparingRef.current;
    prevComparingRef.current = comparing;
    if (!isQueueRunningRef.current || !wasComparing || comparing) return;
    if (!compareStepSucceededRef.current) {
      resetPromptQueue();
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
    async (input: FileList | readonly File[] | null) => {
      if (!input?.length) return;
      // Compare takes audio, so an audio-only 3GP must not be read off its
      // extension as a clip and refused with the video message.
      const files = await classifiedAttachmentFiles(input);
      const next: PendingImage[] = [];
      let droppedImageForUnavailable = false;
      let audioSizeError: string | null = null;
      let videoUnsupported = false;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file) continue;
        // Handle audio files
        if (isAudioAttachmentFile(file)) {
          const sizeError = getAudioSizeError(file.size);
          if (sizeError) {
            audioSizeError ??= sizeError;
            continue;
          }
          fileToBase64(file).then((base64) => {
            setPendingAudio({ name: file.name, base64, contentType: file.type });
            setPendingAudioStore(base64, file.name);
          });
          continue;
        }
        // video_base64 targets the single loaded GGUF, so at most one side of a compare could answer. Say
        // that rather than drop the file.
        if (isVideoFile(file)) {
          videoUnsupported = true;
          continue;
        }
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
      if (audioSizeError) {
        toast.error(audioSizeError);
      }
      if (videoUnsupported) {
        toast.error("Video can't be attached in compare mode", {
          description: "Open a single chat with a video-capable model instead.",
        });
      }
      setPendingImages((prev) => [...prev, ...next]);
    },
    [setPendingAudioStore, attachUnavailableReason],
  );

  const handleFilePaste = useCallback(
    (event: ClipboardEvent<HTMLTextAreaElement>) => {
      pasteClipboardFiles(
        event,
        async (pasted) => {
          // Classify before the check, so a pasted audio-only 3GP is not read
          // as unsupported on its extension alone.
          const files = await classifiedAttachmentFiles(pasted);
          // Let addFiles report audio size errors.
          const supported = files.some(
            (file) =>
              isAudioAttachmentFile(file) ||
              (file.type.match(/^image\/(jpeg|png|webp|gif)$/i) &&
                file.size <= MAX_IMAGE_SIZE),
          );
          if (!supported) throw new Error("Unsupported compare attachment");
          await addFiles(files);
        },
        () =>
          toast.error("Could not paste files.", {
            description: "Compare supports images and audio within the attachment size limits.",
          }),
      );
    },
    [addFiles],
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
    if (composingRef.current) {
      resetPromptQueue();
      return;
    }
    const submittedText = text;
    const submittedImages = pendingImages;
    const submittedAudio = pendingAudio;
    const msg = submittedText.trim();
    if (!msg && submittedImages.length === 0 && !submittedAudio) {
      resetPromptQueue();
      return;
    }
    if (sendUnavailableReason) {
      resetPromptQueue();
      toast.error("Compare unavailable", {
        description: sendUnavailableReason,
      });
      return;
    }

    const hasCompareHandles = Boolean(
      handlesRef.current["model1"] || handlesRef.current["model2"],
    );
    const isGeneralizedCompare =
      hasCompareHandles && Boolean(model1?.id && model2?.id);
    const submittedCompareCheckpoint = requireStableCheckpoint
      ? useChatRuntimeStore.getState().params.checkpoint
      : undefined;

    // Generalized compare requires both panes to have a model: a half-selected send either races to
    // an empty bubble with bogus tok/s (#5569) or leaves the empty pane with a dangling prompt.
    // hasCompareHandles is true only in GeneralCompareContent.
    if (hasCompareHandles && !isGeneralizedCompare) {
      toast.error("Pick a model in each pane to compare", {
        description:
          "Use the model dropdown above each pane, then send your prompt.",
      });
      resetPromptQueue();
      return;
    }

    if (
      submittedImages.length > 0 &&
      !isGeneralizedCompare &&
      imageUnavailableReason
    ) {
      // Single mode: the loaded model's runtime capability is known here. Compare mode defers: each
      // ensureModelLoaded sets loadedIsMultimodal for its side, and the adapter's pre-stream gate
      // runs per side.
      toast.error(imageUnavailableReason);
      resetPromptQueue();
      return;
    }

    const content: CompareMessagePart[] = [];
    for (const { file } of submittedImages) {
      try {
        const image = await fileToBase64DataURL(file);
        content.push({ type: "image", image });
      } catch {
        // skip failed image
      }
    }
    if (submittedAudio) {
      content.push({
        type: "audio",
        name: submittedAudio.name,
        audio: `data:${submittedAudio.contentType};base64,${submittedAudio.base64}`,
      });
    }
    if (msg) {
      content.push({ type: "text", text: msg });
    }
    if (content.length === 0) {
      resetPromptQueue();
      return;
    }

    let compareLifecycleLease: ModelLifecycleLease | null = null;
    if (isGeneralizedCompare) {
      compareLifecycleLease = useChatRuntimeStore
        .getState()
        .beginModelLoading();
      if (compareLifecycleLease === null) {
        toast.info("A model is loading", {
          description: "Wait for it to finish or cancel it first.",
        });
        resetPromptQueue();
        return;
      }
    }
    const releaseCompareModelLifecycle = () => {
      if (compareLifecycleLease === null) {
        return;
      }
      useChatRuntimeStore.getState().endModelLoading(compareLifecycleLease);
      compareLifecycleLease = null;
    };
    const acquireCompareModelLifecycle = () => {
      if (compareLifecycleLease !== null) {
        return;
      }
      compareLifecycleLease = useChatRuntimeStore
        .getState()
        .beginModelLoading();
      if (compareLifecycleLease === null) {
        throw new Error("Another model load started during comparison");
      }
    };
    const submittedDraftIsCurrent = () =>
      textRef.current === submittedText &&
      pendingImagesRef.current === submittedImages &&
      pendingAudioRef.current === submittedAudio;
    const keepChangedDraft = () => {
      releaseCompareModelLifecycle();
      resetPromptQueue();
      toast.info("Message changed while preparing", {
        description: "Your updated draft was kept. Send it again when ready.",
      });
    };
    const clearSubmittedDraft = () => {
      setText("");
      setPendingImages([]);
      setPendingAudio(null);
      clearPendingAudioStore();
      textareaRef.current?.focus();
    };

    let compareStopDecision: Awaited<
      ReturnType<typeof confirmStopRunningChatsIfNeeded>
    > | null = null;
    if (isGeneralizedCompare) {
      try {
        compareStopDecision = await confirmStopRunningChatsIfNeeded(
          "Loading models for comparison",
          "reload",
        );
      } catch (error) {
        releaseCompareModelLifecycle();
        resetPromptQueue();
        toast.error("Compare failed", {
          description: error instanceof Error ? error.message : "Unknown error",
        });
        return;
      }
      if (!compareStopDecision.proceed) {
        releaseCompareModelLifecycle();
        resetPromptQueue();
        return;
      }
    }
    if (!submittedDraftIsCurrent()) {
      keepChangedDraft();
      return;
    }

    // Generalized compare: load each model before dispatching to its side
    if (isGeneralizedCompare) {
      const store = useChatRuntimeStore.getState();
      const trustRemoteCode = store.params.trustRemoteCode ?? false;
      const fallbackTensorParallel = store.tensorParallel;
      const specSettings = resolveSpeculativeSettingsForLoad({
        usePersistedPreference: true,
      });
      let loadedFromConfig = false;

      // Warm the device cache before the snapshot below reconciles the GPU pick: on a cold cache the
      // reconcile passes a stale pick through.
      try {
        if (store.selectedGpuIds != null) {
          await ensureGpuDeviceCache();
        }
      } catch (error) {
        releaseCompareModelLifecycle();
        resetPromptQueue();
        toast.error("Compare failed", {
          description: error instanceof Error ? error.message : "Unknown error",
        });
        return;
      }
      // The GPU/offload knobs both compare loads must use, snapshotted at Send: ensureModelLoaded runs
      // sequentially and the first load's response echo rewrites the live store, so reading the
      // store per load would hand model 2 the first model's echoed defaults.
      // The first load's echo (loadedGpuMemoryFields) rewrites the live store, resetting
      // gpuLayers/nCpuMoe/split/pick to defaults on a non-GGUF or Auto first model.
      const compareLoadKnobs = {
        gpuMemoryMode: store.gpuMemoryMode,
        gpuLayers: store.gpuLayers,
        nCpuMoe: store.nCpuMoe,
        splitRatio: store.splitRatio,
        selectedGpuIds: store.selectedGpuIds,
        selectedGpuIndexKind: store.selectedGpuIndexKind,
      };
      if (!submittedDraftIsCurrent()) {
        keepChangedDraft();
        return;
      }
      clearSubmittedDraft();
      // Set when an accepted transformers install unloaded the active model server-side; a later
      // failure must then clear the stale checkpoint.
      let upgradeUnloadedActive = false;
      const compareSelectionNeedsLoad = (sel: CompareModelSelection) => {
        const currentStore = useChatRuntimeStore.getState();
        const isAlreadyActive =
          currentStore.params.checkpoint === sel.id &&
          (currentStore.activeGgufVariant ?? null) ===
            (sel.ggufVariant ?? null);
        return !isAlreadyActive || sel.config != null || loadedFromConfig;
      };
      const applyCompareStopDecision = () => {
        cancelPreStreamRunReservations(
          compareStopDecision?.preStreamRunTokens ?? [],
        );
        requestLocalPromptQueueStop(compareStopDecision?.promptQueueThreadIds);
      };
      async function ensureModelLoaded(
        sel: CompareModelSelection,
      ): Promise<string> {
        const currentStore = useChatRuntimeStore.getState();
        const config = sel.config ?? null;
        // This pane's effective config: an explicit selection config, else the remembered store config
        // for this model/quant, never the other pane's. No saved config means all-null defaults.
        const resolved = config
          ? { config, remembered: true }
          : resolveInitialConfig(sel.id, sel.ggufVariant ?? null);
        const ownConfig = resolved.config;
        const ownRemembered = resolved.remembered;
        const isAlreadyActive =
          currentStore.params.checkpoint === sel.id &&
          (currentStore.activeGgufVariant ?? null) ===
            (sel.ggufVariant ?? null);
        if (isAlreadyActive && !config && !loadedFromConfig) {
          applyCompareStopDecision();
          return "ready";
        }
        const targetIsGguf =
          (sel.ggufVariant ?? null) != null ||
          sel.id.toLowerCase().endsWith(".gguf");
        const platform = usePlatformStore.getState();
        let resolvedIsDiffusion = sel.isDiffusion;
        // Set when the preflight could not classify the GGUF, so a false resolvedIsDiffusion below must
        // not be read as "ordinary".
        let diffusionUnknown = false;
        if (targetIsGguf && resolvedIsDiffusion === undefined) {
          const preparedToken = await prepareHfTokenForUse(
            currentStore.hfToken,
          );
          if (!preparedToken.proceed) {
            throw new Error("Model load cancelled.");
          }
          const staged = await fetchGgufStagedMetadata({
            model_path: sel.id,
            gguf_variant: sel.ggufVariant ?? null,
            hf_token: preparedToken.token,
          });
          resolvedIsDiffusion = staged.isDiffusion;
          diffusionUnknown = staged.diffusionUnknown;
        }
        // Pass-through arguments can live only in the server's override map while this config comes from
        // local storage, and /load's omission path inherits them from a RESIDENT instance, which a
        // cold compare pane does not have, so the experiment would run a different command.
        if (
          targetIsGguf &&
          // Not for the diffusion runner, which appends none of them.
          resolvedIsDiffusion !== true
        ) {
          try {
            // Sanitised for the same reason the panel sanitises what it hydrates: either list becomes an
            // EXPLICIT /load argument, validated strictly rather than going through the carry-over paths
            // that drop a newly denied flag quietly.
            // A pane on an install upgraded across a denylist change would otherwise answer 400 on a
            // comparison that ran the day before.
            const managed = await loadManagedLlamaFlags();
            const clean = (tokens: readonly string[]) =>
              sanitizeStoredExtraArgs(
                tokens,
                managed?.managed ?? new Set<string>(),
                {
                  maxBytes: managed?.maxBytes,
                  windowsCommandBudget: managed?.windowsCommandBudget,
                },
              );
            const local = ownConfig.llamaExtraArgs;
            if (local === undefined) {
              const resolvedArgs = await fetchLoadExtraArgs(
                sel.id,
                sel.id,
                sel.ggufVariant ?? null,
              );
              const cleaned = clean(resolvedArgs.tokens);
              if (cleaned.length > 0) {
                ownConfig.llamaExtraArgs = cleaned;
              } else if (resolvedArgs.explicit) {
                // An explicit empty row is a cleared box, and this pane has to send it as one: left undefined the
                // field is omitted and /load carries the resident model's arguments into the comparison.
                ownConfig.llamaExtraArgs = [];
              }
            } else if (local !== null && local.length > 0) {
              const cleaned = clean(local);
              if (cleaned.length !== local.length) {
                ownConfig.llamaExtraArgs = cleaned.length > 0 ? cleaned : [];
              }
            }
          } catch {
            // The load still works; a real overrides outage surfaces there.
          }
        }
        // Mirror single-view resolveLoadMaxSeqLength: a pane with no explicit context
        // hands sizing to whichever local backend serves it, not the session
        // maxSeqLength, which would silently shrink the shown context. One no local
        // backend serves falls back to the app default rather than the active model's
        // runtime snapshot, else comparing a saved 128K model against an unconfigured
        // one loads the latter at 128K and OOMs.
        const effectiveMaxSeqLength =
          savedContextPin(ownConfig) ??
          unpinnedLoadContext(
            targetIsGguf,
            isServedByMlx(targetIsGguf, platform.deviceType, platform.chatOnlyReason),
            DEFAULT_MAX_SEQ_LENGTH,
          );
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
        const effectiveTensorParallel = resolvedIsDiffusion
          ? false
          : ownRemembered
            ? ownConfig.tensorParallel
            : fallbackTensorParallel;
        // The diffusion runner has no projector to skip, so the toggle is inert there. A pane with no
        // saved config gets the per-model DEFAULT, not the store's current value: unlike
        // tensorParallel, which stands across models, an unconfigured model is one with vision on.
        const effectiveDisableVision = resolvedIsDiffusion
          ? false
          : ownRemembered
            ? ownConfig.disableVision
            : DEFAULT_PER_MODEL_CONFIG.disableVision;
        if (ownConfig.selectedGpuIds != null) {
          await ensureGpuDeviceCache();
        }
        // A pane's OWN saved split is sent instead of being forced to Auto (#7574); the shared Send-time
        // snapshot is not, since its layer count is bounded by another GGUF. Knobs the runner has no
        // equivalent for stay hard-forced, and an UNCLASSIFIED GGUF is pinned too.
        const {
          gpuMemoryMode: effectiveGpuMemoryMode,
          gpuLayers: effectiveGpuLayers,
        } = resolveComparePlacement(
          ownConfig,
          compareLoadKnobs,
          shouldPinDiffusionPlacement(
            targetIsGguf,
            resolvedIsDiffusion,
            diffusionUnknown,
          ),
        );
        const effectiveNCpuMoe =
          resolvedIsDiffusion
            ? 0
            : (ownConfig.nCpuMoe ?? compareLoadKnobs.nCpuMoe);
        const effectiveSelectedGpuIds =
          ownConfig.selectedGpuIds !== undefined
            ? reconcilePersistedGpuIds(
                ownConfig.selectedGpuIds,
                ownConfig.selectedGpuIndexKind,
                resolvedIsDiffusion === true,
              )
            : reconcilePersistedGpuIds(
                compareLoadKnobs.selectedGpuIds,
                compareLoadKnobs.selectedGpuIndexKind,
                resolvedIsDiffusion === true,
              );
        // A pane's context comes from its own config only: a saved pin, or null. It must not inherit the
        // active model's shared snapshot, which resolveFitMaxSeqLength would treat as a pin and load
        // this pane at the other model's context.
        // A GGUF pane with no explicit context loads at native (0 -> n_ctx_train), not the session
        // maxSeqLength, which would silently shrink the shown context.
        const effectiveCustomContextLength = ownConfig.customContextLength;
        let loadTrustRemoteCode = trustRemoteCode;
        let approvedRemoteCodeFingerprint: string | null = null;
        // Size validation exactly as the load below, so the training-guard preflight checks the footprint
        // that actually loads.
        const compareMaxSeqLength = resolveFitMaxSeqLength(
          targetIsGguf,
          effectiveGpuMemoryMode,
          effectiveGpuLayers,
          // Prefer this pane's own saved context pin over the shared snapshot, falling back to its per-pane
          // effective context.
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
          cache_type_kv: ownConfig.kvCacheDtype ?? null,
          tensor_parallel: effectiveTensorParallel,
          disable_vision: effectiveDisableVision,
          // Scope the validate to the picked GPUs. GGUF-only, like the load below: a non-GGUF target must
          // not inherit a hidden GGUF GPU pick.
          ...(targetIsGguf
            ? {
                gpu_ids: effectiveSelectedGpuIds ?? undefined,
                gpu_memory_mode: effectiveGpuMemoryMode,
                // Sized like the load below: a manual DiffusionGemma split must not be validated as a full-GGUF
                // occupant.
                gpu_layers: effectiveGpuLayers,
                // Slots scale the KV estimate; keep validate sized like the load.
                n_parallel: ownConfig.nParallel ?? null,
                // Only when this panel has read the stored value: omitted, the load inherits it, which is what
                // keeps CLI-set flags working.
                ...(ownConfig.llamaExtraArgs !== undefined
                  ? // biome-ignore lint/style/useNamingConvention: API schema
                    { llama_extra_args: ownConfig.llamaExtraArgs ?? [] }
                  : {}),
                // omitted when blank: a null counts as set and strips inherited -b / -ub
                ...(ownConfig.nBatch != null
                  ? { n_batch: ownConfig.nBatch }
                  : {}),
                ...(ownConfig.nUbatch != null
                  ? { n_ubatch: ownConfig.nUbatch }
                  : {}),
                ...serverTuningLoadPayload(ownConfig),
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
            forceCancelActive:
              compareStopDecision?.forceCancelActive ?? false,
          });
          // The install unloads the active model before the swap even when the swap fails, so if a later
          // gate cancels the UI must stop pointing at that unloaded model.
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
              `${compareModelDisplayName(sel.id)} needs a newer transformers release to load.`,
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
              `${compareModelDisplayName(sel.id)} needs custom code approval to load.`,
            );
          }
        }
        applyCompareStopDecision();
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
          mlx_kv_bits: ownConfig.mlxKvBits ?? null,
          speculative_type: effectiveSpeculativeType,
          spec_draft_n_max: effectiveSpecDraftNMax,
          tensor_parallel: effectiveTensorParallel,
          disable_vision: effectiveDisableVision,
          force_cancel_active:
            compareStopDecision?.forceCancelActive ?? false,
          ...(targetIsGguf
            ? {
                gpu_memory_mode: effectiveGpuMemoryMode,
                gpu_layers: effectiveGpuLayers,
                n_cpu_moe: effectiveNCpuMoe,
                tensor_split: compareLoadKnobs.splitRatio ?? undefined,
                gpu_ids: effectiveSelectedGpuIds ?? undefined,
                n_parallel: ownConfig.nParallel ?? null,
                // Only when this panel has read the stored value: omitted, the load inherits it, which keeps
                // CLI-set flags working.
                ...(ownConfig.llamaExtraArgs !== undefined
                  ? // biome-ignore lint/style/useNamingConvention: API schema
                    { llama_extra_args: ownConfig.llamaExtraArgs ?? [] }
                  : {}),
                ...(ownConfig.nBatch != null
                  ? { n_batch: ownConfig.nBatch }
                  : {}),
                ...(ownConfig.nUbatch != null
                  ? { n_ubatch: ownConfig.nUbatch }
                  : {}),
                ...serverTuningLoadPayload(ownConfig),
              }
            : {}),
        });
        // Keep a compare pane's per-model speculative choice load-local: persist the global preference
        // only when it came from global settings.
        if (ownConfig.speculativeType == null) {
          saveSpeculativeType(effectiveSpeculativeType);
        }
        // Persist the GPU Memory mode on a non-diffusion GGUF compare-load too, so an applied manual
        // choice survives a restart.
        persistGpuMemoryModeOnLoad(resp, effectiveGpuMemoryMode);
        upgradeUnloadedActive = false;
        const store = useChatRuntimeStore.getState();
        store.setCheckpoint(
          resp.model,
          resp.is_gguf ? (sel.ggufVariant ?? undefined) : null,
          // Same cap as the interactive load: this replays the model's remembered settings, and a budget
          // kept from a larger context does not fit the one it just loaded with.
          {
            // The reported window leads and the request stands in only for a backend
            // that sizes nothing, as on the interactive load: an unpinned pane sends
            // the auto-size sentinel, and capping a budget at 0 asks for no output.
            maxTokensCap: replayMaxTokensCap(
              loadedContextFields(resp).loadedContextLength ??
                (!resp.is_gguf && effectiveMaxSeqLength > 0
                  ? effectiveMaxSeqLength
                  : null),
            ),
          },
        );
        store.setModelRequiresTrustRemoteCode(
          resp.requires_trust_remote_code ?? false,
        );
        // This pane's own saved Context Length, not compareMaxSeqLength: the wire
        // value is Auto-resolved for a same-model reload, so pinning it would
        // convert Auto into a number the user never set (see resolveExplicitCtxPin).
        // A non-GGUF pane keeps an MLX pin instead of clearing its baseline.
        const keepCustomCtx = targetIsGguf
          ? resolveExplicitCtxPin(effectiveCustomContextLength)
          : retainedContextPin({
              isMlx: isServedByMlx(
                targetIsGguf,
                platform.deviceType,
                platform.chatOnlyReason,
              ),
              requestedContextLength: compareMaxSeqLength,
            });
        // Slots this compare load committed. Diffusion ignores --parallel, so a
        // count there would mint a phantom override a preset carries onto a GGUF.
        const committedSlots =
          targetIsGguf && !(resp.is_diffusion ?? false)
            ? (ownConfig.nParallel ?? null)
            : null;
        // same rule for the batch sizes
        const committedNBatch =
          targetIsGguf && !(resp.is_diffusion ?? false)
            ? (ownConfig.nBatch ?? null)
            : null;
        const committedNUbatch =
          targetIsGguf && !(resp.is_diffusion ?? false)
            ? (ownConfig.nUbatch ?? null)
            : null;
        useChatRuntimeStore.setState({
          supportsReasoning: resp.supports_reasoning ?? false,
          reasoningAlwaysOn: resp.reasoning_always_on ?? false,
          ...reasoningCapsFromLoad(resp),
          supportsPreserveThinking: resp.supports_preserve_thinking ?? false,
          preserveThinking: resolvePreserveThinkingOnLoad(resp),
          supportsTools: resp.supports_tools ?? false,
          kvCacheDtype: resp.cache_type_kv ?? null,
          loadedKvCacheDtype: resp.cache_type_kv ?? null,
          ...mlxRuntimeStateFrom(resp),
          // Click-time value, not the resolved echo (see the single-model load).
          nParallel: committedSlots,
          loadedNParallel: committedSlots,
          nBatch: committedNBatch,
          loadedNBatch: committedNBatch,
          nUbatch: committedNUbatch,
          loadedNUbatch: committedNUbatch,
          ...(targetIsGguf && !(resp.is_diffusion ?? false)
            ? committedServerTuningState(ownConfig)
            : clearedServerTuningState()),
          // What this pane's launch is running, for a later rollback: the status applier is held off for
          // the whole load, so a switch straight after would snapshot the other model's list.
          loadedLlamaExtraArgs:
            resp.requested_llama_extra_args !== undefined
              ? (resp.requested_llama_extra_args ?? [])
              : (ownConfig.llamaExtraArgs ?? null),
          tensorParallel: resp.tensor_parallel ?? false,
          loadedTensorParallel: resp.tensor_parallel ?? false,
          loadedDisableVision: resp.disable_vision ?? false,
          // Adopted from the echo like the knob above: this pane loaded its own model, so the editable
          // value must follow it or Advanced Settings shows the other pane's Vision state.
          disableVision: resp.disable_vision ?? false,
          defaultChatTemplate: resp.chat_template ?? null,
          chatTemplateOverride: effectiveChatTemplateOverride,
          loadedChatTemplateOverride: effectiveChatTemplateOverride,
          // The context baseline this pane loaded with (see keepCustomCtx above), so a
          // later Apply/Reset can't silently revert the pin it was serving.
          loadedCustomContextLength: keepCustomCtx,
          // Adopt the load response's GPU-memory fields (mode/layers/MoE/split/pick
          // plus loaded baselines) so the GPU controls round-trip. (The context group,
          // customContextLength and native-path token/expiry clear in the tail below.)
          ...loadedGpuMemoryFields(resp),
          // Drives the GPU Memory controls' diffusion gate; set alongside the GPU fields on every load path
          // so the gate cannot read stale.
          loadedIsDiffusion: resp.is_diffusion ?? false,
          loadedIsMultimodal: isMultimodalResponse(resp),
          // Set alongside loadedIsMultimodal so the composer can say WHY images are unavailable in compare mode too.
          loadedVisionDisabledByUser: resp.vision_disabled_by_user ?? false,
          mmprojFallbackReason: resp.mmproj_fallback_reason ?? null,
          activeModelIsLocal: resp.is_local_model ?? false,
          // Same value as the baseline above, so when this pane becomes the active model the UI and a later
          // reload use the context it actually loaded with.
          customContextLength: keepCustomCtx,
          ...loadedContextFields(resp),
          // Compare selections load by repo/variant, never from the file picker,
          // so they carry no native lease. Clear any prior picked file's
          // token/expiry so the reload path never sends a stale lease.
          activeNativePathToken: null,
          activeNativePathExpiresAtMs: null,
          ...resolveLoadedSpeculativeSettings(resp),
        });
        if (!targetIsGguf) {
          // Non-GGUF panes carry their context in params.maxSeqLength.
          const paneParams = useChatRuntimeStore.getState().params;
          store.setParams({
            ...paneParams,
            maxSeqLength: loadedContextForParams(
              loadedContextFields(resp).loadedContextLength,
              effectiveMaxSeqLength,
              paneParams.maxSeqLength,
            ),
          });
        }
        loadedFromConfig = config != null;
        // Sync the models[] entry with the load response so attach/send gates read fresh capabilities:
        // /api/models/list can lag a model's actual state.
        const currentModels = useChatRuntimeStore.getState().models;
        const idx = currentModels.findIndex((m) => m.id === sel.id);
        const synced = {
          isVision: Boolean(resp.is_vision),
          isGguf: Boolean(resp.is_gguf),
          isMlx: Boolean(resp.is_mlx),
          isAudio: Boolean(resp.is_audio),
          audioType: resp.audio_type ?? null,
          hasAudioInput: Boolean(resp.has_audio_input),
          hasVideoInput: Boolean(resp.has_video_input),
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

      const name1 = model1?.id ? compareModelDisplayName(model1.id) : "";
      const name2 = model2?.id ? compareModelDisplayName(model2.id) : "";
      const toastId = toast("Comparing models…", { duration: Infinity });

      setComparing(true);
      try {
        // Side 1: load, generate, wait
        if (handle1 && model1?.id) {
          toast("Loading Model 1…", {
            id: toastId,
            description: name1,
            duration: Infinity,
          });
          const status1 = await ensureModelLoaded(model1);
          releaseCompareModelLifecycle();
          toast("Generating with Model 1…", {
            id: toastId,
            description: `${name1} (${status1})`,
            duration: Infinity,
          });
          const done = handle1.waitForRunEnd();
          handle1.startRun();
          await done;
        }

        // Side 2: load, generate, wait
        if (handle2 && model2?.id) {
          acquireCompareModelLifecycle();
          const needsLoad = compareSelectionNeedsLoad(model2);
          if (needsLoad) {
            const currentStopDecision =
              await confirmStopRunningChatsIfNeeded(
                "Loading the second model for comparison",
                "reload",
              );
            if (!currentStopDecision.proceed) {
              throw new Error("Second comparison model load cancelled.");
            }
            compareStopDecision = currentStopDecision;
            toast("Loading Model 2…", {
              id: toastId,
              description: name2,
              duration: Infinity,
            });
          }
          const status2 = await ensureModelLoaded(model2);
          releaseCompareModelLifecycle();
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
        resetPromptQueue();
        // The install already unloaded the previously active model; drop the checkpoint so the UI does
        // not keep pointing at it.
        if (upgradeUnloadedActive) {
          useChatRuntimeStore.getState().clearCheckpoint();
        }
        toast.error("Compare failed", {
          id: toastId,
          description: err instanceof Error ? err.message : "Unknown error",
          duration: 4000,
        });
      } finally {
        releaseCompareModelLifecycle();
        setComparing(false);
      }
    } else {
      // Original behavior: fire all handles simultaneously
      const liveRuntime = useChatRuntimeStore.getState();
      if (
        requireStableCheckpoint &&
        (liveRuntime.modelLoading ||
          !modelIdsMatch(
            submittedCompareCheckpoint,
            liveRuntime.params.checkpoint,
          ))
      ) {
        resetPromptQueue();
        toast.error("Compare unavailable", {
          description: "The loaded model changed while preparing the message.",
        });
        return;
      }
      const handles = Object.values(handlesRef.current);
      const reservations: symbol[] = [];
      if (requireStableCheckpoint) {
        for (const handle of handles) {
          const token = reservePreStreamRun(handle.threadIds(), {
            usesLocalModel: true,
            cancel: () => handle.cancel(),
          });
          if (!token) {
            for (const reservation of reservations) {
              releasePreStreamRunReservation(reservation);
            }
            resetPromptQueue();
            toast.error("Compare unavailable", {
              description: "A comparison run is already starting.",
            });
            return;
          }
          reservations.push(token);
        }
      }
      clearSubmittedDraft();
      for (const handle of handles) {
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
    // IME composition (JP/CN/KR): Enter commits the candidate, so do not hijack it (#5318). Re-pin
    // composingRef in case the stuck watchdog (#5546) cleared it during a long candidate-window
    // pause, and re-arm the watchdog on the same path, or the WSL+Chrome no-compositionend case
    // pins composingRef forever.
    if (e.nativeEvent.isComposing || e.keyCode === 229) {
      composingRef.current = true;
      refreshStuckImeTimer();
      return;
    }
    // Non-IME key while composingRef is stuck; mirrors the fix in thread.tsx. On macOS, switching
    // input methods without composing can leave composingRef pinned.
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
    !isDictating &&
    !sendUnavailableReason;

  // Compare mode swaps this composer in for the single-chat one and only one is ever on screen, so
  // the chords register in both. Both gate on the chat tab being visible: off-route the pane is
  // hidden, not unmounted.
  const chatActive = useChatActive();
  useShortcut(
    "startDictation",
    () => {
      // As in the single-chat composer: a dialog over Chat leaves this registered, and a microphone
      // opened behind one is neither visible nor stoppable. Stopping first and ungated, so a
      // recording stays stoppable wherever the gate would say no.
      if (isDictating) {
        stopDictation();
        return;
      }
      if (!isSurfaceInForeground(COMPOSER_INPUT_SELECTOR)) return;
      startDictation();
    },
    { enabled: chatActive },
  );
  // Through the existing sendRef: `send` is render-scoped, which the React compiler will not let a hook outlive.
  useShortcut(
    "sendMessage",
    () => {
      // As in the single-chat composer: the draft behind a dialog is not what the user is typing.
      if (!isSurfaceInForeground(COMPOSER_INPUT_SELECTOR)) return;
      sendRef.current?.();
    },
    {
      enabled: chatActive && canSend,
      // As in the single-chat composer: a non-modal popover's search box is a text field the composer
      // is still the foreground behind.
      skipInTextFields: true,
      textFieldException: COMPOSER_INPUT_SELECTOR,
    },
  );
  useShortcut(
    "attachFiles",
    () => {
      // As in the single-chat composer, which would otherwise raise the file chooser from behind a dialog.
      if (!isSurfaceInForeground(COMPOSER_INPUT_SELECTOR)) return;
      fileInputRef.current?.click();
    },
    { enabled: chatActive },
  );

  // Adjustable "+" menu items, keyed by id. Pinned ones render at the top level; the rest fall into
  // the "More" overflow submenu. Core items and "More" itself live outside this map.
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
            { label: "Training JSONL", fn: exportConversationRawJsonl },
            { label: "Message JSONL", fn: exportConversationMessagesJsonl },
            { label: "CSV", fn: exportConversationCsv },
            { label: "ShareGPT JSONL", fn: exportConversationShareGPT },
            {
              label: CONVERSATION_MARKDOWN_LABEL,
              fn: exportConversationMarkdown,
            },
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
      // Compare mode's composer, same as the thread's: find searches the conversation, not the
      // chrome around it. This one is rendered from `chat-page.tsx`, outside the marked one.
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      onDragOver={(e) => {
        if (isTauri || isPortaledDrop(e)) return;
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        // Phase 1 native model drops own Tauri local-path drops. Restore browser attachment drops in
        // Tauri once Phase 1d adds token bridging.
        if (isTauri || isPortaledDrop(e)) return;
        e.preventDefault();
        setDragging(false);
        void addFiles(e.dataTransfer.files);
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
              <span data-reload-snapshot-sensitive className="max-w-48 truncate">
                {pendingAudio.name}
              </span>
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
          // ALWAYS mirror the DOM value into React state, even during IME composition: the controlled
          // `value` must match the DOM at all times, else an unrelated parent re-render reconciles the
          // textarea back to the stored value mid-composition, wiping the preedit (#5318).
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
        onPaste={handleFilePaste}
        onBlur={() => {
          // Mac: switching input methods can fire compositionstart without a matching compositionend,
          // leaving composingRef pinned. The OS always commits or cancels before focus is lost.
          setCompositionState(false);
        }}
        placeholder="Send to both models..."
        // dir="auto" detects RTL from the first strong character; no effect on LTR scripts. Kept next to
        // the placeholder: the IME smoke reads this pair out of the source.
        dir="auto"
        // aui-composer-input carries no styling anywhere; it is the name both composers answer to, so one
        // selector can mean "the composer" whichever is on screen. Escape's decline exception and the
        // dictation foreground check rely on it.
        className="composer-input aui-composer-input"
        rows={1}
      />
      <div className="composer-action-wrapper">
        <div
          ref={pillRowRef}
          className="flex min-w-0 flex-wrap items-center gap-0.5"
          data-pill-compact={pillCompact}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={IMAGE_ACCEPT}
            multiple
            className="hidden"
            onChange={(e) => {
              void addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={audioInputRef}
            type="file"
            accept={AUDIO_PICKER_ACCEPT}
            className="hidden"
            onChange={(e) => {
              void addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <NewProjectDialog
            open={newProjectOpen}
            onOpenChange={setNewProjectOpen}
          />
          {/* Same + menu as single-chat (ComposerToolsMenu), wired to the compare composer's own file/audio
              inputs and tools. */}
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
                {/* Scale, not width: an oversized box pushed the label out of line. */}
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
          {/* Permission-level pill sits immediately after Compare and ahead of every other tool pill so the
              Full access danger state reads first. */}
          <PermissionModeComposerPill side="top" />
          <button
            type="button"
            disabled={searchDisabled}
            onClick={() => {
              const next = !toolsEnabled;
              setToolsEnabled(next);
              // Kimi's $web_search builtin requires thinking=disabled, so toggle the Think pill off when Search
              // is on, mirroring the backend.
              // Per https://platform.kimi.ai/docs/guide/use-web-search.
              // Per https://platform.kimi.ai/docs/guide/use-web-search.
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
        {/* mr-0.5 matches the send button inset from the edge in normal chat; gap-1.5 matches its control spacing. */}
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
                    <ChevronDownIcon strokeWidth={1.5} className="unsloth-thinking-caret size-[15px]" />
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
                              // Mutual exclusion: turning thinking on for a Kimi model forces its
                              // web_search builtin off.
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
                  // Mutual exclusion: Kimi's $web_search builtin requires thinking off, so turning thinking on
                  // flips Search off.
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
                  <MicIcon className="unsloth-dictate-icon size-4" />
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
                  <SquareIcon className="aui-composer-cancel-icon size-3 animate-pulse fill-current" />
                </TooltipIconButton>
              )}
            </>
          }
          {isQueueRunning ? (
            <button
              type="button"
              onClick={() => {
                resetPromptQueue();
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
              className="ml-1.5 size-9 rounded-full"
              onClick={stop}
            >
              <SquareIcon className="aui-composer-cancel-icon size-3 fill-current" />
            </Button>
          ) : (
            <TooltipIconButton
              tooltip={sendUnavailableReason ?? "Send message"}
              side="bottom"
              variant="default"
              size="icon"
              className="ml-1.5 size-9 rounded-full"
              onClick={send}
              disabled={!canSend}
              aria-label="Send message"
            >
              <ArrowUpIcon className="unsloth-send-icon size-[22px] stroke-2" />
            </TooltipIconButton>
          )}
        </div>
      </div>
    </div>
  );
}
