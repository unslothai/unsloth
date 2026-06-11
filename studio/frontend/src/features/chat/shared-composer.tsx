// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  thinkEffortAriaLabel,
  thinkToggleAriaLabel,
} from "@/components/assistant-ui/think-aria-label";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
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
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE, fileToBase64 } from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { isMultimodalResponse } from "./types/api";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import { useAui } from "@assistant-ui/react";
import {
  ArrowUpIcon,
  Columns2Icon,
  FileText,
  GlobeIcon,
  HeadphonesIcon,
  LoaderIcon,
  MoreHorizontalIcon,
  PlusIcon,
  RefreshCwIcon,
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
  Tick02Icon,
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
import { KnowledgeBaseComposerButton } from "@/features/rag/components/knowledge-base-composer-button";
import { NewProjectDialog } from "./components/new-project-dialog";
import { useChatProjects } from "./hooks/use-chat-projects";
import {
  parseExternalModelId,
  providerTypeSupportsVision,
} from "./external-providers";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import {
  type ReasoningEffort,
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
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  getCachedDocumentSupport,
  getDocumentSupport,
  loadModel,
  validateModel,
} from "./api/chat-api";
import {
  AttachmentChipBody,
  AttachmentChipProgress,
  AttachmentChipRemoveButton,
  AttachmentChipRoot,
  AttachmentChipTitle,
  attachmentChipTokens,
} from "./components/attachment-chip-primitives";
import { DocAttachmentChip } from "./components/doc-attachment-chip";
import {
  type DocumentExtractionRunner,
  createDocumentExtractionRunner,
} from "./hooks/use-document-extraction";
import type {
  DocumentExtractionErrorCode,
  PendingDocumentAttachment,
} from "./types";
import {
  DOC_ACCEPT,
  type DocumentVisualPolicy,
  MAX_DOC_SIZE,
  TEXT_ONLY_DOCUMENT_VISUAL_POLICY,
  buildDocumentMessageParts,
  classifyDocumentExtractionError,
  documentParserUnavailableReason,
  documentVisualPayloads,
  documentVisualPolicyFromSupport,
  isDocumentFile,
  markDocumentExtractionRetry,
  normalizeExtractedDocument,
} from "./utils/document-extraction";
import {
  isTemporaryOcrModelBusy,
  subscribeTemporaryOcrModelBusy,
} from "./utils/ocr-model-lock";
import { applyQwenThinkingParams } from "./utils/qwen-params";

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
  /** Returns a promise that resolves when the current or next run finishes.
   *  Pass an AbortSignal so the caller can release the underlying Zustand
   *  subscription if startRun never fires (e.g. it threw synchronously). */
  waitForRunEnd: (signal?: AbortSignal) => Promise<void>;
}

const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp,image/gif";
const MAX_IMAGE_SIZE = 20 * 1024 * 1024;
const MAX_DOCUMENT_RETRIES = 2;
const NON_RETRYABLE_DOCUMENT_ERRORS: ReadonlySet<DocumentExtractionErrorCode> =
  new Set(["aborted", "encrypted", "oversized", "unsupported_type"]);

function canRetryFailedDocument(doc: FailedDocument): boolean {
  return (
    doc.retryCount < MAX_DOCUMENT_RETRIES &&
    !NON_RETRYABLE_DOCUMENT_ERRORS.has(doc.code)
  );
}

async function resolveCurrentDocumentVisualPolicy(): Promise<DocumentVisualPolicy> {
  try {
    return documentVisualPolicyFromSupport(await getDocumentSupport());
  } catch {
    return TEXT_ONLY_DOCUMENT_VISUAL_POLICY;
  }
}

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

const MicIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 256 256"
    fill="currentColor"
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden={true}
  >
    <path d="M128,176a48.05,48.05,0,0,0,48-48V64a48,48,0,0,0-96,0v64A48.05,48.05,0,0,0,128,176ZM96,64a32,32,0,0,1,64,0v64a32,32,0,0,1-64,0Zm40,143.6V232a8,8,0,0,1-16,0V207.6A80.11,80.11,0,0,1,48,128a8,8,0,0,1,16,0,64,64,0,0,0,128,0,8,8,0,0,1,16,0A80.11,80.11,0,0,1,136,207.6Z" />
  </svg>
);

const BulbIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="-10.24 -10.24 1044.48 1044.48"
    fill="currentColor"
    stroke="currentColor"
    strokeWidth={16.384}
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden={true}
  >
    <path d="M511.984 0c-198.032 0-353.12 161.104-353.12 359.136 0 149.2 73.28 220.256 131.185 272.128 37.28 33.424 62.368 53.552 62.368 78.352v54.255c0 1.392.193 2.752.368 4.128h-.72v92.624c.016 97.712 63.2 163.376 161.072 163.376 94.464 0 158.944-65.664 158.944-163.376V768h-.928c.176-1.376.416-2.736.416-4.128v-54.255c0-37.76 28.032-60.592 70.528-97.696 57.504-50.208 123.023-112.688 123.023-252.784C865.136 161.104 710.016 0 511.983 0zm-1.215 960c-59.904 0-94.689-37.152-94.689-99.376l-.463-42.672C438.64 825.824 470 832 512 832c41.424 0 72.848-6.624 96.08-14.768v43.392c0 63.152-35.247 99.376-97.312 99.376zm189.248-396.288c-43.472 37.968-92.433 77.216-92.433 145.904v40.432c-15.183 8.48-43.183 18.56-96.127 18.56-55.569 0-81.92-9.856-95.024-17.473V709.6c0-54.608-42.688-89.297-83.68-126.017-54.32-48.672-109.873-103.84-109.873-224.464-.015-162.72 126.385-295.12 289.104-295.12 162.752 0 289.152 132.4 289.152 295.137 0 111.024-48.463 158.576-101.12 204.576z" />
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
  const [isDictating, setIsDictating] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const start = useCallback(() => {
    const SpeechRecognitionAPI =
      typeof window !== "undefined" &&
      (window.SpeechRecognition ??
        (
          window as unknown as {
            webkitSpeechRecognition?: typeof SpeechRecognition;
          }
        ).webkitSpeechRecognition);
    if (!SpeechRecognitionAPI) {
      return;
    }
    const recognition = new SpeechRecognitionAPI() as SpeechRecognition;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const last = event.resultIndex;
      const result = event.results[last];
      if (!result?.isFinal) return;
      const transcript = result[0]?.transcript?.trim();
      if (transcript) {
        setText((prev) => (prev ? `${prev} ${transcript}` : transcript));
      }
    };
    recognition.onerror = () => {
      setIsDictating(false);
    };
    recognition.onend = () => {
      setIsDictating(false);
    };
    recognition.start();
    recognitionRef.current = recognition;
    setIsDictating(true);
  }, [setText]);

  const stop = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setIsDictating(false);
  }, []);

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  const supported =
    typeof window !== "undefined" &&
    !!(
      window.SpeechRecognition ??
      (window as unknown as { webkitSpeechRecognition?: unknown })
        .webkitSpeechRecognition
    );

  return { isDictating, start, stop, supported };
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
      waitForRunEnd: (signal?: AbortSignal) =>
        new Promise<void>((resolve) => {
          let wasRunning = false;
          let settled = false;
          let unsubscribe: (() => void) | null = null;
          let onAbort: (() => void) | null = null;
          const finish = () => {
            if (settled) return;
            settled = true;
            window.clearTimeout(timeout);
            unsubscribe?.();
            if (onAbort && signal) signal.removeEventListener("abort", onAbort);
            resolve();
          };
          const timeout = window.setTimeout(finish, 120_000);
          unsubscribe = useChatRuntimeStore.subscribe((state) => {
            const anyRunning = Object.keys(state.runningByThreadId).length > 0;
            if (anyRunning) wasRunning = true;
            if (wasRunning && !anyRunning) {
              finish();
            }
          });
          if (signal) {
            if (signal.aborted) {
              finish();
              return;
            }
            onAbort = finish;
            signal.addEventListener("abort", onAbort, { once: true });
          }
        }),
    };
    return () => {
      delete currentHandles[name];
    };
  }, [handlesRef, name, aui]);

  return null;
}

type PendingImage = { id: string; file: File };
type UploadingDocument = { id: string; name: string; progress?: number };
type FailedDocument = {
  id: string;
  name: string;
  file: File;
  message: string;
  code: DocumentExtractionErrorCode;
  retryCount: number;
};

function PendingImageThumb({
  file,
  onRemove,
}: {
  file: File;
  onRemove: () => void;
}): ReactElement {
  const src = useMemo(() => URL.createObjectURL(file), [file]);

  useEffect(() => {
    return () => URL.revokeObjectURL(src);
  }, [src]);

  return (
    <div className={attachmentChipTokens.tile}>
      <img src={src} alt={file.name} className="h-full w-full object-cover" />
      <button
        type="button"
        onClick={onRemove}
        className="absolute top-1 right-1 flex size-6 items-center justify-center rounded-full bg-white text-muted-foreground shadow-sm hover:bg-destructive hover:text-destructive-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
};

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
  const [pendingDocs, setPendingDocs] = useState<PendingDocumentAttachment[]>(
    [],
  );
  const [uploadingDocs, setUploadingDocs] = useState<UploadingDocument[]>([]);
  const [failedDocs, setFailedDocs] = useState<FailedDocument[]>([]);
  const [dragging, setDragging] = useState(false);
  const [isComposing, setIsComposing] = useState(false);
  const [temporaryOcrBusy, setTemporaryOcrBusy] = useState(
    isTemporaryOcrModelBusy,
  );
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [promptStorageOpen, setPromptStorageOpen] = useState(false);
  const [recentPrompts, setRecentPrompts] = useState<PromptEntry[]>([]);
  const refreshRecentPrompts = useCallback(async () => {
    try {
      const rows = await listPromptEntries();
      setRecentPrompts(
        [...rows].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 3),
      );
    } catch {
    }
  }, []);
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
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const modelBusy = modelLoading || temporaryOcrBusy;
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
  const isEffort = effectiveReasoningStyle === "reasoning_effort";
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
  // Above 4 pills, collapse to icons only to cut clutter. Compare, Search and
  // Code always show; the rest are conditional.
  const pillsCompact =
    3 +
      (showImagePill ? 1 : 0) +
      (showRagPill && ragEnabled && !ragDisabled ? 1 : 0) +
      (showWebFetchPill ? 1 : 0) +
      (artifactsEnabled ? 1 : 0) +
      (mcpEnabledForChat ? 1 : 0) >
    4;
  // Backwards-compatible alias for call sites still referencing
  // `toolsDisabled` (rare; both pills used it before).
  const toolsDisabled = codeDisabled;
  const setPendingAudioStore = useChatRuntimeStore((s) => s.setPendingAudio);
  const clearPendingAudioStore = useChatRuntimeStore(
    (s) => s.clearPendingAudio,
  );

  const {
    isDictating,
    start: startDictation,
    stop: stopDictation,
    supported: dictationSupported,
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
        description: "A compare step failed — remaining prompts were not sent.",
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
    const lineHeight = Number.parseFloat(styles.lineHeight) || 20;
    const paddingY =
      Number.parseFloat(styles.paddingTop) +
      Number.parseFloat(styles.paddingBottom);
    const borderY =
      Number.parseFloat(styles.borderTopWidth) +
      Number.parseFloat(styles.borderBottomWidth);
    const maxHeight = lineHeight * 6 + paddingY + borderY;
    const next = Math.min(ta.scrollHeight, maxHeight);
    ta.style.height = `${next}px`;
    ta.style.overflowY = ta.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [text]);

  const docRunnersRef = useRef<Map<string, DocumentExtractionRunner>>(
    new Map(),
  );

  // Abort all in-flight extractions on unmount
  useEffect(() => {
    const runners = docRunnersRef.current;
    return () => {
      for (const runner of runners.values()) {
        runner.abort();
      }
      runners.clear();
    };
  }, []);

  const uploadDocument = useCallback(async (file: File, retryCount = 0) => {
    // Read fresh store state at call time so a settings toggle that
    // lands between file-drop and this callback invocation is honored.
    const current = useChatRuntimeStore.getState().docExtract;
    if (!current.enabled) {
      toast.message("Document extraction is disabled", {
        description: "Enable it in Chat settings before dropping documents.",
      });
      return;
    }
    if (file.size > MAX_DOC_SIZE) {
      toast.error(`${file.name} exceeds 100 MB`);
      return;
    }
    try {
      const support = await getCachedDocumentSupport();
      const unavailableReason = documentParserUnavailableReason(file, support);
      if (unavailableReason) {
        toast.error(`${file.name} is not available for extraction`, {
          description: unavailableReason,
        });
        return;
      }
    } catch {
      // Let the upload path surface the authoritative backend error.
    }
    const placeholderId = crypto.randomUUID();
    const runner = createDocumentExtractionRunner();
    docRunnersRef.current.set(placeholderId, runner);
    setUploadingDocs((prev) => [
      ...prev,
      { id: placeholderId, name: file.name },
    ]);
    setFailedDocs((prev) => prev.filter((doc) => doc.file !== file));
    const captionToastId = `doc-caption-${placeholderId}`;
    let captionToastShown = false;
    try {
      const doc = await runner.run(file, {
        onParseStart: () => {
          setUploadingDocs((prev) =>
            prev.map((item) =>
              item.id === placeholderId
                ? { ...item, progress: Math.max(item.progress ?? 0, 0.1) }
                : item,
            ),
          );
        },
        onCaptionProgress: ({ current, total, page, totalPages }) => {
          if (total <= 0) return;
          const fraction = Math.max(0, Math.min(1, current / total));
          // Map captioning fraction onto the back half of the chip bar
          // so the bar moves through both phases (parse → caption).
          const mapped = 0.2 + fraction * 0.8;
          setUploadingDocs((prev) =>
            prev.map((item) =>
              item.id === placeholderId
                ? { ...item, progress: Math.max(item.progress ?? 0, mapped) }
                : item,
            ),
          );
          const pageSuffix =
            page != null && totalPages > 0
              ? ` · page ${page} of ${totalPages}`
              : "";
          const message = `Captioning images ${current}/${total}${pageSuffix}`;
          const description = `${file.name}`;
          if (!captionToastShown) {
            toast.loading(message, {
              id: captionToastId,
              description,
              duration: Infinity,
            });
            captionToastShown = true;
          } else {
            toast.loading(message, { id: captionToastId, description });
          }
          if (current >= total) {
            toast.success(`Finished captioning ${total} image${total === 1 ? "" : "s"}`, {
              id: captionToastId,
              description,
              duration: 2500,
            });
          }
        },
      });
      // Re-read token budget at send time so Compare Mode sees latest value
      const docSettings = useChatRuntimeStore.getState().docExtract;
      const normalizedDoc = normalizeExtractedDocument(doc);
      const visualPolicy = await resolveCurrentDocumentVisualPolicy();
      const { truncated } = buildDocumentMessageParts(
        {
          filename: normalizedDoc.filename || file.name,
          document: normalizedDoc,
        },
        docSettings.tokenBudget,
        visualPolicy,
        docSettings.maxVisualPayloads,
      );
      const sentImageIndexes = documentVisualPayloads(
        normalizedDoc,
        docSettings.maxVisualPayloads,
        visualPolicy,
      ).map((payload) => payload.index);
      const attachment: PendingDocumentAttachment = {
        id: placeholderId,
        filename: normalizedDoc.filename || file.name,
        sizeBytes: file.size,
        document: normalizedDoc,
        extractedAt: Date.now(),
        truncated,
        sentImageIndexes,
      };
      markDocumentExtractionRetry(file, 0);
      setPendingDocs((prev) => [...prev, attachment]);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        if (captionToastShown) toast.dismiss(captionToastId);
        return;
      }
      if (captionToastShown) toast.dismiss(captionToastId);
      const failure = classifyDocumentExtractionError(err);
      setFailedDocs((prev) => [
        ...prev,
        {
          id: placeholderId,
          name: file.name,
          file,
          message: failure.message,
          code: failure.code,
          retryCount,
        },
      ]);
    } finally {
      docRunnersRef.current.delete(placeholderId);
      setUploadingDocs((prev) => prev.filter((d) => d.id !== placeholderId));
    }
  }, []);

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
          fileToBase64(file)
            .then((base64) => {
              setPendingAudio({ name: file.name, base64 });
              setPendingAudioStore(base64, file.name);
            })
            .catch((err) => {
              const msg = err instanceof Error ? err.message : String(err);
              toast.error(`Failed to encode audio attachment: ${msg}`);
            });
          continue;
        }
        // Handle image files
        if (file.type.match(/^image\/(jpeg|png|webp|gif)$/i)) {
          if (file.size > MAX_IMAGE_SIZE) continue;
          if (attachUnavailableReason) {
            droppedImageForUnavailable = true;
            continue;
          }
          next.push({ id: crypto.randomUUID(), file });
          continue;
        }
        if (isDocumentFile(file)) {
          void uploadDocument(file);
          continue;
        }
        toast.error(`Unsupported file type: ${file.type || file.name}`);
      }
      if (droppedImageForUnavailable && attachUnavailableReason) {
        toast.error(attachUnavailableReason);
      }
      setPendingImages((prev) => [...prev, ...next]);
    },
    [attachUnavailableReason, setPendingAudioStore, uploadDocument],
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

  const removePendingDoc = useCallback((id: string) => {
    const runner = docRunnersRef.current.get(id);
    if (runner) {
      runner.abort();
      docRunnersRef.current.delete(id);
    }
    setPendingDocs((prev) => prev.filter((p) => p.id !== id));
    setUploadingDocs((prev) => prev.filter((d) => d.id !== id));
    setFailedDocs((prev) => prev.filter((d) => d.id !== id));
  }, []);

  const retryFailedDoc = useCallback(
    (doc: FailedDocument) => {
      if (!canRetryFailedDocument(doc)) {
        toast.error("Document retry limit reached", {
          description:
            "Remove the failed attachment or adjust extraction settings before trying again.",
        });
        return;
      }
      const nextRetryCount = doc.retryCount + 1;
      markDocumentExtractionRetry(doc.file, nextRetryCount);
      setFailedDocs((prev) => prev.filter((item) => item.id !== doc.id));
      void uploadDocument(doc.file, nextRetryCount);
    },
    [uploadDocument],
  );

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
    if (
      uploadingDocs.length > 0 ||
      failedDocs.length > 0 ||
      running ||
      comparing ||
      modelBusy
    ) {
      return;
    }

    const msg = text.trim();
    if (
      !msg &&
      pendingImages.length === 0 &&
      !pendingAudio &&
      pendingDocs.length === 0
    ) {
      return;
    }

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

    const documentAttachments = [...pendingDocs];
    const trailingContent: CompareMessagePart[] = [];
    for (const { file } of pendingImages) {
      try {
        const image = await fileToBase64DataURL(file);
        trailingContent.push({ type: "image", image });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        toast.error(`Failed to encode image "${file.name}": ${msg}`);
        // Drop the failing image part; continue with remaining content
      }
    }
    if (pendingAudio) {
      trailingContent.push({ type: "audio", audio: pendingAudio.base64 });
    }
    if (msg) {
      trailingContent.push({ type: "text", text: msg });
    }

    async function buildContentForCurrentModel(): Promise<
      CompareMessagePart[]
    > {
      const visualPolicy = await resolveCurrentDocumentVisualPolicy();
      const docSettings = useChatRuntimeStore.getState().docExtract;
      const content: CompareMessagePart[] = [];
      // Documents first: they provide the reference context the user's
      // message is asking about.
      for (const doc of documentAttachments) {
        const { parts } = buildDocumentMessageParts(
          { filename: doc.filename, document: doc.document },
          docSettings.tokenBudget,
          visualPolicy,
          docSettings.maxVisualPayloads,
        );
        content.push(...parts);
      }
      content.push(...trailingContent);
      return content;
    }

    if (documentAttachments.length === 0 && trailingContent.length === 0)
      return;

    let singleContent: CompareMessagePart[] | null = null;
    if (!isGeneralizedCompare) {
      try {
        singleContent = await buildContentForCurrentModel();
      } catch (err) {
        toast.error("Could not prepare message", {
          description: err instanceof Error ? err.message : "Unknown error",
        });
        return;
      }
    }
    if (
      !isGeneralizedCompare &&
      (!singleContent || singleContent.length === 0)
    ) {
      return;
    }

    setText("");
    setPendingImages([]);
    setPendingAudio(null);
    setPendingDocs([]);
    clearPendingAudioStore();
    textareaRef.current?.focus();

    // Generalized compare: load each model before dispatching to its side
    if (isGeneralizedCompare) {
      const store = useChatRuntimeStore.getState();
      const maxSeqLength = store.params.maxSeqLength;
      const trustRemoteCode = store.params.trustRemoteCode ?? false;
      const chatTemplateOverride = store.chatTemplateOverride;
      const effectiveChatTemplateOverride = chatTemplateOverride?.trim()
        ? chatTemplateOverride
        : null;

      function modelDisplayName(id: string): string {
        const parts = id.split("/");
        return parts[parts.length - 1] || id;
      }

      // Helper: load a model and update store checkpoint
      async function ensureModelLoaded(
        sel: CompareModelSelection,
      ): Promise<string> {
        const currentStore = useChatRuntimeStore.getState();
        const isAlreadyActive =
          currentStore.params.checkpoint === sel.id &&
          (currentStore.activeGgufVariant ?? null) ===
            (sel.ggufVariant ?? null);
        if (!isAlreadyActive) {
          const validation = await validateModel({
            model_path: sel.id,
            hf_token: currentStore.hfToken || null,
            max_seq_length: maxSeqLength,
            load_in_4bit: true,
            is_lora: sel.isLora,
            gguf_variant: sel.ggufVariant ?? null,
            trust_remote_code: trustRemoteCode,
            chat_template_override: effectiveChatTemplateOverride,
          });
          if (validation.requires_trust_remote_code && !trustRemoteCode) {
            throw new Error(
              `${modelDisplayName(sel.id)} needs custom code enabled to load. Turn on "Enable custom code" in Chat Settings, then try again.`,
            );
          }
        }
        const resp = await loadModel({
          model_path: sel.id,
          hf_token: useChatRuntimeStore.getState().hfToken || null,
          max_seq_length: maxSeqLength,
          load_in_4bit: true,
          is_lora: sel.isLora,
          gguf_variant: sel.ggufVariant ?? null,
          trust_remote_code: trustRemoteCode,
          chat_template_override: effectiveChatTemplateOverride,
        });
        const store = useChatRuntimeStore.getState();
        store.setCheckpoint(
          resp.model,
          resp.is_gguf ? (sel.ggufVariant ?? undefined) : null,
        );
        store.setModelRequiresTrustRemoteCode(
          resp.requires_trust_remote_code ?? false,
        );
        useChatRuntimeStore.setState({
          supportsReasoning: resp.supports_reasoning ?? false,
          reasoningAlwaysOn: resp.reasoning_always_on ?? false,
          reasoningStyle: resp.reasoning_style ?? "enable_thinking",
          supportsPreserveThinking: resp.supports_preserve_thinking ?? false,
          supportsTools: resp.supports_tools ?? false,
          loadedIsMultimodal: isMultimodalResponse(resp),
        });
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

      const name1 = model1?.id ? modelDisplayName(model1.id) : "";
      const name2 = model2?.id ? modelDisplayName(model2.id) : "";
      const toastId = toast("Comparing models…", {
        duration: Number.POSITIVE_INFINITY,
      });

      setComparing(true);
      try {
        // Side 1: load → generate → wait
        if (handle1 && model1?.id) {
          toast("Loading Model 1…", {
            id: toastId,
            description: name1,
            duration: Number.POSITIVE_INFINITY,
          });
          const status1 = await ensureModelLoaded(model1);
          toast("Generating with Model 1…", {
            id: toastId,
            description: `${name1} (${status1})`,
            duration: Number.POSITIVE_INFINITY,
          });
          const content1 = await buildContentForCurrentModel();
          handle1.appendMessage(content1);
          const runEndAbort = new AbortController();
          const done = handle1.waitForRunEnd(runEndAbort.signal);
          try {
            handle1.startRun();
          } catch (err) {
            runEndAbort.abort();
            throw err;
          }
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
              duration: Number.POSITIVE_INFINITY,
            });
          }
          const status2 = await ensureModelLoaded(model2);
          toast("Generating with Model 2…", {
            id: toastId,
            description: `${name2} (${status2})`,
            duration: Number.POSITIVE_INFINITY,
          });
          const content2 = await buildContentForCurrentModel();
          handle2.appendMessage(content2);
          const runEndAbort = new AbortController();
          const done = handle2.waitForRunEnd(runEndAbort.signal);
          try {
            handle2.startRun();
          } catch (err) {
            runEndAbort.abort();
            throw err;
          }
          await done;
        }

        compareStepSucceededRef.current = true;
        toast.success("Compare complete", { id: toastId, duration: 2000 });
      } catch (err) {
        compareStepSucceededRef.current = false;
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
        handle.append(singleContent ?? []);
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

  useEffect(() => {
    if (!dragging) return;
    const timeout = window.setTimeout(() => setDragging(false), 3000);
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setDragging(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.clearTimeout(timeout);
      window.removeEventListener("keydown", onKey);
    };
  }, [dragging]);

  useEffect(() => {
    return subscribeTemporaryOcrModelBusy(() => {
      setTemporaryOcrBusy(isTemporaryOcrModelBusy());
    });
  }, []);

  const canSend =
    (text.trim().length > 0 ||
      pendingImages.length > 0 ||
      pendingAudio !== null ||
      pendingDocs.length > 0) &&
    uploadingDocs.length === 0 &&
    failedDocs.length === 0 &&
    !modelBusy &&
    !busy &&
    !isComposing;
  const blockingAttachmentLabel =
    uploadingDocs.length > 0
      ? `Waiting for ${uploadingDocs.length} attachment${
          uploadingDocs.length === 1 ? "" : "s"
        }...`
      : failedDocs.length > 0
        ? `Resolve ${failedDocs.length} failed attachment${
            failedDocs.length === 1 ? "" : "s"
          } before sending.`
      : null;

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
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) {
        send();
      }
    }
  }

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
      {(pendingImages.length > 0 ||
        pendingAudio ||
        pendingDocs.length > 0 ||
        uploadingDocs.length > 0 ||
        failedDocs.length > 0) && (
        <div className="mb-2 flex w-full flex-row flex-wrap items-center gap-2 px-1.5 pt-0.5 pb-1">
          {pendingImages.map(({ id, file }) => (
            <PendingImageThumb
              key={id}
              file={file}
              onRemove={() => removePendingImage(id)}
            />
          ))}
          {pendingDocs.map((doc) => (
            <DocAttachmentChip
              key={doc.id}
              attachment={doc}
              onRemove={() => removePendingDoc(doc.id)}
            />
          ))}
          {uploadingDocs.map((doc) => {
            const pct =
              typeof doc.progress === "number"
                ? Math.round(doc.progress * 100)
                : null;
            return (
              <AttachmentChipRoot
                key={doc.id}
                className="min-w-56 max-w-[min(20rem,calc(100vw-3rem))] items-center pr-9"
                aria-live="polite"
                aria-label={`Extracting ${doc.name}`}
              >
                <span className="flex size-10 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
                  <LoaderIcon
                    className="size-5 animate-spin motion-reduce:animate-none"
                    aria-hidden="true"
                  />
                </span>
                <AttachmentChipBody className="gap-0.5">
                  <AttachmentChipTitle className="text-sm" title={doc.name}>
                    {doc.name}
                  </AttachmentChipTitle>
                  <span className="truncate text-xs text-muted-foreground">
                    {pct !== null ? `Reading… ${pct}%` : "Reading…"}
                  </span>
                  <AttachmentChipProgress
                    value={pct}
                    label={
                      pct !== null ? `${pct}% processed` : `Reading ${doc.name}`
                    }
                    className="mt-1"
                  />
                </AttachmentChipBody>
                <AttachmentChipRemoveButton
                  tooltip="Cancel"
                  onClick={() => removePendingDoc(doc.id)}
                  aria-label={`Cancel extracting ${doc.name}`}
                />
              </AttachmentChipRoot>
            );
          })}
          {failedDocs.map((doc) => {
            const canRetry = canRetryFailedDocument(doc);
            return (
              <AttachmentChipRoot
                key={doc.id}
                className={cn(
                  "min-w-64 max-w-[min(20rem,calc(100vw-3rem))] items-center",
                  canRetry ? "pr-14" : "pr-9",
                )}
                role="alert"
              >
                <span className="flex size-10 shrink-0 items-center justify-center rounded-md bg-destructive/15 text-destructive">
                  <FileText className="size-5" aria-hidden="true" />
                </span>
                <AttachmentChipBody className="gap-0.5">
                  <AttachmentChipTitle className="text-sm" title={doc.name}>
                    {doc.name}
                  </AttachmentChipTitle>
                  <span
                    className="truncate text-xs text-destructive"
                    title={doc.message}
                  >
                    {doc.message}
                  </span>
                </AttachmentChipBody>
                {canRetry ? (
                  <AttachmentChipRemoveButton
                    tooltip="Retry"
                    className="right-7 text-muted-foreground hover:bg-primary/10 hover:text-primary"
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      retryFailedDoc(doc);
                    }}
                    aria-label={`Retry extracting ${doc.name}`}
                  >
                    <RefreshCwIcon className="size-3" aria-hidden="true" />
                  </AttachmentChipRemoveButton>
                ) : null}
                <AttachmentChipRemoveButton
                  tooltip="Remove"
                  onClick={() => removePendingDoc(doc.id)}
                  aria-label={`Remove failed document ${doc.name}`}
                />
              </AttachmentChipRoot>
            );
          })}
          {pendingAudio && (
            <AttachmentChipRoot className="max-w-[min(20rem,calc(100vw-3rem))] items-center pr-9">
              <span className="flex size-10 shrink-0 items-center justify-center rounded-md bg-amber-500/15 text-amber-600 dark:text-amber-400">
                <HeadphonesIcon className="size-5" aria-hidden="true" />
              </span>
              <AttachmentChipBody className="gap-0.5">
                <AttachmentChipTitle
                  className="text-sm"
                  title={pendingAudio.name}
                >
                  {pendingAudio.name}
                </AttachmentChipTitle>
                <span className="truncate text-xs text-muted-foreground">
                  Audio
                </span>
              </AttachmentChipBody>
              <AttachmentChipRemoveButton
                tooltip="Remove audio"
                onClick={() => {
                  setPendingAudio(null);
                  clearPendingAudioStore();
                }}
                aria-label="Remove audio"
              />
            </AttachmentChipRoot>
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
        onPaste={(e) => {
          if (e.clipboardData.files.length > 0) {
            e.preventDefault();
            addFiles(e.clipboardData.files);
          }
        }}
        onKeyDown={onKeyDown}
        placeholder="Send to both models..."
        className="composer-input"
        rows={1}
        // dir="auto" detects RTL (Arabic/Hebrew/Persian/Urdu) from the first
        // strong character; no effect on LTR scripts.
        dir="auto"
      />
      {blockingAttachmentLabel ? (
        <p
          className="px-5 pb-1 text-[11px] text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          {blockingAttachmentLabel}
        </p>
      ) : null}
      <div className="composer-action-wrapper">
        <div
          className="flex items-center gap-0.5"
          data-pill-compact={pillsCompact ? "true" : undefined}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={`${IMAGE_ACCEPT},${DOC_ACCEPT}`}
            multiple={true}
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
              className="unsloth-plus-menu w-[212px]"
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
              <DropdownMenuItem
                disabled={ragDisabled}
                className={
                  ragEnabled && !ragDisabled
                    ? "text-primary font-medium"
                    : undefined
                }
                onSelect={() => setRagEnabled(!ragEnabled)}
              >
                <HugeiconsIcon icon={FileDatabaseIcon} strokeWidth={2} />
                Chat with Files
                {ragEnabled && !ragDisabled ? (
                  <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                    className="ml-auto"
                  />
                ) : null}
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={!supportsTools}
                className={
                  mcpEnabledForChat ? "text-primary font-medium" : undefined
                }
                onSelect={() => setMcpEnabledForChat(!mcpEnabledForChat)}
              >
                <HugeiconsIcon icon={McpServerIcon} strokeWidth={2} />
                MCP
                {mcpEnabledForChat ? (
                  <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                    className="ml-auto"
                  />
                ) : null}
              </DropdownMenuItem>
              {/* RAG hidden temporarily */}
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>
                  <MoreHorizontalIcon className="size-4" />
                  More
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="unsloth-plus-menu w-[200px]">
                  {/* Always active: this menu only renders in compare mode. Ticked
                      like Web search/Code; click toggles it off. */}
                  <DropdownMenuItem
                    className="text-primary font-medium"
                    onSelect={handleExitCompare}
                  >
                    <Columns2Icon />
                    Compare chat
                    <HugeiconsIcon
                      icon={Tick02Icon}
                      strokeWidth={2}
                      className="ml-auto"
                    />
                  </DropdownMenuItem>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger>
                      <HugeiconsIcon icon={Bookmark02Icon} strokeWidth={2} />
                      Saved prompts
                    </DropdownMenuSubTrigger>
                    <DropdownMenuSubContent
                      collisionPadding={16}
                      className="unsloth-plus-menu w-[176px]"
                    >
                      {recentPrompts.map((p) => (
                        <DropdownMenuItem
                          key={p.id}
                          onSelect={() => {
                            setText(p.text);
                            requestAnimationFrame(() =>
                              textareaRef.current?.focus(),
                            );
                          }}
                        >
                          <span className="truncate">{p.name}</span>
                        </DropdownMenuItem>
                      ))}
                      {recentPrompts.length > 0 ? (
                        <DropdownMenuSeparator />
                      ) : null}
                      <DropdownMenuItem
                        onSelect={() => setPromptStorageOpen(true)}
                      >
                        All saved prompts…
                      </DropdownMenuItem>
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger
                      disabled={exportThreadIds.length === 0}
                    >
                      <HugeiconsIcon icon={Download01Icon} strokeWidth={2} />
                      Export chat
                    </DropdownMenuSubTrigger>
                    <DropdownMenuSubContent
                      collisionPadding={16}
                      className="unsloth-plus-menu w-[176px]"
                    >
                      {[
                        { label: "Raw JSONL", fn: exportConversationRawJsonl },
                        { label: "CSV", fn: exportConversationCsv },
                        {
                          label: "ShareGPT JSONL",
                          fn: exportConversationShareGPT,
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
                            Promise.all(
                              exportThreadIds.map((id) => fn(id)),
                            ).catch(() => toast.error("Export failed."));
                          }}
                        >
                          {label}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                  <DropdownMenuItem
                    className={
                      artifactsEnabled ? "text-primary font-medium" : undefined
                    }
                    onSelect={() => setArtifactsEnabled(!artifactsEnabled)}
                  >
                    <HugeiconsIcon icon={PencilRulerIcon} strokeWidth={2} />
                    Canvas
                    {artifactsEnabled ? (
                      <HugeiconsIcon
                        icon={Tick02Icon}
                        strokeWidth={2}
                        className="ml-auto"
                      />
                    ) : null}
                  </DropdownMenuItem>
                </DropdownMenuSubContent>
              </DropdownMenuSub>
              <DropdownMenuSeparator />
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>
                  <HugeiconsIcon icon={Folder01Icon} strokeWidth={2} />
                  Projects
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="unsloth-plus-menu w-[200px]">
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
                    data-active={thinkingActiveLook ? "true" : "false"}
                    aria-label={thinkEffortAriaLabel({
                      modelLoaded,
                      reasoningDisabled,
                      reasoningEffort,
                    })}
                  >
                    <BulbIcon className="size-[15.5px]" />
                    {thinkingActiveLook ? (
                      <span>
                        {isEffort
                          ? `Thinking · ${formatReasoningEffortLabel(
                              reasoningEffort,
                              externalSelection?.modelId,
                            )}`
                          : "Thinking"}
                      </span>
                    ) : null}
                    <ArrowDownStandardIcon className="size-[15px]" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  side="top"
                  align="end"
                  className="unsloth-plus-menu min-w-44"
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
                {thinkingActiveLook ? <span>Thinking</span> : null}
              </button>
            )
          ) : null}
          {dictationSupported && (
            <>
              {isDictating ? (
                <TooltipIconButton
                  tooltip="Stop dictation"
                  side="bottom"
                  variant="ghost"
                  size="icon"
                  className="size-8 rounded-full text-destructive"
                  onClick={stopDictation}
                  aria-label="Stop dictation"
                >
                  <SquareIcon className="size-3 animate-pulse fill-current" />
                </TooltipIconButton>
              ) : (
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
              )}
            </>
          )}
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
              tooltip={blockingAttachmentLabel ?? "Send message"}
              side="bottom"
              variant="default"
              size="icon"
              className={cn(
                "ml-1.5 size-8 rounded-full",
                !canSend && "cursor-not-allowed opacity-50",
              )}
              onClick={() => {
                if (canSend) void send();
              }}
              disabled={!canSend}
              aria-disabled={!canSend}
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
