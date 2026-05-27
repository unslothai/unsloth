// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
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
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE, fileToBase64 } from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { isMultimodalResponse } from "./types/api";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import { useAui } from "@assistant-ui/react";
import {
  ArrowUpIcon,
  DownloadIcon,
  FileText,
  GlobeIcon,
  HeadphonesIcon,
  LightbulbIcon,
  LightbulbOffIcon,
  LoaderIcon,
  MicIcon,
  PlusIcon,
  RefreshCwIcon,
  SquareIcon,
  XIcon,
} from "lucide-react";
import { Image03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "@/lib/toast";
import { parseExternalModelId, providerTypeSupportsVision } from "./external-providers";
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

function isNativeComposing(event: Event) {
  return "isComposing" in event && (event as InputEvent).isComposing === true;
}

// Mirrors the threshold in thread.tsx — see the comment there. Chrome on
// Windows-over-WSL (issue #5546) never fires `compositionend` after the
// IME commit, so the compose flag would otherwise stay true forever.
const IME_STUCK_TIMEOUT_MS = 2500;

function fileToBase64DataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read image file"));
    reader.readAsDataURL(file);
  });
}

function formatReasoningEffortLabel(level: ReasoningEffort, modelId?: string): string {
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
  // Magistral keeps the "none" wire value, but UX should present this floor
  // as "Medium" rather than a disabled state label.
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
        aui.thread().append({
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

export function SharedComposer({
  handlesRef,
  model1,
  model2,
}: {
  handlesRef: CompareHandles;
  model1?: CompareModelSelection;
  model2?: CompareModelSelection;
}): ReactElement {
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
  const supportsReasoningOff = useChatRuntimeStore((s) => s.supportsReasoningOff);
  const reasoningEffortLevels = useChatRuntimeStore((s) => s.reasoningEffortLevels);
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
  const webFetchToolsEnabled = useChatRuntimeStore(
    (s) => s.webFetchToolsEnabled,
  );
  const setWebFetchToolsEnabled = useChatRuntimeStore(
    (s) => s.setWebFetchToolsEnabled,
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
  // Attach-time gate. Compare mode defers to send: the catalog can lag
  // behind a model's real capabilities (e.g., a GGUF whose mmproj
  // arrives after the catalog snapshot), and we only sync the models[]
  // entry after ensureModelLoaded runs at send time. Single mode uses
  // the loaded model's runtime capability.
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
  // Kimi's $web_search builtin mandates thinking=disabled per the docs at
  // https://platform.kimi.ai/docs/guide/use-web-search. Both pills stay
  // clickable for Kimi, but turning one on flips the other off — the
  // click handlers below enforce this mutual exclusion so the visible
  // state always matches what the backend actually sends.
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";
  const effectiveReasoningEnabled = reasoningLockedOn ? true : reasoningEnabled;
  const effectiveReasoningVisualEnabled =
    effectiveReasoningEnabled && reasoningEffort !== "none";
  const reasoningDisabled = !modelLoaded || !effectiveSupportsReasoning;
  const showReasoningControl =
    effectiveSupportsReasoning || effectiveReasoningAlwaysOn;
  // Two-pill gating: Search pill lights up when the runtime has either
  // a local tool runtime (supportsTools, gives us our Code/python + local
  // web_search) OR a server-side web_search the provider runs for us
  // (supportsBuiltinWebSearch, currently OpenAI / Anthropic / OpenRouter
  // / Kimi). Code pill lights up on the local runtime OR when Anthropic
  // is selected with a model that accepts the server-side
  // code_execution_20250825 tool — see
  // providerSupportsBuiltinCodeExecution. Anthropic is the only external
  // provider that ships a code-execution tool today.
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
  const searchDisabled =
    !modelLoaded || !(supportsTools || supportsBuiltinWebSearch);
  const codeDisabled =
    !modelLoaded || !(supportsTools || supportsBuiltinCodeExecution);
  // Images pill is only ever lit on OpenAI cloud's Responses-API models.
  // No local tool runtime fallback because the only image-generation
  // server tool we wire today is OpenAI's; local models cannot dispatch
  // it. Hidden entirely when the active model does not advertise it so
  // the pill row stays compact for providers without the capability.
  const imageDisabled = !modelLoaded || !supportsBuiltinImageGeneration;
  const showImagePill = supportsBuiltinImageGeneration;
  // Fetch pill: Anthropic-only (web_fetch_20250910 / web_fetch_20260209).
  const webFetchDisabled = !modelLoaded || !supportsBuiltinWebFetch;
  const showWebFetchPill = supportsBuiltinWebFetch;
  // Backwards-compatible alias for any other call site that may still
  // reference `toolsDisabled` (rare; both pills used it before).
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
        // Handle document files (PDF / DOCX / MD / HTML)
        if (isDocumentFile(file)) {
          void uploadDocument(file);
          continue;
        }
        // Unsupported file type
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
    // Abort any in-flight extraction for this doc
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

    // Generalized compare requires both panes to have a model. A
    // half-selected send either races to an empty bubble with bogus
    // tok/s (#5569) or leaves the empty pane with a dangling prompt.
    // hasCompareHandles is true only in GeneralCompareContent, so
    // LoraCompare and single-pane chats are unaffected.
    if (hasCompareHandles && !isGeneralizedCompare) {
      toast.error("Pick a model in each pane to compare", {
        description: "Use the model dropdown above each pane, then send your prompt.",
      });
      return;
    }

    if (pendingImages.length > 0 && !isGeneralizedCompare && imageUnavailableReason) {
      // Single mode: the loaded model's runtime capability is known
      // here. Compare mode defers - each ensureModelLoaded below sets
      // loadedIsMultimodal for its side, and the chat-adapter's
      // pre-stream gate runs per-side against that fresh state.
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
      const effectiveChatTemplateOverride =
        chatTemplateOverride?.trim() ? chatTemplateOverride : null;

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
        // Sync the models[] entry with the load response so the
        // attach/send gates read fresh capabilities. /api/models/list
        // can lag behind a model's actual state (e.g., a GGUF whose
        // mmproj was downloaded after the catalog snapshot).
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

        toast.success("Compare complete", { id: toastId, duration: 2000 });
      } catch (err) {
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
    // IME composition (Japanese/Chinese/Korean): Enter commits the candidate.
    // Don't hijack it. See issue #5318. Re-pin composingRef in case the stuck
    // watchdog (#5546) cleared it during a long candidate-window pause; this
    // keeps a follow-up click-Send from submitting preedit text. Re-arm the
    // watchdog on the same path — without it the WSL+Chrome no-compositionend
    // case would leave composingRef pinned forever after an IME keypress and
    // re-lock Send.
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
      className={`chat-composer-surface ${dragging ? "border-ring bg-accent/50" : ""}`}
      onDragOver={(e) => {
        if (isTauri) return;
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        // Phase 1 native model drops own Tauri local-path drops. Restore browser
        // attachment drops in Tauri when Phase 1d adds attachment-token bridging.
        if (isTauri) return;
        e.preventDefault();
        setDragging(false);
        addFiles(e.dataTransfer.files);
      }}
    >
      {dragging ? (
        <div
          className="pointer-events-none absolute inset-1 z-10 flex items-center justify-center rounded-2xl border-2 border-dashed border-ring bg-background/90 text-sm font-medium text-foreground shadow-sm"
          role="region"
          aria-label="Drop to extract document"
        >
          Drop to extract document
        </div>
      ) : null}
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
          // composition. The controlled `value` prop must match the DOM at
          // all times, otherwise any unrelated parent re-render reconciles
          // the textarea back to the stored value mid-composition — wiping
          // the IME preedit AND prior committed text (e.g. Tab cycling
          // candidates erases earlier words). Issue #5318.
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
        // dir="auto" auto-detects RTL (Arabic / Hebrew / Persian / Urdu)
        // from the first strong character; no effect on LTR scripts.
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
        <div className="flex items-center gap-0.5">
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
          <TooltipIconButton
            tooltip="Add files"
            side="bottom"
            variant="ghost"
            size="icon"
            className="size-8.5 rounded-full p-1 font-semibold text-xs hover:bg-muted-foreground/15 dark:border-muted-foreground/15 dark:hover:bg-muted-foreground/30"
            onClick={() => {
              // The picker accepts images, audio, and documents. Don't gate
              // the button on image-availability - addFiles still filters
              // image files per-file when the loaded model can't take
              // them, while audio and documents always work.
              fileInputRef.current?.click();
            }}
            aria-label="Add files"
          >
            <PlusIcon className="size-5 stroke-[1.5px]" />
          </TooltipIconButton>
          {activeModel?.hasAudioInput && (
            <>
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
              <TooltipIconButton
                tooltip="Upload audio"
                side="bottom"
                variant="ghost"
                size="icon"
                className="size-8.5 rounded-full p-1 text-muted-foreground hover:bg-muted-foreground/15"
                onClick={() => audioInputRef.current?.click()}
                aria-label="Upload audio"
              >
                <HeadphonesIcon className="size-4.5 stroke-[1.5px]" />
              </TooltipIconButton>
            </>
          )}
          {showReasoningControl ? (
            effectiveReasoningStyle === "reasoning_effort" ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <button
                  type="button"
                  disabled={reasoningDisabled}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full px-1.5 py-1.5 text-[13px] font-medium text-muted-foreground/70 transition-colors",
                    reasoningDisabled
                      ? "cursor-not-allowed opacity-40"
                      : effectiveReasoningVisualEnabled
                        ? "text-primary hover:bg-primary/10 dark:hover:bg-white/[0.08]"
                        : "hover:bg-primary/10 dark:hover:bg-white/[0.08]",
                  )}
                  aria-label={thinkEffortAriaLabel({
                    modelLoaded,
                    reasoningDisabled,
                    reasoningEffort,
                  })}
                >
                  {effectiveReasoningVisualEnabled ? (
                    <LightbulbIcon className="size-3.5" />
                  ) : (
                    <LightbulbOffIcon className="size-3.5" />
                  )}
                  <span>
                    Think:{" "}
                    {effectiveReasoningVisualEnabled
                      ? formatReasoningEffortLabel(
                          reasoningEffort,
                          externalSelection?.modelId,
                        )
                      : formatReasoningDisabledLabel(
                          effectiveSupportsReasoningOff,
                          isExternalOpenAIReasoning,
                          checkpoint,
                        )}
                  </span>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {effectiveSupportsReasoningOff && (
                  <DropdownMenuItem
                    onSelect={() => {
                      setReasoningEnabled(false);
                      applyQwenThinkingParams(false);
                    }}
                  >
                    {formatReasoningDisabledLabel(
                      effectiveSupportsReasoningOff,
                      isExternalOpenAIReasoning,
                      checkpoint,
                    )}
                    {!effectiveReasoningVisualEnabled ? " \u2713" : ""}
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
                    {formatReasoningEffortLabel(level, externalSelection?.modelId)}
                    {effectiveReasoningVisualEnabled && reasoningEffort === level ? " \u2713" : ""}
                  </DropdownMenuItem>
                ))}
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
                // Mutual exclusion: Kimi's $web_search builtin
                // requires thinking off, so turning thinking on flips
                // the Search pill off (and vice versa).
                if (isKimiExternal && next && toolsEnabled) {
                  setToolsEnabled(false, { persist: false });
                }
              }}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-1.5 py-1.5 text-[13px] font-medium text-muted-foreground/70 transition-colors",
                reasoningLockedOn
                  ? "cursor-not-allowed text-primary"
                  : reasoningDisabled
                    ? "cursor-not-allowed opacity-40"
                    : effectiveReasoningEnabled
                      ? "text-primary hover:bg-primary/10 dark:hover:bg-white/[0.08]"
                      : "hover:bg-primary/10 dark:hover:bg-white/[0.08]",
              )}
              aria-label={thinkToggleAriaLabel({
                reasoningLockedOn,
                modelLoaded,
                reasoningDisabled,
                effectiveReasoningEnabled,
              })}
            >
              {reasoningLockedOn ||
              (effectiveReasoningEnabled && !reasoningDisabled) ? (
                <LightbulbIcon className="size-3.5" />
              ) : (
                <LightbulbOffIcon className="size-3.5" />
              )}
              <span>Think</span>
            </button>
            )
          ) : null}
          {supportsPreserveThinking && (
            <button
              type="button"
              disabled={!modelLoaded}
              onClick={() => setPreserveThinking(!preserveThinking)}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-1.5 py-1.5 text-[13px] font-medium text-muted-foreground/70 transition-colors",
                !modelLoaded
                  ? "cursor-not-allowed opacity-40"
                  : preserveThinking
                    ? "text-primary hover:bg-primary/10 dark:hover:bg-white/[0.08]"
                    : "hover:bg-primary/10 dark:hover:bg-white/[0.08]",
              )}
              aria-label={
                preserveThinking
                  ? "Disable preserve thinking"
                  : "Enable preserve thinking"
              }
            >
              {preserveThinking && modelLoaded ? (
                <LightbulbIcon className="size-3.5" />
              ) : (
                <LightbulbOffIcon className="size-3.5" />
              )}
              <span>Preserve Think</span>
            </button>
          )}
          <button
            type="button"
            disabled={searchDisabled}
            onClick={() => {
              const next = !toolsEnabled;
              setToolsEnabled(next);
              // Kimi's $web_search builtin requires thinking=disabled
              // (https://platform.kimi.ai/docs/guide/use-web-search).
              // Toggle the Think pill off when Search comes on, and
              // back on when Search goes off — mutual exclusion that
              // mirrors what the backend enforces.
              if (isKimiExternal) {
                setReasoningEnabled(!next, { persist: false });
                applyQwenThinkingParams(!next);
              }
            }}
            className="composer-pill-btn"
            data-active={toolsEnabled && !searchDisabled ? "true" : "false"}
            aria-label={toolsEnabled ? "Disable web search" : "Enable web search"}
          >
            <GlobeIcon className="size-3.5" />
            <span>Search</span>
          </button>
          <button
            type="button"
            disabled={codeDisabled}
            onClick={() => setCodeToolsEnabled(!codeToolsEnabled)}
            className="composer-pill-btn"
            data-active={codeToolsEnabled && !codeDisabled ? "true" : "false"}
            aria-label={codeToolsEnabled ? "Disable code execution" : "Enable code execution"}
          >
            <CodeToggleIcon className="size-3.5" />
            <span>Code</span>
          </button>
          {showImagePill && (
            <button
              type="button"
              disabled={imageDisabled}
              onClick={() => setImageToolsEnabled(!imageToolsEnabled)}
              className="composer-pill-btn"
              data-active={imageToolsEnabled && !imageDisabled ? "true" : "false"}
              aria-label={
                imageToolsEnabled ? "Disable image generation" : "Enable image generation"
              }
            >
              <HugeiconsIcon
                icon={Image03Icon}
                className="size-3.5"
                strokeWidth={2}
              />
              <span>Images</span>
            </button>
          )}
          {showWebFetchPill && (
            <button
              type="button"
              disabled={webFetchDisabled}
              onClick={() => setWebFetchToolsEnabled(!webFetchToolsEnabled)}
              className="composer-pill-btn"
              data-active={
                webFetchToolsEnabled && !webFetchDisabled ? "true" : "false"
              }
              aria-label={
                webFetchToolsEnabled ? "Disable URL fetch" : "Enable URL fetch"
              }
            >
              <DownloadIcon className="size-3.5" />
              <span>Fetch</span>
            </button>
          )}
        </div>
        <div className="flex items-center gap-1">
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
          {busy ? (
            <Button
              type="button"
              variant="default"
              size="icon"
              className="size-8 rounded-full"
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
                "size-8 rounded-full",
                !canSend && "cursor-not-allowed opacity-50",
              )}
              onClick={() => {
                if (canSend) void send();
              }}
              disabled={!canSend}
              aria-disabled={!canSend}
              aria-label="Send message"
            >
              <ArrowUpIcon className="size-4" />
            </TooltipIconButton>
          )}
        </div>
      </div>
    </div>
  );
}
