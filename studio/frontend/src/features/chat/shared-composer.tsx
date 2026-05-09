// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
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
import { useAui } from "@assistant-ui/react";
import {
  ArrowUpIcon,
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
import { toast } from "sonner";
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
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
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

function fileToBase64DataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read image file"));
    reader.readAsDataURL(file);
  });
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  const activeModel = useChatRuntimeStore((s) => {
    const checkpoint = s.params.checkpoint;
    return s.models.find((m) => m.id === checkpoint);
  });
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const modelBusy = modelLoading || temporaryOcrBusy;
  const supportsReasoning = useChatRuntimeStore((s) => s.supportsReasoning);
  const reasoningAlwaysOn = useChatRuntimeStore((s) => s.reasoningAlwaysOn);
  const reasoningEnabled = useChatRuntimeStore((s) => s.reasoningEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const reasoningStyle = useChatRuntimeStore((s) => s.reasoningStyle);
  const reasoningEffort = useChatRuntimeStore((s) => s.reasoningEffort);
  const setReasoningEffort = useChatRuntimeStore((s) => s.setReasoningEffort);
  const supportsPreserveThinking = useChatRuntimeStore(
    (s) => s.supportsPreserveThinking,
  );
  const preserveThinking = useChatRuntimeStore((s) => s.preserveThinking);
  const setPreserveThinking = useChatRuntimeStore((s) => s.setPreserveThinking);
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const setCodeToolsEnabled = useChatRuntimeStore((s) => s.setCodeToolsEnabled);
  const reasoningDisabled = !modelLoaded || !supportsReasoning;
  const toolsDisabled = !modelLoaded || !supportsTools;
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
    try {
      const doc = await runner.run(file, {
        onProgress: (pct) => {
          const mapped = pct * 0.7;
          setUploadingDocs((prev) =>
            prev.map((item) =>
              item.id === placeholderId
                ? {
                    ...item,
                    progress: Math.max(
                      item.progress ?? 0,
                      Math.min(0.7, mapped),
                    ),
                  }
                : item,
            ),
          );
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
        return;
      }
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
      setPendingImages((prev) => [...prev, ...next]);
    },
    [setPendingAudioStore, uploadDocument],
  );

  const removePendingImage = useCallback((id: string) => {
    setPendingImages((prev) => prev.filter((p) => p.id !== id));
  }, []);

  function setCompositionState(next: boolean) {
    composingRef.current = next;
    setIsComposing(next);
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

  async function send() {
    if (composingRef.current) return;
    if (uploadingDocs.length > 0 || running || comparing || modelBusy) {
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

    // Generalized compare: load each model before dispatching to its side
    const hasCompareHandles = Boolean(
      handlesRef.current["model1"] || handlesRef.current["model2"],
    );
    const isGeneralizedCompare =
      hasCompareHandles && Boolean(model1?.id || model2?.id);
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
        });
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
    !modelBusy &&
    !busy &&
    !isComposing;
  const waitingAttachmentLabel =
    uploadingDocs.length > 0
      ? `Waiting for ${uploadingDocs.length} attachment${
          uploadingDocs.length === 1 ? "" : "s"
        }...`
      : null;

  function onKeyDown(e: KeyboardEvent) {
    // IME composition (Japanese/Chinese/Korean): Enter commits the candidate.
    // Don't hijack it. See issue #5318.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
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
                    {pct !== null ? `Uploading… ${pct}%` : "Reading…"}
                  </span>
                  <AttachmentChipProgress
                    value={pct}
                    label={
                      pct !== null ? `${pct}% uploaded` : `Reading ${doc.name}`
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
      />
      {waitingAttachmentLabel ? (
        <p
          className="px-5 pb-1 text-[11px] text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          {waitingAttachmentLabel}
        </p>
      ) : null}
      <div className="composer-action-wrapper">

        <div className="flex items-center gap-1">
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
            onClick={() => fileInputRef.current?.click()}
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
          {reasoningStyle === "reasoning_effort" ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <button
                  type="button"
                  disabled={reasoningDisabled}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
                    reasoningDisabled
                      ? "cursor-not-allowed opacity-40"
                      : "bg-primary/10 text-primary hover:bg-primary/20",
                  )}
                  aria-label={`Reasoning effort: ${reasoningEffort}`}
                >
                  <LightbulbIcon className="size-3.5" />
                  <span>
                    Think:{" "}
                    {reasoningEffort.charAt(0).toUpperCase() +
                      reasoningEffort.slice(1)}
                  </span>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {(["low", "medium", "high"] as const).map((level) => (
                  <DropdownMenuItem
                    key={level}
                    onSelect={() => setReasoningEffort(level)}
                  >
                    {level.charAt(0).toUpperCase() + level.slice(1)}
                    {reasoningEffort === level ? " \u2713" : ""}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <button
              type="button"
              disabled={reasoningDisabled}
              onClick={() => {
                if (reasoningAlwaysOn) return;
                const next = !reasoningEnabled;
                setReasoningEnabled(next);
                applyQwenThinkingParams(next);
              }}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
                reasoningDisabled
                  ? "cursor-not-allowed opacity-40"
                  : reasoningEnabled || reasoningAlwaysOn
                    ? "bg-primary/10 text-primary hover:bg-primary/20"
                    : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
              )}
              aria-label={
                reasoningEnabled ? "Disable thinking" : "Enable thinking"
              }
            >
              {(reasoningEnabled || reasoningAlwaysOn) && !reasoningDisabled ? (
                <LightbulbIcon className="size-3.5" />
              ) : (
                <LightbulbOffIcon className="size-3.5" />
              )}
              <span>Think</span>
            </button>
          )}
          {supportsPreserveThinking && (
            <button
              type="button"
              disabled={!modelLoaded}
              onClick={() => setPreserveThinking(!preserveThinking)}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
                modelLoaded
                  ? preserveThinking
                    ? "bg-primary/10 text-primary hover:bg-primary/20"
                    : "bg-muted text-muted-foreground hover:bg-muted-foreground/15"
                  : "cursor-not-allowed opacity-40",
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
            disabled={toolsDisabled}
            onClick={() => setToolsEnabled(!toolsEnabled)}
            className="composer-pill-btn"
            data-active={toolsEnabled && !toolsDisabled ? "true" : "false"}
            aria-label={toolsEnabled ? "Disable web search" : "Enable web search"}
          >
            <GlobeIcon className="size-3.5" />
            <span>Search</span>
          </button>
          <button
            type="button"
            disabled={toolsDisabled}
            onClick={() => setCodeToolsEnabled(!codeToolsEnabled)}
            className="composer-pill-btn"
            data-active={codeToolsEnabled && !toolsDisabled ? "true" : "false"}
            aria-label={codeToolsEnabled ? "Disable code execution" : "Enable code execution"}
          >
            <CodeToggleIcon className="size-3.5" />
            <span>Code</span>
          </button>
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
              tooltip={waitingAttachmentLabel ?? "Send message"}
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
              aria-disabled={!canSend}
            >
              <ArrowUpIcon className="size-4" />
            </TooltipIconButton>
          )}
        </div>
      </div>
    </div>
  );
}
