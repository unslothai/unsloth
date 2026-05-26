// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import {
  thinkEffortAriaLabel,
  thinkToggleAriaLabel,
} from "@/components/assistant-ui/think-aria-label";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { applyQwenThinkingParams } from "@/features/chat/utils/qwen-params";
import type { ModelInventoryFormat } from "@/features/inventory";
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE, fileToBase64 } from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { modelShortName } from "@/lib/format";
import {
  ggufVariantsMatch,
  modelIdsMatch,
} from "./model-config/model-identity";
import type { PerModelConfig } from "./model-config/per-model-config";
import { hydrateRuntimeFromLoadResponse } from "./model-runtime/load-hydration";
import { loadedRuntimeConfigMatches } from "./model-runtime/per-model-load-config";
import {
  shouldLoadFromLocalFilesOnly,
  type LocalModelLoadSource,
} from "./model-runtime/local-files-only";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import { useAui } from "@assistant-ui/react";
import {
  ArrowUpIcon,
  DownloadIcon,
  GlobeIcon,
  HeadphonesIcon,
  LightbulbIcon,
  LightbulbOffIcon,
  MicIcon,
  PlusIcon,
  SquareIcon,
  XIcon,
} from "lucide-react";
import { Image03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "@/lib/toast";
import { getHfToken } from "@/stores/hf-token-store";
import { loadModel, validateModel } from "./api/chat-api";
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
  waitForRunEnd: (signal?: AbortSignal) => Promise<void>;
}

const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp,image/gif";
const MAX_IMAGE_SIZE = 20 * 1024 * 1024;

function isNativeComposing(event: Event) {
  return "isComposing" in event && (event as InputEvent).isComposing === true;
}

function createAbortError(): DOMException {
  return new DOMException("Aborted", "AbortError");
}

function isAbortError(err: unknown): boolean {
  return (err as { name?: string } | null)?.name === "AbortError";
}

function hasRunningThreads(
  state = useChatRuntimeStore.getState(),
): boolean {
  return Object.keys(state.runningByThreadId).length > 0;
}

async function startRunAndWait(
  handle: CompareHandle,
  controller: AbortController,
): Promise<void> {
  if (controller.signal.aborted) throw createAbortError();
  const done = handle.waitForRunEnd(controller.signal);
  try {
    handle.startRun();
  } catch (err) {
    controller.abort();
    await done.catch(() => undefined);
    throw err;
  }
  await done;
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
      (window.SpeechRecognition ?? (window as unknown as { webkitSpeechRecognition?: typeof SpeechRecognition }).webkitSpeechRecognition);
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
    !!(window.SpeechRecognition ?? (window as unknown as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition);

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
    const abortWaiters = new Set<() => void>();
    currentHandles[name] = {
      // fixes occasional reorder on reload.
      append: (content) =>
        aui.thread().append({ role: "user", content, createdAt: new Date() } as never),
      appendMessage: (content) =>
        aui.thread().append({ role: "user", content, createdAt: new Date(), startRun: false } as never),
      startRun: () => {
        const msgs = aui.thread().getState().messages;
        const lastId = msgs.length > 0 ? msgs[msgs.length - 1].id : null;
        aui.thread().startRun({ parentId: lastId });
      },
      cancel: () => aui.thread().cancelRun(),
      isRunning: () => aui.thread().getState().isRunning,
      waitForRunEnd: (signal) =>
        new Promise<void>((resolve, reject) => {
          if (signal?.aborted) {
            reject(createAbortError());
            return;
          }

          let wasRunning = hasRunningThreads();
          let settled = false;
          let unsub: (() => void) | null = null;

          function cleanup() {
            unsub?.();
            unsub = null;
            signal?.removeEventListener("abort", rejectAbort);
            abortWaiters.delete(rejectAbort);
          }

          function settle(finish: () => void) {
            if (settled) return;
            settled = true;
            cleanup();
            finish();
          }

          function rejectAbort() {
            settle(() => reject(createAbortError()));
          }

          function check(state = useChatRuntimeStore.getState()) {
            const anyRunning = hasRunningThreads(state);
            if (anyRunning) {
              wasRunning = true;
              return;
            }
            if (wasRunning) {
              settle(() => resolve());
            }
          }

          abortWaiters.add(rejectAbort);
          signal?.addEventListener("abort", rejectAbort, { once: true });
          unsub = useChatRuntimeStore.subscribe((state) => check(state));
          check();
        }),
    };
    return () => {
      for (const abort of Array.from(abortWaiters)) abort();
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
  if (!src) return <div className="size-14 animate-pulse rounded-[14px] bg-muted" />;
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
  modelFormat?: ModelInventoryFormat | null;
  isDownloaded?: boolean;
  isPartial?: boolean;
  preferLocalCache?: boolean;
  localPath?: string | null;
  expectedBytes?: number;
  config?: PerModelConfig;
  source?: LocalModelLoadSource;
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
  const [comparing, setComparing] = useState(false);
  const [pendingImages, setPendingImages] = useState<PendingImage[]>([]);
  const [pendingAudio, setPendingAudio] = useState<{ name: string; base64: string } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composingRef = useRef(false);
  const stuckImeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);
  const compareAbortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  const running = useChatRuntimeStore((s) =>
    Object.keys(s.runningByThreadId).length > 0,
  );
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
  const supportsPreserveThinking = useChatRuntimeStore((s) => s.supportsPreserveThinking);
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
  const clearPendingAudioStore = useChatRuntimeStore((s) => s.clearPendingAudio);

  const { isDictating, start: startDictation, stop: stopDictation, supported: dictationSupported } = useDictation(
    setText,
  );

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      compareAbortRef.current?.abort();
      compareAbortRef.current = null;
    };
  }, []);

  // Auto-expand textarea up to 6 rows, then scroll (matches regular chat composer).
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    const styles = window.getComputedStyle(ta);
    const lineHeight = parseFloat(styles.lineHeight) || 20;
    const paddingY = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
    const borderY = parseFloat(styles.borderTopWidth) + parseFloat(styles.borderBottomWidth);
    const maxHeight = lineHeight * 6 + paddingY + borderY;
    const next = Math.min(ta.scrollHeight, maxHeight);
    ta.style.height = `${next}px`;
    ta.style.overflowY = ta.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [text]);

  const addFiles = useCallback((files: FileList | null) => {
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
  }, [setPendingAudioStore, attachUnavailableReason]);

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
      // here. Compare mode defers — each ensureModelLoaded below sets
      // loadedIsMultimodal for its side, and the chat-adapter's
      // pre-stream gate runs per-side against that fresh state.
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
      const fallbackMaxSeqLength = store.params.maxSeqLength;

      function resolveSelectionLoad(sel: CompareModelSelection): {
        maxSeqLength: number;
        trustRemoteCode: boolean;
        chatTemplateOverride: string | null;
        cacheTypeKv: string | null;
        speculativeType: string | null;
        specDraftNMax: number | null;
        customContextLength: number | null;
      } {
        const config = sel.config;
        const trustRemoteCode = config?.trustRemoteCode ?? false;
        const trimmedTemplate = config?.chatTemplateOverride?.trim();
        const chatTemplateOverride =
          trimmedTemplate && trimmedTemplate.length > 0
            ? (config?.chatTemplateOverride ?? null)
            : null;
        const lowerSelectionId = sel.id.toLowerCase();
        const isGgufLoad =
          sel.modelFormat === "gguf" ||
          (sel.ggufVariant ?? null) !== null ||
          lowerSelectionId.endsWith(".gguf") ||
          lowerSelectionId.startsWith("ollama-manifest:");
        const customContextLength = config?.customContextLength ?? null;
        const maxSeqLength =
          isGgufLoad && customContextLength != null
            ? customContextLength
            : isGgufLoad
              ? 0
              : fallbackMaxSeqLength;
        return {
          maxSeqLength,
          trustRemoteCode,
          chatTemplateOverride,
          cacheTypeKv: config?.kvCacheDtype ?? null,
          speculativeType: config?.speculativeType ?? null,
          specDraftNMax: config?.specDraftNMax ?? null,
          customContextLength,
        };
      }

      function runtimeMatchesSelectionLoad(sel: CompareModelSelection): boolean {
        return loadedRuntimeConfigMatches({
          state: useChatRuntimeStore.getState(),
          modelId: sel.id,
          ggufVariant: sel.ggufVariant ?? null,
          config: sel.config,
        });
      }

      function applySelectionLoadToRuntime(
        load: ReturnType<typeof resolveSelectionLoad>,
      ): void {
        const current = useChatRuntimeStore.getState();
        useChatRuntimeStore.setState({
          kvCacheDtype: load.cacheTypeKv,
          speculativeType: load.speculativeType,
          specDraftNMax: load.specDraftNMax,
          customContextLength: load.customContextLength,
          chatTemplateOverride: load.chatTemplateOverride,
        });
        current.setParams({
          ...current.params,
          trustRemoteCode: load.trustRemoteCode,
        });
      }

      async function ensureModelLoaded(sel: CompareModelSelection): Promise<string> {
        const currentStore = useChatRuntimeStore.getState();
        const isAlreadyActive =
          modelIdsMatch(currentStore.params.checkpoint, sel.id) &&
          ggufVariantsMatch(currentStore.activeGgufVariant, sel.ggufVariant);
        const load = resolveSelectionLoad(sel);
        const shouldLoad = !isAlreadyActive || !runtimeMatchesSelectionLoad(sel);
        const localFilesOnly = shouldLoadFromLocalFilesOnly({
          modelId: sel.id,
          isCachedLora:
            sel.isLora && (sel.source !== "hub" || Boolean(sel.localPath)),
          selection: sel,
        });
        const localPath = localFilesOnly ? (sel.localPath ?? null) : null;
        if (!shouldLoad) {
          applySelectionLoadToRuntime(load);
          return "already loaded";
        }
        const stateBeforeLoad = useChatRuntimeStore.getState();
        if (shouldLoad) {
          const validation = await validateModel({
            model_path: sel.id,
            hf_token: getHfToken() || null,
            max_seq_length: load.maxSeqLength,
            load_in_4bit: true,
            is_lora: sel.isLora,
            gguf_variant: sel.ggufVariant ?? null,
            model_format: sel.modelFormat ?? null,
            local_files_only: localFilesOnly,
            local_path: localPath,
            trust_remote_code: load.trustRemoteCode,
            chat_template_override: load.chatTemplateOverride,
            cache_type_kv: load.cacheTypeKv,
            speculative_type: load.speculativeType,
            spec_draft_n_max: load.specDraftNMax,
          });
          if (validation.requires_trust_remote_code && !load.trustRemoteCode) {
            throw new Error(
              `${modelShortName(sel.id)} needs custom code enabled to load. Turn on "Enable custom code" in Chat Settings, then try again.`,
            );
          }
        }
        const resp = await loadModel({
          model_path: sel.id,
          hf_token: getHfToken() || null,
          max_seq_length: load.maxSeqLength,
          load_in_4bit: true,
          is_lora: sel.isLora,
          gguf_variant: sel.ggufVariant ?? null,
          model_format: sel.modelFormat ?? null,
          local_files_only: localFilesOnly,
          local_path: localPath,
          trust_remote_code: load.trustRemoteCode,
          chat_template_override: load.chatTemplateOverride,
          cache_type_kv: load.cacheTypeKv,
          speculative_type: load.speculativeType,
          spec_draft_n_max: load.specDraftNMax,
        });
        hydrateRuntimeFromLoadResponse({
          response: resp,
          modelId: sel.id,
          ggufVariant: sel.ggufVariant ?? null,
          requestedConfig: {
            chatTemplateOverride: load.chatTemplateOverride,
            customContextLength: load.customContextLength,
          },
          stateBeforeLoad,
          reloadingSameModel: isAlreadyActive,
        });
        // Sync the models[] entry with the load response so the
        // attach/send gates read fresh capabilities. /api/models/list
        // can lag behind a model's actual state (e.g., a GGUF whose
        // mmproj was downloaded after the catalog snapshot).
        const store = useChatRuntimeStore.getState();
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

      const name1 = model1?.id ? modelShortName(model1.id) : "";
      const name2 = model2?.id ? modelShortName(model2.id) : "";
      const toastId = toast("Comparing models…", { duration: Infinity });
      const compareController = new AbortController();
      compareAbortRef.current?.abort();
      compareAbortRef.current = compareController;

      setComparing(true);
      try {
        // Side 1: load → generate → wait
        if (handle1 && model1?.id) {
          toast("Loading Model 1…", { id: toastId, description: name1, duration: Infinity });
          const status1 = await ensureModelLoaded(model1);
          toast("Generating with Model 1…", { id: toastId, description: `${name1} (${status1})`, duration: Infinity });
          await startRunAndWait(handle1, compareController);
        }

        // Side 2: load → generate → wait
        if (handle2 && model2?.id) {
          const needsLoad =
            !modelIdsMatch(model2.id, model1?.id) ||
            !ggufVariantsMatch(model2.ggufVariant, model1?.ggufVariant);
          if (needsLoad) {
            toast("Loading Model 2…", { id: toastId, description: name2, duration: Infinity });
          }
          const status2 = await ensureModelLoaded(model2);
          toast("Generating with Model 2…", { id: toastId, description: `${name2} (${status2})`, duration: Infinity });
          await startRunAndWait(handle2, compareController);
        }

        toast.success("Compare complete", { id: toastId, duration: 2000 });
      } catch (err) {
        if (isAbortError(err)) {
          toast.dismiss(toastId);
        } else {
          toast.error("Compare failed", {
            id: toastId,
            description: err instanceof Error ? err.message : "Unknown error",
            duration: 4000,
          });
        }
      } finally {
        if (compareAbortRef.current === compareController) {
          compareAbortRef.current = null;
        }
        compareController.abort();
        if (mountedRef.current) setComparing(false);
      }
    } else {
      // Original behavior: fire all handles simultaneously
      for (const handle of Object.values(handlesRef.current)) {
        handle.append(content);
      }
    }
  }

  function stop() {
    compareAbortRef.current?.abort();
    if (isDictating) stopDictation();
    for (const handle of Object.values(handlesRef.current)) {
      handle.cancel();
    }
  }

  const busy = running || comparing;

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
      if (!busy) {
        send();
      }
    }
  }

  const canSend = (text.trim().length > 0 || pendingImages.length > 0 || pendingAudio !== null) && !busy && !isComposing;

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
                onClick={() => { setPendingAudio(null); clearPendingAudioStore(); }}
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
        onKeyDown={onKeyDown}
        placeholder="Send to both models..."
        className="composer-input"
        rows={1}
        // dir="auto" auto-detects RTL (Arabic / Hebrew / Persian / Urdu)
        // from the first strong character; no effect on LTR scripts.
        dir="auto"
      />
      <div className="composer-action-wrapper">
        <div className="flex items-center gap-0.5">
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
          <TooltipIconButton
            tooltip="Add Attachment"
            side="bottom"
            variant="ghost"
            size="icon"
            className="size-8.5 rounded-full p-1 font-semibold text-xs hover:bg-muted-foreground/15 dark:border-muted-foreground/15 dark:hover:bg-muted-foreground/30"
            onClick={() => {
              // The picker accepts both image and audio. Don't gate the
              // button on image-availability — addFiles still filters
              // image files per-file when the loaded model can't take
              // them, while audio attach always works.
              fileInputRef.current?.click();
            }}
            aria-label="Add Attachment"
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
                preserveThinking ? "Disable preserve think" : "Enable preserve think"
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
              tooltip="Send message"
              side="bottom"
              variant="default"
              size="icon"
              className="size-8 rounded-full"
              onClick={send}
              disabled={!canSend}
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
