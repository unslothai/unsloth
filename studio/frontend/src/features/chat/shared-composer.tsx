// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { applyQwenThinkingParams } from "@/features/chat/utils/qwen-params";
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE, fileToBase64 } from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { useAui } from "@assistant-ui/react";
import { ArrowUpIcon, GlobeIcon, HeadphonesIcon, LightbulbIcon, LightbulbOffIcon, MicIcon, PlusIcon, SquareIcon, XIcon } from "lucide-react";
import { toast } from "sonner";
import { loadModel, validateModel } from "./api/chat-api";
import { parseExternalModelId } from "./external-providers";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import {
  type ReasoningEffort,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import { getExternalReasoningCapabilities } from "./provider-capabilities";
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
  waitForRunEnd: () => Promise<void>;
}

const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp,image/gif";
const MAX_IMAGE_SIZE = 20 * 1024 * 1024;

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
  const [pendingAudio, setPendingAudio] = useState<{ name: string; base64: string } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composingRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  const activeModel = useChatRuntimeStore((s) => {
    const checkpoint = s.params.checkpoint;
    return s.models.find((m) => m.id === checkpoint);
  });
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const externalProviders = useExternalProvidersStore((s) => s.providers);
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
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
  const lastOpenRouterChosenModel = useChatRuntimeStore(
    (s) => s.lastOpenRouterChosenModel,
  );
  const externalSelection = parseExternalModelId(checkpoint);
  const selectedExternalProvider =
    externalSelection != null
      ? externalProviders.find((p) => p.id === externalSelection.providerId)
      : undefined;
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
  // (supportsBuiltinWebSearch, currently just OpenAI's /v1/responses).
  // Code pill is gated on `supportsTools` only — external providers
  // never give us code execution, so the pill must stay disabled even
  // when Search is available.
  const searchDisabled =
    !modelLoaded || !(supportsTools || supportsBuiltinWebSearch);
  const codeDisabled = !modelLoaded || !supportsTools;
  // Backwards-compatible alias for any other call site that may still
  // reference `toolsDisabled` (rare; both pills used it before).
  const toolsDisabled = codeDisabled;
  const setPendingAudioStore = useChatRuntimeStore((s) => s.setPendingAudio);
  const clearPendingAudioStore = useChatRuntimeStore((s) => s.clearPendingAudio);

  const { isDictating, start: startDictation, stop: stopDictation, supported: dictationSupported } = useDictation(
    setText,
  );

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
      next.push({ id: crypto.randomUUID(), file });
    }
    setPendingImages((prev) => [...prev, ...next]);
  }, [setPendingAudioStore]);

  const removePendingImage = useCallback((id: string) => {
    setPendingImages((prev) => prev.filter((p) => p.id !== id));
  }, []);

  function setCompositionState(next: boolean) {
    composingRef.current = next;
    setIsComposing(next);
  }

  async function send() {
    if (composingRef.current) return;
    const msg = text.trim();
    if (!msg && pendingImages.length === 0 && !pendingAudio) return;

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
    const hasCompareHandles = Boolean(handlesRef.current["model1"] || handlesRef.current["model2"]);
    const isGeneralizedCompare = hasCompareHandles && Boolean(model1?.id || model2?.id);
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
      async function ensureModelLoaded(sel: CompareModelSelection): Promise<string> {
        const currentStore = useChatRuntimeStore.getState();
        const isAlreadyActive =
          currentStore.params.checkpoint === sel.id &&
          (currentStore.activeGgufVariant ?? null) === (sel.ggufVariant ?? null);
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
          toast("Loading Model 1…", { id: toastId, description: name1, duration: Infinity });
          const status1 = await ensureModelLoaded(model1);
          toast("Generating with Model 1…", { id: toastId, description: `${name1} (${status1})`, duration: Infinity });
          const done = handle1.waitForRunEnd();
          handle1.startRun();
          await done;
        }

        // Side 2: load → generate → wait
        if (handle2 && model2?.id) {
          const needsLoad = model2.id.toLowerCase() !== (model1?.id || "").toLowerCase()
            || (model2.ggufVariant ?? "") !== (model1?.ggufVariant ?? "");
          if (needsLoad) {
            toast("Loading Model 2…", { id: toastId, description: name2, duration: Infinity });
          }
          const status2 = await ensureModelLoaded(model2);
          toast("Generating with Model 2…", { id: toastId, description: `${name2} (${status2})`, duration: Infinity });
          const done = handle2.waitForRunEnd();
          handle2.startRun();
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
        handle.append(content);
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

  function onKeyDown(e: KeyboardEvent) {
    // IME composition (Japanese/Chinese/Korean): Enter commits the candidate.
    // Don't hijack it. See issue #5318.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
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
        onCompositionEnd={(e: CompositionEvent<HTMLTextAreaElement>) => {
          setCompositionState(false);
          setText(e.currentTarget.value);
        }}
        onKeyDown={onKeyDown}
        placeholder="Send to both models..."
        className="composer-input"
        rows={1}
      />
      <div className="composer-action-wrapper">
        <div className="flex items-center gap-1">
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
            onClick={() => fileInputRef.current?.click()}
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
                    "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
                    reasoningDisabled
                      ? "cursor-not-allowed opacity-40"
                      : effectiveReasoningVisualEnabled
                        ? "bg-primary/10 text-primary hover:bg-primary/20"
                        : "text-muted-foreground hover:bg-muted-foreground/15",
                  )}
                  aria-label={`Reasoning effort: ${reasoningEffort}`}
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
                        setToolsEnabled(false);
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
                  setToolsEnabled(false);
                }
              }}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
                reasoningLockedOn
                  ? "cursor-not-allowed bg-primary/10 text-primary"
                  : reasoningDisabled
                    ? "cursor-not-allowed opacity-40"
                    : effectiveReasoningEnabled
                      ? "bg-primary/10 text-primary hover:bg-primary/20"
                      : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
              )}
              aria-label={
                reasoningLockedOn
                  ? "Thinking is required for this model"
                  : effectiveReasoningEnabled
                    ? "Disable thinking"
                    : "Enable thinking"
              }
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
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
                !modelLoaded
                  ? "cursor-not-allowed opacity-40"
                  : preserveThinking
                    ? "bg-primary/10 text-primary hover:bg-primary/20"
                    : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
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
                setReasoningEnabled(!next);
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
