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
import { openLink } from "@/lib/open-link";
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
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE, fileToBase64 } from "@/lib/audio-utils";
import { isTauri } from "@/lib/api-base";
import { isMultimodalResponse } from "./types/api";
import { getImageInputUnavailableReason } from "./utils/image-input-support";
import { useAui } from "@assistant-ui/react";
import {
  ArrowUpIcon,
  CheckIcon,
  Columns2Icon,
  DownloadIcon,
  GlobeIcon,
  HeadphonesIcon,
  PlusIcon,
  SquareIcon,
  XIcon,
} from "lucide-react";
import {
  AttachmentIcon,
  CodeIcon,
  DatabaseIcon,
  Folder01Icon,
  FolderAddIcon,
  Image03Icon,
  McpServerIcon,
  PencilRulerIcon,
} from "@hugeicons/core-free-icons";
import { useNavigate } from "@tanstack/react-router";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "@/lib/toast";
import { ChatMcpServersDialog } from "./chat-mcp-servers-dialog";
import { loadModel, validateModel } from "./api/chat-api";
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
  useRef,
  useState,
} from "react";

// Projects is still in development; its menu entries link to the tracking PR.
const PROJECTS_PR_URL = "https://github.com/unslothai/unsloth/pull/5725";

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

// Inlined to avoid a new icon dependency. Kept in sync with the main composer.
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
};

// Tool icon plus an X overlay the CSS reveals on hover when the pill is active.
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
}: {
  handlesRef: CompareHandles;
  model1?: CompareModelSelection;
  model2?: CompareModelSelection;
  onExitCompare?: () => void;
}): ReactElement {
  const navigate = useNavigate();
  // Exit compare. Uses the parent's restore handler, or a fresh chat when
  // compare was opened by direct URL.
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
  const [mcpOpen, setMcpOpen] = useState(false);
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
  const isEffort = effectiveReasoningStyle === "reasoning_effort";
  const thinkingActiveLook = isEffort
    ? reasoningLockedOn || (effectiveReasoningVisualEnabled && !reasoningDisabled)
    : reasoningLockedOn || (effectiveReasoningEnabled && !reasoningDisabled);
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
  // Gemini rejects codeExecution alongside image modalities. Search is
  // blocked on older Gemini image ids but allowed on Gemini 3 image
  // models -- supportsBuiltinWebSearch already encodes the per-model
  // allowance, so we only disable Code unconditionally in Gemini
  // image mode.
  const isExternalGemini = selectedExternalProvider?.providerType === "gemini";
  const imageDisabled = !modelLoaded || !supportsBuiltinImageGeneration;
  const imageModeDisablesCode =
    isExternalGemini && imageToolsEnabled && !imageDisabled;
  // Image-tier Gemini models always reject codeExecution and reject
  // web_search on older ids (Gemini 3.x Pro/Flash allow it -- encoded
  // in supportsBuiltinWebSearch). Don't let the local `supportsTools`
  // runtime flag re-enable a pill the Gemini backend will silently
  // drop. Detect "external provider is Gemini AND model is image-tier"
  // and gate strictly on the provider builtin support.
  const isGeminiImageTier =
    isExternalGemini && supportsBuiltinImageGeneration;
  // Disable only when a loaded model lacks the capability; with no model the
  // tool can still be pre-selected and reflected, matching the + menu.
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
  // Images pill is only ever lit on OpenAI cloud's Responses-API models
  // and Gemini Nano Banana family. No local tool runtime fallback.
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

    // Generalized compare requires both panes to have a model. A
    // half-selected send either races to an empty bubble with bogus
    // tok/s (#5569) or leaves the empty pane with a dangling prompt.
    // hasCompareHandles is true only in GeneralCompareContent, so
    // LoraCompare and single-pane chats are unaffected.
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

  const canSend =
    (text.trim().length > 0 ||
      pendingImages.length > 0 ||
      pendingAudio !== null) &&
    !busy &&
    !isComposing;

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
        // Phase 1 native model drops own Tauri local-path drops. Restore browser
        // attachment drops in Tauri when Phase 1d adds attachment-token bridging.
        if (isTauri) return;
        e.preventDefault();
        setDragging(false);
        addFiles(e.dataTransfer.files);
      }}
    >
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
          {/* Same + side menu as the single-chat composer (ComposerToolsMenu),
              wired to the compare composer's own file/audio inputs and tools. */}
          <DropdownMenu>
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
              sideOffset={2}
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
                className={toolsEnabled ? "text-primary font-medium" : undefined}
                onSelect={() => setToolsEnabled(!toolsEnabled)}
              >
                <GlobeIcon />
                Web search
                {toolsEnabled ? <CheckIcon className="ml-auto" /> : null}
              </DropdownMenuItem>
              <DropdownMenuItem
                className={
                  codeToolsEnabled ? "text-primary font-medium" : undefined
                }
                onSelect={() => setCodeToolsEnabled(!codeToolsEnabled)}
              >
                <HugeiconsIcon
                  icon={CodeIcon}
                  strokeWidth={2}
                  className="size-[1.175rem]!"
                />
                Code
                {codeToolsEnabled ? <CheckIcon className="ml-auto" /> : null}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => setMcpOpen(true)}>
                <HugeiconsIcon
                  icon={McpServerIcon}
                  strokeWidth={2}
                  className="size-[1.175rem]!"
                />
                MCP
              </DropdownMenuItem>
              <DropdownMenuItem
                className={
                  artifactsEnabled ? "text-primary font-medium" : undefined
                }
                onSelect={() => setArtifactsEnabled(!artifactsEnabled)}
              >
                <HugeiconsIcon icon={PencilRulerIcon} strokeWidth={2} />
                Canvas
                {artifactsEnabled ? <CheckIcon className="ml-auto" /> : null}
              </DropdownMenuItem>
              <DropdownMenuItem>
                <HugeiconsIcon icon={DatabaseIcon} strokeWidth={2} />
                RAG
              </DropdownMenuItem>
              {/* Always active: this menu only renders in compare mode.
                  Ticked like Web search/Code; click toggles it off. */}
              <DropdownMenuItem
                className="text-primary font-medium"
                onSelect={handleExitCompare}
              >
                <Columns2Icon />
                Compare chat
                <CheckIcon className="ml-auto" />
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>
                  <HugeiconsIcon icon={Folder01Icon} strokeWidth={2} />
                  Projects
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="unsloth-plus-menu w-[200px]">
                  <DropdownMenuItem onSelect={() => openLink(PROJECTS_PR_URL)}>
                    <HugeiconsIcon icon={FolderAddIcon} strokeWidth={2} />
                    New project
                  </DropdownMenuItem>
                  <DropdownMenuLabel>Recents</DropdownMenuLabel>
                  <DropdownMenuItem onSelect={() => openLink(PROJECTS_PR_URL)}>
                    No recent projects
                  </DropdownMenuItem>
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            </DropdownMenuContent>
          </DropdownMenu>
          <ChatMcpServersDialog open={mcpOpen} onOpenChange={setMcpOpen} />
          <button
            type="button"
            disabled={searchDisabled}
            onClick={() => {
              const next = !toolsEnabled;
              setToolsEnabled(next);
              // Kimi's $web_search builtin requires thinking=disabled
              // (https://platform.kimi.ai/docs/guide/use-web-search). Toggle
              // the Think pill off when Search is on, mirroring the backend.
              if (isKimiExternal) {
                setReasoningEnabled(!next, { persist: false });
                applyQwenThinkingParams(!next);
              }
            }}
            className="composer-pill-btn"
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
            data-active={codeToolsEnabled && !codeDisabled ? "true" : "false"}
            aria-label={
              codeToolsEnabled
                ? "Disable code execution"
                : "Enable code execution"
            }
          >
            <PillGlyph>
              <CodeToggleIcon className="size-[15px]" />
            </PillGlyph>
            <span>Code</span>
          </button>
          {/* Active in compare mode; click to exit back to single chat. */}
          <button
            type="button"
            onClick={handleExitCompare}
            className="composer-pill-btn"
            data-active="true"
            aria-label="Exit compare chat"
          >
            <PillGlyph>
              <Columns2Icon className="size-[15px]" />
            </PillGlyph>
            <span>Compare</span>
          </button>
          {showImagePill && (
            <button
              type="button"
              disabled={imageDisabled}
              onClick={() => setImageToolsEnabled(!imageToolsEnabled)}
              className="composer-pill-btn"
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
          {artifactsEnabled && (
            <button
              type="button"
              onClick={() => setArtifactsEnabled(false)}
              className="composer-pill-btn"
              data-active="true"
              aria-label="Disable canvas"
            >
              <PillGlyph>
                <HugeiconsIcon
                  icon={PencilRulerIcon}
                  className="size-3.5"
                  strokeWidth={2}
                />
              </PillGlyph>
              <span>Canvas</span>
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
              <PillGlyph>
                <DownloadIcon className="size-3.5" />
              </PillGlyph>
              <span>Fetch</span>
            </button>
          )}
        </div>
        <div className="ml-auto flex items-center gap-1">
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
                    <BulbIcon className="size-[15px]" />
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
                          <CheckIcon
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
                            <CheckIcon
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
                        <CheckIcon
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
                      <CheckIcon
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
                  // Mutual exclusion: Kimi's $web_search builtin
                  // requires thinking off, so turning thinking on flips
                  // the Search pill off (and vice versa).
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
                  <BulbIcon className="size-[15px]" />
                </PillGlyph>
                {thinkingActiveLook ? <span>Thinking</span> : null}
              </button>
            )
          ) : null}
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
              <ArrowUpIcon className="size-[22px] stroke-2" />
            </TooltipIconButton>
          )}
        </div>
      </div>
    </div>
  );
}
