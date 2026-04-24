// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ComposerAddAttachment,
  ComposerAttachments,
  UserMessageAttachments,
} from "@/components/assistant-ui/attachment";
import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { MessageTiming } from "@/components/assistant-ui/message-timing";
import { Reasoning, ReasoningGroup } from "@/components/assistant-ui/reasoning";
import { Sources, SourcesGroup } from "@/components/assistant-ui/sources";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { ToolGroup } from "@/components/assistant-ui/tool-group";
import { PythonToolUI } from "@/components/assistant-ui/tool-ui-python";
import { TerminalToolUI } from "@/components/assistant-ui/tool-ui-terminal";
import { WebSearchToolUI } from "@/components/assistant-ui/tool-ui-web-search";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  IntentAwareScrollProvider,
  useIntentAwareAutoScroll,
  useIsThreadAtBottom,
  useScrollThreadToBottom,
} from "@/components/assistant-ui/use-intent-aware-autoscroll";
import { Button } from "@/components/ui/button";
import { sentAudioNames } from "@/features/chat/api/chat-adapter";
import { db, useLiveQuery } from "@/features/chat/db";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { usePromptEvalStore } from "@/features/chat/stores/use-prompt-eval-store";
import {
  downloadPromptEvalCsv,
  downloadPromptEvalJsonl,
} from "@/features/chat/prompt-eval/utils/export-prompt-eval";
import { PromptStorageDialog } from "@/features/chat/prompt-eval/prompt-storage-dialog";
import { deleteThreadMessage } from "@/features/chat/utils/delete-thread-message";
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE, fileToBase64 } from "@/lib/audio-utils";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import {
  ActionBarMorePrimitive,
  ActionBarPrimitive,
  AuiIf,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAui,
  useAuiEvent,
  useAuiState,
} from "@assistant-ui/react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  FlaskConicalIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  GlobeIcon,
  HeadphonesIcon,
  LightbulbIcon,
  LightbulbOffIcon,
  LoaderIcon,
  MicIcon,
  MoreHorizontalIcon,
  BookmarkIcon,
  PencilIcon,
  RefreshCwIcon,
  SquareIcon,
  TerminalIcon,
  Trash2Icon,
  XIcon,
} from "lucide-react";
import { motion } from "motion/react";
import {
  type FC,
  type FormEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";

export const Thread: FC<{
  hideComposer?: boolean;
  hideWelcome?: boolean;
  targetThreadId?: string;
  onPromptEvalSend?: (text: string) => void;
}> = ({
  hideComposer,
  hideWelcome,
  targetThreadId,
  onPromptEvalSend,
}) => {
  // Intent-aware autoscroll: replaces assistant-ui's built-in autoscroll
  // to prevent the streaming-mutation race that makes the viewport snap
  // back to the bottom while the user is scrolling up (see the hook for
  // the full explanation).
  const { ref: viewportRef, context: autoScrollContext } =
    useIntentAwareAutoScroll();

  const isComposerAttachPending = useAuiState(({ threads }) =>
    targetThreadId ? threads.mainThreadId !== targetThreadId : false,
  );

  return (
    <ThreadPrimitive.Root
      className="aui-root aui-thread-root @container relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden"
      style={{
        ["--thread-max-width" as string]: "44rem",
        ["--thread-content-max-width" as string]:
          "calc(var(--thread-max-width) - 2.5rem)",
      }}
    >
      <IntentAwareScrollProvider value={autoScrollContext}>
        <ThreadPrimitive.Viewport
          ref={viewportRef}
          autoScroll={false}
          scrollToBottomOnRunStart={false}
          scrollToBottomOnInitialize={false}
          scrollToBottomOnThreadSwitch={false}
          className={cn(
            "aui-thread-viewport relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-x-auto overflow-y-auto scroll-smooth px-5",
            hideComposer ? "pt-4" : "pt-[48px]",
          )}
        >
          {!hideWelcome && (
            <AuiIf condition={({ thread }) => thread.isEmpty && !thread.isLoading}>
              <ThreadWelcome hideComposer={hideComposer} onPromptEvalSend={onPromptEvalSend} />
            </AuiIf>
          )}

          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              EditComposer,
              AssistantMessage,
            }}
          />

          {/* Bottom slack so the last message has breathing room above the
            sticky scroll-to-bottom button (and the floating composer in
            single mode). Without this, content would butt against the
            sticky footer and feel cramped. */}
          <AuiIf condition={({ thread }) => hideWelcome || !thread.isEmpty}>
            <div
              className={cn("shrink-0", hideComposer ? "h-16" : "h-40")}
              aria-hidden={true}
            />
          </AuiIf>

          <AuiIf condition={({ thread }) => hideWelcome || !thread.isEmpty}>
            <ThreadPrimitive.ViewportFooter
              className={cn(
                "aui-thread-viewport-footer pointer-events-none sticky z-20 flex w-full justify-center bg-transparent",
                hideComposer ? "bottom-3" : "bottom-[140px]",
              )}
            >
              <ThreadScrollToBottom />
            </ThreadPrimitive.ViewportFooter>
          </AuiIf>
        </ThreadPrimitive.Viewport>

        {!hideComposer && (
          <AuiIf condition={({ thread }) => hideWelcome || !thread.isEmpty}>
            <div className="aui-thread-composer-dock pointer-events-none absolute bottom-0 left-0 right-0 md:right-2 z-20">
              <div
                aria-hidden={true}
                className="absolute inset-x-0 bottom-0 top-[10px] bg-background"
              />
              <div className="relative px-5 pb-2">
                <div className="pointer-events-auto mx-auto w-full max-w-(--thread-max-width)">
                  <ComposerAnimated disabled={isComposerAttachPending} onPromptEvalSend={onPromptEvalSend} />
                </div>
                <p className="mt-1.5 text-center text-[11px] text-muted-foreground">
                  LLMs can make mistakes. Double-check all responses.
                </p>
              </div>
            </div>
          </AuiIf>
        )}
      </IntentAwareScrollProvider>
    </ThreadPrimitive.Root>
  );
};

const ThreadScrollToBottom: FC = () => {
  // State and action both come from our IntentAwareScrollProvider (scoped
  // per Thread, so compare panes are independent). We deliberately
  // avoid `ThreadPrimitive.ScrollToBottom` + `useThreadViewport` to
  // stay off assistant-ui's internal autoscroll path — see the hook
  // for why. The button stays mounted and toggles via CSS; unmounting
  // would trip the hook's MutationObserver as a content change.
  const isAtBottom = useIsThreadAtBottom();
  const scrollToBottom = useScrollThreadToBottom();
  return (
    <TooltipIconButton
      tooltip="Scroll to bottom"
      variant="outline"
      onClick={() => scrollToBottom("auto")}
      className={cn(
        "aui-thread-scroll-to-bottom pointer-events-auto rounded-full p-4 bg-background hover:bg-accent dark:bg-background dark:hover:bg-accent",
        isAtBottom && "invisible pointer-events-none",
      )}
    >
      <ArrowDownIcon />
    </TooltipIconButton>
  );
};

const ThreadWelcome: FC<{ hideComposer?: boolean; onPromptEvalSend?: (text: string) => void }> = ({ hideComposer, onPromptEvalSend }) => {
  return (
    <div className="aui-thread-welcome-root mx-auto my-auto flex w-full max-w-(--thread-max-width) grow flex-col">
      <div className="aui-thread-welcome-center flex w-full grow flex-col items-center justify-center pb-[48px]">
        <div className="aui-thread-welcome-message flex w-full flex-col justify-center gap-6 px-4">
          <div className="flex flex-col items-center gap-2 text-center">
            <img
              src="/Sloth emojis/sloth pc square.png"
              alt="Sloth mascot"
              className="size-20"
            />
            <h1 className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-1 animate-in font-heading font-semibold text-2xl tracking-[-0.02em] duration-200">
              Chat with your model
            </h1>
            <p className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-1 -mt-1 animate-in font-heading font-normal text-muted-foreground text-sm delay-75 duration-200">
              Run GGUFs, safetensors, vision and audio models
            </p>
          </div>
          <GeneratingSpinner />
          {!hideComposer && <ComposerAnimated onPromptEvalSend={onPromptEvalSend} />}
        </div>
      </div>
    </div>
  );
};

const GeneratingSpinner: FC = () => {
  const status = useChatRuntimeStore((s) => s.generatingStatus);
  if (!status) {
    return null;
  }
  return (
    <div className="mx-auto flex w-full max-w-(--thread-max-width) items-center justify-center py-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <LoaderIcon className="size-3.5 animate-spin" />
        <span>Generating</span>
      </div>
    </div>
  );
};

const ComposerAnimated: FC<{ disabled?: boolean; onPromptEvalSend?: (text: string) => void }> = ({ disabled, onPromptEvalSend }) => {
  return (
    <div className="relative mx-auto min-w-0 w-full max-w-(--thread-max-width)">
      <motion.div
        layout={true}
        layoutId="composer"
        transition={{ type: "spring", bounce: 0.15, duration: 0.5 }}
        className="relative z-10 w-full"
      >
        <Composer disabled={disabled} onPromptEvalSend={onPromptEvalSend} />
      </motion.div>
    </div>
  );
};

const PendingAudioChip: FC = () => {
  const audioName = useChatRuntimeStore((s) => s.pendingAudioName);
  const clearPendingAudio = useChatRuntimeStore((s) => s.clearPendingAudio);
  if (!audioName) {
    return null;
  }
  return (
    <div className="mb-2 flex w-full flex-row items-center gap-2 px-1.5 pt-0.5 pb-1">
      <div className="flex items-center gap-2 rounded-lg border border-foreground/20 bg-muted px-3 py-1.5 text-xs">
        <HeadphonesIcon className="size-3.5 text-muted-foreground" />
        <span className="max-w-48 truncate">{audioName}</span>
        <button
          type="button"
          onClick={clearPendingAudio}
          className="flex size-4 items-center justify-center rounded-full hover:bg-destructive hover:text-destructive-foreground"
          aria-label="Remove audio"
        >
          <XIcon className="size-3" />
        </button>
      </div>
    </div>
  );
};

const Composer: FC<{ disabled?: boolean; onPromptEvalSend?: (text: string) => void }> = ({ disabled, onPromptEvalSend }) => {
  const promptEvalMode = usePromptEvalStore((s) => s.promptEvalMode);
  const promptEvalName = usePromptEvalStore((s) => s.promptEvalName);
  const setPromptEvalName = usePromptEvalStore((s) => s.setPromptEvalName);
  const pendingComposerText = usePromptEvalStore((s) => s.pendingComposerText);
  const setPendingComposerText = usePromptEvalStore((s) => s.setPendingComposerText);
  const [promptStorageOpen, setPromptStorageOpen] = useState(false);

  // Detect if a different thread (not the currently visible one) is generating.
  // In that case we block send but keep typing/uploads/prompt-storage enabled.
  // We check both the assistant-ui isRunning flag AND the store's runningByThreadId
  // so that the eval hidden host's generation (which uses its own ChatRuntimeProvider)
  // is recognised as "this thread running" when the visible thread is the same DB thread.
  const thisThreadIsRunning = useAuiState(({ thread }) => thread.isRunning);
  const mainThreadId = useAuiState(({ threads }) => threads.mainThreadId);
  const thisThreadInStore = useChatRuntimeStore((s) =>
    mainThreadId ? Boolean(s.runningByThreadId[mainThreadId]) : false,
  );
  const anyRunning = useChatRuntimeStore((s) => Object.values(s.runningByThreadId).some(Boolean));
  const anotherThreadRunning = !thisThreadIsRunning && !thisThreadInStore && anyRunning;

  // When a prompt is loaded from Prompt Storage, inject it into the textarea.
  // Only inject into a VISIBLE textarea (offsetParent !== null) so that the
  // hidden background eval SingleContent doesn't consume the event.
  useEffect(() => {
    if (!pendingComposerText) return;
    // Find the first visible .aui-composer-input (not inside display:none).
    const all = document.querySelectorAll<HTMLTextAreaElement>(".aui-composer-input");
    let textarea: HTMLTextAreaElement | null = null;
    for (const el of all) {
      if (el.offsetParent !== null) { textarea = el; break; }
    }
    if (textarea) {
      const nativeSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      if (nativeSetter) {
        nativeSetter.call(textarea, pendingComposerText);
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.focus();
      }
    }
    setPendingComposerText(null);
  }, [pendingComposerText, setPendingComposerText]);

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      if (disabled || anotherThreadRunning) {
        event.preventDefault();
        return;
      }
      if (promptEvalMode && onPromptEvalSend) {
        const textarea = event.currentTarget.querySelector('textarea') as HTMLTextAreaElement | null;
        const text = textarea?.value?.trim() ?? "";
        if (text) {
          event.preventDefault();
          const nativeSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
          if (nativeSetter && textarea) {
            nativeSetter.call(textarea, '');
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
          }
          onPromptEvalSend(text);
        } else {
          event.preventDefault();
        }
      }
    },
    [disabled, anotherThreadRunning, promptEvalMode, onPromptEvalSend],
  );

  return (
    <>
      {promptEvalMode && (
        <PromptStorageDialog open={promptStorageOpen} onOpenChange={setPromptStorageOpen} />
      )}
      <ComposerPrimitive.Root
        className="aui-composer-root relative flex w-full flex-col"
        aria-disabled={disabled}
        onSubmit={handleSubmit}
      >
        <ComposerPrimitive.AttachmentDropzone className="aui-composer-attachment-dropzone chat-composer-surface flex w-full flex-col rounded-3xl bg-background dark:bg-card px-1 pt-2 outline-none transition-shadow data-[dragging=true]:border-ring data-[dragging=true]:bg-accent/50">
          {anotherThreadRunning && (
            <div className="mb-2 mx-2 mt-1 flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 dark:border-amber-800/40 dark:bg-amber-950/20">
              <span className="text-xs text-amber-700 dark:text-amber-400">
                Another model is generating — you can send a message once it finishes.
              </span>
            </div>
          )}
          {promptEvalMode && (
            <div className="mb-2 flex items-center gap-2 rounded-xl bg-primary/5 border border-primary/20 px-3 py-2 mx-2 mt-1">
              <label className="text-xs font-medium text-primary whitespace-nowrap">Prompt Eval name:</label>
              <input
                value={promptEvalName}
                onChange={(e) => setPromptEvalName(e.target.value)}
                className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                placeholder="Prompt Eval name..."
              />
              <button
                type="button"
                onClick={() => setPromptStorageOpen(true)}
                className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium text-primary/80 hover:bg-primary/10 hover:text-primary transition-colors shrink-0"
                title="Open Prompt Storage"
              >
                <BookmarkIcon className="size-3.5" />
                Prompt Storage
              </button>
            </div>
          )}
          <ComposerAttachments />
          <PendingAudioChip />
          <ToolStatusDisplay />
          <ComposerPrimitive.Input
            placeholder={promptEvalMode ? "Send prompt to all selected models..." : "Send a message..."}
            className="aui-composer-input mb-1 min-h-12 w-full resize-none overflow-y-auto bg-transparent pl-5 pr-4 pt-2 pb-3 text-sm font-[450] outline-none placeholder:text-muted-foreground focus-visible:ring-0"
            minRows={1}
            maxRows={6}
            autoFocus={!disabled}
            disabled={disabled}
            aria-label="Message input"
          />
          <ComposerAction disabled={disabled} sendDisabled={anotherThreadRunning} />
        </ComposerPrimitive.AttachmentDropzone>
      </ComposerPrimitive.Root>
    </>
  );
};

const ComposerAudioUpload: FC = () => {
  const audioInputRef = useRef<HTMLInputElement>(null);
  const setPendingAudio = useChatRuntimeStore((s) => s.setPendingAudio);
  const activeModel = useChatRuntimeStore((s) => {
    const checkpoint = s.params.checkpoint;
    return s.models.find((m) => m.id === checkpoint);
  });

  const handleAudioFile = useCallback(
    async (file: File) => {
      if (file.size > MAX_AUDIO_SIZE) {
        return;
      }
      try {
        const base64 = await fileToBase64(file);
        setPendingAudio(base64, file.name);
      } catch {
        // skip
      }
    },
    [setPendingAudio],
  );

  if (!activeModel?.hasAudioInput) {
    return null;
  }

  return (
    <>
      <input
        ref={audioInputRef}
        type="file"
        accept={AUDIO_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            handleAudioFile(file);
          }
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
  );
};

/** Qwen3/3.5 recommended params differ between thinking on/off. */
function applyQwenThinkingParams(thinkingOn: boolean): void {
  const store = useChatRuntimeStore.getState();
  const checkpoint = store.params.checkpoint?.toLowerCase() ?? "";
  if (!checkpoint.includes("qwen3")) {
    return;
  }
  const params = thinkingOn
    ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
  store.setParams({ ...store.params, ...params });
}

const ReasoningToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const supportsReasoning = useChatRuntimeStore((s) => s.supportsReasoning);
  const reasoningEnabled = useChatRuntimeStore((s) => s.reasoningEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const disabled = !(modelLoaded && supportsReasoning);

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => {
        const next = !reasoningEnabled;
        setReasoningEnabled(next);
        applyQwenThinkingParams(next);
      }}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        disabled
          ? "cursor-not-allowed opacity-40"
          : reasoningEnabled
            ? "bg-primary/10 text-primary hover:bg-primary/20"
            : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
      )}
      aria-label={reasoningEnabled ? "Disable thinking" : "Enable thinking"}
    >
      {reasoningEnabled && !disabled ? (
        <LightbulbIcon className="size-3.5" />
      ) : (
        <LightbulbOffIcon className="size-3.5" />
      )}
      <span>Think</span>
    </button>
  );
};

const WebSearchToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const disabled = !(modelLoaded && supportsTools);

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => setToolsEnabled(!toolsEnabled)}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        disabled
          ? "cursor-not-allowed opacity-40"
          : toolsEnabled
            ? "bg-primary/10 text-primary hover:bg-primary/20"
            : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
      )}
      aria-label={toolsEnabled ? "Disable web search" : "Enable web search"}
    >
      <GlobeIcon className="size-3.5" />
      <span>Search</span>
    </button>
  );
};

const CodeToolsToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const setCodeToolsEnabled = useChatRuntimeStore((s) => s.setCodeToolsEnabled);
  const disabled = !(modelLoaded && supportsTools);

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => setCodeToolsEnabled(!codeToolsEnabled)}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        disabled
          ? "cursor-not-allowed opacity-40"
          : codeToolsEnabled
            ? "bg-primary/10 text-primary hover:bg-primary/20"
            : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
      )}
      aria-label={
        codeToolsEnabled ? "Disable code execution" : "Enable code execution"
      }
    >
      <CodeToggleIcon className="size-3.5" />
      <span>Code</span>
    </button>
  );
};

const ToolStatusDisplay: FC = () => {
  const toolStatus = useChatRuntimeStore((s) => s.toolStatus);
  const isThreadRunning = useAuiState(({ thread }) => thread.isRunning);
  const [elapsed, setElapsed] = useState(0);
  const [visible, setVisible] = useState(false);
  const visibleRef = useRef(false);

  useEffect(() => {
    visibleRef.current = visible;
  }, [visible]);

  useEffect(() => {
    if (!toolStatus) {
      setElapsed(0);
      if (!isThreadRunning) {
        setVisible(false);
      }
      return;
    }

    setElapsed(0);

    // Debounce badge visibility by 300ms when the badge is not
    // already on screen. Once visible from a prior tool, consecutive
    // tools show immediately so the badge does not flicker. Fast
    // tool calls that all complete under 300ms never show the badge.
    let showTimer: ReturnType<typeof setTimeout> | undefined;
    if (!visibleRef.current) {
      showTimer = setTimeout(() => setVisible(true), 300);
    }

    const interval = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => {
      clearInterval(interval);
      if (showTimer) {
        clearTimeout(showTimer);
      }
    };
  }, [toolStatus, isThreadRunning]);

  if (!(toolStatus && visible)) {
    return null;
  }
  const isRunning = toolStatus.startsWith("Running");
  const StatusIcon = isRunning ? TerminalIcon : GlobeIcon;
  return (
    <div className="mb-2 flex w-full flex-row items-center gap-2 px-1.5 pt-0.5 pb-1">
      <div className="flex animate-pulse items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-3 py-1.5 text-xs text-primary">
        <StatusIcon className="size-3.5" />
        <span>{toolStatus}</span>
        <span className="tabular-nums opacity-60">{elapsed}s</span>
      </div>
    </div>
  );
};

const PromptEvalToggle: FC = () => {
  const promptEvalMode = usePromptEvalStore((s) => s.promptEvalMode);
  const togglePromptEvalMode = usePromptEvalStore((s) => s.togglePromptEvalMode);
  const toggleModelSelection = usePromptEvalStore((s) => s.toggleModelSelection);

  const handleToggle = () => {
    // When enabling Prompt Eval mode, auto-add the currently loaded model
    if (!promptEvalMode) {
      const { params, activeGgufVariant } = useChatRuntimeStore.getState();
      const checkpoint = params.checkpoint;
      if (checkpoint) {
        const storeId = activeGgufVariant ? `${checkpoint}::${activeGgufVariant}` : checkpoint;
        const currentIds = usePromptEvalStore.getState().promptEvalSelectedModelIds;
        if (!currentIds.includes(storeId)) {
          toggleModelSelection(storeId);
        }
      }
    }
    togglePromptEvalMode();
  };

  return (
    <button
      type="button"
      onClick={handleToggle}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        promptEvalMode
          ? "bg-primary/10 text-primary hover:bg-primary/20"
          : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
      )}
      aria-label={promptEvalMode ? "Disable Prompt Eval mode" : "Enable Prompt Eval mode"}
    >
      <FlaskConicalIcon className="size-3.5" />
      <span>Prompt Eval</span>
    </button>
  );
};

const ExportPromptEvalButton: FC = () => {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const thread = useLiveQuery(
    () => (activeThreadId ? db.threads.get(activeThreadId) : Promise.resolve(undefined)),
    [activeThreadId],
  );
  const [open, setOpen] = useState(false);
  const [exporting, setExporting] = useState(false);

  const promptEvalId = thread?.promptEvalId;

  if (!promptEvalId) return null;

  const handleExport = async (format: "jsonl" | "csv") => {
    setExporting(true);
    try {
      if (format === "jsonl") {
        await downloadPromptEvalJsonl(promptEvalId);
      } else {
        await downloadPromptEvalCsv(promptEvalId);
      }
    } catch {
      // user cancelled file picker — ignore
    } finally {
      setExporting(false);
      setOpen(false);
    }
  };

  return (
    <div className="relative">
      <TooltipIconButton
        tooltip="Export Prompt Eval"
        variant="ghost"
        className="size-8 rounded-full text-muted-foreground"
        onClick={() => setOpen((v) => !v)}
        type="button"
      >
        <DownloadIcon className="size-4" />
      </TooltipIconButton>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute bottom-10 right-0 z-50 flex flex-col gap-1 rounded-xl border border-border bg-popover p-2 shadow-md">
            <button
              type="button"
              disabled={exporting}
              onClick={() => { void handleExport("jsonl"); }}
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium hover:bg-accent"
            >
              <DownloadIcon className="size-3.5" />
              Export JSONL
            </button>
            <button
              type="button"
              disabled={exporting}
              onClick={() => { void handleExport("csv"); }}
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium hover:bg-accent"
            >
              <DownloadIcon className="size-3.5" />
              Export CSV
            </button>
          </div>
        </>
      )}
    </div>
  );
};

const ComposerAction: FC<{ disabled?: boolean; sendDisabled?: boolean }> = ({ disabled, sendDisabled }) => {
  const promptEvalMode = usePromptEvalStore((s) => s.promptEvalMode);
  const promptEvalSendFn = usePromptEvalStore((s) => s.promptEvalSendFn);
  // Custom send handler for prompt eval mode.
  // ComposerPrimitive.Send's own onClick bypasses our form's onSubmit (it calls
  // the runtime directly), so in eval mode we replace it with a plain button
  // that reads the textarea value and fires the eval runner.
  const handleEvalSend = useCallback(() => {
    if (!promptEvalSendFn) return;
    const textarea = document.querySelector<HTMLTextAreaElement>(".aui-composer-input:not(.hidden *)");
    // Fall back to any visible composer input (not inside a display:none container)
    const visibleTextarea = (() => {
      const all = document.querySelectorAll<HTMLTextAreaElement>(".aui-composer-input");
      for (const el of all) {
        if (el.offsetParent !== null) return el;
      }
      return textarea;
    })();
    const text = visibleTextarea?.value?.trim() ?? "";
    if (!text) return;
    const nativeSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    if (nativeSetter && visibleTextarea) {
      nativeSetter.call(visibleTextarea, "");
      visibleTextarea.dispatchEvent(new Event("input", { bubbles: true }));
    }
    promptEvalSendFn([text]);
  }, [promptEvalSendFn]);

  return (
    <div className="aui-composer-action-wrapper relative mx-2 mb-2 flex items-center justify-between">
      <div className="flex items-center gap-1">
        <ComposerAddAttachment />
        <ComposerAudioUpload />
        <ReasoningToggle />
        <WebSearchToggle />
        <CodeToolsToggle />
        <PromptEvalToggle />
      </div>
      <div className="flex items-center gap-1">
        <ExportPromptEvalButton />
        <ComposerPrimitive.If dictation={false}>
          <ComposerPrimitive.Dictate asChild={true}>
            <TooltipIconButton
              tooltip="Dictate"
              variant="ghost"
              className="size-8 rounded-full text-muted-foreground"
            >
              <MicIcon className="size-4" />
            </TooltipIconButton>
          </ComposerPrimitive.Dictate>
        </ComposerPrimitive.If>
        <ComposerPrimitive.If dictation={true}>
          <ComposerPrimitive.StopDictation asChild={true}>
            <TooltipIconButton
              tooltip="Stop dictation"
              variant="ghost"
              className="size-8 rounded-full text-destructive"
            >
              <SquareIcon className="size-3 animate-pulse fill-current" />
            </TooltipIconButton>
          </ComposerPrimitive.StopDictation>
        </ComposerPrimitive.If>
        <AuiIf condition={({ thread }) => !thread.isRunning}>
          {promptEvalMode ? (
            // In Prompt Eval mode, bypass ComposerPrimitive.Send entirely —
            // its onClick calls the runtime directly, skipping our onSubmit.
            // This plain button reads the textarea and fires the eval runner.
            <TooltipIconButton
              tooltip={sendDisabled ? "Another model is generating" : "Send to all selected models"}
              side="bottom"
              type="button"
              variant="default"
              size="icon"
              disabled={disabled || sendDisabled}
              className="aui-composer-send size-8 rounded-full"
              aria-label="Send to all selected models"
              onClick={handleEvalSend}
            >
              <ArrowUpIcon className="aui-composer-send-icon size-4" />
            </TooltipIconButton>
          ) : (
            <ComposerPrimitive.Send asChild={true}>
              <TooltipIconButton
                tooltip={sendDisabled ? "Another model is generating" : "Send message"}
                side="bottom"
                type="submit"
                variant="default"
                size="icon"
                disabled={disabled || sendDisabled}
                className="aui-composer-send size-8 rounded-full"
                aria-label="Send message"
              >
                <ArrowUpIcon className="aui-composer-send-icon size-4" />
              </TooltipIconButton>
            </ComposerPrimitive.Send>
          )}
        </AuiIf>
        <AuiIf condition={({ thread }) => thread.isRunning}>
          <ComposerPrimitive.Cancel asChild={true}>
            <Button
              type="button"
              variant="default"
              size="icon"
              className="aui-composer-cancel size-8 rounded-full"
              aria-label="Stop generating"
            >
              <SquareIcon className="aui-composer-cancel-icon size-3 fill-current" />
            </Button>
          </ComposerPrimitive.Cancel>
        </AuiIf>
      </div>
    </div>
  );
};

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="aui-message-error-root mt-2 rounded-md border border-destructive bg-destructive/10 p-3 text-destructive text-sm dark:bg-destructive/5 dark:text-red-200">
        <ErrorPrimitive.Message className="aui-message-error-message line-clamp-2" />
      </ErrorPrimitive.Root>
    </MessagePrimitive.Error>
  );
};

const GeneratingIndicator: FC = () => {
  const show = useAuiState(
    ({ message }) =>
      message.content.length === 0 && message.status?.type === "running",
  );
  if (!show) {
    return null;
  }
  return <span className="text-sm text-muted-foreground">Generating...</span>;
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="aui-assistant-message-root fade-in slide-in-from-bottom-1 relative mx-auto min-w-0 w-full max-w-(--thread-content-max-width) animate-in py-0.5 text-[15.5px] font-[450] duration-150"
      data-role="assistant"
    >
      <div className="aui-assistant-message-content wrap-break-word min-w-0 text-foreground leading-relaxed">
        <GeneratingIndicator />
        <MessagePrimitive.Parts
          components={{
            Text: MarkdownText,
            Reasoning: Reasoning,
            ReasoningGroup: ReasoningGroup,
            Source: Sources,
            ToolGroup: ToolGroup,
            tools: {
              by_name: {
                web_search: WebSearchToolUI,
                python: PythonToolUI,
                terminal: TerminalToolUI,
              },
              Fallback: ToolFallback,
            },
          }}
        />
        <SourcesGroup />
        <MessageError />
      </div>

      <div className="aui-assistant-message-footer mt-1 flex">
        <BranchPicker />
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const COPY_RESET_MS = 2000;

const DeleteMessageButton: FC = () => {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);

  const handleDelete = async () => {
    const remoteId = aui.threadListItem().getState().remoteId;
    const thread = aui.thread();
    try {
      await deleteThreadMessage({
        thread: {
          export: () => thread.export(),
          import: (data) => thread.import(data),
        },
        messageId,
        remoteId,
      });
    } catch (error) {
      console.error("Failed to delete message", error);
      toast.error("Failed to delete message");
    }
  };

  return (
    <TooltipIconButton
      tooltip="Delete message"
      disabled={isRunning}
      onClick={handleDelete}
      className="text-muted-foreground hover:text-destructive"
    >
      <Trash2Icon className="size-4" />
    </TooltipIconButton>
  );
};

const CopyButton: FC = () => {
  const aui = useAui();
  const [copied, setCopied] = useState(false);
  const resetTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleCopy = async () => {
    const text = aui.message().getCopyText();
    if (await copyToClipboard(text)) {
      setCopied(true);
      if (resetTimeoutRef.current) {
        clearTimeout(resetTimeoutRef.current);
      }
      resetTimeoutRef.current = setTimeout(() => {
        setCopied(false);
        resetTimeoutRef.current = null;
      }, COPY_RESET_MS);
    }
  };

  return (
    <TooltipIconButton tooltip="Copy" onClick={handleCopy}>
      {copied ? <CheckIcon /> : <CopyIcon />}
    </TooltipIconButton>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning={true}
      autohide="always"
      autohideFloat="single-branch"
      className="aui-assistant-action-bar-root col-start-3 row-start-2 -ml-1 flex gap-1 text-muted-foreground data-floating:absolute"
    >
      <CopyButton />
      <ActionBarPrimitive.Reload asChild={true}>
        <TooltipIconButton tooltip="Refresh">
          <RefreshCwIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Reload>
      <DeleteMessageButton />
      <MessageTiming side="top" />
      <ActionBarMorePrimitive.Root>
        <ActionBarMorePrimitive.Trigger asChild={true}>
          <TooltipIconButton
            tooltip="More"
            className="data-[state=open]:bg-accent"
          >
            <MoreHorizontalIcon />
          </TooltipIconButton>
        </ActionBarMorePrimitive.Trigger>
        <ActionBarMorePrimitive.Content
          side="bottom"
          align="start"
          className="aui-action-bar-more-content z-50 min-w-32 overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
        >
          <ActionBarPrimitive.ExportMarkdown asChild={true}>
            <ActionBarMorePrimitive.Item className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
              <DownloadIcon className="size-4" />
              Export as Markdown
            </ActionBarMorePrimitive.Item>
          </ActionBarPrimitive.ExportMarkdown>
        </ActionBarMorePrimitive.Content>
      </ActionBarMorePrimitive.Root>
    </ActionBarPrimitive.Root>
  );
};

const UserMessageAudio: FC = () => {
  const audioName = useAuiState(({ message }) =>
    sentAudioNames.get(message.id),
  );
  if (!audioName) {
    return null;
  }
  return (
    <div className="col-start-2 flex justify-end">
      <div className="flex items-center gap-2 rounded-lg border border-foreground/20 bg-muted px-3 py-1.5 text-xs">
        <HeadphonesIcon className="size-3.5 text-muted-foreground" />
        <span className="max-w-48 truncate">{audioName}</span>
      </div>
    </div>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="aui-user-message-root fade-in slide-in-from-bottom-1 mx-auto flex w-full max-w-(--thread-content-max-width) animate-in flex-col items-end gap-y-2 pt-6 pb-0.5 text-[15.5px] font-[450] duration-150"
      data-role="user"
    >
      <UserMessageAttachments />
      <UserMessageAudio />

      <div className="aui-user-message-content-wrapper flex max-w-[80%] min-w-0 flex-col items-end">
        <div className="aui-user-message-content wrap-break-word w-fit rounded-[16px] rounded-tr-[4px] bg-[#f5f5f5] px-4 py-2.5 text-foreground dark:bg-card">
          <MessagePrimitive.Parts />
        </div>
        <div className="mt-1 flex min-h-6">
          <UserActionBar />
        </div>
      </div>

      <BranchPicker className="aui-user-branch-picker -mr-1 justify-end" />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      autohide="always"
      className="aui-user-action-bar-root -mr-1 flex gap-1 text-muted-foreground"
    >
      <CopyButton />
      <ActionBarPrimitive.Edit asChild={true}>
        <TooltipIconButton tooltip="Edit" className="aui-user-action-edit">
          <PencilIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Edit>
      <DeleteMessageButton />
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  const aui = useAui();
  const resendAfterCancelRef = useRef(false);

  useAuiEvent("thread.runEnd", () => {
    if (!resendAfterCancelRef.current) {
      return;
    }
    resendAfterCancelRef.current = false;
    aui.composer().send();
  });

  return (
    <MessagePrimitive.Root className="aui-edit-composer-wrapper mx-auto flex w-full max-w-(--thread-content-max-width) flex-col py-3">
      <ComposerPrimitive.Root className="aui-edit-composer-root ml-auto flex w-full max-w-[85%] flex-col rounded-2xl bg-muted">
        <ComposerPrimitive.Input
          className="aui-edit-composer-input min-h-14 w-full resize-none bg-transparent p-4 text-foreground text-sm font-[450] outline-none"
          autoFocus={true}
        />
        <div className="aui-edit-composer-footer mx-3 mb-3 flex items-center gap-2 self-end">
          <ComposerPrimitive.Cancel asChild={true}>
            <Button variant="ghost" size="sm">
              Cancel
            </Button>
          </ComposerPrimitive.Cancel>
          <Button
            size="sm"
            onClick={() => {
              const newText = aui.composer().getState().text;
              const originalText = aui.message().getCopyText();

              if (newText === originalText) {
                aui.composer().cancel();
                return;
              }

              if (aui.thread().getState().isRunning) {
                resendAfterCancelRef.current = true;
                aui.thread().cancelRun();
                return;
              }
              aui.composer().send();
            }}
          >
            Update
          </Button>
        </div>
      </ComposerPrimitive.Root>
    </MessagePrimitive.Root>
  );
};

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({
  className,
  ...rest
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch={true}
      className={cn(
        "aui-branch-picker-root mr-2 -ml-2 inline-flex items-center text-muted-foreground text-xs",
        className,
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous asChild={true}>
        <TooltipIconButton tooltip="Previous">
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="aui-branch-picker-state font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild={true}>
        <TooltipIconButton tooltip="Next">
          <ChevronRightIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};
