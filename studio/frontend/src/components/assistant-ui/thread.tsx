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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { sentAudioNames } from "@/features/chat/api/chat-adapter";
import { parseExternalModelId } from "@/features/chat/external-providers";
import { getExternalReasoningCapabilities } from "@/features/chat/provider-capabilities";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { applyQwenThinkingParams } from "@/features/chat/utils/qwen-params";
import { isTauri } from "@/lib/api-base";
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
import { flushResourcesSync } from "@assistant-ui/tap";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  DownloadIcon,
  GlobeIcon,
  HeadphonesIcon,
  LightbulbIcon,
  LightbulbOffIcon,
  LoaderIcon,
  MicIcon,
  MoreHorizontalIcon,
  RefreshCwIcon,
  SquareIcon,
  TerminalIcon,
  XIcon,
} from "lucide-react";
import { Copy01Icon, Delete02Icon, Edit03Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ChangeEvent,
  type CompositionEvent,
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
}> = ({
  hideComposer,
  hideWelcome,
  targetThreadId,
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
        ["--thread-max-width" as string]: "48rem",
        ["--thread-content-max-width" as string]:
          "calc(var(--thread-max-width) - 1.5rem)",
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
            "aui-thread-viewport aui-stream-viewport relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-x-auto overflow-y-auto scroll-smooth px-5",
            hideComposer ? "pt-4" : "pt-[48px]",
          )}
        >
          {!hideWelcome && (
            <AuiIf condition={({ thread }) => thread.isEmpty && !thread.isLoading}>
              <ThreadWelcome hideComposer={hideComposer} />
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
            <div className="aui-thread-composer-dock pointer-events-none absolute bottom-0 left-0 right-0 md:right-[10px] z-20">
              <div
                aria-hidden={true}
                className="absolute inset-x-0 bottom-0 top-[10px] bg-background"
              />
              <div className="relative px-5 pb-2">
                <div className="pointer-events-auto mx-auto w-full max-w-(--thread-max-width)">
                  <ComposerAnimated disabled={isComposerAttachPending} />
                </div>
                <p className="composer-footer-note">
                  LLMs can make mistakes. Double-check responses.
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
      <ArrowDownIcon strokeWidth={1.75} className="size-icon" />
    </TooltipIconButton>
  );
};

const ThreadWelcome: FC<{ hideComposer?: boolean }> = ({ hideComposer }) => {
  const [currentEmoji, setCurrentEmoji] = useState("large sloth drink.png");

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour >= 6 && hour < 12) setCurrentEmoji("large sloth drink.png");
    else if (hour >= 12 && hour < 17) setCurrentEmoji("sloth magnify final.png");
    else if (hour >= 17 && hour < 21) setCurrentEmoji("sloth shy large.png");
    else setCurrentEmoji("unsloth-gem.png");
  }, []);

  const currentEmojiSrc =
    currentEmoji === "unsloth-gem.png"
      ? `/${currentEmoji}`
      : `/Sloth emojis/${currentEmoji}`;

  return (
    <div className="aui-thread-welcome-root mx-auto my-auto flex w-full max-w-(--thread-max-width) grow flex-col">
      <div className="aui-thread-welcome-center flex w-full grow flex-col items-center justify-center pb-[48px]">
        <div className="aui-thread-welcome-message flex w-full flex-col justify-center gap-6 px-4">
          <div className="flex flex-col items-center gap-2 text-center">
            <img
              src={currentEmojiSrc}
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
          {!hideComposer && <ComposerAnimated />}
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

const ComposerAnimated: FC<{ disabled?: boolean }> = ({ disabled }) => {
  return (
    <div className="relative mx-auto min-w-0 w-full max-w-(--thread-max-width)">
      <div className="relative z-10 w-full">
        <Composer disabled={disabled} />
      </div>
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

const Composer: FC<{ disabled?: boolean }> = ({ disabled }) => {
  const { inputProps, isComposing, isComposingRef } = useImeComposerInputHandlers();

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      if (disabled || isComposingRef.current) {
        event.preventDefault();
      }
    },
    [disabled, isComposingRef],
  );

  const composerContent = (
    <>
      <ComposerAttachments />
      <PendingAudioChip />
      <ToolStatusDisplay />
      <ComposerPrimitive.Input
        placeholder="Send a message..."
        className="aui-composer-input composer-input"
        minRows={1}
        maxRows={6}
        autoFocus={!disabled}
        disabled={disabled}
        aria-label="Message input"
        {...inputProps}
      />
      <ComposerAction
        disabled={disabled || isComposing}
        blockSend={() => isComposingRef.current}
      />
    </>
  );

  return (
    <ComposerPrimitive.Root
      className="aui-composer-root relative flex w-full flex-col"
      aria-disabled={disabled}
      onSubmit={handleSubmit}
    >
      {isTauri ? (
        // Phase 1 native model drops own Tauri local-path drops. Restore browser
        // attachment drops in Tauri when Phase 1d adds attachment-token bridging.
        <div className="aui-composer-attachment-dropzone chat-composer-surface">
          {composerContent}
        </div>
      ) : (
        <ComposerPrimitive.AttachmentDropzone className="aui-composer-attachment-dropzone chat-composer-surface data-[dragging=true]:border-ring data-[dragging=true]:bg-accent/50">
          {composerContent}
        </ComposerPrimitive.AttachmentDropzone>
      )}
    </ComposerPrimitive.Root>
  );
};

function isNativeComposing(event: Event) {
  return "isComposing" in event && (event as InputEvent).isComposing === true;
}

function useImeComposerInputHandlers() {
  const aui = useAui();
  const composingRef = useRef(false);
  const [isComposing, setIsComposing] = useState(false);

  const setCompositionState = useCallback((next: boolean) => {
    composingRef.current = next;
    setIsComposing(next);
  }, []);

  const setComposerText = useCallback(
    (value: string) => {
      const composer = aui.composer();
      if (!composer.getState().isEditing) {
        return;
      }
      flushResourcesSync(() => {
        composer.setText(value);
      });
    },
    [aui],
  );

  const onCompositionStart = useCallback(() => {
    setCompositionState(true);
  }, [setCompositionState]);

  const onCompositionEnd = useCallback(
    (e: CompositionEvent<HTMLTextAreaElement>) => {
      setCompositionState(false);
      setComposerText(e.currentTarget.value);
    },
    [setComposerText, setCompositionState],
  );

  const onChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      setCompositionState(isNativeComposing(e.nativeEvent));
      setComposerText(e.target.value);
    },
    [setComposerText, setCompositionState],
  );

  return {
    inputProps: {
      onCompositionStart,
      onCompositionEnd,
      onChange,
    },
    isComposing,
    isComposingRef: composingRef,
  };
}

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


const ReasoningToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const supportsReasoning = useChatRuntimeStore((s) => s.supportsReasoning);
  const reasoningAlwaysOn = useChatRuntimeStore((s) => s.reasoningAlwaysOn);
  const reasoningEnabled = useChatRuntimeStore((s) => s.reasoningEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const reasoningStyle = useChatRuntimeStore((s) => s.reasoningStyle);
  const reasoningEffort = useChatRuntimeStore((s) => s.reasoningEffort);
  const supportsReasoningOff = useChatRuntimeStore((s) => s.supportsReasoningOff);
  const reasoningEffortLevels = useChatRuntimeStore((s) => s.reasoningEffortLevels);
  const setReasoningEffort = useChatRuntimeStore((s) => s.setReasoningEffort);
  const lastOpenRouterChosenModel = useChatRuntimeStore(
    (s) => s.lastOpenRouterChosenModel,
  );
  const externalProviders = useExternalProvidersStore((s) => s.providers);
  const externalSelection = parseExternalModelId(checkpoint);
  const selectedExternalProvider =
    externalSelection != null
      ? externalProviders.find((p) => p.id === externalSelection.providerId)
      : undefined;
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
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
  const effectiveReasoningEnabled = reasoningLockedOn ? true : reasoningEnabled;
  const effectiveReasoningVisualEnabled =
    effectiveReasoningEnabled && reasoningEffort !== "none";
  const disabled = !(modelLoaded && effectiveSupportsReasoning);
  const formatEffortLabel = (level: typeof reasoningEffort): string => {
    if (level !== "xhigh") return level.charAt(0).toUpperCase() + level.slice(1);
    const normalized = externalSelection?.modelId?.trim().toLowerCase() ?? "";
    if (
      normalized.startsWith("claude-opus-4-6") ||
      normalized.startsWith("claude-sonnet-4-6")
    ) {
      return "Max";
    }
    return "Extra High";
  };
  const effortLabel = formatEffortLabel(reasoningEffort);

  if (effectiveReasoningStyle === "reasoning_effort") {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            disabled={disabled}
            className={cn(
              "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
              disabled
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
              Think: {effectiveReasoningVisualEnabled ? effortLabel : "None"}
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
              None
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
                // Kimi's $web_search builtin forbids thinking, so
                // enabling thinking flips the Search pill off.
                if (isKimiExternal && toolsEnabled) {
                  setToolsEnabled(false);
                }
              }}
            >
              {formatEffortLabel(level)}
              {effectiveReasoningVisualEnabled && reasoningEffort === level ? " \u2713" : ""}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  return (
    <button
      type="button"
      disabled={disabled || reasoningLockedOn}
      aria-disabled={disabled || reasoningLockedOn}
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
        // Mutual exclusion with the Search pill on Kimi — see the
        // dropdown branch above and shared-composer for the same rule.
        if (isKimiExternal && next && toolsEnabled) {
          setToolsEnabled(false);
        }
      }}
      className="composer-pill-btn"
      data-active={
        reasoningLockedOn || (effectiveReasoningEnabled && !disabled)
          ? "true"
          : "false"
      }
      aria-label={
        reasoningLockedOn
          ? "Thinking is required for this model"
          : effectiveReasoningEnabled
            ? "Disable thinking"
            : "Enable thinking"
      }
    >
      {reasoningLockedOn || (effectiveReasoningEnabled && !disabled) ? (
        <LightbulbIcon className="size-3.5" />
      ) : (
        <LightbulbOffIcon className="size-3.5" />
      )}
      <span>Think</span>
    </button>
  );
};

const PreserveThinkingToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const supportsPreserveThinking = useChatRuntimeStore(
    (s) => s.supportsPreserveThinking,
  );
  const preserveThinking = useChatRuntimeStore((s) => s.preserveThinking);
  const setPreserveThinking = useChatRuntimeStore((s) => s.setPreserveThinking);
  if (!supportsPreserveThinking) return null;
  const disabled = !modelLoaded;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => setPreserveThinking(!preserveThinking)}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        disabled
          ? "cursor-not-allowed opacity-40"
          : preserveThinking
            ? "bg-primary/10 text-primary hover:bg-primary/20"
            : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
      )}
      aria-label={
        preserveThinking ? "Disable preserve think" : "Enable preserve think"
      }
    >
      {preserveThinking && !disabled ? (
        <LightbulbIcon className="size-3.5" />
      ) : (
        <LightbulbOffIcon className="size-3.5" />
      )}
      <span>Preserve Think</span>
    </button>
  );
};

const WebSearchToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  // External providers (OpenAI today) expose a server-side web_search tool
  // even when the local tool runtime is unavailable — gate the Search pill
  // on either source so it lights up on external models too. Mirror of
  // shared-composer's searchDisabled.
  const supportsBuiltinWebSearch = useChatRuntimeStore(
    (s) => s.supportsBuiltinWebSearch,
  );
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const externalProviders = useExternalProvidersStore((s) => s.providers);
  const externalSelection = parseExternalModelId(checkpoint);
  const selectedExternalProvider =
    externalSelection != null
      ? externalProviders.find((p) => p.id === externalSelection.providerId)
      : undefined;
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";
  const disabled =
    !modelLoaded || !(supportsTools || supportsBuiltinWebSearch);

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => {
        const next = !toolsEnabled;
        setToolsEnabled(next);
        // Kimi's $web_search builtin requires thinking=disabled (see
        // https://platform.kimi.ai/docs/guide/use-web-search). Keep
        // the two pills mutually exclusive so the visible state always
        // matches what the backend ends up sending.
        if (isKimiExternal) {
          setReasoningEnabled(!next);
          applyQwenThinkingParams(!next);
        }
      }}
      className="composer-pill-btn"
      data-active={toolsEnabled && !disabled ? "true" : "false"}
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
      className="composer-pill-btn"
      data-active={codeToolsEnabled && !disabled ? "true" : "false"}
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

const ComposerAction: FC<{ disabled?: boolean; blockSend?: () => boolean }> = ({
  disabled,
  blockSend,
}) => {
  return (
    <div className="aui-composer-action-wrapper composer-action-wrapper">
      <div className="flex items-center gap-1">
        <ComposerAddAttachment />
        <ComposerAudioUpload />
        <ReasoningToggle />
        <PreserveThinkingToggle />
        <WebSearchToggle />
        <CodeToolsToggle />
      </div>
      <div className="flex items-center gap-1">
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
          <ComposerPrimitive.Send asChild={true}>
            <TooltipIconButton
              tooltip="Send message"
              side="bottom"
              type="submit"
              variant="default"
              size="icon"
              disabled={disabled}
              onClick={(event) => {
                if (blockSend?.()) {
                  event.preventDefault();
                }
              }}
              className="aui-composer-send size-8 rounded-full"
              aria-label="Send message"
            >
              <ArrowUpIcon className="aui-composer-send-icon size-4" />
            </TooltipIconButton>
          </ComposerPrimitive.Send>
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
      className="aui-assistant-message-root relative mx-auto min-w-0 w-full max-w-(--thread-content-max-width) pt-0.5 pb-4 text-[15.5px] [font-weight:410] tracking-[0.01em] dark:tracking-[0.02em]"
      data-role="assistant"
    >
      <div className="aui-assistant-message-content wrap-break-word min-w-0 text-[#0d0d0d] dark:text-foreground leading-relaxed">
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

      <div className="aui-assistant-message-footer mt-1.5 -ml-[var(--icon-btn-inset)] flex min-h-8">
        <BranchPicker className="mr-0.5" />
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
      className="text-chat-icon-fg hover:text-destructive"
    >
      <HugeiconsIcon
        icon={Delete02Icon}
        strokeWidth={1.75}
        className="size-icon"
      />
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
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        strokeWidth={1.75}
        className="size-icon"
      />
    </TooltipIconButton>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning={true}
      className="aui-assistant-action-bar-root col-start-3 row-start-2 flex items-center gap-1 text-chat-icon-fg [&_button:not([data-slot=message-timing-trigger])]:size-8 [&_button]:!rounded-[10px] [&_button:hover]:bg-chat-icon-bg-hover [&_button:hover]:text-chat-icon-fg-hover"
    >
      <CopyButton />
      <ActionBarPrimitive.Reload asChild={true}>
        <TooltipIconButton tooltip="Refresh">
          <RefreshCwIcon strokeWidth={1.75} className="size-icon" />
        </TooltipIconButton>
      </ActionBarPrimitive.Reload>
      <DeleteMessageButton />
      <ActionBarMorePrimitive.Root>
        <ActionBarMorePrimitive.Trigger asChild={true}>
          <TooltipIconButton
            tooltip="More"
            className="data-[state=open]:bg-accent"
          >
            <MoreHorizontalIcon strokeWidth={1.75} className="size-icon" />
          </TooltipIconButton>
        </ActionBarMorePrimitive.Trigger>
        <ActionBarMorePrimitive.Content
          side="bottom"
          align="start"
          onCloseAutoFocus={(e) => e.preventDefault()}
          className="aui-action-bar-more-content z-50 min-w-32 overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
        >
          <ActionBarPrimitive.ExportMarkdown asChild={true}>
            <ActionBarMorePrimitive.Item className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
              <DownloadIcon strokeWidth={1.75} className="size-icon" />
              Export as Markdown
            </ActionBarMorePrimitive.Item>
          </ActionBarPrimitive.ExportMarkdown>
        </ActionBarMorePrimitive.Content>
      </ActionBarMorePrimitive.Root>
      <MessageTiming side="top" className="h-8 px-2" />
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
      className="aui-user-message-root fade-in slide-in-from-bottom-1 mx-auto flex w-full max-w-(--thread-content-max-width) animate-in flex-col items-end gap-y-2 pt-6 pb-4 text-[15.5px] [font-weight:410] tracking-[0.01em] dark:tracking-[0.02em] duration-150"
      data-role="user"
    >
      <UserMessageAttachments />
      <UserMessageAudio />

      <div className="aui-user-message-content-wrapper flex max-w-[80%] min-w-0 flex-col items-end">
        <div className="aui-user-message-content wrap-break-word w-fit rounded-[24px] bg-[#f5f5f5] px-4 py-2.5 text-[#0d0d0d] dark:text-foreground dark:bg-card">
          <MessagePrimitive.Parts />
        </div>
        <div className="mt-1 -mr-[var(--icon-btn-inset)] flex min-h-8 items-center">
          <UserActionBar />
          <BranchPicker className="aui-user-branch-picker ml-0.5" />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      autohide="always"
      className="aui-user-action-bar-root flex gap-1 text-chat-icon-fg [&_button]:size-8 [&_button]:!rounded-[10px] [&_button:hover]:bg-chat-icon-bg-hover [&_button:hover]:text-chat-icon-fg-hover"
    >
      <CopyButton />
      <ActionBarPrimitive.Edit asChild={true}>
        <TooltipIconButton tooltip="Edit" className="aui-user-action-edit">
          <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
        </TooltipIconButton>
      </ActionBarPrimitive.Edit>
      <DeleteMessageButton />
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  const aui = useAui();
  const { inputProps, isComposingRef } = useImeComposerInputHandlers();
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
          {...inputProps}
        />
        <div className="aui-edit-composer-footer mx-3 mb-3 flex items-center gap-2 self-end">
          <ComposerPrimitive.Cancel asChild={true}>
            <Button type="button" variant="ghost" size="sm">
              Cancel
            </Button>
          </ComposerPrimitive.Cancel>
          <Button
            type="button"
            size="sm"
            onClick={(event) => {
              if (isComposingRef.current) {
                event.preventDefault();
                return;
              }
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
        "aui-branch-picker-root inline-flex items-center text-chat-icon-fg text-[13px]",
        className,
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous asChild={true}>
        <button
          type="button"
          aria-label="Previous"
          className="aui-branch-chevron-btn"
        >
          <ChevronLeftIcon strokeWidth={1.25} className="size-[36px]" />
        </button>
      </BranchPickerPrimitive.Previous>
      <span className="aui-branch-picker-state font-mono text-[13px] tabular-nums">
        <BranchPickerPrimitive.Number />/<BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild={true}>
        <button
          type="button"
          aria-label="Next"
          className="aui-branch-chevron-btn"
        >
          <ChevronRightIcon strokeWidth={1.25} className="size-[36px]" />
        </button>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};
