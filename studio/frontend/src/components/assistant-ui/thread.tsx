// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ComposerAddAttachment,
  ComposerAttachments,
  UserMessageAttachments,
} from "@/components/assistant-ui/attachment";
import { MessageTiming } from "@/components/assistant-ui/message-timing";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { Reasoning, ReasoningGroup } from "@/components/assistant-ui/reasoning";
import { Sources } from "@/components/assistant-ui/sources";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { ToolGroup } from "@/components/assistant-ui/tool-group";
import { WebSearchToolUI } from "@/components/assistant-ui/tool-ui-web-search";
import { PythonToolUI } from "@/components/assistant-ui/tool-ui-python";
import { TerminalToolUI } from "@/components/assistant-ui/tool-ui-terminal";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { sentAudioNames } from "@/features/chat/api/chat-adapter";
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
  SuggestionPrimitive,
  ThreadPrimitive,
  useAui,
  useAuiEvent,
  useAuiState,
} from "@assistant-ui/react";
import { motion } from "framer-motion";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  GlobeIcon,
  HeadphonesIcon,
  LightbulbIcon,
  LightbulbOffIcon,
  MicIcon,
  MoreHorizontalIcon,
  LoaderIcon,
  PencilIcon,
  RefreshCwIcon,
  SquareIcon,
  TerminalIcon,
  XIcon,
} from "lucide-react";
import { type FC, useCallback, useEffect, useRef, useState } from "react";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

export const Thread: FC<{ hideComposer?: boolean; hideWelcome?: boolean }> = ({
  hideComposer,
  hideWelcome,
}) => {
  return (
    <ThreadPrimitive.Root
      className="aui-root aui-thread-root @container flex h-full flex-col "
      style={{
        ["--thread-max-width" as string]: "44rem",
      }}
    >
      <ThreadPrimitive.Viewport
        className="aui-thread-viewport relative flex flex-1 flex-col overflow-x-auto overflow-y-scroll scroll-smooth px-4 pt-4"
      >
        {!hideWelcome && (
          <AuiIf condition={({ thread }) => thread.isEmpty}>
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

        <ThreadPrimitive.ViewportFooter className="aui-thread-viewport-footer sticky bottom-0 mt-auto flex w-full flex-col gap-4 overflow-visible bg-background pb-4 md:pb-4">
          <ThreadScrollToBottom />
          <GeneratingSpinner />
          <AuiIf condition={({ thread }) => !thread.isEmpty}>
            {!hideComposer && <ComposerAnimated />}
          </AuiIf>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild={true}>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="aui-thread-scroll-to-bottom absolute -top-12 z-10 self-center rounded-full p-4 disabled:invisible dark:bg-background dark:hover:bg-accent"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const SuggestionItem: FC = () => {
  const aui = useAui();
  const prompt = useAuiState(({ suggestion }) => suggestion.prompt);
  const isDisabled = useAuiState(({ thread }) => thread.isDisabled);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);

  return (
    <button
      type="button"
      onClick={() => {
        if (!isDisabled && !isRunning) {
          aui.thread().append(prompt);
          aui.composer().setText("");
          return;
        }
        aui.composer().setText(prompt);
      }}
      className="fade-in slide-in-from-bottom-1 animate-in cursor-pointer corner-squircle rounded-xl border bg-background px-4 py-2.5 text-left text-sm text-foreground shadow-sm transition-colors duration-150 hover:bg-accent"
    >
      <SuggestionPrimitive.Title />
    </button>
  );
};

const ThreadWelcome: FC<{ hideComposer?: boolean }> = ({ hideComposer }) => {
  return (
    <div className="aui-thread-welcome-root mx-auto my-auto flex w-full max-w-(--thread-max-width) grow flex-col">
      <div className="aui-thread-welcome-center flex w-full grow flex-col items-center justify-center">
        <div className="aui-thread-welcome-message flex w-full flex-col justify-center gap-6 px-4">
          <div className="flex flex-col items-center gap-2 text-center">
            <img
              src="/Sloth emojis/sloth pc square.png"
              alt="Sloth mascot"
              className="size-20"
            />
            <h1 className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-1 animate-in font-semibold text-2xl duration-200">
              Chat with your model
            </h1>
            <p className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-1 animate-in text-muted-foreground text-base delay-75 duration-200">
              Run GGUFs, safetensors, vision and audio models!
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <ThreadPrimitive.Suggestions
              components={{ Suggestion: SuggestionItem }}
            />
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
  if (!status) return null;
  return (
    <div className="mx-auto flex w-full max-w-(--thread-max-width) items-center justify-center py-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <LoaderIcon className="size-3.5 animate-spin" />
        <span>Generating</span>
      </div>
    </div>
  );
};

const ComposerAnimated: FC = () => {
  return (
    <motion.div
      layout={true}
      layoutId="composer"
      transition={{ type: "spring", bounce: 0.15, duration: 0.5 }}
      className="mx-auto w-full max-w-(--thread-max-width)"
    >
      <Composer />
    </motion.div>
  );
};

const PendingAudioChip: FC = () => {
  const audioName = useChatRuntimeStore((s) => s.pendingAudioName);
  const clearPendingAudio = useChatRuntimeStore((s) => s.clearPendingAudio);
  if (!audioName) return null;
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

const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="aui-composer-root relative flex w-full flex-col">
      <ComposerPrimitive.AttachmentDropzone className="aui-composer-attachment-dropzone shadow-border ring-1 ring-border flex w-full flex-col rounded-2xl bg-background px-1 pt-2 outline-none transition-shadow data-[dragging=true]:ring-ring data-[dragging=true]:bg-accent/50">
        <ComposerAttachments />
        <PendingAudioChip />
        <ToolStatusDisplay />
        <ComposerPrimitive.Input
          placeholder="Send a message..."
          className="aui-composer-input mb-1 max-h-32 min-h-12 w-full resize-none bg-transparent px-4 pt-2 pb-3 text-sm outline-none placeholder:text-muted-foreground focus-visible:ring-0"
          rows={1}
          autoFocus={true}
          aria-label="Message input"
        />
        <ComposerAction />
      </ComposerPrimitive.AttachmentDropzone>
    </ComposerPrimitive.Root>
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
      if (file.size > MAX_AUDIO_SIZE) return;
      try {
        const base64 = await fileToBase64(file);
        setPendingAudio(base64, file.name);
      } catch {
        // skip
      }
    },
    [setPendingAudio],
  );

  if (!activeModel?.hasAudioInput) return null;

  return (
    <>
      <input
        ref={audioInputRef}
        type="file"
        accept={AUDIO_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleAudioFile(file);
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
  if (!checkpoint.includes("qwen3")) return;
  // Qwen3 & Qwen3.5 share the same recommended settings:
  // Thinking ON (general): temp=1.0, top_p=0.95, top_k=20
  // Thinking OFF (general): temp=0.7, top_p=0.8, top_k=20
  const params = thinkingOn
    ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
  store.setParams({ ...store.params, ...params });
}

const ReasoningToggle: FC = () => {
  const supportsReasoning = useChatRuntimeStore((s) => s.supportsReasoning);
  const reasoningEnabled = useChatRuntimeStore((s) => s.reasoningEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);

  if (!supportsReasoning) return null;

  return (
    <button
      type="button"
      onClick={() => {
        const next = !reasoningEnabled;
        setReasoningEnabled(next);
        applyQwenThinkingParams(next);
      }}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        reasoningEnabled
          ? "bg-primary/10 text-primary hover:bg-primary/20"
          : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
      )}
      aria-label={reasoningEnabled ? "Disable thinking" : "Enable thinking"}
    >
      {reasoningEnabled ? (
        <LightbulbIcon className="size-3.5" />
      ) : (
        <LightbulbOffIcon className="size-3.5" />
      )}
      <span>Think</span>
    </button>
  );
};

const WebSearchToggle: FC = () => {
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);

  if (!supportsTools) return null;

  return (
    <button
      type="button"
      onClick={() => setToolsEnabled(!toolsEnabled)}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        toolsEnabled
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
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const setCodeToolsEnabled = useChatRuntimeStore(
    (s) => s.setCodeToolsEnabled,
  );

  if (!supportsTools) return null;

  return (
    <button
      type="button"
      onClick={() => setCodeToolsEnabled(!codeToolsEnabled)}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        codeToolsEnabled
          ? "bg-primary/10 text-primary hover:bg-primary/20"
          : "bg-muted text-muted-foreground hover:bg-muted-foreground/15",
      )}
      aria-label={codeToolsEnabled ? "Disable code execution" : "Enable code execution"}
    >
      <TerminalIcon className="size-3.5" />
      <span>Code</span>
    </button>
  );
};

const ToolStatusDisplay: FC = () => {
  const toolStatus = useChatRuntimeStore((s) => s.toolStatus);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!toolStatus) {
      setElapsed(0);
      return;
    }
    setElapsed(0);
    const interval = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [toolStatus]);

  if (!toolStatus) return null;
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

const ComposerAction: FC = () => {
  return (
    <div className="aui-composer-action-wrapper relative mx-2 mb-2 flex items-center justify-between">
      <div className="flex items-center gap-1">
        <ComposerAddAttachment />
        <ComposerAudioUpload />
        <ReasoningToggle />
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

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="aui-assistant-message-root fade-in slide-in-from-bottom-1 relative mx-auto w-full max-w-(--thread-max-width) animate-in py-3 duration-150"
      data-role="assistant"
    >
      <div className="aui-assistant-message-content wrap-break-word px-2 text-foreground leading-relaxed">
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
        <MessageError />
      </div>

      <div className="aui-assistant-message-footer mt-1 ml-2 flex">
        <BranchPicker />
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const COPY_RESET_MS = 2000;

const CopyButton: FC = () => {
  const aui = useAui();
  const [copied, setCopied] = useState(false);
  const resetTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleCopy = () => {
    const text = aui.message().getCopyText();
    if (copyToClipboard(text)) {
      setCopied(true);
      if (resetTimeoutRef.current) clearTimeout(resetTimeoutRef.current);
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
      autohide="not-last"
      autohideFloat="single-branch"
      className="aui-assistant-action-bar-root col-start-3 row-start-2 -ml-1 flex gap-1 text-muted-foreground data-floating:absolute data-floating:rounded-md data-floating:border data-floating:bg-background data-floating:p-1 data-floating:shadow-sm"
    >
      <CopyButton />
      <ActionBarPrimitive.Reload asChild={true}>
        <TooltipIconButton tooltip="Refresh">
          <RefreshCwIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Reload>
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
  const audioName = useAuiState(({ message }) => sentAudioNames.get(message.id));
  if (!audioName) return null;
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
      className="aui-user-message-root  fade-in slide-in-from-bottom-1 mx-auto grid w-full max-w-(--thread-max-width) animate-in auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] content-start gap-y-2 px-2 py-3 duration-150 [&:where(>*)]:col-start-2"
      data-role="user"
    >
      <UserMessageAttachments />
      <UserMessageAudio />

      <div className="aui-user-message-content-wrapper relative col-start-2 min-w-0">
        <div className="aui-user-message-content wrap-break-word rounded-2xl bg-muted  px-4 py-2.5 text-foreground">
          <MessagePrimitive.Parts />
        </div>
        <div className="aui-user-action-bar-wrapper absolute top-1/2 left-0 -translate-x-full -translate-y-1/2 pr-2">
          <UserActionBar />
        </div>
      </div>

      <BranchPicker className="aui-user-branch-picker col-span-full col-start-1 row-start-3 -mr-1 justify-end" />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      autohide="not-last"
      className="aui-user-action-bar-root flex items-center"
    >
      <CopyButton />
      <ActionBarPrimitive.Edit asChild={true}>
        <TooltipIconButton tooltip="Edit" className="aui-user-action-edit">
          <PencilIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Edit>
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
    <MessagePrimitive.Root className="aui-edit-composer-wrapper mx-auto flex w-full max-w-(--thread-max-width) flex-col px-2 py-3">
      <ComposerPrimitive.Root className="aui-edit-composer-root ml-auto flex w-full max-w-[85%] flex-col rounded-2xl bg-muted">
        <ComposerPrimitive.Input
          className="aui-edit-composer-input min-h-14 w-full resize-none bg-transparent p-4 text-foreground text-sm outline-none"
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
