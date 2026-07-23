// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ComposerAttachments,
  UserMessageAttachments,
} from "@/components/assistant-ui/attachment";
import {
  GeneratedImageOverlayProvider,
  useGeneratedImageOverlay,
} from "@/components/assistant-ui/generated-image-overlay-context";
import { downloadImagePart } from "@/components/assistant-ui/image";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { MessageTiming } from "@/components/assistant-ui/message-timing";
import { Reasoning, ReasoningGroup } from "@/components/assistant-ui/reasoning";
import { RagSourcesGroup } from "@/components/assistant-ui/rag-sources";
import { Sources, SourcesGroup } from "@/components/assistant-ui/sources";
import {
  thinkEffortAriaLabel,
  thinkToggleAriaLabel,
} from "@/components/assistant-ui/think-aria-label";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { ToolGroup } from "@/components/assistant-ui/tool-group";
import { CodeExecutionToolUI } from "@/components/assistant-ui/tool-ui-code-execution";
import { ImageGenerationToolUI } from "@/components/assistant-ui/tool-ui-image-generation";
import { KnowledgeBaseToolUI } from "@/components/assistant-ui/tool-ui-knowledge-base";
import { RenderHtmlToolUI } from "@/components/assistant-ui/tool-ui-render-html";
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
import { Spinner } from "@/components/ui/spinner";
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
import { sentAudioNames } from "@/features/chat/api/chat-adapter";
import {
  PromptStorageDialog,
  exportConversationShareGPT,
  exportConversationRawJsonl,
  exportConversationCsv,
} from "@/features/chat/prompt-storage/prompt-storage-dialog";
import {
  listPromptEntries,
  type PromptEntry,
} from "@/features/chat/api/prompts-api";
import { useChatProjects } from "@/features/chat/hooks/use-chat-projects";
import { NewProjectDialog } from "@/features/chat/components/new-project-dialog";
import { parseExternalModelId } from "@/features/chat/external-providers";
import { McpComposerButton } from "@/features/chat/mcp-composer-button";
import { getExternalReasoningCapabilities } from "@/features/chat/provider-capabilities";
import { useRagToolDisabled } from "@/features/chat/hooks/use-rag-tool-disabled";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { deleteThreadMessage } from "@/features/chat/utils/delete-thread-message";
import { ThreadDocumentsBar } from "@/features/rag/components/thread-documents-bar";
import { KnowledgeBaseComposerButton } from "@/features/rag/components/knowledge-base-composer-button";
import { DocumentPreviewMount } from "@/features/rag/components/document-preview-mount";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { applyQwenThinkingParams } from "@/features/chat/utils/qwen-params";
import { isTauri } from "@/lib/api-base";
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE, fileToBase64 } from "@/lib/audio-utils";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
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
  AttachmentIcon,
  Bookmark02Icon,
  CodeIcon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  FileDatabaseIcon,
  Folder01Icon,
  FolderAddIcon,
  Image03Icon,
  McpServerIcon,
  PencilRulerIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  Columns2Icon,
  GlobeIcon,
  HeadphonesIcon,
  MoreHorizontalIcon,
  PlusIcon,
  RefreshCwIcon,
  SquareIcon,
  TerminalIcon,
  XIcon,
} from "lucide-react";
import {
  type ChangeEvent,
  type ComponentProps,
  type CompositionEvent,
  type FC,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

// True while a file is dragged anywhere over the chat page, so the composer
// can show its "Drop files here" affordance.
const PageDragContext = createContext(false);

// Single-chat prompt queue. State lives at module level so it survives the
// Composer remount when the first queued message creates a new thread, and
// detection subscribes to the store's runningByThreadId rather than
// aui.thread() (unbound on the welcome screen).

import { create as _createZustand } from "zustand";

// Module-level Zustand so ComposerRightControls re-renders across Composer mounts.
interface _QueueUIState { isRunning: boolean; current: number; total: number; }
const _useQueueUI = _createZustand<_QueueUIState>(() => ({
  isRunning: false, current: 0, total: 0,
}));

let _qItems: string[] = [];
let _qIndex = 0;
let _qIsRunning = false;
let _qPrevStoreRunning = false;
let _qStoreUnsub: (() => void) | null = null;

// Points to the current Composer's aui (updated every render), so it stays valid
// after a remount.
let _qGetAui: () => ReturnType<typeof useAui> = () => {
  throw new Error("aui not initialised");
};

function _qStopSubscription() {
  if (_qStoreUnsub) { _qStoreUnsub(); _qStoreUnsub = null; }
  _qPrevStoreRunning = false;
}

function _qAdvance() {
  const nextIndex = _qIndex + 1;
  if (nextIndex >= _qItems.length) {
    _qIsRunning = false;
    _qItems = [];
    _qIndex = 0;
    _qStopSubscription();
    _useQueueUI.setState({ isRunning: false, current: 0, total: 0 });
    toast.success("Prompt queue complete");
    return;
  }
  _qIndex = nextIndex;
  _useQueueUI.setState({ current: nextIndex + 1, total: _qItems.length });
  const next = _qItems[nextIndex];
  toast(`Prompt ${nextIndex + 1} / ${_qItems.length}`, {
    description: next.length > 80 ? next.slice(0, 80) + "…" : next,
  });
  _qPrevStoreRunning = false; // catch the next run
  setTimeout(() => {
    _qGetAui().thread().append({
      role: "user",
      content: [{ type: "text", text: next }],
      createdAt: new Date(),
    } as never);
  }, 100);
}

function _qStartSubscription() {
  _qStopSubscription();
  // runningByThreadId tracks the actual thread (not aui.thread()), so detection
  // survives navigation.
  _qStoreUnsub = useChatRuntimeStore.subscribe((state) => {
    if (!_qIsRunning) { _qStopSubscription(); return; }
    const isRunning = Object.keys(state.runningByThreadId).length > 0;
    const wasRunning = _qPrevStoreRunning;
    _qPrevStoreRunning = isRunning;
    if (wasRunning && !isRunning) {
      _qAdvance();
    }
  });
}

interface _QueueCallbacks { startQueue: (items: string[]) => void; stopQueue: () => void; }
const PromptQueueContext = createContext<_QueueCallbacks>({
  startQueue: () => {}, stopQueue: () => {},
});

// Gap (px) between last message and floating composer; bottom spacer tracks
// composer height plus this gap so chat can scroll fully above the composer.
const COMPOSER_SCROLL_GAP_PX = 24;
// The scroll-to-bottom footer sits 10px below the spacer top.
const FOOTER_GAP_BELOW_SPACER_PX = 10;
// Window after a run start during which composer shrinks apply immediately:
// the run-start pin owns the bottom, so the clamp is the intended glide.
// Covers instant responses where isRunning is already false by resize time.
const RUN_SHRINK_WINDOW_MS = 1000;

export const Thread: FC<{
  hideComposer?: boolean;
  hideWelcome?: boolean;
  targetThreadId?: string;
}> = ({ hideComposer, hideWelcome, targetThreadId }) => {
  // Intent-aware autoscroll replaces assistant-ui's built-in autoscroll to
  // prevent the streaming-mutation race that snaps the viewport back to the
  // bottom while the user scrolls up (see the hook for the full explanation).
  const { ref: viewportRef, context: autoScrollContext } =
    useIntentAwareAutoScroll();

  const isComposerAttachPending = useAuiState(({ threads }) =>
    targetThreadId ? threads.mainThreadId !== targetThreadId : false,
  );
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const threadId = targetThreadId ?? activeThreadId ?? null;
  const aui = useAui();

  // Measured height of the floating composer dock (null until measured).
  // Drives the bottom spacer and the scroll-to-bottom footer offset.
  const [composerHeight, setComposerHeight] = useState<number | null>(null);
  const footerBottomPx =
    composerHeight == null
      ? null
      : composerHeight + COMPOSER_SCROLL_GAP_PX - FOOTER_GAP_BELOW_SPACER_PX;

  // Viewport element is owned by the autoscroll hook; mirror it locally for
  // the spacer clamp math. State, not a ref: the keyed provider remounts the
  // viewport on thread switches and the scroll listener must re-attach.
  const [viewportEl, setViewportEl] = useState<HTMLElement | null>(null);
  const composedViewportRef = useCallback(
    (node: HTMLElement | null) => {
      setViewportEl(node);
      viewportRef(node);
    },
    [viewportRef],
  );

  // Bottom spacer sizing. Invariant: chat never moves on its own on composer
  // resize.
  // - Grow (attachment added, multiline): grow at once; growth below the
  //   scroll position is invisible and only adds room.
  // - Shrink (attachment removed): shrinking scrollHeight near the bottom
  //   clamps scrollTop and yanks the chat down. Defer until invisible (user
  //   scrolled up) or a bottom-pinning moment.
  // Applied imperatively so a remounted spacer can be sized from refs even
  // when composerHeight did not change (e.g. thread switch).
  const spacerElRef = useRef<HTMLDivElement | null>(null);
  const desiredSpacerPxRef = useRef<number | null>(null);
  const appliedSpacerPxRef = useRef<number | null>(null);

  const applySpacerPx = useCallback((px: number) => {
    appliedSpacerPxRef.current = px;
    const node = spacerElRef.current;
    if (node) {
      node.style.height = `${px}px`;
    }
  }, []);

  // Release any deferred shrink; used at moments that pin to the bottom
  // anyway, where the clamp is the intended motion.
  const releaseSpacerExcess = useCallback(() => {
    const desired = desiredSpacerPxRef.current;
    const applied = appliedSpacerPxRef.current;
    if (desired != null && applied != null && applied > desired) {
      applySpacerPx(desired);
    }
  }, [applySpacerPx]);

  const spacerRef = useCallback(
    (node: HTMLDivElement | null) => {
      spacerElRef.current = node;
      // Fresh mounts (thread switch, first message) start at desired size;
      // deferral state from a previous mount is moot.
      const desired = desiredSpacerPxRef.current;
      if (node && desired != null) {
        applySpacerPx(desired);
      }
    },
    [applySpacerPx],
  );

  const prevComposerHeightRef = useRef<number | null>(null);
  // Set on thread.runStart; see RUN_SHRINK_WINDOW_MS.
  const runStartAtRef = useRef(0);
  useLayoutEffect(() => {
    const prev = prevComposerHeightRef.current;
    prevComposerHeightRef.current = composerHeight;
    if (composerHeight == null || hideComposer) {
      desiredSpacerPxRef.current = null;
      appliedSpacerPxRef.current = null;
      spacerElRef.current?.style.removeProperty("height");
      return;
    }
    const desired = composerHeight + COMPOSER_SCROLL_GAP_PX;
    desiredSpacerPxRef.current = desired;
    const applied = appliedSpacerPxRef.current;
    if (applied == null || desired >= applied) {
      applySpacerPx(desired);
    } else {
      const distance = viewportEl
        ? viewportEl.scrollHeight - viewportEl.scrollTop - viewportEl.clientHeight
        : Number.POSITIVE_INFINITY;
      const runOwnsBottom =
        aui.thread().getState().isRunning ||
        performance.now() - runStartAtRef.current < RUN_SHRINK_WINDOW_MS;
      // At the bottom the shrink only drops blank spacer, so apply it now
      // rather than strand dead space until the next pin.
      if (
        runOwnsBottom ||
        distance >= applied - desired ||
        autoScrollContext.getIsAtBottom()
      ) {
        applySpacerPx(desired);
      }
      // else: deferred; released on scroll or a bottom-pinning event.
    }
    if (prev != null && composerHeight > prev) {
      // Chat is now above the new bottom. Detach as if the user scrolled up
      // so no later signal re-pins and shoves the chat up (scrolling back
      // down re-attaches; explicit pins still work). Skip mid-run: that
      // growth is tool-status rows, not the user, and detaching would break
      // streaming autoscroll.
      if (!aui.thread().getState().isRunning) {
        autoScrollContext.detachFromBottom();
      }
    }
  }, [composerHeight, hideComposer, autoScrollContext, aui, applySpacerPx, viewportEl]);

  // Drop deferred spacer excess once the user has scrolled far enough above
  // the bottom that the shrink cannot clamp scrollTop. Keyed on viewportEl
  // so the listener follows viewport remounts.
  useEffect(() => {
    const el = viewportEl;
    if (!el) {
      return;
    }
    const onScroll = () => {
      const desired = desiredSpacerPxRef.current;
      const applied = appliedSpacerPxRef.current;
      if (desired == null || applied == null || applied <= desired) {
        return;
      }
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (distance >= applied - desired) {
        applySpacerPx(desired);
      }
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [viewportEl, applySpacerPx]);

  // These pin to the bottom, so releasing the excess here is invisible.
  // runStart also opens the shrink window for the send-clears-chips case.
  useAuiEvent("thread.runStart", () => {
    runStartAtRef.current = performance.now();
    releaseSpacerExcess();
  });
  useAuiEvent("thread.initialize", releaseSpacerExcess);
  useAuiEvent("threadListItem.switchedTo", releaseSpacerExcess);

  // Page-wide drag-and-drop: dropping a file anywhere on the chat page
  // attaches it and shows the composer drop affordance. The composer's own
  // dropzone handles drops on the box and calls preventDefault, so the page
  // handler skips them (no double-add).
  const [pageDragging, setPageDragging] = useState(false);
  const dragDepth = useRef(0);
  const hasFiles = (e: ReactDragEvent) =>
    Array.from(e.dataTransfer?.types ?? []).includes("Files");
  const onDragEnter = (e: ReactDragEvent) => {
    if (isTauri || !hasFiles(e)) return;
    dragDepth.current += 1;
    setPageDragging(true);
  };
  const onDragOver = (e: ReactDragEvent) => {
    if (isTauri || !hasFiles(e)) return;
    e.preventDefault();
  };
  const onDragLeave = (e: ReactDragEvent) => {
    if (isTauri || !hasFiles(e)) return;
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setPageDragging(false);
  };
  const onDrop = (e: ReactDragEvent) => {
    if (isTauri) return;
    dragDepth.current = 0;
    setPageDragging(false);
    // Compare panes hide this composer and use the shared composer's own
    // dropzone, so don't capture drops into a hidden composer here.
    if (hideComposer) return;
    // Drops on the composer box are handled by its dropzone (preventDefault);
    // skip those here so the file isn't added twice.
    if (e.defaultPrevented) return;
    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    e.preventDefault();
    for (const file of files) {
      aui
        .composer()
        .addAttachment(file)
        .catch(() => {
          // Adapter shows its own toast (e.g. "Load a model before adding images").
        });
    }
  };

  return (
    <GeneratedImageOverlayProvider key={threadId ?? "default"} threadId={threadId}>
      <PageDragContext.Provider value={pageDragging}>
      <ThreadPrimitive.Root
        className="aui-root aui-thread-root @container relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden"
        style={{
          ["--thread-max-width" as string]: "48rem",
          ["--thread-content-max-width" as string]:
            "calc(var(--thread-max-width) - 1.5rem)",
        }}
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <IntentAwareScrollProvider value={autoScrollContext}>
          <ThreadPrimitive.Viewport
            ref={composedViewportRef}
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
              <AuiIf
                condition={({ thread }) => thread.isEmpty && !thread.isLoading}
              >
                <ThreadWelcome hideComposer={hideComposer} threadId={threadId} />
              </AuiIf>
            )}

            <ThreadPrimitive.Messages
              components={{
                UserMessage,
                EditComposer,
                AssistantMessage,
              }}
            />

            {/* Bottom slack so the last message has room above the sticky
            scroll-to-bottom button (and floating composer in single mode),
            instead of butting against the footer. */}
            <AuiIf condition={({ thread }) => hideWelcome || !thread.isEmpty}>
              <div
                ref={spacerRef}
                className={cn(
                  "shrink-0",
                  hideComposer
                    ? "h-16"
                    : composerHeight == null
                      ? "h-40"
                      : undefined,
                )}
                aria-hidden={true}
              />
            </AuiIf>

            <AuiIf condition={({ thread }) => hideWelcome || !thread.isEmpty}>
              <ThreadPrimitive.ViewportFooter
                className={cn(
                  "aui-thread-viewport-footer pointer-events-none sticky z-20 flex w-full justify-center bg-transparent",
                  // 150px (was 140px) to add a small gap above the composer
                  hideComposer
                    ? "bottom-3"
                    : footerBottomPx == null
                      ? "bottom-[150px]"
                      : undefined,
                )}
                style={
                  !hideComposer && footerBottomPx != null
                    ? { bottom: footerBottomPx }
                    : undefined
                }
              >
                <ThreadScrollToBottom />
              </ThreadPrimitive.ViewportFooter>
            </AuiIf>
          </ThreadPrimitive.Viewport>

          <GeneratedImageViewportOverlay
            hideComposer={hideComposer}
            bottomOffsetPx={footerBottomPx}
          />

          {!hideComposer && (
            <AuiIf condition={({ thread }) => hideWelcome || !thread.isEmpty}>
              <ThreadComposerDock
                disabled={isComposerAttachPending}
                threadId={threadId}
                onHeightChange={setComposerHeight}
              />
            </AuiIf>
          )}
        </IntentAwareScrollProvider>
      </ThreadPrimitive.Root>
      {/* Document preview, opened by citation badges. */}
      <DocumentPreviewMount />
      </PageDragContext.Provider>
    </GeneratedImageOverlayProvider>
  );
};

const GeneratedImageViewportOverlay: FC<{
  hideComposer?: boolean;
  bottomOffsetPx?: number | null;
}> = ({ hideComposer, bottomOffsetPx }) => {
  const { overlay, closeOverlay } = useGeneratedImageOverlay();

  useEffect(() => {
    if (!overlay) {
      return;
    }
    document.querySelector<HTMLTextAreaElement>(".aui-composer-input")?.focus();
  }, [overlay]);

  if (!overlay) {
    return null;
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-30">
      <button
        type="button"
        className="pointer-events-auto absolute inset-0 bg-background/65 backdrop-blur-[1px] dark:bg-background/55"
        onClick={closeOverlay}
        aria-label="Close generated image preview"
      />
      <section
        className={cn(
          "pointer-events-none absolute inset-x-5 top-[48px] flex flex-col items-center",
          hideComposer
            ? "bottom-4"
            : bottomOffsetPx == null
              ? "bottom-[150px]"
              : undefined,
        )}
        style={
          !hideComposer && bottomOffsetPx != null
            ? { bottom: bottomOffsetPx }
            : undefined
        }
        aria-label="Generated image preview"
      >
        <div className="pointer-events-auto relative flex min-h-0 w-full max-w-[1100px] flex-1 flex-col items-center justify-center gap-3 rounded-3xl bg-muted/10 p-3 ring-1 ring-border/20">
          <div className="absolute inset-x-3 top-3 z-10 flex justify-end">
            <div className="flex shrink-0 items-center gap-1 rounded-full bg-background/70 p-1 ring-1 ring-border/20 backdrop-blur-sm">
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="size-7 rounded-full"
                onClick={() =>
                  downloadImagePart({
                    image: overlay.image,
                    filename: overlay.filename,
                  })
                }
                aria-label="Download generated image"
              >
                <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="size-7 rounded-full"
                onClick={closeOverlay}
                aria-label="Close generated image preview"
              >
                <XIcon className="size-3.5" />
              </Button>
            </div>
          </div>
          <div className="flex min-h-0 flex-1 items-center justify-center pt-1">
            <img
              src={overlay.image}
              alt={overlay.title}
              className="max-h-full max-w-full object-contain"
            />
          </div>
          <div
            className="w-full max-w-[min(100%,46rem)] shrink-0 text-center"
            title={overlay.title}
          >
            <p className="truncate text-xs font-semibold text-foreground/80">
              Generated image
            </p>
            {overlay.metadata ? (
              <p className="truncate text-[11px] font-medium text-muted-foreground">
                {overlay.metadata}
              </p>
            ) : null}
            {hideComposer ? null : (
              <p className="mx-auto mt-2 inline-flex rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                Type edits below, then send
              </p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
};

const ThreadComposerDock: FC<{
  disabled?: boolean;
  threadId?: string | null;
  onHeightChange?: (height: number | null) => void;
}> = ({ disabled, threadId, onHeightChange }) => {
  const { overlay } = useGeneratedImageOverlay();

  // Report dock height so the viewport reserves matching scroll space when
  // attachments or multiline input grow the composer.
  const dockRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = dockRef.current;
    if (!el || !onHeightChange) return;
    const measure = () => onHeightChange(el.offsetHeight);
    measure();
    const resizeObserver = new ResizeObserver(measure);
    resizeObserver.observe(el);
    return () => {
      resizeObserver.disconnect();
      onHeightChange(null);
    };
  }, [onHeightChange]);

  return (
    <div
      ref={dockRef}
      className={cn(
        "aui-thread-composer-dock pointer-events-none absolute bottom-0 left-0 right-0 md:right-[10px]",
        overlay ? "z-40" : "z-20",
      )}
    >
      {/* Fade the top edge so scrolling text is not cut off by a hard line. */}
      <div
        aria-hidden={true}
        className="absolute inset-x-0 bottom-0 top-[10px] bg-gradient-to-t from-background from-[calc(100%_-_28px)] to-transparent"
      />
      <div className="relative px-5 pb-2">
        <div className="pointer-events-auto mx-auto w-full max-w-(--thread-max-width)">
          <ComposerAnimated
            disabled={disabled}
            threadId={threadId}
            menuSide="top"
          />
        </div>
        <p className="composer-footer-note">
          LLMs can make mistakes. Double-check responses.
        </p>
      </div>
    </div>
  );
};

const ThreadScrollToBottom: FC = () => {
  // State and action both come from our IntentAwareScrollProvider (per-Thread
  // scope, so compare panes are independent). We avoid
  // `ThreadPrimitive.ScrollToBottom` + `useThreadViewport` to stay off
  // assistant-ui's internal autoscroll path (see the hook). The button stays
  // mounted and toggles via CSS; unmounting would trip the hook's
  // MutationObserver as a content change.
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

const pickRandom = <T,>(arr: T[]): T =>
  arr[Math.floor(Math.random() * arr.length)];

// Each greeting carries its matching sloth picture so a line always shows the
// same mascot. Greeting varies by local time; name-bearing lines drop the
// name when none is set.
type Welcome = { text: string; sloth: string };
const DEFAULT_WELCOME: Welcome = {
  text: "What’s on your mind today?",
  sloth: "sloth magnify final.png",
};

function buildWelcome(hour: number, name: string): Welcome {
  const g = (text: string, sloth: string): Welcome => ({ text, sloth });
  // Use the name on ~a third of lines (only direct salutations where it reads
  // naturally); the rest stay name-free so greetings don't feel repetitive.
  const base: Welcome[] = [
    g(name ? `Good to see you, ${name}` : "Good to see you", "large sloth wave.png"),
    g("Ready when you are", "large sloth thumbs.png"),
    DEFAULT_WELCOME,
    g("How can I help?", "sloth sir large.png"),
  ];
  if (hour >= 4 && hour < 9) {
    const morning = g(name ? `Good morning, ${name}` : "Good morning", "large sloth drink.png");
    return pickRandom([...base, morning]);
  }
  if (hour >= 17 && hour < 23) {
    const evening: Welcome[] = [
      g(name ? `Good evening, ${name}` : "Good evening", "sloth shy large.png"),
      g("What’s on for tonight?", "large sloth glasses.png"),
    ];
    // Lean toward an evening line, but a base greeting can still appear.
    return pickRandom(Math.random() < 0.75 ? evening : base);
  }
  if (hour >= 23 || hour < 4) {
    return pickRandom([
      g("Night owl mode?", "large sloth glasses.png"),
      g("Late night ideas?", "large sloth yay.png"),
      g("Up late with an idea?", "large sloth heart.png"),
      g(name ? `The night shift begins, ${name}` : "The night shift begins", "large sloth drink.png"),
    ]);
  }
  return pickRandom(base);
}

const ThreadWelcome: FC<{
  hideComposer?: boolean;
  threadId?: string | null;
}> = ({ hideComposer, threadId }) => {
  const displayName = useUserProfileStore((s) => s.displayName);
  const [welcome, setWelcome] = useState<Welcome>(DEFAULT_WELCOME);

  useEffect(() => {
    // First name only, for a natural greeting; blank falls back to no name.
    const name = displayName.trim().split(/\s+/)[0] ?? "";
    setWelcome(buildWelcome(new Date().getHours(), name));
  }, [displayName]);

  const currentEmojiSrc = `/Sloth emojis/${welcome.sloth}`;

  return (
    <div className="aui-thread-welcome-root mx-auto my-auto flex w-full max-w-(--thread-max-width) grow flex-col">
      <div className="aui-thread-welcome-center flex w-full grow flex-col items-center justify-start pt-[28.5vh]">
        <div className="aui-thread-welcome-message flex w-full flex-col justify-center gap-9 px-4">
          {/* Center the greeting (sloth + title) over the composer. */}
          <div className="flex flex-row items-center justify-center gap-[15px]">
            <img
              src={currentEmojiSrc}
              alt="Sloth mascot"
              className="size-[44px] -translate-y-[2px]"
            />
            <h1 className="aui-thread-welcome-message-inner unsloth-welcome-title fade-in slide-in-from-bottom-1 animate-in text-3xl tracking-[-0.02em] duration-200">
              {welcome.text}
            </h1>
          </div>
          {!hideComposer && <ComposerAnimated threadId={threadId} />}
        </div>
      </div>
    </div>
  );
};

export const ProjectComposer: FC<{
  disabled?: boolean;
  placeholder?: string;
}> = ({ disabled, placeholder }) => {
  return (
    <GeneratedImageOverlayProvider>
      <ComposerAnimated disabled={disabled} placeholder={placeholder} />
    </GeneratedImageOverlayProvider>
  );
};

const ComposerAnimated: FC<{
  disabled?: boolean;
  placeholder?: string;
  threadId?: string | null;
  menuSide?: "top" | "bottom";
}> = ({ disabled, threadId, menuSide }) => {
  return (
    <div className="relative mx-auto min-w-0 w-full max-w-[46rem]">
      <div className="relative z-10 w-full">
        <Composer disabled={disabled} threadId={threadId} menuSide={menuSide} />
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

const Composer: FC<{
  disabled?: boolean;
  placeholder?: string;
  threadId?: string | null;
  menuSide?: "top" | "bottom";
}> = ({ disabled, threadId, menuSide }) => {
  const aui = useAui();
  useAuiEvent("thread.runStart", () => {
    void aui.composer().reset();
  });
  const pageDragging = useContext(PageDragContext);
  const { overlay, closeOverlay } = useGeneratedImageOverlay();
  const setImageToolsEnabled = useChatRuntimeStore(
    (s) => s.setImageToolsEnabled,
  );
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const imageToolsEnabled = useChatRuntimeStore((s) => s.imageToolsEnabled);
  const supportsBuiltinImageGeneration = useChatRuntimeStore(
    (s) => s.supportsBuiltinImageGeneration,
  );
  const artifactsEnabled = useChatRuntimeStore((s) => s.artifactsEnabled);
  const mcpEnabledForChat = useChatRuntimeStore((s) => s.mcpEnabledForChat);
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  // More than 4 pills: collapse to icons only. Search and Code always show;
  // Images, RAG, Canvas and MCP are conditional.
  const pillsCompact =
    2 +
      (ragEnabled ? 1 : 0) +
      (supportsBuiltinImageGeneration ? 1 : 0) +
      (artifactsEnabled ? 1 : 0) +
      (mcpEnabledForChat ? 1 : 0) >
    4;
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const setPendingImageEditReference = useChatRuntimeStore(
    (s) => s.setPendingImageEditReference,
  );
  const { inputProps, isComposing, isComposingRef } =
    useImeComposerInputHandlers();
  const composerText = useAuiState(({ composer }) => composer.text);
  // Expand only once the input wraps to a second line, not on first keystroke.
  // Latch until cleared so it can't flip-flop at the wrap boundary.
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Cache line metrics so getComputedStyle runs once, not per keystroke.
  const lineMetricsRef = useRef<{ lineHeight: number; padding: number } | null>(
    null,
  );
  const [isMultiline, setIsMultiline] = useState(false);
  useEffect(() => {
    if (composerText.length === 0) {
      setIsMultiline(false);
      lineMetricsRef.current = null;
      return;
    }
    // Latched on: stays until the text clears, so skip re-measuring.
    if (isMultiline) return;
    const el = inputRef.current;
    if (!el) {
      return;
    }
    if (!lineMetricsRef.current) {
      const cs = getComputedStyle(el);
      const lineHeight = Number.parseFloat(cs.lineHeight) || 24;
      const padTop = Number.parseFloat(cs.paddingTop) || 0;
      const padBottom = Number.parseFloat(cs.paddingBottom) || 0;
      lineMetricsRef.current = { lineHeight, padding: padTop + padBottom };
    }
    const { lineHeight, padding } = lineMetricsRef.current;
    const contentHeight = el.scrollHeight - padding;
    if (contentHeight > lineHeight * 1.5) setIsMultiline(true);
  }, [composerText, isMultiline]);
  const hasAttachments = useAuiState(
    ({ composer }) => composer.attachments.length > 0,
  );
  const hasPendingAttachments = useAuiState(({ composer }) =>
    composer.attachments.some(
      (attachment) => attachment.status.type === "running",
    ),
  );
  const hasPendingAudio = useChatRuntimeStore((s) =>
    Boolean(s.pendingAudioName),
  );
  const referenceThreadId = threadId ?? activeThreadId ?? null;
  const hasSendableContent =
    composerText.trim().length > 0 || hasAttachments || hasPendingAudio;
  // Two-row layout shows once the input wraps or a tool is on. Tools can
  // pre-select before a model loads, so an active toggle expands it either way.
  const composerExpanded =
    isMultiline ||
    hasAttachments ||
    hasPendingAudio ||
    toolsEnabled ||
    codeToolsEnabled ||
    imageToolsEnabled ||
    ragEnabled ||
    artifactsEnabled ||
    mcpEnabledForChat;
  // react-textarea-autosize re-measures only on value change or window resize,
  // not on the width swap from expanding, so it keeps the taller height and
  // leaves a stray blank row. Nudge a resize whenever input width changes.
  useEffect(() => {
    const el = inputRef.current;
    if (!el || typeof ResizeObserver === "undefined") {
      return;
    }
    let lastWidth = -1;
    const pending: Array<ReturnType<typeof setTimeout>> = [];
    const observer = new ResizeObserver((entries) => {
      const width = Math.round(entries[0]?.contentRect.width ?? 0);
      // Width changes only; reacting to autosize's height change would loop.
      if (width === lastWidth) {
        return;
      }
      lastWidth = width;
      // Re-measure after layout settles. An immediate dispatch races
      // autosize's own measurement (stale pre-expand width); 0ms + 64ms wins.
      while (pending.length) {
        clearTimeout(pending.pop());
      }
      for (const delay of [0, 64]) {
        pending.push(
          setTimeout(() => {
            window.dispatchEvent(new Event("resize"));
          }, delay),
        );
      }
    });
    observer.observe(el);
    return () => {
      while (pending.length) {
        clearTimeout(pending.pop());
      }
      observer.disconnect();
    };
  }, []);
  // Docked composer opens upward; the welcome composer opens downward by
  // default and only flips up via collision detection when it won't fit.
  const effectiveMenuSide = menuSide ?? "bottom";

  // While this thread's docs index, hold the send and fire it once they finish so
  // retrieval covers all of them.
  const [indexingActive, setIndexingActive] = useState(false);
  const [pendingSend, setPendingSend] = useState(false);
  const pendingSendRef = useRef(false);
  const waitToastRef = useRef<string | number | null>(null);

  const dismissWaitToast = useCallback(() => {
    if (waitToastRef.current !== null) {
      toast.dismiss(waitToastRef.current);
      waitToastRef.current = null;
    }
  }, []);

  const cancelQueuedSend = useCallback(() => {
    pendingSendRef.current = false;
    setPendingSend(false);
    dismissWaitToast();
  }, [dismissWaitToast]);

  const enqueueSend = useCallback(() => {
    if (pendingSendRef.current) return;
    pendingSendRef.current = true;
    setPendingSend(true);
    waitToastRef.current = toast("Waiting for documents to finish indexing", {
      description:
        "Your message will send automatically once indexing finishes.",
      duration: Infinity,
      cancel: { label: "Cancel", onClick: cancelQueuedSend },
    });
  }, [cancelQueuedSend]);

  const shouldBlockSend = useCallback(
    () =>
      !hasSendableContent || isComposingRef.current || hasPendingAttachments,
    [hasPendingAttachments, hasSendableContent, isComposingRef],
  );

  // Gate for both form submit and the Send button. Returns true when it handled
  // the event (blocked or queued) so callers stop.
  const interceptSend = useCallback(
    (event: { preventDefault: () => void }) => {
      if (disabled || shouldBlockSend()) {
        event.preventDefault();
        return true;
      }
      if (indexingActive && !overlay) {
        event.preventDefault();
        enqueueSend();
        return true;
      }
      return false;
    },
    [disabled, shouldBlockSend, indexingActive, overlay, enqueueSend],
  );

  // Fire the parked send once indexing clears, unless the user emptied the
  // composer while waiting (then drop it quietly).
  useEffect(() => {
    if (!pendingSend || indexingActive) return;
    const { text, attachments } = aui.composer().getState();
    pendingSendRef.current = false;
    setPendingSend(false);
    dismissWaitToast();
    if (text.trim().length > 0 || attachments.length > 0) {
      aui.composer().send();
    }
  }, [pendingSend, indexingActive, aui, dismissWaitToast]);

  // Drop any queued send + toast on unmount (e.g. thread switch).
  useEffect(
    () => () => {
      pendingSendRef.current = false;
      if (waitToastRef.current !== null) toast.dismiss(waitToastRef.current);
    },
    [],
  );

  const handleSubmit = useCallback(
    (event: Parameters<NonNullable<ComponentProps<"form">["onSubmit"]>>[0]) => {
      if (interceptSend(event)) return;

      if (overlay) {
        const trimmed = composerText.trim();
        if (!trimmed) {
          event.preventDefault();
          return;
        }
        if (!overlay.openaiImageGenerationCallId) {
          event.preventDefault();
          toast.error("This generated image cannot be edited", {
            description:
              "The original image reference is missing. Generate the image again, then retry the edit.",
          });
          closeOverlay();
          return;
        }
        if ((overlay.threadId ?? null) !== referenceThreadId) {
          event.preventDefault();
          toast.error("This generated image belongs to another chat", {
            description: "Open the original chat and retry the edit.",
          });
          closeOverlay();
          return;
        }
        setImageToolsEnabled(true);
        setPendingImageEditReference({
          threadId: overlay.threadId ?? referenceThreadId,
          openaiImageGenerationCallId: overlay.openaiImageGenerationCallId,
          ...(overlay.openaiResponseId
            ? { openaiResponseId: overlay.openaiResponseId }
            : {}),
          openaiReasoningItem: overlay.openaiReasoningItem,
        });
        flushResourcesSync(() => {
          aui
            .composer()
            .setText(
              `Use the selected generated image as the reference and apply this edit: ${trimmed}. Preserve everything else exactly.`,
            );
        });
        closeOverlay();
      }
    },
    [
      aui,
      closeOverlay,
      composerText,
      interceptSend,
      overlay,
      referenceThreadId,
      setImageToolsEnabled,
      setPendingImageEditReference,
    ],
  );

  // Update the getter every render so the queue always calls the current
  // Composer's aui (post-remount).
  _qGetAui = () => aui;

  const stopQueue = useCallback(() => {
    _qIsRunning = false;
    _qStopSubscription();
    _useQueueUI.setState({ isRunning: false, current: 0, total: 0 });
    _qItems = [];
    _qIndex = 0;
    try { _qGetAui().thread().cancelRun(); } catch {}
  }, []);

  const startQueue = useCallback((items: string[]) => {
    const filtered = items.filter((p) => p.trim());
    if (!filtered.length) return;
    _qItems = filtered;
    _qIndex = 0;
    _qIsRunning = true;
    _useQueueUI.setState({ isRunning: true, current: 1, total: filtered.length });
    toast(`Prompt 1 / ${filtered.length}`, {
      description: filtered[0].length > 80 ? filtered[0].slice(0, 80) + "…" : filtered[0],
    });
    // Subscribe BEFORE appending so we don't miss a very fast completion.
    _qStartSubscription();
    setTimeout(() => {
      _qGetAui().thread().append({
        role: "user",
        content: [{ type: "text", text: filtered[0] }],
        createdAt: new Date(),
      } as never);
    }, 50);
  }, []);

  const queueContextValue: _QueueCallbacks = { startQueue, stopQueue };

  const composerContent = (
    <>
      <ComposerAttachments />
      <PendingAudioChip />
      <ThreadDocumentsBar
        threadId={referenceThreadId}
        onIndexingChange={setIndexingActive}
      />
      <ToolStatusDisplay />
      <div
        className="unsloth-composer-line"
        data-expanded={composerExpanded ? "true" : "false"}
      >
        <div
          className="unsloth-composer-left"
          data-pill-compact={pillsCompact ? "true" : undefined}
        >
          <ComposerToolsMenu side={effectiveMenuSide} />
          {composerExpanded ? (
            <>
              <WebSearchToggle />
              <CodeToolsToggle />
              <ImagesToggle />
              <KnowledgeBaseComposerButton side={effectiveMenuSide} />
              {artifactsEnabled ? <ArtifactsToggle /> : null}
              {mcpEnabledForChat ? (
                <McpComposerButton side={effectiveMenuSide} />
              ) : null}
            </>
          ) : null}
        </div>
        <ComposerPrimitive.Input
          placeholder={
            overlay ? "Type your edits for your image" : "Ask anything"
          }
          ref={inputRef}
          className="aui-composer-input unsloth-composer-input"
          minRows={1}
          maxRows={12}
          autoFocus={!disabled}
          disabled={disabled}
          aria-label={overlay ? "Image edit instructions" : "Message input"}
          // dir="auto": browser picks LTR/RTL from the first strong char;
          // no effect on Latin / CJK / Devanagari.
          dir="auto"
          {...inputProps}
        />
        <ComposerRightControls
          disabled={
            disabled ||
            !hasSendableContent ||
            isComposing ||
            hasPendingAttachments
          }
          onSendClick={interceptSend}
          pendingSend={pendingSend}
          menuSide={effectiveMenuSide}
        />
      </div>
    </>
  );

  return (
    <PromptQueueContext.Provider value={queueContextValue}>
    <ComposerPrimitive.Root
      className="aui-composer-root relative flex w-full flex-col"
      aria-disabled={disabled}
      onSubmit={handleSubmit}
    >
      {isTauri ? (
        // Phase 1 native model owns Tauri local-path drops. Restore browser
        // attachment drops in Tauri once Phase 1d adds token bridging.
        <div className="aui-composer-attachment-dropzone unsloth-composer-surface">
          {composerContent}
        </div>
      ) : (
        <ComposerPrimitive.AttachmentDropzone className="group/dropzone aui-composer-attachment-dropzone unsloth-composer-surface relative">
          {composerContent}
          {/* Gemini-style drop affordance, shown while a file is dragged over
              the composer. Absolute + pointer-events-none so the outline adds
              no layout shift and the drop still lands. */}
          <div
            className={cn(
              "aui-composer-drop-overlay pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center gap-1 overflow-hidden rounded-[32px] bg-background/90 opacity-0 backdrop-blur-sm transition-opacity duration-150 group-data-[dragging=true]/dropzone:opacity-100 dark:bg-card/90",
              pageDragging && "opacity-100",
            )}
          >
            <HugeiconsIcon
              icon={AttachmentIcon}
              strokeWidth={2}
              className="size-6 text-primary"
            />
            <span className="text-sm font-medium text-primary">
              Drop files here
            </span>
          </div>
        </ComposerPrimitive.AttachmentDropzone>
      )}
    </ComposerPrimitive.Root>
    </PromptQueueContext.Provider>
  );
};

function isNativeComposing(event: Event) {
  return "isComposing" in event && (event as InputEvent).isComposing === true;
}

// Fallback timeout for stuck IME composition. With Chrome on Windows against
// a WSL-hosted Studio (issue #5546), `compositionend` never fires after the
// candidate commits, so `composingRef` stays true and Send stays disabled.
// Every compositionupdate / non-composing input resets the timer; only a true
// gap-after-commit lets it fire. 2500ms is above a normal candidate-window
// pause but short enough to recover before the user notices Send is stuck.
const IME_STUCK_TIMEOUT_MS = 2500;

function useImeComposerInputHandlers() {
  const aui = useAui();
  const composingRef = useRef(false);
  const [isComposing, setIsComposing] = useState(false);
  const stuckTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearStuckTimer = useCallback(() => {
    if (stuckTimerRef.current) {
      clearTimeout(stuckTimerRef.current);
      stuckTimerRef.current = null;
    }
  }, []);

  const setCompositionState = useCallback(
    (next: boolean) => {
      composingRef.current = next;
      setIsComposing(next);
      clearStuckTimer();
      if (next) {
        stuckTimerRef.current = setTimeout(() => {
          stuckTimerRef.current = null;
          composingRef.current = false;
          setIsComposing(false);
        }, IME_STUCK_TIMEOUT_MS);
      }
    },
    [clearStuckTimer],
  );

  const refreshStuckTimer = useCallback(() => {
    if (!composingRef.current) {
      return;
    }
    clearStuckTimer();
    stuckTimerRef.current = setTimeout(() => {
      stuckTimerRef.current = null;
      composingRef.current = false;
      setIsComposing(false);
    }, IME_STUCK_TIMEOUT_MS);
  }, [clearStuckTimer]);

  useEffect(() => clearStuckTimer, [clearStuckTimer]);

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

  const onCompositionUpdate = useCallback(() => {
    refreshStuckTimer();
  }, [refreshStuckTimer]);

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

  // If the watchdog cleared the composing flags during a long candidate-window
  // pause, a later IME keypress (isComposing=true / keyCode 229) would reach
  // handleSubmit with composingRef=false and submit the preedit text. Re-arm
  // composingRef synchronously from the native event so the submit gate keeps
  // blocking until compositionend. Re-arm the watchdog too, or the WSL+Chrome
  // path (no compositionend, no follow-up input) would pin composingRef true
  // forever and block Send again.
  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.nativeEvent.isComposing || e.keyCode === 229) {
        composingRef.current = true;
        refreshStuckTimer();
      }
    },
    [refreshStuckTimer],
  );

  return {
    inputProps: {
      onCompositionStart,
      onCompositionUpdate,
      onCompositionEnd,
      onChange,
      onKeyDown,
    },
    isComposing,
    isComposingRef: composingRef,
  };
}

// Audio upload row, only for audio-input models.
const ComposerAudioMenuItem: FC = () => {
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

  // Build the input on document.body, not in the menu: selecting the item
  // closes the dropdown, unmounting a menu-rendered input before the OS picker
  // returns and dropping the file.
  const pickAudio = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = AUDIO_ACCEPT;
    input.hidden = true;
    document.body.appendChild(input);
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) handleAudioFile(file);
      document.body.removeChild(input);
    };
    input.oncancel = () => {
      if (!input.files || input.files.length === 0) {
        document.body.removeChild(input);
      }
    };
    input.click();
  }, [handleAudioFile]);

  if (!activeModel?.hasAudioInput) {
    return null;
  }

  return (
    <DropdownMenuItem onSelect={() => pickAudio()}>
      <HeadphonesIcon />
      Upload audio
    </DropdownMenuItem>
  );
};

// Phosphor microphone. Inlined to avoid a new icon dependency.
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

// HugeIcons arrow-down-01 (stroke-standard): straight-line chevron.
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

// svgrepo.com lightbulb (filled, with base).
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

// Same bulb in every state; greyed by the pill's muted color when off.
const ThinkIcon: FC = () => <BulbIcon className="size-[15.5px]" />;

const ReasoningToggle: FC<{ side?: "top" | "bottom" }> = ({
  side = "bottom",
}) => {
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
  const supportsReasoningOff = useChatRuntimeStore(
    (s) => s.supportsReasoningOff,
  );
  const reasoningEffortLevels = useChatRuntimeStore(
    (s) => s.reasoningEffortLevels,
  );
  const setReasoningEffort = useChatRuntimeStore((s) => s.setReasoningEffort);
  const lastOpenRouterChosenModel = useChatRuntimeStore(
    (s) => s.lastOpenRouterChosenModel,
  );
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const externalProvidersAll = useExternalProvidersStore((s) => s.providers);
  const externalProviders = connectionsEnabled ? externalProvidersAll : [];
  const externalSelection = parseExternalModelId(checkpoint);
  const selectedExternalProvider =
    externalSelection != null
      ? externalProviders.find((p) => p.id === externalSelection.providerId)
      : undefined;
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const supportsPreserveThinking = useChatRuntimeStore(
    (s) => s.supportsPreserveThinking,
  );
  const preserveThinking = useChatRuntimeStore((s) => s.preserveThinking);
  const setPreserveThinking = useChatRuntimeStore((s) => s.setPreserveThinking);
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
            // Lets the resolver detect custom Gemini OAI-compat gateways.
            baseUrl: selectedExternalProvider?.baseUrl ?? null,
          },
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
    if (level !== "xhigh")
      return level.charAt(0).toUpperCase() + level.slice(1);
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

  // Only rendered for models that can reason.
  if (!effectiveSupportsReasoning) {
    return null;
  }

  const isEffort = effectiveReasoningStyle === "reasoning_effort";
  // Dropdown when there are effort levels or preserve-thinking; else a toggle.
  const useDropdown = isEffort || supportsPreserveThinking;
  const activeLook = isEffort
    ? reasoningLockedOn || (effectiveReasoningVisualEnabled && !disabled)
    : reasoningLockedOn || (effectiveReasoningEnabled && !disabled);

  if (useDropdown) {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            disabled={disabled}
            className="unsloth-thinking-pill"
            data-active={activeLook ? "true" : "false"}
            aria-label={thinkEffortAriaLabel({
              modelLoaded,
              reasoningDisabled: disabled,
              reasoningEffort,
            })}
          >
            <ThinkIcon />
            {activeLook ? (
              <span>{isEffort ? `Thinking · ${effortLabel}` : "Thinking"}</span>
            ) : null}
            <ArrowDownStandardIcon className="size-[15px]" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side={side}
          align="end"
          avoidCollisions={true}
          className="unsloth-plus-menu unsloth-thinking-menu min-w-0 w-[176px]"
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
                  None
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
                    {formatEffortLabel(level)}
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
              disabled={disabled}
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
        // Mutually exclusive with Search on Kimi (see dropdown branch).
        if (isKimiExternal && next && toolsEnabled) {
          setToolsEnabled(false, { persist: false });
        }
      }}
      className="unsloth-thinking-pill"
      data-active={activeLook ? "true" : "false"}
      aria-label={thinkToggleAriaLabel({
        reasoningLockedOn,
        modelLoaded,
        reasoningDisabled: disabled,
        effectiveReasoningEnabled,
      })}
    >
      <PillGlyph>
        <ThinkIcon />
      </PillGlyph>
      {activeLook ? <span>Thinking</span> : null}
    </button>
  );
};

// Tool icon plus an X overlay the CSS reveals on hover when the pill is active.
const PillGlyph: FC<{ children: ReactNode }> = ({ children }) => (
  <span className="composer-pill-glyph">
    {children}
    <XIcon className="composer-pill-x" />
  </span>
);

const WebSearchToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  // External providers (OpenAI today) expose a server-side web_search tool
  // even without the local tool runtime; gate the pill on either source so it
  // lights up on external models too. Mirror of shared-composer's searchDisabled.
  const supportsBuiltinWebSearch = useChatRuntimeStore(
    (s) => s.supportsBuiltinWebSearch,
  );
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const externalProvidersAll = useExternalProvidersStore((s) => s.providers);
  const externalProviders = connectionsEnabled ? externalProvidersAll : [];
  const externalSelection = parseExternalModelId(checkpoint);
  const selectedExternalProvider =
    externalSelection != null
      ? externalProviders.find((p) => p.id === externalSelection.providerId)
      : undefined;
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";
  // Disable only when a loaded model lacks the capability; with no model the
  // tool can still be pre-selected, matching the + menu.
  const disabled = modelLoaded && !(supportsTools || supportsBuiltinWebSearch);

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => {
        const next = !toolsEnabled;
        setToolsEnabled(next);
        // Kimi's $web_search builtin requires thinking=disabled (see
        // https://platform.kimi.ai/docs/guide/use-web-search). Keep the two
        // pills mutually exclusive so visible state matches what's sent.
        if (isKimiExternal) {
          setReasoningEnabled(!next, { persist: false });
          applyQwenThinkingParams(!next);
        }
      }}
      className="composer-pill-btn"
      data-active={toolsEnabled && !disabled ? "true" : "false"}
      aria-label={toolsEnabled ? "Disable web search" : "Enable web search"}
    >
      <PillGlyph>
        <GlobeIcon className="size-[15px]" />
      </PillGlyph>
      <span>Search</span>
    </button>
  );
};

const CodeToolsToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  // External providers have no local tool runtime, but Anthropic's Claude 4.x
  // dispatches code_execution_20250825 server-side; the chat-page resolver
  // stashes that capability in the runtime store (next to
  // supportsBuiltinWebSearch). Mirror of shared-composer's codeDisabled.
  const supportsBuiltinCodeExecution = useChatRuntimeStore(
    (s) => s.supportsBuiltinCodeExecution,
  );
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const setCodeToolsEnabled = useChatRuntimeStore((s) => s.setCodeToolsEnabled);
  // Disable only when a loaded model lacks the capability; with no model the
  // tool can still be pre-selected, matching the + menu.
  const disabled = modelLoaded && !(supportsTools || supportsBuiltinCodeExecution);

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
      <PillGlyph>
        <HugeiconsIcon
          icon={CodeIcon}
          className="size-[18.5px]"
          strokeWidth={2}
        />
      </PillGlyph>
      <span>Code</span>
    </button>
  );
};

const ImagesToggle: FC = () => {
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  // OpenAI cloud Responses-API models advertise image_generation as a
  // server-side tool; no local runtime fallback. Mirror of shared-composer's
  // imageDisabled / showImagePill so this composer matches the empty state.
  const supportsBuiltinImageGeneration = useChatRuntimeStore(
    (s) => s.supportsBuiltinImageGeneration,
  );
  const imageToolsEnabled = useChatRuntimeStore((s) => s.imageToolsEnabled);
  const setImageToolsEnabled = useChatRuntimeStore(
    (s) => s.setImageToolsEnabled,
  );
  if (!supportsBuiltinImageGeneration) {
    return null;
  }
  const disabled = !modelLoaded;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => setImageToolsEnabled(!imageToolsEnabled)}
      className="composer-pill-btn"
      data-active={imageToolsEnabled && !disabled ? "true" : "false"}
      aria-label={
        imageToolsEnabled
          ? "Disable image generation"
          : "Enable image generation"
      }
    >
      <PillGlyph>
        <HugeiconsIcon icon={Image03Icon} className="size-3.5" strokeWidth={2} />
      </PillGlyph>
      <span>Images</span>
    </button>
  );
};

const ArtifactsToggle: FC = () => {
  const artifactsEnabled = useChatRuntimeStore((s) => s.artifactsEnabled);
  const setArtifactsEnabled = useChatRuntimeStore((s) => s.setArtifactsEnabled);
  // Canvas is opt-in; the pill only shows once it is toggled on from the menu.
  if (!artifactsEnabled) return null;

  return (
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
          className="size-[15.5px]"
          strokeWidth={2}
        />
      </PillGlyph>
      <span>Canvas</span>
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

    // Debounce visibility by 300ms when the badge isn't already on screen.
    // Once visible from a prior tool, later tools show immediately so it
    // doesn't flicker; tool calls under 300ms never show the badge.
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
// Plus menu: attachment and workflow actions. Opens downward in the welcome
// composer; the docked composer passes side="top" to open upward.
const ComposerToolsMenu: FC<{ side?: "top" | "bottom" }> = ({
  side = "bottom",
}) => {
  const navigate = useNavigate();
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const setCodeToolsEnabled = useChatRuntimeStore((s) => s.setCodeToolsEnabled);
  const artifactsEnabled = useChatRuntimeStore((s) => s.artifactsEnabled);
  const setArtifactsEnabled = useChatRuntimeStore((s) => s.setArtifactsEnabled);
  const mcpEnabledForChat = useChatRuntimeStore((s) => s.mcpEnabledForChat);
  const setMcpEnabledForChat = useChatRuntimeStore(
    (s) => s.setMcpEnabledForChat,
  );
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const setRagEnabled = useChatRuntimeStore((s) => s.setRagEnabled);
  // Shared gate so the menu row agrees with the RAG pill.
  const ragDisabled = useRagToolDisabled();
  // Capability gating mirrors the visible pills so menu and pills agree on
  // what a loaded model supports (a tool the backend drops must not look on).
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const supportsBuiltinWebSearch = useChatRuntimeStore(
    (s) => s.supportsBuiltinWebSearch,
  );
  const supportsBuiltinCodeExecution = useChatRuntimeStore(
    (s) => s.supportsBuiltinCodeExecution,
  );
  const supportsBuiltinImageGeneration = useChatRuntimeStore(
    (s) => s.supportsBuiltinImageGeneration,
  );
  const imageToolsEnabled = useChatRuntimeStore((s) => s.imageToolsEnabled);
  const setImageToolsEnabled = useChatRuntimeStore(
    (s) => s.setImageToolsEnabled,
  );
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const externalProvidersAll = useExternalProvidersStore((s) => s.providers);
  const externalProviders = connectionsEnabled ? externalProvidersAll : [];
  const externalSelection = parseExternalModelId(checkpoint);
  const selectedExternalProvider =
    externalSelection != null
      ? externalProviders.find((p) => p.id === externalSelection.providerId)
      : undefined;
  const isKimiExternal = selectedExternalProvider?.providerType === "kimi";
  // Disable only when a loaded model lacks the capability; with no model the
  // tool can still be pre-selected, matching the pill logic above.
  const searchDisabled =
    modelLoaded && !(supportsTools || supportsBuiltinWebSearch);
  const codeDisabled =
    modelLoaded && !(supportsTools || supportsBuiltinCodeExecution);
  const imageDisabled = !modelLoaded;
  // Like Search/Code: disabled only when a loaded model lacks tool support.
  const mcpDisabled = modelLoaded && !supportsTools;
  // Three most recently updated projects for the quick-access submenu.
  const { projects } = useChatProjects();
  const recentProjects = [...projects]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 3);
  const openProject = (projectId: string) => {
    useChatRuntimeStore.getState().setActiveProjectId(projectId);
    navigate({ to: "/chat", search: { project: projectId } });
  };

  const startCompare = useCallback(() => {
    const store = useChatRuntimeStore.getState();
    store.setActiveThreadId(null);
    store.setContextUsage(null);
    // crypto.randomUUID is undefined in non-secure contexts (HTTP over a LAN IP).
    const compareId =
      typeof globalThis.crypto?.randomUUID === "function"
        ? globalThis.crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    navigate({ to: "/chat", search: { compare: compareId } });
  }, [navigate]);

  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [promptStorageOpen, setPromptStorageOpen] = useState(false);
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const aui = useAui();
  // Disable Export chat until the thread has content.
  const messageCount = useAuiState(({ thread }) => thread.messages.length);
  const { startQueue } = useContext(PromptQueueContext);

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

  return (
    <>
    <PromptStorageDialog
      open={promptStorageOpen}
      onOpenChange={setPromptStorageOpen}
      onUse={(text) => {
        aui.composer().setText(text);
      }}
      onRunList={(items) => {
        setPromptStorageOpen(false);
        startQueue(items);
      }}
    />
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
        side={side}
        align="start"
        sideOffset={0}
        avoidCollisions={true}
        className="unsloth-plus-menu w-[212px]"
        // Don't refocus the + on close; restored focus showed a stray ring.
        onCloseAutoFocus={(event) => event.preventDefault()}
      >
        <ComposerPrimitive.AddAttachment asChild={true}>
          <DropdownMenuItem>
            <HugeiconsIcon icon={AttachmentIcon} strokeWidth={2} />
            Add photos &amp; files
          </DropdownMenuItem>
        </ComposerPrimitive.AddAttachment>
        <ComposerAudioMenuItem />
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
        {supportsBuiltinImageGeneration && (
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
            ragEnabled && !ragDisabled ? "text-primary font-medium" : undefined
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
          disabled={mcpDisabled}
          className={
            mcpEnabledForChat && !mcpDisabled
              ? "text-primary font-medium"
              : undefined
          }
          onSelect={() => setMcpEnabledForChat(!mcpEnabledForChat)}
        >
          <HugeiconsIcon icon={McpServerIcon} strokeWidth={2} />
          MCP
          {mcpEnabledForChat && !mcpDisabled ? (
            <HugeiconsIcon
              icon={Tick02Icon}
              strokeWidth={2}
              className="ml-auto"
            />
          ) : null}
        </DropdownMenuItem>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>
            <MoreHorizontalIcon className="size-4" />
            More
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-[200px]">
            <DropdownMenuItem onSelect={() => startCompare()}>
              <Columns2Icon />
              Compare chat
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
                    onSelect={() => aui.composer().setText(p.text)}
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
            <DropdownMenuSub>
              <DropdownMenuSubTrigger
                disabled={!activeThreadId || messageCount === 0}
              >
                <HugeiconsIcon icon={Download01Icon} strokeWidth={2} />
                Export chat
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent
                collisionPadding={16}
                className="unsloth-plus-menu w-[176px]"
              >
                <DropdownMenuItem
                  onSelect={() => {
                    if (!activeThreadId) return;
                    exportConversationRawJsonl(activeThreadId).catch(() =>
                      toast.error("Export failed."),
                    );
                  }}
                >
                  Raw JSONL
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() => {
                    if (!activeThreadId) return;
                    exportConversationCsv(activeThreadId).catch(() =>
                      toast.error("Export failed."),
                    );
                  }}
                >
                  CSV
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() => {
                    if (!activeThreadId) return;
                    exportConversationShareGPT(activeThreadId).catch(() =>
                      toast.error("Export failed."),
                    );
                  }}
                >
                  ShareGPT JSONL
                </DropdownMenuItem>
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
      <NewProjectDialog
        open={newProjectOpen}
        onOpenChange={setNewProjectOpen}
      />
    </>
  );
};

const ComposerRightControls: FC<{
  disabled?: boolean;
  onSendClick?: (event: { preventDefault: () => void }) => void;
  pendingSend?: boolean;
  menuSide?: "top" | "bottom";
}> = ({ disabled, onSendClick, pendingSend, menuSide }) => {
  const isQueueRunning = _useQueueUI((s) => s.isRunning);
  const queueCurrent = _useQueueUI((s) => s.current);
  const queueTotal = _useQueueUI((s) => s.total);
  const { stopQueue } = useContext(PromptQueueContext);
  return (
    <div className="aui-composer-action-wrapper flex shrink-0 items-center gap-1.5">
      <ReasoningToggle side={menuSide} />
      <ComposerPrimitive.If dictation={false}>
        <ComposerPrimitive.Dictate asChild={true}>
          <TooltipIconButton
            tooltip="Dictate"
            aria-label="Dictate"
            variant="ghost"
            className="size-8 rounded-full text-foreground"
          >
            <MicIcon className="size-5" />
          </TooltipIconButton>
        </ComposerPrimitive.Dictate>
      </ComposerPrimitive.If>
      <ComposerPrimitive.If dictation={true}>
        <ComposerPrimitive.StopDictation asChild={true}>
          <TooltipIconButton
            tooltip="Stop dictation"
            aria-label="Stop dictation"
            variant="ghost"
            className="size-8 rounded-full text-destructive"
          >
            <SquareIcon className="size-3 animate-pulse fill-current" />
          </TooltipIconButton>
        </ComposerPrimitive.StopDictation>
      </ComposerPrimitive.If>
      {isQueueRunning ? (
        <button
          type="button"
          onClick={stopQueue}
          aria-label="Stop prompt queue"
          className="ml-1.5 flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/60 px-2.5 py-1 text-xs font-semibold text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <SquareIcon className="size-2.5 shrink-0 fill-current" />
          <span className="tabular-nums">
            Stop queue {queueCurrent}/{queueTotal}
          </span>
        </button>
      ) : (
        <>
          <AuiIf condition={({ thread }) => !thread.isRunning}>
            <ComposerPrimitive.Send asChild={true}>
              <TooltipIconButton
                tooltip={pendingSend ? "Waiting for documents…" : "Send message"}
                side="bottom"
                type="submit"
                variant="default"
                size="icon"
                // Stay clickable while docs index so a click can queue the send;
                // disabled only once a send is parked.
                disabled={disabled || pendingSend}
                onClick={(event) => onSendClick?.(event)}
                className="aui-composer-send ml-1.5 size-8 rounded-full"
                aria-label="Send message"
              >
                {pendingSend ? (
                  <Spinner className="size-[18px]" />
                ) : (
                  <ArrowUpIcon className="aui-composer-send-icon size-[21px] stroke-2" />
                )}
              </TooltipIconButton>
            </ComposerPrimitive.Send>
          </AuiIf>
          <AuiIf condition={({ thread }) => thread.isRunning}>
            <ComposerPrimitive.Cancel asChild={true}>
              <Button
                type="button"
                variant="default"
                size="icon"
                className="aui-composer-cancel ml-1.5 size-8 rounded-full"
                aria-label="Stop generating"
              >
                <SquareIcon className="aui-composer-cancel-icon size-3 fill-current" />
              </Button>
            </ComposerPrimitive.Cancel>
          </AuiIf>
        </>
      )}
    </div>
  );
};

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="aui-message-error-root mt-2 rounded-md bg-destructive/10 p-3 text-destructive text-sm dark:bg-destructive/5 dark:text-red-200">
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

// Placeholder when stop fires before any visible content (e.g. mid-think).
const CancelledIndicator: FC = () => {
  const show = useAuiState(
    ({ message }) =>
      message.content.length === 0 &&
      message.status?.type === "incomplete" &&
      message.status?.reason === "cancelled",
  );
  if (!show) {
    return null;
  }
  return (
    <span className="aui-cancelled-indicator text-sm italic text-muted-foreground">
      Cancelled.
    </span>
  );
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="aui-assistant-message-root relative mx-auto min-w-0 w-full max-w-(--thread-content-max-width) pt-0.5 pb-4 text-[15.5px] [font-weight:410] tracking-[0.01em] dark:tracking-[0.02em]"
      data-role="assistant"
    >
      <div className="aui-assistant-message-content wrap-break-word min-w-0 text-[#0d0d0d] dark:text-foreground leading-relaxed">
        <GeneratingIndicator />
        <CancelledIndicator />
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
                search_knowledge_base: KnowledgeBaseToolUI,
                python: PythonToolUI,
                terminal: TerminalToolUI,
                code_execution: CodeExecutionToolUI,
                image_generation: ImageGenerationToolUI,
                render_html: RenderHtmlToolUI,
              },
              Fallback: ToolFallback,
            },
          }}
        />
        <SourcesGroup />
        <RagSourcesGroup />
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
          className="aui-action-bar-more-content z-50 min-w-32 overflow-hidden rounded-md [--radius:1.1rem] bg-popover p-1 text-popover-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-none"
        >
          <ActionBarPrimitive.ExportMarkdown asChild={true}>
            <ActionBarMorePrimitive.Item className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
              <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon" />
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
          <HugeiconsIcon
            icon={Edit03Icon}
            strokeWidth={1.75}
            className="size-icon"
          />
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
          // See main composer above for the dir="auto" rationale.
          dir="auto"
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
