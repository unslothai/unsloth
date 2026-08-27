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
import { CompactionNotice } from "@/components/assistant-ui/compaction-notice";
import {
  compactionBoundary,
  type ContextTruncation,
} from "@/features/chat/utils/context-truncation";
import { downloadImagePart } from "@/components/assistant-ui/image";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { MessageHtmlArtifacts } from "@/components/assistant-ui/message-html-artifacts";
import {
  MessageResponseDetailsSheet,
  MessageResponseModelBadge,
} from "@/components/assistant-ui/message-response-details-sheet";
import { ProgressiveMessages } from "@/components/assistant-ui/progressive-messages";
import { MessageTiming } from "@/components/assistant-ui/message-timing";
import { attachThreadFastCopy } from "@/components/assistant-ui/thread-fast-copy";
import { threadHasResearchMessage } from "@/components/assistant-ui/thread-research-presence";
import { Reasoning, ReasoningGroup } from "@/components/assistant-ui/reasoning";
import { RagSourcesGroup } from "@/components/assistant-ui/rag-sources";
import { researchReplyOwners } from "@/components/assistant-ui/research-reply-owners";
import { Sources, SourcesGroup } from "@/components/assistant-ui/sources";
import {
  proplessSlot,
  threadMessageKind,
} from "@/components/assistant-ui/thread-message-slot";
import {
  thinkEffortAriaLabel,
  thinkToggleAriaLabel,
} from "@/components/assistant-ui/think-aria-label";
import { withToolConfirmation } from "@/components/assistant-ui/tool-confirmation-controls";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { ToolGroup } from "@/components/assistant-ui/tool-group";
import { CodeExecutionToolUI } from "@/components/assistant-ui/tool-ui-code-execution";
import { ImageGenerationToolUI } from "@/components/assistant-ui/tool-ui-image-generation";
import { KnowledgeBaseToolUI } from "@/components/assistant-ui/tool-ui-knowledge-base";
import { RenderHtmlToolUI } from "@/components/assistant-ui/tool-ui-render-html";
import { PythonToolUI } from "@/components/assistant-ui/tool-ui-python";
import { TerminalToolUI } from "@/components/assistant-ui/tool-ui-terminal";
import { WebSearchToolUI } from "@/components/assistant-ui/tool-ui-web-search";
import { ChatDictationBar } from "@/components/assistant-ui/chat-dictation-bar";
import {
  PROMPT_QUEUE_DRAG_TYPE,
  attachmentsPastedText,
  hasPendingPromptQueueStart,
  isPastedTextFile,
  isPromptQueueChord,
  isPromptQueueDragTypes,
  pastedTextQueueKey,
  promptQueueActiveItemChanged,
  reorderPromptQueueItems,
  pasteClipboardFiles,
  extractYoutubeVideoId,
  pasteLongTextAsFile,
  isPlainPasteChord,
  plainPasteStillCounts,
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
  YoutubeTranscriptPrompt,
  stripSearchImageTokens,
  useChatActive,
  useInComparePane,
} from "@/features/chat";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  IntentAwareScrollProvider,
  useIntentAwareAutoScroll,
  useIsThreadAtBottom,
  useScrollThreadToBottom,
} from "@/components/assistant-ui/use-intent-aware-autoscroll";
import { Button } from "@/components/ui/button";
import { MascotImg } from "@/components/mascot-img";
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
import { forkChatThread } from "@/features/chat/api/chat-api";
import {
  findLatestUserAudioBase64,
  resolveProjectId,
  sentAudioNames,
} from "@/features/chat/api/chat-adapter";
import {
  PromptStorageDialog,
  exportConversationShareGPT,
  exportConversationRawJsonl,
  exportConversationMessagesJsonl,
  exportConversationCsv,
  exportConversationMarkdown,
} from "@/features/chat/prompt-storage/prompt-storage-dialog";
import {
  listPromptEntries,
  type PromptEntry,
} from "@/features/chat/api/prompts-api";
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { useChatProjects } from "@/features/chat/hooks/use-chat-projects";
import { NewProjectDialog } from "@/features/chat/components/new-project-dialog";
import { ProjectGoalBar } from "@/features/chat/components/project-goal-bar";
import { ResearchMessage } from "@/features/chat/components/research-message";
import {
  DeepResearchComposerButton,
  DeepResearchWebsiteAccessDialog,
} from "@/features/chat/components/deep-research-composer-button";
import {
  type NativeIntent,
  useNativeAttachmentTargetKey,
  useNativeIntentStore,
} from "@/features/native-intents";
import { nativeAttachmentIntentToFile } from "@/features/native-intents/native-attachment-file";
import { cancelResearchRun } from "@/features/chat/api/research-api";
import {
  ingestResearchUpdate,
  useResearchRunStore,
} from "@/features/chat/stores/research-run-store";
import { researchReplyOwnsRun } from "@/features/chat/utils/research-run-binding";
import {
  parseExternalModelId,
  providerModelSupportsStudioTools,
} from "@/features/chat/external-providers";
import { toolStatusKind } from "@/features/chat/utils/tool-status";
import { replySourceMarkdown } from "@/features/chat/utils/reply-source-markdown";
import { toolResultModelText } from "@/features/chat/api/chat-adapter";
import {
  CONTINUATION_RUN_CONFIG_KEY,
  incompleteLabel,
  isContinuableContent,
  modeAllowsContinuation,
  readIncompleteInfo,
  readTextThoughtSignature,
  claimAutoContinue,
  forgetAutoContinue,
  recordAutoContinue,
  shouldAutoContinueMessage,
} from "@/features/chat/utils/continuation";
import { holdAutoContinueRun } from "@/features/chat/utils/auto-continue-run-keeper";
import { McpComposerButton } from "@/features/chat/mcp-composer-button";
import {
  COMPOSER_INPUT_SELECTOR,
  isSurfaceInForeground,
  useShortcut,
} from "@/features/settings";
import { create } from "zustand";
import { getExternalReasoningCapabilities } from "@/features/chat/provider-capabilities";
import { useRagToolDisabled } from "@/features/chat/hooks/use-rag-tool-disabled";
import { BypassPermissionsMenuItem } from "@/features/chat/bypass-permissions-menu-item";
import { PermissionModeComposerPill } from "@/features/chat/permission-mode-select";
import {
  settleThreadScopedSettingsForCopy,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { saveMarkdownAsProjectSource } from "@/features/rag";
import {
  PLUS_MENU_ORDER,
  CONVERSATION_MARKDOWN_LABEL,
  PROMPT_QUEUE_RUN_FAILED_EVENT,
  PROMPT_QUEUE_STOP_EVENT,
  addQueuedChatRunSettingsThreadIds,
  adoptPreStreamRunReservation,
  chatHistoryClearBoundary,
  deleteStoredChatThreads,
  discardQueuedChatRunSettings,
  discardQueuedChatRunSettingsForThread,
  hasPreStreamRunReservation,
  localPromptQueueModelBoundary,
  notifyPromptQueueRunFailed,
  planLocalPromptQueueStop,
  registerQueuedChatRunSettings,
  releasePreStreamRunReservation,
  reservePreStreamRun,
  claimThreadCreation,
  useChatProjectScope,
  shouldAbortPendingQueueForModelBoundary,
  shouldAbortPendingQueueForSettingsChange,
  snapshotQueuedChatRunSettings,
  composerDraftKey,
  composerPasteDraftKey,
  createPastedTextFile,
  pastedTextOf,
  readPasteDraft,
  writePasteDraft,
  markThreadIncognito,
  markChatThreadDeleted,
  type PromptQueueRunFailedEventDetail,
  type PromptQueueStopEventDetail,
  dictationFailed,
  dictationProducedTranscript,
  readComposerDraft,
  type PromptQueueUIEntry,
  type PromptQueueUIItem,
  type PromptQueueUIItemStatus,
  type PromptQueueUIState,
  usePromptQueueUI,
  forkCountFor,
  subscribeForkCounts,
  type PlusMenuItemId,
  usePlusMenuPrefsStore,
  writeComposerDraft,
} from "@/features/chat";
import {
  applySentTextGuard,
  armSentTextGuard,
  isGuardRetiringKey,
  markSentTextGuardUserInput,
  sentTextGuardBlocksDraft,
  type SentTextGuard,
} from "@/features/chat/utils/composer-send-guard";
import { deleteThreadMessage } from "@/features/chat/utils/delete-thread-message";
import {
  getStoredChatThread,
  updateStoredChatThread,
} from "@/features/chat/utils/chat-history-storage";
import {
  dictationSendBlocked,
  shouldSubmitDictation,
} from "@/features/chat/utils/dictation-send";
import {
  isRagClientError,
  listProjectDocuments,
  listThreadDocuments,
  projectWorkCount,
} from "@/features/rag/api/rag-api";
import { useRagAvailabilityStore } from "@/features/rag/api/rag-availability";
import { ThreadDocumentsBar } from "@/features/rag/components/thread-documents-bar";
import { KnowledgeBaseComposerButton } from "@/features/rag/components/knowledge-base-composer-button";
import { DocumentPreviewMount } from "@/features/rag/components/document-preview-mount";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { usePublishedFrame } from "@/features/settings/hooks/use-published-frame";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { applyQwenThinkingParams } from "@/features/chat/utils/qwen-params";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { MenuDismissGuard } from "@/lib/menu-dismiss-guard";
import { MicIcon } from "@/lib/mic-icon";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { Tick02Icon } from "@/lib/tick-icon";
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
  BookOpen01Icon,
  CodeIcon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  FileDatabaseIcon,
  Folder01Icon,
  FolderAddIcon,
  HelpCircleIcon,
  Image03Icon,
  McpServerIcon,
  PencilRulerIcon,
  Telescope02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  Columns2Icon,
  CornerDownRightIcon,
  GitBranchIcon,
  FastForwardIcon,
  GlobeIcon,
  HeadphonesIcon,
  Loader2Icon,
  MoreHorizontalIcon,
  PlusIcon,
  RefreshCwIcon,
  SquareIcon,
  TerminalIcon,
  Volume2Icon,
  VolumeXIcon,
  XIcon,
} from "lucide-react";
import {
  type ChangeEvent,
  type CompositionEvent,
  type ClipboardEvent,
  type FC,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  type FocusEvent as ReactFocusEvent,
  type ReactNode,
  type RefObject,
  Fragment,
  createContext,
  memo,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { extractTaggedText, updateThreadMessage } from "@/features/chat/utils/update-thread-message";
import { useComposerPillFit } from "@/hooks/use-composer-pill-fit";
import { useIsMobile } from "@/hooks/use-mobile";

// True while a file is dragged anywhere over the chat page, so the composer
// can show its "Drop files here" affordance.
const PageDragContext = createContext(false);

// Prompt queues live at module level so they survive Composer remounts,
// including the first queued message that creates a new thread. Each chat gets
// its own queue run; completion detection subscribes to runningByThreadId
// instead of aui.thread() so queues can keep advancing in the background.
type PromptQueueTarget = {
  getDocumentThreadId: () => string | null;
  /** The project this queue was started in, for a chat with no row to read. */
  getQueueProjectId: () => string | null;
  /** A knowledge base replaces every other scope, project sources included. */
  usesKnowledgeBase: boolean;
  getRunningThreadIds: () => string[];
  isRunning: () => boolean;
  append: (prompt: string) => void | Promise<void>;
  complete: () => void;
  cancel: () => void;
  isIndexing: () => boolean;
  usesThreadDocuments: boolean;
  usesLocalModel: boolean;
  usesDeepResearch: boolean;
  /** Whether a research run now holds this queue's thread. */
  researchStarted: () => boolean;
  temporary: boolean;
  consumeDeepResearch: () => void;
};

type PromptQueueItem = {
  id: string;
  prompt: string;
  target: PromptQueueTarget;
  dispatched: boolean;
  dispatchRetries: number;
};

type PromptQueueRun = {
  id: string;
  items: PromptQueueItem[];
  index: number;
  generation: number;
  prevStoreRunning: boolean;
  waitingForTargetIdle: boolean;
  retryTimer: ReturnType<typeof setTimeout> | null;
  deepResearchConsumed: boolean;
};

const PROMPT_QUEUE_INDEXING_RETRY_MS = 500;
const PROMPT_QUEUE_DISPATCH_RETRY_MS = 500;
const PROMPT_QUEUE_TARGET_STATE_POLL_MS = 50;
const PROMPT_QUEUE_MAX_DISPATCH_RETRIES = 5;

const promptQueueRuns = new Map<string, PromptQueueRun>();
const promptQueueActiveRunIds = new Set<string>();
const promptQueueDispatchingRunIds = new Set<string>();
const promptQueueRunOrder: string[] = [];
let promptQueueStoreUnsub: (() => void) | null = null;
let promptQueuePumpTimer: ReturnType<typeof setTimeout> | null = null;
let promptQueueRoundRobinCursor = 0;

function compactIds(ids: Array<string | null | undefined>) {
  return Array.from(new Set(ids.filter((id): id is string => Boolean(id))));
}

function createPromptQueueItemId() {
  return `prompt-queue-${crypto.randomUUID()}`;
}

function createPromptQueueRunId() {
  return `prompt-queue-run-${crypto.randomUUID()}`;
}

function stopPromptQueueSubscription() {
  if (promptQueueStoreUnsub) {
    promptQueueStoreUnsub();
    promptQueueStoreUnsub = null;
  }
}

function clearPromptQueuePumpTimer() {
  if (!promptQueuePumpTimer) {
    return;
  }
  clearTimeout(promptQueuePumpTimer);
  promptQueuePumpTimer = null;
}

function clearPromptQueueRetryTimer(run: PromptQueueRun) {
  if (!run.retryTimer) {
    return;
  }
  clearTimeout(run.retryTimer);
  run.retryTimer = null;
}

function deletePromptQueueRun(run: PromptQueueRun) {
  run.generation += 1;
  clearPromptQueueRetryTimer(run);
  promptQueueActiveRunIds.delete(run.id);
  promptQueueDispatchingRunIds.delete(run.id);
  promptQueueRuns.delete(run.id);
  const orderIndex = promptQueueRunOrder.indexOf(run.id);
  if (orderIndex >= 0) {
    promptQueueRunOrder.splice(orderIndex, 1);
    if (promptQueueRoundRobinCursor > orderIndex) {
      promptQueueRoundRobinCursor -= 1;
    }
    if (promptQueueRunOrder.length > 0) {
      promptQueueRoundRobinCursor %= promptQueueRunOrder.length;
    } else {
      promptQueueRoundRobinCursor = 0;
    }
  }
  if (promptQueueRuns.size === 0) {
    clearPromptQueuePumpTimer();
    stopPromptQueueSubscription();
  }
  syncPromptQueueUI();
}

function resetPromptQueues() {
  for (const run of promptQueueRuns.values()) {
    run.generation += 1;
    clearPromptQueueRetryTimer(run);
  }
  promptQueueRuns.clear();
  promptQueueActiveRunIds.clear();
  promptQueueDispatchingRunIds.clear();
  promptQueueRunOrder.length = 0;
  promptQueueRoundRobinCursor = 0;
  clearPromptQueuePumpTimer();
  stopPromptQueueSubscription();
  syncPromptQueueUI();
}

function requestPromptQueuePumpIfReady(delay = 0) {
  if (hasReadyPromptQueueRun()) {
    requestPromptQueuePump(delay);
  }
}

function handleQueuedPromptAppendFailure(
  run: PromptQueueRun,
  item: PromptQueueItem,
  error: unknown,
) {
  if (!isActivePromptQueueItem(run, item, run.generation)) {
    return;
  }
  item.dispatched = false;
  promptQueueActiveRunIds.delete(run.id);
  syncPromptQueueUI();
  item.dispatchRetries += 1;
  if (item.dispatchRetries > PROMPT_QUEUE_MAX_DISPATCH_RETRIES) {
    console.error("Prompt queue dispatch failed permanently:", error);
    try {
      item.target.cancel();
    } catch (cleanupError) {
      console.error("Prompt queue cleanup failed:", cleanupError);
    }
    deletePromptQueueRun(run);
    requestPromptQueuePumpIfReady();
    return;
  }
  item.target.complete();
  scheduleQueuedPromptDispatch(run, item, PROMPT_QUEUE_DISPATCH_RETRY_MS);
}

function consumePromptQueueDeepResearch(
  run: PromptQueueRun,
  item: PromptQueueItem,
) {
  // The model decides whether an armed prompt becomes research, so the queue's one research
  // is spent only once a run actually started, not on the first prompt that was merely armed.
  if (
    run.deepResearchConsumed ||
    !item.target.usesDeepResearch ||
    !item.target.researchStarted()
  ) {
    return;
  }
  run.deepResearchConsumed = true;
  for (const item of run.items) {
    item.target.consumeDeepResearch();
  }
}

function appendQueuedPrompt(run: PromptQueueRun, item: PromptQueueItem) {
  item.dispatched = true;
  promptQueueActiveRunIds.add(run.id);
  syncPromptQueueUI();
  try {
    const result = item.target.append(item.prompt);
    if (result && typeof result.catch === "function") {
      void result
        .then(() => consumePromptQueueDeepResearch(run, item))
        .catch((error) => {
          handleQueuedPromptAppendFailure(run, item, error);
        });
    } else {
      consumePromptQueueDeepResearch(run, item);
    }
  } catch (error) {
    handleQueuedPromptAppendFailure(run, item, error);
  }
  schedulePromptQueueTargetStatePoll(run);
}

const indexingDocument = (doc: { status: string }) =>
  doc.status === "pending" || doc.status === "running";

async function targetHasIndexingDocuments(item: PromptQueueItem) {
  if (item.target.isIndexing()) {
    return true;
  }
  const threadId = item.target.getDocumentThreadId();
  try {
    if (threadId && item.target.usesThreadDocuments) {
      const documents = await listThreadDocuments(threadId);
      if (documents.some(indexingDocument)) {
        return true;
      }
    }
    // Unless a knowledge base is active: the adapter sends kb_id alone, so the
    // project's sources cannot reach this run and waiting on them only delays it.
    if (item.target.usesKnowledgeBase) {
      return false;
    }
    // Project sources are retrieved whatever the Docs pill says (chat-adapter's
    // rag_scope), and isIndexing() above only answers while the bar that watches
    // them is mounted, which a background queue has not. So ask directly, and
    // for a chat with no row yet use the project the queue was started in.
    // Rethrowing: a row this probe could not read is not a chat with no project,
    // and the catch below is what holds the prompt and asks again. The queue's
    // own project is the fallback wherever the row is still missing, so a poll
    // landing mid-navigation cannot probe the project the user moved to.
    const queueProjectId = item.target.getQueueProjectId();
    const projectId = threadId
      ? await resolveProjectId(threadId, undefined, {
          rethrowReadFailure: true,
          composerProjectId: queueProjectId,
        })
      : queueProjectId;
    if (!projectId) {
      return false;
    }
    if (projectWorkCount(projectId) > 0) {
      return true;
    }
    try {
      const projectDocuments = await listProjectDocuments(projectId);
      return projectDocuments.some(indexingDocument);
    } catch (error) {
      // A project the server will not list (deleted, or a server predating the
      // route) is not one to wait for: the retry below never ends.
      if (isRagClientError(error)) {
        return false;
      }
      throw error;
    }
  } catch {
    // A failed status probe cannot prove that this thread's documents are
    // ready. Keep the queued send pending and retry instead of dispatching
    // without the RAG documents it was explicitly waiting for.
    //
    // Unless RAG cannot run on this host at all: the probe can never succeed
    // there, and dispatchQueuedPrompt reschedules on every "still indexing",
    // so waiting it out means the queued prompt is never sent. There are no
    // documents to wait for, so send it.
    return !useRagAvailabilityStore.getState().isUnavailable();
  }
}

function getActivePromptQueueItem(run: PromptQueueRun) {
  return run.items[Math.max(run.index, 0)];
}

function isActivePromptQueueItem(
  run: PromptQueueRun,
  item: PromptQueueItem,
  generation: number,
) {
  if (promptQueueRuns.get(run.id) !== run || generation !== run.generation) {
    return false;
  }
  return getActivePromptQueueItem(run) === item;
}

function scheduleQueuedPromptDispatch(
  run: PromptQueueRun,
  item: PromptQueueItem,
  delay: number,
  generation = run.generation,
) {
  clearPromptQueueRetryTimer(run);
  run.retryTimer = setTimeout(() => {
    run.retryTimer = null;
    if (isActivePromptQueueItem(run, item, generation)) {
      requestPromptQueuePump();
    }
  }, delay);
}

function isPromptQueueRunReadyToDispatch(run: PromptQueueRun) {
  const item = getActivePromptQueueItem(run);
  return Boolean(
    item &&
      run.index >= 0 &&
      !item.dispatched &&
      !run.waitingForTargetIdle &&
      !run.retryTimer &&
      !promptQueueActiveRunIds.has(run.id) &&
      !promptQueueDispatchingRunIds.has(run.id),
  );
}

function getNextReadyPromptQueueRun() {
  if (promptQueueRunOrder.length === 0) {
    return null;
  }
  const size = promptQueueRunOrder.length;
  for (let offset = 0; offset < size; offset += 1) {
    const orderIndex = (promptQueueRoundRobinCursor + offset) % size;
    const runId = promptQueueRunOrder[orderIndex];
    const run = promptQueueRuns.get(runId);
    if (!run || !isPromptQueueRunReadyToDispatch(run)) {
      continue;
    }
    promptQueueRoundRobinCursor = (orderIndex + 1) % size;
    return run;
  }
  return null;
}

function requestPromptQueuePump(delay = 0) {
  if (promptQueuePumpTimer) {
    return;
  }
  promptQueuePumpTimer = setTimeout(() => {
    promptQueuePumpTimer = null;
    pumpPromptQueues();
  }, delay);
}

function pumpPromptQueues() {
  ensurePromptQueueSubscription();
  // A queue is sequential within its own thread, but independent threads may
  // dispatch together. The inference backend owns its actual concurrency cap
  // and queues excess local generations.
  while (true) {
    const run = getNextReadyPromptQueueRun();
    if (!run) {
      return;
    }
    const item = getActivePromptQueueItem(run);
    if (!item) {
      deletePromptQueueRun(run);
      continue;
    }
    promptQueueDispatchingRunIds.add(run.id);
    dispatchQueuedPrompt(run, item, run.generation)
      .catch(() => undefined)
      .finally(() => {
        promptQueueDispatchingRunIds.delete(run.id);
        syncPromptQueueUI();
        if (!promptQueueActiveRunIds.has(run.id)) {
          requestPromptQueuePump();
        }
      });
  }
}

async function dispatchQueuedPrompt(
  run: PromptQueueRun,
  item: PromptQueueItem,
  generation = run.generation,
) {
  if (!isActivePromptQueueItem(run, item, generation)) {
    return;
  }
  if (
    isPromptQueueTargetRunning(
      item.target,
      useChatRuntimeStore.getState().runningByThreadId,
    )
  ) {
    run.waitingForTargetIdle = true;
    run.prevStoreRunning = true;
    promptQueueActiveRunIds.delete(run.id);
    syncPromptQueueUI();
    ensurePromptQueueSubscription();
    handlePromptQueueRunState(
      run,
      useChatRuntimeStore.getState().runningByThreadId,
    );
    schedulePromptQueueTargetStatePoll(run);
    return;
  }
  const hasIndexingDocuments = await targetHasIndexingDocuments(item);
  if (!isActivePromptQueueItem(run, item, generation)) {
    return;
  }
  if (hasIndexingDocuments) {
    promptQueueActiveRunIds.delete(run.id);
    scheduleQueuedPromptDispatch(run, item, PROMPT_QUEUE_INDEXING_RETRY_MS);
    return;
  }
  if (!isActivePromptQueueItem(run, item, generation)) {
    return;
  }
  appendQueuedPrompt(run, item);
}

function createQueuedPrompt(prompt: string, target: PromptQueueTarget) {
  return {
    id: createPromptQueueItemId(),
    prompt,
    target,
    dispatched: false,
    dispatchRetries: 0,
  };
}

function appendTextToThread(prompt: string) {
  return {
    role: "user",
    content: [{ type: "text", text: prompt }],
    createdAt: new Date(),
  } as never;
}

function getPromptQueueTargetIds(target: PromptQueueTarget) {
  return compactIds([
    ...target.getRunningThreadIds(),
    target.getDocumentThreadId(),
  ]);
}

function getPromptQueueRunTargetIds(run: PromptQueueRun) {
  return compactIds(
    run.items.flatMap((item) => getPromptQueueTargetIds(item.target)),
  );
}

function promptQueueRunUsesLocalModel(run: PromptQueueRun) {
  return run.items
    .slice(Math.max(run.index, 0))
    .some((item) => item.target.usesLocalModel);
}

function promptQueueRunIsTemporary(run: PromptQueueRun) {
  return run.items
    .slice(Math.max(run.index, 0))
    .some((item) => item.target.temporary);
}

function promptQueueRunMatchesThreadIds(
  run: PromptQueueRun,
  threadIds: string[],
) {
  return getPromptQueueRunTargetIds(run).some((id) => threadIds.includes(id));
}

function findPromptQueueRunByTarget(target: PromptQueueTarget) {
  const targetIds = getPromptQueueTargetIds(target);
  if (targetIds.length === 0) {
    return null;
  }
  for (const run of promptQueueRuns.values()) {
    if (promptQueueRunMatchesThreadIds(run, targetIds)) {
      return run;
    }
  }
  return null;
}

function findPromptQueueRunByItemId(itemId: string) {
  for (const run of promptQueueRuns.values()) {
    const itemIndex = run.items.findIndex((item) => item.id === itemId);
    if (itemIndex >= 0) {
      return { run, itemIndex, item: run.items[itemIndex] };
    }
  }
  return null;
}

function findPromptQueueRunByThreadIds(threadIds: string[]) {
  if (threadIds.length === 0) {
    return null;
  }
  for (const run of promptQueueRuns.values()) {
    if (promptQueueRunMatchesThreadIds(run, threadIds)) {
      return run;
    }
  }
  return null;
}

function findPromptQueueEntry(
  state: PromptQueueUIState,
  threadIds: string[],
) {
  for (const threadId of threadIds) {
    const entry = state.byThreadId[threadId];
    if (entry) {
      return entry;
    }
  }
  return null;
}

function canEditPromptQueueItem(item: PromptQueueItem) {
  return !item.dispatched;
}

function canRemovePromptQueueItem(item: PromptQueueItem) {
  return !item.dispatched;
}

function getPromptQueueRunProgress(run: PromptQueueRun) {
  const activeItemIndex = Math.max(run.index, 0);
  const total = run.items.length;
  const current = run.index >= 0 ? Math.min(activeItemIndex + 1, total) : 0;
  return { activeItemIndex, current, total };
}

function getPromptQueueItemStatus(
  run: PromptQueueRun,
  index: number,
  activeItemIndex: number,
): PromptQueueUIItemStatus {
  if (run.index >= 0 && index === activeItemIndex) {
    return run.waitingForTargetIdle ? "waiting" : "next";
  }
  return "queued";
}

function getPromptQueueUIItemsForRun(run: PromptQueueRun) {
  const { activeItemIndex, total } = getPromptQueueRunProgress(run);
  const items: PromptQueueUIItem[] = [];
  for (const [index, item] of run.items.entries()) {
    if (index < activeItemIndex || item.dispatched) {
      continue;
    }
    items.push({
      id: item.id,
      runId: run.id,
      prompt: item.prompt,
      position: index + 1,
      total,
      status: getPromptQueueItemStatus(run, index, activeItemIndex),
      threadIds: getPromptQueueTargetIds(item.target),
      canEdit: canEditPromptQueueItem(item),
      canRemove: canRemovePromptQueueItem(item),
    });
  }
  return items;
}

function syncPromptQueueUI() {
  if (promptQueueRuns.size === 0) {
    usePromptQueueUI.setState({
      byThreadId: {},
      current: 0,
      total: 0,
      items: [],
      isRunning: false,
    });
    return;
  }

  const items: PromptQueueUIItem[] = [];
  const byThreadId: Record<string, PromptQueueUIEntry> = {};
  let current = 0;
  let total = 0;

  for (const run of promptQueueRuns.values()) {
    const { current: runCurrent, total: runTotal } =
      getPromptQueueRunProgress(run);
    current += runCurrent;
    total += runTotal;
    items.push(...getPromptQueueUIItemsForRun(run));

    const ids = getPromptQueueRunTargetIds(run);
    if (ids.length === 0) {
      continue;
    }
    const entry = {
      runId: run.id,
      current: runCurrent,
      total: runTotal,
      local: promptQueueRunUsesLocalModel(run),
      temporary: promptQueueRunIsTemporary(run),
      dispatched: Boolean(getActivePromptQueueItem(run)?.dispatched),
    };
    for (const id of ids) {
      byThreadId[id] = entry;
    }
  }

  usePromptQueueUI.setState({
    byThreadId,
    current,
    total,
    items,
    isRunning: true,
  });
}

function editPromptQueueItem(itemId: string, prompt: string) {
  const nextPrompt = prompt.trim();
  if (!nextPrompt) {
    return false;
  }
  const match = findPromptQueueRunByItemId(itemId);
  if (!match) {
    return false;
  }
  const { item } = match;
  if (!canEditPromptQueueItem(item)) {
    return false;
  }
  item.prompt = nextPrompt;
  syncPromptQueueUI();
  return true;
}

function removePromptQueueItem(itemId: string) {
  const match = findPromptQueueRunByItemId(itemId);
  if (!match) {
    return false;
  }
  const { run, itemIndex, item } = match;
  if (!canRemovePromptQueueItem(item)) {
    return false;
  }

  const wasActive = itemIndex === Math.max(run.index, 0);
  run.items.splice(itemIndex, 1);
  if (run.items.length === 0) {
    deletePromptQueueRun(run);
    return true;
  }

  if (itemIndex < run.index) {
    run.index -= 1;
  }
  if (wasActive && run.index >= run.items.length) {
    deletePromptQueueRun(run);
    return true;
  }

  syncPromptQueueUI();
  if (wasActive) {
    clearPromptQueueRetryTimer(run);
    if (run.index < 0 || run.waitingForTargetIdle) {
      return true;
    }
    run.prevStoreRunning = false;
    const next = run.items[run.index];
    if (next) {
      scheduleQueuedPromptDispatch(run, next, 50);
    }
  }
  return true;
}

/**
 * Move a queued prompt into another's slot. Both must still be pending: a
 * dispatched item is already on its way out, and a run mid-dispatch would race
 * the pump. Insert index is read off the pre-splice array, so a downward drag
 * lands after the target and an upward drag lands before it.
 */
function movePromptQueueItem(itemId: string, targetItemId: string) {
  if (itemId === targetItemId) {
    return false;
  }
  const match = findPromptQueueRunByItemId(itemId);
  const target = findPromptQueueRunByItemId(targetItemId);
  if (!match || !target || match.run !== target.run) {
    return false;
  }
  const { run, itemIndex, item } = match;
  if (item.dispatched || target.item.dispatched) {
    return false;
  }
  if (promptQueueDispatchingRunIds.has(run.id)) {
    return false;
  }
  const activeIndex = Math.max(run.index, 0);
  const before = run.items;
  const after = reorderPromptQueueItems(
    before,
    itemIndex,
    target.itemIndex,
    activeIndex,
  );
  if (!after) {
    return false;
  }
  const activeChanged = promptQueueActiveItemChanged(before, after, run.index);
  run.items = after;
  syncPromptQueueUI();

  // A move across the active slot changes what dispatches next, so retarget the
  // pending send the way a removal does.
  const nowActive = run.items[run.index];
  if (run.index >= 0 && !run.waitingForTargetIdle && nowActive && activeChanged) {
    clearPromptQueueRetryTimer(run);
    run.prevStoreRunning = false;
    scheduleQueuedPromptDispatch(run, nowActive, 50);
  }
  return true;
}

function isPromptQueueTargetRunning(
  target: PromptQueueTarget,
  runningByThreadId: Record<string, boolean>,
) {
  // assistant-ui marks a run synchronously when append starts, while the
  // shared store is set later after model loading and request validation.
  // Reading the target closes the rapid-submit window where another append
  // would otherwise cancel the run that just started.
  try {
    if (target.isRunning()) {
      return true;
    }
  } catch {
    // Fall back to the shared store if the thread runtime is remounting.
  }
  const runningIds = Object.keys(runningByThreadId);
  if (runningIds.length === 0) {
    return false;
  }

  const targetIds = target.getRunningThreadIds();
  if (targetIds.length === 0) {
    // Never borrow another chat's running state. A queue without a resolved
    // target should retry its own dispatch instead of becoming globally gated.
    return false;
  }

  return runningIds.some((threadId) => targetIds.includes(threadId));
}

function isPromptQueueRunTargetRunning(
  run: PromptQueueRun,
  runningByThreadId: Record<string, boolean>,
) {
  const activeItem = getActivePromptQueueItem(run);
  if (!activeItem) {
    return false;
  }
  return isPromptQueueTargetRunning(activeItem.target, runningByThreadId);
}

function advancePromptQueue(run: PromptQueueRun) {
  clearPromptQueueRetryTimer(run);
  promptQueueActiveRunIds.delete(run.id);
  getActivePromptQueueItem(run)?.target.complete();
  const nextIndex = run.index + 1;
  if (nextIndex >= run.items.length) {
    deletePromptQueueRun(run);
    return;
  }
  run.index = nextIndex;
  run.waitingForTargetIdle = false;
  run.prevStoreRunning = false;
  syncPromptQueueUI();
  requestPromptQueuePump(100);
}

function shouldPollPromptQueueTargetState(run: PromptQueueRun) {
  return (
    run.waitingForTargetIdle ||
    run.index < 0 ||
    Boolean(getActivePromptQueueItem(run)?.dispatched)
  );
}

function schedulePromptQueueTargetStatePoll(run: PromptQueueRun) {
  const isWaitingForTargetState = shouldPollPromptQueueTargetState(run);
  if (run.retryTimer || !isWaitingForTargetState) {
    return;
  }
  const generation = run.generation;
  run.retryTimer = setTimeout(() => {
    run.retryTimer = null;
    if (
      promptQueueRuns.get(run.id) !== run ||
      generation !== run.generation ||
      !shouldPollPromptQueueTargetState(run)
    ) {
      return;
    }
    handlePromptQueueRunState(
      run,
      useChatRuntimeStore.getState().runningByThreadId,
    );
    if (
      promptQueueRuns.get(run.id) === run &&
      shouldPollPromptQueueTargetState(run)
    ) {
      schedulePromptQueueTargetStatePoll(run);
    }
  }, PROMPT_QUEUE_TARGET_STATE_POLL_MS);
}

function getRunningThreadCount(runningByThreadId: Record<string, boolean>) {
  return Object.values(runningByThreadId).filter(Boolean).length;
}

function hasReadyPromptQueueRun() {
  return Array.from(promptQueueRuns.values()).some(
    isPromptQueueRunReadyToDispatch,
  );
}

function handlePromptQueueRunState(
  run: PromptQueueRun,
  runningByThreadId: Record<string, boolean>,
) {
  if (!promptQueueRuns.has(run.id)) {
    return;
  }
  const isRunning = isPromptQueueRunTargetRunning(run, runningByThreadId);
  const wasRunning = run.prevStoreRunning;
  run.prevStoreRunning = isRunning;
  if (!wasRunning || isRunning) {
    return;
  }
  if (run.waitingForTargetIdle) {
    clearPromptQueueRetryTimer(run);
    run.waitingForTargetIdle = false;
    const activeItem = run.items[run.index];
    if (activeItem) {
      requestPromptQueuePump(50);
    }
    return;
  }
  advancePromptQueue(run);
  requestPromptQueuePump();
}

function ensurePromptQueueSubscription() {
  if (promptQueueStoreUnsub) {
    return;
  }
  // runningByThreadId tracks the actual thread (not aui.thread()), so detection
  // survives navigation.
  let previousRunningCount = getRunningThreadCount(
    useChatRuntimeStore.getState().runningByThreadId,
  );

  promptQueueStoreUnsub = useChatRuntimeStore.subscribe((state) => {
    if (promptQueueRuns.size === 0) {
      stopPromptQueueSubscription();
      return;
    }
    const nextRunningCount = getRunningThreadCount(state.runningByThreadId);
    for (const run of Array.from(promptQueueRuns.values())) {
      handlePromptQueueRunState(run, state.runningByThreadId);
    }

    if (nextRunningCount < previousRunningCount && hasReadyPromptQueueRun()) {
      requestPromptQueuePump();
    }
    previousRunningCount = nextRunningCount;
  });
}

function startPromptQueue(
  items: string[],
  target: PromptQueueTarget,
  waitForCurrentRun = false,
) {
  const filtered = items.map((item) => item.trim()).filter(Boolean);
  if (filtered.length === 0) {
    return;
  }

  const existingRun = findPromptQueueRunByTarget(target);
  if (existingRun) {
    if (existingRun.deepResearchConsumed) {
      target.consumeDeepResearch();
    }
    existingRun.items.push(
      ...filtered.map((prompt) => createQueuedPrompt(prompt, target)),
    );
    syncPromptQueueUI();
    requestPromptQueuePump();
    return;
  }

  const runningByThreadId = useChatRuntimeStore.getState().runningByThreadId;
  const shouldWaitForCurrentRun =
    waitForCurrentRun &&
    isPromptQueueTargetRunning(target, runningByThreadId);
  const run: PromptQueueRun = {
    id: createPromptQueueRunId(),
    items: filtered.map((prompt) => createQueuedPrompt(prompt, target)),
    index: shouldWaitForCurrentRun ? -1 : 0,
    generation: 0,
    prevStoreRunning: shouldWaitForCurrentRun,
    waitingForTargetIdle: false,
    retryTimer: null,
    deepResearchConsumed: false,
  };
  promptQueueRuns.set(run.id, run);
  promptQueueRunOrder.push(run.id);
  syncPromptQueueUI();
  ensurePromptQueueSubscription();
  if (shouldWaitForCurrentRun) {
    handlePromptQueueRunState(
      run,
      useChatRuntimeStore.getState().runningByThreadId,
    );
    schedulePromptQueueTargetStatePoll(run);
  } else {
    requestPromptQueuePump(50);
  }
}

function getPromptQueueRunsForThreadIds(threadIds?: string[]) {
  if (!threadIds || threadIds.length === 0) {
    return Array.from(promptQueueRuns.values());
  }

  const runs = new Set<PromptQueueRun>();
  for (const id of compactIds(threadIds)) {
    const run = findPromptQueueRunByThreadIds([id]);
    if (run) {
      runs.add(run);
    }
  }
  return Array.from(runs);
}

function stopPromptQueueRun(threadIds?: string[]) {
  for (const run of getPromptQueueRunsForThreadIds(threadIds)) {
    const activeItem = getActivePromptQueueItem(run);
    const activeTarget = activeItem?.target;
    const shouldCancelActiveRun = Boolean(activeItem?.dispatched);
    deletePromptQueueRun(run);
    if (!shouldCancelActiveRun) {
      continue;
    }
    try {
      activeTarget?.cancel();
    } catch {
      // The active run may have already ended.
    }
  }
  requestPromptQueuePumpIfReady();
}

function stopPromptQueueRunForThreadIds(threadIds: string[]) {
  stopPromptQueueRun(threadIds);
}

function waitForPromptQueueTargetIdle(run: PromptQueueRun) {
  clearPromptQueueRetryTimer(run);
  promptQueueActiveRunIds.delete(run.id);
  run.waitingForTargetIdle = true;
  run.prevStoreRunning = true;
  syncPromptQueueUI();
  ensurePromptQueueSubscription();
}

function refreshPromptQueueTargetIdleWait(run: PromptQueueRun) {
  handlePromptQueueRunState(
    run,
    useChatRuntimeStore.getState().runningByThreadId,
  );
  schedulePromptQueueTargetStatePoll(run);
}

function stopLocalPromptQueueRun(run: PromptQueueRun) {
  const activeItem = getActivePromptQueueItem(run);
  const plan = planLocalPromptQueueStop(
    run.items.map((item) => ({
      usesLocalModel: item.target.usesLocalModel,
      dispatched: item.dispatched,
    })),
    run.index,
  );
  if (plan.retainedItemIndexes.length === run.items.length) {
    return;
  }

  run.items = plan.retainedItemIndexes.map((index) => run.items[index]);
  if (!getActivePromptQueueItem(run)) {
    deletePromptQueueRun(run);
    if (!plan.cancelActiveItem) {
      return;
    }
    try {
      activeItem?.target.cancel();
    } catch {
      // The active local run may have already ended.
    }
    return;
  }
  if (plan.activeItemRemoved) {
    clearPromptQueueRetryTimer(run);
  }
  if (plan.cancelActiveItem) {
    waitForPromptQueueTargetIdle(run);
    try {
      activeItem?.target.cancel();
    } catch {
      // The active local run may have already ended.
    }
    refreshPromptQueueTargetIdleWait(run);
    return;
  }
  syncPromptQueueUI();
  if (plan.refreshTargetIdleWait) {
    refreshPromptQueueTargetIdleWait(run);
    return;
  }
  if (plan.activeItemRemoved && run.index >= 0 && !run.waitingForTargetIdle) {
    requestPromptQueuePump(50);
  }
}

function stopLocalPromptQueueRunsForThreadIds(threadIds: string[]) {
  if (threadIds.length === 0) {
    return;
  }
  for (const run of getPromptQueueRunsForThreadIds(threadIds)) {
    stopLocalPromptQueueRun(run);
  }
  requestPromptQueuePumpIfReady();
}

function retainPendingPromptQueueItemsAfterFailure(run: PromptQueueRun) {
  const activeIndex = Math.max(run.index, 0);
  const activeItem = run.items[activeIndex];
  if (run.index < 0 || !activeItem?.dispatched) {
    return false;
  }

  activeItem.target.complete();
  run.items.splice(activeIndex, 1);
  if (!getActivePromptQueueItem(run)) {
    deletePromptQueueRun(run);
    return true;
  }
  waitForPromptQueueTargetIdle(run);
  refreshPromptQueueTargetIdleWait(run);
  return true;
}

function cancelPendingPromptQueueFactoriesForStop<
  T extends { temporary: boolean; cancelled: boolean },
>(
  pendingFactories: Map<string, T>,
  aliases: string[],
  detail: PromptQueueStopEventDetail,
) {
  const { threadIds, temporaryOnly, localOnly } = detail;
  if (localOnly) {
    // Advancing the model boundary invalidates local factories once hydrated.
    // External factories must remain intact.
    return;
  }
  if (
    threadIds &&
    threadIds.length > 0 &&
    !threadIds.some((threadId) => aliases.includes(threadId))
  ) {
    return;
  }
  for (const [key, reservation] of pendingFactories) {
    if (temporaryOnly && !reservation.temporary) {
      continue;
    }
    reservation.cancelled = true;
    pendingFactories.delete(key);
  }
}

function stopAllPromptQueueRuns() {
  const activeRuns = Array.from(promptQueueRuns.values()).map((run) => ({
    activeItem: getActivePromptQueueItem(run),
  }));
  resetPromptQueues();
  for (const { activeItem } of activeRuns) {
    const activeTarget = activeItem?.target;
    const shouldCancelActiveRun = Boolean(activeItem?.dispatched);
    if (!shouldCancelActiveRun) {
      continue;
    }
    try {
      activeTarget?.cancel();
    } catch {
      // The active run may have already ended.
    }
  }
}

function handlePromptQueueRunFailed(threadId?: string | null) {
  if (threadId) {
    const failedRun = findPromptQueueRunByThreadIds([threadId]);
    if (failedRun) {
      if (!retainPendingPromptQueueItemsAfterFailure(failedRun)) {
        // A direct-send preflight failure invalidates follow-ups that were
        // waiting for that run to establish a usable thread.
        discardQueuedChatRunSettingsForThread(threadId);
        deletePromptQueueRun(failedRun);
      }
    } else {
      discardQueuedChatRunSettingsForThread(threadId);
    }
  }
  // A queued adapter can fail validation before its running flag turns on.
  // Pump every other ready queue even when no active run matches the event.
  requestPromptQueuePumpIfReady();
}

if (typeof window !== "undefined") {
  window.addEventListener(PROMPT_QUEUE_STOP_EVENT, (event) => {
    const { threadIds, temporaryOnly, localOnly } =
      (event as CustomEvent<PromptQueueStopEventDetail>).detail ?? {};
    if (localOnly) {
      stopLocalPromptQueueRunsForThreadIds(threadIds ?? []);
      return;
    }
    if (threadIds && threadIds.length > 0) {
      stopPromptQueueRunForThreadIds(threadIds);
      return;
    }
    if (temporaryOnly) {
      return;
    }
    stopAllPromptQueueRuns();
  });
  window.addEventListener(PROMPT_QUEUE_RUN_FAILED_EVENT, (event) => {
    const { threadId } =
      (event as CustomEvent<PromptQueueRunFailedEventDetail>).detail ?? {};
    handlePromptQueueRunFailed(threadId);
  });
}

interface PromptQueueCallbacks {
  startQueue: (
    items: string[],
    waitForCurrentRun?: boolean,
    onAborted?: () => void,
  ) => boolean;
  stopQueue: () => void;
}
const noopStartPromptQueue: PromptQueueCallbacks["startQueue"] = () =>
  false;
const noopStopPromptQueue: PromptQueueCallbacks["stopQueue"] = () => undefined;
const PromptQueueContext = createContext<PromptQueueCallbacks>({
  startQueue: noopStartPromptQueue,
  stopQueue: noopStopPromptQueue,
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

// One message, picked from its role and edit state rather than from a `components` map. See
// thread-message-slot.ts for why the map form costs a full-thread re-render on every delete.
// The selectors are ThreadMessageComponent's own, so what a message subscribes to is unchanged.
const ThreadMessage: FC = () => {
  const role = useAuiState(({ message }) => message.role);
  const isEditing = useAuiState(({ message }) => message.composer.isEditing);
  switch (threadMessageKind(role, isEditing)) {
    case "edit":
      return <EditComposer />;
    case "user":
      return <UserMessage />;
    case "assistant":
      return <AssistantMessage />;
    default:
      return null;
  }
};

// Hoisted, so ThreadPrimitive.Messages sees the same children function on every Thread render. An
// inline arrow changes identity each time, invalidating the memo that keeps the message array from
// being rebuilt, and the bail-out below it would never get to run.
const renderThreadMessage = proplessSlot(ThreadMessage);

// Memoized: chat-page renders this inline in a store-subscribing component, so a parent render
// would otherwise reconcile the whole message list.
export const Thread: FC<{
  hideComposer?: boolean;
  hideWelcome?: boolean;
  targetThreadId?: string;
}> = memo(({ hideComposer, hideWelcome, targetThreadId }) => {
  // Intent-aware autoscroll replaces assistant-ui's built-in autoscroll to
  // prevent the streaming-mutation race that snaps the viewport back to the
  // bottom while the user scrolls up (see the hook for the full explanation).
  const { ref: viewportRef, context: autoScrollContext } =
    useIntentAwareAutoScroll();

  const isComposerAttachPending = useAuiState(({ threads }) =>
    targetThreadId ? threads.mainThreadId !== targetThreadId : false,
  );
  const runtimeThreadId = useAuiState(
    ({ threadListItem }) => threadListItem.id,
  );
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const threadId = targetThreadId ?? activeThreadId ?? null;
  const aui = useAui();
  useThreadForkCounts();

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
  // Same element in an identity-stable ref, so ProgressiveMessages can read the viewport without a
  // prop that would rebuild its row array on thread switch. A ref rather than a document-wide query
  // because the Compare panes each mount their own Thread.
  const viewportElRef = useRef<HTMLElement | null>(null);
  const composedViewportRef = useCallback(
    (node: HTMLElement | null) => {
      viewportElRef.current = node;
      setViewportEl(node);
      viewportRef(node);
    },
    [viewportRef],
  );

  // Copying a selection out of the thread writes the plain text itself rather than letting the
  // browser serialise the selection, which spends over 99% of a long thread's copy building the
  // styled clipboard flavour. thread-fast-copy.ts holds the rule for when that substitution is
  // provably invisible, and hands the event back to the browser whenever it is not.
  useEffect(() => {
    if (!viewportEl) return;
    return attachThreadFastCopy(viewportEl);
  }, [viewportEl]);

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
    <GeneratedImageOverlayProvider key={runtimeThreadId} threadId={threadId}>
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
              hideComposer
                ? "pt-4"
                : // + the chat-model notice, which is an opaque absolute bar
                  // directly under the header. 0px whenever it is not showing,
                  // so every other surface keeps the padding it had.
                  "pt-[calc(var(--studio-content-top-inset,0px)+48px+var(--studio-chat-notice-height,0px))]",
            )}
          >
            {!hideWelcome && (
              <AuiIf
                condition={({ thread }) => thread.isEmpty && !thread.isLoading}
              >
                <ThreadWelcome hideComposer={hideComposer} threadId={threadId} />
              </AuiIf>
            )}

            {/* Drop-in for ThreadPrimitive.Messages that bounds a long thread's first commit to
            the tail and mounts the rest over the following frames. Nothing unmounts and the
            document converges to the tree this rendered before; consumers that cannot wait call
            completeProgressiveMounts. It takes the propless slot #9042 introduced, for the same
            reason: React's bail-out needs one shared element per row. See
            progressive-mount-controller.ts. */}
            <ProgressiveMessages
              renderMessage={renderThreadMessage}
              resetKey={runtimeThreadId}
              viewportRef={viewportElRef}
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
});
Thread.displayName = "Thread";

const GeneratedImageViewportOverlay: FC<{
  hideComposer?: boolean;
  bottomOffsetPx?: number | null;
}> = ({ hideComposer, bottomOffsetPx }) => {
  const { overlay, closeOverlay } = useGeneratedImageOverlay();

  useEffect(() => {
    if (!overlay) {
      return;
    }
    document
      .querySelector<HTMLTextAreaElement>(COMPOSER_INPUT_SELECTOR)
      ?.focus();
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
              <p className="truncate text-ui-11 font-medium text-muted-foreground">
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
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const threadListItemId = useAuiState(
    ({ threadListItem }) => threadListItem.id,
  );
  const threadListItemRemoteId = useAuiState(
    ({ threadListItem }) => threadListItem.remoteId,
  );
  const promptQueueThreadIds = compactIds([
    threadListItemId,
    threadListItemRemoteId,
    threadId,
    activeThreadId,
  ]);
  const queueVisible = usePromptQueueUI(
    (s) => {
      const entry = findPromptQueueEntry(s, promptQueueThreadIds);
      return Boolean(
        entry && s.items.some((item) => item.runId === entry.runId),
      );
    },
  );
  const showModelDisclaimer = useChatPreferencesStore(
    (s) => s.showModelDisclaimer,
  );

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
        className={cn(
          "thread-bottom-fade absolute inset-x-0 bottom-0 bg-gradient-to-t from-background from-[calc(100%_-_28px)] to-[rgb(from_var(--background)_r_g_b/0)]",
          queueVisible
            ? "h-32 backdrop-blur-[1px] [mask-image:linear-gradient(to_top,black_0%,black_58%,transparent_100%)]"
            : "top-[10px]",
        )}
      />
      <div className="relative px-5 pb-2">
        <div className="pointer-events-auto mx-auto w-full max-w-(--thread-max-width)">
          <ComposerAnimated
            disabled={disabled}
            threadId={threadId}
            menuSide="top"
          />
        </div>
        {showModelDisclaimer && (
          <p className="composer-footer-note">
            LLMs can make mistakes. Double-check responses.
          </p>
        )}
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
  const incognito = useChatRuntimeStore((s) => s.incognito);
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const showGreetingSloth = useUserProfileStore((s) => s.showGreetingSloth);
  const [welcome, setWelcome] = useState<Welcome>(DEFAULT_WELCOME);

  useEffect(() => {
    // Prefer the nickname; otherwise first name only. Blank falls back to none.
    const raw = nickname.trim() || (displayName.trim().split(/\s+/)[0] ?? "");
    // Cap very long names so the greeting stays on one line.
    const name = raw.length > 20 ? `${raw.slice(0, 20)}…` : raw;
    setWelcome(buildWelcome(new Date().getHours(), name));
  }, [displayName, nickname]);

  const currentEmojiSrc = `Sloth emojis/${welcome.sloth}`;

  return (
    <div className="aui-thread-welcome-root mx-auto my-auto flex w-full max-w-(--thread-max-width) grow flex-col">
      <div className="aui-thread-welcome-center flex w-full grow flex-col items-center justify-start pt-[27.5dvh]">
        <div className="aui-thread-welcome-message flex w-full flex-col justify-center gap-9 px-4">
          {/* Center the greeting (sloth + title) over the composer. */}
          <div className="flex flex-row items-center justify-center gap-[15px]">
            {/* Temporary chat keeps the title on its own, no mascot. */}
            {showGreetingSloth && !incognito && (
              <MascotImg
                src={currentEmojiSrc}
                className="size-[44px] -translate-y-[2px]"
              />
            )}
            <h1 className="aui-thread-welcome-message-inner unsloth-welcome-title fade-in slide-in-from-bottom-1 animate-in text-3xl tracking-[-0.02em] duration-200">
              {incognito ? "Temporary chat" : welcome.text}
            </h1>
          </div>
          {incognito && (
            <p className="aui-thread-welcome-message-inner fade-in -mt-2 animate-in text-center font-heading font-normal text-muted-foreground text-sm duration-200">
              This chat won't appear in your history and isn't saved. It
              disappears when you leave.
            </p>
          )}
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
      {/* New chat in a project: queuing follow-ups here misbinds the thread,
          so the queue only runs once the user is inside a chat session. */}
      <ComposerAnimated
        disabled={disabled}
        placeholder={placeholder}
        disableQueue
      />
    </GeneratedImageOverlayProvider>
  );
};

const ComposerAnimated: FC<{
  disabled?: boolean;
  placeholder?: string;
  threadId?: string | null;
  menuSide?: "top" | "bottom";
  disableQueue?: boolean;
}> = ({ disabled, threadId, menuSide, disableQueue }) => {
  return (
    <div className="relative mx-auto min-w-0 w-full max-w-[46rem]">
      <div className="relative z-10 w-full">
        <Composer
          disabled={disabled}
          threadId={threadId}
          menuSide={menuSide}
          disableQueue={disableQueue}
        />
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

/** Keep a drop on a portaled child, such as a dialog or its overlay, from also
 * attaching to the composer. React routes portal events through the composer,
 * whose dropzone attaches in the capture phase before the dialog sees them. */
function claimPortaledDrop(event: ReactDragEvent): void {
  const target = event.target as Element | null;
  if (!target?.closest?.(".aui-composer-attachment-dropzone")) {
    event.preventDefault();
  }
}

const Composer: FC<{
  disabled?: boolean;
  placeholder?: string;
  threadId?: string | null;
  menuSide?: "top" | "bottom";
  disableQueue?: boolean;
}> = ({ disabled, threadId, menuSide, disableQueue }) => {
  const aui = useAui();
  const isDictating = useAuiState((s) => s.composer.dictation != null);
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
  const deepResearchEnabled = useChatRuntimeStore(
    (s) => s.deepResearchEnabled,
  );
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const researchThreadId = threadId ?? activeThreadId ?? null;
  const researchThreadClaimed = useResearchRunStore((state) =>
    researchThreadId ? Boolean(state.claimedThreadIds[researchThreadId]) : false,
  );
  const liveResearchRunId = useResearchRunStore((state) =>
    researchThreadId ? state.latestRunByThreadId[researchThreadId] : undefined,
  );
  // Derive in the selector, as useThreadResearchActive does: a bare run selector re-renders the
  // composer on every streamed research delta.
  const isResearchActive = useResearchRunStore((state) => {
    const runId = researchThreadId
      ? state.latestRunByThreadId[researchThreadId]
      : undefined;
    const run = runId ? state.sessions[runId]?.run : undefined;
    return Boolean(
      run && !["completed", "failed", "cancelled"].includes(run.status),
    );
  });
  const hasResearchMessage = useAuiState(({ thread }) =>
    threadHasResearchMessage(thread.messages, liveResearchRunId),
  );
  const researchUsed = researchThreadClaimed || hasResearchMessage;
  const effectiveDeepResearchEnabled = deepResearchEnabled && !researchUsed;
  const [researchWebsiteAccessOpen, setResearchWebsiteAccessOpen] =
    useState(false);
  useEffect(() => {
    if (!researchUsed) return;
    if (hasResearchMessage && researchThreadId) {
      useResearchRunStore.getState().setThreadClaimed(researchThreadId, true);
    }
    if (deepResearchEnabled) {
      useChatRuntimeStore.getState().setDeepResearchEnabled(false);
    }
  }, [deepResearchEnabled, hasResearchMessage, researchThreadId, researchUsed]);
  // More than 4 pills: collapse to icons only. Search, Code, and permissions
  // always show; Images, RAG, Canvas, MCP and Deep Research are conditional.
  // Narrow viewports collapse too: the labelled row is wider than a phone composer.
  const isMobile = useIsMobile();
  const pillCount =
    3 +
    (ragEnabled ? 1 : 0) +
    (supportsBuiltinImageGeneration ? 1 : 0) +
    (artifactsEnabled ? 1 : 0) +
    (mcpEnabledForChat ? 1 : 0) +
    (effectiveDeepResearchEnabled ? 1 : 0);
  // Under the count threshold the row still overflows on long labels ("Run
  // automatically" next to "Deep research"), which dropped the dictate and
  // send buttons onto a second line. Measuring collapses just enough.
  const { pillRowRef, pillCompact } = useComposerPillFit(
    isMobile || pillCount > 4,
  );
  const setPendingImageEditReference = useChatRuntimeStore(
    (s) => s.setPendingImageEditReference,
  );
  const pastedTextMinChars = useChatPreferencesStore(
    (state) => state.pastedTextMinChars,
  );
  // Set by Cmd/Ctrl+Enter and read once by the handleSubmit that requestSubmit
  // reaches synchronously. Armed only when that call will happen: with no form,
  // or no requestSubmit, it would stay armed and queue whatever submit came
  // next.
  const forceQueueRef = useRef(false);
  const queueOnModEnter = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      const form = event.currentTarget.form;
      if (typeof form?.requestSubmit !== "function") {
        return;
      }
      forceQueueRef.current = true;
      try {
        form.requestSubmit();
      } catch {
        forceQueueRef.current = false;
      }
    },
    [],
  );
  // Read by both writers that could put the sent text back, the input handlers
  // and the draft restore. Armed by every path that clears the composer.
  const justSentRef = useRef<SentTextGuard | null>(null);
  // Thread on screen, so the guard can tell whether a write belongs to the
  // thread that sent. Kept in step by the effect alongside pasteDraftKeyRef.
  const draftKeyRef = useRef<string | null>(null);
  const { inputProps, isComposing, isComposingRef } =
    useImeComposerInputHandlers({
      submitOnEnter: true,
      onModEnter: queueOnModEnter,
      justSentRef,
      draftKeyRef,
    });
  // A pasted YouTube link offers a transcript attachment above the composer.
  const [youtubeLink, setYoutubeLink] = useState<string | null>(null);
  // Paste without formatting asks for the clipboard in the field, so the paste
  // it makes stays inline however long it is. A paste event carries no
  // modifiers, so the chord is read from the keydown before it, and the flag
  // lasts only as long as the keys are down: the paste is the keydown's own
  // default action, while a menu the user might reach for instead cannot be
  // opened without letting go first.
  const plainPasteAtRef = useRef(0);
  const notePlainPasteChord = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      plainPasteAtRef.current = isPlainPasteChord(event)
        ? performance.now()
        : 0;
    },
    [],
  );
  // Any release ends it, whichever key of the chord goes first, as does losing
  // the field. The time cap behind them is for a release that never lands,
  // which is what tabbing away mid-chord used to leave behind.
  const endPlainPasteChord = useCallback(() => {
    plainPasteAtRef.current = 0;
  }, []);
  const handleFilePaste = useCallback(
    (event: ClipboardEvent<HTMLTextAreaElement>) => {
      // Read once and cleared here, so a paste with no chord before it, from
      // the menu or a script, is never taken for the plain one.
      const plainPaste = plainPasteStillCounts(
        plainPasteAtRef.current,
        performance.now(),
      );
      plainPasteAtRef.current = 0;
      const pastedText = event.clipboardData?.getData("text/plain") ?? "";
      if (extractYoutubeVideoId(pastedText)) {
        setYoutubeLink(pastedText.trim());
      }
      // Bulk text pastes attach as a file instead of filling the input, except
      // in image-edit mode, whose submit path takes an inline instruction only.
      const input = event.currentTarget;
      // An attachment is serialised after all inline text, so only a paste that
      // was already heading to the end can become one. Mid-text pastes stay
      // inline, where the order the user typed them in survives.
      const pasteGoesLast = input.selectionEnd === input.value.length;
      const { selectionStart, selectionEnd, value } = input;
      // Swallowing the paste also swallowed the replacement the browser would
      // have made. Only once the attachment is in, and only if the composer is
      // still the one that was pasted into, or a failed paste eats the text.
      const dropReplacedSelection = () => {
        if (selectionStart === selectionEnd) return;
        const composer = aui.composer();
        if (composer.getState().text !== value) return;
        composer.setText(value.slice(0, selectionStart) + value.slice(selectionEnd));
      };
      const attachedPastedText =
        !overlay &&
        !plainPaste &&
        pasteGoesLast &&
        pasteLongTextAsFile(
        event,
        async (file) => {
          await aui.composer().addAttachment(file);
          dropReplacedSelection();
        },
        () =>
          toast.error("Could not attach the pasted text.", {
            description: "Paste it again, or paste it in smaller pieces.",
          }),
        pastedTextMinChars,
      );
      if (attachedPastedText) return;
      pasteClipboardFiles(
        event,
        async (files) => {
          await Promise.all(
            files.map((file) => aui.composer().addAttachment(file)),
          );
        },
        () =>
          toast.error("Could not paste files.", {
            description: "The clipboard item is unsupported, unreadable, or exceeds its size limit.",
          }),
      );
      // A paste is a gesture, so it retires the guard and re-pasting the sent
      // prompt goes through. Last, and only when the browser will really insert
      // the text: a payload carrying files is preventDefaulted above, so
      // retiring for it would just free the next queued write to refill.
      if (
        pastedText.length > 0 &&
        !event.defaultPrevented &&
        justSentRef.current?.draftKey === draftKeyRef.current
      ) {
        justSentRef.current = null;
      }
    },
    [aui, overlay, pastedTextMinChars],
  );

  const composerText = useAuiState(({ composer }) => composer.text);
  // Derived, not cleared in an effect: the offer retracts as soon as the link
  // leaves the draft, which also covers sending.
  const youtubeOfferUrl =
    youtubeLink !== null && composerText.includes(youtubeLink)
      ? youtubeLink
      : null;
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
  const attachmentsAreAllPastedText = useAuiState(
    ({ composer }) =>
      composer.attachments.length > 0 &&
      composer.attachments.every((attachment) =>
        isPastedTextFile((attachment as { file?: File }).file),
      ),
  );
  // Identities only: paste autosave keys off this, and the bodies behind it can
  // be megabytes. Every attachment counts, not just pasted ones, so removing an
  // ordinary file also releases the paste restore waiting on it.
  const composerAttachmentSignature = useAuiState(({ composer }) =>
    composer.attachments.map((attachment) => attachment.id).join(","),
  );
  const hasPendingAudio = useChatRuntimeStore((s) =>
    Boolean(s.pendingAudioName),
  );
  const nativeAttachmentTargetKey = useNativeAttachmentTargetKey();
  const nativeAttachmentTargetKeyRef = useRef(nativeAttachmentTargetKey);
  nativeAttachmentTargetKeyRef.current = nativeAttachmentTargetKey;
  const hasPendingImageAttachments = useNativeIntentStore((s) =>
    Boolean(
      nativeAttachmentTargetKey &&
        (s.pendingImageAttachments[nativeAttachmentTargetKey]?.length ?? 0) > 0,
    ),
  );
  const hasPendingOpenDocumentAttachments = useNativeIntentStore((s) =>
    Boolean(
      nativeAttachmentTargetKey &&
        (s.pendingOpenDocumentAttachments[nativeAttachmentTargetKey]?.length ??
          0) > 0,
    ),
  );
  const registeringImageDrops = useNativeIntentStore(
    (s) => s.registeringImageDrops > 0,
  );
  const [materializingDroppedImages, setMaterializingDroppedImages] =
    useState(false);
  const [
    materializingDroppedOpenDocuments,
    setMaterializingDroppedOpenDocuments,
  ] = useState(false);
  const hasPendingAudioAttachments = useNativeIntentStore((s) =>
    Boolean(
      nativeAttachmentTargetKey &&
        (s.pendingAudioAttachments[nativeAttachmentTargetKey]?.length ?? 0) > 0,
    ),
  );
  const registeringAudioDrops = useNativeIntentStore(
    (s) => s.registeringAudioDrops > 0,
  );
  const [materializingDroppedAudio, setMaterializingDroppedAudio] =
    useState(false);
  const hasPendingVideoAttachments = useNativeIntentStore((s) =>
    Boolean(
      nativeAttachmentTargetKey &&
        (s.pendingVideoAttachments[nativeAttachmentTargetKey]?.length ?? 0) > 0,
    ),
  );
  const registeringVideoDrops = useNativeIntentStore(
    (s) => s.registeringVideoDrops > 0,
  );
  const [materializingDroppedVideo, setMaterializingDroppedVideo] =
    useState(false);
  // A parked send must not fire on a failed drop: the user is owed the toast and
  // their text, not a send of the text alone. Assigned below, once the callback exists.
  const cancelQueuedSendRef = useRef<(() => void) | null>(null);
  // Which composer is mounted, for deciding where a drain puts work back.
  const composerIdentityRef = useRef("");
  const imageDropFailures = useNativeIntentStore(
    (s) => (nativeAttachmentTargetKey ? s.imageDropFailures[nativeAttachmentTargetKey] : 0) ?? 0,
  );
  const seenImageDropFailuresRef = useRef(imageDropFailures);
  // Registration fails before an intent exists, so the drain never sees it.
  // Cancel here or the parked send goes out with the text alone.
  useEffect(() => {
    if (seenImageDropFailuresRef.current === imageDropFailures) return;
    seenImageDropFailuresRef.current = imageDropFailures;
    cancelQueuedSendRef.current?.();
  }, [imageDropFailures]);
  const audioDropFailures = useNativeIntentStore(
    (s) => (nativeAttachmentTargetKey ? s.audioDropFailures[nativeAttachmentTargetKey] : 0) ?? 0,
  );
  const seenAudioDropFailuresRef = useRef(audioDropFailures);
  // Cancel the parked send before `endAudioDropRegistration` reopens the gate.
  useEffect(() => {
    if (seenAudioDropFailuresRef.current === audioDropFailures) return;
    seenAudioDropFailuresRef.current = audioDropFailures;
    cancelQueuedSendRef.current?.();
  }, [audioDropFailures]);
  const videoDropFailures = useNativeIntentStore(
    (s) => (nativeAttachmentTargetKey ? s.videoDropFailures[nativeAttachmentTargetKey] : 0) ?? 0,
  );
  const seenVideoDropFailuresRef = useRef(videoDropFailures);
  // Cancel the parked send before `endVideoDropRegistration` reopens the gate.
  useEffect(() => {
    if (seenVideoDropFailuresRef.current === videoDropFailures) return;
    seenVideoDropFailuresRef.current = videoDropFailures;
    cancelQueuedSendRef.current?.();
  }, [videoDropFailures]);
  // Registering and reading a dropped clip is async, so hold the send gate:
  // the composer sees nothing until `addAttachment` lands.
  useEffect(() => {
    if (!nativeAttachmentTargetKey) {
      return;
    }
    const targetKey = nativeAttachmentTargetKey;
    const identityAtSetup = composerIdentityRef.current;
    useNativeIntentStore
      .getState()
      .claimAudioAttachments(identityAtSetup, targetKey);
    let disposed = false;
    let draining = false;

    // A re-key follows the same composer; a thread switch parks the clip back.
    const stillThisComposer = () =>
      composerIdentityRef.current === identityAtSetup;
    // A remount hides the new key, so tag the batch; the next instance claims it.
    const requeue = (intents: NativeIntent[]) => {
      const key = stillThisComposer()
        ? (nativeAttachmentTargetKeyRef.current ?? targetKey)
        : targetKey;
      const store = useNativeIntentStore.getState();
      store.addAudioAttachments(key, intents);
      store.noteAudioDropOwner(key, identityAtSetup);
    };

    const drainPendingAudio = async () => {
      if (disposed || draining) return;
      draining = true;
      setMaterializingDroppedAudio(true);
      try {
        while (!disposed) {
          const intents = useNativeIntentStore
            .getState()
            .takeAudioAttachments(targetKey);
          if (intents.length === 0) break;
          for (const [index, intent] of intents.entries()) {
            if (disposed) {
              requeue(intents.slice(index));
              return;
            }
            let file: File;
            try {
              file = await nativeAttachmentIntentToFile(intent);
            } catch (error) {
              toast.error("Could not attach dropped audio", {
                description:
                  error instanceof Error ? error.message : String(error),
              });
              // Do not let a send parked on this clip go out as bare text.
              if (stillThisComposer()) cancelQueuedSendRef.current?.();
              continue;
            }
            // The read is async; a chat switch in that window must not steal the clip.
            if (
              disposed ||
              nativeAttachmentTargetKeyRef.current !== targetKey
            ) {
              requeue(intents.slice(index));
              return;
            }
            try {
              await aui.composer().addAttachment(file);
            } catch {
              // Chat-wide, not per file (no audio model, too large, already
              // attached), and every adapter path toasted: stop quietly.
              if (stillThisComposer()) cancelQueuedSendRef.current?.();
              return;
            }
          }
        }
      } finally {
        draining = false;
        // A drain for a target already left must not touch the flag; cleanup
        // cleared it, and the live target may have set it again.
        if (!disposed) {
          // The early returns requeue mid-batch, and a drop can land while
          // `draining` gated the subscription.
          const pending =
            useNativeIntentStore.getState().pendingAudioAttachments[targetKey]
              ?.length ?? 0;
          // Only the instance still owning this composer re-drains; otherwise
          // the batch stays parked rather than looping here forever.
          if (pending > 0 && stillThisComposer()) {
            void drainPendingAudio();
          } else {
            setMaterializingDroppedAudio(false);
          }
        }
      }
    };

    const unsubscribe = useNativeIntentStore.subscribe((state) => {
      // A predecessor's requeue can land after setup, so keep watching.
      const orphaned = Object.entries(state.audioDropOwners).some(
        ([key, owner]) => owner === identityAtSetup && key !== targetKey,
      );
      if (orphaned) {
        useNativeIntentStore
          .getState()
          .claimAudioAttachments(identityAtSetup, targetKey);
        return;
      }
      if ((state.pendingAudioAttachments[targetKey]?.length ?? 0) > 0) {
        void drainPendingAudio();
      }
    });
    void drainPendingAudio();

    return () => {
      disposed = true;
      setMaterializingDroppedAudio(false);
      unsubscribe();
    };
  }, [nativeAttachmentTargetKey, aui]);

  // Same drain as audio, one queue over: one clip per message, and the send
  // gate has to hold across the read either way.
  useEffect(() => {
    if (!nativeAttachmentTargetKey) {
      return;
    }
    const targetKey = nativeAttachmentTargetKey;
    const identityAtSetup = composerIdentityRef.current;
    useNativeIntentStore
      .getState()
      .claimVideoAttachments(identityAtSetup, targetKey);
    let disposed = false;
    let draining = false;

    // A re-key follows the same composer; a thread switch parks the clip back.
    const stillThisComposer = () =>
      composerIdentityRef.current === identityAtSetup;
    // A remount hides the new key, so tag the batch; the next instance claims it.
    const requeue = (intents: NativeIntent[]) => {
      const key = stillThisComposer()
        ? (nativeAttachmentTargetKeyRef.current ?? targetKey)
        : targetKey;
      const store = useNativeIntentStore.getState();
      store.addVideoAttachments(key, intents);
      store.noteVideoDropOwner(key, identityAtSetup);
    };

    const drainPendingVideo = async () => {
      if (disposed || draining) return;
      draining = true;
      setMaterializingDroppedVideo(true);
      try {
        while (!disposed) {
          const intents = useNativeIntentStore
            .getState()
            .takeVideoAttachments(targetKey);
          if (intents.length === 0) break;
          for (const [index, intent] of intents.entries()) {
            if (disposed) {
              requeue(intents.slice(index));
              return;
            }
            let file: File;
            try {
              file = await nativeAttachmentIntentToFile(intent);
            } catch (error) {
              toast.error("Could not attach dropped video", {
                description:
                  error instanceof Error ? error.message : String(error),
              });
              // Do not let a send parked on this clip go out as bare text.
              if (stillThisComposer()) cancelQueuedSendRef.current?.();
              continue;
            }
            // The read is async; a chat switch in that window must not steal the clip.
            if (
              disposed ||
              nativeAttachmentTargetKeyRef.current !== targetKey
            ) {
              requeue(intents.slice(index));
              return;
            }
            try {
              await aui.composer().addAttachment(file);
            } catch {
              // Chat-wide, not per file (no video mmproj, no ffmpeg, too large,
              // already attached), and every adapter path toasted: stop quietly.
              if (stillThisComposer()) cancelQueuedSendRef.current?.();
              return;
            }
          }
        }
      } finally {
        draining = false;
        // A drain for a target already left must not touch the flag; cleanup
        // cleared it, and the live target may have set it again.
        if (!disposed) {
          // The early returns requeue mid-batch, and a drop can land while
          // `draining` gated the subscription.
          const pending =
            useNativeIntentStore.getState().pendingVideoAttachments[targetKey]
              ?.length ?? 0;
          // Only the instance still owning this composer re-drains; otherwise
          // the batch stays parked rather than looping here forever.
          if (pending > 0 && stillThisComposer()) {
            void drainPendingVideo();
          } else {
            setMaterializingDroppedVideo(false);
          }
        }
      }
    };

    const unsubscribe = useNativeIntentStore.subscribe((state) => {
      // A predecessor's requeue can land after setup, so keep watching.
      const orphaned = Object.entries(state.videoDropOwners).some(
        ([key, owner]) => owner === identityAtSetup && key !== targetKey,
      );
      if (orphaned) {
        useNativeIntentStore
          .getState()
          .claimVideoAttachments(identityAtSetup, targetKey);
        return;
      }
      if ((state.pendingVideoAttachments[targetKey]?.length ?? 0) > 0) {
        void drainPendingVideo();
      }
    });
    void drainPendingVideo();

    return () => {
      disposed = true;
      setMaterializingDroppedVideo(false);
      unsubscribe();
    };
  }, [nativeAttachmentTargetKey, aui]);

  useEffect(() => {
    if (!nativeAttachmentTargetKey) {
      return;
    }
    const targetKey = nativeAttachmentTargetKey;
    const identityAtSetup = composerIdentityRef.current;
    useNativeIntentStore
      .getState()
      .claimImageAttachments(identityAtSetup, targetKey);
    let disposed = false;
    let draining = false;

    // A fresh chat re-keys from "single:new" to its thread id under the same
    // composer, so follow it; a real thread switch keeps the original target.
    const stillThisComposer = () =>
      composerIdentityRef.current === identityAtSetup;
    const requeueKey = () =>
      stillThisComposer()
        ? (nativeAttachmentTargetKeyRef.current ?? targetKey)
        : targetKey;
    // A fresh chat persisting remounts this composer, so the key it moves to is
    // not visible here. Tag the batch instead; the next instance claims it.
    const requeue = (intents: NativeIntent[]) => {
      const key = requeueKey();
      const store = useNativeIntentStore.getState();
      store.addImageAttachments(key, intents);
      store.noteImageDropOwner(key, identityAtSetup);
    };

    const drainPendingImages = async () => {
      if (disposed || draining) {
        return;
      }
      draining = true;
      setMaterializingDroppedImages(true);
      let readFailures = 0;
      let lastReadError: unknown;
      try {
        while (!disposed) {
          const intents = useNativeIntentStore
            .getState()
            .takeImageAttachments(targetKey);
          if (intents.length === 0) {
            break;
          }
          for (let index = 0; index < intents.length; index += 1) {
            if (disposed) {
              requeue(intents.slice(index));
              return;
            }
            const intent = intents[index]!;
            let file: File;
            try {
              file = await nativeAttachmentIntentToFile(intent);
            } catch (error) {
              // Report once below rather than one toast per file: a whole batch
              // can go unreadable at once (volume ejected, tokens expired).
              readFailures += 1;
              lastReadError = error;
              continue;
            }
            if (
              disposed ||
              nativeAttachmentTargetKeyRef.current !== targetKey
            ) {
              requeue(intents.slice(index));
              return;
            }
            try {
              await aui.composer().addAttachment(file);
            } catch {
              // Chat-wide, not per file (no vision model, or none loaded). The
              // adapter toasted, and the rest would fail alike: stop quietly.
              if (stillThisComposer()) cancelQueuedSendRef.current?.();
              return;
            }
          }
        }
      } finally {
        draining = false;
        if (readFailures > 0) {
          toast.error("Could not attach dropped images", {
            description:
              lastReadError instanceof Error
                ? lastReadError.message
                : String(lastReadError),
          });
          // A re-key still owns the parked send; a real thread switch does not.
          if (stillThisComposer()) cancelQueuedSendRef.current?.();
        }
        // A drain for a target the composer has already left must not touch the
        // flag: cleanup cleared it, and the live target may have set it again.
        if (disposed) {
          return;
        }
        const pending =
          useNativeIntentStore.getState().pendingImageAttachments[targetKey]
            ?.length ?? 0;
        if (pending > 0) {
          void drainPendingImages();
        } else {
          setMaterializingDroppedImages(false);
        }
      }
    };

    const unsubscribe = useNativeIntentStore.subscribe((state) => {
      // The predecessor's requeue can land after the claim at setup, so keep
      // watching rather than claiming once.
      const orphaned = Object.entries(state.imageDropOwners).some(
        ([key, owner]) => owner === identityAtSetup && key !== targetKey,
      );
      if (orphaned) {
        useNativeIntentStore
          .getState()
          .claimImageAttachments(identityAtSetup, targetKey);
        return;
      }
      const pending =
        state.pendingImageAttachments[targetKey]?.length ?? 0;
      if (pending > 0) {
        void drainPendingImages();
      }
    });

    void drainPendingImages();

    return () => {
      disposed = true;
      setMaterializingDroppedImages(false);
      unsubscribe();
    };
  }, [nativeAttachmentTargetKey, aui]);
  useEffect(() => {
    if (!nativeAttachmentTargetKey) {
      return;
    }
    const targetKey = nativeAttachmentTargetKey;
    const identityAtSetup = composerIdentityRef.current;
    useNativeIntentStore
      .getState()
      .claimImageAttachments(identityAtSetup, targetKey);
    let disposed = false;
    let draining = false;

    const stillThisComposer = () =>
      composerIdentityRef.current === identityAtSetup;
    const requeue = (intents: NativeIntent[]) => {
      const key = stillThisComposer()
        ? (nativeAttachmentTargetKeyRef.current ?? targetKey)
        : targetKey;
      const store = useNativeIntentStore.getState();
      store.addOpenDocumentAttachments(key, intents);
      store.noteImageDropOwner(key, identityAtSetup);
    };

    const drainPendingOpenDocuments = async () => {
      if (disposed || draining) return;
      draining = true;
      setMaterializingDroppedOpenDocuments(true);
      try {
        while (!disposed) {
          const intents = useNativeIntentStore
            .getState()
            .takeOpenDocumentAttachments(targetKey);
          if (intents.length === 0) break;
          for (const [index, intent] of intents.entries()) {
            if (disposed) {
              requeue(intents.slice(index));
              return;
            }
            let file: File;
            try {
              file = await nativeAttachmentIntentToFile(intent);
            } catch (error) {
              toast.error("Could not attach dropped document", {
                description:
                  error instanceof Error ? error.message : String(error),
              });
              if (stillThisComposer()) cancelQueuedSendRef.current?.();
              continue;
            }
            if (
              disposed ||
              nativeAttachmentTargetKeyRef.current !== targetKey
            ) {
              requeue(intents.slice(index));
              return;
            }
            try {
              await aui.composer().addAttachment(file);
            } catch {
              if (stillThisComposer()) cancelQueuedSendRef.current?.();
            }
          }
        }
      } finally {
        draining = false;
        if (!disposed) {
          const pending =
            useNativeIntentStore.getState().pendingOpenDocumentAttachments[
              targetKey
            ]?.length ?? 0;
          if (pending > 0 && stillThisComposer()) {
            void drainPendingOpenDocuments();
          } else {
            setMaterializingDroppedOpenDocuments(false);
          }
        }
      }
    };

    const unsubscribe = useNativeIntentStore.subscribe((state) => {
      const orphaned = Object.entries(state.imageDropOwners).some(
        ([key, owner]) => owner === identityAtSetup && key !== targetKey,
      );
      if (orphaned) {
        useNativeIntentStore
          .getState()
          .claimImageAttachments(identityAtSetup, targetKey);
        return;
      }
      if (
        (state.pendingOpenDocumentAttachments[targetKey]?.length ?? 0) > 0
      ) {
        void drainPendingOpenDocuments();
      }
    });
    void drainPendingOpenDocuments();

    return () => {
      disposed = true;
      setMaterializingDroppedOpenDocuments(false);
      unsubscribe();
    };
  }, [nativeAttachmentTargetKey, aui]);
  const hasMaterializingImageAttachments =
    registeringImageDrops ||
    hasPendingImageAttachments ||
    materializingDroppedImages ||
    hasPendingOpenDocumentAttachments ||
    materializingDroppedOpenDocuments;
  const hasMaterializingAudioAttachments =
    registeringAudioDrops ||
    hasPendingAudioAttachments ||
    materializingDroppedAudio;
  const hasMaterializingVideoAttachments =
    registeringVideoDrops ||
    hasPendingVideoAttachments ||
    materializingDroppedVideo;
  const threadIsRunning = useAuiState(({ thread }) => thread.isRunning);
  const threadListItemId = useAuiState(
    ({ threadListItem }) => threadListItem.id,
  );
  const threadListItemRemoteId = useAuiState(
    ({ threadListItem }) => threadListItem.remoteId,
  );
  const referenceThreadId = threadId ?? activeThreadId ?? null;
  // Read at Send time, so a send that materializes after a project switch is still filed
  // where it was made.
  const projectScope = useChatProjectScope();
  const promptQueueThreadIds = compactIds([
    threadListItemId,
    threadListItemRemoteId,
    threadId,
  ]);
  const preStreamThreadIds = compactIds([
    ...promptQueueThreadIds,
    referenceThreadId,
  ]);
  const preStreamRunReservationRef = useRef<symbol | null>(null);
  useEffect(() => {
    const token = preStreamRunReservationRef.current;
    if (!token) {
      return;
    }
    adoptPreStreamRunReservation(token, preStreamThreadIds);
    // Keep the reservation until the adapter consumes or fails it. React can
    // expose isRunning before persistence and model preflight finish; releasing
    // here would hide that accepted send from a concurrent model-change gate.
  }, [preStreamThreadIds]);
  const promptQueueActive = usePromptQueueUI((s) =>
    Boolean(findPromptQueueEntry(s, promptQueueThreadIds)),
  );
  const hasSendableContent =
    composerText.trim().length > 0 || hasAttachments || hasPendingAudio;
  const composerAcceptsQueueing =
    !hasPendingAudio &&
    !isComposing &&
    !hasPendingAttachments &&
    !hasMaterializingImageAttachments &&
    !hasMaterializingAudioAttachments &&
    !hasMaterializingVideoAttachments &&
    !disabled &&
    !overlay;
  const canQueueCurrentPrompt =
    composerText.trim().length > 0 && !hasAttachments && composerAcceptsQueueing;
  // A long paste is text the composer parked in a chip, so it queues like the
  // same text did before it attached, rather than being refused as a file.
  const canQueuePastedTextPrompt =
    attachmentsAreAllPastedText && composerAcceptsQueueing;

  // Per-thread draft autosave: restore on mount, then mirror composer text
  // into localStorage (debounced) so a half-typed message survives a
  // navigation or reload. Cleared once empty (i.e. after a send). Setting the
  // text even when no draft exists keeps a thread from inheriting the
  // previous thread's composer contents.
  const draftThreadId = referenceThreadId;
  const draftKey = draftThreadId ? composerDraftKey(draftThreadId) : null;
  // A pasted attachment is a File held in memory only, so without its own slot
  // an unsent paste is the one draft a reload throws away.
  const pasteDraftKey = draftThreadId
    ? composerPasteDraftKey(draftThreadId)
    : null;
  const lastDraftKeyRef = useRef(draftKey);
  // Which key the paste restore has finished for. The save effect writes only
  // for that key, so a draft is never cleared before it has been put back.
  const restoredPasteKeyRef = useRef<string | null>(null);
  const draftSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    const draft = draftKey ? (readComposerDraft(draftKey) ?? "") : "";
    const composer = aui.composer();
    if (!composer.getState().isEditing) return;
    // A save that raced the send still holds the sent text, so restoring it
    // would undo the clear. Keyed on the sending thread, so another thread's
    // identical draft still restores. Clear rather than return early, which
    // would leave the previous thread's text on screen under this one.
    if (sentTextGuardBlocksDraft(justSentRef.current, draft, draftKey)) {
      // Written inline rather than via clearStoredDraft, which is declared
      // below this effect. Cancel the pending save too, or it rewrites the key.
      if (draftSaveTimerRef.current !== null) {
        clearTimeout(draftSaveTimerRef.current);
        draftSaveTimerRef.current = null;
      }
      if (draftKey) writeComposerDraft(draftKey, "");
      composer.setText("");
      return;
    }
    composer.setText(draft);
  }, [draftKey, aui]);
  // The saved-prompt menu and the prompt storage dialog fill the composer
  // directly, bypassing the guard. Text appearing while the sending thread is
  // on screen was put there deliberately, so retire; the draftKey check keeps
  // another thread's restored draft from doing the same. Read live, or a
  // pending render retires it on stale text. Must stay after the restore
  // above, which clears the raced draft this would otherwise retire on.
  useEffect(() => {
    const guard = justSentRef.current;
    if (guard === null || guard.draftKey !== draftKey) return;
    if (aui.composer().getState().text.length === 0) return;
    justSentRef.current = null;
  }, [composerText, draftKey, aui]);
  // Separate from the text restore above, which must stay keyed on the draft
  // alone: this one retries on attachment changes, and rewriting the composer
  // text on those would drop whatever had been typed since the last autosave.
  useEffect(() => {
    const composer = aui.composer();
    if (!composer.getState().isEditing) return;
    if (restoredPasteKeyRef.current === pasteDraftKey) return;
    // The composer outlives a thread switch, so restore only into an empty one
    // rather than mixing this thread's draft with whatever the last one left.
    // Changing attachments re-runs this effect, which is how the retry happens.
    if (composer.getState().attachments.length > 0) return;
    const stored = pasteDraftKey ? readPasteDraft(pasteDraftKey) : [];
    if (stored.length === 0) {
      restoredPasteKeyRef.current = pasteDraftKey;
      return;
    }
    // Claim the key only once the attachments are in, so the save effect
    // cannot write an empty composer over the draft still being restored.
    void Promise.all(
      stored.map((text) => composer.addAttachment(createPastedTextFile(text))),
    ).finally(() => {
      restoredPasteKeyRef.current = pasteDraftKey;
    });
  }, [pasteDraftKey, composerAttachmentSignature, aui]);
  // Keyed on the paste identities, never their bodies, so typing beside a
  // megabyte paste does not rewrite it to localStorage every 300ms.
  useEffect(() => {
    if (!pasteDraftKey || restoredPasteKeyRef.current !== pasteDraftKey) return;
    const pastes = aui
      .composer()
      .getState()
      .attachments.flatMap((attachment) => {
        const text = pastedTextOf((attachment as { file?: File }).file);
        return text === undefined ? [] : [text];
      });
    writePasteDraft(pasteDraftKey, pastes);
  }, [composerAttachmentSignature, pasteDraftKey, aui]);
  useEffect(() => {
    // After a thread switch composerText can still hold the previous
    // thread's text; skip that cycle so it isn't saved under the new key.
    if (lastDraftKeyRef.current !== draftKey) {
      lastDraftKeyRef.current = draftKey;
      return;
    }
    if (!draftKey) {
      return;
    }
    const t = setTimeout(() => writeComposerDraft(draftKey, composerText), 300);
    draftSaveTimerRef.current = t;
    return () => clearTimeout(t);
  }, [composerText, draftKey]);
  const pasteDraftKeyRef = useRef(pasteDraftKey);
  useEffect(() => {
    draftKeyRef.current = draftKey;
    pasteDraftKeyRef.current = pasteDraftKey;
  }, [draftKey, pasteDraftKey]);
  // Call wherever the composer is emptied because its text left as a message.
  const armJustSent = useCallback((...texts: string[]) => {
    justSentRef.current = armSentTextGuard(texts, draftKeyRef.current);
  }, []);
  const clearStoredDraft = useCallback(() => {
    if (draftSaveTimerRef.current !== null) {
      clearTimeout(draftSaveTimerRef.current);
      draftSaveTimerRef.current = null;
    }
    const key = draftKeyRef.current;
    if (key) {
      writeComposerDraft(key, "");
    }
    const pasteKey = pasteDraftKeyRef.current;
    if (pasteKey) {
      writePasteDraft(pasteKey, []);
    }
  }, []);
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
  const indexingActiveRef = useRef(false);
  const promptQueueTargetMountedRef = useRef(true);
  const promptQueueStartPendingRef = useRef(
    new Map<
      string,
      {
        temporary: boolean;
        cancelled: boolean;
        threadId: string | null;
        localModelBoundaryGeneration: number;
        queuedSettingsEpoch: number;
      }
    >(),
  );
  // Reading a pasted-text attachment happens before the queue start is
  // registered, so the intent is recorded here for the length of the read.
  // Keyed like a reservation so a submit during the read cannot start a second
  // read of the same attachment, and carrying the boundaries the read predates.
  const pastedTextQueuePendingRef = useRef(
    new Map<
      string,
      {
        temporary: boolean;
        cancelled: boolean;
        threadId: string | null;
        localModelBoundaryGeneration: number;
        queuedSettingsEpoch: number;
        historyClearGeneration: number;
      }
    >(),
  );
  useEffect(() => {
    promptQueueTargetMountedRef.current = true;
    return () => {
      promptQueueTargetMountedRef.current = false;
    };
  }, []);
  useEffect(() => {
    const cancelPendingQueueFactories = (event: Event) => {
      const detail =
        (event as CustomEvent<PromptQueueStopEventDetail>).detail ?? {};
      const state = aui.threadListItem().getState();
      const aliases = compactIds([state.id, state.remoteId, referenceThreadId]);
      cancelPendingPromptQueueFactoriesForStop(
        promptQueueStartPendingRef.current,
        aliases,
        detail,
      );
      // A read in flight has no reservation yet, so it needs cancelling here
      // too or Clear all lets it queue a prompt and recreate the chat.
      cancelPendingPromptQueueFactoriesForStop(
        pastedTextQueuePendingRef.current,
        aliases,
        detail,
      );
    };
    window.addEventListener(
      PROMPT_QUEUE_STOP_EVENT,
      cancelPendingQueueFactories,
    );
    return () => {
      window.removeEventListener(
        PROMPT_QUEUE_STOP_EVENT,
        cancelPendingQueueFactories,
      );
    };
  }, [aui, referenceThreadId]);
  const [pendingSend, setPendingSend] = useState(false);
  const pendingSendRef = useRef(false);
  // Whether the parked send is a queue gesture. A chord pressed while this
  // chat's settings load parks like any other send, and the release path would
  // otherwise send the prompt the user asked to stack.
  const pendingSendForceQueueRef = useRef(false);
  const waitToastRef = useRef<string | number | null>(null);
  // This chat's own settings are still on their way; a send now would run on the
  // installation defaults showing in their place.
  const threadScopedSettingsPending = useChatRuntimeStore(
    (s) => s.threadScopedSettingsPending,
  );

  const handleIndexingChange = useCallback((active: boolean) => {
    indexingActiveRef.current = active;
    setIndexingActive(active);
  }, []);

  const createPromptQueueTarget = useCallback(async (): Promise<PromptQueueTarget | null> => {
    const assistantRuntime = aui.threads().__internal_getAssistantRuntime?.();
    const initialState = aui.threadListItem().getState();
    const initialRunningThreadIds = [
      initialState.id,
      initialState.remoteId,
      referenceThreadId,
    ].filter((id): id is string => Boolean(id));
    const initialDocumentThreadId =
      initialState.remoteId ?? referenceThreadId ?? null;
    const historyClearGeneration = chatHistoryClearBoundary.capture();
    await useChatRuntimeStore.getState().hydratePersistedSettings();
    if (
      !promptQueueTargetMountedRef.current ||
      chatHistoryClearBoundary.capture() !== historyClearGeneration
    ) {
      return null;
    }
    const currentState = aui.threadListItem().getState();
    if (
      !compactIds([currentState.id, currentState.remoteId]).some((id) =>
        initialRunningThreadIds.includes(id),
      )
    ) {
      return null;
    }
    const chatStateAtQueueStart = useChatRuntimeStore.getState();
    const incognitoAtQueueStart = chatStateAtQueueStart.incognito;
    // A chat with no row yet has no project to look up, and the store holds
    // whichever project is on screen when the queue polls. Read it here.
    const projectIdAtQueueStart = incognitoAtQueueStart
      ? null
      : (chatStateAtQueueStart.activeProjectId ?? null);
    const usesThreadDocumentsAtQueueStart =
      chatStateAtQueueStart.ragEnabled &&
      chatStateAtQueueStart.ragSource.type === "thread";
    const usesKnowledgeBaseAtQueueStart =
      chatStateAtQueueStart.ragEnabled &&
      chatStateAtQueueStart.ragSource.type === "kb";
    const runSettingsAtQueueStart =
      snapshotQueuedChatRunSettings(chatStateAtQueueStart);
    const getThreadListItemState = () => {
      const runtime =
        assistantRuntime ?? aui.threads().__internal_getAssistantRuntime?.();
      if (!runtime) {
        return null;
      }
      for (const id of initialRunningThreadIds) {
        try {
          return runtime.threads.getItemById(id).getState();
        } catch {
          // Try the next captured id.
        }
      }
      return null;
    };
    const getQueueThreadIds = () => {
      const state = getThreadListItemState();
      return compactIds([
        ...initialRunningThreadIds,
        state?.id,
        state?.remoteId,
      ]);
    };
    const getThreadRuntime = () => {
      const runtime =
        assistantRuntime ?? aui.threads().__internal_getAssistantRuntime?.();
      if (!runtime) {
        return null;
      }
      for (const id of getQueueThreadIds()) {
        try {
          const thread = runtime.threads.getById(id);
          thread.getState();
          return thread;
        } catch {
          // Try the next captured id.
        }
      }
      return null;
    };
    const isTargetCurrentThread = () => {
      const state = aui.threadListItem().getState();
      return compactIds([state.id, state.remoteId]).some((id) =>
        initialRunningThreadIds.includes(id),
      );
    };
    const pendingSettingsIds = new Set<number>();
    let cancelled = false;
    let shouldCorrectPersistedModel: boolean | null = null;
    let initializedFreshThreadId: string | null = null;
    let freshThreadAppendAccepted = false;
    const removeFreshThreadPersistedAfterAbort = () => {
      const historyWasCleared =
        chatHistoryClearBoundary.capture() !== historyClearGeneration;
      if (
        !initializedFreshThreadId ||
        freshThreadAppendAccepted ||
        (!cancelled && !historyWasCleared)
      ) {
        return false;
      }
      // Tombstone synchronously so a late initializer cannot leave an empty
      // record visible while backend cleanup completes.
      markChatThreadDeleted(initializedFreshThreadId);
      // the tombstone is never rolled back: a failed DELETE may still have committed, and the
      // backend tombstones on commit, so resurrecting the id would leave it 410 on every write
      void deleteStoredChatThreads([initializedFreshThreadId]).catch(
        () => undefined,
      );
      if (!historyWasCleared && isTargetCurrentThread()) {
        void Promise.resolve(aui.threads().switchToNewThread()).catch(
          () => undefined,
        );
      }
      return true;
    };
    const discardOldestPendingSettings = () => {
      const settingsId = pendingSettingsIds.values().next().value;
      if (settingsId === undefined) {
        return;
      }
      pendingSettingsIds.delete(settingsId);
      discardQueuedChatRunSettings(settingsId);
    };
    return {
      getDocumentThreadId: () => {
        const state = getThreadListItemState();
        return state?.remoteId ?? referenceThreadId ?? initialDocumentThreadId;
      },
      getRunningThreadIds: () => {
        return getQueueThreadIds();
      },
      isRunning: () =>
        hasPreStreamRunReservation(getQueueThreadIds()) ||
        Boolean(getThreadRuntime()?.getState().isRunning),
      append: async (prompt) => {
        const thread = getThreadRuntime();
        if (!thread) {
          throw new Error("Prompt queue thread runtime is unavailable");
        }
        if (incognitoAtQueueStart) {
          for (const id of getQueueThreadIds()) {
            markThreadIncognito(id);
          }
        }
        const settingsId = registerQueuedChatRunSettings(
          getQueueThreadIds(),
          {
            ...runSettingsAtQueueStart,
            params: { ...runSettingsAtQueueStart.params },
          },
        );
        pendingSettingsIds.add(settingsId);
        try {
          const runtime =
            assistantRuntime ?? aui.threads().__internal_getAssistantRuntime?.();
          const state = getThreadListItemState();
          if (!runtime || !state) {
            throw new Error("Prompt queue thread item is unavailable");
          }
          if (chatHistoryClearBoundary.capture() !== historyClearGeneration) {
            return;
          }
          shouldCorrectPersistedModel ??= !state.remoteId;
          const initializingFreshThread = !state.remoteId;
          // Stamp it with what the queue was STARTED under. This path initializes without
          // going through the composer, and by dispatch time the adapter may be showing a
          // different project, so the chat was filed wherever the user is now.
          if (initializingFreshThread) {
            claimThreadCreation([state.id, state.remoteId], {
              projectId: projectIdAtQueueStart,
              incognito: incognitoAtQueueStart,
              modelId: runSettingsAtQueueStart.params.checkpoint ?? "",
              modelGgufVariant: runSettingsAtQueueStart.activeGgufVariant,
              createdAt: Date.now(),
            });
          }
          // A fresh chat receives its remote id during initialization. Await it
          // before append so the adapter can match the queued settings using
          // unstable_threadId on its first invocation.
          const { remoteId } = await runtime.threads
            .getItemById(state.id)
            .initialize();
          if (initializingFreshThread) {
            initializedFreshThreadId = remoteId;
          }
          if (
            removeFreshThreadPersistedAfterAbort() ||
            cancelled ||
            !pendingSettingsIds.has(settingsId)
          ) {
            return;
          }
          addQueuedChatRunSettingsThreadIds(settingsId, [
            ...getQueueThreadIds(),
            remoteId,
          ]);
          if (shouldCorrectPersistedModel) {
            // initialize() persists a fresh thread using the live global model.
            // Correct that metadata to the model captured for this queued run
            // before any later navigation or compatibility check can observe it.
            await updateStoredChatThread(remoteId, {
              modelId: runSettingsAtQueueStart.params.checkpoint ?? "",
              modelGgufVariant: runSettingsAtQueueStart.activeGgufVariant,
            });
            shouldCorrectPersistedModel = false;
            if (
              removeFreshThreadPersistedAfterAbort() ||
              cancelled ||
              !pendingSettingsIds.has(settingsId)
            ) {
              return;
            }
          }
          // Initialization can replace a fresh thread's local id with a remote
          // id. Refresh queue aliases before the run begins so stop dialogs
          // deduplicate the two identities.
          syncPromptQueueUI();
          const appendResult = thread.append(
            appendTextToThread(prompt),
          ) as unknown;
          freshThreadAppendAccepted = true;
          // Calling append synchronously accepts the user turn; its promise
          // follows the whole provider run. Do not turn a later paid/streaming
          // failure into an automatic duplicate dispatch.
          if (
            appendResult &&
            typeof (appendResult as Promise<void>).catch === "function"
          ) {
            void (appendResult as Promise<void>).catch(() => undefined);
          }
        } catch (error) {
          // A setup failure is retryable. Keep the initialized record unless a
          // concurrent stop or Clear all explicitly invalidated this queue.
          removeFreshThreadPersistedAfterAbort();
          pendingSettingsIds.delete(settingsId);
          discardQueuedChatRunSettings(settingsId);
          throw error;
        }
      },
      complete: discardOldestPendingSettings,
      cancel: () => {
        cancelled = true;
        removeFreshThreadPersistedAfterAbort();
        for (const settingsId of pendingSettingsIds) {
          discardQueuedChatRunSettings(settingsId);
        }
        pendingSettingsIds.clear();
        getThreadRuntime()?.cancelRun();
      },
      isIndexing: () =>
        promptQueueTargetMountedRef.current &&
        isTargetCurrentThread() &&
        indexingActiveRef.current,
      getQueueProjectId: () => projectIdAtQueueStart,
      usesThreadDocuments: usesThreadDocumentsAtQueueStart,
      usesKnowledgeBase: usesKnowledgeBaseAtQueueStart,
      usesLocalModel:
        parseExternalModelId(runSettingsAtQueueStart.params.checkpoint) === null,
      usesDeepResearch: runSettingsAtQueueStart.deepResearchEnabled,
      researchStarted: () => {
        const claimed = useResearchRunStore.getState().claimedThreadIds;
        return getQueueThreadIds().some((id) => Boolean(claimed[id]));
      },
      temporary: incognitoAtQueueStart,
      consumeDeepResearch: () => {
        runSettingsAtQueueStart.deepResearchEnabled = false;
      },
    };
  }, [aui, referenceThreadId]);

  // Whether a pending start is already going to be refused when it resolves,
  // so a retry replaces it rather than being turned away as a duplicate and
  // leaving neither gesture to queue anything. Only the checks that need no
  // queue target are here; the model boundary stays with the reservation,
  // where usesLocalModel is known, so this can never be the stricter of the
  // two and start a second queue for the same prompt.
  const pendingQueueStartIsStale = useCallback(
    (pending: {
      cancelled: boolean;
      temporary: boolean;
      queuedSettingsEpoch: number;
      historyClearGeneration?: number;
    }): boolean => {
      if (pending.cancelled) return true;
      if (
        pending.historyClearGeneration !== undefined &&
        chatHistoryClearBoundary.capture() !== pending.historyClearGeneration
      ) {
        return true;
      }
      const chatState = useChatRuntimeStore.getState();
      return shouldAbortPendingQueueForSettingsChange({
        capturedEpoch: pending.queuedSettingsEpoch,
        currentEpoch: chatState.queuedSettingsEpoch,
        capturedTemporary: pending.temporary,
        currentTemporary: chatState.incognito,
      });
    },
    [],
  );

  const startHydratedPromptQueue = useCallback(
    (
      items: string[],
      waitForCurrentRun = false,
      onStarted?: () => void,
      onAborted?: () => void,
      // Captured before an awaited step that precedes this call, so a boundary
      // or setting changed during that step still invalidates the queue.
      capturedAt?: {
        localModelBoundaryGeneration: number;
        queuedSettingsEpoch: number;
        temporary: boolean;
      },
    ) => {
      const reservationKey = JSON.stringify([
        referenceThreadId,
        items,
        waitForCurrentRun,
      ]);
      // A reservation that is still going to start owns this prompt. One that
      // is already invalid is replaced, so the retry is the one that queues.
      const existing = promptQueueStartPendingRef.current.get(reservationKey);
      if (existing && !pendingQueueStartIsStale(existing)) {
        return false;
      }
      const reservation = {
        temporary:
          capturedAt?.temporary ?? useChatRuntimeStore.getState().incognito,
        cancelled: false,
        threadId: referenceThreadId,
        localModelBoundaryGeneration:
          capturedAt?.localModelBoundaryGeneration ??
          localPromptQueueModelBoundary.capture(),
        queuedSettingsEpoch:
          capturedAt?.queuedSettingsEpoch ??
          useChatRuntimeStore.getState().queuedSettingsEpoch,
      };
      promptQueueStartPendingRef.current.set(reservationKey, reservation);
      void createPromptQueueTarget()
        .then((target) => {
          const currentQueueSettings = useChatRuntimeStore.getState();
          const modelBoundaryInvalidated = target
            ? shouldAbortPendingQueueForModelBoundary({
                capturedGeneration:
                  reservation.localModelBoundaryGeneration,
                usesLocalModel: target.usesLocalModel,
                modelLoading: currentQueueSettings.modelLoading,
              })
            : false;
          const settingsInvalidated =
            shouldAbortPendingQueueForSettingsChange({
              capturedEpoch: reservation.queuedSettingsEpoch,
              currentEpoch: currentQueueSettings.queuedSettingsEpoch,
              capturedTemporary: reservation.temporary,
              currentTemporary: currentQueueSettings.incognito,
            });
          if (
            target &&
            !reservation.cancelled &&
            !modelBoundaryInvalidated &&
            !settingsInvalidated &&
            promptQueueStartPendingRef.current.get(reservationKey) ===
              reservation
          ) {
            startPromptQueue(items, target, waitForCurrentRun);
            onStarted?.();
          } else if (
            promptQueueStartPendingRef.current.get(reservationKey) ===
            reservation
          ) {
            // Superseded reservations stay quiet: the one that replaced this
            // is still going, so nothing has been lost to report.
            onAborted?.();
          }
        })
        .catch((error) => {
          toast.error("Could not start prompt queue", {
            description:
              error instanceof Error ? error.message : "Please try again.",
          });
          onAborted?.();
        })
        .finally(() => {
          if (
            promptQueueStartPendingRef.current.get(reservationKey) ===
            reservation
          ) {
            promptQueueStartPendingRef.current.delete(reservationKey);
          }
        });
      return true;
    },
    [createPromptQueueTarget, pendingQueueStartIsStale, referenceThreadId],
  );

  // The queue carries text, and a long paste is text the composer parked in a
  // chip, so fold it back in rather than refusing to queue it as a file.
  const queuePastedTextPrompt = useCallback(
    (waitForCurrentRun: boolean): boolean => {
      const composer = aui.composer();
      const attachments = composer.getState().attachments;
      const files: File[] = [];
      for (const attachment of attachments) {
        const file = (attachment as { file?: File }).file;
        if (file === undefined || !isPastedTextFile(file)) return false;
        files.push(file);
      }
      if (files.length === 0) return false;

      const attachmentIds = attachments.map((attachment) => attachment.id);
      const textAtQueue = composer.getState().text.trim();
      const queueTexts = (
        texts: string[],
        // Captured before an awaited read, when there was one.
        capturedAt?: {
          localModelBoundaryGeneration: number;
          queuedSettingsEpoch: number;
          temporary: boolean;
        },
      ) => {
        const queuedPrompt = [textAtQueue, ...texts]
          .filter((part) => part.trim().length > 0)
          .join("\n\n");
        if (queuedPrompt.length === 0) return;
        startHydratedPromptQueue(
          [queuedPrompt],
          waitForCurrentRun,
          () => {
            const state = composer.getState();
            // Only clear the composer this prompt was queued from.
            if (
              state.text.trim() !== textAtQueue ||
              state.attachments.length !== attachmentIds.length ||
              !state.attachments.every(
                (attachment, index) => attachment.id === attachmentIds[index],
              )
            ) {
              return;
            }
            void composer.clearAttachments();
            flushResourcesSync(() => {
              composer.setText("");
            });
            clearStoredDraft();
            armJustSent(state.text);
          },
          () => {
            toast.info("Pasted text was not queued", {
              description: "The chat settings changed. Send it again.",
            });
          },
          capturedAt,
        );
      };

      // createPastedTextFile records the body under the File identity matched
      // above, so read it from there: a gesture that awaits the File joins the
      // queue behind any later one that does not, reversing the two.
      const cachedTexts: string[] = [];
      for (const file of files) {
        const text = pastedTextOf(file);
        if (text === undefined) break;
        cachedTexts.push(text);
      }
      if (cachedTexts.length === files.length) {
        queueTexts(cachedTexts);
        return true;
      }

      // Registered before the read, or a submit during it takes the send path
      // and this queues the same text again once the read finishes.
      const pendingKey = pastedTextQueueKey(
        referenceThreadId,
        textAtQueue,
        attachmentIds,
      );
      // The same intent as a read already running: report it handled rather
      // than queue a duplicate. A read whose baselines have gone stale will
      // abort, so it must not absorb the retry either.
      const inFlight = pastedTextQueuePendingRef.current.get(pendingKey);
      if (inFlight && !pendingQueueStartIsStale(inFlight)) return true;
      // Every baseline the reservation would otherwise take after the read, so
      // a setting or boundary changed during it still aborts the queue.
      const chatState = useChatRuntimeStore.getState();
      const pendingRead = {
        temporary: chatState.incognito,
        cancelled: false,
        threadId: referenceThreadId,
        // The chat the read began in. The target is anchored after the read,
        // so a switch mid-read would otherwise dispatch into the new chat.
        composerIdentity: composerIdentityRef.current,
        localModelBoundaryGeneration: localPromptQueueModelBoundary.capture(),
        queuedSettingsEpoch: chatState.queuedSettingsEpoch,
        historyClearGeneration: chatHistoryClearBoundary.capture(),
      };
      // Replaces a stale read under the same key. That read still resolves, but
      // it no longer owns the key, so its own start is skipped and only this
      // one can queue.
      pastedTextQueuePendingRef.current.set(pendingKey, pendingRead);
      void Promise.all(files.map((file) => file.text()))
        .then((texts) => {
          // Stopped, cleared, replaced, or aimed at another chat while the
          // read was in flight.
          if (
            pendingQueueStartIsStale(pendingRead) ||
            composerIdentityRef.current !== pendingRead.composerIdentity ||
            pastedTextQueuePendingRef.current.get(pendingKey) !== pendingRead
          ) {
            return;
          }
          queueTexts(texts, pendingRead);
        })
        .catch(() => {
          toast.error("Could not queue the pasted text.", {
            description: "Show it in the text field, then send it again.",
          });
        })
        .finally(() => {
          if (pastedTextQueuePendingRef.current.get(pendingKey) === pendingRead) {
            pastedTextQueuePendingRef.current.delete(pendingKey);
          }
        });
      return true;
    },
    [
      armJustSent,
      aui,
      clearStoredDraft,
      pendingQueueStartIsStale,
      referenceThreadId,
      startHydratedPromptQueue,
    ],
  );

  // Queue whatever the composer holds. Hoisted out of handleSubmit because the
  // parked-send release needs it too and cannot reach that closure. Reads the
  // live composer, not the rendered text, which at release time can be a commit
  // behind.
  const queueComposerText = useCallback(
    (waitForCurrentRun: boolean) => {
      const queuedPrompt = aui.composer().getState().text.trim();
      if (!queuedPrompt) {
        return;
      }
      startHydratedPromptQueue([queuedPrompt], waitForCurrentRun, () => {
        // Guard the untrimmed text too: that is what a late write carries.
        const cleared = aui.composer().getState().text;
        if (cleared.trim() !== queuedPrompt) {
          return;
        }
        flushResourcesSync(() => {
          aui.composer().setText("");
        });
        clearStoredDraft();
        armJustSent(queuedPrompt, cleared);
      });
    },
    [armJustSent, aui, clearStoredDraft, startHydratedPromptQueue],
  );

  const dismissWaitToast = useCallback(() => {
    if (waitToastRef.current !== null) {
      toast.dismiss(waitToastRef.current);
      waitToastRef.current = null;
    }
  }, []);

  // Declared here because cancelQueuedSend has to clear the dictation hold too.
  const sendAfterDictationRef = useRef(false);
  // Composer text while a send waits on dictationBlocked, so an edit can drop it.
  const heldTextRef = useRef<string | null>(null);

  const cancelQueuedSend = useCallback(() => {
    pendingSendRef.current = false;
    pendingSendForceQueueRef.current = false;
    setPendingSend(false);
    // A dictation send held behind the same block would otherwise fire alone.
    sendAfterDictationRef.current = false;
    heldTextRef.current = null;
    dismissWaitToast();
  }, [dismissWaitToast]);
  cancelQueuedSendRef.current = cancelQueuedSend;

  const enqueueSend = useCallback(
    (
      waitingOn:
        | "indexing"
        | "images"
        | "audio"
        | "video"
        | "settings" = "indexing",
    ) => {
      if (pendingSendRef.current) return;
      pendingSendRef.current = true;
      setPendingSend(true);
      const title =
        waitingOn === "images"
          ? "Waiting for dropped images"
          : waitingOn === "audio"
            ? "Waiting for dropped audio"
            : waitingOn === "video"
              ? "Waiting for dropped video"
              : waitingOn === "settings"
                ? "Loading this chat's settings"
                : "Waiting for documents to finish indexing";
      waitToastRef.current = toast(title, {
        description: "Your message will send automatically once they are ready.",
        duration: Infinity,
        cancel: { label: "Cancel", onClick: cancelQueuedSend },
      });
    },
    [cancelQueuedSend],
  );

  // A materializing image or clip is a wait, not a refusal: park the send.
  // Both gates share this so they cannot disagree on what is recoverable.
  const parkIfWaitingOnAttachments = useCallback(() => {
    if (
      disabled ||
      overlay ||
      (!hasMaterializingImageAttachments &&
        !hasMaterializingAudioAttachments &&
        !hasMaterializingVideoAttachments) ||
      !hasSendableContent ||
      isComposingRef.current ||
      hasPendingAttachments
    ) {
      return;
    }
    // Name what is actually being waited on, or a parked video drop reports
    // itself as audio.
    enqueueSend(
      hasMaterializingImageAttachments
        ? "images"
        : hasMaterializingAudioAttachments
          ? "audio"
          : "video",
    );
  }, [
    disabled,
    overlay,
    hasMaterializingImageAttachments,
    hasMaterializingAudioAttachments,
    hasMaterializingVideoAttachments,
    hasSendableContent,
    hasPendingAttachments,
    isComposingRef,
    enqueueSend,
  ]);

  const shouldBlockSend = useCallback(
    () =>
      !hasSendableContent ||
      isComposingRef.current ||
      hasPendingAttachments ||
      hasMaterializingImageAttachments ||
      hasMaterializingAudioAttachments ||
      hasMaterializingVideoAttachments,
    [
      hasMaterializingAudioAttachments,
      hasMaterializingVideoAttachments,
      hasMaterializingImageAttachments,
      hasPendingAttachments,
      hasSendableContent,
      isComposingRef,
    ],
  );

  // alsoGuard: text the composer showed before this path rewrote it, so a late
  // write carrying what the user actually typed is refused too.
  const sendReservedComposer = useCallback((...alsoGuard: string[]) => {
    const assistantRuntime =
      aui.threads().__internal_getAssistantRuntime?.();
    let reservationToken: symbol | null = null;
    reservationToken = reservePreStreamRun(preStreamThreadIds, {
      usesLocalModel:
        parseExternalModelId(
          useChatRuntimeStore.getState().params.checkpoint,
        ) === null,
      cancel: (reservedThreadIds) => {
        if (preStreamRunReservationRef.current === reservationToken) {
          preStreamRunReservationRef.current = null;
        }
        for (const reservedThreadId of reservedThreadIds) {
          try {
            assistantRuntime?.threads.getById(reservedThreadId).cancelRun();
            return;
          } catch {
            // Thread hydration can retire an alias; try the next captured id.
          }
        }
      },
    });
    if (!reservationToken) {
      toast.error("Wait for the current response to finish");
      return;
    }
    preStreamRunReservationRef.current = reservationToken;
    try {
      const sentText = aui.composer().getState().text;
      // Stamp the send BEFORE send() starts awaiting every incomplete attachment: a document
      // send reaches initialize() seconds later, by which time navigation may have moved the
      // project and cleared the temporary flag. See utils/chat-thread-creation-claim.ts.
      const chatStateAtSend = useChatRuntimeStore.getState();
      claimThreadCreation(preStreamThreadIds, {
        projectId: projectScope,
        incognito: chatStateAtSend.incognito,
        modelId: chatStateAtSend.params.checkpoint ?? "",
        modelGgufVariant: chatStateAtSend.activeGgufVariant,
        createdAt: Date.now(),
      });
      aui.composer().send();
      // Empty texts are dropped, so an attachment-only send still clears.
      armJustSent(sentText, ...alsoGuard);
    } catch (error) {
      if (releasePreStreamRunReservation(reservationToken)) {
        notifyPromptQueueRunFailed(referenceThreadId);
      }
      preStreamRunReservationRef.current = null;
      toast.error("Could not prepare attachments", {
        description:
          error instanceof Error ? error.message : "Please retry the send.",
      });
    }
  }, [aui, armJustSent, preStreamThreadIds, projectScope, referenceThreadId]);

  // Gate for both form submit and the Send button. Returns true when it handled
  // the event (blocked or queued) so callers stop.
  const interceptSend = useCallback(
    (event: { preventDefault: () => void }) => {
      if (disabled || shouldBlockSend()) {
        event.preventDefault();
        parkIfWaitingOnAttachments();
        return true;
      }
      if (indexingActive && !overlay) {
        event.preventDefault();
        enqueueSend();
        return true;
      }
      // This chat's own settings have been asked for and have not arrived, so the store
      // is showing the installation defaults and the run would be captured with them:
      // a chat stored as "ask" could run tools without asking. Park it like any other
      // wait, so the click still counts and the send fires once the snapshot lands.
      if (threadScopedSettingsPending && !overlay) {
        event.preventDefault();
        enqueueSend("settings");
        return true;
      }
      return false;
    },
    [
      disabled,
      shouldBlockSend,
      indexingActive,
      overlay,
      threadScopedSettingsPending,
      enqueueSend,
      parkIfWaitingOnAttachments,
    ],
  );

  // Fire the parked send once indexing clears, unless the user emptied the
  // composer while waiting (then drop it quietly). An image dropped after the
  // send was parked has to land first, or indexing finishing early sends the
  // text without it and the image attaches to the next draft.
  useEffect(() => {
    // pendingSendRef too: a cancel earlier in this same commit has already
    // dropped the send, while `pendingSend` still reads true from this render.
    if (
      !pendingSend ||
      !pendingSendRef.current ||
      indexingActive ||
      threadScopedSettingsPending ||
      hasMaterializingImageAttachments ||
      hasMaterializingAudioAttachments ||
      hasMaterializingVideoAttachments
    ) {
      return;
    }
    const { text, attachments } = aui.composer().getState();
    const forceQueue = pendingSendForceQueueRef.current;
    pendingSendRef.current = false;
    pendingSendForceQueueRef.current = false;
    setPendingSend(false);
    dismissWaitToast();
    if (text.trim().length > 0 || attachments.length > 0) {
      clearStoredDraft();
      if (forceQueue) {
        // Wait mode read now, not carried from the parked submit: a run can
        // start while the settings load, and ignoring it would dispatch on top
        // of the response already streaming.
        const waitForCurrentRun = aui.thread().getState().isRunning;
        // The chord's own two branches from handleSubmit, in the same order. A
        // long paste lives in an attachment, so queueing the text alone queues
        // nothing at all when that is all there is.
        if (canQueueCurrentPrompt) {
          queueComposerText(waitForCurrentRun);
          return;
        }
        if (canQueuePastedTextPrompt && queuePastedTextPrompt(waitForCurrentRun)) {
          return;
        }
        // Nothing queueable: send, as this path did before it carried intent.
      }
      sendReservedComposer();
    }
  }, [
    pendingSend,
    indexingActive,
    threadScopedSettingsPending,
    hasMaterializingImageAttachments,
    hasMaterializingAudioAttachments,
    hasMaterializingVideoAttachments,
    aui,
    canQueueCurrentPrompt,
    canQueuePastedTextPrompt,
    clearStoredDraft,
    dismissWaitToast,
    queueComposerText,
    queuePastedTextPrompt,
    sendReservedComposer,
  ]);

  // Drop any queued send + toast on unmount (e.g. thread switch).
  useEffect(
    () => () => {
      pendingSendRef.current = false;
      pendingSendForceQueueRef.current = false;
      if (waitToastRef.current !== null) toast.dismiss(waitToastRef.current);
    },
    [],
  );

  // Recording bar's send: stop dictating, then submit once the transcript
  // lands. Going through the form keeps queueing, indexing holds and draft
  // clearing identical to a typed send.
  const formRef = useRef<HTMLFormElement | null>(null);
  // Mirrored into state so the publish effect re-runs when the node mounts: a
  // ref mutation does not re-render. See usePublishedFrame.
  const [composerEl, setComposerEl] = useState<HTMLFormElement | null>(null);
  const attachComposer = useCallback((node: HTMLFormElement | null) => {
    formRef.current = node;
    setComposerEl(node);
  }, []);
  // Docked under a thread, the composer sits in the corner the API monitor
  // panel opens in. Published so that panel opens clear of Send. The
  // notification rail does not read this; it is anchored in CSS.
  usePublishedFrame(composerEl);
  const dictationBaseTextRef = useRef("");
  const dictationComposerRef = useRef("");
  // Thread switches reuse this composer, so the send has to know where it
  // started to avoid submitting the destination thread's draft. The list item
  // id, not referenceThreadId: that one moves from null to the remote id when
  // a new chat first persists, which is the same composer.
  const composerIdentity = threadListItemId ?? "";
  composerIdentityRef.current = composerIdentity;
  // Keep the mic clickable: if the engine can't run here, explain and point to
  // the local model instead of disabling the button.
  const startDictation = useCallback(() => {
    if (!isStudioDictationAvailable()) {
      notifyStudioDictationUnavailable();
      return;
    }
    try {
      aui.composer().startDictation();
    } catch {
      notifyStudioDictationUnavailable();
    }
  }, [aui]);
  const sendAfterDictation = useCallback(() => {
    sendAfterDictationRef.current = true;
    dictationComposerRef.current = composerIdentity;
    aui.composer().stopDictation();
  }, [aui, composerIdentity]);

  // One gate for the recording bar's send: it greys the button out, and holds
  // a pending send when the composer changes under it after the press.
  const dictationBlocked = dictationSendBlocked({
    composerDisabled: Boolean(disabled),
    uploading:
      hasPendingAttachments ||
      hasMaterializingImageAttachments ||
      hasMaterializingAudioAttachments ||
      hasMaterializingVideoAttachments,
    researchActive: isResearchActive,
    runActive: threadIsRunning || promptQueueActive,
    queueDisabled: Boolean(disableQueue),
    hasOverlay: Boolean(overlay),
    hasAttachments,
    hasPendingAudio,
  });
  // Both chords live here, not with the controls below: the recording bar
  // replaces those while dictation runs, so a chord registered there could
  // start dictation and never stop it.
  const chatActive = useChatActive();
  useShortcut(
    "startDictation",
    () => {
      // Stopping first and ungated: the recording bar replaces the input, so
      // the gate's selector is gone for exactly as long as there is something
      // to stop.
      if (isDictating) {
        aui.composer().stopDictation();
        return;
      }
      // A dialog over Chat leaves this registered, and a microphone opened
      // behind one is neither visible nor stoppable from where the user is.
      if (!isSurfaceInForeground(COMPOSER_INPUT_SELECTOR)) return;
      startDictation();
    },
    { enabled: chatActive },
  );
  useShortcut(
    "sendMessage",
    () => {
      // While recording, the bar's own send: it stops dictation first and lets
      // the final transcript land, where submitting here would send the text
      // so far and leave the rest of the sentence in an empty composer.
      if (isDictating) {
        if (!dictationBlocked) sendAfterDictation();
        return;
      }
      // A dialog over Chat leaves this registered, and the draft behind it is
      // not what the user is typing. Sending is not undoable, so it asks here.
      if (!isSurfaceInForeground(COMPOSER_INPUT_SELECTOR)) return;
      // requestSubmit, not the runtime's send: it runs handleSubmit first,
      // which parks a send behind indexing, queues it behind a run, or
      // refuses it.
      formRef.current?.requestSubmit();
    },
    {
      enabled: chatActive && !disabled,
      // The model picker is a non-modal popover, so the composer stays the
      // foreground while its search box has focus. Every text field but the
      // composer keeps this chord.
      skipInTextFields: true,
      textFieldException: COMPOSER_INPUT_SELECTOR,
    },
  );
  const wasDictatingRef = useRef(false);
  useEffect(() => {
    if (isDictating) {
      if (wasDictatingRef.current) return;
      wasDictatingRef.current = true;
      // A new recording supersedes a send still held for an upload.
      sendAfterDictationRef.current = false;
      heldTextRef.current = null;
      // Text at session start is the dictation base. Anchor on it, not on the
      // text when send was pressed: the browser engine streams interim results
      // into the composer, so a final matching its interim would look unchanged.
      dictationBaseTextRef.current = aui.composer().getState().text;
      return;
    }
    wasDictatingRef.current = false;
    if (!sendAfterDictationRef.current) return;
    // A partial transcript (a failed chunk, or an engine error after one
    // landed) belongs in the composer, but must not send half a message.
    // Silence, a thread switch mid-transcription, or a plus-menu insertion
    // with no speech: keep the draft, submit nothing. Settled before the hold
    // below, so nothing to send never leaves an intent pending.
    const text = composerText;
    const sendable =
      !dictationFailed() &&
      shouldSubmitDictation({
        originComposer: dictationComposerRef.current,
        currentComposer: composerIdentity,
        producedTranscript: dictationProducedTranscript(),
        baseText: dictationBaseTextRef.current,
        text,
      });
    if (!sendable) {
      sendAfterDictationRef.current = false;
      heldTextRef.current = null;
      return;
    }
    // The plus stays live while transcribing, so an upload or an attachment
    // can appear after the press. Keep the intent until the composer accepts
    // a submit again, rather than spending it on one that would bounce.
    if (dictationBlocked) {
      // The bar is gone by now, so the hold is invisible. It lasts only as
      // long as the transcript it was pressed for: editing hands control
      // back, rather than sending that edit when the block clears.
      if (heldTextRef.current === null) {
        heldTextRef.current = text;
      } else if (heldTextRef.current !== text) {
        sendAfterDictationRef.current = false;
        heldTextRef.current = null;
      }
      return;
    }
    sendAfterDictationRef.current = false;
    heldTextRef.current = null;
    formRef.current?.requestSubmit();
  }, [isDictating, aui, composerIdentity, dictationBlocked, composerText]);

  const handleSubmit = useCallback(
    (event: {
      preventDefault: () => void;
      stopPropagation?: () => void;
    }) => {
      // Read once per submit: a rejected send must not leave it armed.
      const forceQueue = forceQueueRef.current;
      forceQueueRef.current = false;
      if (isResearchActive) {
        event.preventDefault();
        return;
      }
      if (disabled || shouldBlockSend()) {
        event.preventDefault();
        parkIfWaitingOnAttachments();
        return;
      }
      // Before the queue branch below, not after it: a prompt queued while this chat's
      // own settings are still on their way is snapshotted from the installation
      // defaults on screen, so a chat stored as "ask" would queue as "off".
      if (threadScopedSettingsPending && !overlay) {
        event.preventDefault();
        // The intent rides with the parked send; the release reads it back.
        if (forceQueue) {
          pendingSendForceQueueRef.current = true;
        }
        enqueueSend("settings");
        return;
      }

      // React may not have rendered threadIsRunning yet when several submits
      // arrive immediately after a send. The imperative runtime is already
      // current, so use it (and the live queue store) for this decision.
      const liveThreadIsRunning =
        threadIsRunning || aui.thread().getState().isRunning;
      const livePromptQueueActive =
        promptQueueActive ||
        hasPendingPromptQueueStart(
          promptQueueStartPendingRef.current.values(),
          referenceThreadId,
        ) ||
        hasPendingPromptQueueStart(
          pastedTextQueuePendingRef.current.values(),
          referenceThreadId,
        ) ||
        Boolean(
          findPromptQueueEntry(
            usePromptQueueUI.getState(),
            promptQueueThreadIds,
          ),
        );
      const livePreStreamRunActive =
        hasPreStreamRunReservation(preStreamThreadIds);

      if (
        liveThreadIsRunning ||
        livePromptQueueActive ||
        livePreStreamRunActive
      ) {
        event.preventDefault();
        // Project new-chat composer: never queue, just ask the user to wait.
        if (disableQueue) {
          toast.error("Wait for the current response to finish");
          return;
        }
        if (!canQueueCurrentPrompt) {
          if (
            canQueuePastedTextPrompt &&
            queuePastedTextPrompt(liveThreadIsRunning || livePreStreamRunActive)
          ) {
            return;
          }
          if (overlay || hasAttachments || hasPendingAudio) {
            toast.error(
              liveThreadIsRunning
                ? "Wait for the current response to finish"
                : "Wait for the prompt queue to finish",
              {
                description:
                  "Only text prompts can be queued while a response is running or the prompt queue is active.",
              },
            );
          }
          return;
        }
        queueComposerText(liveThreadIsRunning || livePreStreamRunActive);
        return;
      }

      // Cmd/Ctrl+Enter queues even with nothing running, so prompts can be
      // stacked up front. The queue dispatches this one immediately; the next
      // Cmd/Ctrl+Enter lands behind it.
      if (forceQueue && !disableQueue) {
        if (canQueueCurrentPrompt) {
          event.preventDefault();
          queueComposerText(false);
          return;
        }
        if (canQueuePastedTextPrompt) {
          event.preventDefault();
          if (queuePastedTextPrompt(false)) {
            return;
          }
        }
      }

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
        clearStoredDraft();
        setImageToolsEnabled(true);
        setPendingImageEditReference({
          threadId: overlay.threadId ?? referenceThreadId,
          openaiImageGenerationCallId: overlay.openaiImageGenerationCallId,
          ...(overlay.openaiResponseId
            ? { openaiResponseId: overlay.openaiResponseId }
            : {}),
          openaiReasoningItem: overlay.openaiReasoningItem,
        });
        // Live, not composerText: a late DOM write carries exactly what the
        // textarea held, whitespace and all, and that is what must be armed.
        const visibleBeforeWrap = aui.composer().getState().text;
        flushResourcesSync(() => {
          aui
            .composer()
            .setText(
              `Use the selected generated image as the reference and apply this edit: ${trimmed}. Preserve everything else exactly.`,
            );
        });
        closeOverlay();
        event.preventDefault();
        // The wrapper replaced what the user typed, so guard that text as well.
        sendReservedComposer(visibleBeforeWrap, trimmed);
        return;
      }

      if (hasAttachments || hasPendingAudio) {
        event.preventDefault();
        clearStoredDraft();
        sendReservedComposer();
        return;
      }
      event.preventDefault();
      clearStoredDraft();
      sendReservedComposer();
    },
    [
      aui,
      canQueueCurrentPrompt,
      canQueuePastedTextPrompt,
      queueComposerText,
      queuePastedTextPrompt,
      clearStoredDraft,
      closeOverlay,
      composerText,
      disabled,
      disableQueue,
      hasAttachments,
      hasPendingAudio,
      enqueueSend,
      interceptSend,
      isResearchActive,
      overlay,
      parkIfWaitingOnAttachments,
      threadScopedSettingsPending,
      promptQueueActive,
      promptQueueThreadIds,
      preStreamThreadIds,
      referenceThreadId,
      setImageToolsEnabled,
      setPendingImageEditReference,
      sendReservedComposer,
      shouldBlockSend,
      threadIsRunning,
    ],
  );

  const stopQueue = useCallback(() => {
    stopPromptQueueRunForThreadIds(promptQueueThreadIds);
  }, [promptQueueThreadIds]);

  const startQueue = useCallback(
    (
      items: string[],
      waitForCurrentRun =
        threadIsRunning || aui.thread().getState().isRunning,
      onAborted?: () => void,
    ) => {
      // Saved-prompt Run-list calls this directly, so honour disableQueue here
      // too: queuing from the project new-chat composer misbinds the thread.
      if (disableQueue) return false;
      return startHydratedPromptQueue(
        items,
        waitForCurrentRun,
        undefined,
        onAborted,
      );
    },
    [aui, startHydratedPromptQueue, threadIsRunning, disableQueue],
  );

  const queueContextValue: PromptQueueCallbacks = { startQueue, stopQueue };

  const composerContent = (
    <>
      {!isDictating ? (
        <>
          <ComposerAttachments />
          <PendingAudioChip />
        </>
      ) : null}
      {/* Keep indexing state subscribed while dictating, but hide its chips so
          the waveform stays the composer's only status indicator. */}
      <div className={isDictating ? "hidden" : "contents"}>
        <ThreadDocumentsBar
          threadId={referenceThreadId}
          onIndexingChange={handleIndexingChange}
        />
      </div>
      {!isDictating ? <ToolStatusDisplay /> : null}
      <div
        className="unsloth-composer-line"
        // The permission pill is always visible, so keep the two-row layout
        // expanded whenever not dictating; dictation collapses to the bar.
        data-expanded={!isDictating ? "true" : "false"}
        data-dictating={isDictating ? "true" : undefined}
      >
        <div
          ref={pillRowRef}
          className="unsloth-composer-left"
          data-pill-compact={pillCompact}
        >
          <ComposerToolsMenu
            side={effectiveMenuSide}
            researchAvailable={!researchUsed}
          />
          {/* While dictating, show only the "+"; hide the pill and tool toggles
              so the waveform is the sole status indicator. */}
          {!isDictating ? (
            <>
              {/* Permission-level pill: always visible, opens the level dropdown. */}
              <PermissionModeComposerPill side={effectiveMenuSide} />
              {effectiveDeepResearchEnabled ? (
                <DeepResearchComposerButton
                  onConfigure={() => setResearchWebsiteAccessOpen(true)}
                />
              ) : null}
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
        {isDictating ? (
          // The recording UI replaces the input and send controls; only the
          // left plus stays visible alongside it.
          <ChatDictationBar
            onSend={sendAfterDictation}
            // Every state handleSubmit rejects, since it would reject after
            // transcription with the send intent already spent. Text presence
            // is left out: the transcript supplies it.
            sendDisabled={dictationBlocked}
          />
        ) : (
          <>
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
              // Capture, so inputProps keeps the handlers it already owns.
              onKeyDownCapture={notePlainPasteChord}
              onKeyUpCapture={endPlainPasteChord}
              onBlurCapture={endPlainPasteChord}
              addAttachmentOnPaste={false}
              onPaste={handleFilePaste}
            />
            <ComposerRightControls
              disabled={
                disabled ||
                !hasSendableContent ||
                isComposing ||
                hasPendingAttachments
              }
              // disableQueue (project new-chat composer) also blocks the queue
              // button, so a running thread shows Stop instead of Queue.
              queueDisabled={
                disableQueue ||
                !(canQueueCurrentPrompt || canQueuePastedTextPrompt)
              }
              onQueueClick={() => {
                if (disableQueue) return;
                // Same pasted-text path the Enter key takes, or the button
                // would refuse what submitting the form accepts.
                if (
                  canQueuePastedTextPrompt &&
                  queuePastedTextPrompt(true)
                ) {
                  return;
                }
                const queuedPrompt = composerText.trim();
                if (queuedPrompt.length === 0) {
                  return;
                }
                startHydratedPromptQueue(
                  [queuedPrompt],
                  true,
                  () => {
                    const cleared = aui.composer().getState().text;
                    if (cleared.trim() !== queuedPrompt) {
                      return;
                    }
                    flushResourcesSync(() => {
                      aui.composer().setText("");
                    });
                    clearStoredDraft();
                    armJustSent(queuedPrompt, cleared);
                  },
                );
              }}
              // ComposerPrimitive.Send handles clicks itself rather than
              // submitting the form, so run the complete queue/capacity path.
              onSendClick={handleSubmit}
              onStopClick={stopQueue}
              onDictateClick={startDictation}
              pendingSend={pendingSend}
              menuSide={effectiveMenuSide}
              queueThreadIds={promptQueueThreadIds}
            />
          </>
        )}
      </div>
      <DeepResearchWebsiteAccessDialog
        open={researchWebsiteAccessOpen && effectiveDeepResearchEnabled}
        onOpenChange={setResearchWebsiteAccessOpen}
      />
    </>
  );

  return (
    <PromptQueueContext.Provider value={queueContextValue}>
    <ComposerPrimitive.Root
      ref={attachComposer}
      className="aui-composer-root relative flex w-full flex-col"
      aria-disabled={disabled}
      onSubmit={handleSubmit}
    >
      <PromptQueueStack queueThreadIds={promptQueueThreadIds} />
      {youtubeOfferUrl && !isDictating && !disabled ? (
        // Keyed by URL: pasting a second link while the first is still fetching
        // remounts the prompt, so its cleanup aborts the request that is no
        // longer the one on offer.
        <YoutubeTranscriptPrompt
          key={youtubeOfferUrl}
          url={youtubeOfferUrl}
          onClose={() => setYoutubeLink(null)}
        />
      ) : null}
      <ProjectGoalBar />
      {isTauri ? (
        // Phase 1 native model owns Tauri local-path drops. Restore browser
        // attachment drops in Tauri once Phase 1d adds token bridging.
        <div className="aui-composer-attachment-dropzone unsloth-composer-surface relative z-10">
          {composerContent}
        </div>
      ) : (
        <ComposerPrimitive.AttachmentDropzone
          className="group/dropzone aui-composer-attachment-dropzone unsloth-composer-surface relative z-10"
          onDragEnterCapture={claimPortaledDrop}
          onDragOverCapture={claimPortaledDrop}
          onDropCapture={claimPortaledDrop}
        >
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

// An autocorrect commit, never a keystroke, a paste or an undo. Its value can
// differ from what was sent, so equality alone would let it through.
function isTextReplacement(event: Event | undefined) {
  return inputTypeOf(event) === "insertReplacementText";
}

// An IME composition write. Finalisation converts the text, so equality never
// matches it, and it is stale only when the composition began before the send:
// one begun after raises compositionstart, which records user input.
// compositionend counts because onCompositionEnd applies that value itself and
// the browser raises no input event for it.
function isCompositionWrite(event: Event | undefined) {
  return (
    inputTypeOf(event) === "insertCompositionText" ||
    event?.type === "compositionend" ||
    (event !== undefined && isNativeComposing(event))
  );
}

// Input types only a gesture produces, so they apply even when they carry the
// sent text. Drag and drop and yank have no event to hook the way paste does.
const DELIBERATE_INPUT_TYPES = new Set([
  "historyUndo",
  "historyRedo",
  "insertFromPaste",
  "insertFromDrop",
  "insertFromYank",
]);

function isDeliberateWrite(event: Event | undefined) {
  return DELIBERATE_INPUT_TYPES.has(inputTypeOf(event) ?? "");
}

function inputTypeOf(event: Event | undefined): string | undefined {
  if (event === undefined || !("inputType" in event)) return undefined;
  return (event as InputEvent).inputType;
}

// Fallback timeout for stuck IME composition. With Chrome on Windows against
// a WSL-hosted Unsloth (issue #5546), `compositionend` never fires after the
// candidate commits, so `composingRef` stays true and Send stays disabled.
// Every compositionupdate / non-composing input resets the timer; only a true
// gap-after-commit lets it fire. 2500ms is above a normal candidate-window
// pause but short enough to recover before the user notices Send is stuck.
const IME_STUCK_TIMEOUT_MS = 2500;

function useImeComposerInputHandlers({
  submitOnEnter = false,
  onModEnter,
  justSentRef,
  draftKeyRef,
}: {
  submitOnEnter?: boolean;
  /** Cmd/Ctrl+Enter without Shift, claimed before the plain-Enter submit. */
  onModEnter?: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  // Guard armed by the last send or queue. See setComposerText below.
  justSentRef?: RefObject<SentTextGuard | null>;
  // Thread on screen. The composer outlives a thread switch, so this is what
  // says whether the armed guard belongs to the thread being typed into.
  draftKeyRef?: RefObject<string | null>;
} = {}) {
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

  // False when refused, so the caller can preventDefault and stop
  // ComposerPrimitive.Input's own handler applying the same value.
  const setComposerText = useCallback(
    (value: string, nativeEvent?: Event): boolean => {
      const composer = aui.composer();
      if (!composer.getState().isEditing) {
        return false;
      }
      // Refuse a write that is the sent message coming back, but only for the
      // thread that sent: typing in another thread must not retire a guard it
      // does not own, or its raced draft returns.
      const guardOwnsThread =
        justSentRef?.current == null ||
        draftKeyRef === undefined ||
        justSentRef.current.draftKey === draftKeyRef.current;
      if (justSentRef && guardOwnsThread) {
        const result = applySentTextGuard(justSentRef.current, {
          value,
          replacesText: isTextReplacement(nativeEvent),
          isDeliberate: isDeliberateWrite(nativeEvent),
          isComposition: isCompositionWrite(nativeEvent),
          composerIsEmpty: composer.getState().text.length === 0,
        });
        justSentRef.current = result.guard;
        if (!result.accept) {
          return false;
        }
      }
      flushResourcesSync(() => {
        composer.setText(value);
      });
      return true;
    },
    [aui, draftKeyRef, justSentRef],
  );

  const onCompositionStart = useCallback(() => {
    // Dictation and handwriting insert without a keydown, and a composition
    // starting after the send cannot be a write the send queued.
    if (justSentRef) {
      justSentRef.current = markSentTextGuardUserInput(justSentRef.current);
    }
    setCompositionState(true);
  }, [justSentRef, setCompositionState]);

  const onCompositionUpdate = useCallback(() => {
    refreshStuckTimer();
  }, [refreshStuckTimer]);

  const onCompositionEnd = useCallback(
    (e: CompositionEvent<HTMLTextAreaElement>) => {
      setCompositionState(false);
      if (!setComposerText(e.currentTarget.value, e.nativeEvent)) {
        e.preventDefault();
      }
    },
    [setComposerText, setCompositionState],
  );

  const onChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      setCompositionState(isNativeComposing(e.nativeEvent));
      if (!setComposerText(e.target.value, e.nativeEvent)) {
        e.preventDefault();
      }
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
        // Deliberately NOT user input: picking a candidate in a composition the
        // send left open is that composition continuing. One begun after the
        // send is marked by compositionstart instead.
        composingRef.current = true;
        refreshStuckTimer();
        return;
      }
      if (justSentRef && isGuardRetiringKey(e)) {
        justSentRef.current = markSentTextGuardUserInput(justSentRef.current);
      }
      if (composingRef.current) {
        // Candidate-confirming Enter can arrive as non-composing; keep it gated.
        if (e.key === "Enter") {
          if (!e.shiftKey) {
            e.preventDefault();
          }
          refreshStuckTimer();
          return;
        }
        // Non-IME key while composingRef is stuck; the input method was likely
        // switched away on macOS without firing compositionend (issue #5546
        // pattern, but triggered by input-method switch rather than WSL).
        // Clear immediately so Send is unblocked on the first non-IME keystroke
        // rather than waiting for the 2500ms watchdog.
        setCompositionState(false);
      }
      if (onModEnter && isPromptQueueChord(e)) {
        e.preventDefault();
        onModEnter(e);
        return;
      }
      if (submitOnEnter && e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        e.currentTarget.form?.requestSubmit();
      }
    },
    [
      justSentRef,
      onModEnter,
      refreshStuckTimer,
      setCompositionState,
      submitOnEnter,
    ],
  );

  // On macOS, switching input methods (e.g. ABC → Pinyin) while the textarea
  // is focused can fire compositionstart without a matching compositionend,
  // leaving composingRef pinned and Send permanently blocked. The OS always
  // commits or cancels any in-progress composition before surrendering focus,
  // so blur is a safe unconditional reset point.
  const onBlur = useCallback(() => {
    setCompositionState(false);
  }, [setCompositionState]);

  return {
    inputProps: {
      onCompositionStart,
      onCompositionUpdate,
      onCompositionEnd,
      onChange,
      onKeyDown,
      onBlur,
    },
    isComposing,
    isComposingRef: composingRef,
  };
}

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

  // enable_thinking_effort (GLM-5.2: high|max + disable) reuses the effort
  // dropdown; it just also carries an Off row via supportsReasoningOff.
  const isEffort =
    effectiveReasoningStyle === "reasoning_effort" ||
    effectiveReasoningStyle === "enable_thinking_effort";
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
            data-pill-label="Thinking settings"
            data-active={activeLook ? "true" : "false"}
            aria-label={thinkEffortAriaLabel({
              modelLoaded,
              reasoningDisabled: disabled,
              reasoningEffort,
            })}
          >
            <ThinkIcon />
            {activeLook ? (
              <span className="unsloth-thinking-label">
                {isEffort ? `Thinking · ${effortLabel}` : "Thinking"}
              </span>
            ) : null}
            <ArrowDownStandardIcon className="unsloth-thinking-caret size-[15px]" />
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
                // 'none' is a real template level for models like Inkling
                // (effort 0 = thinking off); show it as a pick unless the
                // dedicated off item above already covers it.
                .filter(
                  (level) =>
                    level !== "none" || !effectiveSupportsReasoningOff,
                )
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
      data-pill-label="Thinking"
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
      {activeLook ? (
        <span className="unsloth-thinking-label">Thinking</span>
      ) : null}
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
      data-pill-label="Search"
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
      data-pill-label="Code"
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
      data-pill-label="Images"
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
  );
};

const ToolStatusDisplay: FC = () => {
  // This conversation's tool call only: a global status would put one chat's "Running
  // Python..." above every composer. remoteId, not id: the adapter keys this map by
  // unstable_threadId, so reading id lost the status of every restored chat.
  const threadListItemId = useAuiState(
    ({ threadListItem }) => threadListItem.remoteId,
  );
  const isThreadRunning = useAuiState(({ thread }) => thread.isRunning);
  const entry = useChatRuntimeStore((s) => {
    // A first turn starts before its id is persisted, so the adapter files it under
    // "__default"; only this thread's own run may claim it. Two first turns share that key
    // with nothing to tell them apart, so claim it only when it holds one run.
    const unresolved = s.toolStatusByThreadId.__default;
    const own =
      s.toolStatusByThreadId[threadListItemId ?? ""] ??
      (isThreadRunning && unresolved?.length === 1 ? unresolved : undefined);
    // Newest of the runs behind this key: separate entries, so one finishing cannot blank
    // a sibling still running a tool.
    return own?.[own.length - 1];
  });
  const toolStatus = entry?.status ?? null;
  const startedAt = entry?.startedAt ?? null;
  const [now, setNow] = useState(() => Date.now());
  const [visible, setVisible] = useState(false);
  const visibleRef = useRef(false);

  useEffect(() => {
    visibleRef.current = visible;
  }, [visible]);

  useEffect(() => {
    if (!startedAt) {
      if (!isThreadRunning) {
        setVisible(false);
      }
      return;
    }

    setNow(Date.now());

    // Debounce visibility by 300ms when the badge isn't already on screen.
    // Once visible from a prior tool, later tools show immediately so it
    // doesn't flicker; tool calls under 300ms never show the badge.
    let showTimer: ReturnType<typeof setTimeout> | undefined;
    if (!visibleRef.current) {
      showTimer = setTimeout(() => setVisible(true), 300);
    }

    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      clearInterval(interval);
      if (showTimer) {
        clearTimeout(showTimer);
      }
    };
  }, [startedAt, isThreadRunning]);

  if (!(toolStatus && startedAt && visible)) {
    return null;
  }
  // From the store's start time, so returning to the conversation resumes rather than restarting.
  const elapsed = Math.max(0, Math.floor((now - startedAt) / 1000));
  const kind = toolStatusKind(toolStatus);
  const isNudging = kind === "nudge";
  const StatusIcon = kind === "terminal" ? TerminalIcon : GlobeIcon;
  return (
    <div
      data-testid="composer-tool-status"
      className="mb-2 flex w-full flex-row items-center gap-2 px-1.5 pt-0.5 pb-1"
    >
      <div
        className={cn(
          "flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-3 py-1.5 text-xs text-primary",
          // The spinner is its own motion cue; pulsing too just fades it mid-spin.
          !isNudging && "animate-pulse",
        )}
      >
        {isNudging ? (
          // label, not the default "Loading": the spinner is the badge's only
          // role="status" region, so its name is what gets announced.
          <Spinner className="size-3.5" label={toolStatus} />
        ) : (
          <StatusIcon className="size-3.5" />
        )}
        <span>{toolStatus}</span>
        <span className="tabular-nums opacity-60">{elapsed}s</span>
      </div>
    </div>
  );
};
// Plus menu: attachment and workflow actions. Opens downward in the welcome
// composer; the docked composer passes side="top" to open upward.
const AUDIO_ACCEPT_TOKEN_RE =
  /^(audio\/|\.(?:wav|mp3|m4a|ogg|oga|flac)$)/i;

function attachmentAcceptForPicker(accept: string, audioEnabled: boolean): string {
  if (audioEnabled || accept === "*") {
    return accept;
  }
  const filtered = accept
    .split(",")
    .map((token) => token.trim())
    .filter((token) => token && !AUDIO_ACCEPT_TOKEN_RE.test(token))
    .join(",");
  return filtered || accept;
}

const ComposerToolsMenu: FC<{
  side?: "top" | "bottom";
  researchAvailable: boolean;
}> = ({ side = "bottom", researchAvailable }) => {
  const navigate = useNavigate();
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const setCodeToolsEnabled = useChatRuntimeStore((s) => s.setCodeToolsEnabled);
  const artifactsEnabled = useChatRuntimeStore((s) => s.artifactsEnabled);
  const setArtifactsEnabled = useChatRuntimeStore((s) => s.setArtifactsEnabled);
  const showCanvasMenuItem = useChatRuntimeStore((s) => s.showCanvasMenuItem);
  const mcpEnabledForChat = useChatRuntimeStore((s) => s.mcpEnabledForChat);
  const setMcpEnabledForChat = useChatRuntimeStore(
    (s) => s.setMcpEnabledForChat,
  );
  const deepResearchEnabled = useChatRuntimeStore((s) => s.deepResearchEnabled);
  const setDeepResearchEnabled = useChatRuntimeStore((s) => s.setDeepResearchEnabled);
  const incognito = useChatRuntimeStore((s) => s.incognito);
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const setRagEnabled = useChatRuntimeStore((s) => s.setRagEnabled);
  // Shared gate so the menu row agrees with the RAG pill.
  const ragDisabled = useRagToolDisabled();
  // Capability gating mirrors the visible pills so menu and pills agree on
  // what a loaded model supports (a tool the backend drops must not look on).
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const audioAttachmentsEnabled = useChatRuntimeStore((s) => {
    const activeCheckpoint = s.params.checkpoint;
    if (!activeCheckpoint || s.modelLoading) {
      return false;
    }
    const activeModel = s.models.find((m) => m.id === activeCheckpoint);
    return Boolean(activeModel?.hasAudioInput);
  });
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
  // Match Search and Code: allow pre-selection before a local model loads.
  const researchDisabled =
    !researchAvailable ||
    (Boolean(externalSelection) &&
      providerModelSupportsStudioTools(
        selectedExternalProvider?.providerType,
        externalSelection?.modelId,
      ) !== true) ||
    incognito;
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
  const composerCanAddAttachments = useAuiState(
    ({ composer }) => composer.isEditing,
  );
  const pickAttachment = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.hidden = true;

    const attachmentAccept = attachmentAcceptForPicker(
      aui.composer().getState().attachmentAccept,
      audioAttachmentsEnabled,
    );
    if (attachmentAccept !== "*") {
      input.accept = attachmentAccept;
    }

    document.body.appendChild(input);
    input.onchange = (event) => {
      const files = (event.target as HTMLInputElement).files;
      if (files) {
        for (const file of files) {
          void aui.composer().addAttachment(file);
        }
      }
      document.body.removeChild(input);
    };
    input.oncancel = () => {
      if (!input.files || input.files.length === 0) {
        document.body.removeChild(input);
      }
    };
    input.click();
  }, [aui, audioAttachmentsEnabled]);
  // Straight to the picker, skipping the "+" menu the item lives in. Off-route
  // the chat pane is hidden rather than unmounted, so the chords gate on it
  // being the visible tab; a window listener does not care about `inert`.
  const chatActive = useChatActive();
  useShortcut(
    "attachFiles",
    () => {
      // `chatActive` is the visible tab, not the foreground, so a dialog over
      // Chat left this live, and the OS file chooser is the least dismissable
      // thing a chord can raise.
      if (!isSurfaceInForeground(COMPOSER_INPUT_SELECTOR)) return;
      pickAttachment();
    },
    { enabled: chatActive && composerCanAddAttachments },
  );
  // Exports are storage-backed; temporary chats intentionally never write there.
  const messageCount = useAuiState(({ thread }) => thread.messages.length);
  const exportDisabled = incognito || !activeThreadId || messageCount === 0;
  const { startQueue } = useContext(PromptQueueContext);

  const plusPins = usePlusMenuPrefsStore((s) => s.pins);

  const [recentPrompts, setRecentPrompts] = useState<PromptEntry[]>([]);
  const refreshRecentPrompts = useCallback(async () => {
    try {
      const rows = await listPromptEntries();
      const byRecent = [...rows].sort((a, b) => b.updatedAt - a.updatedAt);
      // Pinned prompts take over the submenu; fall back to the 3 most recent
      // when nothing is pinned.
      const pinnedIds = usePlusMenuPrefsStore.getState().pinnedPromptIds;
      const pinned = byRecent.filter((p) => pinnedIds.includes(p.id));
      setRecentPrompts(pinned.length > 0 ? pinned : byRecent.slice(0, 3));
    } catch {
    }
  }, []);

  // Adjustable "+" menu items, keyed by id. Pinned ones render at the top
  // level; the rest fall into the "More" overflow submenu. The core items
  // (photos, web search, code) and "More" itself are always shown and live
  // outside this map.
  const plusMenuNodes: Record<PlusMenuItemId, ReactNode> = {
    chatWithFiles: (
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
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
        ) : null}
      </DropdownMenuItem>
    ),
    mcp: (
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
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
        ) : null}
      </DropdownMenuItem>
    ),
    savedPrompts: (
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>
          <HugeiconsIcon icon={Bookmark02Icon} strokeWidth={2} />
          Saved prompts
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent
          collisionPadding={16}
          className="unsloth-plus-menu w-[208px]"
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
    ),
    compareChat: (
      <DropdownMenuItem onSelect={() => startCompare()}>
        <Columns2Icon />
        Compare chat
      </DropdownMenuItem>
    ),
    exportChat: (
      <DropdownMenuSub>
        <DropdownMenuSubTrigger disabled={exportDisabled}>
          <HugeiconsIcon icon={Download01Icon} strokeWidth={2} />
          Export chat
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent
          collisionPadding={16}
          className="unsloth-plus-menu w-[208px]"
        >
          <DropdownMenuItem
            onSelect={() => {
              if (!activeThreadId) return;
              exportConversationRawJsonl(activeThreadId).catch((error) => {
                if (!isDownloadCancelled(error)) toast.error("Export failed.");
              });
            }}
          >
            Training JSONL
          </DropdownMenuItem>
          <DropdownMenuItem
            onSelect={() => {
              if (!activeThreadId) return;
              exportConversationMessagesJsonl(activeThreadId).catch((error) => {
                if (!isDownloadCancelled(error)) toast.error("Export failed.");
              });
            }}
          >
            Message JSONL
          </DropdownMenuItem>
          <DropdownMenuItem
            onSelect={() => {
              if (!activeThreadId) return;
              exportConversationCsv(activeThreadId).catch((error) => {
                if (!isDownloadCancelled(error)) toast.error("Export failed.");
              });
            }}
          >
            CSV
          </DropdownMenuItem>
          <DropdownMenuItem
            onSelect={() => {
              if (!activeThreadId) return;
              exportConversationShareGPT(activeThreadId).catch((error) => {
                if (!isDownloadCancelled(error)) toast.error("Export failed.");
              });
            }}
          >
            ShareGPT JSONL
          </DropdownMenuItem>
          <DropdownMenuItem
            onSelect={() => {
              if (!activeThreadId) return;
              exportConversationMarkdown(activeThreadId).catch((error) => {
                if (!isDownloadCancelled(error)) toast.error("Export failed.");
              });
            }}
          >
            {CONVERSATION_MARKDOWN_LABEL}
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    ),
    // Hidden by default; enabled from Settings > Chat > Canvas.
    canvas: showCanvasMenuItem ? (
      <DropdownMenuItem
        className={artifactsEnabled ? "text-primary font-medium" : undefined}
        onSelect={() => setArtifactsEnabled(!artifactsEnabled)}
      >
        <HugeiconsIcon icon={PencilRulerIcon} strokeWidth={2} />
        Canvas
        {artifactsEnabled ? (
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
        ) : null}
      </DropdownMenuItem>
    ) : null,
    bypassPermissions: <BypassPermissionsMenuItem />,
    projects: (
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>
          <HugeiconsIcon icon={Folder01Icon} strokeWidth={2} />
          Projects
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent className="unsloth-plus-menu w-[232px]">
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
    ),
  };
  const pinnedPlusItems = PLUS_MENU_ORDER.filter((id) => plusPins[id]);
  const overflowPlusItems = PLUS_MENU_ORDER.filter((id) => !plusPins[id]);

  return (
    <>
    <PromptStorageDialog
      open={promptStorageOpen}
      onOpenChange={setPromptStorageOpen}
      onUse={(text) => {
        aui.composer().setText(text);
      }}
      onRunList={(items) => {
        const started = startQueue(items, undefined, () => {
          setPromptStorageOpen(true);
          toast.info("Saved list was not queued", {
            description: "The chat changed before the queue was ready. Try again.",
          });
        });
        if (started) {
          setPromptStorageOpen(false);
        }
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
          data-tour="chat-plus-menu"
        >
          <PlusIcon className="size-[22px] stroke-[1.75px]" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side={side}
        align="start"
        sideOffset={0}
        avoidCollisions={true}
        className="unsloth-plus-menu w-[244px]"
        // Don't refocus the + on close; restored focus showed a stray ring.
        onCloseAutoFocus={(event) => event.preventDefault()}
      >
        <DropdownMenuItem
          disabled={!composerCanAddAttachments}
          onSelect={() => pickAttachment()}
        >
          <HugeiconsIcon icon={AttachmentIcon} strokeWidth={2} />
          Add photos &amp; files
        </DropdownMenuItem>
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
        {researchAvailable ? (
          <DropdownMenuItem
            disabled={researchDisabled && !deepResearchEnabled}
            className={
              deepResearchEnabled && !researchDisabled
                ? "text-primary font-medium"
                : undefined
            }
            onSelect={() => setDeepResearchEnabled(!deepResearchEnabled)}
          >
            <HugeiconsIcon icon={Telescope02Icon} strokeWidth={2} />
            Deep research
            {deepResearchEnabled && !researchDisabled ? (
              <HugeiconsIcon
                icon={Tick02Icon}
                strokeWidth={2}
                className="ml-auto"
              />
            ) : null}
          </DropdownMenuItem>
        ) : null}
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
        {pinnedPlusItems.map((id) => (
          <Fragment key={id}>{plusMenuNodes[id]}</Fragment>
        ))}
        {overflowPlusItems.length > 0 ? (
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <MoreHorizontalIcon className="size-4" />
              More
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="unsloth-plus-menu w-[248px]">
              {overflowPlusItems.map((id) => (
                <Fragment key={id}>{plusMenuNodes[id]}</Fragment>
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
      <NewProjectDialog
        open={newProjectOpen}
        onOpenChange={setNewProjectOpen}
      />
    </>
  );
};

function promptQueueStatusLabel(status: PromptQueueUIItemStatus) {
  switch (status) {
    case "running":
      return "Running now";
    case "waiting":
      return "Waiting";
    case "next":
      return "Next";
    case "queued":
      return "Queued";
    default: {
      const exhaustiveStatus: never = status;
      throw new Error(`Unhandled prompt queue status: ${exhaustiveStatus}`);
    }
  }
}

// dataTransfer.getData is blocked during dragover, but types is always readable.
function isPromptQueueDrag(event: ReactDragEvent): boolean {
  return isPromptQueueDragTypes(event.dataTransfer?.types);
}

const PromptQueueStack: FC<{ queueThreadIds: string[] }> = ({
  queueThreadIds,
}) => {
  const queueEntry = usePromptQueueUI((s) =>
    findPromptQueueEntry(s, queueThreadIds),
  );
  const items = usePromptQueueUI((s) => s.items);
  const [editingItemId, setEditingItemId] = useState<string | null>(null);
  const [draftPrompt, setDraftPrompt] = useState("");
  const [draggingItemId, setDraggingItemId] = useState<string | null>(null);
  const [dragOverItemId, setDragOverItemId] = useState<string | null>(null);
  const editInputRef = useRef<HTMLTextAreaElement>(null);
  const visibleItems = queueEntry
    ? items.filter((item) => item.runId === queueEntry.runId)
    : [];
  const editingItem = visibleItems.find((item) => item.id === editingItemId);
  const activeEditingItemId = editingItem ? editingItemId : null;

  useEffect(() => {
    if (!activeEditingItemId) {
      return;
    }
    editInputRef.current?.focus();
    editInputRef.current?.select();
  }, [activeEditingItemId]);

  if (!queueEntry || visibleItems.length === 0) {
    return null;
  }

  const { current, total } = queueEntry;

  const startEditing = (item: PromptQueueUIItem) => {
    if (!item.canEdit) {
      return;
    }
    setEditingItemId(item.id);
    setDraftPrompt(item.prompt);
  };
  const saveEditing = () => {
    if (!activeEditingItemId) {
      return;
    }
    if (editPromptQueueItem(activeEditingItemId, draftPrompt)) {
      setEditingItemId(null);
      setDraftPrompt("");
    }
  };
  const cancelEditing = () => {
    setEditingItemId(null);
    setDraftPrompt("");
  };
  const endDrag = () => {
    setDraggingItemId(null);
    setDragOverItemId(null);
  };
  // Keyboard equivalent of a drag, since HTML5 drag events never fire for keys.
  const moveByOffset = (index: number, offset: number) => {
    const target = visibleItems[index + offset];
    if (!target) {
      return;
    }
    movePromptQueueItem(visibleItems[index].id, target.id);
  };
  const reorderable = visibleItems.length > 1;

  return (
    <div
      className="relative z-0 mx-7 mb-[-8px] max-h-[28dvh] overflow-y-auto rounded-t-[18px] rounded-b-none border border-border/45 bg-background/90 px-5 py-2 text-muted-foreground shadow-none backdrop-blur-md dark:bg-card/85"
      aria-label={`Prompt queue, ${current} of ${total}`}
    >
      <div className="divide-y divide-border/25">
        {visibleItems.map((item, visibleIndex) => {
          const isEditing = item.id === activeEditingItemId;
          const visiblePosition = visibleIndex + 1;
          return (
            <div
              key={item.id}
              className={cn(
                "min-h-10",
                isEditing ? "h-auto" : "h-10",
                draggingItemId === item.id && "opacity-40",
                dragOverItemId === item.id &&
                  draggingItemId !== item.id &&
                  "rounded-md ring-1 ring-ring/60",
              )}
              draggable={reorderable && !isEditing}
              onDragStart={(event) => {
                setDraggingItemId(item.id);
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData(PROMPT_QUEUE_DRAG_TYPE, item.id);
              }}
              onDragEnd={endDrag}
              onDragOver={(event) => {
                // Own type only: a file dragged over a row must reach the page
                // dropzone, which skips events already prevented here.
                if (!isPromptQueueDrag(event) || draggingItemId === item.id) {
                  return;
                }
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
                setDragOverItemId(item.id);
              }}
              onDragLeave={() => {
                setDragOverItemId((id) => (id === item.id ? null : id));
              }}
              onDrop={(event) => {
                if (!isPromptQueueDrag(event)) {
                  return;
                }
                event.preventDefault();
                const sourceId =
                  event.dataTransfer.getData(PROMPT_QUEUE_DRAG_TYPE) ||
                  draggingItemId;
                if (sourceId) {
                  movePromptQueueItem(sourceId, item.id);
                }
                endDrag();
              }}
              aria-label={`${promptQueueStatusLabel(item.status)} prompt ${visiblePosition} of ${visibleItems.length}: ${item.prompt}`}
            >
              {isEditing ? (
                <div className="grid min-h-10 grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-2.5 py-1">
                  <textarea
                    ref={editInputRef}
                    value={draftPrompt}
                    rows={1}
                    onChange={(event) =>
                      setDraftPrompt(event.currentTarget.value)
                    }
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" &&
                        (event.metaKey || event.ctrlKey)
                      ) {
                        event.preventDefault();
                        saveEditing();
                      } else if (event.key === "Escape") {
                        event.preventDefault();
                        cancelEditing();
                      }
                    }}
                    className="max-h-20 min-h-8 min-w-0 resize-none rounded-md border border-border/45 bg-transparent px-2 py-1.5 text-sm leading-5 text-foreground outline-none transition-colors focus-visible:border-ring"
                    aria-label={`Edit queued prompt ${visiblePosition}`}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2 text-xs text-muted-foreground"
                    onClick={cancelEditing}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    disabled={draftPrompt.trim().length === 0}
                    onClick={saveEditing}
                  >
                    Save
                  </Button>
                </div>
              ) : (
                <div className="grid h-10 grid-cols-[minmax(0,1fr)_auto_2rem] items-center gap-2.5">
                  <div className="flex min-w-0 items-center gap-2.5">
                    {reorderable ? (
                      <button
                        type="button"
                        className="shrink-0 cursor-grab text-muted-foreground/50 outline-none hover:text-muted-foreground focus-visible:text-foreground active:cursor-grabbing"
                        aria-label={`Reorder queued prompt ${visiblePosition} of ${visibleItems.length}`}
                        onKeyDown={(event) => {
                          if (event.key !== "ArrowUp" && event.key !== "ArrowDown") {
                            return;
                          }
                          event.preventDefault();
                          moveByOffset(
                            visibleIndex,
                            event.key === "ArrowUp" ? -1 : 1,
                          );
                        }}
                      >
                        <CornerDownRightIcon className="size-4" />
                      </button>
                    ) : (
                      <CornerDownRightIcon className="size-4 shrink-0 text-muted-foreground/50" />
                    )}
                    <div className="truncate text-sm text-muted-foreground">
                      {item.prompt}
                    </div>
                  </div>
                  {item.canEdit ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-7 w-[5.25rem] justify-center gap-1 px-0 text-sm font-normal text-muted-foreground/80 hover:text-foreground"
                      onClick={() => startEditing(item)}
                    >
                      <HugeiconsIcon icon={Edit03Icon} strokeWidth={2} />
                      Edit
                    </Button>
                  ) : null}
                  <TooltipIconButton
                    tooltip="Remove from queue"
                    side="bottom"
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="col-start-3 size-7 justify-self-center text-muted-foreground/70 hover:text-destructive"
                    aria-label={`Remove queued prompt ${visiblePosition}`}
                    disabled={!item.canRemove}
                    onClick={() => removePromptQueueItem(item.id)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} strokeWidth={2} />
                  </TooltipIconButton>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const ComposerRightControls: FC<{
  disabled?: boolean;
  queueDisabled?: boolean;
  onQueueClick?: () => void;
  onSendClick?: (event: { preventDefault: () => void }) => void;
  onStopClick?: () => void;
  onDictateClick?: () => void;
  pendingSend?: boolean;
  menuSide?: "top" | "bottom";
  queueThreadIds: string[];
}> = ({
  disabled,
  queueDisabled,
  onQueueClick,
  onSendClick,
  onStopClick,
  onDictateClick,
  pendingSend,
  menuSide,
  queueThreadIds,
}) => {
  const queueEntry = usePromptQueueUI((s) =>
    findPromptQueueEntry(s, queueThreadIds),
  );
  const isQueueRunning = Boolean(queueEntry);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  // Id and status, not the run: run identity changes on every streamed research delta.
  const activeResearchRunId = useResearchRunStore((state) =>
    activeThreadId ? state.latestRunByThreadId[activeThreadId] : undefined,
  );
  const activeResearchRunStatus = useResearchRunStore((state) => {
    const runId = activeThreadId
      ? state.latestRunByThreadId[activeThreadId]
      : undefined;
    return runId ? state.sessions[runId]?.run.status : undefined;
  });
  const isResearchActive = Boolean(
    activeResearchRunStatus &&
      !["completed", "failed", "cancelled"].includes(activeResearchRunStatus),
  );
  const [stoppingResearchRunId, setStoppingResearchRunId] = useState<
    string | null
  >(null);
  const stoppingResearchRunIdRef = useRef<string | null>(null);
  const researchStopping = Boolean(
    activeResearchRunStatus &&
      (activeResearchRunStatus === "cancelling" ||
        (activeResearchRunId !== undefined &&
          stoppingResearchRunId === activeResearchRunId)),
  );
  useEffect(() => {
    if (
      !isResearchActive ||
      (stoppingResearchRunIdRef.current &&
        stoppingResearchRunIdRef.current !== activeResearchRunId)
    ) {
      stoppingResearchRunIdRef.current = null;
      setStoppingResearchRunId(null);
    }
  }, [activeResearchRunId, isResearchActive]);
  const stop = () => {
    if (isResearchActive && activeResearchRunId) {
      if (
        activeResearchRunStatus === "cancelling" ||
        stoppingResearchRunIdRef.current === activeResearchRunId
      ) {
        return;
      }
      if (isQueueRunning) onStopClick?.();
      stoppingResearchRunIdRef.current = activeResearchRunId;
      setStoppingResearchRunId(activeResearchRunId);
      void cancelResearchRun(activeResearchRunId)
        .then((run) => ingestResearchUpdate(run))
        .catch((error) => {
          stoppingResearchRunIdRef.current = null;
          setStoppingResearchRunId(null);
          toast.error("Could not stop research", {
            description: error instanceof Error ? error.message : undefined,
          });
        });
      return;
    }
    if (isQueueRunning) onStopClick?.();
  };
  return (
    <div className="aui-composer-action-wrapper flex shrink-0 items-center gap-1.5">
      <ReasoningToggle side={menuSide} />
      {/* Starts dictation; the recording bar then covers the input row and owns
          the stop and send actions. */}
      <ComposerPrimitive.If dictation={false}>
        <TooltipIconButton
          tooltip="Dictate"
          aria-label="Dictate"
          type="button"
          variant="ghost"
          className="size-8 rounded-full text-foreground"
          onClick={onDictateClick}
        >
          {/* size-[22px] is the fallback; unsloth-dictate-icon sets the size. */}
          <MicIcon className="unsloth-dictate-icon size-[22px]" />
        </TooltipIconButton>
      </ComposerPrimitive.If>
      <AuiIf
        condition={({ thread }) =>
          !thread.isRunning && !isQueueRunning && !isResearchActive
        }
      >
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
            className="aui-composer-send ml-1.5 size-9 rounded-full"
            aria-label="Send message"
          >
            {pendingSend ? (
              <Spinner className="size-[18px]" />
            ) : (
              <ArrowUpIcon className="unsloth-send-icon aui-composer-send-icon size-[21px] stroke-2" />
            )}
          </TooltipIconButton>
        </ComposerPrimitive.Send>
      </AuiIf>
      {isQueueRunning && !isResearchActive ? (
        <AuiIf condition={({ thread }) => !thread.isRunning}>
          {queueEntry?.dispatched ? (
            <Button
              type="button"
              variant="default"
              size="icon"
              className="aui-composer-cancel ml-1.5 size-9 rounded-full"
              aria-label="Stop queued message"
              onClick={stop}
            >
              <SquareIcon className="aui-composer-cancel-icon size-3 fill-current" />
            </Button>
          ) : (
            <TooltipIconButton
              tooltip="Queue message"
              side="bottom"
              type="button"
              variant="default"
              size="icon"
              disabled={disabled || queueDisabled}
              onClick={onQueueClick}
              className="aui-composer-send ml-1.5 size-9 rounded-full"
              aria-label="Queue message"
            >
              <ArrowUpIcon className="unsloth-send-icon aui-composer-send-icon size-[21px] stroke-2" />
            </TooltipIconButton>
          )}
        </AuiIf>
      ) : null}
      {isResearchActive ? (
        <Button
          type="button"
          variant="default"
          size="icon"
          className="aui-composer-cancel ml-1.5 size-9 rounded-full"
          aria-label={researchStopping ? "Stopping research" : "Stop research"}
          disabled={researchStopping}
          onClick={stop}
        >
          {researchStopping ? (
            <Spinner className="size-3.5" />
          ) : (
            <SquareIcon className="aui-composer-cancel-icon size-3 fill-current" />
          )}
        </Button>
      ) : (
        <AuiIf condition={({ thread }) => thread.isRunning}>
          <div className="ml-1.5 flex items-center">
            {queueDisabled ? (
            <ComposerPrimitive.Cancel asChild={true}>
              <Button
                type="button"
                variant="default"
                size="icon"
                className="aui-composer-cancel size-9 rounded-full"
                aria-label="Stop generating"
                onClick={stop}
              >
                <SquareIcon className="aui-composer-cancel-icon size-3 fill-current" />
              </Button>
            </ComposerPrimitive.Cancel>
            ) : (
            <TooltipIconButton
              tooltip="Queue message"
              side="bottom"
              type="button"
              variant="default"
              size="icon"
              disabled={queueDisabled}
              onClick={onQueueClick}
              className="aui-composer-send size-9 rounded-full"
              aria-label="Queue message"
            >
              <ArrowUpIcon className="unsloth-send-icon aui-composer-send-icon size-[21px] stroke-2" />
            </TooltipIconButton>
            )}
          </div>
        </AuiIf>
      )}
    </div>
  );
};

const MessageError: FC = () => {
  const researchRunId = useResearchMessageRunId();
  const researchActive = useThreadResearchActive();
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="aui-message-error-root mt-2 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md bg-destructive/10 p-3 text-destructive text-sm dark:bg-destructive/5 dark:text-red-200">
        <ErrorPrimitive.Message className="aui-message-error-message line-clamp-2 min-w-0 flex-1" />
        {/* Recovery path for interrupted/failed turns: regenerate in place. */}
        {!researchRunId && !researchActive && (
          <ActionBarPrimitive.Reload asChild={true}>
            <button
              type="button"
              className="aui-message-error-retry inline-flex shrink-0 items-center gap-1.5 rounded-md border border-destructive/40 px-2.5 py-1 text-xs font-medium transition-colors hover:bg-destructive/15"
            >
              <RefreshCwIcon strokeWidth={1.75} className="size-3.5" />
              Retry
            </button>
          </ActionBarPrimitive.Reload>
        )}
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

/** Text of an assistant turn: what a continuation resumes from.
 *
 * Text parts only: a continuation resumes the visible answer, not its private reasoning.
 * Joined with nothing, like the backend's `trailing_assistant_text`: a turn split around
 * a reasoning part never had a newline between its halves, and inventing one moves the
 * boundary. */
function assistantMessageText(content: readonly unknown[] | undefined): string {
  if (!content) {
    return "";
  }
  return content
    .filter(
      (part): part is { type: "text"; text: string } =>
        (part as { type?: string })?.type === "text" &&
        typeof (part as { text?: unknown })?.text === "string",
    )
    .map((part) => part.text)
    .join("");
}

/**
 * Resume a response that stopped early instead of regenerating it. Shown under the last
 * assistant turn when Max Tokens ran out, Stop was pressed, or the stream dropped.
 * Retry keeps its old meaning: drop the partial and start over.
 */
const ContinueMessageBar: FC = () => {
  // One subscription, not ten, on every message that is not the newest.
  //
  // The bar mounts under every assistant message and returns null unless it is the last, but the
  // ten `useAuiState` calls below ran first, each a subscription whose selector re-runs on EVERY
  // store update -- one per character typed (220 messages, 300K characters: 10,193 subscriptions,
  // 10,258 selector runs per keystroke).
  //
  // `isLast` is the same condition the body below already gates on, asked before the work rather
  // than after it, so nothing that used to render stops rendering.
  const isLast = useAuiState(({ message }) => message.isLast);
  if (!isLast) {
    return null;
  }
  return <ContinueMessageBarForLastMessage />;
};

const ContinueMessageBarForLastMessage: FC = () => {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const isLast = useAuiState(({ message }) => message.isLast);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const researchRunId = useResearchMessageRunId();
  const researchActive = useThreadResearchActive();
  const status = useAuiState(({ message }) => message.status);
  const metadata = useAuiState(({ message }) => message.metadata);
  const partial = useAuiState(({ message }) =>
    assistantMessageText(message.content),
  );
  // A tool-calling turn cannot be resumed: the continuation runs as a sibling, so the
  // call and its result would be missing from the outbound history.
  const continuable = useAuiState(({ message }) =>
    isContinuableContent(message.content),
  );
  // Gemini signs its text parts, and the resumed turn is replayed from this branch,
  // so the signature travels with the partial.
  const thoughtSignature = useAuiState(({ message }) =>
    readTextThoughtSignature(message.content),
  );
  // Audio input re-listens to the recording and answers afresh rather than resuming,
  // so continuing there would append a second answer.
  const fromAudioInput = useAuiState(({ thread }) =>
    Boolean(findLatestUserAudioBase64(thread.messages, false)),
  );
  // An audio-output model regenerates the whole clip and never reads the request.
  const audioOutputModel = useChatRuntimeStore((s) => {
    const activeModel = s.models.find((m) => m.id === s.params.checkpoint);
    return Boolean(activeModel?.isAudio && !activeModel.hasAudioInput);
  });
  // Cancelled comes through status (the adapter yields nothing after an abort); the
  // other two are stamped on metadata so they survive a reload.
  const stamped = readIncompleteInfo(metadata);
  const cancelled =
    status?.type === "incomplete" && status?.reason === "cancelled";
  const reason = cancelled ? ("cancelled" as const) : stamped?.reason;

  // Every gate the bar itself answers to. Resuming without asking has to clear the same
  // ones, or it would resume a turn the bar would have refused to offer.
  const resumable =
    Boolean(reason) &&
    isLast &&
    !isRunning &&
    !researchRunId &&
    !researchActive &&
    continuable &&
    modeAllowsContinuation({
      fromAudioInput,
      audioOutputModel,
    }) &&
    Boolean(partial.trim());

  // The parent is what every round of one logical turn shares; the message id changes
  // each round, because a continuation runs as a sibling.
  const parentId = useAuiState(({ thread, message }) => {
    const index = thread.messages.findIndex((m) => m.id === message.id);
    return index > 0 ? thread.messages[index - 1].id : null;
  });

  const startContinuation = useCallback(() => {
    const messages = aui.thread().getState().messages;
    const index = messages.findIndex((message) => message.id === messageId);
    if (index < 0) {
      return;
    }
    // Sibling of the truncated turn, so the branch picker can still reach the partial.
    const parent = index > 0 ? messages[index - 1].id : null;
    aui.thread().startRun({
      parentId: parent,
      runConfig: {
        custom: {
          [CONTINUATION_RUN_CONFIG_KEY]: { partial, thoughtSignature },
        },
      },
    });
  }, [aui, messageId, partial, thoughtSignature]);

  // The resumed turn's own fit. Resuming replays the partial as the final assistant turn,
  // which the fit protects, so a partial too big to sit beside the system turn makes the
  // request irreducible and every further round fails identically.
  const truncation = useAuiState(({ message }) => {
    const custom = (message.metadata as { custom?: Record<string, unknown> } | undefined)
      ?.custom;
    return (custom?.contextTruncation ?? null) as ContextTruncation | null;
  });

  // Hitting Max Tokens is the reply running out of room mid-sentence, not a decision the
  // user made, so it resumes on its own and the bar never appears. Bounded, and only for
  // `length`: see `shouldAutoContinue`. Asked per MESSAGE, not just per turn: the round
  // budget belongs to the turn and one spent round out of three still says yes, so
  // arriving at a message the claim below has already taken -- the branch picker back to
  // the truncated sibling, or returning to the chat -- would otherwise show a spinner for
  // a run `claimAutoContinue` refuses to start, on top of the Continue button it hides.
  //
  // Another tab won the message. The claim resolves after this component has already
  // rendered off `shouldAutoContinueMessage`, which cannot see a race the lock decides,
  // so the answer has to come back as state: without it this tab keeps a spinner for a
  // run it never started, with the manual Continue button hidden behind it.
  //
  // Remembered as the message the answer was decided for, not as a bare flag: rows are
  // mounted by INDEX (`<MessageByIndexProvider key={index}>` in progressive-messages.tsx),
  // so selecting a different truncated branch at the same index re-renders THIS component
  // instead of remounting it. A boolean survived that and suppressed the automatic
  // continuation of a message no other tab had claimed, for as long as the row lived.
  // Comparing ids re-answers per message while still refusing the one that really lost.
  const [heldElsewhereFor, setHeldElsewhereFor] = useState<string | null>(null);
  const claimHeldElsewhere = heldElsewhereFor === messageId;
  // The runtime this bar belongs to, so its keeper renews and releases this claim and no
  // other pane's.
  // The thread this run will file itself under, which is what the lease belongs to and
  // what its lifetime is read from. `remoteId`, not `id`: it is the value assistant-ui
  // passes the adapter as `unstable_threadId` and the key the run appears under in
  // `runningByThreadId`, and an uninitialized thread has an `id` but no `remoteId`.
  const runThreadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const autoContinuing =
    !claimHeldElsewhere &&
    resumable &&
    shouldAutoContinueMessage(messageId, reason, parentId, {
      fits: truncation?.fits,
      // The same cheap estimator the backend fit uses, which is all that is needed to
      // spot a partial that has already eaten the whole budget.
      partialTokens: Math.ceil(partial.length / 4),
      promptTarget: truncation?.prompt_target,
    });
  useEffect(() => {
    if (!autoContinuing || !parentId) {
      return;
    }
    let mounted = true;
    // Claimed in module scope, not a ref, and under a cross-tab lock. `<StrictMode>` in
    // src/main.tsx replays this effect on the same fiber with the same `autoContinuing`,
    // so nothing inside would have differed, and rechecking the round budget would not
    // help either: one recorded round still leaves the limit unspent. A ref fixed the
    // replay but not a real remount, so leaving the chat with a truncated branch selected
    // and returning fired it again, creating another sibling and another paid request.
    // A module claim survived both but not a second TAB, which has its own module scope
    // and its own empty claim; the lease behind this one is shared and settles that.
    void claimAutoContinue(messageId, runThreadId ?? "").then((claim) => {
      if (claim === "started") {
        // Is there still a message to resume? `aui.thread()` follows the SELECTION, not
        // the thread this bar belongs to, so a chat or branch switch inside the window the
        // Web Lock is pending leaves `startContinuation` looking at a different list, where
        // it finds nothing and issues no run at all.
        //
        // Asked BEFORE anything is held, because a hold whose run never appears is renewed
        // forever on purpose -- preflight has no upper bound, so no deadline can separate
        // "never coming" from "still on its way". A hold taken for a run that was never
        // issued therefore renews its lease for the life of the tab, and every other tab
        // reads that lease as live and refuses the message for just as long.
        //
        // Not the preflight case, and it cannot become it: this is `startRun` never having
        // been called, decided synchronously off the same store `startContinuation` reads a
        // line later in the same tick. A run that HAS been issued and is merely slow to
        // begin passes here and keeps its hold and its renewals.
        const stillThere = aui
          .thread()
          .getState()
          .messages.some((message) => message.id === messageId);
        if (!stillThere) {
          // Nothing held and nothing recorded, so the lease this claim took runs out its
          // own TTL -- the same thing a tab that closed mid-claim leaves behind -- and the
          // turn keeps the round no request was ever made for.
          //
          // The claim itself is given back, and only inside this tab: it is what makes
          // `claimAutoContinue` answer "skipped" for a message it has already continued,
          // and a message nothing was issued for has not been continued at all. Left in,
          // returning to this branch found the message skipped for the life of the tab.
          // The lease stays, so no second tab may start while this one is still deciding.
          forgetAutoContinue(messageId);
          return;
        }
        // Held for as long as THIS thread's run generates, wherever the user navigates
        // to meanwhile. The bar cannot hold it itself: the continuation's sibling becomes
        // the selected branch and unmounts this component almost at once.
        holdAutoContinueRun(messageId, runThreadId);
        // Started whether or not this component is still mounted: the run belongs to the
        // thread, not to the bar, and a claim taken and then dropped would leave the
        // message continued by nobody.
        //
        // Recorded BEFORE the run, so a round that produces nothing still spends its
        // budget instead of re-firing this effect forever.
        recordAutoContinue(parentId);
        startContinuation();
        return;
      }
      // `skipped` is this tab's own duplicate call, where the run is coming from the
      // other one and nothing on screen should move.
      if (claim === "held-elsewhere" && mounted) {
        setHeldElsewhereFor(messageId);
      }
    });
    return () => {
      mounted = false;
    };
  }, [
    aui,
    autoContinuing,
    parentId,
    messageId,
    startContinuation,
    runThreadId,
  ]);

  // Newest turn only: appending to an older one would strand the replies after it.
  // A turn cut mid-thought has no text to resume from, so Retry stays the way out.
  // `reason` is repeated rather than left to `resumable`, which is a boolean and so
  // narrows nothing: the label below needs it proven non-undefined.
  if (!resumable || !reason) {
    return null;
  }
  if (autoContinuing) {
    // The run is starting this tick; showing the bar first would flash a question that is
    // already being answered.
    return (
      <div className="aui-continue-bar mt-2 flex items-center gap-2 rounded-md border border-border/70 bg-muted/50 p-2.5 text-sm text-muted-foreground">
        <Loader2Icon className="size-3.5 animate-spin" strokeWidth={1.75} />
        Continuing automatically.
      </div>
    );
  }

  const handleContinue = () => {
    const messages = aui.thread().getState().messages;
    const index = messages.findIndex((message) => message.id === messageId);
    if (index < 0) {
      return;
    }
    // Sibling of the truncated turn, so the branch picker can still reach the partial.
    const parentId = index > 0 ? messages[index - 1].id : null;
    aui.thread().startRun({
      parentId,
      runConfig: {
        custom: {
          [CONTINUATION_RUN_CONFIG_KEY]: { partial, thoughtSignature },
        },
      },
    });
  };

  return (
    <div className="aui-continue-bar mt-2 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-border/70 bg-muted/50 p-2.5 text-sm">
      <span className="min-w-0 flex-1 text-muted-foreground">
        {incompleteLabel(reason)}.
      </span>
      <Button
        type="button"
        size="sm"
        variant="secondary"
        className="h-7 shrink-0 gap-1.5 text-xs"
        onClick={handleContinue}
      >
        <FastForwardIcon strokeWidth={1.75} className="size-3.5" />
        Continue
      </Button>
    </div>
  );
};

const WebSearchToolUIConfirmable = withToolConfirmation(WebSearchToolUI);
const KnowledgeBaseToolUIConfirmable =
  withToolConfirmation(KnowledgeBaseToolUI);
const PythonToolUIConfirmable = withToolConfirmation(PythonToolUI);
const TerminalToolUIConfirmable = withToolConfirmation(TerminalToolUI);
const CodeExecutionToolUIConfirmable =
  withToolConfirmation(CodeExecutionToolUI);
const ImageGenerationToolUIConfirmable = withToolConfirmation(
  ImageGenerationToolUI,
);
const RenderHtmlToolUIConfirmable = withToolConfirmation(RenderHtmlToolUI);
const ToolFallbackConfirmable = withToolConfirmation(ToolFallback);

/**
 * At module scope on purpose. The memo comparator in
 * `MessagePrimitivePartByIndex` checks `components.tools` by identity, so an
 * inline literal handed it a fresh object every render, failed the comparator
 * and rebuilt every already-finished part of a streaming reply on each chunk.
 *
 * Anything added here must stay referentially stable; an entry that really
 * depends on props or state belongs in a `useMemo`, not inline in the JSX.
 */
const ASSISTANT_PART_COMPONENTS = {
  Text: MarkdownText,
  Reasoning: Reasoning,
  ReasoningGroup: ReasoningGroup,
  Source: Sources,
  ToolGroup: ToolGroup,
  tools: {
    by_name: {
      web_search: WebSearchToolUIConfirmable,
      search_knowledge_base: KnowledgeBaseToolUIConfirmable,
      python: PythonToolUIConfirmable,
      terminal: TerminalToolUIConfirmable,
      code_execution: CodeExecutionToolUIConfirmable,
      image_generation: ImageGenerationToolUIConfirmable,
      render_html: RenderHtmlToolUIConfirmable,
    },
    Fallback: ToolFallbackConfirmable,
  },
} as const;

// Live in-place denoising canvas for DiffusionGemma: while generating, render the
// latest per-step canvas snapshot in the bubble so the user watches the answer resolve
// out of noise. Transient (store-only, cleared on run end), so the finished message
// keeps only the committed markdown.
const DiffusionCanvas: FC = () => {
  const isRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  // Only this conversation's own frames render here; a first turn has no id yet, so it reads
  // "__default", which is where its run files them until the thread persists.
  const threadKey =
    useAuiState(({ threadListItem }) => threadListItem.remoteId) ?? "__default";
  // A canvas is set only by diffusion_frame events, so its presence is a sufficient gate;
  // loadedIsDiffusion can lag the first frame on a fresh load.
  const canvas = useChatRuntimeStore(
    (s) => s.activeDiffusionCanvasByThreadId[threadKey],
  );
  if (!isRunning || !canvas) {
    return null;
  }
  const stepLabel =
    canvas.total > 0 ? `step ${canvas.step + 1}/${canvas.total}` : "denoising";
  return (
    <div className="aui-diffusion-canvas my-1.5 overflow-hidden rounded-lg border border-primary/20 bg-primary/[0.03]">
      <div className="flex items-center gap-2 border-b border-primary/10 px-3 py-1.5 text-ui-11 font-medium text-primary/80">
        <span className="inline-block size-1.5 animate-pulse rounded-full bg-primary" />
        <span>Denoising</span>
        <span className="opacity-60">
          block {canvas.block + 1} - {stepLabel}
        </span>
      </div>
      <pre className="max-h-[60dvh] overflow-auto whitespace-pre-wrap px-3 py-2 font-mono text-ui-12p5 leading-relaxed text-foreground/90">
        {canvas.text}
      </pre>
    </div>
  );
};

/**
 * Mounts an autohidden action bar while focus is inside the message, the way hovering it does.
 *
 * `autohide="not-last"` UNMOUNTS every bar but the newest reply's, so Copy, Edit, Refresh,
 * Delete, Read aloud and More leave the tab order on older messages and a keyboard or screen
 * reader user has no way back: `:focus-within` in CSS cannot help, there is nothing to style.
 * The reveal has to be JS, and it drives `message.setIsHovering`, the same flag the library's
 * own `mouseenter`/`mouseleave` (MessagePrimitive.Root) writes and the only input to
 * `useActionBarFloatStatus` besides the More menu's interaction lock. Reusing it rather than
 * layering a second visibility source is what keeps the two from disagreeing.
 *
 * One flag, two writers, so the two clobber each other unless this hook covers both crossings:
 *   - pointer leaves while focus is inside (a Tab that scrolls the message under a parked
 *     cursor does exactly this): the library clears the flag, which would unmount the element
 *     that currently has focus. `reassert` below sets it back inside the same event.
 *   - focus leaves while the pointer is still over the message: clearing would unmount a bar
 *     the user is pointing at, and no second `mouseenter` is coming. The `:hover` test defers
 *     to the library's own `mouseleave` instead.
 */
function useActionBarFocusReveal() {
  const aui = useAui();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const focusWithinRef = useRef(false);
  const clearFrameRef = useRef<number | null>(null);

  // The More menu is portaled OUTSIDE the message, so focus entering it looks like a blur.
  // Its own interaction lock keeps the bar mounted meanwhile, but the trigger this hook has to
  // hand focus back to lives in that bar, so a popup this message owns counts as engaged.
  // Scoped to the action bar, NOT to every expanded descendant. Reasoning and tool cards are
  // Radix CollapsibleTriggers and render aria-expanded="true" while open, which is the resting
  // state of a message whose tool output the reader has expanded. An unscoped lookup treated
  // those as an open popup, so `decide` rescheduled itself every frame for as long as the
  // disclosure stayed open, held focusWithinRef and the synthetic hover set, and left the bar
  // mounted: a per-frame DOM query per such message, which is the slowdown this branch removes.
  const openPopupTrigger = useCallback(
    () =>
      rootRef.current?.querySelector(
        '.aui-assistant-action-bar-root [aria-expanded="true"]',
      ) ?? null,
    [],
  );

  const isEngaged = useCallback(() => {
    const el = rootRef.current;
    if (!el) return false;
    const active = document.activeElement;
    if (active && el.contains(active)) return true;
    return openPopupTrigger() !== null;
  }, [openPopupTrigger]);

  const cancelPendingClear = useCallback(() => {
    if (clearFrameRef.current !== null) {
      cancelAnimationFrame(clearFrameRef.current);
      clearFrameRef.current = null;
    }
  }, []);

  /**
   * Decide, a frame from now, whether focus has really left, and keep asking until it has.
   *
   * Deferred rather than read off `relatedTarget`: that is null both for focus going to the
   * browser chrome and for focus entering a portal, and it says nothing at all when the
   * focused element is REMOVED, which is how a menu closes and which Chrome reports with no
   * focusout event whatsoever. Reading `document.activeElement` a frame later answers all of
   * them. Clearing late costs a frame of a mounted bar; clearing early destroys the element
   * the user is on, so late is the safe direction.
   */
  const scheduleClear = useCallback(
    (restart: boolean) => {
      if (clearFrameRef.current !== null) {
        if (!restart) return;
        cancelAnimationFrame(clearFrameRef.current);
      }
      const decide = () => {
        clearFrameRef.current = null;
        const el = rootRef.current;
        if (!el || !focusWithinRef.current) return;
        const active = document.activeElement;
        if (active && el.contains(active)) return;
        if (openPopupTrigger()) {
          // Focus is in this message's own portaled menu, whose interaction lock is holding
          // the bar open anyway. Deciding now would be wrong and deciding never would pin the
          // bar open for good, so ask again next frame; the loop lasts only as long as the
          // menu is open on this one message.
          clearFrameRef.current = requestAnimationFrame(decide);
          return;
        }
        focusWithinRef.current = false;
        if (!el.matches(":hover")) {
          aui.message().setIsHovering(false);
        }
      };
      clearFrameRef.current = requestAnimationFrame(decide);
    },
    [aui, openPopupTrigger],
  );

  // onFocus/onBlur on a container are focusin/focusout in React, so they give focus-within.
  const handleFocus = useCallback(
    (event: ReactFocusEvent<HTMLDivElement>) => {
      const el = rootRef.current;
      const target = event.target as Node | null;
      if (el && target && !el.contains(target)) {
        // React bubbles focus events out of PORTALS along the React tree, so this is this
        // message's own menu, rendered into document.body. Focus is not in the subtree, so do
        // not cancel the watchdog -- the menu will take focus with it when it unmounts, and
        // that removal fires no focusout to wake us up again.
        scheduleClear(false);
        return;
      }
      cancelPendingClear();
      if (focusWithinRef.current) return;
      focusWithinRef.current = true;
      aui.message().setIsHovering(true);
    },
    [aui, cancelPendingClear, scheduleClear],
  );

  const handleBlur = useCallback(() => {
    if (!focusWithinRef.current) return;
    scheduleClear(true);
  }, [scheduleClear]);

  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    // From an effect on purpose: MessagePrimitive.Root binds its own mouseleave from a ref
    // callback, which commits before effects run, so this listener is registered second and
    // runs second on the same element. Both writes land in one dispatch, the store settles on
    // `true`, and React never renders the intermediate `false` -- so the bar does not unmount
    // and the focused control is not destroyed under the user.
    const reassert = () => {
      if (focusWithinRef.current && isEngaged()) {
        aui.message().setIsHovering(true);
      }
    };
    el.addEventListener("mouseleave", reassert);
    return () => {
      el.removeEventListener("mouseleave", reassert);
      cancelPendingClear();
    };
  }, [aui, isEngaged, cancelPendingClear]);

  return { ref: rootRef, onFocus: handleFocus, onBlur: handleBlur };
}

const ResearchMessageRunIdContext = createContext<string | null>(null);

/**
 * AssistantMessage handles the display and inline-editing of AI responses.
 *
 * It utilizes a "Tagged Text" system (<THINK> and <TOOL> tags) to allow users
 * to edit structured reasoning and tool outputs within a plain-text textarea
 * while preserving the underlying data schema and tool-call metadata.
 */
const AssistantMessage: FC = () => {
  const aui = useAui();
  const focusReveal = useActionBarFocusReveal();
  const messageId = useAuiState(({ message }) => message.id);
  const messageContent = useAuiState(({ message }) => message.content);
  const metadataResearchRunId = useAuiState(({ message }) =>
    getResearchRunId(message.metadata),
  );
  const boundResearchAssistantMessageId = useResearchRunStore((state) =>
    metadataResearchRunId
      ? state.sessions[metadataResearchRunId]?.run?.assistantMessageId
      : undefined,
  );
  const researchRunId =
    metadataResearchRunId &&
    researchReplyOwnsRun(boundResearchAssistantMessageId, messageId)
      ? metadataResearchRunId
      : null;
  // Persisted on the assistant turn that compacted, so the notice survives a reload.
  const contextTruncation = useAuiState(({ message }) => {
    const custom = (
      message.metadata as
        | { custom?: { contextTruncation?: unknown } }
        | undefined
    )?.custom;
    const value = custom?.contextTruncation;
    return value && typeof value === "object"
      ? (value as ContextTruncation)
      : null;
  });
  // Once a thread outgrows the window every request runs the fit, so "this turn
  // compacted" is true of every later reply and would put a notice on all of them. What
  // matters is when MORE of the conversation fell out of view: the eviction boundary
  // rising above the last turn that reported one. Between moves the model sees the same
  // history, so there is nothing new to say.
  const showsNotice = useAuiState(({ thread }) => {
    let previousDropped = 0;
    for (const message of thread.messages) {
      if (message.role !== "assistant") continue;
      const value = (
        message.metadata as
          | { custom?: { contextTruncation?: unknown } }
          | undefined
      )?.custom?.contextTruncation as ContextTruncation | undefined;
      const dropped = compactionBoundary(value);
      if (dropped > previousDropped) {
        if (message.id === messageId) return true;
        previousDropped = dropped;
      } else if (message.id === messageId) {
        return false;
      }
    }
    return false;
  });
  const incognito = useChatRuntimeStore((s) => s.incognito);

  // Use global store for editing state to ensure a single source of truth
  const editingId = useChatRuntimeStore((s) => s.editingMessageId);
  const setEditingId = useChatRuntimeStore((s) => s.setEditingMessageId);
  const isEditing = editingId === messageId;

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow textarea height based on content
  const adjustHeight = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight}px`;
    }
  };

  useEffect(() => {
    if (isEditing) setTimeout(adjustHeight, 0);
  }, [isEditing]);

  const handleSave = async () => {
    const finalText = textareaRef.current?.value || "";

    // Prioritize the specific thread item ID, then fallback to the global active thread ID
    const remoteId = aui.threadListItem().getState().remoteId
                  || useChatRuntimeStore.getState().activeThreadId;

    if (!remoteId || remoteId === "" || remoteId === "/") {
      toast.error("Save failed: No thread ID found.");
      setEditingId(null);
      return;
    }

    try {
      await updateThreadMessage({
        thread: {
          export: () => aui.thread().export(),
          import: (data) => aui.thread().import(data)
        },
        messageId,
        remoteId,
        newText: finalText,
        isIncognito: incognito,
      });
    } catch (error) {
      console.error("UI: Error during save:", error);
      toast.error("Failed to save message edits.");
    } finally {
      setEditingId(null);
    }
  };

  return (
    <ResearchMessageRunIdContext.Provider value={researchRunId}>
      <MessagePrimitive.Root
      className="group/assistant-message aui-assistant-message-root relative mx-auto min-w-0 w-full max-w-(--thread-content-max-width) pt-0.5 pb-4 text-ui-15p5 [font-weight:410] tracking-[0.01em] dark:tracking-[0.02em]"
      data-role="assistant"
      // The message itself is the tab stop that lets the reveal below fire. Without it, a reply
      // whose body is plain prose -- no link, no image, no code fence and so not even
      // Streamdown's per-fence Copy button -- contains nothing focusable once `autohide` has
      // unmounted its action bar, and Tab has no way into the message at all: Copy, Edit,
      // Delete and More are unreachable for the whole thread except its newest reply.
      // A tabIndex rather than a visually hidden button on purpose: it adds no DOM node (this
      // PR exists to cut per-message weight) and it draws nothing at rest. The app's own
      // `:focus-visible` rule in index.css gives it the same soft 1px keyboard indicator every
      // other focusable container gets, and `:focus-visible` means a mouse click on a reply
      // still draws nothing.
      tabIndex={0}
      ref={focusReveal.ref}
      onFocus={focusReveal.onFocus}
      onBlur={focusReveal.onBlur}
    >
      <div className="aui-assistant-message-content wrap-break-word min-w-0 text-[#0d0d0d] dark:text-foreground leading-relaxed">
        {contextTruncation && showsNotice && !isEditing && (
          <CompactionNotice truncation={contextTruncation} />
        )}
        {isEditing ? (
          <div className="flex flex-col gap-2 w-full">
            <textarea
              ref={textareaRef}
              defaultValue={extractTaggedText(messageContent)}
              className="w-full p-3 rounded-xl bg-muted border border-border text-foreground focus:ring-1 focus:ring-ring outline-none overflow-y-auto resize-none font-mono text-sm max-h-[70dvh]"
              autoFocus
              onInput={adjustHeight}
              onKeyDown={(e) => {
                e.stopPropagation();
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  handleSave();
                }
                if (e.key === 'Escape') {
                  setEditingId(null); // UX: Close editor on Escape
                }
              }}
            />
            <div className="flex justify-end gap-2">
              <Button size="sm" variant="ghost" onClick={() => setEditingId(null)} className="h-8 text-xs">Cancel</Button>
              <Button size="sm" onClick={handleSave} className="h-8 text-xs">Save</Button>
            </div>
          </div>
        ) : (
          <>
            <div className="pointer-events-none relative h-0 min-w-0">
              <MessageResponseModelBadge className="absolute -top-6 left-0 max-w-[min(22rem,100%)]" />
            </div>
            {researchRunId ? (
              <ResearchMessage />
            ) : (
              <>
                <GeneratingIndicator />
                <CancelledIndicator />
                <DiffusionCanvas />

            {/*
                We use the standard MessagePrimitive.Parts. This ensures that
                edited messages maintain the same professional styling,
                Markdown rendering, and tool-call components as original responses.
            */}
                <MessagePrimitive.Parts components={ASSISTANT_PART_COMPONENTS} />
                <SourcesGroup />
                <RagSourcesGroup />
                <MessageHtmlArtifacts />
                <ContinueMessageBar />
              </>
            )}
            <MessageError />
          </>
        )}
      </div>

      <div className="aui-assistant-message-footer mt-1.5 -ml-[var(--icon-btn-inset)] flex min-h-8">
        <BranchPicker className="mr-0.5" />
        <AssistantActionBar />
      </div>
      {/* Renders nothing. `If last` keeps the hook off the other N-1. */}
      <MessagePrimitive.If last={true}>
        <ForkChatShortcut />
      </MessagePrimitive.If>

      {/*
        The same reveal, for the other traversal direction.

        The tabIndex on the root above only works going FORWARD. A container is reached before its
        own descendants, so Shift+Tab arriving from the message below lands on the last tabbable
        thing in this message -- and with the bar unmounted that is the root, which sits BEFORE
        the bar in DOM order. Focusing it mounts the controls and then the next Shift+Tab steps
        past them to the previous message, so Copy, Edit, Delete and More are reachable going
        forward and unreachable going backward.

        A sentinel AFTER the bar is what makes the backward pass land inside the message: focus
        stops here, the bar mounts, and the next Shift+Tab goes into the last control rather than
        out of the message. Deliberately not a focus redirect to that control, which would trap
        the forward pass in a loop between the last button and this element.

        A span with no content rather than a visually hidden button: it is one node with no text,
        which matters in a PR whose point is per-message weight, and it draws nothing at rest for
        the same :focus-visible reason as the root.

        No onFocus/onBlur of its own: React's onFocus is focusin and bubbles, so focus landing
        here already reaches the root's handler and mounts the bar. Duplicating them would run
        the same handler twice per focus change for no effect. No role either -- it performs no
        action, so labelling it as a button would misdescribe it; the aria-label is what stops it
        being an unannounced stop for a screen reader.
      */}
      <span
        className="aui-assistant-reveal-sentinel"
        tabIndex={0}
        aria-label="Message actions"
      />
      </MessagePrimitive.Root>
    </ResearchMessageRunIdContext.Provider>
  );
};

const COPY_RESET_MS = 2000;

/**
 * One fork-count subscription for as long as the thread is on screen.
 *
 * The badges below sit inside action bars that autohide, so at rest at most the newest reply
 * has one, and none at all while the thread is running or while its last message is a prompt.
 * Left to them, the last badge leaving would drop the thread's counts and the next hover would
 * fetch them all again -- one whole-thread request per message the pointer crosses, with the
 * badge arriving a round trip after the bar it sits in.
 */
const useThreadForkCounts = (): void => {
  const remoteId =
    useAuiState(({ threadListItem }) => threadListItem.remoteId) ?? null;
  useEffect(() => {
    if (!remoteId) return;
    return subscribeForkCounts(remoteId, () => {});
  }, [remoteId]);
};

const ForkCountBadge: FC = () => {
  const remoteId =
    useAuiState(({ threadListItem }) => threadListItem.remoteId) ?? null;
  const messageId = useAuiState(({ message }) => message.id);
  const subscribe = useCallback(
    (onChange: () => void) =>
      remoteId ? subscribeForkCounts(remoteId, onChange) : () => {},
    [remoteId],
  );
  const getSnapshot = useCallback(
    () => (remoteId ? forkCountFor(remoteId, messageId) : 0),
    [remoteId, messageId],
  );
  const count = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  if (count <= 0) return null;
  return (
    <span
      className="mx-1 inline-flex items-center gap-1 rounded-sm bg-primary/10 px-1.5 py-0.5 text-ui-10 font-medium text-primary"
      title={`${count} fork${count === 1 ? "" : "s"} from this message`}
    >
      <GitBranchIcon strokeWidth={1.75} className="size-3" />
      {count}
    </span>
  );
};

/**
 * One fork at a time, across every caller of the hook below.
 *
 * The chord and the button each hold their own instance, so a `useState` flag
 * only disables the one that was used: pressing the chord and then clicking
 * Fork before the first request lands would post two, each with its own new
 * thread id, and race their navigations. A store is what both of them read.
 */
const useForkInFlight = create<{
  forking: boolean;
  setForking: (forking: boolean) => void;
}>((set) => ({
  forking: false,
  setForking: (forking) => set({ forking }),
}));

const useForkMessageAction = () => {
  const aui = useAui();
  const navigate = useNavigate();
  const messageId = useAuiState(({ message }) => message.id);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const pending = useForkInFlight((s) => s.forking);
  const setPending = useForkInFlight((s) => s.setForking);

  const handleFork = async () => {
    // Read, do not trust the render: two handlers can run in one tick, before
    // either sees the other's state.
    if (useForkInFlight.getState().forking) return;
    const remoteId = aui.threadListItem().getState().remoteId;
    if (!remoteId) {
      toast.error("Cannot fork an unsaved chat");
      return;
    }
    setPending(true);
    try {
      // The fork copies settings_json inside its own transaction, so anything not yet
      // in the row is not in the copy: a pill toggled moments ago and still in the
      // 400ms debounce, or one held because this chat's own read has not landed. The
      // fork would otherwise open on the modes the chat had before, not the ones on
      // screen when it was made.
      try {
        await settleThreadScopedSettingsForCopy(remoteId);
      } catch {
        // The row does not hold what is on screen, so a fork made now would carry the
        // pre-edit modes and look like it had lost the change. Better to say so.
        toast.error("Could not fork this chat", {
          description:
            "Its settings could not be saved, so the fork would not match. Please retry.",
        });
        return;
      }
      const result = await forkChatThread(remoteId, {
        messageId,
        newThreadId: crypto.randomUUID(),
        createdAt: Date.now(),
      });
      useChatRuntimeStore.getState().setActiveThreadId(result.thread.id);
      navigate({
        to: "/chat",
        search: { thread: result.thread.id },
        replace: false,
      });
      if (result.containerSnapshotWarning) {
        toast.info("Fork created", {
          description: result.containerSnapshotWarning,
        });
      } else {
        toast.success("Fork created");
      }
    } catch (error) {
      console.error("Failed to fork", error);
      toast.error("Failed to fork", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setPending(false);
    }
  };

  return {
    forkMessage: handleFork,
    forkDisabled: isRunning || pending,
  };
};

/**
 * The chord's registration, which no action bar can hold.
 *
 * The button below is the user bar's, and that bar is `autohide="always"`, so
 * ActionBarPrimitive.Root returns null and takes the registration with it on
 * every message that is not hovered. The assistant bar has its own fork call
 * and never mounts the button at all, so on a thread that ended the ordinary
 * way, with a reply, no message carried the chord.
 *
 * Mounted from both message roots under `If last`, so it exists once, for
 * whichever message is last, whatever its role and wherever the pointer is.
 */
const ForkChatShortcut: FC = () => {
  const { forkMessage, forkDisabled } = useForkMessageAction();
  const chatActive = useChatActive();
  // Compare mounts a thread in each pane, and the chord would go to whichever
  // registered first. Fork from the button there.
  const inComparePane = useInComparePane();
  useShortcut(
    "forkChat",
    () => {
      // `chatActive` is the visible tab, not the foreground, so a dialog over
      // Chat would otherwise fork the conversation behind it.
      if (!isSurfaceInForeground(COMPOSER_INPUT_SELECTOR)) return;
      void forkMessage();
    },
    { enabled: chatActive && !inComparePane && !forkDisabled },
  );
  return null;
};

const ForkMessageButton: FC = () => {
  const { forkMessage, forkDisabled } = useForkMessageAction();

  return (
    <TooltipIconButton
      tooltip="Fork from here"
      disabled={forkDisabled}
      onClick={forkMessage}
    >
      <GitBranchIcon strokeWidth={1.75} className="size-icon" />
    </TooltipIconButton>
  );
};

const getResearchRunId = (metadata: unknown): string | null => {
  const custom = (
    metadata as
      | {
          custom?: {
            researchRunId?: unknown;
            researchRun?: { id?: unknown };
          };
        }
      | undefined
  )?.custom;
  const runId = custom?.researchRunId ?? custom?.researchRun?.id;
  return typeof runId === "string" ? runId : null;
};

const useResearchMessageRunId = () => {
  return useContext(ResearchMessageRunIdContext);
};

// Boolean(), not `!== null`: getResearchRunId returns whatever string it found, and an empty one
// counted as "no research reply" before. Keeping that stops an empty id hiding a message's edit
// and delete controls.
const hasResearchRunId = (metadata: unknown): boolean =>
  Boolean(getResearchRunId(metadata));

const useOwnsResearchMessage = () => {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  // The ANSWER is selected, not the message array: selecting the array subscribed every user
  // message's action bar (and its tooltips) to every thread change, so one delete re-rendered all
  // of them even when the answer had not moved. The export is shared across one revision.
  return useAuiState(({ thread }) => {
    if (thread.messages.length === 0) {
      return false;
    }
    return researchReplyOwners(
      thread.messages,
      () => aui.thread().export().messages,
      hasResearchRunId,
    ).has(messageId);
  });
};

// Whether the active thread has a non-terminal durable research run. After a reload the
// research store follows the run instead of an assistant-ui run, so `thread.isRunning` is
// false while research is active; edit/reload/branch must also gate on this to keep
// one run per chat.
const useThreadResearchActive = (): boolean => {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  return useResearchRunStore((state) => {
    const runId = activeThreadId
      ? state.latestRunByThreadId[activeThreadId]
      : undefined;
    const run = runId ? state.sessions[runId]?.run : undefined;
    return Boolean(
      run && !["completed", "failed", "cancelled"].includes(run.status),
    );
  });
};

const DeleteMessageButton: FC = () => {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const researchRunId = useResearchMessageRunId();
  const ownsResearchMessage = useOwnsResearchMessage();

  const handleDelete = async () => {
    const thread = aui.thread();
    // Deleting a message, and for a user prompt its cascaded assistant replies,
    // unmounts their only Stop reading control. Stop read-aloud first when the
    // spoken message is among those removed. Read speech state at click time and
    // guard the call, which throws if playback already ended.
    const speakingId = thread.getState().speech?.messageId;
    if (speakingId) {
      const { messages } = thread.export();
      const target = messages.find(({ message }) => message.id === messageId);
      const removed = new Set<string>([messageId]);
      if (target?.message.role === "user") {
        for (const { parentId, message } of messages) {
          if (parentId === messageId && message.role === "assistant") {
            removed.add(message.id);
          }
        }
      }
      if (removed.has(speakingId)) {
        try {
          thread.stopSpeaking();
        } catch {
          // Playback ended between reading the state and stopping it.
        }
      }
    }

    const remoteId = aui.threadListItem().getState().remoteId;
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

  if (researchRunId || ownsResearchMessage) {
    return null;
  }

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
    // getCopyText reads content only, and a long paste sits in an attachment.
    const pasted = attachmentsPastedText(aui.message().getState().attachments);
    // The image tokens are renderer markup, not prose: strip them or the clipboard
    // gets `[[img:0123456789ab]]` where the picture was.
    const text = [stripSearchImageTokens(aui.message().getCopyText()), pasted]
      .filter((part) => part.length > 0)
      .join("\n\n");
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

const EditAssistantMessageButton: FC = () => {
  const messageId = useAuiState(({ message }) => message.id);
  const researchRunId = useResearchMessageRunId();
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const researchActive = useThreadResearchActive();
  const setEditingId = useChatRuntimeStore((s) => s.setEditingMessageId);

  if (researchRunId) return null;

  return (
    <TooltipIconButton
      tooltip="Edit response"
      disabled={isRunning || researchActive}
      onClick={() => setEditingId(messageId)}
    >
      <HugeiconsIcon
        icon={Edit03Icon}
        strokeWidth={1.75}
        className="size-icon"
      />
    </TooltipIconButton>
  );
};

async function exportMessageMarkdown(content: string): Promise<void> {
  try {
    await downloadFile(
      // Same rule as the copy button and the whole-chat export: the tokens are
      // renderer markup, so a saved answer must not carry them as prose.
      stripSearchImageTokens(content),
      `message-${Date.now()}.md`,
      "text/markdown",
    );
  } catch (error) {
    if (!isDownloadCancelled(error)) {
      toast.error("Could not save Markdown export.", {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }
}
const AssistantActionBar: FC = () => {
  const aui = useAui();
  const moreMenuTriggerRef = useRef<HTMLButtonElement>(null);
  const { forkMessage, forkDisabled } = useForkMessageAction();
  const researchRunId = useResearchMessageRunId();
  const researchActive = useThreadResearchActive();
  const activeProjectId = useChatRuntimeStore((s) => s.activeProjectId);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const ttsEnabled = useVoiceSettingsStore((s) => s.ttsEnabled);
  // hideWhenRunning is thread-level, so a new run would hide this bar and its
  // only Stop reading control while read-aloud keeps playing; keep it shown.
  const speaking = useAuiState(({ message }) => message.speech != null);

  return (
    <>
      <ActionBarPrimitive.Root
        hideWhenRunning={!speaking}
        // Unmounts the bar on every message that is not hovered, as the user bar already does.
        // Mounted, each one holds ~8 tooltips subscribed to the global modal-layer store, so
        // every menu open fanned out across the whole thread.
        //
        // "not-last", not "always": an unmounted bar is out of the tab order too, and these are
        // the only Copy, Refresh, Read aloud and More controls a message has. The newest reply
        // keeps its bar, so a keyboard user still reaches the message they are acting on, and
        // the other N-1 still go: 8 tooltip subscriptions on a 500-message thread instead of
        // ~250. "never" while speaking because this bar carries the only Stop reading control,
        // which neither hover nor a later reply must take away.
        //
        // The older N-1 are deferred, not lost: useActionBarFocusReveal on the message root
        // remounts a bar when focus enters that message, so tabbing brings back what hovering
        // brings back and the controls return to the accessibility tree with it.
        autohide={speaking ? "never" : "not-last"}
        className="aui-assistant-action-bar-root col-start-3 row-start-2 flex items-center gap-1 text-chat-icon-fg [&_button:not([data-slot=message-timing-trigger])]:size-8 [&_button]:!rounded-full [&_button:hover]:bg-chat-icon-bg-hover [&_button:hover]:text-chat-icon-fg-hover"
      >
        <CopyButton />
        <EditAssistantMessageButton />
        {!researchRunId && !researchActive && (
          <ActionBarPrimitive.Reload asChild={true}>
            <TooltipIconButton tooltip="Refresh">
              <RefreshCwIcon strokeWidth={1.75} className="size-icon" />
            </TooltipIconButton>
          </ActionBarPrimitive.Reload>
        )}
        <ForkCountBadge />
        <DeleteMessageButton />
        {ttsEnabled && (
          <MessagePrimitive.If speaking={false}>
            <ActionBarPrimitive.Speak asChild={true}>
              <TooltipIconButton tooltip="Read aloud" aria-label="Read aloud">
                <Volume2Icon strokeWidth={1.75} className="size-icon" />
              </TooltipIconButton>
            </ActionBarPrimitive.Speak>
          </MessagePrimitive.If>
        )}
        {/* Not gated on ttsEnabled: turning the setting off while a message
            is being read aloud must not remove the only stop control. */}
        <MessagePrimitive.If speaking={true}>
          <ActionBarPrimitive.StopSpeaking asChild={true}>
            <TooltipIconButton
              tooltip="Stop reading"
              aria-label="Stop reading"
              className="text-destructive"
            >
              <VolumeXIcon strokeWidth={1.75} className="size-icon" />
            </TooltipIconButton>
          </ActionBarPrimitive.StopSpeaking>
        </MessagePrimitive.If>
        {/* Non-modal: a modal Radix menu writes `pointer-events: none` on <body>, and
            that is an INHERITED property, so every open invalidates style for the whole
            document. On a long thread that recalc is the bulk of the open+close cost. */}
        <ActionBarMorePrimitive.Root modal={false}>
          <ActionBarMorePrimitive.Trigger asChild={true}>
            <TooltipIconButton
              ref={moreMenuTriggerRef}
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
            className="aui-action-bar-more-content z-50 min-w-32 overflow-hidden rounded-[21px] bg-popover px-[9px] py-2 text-popover-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-none"
          >
            {/* Prevent an outside dismissal from triggering Delete. */}
            <MenuDismissGuard triggerRef={moreMenuTriggerRef} />
            <ActionBarMorePrimitive.Item
              disabled={forkDisabled}
              onSelect={() => void forkMessage()}
              className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-[12px] px-3 py-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
            >
              <GitBranchIcon strokeWidth={1.75} className="size-icon" />
              Fork in new chat
            </ActionBarMorePrimitive.Item>
            <ActionBarPrimitive.ExportMarkdown
              asChild={true}
              onExport={exportMessageMarkdown}
            >
              <ActionBarMorePrimitive.Item className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-[12px] px-3 py-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                <HugeiconsIcon
                  icon={Download01Icon}
                  strokeWidth={1.75}
                  className="size-icon"
                />
                Export as markdown
              </ActionBarMorePrimitive.Item>
            </ActionBarPrimitive.ExportMarkdown>
            {activeProjectId && (
              <ActionBarMorePrimitive.Item
                onSelect={() => {
                  // Not getCopyText: it joins text parts alone, so a reply's
                  // reasoning, tool calls and citations would be dropped and a
                  // tool-only reply would read as empty. Same conversion the
                  // whole-chat save runs.
                  // Stripped: a project source is retrieved back into context, so
                  // saved tokens would teach the model ids that resolve to nothing.
                  const text = stripSearchImageTokens(
                    replySourceMarkdown(
                      aui.message().getState().content,
                      toolResultModelText,
                    ),
                  );
                  if (!text.trim()) {
                    toast.info("No content to save.");
                    return;
                  }
                  const state = aui.threadListItem().getState();
                  // The list item's title belongs to the whole chat, so mark the
                  // reply apart or saving both lists two identical names.
                  const title = state.title ? `${state.title} - reply` : "reply";
                  // activeProjectId can lag a thread switch while the stored
                  // thread loads; resolve the destination from this thread.
                  const remoteId =
                    state.remoteId ||
                    useChatRuntimeStore.getState().activeThreadId;
                  void (async () => {
                    const thread = remoteId
                      ? await getStoredChatThread(remoteId).catch(() => null)
                      : null;
                    if (!thread?.projectId) {
                      toast.info("This chat isn't in a project.");
                      return;
                    }
                    await saveMarkdownAsProjectSource(
                      thread.projectId,
                      text,
                      title,
                    );
                  })();
                }}
                className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-[12px] px-3 py-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
              >
                <HugeiconsIcon
                  icon={BookOpen01Icon}
                  strokeWidth={1.75}
                  className="size-icon"
                />
                Save to project sources
              </ActionBarMorePrimitive.Item>
            )}
            <ActionBarMorePrimitive.Item
              onSelect={() => setDetailsOpen(true)}
              className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-[12px] px-3 py-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
            >
              <HugeiconsIcon
                icon={HelpCircleIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
              See response details
            </ActionBarMorePrimitive.Item>
          </ActionBarMorePrimitive.Content>
        </ActionBarMorePrimitive.Root>
        <MessageTiming side="top" className="h-8 px-2" />
      </ActionBarPrimitive.Root>
      <MessageResponseDetailsSheet
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
      />
    </>
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
      className="aui-user-message-root fade-in slide-in-from-bottom-1 mx-auto flex w-full max-w-(--thread-content-max-width) animate-in flex-col items-end gap-y-2 pt-6 pb-4 text-ui-15p5 [font-weight:410] tracking-[0.01em] dark:tracking-[0.02em] duration-150"
      data-role="user"
    >
      <UserMessageAttachments />
      <UserMessageAudio />

      <div className="aui-user-message-content-wrapper flex max-w-[80%] min-w-0 flex-col items-end">
        <div className="aui-user-message-content wrap-break-word w-fit max-w-full rounded-[24px] bg-[#f5f5f5] px-4 py-2.5 text-[#0d0d0d] dark:text-foreground dark:bg-card">
          <MessagePrimitive.Parts />
        </div>
        <div className="mt-1 -mr-[var(--icon-btn-inset)] flex min-h-8 items-center">
          <UserActionBar />
          <BranchPicker className="aui-user-branch-picker ml-0.5" />
        </div>
      </div>
      {/* The other half of the pair: last is a user message while a reply is
          still to come, or once one has been deleted. */}
      <MessagePrimitive.If last={true}>
        <ForkChatShortcut />
      </MessagePrimitive.If>
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  const ownsResearchMessage = useOwnsResearchMessage();
  const researchActive = useThreadResearchActive();
  return (
    <ActionBarPrimitive.Root
      autohide="always"
      className="aui-user-action-bar-root flex gap-1 text-chat-icon-fg [&_button]:size-8 [&_button]:!rounded-full [&_button:hover]:bg-chat-icon-bg-hover [&_button:hover]:text-chat-icon-fg-hover"
    >
      <CopyButton />
      {!ownsResearchMessage && !researchActive && (
        <ActionBarPrimitive.Edit asChild={true}>
          <TooltipIconButton tooltip="Edit" className="aui-user-action-edit">
            <HugeiconsIcon
              icon={Edit03Icon}
              strokeWidth={1.75}
              className="size-icon"
            />
          </TooltipIconButton>
        </ActionBarPrimitive.Edit>
      )}
      <ForkCountBadge />
      <ForkMessageButton />
      <DeleteMessageButton />
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  const aui = useAui();
  const { inputProps, isComposingRef } = useImeComposerInputHandlers();
  const resendAfterCancelRef = useRef(false);
  const researchActive = useThreadResearchActive();

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
            disabled={researchActive}
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
        "aui-branch-picker-root inline-flex items-center text-chat-icon-fg text-ui-13",
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
      <span className="aui-branch-picker-state font-mono text-ui-13 tabular-nums">
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
