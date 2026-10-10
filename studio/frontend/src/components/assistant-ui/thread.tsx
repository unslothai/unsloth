// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  canQueueTextAttachment,
  normalizeQueuedPrompt,
  prepareQueuedPromptFiles,
  queuedPromptHasContent,
  queuedPromptMessage,
  snapshotQueuedTextPrompt,
  type QueuedPrompt,
} from "@/features/chat/utils/queued-text-attachments";
import { getAuthSessionEpoch } from "@/features/auth";

import {
  ComposerAttachments,
  UserMessageAttachments,
} from "@/components/assistant-ui/attachment";
import {
  GeneratedImageOverlayProvider,
  useGeneratedImageOverlay,
} from "@/components/assistant-ui/generated-image-overlay-context";
import { CompactionNotice } from "@/components/assistant-ui/compaction-notice";
import { compactionNoticeMessageIds } from "@/components/assistant-ui/message-derived";
import type { ContextTruncation } from "@/features/chat/utils/context-truncation";
import { downloadImagePart } from "@/components/assistant-ui/image";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { MessageHtmlArtifacts } from "@/components/assistant-ui/message-html-artifacts";
import {
  MessageResponseDetailsSheet,
  MessageResponseModelBadge,
} from "@/components/assistant-ui/message-response-details-sheet";
import { ComposerDraftPreview } from "@/components/assistant-ui/composer-draft-preview";
import { PromptQueueList } from "@/components/assistant-ui/lazy-prompt-queue-list";
import { QueueResumeIcon } from "@/components/assistant-ui/queue-resume-icon";
import { ProgressiveMessages } from "@/components/assistant-ui/progressive-messages";
import { MessageMenuTime } from "@/components/assistant-ui/message-menu-time";
import { UserMessageActionBar, UserMessageFooter } from "@/components/assistant-ui/user-message-actions";
import { useActionBarFocusReveal } from "@/components/assistant-ui/use-action-bar-focus-reveal";
import { MessageTiming } from "@/components/assistant-ui/message-timing";
import { attachThreadFastCopy } from "@/components/assistant-ui/thread-fast-copy";
import { attachWheelHoverSuppression } from "@/components/assistant-ui/thread-wheel-hover";
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
import { ReadSkillToolUI } from "@/components/assistant-ui/tool-ui-read-skill";
import { SkillMentionPopover } from "@/components/assistant-ui/skill-mentions";
import { RenderHtmlToolUI } from "@/components/assistant-ui/tool-ui-render-html";
import { PythonToolUI } from "@/components/assistant-ui/tool-ui-python";
import { TerminalToolUI } from "@/components/assistant-ui/tool-ui-terminal";
import { WebSearchToolUI } from "@/components/assistant-ui/tool-ui-web-search";
import { ChatDictationBar } from "@/components/assistant-ui/chat-dictation-bar";
import {
  ChatAudioUploadMount,
  ChatSkillsDialog,
  composerSubmitIntent,
  composerFollowUpBehavior,
  composerShortcutLabels,
  composerKeyEventForImeSubmit,
  effectiveSendShortcut,
  imeKeydownBlocksComposerSubmit,
  followUpSubmitIntent,
  steeringInsertionIndex,
  cancelPreStreamRunForThreadIds,
  type ComposerSendShortcut,
  type ComposerSubmitIntent,
  type ComposerFollowUpBehavior,
  attachmentsPastedText,
  hasPendingPromptQueueStart,
  isPastedTextFile,
  pastedTextQueueKey,
  promptQueueActiveItemChanged,
  reorderPromptQueueItems,
  pasteClipboardFiles,
  extractYoutubeVideoUrlFromClipboard,
  pasteLongTextAsFile,
  isPlainPasteChord,
  plainPasteStillCounts,
  currentDictationEntryMode,
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
  YoutubeTranscriptPrompt,
  stripSearchImageTokens,
  useChatActive,
  useChatAudioUpload,
  useInComparePane,
  refreshSkillsCatalog,
  stopRecoveredRun,
  pythonToolRunsInStudio,
  withAttachmentOriginal,
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
  listPromptLists,
  type PromptEntry,
  type PromptListEntry,
} from "@/features/chat/api/prompts-api";
import { PromptCountBadge } from "@/features/chat/prompt-storage/prompt-count-badge";
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { useChatProjects } from "@/features/chat/hooks/use-chat-projects";
import { NewProjectDialog } from "@/features/chat/components/new-project-dialog";
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
import {
  attachLibraryChatFiles,
  useLibraryChatHandoffStore,
} from "@/features/library/chat-handoff-store";
import { resumesThought } from "@/features/model-picker";
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
  collectGeminiAnswerReplayParts,
  collectGeminiThoughtReplayParts,
  continuationGeminiReplayTurns,
} from "@/features/chat/gemini-thought-replay";
import {
  CONTINUATION_RUN_CONFIG_KEY,
  type ContinuationRequest,
  incompleteLabel,
  incompleteRemedy,
  isContinuableContent,
  isProviderReportedReason,
  modeAllowsContinuation,
  providerCompactionContinuationFields,
  readContinuationSource,
  readIncompleteInfo,
  readTextThoughtSignature,
  claimAutoContinue,
  forgetAutoContinue,
  recordAutoContinue,
  shouldAutoContinueMessage,
} from "@/features/chat/utils/continuation";
import {
  holdAutoContinueRun,
  watchAutoContinueRun,
} from "@/features/chat/utils/auto-continue-run-keeper";
import { McpComposerButton } from "@/features/chat/mcp-composer-button";
import { SkillsComposerButton } from "@/features/chat/skills-composer-button";
import { pickerAcceptForTextBasenames } from "@/features/chat/text-attachment-accept";
import {
  COMPOSER_INPUT_SELECTOR,
  isSurfaceInForeground,
  shortcutMatchingEvent,
  useKeyboardShortcutsStore,
  useShortcut,
  useSettingsDialogStore,
  isMacPlatform,
} from "@/features/settings";
import { FIND_SKIP_ATTRIBUTE } from "@/features/find-in-page";
import { translate, useT } from "@/i18n";
import {
  clampReasoningEffortToLevels,
  getExternalReasoningCapabilities,
  modelCatalogVersion,
  subscribeModelCatalog,
} from "@/features/chat/provider-capabilities";
import { useRagToolDisabled } from "@/features/chat/hooks/use-rag-tool-disabled";
import { BypassPermissionsMenuItem } from "@/features/chat/bypass-permissions-menu-item";
import { PermissionModeComposerPill } from "@/features/chat/permission-mode-select";
import {
  codeToolsOn,
  settleThreadScopedSettingsForCopy,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import {
  forkBoundaryAnchor,
  setForkBoundaryAnchor,
  useForkBoundaryStore,
} from "@/features/chat/stores/fork-boundary-store";
import {
  PROMPT_QUEUE_RUN_FAILED_EVENT,
  PROMPT_QUEUE_STOP_EVENT,
} from "@/features/chat/utils/prompt-queue-events";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { saveMarkdownAsProjectSource } from "@/features/rag";
import {
  PLUS_MENU_ORDER,
  CONVERSATION_MARKDOWN_LABEL,
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
  planUserPromptQueueStop,
  userStopTargetCancelMode,
  registerQueuedChatRunSettings,
  releasePreStreamRunReservation,
  reservePreStreamRun,
  subscribePreStreamRunReservations,
  claimThreadCreation,
  useChatProjectScope,
  shouldAbortPendingQueueForModelBoundary,
  shouldAbortPendingQueueForSettingsChange,
  resolveDeferredQueuedModelSettings,
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
  useForkInFlight,
  showForkCreatedToast,
  type PlusMenuItemId,
  usePlusMenuPrefsStore,
  writeComposerDraft,
  normalizeChatImage,
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
  readBackendChatThread,
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
import { InternetGlyph } from "@/lib/internet-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { MenuDismissGuard } from "@/lib/menu-dismiss-guard";
import { useWindowChromeCollisionPadding } from "@/lib/window-chrome";
import { NonModalDropdownMenu } from "@/components/ui/non-modal-dropdown-menu";
import { MicIcon } from "@/lib/mic-icon";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { Tick02Icon } from "@/lib/tick-icon";
import {
  BranchNextIcon,
  BranchPrevIcon,
  ContinueArrowIcon,
  EditResponseIcon,
  ReadAloudIcon,
} from "@/lib/action-bar-icons";
import { ForkIcon } from "@/lib/fork-icon";
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
  FolderAttachmentIcon,
  Folder01Icon,
  FolderAddIcon,
  Image03Icon,
  McpServerIcon,
  Scroll01Icon,
  Telescope02Icon,
  VolumeMute02Icon,
  WorkflowCircle05Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { RefreshGlyph } from "@/lib/refresh-icon";
import { useNavigate } from "@tanstack/react-router";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  ChevronDownIcon,
  Columns2Icon,
  SlidersHorizontalIcon,
  HeadphonesIcon,
  Loader2Icon,
  MoreHorizontalIcon,
  PlusIcon,
  SquareIcon,
  TerminalIcon,
  XIcon,
} from "lucide-react";
import {
  type ChangeEvent,
  type CompositionEvent,
  type ClipboardEvent,
  type CSSProperties,
  type FC,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  type ReactNode,
  type RefObject,
  Fragment,
  createContext,
  memo,
  useCallback,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { extractTaggedText, updateThreadMessage } from "@/features/chat/utils/update-thread-message";
import { useComposerPillFit } from "@/hooks/use-composer-pill-fit";
import { useIsMobile } from "@/hooks/use-mobile";
import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";

// True while a file is dragged anywhere over the chat page, so the composer
// can show its "Drop files here" affordance.
const PageDragContext = createContext(false);

/** The follow-up each of the two chords names. */
const FOLLOW_UP_SHORTCUTS = {
  queueMessage: "queue",
  steerMessage: "steer",
} as const satisfies Record<string, ComposerFollowUpBehavior>;

const FOLLOW_UP_SHORTCUT_IDS = Object.keys(
  FOLLOW_UP_SHORTCUTS,
) as (keyof typeof FOLLOW_UP_SHORTCUTS)[];

/** The behaviour a bound queue or steer chord names, if this event fires one. */
function followUpShortcutBehavior(event: {
  code: string;
  key?: string;
  metaKey: boolean;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
}): ComposerFollowUpBehavior | null {
  const id = shortcutMatchingEvent(
    useKeyboardShortcutsStore.getState().overrides,
    FOLLOW_UP_SHORTCUT_IDS,
    event,
  );
  return id ? FOLLOW_UP_SHORTCUTS[id] : null;
}

// module scope and runningByThreadId let queues survive Composer remounts and advance off-screen.
type PromptQueueTarget = {
  getDocumentThreadId: () => string | null;
  /** project captured when a new chat has no persisted row to read. */
  getQueueProjectId: () => string | null;
  /** a knowledge base replaces every other scope, project sources included. */
  usesKnowledgeBase: boolean;
  getRunningThreadIds: () => string[];
  isRunning: () => boolean;
  append: (prompt: QueuedPrompt) => void | Promise<void>;
  complete: () => void;
  cancel: () => void;
  cancelActiveRun: () => void;
  /** Forget the captured local model so a held prompt resolves the one loaded when it is sent. */
  releaseModel: () => void;
  isIndexing: () => boolean;
  usesThreadDocuments: boolean;
  usesLocalModel: boolean;
  usesDeepResearch: boolean;
  /** whether a research run now holds this queue's thread. */
  researchStarted: () => boolean;
  temporary: boolean;
  consumeDeepResearch: () => void;
};

type PromptQueueItem = QueuedPrompt & {
  id: string;
  target: PromptQueueTarget;
  dispatched: boolean;
  dispatchRetries: number;
  blockedByModelFailure?: boolean;
};

type PromptQueueRun = {
  id: string;
  items: PromptQueueItem[];
  index: number;
  generation: number;
  prevStoreRunning: boolean;
  waitingForTargetIdle: boolean;
  paused: boolean;
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
    const result = item.target.append(item);
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
  return getActivePromptQueueItem(run) === item && !item.blockedByModelFailure;
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
      !item.blockedByModelFailure &&
      !run.waitingForTargetIdle &&
      !run.paused &&
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
    const dispatchGeneration = run.generation;
    promptQueueDispatchingRunIds.add(run.id);
    dispatchQueuedPrompt(run, item, run.generation)
      .catch(() => undefined)
      .finally(() => {
        // Releasing a live attempt's flag makes Stop then Resume append twice.
        if (
          promptQueueRuns.get(run.id) === run &&
          dispatchGeneration !== run.generation
        ) {
          return;
        }
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
  // Keep prompts editable until the local model finishes loading.
  if (item.target.usesLocalModel && useChatRuntimeStore.getState().modelLoading) {
    scheduleQueuedPromptDispatch(run, item, PROMPT_QUEUE_DISPATCH_RETRY_MS);
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
  // recheck loading after the document probe.
  if (
    hasIndexingDocuments ||
    (item.target.usesLocalModel && useChatRuntimeStore.getState().modelLoading)
  ) {
    promptQueueActiveRunIds.delete(run.id);
    scheduleQueuedPromptDispatch(run, item, PROMPT_QUEUE_INDEXING_RETRY_MS);
    return;
  }
  if (!isActivePromptQueueItem(run, item, generation)) {
    return;
  }
  appendQueuedPrompt(run, item);
}

function createQueuedPrompt(prompt: QueuedPrompt, target: PromptQueueTarget) {
  return {
    id: createPromptQueueItemId(),
    ...prompt,
    target,
    dispatched: false,
    dispatchRetries: 0,
  };
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
  if (run.items[index]?.blockedByModelFailure) return "paused";
  if (run.paused && run.index >= 0 && index === activeItemIndex) {
    return "paused";
  }
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
      attachmentNames: item.attachments?.map((attachment) => attachment.name),
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
      paused: run.paused || run.items.some((item) => item.blockedByModelFailure),
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
  const match = findPromptQueueRunByItemId(itemId);
  if (!match) {
    return false;
  }
  const { item } = match;
  if (!canEditPromptQueueItem(item) || !queuedPromptHasContent({ ...item, prompt: nextPrompt })) {
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
  if (run.paused) {
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

function steerPromptQueueTarget(target: PromptQueueTarget) {
  const targetIds = getPromptQueueTargetIds(target);
  const run = findPromptQueueRunByTarget(target);
  const active = run && run.index >= 0 ? getActivePromptQueueItem(run) : null;
  const cancelledTarget = active?.dispatched ? active.target : null;
  pausePromptQueueRun(targetIds);
  cancelPreStreamRunForThreadIds(targetIds);
  try {
    // Pausing already cancels the dispatched target once.
    if (cancelledTarget !== target) target.cancelActiveRun();
  } catch {
    toast.info("Your follow-up is queued next", {
      description: "The current response could not be interrupted.",
    });
  }
  // Dispatch waits for cancellation to finish.
  resumePromptQueueRun(targetIds);
}

function steerPromptQueueItem(itemId: string) {
  const match = findPromptQueueRunByItemId(itemId);
  if (!match) return false;
  const { run, itemIndex, item } = match;
  if (
    item.dispatched ||
    itemIndex < Math.max(run.index, 0) ||
    getPromptQueueTargetIds(item.target).length === 0
  ) {
    return false;
  }
  if (item.target.researchStarted()) {
    toast.info("Research is still running", {
      description: "Stop research before steering with a queued prompt.",
    });
    return false;
  }
  // move the item so its captured settings and identity stay intact.
  run.items.splice(itemIndex, 1);
  run.items.splice(steeringInsertionIndex(run.items, run.index), 0, item);
  steerPromptQueueTarget(item.target);
  return true;
}

function startPromptQueue(
  items: Array<string | QueuedPrompt>,
  target: PromptQueueTarget,
  waitForCurrentRun = false,
  behavior: ComposerFollowUpBehavior = "queue",
) {
  const filtered = items.map(normalizeQueuedPrompt).filter(queuedPromptHasContent);
  if (filtered.length === 0) {
    return;
  }

  const steering = behavior === "steer";
  const targetIds = getPromptQueueTargetIds(target);
  if (steering && targetIds.length === 0) {
    throw new Error("The chat is no longer available for steering.");
  }
  const existingRun = findPromptQueueRunByTarget(target);
  if (existingRun) {
    if (existingRun.deepResearchConsumed) {
      target.consumeDeepResearch();
    }
    const newItems = filtered.map((prompt) => createQueuedPrompt(prompt, target));
    if (steering) {
      existingRun.items.splice(
        steeringInsertionIndex(existingRun.items, existingRun.index),
        0,
        ...newItems,
      );
      steerPromptQueueTarget(target);
      return;
    }
    existingRun.items.push(...newItems);
    syncPromptQueueUI();
    requestPromptQueuePump();
    return;
  }

  const runningByThreadId = useChatRuntimeStore.getState().runningByThreadId;
  const shouldWaitForCurrentRun =
    (waitForCurrentRun || steering) &&
    isPromptQueueTargetRunning(target, runningByThreadId);
  const run: PromptQueueRun = {
    id: createPromptQueueRunId(),
    items: filtered.map((prompt) => createQueuedPrompt(prompt, target)),
    index: shouldWaitForCurrentRun ? -1 : 0,
    generation: 0,
    prevStoreRunning: shouldWaitForCurrentRun,
    waitingForTargetIdle: false,
    paused: false,
    retryTimer: null,
    deepResearchConsumed: false,
  };
  promptQueueRuns.set(run.id, run);
  promptQueueRunOrder.push(run.id);
  syncPromptQueueUI();
  ensurePromptQueueSubscription();
  if (steering) {
    steerPromptQueueTarget(target);
    return;
  }
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

function pausePromptQueueRun(threadIds?: string[]) {
  for (const run of getPromptQueueRunsForThreadIds(threadIds)) {
    const activeItem = getActivePromptQueueItem(run);
    const plan = planUserPromptQueueStop(
      run.items.map((item) => ({ dispatched: item.dispatched })),
      run.index,
    );
    const cancelMode = userStopTargetCancelMode(plan);
    if (plan.retainedItemIndexes.length === 0) {
      deletePromptQueueRun(run);
    } else {
      run.items = plan.retainedItemIndexes.map((index) => run.items[index]);
      // From 0 this rewinds onto an item already passed, replaying it out of order.
      const resumeFrom = plan.retainedItemIndexes.findIndex(
        (index) => index >= Math.max(run.index, 0),
      );
      const searchFrom = resumeFrom < 0 ? 0 : resumeFrom;
      const nextIndex = run.items.findIndex(
        (item, index) => index >= searchFrom && !item.dispatched,
      );
      if (nextIndex < 0) {
        deletePromptQueueRun(run);
      } else {
        run.generation += 1;
        run.index = nextIndex;
        run.paused = plan.pause;
        run.waitingForTargetIdle = false;
        run.prevStoreRunning = false;
        clearPromptQueueRetryTimer(run);
        promptQueueActiveRunIds.delete(run.id);
        promptQueueDispatchingRunIds.delete(run.id);
        syncPromptQueueUI();
      }
    }
    if (cancelMode === "none") {
      continue;
    }
    try {
      if (cancelMode === "permanent") {
        activeItem?.target.cancel();
      } else {
        activeItem?.target.cancelActiveRun();
      }
    } catch {
      // The active run may have already ended.
    }
  }
  requestPromptQueuePumpIfReady();
}

function resumePromptQueueRun(threadIds?: string[]) {
  for (const run of getPromptQueueRunsForThreadIds(threadIds)) {
    if (run.items.some((item) => item.blockedByModelFailure)) {
      for (const item of run.items) item.blockedByModelFailure = false;
      syncPromptQueueUI();
    }
    if (!run.paused) {
      continue;
    }
    run.paused = false;
    // prevStoreRunning outlives the paused early-return; a stale edge skips a prompt.
    run.waitingForTargetIdle = false;
    run.prevStoreRunning = false;
    clearPromptQueueRetryTimer(run);
    syncPromptQueueUI();
  }
  requestPromptQueuePumpIfReady();
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
  if (!plan.cancelActiveItem && plan.heldItemIndexes.length === 0) {
    return;
  }

  const heldItems = plan.heldItemIndexes.map((index) => run.items[index]);
  run.items = plan.retainedItemIndexes.map((index) => run.items[index]);
  for (const item of heldItems) {
    // Held like a failed load: the row stays editable and Resume sends it to the model loaded then.
    item.blockedByModelFailure = true;
    item.target.releaseModel();
  }
  if (!getActivePromptQueueItem(run)) {
    deletePromptQueueRun(run);
    try {
      activeItem?.target.cancel();
    } catch {
      // The active local run may have already ended.
    }
    return;
  }
  if (plan.cancelActiveItem) {
    waitForPromptQueueTargetIdle(run);
    try {
      // A permanent cancel would also void the held prompts that share this target.
      if (run.items.some((item) => item.target === activeItem?.target)) {
        activeItem?.target.cancelActiveRun();
      } else {
        activeItem?.target.cancel();
      }
    } catch {
      // The active local run may have already ended.
    }
    refreshPromptQueueTargetIdleWait(run);
    return;
  }
  if (getActivePromptQueueItem(run)?.blockedByModelFailure) {
    // Invalidate an attempt already underway for the now-held item.
    run.generation += 1;
    clearPromptQueueRetryTimer(run);
    promptQueueDispatchingRunIds.delete(run.id);
    promptQueueActiveRunIds.delete(run.id);
    // The cleared timer may be the poll that sees the run this queue waits behind end.
    if (shouldPollPromptQueueTargetState(run)) {
      refreshPromptQueueTargetIdleWait(run);
    }
  }
  syncPromptQueueUI();
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
  if (run.paused) {
    return true;
  }
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

function handlePromptQueueRunFailed(threadId?: string | null, localOnly = false) {
  if (localOnly) {
    for (const run of promptQueueRuns.values()) {
      for (const item of run.items.slice(Math.max(run.index, 0))) {
        if (item.target.usesLocalModel && !item.dispatched) {
          item.blockedByModelFailure = true;
        }
      }
      if (getActivePromptQueueItem(run)?.blockedByModelFailure) {
        // Invalidate the failed local attempt without disturbing external work.
        run.generation += 1;
        clearPromptQueueRetryTimer(run);
        promptQueueDispatchingRunIds.delete(run.id);
        promptQueueActiveRunIds.delete(run.id);
      }
    }
    syncPromptQueueUI();
    return;
  }
  if (threadId) {
    const failedRun = findPromptQueueRunByThreadIds([threadId]);
    if (failedRun) {
      if (!retainPendingPromptQueueItemsAfterFailure(failedRun)) {
        // Keep accepted follow-ups editable after a failed load or preflight.
        discardQueuedChatRunSettingsForThread(threadId);
        pausePromptQueueRun([threadId]);
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
    const { threadId, localOnly } =
      (event as CustomEvent<PromptQueueRunFailedEventDetail>).detail ?? {};
    handlePromptQueueRunFailed(threadId, localOnly);
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
  let body: ReactNode = null;
  switch (threadMessageKind(role, isEditing)) {
    case "edit":
      body = <EditComposer />;
      break;
    case "user":
      body = <UserMessage />;
      break;
    case "assistant":
      body = <AssistantMessage />;
      break;
    default:
      return null;
  }
  return (
    <>
      {body}
      <ForkContinuationRule />
    </>
  );
};

/**
 * Resolves the divider against the branch on screen, once for the thread.
 *
 * Which inherited message closes the history depends on the branch: editing an inherited turn
 * starts a sibling and leaves the fork's anchor off screen with earlier inherited messages
 * still above it. Selecting the message array in each ROW is what the delete render budget
 * forbids, so it is selected here, in one component, and the rows read the id it publishes.
 * The walk stops at the first message the fork did not inherit, so it costs the inherited
 * count rather than the thread length.
 */
const useTrackForkBoundaryAnchor = (threadId: string | null): void => {
  const inherited = useForkBoundaryStore((s) =>
    threadId === null
      ? undefined
      : s.boundaryByThreadId[threadId]?.messageIds,
  );
  const anchor = useAuiState(({ thread }) =>
    forkBoundaryAnchor(thread.messages, inherited),
  );
  useEffect(() => {
    setForkBoundaryAnchor(threadId, anchor);
  }, [threadId, anchor]);
};

// Closes the history a fork inherited. Rendered by the message it follows, since the row slot
// is propless and the boundary arrives through the store.
const ForkContinuationRule: FC = () => {
  const threadId = useChatRuntimeStore((s) => s.activeThreadId);
  const messageId = useAuiState(({ message }) => message.id);
  // Two plain values rather than the record: both stay identical between renders, so a row
  // subscribed to them does not re-render when an unrelated thread publishes.
  const anchor = useForkBoundaryStore((s) =>
    threadId === null ? undefined : s.anchorByThreadId[threadId],
  );
  const sourceThreadId = useForkBoundaryStore((s) =>
    threadId === null
      ? null
      : (s.boundaryByThreadId[threadId]?.sourceThreadId ?? null),
  );
  const navigate = useNavigate();
  if (anchor === undefined || anchor !== messageId) return null;
  const label = (
    <>
      <HugeiconsIcon icon={WorkflowCircle05Icon} strokeWidth={1.75} className="size-3.5" />
      Continued from chat
    </>
  );
  const labelClass =
    "inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap leading-none";
  return (
    <div
      data-slot="fork-continuation-rule"
      // Same column as the messages it sits between: it is their sibling, not their child,
      // so it takes the width constraint every message root applies to itself.
      className="mx-auto mt-6 mb-2 flex w-full max-w-(--thread-content-max-width) items-center gap-3 text-muted-foreground text-sm"
    >
      <span aria-hidden={true} className="h-px flex-1 bg-border" />
      {sourceThreadId ? (
        <button
          type="button"
          data-slot="fork-continuation-link"
          title="Open the chat this was forked from"
          // Checked on the way out, not on render: the tombstone set is this tab's own, so a
          // source deleted on another device still looks openable until something asks for it.
          // Only a definite "no" stops the trip; an unreachable backend is not a deletion.
          onClick={async () => {
            const source = await readBackendChatThread(sourceThreadId);
            if (source === null) {
              toast.info("That chat has been deleted.");
              return;
            }
            navigate({
              to: "/chat",
              // A paired source is one half of a comparison, and the fork button is offered
              // inside those panes. Opening it as a single chat would show one model's side
              // rather than the view it was forked from. Same shape the sidebar opens a pair
              // with. An unreachable backend has no record to ask, so it falls through.
              search: source?.pairId
                ? { compare: source.pairId }
                : { thread: sourceThreadId },
              replace: false,
            });
          }}
          className={cn(
            labelClass,
            // An answer link's colour, underlined on hover only.
            "cursor-pointer rounded-sm text-primary underline decoration-transparent underline-offset-2 transition-colors hover:decoration-primary focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-2",
          )}
        >
          {label}
        </button>
      ) : (
        // The source is gone, so the text stays but leads nowhere.
        <span className={labelClass}>{label}</span>
      )}
      <span aria-hidden={true} className="h-px flex-1 bg-border" />
    </div>
  );
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
  useTrackForkBoundaryAnchor(threadId);

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

  // plain-text copy skips styled data that consumes over 99% of long-thread copy time.
  // thread-fast-copy.ts falls back to browser copying unless the substitution is invisible.
  useEffect(() => {
    if (!viewportEl) return;
    return attachThreadFastCopy(viewportEl);
  }, [viewportEl]);

  useEffect(() => {
    if (!viewportEl) return;
    return attachWheelHoverSuppression(viewportEl);
  }, [viewportEl]);

  // composer growth applies immediately because content below the scroll position is invisible.
  // composer shrink waits until clamping scrollTop is safe or a bottom pin owns the motion.
  // imperative sizing restores remounted spacers even when composerHeight is unchanged.
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
          ["--thread-max-width" as string]: "var(--custom-chat-max-width, 48rem)",
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
                  "[--thread-header-offset:calc(var(--studio-content-top-inset,0px)+var(--studio-chat-header-height,48px)+var(--studio-chat-notice-height,0px))] pt-[var(--thread-header-offset)]",
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
                      ? "bottom-[calc(150px*var(--ui-space-scale,1))]"
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
          "pointer-events-none absolute inset-x-5 top-[calc(48px*var(--ui-space-scale,1))] flex flex-col items-center",
          hideComposer
            ? "bottom-4"
            : bottomOffsetPx == null
              ? "bottom-[calc(150px*var(--ui-space-scale,1))]"
              : undefined,
        )}
        style={
          !hideComposer && bottomOffsetPx != null
            ? { bottom: bottomOffsetPx }
            : undefined
        }
        aria-label="Generated image preview"
      >
        <div className="pointer-events-auto relative flex min-h-0 w-full max-w-[calc(1100px*var(--ui-space-scale,1))] flex-1 flex-col items-center justify-center gap-3 rounded-3xl bg-muted/10 p-3 ring-1 ring-border/20">
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
        // Inset both sides, not just the right: the offset keeps the bottom
        // fade off the scrollbar, and a one-sided one also moves the centre.
        "aui-thread-composer-dock pointer-events-none absolute bottom-0 left-0 right-0 md:left-[var(--thread-scrollbar-gutter,10px)] md:right-[var(--thread-scrollbar-gutter,10px)]",
        overlay ? "z-40" : "z-20",
      )}
    >
      {/* Column width only: across empty gutters the gradient rounds to a visible seam. */}
      <div
        aria-hidden={true}
        className={cn(
          "thread-bottom-fade absolute bottom-0 left-1/2 w-full max-w-(--thread-max-width) -translate-x-1/2 bg-gradient-to-t from-background from-[calc(100%_-_28px)] to-[rgb(from_var(--background)_r_g_b/0)]",
          queueVisible
            ? "h-32 backdrop-blur-[1px] [mask-image:linear-gradient(to_top,black_0%,black_58%,transparent_100%)]"
            : "top-[calc(10px*var(--ui-space-scale,1))]",
        )}
      />
      {/* Narrow panes spend the gutter on the composer instead; index.css
          trims it off the pane's width, not the window's. */}
      <div className="unsloth-composer-dock-inner relative px-5 pb-2">
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
  const enabled = useChatPreferencesStore(
    (state) => state.showScrollToBottomButton,
  );
  return (
    <TooltipIconButton
      tooltip="Scroll to bottom"
      variant="outline"
      onClick={() => scrollToBottom("auto")}
      className={cn(
        // Muted in dark mode: the page background made it disappear.
        "aui-thread-scroll-to-bottom pointer-events-auto rounded-full p-0 size-[calc(28px*var(--ui-space-scale,1))] bg-background hover:bg-accent dark:bg-muted dark:hover:bg-accent",
        (isAtBottom || !enabled) && "invisible pointer-events-none",
      )}
    >
      <ArrowDownIcon
        strokeWidth={1.75}
        className="size-[calc(var(--ui-icon-size)*1.125)]"
      />
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
        {/* No padding, so the composer here is as wide as once it docks. */}
        <div className="aui-thread-welcome-message flex w-full flex-col justify-center gap-9">
          {/* Center the greeting (sloth + title) over the composer. */}
          <div className="unsloth-welcome-greeting flex flex-row items-center justify-center gap-[calc(15px*var(--ui-space-scale,1))]">
            {/* Temporary chat keeps the title on its own, no mascot. */}
            {showGreetingSloth && !incognito && (
              <MascotImg
                src={currentEmojiSrc}
                className="unsloth-welcome-sloth size-[calc(44px*var(--ui-space-scale,1))] -translate-y-[2px]"
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
    // unsloth-composer-shell is the size container the tight (mobile) layout
    // in index.css queries. It sits outside the surface so those rules can
    // trim the surface's own padding.
    // Same width as the message column. Full chat width sets its own variable, since
    // its percentage would otherwise resolve against this narrower parent.
    <div className="unsloth-composer-shell relative mx-auto min-w-0 w-full max-w-[var(--custom-chat-shell-max-width,var(--thread-content-max-width,46rem))]">
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
      <div className="flex items-center gap-2 rounded-lg border border-[color-mix(in_oklab,var(--foreground)_calc(20%*var(--contrast-edge-gain,1)),transparent)] bg-muted px-3 py-1.5 text-xs">
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

  const supportsTools = useChatRuntimeStore((s) => s.supportsTools);
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  // Effective Code (Full Access implies it), the same gate the request uses to offer read_skill.
  const codeToolsEffective = useChatRuntimeStore(codeToolsOn);
  const imageToolsEnabled = useChatRuntimeStore((s) => s.imageToolsEnabled);
  const supportsBuiltinImageGeneration = useChatRuntimeStore(
    (s) => s.supportsBuiltinImageGeneration,
  );
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
  // always show; Images, RAG, MCP and Deep Research are conditional.
  // Narrow viewports collapse too: the labelled row is wider than a phone composer.
  const isMobile = useIsMobile();
  const pillCount =
    3 +
    (ragEnabled ? 1 : 0) +
    (supportsBuiltinImageGeneration ? 1 : 0) +
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
  const sendShortcut = useChatPreferencesStore((s) => s.sendShortcut);
  const submitIntentRef = useRef<ComposerSubmitIntent>("default");
  const submitOnKey = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>, intent: ComposerSubmitIntent) => {
      const form = event.currentTarget.form;
      if (typeof form?.requestSubmit !== "function") return;
      // A queue or steer chord bound onto an Enter combination lands here
      // first, and preventDefault keeps it from ever reaching useShortcut. Run
      // the behaviour it names, rather than the one the send chord implies.
      const named = followUpShortcutBehavior(event);
      submitIntentRef.current = named
        ? followUpSubmitIntent(
            useChatPreferencesStore.getState().followUpBehavior,
            named,
          )
        : intent;
      try {
        form.requestSubmit();
      } finally {
        submitIntentRef.current = "default";
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
  // True while the @skill picker has a row to pick, so Enter selects it instead of sending.
  const mentionConsumesEnterRef = useRef(false);
  const setMentionConsumesEnter = useCallback((consumesEnter: boolean) => {
    mentionConsumesEnterRef.current = consumesEnter;
  }, []);
  // True while the @skill picker is open, so Escape closes it without collapsing the composer.
  const mentionOpenRef = useRef(false);
  const setMentionOpen = useCallback((open: boolean) => {
    mentionOpenRef.current = open;
  }, []);
  const { inputProps, isComposing, isComposingRef } =
    useImeComposerInputHandlers({
      submitOnEnter: true,
      skipEnterRef: mentionConsumesEnterRef,
      sendShortcut,
      onSubmitKey: submitOnKey,
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
      // A fresh @ re-reads the skill folders, so a skill written since page load is offered.
      if (event.key === "@") refreshSkillsCatalog();
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
      const pastedYoutubeUrl = extractYoutubeVideoUrlFromClipboard(
        event.clipboardData,
      );
      // Bulk text pastes attach as a file instead of filling the input, except
      // in image-edit mode, whose submit path takes an inline instruction only.
      const input = event.currentTarget;
      const { selectionStart, selectionEnd, value } = input;
      // An attachment is serialised after all inline text, so only a paste that
      // was already heading to the end can become one. Mid-text pastes stay
      // inline, where the order the user typed them in survives.
      const pasteGoesLast = input.selectionEnd === input.value.length;
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
      if (event.defaultPrevented) return;
      if (pastedYoutubeUrl) {
        setYoutubeLink(pastedYoutubeUrl);
        if (!pastedText.includes(pastedYoutubeUrl)) {
          event.preventDefault();
          const youtubePasteText =
            pastedText.length === 0
              ? pastedYoutubeUrl
              : `${pastedText}${pastedText.endsWith("\n") ? "" : "\n"}${pastedYoutubeUrl}`;
          const caret = selectionStart + youtubePasteText.length;
          aui
            .composer()
            .setText(
              value.slice(0, selectionStart) +
                youtubePasteText +
                value.slice(selectionEnd),
            );
          requestAnimationFrame(() => input.setSelectionRange(caret, caret));
          if (justSentRef.current?.draftKey === draftKeyRef.current) {
            justSentRef.current = null;
          }
          return;
        }
      }
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
  const editorRef = useRef<HTMLDivElement>(null);
  const inputId = useId();
  // One empty row, at whatever the UI font size makes a row.
  const uiSpaceScale = useUiSpaceScale();
  const oneRowHeight = Math.round(40 * uiSpaceScale);
  const [editorHeight, setEditorHeight] = useState(40);
  const [isWritingExpanded, setIsWritingExpanded] = useState(false);
  const toggleWritingExpanded = () => {
    setIsWritingExpanded((expanded) => !expanded);
    inputRef.current?.focus({ preventScroll: true });
  };
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
  // Autosize's own count: it measures a detached clone, so the expanded
  // editor's min/max-height can't inflate it the way scrollHeight would.
  const [editorRows, setEditorRows] = useState(1);
  const handleEditorHeightChange = useCallback(
    (height: number, meta: { rowHeight: number }) => {
      setEditorHeight(height);
      if (meta.rowHeight <= 0) return;
      const el = inputRef.current;
      if (el && !lineMetricsRef.current) {
        const cs = getComputedStyle(el);
        const lineHeight = Number.parseFloat(cs.lineHeight) || 24;
        const padTop = Number.parseFloat(cs.paddingTop) || 0;
        const padBottom = Number.parseFloat(cs.paddingBottom) || 0;
        lineMetricsRef.current = { lineHeight, padding: padTop + padBottom };
      }
      const padding = lineMetricsRef.current?.padding ?? 0;
      setEditorRows(Math.round((height - padding) / meta.rowHeight));
    },
    [],
  );
  const showWritingToggle = composerText.includes("\n") || editorRows > 3;
  const hasAttachments = useAuiState(
    ({ composer }) => composer.attachments.length > 0,
  );
  const hasPendingAttachments = useAuiState(({ composer }) =>
    composer.attachments.some(
      (attachment) => attachment.status.type === "running",
    ),
  );
  const attachmentsAreQueueableText = useAuiState(
    ({ composer }) =>
      composer.attachments.length > 0 &&
      composer.attachments.every((attachment) =>
        canQueueTextAttachment(attachment) ||
        isPastedTextFile((attachment as { file?: File }).file),
      ),
  );
  // track every attachment id so removing any file releases paste restore without hashing bodies
  const composerAttachmentSignature = useAuiState(({ composer }) =>
    composer.attachments.map((attachment) => attachment.id).join(","),
  );
  const hasPendingAudio = useChatRuntimeStore((s) =>
    Boolean(s.pendingAudioName),
  );
  const nativeAttachmentTargetKey = useNativeAttachmentTargetKey();
  const nativeAttachmentTargetKeyRef = useRef(nativeAttachmentTargetKey);
  nativeAttachmentTargetKeyRef.current = nativeAttachmentTargetKey;

  useEffect(() => {
    if (!nativeAttachmentTargetKey) return;
    const targetKey = nativeAttachmentTargetKey;
    let disposed = false;
    // aui.composer() is whichever chat is open now: a switch mid-batch must not take the rest.
    const add = async (file: File) => {
      if (disposed || nativeAttachmentTargetKeyRef.current !== targetKey) {
        throw new Error("The chat changed before this file was attached.");
      }
      await aui.composer().addAttachment(file);
    };
    const drain = async () => {
      const held = await attachLibraryChatFiles(targetKey, add);
      if (held > 0) toast(translate("library.toast.chatFilesWaiting", { count: held }));
    };
    void drain();
    const offers = useLibraryChatHandoffStore.subscribe((state) => {
      if (state.pending?.targetKey === targetKey) void drain();
    });
    let retrying = false;
    let again = false;
    const retry = async () => {
      if (retrying) {
        again = true;
        return;
      }
      retrying = true;
      do {
        again = false;
        await attachLibraryChatFiles(targetKey, add, true);
      } while (again);
      retrying = false;
    };
    const loads = useChatRuntimeStore.subscribe((state, prev) => {
      if (state.modelLoading) return;
      if (
        prev.modelLoading ||
        state.params.checkpoint !== prev.params.checkpoint ||
        state.residentCheckpoint !== prev.residentCheckpoint ||
        state.loadedIsMultimodal !== prev.loadedIsMultimodal ||
        state.codeToolsEnabled !== prev.codeToolsEnabled ||
        state.supportsTools !== prev.supportsTools
      ) {
        void retry();
      }
    });
    return () => {
      disposed = true;
      offers();
      loads();
    };
  }, [nativeAttachmentTargetKey, aui]);
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
              // The adapter toasted. Keep going: a later, smaller clip may still fit.
              if (stillThisComposer()) cancelQueuedSendRef.current?.();
              continue;
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

  // Same drain as audio, one queue over: video is one clip per message, and the
  // send gate has to hold across the read either way.
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
              file = await normalizeChatImage(
                await nativeAttachmentIntentToFile(intent),
              );
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
  // Not referenceThreadId: it moves null -> remote id on first persist of the same composer.
  const composerIdentity = threadListItemId ?? "";
  composerIdentityRef.current = composerIdentity;
  const chatActive = useChatActive();
  const readAudioUploadDraft = useCallback(
    () => aui.composer().getState().text,
    [aui],
  );
  const writeAudioUploadDraft = useCallback(
    (value: string) => aui.composer().setText(value),
    [aui],
  );
  const focusAudioUploadDraft = useCallback(() => {
    inputRef.current?.focus({ preventScroll: true });
  }, []);
  const dictationEntryDisabled = !chatActive;
  const audioUpload = useChatAudioUpload({
    owner: composerIdentity,
    chatId: referenceThreadId,
    disabled: dictationEntryDisabled || isDictating,
    readDraft: readAudioUploadDraft,
    writeDraft: writeAudioUploadDraft,
    focusDraft: focusAudioUploadDraft,
  });
  const cancelAudioUpload = audioUpload.cancel;
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
  // Wakes a parked attachment when a preflight releases without ever streaming.
  const preStreamRunActive = useSyncExternalStore(
    subscribePreStreamRunReservations,
    () => hasPreStreamRunReservation(preStreamThreadIds),
    () => false,
  );
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
  // validated text uploads and long pastes can join the per-chat queue
  const canQueueTextAttachmentsPrompt =
    attachmentsAreQueueableText && composerAcceptsQueueing;
  // attachments without prepared text stay parked until the run and queue are idle
  const canQueueAttachmentPrompt =
    hasAttachments && !attachmentsAreQueueableText && composerAcceptsQueueing;

  // mirror each thread's draft to localStorage and restore it on mount
  const draftThreadId = referenceThreadId;
  const draftKey = draftThreadId ? composerDraftKey(draftThreadId) : null;
  // unsent pasted File objects need a separate slot because they exist only in memory
  const pasteDraftKey = draftThreadId
    ? composerPasteDraftKey(draftThreadId)
    : null;
  const lastDraftKeyRef = useRef(draftKey);
  // save only after this key's paste restore finishes to avoid premature clearing
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
    // Here, not beside send(): handleSubmit returns early on the three queueing
    // paths, which empty the composer too.
    setIsWritingExpanded(false);
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
        waitForCurrentRun: boolean;
        behavior: ComposerFollowUpBehavior;
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
  // Retain follow-up behavior across preflight waits.
  const pendingFollowUpBehaviorRef = useRef<ComposerFollowUpBehavior>("queue");
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
    let deferModelResolution =
      chatStateAtQueueStart.modelLoading &&
      parseExternalModelId(
        chatStateAtQueueStart.loadingModelPick &&
        !chatStateAtQueueStart.loadingModelPick.selectionSuperseded
          ? chatStateAtQueueStart.loadingModelPick.id
          : chatStateAtQueueStart.params.checkpoint,
      ) === null;
    const runSettingsAtQueueStart = snapshotQueuedChatRunSettings(
      chatStateAtQueueStart,
      { deferModelResolution },
    );
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
    let appendEpoch = 0;
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
        const epoch = appendEpoch;
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
            epoch !== appendEpoch ||
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
              epoch !== appendEpoch ||
              !pendingSettingsIds.has(settingsId)
            ) {
              return;
            }
          }
          // refresh aliases after initialization replaces a local id so stop dialogs deduplicate it
          syncPromptQueueUI();
          const resolvedRunSettings = deferModelResolution
            ? resolveDeferredQueuedModelSettings(
                runSettingsAtQueueStart,
                useChatRuntimeStore.getState(),
              )
            : runSettingsAtQueueStart;
          const keepQueuedFilesForPython = pythonToolRunsInStudio(
            resolvedRunSettings,
          );
          const authSessionEpoch = getAuthSessionEpoch();
          const readyPrompt = await prepareQueuedPromptFiles(
            prompt,
            (file, attachment) =>
              withAttachmentOriginal(
                { file },
                attachment,
                incognitoAtQueueStart,
                authSessionEpoch,
                keepQueuedFilesForPython,
              ),
          );
          prompt.attachments = readyPrompt.attachments;
          prompt.attachmentFiles = readyPrompt.attachmentFiles;
          if (
            removeFreshThreadPersistedAfterAbort() ||
            cancelled ||
            epoch !== appendEpoch ||
            !pendingSettingsIds.has(settingsId)
          ) {
            return;
          }
          const appendResult = thread.append(
            queuedPromptMessage(readyPrompt),
          ) as unknown;
          freshThreadAppendAccepted = true;
          // thread.append accepts synchronously; later provider failures must not dispatch a duplicate
          if (
            appendResult &&
            typeof (appendResult as Promise<void>).catch === "function"
          ) {
            void (appendResult as Promise<void>).catch(() => undefined);
          }
        } catch (error) {
          // retry setup failures unless stop or Clear all invalidated this queue
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
      cancelActiveRun: () => {
        appendEpoch += 1;
        discardOldestPendingSettings();
        getThreadRuntime()?.cancelRun();
      },
      releaseModel: () => {
        // The same snapshot a queue started during a model load takes.
        deferModelResolution = true;
        runSettingsAtQueueStart.params.checkpoint = "";
        runSettingsAtQueueStart.activeGgufVariant = null;
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
      items: Array<string | QueuedPrompt>,
      waitForCurrentRun = false,
      onStarted?: () => void,
      onAborted?: () => void,
      // capture before awaiting so intervening boundary or setting changes invalidate the queue
      capturedAt?: {
        localModelBoundaryGeneration: number;
        queuedSettingsEpoch: number;
        temporary: boolean;
      },
      behavior: ComposerFollowUpBehavior = "queue",
    ) => {
      const reservationKey = JSON.stringify([referenceThreadId, items]);
      // replace only invalid reservations so one attempt owns each prompt
      const existing = promptQueueStartPendingRef.current.get(reservationKey);
      if (existing && !pendingQueueStartIsStale(existing)) {
        // A new shortcut updates the pending draft instead of sending it twice.
        existing.waitForCurrentRun = waitForCurrentRun;
        existing.behavior = behavior;
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
        waitForCurrentRun,
        behavior,
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
            cancelAudioUpload();
            startPromptQueue(
              items,
              target,
              reservation.waitForCurrentRun,
              reservation.behavior,
            );
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
    [
      cancelAudioUpload,
      createPromptQueueTarget,
      pendingQueueStartIsStale,
      referenceThreadId,
    ],
  );

  // preserve pure pastes as editable text while snapshotting uploaded text files
  const queueTextAttachmentsPrompt = useCallback(
    (
      waitForCurrentRun: boolean,
      behavior: ComposerFollowUpBehavior = "queue",
    ): boolean => {
      const composer = aui.composer();
      const attachments = composer.getState().attachments;
      // keep validated decoded file payloads separate from the editable prompt
      if (!attachments.every((attachment) => isPastedTextFile(attachment.file))) {
        const text = composer.getState().text;
        const prepared = snapshotQueuedTextPrompt(text, attachments);
        if (!prepared) return false;
        const ids = attachments.map((attachment) => attachment.id);
        startHydratedPromptQueue(
          [prepared],
          waitForCurrentRun,
          () => {
            const state = composer.getState();
            if (
              state.text !== text ||
              state.attachments.length !== ids.length ||
              !state.attachments.every((attachment, index) => attachment.id === ids[index])
            ) {
              return;
            }
            void composer.clearAttachments();
            flushResourcesSync(() => composer.setText(""));
            clearStoredDraft();
            armJustSent(text);
          },
          () => {
            toast.info("Text attachments were not queued", {
              description: "The chat or settings changed. Send them again.",
            });
          },
          undefined,
          behavior,
        );
        return true;
      }
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
          behavior,
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
    (waitForCurrentRun: boolean, behavior: ComposerFollowUpBehavior = "queue") => {
      const queuedPrompt = aui.composer().getState().text.trim();
      if (!queuedPrompt) {
        return;
      }
      startHydratedPromptQueue(
        [queuedPrompt],
        waitForCurrentRun,
        () => {
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
        },
        undefined,
        undefined,
        behavior,
      );
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
    pendingFollowUpBehaviorRef.current = "queue";
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
        | "running"
        | "settings" = "indexing",
    ) => {
      if (pendingSendRef.current) return;
      pendingSendRef.current = true;
      setPendingSend(true);
      const title =
        waitingOn === "running"
          ? "Waiting for the current response to finish"
          : waitingOn === "images"
            ? "Waiting for dropped images"
            : waitingOn === "audio"
              ? "Waiting for dropped audio"
              : waitingOn === "video"
                ? "Waiting for dropped video"
                : waitingOn === "settings"
                  ? "Loading this chat's settings"
                  : "Waiting for documents to finish indexing";
      waitToastRef.current = toast(title, {
        description: "Your message will send automatically once it is ready.",
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
      // Only after reservation succeeds: a refused send keeps the in-flight transcript.
      cancelAudioUpload();
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
  }, [
    aui,
    armJustSent,
    cancelAudioUpload,
    preStreamThreadIds,
    projectScope,
    referenceThreadId,
  ]);

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

  // Fire the parked send once all waits clear, unless the user emptied the
  // composer while waiting (then drop it quietly). An image dropped after the
  // send was parked has to land first, or indexing finishing early sends the
  // text without it and the image attaches to the next draft.
  useEffect(() => {
    const liveThreadIsRunning =
      threadIsRunning || aui.thread().getState().isRunning;
    const livePromptQueueActive = Boolean(
      findPromptQueueEntry(usePromptQueueUI.getState(), promptQueueThreadIds),
    );
    const livePreStreamRunActive =
      hasPreStreamRunReservation(preStreamThreadIds);
    // pendingSendRef is authoritative because pendingSend still reflects the pre-cancel render.
    if (
      !pendingSend ||
      !pendingSendRef.current ||
      indexingActive ||
      threadScopedSettingsPending ||
      (hasAttachments &&
        !attachmentsAreQueueableText &&
        (liveThreadIsRunning ||
          livePromptQueueActive ||
          livePreStreamRunActive)) ||
      hasMaterializingImageAttachments ||
      hasMaterializingAudioAttachments ||
      hasMaterializingVideoAttachments
    ) {
      return;
    }
    const { text, attachments } = aui.composer().getState();
    const behavior = pendingFollowUpBehaviorRef.current;
    pendingSendRef.current = false;
    pendingFollowUpBehaviorRef.current = "queue";
    setPendingSend(false);
    dismissWaitToast();
    if (text.trim().length > 0 || attachments.length > 0) {
      // Wait mode read now, not carried from the parked submit: a run can
      // start while the settings load, and ignoring it would dispatch on top
      // of the response already streaming.
      const waitForCurrentRun =
        aui.thread().getState().isRunning ||
        hasPreStreamRunReservation(preStreamThreadIds);
      // A parked send is a submit arriving late, so mirror handleSubmit's
      // branches. Sending regardless is how a message vanished: on a throttled
      // browser a follow-up parked 786 ms in was released 236 ms later with
      // the first turn still streaming, went to sendReservedComposer(), and
      // was refused by the runtime -- neither queued nor sent, and the wait
      // toast already dismissed above, so nothing on screen said so.
      //
      // Research refuses every submit and swaps Send for Stop research, so a
      // release here would start a turn from a state where input is disabled.
      if (isResearchActive) {
        return;
      }
      const queueAlreadyActive = Boolean(
        findPromptQueueEntry(usePromptQueueUI.getState(), preStreamThreadIds),
      );
      if (waitForCurrentRun || queueAlreadyActive) {
        // queueing here would bind the follow-up to a nonexistent project new-chat thread.
        if (disableQueue) {
          toast.error("Wait for the current response to finish");
          return;
        }
        // queueComposerText preserves the draft until the queue starts.
        if (canQueueCurrentPrompt) {
          queueComposerText(waitForCurrentRun, behavior);
          return;
        }
        // a long paste has no separate prompt text to queue.
        if (
          canQueueTextAttachmentsPrompt &&
          queueTextAttachmentsPrompt(waitForCurrentRun, behavior)
        ) {
          return;
        }
        // sending here would add the attachment to the active thread.
        if (overlay || hasAttachments || hasPendingAudio) {
          toast.error("Wait for the current response to finish", {
            description:
              "Only text prompts can be queued while a response is running or the prompt queue is active.",
          });
        }
        return;
      }
      clearStoredDraft();
      // keep this synchronous so stale run state cannot drop the send after dismissing the toast.
      // eslint-disable-next-line react-hooks/set-state-in-effect -- see above
      sendReservedComposer();
    }
  }, [
    pendingSend,
    indexingActive,
    threadScopedSettingsPending,
    preStreamRunActive,
    threadIsRunning,
    promptQueueActive,
    promptQueueThreadIds,
    attachmentsAreQueueableText,
    hasMaterializingImageAttachments,
    hasMaterializingAudioAttachments,
    hasMaterializingVideoAttachments,
    aui,
    canQueueCurrentPrompt,
    canQueueTextAttachmentsPrompt,
    clearStoredDraft,
    dismissWaitToast,
    queueComposerText,
    queueTextAttachmentsPrompt,
    sendReservedComposer,
    preStreamThreadIds,
    disableQueue,
    isResearchActive,
    overlay,
    hasAttachments,
    hasPendingAudio,
  ]);

  useEffect(
    () => () => {
      pendingSendRef.current = false;
      pendingFollowUpBehaviorRef.current = "queue";
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
  useEffect(() => {
    setIsWritingExpanded(false);
  }, [composerIdentity]);
  // Window capture runs before the document listeners where the @-mention popover
  // closes and cancelOnEscape preventDefaults every Escape (canCancel is a runtime
  // capability, not a live run), so defaultPrevented cannot tell them apart.
  useEffect(() => {
    const collapseOnEscape = (event: globalThis.KeyboardEvent) => {
      if (
        event.key === "Escape" &&
        !event.isComposing &&
        !mentionOpenRef.current &&
        event.target instanceof Node &&
        editorRef.current?.contains(event.target)
      ) {
        setIsWritingExpanded(false);
      }
    };
    window.addEventListener("keydown", collapseOnEscape, true);
    return () => window.removeEventListener("keydown", collapseOnEscape, true);
  }, []);
  // Keep the mic clickable: if the engine can't run here, explain and point to
  // the local model instead of disabling the button.
  const startDictation = useCallback(() => {
    if (audioUpload.busy || dictationEntryDisabled) return;
    if (currentDictationEntryMode() === "recording-file") {
      audioUpload.openDialog();
      return;
    }
    if (!isStudioDictationAvailable()) {
      notifyStudioDictationUnavailable();
      return;
    }
    try {
      aui.composer().startDictation();
    } catch {
      notifyStudioDictationUnavailable();
    }
  }, [aui, audioUpload, dictationEntryDisabled]);
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
  // Same send as above, with the follow-up named rather than left to the
  // preference. handleSubmit reads the intent ref, so ask it for the opposite
  // whenever the preference is not already the behaviour wanted here.
  const submitWithFollowUp = useCallback(
    (behavior: ComposerFollowUpBehavior) => {
      // No dictation branch: the recording bar replaces the composer, so the
      // foreground gate below already turns these into a no-op there.
      if (!isSurfaceInForeground(COMPOSER_INPUT_SELECTOR)) return;
      submitIntentRef.current = followUpSubmitIntent(
        useChatPreferencesStore.getState().followUpBehavior,
        behavior,
      );
      try {
        formRef.current?.requestSubmit();
      } finally {
        submitIntentRef.current = "default";
      }
    },
    [],
  );
  const followUpShortcutOptions = {
    enabled: chatActive && !disabled,
    skipInTextFields: true,
    textFieldException: COMPOSER_INPUT_SELECTOR,
  };
  useShortcut(
    "queueMessage",
    () => submitWithFollowUp("queue"),
    followUpShortcutOptions,
  );
  useShortcut(
    "steerMessage",
    () => submitWithFollowUp("steer"),
    followUpShortcutOptions,
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
      const behavior = composerFollowUpBehavior(
        useChatPreferencesStore.getState().followUpBehavior,
        submitIntentRef.current,
      );
      submitIntentRef.current = "default";
      pendingFollowUpBehaviorRef.current = behavior;
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
        // the project new-chat composer has no thread to bind a queue to.
        if (disableQueue) {
          toast.error("Wait for the current response to finish");
          return;
        }
        if (!canQueueCurrentPrompt) {
          if (
            canQueueTextAttachmentsPrompt &&
            queueTextAttachmentsPrompt(
              liveThreadIsRunning || livePreStreamRunActive,
              behavior,
            )
          ) {
            return;
          }
          if (canQueueAttachmentPrompt) {
            enqueueSend("running");
            return;
          }
          if (overlay || hasAttachments || hasPendingAudio) {
            toast.error(
              liveThreadIsRunning
                ? "Wait for the current response to finish"
                : "Wait for the prompt queue to finish",
              {
                description:
                  "Only text prompts and ready attachments can be queued while a response is running or the prompt queue is active.",
              },
            );
          }
          return;
        }
        queueComposerText(
          liveThreadIsRunning || livePreStreamRunActive,
          behavior,
        );
        return;
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
      canQueueAttachmentPrompt,
      canQueueCurrentPrompt,
      canQueueTextAttachmentsPrompt,
      queueComposerText,
      queueTextAttachmentsPrompt,
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
    pausePromptQueueRun(promptQueueThreadIds);
  }, [promptQueueThreadIds]);

  const resumeQueue = useCallback(() => {
    resumePromptQueueRun(promptQueueThreadIds);
  }, [promptQueueThreadIds]);

  const startQueue = useCallback(
    (
      items: Array<string | QueuedPrompt>,
      waitForCurrentRun =
        threadIsRunning || aui.thread().getState().isRunning,
      onAborted?: () => void,
    ) => {
      // saved-prompt calls bypass the button, so project new-chat must still refuse queues.
      if (disableQueue) return false;
      // false means an identical start is pending, not refused.
      startHydratedPromptQueue(items, waitForCurrentRun, undefined, onAborted);
      return true;
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
      {!isDictating ? <ComposerDraftPreview text={composerText} /> : null}
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
            audioUploadBusy={audioUpload.busy}
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
              {mcpEnabledForChat ? (
                <McpComposerButton side={effectiveMenuSide} />
              ) : null}
              <SkillsComposerButton side={effectiveMenuSide} />
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
            <div
              ref={editorRef}
              className="unsloth-composer-editor"
              style={
                {
                  "--composer-editor-height": `${composerText.length === 0 ? oneRowHeight : Math.max(oneRowHeight, editorHeight)}px`,
                } as CSSProperties
              }
            >
              <ComposerPrimitive.Input
                id={inputId}
                submitMode="none"
                placeholder={
                  overlay ? "Type your edits for your image" : "Ask anything"
                }
                ref={inputRef}
                data-type-to-activate="composer"
                className="aui-composer-input unsloth-composer-input"
                minRows={1}
                maxRows={12}
                onHeightChange={handleEditorHeightChange}
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
              {(showWritingToggle || isWritingExpanded) && (
                <TooltipIconButton
                  type="button"
                  tooltip={
                    isWritingExpanded ? "Collapse composer" : "Expand composer"
                  }
                  aria-expanded={isWritingExpanded}
                  aria-controls={inputId}
                  disabled={disabled}
                  className="unsloth-composer-expand absolute -right-1 top-0 size-8 rounded-md bg-transparent text-muted-foreground hover:bg-transparent hover:text-muted-foreground dark:hover:bg-transparent aria-expanded:bg-transparent aria-expanded:text-muted-foreground"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={toggleWritingExpanded}
                >
                  <svg
                    viewBox="0 0 20 20"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2.25}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="size-4"
                    aria-hidden={true}
                  >
                    <path
                      d={
                        isWritingExpanded
                          ? "M13 2v5h5M2 13h5v5"
                          : "M11 4h5v5M4 11v5h5"
                      }
                    />
                  </svg>
                </TooltipIconButton>
              )}
            </div>
            <ComposerRightControls
              disabled={
                disabled ||
                !hasSendableContent ||
                isComposing ||
                hasPendingAttachments
              }
              dictationDisabled={dictationEntryDisabled}
              // disableQueue makes running threads show Stop instead of Queue.
              queueDisabled={
                disableQueue ||
                !(
                  canQueueCurrentPrompt ||
                  canQueueTextAttachmentsPrompt ||
                  canQueueAttachmentPrompt
                )
              }
              onQueueClick={() => formRef.current?.requestSubmit()}
              // ComposerPrimitive.Send skips form submit, so run the full queue and capacity path.
              onSendClick={handleSubmit}
              onStopClick={stopQueue}
              onResumeClick={resumeQueue}
              onDictateClick={startDictation}
              audioUpload={audioUpload}
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
      <ChatAudioUploadMount audioUpload={audioUpload} />
    </>
  );

  return (
    <PromptQueueContext.Provider value={queueContextValue}>
    <ComposerPrimitive.Unstable_TriggerPopoverRoot>
      <SkillMentionPopover
        enabled={supportsTools && codeToolsEffective}
        onConsumesEnterChange={setMentionConsumesEnter}
        onOpenChange={setMentionOpen}
      />
    <ComposerPrimitive.Root
      ref={attachComposer}
      // Out of find-in-page's reach: the draft itself lives in a textarea the index cannot read, so
      // all this leaves to find are the pill labels, and a search for "code" or "images" would land
      // on the toolbar instead of on the conversation.
      {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
      className="aui-composer-root relative flex w-full flex-col"
      data-writing-expanded={
        isWritingExpanded && !isDictating ? "true" : undefined
      }
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
              "aui-composer-drop-overlay pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center gap-1 overflow-hidden rounded-[inherit] bg-background/90 opacity-0 backdrop-blur-sm transition-opacity duration-150 group-data-[dragging=true]/dropzone:opacity-100 dark:bg-card/90",
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
    </ComposerPrimitive.Unstable_TriggerPopoverRoot>
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
  skipEnterRef,
  onSubmitKey,
  sendShortcut = "enter",
  justSentRef,
  draftKeyRef,
}: {
  submitOnEnter?: boolean;
  /** Set while a composer popover will consume plain Enter itself. */
  skipEnterRef?: RefObject<boolean>;
  /** The selected send chord, carrying the one-message follow-up override. */
  onSubmitKey?: (
    event: KeyboardEvent<HTMLTextAreaElement>,
    intent: ComposerSubmitIntent,
  ) => void;
  sendShortcut?: ComposerSendShortcut;
  // Guard armed by the last send or queue. See setComposerText below.
  justSentRef?: RefObject<SentTextGuard | null>;
  // Thread on screen. The composer outlives a thread switch, so this is what
  // says whether the armed guard belongs to the thread being typed into.
  draftKeyRef?: RefObject<string | null>;
} = {}) {
  const aui = useAui();
  const composingRef = useRef(false);
  const imeSessionOpenRef = useRef(false);
  const compositionEndedAtRef = useRef(-Infinity);
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
    imeSessionOpenRef.current = true;
    setCompositionState(true);
  }, [justSentRef, setCompositionState]);

  const onCompositionUpdate = useCallback(() => {
    refreshStuckTimer();
  }, [refreshStuckTimer]);

  const onCompositionEnd = useCallback(
    (e: CompositionEvent<HTMLTextAreaElement>) => {
      imeSessionOpenRef.current = false;
      compositionEndedAtRef.current = e.timeStamp;
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
      const msSinceCompositionEnd = e.timeStamp - compositionEndedAtRef.current;
      compositionEndedAtRef.current = -Infinity;
      const imeKey = e.nativeEvent.isComposing || e.keyCode === 229;
      if (imeKey) {
        if (
          imeKeydownBlocksComposerSubmit(
            e,
            imeSessionOpenRef.current,
            msSinceCompositionEnd,
          )
        ) {
          // Deliberately NOT user input: picking a candidate in a composition the
          // send left open is that composition continuing. One begun after the
          // send is marked by compositionstart instead.
          composingRef.current = true;
          refreshStuckTimer();
          return;
        }
        setCompositionState(false);
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
      if (submitOnEnter && !skipEnterRef?.current) {
        const intent = composerSubmitIntent(
          imeKey ? composerKeyEventForImeSubmit(e) : e,
          sendShortcut,
          e.currentTarget?.value,
        );
        if (intent) {
          e.preventDefault();
          if (onSubmitKey) onSubmitKey(e, intent);
          else e.currentTarget.form?.requestSubmit();
        }
      }
    },
    [
      justSentRef,
      onSubmitKey,
      sendShortcut,
      refreshStuckTimer,
      setCompositionState,
      skipEnterRef,
      submitOnEnter,
    ],
  );

  // On macOS, switching input methods (e.g. ABC → Pinyin) while the textarea
  // is focused can fire compositionstart without a matching compositionend,
  // leaving composingRef pinned and Send permanently blocked. The OS always
  // commits or cancels any in-progress composition before surrendering focus,
  // so blur is a safe unconditional reset point.
  const onBlur = useCallback(() => {
    imeSessionOpenRef.current = false;
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
const ThinkIcon: FC = () => <BulbIcon className="size-[calc(15.5px*var(--ui-space-scale,1))]" />;

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
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  const externalReasoningCaps =
    externalSelection != null
      ? getExternalReasoningCapabilities(
          selectedExternalProvider?.providerType,
          // The adapter resolves reasoning for the selected id; openrouter/free can route each turn elsewhere.
          externalSelection?.modelId,
          {
            isReasoningProvider:
              selectedExternalProvider?.isReasoningModel === true,
            // Lets the resolver detect custom Gemini OAI-compat gateways.
            baseUrl: selectedExternalProvider?.baseUrl ?? null,
            apiType: selectedExternalProvider?.apiType,
            reasoningConfig: selectedExternalProvider?.reasoningConfig,
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
  // What the adapter sends: the stored effort clamped to the current ladder, so a catalog refresh that
  // drops the stored level is shown truthfully without overwriting the choice.
  const displayedEffort =
    effectiveReasoningEffortLevels.length > 0
      ? clampReasoningEffortToLevels(reasoningEffort, effectiveReasoningEffortLevels)
      : reasoningEffort;
  const effectiveReasoningVisualEnabled =
    effectiveReasoningEnabled && displayedEffort !== "none";
  const disabled =
    !modelLoaded || !(effectiveSupportsReasoning || supportsPreserveThinking);
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
  const effortLabel = formatEffortLabel(displayedEffort);

  // A connection may support history preservation without a generation toggle.
  if (!effectiveSupportsReasoning && !supportsPreserveThinking) {
    return null;
  }

  // enable_thinking_effort (GLM-5.2: high|max + disable) reuses the effort
  // dropdown; it just also carries an Off row via supportsReasoningOff.
  const isEffort =
    effectiveReasoningStyle === "reasoning_effort" ||
    effectiveReasoningStyle === "enable_thinking_effort";
  // Dropdown when there are effort levels or preserve-thinking; else a toggle.
  const useDropdown = isEffort || supportsPreserveThinking;
  const activeLook = !effectiveSupportsReasoning
    ? preserveThinking && !disabled
    : isEffort
      ? reasoningLockedOn || (effectiveReasoningVisualEnabled && !disabled)
      : reasoningLockedOn || (effectiveReasoningEnabled && !disabled);

  if (useDropdown) {
    return (
      <NonModalDropdownMenu
        side={side}
        align="end"
        avoidCollisions={true}
        className="unsloth-plus-menu unsloth-thinking-menu min-w-0 w-[calc(176px*var(--ui-space-scale,1))]"
        trigger={(triggerRef) => (
          <button
            ref={triggerRef}
            type="button"
            disabled={disabled}
            className="unsloth-thinking-pill"
            data-pill-label="Thinking settings"
            data-active={activeLook ? "true" : "false"}
            aria-label={thinkEffortAriaLabel({
              modelLoaded,
              reasoningDisabled: disabled,
              reasoningEffort: displayedEffort,
            })}
          >
            <ThinkIcon />
            {activeLook ? (
              <span className="unsloth-thinking-label">
                {isEffort ? `Thinking · ${effortLabel}` : "Thinking"}
              </span>
            ) : null}
            <ChevronDownIcon strokeWidth={1.5} className="unsloth-thinking-caret size-[calc(15px*var(--ui-space-scale,1))]" />
          </button>
        )}
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
                None
                <HugeiconsIcon
                  icon={Tick02Icon}
                  strokeWidth={2}
                  className={cn(
                    "unsloth-tick ms-auto size-4",
                    effectiveReasoningVisualEnabled && "opacity-0",
                  )}
                />
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
                  {formatEffortLabel(level)}
                  <HugeiconsIcon
                  icon={Tick02Icon}
                  strokeWidth={2}
                    className={cn(
                      "unsloth-tick ms-auto size-4",
                      !(
                        effectiveReasoningVisualEnabled &&
                        displayedEffort === level
                      ) && "opacity-0",
                    )}
                  />
                </DropdownMenuItem>
              ))}
          </>
        ) : (
          effectiveSupportsReasoning &&
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
              Thinking
              <HugeiconsIcon
                  icon={Tick02Icon}
                  strokeWidth={2}
                className={cn(
                  "unsloth-tick ms-auto size-4",
                  !effectiveReasoningEnabled && "opacity-0",
                )}
              />
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
              // Only local models couple this setting to generation controls.
              if (next && externalSelection === null) {
                setReasoningEnabled(true);
                applyQwenThinkingParams(true);
              }
            }}
          >
            Preserve thinking
            <HugeiconsIcon
                  icon={Tick02Icon}
                  strokeWidth={2}
              className={cn(
                "unsloth-tick ms-auto size-4",
                !preserveThinking && "opacity-0",
              )}
            />
          </DropdownMenuItem>
        )}
      </NonModalDropdownMenu>
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
        <InternetGlyph className="size-[calc(15px*var(--ui-space-scale,1))]" />
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
  const codeToolsEnabled = useChatRuntimeStore(codeToolsOn);
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
          className="size-[calc(18.5px*var(--ui-space-scale,1))]"
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
  const StatusIcon = kind === "terminal" ? TerminalIcon : InternetGlyph;
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
  /^(audio\/|\.(?:wav|mp3|mp2|m4a|ogg|oga|opus|flac|aac|aiff|aif|aifc|caf|wma|amr)$)/i;

function attachmentAcceptForPicker(accept: string, audioEnabled: boolean): string {
  const enabledAccept =
    audioEnabled || accept === "*"
      ? accept
      : accept
          .split(",")
          .map((token) => token.trim())
          .filter((token) => token && !AUDIO_ACCEPT_TOKEN_RE.test(token))
          .join(",") || accept;
  return pickerAcceptForTextBasenames(enabledAccept);
}

const ComposerToolsMenu: FC<{
  side?: "top" | "bottom";
  researchAvailable: boolean;
  audioUploadBusy: boolean;
}> = ({ side = "bottom", researchAvailable, audioUploadBusy }) => {
  const t = useT();
  const navigate = useNavigate();
  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);
  const codeToolsEnabled = useChatRuntimeStore(codeToolsOn);
  const setCodeToolsEnabled = useChatRuntimeStore((s) => s.setCodeToolsEnabled);
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
  // The permission pill is hidden while recording, so the menu carries it then.
  const isDictating = useAuiState((s) => s.composer.dictation != null);
  // Capability gating mirrors the visible pills so menu and pills agree on
  // what a loaded model supports (a tool the backend drops must not look on).
  const modelLoaded = useChatRuntimeStore(
    (s) => !!s.params.checkpoint && !s.modelLoading,
  );
  const audioAttachmentsEnabled = useChatRuntimeStore((s) => {
    const activeCheckpoint = s.params.checkpoint;
    // No model yet: offer audio too, since files attached now wait for the model loaded next.
    if (!activeCheckpoint || s.modelLoading) {
      return true;
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
  const [skillsOpen, setSkillsOpen] = useState(false);
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
  const { overlay: generatedImageOverlay } = useGeneratedImageOverlay();
  const menuIsDictating = useAuiState((s) => s.composer.dictation != null);

  const plusPins = usePlusMenuPrefsStore((s) => s.pins);

  const [recentPrompts, setRecentPrompts] = useState<PromptEntry[]>([]);
  const [recentLists, setRecentLists] = useState<PromptListEntry[]>([]);
  const recentSeqRef = useRef(0);
  const refreshRecentPrompts = useCallback(async () => {
    recentSeqRef.current += 1;
    const seq = recentSeqRef.current;
    try {
      const rows = await listPromptEntries();
      if (seq !== recentSeqRef.current) return;
      const byRecent = [...rows].sort((a, b) => b.updatedAt - a.updatedAt);
      // Pinned prompts take over the submenu; fall back to the 3 most recent
      // when nothing is pinned.
      const pinnedIds = usePlusMenuPrefsStore.getState().pinnedPromptIds;
      const pinned = byRecent.filter((p) => pinnedIds.includes(p.id));
      setRecentPrompts(pinned.length > 0 ? pinned : byRecent.slice(0, 3));
    } catch {
      // Clear, don't keep: a stale list row would run its cached items.
      if (seq === recentSeqRef.current) setRecentPrompts([]);
    }
    try {
      const rows = await listPromptLists();
      if (seq !== recentSeqRef.current) return;
      const pinnedIds = usePlusMenuPrefsStore.getState().pinnedListIds;
      setRecentLists(rows.filter((l) => pinnedIds.includes(l.id)));
    } catch {
      if (seq === recentSeqRef.current) setRecentLists([]);
    }
  }, []);

  const runPromptList = useCallback(
    (items: string[], fromDialog = false) => {
      // A queue started while recording would swallow the held transcript send.
      if (menuIsDictating) {
        toast.error("Finish dictating before running a list");
        return;
      }
      // Starting the queue cancels an in-flight transcription, which would discard it.
      if (audioUploadBusy) {
        toast.error("Wait for the transcription to finish before running a list");
        return;
      }
      // Mid image edit, startQueue would bypass the overlay's prompt rewrite.
      if (generatedImageOverlay) {
        toast.error("Close the image editor before running a list", {
          description: "Saved lists cannot be applied to a generated image.",
        });
        return;
      }
      const started = startQueue(items, undefined, () => {
        if (fromDialog) setPromptStorageOpen(true);
        toast.info("Saved list was not queued", {
          description: "The chat changed before the queue was ready. Try again.",
        });
      });
      if (started) {
        setPromptStorageOpen(false);
        return;
      }
      // startQueue refuses synchronously without calling onAborted.
      toast.error("Couldn't queue that list here", {
        description: "Open a chat first, then run the list.",
      });
    },
    [startQueue, generatedImageOverlay, menuIsDictating, audioUploadBusy, setPromptStorageOpen],
  );

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
        Chat with files
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
    skills: (
      <DropdownMenuItem onSelect={() => setSkillsOpen(true)}>
        <HugeiconsIcon icon={Scroll01Icon} strokeWidth={2} />
        Skills
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
          className="unsloth-plus-menu w-[calc(208px*var(--ui-space-scale,1))]"
        >
          {recentPrompts.map((p) => (
            <DropdownMenuItem
              key={`prompt:${p.id}`}
              onSelect={() => aui.composer().setText(p.text)}
            >
              <span className="truncate">{p.name}</span>
            </DropdownMenuItem>
          ))}
          {recentLists.map((l) => (
            <DropdownMenuItem key={`list:${l.id}`} onSelect={() => runPromptList(l.items)}>
              <span className="truncate">{l.name}</span>
              <PromptCountBadge count={l.items.length} />
            </DropdownMenuItem>
          ))}
          {recentPrompts.length > 0 || recentLists.length > 0 ? (
            <DropdownMenuSeparator />
          ) : null}
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
          className="unsloth-plus-menu w-[calc(208px*var(--ui-space-scale,1))]"
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
    projects: (
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>
          <HugeiconsIcon icon={Folder01Icon} strokeWidth={2} />
          Projects
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent className="unsloth-plus-menu w-[calc(232px*var(--ui-space-scale,1))]">
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
    <ChatSkillsDialog open={skillsOpen} onOpenChange={setSkillsOpen} />
    <PromptStorageDialog
      open={promptStorageOpen}
      onOpenChange={setPromptStorageOpen}
      onUse={(text) => {
        aui.composer().setText(text);
      }}
      onRunList={(items) => runPromptList(items, true)}
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
          <PlusIcon className="size-[calc(22px*var(--ui-space-scale,1))] stroke-[1.75px]" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side={side}
        align="start"
        sideOffset={0}
        avoidCollisions={true}
        className="unsloth-plus-menu w-[calc(244px*var(--ui-space-scale,1))]"
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
          <InternetGlyph />
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
        {isDictating ? <BypassPermissionsMenuItem /> : null}
        {pinnedPlusItems.map((id) => (
          <Fragment key={id}>{plusMenuNodes[id]}</Fragment>
        ))}
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>
            <MoreHorizontalIcon className="size-4" />
            More
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="unsloth-plus-menu w-[calc(248px*var(--ui-space-scale,1))]">
            {overflowPlusItems.map((id) => (
              <Fragment key={id}>{plusMenuNodes[id]}</Fragment>
            ))}
            {overflowPlusItems.length > 0 && <DropdownMenuSeparator />}
            <DropdownMenuItem
              onSelect={() => useSettingsDialogStore.getState().openDialog("chat", {
                scrollTarget: "chat-composer",
              })}
            >
              <SlidersHorizontalIcon className="size-4" />
              {t("composerSettings.settings")}
            </DropdownMenuItem>
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

const PromptQueueStack: FC<{ queueThreadIds: string[] }> = ({
  queueThreadIds,
}) => {
  const queueEntry = usePromptQueueUI((s) =>
    findPromptQueueEntry(s, queueThreadIds),
  );
  const items = usePromptQueueUI((s) => s.items);
  const visibleItems = queueEntry
    ? items.filter((item) => item.runId === queueEntry.runId)
    : [];

  if (!queueEntry || visibleItems.length === 0) return null;

  return (
    <PromptQueueList
      key={queueEntry.runId}
      entry={queueEntry}
      items={visibleItems}
      onEdit={editPromptQueueItem}
      onRemove={removePromptQueueItem}
      onMove={movePromptQueueItem}
      onSteer={steerPromptQueueItem}
      onResume={() => resumePromptQueueRun(queueThreadIds)}
    />
  );
};

const ComposerRightControls: FC<{
  disabled?: boolean;
  dictationDisabled?: boolean;
  queueDisabled?: boolean;
  onQueueClick?: () => void;
  onSendClick?: (event: { preventDefault: () => void }) => void;
  onStopClick?: () => void;
  onResumeClick?: () => void;
  onDictateClick?: () => void;
  audioUpload: ReturnType<typeof useChatAudioUpload>;
  pendingSend?: boolean;
  menuSide?: "top" | "bottom";
  queueThreadIds: string[];
}> = ({
  disabled,
  dictationDisabled,
  queueDisabled,
  onQueueClick,
  onSendClick,
  onStopClick,
  onResumeClick,
  onDictateClick,
  audioUpload,
  pendingSend,
  menuSide,
  queueThreadIds,
}) => {
  const t = useT();
  const followUpBehavior = useChatPreferencesStore((s) => s.followUpBehavior);
  const sendShortcut = useChatPreferencesStore((s) => s.sendShortcut);
  // A boolean, so typing re-renders this only when a line break comes or goes.
  const multiline = useAuiState(({ composer }) => composer.text.includes("\n"));
  const shortcutLabels = composerShortcutLabels(
    sendShortcut,
    isMacPlatform(),
    multiline ? "\n" : "",
  );
  const followUpLabel = t(
    followUpBehavior === "queue"
      ? "promptQueue.queueButton"
      : "promptQueue.steerButton",
  );
  const followUpTooltip = t("promptQueue.followUpTooltip", {
    action: followUpLabel,
    send: shortcutLabels.send,
    opposite: shortcutLabels.opposite,
  });
  const queueEntry = usePromptQueueUI((s) =>
    findPromptQueueEntry(s, queueThreadIds),
  );
  const isQueueRunning = Boolean(queueEntry);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const threadRemoteId = useAuiState(
    ({ threadListItem }) => threadListItem.remoteId,
  );
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
    // A reply replayed after a reload has no adapter run for Cancel to abort.
    stopRecoveredRun(threadRemoteId);
  };
  return (
    <div className="aui-composer-action-wrapper flex shrink-0 items-center gap-1.5">
      <ReasoningToggle side={menuSide} />
      {/* Starts dictation; the recording bar then covers the input row and owns
          the stop and send actions. */}
      <ComposerPrimitive.If dictation={false}>
        {audioUpload.busy ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-9 gap-1.5 rounded-full px-2.5 text-muted-foreground"
            aria-label={t("settings.voice.dictation.audioUploadCancel")}
            title={t("settings.voice.dictation.audioUploadCancel")}
            onClick={audioUpload.cancel}
          >
            <Spinner className="size-4" />
            <span>{t("settings.voice.dictation.audioUploadTranscribing")}</span>
            <XIcon className="size-3.5" aria-hidden="true" />
          </Button>
        ) : (
          <TooltipIconButton
            tooltip="Dictate"
            aria-label="Dictate"
            type="button"
            variant="ghost"
            className="size-9 rounded-full text-foreground"
            disabled={dictationDisabled}
            onClick={onDictateClick}
          >
            <MicIcon className="unsloth-dictate-icon size-6" />
          </TooltipIconButton>
        )}
      </ComposerPrimitive.If>
      <AuiIf
        condition={({ thread }) =>
          !thread.isRunning && !isQueueRunning && !isResearchActive
        }
      >
        <ComposerPrimitive.Send asChild={true}>
          <TooltipIconButton
            tooltip={
              pendingSend
                ? "Waiting for documents…"
                : t("promptQueue.sendTooltip", { shortcut: shortcutLabels.send })
            }
            side="bottom"
            type="submit"
            variant="default"
            size="icon"
            // Stay clickable while docs index so a click can queue the send;
            // disabled only once a send is parked.
            disabled={disabled || pendingSend}
            onClick={(event) => onSendClick?.(event)}
            className="aui-composer-send ml-1.5 size-9 rounded-full"
            aria-label={t("promptQueue.sendLabel")}
          >
            {pendingSend ? (
              <Spinner className="size-[calc(18px*var(--ui-space-scale,1))]" />
            ) : (
              <ArrowUpIcon className="unsloth-send-icon aui-composer-send-icon size-[calc(21px*var(--ui-space-scale,1))] stroke-2" />
            )}
          </TooltipIconButton>
        </ComposerPrimitive.Send>
      </AuiIf>
      {isQueueRunning && !isResearchActive ? (
        <AuiIf condition={({ thread }) => !thread.isRunning}>
          {queueEntry?.paused && queueDisabled ? (
            <TooltipIconButton
              tooltip="Resume queue"
              side="bottom"
              type="button"
              variant="default"
              size="icon"
              onClick={onResumeClick}
              className="aui-composer-send ml-1.5 size-9 rounded-full"
              aria-label="Resume queue"
            >
              {/* Solid glyph, so it is smaller than the stroked send arrow. */}
              <QueueResumeIcon className="size-4" />
            </TooltipIconButton>
          ) : queueEntry?.dispatched && !queueEntry.paused ? (
            <Button
              type="button"
              variant="default"
              size="icon"
              className="aui-composer-cancel ml-1.5 size-9 rounded-full"
              aria-label="Stop queued message"
              onClick={stop}
            >
              <SquareIcon className="size-3 fill-current" />
            </Button>
          ) : (
            <TooltipIconButton
              tooltip={followUpTooltip}
              side="bottom"
              type="button"
              variant="default"
              size="icon"
              disabled={disabled || queueDisabled}
              onClick={onQueueClick}
              className="aui-composer-send ml-1.5 size-9 rounded-full"
              aria-label={followUpLabel}
            >
              <ArrowUpIcon className="unsloth-send-icon aui-composer-send-icon size-[calc(21px*var(--ui-space-scale,1))] stroke-2" />
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
            <SquareIcon className="size-3 fill-current" />
          )}
        </Button>
      ) : (
        <AuiIf condition={({ thread }) => thread.isRunning}>
          {/* Classed so the narrow-screen rules can treat this like the
              sibling send/stop buttons; it is the flex item, not the button. */}
          <div className="aui-composer-run-controls ml-1.5 flex items-center gap-1.5">
            <ComposerPrimitive.Cancel asChild={true}>
              <Button
                type="button"
                variant="default"
                size="icon"
                className="aui-composer-cancel size-9 rounded-full"
                aria-label="Stop generating"
                // Cancel only ends the reply; handlePromptQueueRunState then
                // dispatches the next queued prompt. stop() ends the run.
                onClick={stop}
              >
                <SquareIcon className="size-3 fill-current" />
              </Button>
            </ComposerPrimitive.Cancel>
            {!queueDisabled ? (
              <TooltipIconButton
                tooltip={followUpTooltip}
                side="bottom"
                type="button"
                variant="default"
                size="icon"
                onClick={onQueueClick}
                className="aui-composer-send size-9 rounded-full"
                aria-label={followUpLabel}
              >
                <ArrowUpIcon className="unsloth-send-icon aui-composer-send-icon size-[calc(21px*var(--ui-space-scale,1))] stroke-2" />
              </TooltipIconButton>
            ) : null}
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
              <RefreshGlyph strokeWidth={1.75} className="size-3.5" />
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

function readThoughtDuration(metadata: unknown): number | undefined {
  const custom = (metadata as { custom?: Record<string, unknown> } | undefined)
    ?.custom;
  const durations = custom?.reasoningDurations;
  const value = Array.isArray(durations) ? durations[0] : custom?.reasoningDuration;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/** Shared eligibility and run setup for Resume and Continue response. */
function useContinuation() {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const isLast = useAuiState(({ message }) => message.isLast);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const researchRunId = useResearchMessageRunId();
  const researchActive = useThreadResearchActive();
  const status = useAuiState(({ message }) => message.status);
  const metadata = useAuiState(({ message }) => message.metadata);
  const thoughtResumable = useChatRuntimeStore((s) =>
    resumesThought({
      loadedIsGguf: s.loadedIsGguf,
      loadedIsMlx: s.loadedIsMlx,
      activeGgufVariant: s.activeGgufVariant,
      activeNativePathToken: s.activeNativePathToken,
      checkpoint: s.params.checkpoint,
    }),
  );
  const partial = useAuiState(
    ({ message }) => readContinuationSource(message.content).partial,
  );
  const reasoning = useAuiState(
    ({ message }) => readContinuationSource(message.content).reasoning,
  );
  // Gemini signs answer and thought parts. A continuation runs from a sibling branch,
  // so both kinds of replay metadata travel with the partial.
  const thoughtSignature = useAuiState(({ message }) =>
    readTextThoughtSignature(message.content),
  );
  const messageContent = useAuiState(({ message }) => message.content);
  const thoughtParts = useMemo(
    () => collectGeminiThoughtReplayParts(messageContent),
    [messageContent],
  );
  const answerParts = useMemo(
    () => collectGeminiAnswerReplayParts(messageContent),
    [messageContent],
  );
  const geminiReplayTurns = useMemo(
    () =>
      continuationGeminiReplayTurns(metadata, {
        text: partial,
        ...(thoughtSignature ? { thoughtSignature } : {}),
        ...(thoughtParts.length > 0 ? { thoughtParts } : {}),
        ...(answerParts.length > 0 ? { answerParts } : {}),
      }),
    [metadata, partial, thoughtSignature, thoughtParts, answerParts],
  );
  // A tool-calling turn cannot be resumed: the continuation runs as a sibling, so the
  // call and its result would be missing from the outbound history. A Gemini ledger is
  // independently replayable even when its signed thought produced no visible text.
  const continuable = useMemo(
    () =>
      isContinuableContent(messageContent, {
        thought: thoughtResumable,
        replay: geminiReplayTurns.length > 0,
      }),
    [messageContent, thoughtResumable, geminiReplayTurns],
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
  // other two are stamped on metadata so they survive a reload. A provider-reported reason
  // is on the metadata either way, and outranks a cancelled status.
  const stamped = readIncompleteInfo(metadata);
  const cancelled =
    status?.type === "incomplete" && status?.reason === "cancelled";
  const reason =
    cancelled && !isProviderReportedReason(stamped?.reason)
      ? ("cancelled" as const)
      : stamped?.reason;
  const carriedReasoning = thoughtResumable ? reasoning : "";

  const canResume =
    isLast &&
    !isRunning &&
    !researchRunId &&
    !researchActive &&
    continuable &&
    modeAllowsContinuation({
      fromAudioInput,
      audioOutputModel,
    }) &&
    Boolean(
      partial.trim() ||
        carriedReasoning.trim() ||
        geminiReplayTurns.length > 0,
    );

  const reasoningDuration = readThoughtDuration(metadata);
  // Hands the started run back, untyped: the only handle identified with THIS run.
  const startContinuation = useCallback((): unknown => {
    const messages = aui.thread().getState().messages;
    const index = messages.findIndex((message) => message.id === messageId);
    if (index < 0) {
      return undefined;
    }
    // Sibling of the resumed turn, so the branch picker can still reach the original.
    const parent = index > 0 ? messages[index - 1].id : null;
    const request: ContinuationRequest = {
      partial,
      ...(carriedReasoning ? { reasoning: carriedReasoning, reasoningDuration } : {}),
      ...(thoughtSignature ? { thoughtSignature } : {}),
      ...(thoughtParts.length > 0 ? { thoughtParts } : {}),
      ...(answerParts.length > 0 ? { answerParts } : {}),
      ...(geminiReplayTurns.length > 0 ? { geminiReplayTurns } : {}),
      ...providerCompactionContinuationFields(metadata),
    };
    return aui.thread().startRun({
      parentId: parent,
      runConfig: {
        custom: { [CONTINUATION_RUN_CONFIG_KEY]: request },
      },
    });
  }, [
    aui,
    messageId,
    partial,
    carriedReasoning,
    reasoningDuration,
    thoughtSignature,
    thoughtParts,
    answerParts,
    geminiReplayTurns,
    metadata,
  ]);

  return {
    messageId,
    reason,
    completed: status?.type === "complete",
    canResume,
    resumedChars: partial.length + carriedReasoning.length,
    startContinuation,
  };
}

const ContinueMessageBar: FC = () => {
  // Mount the full subscriptions only for the newest message to keep typing responsive.
  const isLast = useAuiState(({ message }) => message.isLast);
  if (!isLast) {
    return null;
  }
  return <ContinueMessageBarForLastMessage />;
};

const ContinueMessageBarForLastMessage: FC = () => {
  const aui = useAui();
  const { messageId, reason, canResume, resumedChars, startContinuation } =
    useContinuation();

  const resumable = Boolean(reason) && canResume;

  // A cut with a remedy is one resuming cannot undo, so the way out replaces the button.
  const remedy = reason ? incompleteRemedy(reason) : null;

  // The parent is what every round of one logical turn shares; the message id changes
  // each round, because a continuation runs as a sibling.
  const parentId = useAuiState(({ thread, message }) => {
    const index = thread.messages.findIndex((m) => m.id === message.id);
    return index > 0 ? thread.messages[index - 1].id : null;
  });

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
      partialTokens: Math.ceil(resumedChars / 4),
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
        // The run's own promise is what ends the hold if this preflight is stopped.
        watchAutoContinueRun(messageId, runThreadId, startContinuation());
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
  // The remedy is owed even when nothing can be resumed: a tool-calling turn never can be.
  if (!reason || (!remedy && !resumable)) {
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

  return (
    <div className="aui-continue-bar mt-2 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-border/70 bg-muted/50 p-2.5 text-sm">
      <span className="min-w-0 flex-1 text-muted-foreground">
        {incompleteLabel(reason)}.{remedy ? ` ${remedy}.` : ""}
      </span>
      {remedy ? null : (
        <Button
          type="button"
          size="sm"
          variant="secondary"
          className="h-7 shrink-0 gap-1.5 text-xs"
          onClick={() => {
            startContinuation();
          }}
        >
          <QueueResumeIcon className="size-3.5" />
          Resume
        </Button>
      )}
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
// Read at render time, not module scope: the skill modules reach the chat barrel.
const ReadSkillToolUIConfirmable = withToolConfirmation((props) => (
  <ReadSkillToolUI {...props} />
));
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
      read_skill: ReadSkillToolUIConfirmable,
      studio_load_skill: ReadSkillToolUIConfirmable,
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
  // rising above the last turn that reported one, or a checkpoint starting inside a tool
  // loop (which evicts without moving the boundary). Sticky replays stay quiet.
  const showsNotice = useAuiState(({ thread }) =>
    compactionNoticeMessageIds(thread.messages).has(messageId),
  );
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
            {/* Borderless textarea, so auto-grow fits with no scrollbar; the wrapper keeps corners round. */}
            <div className="overflow-hidden rounded-xl border-[0.5px] border-border bg-muted focus-within:border-ring">
              <textarea
                ref={textareaRef}
                defaultValue={extractTaggedText(messageContent)}
                className="block w-full p-3 bg-transparent text-foreground outline-none overflow-y-auto resize-none font-mono text-sm max-h-[70dvh]"
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
            </div>
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
      <HugeiconsIcon icon={ForkIcon} strokeWidth={1.75} className="size-3" />
      {count}
    </span>
  );
};

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
      showForkCreatedToast(result.containerSnapshotWarning);
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
      tooltip="Fork in new chat"
      disabled={forkDisabled}
      onClick={forkMessage}
    >
      <HugeiconsIcon icon={ForkIcon} strokeWidth={1.75} className="size-[calc(var(--icon-size)*0.97)]" />
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

/** Deletes this message (a prompt with its replies); `hidden` for research messages. */
function useDeleteMessage() {
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

  return { handleDelete, isRunning, hidden: Boolean(researchRunId || ownsResearchMessage) };
}

// The More menu's Delete, last and in red.
const DeleteMessageMenuItem: FC = () => {
  const { handleDelete, isRunning, hidden } = useDeleteMessage();
  if (hidden) {
    return null;
  }
  return (
    <ActionBarMorePrimitive.Item
      disabled={isRunning}
      onSelect={() => void handleDelete()}
      className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-[12px] px-3 py-2 text-sm text-destructive outline-none hover:bg-destructive/10 focus:bg-destructive/10 data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
    >
      <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
      Delete
    </ActionBarMorePrimitive.Item>
  );
};

const MORE_MENU_CONTENT_CLASS =
  "aui-action-bar-more-content dropdown-surface z-50 min-w-32 max-h-(--radix-dropdown-menu-content-available-height) flex flex-col overflow-hidden rounded-[21px] bg-popover px-[calc(9px*var(--ui-space-scale,1))] py-2 text-popover-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)]";

const ForkMessageMenuItem: FC = () => {
  const { forkMessage, forkDisabled } = useForkMessageAction();
  return (
    <ActionBarMorePrimitive.Item
      disabled={forkDisabled}
      onSelect={() => void forkMessage()}
      className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-[12px] px-3 py-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
    >
      <HugeiconsIcon icon={ForkIcon} strokeWidth={1.75} className="size-icon" />
      Fork in new chat
    </ActionBarMorePrimitive.Item>
  );
};

// A prompt's More menu: Fork, then Delete.
const UserMoreMenu: FC = () => {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const collisionPadding = useWindowChromeCollisionPadding(undefined);
  return (
    <ActionBarMorePrimitive.Root modal={false}>
      <ActionBarMorePrimitive.Trigger asChild={true}>
        <TooltipIconButton ref={triggerRef} tooltip="More" className="data-[state=open]:bg-accent">
          <MoreHorizontalIcon strokeWidth={1.75} className="size-icon" />
        </TooltipIconButton>
      </ActionBarMorePrimitive.Trigger>
      <ActionBarMorePrimitive.Content
        side="bottom"
        align="end"
        collisionPadding={collisionPadding}
        onCloseAutoFocus={(e) => e.preventDefault()}
        className={MORE_MENU_CONTENT_CLASS}
      >
        <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
          <MenuDismissGuard triggerRef={triggerRef} />
          <ForkMessageMenuItem />
          <DeleteMessageMenuItem />
        </div>
      </ActionBarMorePrimitive.Content>
    </ActionBarMorePrimitive.Root>
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
        icon={EditResponseIcon}
        strokeWidth={1.75}
        className="size-icon"
      />
    </TooltipIconButton>
  );
};

/** The More menu's Continue response, for the newest finished reply; incomplete replies use the
 *  Resume bar. */
const ContinueResponseMenuItem: FC = () => {
  const isLast = useAuiState(({ message }) => message.isLast);
  if (!isLast) {
    return null;
  }
  return <ContinueResponseMenuItemForLastMessage />;
};

const ContinueResponseMenuItemForLastMessage: FC = () => {
  const { messageId, reason, completed, canResume, startContinuation } =
    useContinuation();
  const editing = useChatRuntimeStore((s) => s.editingMessageId === messageId);
  // The sibling carries no citations, and a continuation restarts their [N] numbering.
  const cited = useAuiState(({ message }) =>
    message.content.some((part) => part.type === "source"),
  );
  if (!completed || reason || !canResume || editing || cited) {
    return null;
  }
  return (
    <ActionBarMorePrimitive.Item
      onSelect={() => {
        startContinuation();
      }}
      className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-[12px] px-3 py-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
    >
      <HugeiconsIcon icon={ContinueArrowIcon} strokeWidth={1.75} className="size-icon" />
      Continue response
    </ActionBarMorePrimitive.Item>
  );
};

// The More menu's Edit response, shown when the button is not pinned to the bar.
const EditAssistantMessageMenuItem: FC = () => {
  const messageId = useAuiState(({ message }) => message.id);
  const researchRunId = useResearchMessageRunId();
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const researchActive = useThreadResearchActive();
  const setEditingId = useChatRuntimeStore((s) => s.setEditingMessageId);

  if (researchRunId) return null;

  return (
    <ActionBarMorePrimitive.Item
      disabled={isRunning || researchActive}
      onSelect={() => setEditingId(messageId)}
      className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-[12px] px-3 py-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
    >
      <HugeiconsIcon icon={EditResponseIcon} strokeWidth={1.75} className="size-icon" />
      Edit response
    </ActionBarMorePrimitive.Item>
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
  // Not built on DropdownMenuContent, so clear the titlebar and cap the height here.
  const moreMenuCollisionPadding = useWindowChromeCollisionPadding(undefined);
  const researchRunId = useResearchMessageRunId();
  const researchActive = useThreadResearchActive();
  const activeProjectId = useChatRuntimeStore((s) => s.activeProjectId);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const ttsEnabled = useVoiceSettingsStore((s) => s.ttsEnabled);
  // Off by default: Edit response lives in the More menu.
  const inlineEdit = useChatPreferencesStore((s) => s.showInlineEditResponse);
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
        {inlineEdit && <EditAssistantMessageButton />}
        <ForkCountBadge />
        <ForkMessageButton />
        {ttsEnabled && (
          <MessagePrimitive.If speaking={false}>
            <ActionBarPrimitive.Speak asChild={true}>
              <TooltipIconButton tooltip="Read aloud" aria-label="Read aloud">
                <HugeiconsIcon icon={ReadAloudIcon} strokeWidth={1.75} className="size-icon" />
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
              <HugeiconsIcon icon={VolumeMute02Icon} strokeWidth={1.75} className="size-icon" />
            </TooltipIconButton>
          </ActionBarPrimitive.StopSpeaking>
        </MessagePrimitive.If>
        {!researchRunId && !researchActive && (
          <ActionBarPrimitive.Reload asChild={true}>
            <TooltipIconButton tooltip="Refresh">
              <RefreshGlyph strokeWidth={1.75} className="size-icon" />
            </TooltipIconButton>
          </ActionBarPrimitive.Reload>
        )}
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
            collisionPadding={moreMenuCollisionPadding}
            onCloseAutoFocus={(e) => e.preventDefault()}
            className={MORE_MENU_CONTENT_CLASS}
          >
            {/* Scroll an inner viewport: a scrollbar on the rounded surface squares its corners.
                The surface padding insets it clear of the curve. */}
            <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
              {/* Keep the click that dismisses the menu off the bar's buttons. */}
              <MenuDismissGuard triggerRef={moreMenuTriggerRef} />
              <MessageMenuTime onShowDetails={() => setDetailsOpen(true)} />
              <ContinueResponseMenuItem />
              {!inlineEdit && <EditAssistantMessageMenuItem />}
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
                    icon={FolderAttachmentIcon}
                    strokeWidth={1.75}
                    className="size-icon"
                  />
                  Save to project sources
                </ActionBarMorePrimitive.Item>
              )}
              <DeleteMessageMenuItem />
            </div>
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
      <div className="flex items-center gap-2 rounded-lg border border-[color-mix(in_oklab,var(--foreground)_calc(20%*var(--contrast-edge-gain,1)),transparent)] bg-muted px-3 py-1.5 text-xs">
        <HeadphonesIcon className="size-3.5 text-muted-foreground" />
        <span className="max-w-48 truncate">{audioName}</span>
      </div>
    </div>
  );
};

const UserMessage: FC = () => {
  const focusReveal = useActionBarFocusReveal();
  // Attachments alone (annotations, images) get no empty bubble under them.
  const hasContent = useAuiState(({ message }) =>
    message.content.some((part) => part.type !== "text" || part.text.trim() !== ""),
  );
  return (
    <MessagePrimitive.Root
      className="aui-user-message-root fade-in slide-in-from-bottom-1 mx-auto flex w-full max-w-(--thread-content-max-width) animate-in flex-col items-end gap-y-2 pt-6 pb-4 text-ui-15p5 [font-weight:410] tracking-[0.01em] dark:tracking-[0.02em] duration-150"
      data-role="user"
      tabIndex={0}
      {...focusReveal}
    >
      <UserMessageAttachments />
      <UserMessageAudio />

      <div className="aui-user-message-content-wrapper flex w-full min-w-0 flex-col items-end">
        {hasContent ? (
          <div className="aui-user-message-content wrap-break-word w-fit max-w-[80%] rounded-[24px] bg-[#f5f5f5] px-4 py-2.5 text-[#0d0d0d] dark:text-foreground dark:bg-card">
            <MessagePrimitive.Parts />
          </div>
        ) : null}
        <UserMessageFooter>
          <UserActionBar />
          <BranchPicker className="aui-user-branch-picker ml-[calc(6px*var(--ui-space-scale,1))] shrink-0" />
        </UserMessageFooter>
      </div>
      {/* The other half of the pair: last is a user message while a reply is
          still to come, or once one has been deleted. */}
      <MessagePrimitive.If last={true}>
        <ForkChatShortcut />
      </MessagePrimitive.If>
      {/* Reverse traversal reaches a trailing stop before the root. Focusing it
          mounts the autohidden controls, so the next Shift+Tab enters More
          instead of skipping the action bar and leaving the message. */}
      <span
        className="aui-user-reveal-sentinel"
        tabIndex={0}
        aria-label="Message actions"
      />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  const ownsResearchMessage = useOwnsResearchMessage();
  const researchActive = useThreadResearchActive();
  return (
    <UserMessageActionBar>
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
      <UserMoreMenu />
    </UserMessageActionBar>
  );
};

const EditComposer: FC = () => {
  const aui = useAui();
  const sendShortcut = useChatPreferencesStore((s) => s.sendShortcut);
  const editMultiline = useAuiState(({ composer }) => composer.text.includes("\n"));
  const { inputProps, isComposingRef } = useImeComposerInputHandlers();
  const resendAfterCancelRef = useRef(false);
  const researchActive = useThreadResearchActive();
  // send() drops an empty composer, e.g. a paste-only message whose chip was removed.
  const editEmpty = useAuiState(({ composer }) => composer.isEmpty);

  useAuiEvent("thread.runEnd", () => {
    if (!resendAfterCancelRef.current) {
      return;
    }
    resendAfterCancelRef.current = false;
    aui.composer().send({ startRun: true });
  });

  // The one submit path. ComposerPrimitive.Root is a form and its Enter handler sends
  // without startRun, which drops an unchanged edit, so preventDefault here and take over.
  const submitEdit = useCallback(() => {
    if (isComposingRef.current) return;
    if (aui.thread().getState().isRunning) {
      resendAfterCancelRef.current = true;
      aui.thread().cancelRun();
      return;
    }
    // startRun forces a run when nothing was typed: the edit composer's send
    // drops a message whose text and attachments are unchanged.
    aui.composer().send({ startRun: true });
  }, [aui, isComposingRef]);

  return (
    <MessagePrimitive.Root className="aui-edit-composer-wrapper mx-auto flex w-full max-w-(--thread-content-max-width) flex-col py-3">
      <ComposerPrimitive.Root
        className="aui-edit-composer-root ml-auto flex w-full max-w-[85%] flex-col rounded-2xl bg-muted"
        onSubmit={(event) => {
          event.preventDefault();
          submitEdit();
        }}
      >
        <ComposerAttachments className="mb-0 px-3 pt-3" />
        <ComposerPrimitive.Input
          submitMode={
            effectiveSendShortcut(sendShortcut, editMultiline ? "\n" : "") === "mod-enter"
              ? "ctrlEnter"
              : "enter"
          }
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
          <Button type="submit" size="sm" disabled={researchActive || editEmpty}>
            Send
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
          <HugeiconsIcon icon={BranchPrevIcon} strokeWidth={1.75} className="size-4" />
        </button>
      </BranchPickerPrimitive.Previous>
      <span className="aui-branch-picker-state text-ui-13 leading-none tabular-nums">
        <BranchPickerPrimitive.Number />/<BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild={true}>
        <button
          type="button"
          aria-label="Next"
          className="aui-branch-chevron-btn"
        >
          <HugeiconsIcon icon={BranchNextIcon} strokeWidth={1.75} className="size-4" />
        </button>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};
