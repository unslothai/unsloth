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
import { MessageHtmlArtifacts } from "@/components/assistant-ui/message-html-artifacts";
import {
  MessageResponseDetailsSheet,
  MessageResponseModelBadge,
} from "@/components/assistant-ui/message-response-details-sheet";
import { MessageTiming } from "@/components/assistant-ui/message-timing";
import { Reasoning, ReasoningGroup } from "@/components/assistant-ui/reasoning";
import { RagSourcesGroup } from "@/components/assistant-ui/rag-sources";
import { Sources, SourcesGroup } from "@/components/assistant-ui/sources";
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
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
} from "@/features/chat/adapters/studio-dictation-adapter";
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
import {
  CHAT_HISTORY_UPDATED_EVENT,
  forkChatThread,
  getForkCount,
} from "@/features/chat/api/chat-api";
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
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { useChatProjects } from "@/features/chat/hooks/use-chat-projects";
import { NewProjectDialog } from "@/features/chat/components/new-project-dialog";
import { parseExternalModelId } from "@/features/chat/external-providers";
import { McpComposerButton } from "@/features/chat/mcp-composer-button";
import { getExternalReasoningCapabilities } from "@/features/chat/provider-capabilities";
import { useRagToolDisabled } from "@/features/chat/hooks/use-rag-tool-disabled";
import { BypassPermissionsMenuItem } from "@/features/chat/bypass-permissions-menu-item";
import { PermissionModeComposerPill } from "@/features/chat/permission-mode-select";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { PROMPT_QUEUE_STOP_EVENT } from "@/features/chat/utils/prompt-queue-boundary";
import {
  PLUS_MENU_ORDER,
  composerDraftKey,
  readComposerDraft,
  type PlusMenuItemId,
  usePlusMenuPrefsStore,
  writeComposerDraft,
} from "@/features/chat";
import { deleteThreadMessage } from "@/features/chat/utils/delete-thread-message";
import { listThreadDocuments } from "@/features/rag/api/rag-api";
import { ThreadDocumentsBar } from "@/features/rag/components/thread-documents-bar";
import { KnowledgeBaseComposerButton } from "@/features/rag/components/knowledge-base-composer-button";
import { DocumentPreviewMount } from "@/features/rag/components/document-preview-mount";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { applyQwenThinkingParams } from "@/features/chat/utils/qwen-params";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
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
  GlobeIcon,
  HeadphonesIcon,
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
  type ComponentProps,
  type CompositionEvent,
  type FC,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  type ReactNode,
  Fragment,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { create } from "zustand";
import { extractTaggedText, updateThreadMessage } from "@/features/chat/utils/update-thread-message";
import { useIsMobile } from "@/hooks/use-mobile";

// True while a file is dragged anywhere over the chat page, so the composer
// can show its "Drop files here" affordance.
const PageDragContext = createContext(false);

// Single-chat prompt queue. State lives at module level so it survives the
// Composer remount when the first queued message creates a new thread, and
// detection subscribes to runningByThreadId instead of aui.thread() so the
// welcome-screen composer can queue safely before a thread is bound.
type PromptQueueUIEntry = {
  current: number;
  total: number;
};

type PromptQueueUIItemStatus = "queued" | "next" | "waiting" | "running";

type PromptQueueUIItem = {
  id: string;
  prompt: string;
  position: number;
  total: number;
  status: PromptQueueUIItemStatus;
  threadIds: string[];
  canEdit: boolean;
  canRemove: boolean;
};

interface PromptQueueUIState {
  byThreadId: Record<string, PromptQueueUIEntry>;
  current: number;
  total: number;
  items: PromptQueueUIItem[];
  isRunning: boolean;
}

const usePromptQueueUI = create<PromptQueueUIState>(() => ({
  byThreadId: {},
  current: 0,
  total: 0,
  items: [],
  isRunning: false,
}));

type PromptQueueTarget = {
  getDocumentThreadId: () => string | null;
  getRunningThreadIds: () => string[];
  append: (prompt: string) => void;
  cancel: () => void;
  isIndexing: () => boolean;
};

type PromptQueueItem = {
  id: string;
  prompt: string;
  target: PromptQueueTarget;
  dispatched: boolean;
};

const PROMPT_QUEUE_INDEXING_RETRY_MS = 500;

let promptQueueItems: PromptQueueItem[] = [];
let promptQueueIndex = 0;
let promptQueueIsRunning = false;
let promptQueueGeneration = 0;
let promptQueuePrevStoreRunning = false;
let promptQueueWaitingForTargetIdle = false;
let promptQueueStoreUnsub: (() => void) | null = null;
let promptQueueRetryTimer: ReturnType<typeof setTimeout> | null = null;

function compactIds(ids: Array<string | null | undefined>) {
  return Array.from(new Set(ids.filter((id): id is string => Boolean(id))));
}

function createPromptQueueItemId() {
  return `prompt-queue-${crypto.randomUUID()}`;
}

function stopPromptQueueSubscription({
  resetRunningState = true,
}: {
  resetRunningState?: boolean;
} = {}) {
  if (promptQueueStoreUnsub) {
    promptQueueStoreUnsub();
    promptQueueStoreUnsub = null;
  }
  if (resetRunningState) {
    promptQueuePrevStoreRunning = false;
  }
}

function resetPromptQueue() {
  promptQueueGeneration += 1;
  promptQueueIsRunning = false;
  promptQueueItems = [];
  promptQueueIndex = 0;
  promptQueueWaitingForTargetIdle = false;
  if (promptQueueRetryTimer) {
    clearTimeout(promptQueueRetryTimer);
    promptQueueRetryTimer = null;
  }
  stopPromptQueueSubscription();
  syncPromptQueueUI();
}

function appendQueuedPrompt(item: PromptQueueItem) {
  item.dispatched = true;
  syncPromptQueueUI();
  item.target.append(item.prompt);
}

async function targetHasIndexingDocuments(item: PromptQueueItem) {
  if (item.target.isIndexing()) {
    return true;
  }
  const state = useChatRuntimeStore.getState();
  if (
    !state.ragEnabled ||
    state.ragSource.type !== "thread"
  ) {
    return false;
  }
  const threadId = item.target.getDocumentThreadId();
  if (!threadId) {
    return false;
  }
  try {
    const documents = await listThreadDocuments(threadId);
    return documents.some(
      (doc) => doc.status === "pending" || doc.status === "running",
    );
  } catch {
    return item.target.isIndexing();
  }
}

function isActivePromptQueueItem(item: PromptQueueItem, generation: number) {
  if (!promptQueueIsRunning || generation !== promptQueueGeneration) {
    return false;
  }
  return promptQueueItems[Math.max(promptQueueIndex, 0)] === item;
}

function scheduleQueuedPromptDispatch(
  item: PromptQueueItem,
  delay: number,
  generation = promptQueueGeneration,
) {
  if (promptQueueRetryTimer) {
    clearTimeout(promptQueueRetryTimer);
  }
  promptQueueRetryTimer = setTimeout(() => {
    promptQueueRetryTimer = null;
    void dispatchQueuedPrompt(item, generation);
  }, delay);
}

async function dispatchQueuedPrompt(
  item: PromptQueueItem,
  generation = promptQueueGeneration,
) {
  if (!isActivePromptQueueItem(item, generation)) {
    return;
  }
  if (
    isPromptQueueTargetRunning(
      item.target,
      useChatRuntimeStore.getState().runningByThreadId,
    )
  ) {
    promptQueueWaitingForTargetIdle = true;
    promptQueuePrevStoreRunning = true;
    syncPromptQueueUI();
    startPromptQueueSubscription();
    return;
  }
  const hasIndexingDocuments = await targetHasIndexingDocuments(item);
  if (!isActivePromptQueueItem(item, generation)) {
    return;
  }
  if (hasIndexingDocuments) {
    scheduleQueuedPromptDispatch(item, PROMPT_QUEUE_INDEXING_RETRY_MS);
    return;
  }
  if (!isActivePromptQueueItem(item, generation)) {
    return;
  }
  appendQueuedPrompt(item);
}

function createQueuedPrompt(prompt: string, target: PromptQueueTarget) {
  return {
    id: createPromptQueueItemId(),
    prompt,
    target,
    dispatched: false,
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

function promptQueueItemMatchesThreadIds(
  item: PromptQueueUIItem,
  threadIds: string[],
) {
  return item.threadIds.some((threadId) => threadIds.includes(threadId));
}

function syncPromptQueueUI() {
  if (!promptQueueIsRunning || promptQueueItems.length === 0) {
    usePromptQueueUI.setState({
      byThreadId: {},
      current: 0,
      total: 0,
      items: [],
      isRunning: false,
    });
    return;
  }

  const activeItemIndex = Math.max(promptQueueIndex, 0);
  const total = promptQueueItems.length;
  const current = promptQueueIndex >= 0 ? Math.min(activeItemIndex + 1, total) : 0;
  const items = promptQueueItems
    .map((item, index): PromptQueueUIItem | null => {
      if (index < activeItemIndex || item.dispatched) {
        return null;
      }
      const threadIds = getPromptQueueTargetIds(item.target);
      const isActive = promptQueueIndex >= 0 && index === activeItemIndex;
      const status: PromptQueueUIItemStatus = item.dispatched
        ? "running"
        : isActive
          ? promptQueueWaitingForTargetIdle
            ? "waiting"
            : "next"
          : "queued";
      return {
        id: item.id,
        prompt: item.prompt,
        position: index + 1,
        total,
        status,
        threadIds,
        canEdit: canEditPromptQueueItem(item),
        canRemove: canRemovePromptQueueItem(item),
      };
    })
    .filter((item): item is PromptQueueUIItem => Boolean(item));
  const groups: Array<{
    ids: Set<string>;
    current: number;
    total: number;
    active: boolean;
  }> = [];

  for (const [index, item] of promptQueueItems.entries()) {
    const ids = getPromptQueueTargetIds(item.target);
    if (ids.length === 0) {
      continue;
    }
    let group = groups.find((candidate) =>
      ids.some((id) => candidate.ids.has(id)),
    );
    if (!group) {
      group = {
        ids: new Set<string>(),
        current: 0,
        total: 0,
        active: false,
      };
      groups.push(group);
    }
    ids.forEach((id) => group.ids.add(id));
    group.total += 1;
    if (promptQueueIndex >= 0 && index <= promptQueueIndex) {
      group.current += 1;
    }
    if (index === activeItemIndex) {
      group.active = true;
    }
  }

  const byThreadId: Record<string, PromptQueueUIEntry> = {};
  for (const group of groups) {
    if (!group.active && group.current >= group.total) {
      continue;
    }
    const entry = {
      current: Math.min(group.current, group.total),
      total: group.total,
    };
    group.ids.forEach((id) => {
      byThreadId[id] = entry;
    });
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
  const itemIndex = promptQueueItems.findIndex(
    (candidate) => candidate.id === itemId,
  );
  if (itemIndex < 0) {
    return false;
  }
  const item = promptQueueItems[itemIndex];
  if (!canEditPromptQueueItem(item)) {
    return false;
  }
  item.prompt = nextPrompt;
  syncPromptQueueUI();
  return true;
}

function clearPromptQueueRetryTimer() {
  if (!promptQueueRetryTimer) {
    return;
  }
  clearTimeout(promptQueueRetryTimer);
  promptQueueRetryTimer = null;
}

function removePromptQueueItem(itemId: string) {
  const itemIndex = promptQueueItems.findIndex((item) => item.id === itemId);
  if (itemIndex < 0) {
    return false;
  }
  const item = promptQueueItems[itemIndex];
  if (!canRemovePromptQueueItem(item)) {
    return false;
  }

  const wasActive =
    promptQueueIndex >= 0 && itemIndex === Math.max(promptQueueIndex, 0);
  promptQueueItems.splice(itemIndex, 1);
  if (promptQueueItems.length === 0) {
    resetPromptQueue();
    return true;
  }

  if (itemIndex < promptQueueIndex) {
    promptQueueIndex -= 1;
  }
  if (wasActive && promptQueueIndex >= promptQueueItems.length) {
    resetPromptQueue();
    return true;
  }

  syncPromptQueueUI();
  if (wasActive) {
    clearPromptQueueRetryTimer();
    promptQueueWaitingForTargetIdle = false;
    promptQueuePrevStoreRunning = false;
    const next = promptQueueItems[promptQueueIndex];
    if (next) {
      scheduleQueuedPromptDispatch(next, 50);
    }
  }
  return true;
}

function isPromptQueueTargetRunning(
  target: PromptQueueTarget,
  runningByThreadId: Record<string, boolean>,
) {
  const runningIds = Object.keys(runningByThreadId);
  if (runningIds.length === 0) {
    return false;
  }

  const targetIds = target.getRunningThreadIds();
  if (targetIds.length === 0) {
    return runningIds.length > 0;
  }

  return runningIds.some((threadId) => targetIds.includes(threadId));
}

function isActivePromptQueueTargetRunning(
  runningByThreadId: Record<string, boolean>,
) {
  const activeItem = promptQueueItems[Math.max(promptQueueIndex, 0)];
  if (!activeItem) {
    return false;
  }
  return isPromptQueueTargetRunning(activeItem.target, runningByThreadId);
}

function advancePromptQueue() {
  const nextIndex = promptQueueIndex + 1;
  if (nextIndex >= promptQueueItems.length) {
    resetPromptQueue();
    return;
  }
  promptQueueIndex = nextIndex;
  syncPromptQueueUI();
  const next = promptQueueItems[nextIndex];
  promptQueueWaitingForTargetIdle = false;
  promptQueuePrevStoreRunning = false;
  scheduleQueuedPromptDispatch(next, 100);
}

function startPromptQueueSubscription() {
  const wasWaitingForRun = promptQueuePrevStoreRunning;
  stopPromptQueueSubscription({ resetRunningState: false });
  promptQueuePrevStoreRunning = wasWaitingForRun;
  // runningByThreadId tracks the actual thread (not aui.thread()), so detection
  // survives navigation.
  promptQueueStoreUnsub = useChatRuntimeStore.subscribe((state) => {
    if (!promptQueueIsRunning) {
      stopPromptQueueSubscription();
      return;
    }
    const isRunning = isActivePromptQueueTargetRunning(state.runningByThreadId);
    const wasRunning = promptQueuePrevStoreRunning;
    promptQueuePrevStoreRunning = isRunning;
    if (wasRunning && !isRunning) {
      if (promptQueueWaitingForTargetIdle) {
        promptQueueWaitingForTargetIdle = false;
        const activeItem = promptQueueItems[promptQueueIndex];
        if (activeItem) {
          scheduleQueuedPromptDispatch(activeItem, 50);
        }
        return;
      }
      advancePromptQueue();
    }
  });

  const isRunningNow = isActivePromptQueueTargetRunning(
    useChatRuntimeStore.getState().runningByThreadId,
  );
  if (promptQueuePrevStoreRunning && !isRunningNow) {
    promptQueuePrevStoreRunning = false;
    if (promptQueueWaitingForTargetIdle) {
      promptQueueWaitingForTargetIdle = false;
      const activeItem = promptQueueItems[promptQueueIndex];
      if (activeItem) {
        scheduleQueuedPromptDispatch(activeItem, 50);
      }
      return;
    }
    advancePromptQueue();
  }
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

  if (promptQueueIsRunning) {
    promptQueueItems.push(
      ...filtered.map((prompt) => createQueuedPrompt(prompt, target)),
    );
    syncPromptQueueUI();
    return;
  }

  const runningByThreadId = useChatRuntimeStore.getState().runningByThreadId;
  const shouldWaitForCurrentRun =
    waitForCurrentRun &&
    isPromptQueueTargetRunning(target, runningByThreadId);
  promptQueueGeneration += 1;
  promptQueueItems = filtered.map((prompt) =>
    createQueuedPrompt(prompt, target),
  );
  promptQueueIndex = shouldWaitForCurrentRun ? -1 : 0;
  promptQueueIsRunning = true;
  promptQueuePrevStoreRunning = shouldWaitForCurrentRun;
  syncPromptQueueUI();
  startPromptQueueSubscription();
  if (!shouldWaitForCurrentRun) {
    const first = promptQueueItems[0];
    if (first) {
      scheduleQueuedPromptDispatch(first, 50);
    }
  }
}

function stopPromptQueueRun() {
  const activeItem = promptQueueItems[Math.max(promptQueueIndex, 0)];
  const activeTarget = activeItem?.target;
  const shouldCancelActiveRun = Boolean(activeItem?.dispatched);
  resetPromptQueue();
  if (!shouldCancelActiveRun) {
    return;
  }
  try {
    activeTarget.cancel();
  } catch {
    // The active run may have already ended.
  }
}

if (typeof window !== "undefined") {
  window.addEventListener(PROMPT_QUEUE_STOP_EVENT, () => stopPromptQueueRun());
}

interface PromptQueueCallbacks {
  startQueue: (items: string[], waitForCurrentRun?: boolean) => void;
  stopQueue: () => void;
}
const noopStartPromptQueue: PromptQueueCallbacks["startQueue"] = () =>
  undefined;
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
              hideComposer
                ? "pt-4"
                : "pt-[calc(var(--studio-content-top-inset,0px)+48px)]",
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
    (s) =>
      Boolean(findPromptQueueEntry(s, promptQueueThreadIds)) &&
      s.items.some((item) =>
        promptQueueItemMatchesThreadIds(item, promptQueueThreadIds),
      ),
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
      <div className="aui-thread-welcome-center flex w-full grow flex-col items-center justify-start pt-[27.5vh]">
        <div className="aui-thread-welcome-message flex w-full flex-col justify-center gap-9 px-4">
          {/* Center the greeting (sloth + title) over the composer. */}
          <div className="flex flex-row items-center justify-center gap-[15px]">
            {showGreetingSloth && (
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
  // More than 4 pills: collapse to icons only. Search, Code, and permissions
  // always show; Images, RAG, Canvas and MCP are conditional. Narrow viewports
  // collapse too: the labelled row is wider than a phone-width composer.
  const isMobile = useIsMobile();
  const pillCount =
    3 +
    (ragEnabled ? 1 : 0) +
    (supportsBuiltinImageGeneration ? 1 : 0) +
    (artifactsEnabled ? 1 : 0) +
    (mcpEnabledForChat ? 1 : 0);
  const pillsCompact = isMobile || pillCount > 4;
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const setPendingImageEditReference = useChatRuntimeStore(
    (s) => s.setPendingImageEditReference,
  );
  const { inputProps, isComposing, isComposingRef } =
    useImeComposerInputHandlers({ submitOnEnter: true });
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
  const threadIsRunning = useAuiState(({ thread }) => thread.isRunning);
  const threadListItemId = useAuiState(
    ({ threadListItem }) => threadListItem.id,
  );
  const threadListItemRemoteId = useAuiState(
    ({ threadListItem }) => threadListItem.remoteId,
  );
  const referenceThreadId = threadId ?? activeThreadId ?? null;
  const promptQueueThreadIds = compactIds([
    threadListItemId,
    threadListItemRemoteId,
    threadId,
  ]);
  const promptQueueActive = usePromptQueueUI((s) =>
    Boolean(findPromptQueueEntry(s, promptQueueThreadIds)),
  );
  useEffect(() => {
    if (threadId != null || activeThreadId != null) {
      return;
    }
    stopPromptQueueRun();
  }, [activeThreadId, threadId]);
  const hasSendableContent =
    composerText.trim().length > 0 || hasAttachments || hasPendingAudio;
  const canQueueCurrentPrompt =
    composerText.trim().length > 0 &&
    !hasAttachments &&
    !hasPendingAudio &&
    !isComposing &&
    !hasPendingAttachments &&
    !disabled &&
    !overlay;

  // Per-thread draft autosave: restore on mount, then mirror composer text
  // into localStorage (debounced) so a half-typed message survives a
  // navigation or reload. Cleared once empty (i.e. after a send). Setting the
  // text even when no draft exists keeps a thread from inheriting the
  // previous thread's composer contents.
  const draftThreadId = referenceThreadId;
  const draftKey = draftThreadId ? composerDraftKey(draftThreadId) : null;
  const lastDraftKeyRef = useRef(draftKey);
  useEffect(() => {
    const draft = draftKey ? (readComposerDraft(draftKey) ?? "") : "";
    const composer = aui.composer();
    if (composer.getState().isEditing) {
      composer.setText(draft);
    }
  }, [draftKey, aui]);
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
    return () => clearTimeout(t);
  }, [composerText, draftKey]);
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
  const [pendingSend, setPendingSend] = useState(false);
  const pendingSendRef = useRef(false);
  const waitToastRef = useRef<string | number | null>(null);

  const handleIndexingChange = useCallback((active: boolean) => {
    indexingActiveRef.current = active;
    setIndexingActive(active);
  }, []);

  const createPromptQueueTarget = useCallback((): PromptQueueTarget => {
    const thread = aui.thread();
    const threadListItem = aui.threadListItem();
    const initialState = threadListItem.getState();
    const initialRunningThreadIds = [
      initialState.id,
      initialState.remoteId,
      referenceThreadId,
    ].filter((id): id is string => Boolean(id));
    const initialDocumentThreadId =
      initialState.remoteId ?? referenceThreadId ?? null;
    return {
      getDocumentThreadId: () => {
        const state = threadListItem.getState();
        return state.remoteId ?? referenceThreadId ?? initialDocumentThreadId;
      },
      getRunningThreadIds: () => {
        const state = threadListItem.getState();
        return Array.from(
          new Set(
            [
              ...initialRunningThreadIds,
              state.id,
              state.remoteId,
              referenceThreadId,
            ].filter((id): id is string => Boolean(id)),
          ),
        );
      },
      append: (prompt) => {
        thread.append(appendTextToThread(prompt));
      },
      cancel: () => thread.cancelRun(),
      isIndexing: () => indexingActiveRef.current,
    };
  }, [aui, referenceThreadId]);

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
      if (disabled || shouldBlockSend()) {
        event.preventDefault();
        return;
      }

      if (threadIsRunning || promptQueueActive) {
        event.preventDefault();
        // Project new-chat composer: never queue, just ask the user to wait.
        if (disableQueue) {
          toast.error("Wait for the current response to finish");
          return;
        }
        if (!canQueueCurrentPrompt) {
          if (overlay || hasAttachments || hasPendingAudio) {
            toast.error(
              threadIsRunning
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
        const queuedPrompt = composerText.trim();
        flushResourcesSync(() => {
          aui.composer().setText("");
        });
        startPromptQueue(
          [queuedPrompt],
          createPromptQueueTarget(),
          threadIsRunning,
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
      canQueueCurrentPrompt,
      closeOverlay,
      composerText,
      createPromptQueueTarget,
      disabled,
      disableQueue,
      hasAttachments,
      hasPendingAudio,
      interceptSend,
      overlay,
      promptQueueActive,
      referenceThreadId,
      setImageToolsEnabled,
      setPendingImageEditReference,
      shouldBlockSend,
      threadIsRunning,
    ],
  );

  const stopQueue = useCallback(() => {
    stopPromptQueueRun();
  }, []);

  const startQueue = useCallback(
    (items: string[], waitForCurrentRun = threadIsRunning) => {
      // Saved-prompt Run-list calls this directly, so honour disableQueue here
      // too: queuing from the project new-chat composer misbinds the thread.
      if (disableQueue) return;
      startPromptQueue(items, createPromptQueueTarget(), waitForCurrentRun);
    },
    [createPromptQueueTarget, threadIsRunning, disableQueue],
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
          the waveform remains the composer's only status indicator. */}
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
          className="unsloth-composer-left"
          data-pill-compact={pillsCompact ? "true" : undefined}
        >
          <ComposerToolsMenu side={effectiveMenuSide} />
          {/* While dictating, show only the "+"; hide the pill and tool
              toggles so the waveform is the sole status indicator. */}
          {!isDictating ? (
            <>
              {/* Permission-level pill: always visible and opens the
                  permission level dropdown. */}
              <PermissionModeComposerPill side={effectiveMenuSide} />
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
          // The recording UI replaces the input and send controls while
          // dictating; only the left plus stays visible alongside it.
          <ChatDictationBar />
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
              queueDisabled={disableQueue || !canQueueCurrentPrompt}
              onQueueClick={() => {
                if (disableQueue) return;
                const queuedPrompt = composerText.trim();
                if (queuedPrompt.length === 0) {
                  return;
                }
                flushResourcesSync(() => {
                  aui.composer().setText("");
                });
                startPromptQueue([queuedPrompt], createPromptQueueTarget(), true);
              }}
              onSendClick={interceptSend}
              onStopClick={stopQueue}
              pendingSend={pendingSend}
              menuSide={effectiveMenuSide}
              queueThreadIds={promptQueueThreadIds}
            />
          </>
        )}
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
      <PromptQueueStack queueThreadIds={promptQueueThreadIds} />
      {isTauri ? (
        // Phase 1 native model owns Tauri local-path drops. Restore browser
        // attachment drops in Tauri once Phase 1d adds token bridging.
        <div className="aui-composer-attachment-dropzone unsloth-composer-surface relative z-10">
          {composerContent}
        </div>
      ) : (
        <ComposerPrimitive.AttachmentDropzone className="group/dropzone aui-composer-attachment-dropzone unsloth-composer-surface relative z-10">
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
// a WSL-hosted Unsloth (issue #5546), `compositionend` never fires after the
// candidate commits, so `composingRef` stays true and Send stays disabled.
// Every compositionupdate / non-composing input resets the timer; only a true
// gap-after-commit lets it fire. 2500ms is above a normal candidate-window
// pause but short enough to recover before the user notices Send is stuck.
const IME_STUCK_TIMEOUT_MS = 2500;

function useImeComposerInputHandlers({
  submitOnEnter = false,
}: {
  submitOnEnter?: boolean;
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
        return;
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
      if (submitOnEnter && e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        e.currentTarget.form?.requestSubmit();
      }
    },
    [refreshStuckTimer, setCompositionState, submitOnEnter],
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
  const showCanvasMenuItem = useChatRuntimeStore((s) => s.showCanvasMenuItem);
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
  const incognito = useChatRuntimeStore((s) => s.incognito);
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
            Raw JSONL
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

const PromptQueueStack: FC<{ queueThreadIds: string[] }> = ({
  queueThreadIds,
}) => {
  const queueEntry = usePromptQueueUI((s) =>
    findPromptQueueEntry(s, queueThreadIds),
  );
  const items = usePromptQueueUI((s) => s.items);
  const [editingItemId, setEditingItemId] = useState<string | null>(null);
  const [draftPrompt, setDraftPrompt] = useState("");
  const editInputRef = useRef<HTMLTextAreaElement>(null);
  const visibleItems = items.filter((item) =>
    promptQueueItemMatchesThreadIds(item, queueThreadIds),
  );
  const editingItem = visibleItems.find((item) => item.id === editingItemId);
  const editingItemCanEdit = editingItem?.canEdit ?? false;
  const activeEditingItemId = editingItem ? editingItemId : null;

  useEffect(() => {
    if (!activeEditingItemId) {
      return;
    }
    editInputRef.current?.focus();
    editInputRef.current?.select();
  }, [activeEditingItemId]);

  useEffect(() => {
    if (!editingItemId || editingItemCanEdit) {
      return;
    }
    setEditingItemId(null);
    setDraftPrompt("");
  }, [editingItemCanEdit, editingItemId]);

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

  return (
    <div
      className="relative z-0 mx-7 mb-[-8px] max-h-[28vh] overflow-y-auto rounded-t-[18px] rounded-b-none border border-border/45 bg-background/90 px-5 py-2 text-muted-foreground shadow-none backdrop-blur-md dark:bg-card/85"
      aria-label={`Prompt queue, ${current} of ${total}`}
    >
      <div className="divide-y divide-border/25">
        {visibleItems.map((item, visibleIndex) => {
          const isEditing = item.id === activeEditingItemId;
          const visiblePosition = visibleIndex + 1;
          return (
            <div
              key={item.id}
              className={cn("min-h-10", isEditing ? "h-auto" : "h-10")}
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
                    <CornerDownRightIcon className="size-4 shrink-0 text-muted-foreground/50" />
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
  pendingSend?: boolean;
  menuSide?: "top" | "bottom";
  queueThreadIds: string[];
}> = ({
  disabled,
  queueDisabled,
  onQueueClick,
  onSendClick,
  onStopClick,
  pendingSend,
  menuSide,
  queueThreadIds,
}) => {
  const queueEntry = usePromptQueueUI((s) =>
    findPromptQueueEntry(s, queueThreadIds),
  );
  const isQueueRunning = Boolean(queueEntry);
  const aui = useAui();
  // Keep the mic clickable: if the engine can't run here, explain and point to
  // the local model instead of disabling the button.
  const startDictation = () => {
    if (!isStudioDictationAvailable()) {
      notifyStudioDictationUnavailable();
      return;
    }
    try {
      aui.composer().startDictation();
    } catch {
      notifyStudioDictationUnavailable();
    }
  };
  return (
    <div className="aui-composer-action-wrapper flex shrink-0 items-center gap-1.5">
      <ReasoningToggle side={menuSide} />
      {/* Starts dictation; the recording bar then covers the input row and owns
          the stop and discard actions. */}
      <ComposerPrimitive.If dictation={false}>
        <TooltipIconButton
          tooltip="Dictate"
          aria-label="Dictate"
          type="button"
          variant="ghost"
          className="size-8 rounded-full text-foreground"
          onClick={startDictation}
        >
          <MicIcon className="size-5" />
        </TooltipIconButton>
      </ComposerPrimitive.If>
      <AuiIf condition={({ thread }) => !thread.isRunning && !isQueueRunning}>
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
      {isQueueRunning ? (
        <AuiIf condition={({ thread }) => !thread.isRunning}>
          <TooltipIconButton
            tooltip="Queue message"
            side="bottom"
            type="button"
            variant="default"
            size="icon"
            disabled={disabled || queueDisabled}
            onClick={onQueueClick}
            className="aui-composer-send ml-1.5 size-8 rounded-full"
            aria-label="Queue message"
          >
            <ArrowUpIcon className="aui-composer-send-icon size-[21px] stroke-2" />
          </TooltipIconButton>
        </AuiIf>
      ) : null}
      <AuiIf condition={({ thread }) => thread.isRunning}>
        <div className="ml-1.5 flex items-center">
          {queueDisabled ? (
            <ComposerPrimitive.Cancel asChild={true}>
              <Button
                type="button"
                variant="default"
                size="icon"
                className="aui-composer-cancel size-8 rounded-full"
                aria-label="Stop generating"
                onClick={isQueueRunning ? onStopClick : undefined}
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
              className="aui-composer-send size-8 rounded-full"
              aria-label="Queue message"
            >
              <ArrowUpIcon className="aui-composer-send-icon size-[21px] stroke-2" />
            </TooltipIconButton>
          )}
        </div>
      </AuiIf>
    </div>
  );
};

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="aui-message-error-root mt-2 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md bg-destructive/10 p-3 text-destructive text-sm dark:bg-destructive/5 dark:text-red-200">
        <ErrorPrimitive.Message className="aui-message-error-message line-clamp-2 min-w-0 flex-1" />
        {/* Recovery path for interrupted/failed turns: regenerate in place. */}
        <ActionBarPrimitive.Reload asChild={true}>
          <button
            type="button"
            className="aui-message-error-retry inline-flex shrink-0 items-center gap-1.5 rounded-md border border-destructive/40 px-2.5 py-1 text-xs font-medium transition-colors hover:bg-destructive/15"
          >
            <RefreshCwIcon strokeWidth={1.75} className="size-3.5" />
            Retry
          </button>
        </ActionBarPrimitive.Reload>
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

// Live in-place denoising canvas for DiffusionGemma: while generating, render the
// latest per-step canvas snapshot in the bubble so the user watches the answer resolve
// out of noise. Transient (store-only, cleared on run end), so the finished message
// keeps only the committed markdown.
const DiffusionCanvas: FC = () => {
  const isRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  // A non-null canvas is set only by diffusion_frame events (diffusion models only),
  // so it is a sufficient gate; loadedIsDiffusion can lag the first frame on a fresh load.
  const canvas = useChatRuntimeStore((s) => s.activeDiffusionCanvas);
  if (!isRunning || !canvas) {
    return null;
  }
  const stepLabel =
    canvas.total > 0 ? `step ${canvas.step + 1}/${canvas.total}` : "denoising";
  return (
    <div className="aui-diffusion-canvas my-1.5 overflow-hidden rounded-lg border border-primary/20 bg-primary/[0.03]">
      <div className="flex items-center gap-2 border-b border-primary/10 px-3 py-1.5 text-[11px] font-medium text-primary/80">
        <span className="inline-block size-1.5 animate-pulse rounded-full bg-primary" />
        <span>Denoising</span>
        <span className="opacity-60">
          block {canvas.block + 1} - {stepLabel}
        </span>
      </div>
      <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap px-3 py-2 font-mono text-[12.5px] leading-relaxed text-foreground/90">
        {canvas.text}
      </pre>
    </div>
  );
};

/**
 * AssistantMessage handles the display and inline-editing of AI responses.
 *
 * It utilizes a "Tagged Text" system (<THINK> and <TOOL> tags) to allow users
 * to edit structured reasoning and tool outputs within a plain-text textarea
 * while preserving the underlying data schema and tool-call metadata.
 */
const AssistantMessage: FC = () => {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const messageContent = useAuiState(({ message }) => message.content);
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
    <MessagePrimitive.Root
      className="group/assistant-message aui-assistant-message-root relative mx-auto min-w-0 w-full max-w-(--thread-content-max-width) pt-0.5 pb-4 text-[15.5px] [font-weight:410] tracking-[0.01em] dark:tracking-[0.02em]"
      data-role="assistant"
    >
      <div className="aui-assistant-message-content wrap-break-word min-w-0 text-[#0d0d0d] dark:text-foreground leading-relaxed">
        {isEditing ? (
          <div className="flex flex-col gap-2 w-full">
            <textarea
              ref={textareaRef}
              defaultValue={extractTaggedText(messageContent)}
              className="w-full p-3 rounded-xl bg-muted border border-border text-foreground focus:ring-1 focus:ring-ring outline-none overflow-y-auto resize-none font-mono text-sm max-h-[70vh]"
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
            <GeneratingIndicator />
            <CancelledIndicator />
            <DiffusionCanvas />

            {/*
                We use the standard MessagePrimitive.Parts. This ensures that
                edited messages maintain the same professional styling,
                Markdown rendering, and tool-call components as original responses.
            */}
            <MessagePrimitive.Parts
              components={{
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
              }}
            />
            <SourcesGroup />
            <RagSourcesGroup />
            <MessageHtmlArtifacts />
            <MessageError />
          </>
        )}
      </div>

      <div className="aui-assistant-message-footer mt-1.5 -ml-[var(--icon-btn-inset)] flex min-h-8">
        <BranchPicker className="mr-0.5" />
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const COPY_RESET_MS = 2000;

const ForkCountBadge: FC = () => {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const [count, setCount] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      const remoteId = aui.threadListItem().getState().remoteId;
      if (!remoteId) {
        if (!cancelled) setCount(0);
        return;
      }
      void getForkCount(remoteId, messageId)
        .then((n) => {
          if (!cancelled) setCount(n);
        })
        .catch(() => {
          /* swallow: badge is non-critical */
        });
    };
    refresh();
    const handler = () => refresh();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, handler);
    return () => {
      cancelled = true;
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, handler);
    };
  }, [aui, messageId]);

  if (count <= 0) return null;
  return (
    <span
      className="mx-1 inline-flex items-center gap-1 rounded-sm bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary"
      title={`${count} fork${count === 1 ? "" : "s"} from this message`}
    >
      <GitBranchIcon strokeWidth={1.75} className="size-3" />
      {count}
    </span>
  );
};

const useForkMessageAction = () => {
  const aui = useAui();
  const navigate = useNavigate();
  const messageId = useAuiState(({ message }) => message.id);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const [pending, setPending] = useState(false);

  const handleFork = async () => {
    const remoteId = aui.threadListItem().getState().remoteId;
    if (!remoteId) {
      toast.error("Cannot fork an unsaved chat");
      return;
    }
    setPending(true);
    try {
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

const DeleteMessageButton: FC = () => {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);

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

const EditAssistantMessageButton: FC = () => {
  const messageId = useAuiState(({ message }) => message.id);
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const setEditingId = useChatRuntimeStore((s) => s.setEditingMessageId);

  return (
    <TooltipIconButton
      tooltip="Edit response"
      disabled={isRunning}
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
      content,
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
  const { forkMessage, forkDisabled } = useForkMessageAction();
  const [detailsOpen, setDetailsOpen] = useState(false);
  const ttsEnabled = useVoiceSettingsStore((s) => s.ttsEnabled);
  // hideWhenRunning is thread-level, so a new run would hide this bar and its
  // only Stop reading control while read-aloud keeps playing; keep it shown.
  const speaking = useAuiState(({ message }) => message.speech != null);

  return (
    <>
      <ActionBarPrimitive.Root
        hideWhenRunning={!speaking}
        className="aui-assistant-action-bar-root col-start-3 row-start-2 flex items-center gap-1 text-chat-icon-fg [&_button:not([data-slot=message-timing-trigger])]:size-8 [&_button]:!rounded-full [&_button:hover]:bg-chat-icon-bg-hover [&_button:hover]:text-chat-icon-fg-hover"
      >
        <CopyButton />
        <EditAssistantMessageButton />
        <ActionBarPrimitive.Reload asChild={true}>
          <TooltipIconButton tooltip="Refresh">
            <RefreshCwIcon strokeWidth={1.75} className="size-icon" />
          </TooltipIconButton>
        </ActionBarPrimitive.Reload>
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
            className="aui-action-bar-more-content z-50 min-w-32 overflow-hidden rounded-[21px] bg-popover px-[9px] py-2 text-popover-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-none"
          >
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
      className="aui-user-message-root fade-in slide-in-from-bottom-1 mx-auto flex w-full max-w-(--thread-content-max-width) animate-in flex-col items-end gap-y-2 pt-6 pb-4 text-[15.5px] [font-weight:410] tracking-[0.01em] dark:tracking-[0.02em] duration-150"
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
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      autohide="always"
      className="aui-user-action-bar-root flex gap-1 text-chat-icon-fg [&_button]:size-8 [&_button]:!rounded-full [&_button:hover]:bg-chat-icon-bg-hover [&_button:hover]:text-chat-icon-fg-hover"
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
