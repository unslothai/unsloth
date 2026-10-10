// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import {
  applyModelLoadConfigToRuntime,
  clearModelConfigHandoff,
  currentRuntimePerModelConfig,
  type DeletedModelRef,
  type ExternalConnectionRef,
  type ExternalModelOption,
  type LoraModelOption,
  type ModelOption,
  ModelSelector,
  type ModelSelectorChangeMeta,
  type PerModelConfig,
  isServedByMlx,
  loadedContextFields,
  modelConfigHandoffForDestination,
  resolveResidentInitialConfig,
  SidebarModelConfig,
  useActiveModelConfig,
  useModelConfigHandoffStore,
  pinnedReasoningEffort,
  useModelReasoningEffortStore,
} from "@/features/model-picker";
import { ProjectComposer, Thread } from "@/components/assistant-ui/thread";
import { usePlatformStore } from "@/config/env";
import { CopyableErrorChip } from "@/components/ui/copyable-error-chip";
import {
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { NonModalDropdownMenu } from "@/components/ui/non-modal-dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { useSidebar } from "@/components/ui/sidebar";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { useIsMobile } from "@/hooks/use-mobile";
import { holdSidebarPinned, releaseSidebarPinned } from "@/hooks/use-sidebar-pin";
import {
  DOWNLOAD_KIND,
  dismissStartToast,
  dismissStartToastsForModelSelection,
  downloadManager,
  jobKeyOf,
  useRepoDownload,
} from "@/features/hub/download-manager";
import {
  INVENTORY_FRESHNESS_WINDOW_MS,
  useDeviceInventorySources,
} from "@/features/hub/inventory";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import { ChatHeaderMenu } from "./components/chat-header-menu";
import { DeleteChatFilesSwitch } from "./components/delete-chat-files-switch";
import { chatLocalModelOptions } from "./local-model-options";
import {
  type NativeIntent,
  NativeAttachmentTargetContext,
  NativeModelChip,
  NativeModelDropOverlay,
  useNativeIntentStore,
  useNativeModelDrop,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import { isNpuModelId } from "@/features/npu";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { isTauri } from "@/lib/api-base";
import { chatModelLoaded } from "./lib/chat-model-loaded";
import { hasKnownContextWindow } from "./lib/context-window-known";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { setInAppLinkHandler } from "@/lib/open-link";
import {
  BrowserToggleButton,
  openUrlInBrowser,
  pinBrowserPage,
  setBrowserPanelAvailable,
  useBrowserStore,
} from "@/features/browser";
import {
  CONVERSATION_MARKDOWN_FORMAT,
  CONVERSATION_MARKDOWN_LABEL,
} from "./utils/conversation-markdown";
import {
  Archive03Icon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  FolderAttachmentIcon,
  Folder01Icon,
  Folder02Icon,
  FolderExportIcon,
  LayoutAlignRightIcon,
  MoreHorizontalIcon,
  MoreVerticalIcon,
  PinIcon,
  PinOffIcon,
  PencilEdit02Icon,
  Telescope02Icon,
} from "@hugeicons/core-free-icons";
import { useAui, useAuiState } from "@assistant-ui/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  SaveTemporaryChatMenu,
  TemporaryChatSaveBridge,
} from "./components/temporary-chat-save";
import { useT } from "@/i18n";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import {
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactElement,
  createContext,
  lazy,
  memo,
  Suspense,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import type { PanelImperativeHandle } from "react-resizable-panels";
import {
  CHAT_HISTORY_UPDATED_EVENT,
  notifyChatHistoryUpdated,
} from "./api/chat-api";
import { codeToolCanRun } from "./api/code-tool-placement";
import {
  clearAutoOpenedArtifacts,
  useChatArtifactsStore,
} from "./artifacts/store";
import { isKnownTextOnlySelection } from "./utils/model-vision-capability";
import { McpServersDialogMount } from "./mcp-composer-button";
import { ChatSettingsPanel } from "./chat-settings-sheet";
import {
  ResearchActivityPanel,
  ResearchActivitySheet,
} from "./components/research-activity-panel";
import { ChatModelNotice } from "./components/chat-model-notice";
import {
  chatModelSwitchMeta,
  type ChatModelSwitchTarget,
} from "./components/chat-model-notice-switch";
import { ContextUsageBar } from "./components/context-usage-bar";
import { ModelLoadInlineStatus } from "./components/model-load-status";
import { ProjectSwitcher } from "./components/project-switcher";
import { EditProjectDialog } from "./components/edit-project-dialog";
import {
  buildExternalModelId,
  isDecisionConnection,
  isExternalModelId,
  parseExternalModelId,

  providerModelSupportsStudioTools,
} from "./external-providers";
import { useChatModelRuntime } from "./hooks/use-chat-model-runtime";
import type { SelectedModelInput } from "./hooks/use-chat-model-runtime";
import {
  deleteChatProject,
  moveChatItemToProject,
  useChatProjects,
} from "./hooks/use-chat-projects";
import {
  type SidebarItem,
  archiveChatItem,
  deleteChatItem,
  renameChatItem,
  useChatSidebarItems,
} from "./hooks/use-chat-sidebar-items";
import { usePinnedChatsStore } from "./stores/pinned-chats-store";
import {
  normalizeSectionName,
  useSidebarOrganizationStore,
} from "./stores/sidebar-organization-store";
import { SectionNameDialog } from "./components/section-name-dialog";
import { ProjectMenuItems } from "./components/project-menu-items";
import { useFileProjectInSection } from "./hooks/use-file-project-in-section";
import {
  clearTrainingCompareHandoff,
  getTrainingCompareHandoff,
  normalizeModelRef,
  pickTrainingCompareTarget,
  trainingCompareSelection,
} from "./lib/training-compare-handoff";
import {
  externalReasoningTakesEffort,
  getExternalReasoningCapabilities,
  providerSupportsPreserveThinking,
  getProviderCapabilities,
  modelCatalogVersion,
  providerHostsCodeExecution,
  providerSupportsBuiltinCodeExecution,
  providerSupportsBuiltinImageGeneration,
  providerSupportsBuiltinWebFetch,
  providerSupportsBuiltinWebSearch,
  providerSupportsFastMode,
  reasoningFieldsAfterCatalogRefresh,
  resolveExternalReasoningEffort,
  subscribeModelCatalog,
} from "./provider-capabilities";
import {
  COMPOSER_INPUT_SELECTOR,
  isSurfaceBackgrounded,
  useShortcut,
} from "@/features/settings";
import {
  ChatActiveContext,
  ChatRuntimeProvider,
  useChatActive,
} from "./runtime-provider";
import {
  type CompareHandle,
  type CompareHandles,
  CompareHandlesProvider,
  RegisterCompareHandle,
  SharedComposer,
} from "./shared-composer";
import { BypassPermissionsConfirmDialog } from "./bypass-permissions-menu-item";
import { RootSandboxSetupDialog } from "./sandbox-setup-dialog";
import {
  CHAT_CODE_TOOLS_ENABLED_KEY,
  CHAT_IMAGE_TOOLS_ENABLED_KEY,
  CHAT_TOOLS_ENABLED_KEY,
  CHAT_WEB_FETCH_TOOLS_ENABLED_KEY,
  PENDING_CHAT_ATTACHMENT_KEY,
  loadOptionalBool,
  noteEffortDisplacedByPin,
  pinHoldsLiveEffort,
  readPendingAttachmentTargetClaim,
  reconcilePinnedReasoningEffort,
  takeEffortDisplacedByPin,
  threadScopedOverride,
  resolvePreserveThinkingOnLoad,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import { wantsDownloadManagerStaging } from "./utils/model-download-staging";
import { useChatPreferencesStore } from "./stores/chat-preferences-store";
import { useResearchRunStore } from "./stores/research-run-store";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import { buildChatTourSteps } from "./tour";
import type { ChatView, MessageRecord } from "./types";
import {
  type ComparePairReadState,
  checkpointCompareClass,
  comparePairReadState,
  resolveComparePaneThreadIds,
} from "./utils/compare-pane-threads";
import { clearNewChatDraft } from "./utils/composer-draft";
import { isChatThreadDeleted } from "./utils/chat-thread-tombstones";
import {
  getStoredChatThread,
  isExpectedBackgroundChatStorageError,
  listStoredChatMessages,
  listStoredChatThreads,
} from "./utils/chat-history-storage";
import { isCoalescedHistoryEvent } from "./utils/chat-history-revision";
import { attachmentsSample } from "./utils/pasted-text";
import {
  type DocumentAnnotations,
  createAnnotationsFile,
} from "./utils/document-annotations";
import { requestTemporaryPromptQueueStop } from "./utils/prompt-queue-boundary";
import { savedBranchHead } from "./utils/branch-head";
import { estimateContextUsage } from "./utils/estimate-chat-tokens";
import { orderBySelectedBranch } from "./utils/message-order";
import { isAssistantLocalThreadId } from "./utils/thread-ids";
import {
  consumeProjectSourcesPending,
  hasProjectSourcesPending,
  noteProjectLandingMounted,
} from "@/features/rag/components/project-source-dropzone";
import {
  exportConversationCsv,
  exportConversationMarkdown,
  exportConversationMessagesJsonl,
  exportConversationRawJsonl,
  exportConversationShareGPT,
  saveChatItemAsProjectSource,
} from "./prompt-storage/prompt-storage-dialog";

const BrowserPanel = lazy(() =>
  import("@/features/browser/browser-panel").then((module) => ({
    default: module.BrowserPanel,
  })),
);
const FullViewChatBar = lazy(() =>
  import("@/features/browser/full-view-chat").then((module) => ({
    default: module.FullViewChatBar,
  })),
);
const FullViewChatButton = lazy(() =>
  import("@/features/browser/full-view-chat").then((module) => ({
    default: module.FullViewChatButton,
  })),
);

// Sandboxed page frames stay outside the trap: same-origin access would weaken their isolation.
const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

// Only what is on screen: background tabs stay mounted under hidden or aria-hidden wrappers.
function focusableIn(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (element) =>
      element.tabIndex !== -1 &&
      !element.closest('[aria-hidden="true"], [inert]') &&
      element.getClientRects().length > 0 &&
      getComputedStyle(element).visibility !== "hidden",
  );
}

/** Puts a staged fix prompt in this view's composer once it is on screen. */
function useStagedFixPrompt(pendingFixPrompt: string | null, active: boolean): void {
  const aui = useAui();
  useEffect(() => {
    if (!pendingFixPrompt || !active) return;
    useChatArtifactsStore.getState().clearFixPrompt();
    const composer = aui.composer();
    const current = composer.getState().text;
    composer.setText(
      current.trim().length > 0
        ? `${current}\n\n${pendingFixPrompt}`
        : pendingFixPrompt,
    );
    // Focus the composer after the overlay returns focus to its opener.
    window.setTimeout(() => {
      document
        .querySelector<HTMLTextAreaElement>(COMPOSER_INPUT_SELECTOR)
        ?.focus();
    }, 0);
  }, [pendingFixPrompt, aui, active]);
}

// Compare keeps the base view mounted but hidden; the overlay owns the browser then, so it runs once.
const BrowserOverlaidContext = createContext(false);

/** The browser over the chat, where it cannot sit beside it. A modal: focus moves in, stays in, and returns on close. */
function BrowserOverlay(): ReactElement {
  const t = useT();
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement;
    const id = window.setTimeout(() => {
      const dialog = dialogRef.current;
      if (dialog && !dialog.contains(document.activeElement)) (focusableIn(dialog)[0] ?? dialog).focus();
    }, 0);
    return () => {
      window.clearTimeout(id);
      if (previous instanceof HTMLElement && previous.isConnected) previous.focus();
    };
  }, []);
  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      // Menus, fields and annotating handle their own Escape first.
      const target = event.target as HTMLElement;
      const typing = target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (event.defaultPrevented || typing || useBrowserStore.getState().annotateTabId !== null) return;
      event.preventDefault();
      useBrowserStore.getState().closePanel();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = focusableIn(event.currentTarget);
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!first || !last) {
      event.preventDefault();
      event.currentTarget.focus();
    } else if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm sm:p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          useBrowserStore.getState().closePanel();
        }
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal={true}
        aria-label={t("browser.title")}
        tabIndex={-1}
        onKeyDown={onKeyDown}
        className="size-full overflow-hidden border-border bg-background outline-none sm:h-[min(92dvh,900px)] sm:w-[min(96vw,1200px)] sm:rounded-2xl sm:border sm:shadow-xl"
      >
        <Suspense fallback={null}>
          <BrowserPanel active={true} />
        </Suspense>
      </div>
    </div>
  );
}

const ProjectSourcesPanel = lazy(() =>
  import("@/features/rag/components/project-sources-panel").then((module) => ({
    default: module.ProjectSourcesPanel,
  })),
);

const EXTERNAL_PROVIDER_DROPDOWN_ORDER: Record<string, number> = {
  openai: 0,
  anthropic: 1,
};

function getExternalProviderDropdownRank(providerType: string): number {
  return EXTERNAL_PROVIDER_DROPDOWN_ORDER[providerType] ?? 2;
}

type RuntimeStoreState = ReturnType<typeof useChatRuntimeStore.getState>;

// The newest saved usage if the active checkpoint and window could have produced it, else null.
function savedUsageFor(
  messages: MessageRecord[],
  store: RuntimeStoreState,
): RuntimeStoreState["contextUsage"] {
  const msg = [...messages].sort((a, b) => b.createdAt - a.createdAt)[0];
  const usage = msg?.metadata?.contextUsage as RuntimeStoreState["contextUsage"];
  if (!usage) return null;
  const activeCheckpoint = store.params.checkpoint;
  const usageModelId = (usage as { modelId?: unknown }).modelId;
  if (typeof usageModelId === "string" && usageModelId) {
    if (!activeCheckpoint || usageModelId !== activeCheckpoint) {
      return null;
    }
  }
  // llama.cpp stops at the window, so a count past it is stale; MLX runs past it, so its count stands.
  const limit = store.loadedIsGguf ? store.loadedContextLength : null;
  if (typeof limit === "number" && limit > 0 && (usage.contextTokens ?? usage.totalTokens ?? 0) > limit) {
    return null;
  }
  return usage;
}

function messageHasImage(message: MessageRecord): boolean {
  const contentParts = Array.isArray(message.content) ? message.content : [];
  if (contentParts.some((part) => part.type === "image")) {
    return true;
  }
  const attachments = Array.isArray(message.attachments)
    ? message.attachments
    : [];
  for (const attachment of attachments) {
    const parts = Array.isArray(attachment.content) ? attachment.content : [];
    for (const part of parts as Array<{ type?: string }>) {
      if (part?.type === "image") {
        return true;
      }
    }
  }
  return false;
}

/** Send browser annotations as their own message via the composer (same checks); a refused send or a draft leaves them staged. */
function sendDocumentAnnotations(
  aui: ReturnType<typeof useAui>,
  annotations: DocumentAnnotations,
  files: File[] = [],
): Promise<boolean> {
  const composer = aui.composer();
  const drafted = () => {
    const state = composer.getState();
    return { text: state.text.trim().length > 0, attachments: state.attachments.length };
  };
  // Text or files the user staged are their next message: the annotations join it unsent.
  const before = drafted();
  const hasDraft = before.text || before.attachments > 0;
  // Extra files first, after the draft check, so they don't count as a draft. They're optional:
  // one the model can't take (a screenshot on a text-only model) is skipped.
  let extras = 0;
  return files
    .reduce(
      (staged, file) =>
        staged.then(() =>
          composer.addAttachment(file).then(
            () => void extras++,
            () => undefined,
          ),
        ),
      Promise.resolve(),
    )
    .then(() => composer.addAttachment(createAnnotationsFile(annotations)))
    .then(() => {
      const form = [
        ...document.querySelectorAll<HTMLFormElement>("form.aui-composer-root"),
      ].find(
        (element) =>
          element.offsetParent !== null &&
          !element.closest('[aria-hidden="true"], [inert]'),
      );
      if (hasDraft || !form) {
        document
          .querySelector<HTMLTextAreaElement>(COMPOSER_INPUT_SELECTOR)
          ?.focus();
        return true;
      }
      // Two frames, so the composer has rendered the attachment it now sends.
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          const now = drafted();
          if (now.text || now.attachments > 1 + extras) return;
          form.requestSubmit();
        }),
      );
      return true;
    })
    .catch(() => false);
}

function useStoredChatTitle(threadId: string | null): string | undefined {
  const [title, setTitle] = useState<{ id: string; title: string } | null>(
    null,
  );
  useEffect(() => {
    if (!threadId) return;
    let live = true;
    const load = () => {
      getStoredChatThread(threadId)
        .then((thread) => {
          if (!live || !thread) return;
          setTitle((current) =>
            current?.id === threadId && current.title === thread.title
              ? current
              : { id: threadId, title: thread.title },
          );
        })
        .catch(() => undefined);
    };
    // Streaming saves fire this per chunk and never rename the chat.
    const onHistoryUpdated = (event: Event) => {
      if (!isCoalescedHistoryEvent(event)) load();
    };
    load();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    return () => {
      live = false;
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    };
  }, [threadId]);
  return title && title.id === threadId ? title.title : undefined;
}

const ARTIFACT_PANEL_DEFAULT_SIZE = "38%";
const BROWSER_PANEL_DEFAULT_SIZE = "50%";
const ARTIFACT_PANEL_TRANSITION_MS = 260;
const ARTIFACT_SURFACE_POP_DELAY_MS = 150;

const SingleContent = memo(function SingleContent({
  threadId,
}: {
  threadId?: string;
}): ReactElement {
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const isMobile = useIsMobile();
  const chatActive = useChatActive();
  const aui = useAui();
  // Compare keeps this view mounted but hidden, so the backgrounded copy leaves the prompt to SharedComposer.
  const pendingFixPrompt = useChatArtifactsStore(
    (state) => state.pendingFixPrompt,
  );
  const browserOpen = useBrowserStore((state) => state.open);
  const browserOpenSequence = useBrowserStore((state) => state.openSequence);
  const closeBrowser = useBrowserStore((state) => state.closePanel);
  useEffect(() => {
    if (!chatActive || isMobile) return;
    setBrowserPanelAvailable(true);
    setInAppLinkHandler(openUrlInBrowser);
    useBrowserStore.setState({
      requestEdits: (prompt) => useChatArtifactsStore.getState().stageFixPrompt(prompt),
      sendAnnotations: (annotations, files) => sendDocumentAnnotations(aui, annotations, files),
      attachToChat: (file) =>
        aui
          .composer()
          .addAttachment(file)
          .then(() => {
            document.querySelector<HTMLTextAreaElement>(COMPOSER_INPUT_SELECTOR)?.focus();
            return true;
          })
          .catch(() => false),
    });
    return () => {
      setBrowserPanelAvailable(false);
      setInAppLinkHandler(null);
      useBrowserStore.setState({ requestEdits: null, sendAnnotations: null, attachToChat: null });
    };
  }, [chatActive, isMobile, aui]);
  useStagedFixPrompt(pendingFixPrompt, chatActive);
  const openResearchRunId = useResearchRunStore((state) => state.openRunId);
  const closeResearchPanel = useResearchRunStore((state) => state.closePanel);
  useEffect(() => {
    if (!activeThreadId || !openResearchRunId) return;
    const openRun =
      useResearchRunStore.getState().sessions[openResearchRunId]?.run;
    if (openRun && openRun.threadId !== activeThreadId) closeResearchPanel();
  }, [activeThreadId, openResearchRunId, closeResearchPanel]);
  // A string, not the run: report deltas replace the run ~12x/s, and this owns the thread pane.
  const openResearchThreadId = useResearchRunStore((state) =>
    openResearchRunId
      ? state.sessions[openResearchRunId]?.run.threadId
      : undefined,
  );
  const artifactPanelRef = useRef<PanelImperativeHandle | null>(null);
  // Sampled on drag end, not onResize: that fires through the open/close animations too.
  const artifactPanelWidthRef = useRef<string | null>(null);
  const rememberArtifactPanelWidth = useCallback(() => {
    const size = artifactPanelRef.current?.getSize().asPercentage;
    if (size == null) return;
    // Dragged shut means closed.
    if (size <= 5) {
      closeBrowser();
      return;
    }
    artifactPanelWidthRef.current = `${size}%`;
  }, [closeBrowser]);
  const hasInitializedArtifactPanelRef = useRef(false);
  const [isArtifactLayoutAnimating, setIsArtifactLayoutAnimating] =
    useState(false);
  const [isArtifactPanelLayoutActive, setIsArtifactPanelLayoutActive] =
    useState(false);
  const [isArtifactSurfaceVisible, setIsArtifactSurfaceVisible] =
    useState(false);
  const researchMatchesThread = Boolean(
    openResearchThreadId &&
      openResearchThreadId === (threadId ?? activeThreadId),
  );
  const showResearchPanel = researchMatchesThread && !isMobile;
  const browserOverlaid = useContext(BrowserOverlaidContext);
  const showBrowserPanel = !showResearchPanel && !isMobile && !browserOverlaid && browserOpen;
  // Research outranks the browser in the side pane, so a new browser open (a card, a link) closes it.
  const handledBrowserOpenRef = useRef(browserOpenSequence);
  useEffect(() => {
    if (handledBrowserOpenRef.current === browserOpenSequence) return;
    handledBrowserOpenRef.current = browserOpenSequence;
    if (showResearchPanel && browserOpen && !browserOverlaid) closeResearchPanel();
  }, [browserOpenSequence, browserOpen, browserOverlaid, showResearchPanel, closeResearchPanel]);
  const browserFullView =
    useBrowserStore((state) => state.fullView) && showBrowserPanel;
  const chatDock = useBrowserStore((state) => state.chatDock);
  const chatOnRight =
    useBrowserStore((state) => state.chatSide === "right") &&
    showBrowserPanel &&
    !browserFullView;
  const defaultPanelSizeRef = useRef(ARTIFACT_PANEL_DEFAULT_SIZE);
  useEffect(() => {
    defaultPanelSizeRef.current = showBrowserPanel
      ? BROWSER_PANEL_DEFAULT_SIZE
      : ARTIFACT_PANEL_DEFAULT_SIZE;
  });
  const showContextPanel = showResearchPanel || showBrowserPanel;

  const artifactLayoutActive = showContextPanel || isArtifactPanelLayoutActive;
  const artifactPanelSettledOpen =
    showContextPanel &&
    isArtifactPanelLayoutActive &&
    !isArtifactLayoutAnimating;

  const artifactPanelThread = threadId ?? activeThreadId ?? null;
  // biome-ignore lint/correctness/useExhaustiveDependencies: resetting is the effect
  useEffect(() => {
    artifactPanelWidthRef.current = null;
  }, [artifactPanelThread]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: the open sequence is the re-expand trigger
  useEffect(() => {
    if (!showContextPanel || browserOpenSequence === 0) return;
    const panel = artifactPanelRef.current;
    if (!panel) return;
    if (!panel.isCollapsed() && panel.getSize().asPercentage > 5) return;
    // expand() alone restores the pre-collapse width, which is zero after a drag shut.
    panel.expand();
    panel.resize(artifactPanelWidthRef.current ?? defaultPanelSizeRef.current);
  }, [browserOpenSequence, showContextPanel]);

  useEffect(() => {
    const panel = artifactPanelRef.current;
    if (!panel) return;

    setIsArtifactSurfaceVisible(false);

    if (!hasInitializedArtifactPanelRef.current) {
      hasInitializedArtifactPanelRef.current = true;
       if (!showContextPanel) {
        panel.resize("0%");
        return;
      }
    }

    setIsArtifactPanelLayoutActive(true);
    setIsArtifactLayoutAnimating(true);
    let resizeFrameId = 0;
    const prepFrameId = window.requestAnimationFrame(() => {
      resizeFrameId = window.requestAnimationFrame(() => {
        panel.resize(
          showContextPanel
            ? (artifactPanelWidthRef.current ?? defaultPanelSizeRef.current)
            : "0%",
        );
      });
    });
    const surfaceTimerId = showContextPanel
      ? window.setTimeout(() => {
          setIsArtifactSurfaceVisible(true);
        }, ARTIFACT_SURFACE_POP_DELAY_MS)
      : 0;
    const timeoutId = window.setTimeout(() => {
      setIsArtifactLayoutAnimating(false);
      if (!showContextPanel) {
        setIsArtifactPanelLayoutActive(false);
      }
    }, ARTIFACT_PANEL_TRANSITION_MS + 60);
    return () => {
      window.cancelAnimationFrame(prepFrameId);
      if (resizeFrameId) {
        window.cancelAnimationFrame(resizeFrameId);
      }
      if (surfaceTimerId) {
        window.clearTimeout(surfaceTimerId);
      }
      window.clearTimeout(timeoutId);
    };
  }, [showContextPanel]);

  useEffect(() => {
    if (!researchMatchesThread) return;
    closeBrowser();
    useChatRuntimeStore.getState().setSettingsPanelOpen(false);
  }, [researchMatchesThread, closeBrowser]);

  // Close the browser on leaving the chat; the runtime thread id survives a new chat's first save.
  const shownThreadId = useAuiState(({ threads }) => threads.mainThreadId);
  const shownThreadIdRef = useRef(shownThreadId);
  useEffect(() => {
    if (shownThreadIdRef.current === shownThreadId) return;
    shownThreadIdRef.current = shownThreadId;
    closeBrowser();
  }, [shownThreadId, closeBrowser]);
  useEffect(() => {
    if (!chatActive) closeBrowser();
  }, [chatActive, closeBrowser]);

  // Width on header/notice only: on the root it would restyle the whole thread per resize.
  const contextSurfaceRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const surface = contextSurfaceRef.current;
    const root = surface?.closest<HTMLElement>("[data-chat-content-root]");
    if (!surface || !root || !showBrowserPanel) return;
    let width = "";
    const insets = () =>
      root.querySelectorAll<HTMLElement>(":scope > [data-side-panel-inset]");
    const apply = () => {
      for (const element of insets()) {
        element.style.setProperty(
          "--studio-side-panel-width",
          chatOnRight ? "0px" : width,
        );
        element.style.setProperty(
          "--studio-side-panel-left",
          chatOnRight ? width : "0px",
        );
      }
    };
    const resizeObserver = new ResizeObserver(() => {
      const next = `${Math.round(surface.getBoundingClientRect().width)}px`;
      if (next === width) return;
      width = next;
      apply();
    });
    resizeObserver.observe(surface);
    const childObserver = new MutationObserver(apply);
    childObserver.observe(root, { childList: true });
    return () => {
      resizeObserver.disconnect();
      childObserver.disconnect();
      for (const element of insets()) {
        element.style.removeProperty("--studio-side-panel-width");
        element.style.removeProperty("--studio-side-panel-left");
      }
    };
  }, [showBrowserPanel, chatOnRight]);

  // Moving the chat re-sorts panels to default sizes; restore the browser's width (full view fills).
  const browserLayout = `${browserFullView}:${chatOnRight}`;
  const seenBrowserLayoutRef = useRef(browserLayout);
  useEffect(() => {
    if (seenBrowserLayoutRef.current === browserLayout) return;
    seenBrowserLayoutRef.current = browserLayout;
    const panel = artifactPanelRef.current;
    if (!panel || !showBrowserPanel) return;
    const frameId = window.requestAnimationFrame(() => {
      panel.resize(
        artifactPanelWidthRef.current ?? defaultPanelSizeRef.current,
      );
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [browserLayout, showBrowserPanel]);

  // With no chat beside it, the chat header would sit over the browser's tabs.
  useEffect(() => {
    const root = contextSurfaceRef.current?.closest<HTMLElement>(
      "[data-chat-content-root]",
    );
    if (!root || !browserFullView) return;
    const insets = root.querySelectorAll<HTMLElement>(
      ":scope > [data-side-panel-inset]",
    );
    for (const element of insets) element.style.visibility = "hidden";
    return () => {
      for (const element of insets) element.style.removeProperty("visibility");
    };
  }, [browserFullView]);

  const fullViewChatTitle = useStoredChatTitle(
    browserFullView ? artifactPanelThread : null,
  );

  // Kept at one place in the tree in every layout, so switching layouts never remounts the thread.
  const threadPane = (
    <div
      className={cn(
        "chat-thread-pane flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden",
        browserFullView && chatDock === "minimized" && "hidden",
      )}
    >
      {/* A floating chat has no welcome screen, so its composer shows in a new chat too. */}
      <Thread
        hideWelcome={Boolean(threadId) || browserFullView}
        targetThreadId={threadId}
      />
    </div>
  );

  return (
    <>
      <ResizablePanelGroup
        orientation="horizontal"
        data-artifact-layout-animating={
          isArtifactLayoutAnimating ? "true" : "false"
        }
        className={cn(
          "chat-artifact-split relative min-h-0 min-w-0 flex-1 basis-0 overflow-hidden",
          chatOnRight &&
            "[&>#chat-artifact]:order-1 [&>[data-slot=resizable-handle]]:order-2 [&>#chat-thread]:order-3",
        )}
        data-browser-full-view={browserFullView ? "true" : undefined}
        // Not :has(), which re-walks the thread per token (thread-ancestor-has-scope.test.ts).
        data-chat-dock={browserFullView ? chatDock : undefined}
      >
        <ResizablePanel
          id="chat-thread"
          defaultSize="100%"
          // Distinct per layout: re-registering re-sorts panels so resizing follows the chat.
          minSize={
            browserFullView
              ? "0%"
              : chatOnRight
                ? "34%"
                : artifactLayoutActive
                  ? "42%"
                  : "100%"
          }
          className="h-full min-h-0 min-w-0 overflow-hidden"
          style={browserFullView ? { overflow: "visible" } : undefined}
        >
          <div
            data-expanded={chatDock === "expanded" ? "true" : "false"}
            className={cn(
              "flex h-full min-h-0 min-w-0 flex-col overflow-hidden",
              browserFullView &&
                (chatDock === "minimized"
                  ? "chat-full-view-dock-minimized"
                  : "chat-full-view-dock group/dock"),
            )}
          >
            <Suspense fallback={null}>
              {browserFullView && chatDock !== "minimized" ? (
                <FullViewChatBar title={fullViewChatTitle} />
              ) : null}
              {browserFullView && chatDock === "minimized" ? (
                <FullViewChatButton />
              ) : null}
            </Suspense>
            {threadPane}
          </div>
        </ResizablePanel>
        <ResizableHandle
          withHandle={false}
          // The library's double-click reset would shut the panel without closing the artifact.
          disableDoubleClick
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            const unpin = pinBrowserPage(event.currentTarget);
            const release = () => {
              window.removeEventListener("pointerup", release);
              window.removeEventListener("pointercancel", release);
              unpin();
              rememberArtifactPanelWidth();
            };
            window.addEventListener("pointerup", release);
            window.addEventListener("pointercancel", release);
          }}
          onKeyUp={rememberArtifactPanelWidth}
          className={cn(
            "relative z-30 w-5 bg-transparent transition-[width,margin] duration-[260ms] ease-[var(--ease-out-cubic)] hover:bg-transparent hover:shadow-none active:bg-transparent active:shadow-none focus-visible:bg-transparent focus-visible:shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:outline-none",
            chatOnRight ? "-ml-4 -mr-1" : "-ml-1 -mr-4",
            (!artifactLayoutActive || browserFullView) &&
              "pointer-events-none -ml-0 -mr-0 w-0",
          )}
        />
        <ResizablePanel
          panelRef={artifactPanelRef}
          id="chat-artifact"
          defaultSize="0%"
          minSize={
            showResearchPanel
              ? "30%"
              : artifactPanelSettledOpen
                ? "30%"
                : "0%"
          }
          maxSize={
            chatOnRight
              ? "66%"
              : showResearchPanel
              ? "58%"
              : artifactLayoutActive
                ? "58%"
                : "0%"
          }
          collapsible={showBrowserPanel}
          collapsedSize="0%"
          className={cn(
            "h-full min-h-0 min-w-0 overflow-visible",
            !showContextPanel && "pointer-events-none",
          )}
        >
          <div
            ref={contextSurfaceRef}
            data-artifact-surface-visible={
              isArtifactSurfaceVisible ? "true" : "false"
            }
            className={cn(
              "chat-artifact-pop-surface flex h-full min-h-0 min-w-0 flex-col overflow-visible",
              (showResearchPanel || showBrowserPanel) &&
                (chatOnRight ? "border-r" : "border-l"),
              (showResearchPanel || showBrowserPanel) && "border-border/70",
            )}
            style={showBrowserPanel ? { transform: "none" } : undefined}
          >
             {showResearchPanel && openResearchRunId ? (
               <ResearchActivityPanel
                 key={openResearchRunId}
                 runId={openResearchRunId}
                 onClose={closeResearchPanel}
               />
             ) : showBrowserPanel ? (
              <Suspense fallback={null}>
                <BrowserPanel active={chatActive} />
              </Suspense>
            ) : null}
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
      {openResearchRunId && researchMatchesThread ? (
        <ResearchActivitySheet
          runId={openResearchRunId}
          open={chatActive && isMobile}
          onOpenChange={(open) => {
            if (!open) closeResearchPanel();
          }}
        />
      ) : null}
    </>
  );
});

type CompareModelSelection = {
  id: string;
  isLora: boolean;
  ggufVariant?: string;
  isDiffusion?: boolean;
  config?: PerModelConfig;
};

function modelMatchesDeleted(
  model: { id: string; ggufVariant?: string | null },
  deletedModel?: DeletedModelRef,
): boolean {
  if (!deletedModel || model.id !== deletedModel.id) return false;
  return (
    deletedModel.ggufVariant == null ||
    (model.ggufVariant ?? null) === deletedModel.ggufVariant
  );
}

/** LoRA checkpoints can use the fast simultaneous adapter-toggle compare path. */
function useIsLoraCompare(): boolean | null {
  return useChatRuntimeStore((s) =>
    checkpointCompareClass({
      checkpoint: s.params.checkpoint,
      isExternal: isExternalModelId(s.params.checkpoint),
      residentUnknown: s.residentCheckpoint === undefined,
      models: s.models,
      loras: s.loras,
      inventorySettled: s.loraInventorySettled,
    }),
  );
}

/** `pending` while the pair is still being read, so neither component hydrates first. */
function useCompareVariant(pairId: string): {
  state: ComparePairReadState;
  retry: () => void;
} {
  const checkpointIsLora = useIsLoraCompare();
  const [read, setRead] = useState<{
    pairId: string;
    state: ComparePairReadState;
  }>();
  const [storageRetry, setStorageRetry] = useState<{
    pairId: string;
    count: number;
  }>();
  const settled = read?.pairId === pairId ? read.state : undefined;
  const retryCount = storageRetry?.pairId === pairId ? storageRetry.count : 0;

  useEffect(() => {
    if (settled) return;
    let isActive = true;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    const settle = (state: ComparePairReadState) => {
      if (!isActive || state.status === "pending") return;
      if (state.status === "retry") {
        retryTimer = setTimeout(() => {
          if (isActive) setStorageRetry({ pairId, count: retryCount + 1 });
        }, 250);
        return;
      }
      setRead({ pairId, state });
    };
    listStoredChatThreads({ pairId })
      .then((threads) =>
        settle(comparePairReadState({ threads }, checkpointIsLora, retryCount)),
      )
      .catch((error) => {
        if (!isExpectedBackgroundChatStorageError(error)) {
          console.error("Could not read a comparison's stored threads", error);
        }
        settle(
          comparePairReadState({ failed: true }, checkpointIsLora, retryCount),
        );
      });
    return () => {
      isActive = false;
      if (retryTimer !== null) clearTimeout(retryTimer);
    };
  }, [pairId, checkpointIsLora, retryCount, settled]);

  const retry = useCallback(() => {
    setRead(undefined);
    setStorageRetry({ pairId, count: 0 });
  }, [pairId]);

  return { state: settled ?? { status: "pending" }, retry };
}

/** The pair read failed. Its persisted shape is unknown, and picking a renderer from the loaded
 *  checkpoint would relabel existing histories, so offer the read again instead. */
function CompareUnreadable({
  onRetry,
}: {
  onRetry: () => void;
}): ReactElement {
  return (
    <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col items-center justify-center gap-3 p-6 text-center">
      <p className="text-sm text-muted-foreground">
        Could not load this comparison's history.
      </p>
      <Button variant="outline" size="sm" onClick={onRetry}>
        Try again
      </Button>
    </div>
  );
}

const CompareContent = memo(function CompareContent({
  pairId,
  projectId,
  models,
  loraModels,
  externalModels,
  externalConnections,
  onFoldersChange,
  onModelsChange,
  deleteDisabled,
  onExitCompare,
}: {
  pairId: string;
  projectId?: string | null;
  models: ModelOption[];
  loraModels: LoraModelOption[];
  externalModels: ExternalModelOption[];
  externalConnections: ExternalConnectionRef[];
  onFoldersChange?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  onExitCompare?: () => void;
}): ReactElement {
  const { state: compareRead, retry: retryCompareRead } =
    useCompareVariant(pairId);

  if (compareRead.status === "unreadable") {
    return <CompareUnreadable onRetry={retryCompareRead} />;
  }
  if (compareRead.status !== "ready") return <></>;

  return compareRead.variant === "lora" ? (
    <LoraCompareContent
      pairId={pairId}
      onExitCompare={onExitCompare}
      projectId={projectId}
    />
  ) : (
    <GeneralCompareContent
      pairId={pairId}
      projectId={projectId}
      models={models}
      loraModels={loraModels}
      externalModels={externalModels}
      externalConnections={externalConnections}
      onFoldersChange={onFoldersChange}
      onModelsChange={onModelsChange}
      deleteDisabled={deleteDisabled}
      onExitCompare={onExitCompare}
    />
  );
});

/** Panes are `flex-1 basis-0 min-h-0 min-w-0` so they share space and the viewport scrolls. */
function ComparePane({
  modelType,
  pairId,
  projectId,
  initialThreadId,
  handleName,
  header,
  borderClassName,
  onInitialHistoryReady,
}: {
  modelType: "base" | "lora" | "model1" | "model2";
  pairId: string;
  projectId?: string | null;
  initialThreadId: string | undefined;
  handleName: string;
  header: ReactElement;
  borderClassName?: string;
  onInitialHistoryReady?: (pane: string) => void;
}): ReactElement {
  const signalInitialHistoryReady = useMemo(
    () =>
      onInitialHistoryReady
        ? () => onInitialHistoryReady(modelType)
        : undefined,
    [modelType, onInitialHistoryReady],
  );
  return (
    <div
      className={cn(
        "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden",
        borderClassName,
      )}
    >
      {header}
      <div className="relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden [&_.aui-thread-viewport]:px-6 lg:[&_.aui-thread-viewport]:px-10">
        <div
          aria-hidden={true}
          className="compare-pane-fade pointer-events-none absolute top-0 left-0 right-[var(--thread-scrollbar-gutter,10px)] z-20 h-6 bg-gradient-to-b from-background to-transparent"
        />
        <ChatRuntimeProvider
          modelType={modelType}
          pairId={pairId}
          projectId={projectId}
          initialThreadId={initialThreadId}
          syncActiveThreadId={false}
          onInitialHistoryReady={signalInitialHistoryReady}
        >
          <RegisterCompareHandle name={handleName} />
          <Thread hideComposer={true} hideWelcome={true} />
        </ChatRuntimeProvider>
      </div>
    </div>
  );
}

function useCompareReloadReadiness(pairId: string): (pane: string) => void {
  const signalReady = useAppShellReadySignal();
  const stateRef = useRef({
    pairId,
    panes: new Set<string>(),
    sent: false,
  });
  if (stateRef.current.pairId !== pairId) {
    stateRef.current = { pairId, panes: new Set<string>(), sent: false };
  }
  return useCallback(
    (pane: string) => {
      const state = stateRef.current;
      if (state.pairId !== pairId || state.sent) {
        return;
      }
      state.panes.add(pane);
      if (state.panes.size < 2) {
        return;
      }
      state.sent = true;
      signalReady();
    },
    [pairId, signalReady],
  );
}

/** Flex, not grid: grid 1fr rows caused autoscroll resize thrash on breakpoint crossings. */
function CompareShell({
  handlesRef,
  children,
  composer,
}: {
  handlesRef: CompareHandles;
  children: ReactElement;
  composer: ReactElement;
}): ReactElement {
  const showModelDisclaimer = useChatPreferencesStore(
    (s) => s.showModelDisclaimer,
  );
  return (
    <CompareHandlesProvider handlesRef={handlesRef}>
      <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col">
        <div
          data-tour="chat-compare-view"
          className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col pt-[var(--studio-content-top-inset,0px)] lg:flex-row"
        >
          {children}
        </div>
        <div className="shrink-0 bg-background pl-5 pr-5 md:px-[calc(30px*var(--ui-space-scale,1))] pb-2 pt-1">
          {/* unsloth-composer-shell: the size container the single composer's narrow layout queries. */}
          <div className="unsloth-composer-shell mx-auto w-full max-w-[var(--custom-chat-max-width,48rem)]">{composer}</div>
          {showModelDisclaimer && (
            <p className="composer-footer-note">
              LLMs can make mistakes. Double-check responses.
            </p>
          )}
        </div>
      </div>
    </CompareHandlesProvider>
  );
}

/** Fast path: same model, adapter on/off, simultaneous generation. */
const LoraCompareContent = memo(function LoraCompareContent({
  pairId,
  onExitCompare,
  projectId,
}: {
  pairId: string;
  onExitCompare?: () => void;
  projectId?: string | null;
}): ReactElement {
  const handlesRef = useRef<Record<string, CompareHandle>>({});
  const [baseThreadId, setBaseThreadId] = useState<string>();
  const [loraThreadId, setLoraThreadId] = useState<string>();
  const [pairLoraModelId, setPairLoraModelId] = useState<string>();
  const [threadsSettled, setThreadsSettled] = useState(false);
  const markInitialHistoryReady = useCompareReloadReadiness(pairId);
  const active = useChatActive();
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const checkpointIsLora = useIsLoraCompare();

  // Global: a first compare run starts before either thread exists. Learning ids mid-run would
  // point ThreadAutoSwitch (via initialThreadId) at a live thread.
  const anyRunning = useChatRuntimeStore(
    (s) => Object.keys(s.localRunByThreadId).length > 0,
  );
  // Only re-lists wait: the shared provider keeps a base chat's run alive into compare, so gating
  // the first list on anyRunning left compare blank.
  const listedPairRef = useRef<string | null>(null);

  useEffect(() => {
    if (anyRunning && listedPairRef.current === pairId) return;
    listedPairRef.current = pairId;
    let isActive = true;
    setThreadsSettled(false);
    listStoredChatThreads({ pairId })
      .then((threads) => {
        if (!isActive) return;
        // No model1/model2 fallback: a generalized pair never routes here, so adopting one mislabels it.
        const baseThread = threads.find((t) => t.modelType === "base");
        const loraThread = threads.find((t) => t.modelType === "lora");
        setBaseThreadId(baseThread?.id);
        setLoraThreadId(loraThread?.id);
        setPairLoraModelId(
          loraThread?.modelId?.trim() || baseThread?.modelId?.trim() || undefined,
        );
      })
      .catch((error) => {
        if (!isExpectedBackgroundChatStorageError(error)) {
          throw error;
        }
      })
      .finally(() => {
        if (isActive) setThreadsSettled(true);
      });
    return () => {
      isActive = false;
    };
  }, [pairId, anyRunning]);

  useEffect(() => {
    if (!threadsSettled) return;
    if (!baseThreadId) markInitialHistoryReady("base");
    if (!loraThreadId) markInitialHistoryReady("lora");
  }, [
    baseThreadId,
    loraThreadId,
    markInitialHistoryReady,
    threadsSettled,
  ]);

  const sendUnavailableReason = !threadsSettled
    ? "Loading comparison history."
    : checkpointIsLora === null
      ? "Checking the loaded model."
      : !checkpointIsLora ||
          (pairLoraModelId !== undefined &&
            !modelIdsMatch(pairLoraModelId, checkpoint))
        ? "Load the LoRA saved with this comparison before sending."
        : undefined;

  return (
    <CompareShell
      handlesRef={handlesRef}
      composer={
        active ? (
          <SharedComposer
            handlesRef={handlesRef}
            onExitCompare={onExitCompare}
            model1ThreadId={baseThreadId}
            model2ThreadId={loraThreadId}
            sendUnavailableReason={sendUnavailableReason}
            requireStableCheckpoint={true}
          />
        ) : (
          <></>
        )
      }
    >
      <>
        <ComparePane
          modelType="base"
          pairId={pairId}
          projectId={projectId}
          initialThreadId={baseThreadId}
          handleName="base"
          onInitialHistoryReady={
            threadsSettled ? markInitialHistoryReady : undefined
          }
          header={
            <div className="shrink-0 px-3 py-1.5">
              <span className="text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
                Base Model
              </span>
            </div>
          }
        />
        <ComparePane
          modelType="lora"
          pairId={pairId}
          projectId={projectId}
          initialThreadId={loraThreadId}
          handleName="lora"
          onInitialHistoryReady={
            threadsSettled ? markInitialHistoryReady : undefined
          }
          borderClassName="border-t border-border/60 lg:border-t-0 lg:border-l"
          header={
            <div className="shrink-0 px-3 py-1.5 text-start lg:text-end lg:pr-[calc(4rem*var(--ui-space-scale,1)+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]">
              <span className="text-ui-10 font-semibold uppercase tracking-wider text-primary">
                Fine-tuned
              </span>
            </div>
          }
        />
      </>
    </CompareShell>
  );
});

function GeneralCompareHeader({
  models,
  loraModels,
  externalModels,
  externalConnections,
  value,
  selectedConfig,
  selectedGgufVariant,
  onValueChange,
  onFoldersChange,
  onModelsChange,
  deleteDisabled,
  side,
  label,
}: {
  models: ModelOption[];
  loraModels: LoraModelOption[];
  externalModels: ExternalModelOption[];
  externalConnections: ExternalConnectionRef[];
  value: string;
  selectedConfig?: PerModelConfig | null;
  selectedGgufVariant?: string | null;
  onValueChange: (
    id: string,
    meta: ModelSelectorChangeMeta,
  ) => void;
  onFoldersChange?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  side: "left" | "right";
  label?: "Base Model" | "Fine-tuned";
}): ReactElement {
  // Controlled so the body-portaled popover cannot linger over another tab off-route.
  const active = useChatActive();
  const [selectorOpen, setSelectorOpen] = useState(false);

  const { pinned } = useSidebar();
  return (
    <div
      className={cn(
        "pointer-events-none relative z-40 flex h-[calc(48px*var(--ui-space-scale,1))] shrink-0 items-start gap-2 bg-background pt-[var(--studio-chat-header-padding-top,11px)]",
        side === "left"
          ? pinned
            ? "pl-12 pr-3 md:pl-2"
            : isTauri
              ? "pl-12 pr-3 md:pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]"
              : "pl-12 pr-3 md:pl-[calc(0.5rem*var(--ui-space-scale,1)+max(0px,var(--studio-mac-traffic-light-inset,0px)-var(--sidebar-width-icon,3rem)))]"
          : "pl-3 pr-[calc(3rem*var(--ui-space-scale,1)+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]",
      )}
    >
      <ModelSelector
        models={models}
        loraModels={loraModels}
        externalModels={externalModels}
        externalConnections={externalConnections}
        value={value}
        selectedConfig={selectedConfig}
        selectedGgufVariant={selectedGgufVariant}
        onValueChange={onValueChange}
        onFoldersChange={onFoldersChange}
        onModelsChange={onModelsChange}
        deleteDisabled={deleteDisabled}
        variant="ghost"
        className="pointer-events-auto max-w-[80%] !h-[var(--studio-chat-control-height,34px)]"
        open={active && selectorOpen}
        onOpenChange={(open) => setSelectorOpen(active && open)}
      />
      {label ? (
        <span
          className={cn(
            "pointer-events-none hidden h-[var(--studio-chat-control-height,34px)] shrink-0 items-center text-ui-10 font-semibold uppercase tracking-wider sm:flex",
            side === "right" && "ml-auto",
            label === "Fine-tuned" ? "text-primary" : "text-muted-foreground",
          )}
        >
          {label}
        </span>
      ) : null}
    </div>
  );
}

function generalCompareLabels(
  loraModels: LoraModelOption[],
  model1: CompareModelSelection,
  model2: CompareModelSelection,
): ["Base Model" | "Fine-tuned" | undefined, "Base Model" | "Fine-tuned" | undefined] {
  const baseOf = (sel: CompareModelSelection) =>
    sel.isLora
      ? loraModels.find((lora) => lora.id === sel.id)?.baseModel
      : undefined;
  const isTunedFrom = (tuned: CompareModelSelection, base: CompareModelSelection) => {
    const loraBase = baseOf(tuned);
    return Boolean(loraBase && base.id) &&
      normalizeModelRef(loraBase) === normalizeModelRef(base.id);
  };
  if (isTunedFrom(model1, model2)) return ["Fine-tuned", "Base Model"];
  if (isTunedFrom(model2, model1)) return ["Base Model", "Fine-tuned"];
  return [undefined, undefined];
}

const GeneralCompareContent = memo(function GeneralCompareContent({
  pairId,
  projectId,
  models,
  loraModels,
  externalModels,
  externalConnections,
  onFoldersChange,
  onModelsChange,
  deleteDisabled,
  onExitCompare,
}: {
  pairId: string;
  projectId?: string | null;
  models: ModelOption[];
  loraModels: LoraModelOption[];
  externalModels: ExternalModelOption[];
  externalConnections: ExternalConnectionRef[];
  onFoldersChange?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  onExitCompare?: () => void;
}): ReactElement {
  const handlesRef = useRef<Record<string, CompareHandle>>({});
  const [model1ThreadId, setModel1ThreadId] = useState<string>();
  const [model2ThreadId, setModel2ThreadId] = useState<string>();
  const [threadsSettled, setThreadsSettled] = useState(false);
  const markInitialHistoryReady = useCompareReloadReadiness(pairId);

  const globalCheckpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const globalGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const globalIsDiffusion = useChatRuntimeStore((s) => s.loadedIsDiffusion);
  const active = useChatActive();
  // Global, with only RE-lists waiting on it; see the note on the Lora variant above.
  const anyRunning = useChatRuntimeStore(
    (s) => Object.keys(s.runningByThreadId).length > 0,
  );
  // A compare send is idle between the two sequential runs; a re-list there swaps the pane threads mid-send.
  const [comparing, setComparing] = useState(false);
  const listedPairRef = useRef<string | null>(null);
  const [model1, setModel1] = useState<CompareModelSelection>({
    id: globalCheckpoint || "",
    isLora: loraModels.some(
      (lora) => lora.id === globalCheckpoint && lora.exportType === "lora",
    ),
    ggufVariant: globalGgufVariant ?? undefined,
    isDiffusion: globalIsDiffusion,
  });
  const [model2, setModel2] = useState<CompareModelSelection>({
    id: "",
    isLora: false,
  });

  const handleModelsChange = useCallback(
    (deletedModel?: DeletedModelRef) => {
      if (modelMatchesDeleted(model1, deletedModel)) {
        setModel1({ id: "", isLora: false });
      }
      if (modelMatchesDeleted(model2, deletedModel)) {
        setModel2({ id: "", isLora: false });
      }
      onModelsChange?.(deletedModel);
    },
    [model1, model2, onModelsChange],
  );

  useEffect(() => {
    if ((anyRunning || comparing) && listedPairRef.current === pairId) return;
    listedPairRef.current = pairId;
    let isActive = true;
    setThreadsSettled(false);
    listStoredChatThreads({ pairId })
      .then((threads) => {
        if (!isActive) return;
        const pair = resolveComparePaneThreadIds(threads);
        setModel1ThreadId(pair.first);
        setModel2ThreadId(pair.second);
      })
      .catch((error) => {
        if (!isExpectedBackgroundChatStorageError(error)) {
          throw error;
        }
      })
      .finally(() => {
        if (isActive) setThreadsSettled(true);
      });
    return () => {
      isActive = false;
    };
  }, [pairId, anyRunning, comparing]);

  useEffect(() => {
    if (!threadsSettled) return;
    if (!model1ThreadId) markInitialHistoryReady("model1");
    if (!model2ThreadId) markInitialHistoryReady("model2");
  }, [
    markInitialHistoryReady,
    model1ThreadId,
    model2ThreadId,
    threadsSettled,
  ]);

  const [model1Label, model2Label] = generalCompareLabels(
    loraModels,
    model1,
    model2,
  );

  return (
    <CompareShell
      handlesRef={handlesRef}
      composer={
        active ? (
          <SharedComposer
            handlesRef={handlesRef}
            model1={model1}
            model2={model2}
            onExitCompare={onExitCompare}
            onComparingChange={setComparing}
            model1ThreadId={model1ThreadId}
            model2ThreadId={model2ThreadId}
            sendUnavailableReason={
              threadsSettled ? undefined : "Loading comparison history."
            }
          />
        ) : (
          <></>
        )
      }
    >
      <>
        <ComparePane
          modelType="model1"
          pairId={pairId}
          projectId={projectId}
          initialThreadId={model1ThreadId}
          handleName="model1"
          onInitialHistoryReady={
            threadsSettled ? markInitialHistoryReady : undefined
          }
          header={
            <GeneralCompareHeader
              side="left"
              label={model1Label}
              models={models}
              loraModels={loraModels}
              externalModels={externalModels}
              externalConnections={externalConnections}
              value={model1.id}
              selectedConfig={model1.config}
              selectedGgufVariant={model1.ggufVariant}
              onValueChange={(id, meta) =>
                setModel1({
                  id,
                  isLora: meta.isLora,
                  ggufVariant: meta.ggufVariant,
                  isDiffusion: meta.isDiffusion,
                  config: meta.config,
                })
              }
              onFoldersChange={onFoldersChange}
              onModelsChange={handleModelsChange}
              deleteDisabled={deleteDisabled}
            />
          }
        />
        <ComparePane
          modelType="model2"
          pairId={pairId}
          projectId={projectId}
          initialThreadId={model2ThreadId}
          handleName="model2"
          onInitialHistoryReady={
            threadsSettled ? markInitialHistoryReady : undefined
          }
          borderClassName="border-t border-sidebar-border lg:border-t-0 lg:border-l"
          header={
            <GeneralCompareHeader
              side="right"
              label={model2Label}
              models={models}
              loraModels={loraModels}
              externalModels={externalModels}
              externalConnections={externalConnections}
              value={model2.id}
              selectedConfig={model2.config}
              selectedGgufVariant={model2.ggufVariant}
              onValueChange={(id, meta) =>
                setModel2({
                  id,
                  isLora: meta.isLora,
                  ggufVariant: meta.ggufVariant,
                  isDiffusion: meta.isDiffusion,
                  config: meta.config,
                })
              }
              onFoldersChange={onFoldersChange}
              onModelsChange={handleModelsChange}
              deleteDisabled={deleteDisabled}
            />
          }
        />
      </>
    </CompareShell>
  );
});

function formatProjectChatDate(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
  }).format(new Date(timestamp));
}

function createThreadNonce(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

type ProjectChatExportFormat =
  | "raw-jsonl"
  | "messages-jsonl"
  | "csv"
  | "sharegpt-jsonl"
  | typeof CONVERSATION_MARKDOWN_FORMAT;
const PROJECT_CHAT_EXPORT_OPTIONS: Array<{
  label: string;
  format: ProjectChatExportFormat;
}> = [
  { label: "Training JSONL", format: "raw-jsonl" },
  { label: "Message JSONL", format: "messages-jsonl" },
  { label: "CSV", format: "csv" },
  { label: "ShareGPT JSONL", format: "sharegpt-jsonl" },
  {
    label: CONVERSATION_MARKDOWN_LABEL,
    format: CONVERSATION_MARKDOWN_FORMAT,
  },
];

async function exportProjectConversation(
  threadId: string,
  format: ProjectChatExportFormat,
): Promise<void> {
  if (format === "raw-jsonl") return exportConversationRawJsonl(threadId);
  if (format === "messages-jsonl")
    return exportConversationMessagesJsonl(threadId);
  if (format === "csv") return exportConversationCsv(threadId);
  if (format === CONVERSATION_MARKDOWN_FORMAT)
    return exportConversationMarkdown(threadId);
  if (format === "sharegpt-jsonl") return exportConversationShareGPT(threadId);
  const unhandled: never = format;
  throw new Error(`Unhandled export format: ${String(unhandled)}`);
}

async function exportProjectChatItem(
  item: SidebarItem,
  format: ProjectChatExportFormat,
): Promise<void> {
  const ids =
    item.type === "single"
      ? [item.id]
      : (await listStoredChatThreads({ pairId: item.id })).map((t) => t.id);
  for (const id of ids) await exportProjectConversation(id, format);
}

async function saveProjectChatItemAsSource(
  item: SidebarItem,
  projectId: string,
): Promise<void> {
  await saveChatItemAsProjectSource(item, projectId);
}

function extractMessageText(content: MessageRecord["content"]): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .map((part) => {
      if (part.type === "text") {
        return part.text;
      }
      if (part.type === "image") {
        return "Image";
      }
      if (part.type === "audio") {
        return "Audio";
      }
      return "";
    })
    .filter(Boolean)
    .join(" ");
}

function ProjectLanding({
  projectId,
  projectName,
  items,
  newThreadNonce,
  rotateNewThreadNonce,
  dataLoaded,
  runtimeReady,
}: {
  projectId: string;
  projectName: string;
  items: SidebarItem[];
  newThreadNonce: string;
  rotateNewThreadNonce: () => void;
  dataLoaded: boolean;
  // The provider is hoisted above the view switch, so its owner reports reload readiness down.
  runtimeReady: boolean;
}): ReactElement {
  const signalReady = useAppShellReadySignal();
  const navigate = useNavigate();
  const active = useChatActive();
  // The browser shows over a project, so its Request edits lands in this composer.
  useStagedFixPrompt(
    useChatArtifactsStore((state) => state.pendingFixPrompt),
    active,
  );
  const wasActiveRef = useRef(active);
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  // Captured in render: an earlier sibling's mount effect blanks activeThreadId first.
  const [initialActiveThreadId] = useState(
    () => useChatRuntimeStore.getState().activeThreadId,
  );
  const [projectTab, setProjectTab] = useState<"chats" | "sources">(() =>
    hasProjectSourcesPending(projectId) ? "sources" : "chats",
  );
  // Drop the marker once committed: React may replay the initializer above.
  useEffect(() => {
    consumeProjectSourcesPending(projectId);
    return noteProjectLandingMounted(projectId);
  }, [projectId]);
  const [pendingNewThreadId, setPendingNewThreadId] = useState<string | null>(
    null,
  );
  const [previews, setPreviews] = useState<
    Record<string, { snippet: string; date: string }>
  >({});
  const reloadReadySent = useRef(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const skipRenameBlurRef = useRef(false);
  // Optimistic title until the debounced sidebar refresh lands, so the old name does not flash.
  const [pendingRename, setPendingRename] = useState<{
    id: string;
    title: string;
  } | null>(null);

  const [editingProject, setEditingProject] = useState(false);
  const [deletingProject, setDeletingProject] = useState(false);
  const createCustomSection = useSidebarOrganizationStore((s) => s.createCustomSection);
  const fileProjectInSection = useFileProjectInSection();
  const [creatingSection, setCreatingSection] = useState(false);

  /** Always asks: a project workspace is bigger to remove than a chat's sandbox. */
  function openProjectDelete(): void {
    setDeleteFilesOnDelete(false);
    setDeletingProject(true);
  }

  async function commitProjectDelete(): Promise<void> {
    const deleteFiles = deleteFilesOnDelete;
    setDeletingProject(false);
    setDeleteFilesOnDelete(false);
    try {
      await deleteChatProject(projectId, { deleteFiles });
      notifyChatHistoryUpdated();
      useChatRuntimeStore.getState().setActiveProjectId(null);
      navigate({ to: "/chat", search: { new: createThreadNonce() } });
    } catch (err) {
      toast.error("Failed to delete project", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  useEffect(() => {
    useChatRuntimeStore.getState().setActiveThreadId(null);
    useChatRuntimeStore.getState().setContextUsage(null);
    setPendingNewThreadId(null);
    rotateNewThreadNonce();
    setRenamingId(null);
    setPendingRename(null);
  }, [projectId, rotateNewThreadNonce]);

  useEffect(() => {
    if (!pendingRename) return;
    const match = items.find((item) => item.id === pendingRename.id);
    if (match && match.title === pendingRename.title) setPendingRename(null);
  }, [items, pendingRename]);

  const openRename = useCallback((item: SidebarItem) => {
    skipRenameBlurRef.current = false;
    setRenameDraft(item.title);
    setRenamingId(item.id);
  }, []);

  const commitRename = useCallback(
    async (item: SidebarItem) => {
      const trimmed = renameDraft.trim();
      setRenamingId(null);
      if (!trimmed || trimmed === item.title) return;
      setPendingRename({ id: item.id, title: trimmed });
      try {
        await renameChatItem(item, trimmed);
      } catch (err) {
        setPendingRename(null);
        toast.error("Failed to rename chat", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
    },
    [renameDraft],
  );

  const { projects } = useChatProjects();
  const currentProject = useMemo(
    () => projects.find((project) => project.id === projectId),
    [projects, projectId],
  );
  const pinnedChatIds = usePinnedChatsStore((s) => s.pinnedIds);
  const togglePinnedChat = usePinnedChatsStore((s) => s.togglePin);
  const confirmDeleteChats = useChatPreferencesStore(
    (s) => s.confirmDeleteChats,
  );
  const alwaysDeleteChatFiles = useChatPreferencesStore(
    (s) => s.alwaysDeleteChatFiles,
  );
  const pinnedChatIdSet = useMemo(
    () => new Set(pinnedChatIds),
    [pinnedChatIds],
  );
  const [confirmingDelete, setConfirmingDelete] = useState<SidebarItem | null>(
    null,
  );
  const [deleteFilesOnDelete, setDeleteFilesOnDelete] = useState(false);

  const noopView = useCallback(() => {}, []);

  const handleArchive = useCallback(
    async (item: SidebarItem) => {
      try {
        await archiveChatItem(item, activeThreadId ?? undefined, noopView);
      } catch (err) {
        toast.error("Failed to archive chat", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
    },
    [activeThreadId, noopView],
  );

  const runDelete = useCallback(
    async (item: SidebarItem, deleteFiles: boolean) => {
      try {
        await deleteChatItem(item, activeThreadId ?? undefined, noopView, {
          deleteFiles,
        });
      } catch (err) {
        toast.error("Failed to delete chat", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
    },
    [activeThreadId, noopView],
  );

  const handleDelete = useCallback(
    (item: SidebarItem) => {
      if (confirmDeleteChats) {
        setDeleteFilesOnDelete(alwaysDeleteChatFiles);
        setConfirmingDelete(item);
        return;
      }
      void runDelete(item, alwaysDeleteChatFiles);
    },
    [confirmDeleteChats, runDelete, alwaysDeleteChatFiles],
  );

  const handleMoveToProject = useCallback(
    async (item: SidebarItem, targetId: string | null) => {
      try {
        await moveChatItemToProject(item, targetId);
      } catch (err) {
        toast.error("Failed to move chat", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
    },
    [],
  );

  const handleExport = useCallback(
    async (item: SidebarItem, format: ProjectChatExportFormat) => {
      try {
        await exportProjectChatItem(item, format);
      } catch (error) {
        if (!isDownloadCancelled(error)) toast.error("Export failed.");
      }
    },
    [],
  );

  const handleSaveAsSource = useCallback(
    async (item: SidebarItem) => {
      try {
        await saveProjectChatItemAsSource(item, projectId);
      } catch {
        toast.error("Failed to save to project sources.");
      }
    },
    [projectId],
  );

  // No composer records under this claim, so passing it refuses adoption.
  const NO_SUCH_CLAIM = -1;

  // Every fresh composer shares one pending key; only the claim tells them apart.
  const pendingTargetClaimRef = useRef<{
    nonce: string;
    claim: number;
  } | null>(null);
  useEffect(() => {
    return useChatRuntimeStore.subscribe((state) => {
      const pending =
        state.projectAttachmentTargetByThread[PENDING_CHAT_ATTACHMENT_KEY];
      if (pending === undefined) return;
      // By claim, not value: re-picking the same destination writes the same string under a new claim.
      const claim = readPendingAttachmentTargetClaim();
      const captured = pendingTargetClaimRef.current;
      if (captured?.nonce === newThreadNonce && captured.claim === claim) {
        return;
      }
      pendingTargetClaimRef.current = { nonce: newThreadNonce, claim };
    });
  }, [newThreadNonce]);

  useEffect(() => {
    const resumed = active && !wasActiveRef.current;
    wasActiveRef.current = active;
    if (!active) {
      return;
    }
    if (!activeThreadId) {
      if (resumed && pendingNewThreadId) {
        // Unless deleted meanwhile: nothing else clears this id, so restoring it shows a tombstoned chat.
        if (!isChatThreadDeleted(pendingNewThreadId)) {
          useChatRuntimeStore.getState().setActiveThreadId(pendingNewThreadId);
          return;
        }
      }
      // Rotate the nonce so the runtime switches to a fresh thread instead of appending to the old one.
      if (pendingNewThreadId) {
        rotateNewThreadNonce();
        setPendingNewThreadId(null);
      }
      return;
    }
    if (
      activeThreadId === initialActiveThreadId ||
      activeThreadId === pendingNewThreadId
    ) {
      return;
    }
    // Hand off the attach choice now: this swap unmounts the bar holding it. Only this composer's
    // own claim, or a later send would consume another composer's pick.
    const captured = pendingTargetClaimRef.current;
    useChatRuntimeStore
      .getState()
      .adoptPendingProjectAttachmentTarget(
        activeThreadId,
        captured?.nonce === newThreadNonce ? captured.claim : NO_SUCH_CLAIM,
      );
    setPendingNewThreadId(activeThreadId);
  }, [
    active,
    activeThreadId,
    initialActiveThreadId,
    pendingNewThreadId,
    newThreadNonce,
    rotateNewThreadNonce,
  ]);

  useEffect(() => {
    let cancelled = false;

    async function loadPreviews(): Promise<void> {
      const entries = await Promise.all(
        items.map(async (item) => {
          if (item.type !== "single") {
            return [
              item.id,
              {
                snippet: "Compare chat",
                date: formatProjectChatDate(item.createdAt),
              },
            ] as const;
          }
          const messages = await listStoredChatMessages(item.id).catch(
            () => [],
          );
          const firstUserMessage =
            messages.find((message) => message.role === "user") ?? messages[0];
          return [
            item.id,
            {
              snippet: firstUserMessage
                ? extractMessageText(firstUserMessage.content) ||
                  attachmentsSample(firstUserMessage.attachments)
                : "",
              date: formatProjectChatDate(item.createdAt),
            },
          ] as const;
        }),
      );
      if (!cancelled) {
        setPreviews(Object.fromEntries(entries));
      }
    }

    void loadPreviews();
    return () => {
      cancelled = true;
    };
  }, [items]);

  useEffect(() => {
    const previewsReady = items.every((item) => previews[item.id] !== undefined);
    if (
      !dataLoaded ||
      !runtimeReady ||
      !previewsReady ||
      reloadReadySent.current
    ) {
      return;
    }
    reloadReadySent.current = true;
    signalReady();
  }, [dataLoaded, items, previews, runtimeReady, signalReady]);

  return (
    <>
      {pendingNewThreadId ? (
        <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden">
          <Thread hideWelcome={true} targetThreadId={pendingNewThreadId} />
        </div>
      ) : (
        <div
          className="flex min-h-0 min-w-0 flex-1 basis-0 overflow-y-auto px-5"
          style={
            {
              ["--thread-max-width" as string]: "48rem",
            } as CSSProperties
          }
        >
          <div className="mx-auto flex w-full max-w-[calc(44rem*var(--ui-space-scale,1))] flex-col pt-[calc(120px*var(--ui-space-scale,1))] pb-14">
            <div className="mb-12 flex items-center gap-4">
              <span className="flex size-13 shrink-0 items-center justify-center rounded-[18px] bg-muted text-foreground/80">
                <HugeiconsIcon
                  icon={Folder02Icon}
                  strokeWidth={1.75}
                  className="size-6.5"
                />
              </span>
              <h1 className="min-w-0 flex-1 truncate font-sans text-ui-30 font-medium leading-tight tracking-normal text-foreground">
                {projectName}
              </h1>
              <NonModalDropdownMenu
                side="bottom"
                align="end"
                sideOffset={6}
                className="unsloth-plus-menu menu-flat-destructive w-52"
                trigger={(triggerRef) => (
                  <button
                    ref={triggerRef}
                    type="button"
                    aria-label="Project options"
                    className="inline-flex size-9 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring data-[state=open]:bg-muted data-[state=open]:text-foreground"
                  >
                    <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
                  </button>
                )}
              >
                <ProjectMenuItems
                  project={{ id: projectId, name: projectName }}
                  chatCount={items.length}
                  onEdit={() => setEditingProject(true)}
                  onDelete={() => openProjectDelete()}
                  onNewSection={() => setCreatingSection(true)}
                  subClassName="unsloth-plus-menu"
                />
              </NonModalDropdownMenu>
            </div>

            <ProjectComposer
              disabled={Boolean(pendingNewThreadId)}
              placeholder={`New chat in ${projectName}`}
            />

            <div className="mt-9 flex items-center gap-2">
              <button
                type="button"
                onClick={() => setProjectTab("chats")}
                data-active={projectTab === "chats"}
                className="h-10 rounded-full px-5 text-ui-14 font-semibold transition-colors data-[active=true]:bg-muted data-[active=true]:text-foreground data-[active=false]:text-muted-foreground data-[active=false]:hover:bg-nav-surface-hover"
              >
                Chats
              </button>
              <button
                type="button"
                onClick={() => setProjectTab("sources")}
                data-active={projectTab === "sources"}
                className="h-10 rounded-full px-5 text-ui-14 font-semibold transition-colors data-[active=true]:bg-muted data-[active=true]:text-foreground data-[active=false]:text-muted-foreground data-[active=false]:hover:bg-nav-surface-hover"
              >
                Sources
              </button>
            </div>

            {projectTab === "sources" ? (
              <Suspense
                fallback={
                  <div className="mt-8 rounded-[26px] bg-muted/30 px-6 py-10 text-center text-sm text-muted-foreground">
                    Loading sources…
                  </div>
                }
              >
                <ProjectSourcesPanel projectId={projectId} />
              </Suspense>
            ) : (
              <div className="mt-8 flex flex-col gap-1">
                {items.map((item) => {
                  const preview = previews[item.id];
                  const displayTitle =
                    pendingRename?.id === item.id
                      ? pendingRename.title
                      : item.title;
                  if (renamingId === item.id) {
                    return (
                      <div
                        key={`${item.type}:${item.id}`}
                        className="flex min-h-[calc(58px*var(--ui-space-scale,1))] w-full items-center rounded-[14px] px-4 py-2"
                      >
                        <div className="min-w-0 flex-1">
                          <input
                            autoFocus
                            value={renameDraft}
                            onChange={(event) =>
                              setRenameDraft(event.target.value)
                            }
                            onKeyDown={(event) => {
                              // Ignore keydowns mid-IME-composition, Escape included.
                              if (
                                event.nativeEvent.isComposing ||
                                event.keyCode === 229
                              )
                                return;
                              if (event.key === "Enter") {
                                event.preventDefault();
                                skipRenameBlurRef.current = true;
                                void commitRename(item);
                              } else if (event.key === "Escape") {
                                event.preventDefault();
                                skipRenameBlurRef.current = true;
                                setRenamingId(null);
                              }
                            }}
                            onBlur={() => {
                              if (skipRenameBlurRef.current) {
                                skipRenameBlurRef.current = false;
                                return;
                              }
                              void commitRename(item);
                            }}
                            onFocus={(event) => event.currentTarget.select()}
                            maxLength={120}
                            aria-label="Rename chat"
                            className="w-full border-0 bg-transparent text-ui-15 leading-5 text-foreground outline-none"
                          />
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div
                      key={`${item.type}:${item.id}`}
                      className="group relative flex min-h-[calc(58px*var(--ui-space-scale,1))] w-full items-center rounded-[14px] transition-colors hover:bg-nav-surface-hover has-[[data-state=open]]:bg-nav-surface-hover"
                    >
                      <button
                        type="button"
                        onClick={() => {
                          navigate({
                            to: "/chat",
                            search:
                              item.type === "single"
                                ? { thread: item.id, project: projectId }
                                : { compare: item.id, project: projectId },
                          });
                        }}
                        className="flex min-h-[calc(58px*var(--ui-space-scale,1))] min-w-0 flex-1 items-center gap-4 rounded-[14px] px-4 py-2 text-left"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-ui-15 leading-5 text-foreground">
                            {displayTitle}
                          </div>
                        </div>
                        <span className="shrink-0 text-ui-14 text-muted-foreground transition-opacity max-md:opacity-0 pointer-coarse:opacity-0 group-hover:opacity-0 group-has-[[data-state=open]]:opacity-0">
                          {preview?.date ??
                            formatProjectChatDate(item.createdAt)}
                        </span>
                      </button>
                      <NonModalDropdownMenu
                        side="bottom"
                        align="end"
                        sideOffset={4}
                        className="unsloth-plus-menu menu-flat-destructive w-56"
                        trigger={(triggerRef) => (
                          <button
                            ref={triggerRef}
                            type="button"
                            onClick={(event) => event.stopPropagation()}
                            aria-label="Chat options"
                            className="absolute right-3 top-1/2 inline-flex size-8 -translate-y-1/2 cursor-pointer items-center justify-center rounded-full text-muted-foreground outline-none transition-opacity hover:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] md:pointer-fine:opacity-0 md:pointer-fine:pointer-events-none focus-visible:opacity-100 focus-visible:pointer-events-auto group-hover:opacity-100 group-hover:pointer-events-auto data-[state=open]:opacity-100 data-[state=open]:pointer-events-auto"
                          >
                            <HugeiconsIcon
                              icon={MoreVerticalIcon}
                              strokeWidth={1.75}
                              className="size-icon"
                            />
                          </button>
                        )}
                      >
                        <DropdownMenuItem onSelect={() => openRename(item)}>
                          <HugeiconsIcon
                            icon={Edit03Icon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                          <span>Rename</span>
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onSelect={() => togglePinnedChat(item.id)}
                        >
                          <HugeiconsIcon
                            icon={
                              pinnedChatIdSet.has(item.id)
                                ? PinOffIcon
                                : PinIcon
                            }
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                          <span>
                            {pinnedChatIdSet.has(item.id)
                              ? "Unpin chat"
                              : "Pin chat"}
                          </span>
                        </DropdownMenuItem>
                        <DropdownMenuSub>
                          <DropdownMenuSubTrigger>
                            <HugeiconsIcon
                              icon={FolderExportIcon}
                              strokeWidth={1.75}
                              className="size-icon"
                            />
                            <span>Project</span>
                          </DropdownMenuSubTrigger>
                          <DropdownMenuSubContent className="unsloth-plus-menu w-52">
                            <DropdownMenuItem
                              disabled={item.projectId !== projectId}
                              onSelect={() =>
                                void handleMoveToProject(item, null)
                              }
                            >
                              <span>Recents</span>
                            </DropdownMenuItem>
                            {projects.map((p) => (
                              <DropdownMenuItem
                                key={p.id}
                                disabled={item.projectId === p.id}
                                onSelect={() =>
                                  void handleMoveToProject(item, p.id)
                                }
                              >
                                <HugeiconsIcon
                                  icon={Folder01Icon}
                                  strokeWidth={1.75}
                                  className="size-icon"
                                />
                                <span className="truncate">{p.name}</span>
                              </DropdownMenuItem>
                            ))}
                          </DropdownMenuSubContent>
                        </DropdownMenuSub>
                        <DropdownMenuSub>
                          <DropdownMenuSubTrigger>
                            <HugeiconsIcon
                              icon={Download01Icon}
                              strokeWidth={1.75}
                              className="size-icon"
                            />
                            <span>Export</span>
                          </DropdownMenuSubTrigger>
                          <DropdownMenuSubContent className="unsloth-plus-menu w-52">
                            {PROJECT_CHAT_EXPORT_OPTIONS.map(
                              ({ label, format }) => (
                                <DropdownMenuItem
                                  key={format}
                                  onSelect={() =>
                                    void handleExport(item, format)
                                  }
                                >
                                  {label}
                                </DropdownMenuItem>
                              ),
                            )}
                          </DropdownMenuSubContent>
                        </DropdownMenuSub>
                        <DropdownMenuItem
                          onSelect={() => void handleSaveAsSource(item)}
                        >
                          <HugeiconsIcon
                            icon={FolderAttachmentIcon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                          <span>Project sources</span>
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onSelect={() => void handleArchive(item)}
                        >
                          <HugeiconsIcon
                            icon={Archive03Icon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                          <span>Archive</span>
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          variant="destructive"
                          onSelect={() => handleDelete(item)}
                        >
                          <HugeiconsIcon
                            icon={Delete02Icon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                          <span>Delete</span>
                        </DropdownMenuItem>
                      </NonModalDropdownMenu>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
      <AlertDialog
        open={active && confirmingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setConfirmingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete chat</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes "{confirmingDelete?.title}". This cannot
              be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <DeleteChatFilesSwitch
            id="chat-landing-delete-files"
            checked={deleteFilesOnDelete}
            onCheckedChange={setDeleteFilesOnDelete}
          />
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                const target = confirmingDelete;
                const deleteFiles = deleteFilesOnDelete;
                setConfirmingDelete(null);
                if (target) void runDelete(target, deleteFiles);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <SectionNameDialog
        open={active && creatingSection}
        mode="create"
        onOpenChange={(open) => !open && setCreatingSection(false)}
        onSubmit={(name) => {
          const sectionId = createCustomSection(name);
          if (sectionId) {
            fileProjectInSection({ id: projectId, name: projectName }, sectionId, normalizeSectionName(name));
          }
        }}
      />
      <EditProjectDialog
        project={active && editingProject ? (currentProject ?? null) : null}
        onOpenChange={(open) => {
          if (!open) setEditingProject(false);
        }}
        onDelete={() => {
          setEditingProject(false);
          openProjectDelete();
        }}
      />
      <AlertDialog
        open={active && deletingProject}
        onOpenChange={(open) => {
          if (!open) setDeletingProject(false);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete project</AlertDialogTitle>
            <AlertDialogDescription>
              Delete "{projectName}"? Its chats will be permanently deleted.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <DeleteChatFilesSwitch
            id="chat-landing-delete-project-files"
            checked={deleteFilesOnDelete}
            onCheckedChange={setDeleteFilesOnDelete}
            description={
              currentProject?.rootPath ??
              "The project workspace folder will be removed from disk."
            }
          />
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => void commitProjectDelete()}>
              {deleteFilesOnDelete ? "Delete all" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

export type ChatSearch = {
  thread?: string;
  compare?: string;
  new?: string;
  project?: string;
};

export function validateChatSearch(search: Record<string, unknown>): ChatSearch {
  return {
    thread: typeof search.thread === "string" ? search.thread : undefined,
    compare: typeof search.compare === "string" ? search.compare : undefined,
    new: typeof search.new === "string" ? search.new : undefined,
    project: typeof search.project === "string" ? search.project : undefined,
  };
}

type PendingHubAutoLoad = {
  selection: SelectedModelInput;
  contextKey: string;
  originCheckpoint: string;
  originGgufVariant: string | null;
};

// `search` comes from RootLayout so ChatPage stays mounted off-route; `active` is false there.
export function ChatPage({
  search,
  active,
}: { search: ChatSearch; active: boolean }): ReactElement {
  const showContextWindowUsage = useChatPreferencesStore(
    (s) => s.showContextWindowUsage,
  );
  const navigate = useNavigate();

  const settingsOpen = useChatRuntimeStore((s) => s.settingsPanelOpen);
  const setSettingsOpen = useChatRuntimeStore((s) => s.setSettingsPanelOpen);
  const incognito = useChatRuntimeStore((s) => s.incognito);
  const setIncognito = useChatRuntimeStore((s) => s.setIncognito);
  const toggleIncognito = useCallback(() => {
    const store = useChatRuntimeStore.getState();
    const wasIncognito = store.incognito;
    store.setIncognito(!store.incognito);
    // On an empty scratch chat flip in place: navigating would remount the thread. Otherwise start a
    // clean chat so the temporary session cannot inherit or leave a persisted thread.
    const onEmptyScratchChat =
      !search.thread &&
      !search.compare &&
      !search.project &&
      store.activeThreadId == null;
    if (wasIncognito) {
      requestTemporaryPromptQueueStop();
    }
    if (onEmptyScratchChat) return;
    store.setActiveThreadId(null);
    store.setActiveProjectId(null);
    navigate({ to: "/chat", search: { new: crypto.randomUUID() } });
  }, [navigate, search]);
  const hydratePersistedSettings = useChatRuntimeStore(
    (s) => s.hydratePersistedSettings,
  );
  const settingsHydrated = useChatRuntimeStore((s) => s.settingsHydrated);
  const externalProviders = useExternalProvidersStore((s) => s.providers);
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const setExternalProviders = useExternalProvidersStore((s) => s.setProviders);
  const externalProvidersForChat = connectionsEnabled ? externalProviders : [];

  useEffect(() => {
    void hydratePersistedSettings();
  }, [hydratePersistedSettings]);

  useEffect(() => {
    // Off-route ChatPage stays mounted; toast+navigate would yank the user back to chat.
    if (!active) return;
    const threadId = search.thread;
    if (!threadId) return;

    let canceled = false;
    void getStoredChatThread(threadId)
      .then((thread) => {
        if (canceled || thread) return;
        useChatRuntimeStore.getState().setActiveThreadId(null);
        toast.info("Chat not found", {
          description: "That thread no longer exists, so we opened a new chat.",
        });
        navigate({
          to: "/chat",
          search: search.project
            ? { project: search.project }
            : { new: crypto.randomUUID() },
          replace: true,
        });
      })
      .catch(() => {
        if (useChatRuntimeStore.getState().activeThreadId === threadId) {
          useChatRuntimeStore.getState().setActiveThreadId(null);
        }
      });

    return () => {
      canceled = true;
    };
  }, [active, navigate, search.thread]);

  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [projectPickerOpen, setProjectPickerOpen] = useState(false);
  const [modelSelectorLocked, setModelSelectorLocked] = useState(false);
  const modelConfigRequest = useModelConfigHandoffStore((state) =>
    modelConfigHandoffForDestination(state.request, {
      active,
      newChatId: search.new,
      threadId: search.thread,
      compareId: search.compare,
      projectId: search.project,
    }),
  );
  const handleModelConfigRequestAdopted = useCallback(
    (requestId: string) => {
      setSettingsOpen(false);
      setModelSelectorLocked(false);
      setModelSelectorOpen(true);
      clearModelConfigHandoff(requestId);
    },
    [setSettingsOpen],
  );
  const viewBeforeCompareRef = useRef<ChatSearch | null>(null);
  // Fallback for compare opened from a path that does not set viewBeforeCompareRef.
  const lastNonCompareViewRef = useRef<ChatSearch | null>(null);
  useEffect(() => {
    if (!search.compare) {
      lastNonCompareViewRef.current = { ...search };
    }
  }, [search]);
  const inferenceParams = useChatRuntimeStore((state) => state.params);
  const setInferenceParams = useChatRuntimeStore((state) => state.setParams);
  const activeGgufVariant = useChatRuntimeStore(
    (state) => state.activeGgufVariant,
  );
  const residentCheckpoint = useChatRuntimeStore(
    (state) => state.residentCheckpoint,
  );
  const loadedCount = useChatRuntimeStore((state) => state.loadedModels.length);
  const loadedContextLength = useChatRuntimeStore(
    (state) => state.loadedContextLength,
  );
  const nativeContextLength = useChatRuntimeStore(
    (state) => state.nativeContextLength,
  );
  const contextUsage = useChatRuntimeStore((state) => state.contextUsage);
  const loadedIsGguf = useChatRuntimeStore((state) => state.loadedIsGguf);
  const loadedContextUnboundedWhenBatched = useChatRuntimeStore(
    (state) => state.loadedContextUnboundedWhenBatched,
  );
  const loadedParallelSlots = useChatRuntimeStore(
    (state) => state.loadedParallelSlots,
  );
  const loadedContextBudget = useChatRuntimeStore(
    (state) => state.loadedContextBudget,
  );
  const loadedContextEnforced = useChatRuntimeStore(
    (state) => state.loadedContextEnforced,
  );
  const platformDeviceType = usePlatformStore((state) => state.deviceType);
  const platformChatOnlyReason = usePlatformStore(
    (state) => state.chatOnlyReason,
  );
  const modelsFromStore = useChatRuntimeStore((state) => state.models);
  const lorasFromStore = useChatRuntimeStore((state) => state.loras);
  const modelsError = useChatRuntimeStore((state) => state.modelsError);
  const modelLoading = useChatRuntimeStore((state) => state.modelLoading);
  const clearCheckpoint = useChatRuntimeStore((state) => state.clearCheckpoint);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const latestResearchRunId = useResearchRunStore((state) =>
    activeThreadId ? state.latestRunByThreadId[activeThreadId] : undefined,
  );
  // Status, not the run: a run selector here re-rendered the page on every research delta.
  const latestResearchRunStatus = useResearchRunStore((state) =>
    latestResearchRunId
      ? state.sessions[latestResearchRunId]?.run.status
      : undefined,
  );
  const openResearchPanel = useResearchRunStore((state) => state.openPanel);
  const openResearchRunId = useResearchRunStore((state) => state.openRunId);
  const closeResearchPanel = useResearchRunStore((state) => state.closePanel);
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(
    search.project ?? null,
  );
  const { projects, isLoading: projectsLoading } = useChatProjects();
  const currentProject = currentProjectId
    ? (projects.find((project) => project.id === currentProjectId) ?? null)
    : null;
  const { items: currentProjectItems, loaded: currentProjectItemsLoaded } =
    useChatSidebarItems({
    projectId: currentProjectId ?? "__no_project_selected__",
    });
  const currentChatTitle = activeThreadId
    ? currentProjectItems.find((item) => item.id === activeThreadId)?.title
    : undefined;
  const openProjectLanding = useCallback(
    (projectId: string) => {
      useChatRuntimeStore.getState().setActiveThreadId(null);
      useChatRuntimeStore.getState().setActiveProjectId(projectId);
      navigate({ to: "/chat", search: { project: projectId } });
    },
    [navigate],
  );

  const handleDesktopNewChat = useCallback(() => {
    clearNewChatDraft();
    const runtime = useChatRuntimeStore.getState();
    runtime.setActiveThreadId(null);
    runtime.setActiveProjectId(currentProjectId);
    runtime.setIncognito(false);
    navigate({
      to: "/chat",
      search: currentProjectId
        ? { project: currentProjectId }
        : { new: crypto.randomUUID() },
    });
  }, [currentProjectId, navigate]);
  const openProjectsList = useCallback(() => {
    navigate({ to: "/projects" });
  }, [navigate]);
  const persistedActiveThreadId = isAssistantLocalThreadId(activeThreadId)
    ? null
    : activeThreadId;
  // ?new=<nonce> lacks a URL thread; wait for ThreadNewChatSwitch to clear stale activeThreadId.
  const newChatBlankedRef = useRef<string | null>(null);
  if (
    search.new &&
    (activeThreadId === null || isAssistantLocalThreadId(activeThreadId))
  ) {
    newChatBlankedRef.current = search.new;
  }
  const newChatThreadId =
    search.new && newChatBlankedRef.current === search.new
      ? persistedActiveThreadId
      : null;
  // leaving Chat clears activeThreadId and restores it later, so the shown thread id is latched.
  const newChatIdentityBlankedRef = useRef<string | null>(null);
  if (search.new && activeThreadId === null) {
    newChatIdentityBlankedRef.current = search.new;
  }
  // re-latch after every blank: a nonce whose chat was deleted while hidden comes back on a fresh thread.
  const newChatRef = useRef<{ nonce: string; threadId: string } | null>(null);
  if (search.new && activeThreadId && newChatIdentityBlankedRef.current === search.new) {
    newChatRef.current = { nonce: search.new, threadId: activeThreadId };
    newChatIdentityBlankedRef.current = null;
  }
  const modelOperationInProgress = useChatRuntimeStore(
    (state) => state.modelLoading,
  );
  const {
    refresh,
    cancelLoadingForReplacement,
    invalidatePendingModelSelection,
    discardExternalReplacement,
    restoreConfigForExternalReplacement,

    isModelSelectionIntentCurrent,
    selectModel,
    loadNpuModel,
    ejectModel,
    ejectAllModels,
    cancelLoading,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  } = useChatModelRuntime();
  const prevConnectionsEnabledRef = useRef(connectionsEnabled);
  useEffect(() => {
    const turnedOff = prevConnectionsEnabledRef.current && !connectionsEnabled;
    if (!connectionsEnabled && isExternalModelId(inferenceParams.checkpoint)) {
      clearCheckpoint();
      if (turnedOff) {
        toast.info("Connections disabled", {
          description: "Switched away from the hosted model.",
        });
      }
    }
    prevConnectionsEnabledRef.current = connectionsEnabled;
  }, [
    clearCheckpoint,
    connectionsEnabled,
    inferenceParams.checkpoint,
  ]);
  const pendingNativeModelIntent = useNativeIntentStore(
    (state) => state.pendingModelIntent,
  );
  const nativePathLeasesSupported = useNativePathLeasesSupported();
  const refreshRef = useRef(refresh);
  const selectModelRef = useRef(selectModel);

  useEffect(() => {
    refreshRef.current = refresh;
    selectModelRef.current = selectModel;
  }, [refresh, selectModel]);
  const rememberedConfigFor = useCallback(
    (selection: {
      id: string;
      ggufVariant?: string | null;
      source?: string;
    }) => {
      if (selection.source === "external") return null;
      const resolved = resolveResidentInitialConfig(
        selection.id,
        selection.ggufVariant,
      );
      return resolved.remembered ? resolved.config : null;
    },
    [],
  );
  const isExternalModel = useMemo(
    () => isExternalModelId(inferenceParams.checkpoint),
    [inferenceParams.checkpoint],
  );
  const contextWindowKnown = hasKnownContextWindow({
    loadedContextLength,
    modelLoading,
    isExternalModel,
    residentCheckpoint,
  });
  const {
    checkpoint: runtimeCheckpoint,
    isGguf: runtimeModelIsGguf,
    config: activeModelConfig,
  } = useActiveModelConfig();
  const activeModelIsGguf =
    runtimeCheckpoint != null && !isExternalModel && runtimeModelIsGguf;
  const activeModelIsDiffusion = useChatRuntimeStore(
    (s) => s.loadedIsDiffusion,
  );
  const activeModelIsLora = useMemo(() => {
    const checkpoint = inferenceParams.checkpoint;
    if (!checkpoint || isExternalModel) return false;
    const model = modelsFromStore.find((entry) => entry.id === checkpoint);
    if (model) return model.isLora;
    const lora = lorasFromStore.find((entry) => entry.id === checkpoint);
    return lora?.exportType === "lora";
  }, [inferenceParams.checkpoint, isExternalModel, modelsFromStore, lorasFromStore]);
  const reasoningEnabled = useChatRuntimeStore((s) => s.reasoningEnabled);
  const reasoningStyle = useChatRuntimeStore((s) => s.reasoningStyle);
  const reasoningEffort = useChatRuntimeStore((s) => s.reasoningEffort);
  const supportsReasoningOff = useChatRuntimeStore(
    (s) => s.supportsReasoningOff,
  );
  const activeExternalProvider = useMemo(() => {
    const selection = parseExternalModelId(inferenceParams.checkpoint);
    if (!selection) return null;
    return (
      externalProvidersForChat.find((p) => p.id === selection.providerId) ??
      null
    );
  }, [externalProvidersForChat, inferenceParams.checkpoint]);
  const activeExternalProviderType =
    activeExternalProvider?.providerType ?? null;
  const activeProviderCapabilities = useMemo(() => {
    const selection = parseExternalModelId(inferenceParams.checkpoint);
    if (!selection) return null;
    const provider = externalProvidersForChat.find(
      (p) => p.id === selection.providerId,
    );
    const baseCapabilities = getProviderCapabilities(
      provider?.providerType,
      provider?.apiType,
      selection.modelId,
      provider?.baseUrl,
    );
    if (!baseCapabilities) return baseCapabilities;
    const anthropicThinkingEnabled =
      provider?.providerType === "anthropic" &&
      reasoningStyle === "reasoning_effort" &&
      (supportsReasoningOff ? reasoningEnabled : true) &&
      reasoningEffort !== "none";
    if (!anthropicThinkingEnabled) return baseCapabilities;
    return {
      ...baseCapabilities,
      temperature: false,
      topK: false,
    };
  }, [
    externalProvidersForChat,
    inferenceParams.checkpoint,
    reasoningEnabled,
    reasoningStyle,
    reasoningEffort,
    supportsReasoningOff,
  ]);
  useEffect(() => {
    const selection = parseExternalModelId(inferenceParams.checkpoint);
    if (!selection) return;
    const provider = externalProvidersForChat.find(
      (p) => p.id === selection.providerId,
    );
    const reasoningCaps = getExternalReasoningCapabilities(
      provider?.providerType,
      selection.modelId,
      {
        isReasoningProvider: provider?.isReasoningModel === true,
        baseUrl: provider?.baseUrl ?? null,
        apiType: provider?.apiType,
        reasoningConfig: provider?.reasoningConfig,
      },
    );
    const state = useChatRuntimeStore.getState();
    const effortLevels = reasoningCaps.reasoningEffortLevels;
    // Resolve with the pin: without it a resync put the provider default over the pinned level.
    const pinnedEffort = externalReasoningTakesEffort(reasoningCaps)
      ? pinnedReasoningEffort(inferenceParams.checkpoint, effortLevels)
      : null;
    const nextReasoningEffort = resolveExternalReasoningEffort({
      caps: reasoningCaps,
      providerType: provider?.providerType,
      apiType: provider?.apiType,
      current:
        !pinnedEffort && pinHoldsLiveEffort()
          ? (takeEffortDisplacedByPin() ?? state.reasoningEffort)
          : state.reasoningEffort,
      pinned: pinnedEffort,
    });
    // Remember the chat's own level so clearing the pin can restore it.
    if (pinnedEffort && nextReasoningEffort !== state.reasoningEffort) {
      noteEffortDisplacedByPin(state.reasoningEffort);
    }
    const supportsBuiltinWebSearch = providerSupportsBuiltinWebSearch(
      provider?.providerType,
      selection.modelId,
      provider?.baseUrl,
    );
    const supportsBuiltinCodeExecution = providerSupportsBuiltinCodeExecution(
      provider?.providerType,
      selection.modelId,
      provider?.baseUrl,
      provider?.apiType,
    );
    const supportsBuiltinImageGeneration =
      providerSupportsBuiltinImageGeneration(
        provider?.providerType,
        selection.modelId,
        provider?.baseUrl,
        provider?.apiType,
      );
    const supportsBuiltinWebFetch = providerSupportsBuiltinWebFetch(
      provider?.providerType,
    );
    // Kimi k2.x defaults to thinking enabled server-side, so Think starts on.
    const isKimi = provider?.providerType === "kimi";
    // Search defaults on only for Anthropic and OpenAI (structured citations).
    const searchOnByDefault =
      supportsBuiltinWebSearch &&
      (provider?.providerType === "anthropic" ||
        provider?.providerType === "openai");
    // the open chat's own pills win, or selecting a model would revert them to the global ones.
    const storedToolsEnabled =
      threadScopedOverride("toolsEnabled") ??
      loadOptionalBool(CHAT_TOOLS_ENABLED_KEY);
    const storedCodeToolsEnabled =
      threadScopedOverride("codeToolsEnabled") ??
      loadOptionalBool(CHAT_CODE_TOOLS_ENABLED_KEY);
    const storedImageToolsEnabled =
      threadScopedOverride("imageToolsEnabled") ??
      loadOptionalBool(CHAT_IMAGE_TOOLS_ENABLED_KEY);
    const storedWebFetchToolsEnabled =
      threadScopedOverride("webFetchToolsEnabled") ??
      loadOptionalBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY);
    // Unsloth runs Search and Code itself for any capable provider, so a self-hosted connection has no
    // hosted flag to key off; keying on those alone dropped the saved preference.
    const supportsStudioToolsHere =
      providerModelSupportsStudioTools(
        provider?.providerType,
        selection.modelId,
      ) === true;
    const canSearch = supportsBuiltinWebSearch || supportsStudioToolsHere;
    const canRunCode = codeToolCanRun({
      hostedCodeExecutionForThisTurn: supportsBuiltinCodeExecution,
      providerHostsCodeExecution: providerHostsCodeExecution(
        provider?.providerType,
        provider?.baseUrl,
        provider?.apiType,
      ),
      supportsStudioTools: supportsStudioToolsHere,
    });
    const nextToolsEnabled = canSearch
      ? isKimi
        ? false
        : (storedToolsEnabled ?? searchOnByDefault)
      : false;
    useChatRuntimeStore.setState({
      supportsReasoning: reasoningCaps.supportsReasoning,
      reasoningAlwaysOn: reasoningCaps.reasoningAlwaysOn,
      reasoningStyle: reasoningCaps.reasoningStyle,
      supportsReasoningOff: reasoningCaps.supportsReasoningOff,
      reasoningEffortLevels: effortLevels,
      reasoningEffort: nextReasoningEffort,
      reasoningEnabled: reasoningCaps.supportsReasoning
        ? reasoningCaps.supportsReasoningOff
          ? isKimi
            ? true
            : state.reasoningEnabled
          : true
        : state.reasoningEnabled,
      supportsPreserveThinking: providerSupportsPreserveThinking(provider?.providerType),
      preserveThinking: resolvePreserveThinkingOnLoad({
        supports_preserve_thinking: providerSupportsPreserveThinking(provider?.providerType),
      }),
      supportsTools: supportsStudioToolsHere,
      supportsBuiltinWebSearch,
      supportsBuiltinCodeExecution,
      supportsBuiltinImageGeneration,
      supportsBuiltinWebFetch,
      toolsEnabled: nextToolsEnabled,
      codeToolsEnabled: canRunCode ? (storedCodeToolsEnabled ?? false) : false,
      imageToolsEnabled: supportsBuiltinImageGeneration
        ? (storedImageToolsEnabled ?? false)
        : false,
      // Fetch defaults off: Anthropic bills per fetch.
      webFetchToolsEnabled: supportsBuiltinWebFetch
        ? (storedWebFetchToolsEnabled ?? false)
        : false,
    });
    // Rerun on hydration: this clamps the stored pills, so it must apply after them.
  }, [externalProvidersForChat, inferenceParams.checkpoint, settingsHydrated]);
  // Another tab can change the pin, and the effect above reads it without subscribing.
  const activePinnedEffort = useModelReasoningEffortStore(
    (state) => state.effortByModel[inferenceParams.checkpoint],
  );
  const appliedPinnedEffort = useRef(activePinnedEffort);
  useEffect(() => {
    if (appliedPinnedEffort.current === activePinnedEffort) return;
    appliedPinnedEffort.current = activePinnedEffort;
    const selection = parseExternalModelId(inferenceParams.checkpoint);
    if (!selection) return;
    const provider = externalProvidersForChat.find(
      (p) => p.id === selection.providerId,
    );
    const caps = getExternalReasoningCapabilities(
      provider?.providerType,
      selection.modelId,
      {
        isReasoningProvider: provider?.isReasoningModel === true,
        baseUrl: provider?.baseUrl ?? null,
        apiType: provider?.apiType,
        reasoningConfig: provider?.reasoningConfig,
      },
    );
    reconcilePinnedReasoningEffort({
      checkpoint: inferenceParams.checkpoint,
      caps,
      providerType: provider?.providerType,
      apiType: provider?.apiType,
    });
  }, [activePinnedEffort, externalProvidersForChat, inferenceParams.checkpoint]);
  // A late catalog refreshes only the stored reasoning fields, never the selection defaults.
  const modelCatalogChange = useSyncExternalStore(
    subscribeModelCatalog,
    modelCatalogVersion,
  );
  const appliedCatalogChange = useRef(modelCatalogChange);
  useEffect(() => {
    if (appliedCatalogChange.current === modelCatalogChange) return;
    appliedCatalogChange.current = modelCatalogChange;
    const selection = parseExternalModelId(inferenceParams.checkpoint);
    if (!selection) return;
    const { providers, connectionsEnabled: enabled } = useExternalProvidersStore.getState();
    const provider = enabled
      ? providers.find((p) => p.id === selection.providerId)
      : undefined;
    const caps = getExternalReasoningCapabilities(
      provider?.providerType,
      selection.modelId,
      {
        isReasoningProvider: provider?.isReasoningModel === true,
        baseUrl: provider?.baseUrl ?? null,
        apiType: provider?.apiType,
        reasoningConfig: provider?.reasoningConfig,
      },
    );
    useChatRuntimeStore.setState(
      reasoningFieldsAfterCatalogRefresh(useChatRuntimeStore.getState(), caps),
    );
    // After the levels: a pin applies only while the catalog calls it legal.
    reconcilePinnedReasoningEffort({
      checkpoint: inferenceParams.checkpoint,
      caps,
      providerType: provider?.providerType,
      apiType: provider?.apiType,
    });
  }, [modelCatalogChange, inferenceParams.checkpoint]);
  const canCompare = useMemo(() => {
    return Boolean(inferenceParams.checkpoint) && !isExternalModel;
  }, [inferenceParams.checkpoint, isExternalModel]);

  useEffect(() => {
    let canceled = false;

    async function resolveProjectId(): Promise<void> {
      if (search.project) {
        setCurrentProjectId(search.project);
        useChatRuntimeStore.getState().setActiveProjectId(search.project);
        return;
      }

      if (search.thread) {
        const thread = await getStoredChatThread(search.thread).catch(
          () => null,
        );
        if (!canceled) {
          const projectId = thread?.projectId ?? null;
          setCurrentProjectId(projectId);
          useChatRuntimeStore.getState().setActiveProjectId(projectId);
        }
        return;
      }

      if (search.compare) {
        const threads = await listStoredChatThreads({
          pairId: search.compare,
          includeArchived: true,
        }).catch(() => []);
        if (!canceled) {
          const projectId = threads[0]?.projectId ?? null;
          setCurrentProjectId(projectId);
          useChatRuntimeStore.getState().setActiveProjectId(projectId);
        }
        return;
      }

      setCurrentProjectId(null);
      useChatRuntimeStore.getState().setActiveProjectId(null);
    }

    void resolveProjectId();
    return () => {
      canceled = true;
    };
  }, [search.compare, search.project, search.thread]);

  const view = useMemo<ChatView>(() => {
    if (search.compare) {
      return {
        mode: "compare",
        pairId: search.compare,
        projectId: currentProjectId,
      };
    }
    if (search.thread) {
      return {
        mode: "single",
        threadId: search.thread,
        projectId: currentProjectId,
      };
    }
    if (search.new) {
      return {
        mode: "single",
        newThreadNonce: search.new,
        projectId: currentProjectId,
      };
    }
    if (search.project) {
      return {
        mode: "project",
        projectId: search.project,
      };
    }
    if (persistedActiveThreadId) {
      return {
        mode: "single",
        threadId: persistedActiveThreadId,
        projectId: currentProjectId,
      };
    }
    return { mode: "single", projectId: currentProjectId };
  }, [
    search.thread,
    search.compare,
    search.new,
    search.project,
    persistedActiveThreadId,
    currentProjectId,
  ]);

  const [projectNewThreadNonce, setProjectNewThreadNonce] = useState(() =>
    createThreadNonce(),
  );
  const rotateProjectNewThreadNonce = useCallback(() => {
    setProjectNewThreadNonce(createThreadNonce());
  }, []);

  useEffect(() => {
    const onFreshSingleChat = view.mode === "single" && !view.threadId;
    if (incognito && !onFreshSingleChat) {
      setIncognito(false);
    }
  }, [view, incognito, setIncognito]);

  const browserOpen = useBrowserStore((state) => state.open);
  const browserOpenSequence = useBrowserStore((state) => state.openSequence);
  // A project has no side pane: only a browser opened from it (a card, a link) shows over it, not
  // one left open in a chat before.
  const [projectBrowserBaseline, setProjectBrowserBaseline] = useState<number | null>(null);
  const projectViewId = view.mode === "project" ? view.projectId : null;
  useEffect(() => {
    setProjectBrowserBaseline(projectViewId ? useBrowserStore.getState().openSequence : null);
  }, [projectViewId]);
  const artifactViewKey =
    view.mode === "single"
      ? `single:${view.threadId ?? view.newThreadNonce ?? "new"}`
      : view.mode === "compare"
        ? `compare:${view.pairId}`
        : `project:${view.projectId}`;

  const attachmentScope =
    view.mode === "single" && !search.thread && !search.new && !search.project
      ? "single:implicit"
      : artifactViewKey;

  // Kept mounted behind compare: unmounting detaches the runtime and the backend cancels the run.
  const keptBaseViewRef = useRef<{
    view: Exclude<ChatView, { mode: "compare" }>;
    attachmentTargetKey: string;
  } | null>(null);
  if (view.mode !== "compare") {
    keptBaseViewRef.current = { view, attachmentTargetKey: artifactViewKey };
  }
  const baseView = keptBaseViewRef.current?.view ?? null;
  const baseAttachmentTargetKey =
    keptBaseViewRef.current?.attachmentTargetKey ?? artifactViewKey;
  const baseBackgrounded = view.mode === "compare";

  // Keyed by project, not a boolean: the hoisted provider is not remounted on project change.
  const projectLandingId =
    baseView?.mode === "project" ? baseView.projectId : null;
  const [projectRuntimeReadyFor, setProjectRuntimeReadyFor] = useState<
    string | null
  >(null);
  const markProjectRuntimeReady = useCallback(() => {
    setProjectRuntimeReadyFor(projectLandingId);
  }, [projectLandingId]);
  const projectRuntimeReady =
    projectLandingId !== null && projectRuntimeReadyFor === projectLandingId;

  // biome-ignore lint/correctness/useExhaustiveDependencies: a new view is the reset
  useEffect(() => {
    clearAutoOpenedArtifacts();
  }, [artifactViewKey]);

  const newChat = newChatRef.current;
  const newChatShownId =
    newChat && view.mode === "single" && view.newThreadNonce === newChat.nonce ? newChat.threadId : null;
  const projectChatBlankedRef = useRef<{ projectId: string; nonce: string } | null>(null);
  if (view.mode === "project" && activeThreadId === null) {
    projectChatBlankedRef.current = { projectId: view.projectId, nonce: projectNewThreadNonce };
  }
  const projectChatRef = useRef<{ projectId: string; nonce: string; threadId: string } | null>(null);
  const projectChatBlanked = projectChatBlankedRef.current;
  if (
    view.mode === "project" &&
    activeThreadId &&
    projectChatBlanked?.projectId === view.projectId &&
    projectChatBlanked.nonce === projectNewThreadNonce &&
    (projectChatRef.current?.projectId !== view.projectId ||
      projectChatRef.current.nonce !== projectNewThreadNonce)
  ) {
    projectChatRef.current = {
      projectId: view.projectId,
      nonce: projectNewThreadNonce,
      threadId: activeThreadId,
    };
  }
  const projectChat = projectChatRef.current;
  const projectChatShownId =
    projectChat &&
    view.mode === "project" &&
    projectChat.projectId === view.projectId &&
    projectChat.nonce === projectNewThreadNonce
      ? projectChat.threadId
      : null;
  const shownChatKey =
    view.mode === "single"
      ? `single:${view.threadId ?? newChatShownId ?? activeThreadId ?? view.newThreadNonce ?? "new"}`
      : view.mode === "project"
        ? projectChatShownId
          ? `single:${projectChatShownId}`
          : `project:${view.projectId}:${projectNewThreadNonce}`
        : artifactViewKey;
  // biome-ignore lint/correctness/useExhaustiveDependencies: another chat on screen is the reset
  useLayoutEffect(() => {
    useBrowserStore.getState().closeChatPages();
  }, [shownChatKey]);

  const hasActiveModel = Boolean(inferenceParams.checkpoint);
  const chatContextKey = `${view.mode}|${activeThreadId ?? ""}|${search.new ?? ""}|${search.project ?? ""}`;
  const [pendingHubAutoLoad, setPendingHubAutoLoad] =
    useState<PendingHubAutoLoad | null>(null);
  const stageOrLoad = useCallback(
    async (selection: SelectedModelInput) => {
      const store = useChatRuntimeStore.getState();
      const wantManagerStaging = wantsDownloadManagerStaging(selection);

      if (wantManagerStaging) {
        dismissStartToastsForModelSelection();
      }
      if (store.modelLoading) {
        const isLoadingThisPick =
          !!loadingModel &&
          normalizeModelRef(loadingModel.id) ===
            normalizeModelRef(selection.id) &&
          (loadingModel.ggufVariant ?? null) === (selection.ggufVariant ?? null);
        if (isLoadingThisPick) {
          toast.info("This model is already loading", {
            description: "It's downloading as part of the load in progress.",
          });
          // The duplicate click is the only pick this guard refuses.
          return;
        }
        if (wantManagerStaging) {
          const outcome = await downloadManager.requestStart({
            kind: DOWNLOAD_KIND.MODEL,
            repoId: selection.id,
            variant: selection.ggufVariant ?? null,
            expectedBytes: selection.expectedBytes ?? 0,
            presentation: selection.downloadPresentation,
            // Handed over so one start makes one toast.
            callerToast: {
              title: "Downloading in the background",
              description:
                "It'll be ready to load once the current model finishes.",
            },
          });
          if (outcome === "conflict") {
            toast.info("Resume this download from Models", {
              description:
                "An earlier partial download used a different transport. Open the Model hub tab to resume or restart it.",
            });
          } else if (outcome === "busy") {
            toast.info("Download already in progress", {
              description:
                "Another download for this model is still running. Reselect it once that finishes to load it.",
            });
          }
          return;
        }
        // A different pick falls through: selectModel cancels the pending load and keeps the rollback target.
      }
      if (wantManagerStaging) {
        setPendingHubAutoLoad((current) =>
          current &&
          current.selection.id === selection.id &&
          (current.selection.ggufVariant ?? null) ===
            (selection.ggufVariant ?? null) &&
          current.contextKey === chatContextKey &&
          current.originCheckpoint === store.params.checkpoint &&
          current.originGgufVariant === store.activeGgufVariant
            ? current
            : {
                selection,
                contextKey: chatContextKey,
                originCheckpoint: store.params.checkpoint,
                originGgufVariant: store.activeGgufVariant,
              },
        );
        return;
      }
      setPendingHubAutoLoad(null);
      const previousConfig = currentRuntimePerModelConfig({
        includeMaxSeqLength: true,
      });
      const loadConfig =
        selection.config ?? rememberedConfigFor(selection);
      await selectModel({
        ...selection,
        ...(loadConfig ? { config: loadConfig, keepSpeculative: true } : {}),
        previousConfig,
      });
    },
    [selectModel, loadingModel, rememberedConfigFor, chatContextKey],
  );
  useRepoDownload({
    kind: DOWNLOAD_KIND.MODEL,
    repoId: pendingHubAutoLoad?.selection.id ?? "__hub_autoload_idle__",
    activeVariant: pendingHubAutoLoad?.selection.ggufVariant ?? null,
    onComplete: (variant) => {
      const pending = pendingHubAutoLoad;
      if (
        !pending ||
        (pending.selection.ggufVariant ?? null) !== (variant ?? null)
      ) {
        return;
      }
      setPendingHubAutoLoad(null);
      const store = useChatRuntimeStore.getState();
      if (
        !active ||
        pending.contextKey !== chatContextKey ||
        normalizeModelRef(pending.originCheckpoint) !==
          normalizeModelRef(store.params.checkpoint) ||
        pending.originGgufVariant !== store.activeGgufVariant
      ) {
        return;
      }
      void stageOrLoad({ ...pending.selection, isDownloaded: true });
    },
    onError: (variant) => {
      if (
        pendingHubAutoLoad &&
        (pendingHubAutoLoad.selection.ggufVariant ?? null) === (variant ?? null)
      ) {
        setPendingHubAutoLoad(null);
      }
    },
    onCancelled: (variant) => {
      if (
        pendingHubAutoLoad &&
        (pendingHubAutoLoad.selection.ggufVariant ?? null) === (variant ?? null)
      ) {
        setPendingHubAutoLoad(null);
      }
    },
  });
  const pendingAutoLoadKeyRef = useRef<string | null>(null);
  const chatContextKeyRef = useRef(chatContextKey);
  useEffect(() => {
    chatContextKeyRef.current = chatContextKey;
  }, [chatContextKey]);
  useEffect(() => {
    const pending = pendingHubAutoLoad;
    if (!pending) {
      pendingAutoLoadKeyRef.current = null;
      return;
    }
    const pendingKey = jobKeyOf(
      DOWNLOAD_KIND.MODEL,
      pending.selection.id,
      pending.selection.ggufVariant ?? null,
    );
    pendingAutoLoadKeyRef.current = pendingKey;
    let active = true;
    void (async () => {
      const outcome = await downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId: pending.selection.id,
        variant: pending.selection.ggufVariant ?? null,
        expectedBytes: pending.selection.expectedBytes ?? 0,
        presentation: pending.selection.downloadPresentation,
        // Notice-only: this surface has no toast of its own.
        callerToast: {
          title: "Downloading model",
          description: "It'll load automatically once the download finishes.",
          noticeOnly: true,
          // A raise still in flight would promise an auto-load that onComplete then refuses.
          stillValid: () => chatContextKeyRef.current === pending.contextKey,
        },
      });
      if (!active) return;
      if (outcome === "started") {
        return;
      }
      if (outcome === "conflict") {
        // Keep pendingHubAutoLoad bound so cleanup does not wipe the conflict requestStart recorded.
        toast.info("Resume this download from Models", {
          description:
            "An earlier partial download used a different transport. Open the Model hub tab to resume or restart it.",
        });
        return;
      }
      if (outcome === "busy") {
        toast.info("Download already in progress", {
          description:
            "Another download for this model is still running. Reselect it once that finishes to load it.",
        });
      }
      setPendingHubAutoLoad((current) => (current === pending ? null : current));
    })();
    return () => {
      active = false;
      dismissStartToast(pendingKey);
    };
  }, [pendingHubAutoLoad]);
  // Thread/project switches keep the pathname, so neither sweep above runs.
  useEffect(() => {
    return () => {
      const pendingKey = pendingAutoLoadKeyRef.current;
      if (pendingKey) dismissStartToast(pendingKey);
    };
  }, [chatContextKey]);
  const loadNativeModelIntent = useCallback(
    async (intent: NativeIntent, loadingDescription: string) => {
      const label =
        intent.path.displayLabel || intent.displayLabel || "Local GGUF model";
      await stageOrLoad({
        id: label,
        nativePathToken: intent.path.token,
        nativePathExpiresAtMs: intent.path.expiresAtMs ?? null,
        isDownloaded: true,
        loadingDescription,
        forceReload: true,
        throwOnError: true,
      });
      useNativeIntentStore.getState().clearModelIntent(intent.id);
    },
    [stageOrLoad],
  );
  const handleNativeModelDropAutoLoad = useCallback(
    (intent: NativeIntent) =>
      loadNativeModelIntent(
        intent,
        hasActiveModel
          ? "Replacing with dropped local GGUF model."
          : "Loading dropped local GGUF model.",
      ),
    [hasActiveModel, loadNativeModelIntent],
  );
  const handleNativeAttachmentDrop = useCallback(
    (intents: NativeIntent[]) => {
      useNativeIntentStore.getState().addAttachments(artifactViewKey, intents);
    },
    [artifactViewKey],
  );
  const handleNativeImageDrop = useCallback(
    (intents: NativeIntent[]) => {
      useNativeIntentStore.getState().addImageAttachments(artifactViewKey, intents);
    },
    [artifactViewKey],
  );
  const handleNativeOpenDocumentDrop = useCallback(
    (intents: NativeIntent[]) => {
      useNativeIntentStore
        .getState()
        .addOpenDocumentAttachments(artifactViewKey, intents);
    },
    [artifactViewKey],
  );
  const handleNativeAudioDrop = useCallback(
    (intents: NativeIntent[]) => {
      useNativeIntentStore.getState().addAudioAttachments(artifactViewKey, intents);
    },
    [artifactViewKey],
  );
  const handleNativeVideoDrop = useCallback(
    (intents: NativeIntent[]) => {
      useNativeIntentStore.getState().addVideoAttachments(artifactViewKey, intents);
    },
    [artifactViewKey],
  );
  const nativeModelDropState = useNativeModelDrop({
    // Stay enabled in compare and refuse drops out loud; disabling made them vanish silently.
    enabled: active,
    dropsUnsupportedReason:
      view.mode === "single"
        ? undefined
        : "Dropped files need a single chat. Open one, then drop it there.",
    attachmentScope,
    attachmentTargetKey: artifactViewKey,
    nativePathLeasesSupported,
    hasActiveModel,
    isModelLoading: Boolean(loadingModel) || modelLoading,
    onAutoLoad: handleNativeModelDropAutoLoad,
    onAttach: handleNativeAttachmentDrop,
    onAttachImages: handleNativeImageDrop,
    onAttachOpenDocuments: handleNativeOpenDocumentDrop,
    onAttachAudio: handleNativeAudioDrop,
    onAttachVideo: handleNativeVideoDrop,
  });

  const handleCheckpointChange = useCallback(
    (
      value: string,
      meta?: Partial<ModelSelectorChangeMeta>,
    ) => {
      const store = useChatRuntimeStore.getState();
      const currentCheckpoint = store.params.checkpoint;
      const currentVariant = store.activeGgufVariant;
      if (!value) return;
      setPendingHubAutoLoad(null);
      const isExternalSelection =
        meta?.source === "external" || isExternalModelId(value);
      const isActiveModelLoad = modelOperationInProgress || loadingModel;
      const isSameLoadedModel =
        value === currentCheckpoint &&
        (meta?.ggufVariant ?? null) === (currentVariant ?? null);
      if (isSameLoadedModel && !meta?.forceReload) {
        if (!isExternalSelection) return;
        // With only a preflight in flight, invalidate it here even though no loading flag is published.
        if (!isActiveModelLoad) {
          invalidatePendingModelSelection();
          return;
        }
      }
      if (isNpuModelId(value)) {
        void loadNpuModel(value, {
          forceReload: meta?.forceReload,
          config: meta?.config,
        });
        return;
      }
      if (isExternalSelection) {
        const externalIntentId = invalidatePendingModelSelection();
        let externalCapabilityPatch: Partial<ReturnType<typeof useChatRuntimeStore.getState>> | null = null;
        // Stop a local load in flight: it pins modelLoading and its completion would overwrite these
        // capability fields. Invalidate first so a preflight-parked run yields.
        if (isActiveModelLoad) {
          void cancelLoadingForReplacement(externalIntentId).then((stopped) => {
            if (!isModelSelectionIntentCurrent(externalIntentId)) return;
            if (!stopped) {
              // Backend state is uncertain after a failed stop, so drop the rollback marker.
              discardExternalReplacement(externalIntentId);
              return;
            }
            restoreConfigForExternalReplacement(externalIntentId);
            // The cancelled run may have cleared the checkpoint; put the external pick back.
            const live = useChatRuntimeStore.getState();
            if (live.params.checkpoint !== value) {
              live.setCheckpoint(value, null);
            }
            // Cancellation can finish after clearCheckpoint; reapply the full external capability state.
            if (externalCapabilityPatch) {
              useChatRuntimeStore.setState(externalCapabilityPatch);
            }
          });
        }
        const selectedExternal = parseExternalModelId(value);
        const selectedProvider = selectedExternal
          ? externalProvidersForChat.find(
              (p) => p.id === selectedExternal.providerId,
            )
          : null;
        const reasoningCaps = getExternalReasoningCapabilities(
          selectedProvider?.providerType,
          selectedExternal?.modelId,
          {
            isReasoningProvider: selectedProvider?.isReasoningModel === true,
            baseUrl: selectedProvider?.baseUrl ?? null,
            apiType: selectedProvider?.apiType,
            reasoningConfig: selectedProvider?.reasoningConfig,
          },
        );
        const effortLevels = reasoningCaps.reasoningEffortLevels;
        const pinnedEffort = externalReasoningTakesEffort(reasoningCaps)
          ? pinnedReasoningEffort(value, effortLevels)
          : null;
        const nextReasoningEffort = resolveExternalReasoningEffort({
          caps: reasoningCaps,
          providerType: selectedProvider?.providerType,
          apiType: selectedProvider?.apiType,
          // An unpinned target resolves from the chat's own effort, not the outgoing model's pin.
          current:
            !pinnedEffort && pinHoldsLiveEffort()
              ? (takeEffortDisplacedByPin() ?? store.reasoningEffort)
              : store.reasoningEffort,
          pinned: pinnedEffort,
        });
        if (pinnedEffort && nextReasoningEffort !== store.reasoningEffort) {
          noteEffortDisplacedByPin(store.reasoningEffort);
        }
        // Else the chip keeps a stale openrouter/free ":<chosen>" suffix.
        const stillOnOpenRouterFree =
          selectedProvider?.providerType === "openrouter" &&
          selectedExternal?.modelId === "openrouter/free";
        store.setCheckpoint(value, null);
        const supportsBuiltinWebSearch = providerSupportsBuiltinWebSearch(
          selectedProvider?.providerType,
          selectedExternal?.modelId,
          selectedProvider?.baseUrl,
        );
        const supportsBuiltinCodeExecution =
          providerSupportsBuiltinCodeExecution(
            selectedProvider?.providerType,
            selectedExternal?.modelId,
            selectedProvider?.baseUrl,
            selectedProvider?.apiType,
          );
        const supportsBuiltinImageGeneration =
          providerSupportsBuiltinImageGeneration(
            selectedProvider?.providerType,
            selectedExternal?.modelId,
            selectedProvider?.baseUrl,
            selectedProvider?.apiType,
          );
        const supportsBuiltinWebFetch = providerSupportsBuiltinWebFetch(
          selectedProvider?.providerType,
        );
        const isKimi = selectedProvider?.providerType === "kimi";
        const searchOnByDefault =
          supportsBuiltinWebSearch &&
          (selectedProvider?.providerType === "anthropic" ||
            selectedProvider?.providerType === "openai");
        const storedToolsEnabled =
          threadScopedOverride("toolsEnabled") ??
          loadOptionalBool(CHAT_TOOLS_ENABLED_KEY);
        const storedCodeToolsEnabled =
          threadScopedOverride("codeToolsEnabled") ??
          loadOptionalBool(CHAT_CODE_TOOLS_ENABLED_KEY);
        const storedImageToolsEnabled =
          threadScopedOverride("imageToolsEnabled") ??
          loadOptionalBool(CHAT_IMAGE_TOOLS_ENABLED_KEY);
        const storedWebFetchToolsEnabled =
          threadScopedOverride("webFetchToolsEnabled") ??
          loadOptionalBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY);
        const supportsStudioToolsHere =
          providerModelSupportsStudioTools(
            selectedProvider?.providerType,
            selectedExternal?.modelId,
          ) === true;
        const canSearch = supportsBuiltinWebSearch || supportsStudioToolsHere;
        const canRunCode = codeToolCanRun({
          hostedCodeExecutionForThisTurn: supportsBuiltinCodeExecution,
          providerHostsCodeExecution: providerHostsCodeExecution(
            selectedProvider?.providerType,
            selectedProvider?.baseUrl,
            selectedProvider?.apiType,
          ),
          supportsStudioTools: supportsStudioToolsHere,
        });
        const nextToolsEnabled = canSearch
          ? isKimi
            ? false
            : (storedToolsEnabled ?? searchOnByDefault)
          : false;
        externalCapabilityPatch = {
          activeGgufVariant: null,
          ...loadedContextFields(null),
          activeNativePathToken: null,
          activeNativePathExpiresAtMs: null,
          // Clear previous-model counters, including per-thread copies, or stale stats show.
          contextUsage: null,
          contextUsageByThreadId: {},
          supportsReasoning: reasoningCaps.supportsReasoning,
          reasoningAlwaysOn: reasoningCaps.reasoningAlwaysOn,
          reasoningStyle: reasoningCaps.reasoningStyle,
          supportsReasoningOff: reasoningCaps.supportsReasoningOff,
          reasoningEffortLevels: effortLevels,
          reasoningEffort: nextReasoningEffort,
          reasoningEnabled: reasoningCaps.supportsReasoning
            ? reasoningCaps.supportsReasoningOff
              ? isKimi
                ? true
                : store.reasoningEnabled
              : true
            : store.reasoningEnabled,
          supportsPreserveThinking: providerSupportsPreserveThinking(selectedProvider?.providerType),
          preserveThinking: resolvePreserveThinkingOnLoad({
            supports_preserve_thinking: providerSupportsPreserveThinking(selectedProvider?.providerType),
          }),
          supportsTools: supportsStudioToolsHere,
          supportsBuiltinWebSearch,
          supportsBuiltinCodeExecution,
          supportsBuiltinImageGeneration,
          supportsBuiltinWebFetch,
          toolsEnabled: nextToolsEnabled,
          codeToolsEnabled: canRunCode
            ? (storedCodeToolsEnabled ?? false)
            : false,
          imageToolsEnabled: supportsBuiltinImageGeneration
            ? (storedImageToolsEnabled ?? false)
            : false,
          webFetchToolsEnabled: supportsBuiltinWebFetch
            ? (storedWebFetchToolsEnabled ?? false)
            : false,
          ...(stillOnOpenRouterFree ? {} : { lastOpenRouterChosenModel: null }),
        };
        useChatRuntimeStore.setState(externalCapabilityPatch);
        return;
      }
      useChatRuntimeStore.setState({ lastOpenRouterChosenModel: null });
      // Restore the chat's own effort where the load applies its fields: anything after can abort.
      void (async () => {
        let showImageCompatibilityWarning = false;
        if (view.mode === "single" && activeThreadId) {
          const thread = await getStoredChatThread(activeThreadId);
          if (thread?.modelId && thread.modelId !== value) {
            const messages = await listStoredChatMessages(activeThreadId);
            if (messages.length > 0) {
              const hasImage = messages.some(messageHasImage);
              const targetModel = modelsFromStore.find(
                (model) => model.id === value,
              );
              showImageCompatibilityWarning =
                hasImage &&
                isKnownTextOnlySelection(
                  { isVision: meta?.isVision, isGguf: meta?.isGguf },
                  targetModel,
                );
            }
          }
        }

        if (showImageCompatibilityWarning) {
          toast.warning("Selected model may not handle earlier images", {
            description:
              "This chat already includes images. Text-only models can ignore them or fail on follow-up replies.",
            duration: 6000,
          });
        }
        const selection = {
          id: value,
          loadId: meta?.loadId,
          source: meta?.source,
          isLora: meta?.isLora,
          ggufVariant: meta?.ggufVariant,
          isDownloaded: meta?.isDownloaded || isSameLoadedModel,
          expectedBytes: meta?.expectedBytes,
          downloadPresentation: meta?.downloadPresentation,
          isGguf: meta?.isGguf,
          isVision: meta?.isVision,
          isDiffusion: meta?.isDiffusion,
          config: meta?.config,
          nativePathToken: meta?.nativePathToken,
          nativePathExpiresAtMs: meta?.nativePathExpiresAtMs,
          forceReload: meta?.forceReload ?? (isSameLoadedModel || undefined),
        };
        await stageOrLoad(selection);
      })();
    },
    [
      activeThreadId,
      externalProvidersForChat,
      loadNpuModel,
      modelsFromStore,
      stageOrLoad,
      view,
      modelOperationInProgress,
      loadingModel,
      cancelLoadingForReplacement,
      invalidatePendingModelSelection,
      discardExternalReplacement,
      restoreConfigForExternalReplacement,

      isModelSelectionIntentCurrent,
    ],
  );
  const handleReloadActiveModel = useCallback(
    (config: PerModelConfig) => {
      const checkpoint = inferenceParams.checkpoint;
      if (!checkpoint) return;
      const runtime = useChatRuntimeStore.getState();
      const activeLoadId = runtime.activeLoadId;
      const nativeToken = runtime.activeNativePathToken;
      const nativeExpiry = runtime.activeNativePathExpiresAtMs;
      // Native path tokens expire on the desktop host; an expired one fails opaquely, so re-select.
      if (nativeToken && nativeExpiry != null && Date.now() >= nativeExpiry) {
        toast.error("This local model file's access has expired.", {
          description: "Re-select the model file to reload it.",
        });
        return;
      }
      handleCheckpointChange(checkpoint, {
        source: "local",
        isLora: activeModelIsLora,
        loadId: activeLoadId,
        ggufVariant: activeGgufVariant ?? undefined,
        // Without the token the reload treats the display label as a repo id and fails.
        nativePathToken: nativeToken ?? undefined,
        nativePathExpiresAtMs: nativeExpiry,
        isGguf: activeModelIsGguf,
        isDiffusion: activeModelIsDiffusion,
        isDownloaded: true,
        config,
        forceReload: true,
      });
    },
    [
      inferenceParams.checkpoint,
      activeGgufVariant,
      activeModelIsLora,
      activeModelIsGguf,
      activeModelIsDiffusion,
      handleCheckpointChange,
    ],
  );
  const handleEject = useCallback(
    (modelId?: string) => {
      void ejectModel(modelId);
    },
    [ejectModel],
  );
  const handleEjectAll = useCallback(() => {
    void ejectAllModels();
  }, [ejectAllModels]);

  // Pinned open for tour steps so a stray click cannot dismiss it.
  const openModelSelector = useCallback(() => {
    setModelSelectorLocked(true);
    setModelSelectorOpen(true);
  }, []);

  const closeModelSelector = useCallback(() => {
    setModelSelectorLocked(false);
    setModelSelectorOpen(false);
  }, []);

  const toggleModelSelector = useCallback(() => {
    if (modelSelectorLocked) return;
    setModelSelectorOpen((open) => !open);
  }, [modelSelectorLocked]);

  const handleModelSelectorOpenChange = useCallback(
    (open: boolean) => {
      if (!open && modelSelectorLocked) return;
      setModelSelectorOpen(open);
    },
    [modelSelectorLocked],
  );
  const openSettings = useCallback(
    () => setSettingsOpen(true),
    [setSettingsOpen],
  );
  const closeSettings = useCallback(
    () => setSettingsOpen(false),
    [setSettingsOpen],
  );

  // Compare drops the header pickers, so the chord would toggle state nothing renders.
  const headerPickersShown = active && view.mode !== "compare";
  // The page stays mounted under dialogs, so check at press time that the header is not covered.
  const chatCovered = () => isSurfaceBackgrounded(COMPOSER_INPUT_SELECTOR);
  useShortcut(
    "openModelPicker",
    () => {
      if (chatCovered()) return;
      toggleModelSelector();
    },
    { enabled: headerPickersShown },
  );
  const projectSwitcherShown = headerPickersShown && Boolean(currentProjectId);
  useShortcut(
    "openProjectPicker",
    () => {
      if (chatCovered()) return;
      setProjectPickerOpen(true);
    },
    { enabled: projectSwitcherShown },
  );
  // Close a picker that is no longer shown, or it returns as a ghost; adjusted during render.
  if (!projectSwitcherShown && projectPickerOpen) {
    setProjectPickerOpen(false);
  }

  const shiftReasoningEffort = useCallback(
    (delta: number, wrap: boolean) => {
      const state = useChatRuntimeStore.getState();
      const levels = state.reasoningEffortLevels;
      // enable_thinking models keep levels populated but drop the effort; same test as the effort menu.
      const isEffort =
        state.reasoningStyle === "reasoning_effort" ||
        state.reasoningStyle === "enable_thinking_effort";
      if (!state.supportsReasoning || !isEffort || levels.length === 0) {
        toast.info("This model has no reasoning effort setting");
        return;
      }
      const current = levels.indexOf(state.reasoningEffort);
      // The level in force may be gone after a load (indexOf -1); the first press picks the lowest.
      if (current === -1) {
        state.setReasoningEffort(levels[0]);
        return;
      }
      const from = current;
      const next = wrap
        ? (from + delta + levels.length) % levels.length
        : Math.min(Math.max(from + delta, 0), levels.length - 1);
      if (levels[next] === state.reasoningEffort) return;
      state.setReasoningEffort(levels[next]);
    },
    [],
  );
  useShortcut(
    "cycleReasoningEffort",
    () => {
      if (chatCovered()) return;
      shiftReasoningEffort(1, true);
    },
    { enabled: active },
  );
  useShortcut(
    "increaseReasoningEffort",
    () => {
      if (chatCovered()) return;
      shiftReasoningEffort(1, false);
    },
    { enabled: active },
  );
  useShortcut(
    "decreaseReasoningEffort",
    () => {
      if (chatCovered()) return;
      shiftReasoningEffort(-1, false);
    },
    { enabled: active },
  );

  const fastModeSupported = providerSupportsFastMode(
    activeExternalProviderType,
    parseExternalModelId(inferenceParams.checkpoint)?.modelId ?? null,
  );
  useShortcut(
    "toggleFastMode",
    () => {
      if (chatCovered()) return;
      const state = useChatRuntimeStore.getState();
      const next = !state.params.fastMode;
      state.setParams({ ...state.params, fastMode: next });
      toast.success(next ? "Fast mode on" : "Fast mode off");
    },
    { enabled: active && fastModeSupported },
  );
  const { isMobile, pinned } = useSidebar();
  // The tour hold is never persisted, so a tour cannot rewrite the user's pin.
  const showSidebarForTour = holdSidebarPinned;
  const restoreSidebarAfterTour = releaseSidebarPinned;

  const enterCompare = useCallback(() => {
    viewBeforeCompareRef.current = { ...search };
    useChatRuntimeStore.getState().setActiveThreadId(null);
    useChatRuntimeStore.getState().setContextUsage(null);
    navigate({
      to: "/chat",
      search: {
        compare: crypto.randomUUID(),
        ...(currentProjectId ? { project: currentProjectId } : {}),
      },
    });
  }, [currentProjectId, navigate, search]);

  const exitCompare = useCallback(() => {
    // the composer and menu exit paths rely on the last non-compare view.
    const saved = viewBeforeCompareRef.current ?? lastNonCompareViewRef.current;
    // direct compare URLs have no saved view, so return to a fresh chat.
    if (!saved) {
      navigate({ to: "/chat" });
      return;
    }
    viewBeforeCompareRef.current = null;
    navigate({ to: "/chat", search: saved });
    const threadId =
      saved.thread ?? useChatRuntimeStore.getState().activeThreadId;
    if (threadId) {
      void listStoredChatMessages(threadId)
        .then((messages) => {
          const store = useChatRuntimeStore.getState();
          const branch = orderBySelectedBranch(
            messages,
            savedBranchHead(threadId, messages),
          );
          const usage = savedUsageFor(branch, store) ?? estimateContextUsage(branch);
          if (!usage) return;
          // key usage by the restored thread because this read can outlast a thread switch.
          store.setThreadContextUsage(threadId, usage);
          if (store.activeThreadId === threadId) {
            store.setContextUsage(usage);
          }
        })
        .catch((error) => {
          if (!isExpectedBackgroundChatStorageError(error)) {
            throw error;
          }
        });
    }
  }, [navigate]);

  const models = useMemo<ModelOption[]>(
    () =>
      modelsFromStore.map((model) => ({
        id: model.id,
        name: model.name,
        description: model.description,
        isGguf: model.isGguf,
      })),
    [modelsFromStore],
  );
  const lastOpenRouterChosenModel = useChatRuntimeStore(
    (s) => s.lastOpenRouterChosenModel,
  );
  const externalModels = useMemo<ExternalModelOption[]>(
    () =>
      externalProvidersForChat
        .filter((provider) => !isDecisionConnection(provider))
        .sort(
          (a, b) =>
            getExternalProviderDropdownRank(a.providerType) -
            getExternalProviderDropdownRank(b.providerType),
        )
        .flatMap((provider) =>
          provider.models.map((model) => {
            // OpenRouter free router: show `openrouter:<chosen>` minus `/free` and the org prefix.
            let displayName = model;
            if (
              provider.providerType === "openrouter" &&
              model === "openrouter/free" &&
              lastOpenRouterChosenModel
            ) {
              const lastSlash = lastOpenRouterChosenModel.lastIndexOf("/");
              const shortChosen =
                lastSlash >= 0
                  ? lastOpenRouterChosenModel.slice(lastSlash + 1)
                  : lastOpenRouterChosenModel;
              displayName = `openrouter:${shortChosen}`;
            }
            return {
              id: buildExternalModelId(provider.id, model),
              name: displayName,
              providerId: provider.id,
              providerName: provider.name,
              providerType: provider.providerType,
            };
          }),
        ),
    [externalProvidersForChat, lastOpenRouterChosenModel],
  );
  // An unticked model looks like a withdrawn one; the connection's cached catalog tells them apart.
  const externalConnections = useMemo<ExternalConnectionRef[]>(
    () =>
      connectionsEnabled
        ? externalProviders.map((provider) => ({
            id: provider.id,
            name: provider.name,
            providerType: provider.providerType,
            availableModels: provider.availableModels,
          }))
        : [],
    [connectionsEnabled, externalProviders],
  );

  const localModelInventory = useDeviceInventorySources(["localModels"], {
    enabled: active,
  });
  const localModels = useMemo<LoraModelOption[]>(
    () => chatLocalModelOptions(localModelInventory.localModels.rows),
    [localModelInventory.localModels.rows],
  );

  const refreshLocalModels = useCallback(() => {
    void localModelInventory.refresh();
  }, [localModelInventory.refresh]);

  const refreshModelLists = useCallback(
    (deletedModel?: DeletedModelRef) => {
      const { checkpoint } = useChatRuntimeStore.getState().params;
      const activeGgufVariant =
        useChatRuntimeStore.getState().activeGgufVariant;
      if (
        modelMatchesDeleted(
          { id: checkpoint, ggufVariant: activeGgufVariant },
          deletedModel,
        )
      ) {
        useChatRuntimeStore.getState().clearCheckpoint();
      }
      void refresh();
      refreshLocalModels();
    },
    [refresh, refreshLocalModels],
  );

  const loraModels = useMemo<LoraModelOption[]>(() => {
    const fromLoras = lorasFromStore.map((lora) => ({
      id: lora.id,
      name: lora.name,
      baseModel: lora.baseModel,
      updatedAt: lora.updatedAt,
      source: lora.source,
      exportType: lora.exportType,
      sizeBytes: lora.sizeBytes,
      audioType: lora.audioType,
    }));
    return [...fromLoras, ...localModels];
  }, [lorasFromStore, localModels]);

  const selectableModelIds = useMemo(
    () =>
      new Set<string>([
        ...models.map((model) => model.id),
        ...loraModels.map((model) => model.id),
        ...externalModels.map((model) => model.id),
      ]),
    [models, loraModels, externalModels],
  );

  // Pass the row metadata as the picker does: local/fine-tuned rows are in no list, so the bare id
  // loads with different arguments.
  const handleSwitchBackToChatModel = useCallback(
    (target: ChatModelSwitchTarget) => {
      handleCheckpointChange(
        target.modelId,
        chatModelSwitchMeta(target, loraModels),
      );
    },
    [handleCheckpointChange, loraModels],
  );

  const inventoryRefreshStartedRef = useRef(false);
  const refreshDeferredModelInventories = useCallback(() => {
    inventoryRefreshStartedRef.current = true;
    void refresh({ includeLoras: true });
    void localModelInventory.refreshIfOlderThan(INVENTORY_FRESHNESS_WINDOW_MS);
  }, [refresh, localModelInventory.refreshIfOlderThan]);

  useEffect(() => {
    if (getTrainingCompareHandoff()) return;
    const controller = new AbortController();
    // Models and status only: a hanging LoRA scan would empty the picker via Promise.all.
    void refresh({
      includeLoras: false,
      signal: controller.signal,
      waitForServerModel: !useChatRuntimeStore.getState().params.checkpoint,
    });
    const timeoutId = window.setTimeout(() => {
      if (!inventoryRefreshStartedRef.current) {
        refreshDeferredModelInventories();
      }
    }, 1200);
    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [refresh, refreshDeferredModelInventories]);

  useEffect(() => {
    if (!active || !modelSelectorOpen) return;
    refreshDeferredModelInventories();
  }, [active, modelSelectorOpen, refreshDeferredModelInventories]);

  useEffect(() => {
    // ChatPage no longer remounts on navigation, so re-check the handoff on return.
    if (!active) return;
    const handoff = getTrainingCompareHandoff();
    if (!handoff) return;
    console.info("[chat-handoff] received", handoff);
    function clearHandoff(): void {
      clearTrainingCompareHandoff();
    }

    let canceled = false;
    void (async () => {
      try {
        console.info("[chat-handoff] refreshing models+loras");
        await refreshRef.current();
        if (canceled) return;

        const state = useChatRuntimeStore.getState();
        const target = pickTrainingCompareTarget(state.loras, handoff);
        const selectWithConfig = async (
          selection: Pick<
            SelectedModelInput,
            "id" | "isLora" | "isDownloaded"
          >,
        ) => {
          const previousConfig = currentRuntimePerModelConfig({
            includeMaxSeqLength: true,
          });
          const remembered = rememberedConfigFor(selection);
          const hasAppliedConfig = applyModelLoadConfigToRuntime(remembered);
          await selectModelRef.current({
            ...selection,
            ...(hasAppliedConfig ? { keepSpeculative: true } : {}),
            previousConfig,
            // The runtime mirror carries no launch flags, so /load has nothing to inherit them from.
            ...(remembered ? { config: remembered } : {}),
          });
        };
        if (target) {
          const selection = trainingCompareSelection(target);
          console.info("[chat-handoff] loading trained model", {
            ...selection,
            baseModel: target.baseModel,
          });
          await selectWithConfig(selection);
          if (canceled) return;
          useChatRuntimeStore.getState().setActiveThreadId(null);
          useChatRuntimeStore.getState().setContextUsage(null);
          navigate({ to: "/chat", search: { compare: crypto.randomUUID() } });
          clearHandoff();
          console.info("[chat-handoff] loaded trained model + opened compare");
          return;
        }

        if (
          handoff.baseModel &&
          state.models.some((model) => model.id === handoff.baseModel)
        ) {
          console.info("[chat-handoff] no lora match, loading base", {
            id: handoff.baseModel,
          });
          await selectWithConfig({ id: handoff.baseModel, isLora: false });
          if (canceled) return;
        } else {
          console.warn("[chat-handoff] no lora/base match found", {
            requestedBaseModel: handoff.baseModel,
            loraCount: state.loras.length,
            modelCount: state.models.length,
          });
        }
        clearHandoff();
        console.info("[chat-handoff] completed");
      } catch (error) {
        console.error("[chat-handoff] failed", error);
        clearHandoff();
      }
    })();

    return () => {
      canceled = true;
    };
  }, [active, navigate, rememberedConfigFor]);

  const tourSteps = useMemo(
    () =>
      // eslint-disable-next-line react-hooks/refs -- buildChatTourSteps stores callbacks without invoking them during render.
      buildChatTourSteps({
        canShowNav: !isMobile,
        canCompare,
        openModelSelector,
        closeModelSelector,
        openSettings,
        closeSettings,
        enterCompare,
        exitCompare,
      }).map((step) =>
        step.target === "navbar"
          ? {
              ...step,
              onEnter: showSidebarForTour,
              onExit: restoreSidebarAfterTour,
            }
          : step,
      ),
    [
      canCompare,
      closeModelSelector,
      closeSettings,
      enterCompare,
      exitCompare,
      isMobile,
      openModelSelector,
      openSettings,
      restoreSidebarAfterTour,
      showSidebarForTour,
    ],
  );

  const tour = useGuidedTourController({
    id: "chat",
    steps: tourSteps,
    enabled: active,
  });

  useEffect(() => {
    if (tour.open) return;
    if (!modelSelectorLocked) return;
    const timeoutId = window.setTimeout(() => {
      setModelSelectorLocked(false);
      setModelSelectorOpen(false);
    }, 0);
    return () => window.clearTimeout(timeoutId);
  }, [modelSelectorLocked, tour.open]);

  // Compare, projects (no side pane) and phones show the browser over the chat.
  const showBrowserOverlay =
    active &&
    browserOpen &&
    (view.mode === "compare" ||
      isMobile ||
      (view.mode === "project" &&
        projectBrowserBaseline !== null &&
        browserOpenSequence > projectBrowserBaseline));

  return (
    <ChatActiveContext.Provider value={active}>
    <div className="flex min-h-0 min-w-0 flex-1 basis-0 overflow-hidden bg-background">
      {/* Portaled surfaces escape the hidden wrapper, so gate them on `active`. */}
      {active && <GuidedTour {...tour.tourProps} />}
      {/* One app-level mount driven by global state, or Compare's composers each render a copy. */}
      {active && <BypassPermissionsConfirmDialog />}
      {active && <RootSandboxSetupDialog />}
      {/* Mounted always: its chord must work before MCP is on, and it closes itself on route change. */}
      <McpServersDialogMount />
      {/* Declares --studio-chat-notice-height for both. `has-[>...]`, not `has-[...]`: a descendant
          :has() is re-checked on every insertion in the thread. ChatModelNotice is a direct child. */}
      <div
        data-chat-content-root=""
        className="relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden has-[>[data-chat-model-notice]]:[--studio-chat-notice-height:2.25rem]"
      >
        <NativeModelDropOverlay state={nativeModelDropState} />
        {view.mode !== "compare" && (
          <div
            aria-hidden
            data-side-panel-inset=""
            className="chat-header-fade pointer-events-none absolute left-[var(--studio-side-panel-left,0px)] right-[calc(var(--thread-scrollbar-gutter,10px)+var(--studio-side-panel-width,0px))] top-[calc(var(--studio-content-top-inset,0px)+var(--studio-chat-header-height,48px)+var(--studio-chat-notice-height,0px))] z-20 h-6 bg-gradient-to-b from-background to-transparent"
          />
        )}
        <div
          data-side-panel-inset=""
          className={cn(
            "pointer-events-none absolute top-[var(--studio-content-top-inset,0px)] left-[var(--studio-side-panel-left,0px)] right-[calc(var(--thread-scrollbar-gutter,10px)+var(--studio-side-panel-width,0px))] z-40 flex h-[var(--studio-chat-header-height,48px)] shrink-0 items-start bg-background pt-[var(--studio-chat-header-padding-top,11px)] pr-[calc(0.5rem*var(--ui-space-scale,1)+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]",
            isMobile
              ? "pl-12"
              : pinned
                ? "pl-2"
                : isTauri
                  ? "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]"
                  : "pl-[calc(0.5rem*var(--ui-space-scale,1)+max(0px,var(--studio-mac-traffic-light-inset,0px)-var(--sidebar-width-icon,3rem)))]",
            view.mode === "compare" &&
              "right-[var(--thread-scrollbar-gutter,10px)] left-auto w-auto bg-transparent pl-0 pr-[calc(0.5rem*var(--ui-space-scale,1)+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]",
          )}
        >
          <div className="pointer-events-auto flex min-w-0 items-center gap-1">
            {isTauri && !isMobile && !pinned && view.mode !== "compare" && (
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                title="New chat"
                aria-label="New chat"
                onClick={handleDesktopNewChat}
                className="!size-[calc(30px*var(--ui-space-scale,1))] shrink-0 rounded-[10px] text-muted-foreground"
              >
                <HugeiconsIcon
                  icon={PencilEdit02Icon}
                  strokeWidth={1.75}
                  className="size-icon"
                />
              </Button>
            )}
            {view.mode !== "compare" && (
              <ModelSelector
                models={models}
                loraModels={loraModels}
                externalModels={externalModels}
                externalConnections={externalConnections}
                value={inferenceParams.checkpoint}
                // Resident, not merely picked: an image or video load evicts the chat model.
                loaded={chatModelLoaded({
                  checkpoint: inferenceParams.checkpoint,
                  isExternalModel: isExternalModelId(
                    inferenceParams.checkpoint,
                  ),
                  residentCheckpoint,
                })}
                activeGgufVariant={activeGgufVariant}
                activeModelConfig={activeModelConfig}
                activeLoadedContextLength={loadedContextLength}
                configRequest={modelConfigRequest}
                onConfigRequestAdopted={handleModelConfigRequestAdopted}
                onValueChange={handleCheckpointChange}
                onEject={handleEject}
                onEjectAll={handleEjectAll}
                loadedCount={loadedCount}
                onFoldersChange={refreshLocalModels}
                onModelsChange={refreshModelLists}
                deleteDisabled={modelOperationInProgress}
                variant="ghost"
                open={
                  active && (modelSelectorOpen || modelConfigRequest !== null)
                }
                onOpenChange={handleModelSelectorOpenChange}
                triggerDataTour="chat-model-selector"
                contentDataTour="chat-model-selector-popover"
                showCloudIndicator={isExternalModel}
                className="max-w-[62vw] !pr-3 md:max-w-none !h-[var(--studio-chat-control-height,34px)]"
              />
            )}
            {view.mode !== "compare" && currentProjectId && (
              <nav
                aria-label="Project location"
                className="flex h-[var(--studio-chat-control-height,34px)] min-w-0 items-center gap-1.5 self-center text-ui-13p5 tracking-nav text-muted-foreground"
              >
                <ProjectSwitcher
                  currentProject={currentProject}
                  projects={projects}
                  isLoading={projectsLoading}
                  onSelectProject={openProjectLanding}
                  onViewAllProjects={openProjectsList}
                  open={projectPickerOpen}
                  onOpenChange={setProjectPickerOpen}
                />
                {currentProject && activeThreadId ? (
                  <>
                    <span className="shrink-0" aria-hidden={true}>
                      /
                    </span>
                    <span className="min-w-0 truncate">
                      {currentChatTitle ?? "New chat"}
                    </span>
                  </>
                ) : null}
              </nav>
            )}
            {pendingNativeModelIntent && view.mode !== "compare" ? (
              <NativeModelChip
                intent={pendingNativeModelIntent}
                nativeReadsDisabled={!nativePathLeasesSupported}
                onLoad={() =>
                  loadNativeModelIntent(
                    pendingNativeModelIntent,
                    "Loading selected local GGUF model.",
                  )
                }
              />
            ) : null}
            {loadingModel && loadToastDismissed ? (
              <ModelLoadInlineStatus
                label={
                  loadProgress?.phase === "starting"
                    ? "Starting model…"
                    : loadingModel.isDownloaded || loadingModel.isCachedLora
                      ? "Loading model…"
                      : "Downloading model…"
                }
                title={
                  loadingModel.isDownloaded
                    ? `Loading ${loadingModel.displayName} from cache.`
                    : loadingModel.isCachedLora
                      ? `Loading ${loadingModel.displayName} into memory.`
                      : `Loading ${loadingModel.displayName}. This may include downloading.`
                }
                progressPercent={loadProgress?.percent}
                progressLabel={loadProgress?.label}
                onStop={cancelLoading}
              />
            ) : null}
            {!loadingModel && modelsError ? (
              <div
                className="relative top-0.5 pl-0.5"
                role="status"
                aria-live="polite"
              >
                <CopyableErrorChip message={modelsError} />
              </div>
            ) : null}
          </div>
          <div className="pointer-events-auto ml-auto flex min-w-min max-w-max grow basis-0 items-center gap-1 *:shrink-0">
            {showContextWindowUsage &&
            (view.mode === "single" ||
              (view.mode === "project" && activeThreadId != null)) &&
            (contextUsage || contextWindowKnown) ? (
              <ContextUsageBar
                used={contextUsage?.contextTokens ?? contextUsage?.totalTokens ?? null}
                // null on external providers; the bar handles that.
                total={loadedContextLength}
                cached={contextUsage?.cachedTokens}
                cacheWrites={contextUsage?.cacheWriteTokens}
                promptTokens={contextUsage?.promptTokens}
                // A tool turn's completionTokens sums every pass; the context holds only the last one.
                completionTokens={
                  contextUsage?.contextTokens !== undefined
                    ? contextUsage.contextTokens - contextUsage.promptTokens
                    : contextUsage?.completionTokens
                }
                isMlx={isServedByMlx(
                  Boolean(loadedIsGguf),
                  platformDeviceType,
                  platformChatOnlyReason,
                )}
                contextEnforced={loadedContextEnforced}
                contextUnboundedWhenBatched={loadedContextUnboundedWhenBatched}
                parallelSlots={loadedParallelSlots}
                contextBudget={loadedContextBudget}
                estimated={contextUsage?.estimated}
                className="h-[var(--studio-chat-control-height,34px)]"
              />
            ) : null}
            {view.mode === "single" && incognito ? (
              <SaveTemporaryChatMenu
                className="mr-[calc(6px*var(--ui-space-scale,1))]"
                onDiscard={toggleIncognito}
              />
            ) : null}
            {view.mode === "single" && (
              <ChatHeaderMenu temporary={incognito} onToggleTemporary={toggleIncognito} />
            )}
            {view.mode === "single" && !isMobile ? <BrowserToggleButton active={active} /> : null}
            {view.mode === "single" &&
            latestResearchRunId &&
            latestResearchRunStatus ? (
              <Tooltip>
                <TooltipPrimitive.Trigger asChild={true}>
                  <button
                    type="button"
                    onClick={() => {
                      if (openResearchRunId === latestResearchRunId) {
                        closeResearchPanel();
                        return;
                      }
                      setSettingsOpen(false);
                      openResearchPanel(latestResearchRunId);
                    }}
                    className="relative flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:hover:text-white"
                    aria-label="Open research activity"
                    aria-pressed={openResearchRunId === latestResearchRunId}
                  >
                    <HugeiconsIcon
                      icon={Telescope02Icon}
                      className="size-icon"
                      strokeWidth={1.75}
                    />
                    {!['completed', 'failed', 'cancelled'].includes(latestResearchRunStatus) ? (
                      <span className="absolute right-1 top-1 size-1.5 rounded-full bg-primary ring-2 ring-background" />
                    ) : null}
                  </button>
                </TooltipPrimitive.Trigger>
                <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
                  Research activity
                </TooltipContent>
              </Tooltip>
            ) : null}
            {!settingsOpen && (
              <Tooltip>
                <TooltipPrimitive.Trigger asChild={true}>
                  <button
                    type="button"
                    onClick={() => {
                      useResearchRunStore.getState().closePanel();
                      setSettingsOpen(true);
                    }}
                    className="flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    aria-label="Open run settings"
                  >
                    <HugeiconsIcon
                      icon={LayoutAlignRightIcon}
                      strokeWidth={1.75}
                      className="size-icon"
                    />
                  </button>
                </TooltipPrimitive.Trigger>
                <TooltipContent
                  side="bottom"
                  sideOffset={6}
                  className="tooltip-compact"
                >
                  Open run settings
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>

        {view.mode === "single" && (
          <ChatModelNotice
            threadId={view.threadId ?? newChatThreadId ?? undefined}
            checkpoint={inferenceParams.checkpoint}
            activeGgufVariant={activeGgufVariant}
            selectableModelIds={selectableModelIds}
            onSwitch={handleSwitchBackToChatModel}
          />
        )}

        {/* Never keyed: a key remounts the runtime on view switch and aborts generation. Compare is
            a sibling because ComparePane builds its own providers and nesting throws. */}
        {baseView ? (
          <div
            className={
              baseBackgrounded
                ? "hidden"
                : "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden"
            }
            inert={baseBackgrounded || undefined}
          >
            <BrowserOverlaidContext.Provider value={baseBackgrounded}>
            <ChatActiveContext.Provider value={active && !baseBackgrounded}>
              <ChatRuntimeProvider
                modelType="base"
                projectId={baseView.projectId}
                initialThreadId={
                  baseView.mode === "single" ? baseView.threadId : undefined
                }
                newThreadNonce={
                  baseView.mode === "project"
                    ? projectNewThreadNonce
                    : baseView.newThreadNonce
                }
                listThreads={false}
                backgrounded={baseBackgrounded}
                onInitialHistoryReady={
                  baseView.mode === "project"
                    ? markProjectRuntimeReady
                    : undefined
                }
              >
                {baseView.mode === "project" ? (
                  <ProjectLanding
                    key={baseView.projectId}
                    projectId={baseView.projectId}
                    projectName={currentProject?.name ?? "Project"}
                    items={currentProjectItems}
                    newThreadNonce={projectNewThreadNonce}
                    rotateNewThreadNonce={rotateProjectNewThreadNonce}
                    dataLoaded={currentProjectItemsLoaded && !projectsLoading}
                    runtimeReady={projectRuntimeReady}
                  />
                ) : (
                  <NativeAttachmentTargetContext.Provider
                    value={baseAttachmentTargetKey}
                  >
                    <TemporaryChatSaveBridge />
                    <SingleContent threadId={baseView.threadId} />
                  </NativeAttachmentTargetContext.Provider>
                )}
              </ChatRuntimeProvider>
            </ChatActiveContext.Provider>
            </BrowserOverlaidContext.Provider>
          </div>
        ) : null}
        {view.mode === "compare" ? (
          <CompareContent
            key={view.pairId}
            pairId={view.pairId}
            projectId={view.projectId}
            models={models}
            loraModels={loraModels}
            externalModels={externalModels}
            externalConnections={externalConnections}
            onFoldersChange={refreshLocalModels}
            onModelsChange={refreshModelLists}
            deleteDisabled={modelOperationInProgress}
            onExitCompare={exitCompare}
          />
        ) : null}

        {showBrowserOverlay ? <BrowserOverlay /> : null}
      </div>

      <ChatSettingsPanel
        open={active && modelConfigRequest === null && settingsOpen}
        onOpenChange={(open) => {
          setSettingsOpen(open);
        }}
        params={inferenceParams}
        onParamsChange={setInferenceParams}
        modelConfig={
          view.mode !== "compare" && activeModelConfig && !modelLoading ? (
            <SidebarModelConfig
              modelId={inferenceParams.checkpoint}
              ggufVariant={activeGgufVariant ?? null}
              isGguf={activeModelIsGguf}
              isLora={activeModelIsLora}
              isDiffusion={activeModelIsDiffusion}
              nativeContextLength={nativeContextLength}
              loadedContextLength={loadedContextLength}
              loadedConfig={activeModelConfig}
              onReload={handleReloadActiveModel}
            />
          ) : null
        }
        isExternalModel={isExternalModel}
        providerCapabilities={activeProviderCapabilities}
        activeExternalProvider={activeExternalProvider}
        onExternalProviderChange={(updatedProvider) => {
          setExternalProviders(
            externalProviders.map((provider) =>
              provider.id === updatedProvider.id ? updatedProvider : provider,
            ),
          );
        }}
        externalProviderType={activeExternalProviderType}
      />
    </div>
    </ChatActiveContext.Provider>
  );
}
