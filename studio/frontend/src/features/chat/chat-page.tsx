// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  applyModelLoadConfigToRuntime,
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
  resolveInitialConfig,
  SidebarModelConfig,
  useActiveModelConfig,
} from "@/features/model-picker";
import { ProjectComposer, Thread } from "@/components/assistant-ui/thread";
import { usePlatformStore } from "@/config/env";
import { CopyableErrorChip } from "@/components/ui/copyable-error-chip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { useSidebar } from "@/components/ui/sidebar";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { useIsMobile } from "@/hooks/use-mobile";
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
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { isTauri } from "@/lib/api-base";
import { chatModelLoaded } from "./lib/chat-model-loaded";
import { hasKnownContextWindow } from "./lib/context-window-known";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  CONVERSATION_MARKDOWN_FORMAT,
  CONVERSATION_MARKDOWN_LABEL,
} from "./utils/conversation-markdown";
import {
  Archive03Icon,
  BookOpen01Icon,
  BubbleChatTemporaryIcon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
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
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import {
  type CSSProperties,
  type ReactElement,
  lazy,
  memo,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { PanelImperativeHandle } from "react-resizable-panels";
import { notifyChatHistoryUpdated } from "./api/chat-api";
import { codeToolCanRun } from "./api/code-tool-placement";
import { ArtifactSurface } from "./artifacts/artifact-surface";
import {
  clearAutoOpenedArtifacts,
  useChatArtifactsStore,
  useSelectedChatArtifact,
} from "./artifacts/store";
import type { ChatArtifact, ChatArtifactSurface } from "./artifacts/types";
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
import {
  buildExternalModelId,
  isExternalModelId,
  parseExternalModelId,

  providerModelSupportsStudioTools,
} from "./external-providers";
import { useChatModelRuntime } from "./hooks/use-chat-model-runtime";
import type { SelectedModelInput } from "./hooks/use-chat-model-runtime";
import {
  deleteChatProject,
  moveChatItemToProject,
  renameChatProject,
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
import { usePinnedProjectsStore } from "./stores/pinned-projects-store";
import {
  clearTrainingCompareHandoff,
  getTrainingCompareHandoff,
} from "./lib/training-compare-handoff";
import {
  clampReasoningEffortToLevels,
  getExternalReasoningCapabilities,
  getProviderCapabilities,
  providerHostsCodeExecution,
  providerSupportsBuiltinCodeExecution,
  providerSupportsBuiltinImageGeneration,
  providerSupportsBuiltinWebFetch,
  providerSupportsBuiltinWebSearch,
  providerSupportsFastMode,
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
import {
  CHAT_CODE_TOOLS_ENABLED_KEY,
  CHAT_IMAGE_TOOLS_ENABLED_KEY,
  CHAT_TOOLS_ENABLED_KEY,
  CHAT_WEB_FETCH_TOOLS_ENABLED_KEY,
  PENDING_CHAT_ATTACHMENT_KEY,
  loadOptionalBool,
  readPendingAttachmentTargetClaim,
  threadScopedOverride,
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
import { attachmentsSample } from "./utils/pasted-text";
import { requestTemporaryPromptQueueStop } from "./utils/prompt-queue-boundary";
import { isAssistantLocalThreadId } from "./utils/thread-ids";
import {
  consumeProjectSourcesPending,
  hasProjectSourcesPending,
} from "@/features/rag/components/project-source-dropzone";

const ProjectSourcesPanel = lazy(() =>
  import("@/features/rag/components/project-sources-panel").then((module) => ({
    default: module.ProjectSourcesPanel,
  })),
);

type LoraCandidate = {
  id: string;
  baseModel: string;
  updatedAt?: number;
  exportType?: "lora" | "merged" | "gguf";
};

const EXTERNAL_PROVIDER_DROPDOWN_ORDER: Record<string, number> = {
  openai: 0,
  anthropic: 1,
};

function getExternalProviderDropdownRank(providerType: string): number {
  return EXTERNAL_PROVIDER_DROPDOWN_ORDER[providerType] ?? 2;
}

function normalizeModelRef(value: string | null | undefined): string {
  return value?.trim().toLowerCase() ?? "";
}

function pickBestLoraForBase(
  loras: LoraCandidate[],
  baseModel: string | null,
): LoraCandidate | null {
  const adapterOnly = loras.filter((lora) => lora.exportType === "lora");
  if (adapterOnly.length === 0) return null;
  const sorted = [...adapterOnly].sort(
    (a, b) => (b.updatedAt ?? -1) - (a.updatedAt ?? -1),
  );
  const normalizedBase = normalizeModelRef(baseModel);
  if (!normalizedBase) return sorted[0] ?? null;

  const exact = sorted.find(
    (lora) => normalizeModelRef(lora.baseModel) === normalizedBase,
  );
  if (exact) return exact;

  const partial = sorted.find((lora) => {
    const normalizedLoraBase = normalizeModelRef(lora.baseModel);
    if (!normalizedLoraBase) return false;
    return (
      normalizedLoraBase.includes(normalizedBase) ||
      normalizedBase.includes(normalizedLoraBase)
    );
  });
  return partial ?? sorted[0] ?? null;
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

const ARTIFACT_PANEL_DEFAULT_SIZE = "38%";
const ARTIFACT_PANEL_TRANSITION_MS = 260;
const ARTIFACT_SURFACE_POP_DELAY_MS = 150;

const SingleContent = memo(function SingleContent({
  threadId,
  artifact,
  artifactSurface,
  onCloseArtifact,
}: {
  threadId?: string;
  artifact?: ChatArtifact | null;
  artifactSurface: ChatArtifactSurface;
  onCloseArtifact: () => void;
}): ReactElement {
  const openArtifact = useChatArtifactsStore((state) => state.openArtifact);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const isMobile = useIsMobile();
  const chatActive = useChatActive();
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
  // Without a URL threadId the artifact must belong to the active thread.
  const showArtifactPanel = !showResearchPanel && Boolean(
    artifact &&
      artifactSurface === "panel" &&
      (threadId
        ? !artifact.threadId || artifact.threadId === threadId
        : Boolean(artifact.threadId && artifact.threadId === activeThreadId)),
  );
  const showContextPanel = showResearchPanel || showArtifactPanel;

  const artifactLayoutActive = showContextPanel || isArtifactPanelLayoutActive;
  const artifactPanelSettledOpen =
    showContextPanel &&
    isArtifactPanelLayoutActive &&
    !isArtifactLayoutAnimating;

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
        panel.resize(showContextPanel ? ARTIFACT_PANEL_DEFAULT_SIZE : "0%");
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
    onCloseArtifact();
    useChatRuntimeStore.getState().setSettingsPanelOpen(false);
  }, [researchMatchesThread, onCloseArtifact]);

  const threadPane = (
    <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden">
      <Thread hideWelcome={Boolean(threadId)} targetThreadId={threadId} />
    </div>
  );

  return (
    <>
      <ResizablePanelGroup
        orientation="horizontal"
        data-artifact-layout-animating={
          isArtifactLayoutAnimating ? "true" : "false"
        }
        className="chat-artifact-split min-h-0 min-w-0 flex-1 basis-0 overflow-hidden"
      >
        <ResizablePanel
          id="chat-thread"
          defaultSize="100%"
          minSize={artifactLayoutActive ? "42%" : "100%"}
          className="h-full min-h-0 min-w-0 overflow-hidden"
        >
          <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden">
            {threadPane}
          </div>
        </ResizablePanel>
        <ResizableHandle
          withHandle={false}
          className={cn(
            "relative z-30 -ml-1 -mr-4 w-5 bg-transparent transition-[width,margin] duration-[260ms] ease-[var(--ease-out-cubic)] hover:bg-transparent hover:shadow-none active:bg-transparent active:shadow-none focus-visible:bg-transparent focus-visible:shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:outline-none",
            !artifactLayoutActive &&
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
            showResearchPanel
              ? "58%"
              : artifactLayoutActive
                ? "58%"
                : "0%"
          }
          collapsible={showArtifactPanel}
          collapsedSize="0%"
          className={cn(
            "h-full min-h-0 min-w-0 overflow-visible",
            !showContextPanel && "pointer-events-none",
          )}
        >
          <div
            data-artifact-surface-visible={
              isArtifactSurfaceVisible ? "true" : "false"
            }
            className={cn(
              "chat-artifact-pop-surface flex h-full min-h-0 min-w-0 flex-col overflow-visible",
              showResearchPanel && "border-l border-border/70",
            )}
          >
             {showResearchPanel && openResearchRunId ? (
               <ResearchActivityPanel
                 key={openResearchRunId}
                 runId={openResearchRunId}
                 onClose={closeResearchPanel}
               />
             ) : showArtifactPanel && artifact ? (
              <ArtifactSurface
                artifact={artifact}
                variant="panel"
                onClose={onCloseArtifact}
                onOpenFullscreen={() =>
                  openArtifact(artifact, { surface: "overlay" })
                }
              />
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

/** True when the loaded checkpoint is a LoRA, so a base-vs-fine-tuned compare can use the fast
 *  simultaneous adapter-toggle path. */
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

/** One column in the compare layout: a ChatRuntimeProvider and a Thread with hideComposer. Each
 *  pane is `flex-1 basis-0 min-h-0 min-w-0` so panes share space equally and the inner
 *  viewport scrolls instead of spilling. */
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
      <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden [&_.aui-thread-viewport]:px-6 lg:[&_.aui-thread-viewport]:px-10">
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
      window.dispatchEvent(new Event("unsloth:app-shell-ready"));
    },
    [pairId],
  );
}

/** Shared shell for both compare variants: a flex column with the two panes as siblings and the
 *  shared composer docked at the bottom. Flex, not grid: grid rows with 1fr triggered resize
 *  thrash in assistant-ui's autoscroll on breakpoint crossings. */
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
          className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col pt-[var(--studio-content-top-inset,0px)] md:flex-row"
        >
          {children}
        </div>
        <div className="shrink-0 bg-background pl-5 pr-5 md:pr-[30px] pb-2 pt-1">
          <div className="mx-auto w-full max-w-[48rem]">{composer}</div>
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

  // Global on purpose: a first compare run starts before either thread exists, so there is no pair
  // id to scope BY. The gate exists at all because these ids feed ComparePane's
  // `initialThreadId`, so learning them mid-run points ThreadAutoSwitch at a live thread.
  const anyRunning = useChatRuntimeStore(
    (s) => Object.keys(s.localRunByThreadId).length > 0,
  );
  // ...but only RE-lists wait. The shared provider (#8908) keeps a base chat's run alive across
  // the switch into compare, so `anyRunning` is true on arrival for an unrelated reason, and
  // gating the FIRST list on it left an existing compare on blank runtimes.
  const listedPairRef = useRef<string | null>(null);

  useEffect(() => {
    if (anyRunning && listedPairRef.current === pairId) return;
    listedPairRef.current = pairId;
    let isActive = true;
    setThreadsSettled(false);
    listStoredChatThreads({ pairId })
      .then((threads) => {
        if (!isActive) return;
        // No model1/model2 fallback: useCompareVariant never routes a generalized pair here, so adopting
        // one could only mislabel it.
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
          borderClassName="border-t border-border/60 md:border-t-0 md:border-l"
          header={
            <div className="shrink-0 px-3 py-1.5 text-start md:text-end md:pr-[calc(4rem+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]">
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

/** Per-pane header with the model selector, aligned to the global topbar height. Left pane
 *  reserves room for the mobile sidebar trigger; right pane for the settings button. */
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
}): ReactElement {
  // Controlled so the body-portaled popover cannot linger over another tab off-route.
  const active = useChatActive();
  const [selectorOpen, setSelectorOpen] = useState(false);

  const { pinned } = useSidebar();
  return (
    <div
      className={cn(
        "pointer-events-none relative z-40 flex h-[48px] shrink-0 items-start gap-2 bg-background pt-[var(--studio-chat-header-padding-top,11px)]",
        side === "left"
          ? pinned
            ? "pl-12 pr-3 md:pl-2"
            : "pl-12 pr-3 md:pl-[calc(0.5rem+max(0px,var(--studio-mac-traffic-light-inset,0px)-var(--sidebar-width-icon,3rem)))]"
          : "pl-3 pr-[calc(3rem+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]",
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
    </div>
  );
}

/** General path: any two models, sequential load → generate. */
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
  const listedPairRef = useRef<string | null>(null);
  const [model1, setModel1] = useState<CompareModelSelection>({
    id: globalCheckpoint || "",
    isLora: false,
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
    if (anyRunning && listedPairRef.current === pairId) return;
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
  }, [pairId, anyRunning]);

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
            model1ThreadId={model1ThreadId}
            model2ThreadId={model2ThreadId}
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
          borderClassName="border-t border-sidebar-border md:border-t-0 md:border-l"
          header={
            <GeneralCompareHeader
              side="right"
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

// Unique thread nonce; falls back off crypto.randomUUID for non-secure (HTTP LAN) contexts.
function createThreadNonce(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

// Chat export formats, mirroring the sidebar chat menu.
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
  const exports = await import("./prompt-storage/prompt-storage-dialog");
  if (format === "raw-jsonl") return exports.exportConversationRawJsonl(threadId);
  if (format === "messages-jsonl")
    return exports.exportConversationMessagesJsonl(threadId);
  if (format === "csv") return exports.exportConversationCsv(threadId);
  if (format === CONVERSATION_MARKDOWN_FORMAT)
    return exports.exportConversationMarkdown(threadId);
  if (format === "sharegpt-jsonl") return exports.exportConversationShareGPT(threadId);
  // Was a fallthrough return, so an unhandled format silently exported ShareGPT.
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
  const { saveChatItemAsProjectSource } = await import(
    "./prompt-storage/prompt-storage-dialog"
  );
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
  // #9251 holds the reload shell until the landing can be drawn. Its provider is hoisted above the
  // view switch now (#8908), so the owner of that one reports readiness down.
  runtimeReady: boolean;
}): ReactElement {
  const navigate = useNavigate();
  // Gates body-portaled surfaces so they cannot linger or act while the landing is off-route.
  const active = useChatActive();
  const wasActiveRef = useRef(active);
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  // Captured in render, not an effect: the shared provider's ThreadNewChatSwitch is an earlier
  // sibling, so its mount effect blanks activeThreadId before an effect here runs.
  const [initialActiveThreadId] = useState(
    () => useChatRuntimeStore.getState().activeThreadId,
  );
  // Land on Sources when the project was just created with dropped files.
  const [projectTab, setProjectTab] = useState<"chats" | "sources">(() =>
    hasProjectSourcesPending(projectId) ? "sources" : "chats",
  );
  // Drop the marker once committed: React may replay the initializer above.
  useEffect(() => {
    consumeProjectSourcesPending(projectId);
  }, [projectId]);
  const [pendingNewThreadId, setPendingNewThreadId] = useState<string | null>(
    null,
  );
  const [previews, setPreviews] = useState<
    Record<string, { snippet: string; date: string }>
  >({});
  const reloadReadySent = useRef(false);
  // Inline rename, mirroring the sidebar recent-row UX. Reuses the projectId-agnostic
  // renameChatItem so behavior matches the sidebar.
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  // Skips the input's blur-commit when Enter/Escape already handled it.
  const skipRenameBlurRef = useRef(false);
  // Optimistic title shown until the debounced sidebar refresh catches up, so the old name does not flash back in.
  const [pendingRename, setPendingRename] = useState<{
    id: string;
    title: string;
  } | null>(null);

  // Project-level options (the header kebab menu).
  const pinnedProjectIds = usePinnedProjectsStore((s) => s.pinnedIds);
  const togglePinProject = usePinnedProjectsStore((s) => s.togglePin);
  const projectPinned = pinnedProjectIds.includes(projectId);
  const [renamingProject, setRenamingProject] = useState(false);
  const [projectNameDraft, setProjectNameDraft] = useState("");
  const [deletingProject, setDeletingProject] = useState(false);

  async function handleProjectExport(
    format: ProjectChatExportFormat,
  ): Promise<void> {
    try {
      const threads = await listStoredChatThreads({
        projectId,
        includeArchived: false,
      });
      const ids = [...new Set(threads.map((t) => t.id))];
      for (const id of ids) await exportProjectConversation(id, format);
    } catch (error) {
      if (!isDownloadCancelled(error)) toast.error("Export failed.");
    }
  }

  async function commitProjectRename(): Promise<void> {
    const name = projectNameDraft.trim();
    setRenamingProject(false);
    if (!name || name === projectName) return;
    try {
      await renameChatProject(projectId, name);
    } catch (err) {
      toast.error("Failed to rename project", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function commitProjectDelete(): Promise<void> {
    setDeletingProject(false);
    try {
      await deleteChatProject(projectId);
      // Refresh chat history so the project's now-deleted chats do not linger in the sidebar, matching
      // the sidebar delete path.
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

  // Full chat actions, matching the sidebar chat menu.
  const { projects } = useChatProjects();
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
  // Preselected from the preference, so the dialog shows what is about to happen and can still be
  // turned off for this one chat.
  const [deleteFilesOnDelete, setDeleteFilesOnDelete] = useState(false);

  // Landing has no active thread selected, so the onView callback is a no-op; the items list
  // refreshes itself once storage emits its update.
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
      // No confirmation to preselect, so the preference is the answer.
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

  // No composer ever records under this, so passing it refuses the adoption
  // (adoptPendingProjectAttachmentTarget only adopts on an exact claim match).
  const NO_SUCH_CLAIM = -1;

  // The claim the composer on screen recorded its attach choice under: every fresh composer shares
  // one pending key, so only the claim tells them apart.
  const pendingTargetClaimRef = useRef<{
    nonce: string;
    claim: number;
  } | null>(null);
  useEffect(() => {
    return useChatRuntimeStore.subscribe((state) => {
      const pending =
        state.projectAttachmentTargetByThread[PENDING_CHAT_ATTACHMENT_KEY];
      if (pending === undefined) return;
      // By claim, not by value: picking the same destination twice rewrites the same string under a
      // new claim, and skipping it reads as somebody else's.
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
        // ...unless it was deleted while another view held the screen. Nothing else clears this id, so
        // restoring a tombstoned thread would put a conversation storage no longer has back on
        // screen. Fall through to the rotate below.
        if (!isChatThreadDeleted(pendingNewThreadId)) {
          useChatRuntimeStore.getState().setActiveThreadId(pendingNewThreadId);
          return;
        }
      }
      // Leaving a created chat for a new one: rotate the nonce so the runtime switches to a fresh
      // thread instead of appending to the old chat.
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
    // Hand the composer's attach choice to the chat it just created: setting this swaps
    // ProjectComposer for Thread, so the bar holding the choice unmounts and its cleanup drops
    // it. Its own choice only, or a later send would consume another composer's pick.
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
              // A paste-only message carries its text in the attachment, so the row would otherwise be blank.
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
    window.dispatchEvent(new Event("unsloth:app-shell-ready"));
  }, [dataLoaded, items, previews, runtimeReady]);

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
          {/* Slightly narrower than the composer max; every block shares this. */}
          <div className="mx-auto flex w-full max-w-[44rem] flex-col pt-[120px] pb-14">
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
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <button
                    type="button"
                    aria-label="Project options"
                    className="inline-flex size-9 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring data-[state=open]:bg-muted data-[state=open]:text-foreground"
                  >
                    <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  side="bottom"
                  align="end"
                  sideOffset={6}
                  className="unsloth-plus-menu menu-flat-destructive w-52"
                >
                  <DropdownMenuItem
                    onSelect={() => {
                      setProjectNameDraft(projectName);
                      setRenamingProject(true);
                    }}
                  >
                    <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                    <span>Rename project</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => togglePinProject(projectId)}>
                    <HugeiconsIcon icon={projectPinned ? PinOffIcon : PinIcon} strokeWidth={1.75} className="size-icon" />
                    <span>{projectPinned ? "Unpin project" : "Pin project"}</span>
                  </DropdownMenuItem>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger>
                      <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon" />
                      <span>Export</span>
                    </DropdownMenuSubTrigger>
                    <DropdownMenuSubContent className="unsloth-plus-menu w-48">
                      {PROJECT_CHAT_EXPORT_OPTIONS.map(({ label, format }) => (
                        <DropdownMenuItem
                          key={format}
                          onSelect={() => void handleProjectExport(format)}
                        >
                          {label}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    variant="destructive"
                    onSelect={() => setDeletingProject(true)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                    <span>Delete project</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
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
                        className="flex min-h-[58px] w-full items-center rounded-full px-4 py-2"
                      >
                        <div className="min-w-0 flex-1">
                          <input
                            autoFocus
                            value={renameDraft}
                            onChange={(event) =>
                              setRenameDraft(event.target.value)
                            }
                            onKeyDown={(event) => {
                              // Ignore keydowns fired mid-IME-composition (CJK) so a candidate-confirming
                              // Enter does not commit the rename. Guarded before the key branch so Escape
                              // is covered too (isComposing on WebKit, 229 on Chromium).
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
                            className="w-full border-0 bg-transparent text-ui-15 font-semibold leading-5 text-foreground outline-none"
                          />
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div
                      key={`${item.type}:${item.id}`}
                      className="group relative flex min-h-[58px] w-full items-center rounded-full transition-colors hover:bg-nav-surface-hover has-[[data-state=open]]:bg-nav-surface-hover"
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
                        className="flex min-h-[58px] min-w-0 flex-1 items-center gap-4 rounded-full px-4 py-2 text-left"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-ui-15 font-semibold leading-5 text-foreground">
                            {displayTitle}
                          </div>
                        </div>
                        <span className="shrink-0 text-ui-14 text-muted-foreground transition-opacity max-md:opacity-0 pointer-coarse:opacity-0 group-hover:opacity-0 group-has-[[data-state=open]]:opacity-0">
                          {preview?.date ??
                            formatProjectChatDate(item.createdAt)}
                        </span>
                      </button>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            type="button"
                            onClick={(event) => event.stopPropagation()}
                            aria-label="Chat options"
                            className="absolute right-3 top-1/2 inline-flex size-8 -translate-y-1/2 cursor-pointer items-center justify-center rounded-full text-muted-foreground outline-none transition-opacity hover:bg-foreground/10 md:pointer-fine:opacity-0 md:pointer-fine:pointer-events-none focus-visible:opacity-100 focus-visible:pointer-events-auto group-hover:opacity-100 group-hover:pointer-events-auto data-[state=open]:opacity-100 data-[state=open]:pointer-events-auto"
                          >
                            <HugeiconsIcon
                              icon={MoreVerticalIcon}
                              strokeWidth={1.75}
                              className="size-icon"
                            />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          side="bottom"
                          align="end"
                          sideOffset={4}
                          className="unsloth-plus-menu menu-flat-destructive w-56"
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
                              <span>Move to project</span>
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
                              icon={BookOpen01Icon}
                              strokeWidth={1.75}
                              className="size-icon"
                            />
                            <span>Save to project sources</span>
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
                        </DropdownMenuContent>
                      </DropdownMenu>
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
      <Dialog
        open={active && renamingProject}
        onOpenChange={(open) => {
          if (!open) setRenamingProject(false);
        }}
      >
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Rename project</DialogTitle>
          </DialogHeader>
          <Input
            value={projectNameDraft}
            onChange={(e) => setProjectNameDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void commitProjectRename();
              }
            }}
            autoFocus={true}
            maxLength={120}
            placeholder="Project name"
            aria-label="Project name"
            className="focus-visible:border-input focus-visible:ring-0"
          />
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button type="button" variant="ghost" onClick={() => setRenamingProject(false)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void commitProjectRename()}
              disabled={
                !projectNameDraft.trim() || projectNameDraft.trim() === projectName
              }
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => void commitProjectDelete()}>
              Delete
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

// `search` comes from RootLayout (not useSearch) so ChatPage stays mounted off-route, frozen to
// the last /chat search. `active` is false off-route: close portaled surfaces and stop
// route-specific listeners.
export function ChatPage({
  search,
  active,
}: { search: ChatSearch; active: boolean }): ReactElement {
  const navigate = useNavigate();

  const settingsOpen = useChatRuntimeStore((s) => s.settingsPanelOpen);
  const setSettingsOpen = useChatRuntimeStore((s) => s.setSettingsPanelOpen);
  const incognito = useChatRuntimeStore((s) => s.incognito);
  const setIncognito = useChatRuntimeStore((s) => s.setIncognito);
  const incognitoLabel = incognito
    ? "Turn off temporary chat"
    : "Turn on temporary chat";
  const toggleIncognito = useCallback(() => {
    const store = useChatRuntimeStore.getState();
    const wasIncognito = store.incognito;
    store.setIncognito(!store.incognito);
    // On an empty scratch chat there is nothing to abandon, so flip in place: navigating would
    // remount the thread and bounce the composer. Otherwise start a clean chat so the temporary
    // session cannot inherit or leave behind a persisted thread.
    const onEmptyScratchChat =
      !search.thread &&
      !search.compare &&
      !search.project &&
      store.activeThreadId == null;
    if (wasIncognito) {
      requestTemporaryPromptQueueStop();
    }
    if (onEmptyScratchChat) return;
    // setActiveThreadId already clears contextUsage.
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
    // Skip while off-route: ChatPage stays mounted, and toast+navigate here would yank the user back
    // to chat from whatever tab they are on.
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
  // Controlled, so the chord can open the switcher and not just its trigger.
  const [projectPickerOpen, setProjectPickerOpen] = useState(false);
  const [modelSelectorLocked, setModelSelectorLocked] = useState(false);
  const viewBeforeCompareRef = useRef<ChatSearch | null>(null);
  // Latest non-compare view, so exiting compare can restore it even when compare was opened from a
  // path that does not set viewBeforeCompareRef.
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
  const loadedContextLength = useChatRuntimeStore(
    (state) => state.loadedContextLength,
  );
  const nativeContextLength = useChatRuntimeStore(
    (state) => state.nativeContextLength,
  );
  const contextUsage = useChatRuntimeStore((state) => state.contextUsage);
  const loadedIsGguf = useChatRuntimeStore((state) => state.loadedIsGguf);
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
  const resetArtifacts = useChatArtifactsStore((state) => state.resetArtifacts);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const latestResearchRunId = useResearchRunStore((state) =>
    activeThreadId ? state.latestRunByThreadId[activeThreadId] : undefined,
  );
  // Status, not the run: this subscribes in ChatPage itself, so a run selector re-rendered the
  // whole page on every streamed research delta.
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
  // A ?new=<nonce> chat has no thread in the URL before or after its first send, and for the first
  // render the store still holds the PREVIOUS chat's id until ThreadNewChatSwitch blanks it,
  // so latch on having seen it blanked for this nonce.
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
  const modelOperationInProgress = useChatRuntimeStore(
    (state) => state.modelLoading,
  );
  const {
    refresh,
    selectModel,
    ejectModel,
    cancelLoading,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  } = useChatModelRuntime();
  const prevConnectionsEnabledRef = useRef(connectionsEnabled);
  useEffect(() => {
    const turnedOff = prevConnectionsEnabledRef.current && !connectionsEnabled;
    if (!connectionsEnabled && isExternalModelId(inferenceParams.checkpoint)) {
      resetArtifacts();
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
    resetArtifacts,
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
      const resolved = resolveInitialConfig(selection.id, selection.ggufVariant);
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
    const baseCapabilities = getProviderCapabilities(provider?.providerType);
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
      },
    );
    const state = useChatRuntimeStore.getState();
    const preferredEffort = state.reasoningEffort;
    const effortLevels = reasoningCaps.reasoningEffortLevels;
    const clampedEffort = clampReasoningEffortToLevels(
      preferredEffort,
      effortLevels,
    );
    // Per-provider default effort: Anthropic gets the highest level, since Claude's adaptive thinking
    // adjusts cost per turn; OpenAI gets "high"; everyone else "medium". Overridable via Think.
    const isAnthropic = provider?.providerType === "anthropic";
    const isOpenAI = provider?.providerType === "openai";
    const anthropicTopEffort = effortLevels.includes("xhigh")
      ? "xhigh"
      : effortLevels.includes("high")
        ? "high"
        : clampedEffort;
    const openaiDefaultEffort = effortLevels.includes("high")
      ? "high"
      : effortLevels.includes("medium")
        ? "medium"
        : clampedEffort;
    const nextReasoningEffort = reasoningCaps.supportsReasoning
      ? isAnthropic
        ? anthropicTopEffort
        : isOpenAI
          ? openaiDefaultEffort
          : effortLevels.includes("medium")
            ? "medium"
            : clampedEffort
      : state.reasoningEffort;
    const supportsBuiltinWebSearch = providerSupportsBuiltinWebSearch(
      provider?.providerType,
      selection.modelId,
      provider?.baseUrl,
    );
    const supportsBuiltinCodeExecution = providerSupportsBuiltinCodeExecution(
      provider?.providerType,
      selection.modelId,
      provider?.baseUrl,
    );
    const supportsBuiltinImageGeneration =
      providerSupportsBuiltinImageGeneration(
        provider?.providerType,
        selection.modelId,
        provider?.baseUrl,
      );
    const supportsBuiltinWebFetch = providerSupportsBuiltinWebFetch(
      provider?.providerType,
    );
    // Kimi's k2.6/k2.5 default to thinking enabled server-side, so the Think pill comes up clicked.
    // Search stays off; the composer's mutual-exclusion handlers flip the two.
    // Per https://platform.kimi.ai/docs/models.
    const isKimi = provider?.providerType === "kimi";
    // Web search on by default only for Anthropic and OpenAI, both with structured citations.
    // OpenRouter and Kimi work on opt-in but are less reliable.
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
    // Unsloth runs Search and Code itself for any provider that advertises the capability, so a
    // self-hosted connection has no hosted builtin to key off. Keying the pill state on the
    // hosted flags alone discarded the saved preference on every reload and sent
    // enable_tools: false.
    const supportsStudioToolsHere =
      providerModelSupportsStudioTools(
        provider?.providerType,
        selection.modelId,
      ) === true;
    const canSearch = supportsBuiltinWebSearch || supportsStudioToolsHere;
    // Read out of the placement rule, not off the Unsloth-tools flag: a model on a sandbox-owning
    // provider that cannot use it runs nothing either way.
    const canRunCode = codeToolCanRun({
      hostedCodeExecutionForThisTurn: supportsBuiltinCodeExecution,
      providerHostsCodeExecution: providerHostsCodeExecution(provider?.providerType),
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
      supportsPreserveThinking: false,
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
      // Default Fetch off (Anthropic bills per fetch); deliberate opt-in.
      webFetchToolsEnabled: supportsBuiltinWebFetch
        ? (storedWebFetchToolsEnabled ?? false)
        : false,
    });
    // Reruns once settings hydrate: this normalization reads the stored pills and clamps them to the
    // model, and hydration refreshes what it reads, so it has to be applied last.
  }, [externalProvidersForChat, inferenceParams.checkpoint, settingsHydrated]);
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

  // Temporary chat only applies to a fresh single-view chat, so exit incognito on anything else
  // (compare, a project, an existing thread) rather than stranding the toggle.
  useEffect(() => {
    const onFreshSingleChat = view.mode === "single" && !view.threadId;
    if (incognito && !onFreshSingleChat) {
      setIncognito(false);
    }
  }, [view, incognito, setIncognito]);

  const selectedArtifact = useSelectedChatArtifact();
  const artifactSurface = useChatArtifactsStore((state) => state.surface);
  const closeArtifactSurface = useChatArtifactsStore(
    (state) => state.closeArtifactSurface,
  );
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

  // Compare replaces the shared provider on screen, so the base view is kept mounted behind it:
  // unmounting runs useLocalRuntime's detach(), the backend cancels on the disconnect, and a
  // project chat's run would die. Frozen to the last non-compare view.
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

  // #9251's reload signal, taken here because the provider is hoisted. Stored as the project it
  // belongs to, not a boolean: the hoisted provider is not remounted when the project changes,
  // so a flag would release the next landing's shell early.
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

  useEffect(() => {
    clearAutoOpenedArtifacts();
    closeArtifactSurface();
  }, [artifactViewKey, closeArtifactSurface]);

  useEffect(() => {
    if (view.mode !== "single") return;
    if (view.threadId || !selectedArtifact) return;
    // Close any canvas that does not belong to the active thread.
    if (
      selectedArtifact.threadId &&
      selectedArtifact.threadId === activeThreadId
    )
      return;
    closeArtifactSurface();
  }, [activeThreadId, closeArtifactSurface, selectedArtifact, view]);

  const hasActiveModel = Boolean(inferenceParams.checkpoint);
  const chatContextKey = `${view.mode}|${activeThreadId ?? ""}|${search.new ?? ""}|${search.project ?? ""}`;
  const [pendingHubAutoLoad, setPendingHubAutoLoad] =
    useState<PendingHubAutoLoad | null>(null);
  const stageOrLoad = useCallback(
    async (selection: SelectedModelInput) => {
      const store = useChatRuntimeStore.getState();
      const wantManagerStaging = wantsDownloadManagerStaging(selection);

      if (wantManagerStaging) {
        // Uncached picks return below and do not reach selectModel until completion.
        // Invalidate the previous model's notice at the actual picker boundary.
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
        } else if (wantManagerStaging) {
          const outcome = await downloadManager.requestStart({
            kind: DOWNLOAD_KIND.MODEL,
            repoId: selection.id,
            variant: selection.ggufVariant ?? null,
            expectedBytes: selection.expectedBytes ?? 0,
            // Handed over, not raised here, so one start makes one toast.
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
        } else {
          toast.info("Another model is already loading", {
            description: "Wait for it to finish or cancel it first.",
          });
        }
        return;
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
  // The pending auto-load's job, for the context-change effect below. Written from an effect, never during render.
  const pendingAutoLoadKeyRef = useRef<string | null>(null);
  // The live context, so a start still in flight can check it is still on screen.
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
        // Notice-only: #9663 removed this surface's own toast, so it must not return on an HTTP start or
        // once the three notices are spent.
        callerToast: {
          title: "Downloading model",
          description: "It'll load automatically once the download finishes.",
          noticeOnly: true,
          // The cleanup below only reaches a toast that already exists; a raise still in flight would
          // promise an auto-load onComplete then refuses.
          stillValid: () => chatContextKeyRef.current === pending.contextKey,
        },
      });
      if (!active) return;
      if (outcome === "started") {
        // No toast here, and none from the manager unless the notice folds the sentence in. The
        // auto-load runs from onComplete.
        return;
      }
      if (outcome === "conflict") {
        // Keep pendingHubAutoLoad bound so this surface's cleanup does not wipe the conflict requestStart
        // just recorded; resolving it from the Hub completes the download and onComplete auto-loads.
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
      // Another model was picked, so this one's completion loads nothing. A no-op once the download
      // finished, which dismisses the same id.
      dismissStartToast(pendingKey);
    };
  }, [pendingHubAutoLoad]);
  // Switching thread or project keeps the pathname and pendingHubAutoLoad, so neither sweep above
  // runs, yet onComplete refuses to load into a different contextKey.
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
  // Dropped documents go to the thread bar, which owns the RAG upload and can materialize a thread
  // id for a chat that has not been sent to yet.
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
    // Compare used to disable this outright, so a drop there vanished with no overlay and no message
    // (#9036). Keep listening and refuse out loud.
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
      // Partial: the switch-back carries only what the resolver cannot recover.
      meta?: Partial<ModelSelectorChangeMeta>,
    ) => {
      const store = useChatRuntimeStore.getState();
      const currentCheckpoint = store.params.checkpoint;
      const currentVariant = store.activeGgufVariant;
      if (!value) return;
      setPendingHubAutoLoad(null);
      const isSameLoadedModel =
        value === currentCheckpoint &&
        (meta?.ggufVariant ?? null) === (currentVariant ?? null);
      if (isSameLoadedModel && !meta?.forceReload) {
        return;
      }
      if (meta?.source === "external" || isExternalModelId(value)) {
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
          },
        );
        const preferredEffort = store.reasoningEffort;
        const effortLevels = reasoningCaps.reasoningEffortLevels;
        const clampedEffort = clampReasoningEffortToLevels(
          preferredEffort,
          effortLevels,
        );
        // Same per-provider default policy as the useEffect above: Anthropic highest level, OpenAI
        // "high", everyone else "medium".
        const isAnthropic = selectedProvider?.providerType === "anthropic";
        const isOpenAI = selectedProvider?.providerType === "openai";
        const anthropicTopEffort = effortLevels.includes("xhigh")
          ? "xhigh"
          : effortLevels.includes("high")
            ? "high"
            : clampedEffort;
        const openaiDefaultEffort = effortLevels.includes("high")
          ? "high"
          : effortLevels.includes("medium")
            ? "medium"
            : clampedEffort;
        const nextReasoningEffort = reasoningCaps.supportsReasoning
          ? isAnthropic
            ? anthropicTopEffort
            : isOpenAI
              ? openaiDefaultEffort
              : effortLevels.includes("medium")
                ? "medium"
                : clampedEffort
          : store.reasoningEffort;
        // Clear any cached router-picked openrouter/free model unless staying on openrouter/free, else
        // the chip keeps a stale ":<chosen>" suffix.
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
          );
        const supportsBuiltinImageGeneration =
          providerSupportsBuiltinImageGeneration(
            selectedProvider?.providerType,
            selectedExternal?.modelId,
            selectedProvider?.baseUrl,
          );
        const supportsBuiltinWebFetch = providerSupportsBuiltinWebFetch(
          selectedProvider?.providerType,
        );
        // See sibling useEffect: Kimi's k2.x default to thinking enabled. Search stays off; the
        // composer's mutual exclusion flips them.
        const isKimi = selectedProvider?.providerType === "kimi";
        // Mirror of sibling useEffect: Anthropic/OpenAI get Search on by default; others stay off.
        const searchOnByDefault =
          supportsBuiltinWebSearch &&
          (selectedProvider?.providerType === "anthropic" ||
            selectedProvider?.providerType === "openai");
        // mirror of the sibling effect: the open chat's own pills win over the global ones.
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
        // Same rule as the selection handler above: a self-hosted connection has no hosted builtin, so
        // keying the pills on those flags threw away the saved preference.
        const supportsStudioToolsHere =
          providerModelSupportsStudioTools(
            selectedProvider?.providerType,
            selectedExternal?.modelId,
          ) === true;
        const canSearch = supportsBuiltinWebSearch || supportsStudioToolsHere;
        // Same placement rule as the selection handler above.
        const canRunCode = codeToolCanRun({
          hostedCodeExecutionForThisTurn: supportsBuiltinCodeExecution,
          providerHostsCodeExecution: providerHostsCodeExecution(
            selectedProvider?.providerType,
          ),
          supportsStudioTools: supportsStudioToolsHere,
        });
        const nextToolsEnabled = canSearch
          ? isKimi
            ? false
            : (storedToolsEnabled ?? searchOnByDefault)
          : false;
        useChatRuntimeStore.setState({
          activeGgufVariant: null,
          ...loadedContextFields(null),
          activeNativePathToken: null,
          activeNativePathExpiresAtMs: null,
          // Clear previous-model counters, else the relaxed external-provider render gate shows stale
          // stats. The per-thread copies go too, so a switch back cannot re-apply.
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
          supportsPreserveThinking: false,
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
        });
        return;
      }
      // Local model picked: drop any cached openrouter/free chosen model.
      useChatRuntimeStore.setState({ lastOpenRouterChosenModel: null });
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
                hasImage && targetModel?.isVision === false;
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
          isGguf: meta?.isGguf,
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
      modelsFromStore,
      stageOrLoad,
      view,
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
      // A file-picked GGUF is reachable only via its native path token, which the desktop host prunes
      // after a TTL. Reusing an expired token fails with an opaque error, so prompt to re-select.
      if (nativeToken && nativeExpiry != null && Date.now() >= nativeExpiry) {
        toast.error("This local model file's access has expired.", {
          description: "Re-select the model file to reload it.",
        });
        return;
      }
      handleCheckpointChange(checkpoint, {
        source: "local",
        isLora: activeModelIsLora,
        // The checkpoint is the id, so a pinned model reloads from that same snapshot.
        loadId: activeLoadId,
        ggufVariant: activeGgufVariant ?? undefined,
        // Without the native token the reload validates the display label as a repo and fails.
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
  const handleEject = useCallback(() => {
    void (async () => {
      if (await ejectModel()) {
        resetArtifacts();
      }
    })();
  }, [ejectModel, resetArtifacts]);

  // Pins the picker open so a stray click cannot dismiss the step under it. Tour steps only: the
  // effect below shuts anything left pinned once the tour is gone.
  const openModelSelector = useCallback(() => {
    setModelSelectorLocked(true);
    setModelSelectorOpen(true);
  }, []);

  const closeModelSelector = useCallback(() => {
    setModelSelectorLocked(false);
    setModelSelectorOpen(false);
  }, []);

  /** The chord's opener: no pin, so the picker stays dismissible. */
  const toggleModelSelector = useCallback(() => {
    // Pinned means a tour step is standing on it, and that step owns it.
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

  // Both controls are the header's, and the header drops them in Compare, where each pane carries
  // its own picker. Without the check the chord would toggle state nothing renders.
  const headerPickersShown = active && view.mode !== "compare";
  // This page stays mounted under a dialog, so `enabled` still says yes while the header is inert;
  // without a press-time check the chord opens a popover on the covered surface.
  // Backgrounded, not "not in the foreground": an unrendered layout is not a covered one.
  const chatCovered = () => isSurfaceBackgrounded(COMPOSER_INPUT_SELECTOR);
  useShortcut(
    "openModelPicker",
    () => {
      if (chatCovered()) return;
      toggleModelSelector();
    },
    { enabled: headerPickersShown },
  );
  // The same condition the switcher renders by, so the chord cannot open a control that is not there.
  const projectSwitcherShown = headerPickersShown && Boolean(currentProjectId);
  useShortcut(
    "openProjectPicker",
    () => {
      if (chatCovered()) return;
      setProjectPickerOpen(true);
    },
    { enabled: projectSwitcherShown },
  );
  // A picker left open would come back on the next visit as a ghost. Off-route is one way to leave
  // it (this page stays mounted), and so is entering Compare or a standalone chat taking the
  // project away. Adjusted during render, as React prescribes for derived state.
  if (!projectSwitcherShown && projectPickerOpen) {
    setProjectPickerOpen(false);
  }

  /** Step the effort level, clamped at both ends unless we are cycling. */
  const shiftReasoningEffort = useCallback(
    (delta: number, wrap: boolean) => {
      const state = useChatRuntimeStore.getState();
      const levels = state.reasoningEffortLevels;
      // Levels stay populated for an enable_thinking model, whose request path drops the effort. Same
      // test as the composer's effort menu.
      const isEffort =
        state.reasoningStyle === "reasoning_effort" ||
        state.reasoningStyle === "enable_thinking_effort";
      if (!state.supportsReasoning || !isEffort || levels.length === 0) {
        toast.info("This model has no reasoning effort setting");
        return;
      }
      const current = levels.indexOf(state.reasoningEffort);
      // Loading a model that drops the level in force leaves the effort set to one that is gone, and
      // indexOf gives -1, so the first press picks the lowest level offered.
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
    // Prefer the explicit save; fall back to the last non-compare view so the composer + menu path
    // also returns where the user started.
    const saved = viewBeforeCompareRef.current ?? lastNonCompareViewRef.current;
    // No saved view (compare opened by direct URL); fall back to a fresh chat.
    if (!saved) {
      navigate({ to: "/chat" });
      return;
    }
    viewBeforeCompareRef.current = null;
    navigate({ to: "/chat", search: saved });
    // Restore usage from the last assistant message, only if it matches the active checkpoint, else
    // the relaxed render gate shows stale stats.
    const threadId =
      saved.thread ?? useChatRuntimeStore.getState().activeThreadId;
    if (threadId) {
      void listStoredChatMessages(threadId)
        .then(
          (messages) =>
            [...messages].sort((a, b) => b.createdAt - a.createdAt)[0],
        )
        .then((msg) => {
          const metadata = msg?.metadata as Record<string, unknown> | undefined;
          const usage = metadata?.contextUsage as ReturnType<
            typeof useChatRuntimeStore.getState
          >["contextUsage"];
          if (!usage) return;
          const store = useChatRuntimeStore.getState();
          const activeCheckpoint = store.params.checkpoint;
          const usageModelId = (usage as { modelId?: unknown }).modelId;
          // Scope by modelId when present; reject if no active checkpoint, since model-scoped usage cannot
          // be attributed to "nothing".
          if (typeof usageModelId === "string" && usageModelId) {
            if (!activeCheckpoint || usageModelId !== activeCheckpoint) {
              return;
            }
          }
          // For local turns, also require the restored count to fit in
          // the active window. Skip when unknown (external provider).
          //
          // llama.cpp only: it stops at the window, so a count past it is stale by
          // definition. MLX generates straight past instead, where an over-window count
          // is the true one and the bar has a state for it.
          const limit = store.loadedIsGguf ? store.loadedContextLength : null;
          if (
            typeof limit === "number" &&
            limit > 0 &&
            (usage.totalTokens ?? 0) > limit
          ) {
            return;
          }
          // Key by the thread this restore read, like the history loader: the await above can outlast a
          // switch away, and an unkeyed write would file this usage under the incoming thread.
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
      [...externalProvidersForChat]
        .sort(
          (a, b) =>
            getExternalProviderDropdownRank(a.providerType) -
            getExternalProviderDropdownRank(b.providerType),
        )
        .flatMap((provider) =>
          provider.models.map((model) => {
            // For OpenRouter's free router the chosen underlying model is latched from `chunk.model`, so
            // render the chip as `openrouter:<short-chosen>`, dropping the redundant `/free` and the
            // chosen id's org prefix: inclusionai/ring-2.6-1t-20260508:free becomes
            // ring-2.6-1t-20260508:free. The `:free` suffix already conveys "free model".
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
  // `externalModels` is flat-mapped from `provider.models`, the ids the user ticked, so a model
  // unticked in the connection dialog looks exactly like one the provider withdrew; the
  // connection's cached catalogue tells the two apart. Depends on the store value and the gate
  // rather than on `externalProvidersForChat`, which is a fresh array each render.
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
      audioType: lora.audioType,
    }));
    return [...fromLoras, ...localModels];
  }, [lorasFromStore, localModels]);

  // Everything the picker can offer right now, so the chat's own model is only proposed when selecting it would work.
  const selectableModelIds = useMemo(
    () =>
      new Set<string>([
        ...models.map((model) => model.id),
        ...loraModels.map((model) => model.id),
        ...externalModels.map((model) => model.id),
      ]),
    [models, loraModels, externalModels],
  );

  // The picker's own handler, reached the way the picker reaches it: with the row's metadata, not
  // the bare id. A local or fine-tuned row is in neither `/api/models/list` nor the external
  // ids, so without it the switch loads on different arguments.
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
    // Models and status only: a LoRA scan that hangs or 500s takes the whole Promise.all
    // with it and leaves the picker empty. The deferred refresh below owns that inventory.
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
    // ChatPage no longer remounts on navigation, so re-check the handoff whenever we return to /chat.
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
        const targetLora = pickBestLoraForBase(state.loras, handoff.baseModel);
        const selectWithConfig = async (
          selection: Pick<SelectedModelInput, "id" | "isLora">,
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
            // As on the Hub launch: the runtime mirror carries no launch flags, and the handoff arrives with
            // another model resident, so there is nothing for /load to inherit them from.
            ...(remembered ? { config: remembered } : {}),
          });
        };
        if (targetLora) {
          console.info("[chat-handoff] loading lora", {
            id: targetLora.id,
            baseModel: targetLora.baseModel,
          });
          await selectWithConfig({ id: targetLora.id, isLora: true });
          if (canceled) return;
          useChatRuntimeStore.getState().setActiveThreadId(null);
          useChatRuntimeStore.getState().setContextUsage(null);
          navigate({ to: "/chat", search: { compare: crypto.randomUUID() } });
          clearHandoff();
          console.info("[chat-handoff] loaded lora + opened compare");
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
        canCompare,
        openModelSelector,
        closeModelSelector,
        openSettings,
        closeSettings,
        enterCompare,
        exitCompare,
      }),
    [
      canCompare,
      closeModelSelector,
      closeSettings,
      enterCompare,
      exitCompare,
      openModelSelector,
      openSettings,
    ],
  );

  const tour = useGuidedTourController({
    id: "chat",
    steps: tourSteps,
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

  const showArtifactOverlay = Boolean(
    selectedArtifact &&
      (view.mode === "compare" || artifactSurface === "overlay"),
  );

  return (
    // Provides `active` to ChatRuntimeProvider (drops the message views while off-route, keeping the
    // runtime alive) and to the compare chrome.
    <ChatActiveContext.Provider value={active}>
    <div className="flex min-h-0 min-w-0 flex-1 basis-0 overflow-hidden bg-background">
      {/* Portaled surfaces render to document.body, escaping the parent's hidden wrapper, so gate them
          on `active` to keep them off other tabs. */}
      {active && <GuidedTour {...tour.tourProps} />}
      {/* Single app-level mount for the Bypass permissions warning: it is driven by global store state,
          so it must live at one stable root, or Compare mode's composers would each render a copy.
          It also portals to body, so gate it on `active`. */}
      {active && <BypassPermissionsConfirmDialog />}
      {/* The MCP servers dialog: its chord has to work before MCP is switched on, and the pill that
          used to own it only renders once it is. Mounted through the route change so it can close
          itself on the way out. */}
      <McpServersDialogMount />
      {/* `--studio-chat-notice-height` is 0 until ChatModelNotice is on screen; the thread viewport
          adds it to the top padding, so without it the first message reads under an opaque bar.
          Declared on the nearest ancestor of BOTH so the two cannot disagree. `has-[>...]`, not
          `has-[...]`: a `:has()` with a DESCENDANT argument is re-checked on any insertion in the
          subtree, which walks the whole thread on every mutation - 17.5 ms per append on a 357k-
          element thread, against 0.10 ms without this rule (Chromium). ChatModelNotice renders a
          direct child, which tests/thread-ancestor-has-scope.test.ts asserts. */}
      <div className="relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden has-[>[data-chat-model-notice]]:[--studio-chat-notice-height:2.25rem]">
        <NativeModelDropOverlay state={nativeModelDropState} />
        {/* Fade under the top bar so messages dissolve as they scroll beneath it, instead of a hard cut. */}
        {view.mode !== "compare" && (
          <div
            aria-hidden
            className="chat-header-fade pointer-events-none absolute left-0 right-[10px] top-[calc(var(--studio-content-top-inset,0px)+var(--studio-chat-header-height,48px)+var(--studio-chat-notice-height,0px))] z-20 h-6 bg-gradient-to-b from-background to-transparent"
          />
        )}
        <div
          className={cn(
            "pointer-events-none absolute top-[var(--studio-content-top-inset,0px)] left-0 right-[10px] z-40 flex h-[var(--studio-chat-header-height,48px)] shrink-0 items-start bg-background pt-[var(--studio-chat-header-padding-top,11px)] pr-[calc(0.5rem+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]",
            isMobile
              ? "pl-12"
              : pinned
                ? "pl-2"
                : isTauri
                  ? "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]"
                  : "pl-[calc(0.5rem+max(0px,var(--studio-mac-traffic-light-inset,0px)-var(--sidebar-width-icon,3rem)))]",
            view.mode === "compare" &&
              "right-[10px] left-auto w-auto bg-transparent pl-0 pr-[calc(0.5rem+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]",
          )}
        >
          <div className="pointer-events-auto flex items-center gap-1">
            {isTauri && !isMobile && !pinned && view.mode !== "compare" && (
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                title="New chat"
                aria-label="New chat"
                onClick={handleDesktopNewChat}
                className="!size-[30px] rounded-[10px] text-muted-foreground"
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
                // Resident, not merely picked: an image or video load evicts the chat model and leaves this
                // selection behind, so the tick stayed on a released model.
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
                onValueChange={handleCheckpointChange}
                onEject={handleEject}
                onFoldersChange={refreshLocalModels}
                onModelsChange={refreshModelLists}
                deleteDisabled={modelOperationInProgress}
                variant="ghost"
                open={active && modelSelectorOpen}
                onOpenChange={handleModelSelectorOpenChange}
                triggerDataTour="chat-model-selector"
                contentDataTour="chat-model-selector-popover"
                showCloudIndicator={isExternalModel}
                className="max-w-[62vw] !pr-3 sm:max-w-none !h-[var(--studio-chat-control-height,34px)]"
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
          <div className="pointer-events-auto ml-auto flex items-center gap-1">
            {view.mode === "single" && (contextUsage || contextWindowKnown) ? (
              <ContextUsageBar
                used={contextUsage?.totalTokens ?? null}
                // null on external providers; the bar handles that.
                total={loadedContextLength}
                cached={contextUsage?.cachedTokens}
                cacheWrites={contextUsage?.cacheWriteTokens}
                promptTokens={contextUsage?.promptTokens}
                completionTokens={contextUsage?.completionTokens}
                isMlx={isServedByMlx(
                  Boolean(loadedIsGguf),
                  platformDeviceType,
                  platformChatOnlyReason,
                )}
                contextEnforced={loadedContextEnforced}
                className="h-[var(--studio-chat-control-height,34px)]"
              />
            ) : null}
            {view.mode === "single" && (
              <Tooltip>
                <TooltipPrimitive.Trigger asChild={true}>
                  <button
                    type="button"
                    onClick={toggleIncognito}
                    className={cn(
                      "flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                      incognito
                        ? "bg-primary/10 text-primary hover:bg-primary/15"
                        : "text-nav-fg hover:bg-nav-surface-hover hover:text-black dark:hover:text-white",
                    )}
                    aria-label={incognitoLabel}
                    aria-pressed={incognito}
                  >
                    <HugeiconsIcon
                      icon={BubbleChatTemporaryIcon}
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
                  {incognitoLabel}
                </TooltipContent>
              </Tooltip>
            )}
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
                      closeArtifactSurface();
                      openResearchPanel(latestResearchRunId);
                    }}
                    className="relative flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:hover:text-white"
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
                    className="flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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

        {/* One provider shared by the project and single views, never keyed on thread / nonce / project,
            so switching between them reattaches the live run instead of remounting the runtime and
            aborting generation (#8908). Any key here brings the bug back. Compare renders as a
            sibling so it hides this rather than unmounting it, since ComparePane builds its own
            providers and nesting throws. */}
        {baseView ? (
          <div
            className={
              baseBackgrounded
                ? "hidden"
                : "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden"
            }
            inert={baseBackgrounded || undefined}
          >
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
                    <SingleContent
                      threadId={baseView.threadId}
                      artifact={selectedArtifact}
                      artifactSurface={artifactSurface}
                      onCloseArtifact={closeArtifactSurface}
                    />
                  </NativeAttachmentTargetContext.Provider>
                )}
              </ChatRuntimeProvider>
            </ChatActiveContext.Provider>
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

        {active && showArtifactOverlay && selectedArtifact ? (
          <ArtifactSurface
            artifact={selectedArtifact}
            variant="overlay"
            onClose={closeArtifactSurface}
          />
        ) : null}
      </div>

      <ChatSettingsPanel
        open={active && settingsOpen}
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
