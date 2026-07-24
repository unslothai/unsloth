// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  applyModelLoadConfigToRuntime,
  currentRuntimePerModelConfig,
  type DeletedModelRef,
  type ExternalModelOption,
  type LoraModelOption,
  type ModelOption,
  ModelSelector,
  type ModelSelectorChangeMeta,
  type PerModelConfig,
  resolveInitialConfig,
  SidebarModelConfig,
  useActiveModelConfig,
} from "@/features/model-picker";
import { ProjectComposer, Thread } from "@/components/assistant-ui/thread";
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
import {
  DOWNLOAD_KIND,
  downloadManager,
  useRepoDownload,
} from "@/features/hub/download-manager";
import {
  type NativeIntent,
  NativeModelChip,
  NativeModelDropOverlay,
  useChooseNativeModel,
  useNativeIntentStore,
  useNativeModelDrop,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { isTauri } from "@/lib/api-base";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Archive03Icon,
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
import { listLocalModels, notifyChatHistoryUpdated } from "./api/chat-api";
import { ArtifactSurface } from "./artifacts/artifact-surface";
import {
  clearAutoOpenedArtifacts,
  useChatArtifactsStore,
  useSelectedChatArtifact,
} from "./artifacts/store";
import type { ChatArtifact, ChatArtifactSurface } from "./artifacts/types";
import { ChatSettingsPanel } from "./chat-settings-sheet";
import { ContextUsageBar } from "./components/context-usage-bar";
import { ModelLoadInlineStatus } from "./components/model-load-status";
import { ProjectSwitcher } from "./components/project-switcher";
import {
  buildExternalModelId,
  isExternalModelId,
  parseExternalModelId,
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
  providerSupportsBuiltinCodeExecution,
  providerSupportsBuiltinImageGeneration,
  providerSupportsBuiltinWebFetch,
  providerSupportsBuiltinWebSearch,
} from "./provider-capabilities";
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
  hasGgufSource,
  isDownloadableHubRepo,
  loadOptionalBool,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import { useChatPreferencesStore } from "./stores/chat-preferences-store";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import { buildChatTourSteps } from "./tour";
import type { ChatView, MessageRecord } from "./types";
import {
  getStoredChatThread,
  isExpectedBackgroundChatStorageError,
  listStoredChatMessages,
  listStoredChatThreads,
} from "./utils/chat-history-storage";
import { isAssistantLocalThreadId } from "./utils/thread-ids";


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
  newThreadNonce,
  projectId,
  artifact,
  artifactSurface,
  onCloseArtifact,
}: {
  threadId?: string;
  newThreadNonce?: string;
  projectId?: string | null;
  artifact?: ChatArtifact | null;
  artifactSurface: ChatArtifactSurface;
  onCloseArtifact: () => void;
}): ReactElement {
  const openArtifact = useChatArtifactsStore((state) => state.openArtifact);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const artifactPanelRef = useRef<PanelImperativeHandle | null>(null);
  const hasInitializedArtifactPanelRef = useRef(false);
  const [isArtifactLayoutAnimating, setIsArtifactLayoutAnimating] =
    useState(false);
  const [isArtifactPanelLayoutActive, setIsArtifactPanelLayoutActive] =
    useState(false);
  const [isArtifactSurfaceVisible, setIsArtifactSurfaceVisible] =
    useState(false);
  // Without a URL threadId the artifact must belong to the active thread.
  const showArtifactPanel = Boolean(
    artifact &&
      artifactSurface === "panel" &&
      (threadId
        ? !artifact.threadId || artifact.threadId === threadId
        : Boolean(artifact.threadId && artifact.threadId === activeThreadId)),
  );

  const artifactLayoutActive = showArtifactPanel || isArtifactPanelLayoutActive;
  const artifactPanelSettledOpen =
    showArtifactPanel &&
    isArtifactPanelLayoutActive &&
    !isArtifactLayoutAnimating;

  useEffect(() => {
    const panel = artifactPanelRef.current;
    if (!panel) return;

    setIsArtifactSurfaceVisible(false);

    if (!hasInitializedArtifactPanelRef.current) {
      hasInitializedArtifactPanelRef.current = true;
      if (!showArtifactPanel) {
        panel.resize("0%");
        return;
      }
    }

    setIsArtifactPanelLayoutActive(true);
    setIsArtifactLayoutAnimating(true);
    let resizeFrameId = 0;
    const prepFrameId = window.requestAnimationFrame(() => {
      resizeFrameId = window.requestAnimationFrame(() => {
        panel.resize(showArtifactPanel ? ARTIFACT_PANEL_DEFAULT_SIZE : "0%");
      });
    });
    const surfaceTimerId = showArtifactPanel
      ? window.setTimeout(() => {
          setIsArtifactSurfaceVisible(true);
        }, ARTIFACT_SURFACE_POP_DELAY_MS)
      : 0;
    const timeoutId = window.setTimeout(() => {
      setIsArtifactLayoutAnimating(false);
      if (!showArtifactPanel) {
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
  }, [showArtifactPanel]);

  const threadPane = (
    <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden">
      <Thread hideWelcome={Boolean(threadId)} targetThreadId={threadId} />
    </div>
  );

  return (
    <ChatRuntimeProvider
      modelType="base"
      initialThreadId={threadId}
      newThreadNonce={newThreadNonce}
      projectId={projectId}
      listThreads={false}
    >
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
            !artifactLayoutActive && "pointer-events-none -ml-0 -mr-0 w-0",
          )}
        />
        <ResizablePanel
          panelRef={artifactPanelRef}
          id="chat-artifact"
          defaultSize="0%"
          minSize={artifactPanelSettledOpen ? "30%" : "0%"}
          maxSize={artifactLayoutActive ? "58%" : "0%"}
          collapsible={true}
          collapsedSize="0%"
          className={cn(
            "h-full min-h-0 min-w-0 overflow-visible",
            !showArtifactPanel && "pointer-events-none",
          )}
        >
          <div
            data-artifact-surface-visible={
              isArtifactSurfaceVisible ? "true" : "false"
            }
            className="chat-artifact-pop-surface flex h-full min-h-0 min-w-0 flex-col overflow-visible"
          >
            {showArtifactPanel && artifact ? (
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
    </ChatRuntimeProvider>
  );
});

type CompareModelSelection = {
  id: string;
  isLora: boolean;
  ggufVariant?: string;
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

/**
 * True when the loaded checkpoint is a LoRA, meaning a base-vs-fine-tuned
 * compare that uses the fast simultaneous adapter-toggle path.
 */
function useIsLoraCompare(): boolean {
  return useChatRuntimeStore((s) => {
    const cp = s.params.checkpoint;
    const selected = cp ? s.loras.find((l) => l.id === cp) : undefined;
    return selected?.exportType === "lora";
  });
}

const CompareContent = memo(function CompareContent({
  pairId,
  projectId,
  models,
  loraModels,
  externalModels,
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
  onFoldersChange?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  onExitCompare?: () => void;
}): ReactElement {
  const isLoraCompare = useIsLoraCompare();

  return isLoraCompare ? (
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
      onFoldersChange={onFoldersChange}
      onModelsChange={onModelsChange}
      deleteDisabled={deleteDisabled}
      onExitCompare={onExitCompare}
    />
  );
});

/**
 * A single column in the compare layout: one ChatRuntimeProvider and one
 * Thread with hideComposer (the composer is shared across panes).
 *
 * Each pane is `flex-1 basis-0 min-h-0 min-w-0` so panes share height
 * (mobile flex-col) or width (desktop flex-row) equally. The `min-*`
 * constraints let the inner viewport scroll instead of spilling.
 */
function ComparePane({
  modelType,
  pairId,
  projectId,
  initialThreadId,
  handleName,
  header,
  borderClassName,
}: {
  modelType: "base" | "lora" | "model1" | "model2";
  pairId: string;
  projectId?: string | null;
  initialThreadId: string | undefined;
  handleName: string;
  header: ReactElement;
  borderClassName?: string;
}): ReactElement {
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
        >
          <RegisterCompareHandle name={handleName} />
          <Thread hideComposer={true} hideWelcome={true} />
        </ChatRuntimeProvider>
      </div>
    </div>
  );
}

/**
 * Shared shell for both compare variants: a flex column with the two panes
 * as siblings and the shared composer docked at the bottom. Panes stack on
 * mobile (flex-col), sit side by side on desktop (md:flex-row).
 *
 * Flex, not grid, for the pane container: grid rows with 1fr triggered
 * resize thrash in assistant-ui's autoscroll on breakpoint crossings,
 * leaving it stuck in a scroll-to-bottom loop.
 */
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
  const active = useChatActive();

  const compareRunning = useChatRuntimeStore(
    (s) => Object.keys(s.runningByThreadId).length > 0,
  );

  useEffect(() => {
    if (compareRunning) return;
    let isActive = true;
    listStoredChatThreads({ pairId })
      .then((threads) => {
        if (!isActive) return;
        setBaseThreadId(threads.find((t) => t.modelType === "base")?.id);
        setLoraThreadId(threads.find((t) => t.modelType === "lora")?.id);
      })
      .catch((error) => {
        if (!isExpectedBackgroundChatStorageError(error)) {
          throw error;
        }
      });
    return () => {
      isActive = false;
    };
  }, [pairId, compareRunning]);

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

/**
 * Per-pane header (inside GeneralCompareContent) with the model selector,
 * aligned to the global topbar height. Left pane reserves room for the
 * mobile sidebar trigger; right pane for the global settings button.
 */
function GeneralCompareHeader({
  models,
  loraModels,
  externalModels,
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
  // Controlled so the body-portaled popover can't linger over another tab off-route.
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
  onFoldersChange?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  onExitCompare?: () => void;
}): ReactElement {
  const handlesRef = useRef<Record<string, CompareHandle>>({});
  const [model1ThreadId, setModel1ThreadId] = useState<string>();
  const [model2ThreadId, setModel2ThreadId] = useState<string>();

  const globalCheckpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const globalGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const active = useChatActive();
  const compareRunning = useChatRuntimeStore(
    (s) => Object.keys(s.runningByThreadId).length > 0,
  );
  const [model1, setModel1] = useState<CompareModelSelection>({
    id: globalCheckpoint || "",
    isLora: false,
    ggufVariant: globalGgufVariant ?? undefined,
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
    if (compareRunning) return;
    let isActive = true;
    listStoredChatThreads({ pairId })
      .then((threads) => {
        if (!isActive) return;
        setModel1ThreadId(
          threads.find(
            (t) => t.modelType === "model1" || t.modelType === "base",
          )?.id,
        );
        setModel2ThreadId(
          threads.find(
            (t) => t.modelType === "model2" || t.modelType === "lora",
          )?.id,
        );
      })
      .catch((error) => {
        if (!isExpectedBackgroundChatStorageError(error)) {
          throw error;
        }
      });
    return () => {
      isActive = false;
    };
  }, [pairId, compareRunning]);

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
          header={
            <GeneralCompareHeader
              side="left"
              models={models}
              loraModels={loraModels}
              externalModels={externalModels}
              value={model1.id}
              selectedConfig={model1.config}
              selectedGgufVariant={model1.ggufVariant}
              onValueChange={(id, meta) =>
                setModel1({
                  id,
                  isLora: meta.isLora,
                  ggufVariant: meta.ggufVariant,
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
          borderClassName="border-t border-sidebar-border md:border-t-0 md:border-l"
          header={
            <GeneralCompareHeader
              side="right"
              models={models}
              loraModels={loraModels}
              externalModels={externalModels}
              value={model2.id}
              selectedConfig={model2.config}
              selectedGgufVariant={model2.ggufVariant}
              onValueChange={(id, meta) =>
                setModel2({
                  id,
                  isLora: meta.isLora,
                  ggufVariant: meta.ggufVariant,
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

// Unique thread nonce; falls back off crypto.randomUUID for non-secure
// (HTTP LAN) contexts where it is unavailable.
function createThreadNonce(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

// Chat export formats, mirroring the sidebar chat menu.
type ProjectChatExportFormat = "raw-jsonl" | "csv" | "sharegpt-jsonl";
const PROJECT_CHAT_EXPORT_OPTIONS: Array<{
  label: string;
  format: ProjectChatExportFormat;
}> = [
  { label: "Raw JSONL", format: "raw-jsonl" },
  { label: "CSV", format: "csv" },
  { label: "ShareGPT JSONL", format: "sharegpt-jsonl" },
];

async function exportProjectConversation(
  threadId: string,
  format: ProjectChatExportFormat,
): Promise<void> {
  const exports = await import("./prompt-storage/prompt-storage-dialog");
  if (format === "raw-jsonl") return exports.exportConversationRawJsonl(threadId);
  if (format === "csv") return exports.exportConversationCsv(threadId);
  return exports.exportConversationShareGPT(threadId);
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
}: {
  projectId: string;
  projectName: string;
  items: SidebarItem[];
}): ReactElement {
  const navigate = useNavigate();
  // Gates body-portaled surfaces so they can't linger or act while the landing
  // is off-route (e.g. behind another tab).
  const active = useChatActive();
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const initialActiveThreadRef = useRef<string | null>(null);
  const [projectTab, setProjectTab] = useState<"chats" | "sources">("chats");
  const [pendingNewThreadId, setPendingNewThreadId] = useState<string | null>(
    null,
  );
  const [newThreadNonce, setNewThreadNonce] = useState(() =>
    createThreadNonce(),
  );
  const [previews, setPreviews] = useState<
    Record<string, { snippet: string; date: string }>
  >({});
  // Inline rename, mirroring the sidebar recent-row UX: edit the title in place,
  // commit on Enter/blur, cancel on Escape. Reuses the projectId-agnostic
  // renameChatItem so behavior matches the sidebar.
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  // Skips the input's blur-commit when Enter/Escape already handled it.
  const skipRenameBlurRef = useRef(false);
  // Optimistic title shown until the debounced sidebar refresh (fired by the
  // rename) catches up, so the old name does not flash back in.
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
      // Refresh chat history so the project's now-deleted chats don't linger
      // in the sidebar, matching the sidebar delete path.
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
    initialActiveThreadRef.current =
      useChatRuntimeStore.getState().activeThreadId;
    useChatRuntimeStore.getState().setActiveThreadId(null);
    useChatRuntimeStore.getState().setContextUsage(null);
    setPendingNewThreadId(null);
    setNewThreadNonce(createThreadNonce());
    setRenamingId(null);
    setPendingRename(null);
  }, [projectId]);

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
  const pinnedChatIdSet = useMemo(
    () => new Set(pinnedChatIds),
    [pinnedChatIds],
  );
  const [confirmingDelete, setConfirmingDelete] = useState<SidebarItem | null>(
    null,
  );

  // Landing has no active thread selected, so the onView callback here is a
  // no-op; the items list refreshes itself once storage emits its update.
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
    async (item: SidebarItem) => {
      try {
        await deleteChatItem(item, activeThreadId ?? undefined, noopView);
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
      if (confirmDeleteChats) setConfirmingDelete(item);
      else void runDelete(item);
    },
    [confirmDeleteChats, runDelete],
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

  useEffect(() => {
    if (!activeThreadId) {
      // Leaving a created chat for a new one: rotate the nonce so the runtime
      // switches to a fresh thread instead of appending to the old chat.
      if (pendingNewThreadId) {
        setNewThreadNonce(createThreadNonce());
        setPendingNewThreadId(null);
      }
      return;
    }
    if (activeThreadId === initialActiveThreadRef.current) {
      return;
    }
    setPendingNewThreadId(activeThreadId);
  }, [activeThreadId, pendingNewThreadId]);

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
                ? extractMessageText(firstUserMessage.content)
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

  return (
    <ChatRuntimeProvider
      key={projectId}
      projectId={projectId}
      newThreadNonce={newThreadNonce}
      listThreads={false}
    >
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
                              // Ignore keydowns fired mid-IME-composition (CJK)
                              // so a candidate-confirming Enter or candidate-
                              // cancelling Escape does not commit/cancel the
                              // rename. Guard before the key branch so Escape is
                              // covered too (isComposing on WebKit, 229 on Chromium).
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
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                const target = confirmingDelete;
                setConfirmingDelete(null);
                if (target) void runDelete(target);
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
    </ChatRuntimeProvider>
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

// `search` comes from RootLayout (not useSearch) so ChatPage stays mounted off-route
// (keeping an in-flight generation alive), frozen to the last /chat search. `active`
// is false off-route: close body-portaled surfaces and stop route-specific listeners
// that would otherwise bleed over the visible tab.
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
    store.setIncognito(!store.incognito);
    // On an empty scratch chat there's nothing to abandon, so flip in
    // place: navigating would remount the thread and bounce the composer
    // (it docks to the bottom before the welcome state re-centers it).
    // Otherwise start a clean chat so the temporary session can't inherit
    // or leave behind a persisted thread (matches ChatGPT / Gemini).
    const onEmptyScratchChat =
      !search.thread &&
      !search.compare &&
      !search.project &&
      store.activeThreadId == null;
    if (onEmptyScratchChat) return;
    // setActiveThreadId already clears contextUsage.
    store.setActiveThreadId(null);
    store.setActiveProjectId(null);
    navigate({ to: "/chat", search: { new: crypto.randomUUID() } });
  }, [navigate, search]);
  const hydratePersistedSettings = useChatRuntimeStore(
    (s) => s.hydratePersistedSettings,
  );
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
    // Skip while off-route: ChatPage stays mounted, and toast+navigate here would
    // yank the user back to chat from whatever tab they're on.
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
  const [modelSelectorLocked, setModelSelectorLocked] = useState(false);
  const viewBeforeCompareRef = useRef<ChatSearch | null>(null);
  // Latest non-compare view, so exiting compare can restore it even when
  // compare was opened from a path that doesn't set viewBeforeCompareRef.
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
  const ggufContextLength = useChatRuntimeStore(
    (state) => state.ggufContextLength,
  );
  const ggufNativeContextLength = useChatRuntimeStore(
    (state) => state.ggufNativeContextLength,
  );
  const contextUsage = useChatRuntimeStore((state) => state.contextUsage);
  const modelsFromStore = useChatRuntimeStore((state) => state.models);
  const lorasFromStore = useChatRuntimeStore((state) => state.loras);
  const modelsError = useChatRuntimeStore((state) => state.modelsError);
  const modelLoading = useChatRuntimeStore((state) => state.modelLoading);
  const clearCheckpoint = useChatRuntimeStore((state) => state.clearCheckpoint);
  const resetArtifacts = useChatArtifactsStore((state) => state.resetArtifacts);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(
    search.project ?? null,
  );
  const { projects, isLoading: projectsLoading } = useChatProjects();
  const currentProject = currentProjectId
    ? (projects.find((project) => project.id === currentProjectId) ?? null)
    : null;
  const { items: currentProjectItems } = useChatSidebarItems({
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
  const openProjectsList = useCallback(() => {
    navigate({ to: "/projects" });
  }, [navigate]);
  const persistedActiveThreadId = isAssistantLocalThreadId(activeThreadId)
    ? null
    : activeThreadId;
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
  const {
    checkpoint: runtimeCheckpoint,
    isGguf: runtimeModelIsGguf,
    config: activeModelConfig,
  } = useActiveModelConfig();
  const activeModelIsGguf =
    runtimeCheckpoint != null && !isExternalModel && runtimeModelIsGguf;
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
    // Per-provider default effort. Anthropic gets the highest level since
    // Claude's adaptive thinking adjusts cost per turn (top of dial =
    // strongest answers, still skips thinking when trivial). OpenAI gets
    // "high" (gpt-5.x accept it across the board; good cost/quality for
    // Responses-API tools). Everyone else "medium". Overridable via Think.
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
    // Kimi's k2.6/k2.5 default to thinking enabled server-side (per
    // https://platform.kimi.ai/docs/models). Mirror that so the Think pill
    // comes up clicked for Kimi models. Search stays off; the composer's
    // mutual-exclusion handlers flip the two when needed.
    const isKimi = provider?.providerType === "kimi";
    // Web search on by default only for the two providers we trust most:
    // Anthropic and OpenAI (both with structured citations). Others stay
    // off-by-default; OpenRouter and Kimi work on opt-in but are less
    // reliable, so we don't pre-enable them.
    const searchOnByDefault =
      supportsBuiltinWebSearch &&
      (provider?.providerType === "anthropic" ||
        provider?.providerType === "openai");
    const storedToolsEnabled = loadOptionalBool(CHAT_TOOLS_ENABLED_KEY);
    const storedCodeToolsEnabled = loadOptionalBool(
      CHAT_CODE_TOOLS_ENABLED_KEY,
    );
    const storedImageToolsEnabled = loadOptionalBool(
      CHAT_IMAGE_TOOLS_ENABLED_KEY,
    );
    const storedWebFetchToolsEnabled = loadOptionalBool(
      CHAT_WEB_FETCH_TOOLS_ENABLED_KEY,
    );
    const nextToolsEnabled = supportsBuiltinWebSearch
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
      // External models have no local tool runtime, so `supportsTools` is
      // false. The `supportsBuiltin*` flags cover providers that run tools
      // server-side: WebSearch lights the Search pill (OpenAI/Anthropic/
      // OpenRouter/Kimi), CodeExecution the Code pill (Claude 4.x, gpt-5.5),
      // ImageGeneration the Images pill (OpenAI cloud Responses-API only).
      supportsTools: false,
      supportsBuiltinWebSearch,
      supportsBuiltinCodeExecution,
      supportsBuiltinImageGeneration,
      supportsBuiltinWebFetch,
      toolsEnabled: nextToolsEnabled,
      codeToolsEnabled: supportsBuiltinCodeExecution
        ? (storedCodeToolsEnabled ?? false)
        : false,
      imageToolsEnabled: supportsBuiltinImageGeneration
        ? (storedImageToolsEnabled ?? false)
        : false,
      // Default Fetch off (Anthropic bills per fetch); deliberate opt-in.
      webFetchToolsEnabled: supportsBuiltinWebFetch
        ? (storedWebFetchToolsEnabled ?? false)
        : false,
    });
  }, [externalProvidersForChat, inferenceParams.checkpoint]);
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

  // Derive view from URL search params
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

  // Temporary chat only applies to a fresh single-view chat. Exit incognito
  // when we land on anything else (compare, a project, or an existing thread
  // via sidebar/deep link/back), so the toggle isn't stranded and the UI
  // never implies a saved thread is temporary.
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

  useEffect(() => {
    clearAutoOpenedArtifacts();
    closeArtifactSurface();
  }, [artifactViewKey, closeArtifactSurface]);

  useEffect(() => {
    if (view.mode !== "single") return;
    if (view.threadId || !selectedArtifact) return;
    // Close any canvas that doesn't belong to the active thread.
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
      const wantManagerDownload =
        isDownloadableHubRepo(selection) && !selection.isDownloaded;
      if (store.modelLoading) {
        const wantBackgroundDownload =
          wantManagerDownload ||
          (selection.source === "hub" &&
            hasGgufSource(selection) &&
            !selection.isDownloaded);
        const isLoadingThisPick =
          !!loadingModel &&
          normalizeModelRef(loadingModel.id) ===
            normalizeModelRef(selection.id) &&
          (loadingModel.ggufVariant ?? null) === (selection.ggufVariant ?? null);
        if (isLoadingThisPick) {
          toast.info("This model is already loading", {
            description: "It's downloading as part of the load in progress.",
          });
        } else if (wantBackgroundDownload) {
          const outcome = await downloadManager.requestStart({
            kind: DOWNLOAD_KIND.MODEL,
            repoId: selection.id,
            variant: selection.ggufVariant ?? null,
            expectedBytes: selection.expectedBytes ?? 0,
          });
          if (outcome === "started") {
            toast.info("Downloading in the background", {
              description:
                "It'll be ready to load once the current model finishes.",
            });
          } else if (outcome === "conflict") {
            toast.info("Resume this download from Models", {
              description:
                "An earlier partial download used a different transport. Open the Models tab to resume or restart it.",
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
      const wantManagerStage =
        wantManagerDownload ||
        (selection.source === "hub" &&
          hasGgufSource(selection) &&
          !selection.isDownloaded);
      if (wantManagerStage) {
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
      const hasAppliedConfig = applyModelLoadConfigToRuntime(
        selection.config ?? rememberedConfigFor(selection),
        // Only the rememberedConfigFor fallback is a storage restore; an
        // explicit selection.config is a fresh pick in the current index space.
        { fromPersisted: !selection.config },
      );
      await selectModel({
        ...selection,
        ...(hasAppliedConfig ? { keepSpeculative: true } : {}),
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
  useEffect(() => {
    const pending = pendingHubAutoLoad;
    if (!pending) return;
    let active = true;
    void (async () => {
      const outcome = await downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId: pending.selection.id,
        variant: pending.selection.ggufVariant ?? null,
        expectedBytes: pending.selection.expectedBytes ?? 0,
      });
      if (!active) return;
      if (outcome === "started") {
        toast.info("Downloading model", {
          description: "It'll load automatically once the download finishes.",
        });
        return;
      }
      if (outcome === "conflict") {
        // Keep pendingHubAutoLoad bound so this surface's cleanup does not wipe
        // the conflict just recorded by requestStart (which the toast points the
        // user to); resolving it from the Hub completes the download and this
        // surface's onComplete auto-loads, mirroring the "started" branch.
        toast.info("Resume this download from Models", {
          description:
            "An earlier partial download used a different transport. Open the Models tab to resume or restart it.",
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
    };
  }, [pendingHubAutoLoad]);
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
  const handleNativeModelPickerAutoLoad = useCallback(
    (intent: NativeIntent) =>
      loadNativeModelIntent(intent, "Loading chosen local GGUF model."),
    [loadNativeModelIntent],
  );
  const canAutoLoadPickedNativeModel = useCallback(() => {
    const store = useChatRuntimeStore.getState();
    return (
      view.mode === "single" &&
      nativePathLeasesSupported &&
      !loadingModel &&
      !modelLoading &&
      !store.modelLoading &&
      !store.params.checkpoint
    );
  }, [loadingModel, modelLoading, nativePathLeasesSupported, view.mode]);
  const chooseNativeModel = useChooseNativeModel({
    shouldAutoLoad: canAutoLoadPickedNativeModel,
    onAutoLoad: handleNativeModelPickerAutoLoad,
  });
  const nativeModelDropState = useNativeModelDrop({
    enabled: active && view.mode === "single",
    nativePathLeasesSupported,
    hasActiveModel,
    isModelLoading: Boolean(loadingModel) || modelLoading,
    onAutoLoad: handleNativeModelDropAutoLoad,
  });

  const handleCheckpointChange = useCallback(
    (
      value: string,
      meta?: ModelSelectorChangeMeta,
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
        // Same per-provider default policy as the useEffect above:
        // Anthropic highest level, OpenAI "high", everyone else "medium".
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
        // Clear any cached router-picked openrouter/free model unless staying
        // on openrouter/free, else the chip keeps a stale ":<chosen>" suffix.
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
        // See sibling useEffect: Kimi's k2.x default to thinking enabled
        // (Think pill clicked). Search stays off; the composer's mutual
        // exclusion flips them.
        const isKimi = selectedProvider?.providerType === "kimi";
        // Mirror of sibling useEffect: Anthropic/OpenAI get Search on by
        // default (structured citations end-to-end); others stay off.
        const searchOnByDefault =
          supportsBuiltinWebSearch &&
          (selectedProvider?.providerType === "anthropic" ||
            selectedProvider?.providerType === "openai");
        const storedToolsEnabled = loadOptionalBool(CHAT_TOOLS_ENABLED_KEY);
        const storedCodeToolsEnabled = loadOptionalBool(
          CHAT_CODE_TOOLS_ENABLED_KEY,
        );
        const storedImageToolsEnabled = loadOptionalBool(
          CHAT_IMAGE_TOOLS_ENABLED_KEY,
        );
        const storedWebFetchToolsEnabled = loadOptionalBool(
          CHAT_WEB_FETCH_TOOLS_ENABLED_KEY,
        );
        const nextToolsEnabled = supportsBuiltinWebSearch
          ? isKimi
            ? false
            : (storedToolsEnabled ?? searchOnByDefault)
          : false;
        useChatRuntimeStore.setState({
          activeGgufVariant: null,
          ggufContextLength: null,
          ggufMaxContextLength: null,
          ggufNativeContextLength: null,
          activeNativePathToken: null,
          activeNativePathExpiresAtMs: null,
          // Clear previous-model counters, else the relaxed external-provider
          // render gate shows stale stats until the next completion.
          contextUsage: null,
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
          // External models have no local tool runtime → supportsTools false.
          // The supportsBuiltin* flags carry server-side capability per pill:
          // Search, Code (Claude 4.x + gpt-5.5), Images (OpenAI cloud
          // Responses-API).
          supportsTools: false,
          supportsBuiltinWebSearch,
          supportsBuiltinCodeExecution,
          supportsBuiltinImageGeneration,
          supportsBuiltinWebFetch,
          toolsEnabled: nextToolsEnabled,
          codeToolsEnabled: supportsBuiltinCodeExecution
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
      // Local model picked → drop any cached openrouter/free chosen model.
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
          source: meta?.source,
          isLora: meta?.isLora,
          ggufVariant: meta?.ggufVariant,
          isDownloaded: meta?.isDownloaded || isSameLoadedModel,
          expectedBytes: meta?.expectedBytes,
          isGguf: meta?.isGguf,
          config: meta?.config,
          nativePathToken: meta?.nativePathToken,
          nativePathExpiresAtMs: meta?.nativePathExpiresAtMs,
          forceReload: isSameLoadedModel || undefined,
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
      const nativeToken = runtime.activeNativePathToken;
      const nativeExpiry = runtime.activeNativePathExpiresAtMs;
      // A file-picked GGUF is reachable only via its native path token, which
      // the desktop host prunes after a TTL. Reusing an expired token makes the
      // reload fail with an opaque error, so prompt the user to re-select the
      // file instead.
      if (nativeToken && nativeExpiry != null && Date.now() >= nativeExpiry) {
        toast.error("This local model file's access has expired.", {
          description: "Re-select the model file to reload it.",
        });
        return;
      }
      handleCheckpointChange(checkpoint, {
        source: "local",
        isLora: activeModelIsLora,
        ggufVariant: activeGgufVariant ?? undefined,
        // Without the native token the reload validates the display label as a
        // repo and fails.
        nativePathToken: nativeToken ?? undefined,
        nativePathExpiresAtMs: nativeExpiry,
        isGguf: activeModelIsGguf,
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

  const openModelSelector = useCallback(() => {
    setModelSelectorLocked(true);
    setModelSelectorOpen(true);
  }, []);

  const closeModelSelector = useCallback(() => {
    setModelSelectorLocked(false);
    setModelSelectorOpen(false);
  }, []);

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
    // Prefer the explicit save; fall back to the last non-compare view so
    // the composer + menu path also returns where the user started.
    const saved = viewBeforeCompareRef.current ?? lastNonCompareViewRef.current;
    // No saved view (compare opened by direct URL); fall back to a fresh chat.
    if (!saved) {
      navigate({ to: "/chat" });
      return;
    }
    viewBeforeCompareRef.current = null;
    navigate({ to: "/chat", search: saved });
    // Restore usage from the last assistant message, only if it matches the
    // active checkpoint, else the relaxed render gate shows stale stats.
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
          // Scope by modelId when present; reject if no active checkpoint
          // (model-scoped usage can't be attributed to "nothing").
          if (typeof usageModelId === "string" && usageModelId) {
            if (!activeCheckpoint || usageModelId !== activeCheckpoint) {
              return;
            }
          }
          // For local turns, also require the restored count to fit in
          // the active window. Skip when unknown (external provider).
          const limit = store.ggufContextLength;
          if (
            typeof limit === "number" &&
            limit > 0 &&
            (usage.totalTokens ?? 0) > limit
          ) {
            return;
          }
          store.setContextUsage(usage);
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
            // For OpenRouter's free router we know which underlying free
            // model the gateway picked once a stream completes (chat-adapter
            // latches `chunk.model`). Render the chip as
            // `openrouter:<short-chosen>`, dropping the redundant `/free`
            // and the chosen id's org prefix (e.g. openrouter/free +
            // inclusionai/ring-2.6-1t-20260508:free ->
            // openrouter:ring-2.6-1t-20260508:free). The `:free` suffix
            // already conveys "free model".
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

  const [localModels, setLocalModels] = useState<LoraModelOption[]>([]);

  const refreshLocalModels = useCallback(() => {
    void listLocalModels()
      .then((res) => {
        setLocalModels(
          res.models
            .filter(
              (m) =>
                m.source === "lmstudio" ||
                m.source === "models_dir" ||
                m.source === "custom",
            )
            .map((m) => ({
              id: m.id,
              name:
                m.source === "lmstudio" && m.model_id
                  ? m.model_id
                  : m.display_name,
              baseModel:
                m.source === "lmstudio"
                  ? "LM Studio"
                  : m.source === "custom"
                    ? "Custom Folders"
                    : "Local models",
              updatedAt: m.updated_at ?? undefined,
              source: "local" as const,
            })),
        );
      })
      .catch(() => {});
  }, [navigate]);

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
    }));
    return [...fromLoras, ...localModels];
  }, [lorasFromStore, localModels]);

  const inventoryRefreshStartedRef = useRef(false);
  const refreshDeferredModelInventories = useCallback(() => {
    inventoryRefreshStartedRef.current = true;
    void refresh({ includeLoras: true });
    refreshLocalModels();
  }, [refresh, refreshLocalModels]);

  useEffect(() => {
    if (getTrainingCompareHandoff()) return;
    void refresh({ includeLoras: false });
    const timeoutId = window.setTimeout(() => {
      if (!inventoryRefreshStartedRef.current) {
        refreshDeferredModelInventories();
      }
    }, 1200);
    return () => window.clearTimeout(timeoutId);
  }, [refresh, refreshDeferredModelInventories]);

  useEffect(() => {
    if (!active || !modelSelectorOpen) return;
    refreshDeferredModelInventories();
  }, [active, modelSelectorOpen, refreshDeferredModelInventories]);

  useEffect(() => {
    // ChatPage no longer remounts on navigation, so re-check the handoff whenever
    // we return to /chat (e.g. from the training progress "compare in chat" action).
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
          const hasAppliedConfig = applyModelLoadConfigToRuntime(
            rememberedConfigFor(selection),
            { fromPersisted: true },
          );
          await selectModelRef.current({
            ...selection,
            ...(hasAppliedConfig ? { keepSpeculative: true } : {}),
            previousConfig,
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
    // Provides `active` to ChatRuntimeProvider (drops the message views/composers
    // while off-route, keeping the runtime alive) and to the compare chrome.
    <ChatActiveContext.Provider value={active}>
    <div className="flex min-h-0 min-w-0 flex-1 basis-0 bg-background overflow-hidden">
      {/* Portaled surfaces render to document.body, escaping the parent's hidden
          wrapper, so gate them on `active` to keep them off other tabs. */}
      {active && <GuidedTour {...tour.tourProps} />}
      {/* Single app-level mount for the Bypass permissions warning. It is driven
          by global store state, so it must live at one stable root (not inside a
          Composer) -- otherwise Compare mode's multiple composers would each
          render their own copy and the shared-composer menu would have none. It
          also portals to body, so gate it on `active` like the tour above. */}
      {active && <BypassPermissionsConfirmDialog />}
      <div className="relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden">
        <NativeModelDropOverlay state={nativeModelDropState} />
        {/* Fade under the top bar so messages dissolve as they scroll
            beneath it, instead of a hard cut. */}
        {view.mode !== "compare" && (
          <div
            aria-hidden
            className="chat-header-fade pointer-events-none absolute left-0 right-[10px] top-[calc(var(--studio-content-top-inset,0px)+var(--studio-chat-header-height,48px))] z-20 h-6 bg-gradient-to-b from-background to-transparent"
          />
        )}
        <div
          className={cn(
            "pointer-events-none absolute top-[var(--studio-content-top-inset,0px)] left-0 right-[10px] z-40 flex h-[var(--studio-chat-header-height,48px)] shrink-0 items-start bg-background pt-[var(--studio-chat-header-padding-top,11px)] pr-[calc(0.5rem+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]",
            isMobile
              ? "pl-12"
              : pinned
                ? "pl-2"
                : "pl-[calc(0.5rem+max(0px,var(--studio-mac-traffic-light-inset,0px)-var(--sidebar-width-icon,3rem)))]",
            view.mode === "compare" &&
              "right-[10px] left-auto w-auto bg-transparent pl-0 pr-[calc(0.5rem+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]",
          )}
        >
          <div className="pointer-events-auto flex items-center gap-1">
            {view.mode !== "compare" && (
              <ModelSelector
                models={models}
                loraModels={loraModels}
                externalModels={externalModels}
                value={inferenceParams.checkpoint}
                activeGgufVariant={activeGgufVariant}
                activeModelConfig={activeModelConfig}
                activeGgufContextLength={ggufContextLength}
                onValueChange={handleCheckpointChange}
                onEject={handleEject}
                onFoldersChange={refreshLocalModels}
                onPickLocalModel={isTauri ? chooseNativeModel : undefined}
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
            {incognito && view.mode === "single" && (
              <div className="flex h-[var(--studio-chat-control-height,34px)] shrink-0 items-center gap-1.5 self-center rounded-full bg-primary/10 px-2.5 font-medium text-ui-13 text-primary">
                <HugeiconsIcon
                  icon={BubbleChatTemporaryIcon}
                  strokeWidth={2}
                  className="size-3.5"
                />
                <span>Temporary</span>
              </div>
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
          <div className="pointer-events-auto ml-auto flex items-center gap-2">
            {view.mode === "single" && contextUsage ? (
              <ContextUsageBar
                used={contextUsage.totalTokens}
                // null on external providers; the bar handles that.
                total={ggufContextLength}
                cached={contextUsage.cachedTokens}
                cacheWrites={contextUsage.cacheWriteTokens}
                promptTokens={contextUsage.promptTokens}
                completionTokens={contextUsage.completionTokens}
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
                      "flex size-[var(--studio-chat-control-height,34px)] cursor-pointer items-center justify-center rounded-[12px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
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
            {!settingsOpen && (
              <Tooltip>
                <TooltipPrimitive.Trigger asChild={true}>
                  <button
                    type="button"
                    onClick={() => setSettingsOpen(true)}
                    className="flex size-[var(--studio-chat-control-height,34px)] translate-x-[2px] cursor-pointer items-center justify-center rounded-[12px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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

        {view.mode === "project" ? (
          <ProjectLanding
            key={view.projectId}
            projectId={view.projectId}
            projectName={currentProject?.name ?? "Project"}
            items={currentProjectItems}
          />
        ) : view.mode === "single" ? (
          // Keyed by project only (not thread / new-chat nonce) so switching threads or
          // starting a New Chat reuses the same provider and switches in place. This keeps
          // an in-flight generation streaming in the background (assistant-ui keeps every
          // alive thread's runtime mounted) instead of remounting the provider and cutting
          // it off; returning to that thread reattaches the live run rather than reloading
          // a half-saved one.
          <SingleContent
            key={view.projectId ?? "single"}
            threadId={view.threadId}
            newThreadNonce={view.newThreadNonce}
            projectId={view.projectId}
            artifact={selectedArtifact}
            artifactSurface={artifactSurface}
            onCloseArtifact={closeArtifactSurface}
          />
        ) : (
          <CompareContent
            key={view.pairId}
            pairId={view.pairId}
            projectId={view.projectId}
            models={models}
            loraModels={loraModels}
            externalModels={externalModels}
            onFoldersChange={refreshLocalModels}
            onModelsChange={refreshModelLists}
            deleteDisabled={modelOperationInProgress}
            onExitCompare={exitCompare}
          />
        )}

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
              nativeContextLength={ggufNativeContextLength}
              loadedContextLength={ggufContextLength}
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
