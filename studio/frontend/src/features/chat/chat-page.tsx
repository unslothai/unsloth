// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatSearch } from "@/app/routes/chat";
import {
  type DeletedModelRef,
  type ExternalModelOption,
  type LoraModelOption,
  type ModelOption,
  ModelSelector,
} from "@/components/assistant-ui/model-selector";
import { ProjectComposer, Thread } from "@/components/assistant-ui/thread";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { useSidebar } from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import {
  NativeModelChip,
  NativeModelDropOverlay,
  type NativeIntent,
  useChooseNativeModel,
  useNativeIntentStore,
  useNativeModelDrop,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import {
  Folder02Icon,
  FolderAddIcon,
  LayoutAlignRightIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState, useSearch } from "@tanstack/react-router";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import type { PanelImperativeHandle } from "react-resizable-panels";
import {
  type CSSProperties,
  type ReactElement,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "@/lib/toast";
import { listLocalModels } from "./api/chat-api";
import { ChatSettingsPanel } from "./chat-settings-sheet";
import { CopyableErrorChip } from "@/components/ui/copyable-error-chip";
import { ContextUsageBar } from "./components/context-usage-bar";
import { ModelLoadInlineStatus } from "./components/model-load-status";
import { ProjectSwitcher } from "./components/project-switcher";
import {
  buildExternalModelId,
  isExternalModelId,
  parseExternalModelId,
} from "./external-providers";
import { useChatModelRuntime } from "./hooks/use-chat-model-runtime";
import { useChatProjects } from "./hooks/use-chat-projects";
import {
  type SidebarItem,
  useChatSidebarItems,
} from "./hooks/use-chat-sidebar-items";
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
import { ChatRuntimeProvider } from "./runtime-provider";
import {
  type CompareHandle,
  type CompareHandles,
  CompareHandlesProvider,
  RegisterCompareHandle,
  SharedComposer,
} from "./shared-composer";
import {
  CHAT_CODE_TOOLS_ENABLED_KEY,
  CHAT_IMAGE_TOOLS_ENABLED_KEY,
  CHAT_TOOLS_ENABLED_KEY,
  CHAT_WEB_FETCH_TOOLS_ENABLED_KEY,
  loadOptionalBool,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import { buildChatTourSteps } from "./tour";
import { ArtifactSurface } from "./artifacts/artifact-surface";
import {
  clearAutoOpenedArtifacts,
  useChatArtifactsStore,
  useSelectedChatArtifact,
} from "./artifacts/store";
import type { ChatArtifact, ChatArtifactSurface } from "./artifacts/types";
import type { ChatView, MessageRecord } from "./types";
import {
  getStoredChatThread,
  isExpectedBackgroundChatStorageError,
  listStoredChatMessages,
  listStoredChatThreads,
} from "./utils/chat-history-storage";

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

function isAssistantLocalThreadId(
  threadId: string | null | undefined,
): boolean {
  return Boolean(threadId?.startsWith("__LOCALID_"));
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
  const showArtifactPanel = Boolean(
    artifact &&
      artifactSurface === "panel" &&
      (threadId
        ? !artifact.threadId || artifact.threadId === threadId
        : Boolean(newThreadNonce) ||
          Boolean(artifact.threadId && artifact.threadId === activeThreadId)),
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
 * Detect if this is a LoRA base-vs-fine-tuned compare.
 * Returns true when the loaded checkpoint is a LoRA — in that case
 * we use the fast simultaneous base/lora adapter-toggle path.
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
 * A single column in the compare layout. Hosts one ChatRuntimeProvider
 * and one Thread rendered with hideComposer — the composer is shared
 * across both panes and rendered outside the pane flex.
 *
 * Each pane is a flex item with `flex-1 basis-0 min-h-0 min-w-0` so on
 * mobile (flex-col) they share height equally, and on desktop (flex-row)
 * they share width equally. The `min-*` constraints are required for
 * the inner viewport to scroll internally instead of spilling into the
 * page.
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
 * Shared shell for both compare variants. A vertical flex column with
 * the two panes as siblings and the shared composer docked at the
 * bottom. On mobile the panes stack (flex-col); on desktop they sit
 * side by side (md:flex-row).
 *
 * Flex is used rather than CSS grid for the pane container so that
 * viewport sizing stays stable across viewport-size transitions. Grid
 * rows with 1fr were triggering resize thrash in assistant-ui's
 * autoscroll hook on breakpoint crossings, leaving it stuck in a
 * scroll-to-bottom loop.
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
  return (
    <CompareHandlesProvider handlesRef={handlesRef}>
      <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col">
        <div
          data-tour="chat-compare-view"
          className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col md:flex-row"
        >
          {children}
        </div>
        <div className="shrink-0 bg-background pl-5 pr-5 md:pr-[30px] pb-2 pt-1">
          <div className="mx-auto w-full max-w-[48rem]">{composer}</div>
          <p className="composer-footer-note">
            LLMs can make mistakes. Double-check responses.
          </p>
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

  useEffect(() => {
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
  }, [pairId]);

  return (
    <CompareShell
      handlesRef={handlesRef}
      composer={
        <SharedComposer
          handlesRef={handlesRef}
          onExitCompare={onExitCompare}
        />
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
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
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
            <div className="shrink-0 px-3 py-1.5 text-start md:text-end">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-primary">
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
 * Per-pane header rendered inside GeneralCompareContent. Contains the
 * model selector aligned with the global topbar height. The left pane
 * reserves room for the mobile sidebar trigger; the right pane reserves
 * room for the global settings button.
 */
function GeneralCompareHeader({
  models,
  loraModels,
  externalModels,
  value,
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
  onValueChange: (
    id: string,
    meta: { isLora: boolean; ggufVariant?: string },
  ) => void;
  onFoldersChange?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  side: "left" | "right";
}): ReactElement {
  return (
    <div
      className={cn(
        "flex h-[48px] shrink-0 items-start pt-[11px] gap-2 bg-background",
        side === "left" ? "pl-12 pr-3 md:pl-2" : "pl-3 pr-12",
      )}
    >
      <ModelSelector
        models={models}
        loraModels={loraModels}
        externalModels={externalModels}
        value={value}
        onValueChange={onValueChange}
        onFoldersChange={onFoldersChange}
        onModelsChange={onModelsChange}
        deleteDisabled={deleteDisabled}
        variant="ghost"
        className="max-w-[80%] !h-[34px]"
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
  }, [pairId]);

  return (
    <CompareShell
      handlesRef={handlesRef}
      composer={
        <SharedComposer
          handlesRef={handlesRef}
          model1={model1}
          model2={model2}
          onExitCompare={onExitCompare}
        />
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
              onValueChange={(id, meta) =>
                setModel1({
                  id,
                  isLora: meta.isLora,
                  ggufVariant: meta.ggufVariant,
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
              onValueChange={(id, meta) =>
                setModel2({
                  id,
                  isLora: meta.isLora,
                  ggufVariant: meta.ggufVariant,
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
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const initialActiveThreadRef = useRef<string | null>(null);
  const [projectTab, setProjectTab] = useState<"chats" | "sources">("chats");
  const [pendingNewThreadId, setPendingNewThreadId] = useState<string | null>(
    null,
  );
  const [newThreadNonce, setNewThreadNonce] = useState(() =>
    crypto.randomUUID(),
  );
  const [previews, setPreviews] = useState<
    Record<string, { snippet: string; date: string }>
  >({});

  useEffect(() => {
    initialActiveThreadRef.current =
      useChatRuntimeStore.getState().activeThreadId;
    useChatRuntimeStore.getState().setActiveThreadId(null);
    useChatRuntimeStore.getState().setContextUsage(null);
    setPendingNewThreadId(null);
    setNewThreadNonce(crypto.randomUUID());
  }, [projectId]);

  useEffect(() => {
    if (!activeThreadId) {
      setPendingNewThreadId(null);
      return;
    }
    if (activeThreadId === initialActiveThreadRef.current) {
      return;
    }
    setPendingNewThreadId(activeThreadId);
  }, [activeThreadId]);

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
          const messages = await listStoredChatMessages(item.id).catch(() => []);
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
          <div className="mx-auto flex w-full max-w-[48rem] flex-col pt-[120px] pb-14">
            <div className="mb-12 flex items-center gap-3">
              <HugeiconsIcon
                icon={Folder02Icon}
                strokeWidth={1.75}
                className="size-9 shrink-0 text-foreground"
              />
              <h1 className="truncate font-sans text-[30px] font-medium leading-tight tracking-normal text-foreground">
                {projectName}
              </h1>
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
                className="h-10 rounded-full border px-5 text-[14px] font-semibold transition-colors data-[active=true]:border-border data-[active=true]:bg-muted data-[active=true]:text-foreground data-[active=false]:border-transparent data-[active=false]:text-muted-foreground data-[active=false]:hover:bg-nav-surface-hover"
              >
                Chats
              </button>
              <button
                type="button"
                onClick={() => setProjectTab("sources")}
                data-active={projectTab === "sources"}
                className="h-10 rounded-full border px-5 text-[14px] font-semibold transition-colors data-[active=true]:border-border data-[active=true]:bg-muted data-[active=true]:text-foreground data-[active=false]:border-transparent data-[active=false]:text-muted-foreground data-[active=false]:hover:bg-nav-surface-hover"
              >
                Sources
              </button>
            </div>

            {projectTab === "sources" ? (
              <div className="mt-8 flex flex-col items-center justify-center gap-3 rounded-[16px] border border-dashed border-border/70 bg-muted/30 px-6 py-16 text-center">
                <span className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
                  <HugeiconsIcon
                    icon={FolderAddIcon}
                    strokeWidth={1.75}
                    className="size-6"
                  />
                </span>
                <div className="space-y-1">
                  <p className="text-[15px] font-semibold text-foreground">
                    Give this project context
                  </p>
                  <p className="max-w-sm text-sm text-muted-foreground">
                    Upload PDFs, documents, or other text. The model can
                    reference them in every chat in this project.
                  </p>
                </div>
                <Button type="button" className="mt-1" disabled>
                  Add sources
                </Button>
                <p className="text-[11px] text-muted-foreground">Coming soon</p>
              </div>
            ) : (
            <div className="mt-8 flex flex-col gap-1">
              {items.map((item) => {
                const preview = previews[item.id];
                return (
                  <button
                    key={`${item.type}:${item.id}`}
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
                    className="group flex min-h-[58px] w-full items-center gap-4 rounded-[10px] px-4 py-2 text-left transition-colors hover:bg-nav-surface-hover"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[15px] font-semibold leading-5 text-foreground">
                        {item.title}
                      </div>
                      {preview?.snippet ? (
                        <div className="mt-0.5 truncate text-[14px] leading-5 text-muted-foreground">
                          {preview.snippet}
                        </div>
                      ) : null}
                    </div>
                    <span className="shrink-0 text-[14px] text-muted-foreground">
                      {preview?.date ?? formatProjectChatDate(item.createdAt)}
                    </span>
                  </button>
                );
              })}
            </div>
            )}
          </div>
        </div>
      )}
    </ChatRuntimeProvider>
  );
}

export function ChatPage(): ReactElement {
  const search = useSearch({ from: "/chat" });
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isCurrentChatRoute = pathname.startsWith("/chat");

  const settingsOpen = useChatRuntimeStore((s) => s.settingsPanelOpen);
  const setSettingsOpen = useChatRuntimeStore((s) => s.setSettingsPanelOpen);
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
  }, [navigate, search.thread]);

  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [modelSelectorLocked, setModelSelectorLocked] = useState(false);
  const viewBeforeCompareRef = useRef<ChatSearch | null>(null);
  const inferenceParams = useChatRuntimeStore((state) => state.params);
  const setInferenceParams = useChatRuntimeStore((state) => state.setParams);
  const activeGgufVariant = useChatRuntimeStore(
    (state) => state.activeGgufVariant,
  );
  const ggufContextLength = useChatRuntimeStore(
    (state) => state.ggufContextLength,
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
  const isExternalModel = useMemo(
    () => isExternalModelId(inferenceParams.checkpoint),
    [inferenceParams.checkpoint],
  );
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
    // Per-provider default effort. Anthropic gets the highest available
    // level (xhigh on 4.6/4.7, high on 4.5) since Claude's adaptive
    // thinking adjusts cost per turn — sitting at the top of the dial
    // gives users the strongest answers and the model can still skip
    // thinking when the turn is trivial. OpenAI gets "high" by default
    // — the gpt-5.x reasoning models accept high across the board and
    // it's the right cost/quality sweet spot for Responses-API tools
    // (web search included). Everyone else gets "medium" as a balanced
    // default. Users can pick another level via the Think dropdown.
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
    // Kimi's k2.6/k2.5 default to thinking enabled on the server side
    // (per https://platform.kimi.ai/docs/models). Mirror that default
    // in the UI so the Think pill comes up clicked when the user picks
    // a Kimi model. The Search pill stays off by default; the mutual-
    // exclusion handlers in the composer flip the two when needed.
    const isKimi = provider?.providerType === "kimi";
    // Web search is on by default for the two providers we trust most
    // for it: Anthropic (web_search_20250305 server tool, structured
    // citations) and OpenAI (/v1/responses web_search, structured
    // citations). Other providers stay off-by-default — OpenRouter's
    // plugins shape and Kimi's $web_search builtin still work when the
    // user opts in via the pill, but they're a notch less reliable so
    // we don't pre-enable them.
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
      // External models never give us a local tool runtime (no
      // python sandbox), so `supportsTools` must be false. The three
      // `supportsBuiltin*` flags pick up the slack for providers that
      // run the tool server-side: `supportsBuiltinWebSearch` lights
      // up the Search pill (OpenAI / Anthropic / OpenRouter / Kimi),
      // `supportsBuiltinCodeExecution` lights up the Code pill
      // (Anthropic Claude 4.x and OpenAI gpt-5.5), and
      // `supportsBuiltinImageGeneration` lights up the Images pill
      // (OpenAI cloud Responses-API models only).
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
        const thread = await getStoredChatThread(search.thread).catch(() => null);
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
    if (view.threadId || view.newThreadNonce || !selectedArtifact) return;
    // view intentionally excludes __LOCALID_ threads (they fall through to
    // { mode: "single" } with no threadId/nonce).  Don't close an artifact
    // whose thread is the currently active local thread.
    if (
      selectedArtifact.threadId &&
      selectedArtifact.threadId === activeThreadId
    )
      return;
    closeArtifactSurface();
  }, [activeThreadId, closeArtifactSurface, selectedArtifact, view]);

  const hasActiveModel = Boolean(inferenceParams.checkpoint);
  const loadNativeModelIntent = useCallback(
    async (intent: NativeIntent, loadingDescription: string) => {
      const label =
        intent.path.displayLabel || intent.displayLabel || "Local GGUF model";
      await selectModel({
        id: label,
        nativePathToken: intent.path.token,
        isDownloaded: true,
        loadingDescription,
        forceReload: true,
        throwOnError: true,
      });
      useNativeIntentStore.getState().clearModelIntent(intent.id);
    },
    [selectModel],
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
    enabled: view.mode === "single",
    nativePathLeasesSupported,
    hasActiveModel,
    isModelLoading: Boolean(loadingModel) || modelLoading,
    onAutoLoad: handleNativeModelDropAutoLoad,
  });

  const handleCheckpointChange = useCallback(
    (
      value: string,
      meta?: {
        source?: string;
        isLora: boolean;
        ggufVariant?: string;
        isDownloaded?: boolean;
        expectedBytes?: number;
      },
    ) => {
      const store = useChatRuntimeStore.getState();
      const currentCheckpoint = store.params.checkpoint;
      const currentVariant = store.activeGgufVariant;
      if (
        !value ||
        (value === currentCheckpoint &&
          (meta?.ggufVariant ?? null) === (currentVariant ?? null))
      )
        return;
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
            isReasoningProvider:
              selectedProvider?.isReasoningModel === true,
            baseUrl: selectedProvider?.baseUrl ?? null,
          },
        );
        const preferredEffort = store.reasoningEffort;
        const effortLevels = reasoningCaps.reasoningEffortLevels;
        const clampedEffort = clampReasoningEffortToLevels(
          preferredEffort,
          effortLevels,
        );
        // Same per-provider default policy as the useEffect path above:
        // Anthropic picks the highest available level, OpenAI picks
        // "high", everyone else picks "medium".
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
        // Clear any cached router-picked openrouter/free model unless the
        // user is staying on openrouter/free — otherwise the chip would
        // keep showing a stale ":<chosen>" suffix from a previous model.
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
        // See sibling useEffect above: Kimi's k2.x default to thinking
        // enabled, so the Think pill comes up clicked. Search pill stays
        // off by default; mutual exclusion flips them via the composer.
        const isKimi = selectedProvider?.providerType === "kimi";
        // Mirror of sibling useEffect: Anthropic and OpenAI get Search
        // on-by-default since their server tools emit structured
        // citations end-to-end. OpenRouter and Kimi stay off-by-default.
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
          // Clear previous-model counters; the relaxed external-provider
          // render gate would otherwise show stale stats until the next
          // completion overwrites them.
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
          // External models have no local tool runtime → supportsTools
          // stays false. The three supportsBuiltin* flags carry the
          // server-side capability info for each pill:
          //   - Search → providerSupportsBuiltinWebSearch
          //   - Code   → providerSupportsBuiltinCodeExecution
          //              (Anthropic Claude 4.x + OpenAI gpt-5.5)
          //   - Images → providerSupportsBuiltinImageGeneration
          //              (OpenAI cloud Responses-API models)
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
        await selectModel({
          id: value,
          isLora: meta?.isLora,
          ggufVariant: meta?.ggufVariant,
          isDownloaded: meta?.isDownloaded,
          expectedBytes: meta?.expectedBytes,
        });
      })();
    },
    [
      activeThreadId,
      externalProvidersForChat,
      modelsFromStore,
      selectModel,
      view,
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
  const { setPinned, isMobile } = useSidebar();
  const openSidebar = useCallback(() => setPinned(true), [setPinned]);

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
    const saved = viewBeforeCompareRef.current;
    // No saved view (compare opened by direct URL); fall back to a fresh chat.
    if (!saved) {
      navigate({ to: "/chat" });
      return;
    }
    viewBeforeCompareRef.current = null;
    navigate({ to: "/chat", search: saved });
    // Restore usage from the last assistant message, but only if it
    // matches the currently active checkpoint. Without this guard the
    // relaxed render gate would show stale stats from another model.
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
          const usageModelId =
            (usage as { modelId?: unknown }).modelId;
          // Scope by modelId when present; reject if no active checkpoint
          // (model-scoped usage cannot be attributed to "nothing").
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
            // model the gateway actually picked once a stream completes
            // (chat-adapter latches `chunk.model` into the runtime store).
            // Render the chip as `openrouter:<short-chosen>` — drop the
            // redundant `/free` from the router id and the org prefix
            // from the chosen id (e.g.
            //   openrouter/free + inclusionai/ring-2.6-1t-20260508:free
            //     -> openrouter:ring-2.6-1t-20260508:free
            // ). The `:free` suffix on the chosen id already conveys
            // 'free model', so the leading `/free` is noise.
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

  useEffect(() => {
    if (getTrainingCompareHandoff()) return;
    void refresh();
    refreshLocalModels();
  }, [refresh, refreshLocalModels]);

  useEffect(() => {
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
        if (targetLora) {
          console.info("[chat-handoff] loading lora", {
            id: targetLora.id,
            baseModel: targetLora.baseModel,
          });
          await selectModelRef.current({ id: targetLora.id, isLora: true });
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
          await selectModelRef.current({
            id: handoff.baseModel,
            isLora: false,
          });
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
  }, []);

  const tourSteps = useMemo(
    () =>
      // eslint-disable-next-line react-hooks/refs -- buildChatTourSteps stores callbacks without invoking them during render.
      buildChatTourSteps({
        canCompare,
        openModelSelector,
        closeModelSelector,
        openSettings,
        closeSettings,
        openSidebar,
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
      openSidebar,
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

  if (!isCurrentChatRoute) {
    return (
      <div className="flex min-h-0 min-w-0 flex-1 basis-0 bg-background" />
    );
  }

  return (
    <div className="flex min-h-0 min-w-0 flex-1 basis-0 bg-background overflow-hidden">
      <GuidedTour {...tour.tourProps} />
      <div className="relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden">
        <NativeModelDropOverlay state={nativeModelDropState} />
        {/* Bottom fade under the top bar so messages dissolve as they scroll
            beneath it (Gemini / unsloth-sidebar style), instead of a hard cut. */}
        {view.mode !== "compare" && (
          <div
            aria-hidden
            className="pointer-events-none absolute left-0 right-[10px] top-[48px] z-20 h-6 bg-gradient-to-b from-background to-transparent"
          />
        )}
        <div
          className={cn(
            "absolute top-0 left-0 right-[10px] z-30 flex h-[48px] shrink-0 items-start pt-[11px] pr-2 bg-background",
            isMobile ? "pl-12 pr-1.5" : "pl-2",
            view.mode === "compare" &&
              "right-[10px] left-auto w-auto bg-transparent pl-0 pr-2",
          )}
        >
          <div className="flex items-center gap-1">
            {view.mode !== "compare" && (
              <ModelSelector
                models={models}
                loraModels={loraModels}
                externalModels={externalModels}
                value={inferenceParams.checkpoint}
                activeGgufVariant={activeGgufVariant}
                onValueChange={handleCheckpointChange}
                onEject={handleEject}
                onFoldersChange={refreshLocalModels}
                onPickLocalModel={isTauri ? chooseNativeModel : undefined}
                onModelsChange={refreshModelLists}
                deleteDisabled={modelOperationInProgress}
                variant="ghost"
                open={modelSelectorOpen}
                onOpenChange={handleModelSelectorOpenChange}
                triggerDataTour="chat-model-selector"
                contentDataTour="chat-model-selector-popover"
                showCloudIndicator={isExternalModel}
                className="max-w-[62vw] !pr-3 sm:max-w-none !h-[34px]"
              />
            )}
            {view.mode !== "compare" && currentProjectId && (
              <nav
                aria-label="Project location"
                className="flex h-[34px] min-w-0 items-center gap-1.5 self-center text-[13.5px] tracking-nav text-muted-foreground"
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
                    <span className="shrink-0" aria-hidden>
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
                onLoad={(selection) => selectModel(selection)}
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
          <div className="ml-auto flex items-center gap-2">
            {view.mode === "single" && contextUsage ? (
              <ContextUsageBar
                used={contextUsage.totalTokens}
                // null on external providers; the bar handles that.
                total={ggufContextLength}
                cached={contextUsage.cachedTokens}
                cacheWrites={contextUsage.cacheWriteTokens}
                promptTokens={contextUsage.promptTokens}
                completionTokens={contextUsage.completionTokens}
                className="h-[34px]"
              />
            ) : null}
            {!settingsOpen && (
              <Tooltip>
                <TooltipPrimitive.Trigger asChild={true}>
                  <button
                    type="button"
                    onClick={() => setSettingsOpen(true)}
                    className="flex h-[34px] w-[34px] translate-x-[2px] cursor-pointer items-center justify-center rounded-[12px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label="Open run settings"
                    data-tour="chat-settings"
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
          <SingleContent
            key={view.threadId ?? view.newThreadNonce ?? "single"}
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

        {showArtifactOverlay && selectedArtifact ? (
          <ArtifactSurface
            artifact={selectedArtifact}
            variant="overlay"
            onClose={closeArtifactSurface}
          />
        ) : null}
      </div>

      <ChatSettingsPanel
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        params={inferenceParams}
        onParamsChange={setInferenceParams}
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
        onReloadModel={() => {
          const state = useChatRuntimeStore.getState();
          if (state.params.checkpoint) {
            selectModel({
              id: state.params.checkpoint,
              ggufVariant: state.activeGgufVariant ?? undefined,
              forceReload: true,
              isDownloaded: true,
              loadingDescription: "Reloading with updated chat template.",
            });
          }
        }}
      />
    </div>
  );
}
