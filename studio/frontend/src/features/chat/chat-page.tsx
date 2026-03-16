// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type LoraModelOption,
  type ModelOption,
  ModelSelector,
} from "@/components/assistant-ui/model-selector";
import { Thread } from "@/components/assistant-ui/thread";
import { Button } from "@/components/ui/button";
import { SidebarProvider, SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  ColumnInsertIcon,
  PencilEdit02Icon,
  Settings04Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  type ReactElement,
  type ReactNode,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { ChatSettingsPanel } from "./chat-settings-sheet";
import { ModelLoadInlineStatus } from "./components/model-load-status";
import { db } from "./db";
import { useChatModelRuntime } from "./hooks/use-chat-model-runtime";
import {
  clearTrainingCompareHandoff,
  getTrainingCompareHandoff,
} from "./lib/training-compare-handoff";
import { ChatRuntimeProvider } from "./runtime-provider";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import {
  type CompareHandle,
  CompareHandlesProvider,
  RegisterCompareHandle,
  SharedComposer,
} from "./shared-composer";
import { ThreadSidebar } from "./thread-sidebar";
import type { ChatView, MessageRecord } from "./types";
import { buildChatTourSteps } from "./tour";

type LoraCandidate = {
  id: string;
  baseModel: string;
  updatedAt?: number;
};

function normalizeModelRef(value: string | null | undefined): string {
  return value?.trim().toLowerCase() ?? "";
}

function pickBestLoraForBase(
  loras: LoraCandidate[],
  baseModel: string | null,
): LoraCandidate | null {
  if (loras.length === 0) return null;
  const sorted = [...loras].sort(
    (a, b) => (b.updatedAt ?? -1) - (a.updatedAt ?? -1),
  );
  const normalizedBase = normalizeModelRef(baseModel);
  if (!normalizedBase) return sorted[0];

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
  return partial ?? sorted[0];
}

function messageHasImage(message: MessageRecord): boolean {
  const contentParts = Array.isArray(message.content) ? message.content : [];
  if (contentParts.some((part) => part.type === "image")) {
    return true;
  }
  const attachments = Array.isArray(message.attachments) ? message.attachments : [];
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

const SingleContent = memo(function SingleContent({
  threadId,
  newThreadNonce,
}: { threadId?: string; newThreadNonce?: string }): ReactElement {
  return (
    <ChatRuntimeProvider
      modelType="base"
      initialThreadId={threadId}
      newThreadNonce={newThreadNonce}
    >
      <div className="min-h-0 flex-1">
        <Thread />
      </div>
    </ChatRuntimeProvider>
  );
});

const CompareContent = memo(function CompareContent({
  pairId,
}: { pairId: string }): ReactElement {
  const handlesRef = useRef<Record<string, CompareHandle>>({});
  const [baseThreadId, setBaseThreadId] = useState<string>();
  const [loraThreadId, setLoraThreadId] = useState<string>();

  useEffect(() => {
    let isActive = true;
    db.threads
      .where("pairId")
      .equals(pairId)
      .toArray()
      .then((threads) => {
        if (!isActive) {
          return;
        }
        setBaseThreadId(threads.find((t) => t.modelType === "base")?.id);
        setLoraThreadId(threads.find((t) => t.modelType === "lora")?.id);
      });
    return () => {
      isActive = false;
    };
  }, [pairId]);

  return (
    <CompareHandlesProvider handlesRef={handlesRef}>
      <div className="flex min-h-0 flex-1 flex-col">
        <div
          data-tour="chat-compare-view"
          className="grid min-h-0 flex-1 grid-cols-1 px-0 md:grid-cols-2"
        >
          <div className="flex min-h-0 flex-col">
            <div className="px-3 py-1.5">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Base Model
              </span>
            </div>
            <div className="min-h-0 flex-1">
              <ChatRuntimeProvider
                modelType="base"
                pairId={pairId}
                initialThreadId={baseThreadId}
              >
                <RegisterCompareHandle name="base" />
                <Thread hideComposer={true} hideWelcome={true} />
              </ChatRuntimeProvider>
            </div>
          </div>
          <div className="flex min-h-0 flex-col border-t border-border/60 md:border-t-0 md:border-l">
            <div className="px-3 py-1.5 text-start md:text-end">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-primary">
                Fine-tuned (LoRA)
              </span>
            </div>
            <div className="min-h-0 flex-1">
              <ChatRuntimeProvider
                modelType="lora"
                pairId={pairId}
                initialThreadId={loraThreadId}
              >
                <RegisterCompareHandle name="lora" />
                <Thread hideComposer={true} hideWelcome={true} />
              </ChatRuntimeProvider>
            </div>
          </div>
        </div>
        <div className="mx-auto w-full max-w-4xl px-4 py-4">
          <SharedComposer handlesRef={handlesRef} />
        </div>
      </div>
    </CompareHandlesProvider>
  );
});

function InlineSidebar({
  children,
  side = "left",
}: {
  children: ReactNode;
  side?: "left" | "right";
}) {
  const { state, isMobile, openMobile, setOpenMobile } = useSidebar();
  const collapsed = state === "collapsed";

  if (isMobile) {
    return (
      <Sheet open={openMobile} onOpenChange={setOpenMobile}>
        <SheetContent side={side} className="w-[18rem] p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Chat sidebar</SheetTitle>
            <SheetDescription>Chat threads and actions</SheetDescription>
          </SheetHeader>
          <div className="h-full overflow-auto">{children}</div>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <div
      className="group shrink-0 h-full pb-3.5"
      data-state={state}
      data-collapsible={collapsed ? "offcanvas" : ""}
      data-side={side}
    >
      <aside
        data-sidebar="sidebar"
        className={cn(
          "bg-muted/70 text-sidebar-foreground h-full overflow-hidden rounded-2xl corner-squircle transition-[width] duration-200 ease-linear",
          !collapsed &&
          side === "right" && "border-l border-sidebar-border/70",
          collapsed ? "w-0" : "w-(--sidebar-width)",
        )}
      >
        <div className="flex h-full w-(--sidebar-width) flex-col">
          {children}
        </div>
      </aside>
    </div>
  );
}

function TopBarActions({
  onNewThread,
  onNewCompare,
  showCompare,
}: { onNewThread: () => void; onNewCompare: () => void; showCompare: boolean }) {
  const { state } = useSidebar();
  if (state !== "collapsed") {
    return null;
  }
  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button variant="ghost" size="icon-sm" onClick={onNewThread}>
            <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={2} />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">New Chat</TooltipContent>
      </Tooltip>
      {showCompare ? (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <Button variant="ghost" size="icon-sm" onClick={onNewCompare}>
              <HugeiconsIcon icon={ColumnInsertIcon} strokeWidth={2} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Compare</TooltipContent>
        </Tooltip>
      ) : null}
    </>
  );
}

export function ChatPage(): ReactElement {
  const [view, setView] = useState<ChatView>({
    mode: "single",
    newThreadNonce: crypto.randomUUID(),
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [modelSelectorLocked, setModelSelectorLocked] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [viewBeforeCompare, setViewBeforeCompare] = useState<ChatView | null>(
    null,
  );
  const inferenceParams = useChatRuntimeStore((state) => state.params);
  const setInferenceParams = useChatRuntimeStore((state) => state.setParams);
  const activeGgufVariant = useChatRuntimeStore((state) => state.activeGgufVariant);
  const autoTitle = useChatRuntimeStore((state) => state.autoTitle);
  const setAutoTitle = useChatRuntimeStore((state) => state.setAutoTitle);
  const modelsFromStore = useChatRuntimeStore((state) => state.models);
  const lorasFromStore = useChatRuntimeStore((state) => state.loras);
  const modelsError = useChatRuntimeStore((state) => state.modelsError);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const {
    refresh,
    selectModel,
    ejectModel,
    cancelLoading,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  } =
    useChatModelRuntime();
  const refreshRef = useRef(refresh);
  const selectModelRef = useRef(selectModel);

  useEffect(() => {
    refreshRef.current = refresh;
    selectModelRef.current = selectModel;
  }, [refresh, selectModel]);
  const canCompare = useMemo(() => {
    const selected = inferenceParams.checkpoint;
    if (!selected) return false;
    return lorasFromStore.some((lora) => lora.id === selected);
  }, [inferenceParams.checkpoint, lorasFromStore]);

  const handleCheckpointChange = useCallback(
    (value: string, meta?: { isLora: boolean; ggufVariant?: string; isDownloaded?: boolean; expectedBytes?: number }) => {
      const store = useChatRuntimeStore.getState();
      const currentCheckpoint = store.params.checkpoint;
      const currentVariant = store.activeGgufVariant;
      if (!value || (value === currentCheckpoint && (meta?.ggufVariant ?? null) === (currentVariant ?? null))) return;
      void (async () => {
        let showImageCompatibilityWarning = false;
        if (view.mode === "single" && activeThreadId) {
          const thread = await db.threads.get(activeThreadId);
          if (thread?.modelId && thread.modelId !== value) {
            const messages = await db.messages
              .where("threadId")
              .equals(activeThreadId)
              .toArray();
            if (messages.length > 0) {
              const hasImage = messages.some(messageHasImage);
              const targetModel = modelsFromStore.find((model) => model.id === value);
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
    [activeThreadId, modelsFromStore, selectModel, view],
  );
  const handleEject = useCallback(() => {
    void ejectModel();
  }, [ejectModel]);
  const handleNewThread = useCallback(
    () => {
      useChatRuntimeStore.getState().setActiveThreadId(null);
      setView({ mode: "single", newThreadNonce: crypto.randomUUID() });
    },
    [],
  );
  const handleNewCompare = useCallback(
    () => setView({ mode: "compare", pairId: crypto.randomUUID() }),
    [],
  );

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
  const openSettings = useCallback(() => setSettingsOpen(true), []);
  const closeSettings = useCallback(() => setSettingsOpen(false), []);
  const openSidebar = useCallback(() => setSidebarOpen(true), []);

  const enterCompare = useCallback(() => {
    setViewBeforeCompare((prev) => prev ?? view);
    setView({ mode: "compare", pairId: crypto.randomUUID() });
  }, [view]);

  const exitCompare = useCallback(() => {
    if (!viewBeforeCompare) return;
    setView(viewBeforeCompare);
    setViewBeforeCompare(null);
  }, [viewBeforeCompare]);

  const handleThreadSelect = useCallback(
    (nextView: ChatView) => {
      setView(nextView);
    },
    [],
  );

  const models = useMemo<ModelOption[]>(
    () =>
      modelsFromStore.map((model) => ({
        id: model.id,
        name: model.name,
        description: model.description,
      })),
    [modelsFromStore],
  );

  const loraModels = useMemo<LoraModelOption[]>(
    () =>
      lorasFromStore.map((lora) => ({
        id: lora.id,
        name: lora.name,
        baseModel: lora.baseModel,
        updatedAt: lora.updatedAt,
        source: lora.source,
        exportType: lora.exportType,
      })),
    [lorasFromStore],
  );

  useEffect(() => {
    if (getTrainingCompareHandoff()) return;
    void refresh();
  }, [refresh]);

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
          setView({ mode: "compare", pairId: crypto.randomUUID() });
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
          await selectModelRef.current({ id: handoff.baseModel, isLora: false });
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

  return (
    <div className="h-[calc(100dvh-4rem)] bg-background overflow-hidden">
      <GuidedTour {...tour.tourProps} />
      <SidebarProvider
        defaultOpen={true}
        open={sidebarOpen}
        onOpenChange={setSidebarOpen}
        className="!min-h-0 h-full w-full max-w-7xl mx-auto px-2 sm:px-4"
        style={
          {
            "--sidebar-width": "14rem",
            "--sidebar-width-icon": "3rem",
          } as CSSProperties
        }
      >
        <InlineSidebar>
          <ThreadSidebar
            view={view}
            onSelect={handleThreadSelect}
            onNewThread={handleNewThread}
            onNewCompare={handleNewCompare}
            showCompare={canCompare}
          />
        </InlineSidebar>

        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="flex h-11 shrink-0 items-center px-1.5 sm:px-2">
            <div className="flex items-center gap-1">
              <SidebarTrigger />
              <TopBarActions
                onNewThread={handleNewThread}
                onNewCompare={handleNewCompare}
                showCompare={canCompare}
              />
              <ModelSelector
                models={models}
                loraModels={loraModels}
                value={inferenceParams.checkpoint}
                activeGgufVariant={activeGgufVariant}
                onValueChange={handleCheckpointChange}
                onEject={handleEject}
                variant="ghost"
                open={modelSelectorOpen}
                onOpenChange={handleModelSelectorOpenChange}
                triggerDataTour="chat-model-selector"
                contentDataTour="chat-model-selector-popover"
                className="max-w-[62vw] sm:max-w-none"
              />
              {loadingModel && loadToastDismissed ? (
                <ModelLoadInlineStatus
                  label={
                    loadProgress?.phase === "starting"
                      ? "Starting model…"
                      : loadingModel.isDownloaded
                        ? "Loading model…"
                        : "Downloading model…"
                  }
                  title={loadingModel.isDownloaded
                    ? `Loading ${loadingModel.displayName} from cache.`
                    : `Loading ${loadingModel.displayName}. This may include downloading.`}
                  progressPercent={loadProgress?.percent}
                  progressLabel={loadProgress?.label}
                  onStop={cancelLoading}
                />
              ) : null}
            </div>
            {modelsError && (
              <div className="ml-2 text-xs text-destructive truncate max-w-[28rem]">
                {modelsError}
              </div>
            )}
            <div className="flex-1" />
            <button
              type="button"
              onClick={() => setSettingsOpen((o) => !o)}
              className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              title="Inference settings"
              data-tour="chat-settings"
            >
              <HugeiconsIcon icon={Settings04Icon} className="size-5" />
            </button>
          </div>

          {view.mode === "single" ? (
            <SingleContent
              key={view.threadId ?? view.newThreadNonce ?? "new"}
              threadId={view.threadId}
              newThreadNonce={view.newThreadNonce}
            />
          ) : (
            <CompareContent key={view.pairId} pairId={view.pairId} />
          )}
        </div>

        <ChatSettingsPanel
          open={settingsOpen}
          params={inferenceParams}
          onParamsChange={setInferenceParams}
          autoTitle={autoTitle}
          onAutoTitleChange={setAutoTitle}
        />
      </SidebarProvider>
    </div>
  );
}
