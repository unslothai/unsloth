// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type LoraModelOption,
  type ModelOption,
  ModelSelector,
} from "@/components/assistant-ui/model-selector";
import { Thread } from "@/components/assistant-ui/thread";
import { cn } from "@/lib/utils";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { useSidebar } from "@/components/ui/sidebar";
import { Settings05Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { useNavigate, useSearch } from "@tanstack/react-router";
import {
  type ReactElement,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import type { ChatSearch } from "@/app/routes/chat";
import { listLocalModels } from "./api/chat-api";
import { ChatSettingsPanel } from "./chat-settings-sheet";
import { ContextUsageBar } from "./components/context-usage-bar";
import { ModelLoadInlineStatus } from "./components/model-load-status";
import { db } from "./db";
import { useChatModelRuntime } from "./hooks/use-chat-model-runtime";
import {
  clearTrainingCompareHandoff,
  getTrainingCompareHandoff,
} from "./lib/training-compare-handoff";
import { ChatRuntimeProvider } from "./runtime-provider";
import {
  type CompareHandle,
  type CompareHandles,
  CompareHandlesProvider,
  RegisterCompareHandle,
  SharedComposer,
} from "./shared-composer";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import { buildChatTourSteps } from "./tour";
import type { ChatView, MessageRecord } from "./types";

type LoraCandidate = {
  id: string;
  baseModel: string;
  updatedAt?: number;
  exportType?: "lora" | "merged" | "gguf";
};

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
      <div className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden">
        <Thread hideWelcome={Boolean(threadId)} targetThreadId={threadId} />
      </div>
    </ChatRuntimeProvider>
  );
});

type CompareModelSelection = {
  id: string;
  isLora: boolean;
  ggufVariant?: string;
};

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
  models,
  loraModels,
  onFoldersChange,
}: {
  pairId: string;
  models: ModelOption[];
  loraModels: LoraModelOption[];
  onFoldersChange?: () => void;
}): ReactElement {
  const isLoraCompare = useIsLoraCompare();

  return isLoraCompare ? (
    <LoraCompareContent pairId={pairId} />
  ) : (
    <GeneralCompareContent
      pairId={pairId}
      models={models}
      loraModels={loraModels}
      onFoldersChange={onFoldersChange}
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
  initialThreadId,
  handleName,
  header,
  borderClassName,
}: {
  modelType: "base" | "lora" | "model1" | "model2";
  pairId: string;
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
        <div className="shrink-0 bg-background px-5 pb-2 pt-1">
          <div className="mx-auto w-full max-w-[44rem]">{composer}</div>
          <p className="mt-1.5 text-center text-[11px] text-muted-foreground">
            LLMs can make mistakes. Double-check all responses.
          </p>
        </div>
      </div>
    </CompareHandlesProvider>
  );
}

/** Fast path: same model, adapter on/off, simultaneous generation. */
const LoraCompareContent = memo(function LoraCompareContent({
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
        if (!isActive) return;
        setBaseThreadId(threads.find((t) => t.modelType === "base")?.id);
        setLoraThreadId(threads.find((t) => t.modelType === "lora")?.id);
      });
    return () => {
      isActive = false;
    };
  }, [pairId]);

  return (
    <CompareShell
      handlesRef={handlesRef}
      composer={<SharedComposer handlesRef={handlesRef} />}
    >
      <>
        <ComparePane
          modelType="base"
          pairId={pairId}
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
  value,
  onValueChange,
  onFoldersChange,
  side,
}: {
  models: ModelOption[];
  loraModels: LoraModelOption[];
  value: string;
  onValueChange: (
    id: string,
    meta: { isLora: boolean; ggufVariant?: string },
  ) => void;
  onFoldersChange?: () => void;
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
        value={value}
        onValueChange={onValueChange}
        onFoldersChange={onFoldersChange}
        variant="ghost"
        className="max-w-[80%] !h-[34px]"
      />
    </div>
  );
}

/** General path: any two models, sequential load → generate. */
const GeneralCompareContent = memo(function GeneralCompareContent({
  pairId,
  models,
  loraModels,
  onFoldersChange,
}: {
  pairId: string;
  models: ModelOption[];
  loraModels: LoraModelOption[];
  onFoldersChange?: () => void;
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

  useEffect(() => {
    let isActive = true;
    db.threads
      .where("pairId")
      .equals(pairId)
      .toArray()
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
        />
      }
    >
      <>
        <ComparePane
          modelType="model1"
          pairId={pairId}
          initialThreadId={model1ThreadId}
          handleName="model1"
          header={
            <GeneralCompareHeader
              side="left"
              models={models}
              loraModels={loraModels}
              value={model1.id}
              onValueChange={(id, meta) =>
                setModel1({
                  id,
                  isLora: meta.isLora,
                  ggufVariant: meta.ggufVariant,
                })
              }
              onFoldersChange={onFoldersChange}
            />
          }
        />
        <ComparePane
          modelType="model2"
          pairId={pairId}
          initialThreadId={model2ThreadId}
          handleName="model2"
          borderClassName="border-t border-sidebar-border md:border-t-0 md:border-l"
          header={
            <GeneralCompareHeader
              side="right"
              models={models}
              loraModels={loraModels}
              value={model2.id}
              onValueChange={(id, meta) =>
                setModel2({
                  id,
                  isLora: meta.isLora,
                  ggufVariant: meta.ggufVariant,
                })
              }
              onFoldersChange={onFoldersChange}
            />
          }
        />
      </>
    </CompareShell>
  );
});

export function ChatPage(): ReactElement {
  const search = useSearch({ from: "/chat" });
  const navigate = useNavigate();

  const settingsOpen = useChatRuntimeStore((s) => s.settingsPanelOpen);
  const setSettingsOpen = useChatRuntimeStore((s) => s.setSettingsPanelOpen);

  useEffect(() => {
    return () => setSettingsOpen(false);
  }, [setSettingsOpen]);

  useEffect(() => {
    const threadId = search.thread;
    if (!threadId) return;

    let canceled = false;
    void db.threads
      .get(threadId)
      .then((thread) => {
        if (canceled || thread) return;
        useChatRuntimeStore.getState().setActiveThreadId(null);
        toast.info("Chat not found", {
          description: "That thread no longer exists, so we opened a new chat.",
        });
        navigate({
          to: "/chat",
          search: { new: crypto.randomUUID() },
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
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const {
    refresh,
    selectModel,
    ejectModel,
    cancelLoading,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  } = useChatModelRuntime();
  const refreshRef = useRef(refresh);
  const selectModelRef = useRef(selectModel);

  useEffect(() => {
    refreshRef.current = refresh;
    selectModelRef.current = selectModel;
  }, [refresh, selectModel]);
  const canCompare = useMemo(() => {
    return Boolean(inferenceParams.checkpoint);
  }, [inferenceParams.checkpoint]);

  // Derive view from URL search params
  const view = useMemo<ChatView>(() => {
    if (search.compare) {
      return {
        mode: "compare",
        pairId: search.compare,
      };
    }
    if (search.thread) {
      return { mode: "single", threadId: search.thread };
    }
    if (activeThreadId && !activeThreadId.startsWith("__LOCALID_")) {
      return { mode: "single", threadId: activeThreadId };
    }
    if (search.new) {
      return { mode: "single", newThreadNonce: search.new };
    }
    return { mode: "single" };
  }, [search.thread, search.compare, search.new, activeThreadId]);

  const handleCheckpointChange = useCallback(
    (
      value: string,
      meta?: {
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
    [activeThreadId, modelsFromStore, selectModel, view],
  );
  const handleEject = useCallback(() => {
    void ejectModel();
  }, [ejectModel]);

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
    navigate({ to: "/chat", search: { compare: crypto.randomUUID() } });
  }, [navigate, search]);

  const exitCompare = useCallback(() => {
    const saved = viewBeforeCompareRef.current;
    if (!saved) return;
    viewBeforeCompareRef.current = null;
    navigate({ to: "/chat", search: saved });
    // Restore context usage from the active thread's last assistant message.
    const threadId =
      saved.thread ?? useChatRuntimeStore.getState().activeThreadId;
    if (threadId) {
      void db.messages
        .where("threadId")
        .equals(threadId)
        .reverse()
        .first()
        .then((msg) => {
          const metadata = msg?.metadata as Record<string, unknown> | undefined;
          const usage = metadata?.contextUsage as ReturnType<
            typeof useChatRuntimeStore.getState
          >["contextUsage"];
          if (usage) useChatRuntimeStore.getState().setContextUsage(usage);
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
    <div className="flex min-h-0 min-w-0 flex-1 basis-0 bg-background overflow-hidden">
      <GuidedTour {...tour.tourProps} />
      <div className="relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden">
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
                value={inferenceParams.checkpoint}
                activeGgufVariant={activeGgufVariant}
                onValueChange={handleCheckpointChange}
                onEject={handleEject}
                onFoldersChange={refreshLocalModels}
                variant="ghost"
                open={modelSelectorOpen}
                onOpenChange={handleModelSelectorOpenChange}
                triggerDataTour="chat-model-selector"
                contentDataTour="chat-model-selector-popover"
                className="max-w-[62vw] sm:max-w-none !h-[34px]"
              />
            )}
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
          </div>
          {modelsError && (
            <div className="ml-2 text-xs text-destructive truncate max-w-[28rem]">
              {modelsError}
            </div>
          )}
          <div className="ml-auto flex items-center gap-2">
            {view.mode === "single" && ggufContextLength && contextUsage ? (
              <ContextUsageBar
                used={contextUsage.totalTokens}
                total={ggufContextLength}
                cached={contextUsage.cachedTokens}
                promptTokens={contextUsage.promptTokens}
                completionTokens={contextUsage.completionTokens}
                className="h-[34px]"
              />
            ) : null}
            {!settingsOpen && (
              <Tooltip>
                <TooltipPrimitive.Trigger asChild>
                  <button
                    type="button"
                    onClick={() => setSettingsOpen(true)}
                    className="flex h-[34px] w-[34px] items-center justify-center rounded-[8px] text-[#383835] dark:text-[#c7c7c4] transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label="Open configuration"
                    data-tour="chat-settings"
                  >
                    <HugeiconsIcon icon={Settings05Icon} className="size-5" />
                  </button>
                </TooltipPrimitive.Trigger>
                <TooltipContent side="bottom" sideOffset={6}>
                  Open configuration
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>

        {view.mode === "single" ? (
          <SingleContent
            key={view.threadId ?? "single"}
            threadId={view.threadId}
            newThreadNonce={view.newThreadNonce}
          />
        ) : (
          <CompareContent
            key={view.pairId}
            pairId={view.pairId}
            models={models}
            loraModels={loraModels}
            onFoldersChange={refreshLocalModels}
          />
        )}
      </div>

      <ChatSettingsPanel
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        params={inferenceParams}
        onParamsChange={setInferenceParams}
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
