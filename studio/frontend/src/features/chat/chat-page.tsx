import {
  type LoraModelOption,
  type ModelOption,
  ModelSelector,
} from "@/components/assistant-ui/model-selector";
import { Thread } from "@/components/assistant-ui/thread";
import { Button } from "@/components/ui/button";
import {
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
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
import { ChatSettingsPanel } from "./chat-settings-sheet";
import { db } from "./db";
import { useChatModelRuntime } from "./hooks/use-chat-model-runtime";
import { ChatRuntimeProvider } from "./runtime-provider";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import {
  type CompareHandle,
  CompareHandlesProvider,
  RegisterCompareHandle,
  SharedComposer,
} from "./shared-composer";
import { ThreadSidebar } from "./thread-sidebar";
import type { ChatView } from "./types";

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
        <div className="grid min-h-0 flex-1 grid-cols-2 px-0">
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
          <div className="flex min-h-0 flex-col">
            <div className="text-end px-3 py-1.5">
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
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  return (
    <div
      className="group shrink-0 h-full"
      data-state={state}
      data-collapsible={collapsed ? "offcanvas" : ""}
      data-side={side}
    >
      <aside
        data-sidebar="sidebar"
        className={cn(
          "bg-sidebar text-sidebar-foreground h-full overflow-hidden rounded-2xl corner-squircle transition-[width] duration-200 ease-linear",
          !collapsed &&
            (side === "left"
              ? "border-r border-0 border-sidebar-border"
              : "border-l border-0 border-sidebar-border"),
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
  const inferenceParams = useChatRuntimeStore((state) => state.params);
  const setInferenceParams = useChatRuntimeStore((state) => state.setParams);
  const autoTitle = useChatRuntimeStore((state) => state.autoTitle);
  const setAutoTitle = useChatRuntimeStore((state) => state.setAutoTitle);
  const modelsFromStore = useChatRuntimeStore((state) => state.models);
  const lorasFromStore = useChatRuntimeStore((state) => state.loras);
  const modelsError = useChatRuntimeStore((state) => state.modelsError);
  const { refresh, selectModel, ejectModel } = useChatModelRuntime();
  const canCompare = useMemo(() => {
    const selected = inferenceParams.checkpoint;
    if (!selected) return false;
    return lorasFromStore.some((lora) => lora.id === selected);
  }, [inferenceParams.checkpoint, lorasFromStore]);

  const handleCheckpointChange = useCallback(
    (value: string, meta?: { isLora: boolean }) => {
      void selectModel({ id: value, isLora: meta?.isLora });
    },
    [selectModel],
  );
  const handleEject = useCallback(() => {
    void ejectModel();
  }, [ejectModel]);
  const handleNewThread = useCallback(
    () => setView({ mode: "single", newThreadNonce: crypto.randomUUID() }),
    [],
  );
  const handleNewCompare = useCallback(
    () => setView({ mode: "compare", pairId: crypto.randomUUID() }),
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
      })),
    [lorasFromStore],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <div className="h-[calc(100vh-4rem)] bg-background overflow-hidden">
    <SidebarProvider
      defaultOpen={true}
      className="!min-h-0 h-full max-w-7xl mx-auto px-4"
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
          onSelect={setView}
          onNewThread={handleNewThread}
          onNewCompare={handleNewCompare}
          showCompare={canCompare}
        />
      </InlineSidebar>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex h-11 shrink-0 items-center px-2">
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
              onValueChange={handleCheckpointChange}
              onEject={handleEject}
              variant="ghost"
            />
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
