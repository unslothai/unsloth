import {
  type ModelOption,
  ModelSelector,
} from "@/components/assistant-ui/model-selector";
import { Thread } from "@/components/assistant-ui/thread";
import {
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { Settings04Icon } from "@hugeicons/core-free-icons";
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
import {
  ChatSettingsPanel,
  type InferenceParams,
  defaultInferenceParams,
} from "./chat-settings-sheet";
import { db } from "./db";
import { ChatRuntimeProvider } from "./runtime-provider";
import {
  type CompareHandle,
  CompareHandlesProvider,
  RegisterCompareHandle,
  SharedComposer,
} from "./shared-composer";
import { ThreadSidebar } from "./thread-sidebar";
import type { ChatView } from "./types";

// TODO: fetch from API at runtime
const LORA_MODELS: ModelOption[] = [
  {
    id: "outputs/llama-3.1-8b-instruct-lora",
    name: "meta-llama/Llama-3.1-8B-Instruct",
    description: "LoRA v1",
  },
  {
    id: "outputs/qwen2.5-7b-lora",
    name: "Qwen/Qwen2.5-7B-Instruct",
    description: "LoRA v2",
  },
  {
    id: "outputs/mistral-7b-v0.3-lora",
    name: "mistralai/Mistral-7B-Instruct-v0.3",
    description: "LoRA v1",
  },
];

const GGUF_MODELS: ModelOption[] = [
  {
    id: "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    name: "Meta-Llama-3.1-8B-Instruct",
    description: "Q4_K_M",
  },
  {
    id: "models/Qwen2.5-7B-Instruct-Q5_K_M.gguf",
    name: "Qwen2.5-7B-Instruct",
    description: "Q5_K_M",
  },
  {
    id: "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    name: "Mistral-7B-Instruct-v0.3",
    description: "Q4_K_M",
  },
];

type SingleContentProps = {
  threadId?: string;
};

type CompareContentProps = {
  pairId: string;
};

const SingleContent = memo(function SingleContent({
  threadId,
}: SingleContentProps): ReactElement {
  return (
    <ChatRuntimeProvider modelType="base" initialThreadId={threadId}>
      <div className="min-h-0 flex-1">
        <Thread />
      </div>
    </ChatRuntimeProvider>
  );
});

const CompareContent = memo(function CompareContent({
  pairId,
}: CompareContentProps): ReactElement {
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
        <div className="grid min-h-0 flex-1 grid-cols-2  px-0">
          <div className="flex min-h-0 flex-col">
            <div className=" px-3 py-1.5">
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
            <div className="  text-end px-3 py-1.5">
              <span className="text-[10px] font-semibold uppercase  tracking-wider text-primary">
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
          "bg-sidebar text-sidebar-foreground h-full overflow-hidden transition-[width] duration-200 ease-linear",
          !collapsed && (side === "left" ? "border-r border-sidebar-border" : "border-l border-sidebar-border"),
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

export function ChatPage(): ReactElement {
  const [view, setView] = useState<ChatView>({ mode: "single" });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inferenceParams, setInferenceParams] = useState<InferenceParams>(
    defaultInferenceParams,
  );

  const handleCheckpointChange = useCallback(
    (v: string) => setInferenceParams((p) => ({ ...p, checkpoint: v })),
    [],
  );
  const handleEject = useCallback(
    () => setInferenceParams((p) => ({ ...p, checkpoint: "" })),
    [],
  );

  const models = useMemo(
    () =>
      inferenceParams.inferenceEngine === "llama-cpp"
        ? GGUF_MODELS
        : LORA_MODELS,
    [inferenceParams.inferenceEngine],
  );

  return (
    <SidebarProvider
      defaultOpen={false}
      className="!min-h-0 h-[calc(100vh-4rem)] max-w-7xl mx-auto px-4"
      style={{ "--sidebar-width": "14rem", "--sidebar-width-icon": "3rem" } as CSSProperties}
    >
      <InlineSidebar>
        <ThreadSidebar
          view={view}
          onSelect={setView}
          onNewThread={() => setView({ mode: "single" })}
          onNewCompare={() =>
            setView({ mode: "compare", pairId: crypto.randomUUID() })
          }
        />
      </InlineSidebar>

      {/* main chat area */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {/* top bar */}
        <div className="flex h-11 shrink-0 items-center px-2">
          <div className="flex items-center gap-1.5">
            <SidebarTrigger />
            <ModelSelector
              models={models}
              value={inferenceParams.checkpoint}
              onValueChange={handleCheckpointChange}
              onEject={handleEject}
              variant="ghost"
            />
          </div>
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
            key={view.threadId ?? "new"}
            threadId={view.threadId}
          />
        ) : (
          <CompareContent key={view.pairId} pairId={view.pairId} />
        )}
      </div>

      {/* inline settings panel on right */}
      <ChatSettingsPanel
        open={settingsOpen}
        params={inferenceParams}
        onParamsChange={setInferenceParams}
      />
    </SidebarProvider>
  );
}
