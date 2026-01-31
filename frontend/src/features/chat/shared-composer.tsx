import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { useAssistantRuntime } from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import {
  type KeyboardEvent,
  type MutableRefObject,
  type ReactElement,
  type ReactNode,
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

export interface CompareHandle {
  append: (content: { type: "text"; text: string }[]) => void;
  cancel: () => void;
  isRunning: () => boolean;
}

export type CompareHandles = MutableRefObject<Record<string, CompareHandle>>;

const CompareHandlesContext = createContext<CompareHandles | null>(null);

export function CompareHandlesProvider({
  handlesRef,
  children,
}: {
  handlesRef: CompareHandles;
  children: ReactNode;
}): ReactElement {
  return (
    <CompareHandlesContext.Provider value={handlesRef}>
      {children}
    </CompareHandlesContext.Provider>
  );
}

export function RegisterCompareHandle({
  name,
}: {
  name: string;
}): ReactElement | null {
  const handlesRef = useContext(CompareHandlesContext);
  const runtime = useAssistantRuntime();

  useEffect(() => {
    if (!handlesRef) {
      return;
    }
    const currentHandles = handlesRef.current;
    currentHandles[name] = {
      append: (content) => runtime.thread.append({ role: "user", content }),
      cancel: () => runtime.thread.cancelRun(),
      isRunning: () => runtime.thread.getState().isRunning,
    };
    return () => {
      delete currentHandles[name];
    };
  }, [handlesRef, name, runtime]);

  return null;
}

export function SharedComposer({
  handlesRef,
}: {
  handlesRef: CompareHandles;
}): ReactElement {
  const [text, setText] = useState("");
  const [running, setRunning] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const id = setInterval(() => {
      const handles = handlesRef.current;
      const any = Object.values(handles).some((h) => h.isRunning());
      setRunning(any);
    }, 200);
    return () => clearInterval(id);
  }, [handlesRef]);

  function send() {
    const msg = text.trim();
    if (!msg) {
      return;
    }

    const content: { type: "text"; text: string }[] = [
      { type: "text", text: msg },
    ];
    for (const handle of Object.values(handlesRef.current)) {
      handle.append(content);
    }
    setText("");
    textareaRef.current?.focus();
  }

  function stop() {
    for (const handle of Object.values(handlesRef.current)) {
      handle.cancel();
    }
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!running) {
        send();
      }
    }
  }

  return (
    <div className="shadow-border ring-1 ring-border relative flex w-full flex-col rounded-2xl bg-background px-1 pt-2 transition-shadow has-[textarea:focus-visible]:ring-2 has-[textarea:focus-visible]:ring-ring/20">
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Send to both models..."
        className="mb-1 max-h-32 min-h-14 w-full resize-none bg-transparent px-4 pt-2 pb-3 text-sm outline-none placeholder:text-muted-foreground"
        rows={1}
      />
      <div className="relative mx-2 mb-2 flex items-center justify-end">
        {running ? (
          <Button
            type="button"
            variant="default"
            size="icon"
            className="size-8 rounded-full"
            onClick={stop}
          >
            <SquareIcon className="size-3 fill-current" />
          </Button>
        ) : (
          <TooltipIconButton
            tooltip="Send message"
            side="bottom"
            variant="default"
            size="icon"
            className="size-8 rounded-full"
            onClick={send}
            disabled={!text.trim()}
          >
            <ArrowUpIcon className="size-4" />
          </TooltipIconButton>
        )}
      </div>
    </div>
  );
}
