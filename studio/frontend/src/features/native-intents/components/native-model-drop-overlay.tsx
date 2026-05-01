import { cn } from "@/lib/utils";
import { FileUpIcon } from "lucide-react";
import type { NativeModelDropState } from "../use-native-drop";

function overlayCopy(state: NativeModelDropState): { title: string; description: string } {
  if (state.status === "invalid") {
    return {
      title: "GGUF models only",
      description: "Other files are not handled here yet.",
    };
  }
  if (state.status === "valid" && state.action === "replace") {
    return {
      title: "Drop to replace model",
      description: "Current model will unload first.",
    };
  }
  if (state.status === "valid" && state.action === "load") {
    return {
      title: "Drop to load model",
      description: "Adds it as the active chat model.",
    };
  }
  return {
    title: "Drop to add model chip",
    description: "Review it before loading.",
  };
}

export function NativeModelDropOverlay({ state }: { state: NativeModelDropState }) {
  const isIdle = state.status === "idle";
  const isAutoLoad = state.status === "valid" && state.action !== "chip";
  const isInvalid = state.status === "invalid";
  const { title, description } = overlayCopy(state);

  return (
    <div
      className={cn(
        "pointer-events-none absolute left-1/2 top-4 z-50 w-[clamp(16rem,28vw,22rem)] max-w-[calc(100vw-1rem)] -translate-x-1/2 transition-all duration-200 ease-out",
        isIdle ? "-translate-y-1 opacity-0" : "translate-y-0 opacity-100",
      )}
      role="status"
      aria-live="polite"
      aria-hidden={isIdle}
    >
      <div className="flex items-center gap-2.5 rounded-xl border border-border/70 bg-card/96 px-3 py-2.5 shadow-sm shadow-black/10 backdrop-blur-sm dark:shadow-black/30">
        <div
          className={cn(
            "flex size-8 shrink-0 items-center justify-center rounded-lg border transition-colors duration-200 ease-out",
            isAutoLoad
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
              : isInvalid
                ? "border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-300"
                : "border-border bg-muted text-muted-foreground",
          )}
        >
          <FileUpIcon className="size-4" aria-hidden="true" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-xs font-medium text-foreground">
            {title}
          </div>
          <div className="mt-0.5 truncate text-[11px] leading-4 text-muted-foreground">
            {description}
          </div>
        </div>
      </div>
    </div>
  );
}
