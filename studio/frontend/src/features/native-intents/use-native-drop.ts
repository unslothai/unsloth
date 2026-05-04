import { isTauri } from "@/lib/api-base";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { registerNativeModelPath } from "./api";
import { useNativeIntentStore } from "./store";
import type { NativeIntent } from "./types";

export type NativeModelDropState =
  | { status: "idle" }
  | { status: "valid"; action: "load" | "replace" | "chip" }
  | { status: "invalid" };

interface NativeModelDropOptions {
  enabled?: boolean;
  nativePathLeasesSupported: boolean;
  hasActiveModel: boolean;
  isModelLoading: boolean;
  onAutoLoad?: (intent: NativeIntent) => Promise<void> | void;
}

function ggufPaths(paths: string[]): string[] {
  return paths.filter((path) => path.toLowerCase().endsWith(".gguf"));
}

function canAutoLoadPaths(paths: string[], options: NativeModelDropOptions): boolean {
  return (
    paths.length === 1 &&
    ggufPaths(paths).length === 1 &&
    options.nativePathLeasesSupported &&
    !options.isModelLoading &&
    Boolean(options.onAutoLoad)
  );
}

function dropStateForPaths(
  paths: string[],
  options: NativeModelDropOptions,
): NativeModelDropState {
  if (paths.length === 0) {
    return { status: "idle" };
  }
  const ggufs = ggufPaths(paths);
  if (paths.length !== 1 || ggufs.length !== 1) {
    return { status: "invalid" };
  }
  if (!canAutoLoadPaths(paths, options)) {
    return { status: "valid", action: "chip" };
  }
  return {
    status: "valid",
    action: options.hasActiveModel ? "replace" : "load",
  };
}

export function useNativeModelDrop(options: NativeModelDropOptions): NativeModelDropState {
  const { enabled = true } = options;
  const addIntent = useNativeIntentStore((state) => state.addIntent);
  const [dropState, setDropState] = useState<NativeModelDropState>({ status: "idle" });
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    if (!isTauri || !enabled) {
      setDropState({ status: "idle" });
      return;
    }
    let disposed = false;
    let unlisten: (() => void) | undefined;

    void import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => getCurrentWindow().onDragDropEvent(async (event) => {
        const currentOptions = optionsRef.current;
        if (event.payload.type === "enter") {
          setDropState(dropStateForPaths(event.payload.paths, currentOptions));
          return;
        }
        if (event.payload.type === "leave") {
          setDropState({ status: "idle" });
          return;
        }
        if (event.payload.type !== "drop") return;
        setDropState({ status: "idle" });
        const ggufs = ggufPaths(event.payload.paths);
        if (event.payload.paths.length !== 1 || ggufs.length !== 1) {
          if (event.payload.paths.length > 0) {
            toast.error(
              ggufs.length === 0
                ? "Only .gguf model files can be dropped here."
                : "Drop a single .gguf model file.",
            );
          }
          return;
        }
        const ggufPath = ggufs[0];
        try {
          const intent = await registerNativeModelPath(ggufPath);
          if (disposed) return;
          if (!canAutoLoadPaths(event.payload.paths, currentOptions)) {
            addIntent(intent);
            return;
          }
          try {
            await currentOptions.onAutoLoad?.(intent);
          } catch (error) {
            addIntent(intent);
            toast.error("Could not load dropped model", {
              description: error instanceof Error ? error.message : String(error),
            });
          }
        } catch (error) {
          toast.error("Could not use dropped model", {
            description: error instanceof Error ? error.message : String(error),
          });
        }
      }))
      .then((cleanup) => {
        if (disposed) {
          cleanup();
        } else {
          unlisten = cleanup;
        }
      })
      .catch(() => undefined);

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [addIntent, enabled]);

  return dropState;
}
