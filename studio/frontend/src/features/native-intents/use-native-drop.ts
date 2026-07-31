import { isTauri } from "@/lib/api-base";
import { useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { registerNativeAttachmentPath, registerNativeModelPath } from "./api";
import { classifyDropPaths, SUPPORTED_DROP_HINT } from "./drop-paths";
import { useNativeIntentStore } from "./store";
import type { NativeIntent } from "./types";

export type NativeModelDropState =
  | { status: "idle" }
  | { status: "valid"; action: "load" | "replace" | "chip" }
  | { status: "attach"; count: number }
  | { status: "invalid" };

interface NativeModelDropOptions {
  enabled?: boolean;
  attachmentScope?: string;
  nativePathLeasesSupported: boolean;
  hasActiveModel: boolean;
  isModelLoading: boolean;
  onAutoLoad?: (intent: NativeIntent) => Promise<void> | void;
  onAttach?: (intents: NativeIntent[]) => Promise<void> | void;
}

function canAttachDocs(options: NativeModelDropOptions): boolean {
  return options.nativePathLeasesSupported && Boolean(options.onAttach);
}

function canAutoLoadModel(options: NativeModelDropOptions): boolean {
  return (
    options.nativePathLeasesSupported &&
    !options.isModelLoading &&
    Boolean(options.onAutoLoad)
  );
}

function dropStateForPaths(
  paths: string[],
  options: NativeModelDropOptions,
): NativeModelDropState {
  const dropped = classifyDropPaths(paths);
  if (dropped.kind === "none") return { status: "idle" };
  if (dropped.kind === "docs") {
    // Unlike a browser upload, a document drop only reaches the ingest through a signed
    // lease, so don't offer it as a target before the backend can verify one.
    return canAttachDocs(options)
      ? { status: "attach", count: dropped.paths.length }
      : { status: "invalid" };
  }
  if (dropped.kind === "unsupported") return { status: "invalid" };
  if (!canAutoLoadModel(options)) {
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
        const dropped = classifyDropPaths(event.payload.paths);
        if (dropped.kind === "none") return;
        if (dropped.kind === "unsupported") {
          toast.error(SUPPORTED_DROP_HINT);
          return;
        }
        if (dropped.kind === "docs") {
          if (!canAttachDocs(currentOptions)) {
            toast.error("Attaching files needs the desktop backend", {
              description: "Retry once Studio has finished starting up.",
            });
            return;
          }
          try {
            const intents = await Promise.all(
              dropped.paths.map(registerNativeAttachmentPath),
            );
            if (disposed) return;
            const latestOptions = optionsRef.current;
            const attachOptions =
              latestOptions.attachmentScope === currentOptions.attachmentScope
                ? latestOptions
                : currentOptions;
            await attachOptions.onAttach?.(intents);
          } catch (error) {
            toast.error("Could not attach dropped files", {
              description: error instanceof Error ? error.message : String(error),
            });
          }
          return;
        }
        try {
          const intent = await registerNativeModelPath(dropped.path);
          if (disposed) return;
          if (!canAutoLoadModel(currentOptions)) {
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
