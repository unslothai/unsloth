import { useCallback, useRef } from "react";
import { toast } from "sonner";
import { pickNativeModel } from "./api";
import { useNativeIntentStore } from "./store";
import type { NativeIntent } from "./types";

interface ChooseNativeModelOptions {
  shouldAutoLoad?: (intent: NativeIntent) => boolean;
  onAutoLoad?: (intent: NativeIntent) => Promise<void> | void;
}

function isGgufModelIntent(intent: NativeIntent): boolean {
  const label = intent.path.displayLabel || intent.displayLabel;
  return (
    intent.kind === "model" &&
    intent.path.kind === "model" &&
    label.toLowerCase().endsWith(".gguf") &&
    intent.path.allowedOperations.includes("validate-model") &&
    intent.path.allowedOperations.includes("load-model")
  );
}

export function useChooseNativeModel(options: ChooseNativeModelOptions = {}) {
  const addIntent = useNativeIntentStore((state) => state.addIntent);
  const pickingRef = useRef(false);
  const onAutoLoad = options.onAutoLoad;
  const shouldAutoLoad = options.shouldAutoLoad;

  return useCallback(async () => {
    if (pickingRef.current) return;
    pickingRef.current = true;
    try {
      let intent: NativeIntent | null = null;
      try {
        intent = await pickNativeModel();
      } catch (error) {
        toast.error("Could not choose local model", {
          description: error instanceof Error ? error.message : String(error),
        });
        return;
      }

      if (!intent) return;

      let runAutoLoad = false;
      try {
        runAutoLoad =
          Boolean(onAutoLoad) &&
          isGgufModelIntent(intent) &&
          shouldAutoLoad?.(intent) === true;
      } catch {
        runAutoLoad = false;
      }
      if (!runAutoLoad) {
        addIntent(intent);
        return;
      }

      try {
        await onAutoLoad?.(intent);
      } catch {
        addIntent(intent);
      }
    } finally {
      pickingRef.current = false;
    }
  }, [addIntent, onAutoLoad, shouldAutoLoad]);
}
