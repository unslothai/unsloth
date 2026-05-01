import { isTauri } from "@/lib/api-base";
import { useEffect } from "react";
import { toast } from "sonner";
import { registerNativeModelPath } from "./api";
import { useNativeIntentStore } from "./store";

export function useNativeModelDrop(enabled = true) {
  const addIntent = useNativeIntentStore((state) => state.addIntent);

  useEffect(() => {
    if (!isTauri || !enabled) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;

    void import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => getCurrentWindow().onDragDropEvent(async (event) => {
        if (event.payload.type !== "drop") return;
        const ggufPath = event.payload.paths.find((path: string) => path.toLowerCase().endsWith(".gguf"));
        if (!ggufPath) return;
        try {
          const intent = await registerNativeModelPath(ggufPath);
          if (!disposed) addIntent(intent);
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
}
