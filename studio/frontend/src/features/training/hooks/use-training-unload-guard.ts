


import { isTauri } from "@/lib/api-base";
import { useEffect } from "react";
import { subscribeToTrainingActivity } from "../lib/training-activity";
import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";

let currentHandler: ((e: BeforeUnloadEvent) => void) | null = null;

// Desktop quit never fires beforeunload; Rust asks instead, from the tray and (outside macOS) the close button.
function publishTrainingActive(active: boolean): void {
  if (!isTauri) return;
  void import("@tauri-apps/api/core")
    .then(({ invoke }) => invoke("set_training_active", { active }))
    .catch(() => {});
}

/** Mounts a beforeunload guard that warns while training is starting or active.
* Call once at the app root. */
export function useTrainingUnloadGuard() {
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (!isTrainingStartPending(useTrainingRuntimeStore.getState())) {
        return;
      }
      e.preventDefault();
      e.returnValue = "";
    };
    currentHandler = handler;
    window.addEventListener("beforeunload", handler);
    return () => {
      if (currentHandler === handler) {
        currentHandler = null;
      }
      window.removeEventListener("beforeunload", handler);
    };
  }, []);

  useEffect(() => {
    return subscribeToTrainingActivity(publishTrainingActive);
  }, []);
}

/**
* Removes the active beforeunload guard (if any). Call before intentionally ending the session
* so the "Server stopped" page renders without the browser prompting to confirm leaving. */
export function removeTrainingUnloadGuard() {
  if (currentHandler) {
    window.removeEventListener("beforeunload", currentHandler);
    currentHandler = null;
  }
  publishTrainingActive(false);
}
