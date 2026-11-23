// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { useEffect, useState } from "react";
import { type EngineStatus, listEngines, changeEngine } from "../api/engines";

import {
  startExternalJob,
  isExternalJob,
  updateExternalActivity,
  finishExternalJob,
} from "@/features/hub/download-manager/external-jobs";
import { startVisibilityPoll } from "@/features/hub/download-manager/visibility-poll";
import { enginePollDelayMs } from "./engine-poll-delay";

export function useEngines(enabled = true, background = false) {
  const [engines, setEngines] = useState<EngineStatus[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!enabled) return;
    let disposed = false;
    let installing = false;
    const refresh = async () => {
      try {
        const result = await listEngines();
        if (!disposed) {
          installing = result.some((engine) => engine.job.state === "running");
          for (const engine of result) {
            const key = `engine:${engine.engine}`;
            if (engine.job.state === "running") {
              if (!isExternalJob(key))
                startExternalJob({
                  key,
                  repoId: engine.engine === "vllm" ? "vLLM" : "SGLang",
                  variant: engine.version,
                  expectedBytes: 0,
                  cancel: () => changeEngine(engine.engine, "cancel"),
                });
              updateExternalActivity(
                key,
                [engine.job.message, engine.job.activity]
                  .filter(Boolean)
                  .join(": "),
                engine.job.log ?? [],
              );
            } else if (isExternalJob(key)) {
              finishExternalJob(
                key,
                engine.job.state === "success"
                  ? "complete"
                  : engine.job.state === "cancelled"
                    ? "cancelled"
                    : "error",
                engine.job.message,
              );
            }
          }
          setEngines(result);
          setError("");
        }
      } catch (err) {
        if (!disposed) {
          setError(
            err instanceof Error
              ? err.message
              : "Could not read inference engines",
          );
        }
      }
      return !disposed;
    };
    // Pickers keep their timer-only cadence; only the parked background poll wakes on show.
    const poller = startVisibilityPoll(
      refresh,
      (hidden) => enginePollDelayMs(hidden, installing, background),
      { wakeOnVisible: background },
    );
    const changed = () => poller.refresh();
    poller.restart();
    window.addEventListener("studio-engines-changed", changed);
    return () => {
      disposed = true;
      poller.stop();
      window.removeEventListener("studio-engines-changed", changed);
    };
  }, [enabled, background]);
  return { engines, error };
}
