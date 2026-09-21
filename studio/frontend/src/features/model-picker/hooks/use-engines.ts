// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { useEffect, useState } from "react";
import { type EngineStatus, listEngines } from "../api/engines";

export function useEngines() {
  const [engines, setEngines] = useState<EngineStatus[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    let fetching = false;
    const refresh = async () => {
      if (fetching || disposed) {
        return;
      }
      fetching = true;
      try {
        const result = await listEngines();
        if (!disposed) {
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
      } finally {
        fetching = false;
        if (!disposed) {
          timer = setTimeout(refresh, document.hidden ? 30000 : 3000);
        }
      }
    };
    const changed = () => {
      clearTimeout(timer);
      void refresh();
    };
    void refresh();
    window.addEventListener("studio-engines-changed", changed);
    return () => {
      disposed = true;
      clearTimeout(timer);
      window.removeEventListener("studio-engines-changed", changed);
    };
  }, []);
  return { engines, error };
}
