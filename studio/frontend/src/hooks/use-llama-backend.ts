import { useEffect, useState } from "react";

import { loadLlamaBackendStatus } from "@/features/settings/api/llama-backend";
import { resolveLlamaBackendForWarning } from "./llama-backend-warning";

export function useLlamaCppBackend(): string | null {
  const [backend, setBackend] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadLlamaBackendStatus()
      .then((status) => {
        if (!cancelled) setBackend(resolveLlamaBackendForWarning(status));
      })
      .catch(() => {
        if (!cancelled) setBackend(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return backend;
}
