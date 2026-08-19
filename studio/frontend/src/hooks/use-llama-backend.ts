import { useEffect, useState } from "react";

import { loadLlamaBackendStatus } from "@/features/settings/api/llama-backend";

export function useLlamaCppBackend(): string | null {
  const [backend, setBackend] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadLlamaBackendStatus()
      .then((status) => {
        if (!cancelled) setBackend(status.backend);
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
