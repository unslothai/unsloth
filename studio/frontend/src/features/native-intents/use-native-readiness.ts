import { apiUrl, isTauri } from "@/lib/api-base";
import { useEffect, useState } from "react";

const MAX_READINESS_POLLS = 720;

export function useNativePathLeasesSupported(): boolean {
  const [supported, setSupported] = useState(false);

  useEffect(() => {
    if (!isTauri) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    let polls = 0;

    function check(delay = 0) {
      if (polls >= MAX_READINESS_POLLS) return;
      polls += 1;
      timer = setTimeout(() => {
        controller = new AbortController();
        fetch(apiUrl("/api/health"), { signal: controller.signal })
          .then((response) => response.json())
          .then(async (health) => {
            if (disposed) return;
            if (health?.native_path_leases_supported !== true) {
              check(5000);
              return;
            }
            // The health bit says the backend holds a key, not that it holds
            // OURS. A survivor adopted from a dead previous app has one of its
            // own, so the grant would fail on the signature instead. Only the
            // app knows which backend it spawned.
            const { invoke } = await import("@tauri-apps/api/core");
            const usable = await invoke<boolean>("native_path_leases_usable");
            if (disposed) return;
            if (usable) setSupported(true);
            else check(5000);
          })
          .catch(() => {
            if (!disposed) check(5000);
          });
      }, delay);
    }

    check();
    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      controller?.abort();
    };
  }, []);

  return supported;
}
