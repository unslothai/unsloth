import { apiUrl, isTauri } from "@/lib/api-base";
import { isPlatformAuthEnabled } from "@/integrations/platform-backend/config";
import { useEffect, useState } from "react";

const MAX_READINESS_POLLS = 720;

export function useNativePathLeasesSupported(): boolean {
  const [supported, setSupported] = useState(false);
  const platformAuthEnabled = isPlatformAuthEnabled();

  useEffect(() => {
    // Native path leases belong to the legacy Studio /api/health contract.
    // Rag Platform does not consume them, so polling can never become useful.
    if (!isTauri || platformAuthEnabled) return;
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
          .then((health) => {
            if (disposed) return;
            if (health?.native_path_leases_supported === true) {
              setSupported(true);
            } else {
              check(5000);
            }
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
  }, [platformAuthEnabled]);

  return supported;
}
