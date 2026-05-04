import { apiUrl, isTauri } from "@/lib/api-base";
import { useEffect, useState } from "react";

export function useNativePathLeasesSupported(): boolean {
  const [supported, setSupported] = useState(false);

  useEffect(() => {
    if (!isTauri) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    function check(delay = 0) {
      timer = setTimeout(() => {
        fetch(apiUrl("/api/health"))
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
    };
  }, []);

  return supported;
}
