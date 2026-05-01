import { apiUrl, isTauri } from "@/lib/api-base";
import { useEffect, useState } from "react";

export function useNativePathLeasesSupported(): boolean {
  const [supported, setSupported] = useState(false);

  useEffect(() => {
    if (!isTauri) return;
    let disposed = false;
    fetch(apiUrl("/api/health"))
      .then((response) => response.json())
      .then((health) => {
        if (!disposed) setSupported(health?.native_path_leases_supported === true);
      })
      .catch(() => {
        if (!disposed) setSupported(false);
      });
    return () => {
      disposed = true;
    };
  }, []);

  return supported;
}
