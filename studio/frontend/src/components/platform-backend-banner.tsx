import { Button } from "@/components/ui/button";
import { isPlatformOnlyMode } from "@/config/platform-capabilities";
import { usePlatformConnectionStore } from "@/integrations/platform-backend";
import { useCallback, useEffect, useRef } from "react";

const CHECK_INTERVAL_MS = 30_000;

export function PlatformBackendBanner() {
  const status = usePlatformConnectionStore((state) => state.status);
  const checkConnection = usePlatformConnectionStore(
    (state) => state.checkConnection,
  );
  const controllerRef = useRef<AbortController | null>(null);

  const runCheck = useCallback(() => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    void checkConnection(controller.signal);
  }, [checkConnection]);

  useEffect(() => {
    if (!isPlatformOnlyMode()) return;
    runCheck();
    const interval = window.setInterval(() => {
      if (document.visibilityState === "visible") runCheck();
    }, CHECK_INTERVAL_MS);
    return () => {
      window.clearInterval(interval);
      controllerRef.current?.abort();
    };
  }, [runCheck]);

  if (
    !isPlatformOnlyMode() ||
    status === "idle" ||
    status === "checking" ||
    status === "connected"
  ) {
    return null;
  }
  const disconnected = status === "disconnected";
  const message =
    status === "unauthorized"
      ? "Rag Platform oturumu doğrulanamadı. Yeniden giriş yapın."
      : disconnected
        ? "Rag Platform bağlantısı kesildi. Veriler boş değil; şu anda yüklenemiyor."
        : "Rag Platform kısmi hizmet veriyor. Bazı işlemler geçici olarak kullanılamayabilir.";

  return (
    <output
      aria-live="polite"
      className={
        disconnected
          ? "flex items-center justify-between gap-3 border-b border-destructive/20 bg-destructive/10 px-4 py-2 text-xs text-destructive"
          : "flex items-center justify-between gap-3 border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs text-amber-800 dark:text-amber-300"
      }
    >
      <span>{message}</span>
      {status !== "unauthorized" && (
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={runCheck}
        >
          Yeniden dene
        </Button>
      )}
    </output>
  );
}
