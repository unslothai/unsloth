import { useEffect } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

import { getPlatformBackendConfig } from "./config";
import {
  type PlatformConnectionStatus,
  usePlatformConnectionStore,
} from "./connection-store";

const STATUS_LABEL: Record<PlatformConnectionStatus, string> = {
  idle: "Not checked",
  checking: "Checking…",
  connected: "Connected",
  degraded: "Degraded",
  disconnected: "Disconnected",
  unauthorized: "Permission required",
};

const STATUS_STYLE: Record<PlatformConnectionStatus, string> = {
  idle: "bg-muted text-muted-foreground",
  checking: "bg-muted text-foreground",
  connected: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  degraded: "bg-amber-500/10 text-amber-700 dark:text-amber-300",
  disconnected: "bg-destructive/10 text-destructive",
  unauthorized: "bg-amber-500/10 text-amber-700 dark:text-amber-300",
};

interface PlatformBackendConnectionStatusProps {
  enabled?: boolean;
}

export function PlatformBackendConnectionStatus({
  enabled = getPlatformBackendConfig().enabled,
}: PlatformBackendConnectionStatusProps) {
  const status = usePlatformConnectionStore((state) => state.status);
  const version = usePlatformConnectionStore((state) => state.version);
  const error = usePlatformConnectionStore((state) => state.error);
  const lastCheckedAt = usePlatformConnectionStore(
    (state) => state.lastCheckedAt,
  );
  const checkConnection = usePlatformConnectionStore(
    (state) => state.checkConnection,
  );

  useEffect(() => {
    // The store survives Settings tab remounts. Probe once per app session and
    // leave explicit refreshes to the button instead of repeatedly hitting all
    // three readiness endpoints whenever the tab is reopened.
    if (!enabled || lastCheckedAt) return;
    const controller = new AbortController();
    void checkConnection(controller.signal);
    return () => controller.abort();
  }, [checkConnection, enabled, lastCheckedAt]);

  const visibleStatus = enabled ? status : "idle";

  return (
    <section
      aria-labelledby="rag-platform-connection-heading"
      className="rounded-2xl border border-border bg-card p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2
            id="rag-platform-connection-heading"
            className="text-sm font-semibold text-foreground"
          >
            Rag Platform backend
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Connection, version and dependency readiness
          </p>
        </div>
        <span
          aria-live="polite"
          className={cn(
            "rounded-full px-2.5 py-1 text-xs font-medium",
            enabled ? STATUS_STYLE[visibleStatus] : STATUS_STYLE.idle,
          )}
        >
          {enabled ? STATUS_LABEL[visibleStatus] : "Disabled"}
        </span>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs text-muted-foreground">
          {version ? (
            <span>Version {version}</span>
          ) : (
            <span>No version yet</span>
          )}
          {lastCheckedAt ? (
            <span className="ml-2">
              Checked {new Date(lastCheckedAt).toLocaleTimeString()}
            </span>
          ) : null}
        </div>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={!enabled || status === "checking"}
          onClick={() => void checkConnection()}
        >
          {status === "checking" ? "Checking…" : "Check connection"}
        </Button>
      </div>

      {error ? (
        <p role="alert" className="mt-3 text-xs text-destructive">
          {error.message}
        </p>
      ) : null}
    </section>
  );
}
