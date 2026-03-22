// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

interface ConnectionOverlayProps {
  status: "starting" | "stopped" | "error";
  error: string | null;
  logs: string[];
  onRetry: () => void;
  onViewLogs: () => void;
}

export function ConnectionOverlay({
  status,
  error,
  onRetry,
  onViewLogs,
}: ConnectionOverlayProps) {
  const messages = {
    starting: "Starting server...",
    stopped: "Server stopped",
    error: error ?? "An error occurred",
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div className="max-w-md space-y-4 rounded-lg border bg-card p-6 shadow-lg">
        <h2 className="text-lg font-semibold">{messages[status]}</h2>
        {status === "starting" && (
          <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-primary animate-pulse w-full" />
          </div>
        )}
        {status === "error" && (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">{error}</p>
            <div className="flex gap-2">
              <button
                onClick={onRetry}
                className="rounded bg-primary px-4 py-2 text-sm text-primary-foreground"
              >
                Retry
              </button>
              <button
                onClick={onViewLogs}
                className="rounded bg-muted px-4 py-2 text-sm"
              >
                View Logs
              </button>
            </div>
          </div>
        )}
        {status === "stopped" && (
          <button
            onClick={onRetry}
            className="rounded bg-primary px-4 py-2 text-sm text-primary-foreground"
          >
            Start Server
          </button>
        )}
      </div>
    </div>
  );
}
