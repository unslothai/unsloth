// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ConnectionOverlay } from "@/components/tauri/connection-overlay";
import { SetupWizard } from "@/components/tauri/setup-wizard";
import { Toaster } from "@/components/ui/sonner";
import { useTauriBackend } from "@/hooks/use-tauri-backend";
import { isTauri } from "@/lib/api-base";
import { ThemeProvider } from "next-themes";
import { useCallback, useState } from "react";
import type { ReactNode } from "react";

interface AppProviderProps {
  children: ReactNode;
}

function TauriWrapper({ children }: { children: ReactNode }) {
  const { status, logs, error, startInstall, retry } = useTauriBackend();
  const [showLogs, setShowLogs] = useState(false);

  const handleViewLogs = useCallback(async () => {
    if (isTauri) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("open_logs_dir");
      } catch {
        setShowLogs(true);
      }
    } else {
      setShowLogs(true);
    }
  }, []);

  // Non-Tauri mode: render children directly
  if (!isTauri) return <>{children}</>;

  // First-time install flow
  if (status === "not-installed" || status === "installing") {
    return (
      <SetupWizard logs={logs} onInstall={startInstall} status={status} />
    );
  }

  // Checking status — show nothing while determining state
  if (status === "checking") {
    return null;
  }

  // Starting, stopped, or error: show children with overlay
  if (status === "starting" || status === "stopped" || status === "error") {
    return (
      <>
        {children}
        <ConnectionOverlay
          status={status}
          error={error}
          logs={logs}
          onRetry={retry}
          onViewLogs={handleViewLogs}
        />
        {showLogs && (
          <div className="fixed inset-0 z-[60] flex items-center justify-center bg-background/90">
            <div className="w-full max-w-3xl space-y-4 p-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Server Logs</h3>
                <button
                  onClick={() => setShowLogs(false)}
                  className="rounded bg-muted px-3 py-1 text-sm"
                >
                  Close
                </button>
              </div>
              <div className="h-96 overflow-y-auto rounded-lg bg-muted p-4 font-mono text-xs">
                {logs.map((line, i) => (
                  <div key={i} className="whitespace-pre-wrap">
                    {line}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </>
    );
  }

  // Running: render children normally
  return <>{children}</>;
}

export function AppProvider({ children }: AppProviderProps) {
  return (
    <ThemeProvider attribute="class" defaultTheme="light">
      <TauriWrapper>
        {children}
      </TauriWrapper>
      <Toaster position="top-right" visibleToasts={2} expand={true} />
    </ThemeProvider>
  );
}
