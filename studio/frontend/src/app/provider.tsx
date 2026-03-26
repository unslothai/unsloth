// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StartupScreen } from "@/components/tauri/startup-screen";
import { Toaster } from "@/components/ui/sonner";
import { useTauriBackend } from "@/hooks/use-tauri-backend";
import { isTauri } from "@/lib/api-base";
import { ThemeProvider } from "next-themes";
import type { ReactNode } from "react";

interface AppProviderProps {
  children: ReactNode;
}

function TauriWrapper({ children }: { children: ReactNode }) {
  const {
    status, logs, error,
    currentStepIndex, progressDetail, elevationPackages,
    startInstall, retry, retryInstall, approveElevation,
  } = useTauriBackend();

  if (!isTauri) return <>{children}</>;
  if (status === "running") return <>{children}</>;

  return (
    <StartupScreen
      status={status}
      logs={logs}
      error={error}
      currentStepIndex={currentStepIndex}
      progressDetail={progressDetail}
      elevationPackages={elevationPackages}
      onInstall={startInstall}
      onRetry={retry}
      onRetryInstall={retryInstall}
      onApproveElevation={approveElevation}
      onStartServer={retry}
    />
  );
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
