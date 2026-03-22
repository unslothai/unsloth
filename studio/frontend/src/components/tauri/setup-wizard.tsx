// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

interface SetupWizardProps {
  logs: string[];
  onInstall: () => void;
  status: "not-installed" | "installing";
  error?: string | null;
  onRetry?: () => void;
}

export function SetupWizard({ logs, onInstall, status, error, onRetry }: SetupWizardProps) {
  return (
    <div className="flex h-screen items-center justify-center bg-background">
      <div className="w-full max-w-2xl space-y-6 p-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold">Unsloth Studio</h1>
          <p className="text-muted-foreground">
            {status === "not-installed"
              ? "First-time setup required. This will install the ML backend."
              : "Setting up Unsloth Studio..."}
          </p>
        </div>

        {error && (
          <div className="space-y-3 rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="text-sm font-medium text-destructive">Installation failed</p>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">{error}</p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="rounded bg-primary px-4 py-2 text-sm text-primary-foreground"
              >
                Try Again
              </button>
            )}
          </div>
        )}

        {status === "not-installed" && !error && (
          <div className="text-center">
            <button
              onClick={onInstall}
              className="rounded-lg bg-primary px-6 py-3 text-primary-foreground font-medium hover:bg-primary/90"
            >
              Install Backend
            </button>
          </div>
        )}

        {(status === "installing" || (error && logs.length > 0)) && (
          <div className="space-y-4">
            {status === "installing" && (
              <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-primary animate-pulse w-full" />
              </div>
            )}
            <div className="h-64 overflow-y-auto rounded-lg bg-muted p-4 font-mono text-xs">
              {logs.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap">
                  {line}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
