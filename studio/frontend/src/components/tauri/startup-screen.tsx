// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SetupLogDetails } from "@/components/tauri/setup-log";
import {
  installProgressMessage,
  startupWaitingMessage,
  STATUS_MESSAGE_ROTATION_MS,
  type StartupMessage,
} from "@/components/tauri/startup-messages";
import { Spinner } from "@/components/ui/spinner";
import type { BackendStatus } from "@/hooks/use-tauri-backend";
import type { CopySupportDiagnosticsResult } from "@/lib/tauri-diagnostics";

import { AnimatePresence, motion } from "motion/react";
import { type ReactNode, useEffect, useState } from "react";

interface StartupScreenProps {
  status: BackendStatus;
  logs: string[];
  error: string | null;
  currentStepIndex: number;
  progressDetail: string | null;
  startupMessage: StartupMessage;
  elevationPackages: string[];
  onInstall: () => void;
  onRetry: () => void;
  onRetryInstall: () => void;
  onApproveElevation: () => void;
  onStartServer: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}

function DiagnosticsCopyActions({
  onCopyDiagnostics,
  children,
}: {
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
  children: React.ReactNode;
}) {
  const [copying, setCopying] = useState(false);
  const [manualReport, setManualReport] = useState<string | null>(null);
  const [manualMessage, setManualMessage] = useState<string | null>(null);

  async function handleCopyDiagnostics() {
    setCopying(true);
    try {
      const result = await onCopyDiagnostics();
      if (result.ok) {
        setManualReport(null);
        setManualMessage(null);
      } else {
        setManualReport(result.report);
        setManualMessage(result.error ?? "Clipboard copy failed. Select and copy the diagnostics below.");
      }
    } catch (error) {
      setManualReport(null);
      setManualMessage(`Diagnostics copy failed: ${String(error)}`);
    } finally {
      setCopying(false);
    }
  }

  return (
    <div className="mt-4 flex w-full flex-col items-center gap-3">
      <div className="flex gap-3">
        <ActionButton
          variant="secondary"
          onClick={() => void handleCopyDiagnostics()}
        >
          {copying ? "Copying..." : "Copy Diagnostics"}
        </ActionButton>
        {children}
      </div>
      {manualMessage && (
        <p className="max-w-md text-center text-xs text-destructive">{manualMessage}</p>
      )}
      {manualReport && (
        <textarea
          readOnly
          value={manualReport}
          onFocus={(event) => event.currentTarget.select()}
          className="h-32 w-full max-w-md resize-none rounded-lg border border-border/50 bg-muted/30 p-2 font-mono text-ui-10 text-muted-foreground"
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------


const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Logo() {
  return (
    <div className="flex items-center justify-center gap-3">
      <img
        src="/sticker.png"
        alt=""
        aria-hidden="true"
        className="h-[60px] w-[60px] object-contain"
      />
      <span
        className="text-ui-50 font-semibold leading-none tracking-[-0.02em] text-foreground"
        style={{ fontFamily: '"Hellix", sans-serif' }}
      >
        unsloth
      </span>
    </div>
  );
}

function ActionButton({
  onClick,
  variant = "primary",
  children,
}: {
  onClick: () => void;
  variant?: "primary" | "secondary";
  children: React.ReactNode;
}) {
  const base = "rounded-lg px-5 py-2.5 text-sm font-medium cursor-pointer transition-colors";
  const styles =
    variant === "primary"
      ? `${base} bg-primary text-primary-foreground hover:bg-primary/80`
      : `${base} bg-muted text-foreground hover:bg-muted/80`;
  return (
    <button type="button" className={styles} onClick={onClick}>
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Per-status renderers
// ---------------------------------------------------------------------------

function CheckingContent() {
  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm text-muted-foreground">Checking...</p>
      </div>
    </div>
  );
}

function NotInstalledContent({ onInstall }: { onInstall: () => void }) {
  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 flex-col items-center justify-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-3">
        <p
          className="text-ui-13 font-semibold tracking-[-0.01em] text-muted-foreground"
          style={{ fontFamily: '"Hellix", sans-serif' }}
        >
          To install Unsloth, click Get Started.
        </p>
        <ActionButton onClick={onInstall}>
          Get Started
        </ActionButton>
      </div>
    </div>
  );
}

function useRotatingMessageIndex(): number {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(
      () => setMessageIndex((current) => current + 1),
      STATUS_MESSAGE_ROTATION_MS,
    );
    return () => window.clearInterval(interval);
  }, []);

  return messageIndex;
}

function InstallingContent({
  logs,
  currentStepIndex,
  progressDetail,
}: {
  logs: string[];
  currentStepIndex: number;
  progressDetail: string | null;
}) {
  const messageIndex = useRotatingMessageIndex();
  const message = installProgressMessage(currentStepIndex, messageIndex);
  const detailLines = progressDetail
    ? [...logs, progressDetail]
    : logs;

  return (
    <div className="flex h-full w-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex w-full flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm font-bold text-foreground" aria-live="polite">
          {message.title}
        </p>
        <p className="text-sm text-muted-foreground">{message.subtitle}</p>
        <SetupLogDetails label="installation" lines={detailLines} />
      </div>
    </div>
  );
}

function RepairingContent({
  logs,
  progressDetail,
}: {
  logs: string[];
  progressDetail: string | null;
}) {
  const detailLines = progressDetail
    ? [...logs, progressDetail]
    : logs;

  return (
    <div className="flex h-full w-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex w-full flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm font-bold text-foreground">Getting things ready...</p>
        <p className="text-sm text-muted-foreground">This won’t take long.</p>
        <SetupLogDetails label="setup" lines={detailLines} />
      </div>
    </div>
  );
}

function ClosingContent() {
  return (
    <div className="flex h-full w-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex w-full flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm font-bold text-foreground" aria-live="polite">
          Closing Unsloth Desktop...
        </p>
        <p className="text-sm text-muted-foreground">Shutting down the backend.</p>
      </div>
    </div>
  );
}

function InstallErrorContent({
  error,
  onRetryInstall,
  onCopyDiagnostics,
}: {
  error: string | null;
  onRetryInstall: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-destructive">Setup ran into a problem</p>
        {error && (
          <p className="max-w-xs text-center text-xs text-muted-foreground">{error}</p>
        )}
        <DiagnosticsCopyActions onCopyDiagnostics={onCopyDiagnostics}>
          <ActionButton onClick={onRetryInstall}>Try Again</ActionButton>
        </DiagnosticsCopyActions>
      </div>
    </>
  );
}

function RepairErrorContent({
  error,
  onRetry,
  onCopyDiagnostics,
}: {
  error: string | null;
  onRetry: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-destructive">Update failed</p>
        {error && (
          <p className="max-w-md text-center text-xs text-muted-foreground">{error}</p>
        )}
        <DiagnosticsCopyActions onCopyDiagnostics={onCopyDiagnostics}>
          <ActionButton onClick={onRetry}>Retry</ActionButton>
        </DiagnosticsCopyActions>
      </div>
    </>
  );
}

function NeedsElevationContent({
  elevationPackages,
  onApproveElevation,
  onRetryInstall,
}: {
  elevationPackages: string[];
  onApproveElevation: () => void;
  onRetryInstall: () => void;
}) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-foreground">Permission needed</p>
        <p className="text-xs text-muted-foreground">
          The following system packages need to be installed:
        </p>
        <div className="mt-2 w-full max-w-xs rounded-lg bg-muted p-3 font-mono text-xs">
          {elevationPackages.map((pkg) => (
            <div key={pkg}>{pkg}</div>
          ))}
        </div>
        <div className="mt-4 flex gap-3">
          <ActionButton variant="secondary" onClick={onRetryInstall}>Cancel</ActionButton>
          <ActionButton onClick={onApproveElevation}>Allow</ActionButton>
        </div>
      </div>
    </>
  );
}

function StartingContent({ message }: { message: StartupMessage }) {
  const messageIndex = useRotatingMessageIndex();
  const displayMessage = startupWaitingMessage(message, messageIndex);

  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm text-muted-foreground">{displayMessage}</p>
      </div>
    </div>
  );
}

function StoppedContent({ onStartServer }: { onStartServer: () => void }) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-foreground">Server stopped</p>
        <div className="mt-4">
          <ActionButton onClick={onStartServer}>Start Server</ActionButton>
        </div>
      </div>
    </>
  );
}

function ErrorContent({
  error,
  onRetry,
  onCopyDiagnostics,
}: {
  error: string | null;
  onRetry: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-destructive">Something went wrong</p>
        {error && (
          <p className="max-w-md text-center text-xs text-muted-foreground">{error}</p>
        )}
        <DiagnosticsCopyActions onCopyDiagnostics={onCopyDiagnostics}>
          <ActionButton onClick={onRetry}>Retry</ActionButton>
        </DiagnosticsCopyActions>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function StartupScreen({
  status,
  logs,
  error,
  currentStepIndex,
  progressDetail,
  startupMessage,
  elevationPackages,
  onInstall,
  onRetry,
  onRetryInstall,
  onApproveElevation,
  onStartServer,
  onCopyDiagnostics,
}: StartupScreenProps) {
  function renderContent() {
    switch (status) {
      case "checking":
        return <CheckingContent />;
      case "not-installed":
        return <NotInstalledContent onInstall={onInstall} />;
      case "installing":
        return (
          <InstallingContent
            logs={logs}
            currentStepIndex={currentStepIndex}
            progressDetail={progressDetail}
          />
        );
      case "install-error":
        return (
          <InstallErrorContent
            error={error}
            onRetryInstall={onRetryInstall}
            onCopyDiagnostics={onCopyDiagnostics}
          />
        );
      case "repairing":
        return <RepairingContent logs={logs} progressDetail={progressDetail} />;
      case "repair-error":
        return (
          <RepairErrorContent
            error={error}
            onRetry={onRetry}
            onCopyDiagnostics={onCopyDiagnostics}
          />
        );
      case "needs-elevation":
        return (
          <NeedsElevationContent
            elevationPackages={elevationPackages}
            onApproveElevation={onApproveElevation}
            onRetryInstall={onRetryInstall}
          />
        );
      case "starting":
        return <StartingContent key={startupMessage} message={startupMessage} />;
      case "running":
        return null;
      case "stopped":
        return <StoppedContent onStartServer={onStartServer} />;
      case "error":
        return (
          <ErrorContent
            error={error}
            onRetry={onRetry}
            onCopyDiagnostics={onCopyDiagnostics}
          />
        );
    }
  }

  return (
    <StartupSurface>
      <AnimatePresence mode="wait">
        <motion.div
          key={status}
          className="flex h-full w-full flex-col items-center justify-center text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2, ease: EASE_OUT_QUART }}
        >
          {renderContent()}
        </motion.div>
      </AnimatePresence>
    </StartupSurface>
  );
}

/** The chrome both full-window screens sit in, so they agree on insets and scrolling. */
function StartupSurface({ children }: { children: ReactNode }) {
  return (
    <div className="box-border flex h-full w-full flex-col items-center overflow-y-auto bg-background pb-6 pt-[var(--studio-startup-top-inset,0px)]">
      <div className="flex min-h-0 flex-1 w-full max-w-md items-center justify-center px-6">
        {children}
      </div>
    </div>
  );
}

/**
 * Shown from the moment a quit is requested until the process is gone. Separate from
 * StartupScreen because a quit can come from the running app, where no backend status
 * applies, and unanimated because it has to be on screen for the very next paint.
 *
 * A layer over the app rather than a replacement for it: a declined quit has to hand the
 * user back the tree they had, in-flight generations and unsaved drafts included. The
 * z-index clears the titlebar, the download stack and the floating panels above it:
 * it is Z_LAYER.STARTUP_SCREEN, which lib/z-layers puts over both.
 */
export function ClosingScreen() {
  return (
    // pointer-events-auto, not the inherited default: Radix parks pointer-events:none on
    // <body> while any modal layer is open, and a quit raised from the window controls,
    // the tray or Alt+F4 never closes that layer. Inheriting it would make the overlay
    // click-through onto the dialog it is hiding, so clicks meant for a screen that says
    // the app is closing would land on buttons the user can no longer see.
    <div className="pointer-events-auto fixed inset-0 z-[9999]">
      <StartupSurface>
        <div className="flex h-full w-full flex-col items-center justify-center text-center">
          <ClosingContent />
        </div>
      </StartupSurface>
    </div>
  );
}
