// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ShimmerButton } from "@/components/ui/shimmer-button";
import type { BackendStatus } from "@/hooks/use-tauri-backend";
import { AnimatePresence, motion } from "motion/react";

interface StartupScreenProps {
  status: BackendStatus;
  logs: string[];
  error: string | null;
  currentStepIndex: number;
  progressDetail: string | null;
  elevationPackages: string[];
  onInstall: () => void;
  onRetry: () => void;
  onRetryInstall: () => void;
  onApproveElevation: () => void;
  onStartServer: () => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INSTALL_STEPS = [
  "Detecting your system",
  "Checking dependencies",
  "Setting up package manager",
  "Creating Python environment",
  "Installing ML framework",
  "Installing Unsloth",
  "Finalizing setup",
] as const;

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TealSpinner({ size = 24 }: { size?: number }) {
  return (
    <span
      className="inline-block animate-spin rounded-full border-2 border-primary border-t-transparent"
      style={{ width: size, height: size, animationDuration: "0.8s" }}
    />
  );
}

function Logo() {
  return (
    <div className="flex flex-col items-center gap-4">
      <img src="/sticker.png" alt="Unsloth" className="h-[72px] w-[72px] object-contain" />
      <img src="/studio.png" alt="Unsloth Studio" className="h-auto w-[250px] object-contain dark:invert" />
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
        <TealSpinner />
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
        <p className="mt-4 text-xs font-bold text-muted-foreground">
          To install Unsloth, click Get Started.
        </p>
      </div>
      <div className="mb-10">
        <ShimmerButton
          onClick={onInstall}
          shimmerColor="#a7f3d0"
          background="oklch(0.696 0.17 162.48)"
          className="text-sm font-medium"
        >
          Get Started
        </ShimmerButton>
      </div>
    </div>
  );
}

function InstallingContent({
  currentStepIndex,
  progressDetail,
}: {
  currentStepIndex: number;
  progressDetail: string | null;
}) {
  const stepNum = Math.max(0, currentStepIndex) + 1;
  const stepLabel = INSTALL_STEPS[Math.min(currentStepIndex, INSTALL_STEPS.length - 1)];

  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-2">
        <TealSpinner />
        <p className="text-sm font-bold text-foreground">Installing...</p>
        <p className="text-sm font-bold text-muted-foreground">
          Please wait a few mins, then you can start training.
        </p>
        {currentStepIndex >= 0 && (
          <p className="mt-1 text-xs font-bold text-muted-foreground">
            Step {stepNum} of {INSTALL_STEPS.length}: {stepLabel}
          </p>
        )}
        {progressDetail && (
          <p className="text-xs text-muted-foreground/70">{progressDetail}</p>
        )}
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
  const latest = progressDetail ?? logs.at(-1);

  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-2">
        <TealSpinner />
        <p className="text-sm font-bold text-foreground">Updating existing Studio install...</p>
        {latest && (
          <p className="max-w-xs text-center text-xs text-muted-foreground">{latest}</p>
        )}
      </div>
    </div>
  );
}

function InstallErrorContent({
  error,
  logs,
  onRetryInstall,
}: {
  error: string | null;
  logs: string[];
  onRetryInstall: () => void;
}) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-destructive">Setup ran into a problem</p>
        {error && (
          <p className="max-w-xs text-center text-xs text-muted-foreground">{error}</p>
        )}
        <div className="mt-4 flex gap-3">
          <ActionButton
            variant="secondary"
            onClick={() => void navigator.clipboard.writeText(logs.join("\n"))}
          >
            Copy Logs
          </ActionButton>
          <ActionButton onClick={onRetryInstall}>Try Again</ActionButton>
        </div>
      </div>
    </>
  );
}

function RepairErrorContent({
  error,
  logs,
  onRetry,
}: {
  error: string | null;
  logs: string[];
  onRetry: () => void;
}) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-destructive">Update failed</p>
        {error && (
          <p className="max-w-md text-center text-xs text-muted-foreground">{error}</p>
        )}
        <div className="mt-4 flex gap-3">
          <ActionButton
            variant="secondary"
            onClick={() => void navigator.clipboard.writeText(logs.join("\n"))}
          >
            Copy Logs
          </ActionButton>
          <ActionButton onClick={onRetry}>Retry</ActionButton>
        </div>
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

function StartingContent() {
  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-2">
        <TealSpinner />
        <p className="text-sm text-muted-foreground">Starting server...</p>
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
  logs,
  onRetry,
}: {
  error: string | null;
  logs: string[];
  onRetry: () => void;
}) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-destructive">Something went wrong</p>
        {error && (
          <p className="max-w-md text-center text-xs text-muted-foreground">{error}</p>
        )}
        <div className="mt-4 flex gap-3">
          <ActionButton
            variant="secondary"
            onClick={() => void navigator.clipboard.writeText(logs.join("\n"))}
          >
            Copy Logs
          </ActionButton>
          <ActionButton onClick={onRetry}>Retry</ActionButton>
        </div>
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
  elevationPackages,
  onInstall,
  onRetry,
  onRetryInstall,
  onApproveElevation,
  onStartServer,
}: StartupScreenProps) {
  function renderContent() {
    switch (status) {
      case "checking":
        return <CheckingContent />;
      case "not-installed":
        return <NotInstalledContent onInstall={onInstall} />;
      case "installing":
        return <InstallingContent currentStepIndex={currentStepIndex} progressDetail={progressDetail} />;
      case "install-error":
        return <InstallErrorContent error={error} logs={logs} onRetryInstall={onRetryInstall} />;
      case "repairing":
        return <RepairingContent logs={logs} progressDetail={progressDetail} />;
      case "repair-error":
        return <RepairErrorContent error={error} logs={logs} onRetry={onRetry} />;
      case "needs-elevation":
        return (
          <NeedsElevationContent
            elevationPackages={elevationPackages}
            onApproveElevation={onApproveElevation}
            onRetryInstall={onRetryInstall}
          />
        );
      case "starting":
        return <StartingContent />;
      case "running":
        return null;
      case "stopped":
        return <StoppedContent onStartServer={onStartServer} />;
      case "error":
        return <ErrorContent error={error} logs={logs} onRetry={onRetry} />;
    }
  }

  return (
    <div className="flex h-screen w-full flex-col items-center bg-background">
      <div className="flex flex-1 w-full max-w-md items-center justify-center px-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={status}
            className="flex h-full w-full flex-col items-center text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2, ease: EASE_OUT_QUART }}
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
