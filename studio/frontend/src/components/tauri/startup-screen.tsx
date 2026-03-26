// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type BackendStatus =
  | "checking"
  | "not-installed"
  | "installing"
  | "install-error"
  | "needs-elevation"
  | "starting"
  | "running"
  | "stopped"
  | "error";

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
  { id: "platform", label: "Detecting your system" },
  { id: "dependencies", label: "Checking dependencies" },
  { id: "uv", label: "Setting up package manager" },
  { id: "venv", label: "Creating Python environment" },
  { id: "pytorch", label: "Installing ML framework" },
  { id: "unsloth", label: "Installing Unsloth" },
  { id: "setup", label: "Finalizing setup" },
] as const;

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

const MASCOT = {
  default: "/Sloth emojis/sloth pc emoji.png",
  welcome: "/Sloth emojis/Sloth w PC no Logo.png",
  sad: "/Sloth emojis/large sloth sad.png",
  success: "/Sloth emojis/UnSloth Sparkling large.png",
} as const;

// ---------------------------------------------------------------------------
// Sub-components (inline, not exported)
// ---------------------------------------------------------------------------

function Mascot({ src, size }: { src: string; size: number }) {
  return (
    <motion.img
      src={src}
      alt="Unsloth mascot"
      style={{ width: size, height: size }}
      className="object-contain"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, ease: EASE_OUT_QUART }}
    />
  );
}

function TealSpinner({ size = 16 }: { size?: number }) {
  return (
    <span
      className="inline-block rounded-full border-2 border-primary border-t-transparent"
      style={{
        width: size,
        height: size,
        animation: "spin 0.8s linear infinite",
      }}
    />
  );
}

function AnimatedDots() {
  return (
    <span className="inline-flex gap-0.5" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="text-muted-foreground"
          style={{
            animation: "dotPulse 1.4s ease-in-out infinite",
            animationDelay: `${i * 0.2}s`,
          }}
        >
          .
        </span>
      ))}
      <style>{`
        @keyframes dotPulse {
          0%, 80%, 100% { opacity: 0.2; }
          40% { opacity: 1; }
        }
      `}</style>
    </span>
  );
}

function LogDrawer({
  logs,
  defaultOpen = false,
}: {
  logs: string[];
  defaultOpen?: boolean;
}) {
  const [showLogs, setShowLogs] = useState(defaultOpen);
  const scrollRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Auto-expand when defaultOpen changes to true
  useEffect(() => {
    if (defaultOpen) setShowLogs(true);
  }, [defaultOpen]);

  // Auto-scroll when new logs arrive
  useEffect(() => {
    if (showLogs && endRef.current) {
      endRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, showLogs]);

  return (
    <div className="mt-4 w-full">
      <button
        type="button"
        className="text-sm text-muted-foreground underline cursor-pointer"
        onClick={() => setShowLogs((v) => !v)}
      >
        {showLogs ? "Hide details" : "Show details"}
      </button>

      {showLogs && (
        <div className="relative mt-2">
          <div
            ref={scrollRef}
            className="bg-muted rounded-lg p-3 max-h-48 overflow-y-auto font-mono text-xs text-muted-foreground"
          >
            {logs.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
            <div ref={endRef} />
          </div>
          <button
            type="button"
            className="absolute top-2 right-2 text-xs text-muted-foreground hover:text-foreground cursor-pointer"
            onClick={() => {
              void navigator.clipboard.writeText(logs.join("\n"));
            }}
          >
            Copy to clipboard
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-status renderers
// ---------------------------------------------------------------------------

function CheckingContent() {
  return (
    <>
      <Mascot src={MASCOT.default} size={80} />
      <h1 className="mt-4 text-xl font-semibold">
        Unsloth Studio
        <AnimatedDots />
      </h1>
    </>
  );
}

function NotInstalledContent({ onInstall }: { onInstall: () => void }) {
  return (
    <>
      <Mascot src={MASCOT.welcome} size={120} />
      <h1 className="mt-4 text-xl font-semibold">Welcome to Unsloth Studio</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Let's set up your ML environment. This usually takes 5-10 minutes.
      </p>
      <button
        type="button"
        onClick={onInstall}
        className="mt-6 rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/80 cursor-pointer"
      >
        Get Started
      </button>
    </>
  );
}

function InstallingContent({
  currentStepIndex,
  progressDetail,
  logs,
}: {
  currentStepIndex: number;
  progressDetail: string | null;
  logs: string[];
}) {
  return (
    <>
      <Mascot src={MASCOT.default} size={80} />
      <h1 className="mt-4 text-xl font-semibold">Setting up...</h1>

      {/* Step list */}
      <div className="mt-5 w-full space-y-2.5">
        {INSTALL_STEPS.map((step, idx) => {
          const isCompleted = idx < currentStepIndex;
          const isCurrent = idx === currentStepIndex;
          const isPending = idx > currentStepIndex;

          return (
            <div key={step.id}>
              <div className="flex items-center gap-3">
                {/* Icon column — fixed 20px */}
                <div className="flex w-5 shrink-0 items-center justify-center">
                  {isCompleted && (
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 20 20"
                      fill="none"
                      className="text-primary"
                    >
                      <circle cx="10" cy="10" r="10" fill="currentColor" />
                      <path
                        d="M6 10.5l2.5 2.5L14 8"
                        stroke="white"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  )}
                  {isCurrent && <TealSpinner size={16} />}
                  {isPending && (
                    <span className="block h-4 w-4 rounded-full border-2 border-muted-foreground/30" />
                  )}
                </div>

                {/* Label */}
                <span
                  className={
                    isCurrent
                      ? "text-sm font-medium text-foreground"
                      : isPending
                        ? "text-sm text-muted-foreground"
                        : "text-sm text-foreground"
                  }
                >
                  {step.label}
                </span>
              </div>

              {/* Progress detail for current step */}
              {isCurrent && progressDetail && (
                <p className="ml-8 mt-0.5 text-xs text-muted-foreground">
                  {progressDetail}
                </p>
              )}
            </div>
          );
        })}
      </div>

      <LogDrawer logs={logs} />
    </>
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
      <Mascot src={MASCOT.sad} size={96} />
      <h1 className="mt-4 text-xl font-semibold text-destructive">
        Setup ran into a problem
      </h1>
      {error && (
        <p className="mt-1 text-sm text-muted-foreground">{error}</p>
      )}
      <div className="mt-5 flex gap-3">
        <button
          type="button"
          className="rounded-lg bg-muted px-4 py-2 text-sm font-medium text-foreground hover:bg-muted/80 cursor-pointer"
          onClick={() => {
            void navigator.clipboard.writeText(logs.join("\n"));
          }}
        >
          Copy Logs
        </button>
        <button
          type="button"
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/80 cursor-pointer"
          onClick={onRetryInstall}
        >
          Try Again
        </button>
      </div>
      <LogDrawer logs={logs} defaultOpen />
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
      <Mascot src={MASCOT.default} size={80} />
      <h1 className="mt-4 text-xl font-semibold">Permission needed</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        The following system packages need to be installed:
      </p>
      <div className="mt-3 w-full bg-muted rounded-lg p-3 font-mono text-sm">
        {elevationPackages.map((pkg) => (
          <div key={pkg}>{pkg}</div>
        ))}
      </div>
      <div className="mt-5 flex gap-3">
        <button
          type="button"
          className="rounded-lg bg-muted px-4 py-2 text-sm font-medium text-foreground hover:bg-muted/80 cursor-pointer"
          onClick={onRetryInstall}
        >
          Cancel
        </button>
        <button
          type="button"
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/80 cursor-pointer"
          onClick={onApproveElevation}
        >
          Allow
        </button>
      </div>
    </>
  );
}

function StartingContent() {
  return (
    <>
      <Mascot src={MASCOT.default} size={80} />
      <h1 className="mt-4 text-xl font-semibold">
        Starting server
        <AnimatedDots />
      </h1>
    </>
  );
}

function StoppedContent({ onStartServer }: { onStartServer: () => void }) {
  return (
    <>
      <Mascot src={MASCOT.default} size={80} />
      <h1 className="mt-4 text-xl font-semibold">Server stopped</h1>
      <button
        type="button"
        className="mt-5 rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/80 cursor-pointer"
        onClick={onStartServer}
      >
        Start Server
      </button>
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
      <Mascot src={MASCOT.sad} size={96} />
      <h1 className="mt-4 text-xl font-semibold text-destructive">
        Something went wrong
      </h1>
      {error && (
        <p className="mt-1 text-sm text-muted-foreground">{error}</p>
      )}
      <div className="mt-5 flex gap-3">
        <button
          type="button"
          className="rounded-lg bg-muted px-4 py-2 text-sm font-medium text-foreground hover:bg-muted/80 cursor-pointer"
          onClick={() => {
            void navigator.clipboard.writeText(logs.join("\n"));
          }}
        >
          Copy Logs
        </button>
        <button
          type="button"
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/80 cursor-pointer"
          onClick={onRetry}
        >
          Retry
        </button>
      </div>
      <LogDrawer logs={logs} defaultOpen />
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
  // CSS keyframe for the spinner (injected once)
  const spinStyle = (
    <style>{`
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
    `}</style>
  );

  function renderContent() {
    switch (status) {
      case "checking":
        return <CheckingContent />;
      case "not-installed":
        return <NotInstalledContent onInstall={onInstall} />;
      case "installing":
        return (
          <InstallingContent
            currentStepIndex={currentStepIndex}
            progressDetail={progressDetail}
            logs={logs}
          />
        );
      case "install-error":
        return (
          <InstallErrorContent
            error={error}
            logs={logs}
            onRetryInstall={onRetryInstall}
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
        return <StartingContent />;
      case "running":
        // Should not be visible — parent hides the overlay when running.
        return null;
      case "stopped":
        return <StoppedContent onStartServer={onStartServer} />;
      case "error":
        return (
          <ErrorContent error={error} logs={logs} onRetry={onRetry} />
        );
    }
  }

  return (
    <div className="flex h-screen w-full items-center justify-center bg-background">
      {spinStyle}
      <div className="w-full max-w-md px-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={status}
            className="flex flex-col items-center text-center"
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
