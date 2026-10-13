// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { create } from "zustand";

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Spinner } from "@/components/ui/spinner";
import { useSettingsDialogStore } from "@/features/settings";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  type SandboxCapability,
  type SandboxSetupJob,
  forgetSandboxCapability,
  loadSandboxSetup,
  loadSettledSandboxCapability,
  sandboxReady,
  startSandboxSetup,
} from "./api/sandbox-capability";
import { SANDBOX_SETUP_POLL_MS, sandboxSetupView } from "./sandbox-setup-state";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

/** Mounted once at the chat-page root so it outlives the menu that opened it. */
export const useSandboxSetupDialogStore = create<{
  open: boolean;
  setOpen: (open: boolean) => void;
}>((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
}));

const NOTE_CLASS = "text-xs leading-relaxed";

/** Switching the sandbox to High without an OS sandbox. Opening it never starts a setup; only the
 *  install button does, and a setup that works turns High on. */
export function SandboxSetupDialog({
  open,
  onOpenChange,
  onLearnMore,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Defaults to opening Settings > Sandbox at Permissions. */
  onLearnMore?: () => void;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      {/* Mounted per opening, so every opening starts from a fresh check. */}
      {open ? (
        <SandboxSetupContent onOpenChange={onOpenChange} onLearnMore={onLearnMore} />
      ) : null}
    </AlertDialog>
  );
}

function SandboxSetupContent({
  onOpenChange,
  onLearnMore,
}: {
  onOpenChange: (open: boolean) => void;
  onLearnMore?: () => void;
}) {
  const t = useT();
  const setSandboxLevel = useChatRuntimeStore((s) => s.setSandboxLevel);
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const [capability, setCapability] = useState<SandboxCapability | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [job, setJob] = useState<SandboxSetupJob | null>(null);
  const [consent, setConsent] = useState(false);
  const [stillUnavailable, setStillUnavailable] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const mounted = useRef(true);
  // Only the newest check applies (a retry can overtake a slow first read).
  const checks = useRef(0);
  // A level picked while the setup ran wins over turning High on at the end.
  const levelAtOpen = useRef(useChatRuntimeStore.getState().sandboxLevel);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const check = useCallback(() => {
    const id = ++checks.current;
    void loadSettledSandboxCapability().then((next) => {
      if (!mounted.current || id !== checks.current) return;
      if (next === null) {
        setLoadFailed(true);
        return;
      }
      setCapability(next);
      if (next.canRunSetup) {
        void loadSandboxSetup(t("sandboxSetup.startError"))
          .then((current) => {
            if (mounted.current && current.state === "running") setJob(current);
          })
          .catch(() => undefined);
      }
    });
  }, [t]);

  useEffect(() => {
    check();
  }, [check]);

  const retry = () => {
    setLoadFailed(false);
    check();
  };

  const finish = useCallback(
    (finished: SandboxSetupJob) => {
      if (finished.state !== "succeeded") return;
      forgetSandboxCapability();
      void loadSettledSandboxCapability().then((next) => {
        if (!mounted.current) return;
        if (next) setCapability(next);
        if (next && sandboxReady(next)) {
          if (useChatRuntimeStore.getState().sandboxLevel === levelAtOpen.current) {
            setSandboxLevel("high");
          }
          toast.success(t("sandboxSetup.succeeded"));
          onOpenChange(false);
        } else {
          setStillUnavailable(true);
        }
      });
    },
    [onOpenChange, setSandboxLevel, t],
  );

  useEffect(() => {
    if (job?.state !== "running") return;
    const timer = window.setTimeout(() => {
      void loadSandboxSetup(t("sandboxSetup.startError"))
        .then((next) => {
          if (!mounted.current) return;
          setJob(next);
          if (next.state !== "running") finish(next);
        })
        .catch((pollError) => {
          if (!mounted.current) return;
          setActionError(
            pollError instanceof Error
              ? pollError.message
              : t("sandboxSetup.startError"),
          );
          setJob(null);
        });
    }, SANDBOX_SETUP_POLL_MS);
    return () => window.clearTimeout(timer);
  }, [job, t, finish]);

  const view = sandboxSetupView({ capability, job, consent, loadFailed });

  const install = async () => {
    const operation = capability?.setupAction;
    if (!operation) return;
    setActionError(null);
    setStillUnavailable(false);
    try {
      const started = await startSandboxSetup(
        operation,
        { consentDaclFallback: view.showConsent && consent },
        t("sandboxSetup.startError"),
      );
      if (!mounted.current) return;
      setJob(started);
      if (started.state !== "running") finish(started);
    } catch (startError) {
      if (!mounted.current) return;
      setActionError(
        startError instanceof Error
          ? startError.message
          : t("sandboxSetup.startError"),
      );
    }
  };

  const copy = async () => {
    if (await copyToClipboard(view.command)) {
      // The command itself stays out of the popup; Settings > Sandbox shows it in full.
      toast.success(t("sandboxSetup.copiedRunIt"));
    } else {
      toast.error(t("sandboxSetup.copyFailed"));
    }
  };

  return (
    <AlertDialogContent>
      <AlertDialogHeader>
        <AlertDialogTitle>{t("sandboxSetup.levelTitle")}</AlertDialogTitle>
        <AlertDialogDescription>
          {t("sandboxSetup.levelDescription")}
        </AlertDialogDescription>
      </AlertDialogHeader>

      <div className="flex min-w-0 flex-col gap-3">
        {view.loadFailed ? (
          <p className={`${NOTE_CLASS} text-destructive`}>
            {t("sandboxSetup.checkFailed")}
          </p>
        ) : null}
        {view.checking ? (
          <p
            className={`${NOTE_CLASS} flex items-center gap-2 text-muted-foreground`}
          >
            <Spinner />
            {t("sandboxSetup.checking")}
          </p>
        ) : null}
        {view.showConsent ? (
          <div className="flex flex-col gap-2">
            <p className={`${NOTE_CLASS} text-muted-foreground`}>
              {t("sandboxSetup.windowsNote")}
            </p>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={consent}
                disabled={view.running}
                onCheckedChange={(checked) => setConsent(checked === true)}
              />
              {t("settings.sandbox.optInLabel")}
            </label>
          </div>
        ) : null}
        {view.running ? (
          <p
            className={`${NOTE_CLASS} flex items-center gap-2 text-muted-foreground`}
          >
            <Spinner />
            {t("sandboxSetup.running")}
          </p>
        ) : null}
        {view.running ? (
          <p className={`${NOTE_CLASS} text-muted-foreground`}>
            {t("sandboxSetup.keepsRunning")}
          </p>
        ) : null}
        {stillUnavailable ? (
          <p className={`${NOTE_CLASS} text-destructive`}>
            {t("sandboxSetup.stillUnavailable")}
          </p>
        ) : null}
        {view.result === "declined" ? (
          <p className={`${NOTE_CLASS} text-destructive`}>
            {t("sandboxSetup.declined")}
          </p>
        ) : null}
        {view.result === "failed" ? (
          <p className={`${NOTE_CLASS} text-destructive`}>
            {t("sandboxSetup.failed")}
          </p>
        ) : null}
        {view.note ? (
          <p className={`${NOTE_CLASS} text-destructive`}>{view.note}</p>
        ) : null}
        {view.outputLines.length > 0 ? (
          <pre className="whitespace-pre-wrap break-words font-mono text-ui-11 text-muted-foreground">
            {view.outputLines.join("\n")}
          </pre>
        ) : null}
        {actionError ? (
          <p className={`${NOTE_CLASS} text-destructive`}>{actionError}</p>
        ) : null}
        {view.showOwnerOnly ? (
          <p className={`${NOTE_CLASS} text-muted-foreground`}>
            {t("sandboxSetup.ownerOnly")}
          </p>
        ) : null}
      </div>

      <AlertDialogFooter className="flex-wrap">
        {view.loadFailed ? (
          <Button size="sm" onClick={retry}>
            {t("sandboxSetup.retry")}
          </Button>
        ) : null}
        {view.install ? (
          <Button
            size="sm"
            disabled={view.installDisabled}
            onClick={() => void install()}
          >
            {view.running ? <Spinner /> : null}
            {view.install === "windows"
              ? t("sandboxSetup.windowsSetup")
              : t("sandboxSetup.install")}
          </Button>
        ) : null}
        {view.command ? (
          <Button
            size="sm"
            variant="outline"
            title={view.command}
            onClick={() => void copy()}
          >
            {t("sandboxSetup.copyCommand")}
          </Button>
        ) : null}
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            // Closing does not stop a running setup; its result shows in Settings > Sandbox.
            onOpenChange(false);
            if (onLearnMore) {
              onLearnMore();
            } else {
              // Deferred past the dialog's focus restore.
              setTimeout(
                () => openSettings("sandbox", { scrollTarget: "sandbox-permissions" }),
                0,
              );
            }
          }}
        >
          {t("settings.sandbox.learnMore")}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            setSandboxLevel("low");
            onOpenChange(false);
          }}
        >
          {t("sandboxSetup.useLow")}
        </Button>
      </AlertDialogFooter>
    </AlertDialogContent>
  );
}

export function RootSandboxSetupDialog() {
  const open = useSandboxSetupDialogStore((s) => s.open);
  const setOpen = useSandboxSetupDialogStore((s) => s.setOpen);
  return <SandboxSetupDialog open={open} onOpenChange={setOpen} />;
}
