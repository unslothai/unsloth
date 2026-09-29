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

/** Open state for the copy mounted once at the chat-page root, which the composer pill and the
 *  dictation menu open: both unmount with their menu, and the dialog must outlive it. */
export const useSandboxSetupDialogStore = create<{
  open: boolean;
  setOpen: (open: boolean) => void;
}>((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
}));

const NOTE_CLASS = "text-xs leading-relaxed";

/** Shown when "Full access in sandbox" is picked on a computer whose OS sandbox does not work
 *  yet. Picking the mode never starts a setup; only the install button does. Cancel leaves the
 *  previous mode, "Use it anyway" keeps the pick with risky calls still asking. */
export function SandboxSetupDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      {/* Mounted per opening, so every opening starts from a fresh check. */}
      {open ? <SandboxSetupContent onOpenChange={onOpenChange} /> : null}
    </AlertDialog>
  );
}

function SandboxSetupContent({
  onOpenChange,
}: {
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);
  const [capability, setCapability] = useState<SandboxCapability | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [job, setJob] = useState<SandboxSetupJob | null>(null);
  const [consent, setConsent] = useState(false);
  const [stillUnavailable, setStillUnavailable] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const check = useCallback(() => {
    void loadSettledSandboxCapability().then((next) => {
      if (!mounted.current) return;
      if (next === null) {
        // Never an endless spinner: say the check failed and offer Retry.
        setLoadFailed(true);
        return;
      }
      setCapability(next);
      // A setup already running (another window, or before this dialog reopened) keeps
      // reporting here. Only the owner may read it.
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
          setPermissionMode("off");
          toast.success(t("sandboxSetup.succeeded"));
          onOpenChange(false);
        } else {
          setStillUnavailable(true);
        }
      });
    },
    [onOpenChange, setPermissionMode, t],
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
      toast.success(t("sandboxSetup.copied"));
    } else {
      toast.error(t("sandboxSetup.copyFailed"));
    }
  };

  return (
    <AlertDialogContent>
      <AlertDialogHeader>
        <AlertDialogTitle>{t("sandboxSetup.title")}</AlertDialogTitle>
        <AlertDialogDescription>
          {t("sandboxSetup.description")}
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
        {capability?.reason ? (
          <p className={`${NOTE_CLASS} text-muted-foreground`}>
            {capability.reason}
          </p>
        ) : null}
        {view.showConsent ? (
          <div className="flex flex-col gap-2">
            <p className={`${NOTE_CLASS} text-muted-foreground`}>
              {t("settings.sandbox.disclosure")}
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
          <pre className="whitespace-pre-wrap break-words font-mono text-[11px] text-muted-foreground">
            {view.outputLines.join("\n")}
          </pre>
        ) : null}
        {actionError ? (
          <p className={`${NOTE_CLASS} text-destructive`}>{actionError}</p>
        ) : null}
        {view.showOwnerOnly && !view.command ? (
          <p className={`${NOTE_CLASS} text-muted-foreground`}>
            {t("sandboxSetup.ownerOnly")}
          </p>
        ) : null}
        {view.command ? (
          <div className="flex min-w-0 flex-col gap-1">
            <p className={`${NOTE_CLASS} text-muted-foreground`}>
              {view.showOwnerOnly
                ? t("sandboxSetup.ownerOnly")
                : view.showRunInTerminal
                  ? t("sandboxSetup.runInTerminal")
                  : t("sandboxSetup.commandHint")}
            </p>
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted px-2 py-1.5 font-mono text-[11px]">
              {view.command}
            </pre>
          </div>
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
          <Button size="sm" variant="outline" onClick={() => void copy()}>
            {t("sandboxSetup.copyCommand")}
          </Button>
        ) : null}
        <Button
          size="sm"
          variant="outline"
          disabled={view.checking}
          onClick={() => {
            setPermissionMode("off");
            onOpenChange(false);
          }}
        >
          {t("sandboxSetup.useAnyway")}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => onOpenChange(false)}>
          {/* Closing does not stop a running setup; its result shows in Settings > Sandbox. */}
          {view.running ? t("sandboxSetup.close") : t("sandboxSetup.cancel")}
        </Button>
      </AlertDialogFooter>
    </AlertDialogContent>
  );
}

/** The copy mounted once at the chat-page root, beside the Bypass permissions confirmation. */
export function RootSandboxSetupDialog() {
  const open = useSandboxSetupDialogStore((s) => s.open);
  const setOpen = useSandboxSetupDialogStore((s) => s.setOpen);
  return <SandboxSetupDialog open={open} onOpenChange={setOpen} />;
}
