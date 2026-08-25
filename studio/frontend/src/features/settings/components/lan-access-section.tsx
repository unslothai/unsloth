// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import {
  loadLanAccess,
  startLanAccess,
  stopLanAccess,
  updateLanAccessAutoStart,
} from "@/features/settings/api/lan-access";
import {
  LAN_ACCESS_POLL_MS,
  type LanAccessStatus,
  lanAccessAutoStartReadOnly,
  lanAccessBlockMessage,
  lanAccessErrorMessage,
  lanAccessStopDisconnectsOrigin,
} from "@/features/settings/api/lan-access-state";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon, QrCodeIcon, Wifi01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import QRCode from "react-qr-code";
import { SettingsRow } from "./settings-row";

type LanAccessOperation = "start" | "stop" | "auto";

const STATE_LABEL: Record<LanAccessStatus["state"], string> = {
  off: "Off",
  online: "Online",
  error: "Error",
};

const OWNER_LABEL: Record<
  Exclude<LanAccessStatus["managedBy"], null>,
  string
> = {
  launch: "Launch managed",
  settings: "Settings managed",
};

function stateDotClass(state?: LanAccessStatus["state"]): string {
  if (state === "online") {
    return "bg-emerald-500";
  }
  return state === "error" ? "bg-red-500" : "bg-muted-foreground";
}

function AccessStatus({ status }: { status: LanAccessStatus | null }) {
  const owner = status?.managedBy ? OWNER_LABEL[status.managedBy] : null;
  return (
    <output
      className="flex items-center gap-1.5 text-xs text-muted-foreground"
      aria-live="polite"
    >
      <span
        className={cn("size-2 rounded-full", stateDotClass(status?.state))}
      />
      {status ? STATE_LABEL[status.state] : "Unavailable"}
      {owner ? ` · ${owner}` : ""}
    </output>
  );
}

function CopyLanUrlButton({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (copyTimer.current !== null) {
        window.clearTimeout(copyTimer.current);
      }
    };
  }, []);
  return (
    <Button
      type="button"
      size="sm"
      variant="outline"
      className="gap-1.5"
      aria-label={`Copy ${url}`}
      onClick={async () => {
        if (!(await copyToClipboard(url))) {
          return;
        }
        setCopied(true);
        if (copyTimer.current !== null) {
          window.clearTimeout(copyTimer.current);
        }
        copyTimer.current = window.setTimeout(() => setCopied(false), 1800);
      }}
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        className="size-3.5"
      />
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

function LanUrlQrButton({ url }: { url: string }) {
  return (
    <Dialog>
      <DialogTrigger asChild={true}>
        <Button type="button" size="sm" variant="outline" className="gap-1.5">
          <HugeiconsIcon icon={QrCodeIcon} className="size-3.5" />
          QR
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xs">
        <DialogHeader>
          <DialogTitle>Open on your phone</DialogTitle>
          <DialogDescription>
            Scan from a device on the same network to open this address.
          </DialogDescription>
        </DialogHeader>
        <div className="mx-auto mt-2 rounded-md bg-white p-3">
          <QRCode value={url} size={192} />
        </div>
        <code className="block break-all text-center font-mono text-xs text-muted-foreground">
          {url}
        </code>
      </DialogContent>
    </Dialog>
  );
}

function StatusMessage({
  message,
  destructive,
}: {
  message?: string | null;
  destructive?: boolean;
}) {
  if (!message) {
    return null;
  }
  return (
    <p
      className={cn(
        "border-t border-border/60 px-4 py-2.5 text-xs leading-snug",
        destructive ? "text-destructive" : "text-muted-foreground",
      )}
    >
      {message}
    </p>
  );
}

function LanUrlPanel({ status }: { status: LanAccessStatus | null }) {
  if (!status || status.urls.length === 0) {
    return null;
  }
  const [primary] = status.urls;
  return (
    <div className="flex flex-col gap-1.5 border-t border-border/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-foreground">
          {status.urls.length > 1 ? "Network addresses" : "Network address"}
        </span>
        <LanUrlQrButton url={primary} />
      </div>
      {status.urls.map((url) => (
        <div key={url} className="flex items-center gap-2">
          <code className="block w-full min-w-0 break-all rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs text-foreground">
            {url}
          </code>
          <CopyLanUrlButton url={url} />
        </div>
      ))}
      {status.publicUrls.length > 0 ? (
        <span className="text-xs text-destructive leading-snug">
          {status.publicUrls[0]} is a public internet address, so this reaches
          beyond your local network. Anyone who has the password or an API key
          can sign in and run code on this machine.
        </span>
      ) : (
        <span className="text-xs text-muted-foreground leading-snug">
          {status.servesWebUi
            ? "Anyone on this network who has the password or an API key can sign in and run code on this machine."
            : "This launch serves the API only, so devices on the network can call the API but not open the web UI."}
        </span>
      )}
    </div>
  );
}

export function LanAccessSection() {
  const [status, setStatus] = useState<LanAccessStatus | null>(null);
  const [busy, setBusy] = useState<LanAccessOperation | null>(null);
  const [pollRevision, setPollRevision] = useState(0);
  const [pollEnabled, setPollEnabled] = useState(true);
  const mutationEpoch = useRef(0);
  const pollSuppressed = useRef(false);
  const selfStopDisconnectExpected = useRef(false);

  const applyStatus = useCallback((next: LanAccessStatus) => {
    setStatus(next);
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: pollRevision intentionally restarts polling after a mutation
  useEffect(() => {
    if (!pollEnabled) {
      return;
    }
    let stopped = false;
    let timer: number | null = null;
    const schedule = () => {
      if (!stopped && !pollSuppressed.current) {
        timer = window.setTimeout(poll, LAN_ACCESS_POLL_MS);
      }
    };
    const poll = () => {
      if (pollSuppressed.current) {
        return;
      }
      const epoch = mutationEpoch.current;
      loadLanAccess()
        .then((next) => {
          if (
            !stopped &&
            !pollSuppressed.current &&
            mutationEpoch.current === epoch
          ) {
            selfStopDisconnectExpected.current = false;
            applyStatus(next);
          }
          schedule();
        })
        .catch(() => {
          if (mutationEpoch.current !== epoch) {
            return;
          }
          // a stop from a LAN address kills this page's own origin, so stop polling
          if (selfStopDisconnectExpected.current) {
            setPollEnabled(false);
            return;
          }
          schedule();
        });
    };
    poll();
    return () => {
      stopped = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [applyStatus, pollEnabled, pollRevision]);

  const perform = async (
    operation: LanAccessOperation,
    request: () => Promise<LanAccessStatus>,
    pausePollingAfterSuccess = false,
  ) => {
    mutationEpoch.current += 1;
    pollSuppressed.current = true;
    setBusy(operation);
    try {
      applyStatus(await request());
      if (pausePollingAfterSuccess) {
        selfStopDisconnectExpected.current = true;
      }
    } catch {
      // polling resumes below and reconciles the visible state
    } finally {
      setBusy(null);
      pollSuppressed.current = false;
      setPollEnabled(true);
      setPollRevision((revision) => revision + 1);
    }
  };

  const start = () => perform("start", startLanAccess);
  const stop = () =>
    perform(
      "stop",
      stopLanAccess,
      lanAccessStopDisconnectsOrigin(
        status?.urls ?? [],
        typeof window === "undefined" ? "" : window.location.origin,
      ),
    );
  const setAutoStart = (enabled: boolean) =>
    perform("auto", () => updateLanAccessAutoStart(enabled));

  const blockMessage = lanAccessBlockMessage(
    status?.blockReason ?? null,
    isTauri,
  );
  const errorMessage = lanAccessErrorMessage(status?.error ?? null);
  const stopAction = status?.state === "online";
  const actionDisabled =
    busy !== null || (stopAction ? !status?.canStop : !status?.canStart);
  const actionLabel =
    busy === "start"
      ? "Starting…"
      : busy === "stop"
        ? "Stopping…"
        : stopAction
          ? "Stop"
          : "Start";

  return (
    <section
      data-settings-label="LAN access"
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <HugeiconsIcon
              icon={Wifi01Icon}
              className="size-4 text-foreground"
            />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-base font-semibold font-heading text-foreground">
                LAN access
              </h2>
              <AccessStatus status={status} />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Use Unsloth and its APIs from other devices on your Wi-Fi or wired
              network.
            </p>
          </div>
        </div>
        <Button
          type="button"
          size="sm"
          variant={stopAction ? "outline" : "default"}
          className="min-w-20 shrink-0"
          onClick={stopAction ? stop : start}
          disabled={actionDisabled}
        >
          {actionLabel}
        </Button>
      </div>

      <StatusMessage
        message={blockMessage ?? errorMessage}
        destructive={!blockMessage}
      />
      <LanUrlPanel status={status} />

      <div className="border-t border-border/60 px-4 py-1">
        <SettingsRow
          label="Start automatically"
          description="Put Unsloth on the network each time it starts. Stopping LAN access now won’t turn this off."
        >
          <Switch
            checked={status?.autoStart ?? false}
            disabled={busy !== null || lanAccessAutoStartReadOnly(status)}
            onCheckedChange={setAutoStart}
            aria-label="Start automatically"
          />
        </SettingsRow>
      </div>
    </section>
  );
}
