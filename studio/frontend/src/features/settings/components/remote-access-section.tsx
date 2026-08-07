


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
import { usePlatformStore } from "@/config/env";
import {
  loadRemoteAccess,
  startRemoteAccess,
  stopRemoteAccess,
  updateRemoteAccessAutoStart,
} from "@/features/settings/api/remote-access";
import {
  type RemoteAccessStatus,
  remoteAccessAutoStartReadOnly,
  remoteAccessBlockMessage,
  remoteAccessPollDelay,
  remoteAccessSelfStopPoll,
  remoteAccessStopDisconnectsOrigin,
} from "@/features/settings/api/remote-access-state";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  Copy01Icon,
  Globe02Icon,
  QrCodeIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import QRCode from "react-qr-code";
import { ChangePasswordDialog } from "./change-password-dialog";
import { SettingsRow } from "./settings-row";

type RemoteAccessOperation = "start" | "stop" | "auto";

const STATE_LABEL: Record<RemoteAccessStatus["state"], string> = {
  off: "Off",
  starting: "Starting",
  online: "Online",
  stopping: "Stopping",
  error: "Error",
};

const OWNER_LABEL: Record<
  Exclude<RemoteAccessStatus["managedBy"], null>,
  string
> = {
  launch: "Launch managed",
  settings: "Settings managed",
  colab: "Colab managed",
};

function stateDotClass(state?: RemoteAccessStatus["state"]): string {
  if (state === "online") {
    return "bg-emerald-500";
  }
  if (state === "starting" || state === "stopping") {
    return "animate-pulse bg-blue-500";
  }
  return state === "error" ? "bg-red-500" : "bg-muted-foreground";
}

function AccessStatus({ status }: { status: RemoteAccessStatus | null }) {
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

function CopyRemoteUrlButton({ url }: { url: string }) {
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
      {copied ? "Copied" : "Copy URL"}
    </Button>
  );
}

function RemoteUrlQrButton({ url }: { url: string }) {
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
            Scan to open the remote URL in your phone’s browser.
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

function RemoteUrlPanel({ url }: { url: string | null }) {
  if (!url) {
    return null;
  }
  return (
    <div className="flex flex-col gap-1.5 border-t border-border/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-foreground">Remote URL</span>
        <div className="flex items-center gap-2">
          <RemoteUrlQrButton url={url} />
          <CopyRemoteUrlButton url={url} />
        </div>
      </div>
      <code className="block w-full break-all rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-xs text-foreground">
        {url}
      </code>
      <span className="text-xs text-muted-foreground leading-snug">
        Anyone with this URL and the remote password can sign in.
      </span>
    </div>
  );
}

// Desktop signs in with a local secret, so the account password exists only for
// remote browsers and is managed here rather than in the General tab.
function RemotePasswordRow({
  status,
  onDone,
}: {
  status: RemoteAccessStatus | null;
  onDone: () => void;
}) {
  if (!(isTauri && status)) {
    return null;
  }
  return (
    <SettingsRow
      label="Remote password"
      description="Remote browsers sign in as unsloth. The Unsloth Desktop App keeps signing in automatically."
    >
      <ChangePasswordDialog initial={status.passwordPending} onDone={onDone} />
    </SettingsRow>
  );
}

export function RemoteAccessSection() {
  const [status, setStatus] = useState<RemoteAccessStatus | null>(null);
  const [busy, setBusy] = useState<RemoteAccessOperation | null>(null);
  const [pollRevision, setPollRevision] = useState(0);
  const [pollEnabled, setPollEnabled] = useState(true);
  const mutationEpoch = useRef(0);
  const pollSuppressed = useRef(false);
  const selfStopDisconnectExpected = useRef(false);

  const applyStatus = useCallback((next: RemoteAccessStatus) => {
    setStatus(next);
    usePlatformStore.setState({ cloudflareUrl: next.url });
  }, []);

  // A password change rotates credentials outside this section's requests;
  // discard any in-flight poll and re-read so the block resolves at once.
  const refreshStatus = useCallback(() => {
    mutationEpoch.current += 1;
    setPollEnabled(true);
    setPollRevision((revision) => revision + 1);
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: pollRevision intentionally restarts polling after a mutation
  useEffect(() => {
    if (!pollEnabled) {
      return;
    }
    let stopped = false;
    let timer: number | null = null;
    const schedule = (next: RemoteAccessStatus | null) => {
      if (!stopped && !pollSuppressed.current) {
        timer = window.setTimeout(poll, remoteAccessPollDelay(next));
      }
    };
    const poll = () => {
      if (pollSuppressed.current) {
        return;
      }
      const epoch = mutationEpoch.current;
      loadRemoteAccess()
        .then((next) => {
          if (
            !stopped &&
            !pollSuppressed.current &&
            mutationEpoch.current === epoch
          ) {
            const settled = remoteAccessSelfStopPoll(
              next,
              selfStopDisconnectExpected.current,
            );
            selfStopDisconnectExpected.current = settled.expectingDisconnect;
            if (settled.apply) {
              applyStatus(next);
            }
          }
          schedule(next);
        })
        .catch(() => {
          if (mutationEpoch.current !== epoch) {
            return;
          }
          if (selfStopDisconnectExpected.current) {
            setPollEnabled(false);
            return;
          }
          schedule(null);
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
    operation: RemoteAccessOperation,
    request: () => Promise<RemoteAccessStatus>,
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
      // Polling resumes below and reconciles the visible state.
    } finally {
      setBusy(null);
      pollSuppressed.current = false;
      // A stop through the tunnel latches polling off when its own origin dies.
      // Any later action means the user is on a reachable origin, so resume.
      setPollEnabled(true);
      setPollRevision((revision) => revision + 1);
    }
  };

  const start = () => perform("start", startRemoteAccess);
  const stop = () =>
    perform(
      "stop",
      stopRemoteAccess,
      remoteAccessStopDisconnectsOrigin(
        status?.url ?? null,
        typeof window === "undefined" ? "" : window.location.origin,
      ),
    );
  const setAutoStart = (enabled: boolean) =>
    perform("auto", () => updateRemoteAccessAutoStart(enabled));

  const blockMessage = remoteAccessBlockMessage(
    status?.blockReason ?? null,
    isTauri,
  );
  const stopAction =
    status?.canStop === true ||
    status?.state === "starting" ||
    status?.state === "online" ||
    status?.state === "stopping";
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
      data-settings-label="Remote access"
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <HugeiconsIcon
              icon={Globe02Icon}
              className="size-4 text-foreground"
            />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-base font-semibold font-heading text-foreground">
                Remote access
              </h2>
              <AccessStatus status={status} />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Use Unsloth and its APIs from other devices through a secure,
              temporary Cloudflare URL.
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
        message={blockMessage ?? status?.error}
        destructive={!blockMessage}
      />
      <RemoteUrlPanel url={status?.url ?? null} />

      <div className="border-t border-border/60 px-4 py-1">
        <RemotePasswordRow status={status} onDone={refreshStatus} />

        <SettingsRow
          label="Start automatically"
          description="Create a new remote URL each time Unsloth starts. Stopping remote access now won’t turn this off."
        >
          <Switch
            checked={status?.autoStart ?? false}
            disabled={busy !== null || remoteAccessAutoStartReadOnly(status)}
            onCheckedChange={setAutoStart}
            aria-label="Start automatically"
          />
        </SettingsRow>
      </div>
    </section>
  );
}
