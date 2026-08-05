// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
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
  remoteAccessStopDisconnectsOrigin,
} from "@/features/settings/api/remote-access-state";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChangePasswordDialog } from "./change-password-dialog";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

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
            if (selfStopDisconnectExpected.current && next.canStop) {
              selfStopDisconnectExpected.current = false;
            }
            applyStatus(next);
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

  const statusDescription =
    remoteAccessBlockMessage(status?.blockReason ?? null, isTauri) ??
    status?.error;
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
    <SettingsSection
      title="Remote access"
      description="Make Unsloth and its APIs available through a temporary Remote Secure Cloudflare URL."
    >
      <SettingsRow
        label="Status"
        labelAccessory={<AccessStatus status={status} />}
        description={statusDescription}
      >
        <Button
          type="button"
          size="sm"
          variant={stopAction ? "outline" : "default"}
          onClick={stopAction ? stop : start}
          disabled={actionDisabled}
        >
          {actionLabel}
        </Button>
      </SettingsRow>

      <RemotePasswordRow status={status} onDone={refreshStatus} />

      {status?.url ? (
        <SettingsRow
          label="Remote Secure Cloudflare URL"
          description={
            <code className="block break-all whitespace-normal font-mono">
              {status.url}
            </code>
          }
        >
          <CopyRemoteUrlButton url={status.url} />
        </SettingsRow>
      ) : null}

      <SettingsRow
        label="Start remote access when Unsloth starts"
        description="Unsloth will create a new remote URL each time it starts. Stopping remote access now won’t turn this setting off."
      >
        <Switch
          checked={status?.autoStart ?? false}
          disabled={busy !== null || remoteAccessAutoStartReadOnly(status)}
          onCheckedChange={setAutoStart}
          aria-label="Start remote access when Unsloth starts"
        />
      </SettingsRow>
    </SettingsSection>
  );
}
