// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { usePlatformStore } from "@/config/env";
import {
  loadPublicAccess,
  startPublicAccess,
  stopPublicAccess,
  updatePublicAccessAutoStart,
} from "@/features/settings/api/public-access";
import {
  type PublicAccessStatus,
  publicAccessAutoStartReadOnly,
  publicAccessBlockMessage,
  publicAccessPollDelay,
  publicAccessStopDisconnectsOrigin,
} from "@/features/settings/api/public-access-state";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

type PublicAccessOperation = "start" | "stop" | "auto";

const STATE_LABEL: Record<PublicAccessStatus["state"], string> = {
  off: "Off",
  starting: "Starting",
  online: "Online",
  stopping: "Stopping",
  error: "Error",
};

const OWNER_LABEL: Record<
  Exclude<PublicAccessStatus["managedBy"], null>,
  string
> = {
  launch: "Launch managed",
  settings: "Settings managed",
  colab: "Colab managed",
};

function stateDotClass(state?: PublicAccessStatus["state"]): string {
  if (state === "online") {
    return "bg-emerald-500";
  }
  if (state === "starting" || state === "stopping") {
    return "animate-pulse bg-blue-500";
  }
  return state === "error" ? "bg-red-500" : "bg-muted-foreground";
}

function AccessStatus({ status }: { status: PublicAccessStatus | null }) {
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

function CopyPublicUrlButton({ url }: { url: string }) {
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

export function PublicAccessSection() {
  const [status, setStatus] = useState<PublicAccessStatus | null>(null);
  const [busy, setBusy] = useState<PublicAccessOperation | null>(null);
  const [pollRevision, setPollRevision] = useState(0);
  const [pollEnabled, setPollEnabled] = useState(true);
  const mutationEpoch = useRef(0);
  const pollSuppressed = useRef(false);
  const selfStopDisconnectExpected = useRef(false);

  const applyStatus = useCallback((next: PublicAccessStatus) => {
    setStatus(next);
    usePlatformStore.setState({ cloudflareUrl: next.url });
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: pollRevision intentionally restarts polling after a mutation
  useEffect(() => {
    if (!pollEnabled) {
      return;
    }
    let stopped = false;
    let timer: number | null = null;
    const schedule = (next: PublicAccessStatus | null) => {
      if (!stopped && !pollSuppressed.current) {
        timer = window.setTimeout(poll, publicAccessPollDelay(next));
      }
    };
    const poll = () => {
      if (pollSuppressed.current) {
        return;
      }
      const epoch = mutationEpoch.current;
      loadPublicAccess()
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
    operation: PublicAccessOperation,
    request: () => Promise<PublicAccessStatus>,
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

  const start = () => perform("start", startPublicAccess);
  const stop = () =>
    perform(
      "stop",
      stopPublicAccess,
      publicAccessStopDisconnectsOrigin(
        status?.url ?? null,
        typeof window === "undefined" ? "" : window.location.origin,
      ),
    );
  const setAutoStart = (enabled: boolean) =>
    perform("auto", () => updatePublicAccessAutoStart(enabled));

  const blockMessage = publicAccessBlockMessage(status?.blockReason ?? null);
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
      title="Public access"
      description="Make Unsloth and its APIs available through a temporary Cloudflare Quick Tunnel HTTPS URL."
    >
      <SettingsRow
        label="Status"
        labelAccessory={<AccessStatus status={status} />}
        description={blockMessage}
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

      {status?.url ? (
        <SettingsRow
          label="Public URL"
          description={
            <code className="block break-all whitespace-normal font-mono">
              {status.url}
            </code>
          }
        >
          <CopyPublicUrlButton url={status.url} />
        </SettingsRow>
      ) : null}

      <SettingsRow
        label="Start public access when Unsloth starts"
        description="Unsloth will create a new public URL each time it starts. Stopping public access now won’t turn this setting off."
      >
        <Switch
          checked={status?.autoStart ?? false}
          disabled={busy !== null || publicAccessAutoStartReadOnly(status)}
          onCheckedChange={setAutoStart}
          aria-label="Start public access when Unsloth starts"
        />
      </SettingsRow>
    </SettingsSection>
  );
}
