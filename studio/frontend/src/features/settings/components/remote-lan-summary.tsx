// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ArrowRight01Icon, HomeWifiIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import {
  type LanAccessStatus,
  loadLanAccess,
  publishLanUrls,
} from "../api/lan-access";
import {
  type RemoteAccessStatus,
  loadRemoteAccess,
} from "../api/remote-access";
import { remoteAccessPollDelay } from "../api/remote-access-state";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

type AccessState = "off" | "starting" | "online" | "stopping" | "error";

const STATE_LABEL: Record<AccessState, string> = {
  off: "Off",
  starting: "Starting",
  online: "Online",
  stopping: "Stopping",
  error: "Error",
};

function stateDotClass(state?: AccessState): string {
  if (state === "online") return "bg-emerald-500";
  if (state === "starting" || state === "stopping") {
    return "animate-pulse bg-blue-500";
  }
  return state === "error" ? "bg-red-500" : "bg-muted-foreground";
}

function AccessLine({
  label,
  state,
  url,
}: {
  label: string;
  state: AccessState | "unavailable" | null;
  url: string | null;
}) {
  const known = state !== null && state !== "unavailable" ? state : undefined;
  return (
    <div className="flex min-w-0 items-center gap-3 px-4 py-2.5">
      <span className="w-28 shrink-0 text-sm text-foreground">{label}</span>
      <span className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
        <span
          className={cn("size-2 rounded-full", stateDotClass(known))}
          aria-hidden={true}
        />
        {known
          ? STATE_LABEL[known]
          : state === "unavailable"
            ? "Unavailable"
            : "Checking…"}
      </span>
      {url ? (
        <span className="min-w-0 truncate font-mono text-xs text-muted-foreground">
          {url}
        </span>
      ) : null}
    </div>
  );
}

/**
 * Reads each once instead of mounting both panels and their five-second polls here.
 */
export function RemoteLanSummary() {
  const t = useT();
  const [remote, setRemote] = useState<RemoteAccessStatus | null>(null);
  const [lan, setLan] = useState<LanAccessStatus | null>(null);
  const [remoteFailed, setRemoteFailed] = useState(false);
  const [lanFailed, setLanFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let settling = false;
    // Re-read only mid-transition: a Start left on another tab would otherwise read "Starting" forever.
    // A failed read while settling retries too, or the last transitional state would stick.
    const readRemote = () => {
      loadRemoteAccess()
        .then((next) => {
          if (cancelled) return;
          setRemote(next);
          usePlatformStore.setState({ cloudflareUrl: next.url });
          settling = next.state === "starting" || next.state === "stopping";
          if (settling) {
            timer = window.setTimeout(readRemote, remoteAccessPollDelay(next));
          }
        })
        .catch(() => {
          if (cancelled) return;
          setRemoteFailed(true);
          if (settling) {
            timer = window.setTimeout(readRemote, remoteAccessPollDelay(null));
          }
        });
    };
    readRemote();
    loadLanAccess()
      .then((next) => {
        if (cancelled) return;
        setLan(next);
        publishLanUrls(next);
      })
      .catch(() => {
        if (!cancelled) setLanFailed(true);
      });
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  return (
    <section
      data-settings-label={t("settings.remoteLan.title")}
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <HugeiconsIcon
              icon={HomeWifiIcon}
              className="size-4 text-foreground"
            />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <h2 className="settings-heading text-base font-semibold font-heading">
              {t("settings.remoteLan.title")}
            </h2>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {t("settings.remoteLan.description")}
            </p>
          </div>
        </div>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="shrink-0 gap-1.5"
          onClick={() =>
            useSettingsDialogStore.getState().setActiveTab("remote-lan")
          }
        >
          Manage
          <HugeiconsIcon icon={ArrowRight01Icon} className="size-3.5" />
        </Button>
      </div>
      <div className="flex flex-col py-1">
        <AccessLine
          label="Remote access"
          state={remote?.state ?? (remoteFailed ? "unavailable" : null)}
          url={remote?.state === "online" ? remote.url : null}
        />
        <AccessLine
          label="LAN access"
          state={lan?.state ?? (lanFailed ? "unavailable" : null)}
          url={lan?.state === "online" ? (lan.urls[0] ?? null) : null}
        />
      </div>
    </section>
  );
}
