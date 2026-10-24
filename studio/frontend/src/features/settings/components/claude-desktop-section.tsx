// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ClaudeDesktopStatus,
  ClaudeDesktopUnavailable,
  connectClaudeDesktop,
  disconnectClaudeDesktop,
  fetchClaudeDesktopStatus,
} from "../api/claude-desktop";
import { AgentIcon } from "./agent-command-block";
import { ClaudeDesktopModels } from "./claude-desktop-models";
import { claudeDesktopView } from "./claude-desktop-state";
import { CodexDesktopCard } from "./codex-desktop-card";
import { pinnedModels } from "./desktop-app-models";
import { DesktopAppRunSettings } from "./desktop-app-run-settings";
import { SettingsSection } from "./settings-section";

const POLL_MS = 5000;

type Busy = "connect" | "disconnect" | null;

/** Settings → Agents → Desktop apps: switch Claude Desktop and the Codex app on this Mac to
 *  Studio and back. The backend owns each switch, so it lasts exactly as long as Studio runs; no
 *  terminal involved. A card for an app this Studio can't offer (an older backend, a non-owner)
 *  hides itself, and the section goes with the last one. */
export function ClaudeDesktopSection() {
  const t = useT();
  const [hiddenApps, setHiddenApps] = useState<string[]>([]);
  const hide = useCallback(
    (app: string) =>
      setHiddenApps((apps) => (apps.includes(app) ? apps : [...apps, app])),
    [],
  );
  return (
    <div hidden={hiddenApps.length >= 2}>
      <SettingsSection
        title={t("settings.agents.desktopApps.title")}
        description={t("settings.agents.desktopApps.description")}
      >
        <div className="mt-2 flex min-w-0 flex-col gap-2">
          <ClaudeDesktopCard onUnavailable={hide} />
          <CodexDesktopCard onUnavailable={hide} />
          <DesktopAppRunSettings preferred={[]} />
        </div>
      </SettingsSection>
    </div>
  );
}

function ClaudeDesktopCard({
  onUnavailable,
}: {
  onUnavailable: (app: string) => void;
}) {
  const t = useT();
  const [status, setStatus] = useState<ClaudeDesktopStatus | null>(null);
  const [hidden, setHidden] = useState(false);
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [askRestart, setAskRestart] = useState(false);
  const [pendingRestart, setPendingRestart] = useState(false);
  const mounted = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchClaudeDesktopStatus();
      if (mounted.current) {
        setStatus(next);
        if (!next.connected) {
          setPendingRestart(false);
        }
      }
    } catch (err) {
      if (mounted.current && err instanceof ClaudeDesktopUnavailable) {
        setHidden(true);
        onUnavailable("claude");
      }
    }
  }, [onUnavailable]);

  useEffect(() => {
    mounted.current = true;
    const first = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => {
      mounted.current = false;
      window.clearTimeout(first);
      window.clearInterval(timer);
    };
  }, [refresh]);

  const connect = useCallback(async (restart?: boolean) => {
    setBusy("connect");
    setError(null);
    setNotes([]);
    try {
      const result = await connectClaudeDesktop(restart);
      if (!mounted.current) return;
      setStatus(result.claude);
      if (result.status === "needs_restart") {
        setAskRestart(true);
      } else {
        setPendingRestart(result.status === "pending_restart");
      }
    } catch (err) {
      if (mounted.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mounted.current) setBusy(null);
    }
  }, []);

  const disconnect = useCallback(async () => {
    setBusy("disconnect");
    setError(null);
    try {
      const result = await disconnectClaudeDesktop();
      if (!mounted.current) return;
      setStatus(result.claude);
      setNotes(result.notes);
      setPendingRestart(false);
    } catch (err) {
      if (mounted.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mounted.current) setBusy(null);
    }
  }, []);

  if (hidden) {
    return null;
  }

  const view = claudeDesktopView(status, pendingRestart);
  let detail: string;
  switch (view.kind) {
    case "loading":
      detail = t("settings.agents.desktopApps.checking");
      break;
    case "remote":
      detail = t("settings.agents.desktopApps.remote");
      break;
    case "unsupported":
      detail = t("settings.agents.desktopApps.notMac");
      break;
    case "not_installed":
      detail = t("settings.agents.desktopApps.notInstalled");
      break;
    case "held":
      detail = t("settings.agents.desktopApps.heldByCli", {
        pid: String(view.pid ?? "?"),
      });
      break;
    case "connected":
      detail = view.pendingRestart
        ? t("settings.agents.desktopApps.pendingRestart")
        : status && pinnedModels(status.mappings ?? {}).length > 0
          ? t("settings.agents.desktopApps.connectedMapped")
          : view.model
            ? t("settings.agents.desktopApps.connected", { model: view.model })
            : t("settings.agents.desktopApps.connectedNoModel");
      break;
    case "idle":
      detail = view.model
        ? t("settings.agents.desktopApps.idle", { model: view.model })
        : t("settings.agents.desktopApps.idleNoModel");
      break;
  }

  const connected = view.kind === "connected";
  const canConnect = view.kind === "idle";

  return (
    <>
      <div
        data-testid="claude-desktop-card"
        data-state={view.kind}
        className="flex min-w-0 flex-col gap-2 rounded-2xl border border-border/60 px-3.5 py-3"
      >
        <div className="flex min-w-0 items-center gap-3">
          <AgentIcon logo="anthropic" />
          <div className="flex min-w-0 flex-1 flex-col gap-0.5">
            <div className="flex min-w-0 items-center gap-2">
              <span className="truncate text-sm font-medium text-foreground">
                {t("settings.agents.desktopApps.claudeName")}
              </span>
              <span
                className={cn(
                  "inline-flex shrink-0 items-center gap-1 rounded-full px-1.5 py-0.5 text-ui-10 font-medium",
                  connected
                    ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                    : "bg-muted text-muted-foreground",
                )}
              >
                <span
                  aria-hidden={true}
                  className={cn(
                    "size-1.5 rounded-full",
                    connected ? "bg-emerald-500" : "bg-muted-foreground/50",
                  )}
                />
                {connected
                  ? t("settings.agents.desktopApps.badgeConnected")
                  : t("settings.agents.desktopApps.badgeOff")}
              </span>
            </div>
            <p className="text-ui-11 leading-relaxed text-muted-foreground">
              {detail}
            </p>
          </div>
          {view.kind === "not_installed" ? (
            <Button asChild={true} size="sm" variant="outline">
              <a
                href={view.downloadUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                {t("settings.agents.desktopApps.download")}
                <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
              </a>
            </Button>
          ) : connected ? (
            <Button
              size="sm"
              variant="outline"
              disabled={busy !== null}
              onClick={() => void disconnect()}
            >
              {busy === "disconnect" ? <Spinner className="size-3.5" /> : null}
              {busy === "disconnect"
                ? t("settings.agents.desktopApps.disconnecting")
                : t("settings.agents.desktopApps.disconnect")}
            </Button>
          ) : (
            <Button
              size="sm"
              variant="dark"
              disabled={!canConnect || busy !== null}
              onClick={() => void connect()}
            >
              {busy === "connect" ? <Spinner className="size-3.5" /> : null}
              {busy === "connect"
                ? t("settings.agents.desktopApps.connecting")
                : t("settings.agents.desktopApps.connect")}
            </Button>
          )}
        </div>
        {status && (view.kind === "idle" || connected) ? (
          <ClaudeDesktopModels
            status={status}
            disabled={busy !== null}
            onStatus={(next, pending) => {
              setStatus(next);
              if (pending !== undefined) setPendingRestart(pending);
            }}
          />
        ) : null}
        {connected ? (
          <p className="text-ui-11 leading-relaxed text-muted-foreground">
            {t("settings.agents.desktopApps.lifecycle")}
          </p>
        ) : null}
        {error ? (
          <p
            role="alert"
            className="text-ui-11 leading-relaxed text-destructive"
          >
            {error}
          </p>
        ) : null}
        {notes.map((note) => (
          <p
            key={note}
            className="text-ui-11 leading-relaxed text-muted-foreground"
          >
            {note}
          </p>
        ))}
      </div>

      <AlertDialog open={askRestart} onOpenChange={setAskRestart}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.agents.desktopApps.restartTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("settings.agents.desktopApps.restartBody")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <Button
              variant="outline"
              onClick={() => {
                setAskRestart(false);
                void connect(false);
              }}
            >
              {t("settings.agents.desktopApps.restartLater")}
            </Button>
            <AlertDialogAction
              onClick={(event) => {
                event.preventDefault();
                setAskRestart(false);
                void connect(true);
              }}
            >
              {t("settings.agents.desktopApps.restartNow")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
