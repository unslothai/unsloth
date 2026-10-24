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
import type { DesktopAppModel } from "../api/claude-desktop";
import {
  type CodexDesktopStatus,
  CodexDesktopUnavailable,
  connectCodexDesktop,
  disconnectCodexDesktop,
  fetchCodexDesktopModels,
  fetchCodexDesktopStatus,
  setCodexDesktopModels,
} from "../api/codex-desktop";
import { AgentIcon } from "./agent-command-block";
import { claudeDesktopView } from "./claude-desktop-state";
import { codexPicks, codexTarget, sameModels } from "./codex-desktop-state";
import { DesktopAppModelListPicker } from "./desktop-app-model-list-picker";

const POLL_MS = 5000;

type Busy = "connect" | "disconnect" | "model" | null;
// What the open restart dialog will do when confirmed.
type Ask = "connect" | "disconnect" | "model" | null;

/** Settings → Agents → Desktop apps → Codex Desktop: the same switch as `unsloth start codex
 *  --app`, held by the backend. A list of Studio models, like Ollama's ChatGPT settings: each one
 *  is in Codex's own model picker (the first is the default) and Studio loads it when Codex asks.
 *  Codex reads the list at startup, so a change while connected restarts it. */
export function CodexDesktopCard({
  onUnavailable,
}: {
  onUnavailable: (app: string) => void;
}) {
  const t = useT();
  const [status, setStatus] = useState<CodexDesktopStatus | null>(null);
  const [hidden, setHidden] = useState(false);
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [ask, setAsk] = useState<Ask>(null);
  const [draft, setDraft] = useState<string[] | null>(null);
  const [pendingRestart, setPendingRestart] = useState(false);
  const [models, setModels] = useState<DesktopAppModel[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const mounted = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchCodexDesktopStatus();
      if (mounted.current) {
        setStatus(next);
        if (!next.connected) setPendingRestart(false);
      }
    } catch (err) {
      if (mounted.current && err instanceof CodexDesktopUnavailable) {
        setHidden(true);
        onUnavailable("codex");
      }
    }
  }, [onUnavailable]);

  const loadModels = useCallback(async () => {
    setLoadingModels(true);
    try {
      const next = await fetchCodexDesktopModels();
      if (mounted.current) setModels(next);
    } catch {
      // The picker still offers "Studio's loaded model".
    } finally {
      if (mounted.current) setLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    const first = window.setTimeout(() => {
      void refresh();
      void loadModels();
    }, 0);
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => {
      mounted.current = false;
      window.clearTimeout(first);
      window.clearInterval(timer);
    };
  }, [refresh, loadModels]);

  const run = useCallback(
    async (kind: Busy & string, action: () => Promise<void>) => {
      setBusy(kind);
      setError(null);
      try {
        await action();
      } catch (err) {
        if (mounted.current) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (mounted.current) setBusy(null);
      }
    },
    [],
  );

  const connect = (restart?: boolean) =>
    run("connect", async () => {
      setNotes([]);
      const result = await connectCodexDesktop(restart);
      if (!mounted.current) return;
      setStatus(result.codex);
      if (result.status === "needs_restart") setAsk("connect");
      else setPendingRestart(result.status === "pending_restart");
    });

  const disconnect = (restart?: boolean) =>
    run("disconnect", async () => {
      const result = await disconnectCodexDesktop(restart);
      if (!mounted.current) return;
      setStatus(result.codex);
      if (result.status === "needs_restart") {
        setAsk("disconnect");
        return;
      }
      setNotes(result.notes);
      setPendingRestart(false);
      setDraft(null);
    });

  const pickModels = (picks: string[], restart?: boolean) =>
    run("model", async () => {
      const result = await setCodexDesktopModels(picks, restart);
      if (!mounted.current) return;
      setStatus(result.codex);
      if (result.status === "needs_restart") {
        setAsk("model");
        return;
      }
      setDraft(null);
      setPendingRestart(result.status === "pending_restart");
    });

  if (hidden) {
    return null;
  }

  const view = claudeDesktopView(
    status && {
      ...status,
      model: status.connected ? status.model : codexTarget(status),
    },
    pendingRestart,
  );
  let detail: string;
  switch (view.kind) {
    case "loading":
      detail = t("settings.agents.desktopApps.codex.checking");
      break;
    case "remote":
      detail = t("settings.agents.desktopApps.codex.remote");
      break;
    case "unsupported":
      detail = t("settings.agents.desktopApps.codex.notMac");
      break;
    case "not_installed":
      detail = t("settings.agents.desktopApps.codex.notInstalled");
      break;
    case "held":
      detail = t("settings.agents.desktopApps.codex.heldByCli", {
        pid: String(view.pid ?? "?"),
      });
      break;
    case "connected":
      detail = view.pendingRestart
        ? t("settings.agents.desktopApps.codex.pendingRestart")
        : (status?.models.length ?? 0) > 1
          ? t("settings.agents.desktopApps.codex.connectedMany", {
              count: String(status?.models.length ?? 0),
              model: view.model ?? "?",
            })
          : t("settings.agents.desktopApps.codex.connected", {
              model: view.model ?? "?",
            });
      break;
    case "idle":
      detail = view.model
        ? t("settings.agents.desktopApps.codex.idle", { model: view.model })
        : t("settings.agents.desktopApps.codex.idleNoModel");
      break;
  }

  const connected = view.kind === "connected";
  const canConnect = view.kind === "idle" && view.model !== null;
  const current = status ? codexPicks(status) : [];
  const value = draft ?? current;
  const changed = connected && draft !== null && !sameModels(draft, current);
  const running = status?.running ?? false;
  const applyModel = () => {
    if (draft === null) {
      void pickModels(current, true);
    } else if (running) {
      setAsk("model");
    } else {
      void pickModels(draft, true);
    }
  };

  return (
    <>
      <div
        data-testid="codex-desktop-card"
        data-state={view.kind}
        className="flex min-w-0 flex-col gap-2 rounded-2xl border border-border/60 px-3.5 py-3"
      >
        <div className="flex min-w-0 items-center gap-3">
          <AgentIcon logo="openai" />
          <div className="flex min-w-0 flex-1 flex-col gap-0.5">
            <div className="flex min-w-0 items-center gap-2">
              <span className="truncate text-sm font-medium text-foreground">
                {t("settings.agents.desktopApps.codex.name")}
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
                {t("settings.agents.desktopApps.codex.download")}
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
          <div
            data-testid="codex-desktop-models"
            className="mt-1 flex min-w-0 flex-col gap-1 border-t border-border/60 pt-2"
          >
            <div className="grid min-h-10 grid-cols-[5.5rem_minmax(0,1fr)_auto] items-start gap-2 max-sm:grid-cols-1">
              <span className="pt-1.5 text-ui-12 font-medium text-foreground">
                {t("settings.agents.desktopApps.codex.modelLabel")}
              </span>
              <div className="w-full min-w-0 max-w-md">
                <DesktopAppModelListPicker
                  id="codex-models"
                  value={value}
                  models={models}
                  max={status.max_models}
                  loadedModel={status.loaded_model}
                  loading={loadingModels}
                  disabled={busy !== null}
                  onOpen={() => void loadModels()}
                  onChange={(picks) => {
                    if (connected) setDraft(picks);
                    else void pickModels(picks);
                  }}
                />
              </div>
              {connected ? (
                <Button
                  size="sm"
                  variant="outline"
                  data-testid="codex-desktop-apply"
                  disabled={busy !== null || (running && !changed)}
                  onClick={applyModel}
                >
                  {busy === "model" ? <Spinner className="size-3.5" /> : null}
                  {busy === "model"
                    ? running
                      ? t("settings.agents.desktopApps.models.restarting")
                      : t("settings.agents.desktopApps.models.starting")
                    : running
                      ? t("settings.agents.desktopApps.codex.restartNow")
                      : t("settings.agents.desktopApps.codex.start")}
                </Button>
              ) : null}
            </div>
            <p className="text-ui-11 leading-relaxed text-muted-foreground">
              {changed && running
                ? t("settings.agents.desktopApps.codex.restartGuidance")
                : t("settings.agents.desktopApps.codex.modelHint")}
            </p>
          </div>
        ) : null}
        {connected ? (
          <p className="text-ui-11 leading-relaxed text-muted-foreground">
            {t("settings.agents.desktopApps.codex.lifecycle")}
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

      <AlertDialog
        open={ask !== null}
        onOpenChange={(open) => {
          if (!open) setAsk(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.agents.desktopApps.codex.restartTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {ask === "disconnect"
                ? t("settings.agents.desktopApps.codex.disconnectBody")
                : ask === "model"
                  ? t("settings.agents.desktopApps.codex.modelBody")
                  : t("settings.agents.desktopApps.codex.restartBody")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            {ask !== "model" ? (
              <Button
                variant="outline"
                onClick={() => {
                  const kind = ask;
                  setAsk(null);
                  if (kind === "disconnect") void disconnect(false);
                  else void connect(false);
                }}
              >
                {ask === "disconnect"
                  ? t("settings.agents.desktopApps.codex.leaveOpen")
                  : t("settings.agents.desktopApps.restartLater")}
              </Button>
            ) : null}
            <AlertDialogAction
              onClick={(event) => {
                event.preventDefault();
                const kind = ask;
                setAsk(null);
                if (kind === "disconnect") void disconnect(true);
                else if (kind === "model") void pickModels(value, true);
                else void connect(true);
              }}
            >
              {t("settings.agents.desktopApps.codex.restartNow")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
