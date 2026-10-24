// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ClaudeDesktopStatus,
  type DesktopAppModel,
  fetchClaudeDesktopModels,
  setClaudeDesktopMappings,
} from "../api/claude-desktop";
import { DesktopAppModelPicker } from "./desktop-app-model-picker";
import {
  CLAUDE_ROUTES,
  type Mappings,
  cleanMappings,
  mappingsEqual,
  pinnedModels,
  sharedMappingModel,
  updateAllMappings,
  updateMapping,
} from "./desktop-app-models";

type Props = {
  status: ClaudeDesktopStatus;
  disabled: boolean;
  onStatus: (status: ClaudeDesktopStatus, pendingRestart?: boolean) => void;
};

const ROW =
  "relative grid min-h-10 grid-cols-[5.5rem_1.5rem_minmax(0,1fr)] items-center gap-2 py-0.5 max-sm:grid-cols-1";

/** Which Studio model each of Claude's five model options uses, 1:1 with Ollama's Claude settings:
 *  an "All" picker that sets every option, then one picker per option. Disconnected, a pick is
 *  saved at once; connected, the picks are applied with Start / Restart Claude, because Claude
 *  reads its model list only at startup (requests follow a new pick right away). */
export function ClaudeDesktopModels({ status, disabled, onStatus }: Props) {
  const t = useT();
  const routes = status.routes?.length ? status.routes : CLAUDE_ROUTES;
  const saved = cleanMappings(status.mappings ?? {}, routes);
  const savedKey = JSON.stringify(saved);
  const [syncedKey, setSyncedKey] = useState(savedKey);
  const [draft, setDraft] = useState<Mappings>(saved);
  const [models, setModels] = useState<DesktopAppModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [askRestart, setAskRestart] = useState(false);
  const mounted = useRef(true);

  // Saved choices changed underneath (another window, or after Apply): follow them unless the
  // user is mid-edit.
  if (savedKey !== syncedKey) {
    const before = JSON.parse(syncedKey) as Mappings;
    setSyncedKey(savedKey);
    if (mappingsEqual(draft, before, routes)) {
      setDraft(saved);
    }
  }

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const next = await fetchClaudeDesktopModels();
      if (mounted.current) setModels(next);
    } catch (err) {
      if (mounted.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    const first = window.setTimeout(() => void loadModels(), 0);
    return () => {
      mounted.current = false;
      window.clearTimeout(first);
    };
  }, [loadModels]);

  const save = useCallback(
    async (mappings: Mappings, restart?: boolean) => {
      setBusy(true);
      setError(null);
      try {
        const result = await setClaudeDesktopMappings(
          cleanMappings(mappings, routes),
          restart,
        );
        if (!mounted.current) return;
        if (result.status === "needs_restart") {
          setAskRestart(true);
          return;
        }
        onStatus(result.claude, result.status === "pending_restart");
      } catch (err) {
        if (mounted.current) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (mounted.current) setBusy(false);
      }
    },
    [onStatus, routes],
  );

  const change = (next: Mappings) => {
    setDraft(next);
    if (!status.connected && Object.keys(cleanMappings(next, routes)).length) {
      void save(next);
    }
  };

  const assigned = Object.keys(cleanMappings(draft, routes)).length;
  const changed = !mappingsEqual(draft, saved, routes);
  const shared = sharedMappingModel(draft, routes);
  const pinned = pinnedModels(cleanMappings(draft, routes));
  const apply = () => {
    if (status.running && changed) {
      setAskRestart(true);
    } else {
      void save(draft, true);
    }
  };
  const pickerProps = {
    models,
    loadedModel: status.model,
    loading,
    disabled: disabled || busy,
    onOpen: () => void loadModels(),
  };

  return (
    <div
      data-testid="claude-desktop-models"
      className="mt-1 flex min-w-0 flex-col gap-1 border-t border-border/60 pt-2"
    >
      <div className="flex min-w-0 items-start justify-between gap-3">
        <p className="text-ui-11 leading-relaxed text-muted-foreground">
          {t("settings.agents.desktopApps.models.claudeDescription")}
        </p>
        {status.connected ? (
          <Button
            size="sm"
            variant="outline"
            data-testid="claude-desktop-apply"
            disabled={
              disabled || busy || assigned === 0 || (status.running && !changed)
            }
            onClick={apply}
          >
            {busy ? <Spinner className="size-3.5" /> : null}
            {busy
              ? status.running
                ? t("settings.agents.desktopApps.models.restarting")
                : t("settings.agents.desktopApps.models.starting")
              : status.running
                ? t("settings.agents.desktopApps.models.restartClaude")
                : t("settings.agents.desktopApps.models.startClaude")}
          </Button>
        ) : null}
      </div>
      <div className="w-full max-w-xl">
        <div className="mb-1 border-b border-border/60 pb-1.5">
          <div className={ROW}>
            <span className="text-ui-12 font-medium text-foreground">
              {t("settings.agents.desktopApps.models.all")}
            </span>
            <HugeiconsIcon
              icon={ArrowRight02Icon}
              aria-hidden={true}
              className="size-3.5 text-muted-foreground/60 max-sm:hidden"
            />
            <div className="w-full min-w-0 sm:w-2/3">
              <DesktopAppModelPicker
                id="claude-route-all"
                routeName={t("settings.agents.desktopApps.models.allRouteName")}
                value={shared ?? ""}
                placeholder={
                  assigned > 0
                    ? t("settings.agents.desktopApps.models.mixed")
                    : t("settings.agents.desktopApps.models.select")
                }
                onChange={(model) => change(updateAllMappings(model, routes))}
                {...pickerProps}
              />
            </div>
          </div>
          <p className="text-ui-11 leading-5 text-muted-foreground">
            {t("settings.agents.desktopApps.models.allHint")}
          </p>
        </div>
        {routes.map((route) => (
          <div key={route.id} className={ROW}>
            <span className="text-ui-12 font-medium text-foreground">
              {route.name}
            </span>
            <HugeiconsIcon
              icon={ArrowRight02Icon}
              aria-hidden={true}
              className="size-3.5 text-muted-foreground/60 max-sm:hidden"
            />
            <div className="w-full min-w-0 sm:w-2/3">
              <DesktopAppModelPicker
                id={`claude-route-${route.id}`}
                routeName={route.name}
                value={draft[route.id] ?? ""}
                onChange={(model) =>
                  change(updateMapping(draft, route.id, model))
                }
                {...pickerProps}
              />
            </div>
          </div>
        ))}
      </div>
      {status.connected && changed && status.running ? (
        <p className="text-ui-11 leading-relaxed text-muted-foreground">
          {t("settings.agents.desktopApps.models.restartGuidance")}
        </p>
      ) : null}
      {pinned.length > 1 ? (
        <p className="text-ui-11 leading-relaxed text-muted-foreground">
          {t("settings.agents.desktopApps.models.multiModelHint")}
        </p>
      ) : null}
      {error ? (
        <p role="alert" className="text-ui-11 leading-relaxed text-destructive">
          {error}
        </p>
      ) : null}

      <AlertDialog open={askRestart} onOpenChange={setAskRestart}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.agents.desktopApps.models.restartConfirm")}
            </AlertDialogTitle>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={(event) => {
                event.preventDefault();
                setAskRestart(false);
                void save(draft, true);
              }}
            >
              {t("settings.agents.desktopApps.models.restartClaude")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
