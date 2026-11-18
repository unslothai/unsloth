// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { FolderBrowser } from "@/features/model-picker";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type ReactElement, useEffect, useState } from "react";
import {
  type OfflineDocsSettings,
  loadOfflineDocsSettings,
  refreshOfflineDocs,
  updateOfflineDocsSettings,
} from "../api/offline-docs";
import { SettingsRow } from "./settings-row";

const KEY = "settings.general.rag.offlineDocs";
const message = (reason: unknown) =>
  reason instanceof Error ? reason.message : String(reason);

export function OfflineDocsRows(): ReactElement {
  const t = useT();
  const [settings, setSettings] = useState<OfflineDocsSettings | null>(null);
  const [draftPath, setDraftPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apply = (next: OfflineDocsSettings) => {
    setSettings(next);
    setDraftPath(next.path ?? "");
  };
  const fail = (reason: unknown) => {
    setError(message(reason));
    toast.error(t("common.error"), { description: message(reason) });
  };

  useEffect(() => {
    let active = true;
    loadOfflineDocsSettings()
      .then((next) => active && apply(next))
      .catch((reason) => active && setError(message(reason)));
    return () => {
      active = false;
    };
  }, []);

  // A startup refresh may be running when the rows open; poll until it ends.
  const serverRefreshing = Boolean(settings?.refreshing) && !downloading;
  useEffect(() => {
    if (!serverRefreshing) {
      return;
    }
    const timer = window.setInterval(() => {
      loadOfflineDocsSettings()
        .then(setSettings)
        .catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [serverRefreshing]);

  const download = async () => {
    setDownloading(true);
    setError(null);
    try {
      setSettings(await refreshOfflineDocs());
    } catch (reason) {
      fail(reason);
      // The failure is recorded server-side; re-read so the status line shows it.
      loadOfflineDocsSettings()
        .then(setSettings)
        .catch(() => undefined);
    } finally {
      setDownloading(false);
    }
  };

  const save = async (update: {
    path?: string | null;
    autoUpdate?: boolean;
  }) => {
    setSaving(true);
    setError(null);
    try {
      const next = await updateOfflineDocsSettings(update);
      apply(next);
      setBrowserOpen(false);
      if (next.path && !next.file && (update.path || update.autoUpdate)) {
        await download();
      }
    } catch (reason) {
      fail(reason);
    } finally {
      setSaving(false);
    }
  };

  const busy = saving || downloading || Boolean(settings?.refreshing);
  const pathDirty = Boolean(
    settings && draftPath.trim() !== (settings.path ?? ""),
  );
  const savePath = () => save({ path: draftPath.trim() || null });
  let status: string | null = null;
  if (downloading || settings?.refreshing) {
    status = t(`${KEY}.downloading`);
  } else if (settings?.file && settings.updatedAt) {
    status = t(`${KEY}.lastUpdated`, {
      date: new Date(settings.updatedAt).toLocaleDateString(),
      release: settings.release ?? settings.currentRelease,
    });
  } else if (settings?.path) {
    status = t(`${KEY}.notDownloaded`);
  }
  const failure = error ?? settings?.error ?? null;

  return (
    <>
      <SettingsRow
        label={t(`${KEY}.folderLabel`)}
        description={t(`${KEY}.folderDescription`)}
        hint={t(`${KEY}.folderHint`)}
        className="max-[840px]:flex-col max-[840px]:items-stretch max-[840px]:gap-2"
        below={
          status || failure ? (
            <div className="flex w-[calc(392px*var(--ui-space-scale,1))] min-w-0 flex-col gap-0.5 pl-3.5 pr-1 text-xs max-[840px]:w-full">
              {status && (
                <span
                  title={settings?.file ?? status}
                  className="min-w-0 truncate text-muted-foreground"
                  data-testid="offline-docs-status"
                >
                  {status}
                </span>
              )}
              {failure && (
                <span
                  className="text-destructive"
                  data-testid="offline-docs-error"
                >
                  {failure}
                </span>
              )}
            </div>
          ) : null
        }
      >
        <div className="grid w-[calc(392px*var(--ui-space-scale,1))] min-w-0 grid-cols-[minmax(0,1fr)_auto_auto] gap-x-2 max-[840px]:w-full">
          <Input
            aria-label={t(`${KEY}.folderLabel`)}
            value={draftPath}
            placeholder={settings ? "" : t("common.loading")}
            disabled={!settings}
            onChange={(event) => setDraftPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && pathDirty && !busy) {
                savePath();
              }
            }}
            title={settings?.path ?? undefined}
            className="h-8 min-w-0 font-mono text-xs"
            data-testid="offline-docs-path-input"
          />
          <Button
            variant="outline"
            size="sm"
            disabled={!pathDirty || busy}
            onClick={savePath}
            data-testid="offline-docs-path-save"
          >
            {saving ? t("common.saving") : t("common.save")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={!settings || busy}
            onClick={() => setBrowserOpen(true)}
            data-testid="offline-docs-path-change"
          >
            {t(`${KEY}.choose`)}
          </Button>
        </div>
      </SettingsRow>
      <SettingsRow
        label={t(`${KEY}.autoUpdateLabel`)}
        description={t(`${KEY}.autoUpdateDescription`)}
      >
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            disabled={!settings?.path || busy}
            onClick={download}
            data-testid="offline-docs-download"
          >
            {downloading && <Spinner />}
            {t(`${KEY}.downloadNow`)}
          </Button>
          <Switch
            aria-label={t(`${KEY}.autoUpdateLabel`)}
            checked={settings?.autoUpdate ?? false}
            disabled={!settings?.path || busy}
            onCheckedChange={(autoUpdate) => save({ autoUpdate })}
            data-testid="offline-docs-auto-update"
          />
        </div>
      </SettingsRow>
      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(path) => save({ path })}
        initialPath={settings?.path ?? undefined}
        title={t(`${KEY}.folderLabel`)}
        description={t(`${KEY}.folderDescription`)}
        confirmLabel={t(`${KEY}.choose`)}
        showModelHints={false}
      />
    </>
  );
}
