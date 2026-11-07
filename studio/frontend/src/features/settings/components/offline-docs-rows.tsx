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

/** A local copy of the Unsloth docs, so RAG still knows them offline once the folder is linked to a knowledge base. */
export function OfflineDocsRows(): ReactElement {
  const t = useT();
  const [settings, setSettings] = useState<OfflineDocsSettings | null>(null);
  const [draftPath, setDraftPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void loadOfflineDocsSettings()
      .then((next) => {
        if (!active) return;
        setSettings(next);
        setDraftPath(next.path ?? "");
      })
      .catch((reason) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      active = false;
    };
  }, []);

  const download = async () => {
    setDownloading(true);
    setError(null);
    try {
      setSettings(await refreshOfflineDocs());
      toast.success(t("settings.general.rag.offlineDocs.downloaded"));
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
      toast.error(t("settings.general.rag.offlineDocs.downloadError"), {
        description: message,
      });
      // The failure is recorded server-side; re-read so the status line matches it.
      void loadOfflineDocsSettings()
        .then(setSettings)
        .catch(() => {});
    } finally {
      setDownloading(false);
    }
  };

  const save = async (update: { path?: string | null; autoUpdate?: boolean }) => {
    setSaving(true);
    setError(null);
    try {
      const next = await updateOfflineDocsSettings(update);
      setSettings(next);
      setDraftPath(next.path ?? "");
      setBrowserOpen(false);
      // A newly chosen folder, or auto-update switched on, with no copy there yet gets one now.
      if (next.path && !next.file && (update.path || update.autoUpdate)) {
        void download();
      } else {
        toast.success(t("settings.general.rag.offlineDocs.saved"));
      }
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
      toast.error(t("settings.general.rag.offlineDocs.saveError"), {
        description: message,
      });
    } finally {
      setSaving(false);
    }
  };

  const busy = saving || downloading || Boolean(settings?.refreshing);
  const pathDirty = Boolean(settings && draftPath.trim() !== (settings.path ?? ""));
  const updatedOn = settings?.updatedAt
    ? new Date(settings.updatedAt).toLocaleDateString()
    : null;
  const status = !settings
    ? null
    : downloading || settings.refreshing
      ? t("settings.general.rag.offlineDocs.downloading")
      : settings.file && updatedOn
        ? t("settings.general.rag.offlineDocs.lastUpdated", {
            date: updatedOn,
            release: settings.release ?? settings.currentRelease,
          })
        : settings.path
          ? t("settings.general.rag.offlineDocs.notDownloaded")
          : null;
  const failure = error ?? settings?.error ?? null;

  return (
    <>
      <SettingsRow
        label={t("settings.general.rag.offlineDocs.folderLabel")}
        description={t("settings.general.rag.offlineDocs.folderDescription")}
        hint={t("settings.general.rag.offlineDocs.folderHint")}
        className="max-[840px]:flex-col max-[840px]:items-stretch max-[840px]:gap-2"
        below={
          status || failure ? (
            <div className="flex w-[calc(392px*var(--ui-space-scale,1))] min-w-0 flex-col gap-0.5 pl-3.5 pr-1 text-xs max-[840px]:w-full">
              {status ? (
                <span
                  title={settings?.file ?? status}
                  className="min-w-0 truncate text-muted-foreground"
                  data-testid="offline-docs-status"
                >
                  {status}
                </span>
              ) : null}
              {failure ? (
                <span className="text-destructive" data-testid="offline-docs-error">
                  {failure}
                </span>
              ) : null}
            </div>
          ) : null
        }
      >
        <div className="grid w-[calc(392px*var(--ui-space-scale,1))] min-w-0 grid-cols-[minmax(0,1fr)_auto_auto] gap-x-2 max-[840px]:w-full">
          <Input
            aria-label={t("settings.general.rag.offlineDocs.folderLabel")}
            value={draftPath}
            placeholder={
              settings
                ? t("settings.general.rag.offlineDocs.noFolder")
                : t("common.loading")
            }
            disabled={!settings}
            onChange={(event) => setDraftPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && pathDirty && !busy) {
                void save({ path: draftPath.trim() || null });
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
            onClick={() => void save({ path: draftPath.trim() || null })}
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
            {t("settings.general.rag.offlineDocs.choose")}
          </Button>
        </div>
      </SettingsRow>
      <SettingsRow
        label={t("settings.general.rag.offlineDocs.autoUpdateLabel")}
        description={t("settings.general.rag.offlineDocs.autoUpdateDescription")}
      >
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            disabled={!settings?.path || busy}
            onClick={() => void download()}
            data-testid="offline-docs-download"
          >
            {downloading ? <Spinner /> : null}
            {t("settings.general.rag.offlineDocs.downloadNow")}
          </Button>
          <Switch
            aria-label={t("settings.general.rag.offlineDocs.autoUpdateLabel")}
            checked={settings?.autoUpdate ?? false}
            disabled={!settings?.path || busy}
            onCheckedChange={(autoUpdate) => void save({ autoUpdate })}
            data-testid="offline-docs-auto-update"
          />
        </div>
      </SettingsRow>

      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(path) => void save({ path })}
        initialPath={settings?.path ?? undefined}
        title={t("settings.general.rag.offlineDocs.chooseTitle")}
        description={t("settings.general.rag.offlineDocs.folderDescription")}
        confirmLabel={t("settings.general.rag.offlineDocs.chooseAction")}
        showModelHints={false}
      />
    </>
  );
}
