// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { FolderBrowser } from "@/features/model-picker";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, useRef, useState } from "react";
import {
  type ProjectsHomeSettings,
  loadProjectsHomeSettings,
  updateProjectsHomeSettings,
} from "../api/projects-home";
import { SettingsCardRow } from "./storage-overview";

export function ProjectsHomeSetting() {
  const t = useT();
  const [settings, setSettings] = useState<ProjectsHomeSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const mounted = useRef(false);
  const readVersion = useRef(0);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    const version = readVersion.current;
    const current = () => active && version === readVersion.current;
    void loadProjectsHomeSettings(
      t("settings.resources.storage.projectsFolder.loadError"),
    )
      .then((next) => {
        if (current()) {
          setSettings(next);
          setError(null);
        }
      })
      .catch((reason) => {
        if (current()) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      })
      .finally(() => {
        if (current()) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [refreshVersion, t]);

  const save = async (path: string | null) => {
    if (!settings?.editable || loading || saving) return;
    // A locale change can start a read while this row is open. Its older
    // snapshot must not replace the result of a folder change or reset.
    readVersion.current += 1;
    setSaving(true);
    setError(null);
    try {
      const next = await updateProjectsHomeSettings(
        path,
        t("settings.resources.storage.projectsFolder.saveError"),
      );
      if (!mounted.current) return;
      setSettings(next);
      setError(null);
      toast.success(t("settings.resources.storage.projectsFolder.saved"));
    } catch (reason) {
      if (!mounted.current) return;
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      readVersion.current += 1;
      if (mounted.current) setSaving(false);
    }
  };

  const disabled = !settings?.editable || loading || saving;

  return (
    <>
      <SettingsCardRow
        label={t("settings.resources.storage.projectsFolder.label")}
        description={
          <div className="flex min-w-0 flex-col gap-1">
            <span
              className="truncate font-mono text-ui-11"
              title={settings?.projectsHome}
              data-testid="projects-home-path"
            >
              {settings?.projectsHome ?? (loading ? t("common.loading") : "—")}
            </span>
            <span>
              {t("settings.resources.storage.projectsFolder.description")}
            </span>
            {settings?.source === "environment" ? (
              <span>
                {t("settings.resources.storage.environmentManaged", {
                  variable:
                    settings.environmentVariable ??
                    "UNSLOTH_STUDIO_PROJECTS_HOME",
                })}
              </span>
            ) : null}
            {error ? (
              <span
                role="alert"
                className="text-destructive"
                data-testid="projects-home-error"
              >
                {error}
              </span>
            ) : null}
          </div>
        }
      >
        {!settings && !loading ? (
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setLoading(true);
              setError(null);
              setRefreshVersion((value) => value + 1);
            }}
            data-testid="projects-home-refresh"
          >
            {t("settings.resources.storage.refreshAction")}
          </Button>
        ) : null}
        {settings?.isCustom ? (
          <Button
            variant="outline"
            size="sm"
            disabled={disabled}
            onClick={() => void save(null)}
            data-testid="projects-home-reset"
          >
            {t("settings.resources.storage.resetAction")}
          </Button>
        ) : null}
        <Button
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={() => setBrowserOpen(true)}
          data-testid="projects-home-change"
        >
          {saving
            ? t("common.saving")
            : t("settings.resources.storage.changeAction")}
        </Button>
      </SettingsCardRow>
      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(path) => void save(path)}
        initialPath={settings?.projectsHome}
        title={t("settings.resources.storage.projectsFolder.chooseTitle")}
        description={t("settings.resources.storage.projectsFolder.description")}
        confirmLabel={t(
          "settings.resources.storage.projectsFolder.chooseAction",
        )}
        showModelHints={false}
      />
    </>
  );
}
