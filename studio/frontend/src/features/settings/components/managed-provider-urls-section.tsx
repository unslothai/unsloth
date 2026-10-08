// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type ManagedProviderUrlSettings,
  loadManagedProviderUrls,
  updateManagedProviderUrls,
} from "../api/managed-provider-urls";
import { isSettingsRouteAbsent } from "../api/settings-route-absent";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

// Owner only: rendered from the Accounts tab.
export function ManagedProviderUrlsSection() {
  const t = useT();
  const [settings, setSettings] = useState<ManagedProviderUrlSettings | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // A backend that does not serve the route has no such setting to show.
  const [absent, setAbsent] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadManagedProviderUrls()
      .then((loaded) => {
        if (cancelled) return;
        setSettings(loaded);
        setError(null);
      })
      .catch((loadError) => {
        if (cancelled) return;
        if (isSettingsRouteAbsent(loadError)) {
          setAbsent(true);
          return;
        }
        setError(
          loadError instanceof Error
            ? loadError.message
            : t("settings.general.managedProviderUrls.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const save = async (allowed: boolean) => {
    setSaving(true);
    setError(null);
    try {
      setSettings(await updateManagedProviderUrls(allowed));
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.general.managedProviderUrls.saveError"),
      );
    } finally {
      setSaving(false);
    }
  };

  if (absent) return null;

  return (
    <SettingsSection
      title={t("settings.general.managedProviderUrls.sectionTitle")}
    >
      <SettingsRow
        label={t("settings.general.managedProviderUrls.enableLabel")}
        description={t(
          "settings.general.managedProviderUrls.enableDescription",
        )}
        below={
          settings?.lockedByEnvironment || error ? (
            <div className="flex flex-col items-end gap-1">
              {settings?.lockedByEnvironment ? (
                <span className="max-w-[calc(260px*var(--ui-space-scale,1))] text-right text-xs text-muted-foreground">
                  {t(
                    "settings.general.managedProviderUrls.lockedByEnvironment",
                  )}
                </span>
              ) : null}
              {error ? (
                <span className="max-w-[calc(260px*var(--ui-space-scale,1))] text-right text-xs text-destructive">
                  {error}
                </span>
              ) : null}
            </div>
          ) : null
        }
      >
        <Switch
          checked={settings?.allowed ?? false}
          disabled={!settings || saving || settings.lockedByEnvironment}
          onCheckedChange={(allowed) => void save(allowed)}
        />
      </SettingsRow>
    </SettingsSection>
  );
}
