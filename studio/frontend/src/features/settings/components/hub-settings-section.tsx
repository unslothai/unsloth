// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useIsAccountOwner } from "@/features/auth";
import { useT } from "@/i18n";
import { DEFAULT_HF_ENDPOINT, type HubSource } from "@/lib/hf-endpoint";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import {
  type HubEndpointSettings,
  type HubSettings,
  InvalidHubEndpointError,
  loadHubSettings,
  updateHubSettings,
  updateHubSource,
} from "../api/hub-settings";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const SOURCES: { value: HubSource; label: string }[] = [
  { value: "huggingface", label: "Hugging Face" },
  { value: "modelscope", label: "ModelScope" },
];

export function HubSettingsSection() {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const [settings, setSettings] = useState<HubSettings | null>(null);
  const [draftEndpoint, setDraftEndpoint] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadHubSettings()
      .then((loaded) => {
        if (cancelled) {
          return;
        }
        setSettings(loaded);
        setDraftEndpoint(loaded.hfEndpoint);
      })
      .catch(() => {
        if (!cancelled) {
          setError(t("settings.general.hub.loadFailed"));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const save = async (next: HubEndpointSettings) => {
    setSaving(true);
    setError(null);
    try {
      const saved = await updateHubSettings(next);
      setSettings(saved);
      // A toggle saves the committed endpoint and leaves an unsaved draft alone.
      if (next.hfEndpoint === draftEndpoint) {
        setDraftEndpoint(saved.hfEndpoint);
      }
    } catch (err) {
      setError(
        err instanceof InvalidHubEndpointError
          ? t("settings.general.hub.invalidEndpoint")
          : t("settings.general.hub.saveFailed"),
      );
    } finally {
      setSaving(false);
    }
  };

  const locked = !settings || saving || !isOwner;
  const endpointChanged =
    settings !== null && draftEndpoint.trim() !== settings.hfEndpoint;
  const saveSource = async (source: HubSource) => {
    setSaving(true);
    setError(null);
    try {
      setSettings(await updateHubSource(source));
    } catch {
      setError(t("settings.general.hub.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  // Hidden while ModelScope serves; a ModelScope that failed to start falls back to the endpoint.
  const showEndpointRows = settings?.activeSource !== "modelscope";
  const errorNote = error ? (
    <span className="max-w-[300px] text-right text-xs text-destructive">
      {error}
    </span>
  ) : null;

  const saveEndpoint = () =>
    settings &&
    void save({
      hfEndpoint: draftEndpoint,
      // Nothing to follow once the endpoint is cleared.
      datasetsServerFollowsEndpoint:
        settings.datasetsServerFollowsEndpoint && draftEndpoint.trim() !== "",
    });

  return (
    <SettingsSection title={t("settings.general.hub.sectionTitle")}>
      <SettingsRow
        alignTop={true}
        label={t("settings.general.hub.source")}
        description={t("settings.general.hub.sourceDescription")}
      >
        <div className="flex flex-col items-end gap-1">
          <div
            role="radiogroup"
            aria-label={t("settings.general.hub.source")}
            className="hub-tab-toggle inline-flex h-8 items-center rounded-full"
          >
            {SOURCES.map((option) => {
              const active = settings?.source === option.value;
              return (
                <button
                  key={option.value}
                  type="button"
                  role="radio"
                  data-settings-label={option.label}
                  aria-checked={active}
                  disabled={locked}
                  onClick={() => {
                    if (!active) void saveSource(option.value);
                  }}
                  className={cn(
                    "relative flex h-8 items-center rounded-full px-3 text-xs font-medium transition-colors disabled:cursor-not-allowed",
                    active
                      ? "hub-tab-toggle-pill text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  <span className="relative z-10">{option.label}</span>
                </button>
              );
            })}
          </div>
          {settings && settings.source !== settings.activeSource ? (
            <span className="max-w-[300px] text-right text-xs text-destructive">
              {t("settings.general.hub.sourceFallback")}
            </span>
          ) : null}
          {showEndpointRows ? null : errorNote}
        </div>
      </SettingsRow>
      {showEndpointRows ? (
        <>
          <SettingsRow
            alignTop={true}
            label={t("settings.general.hub.endpoint")}
            description={t("settings.general.hub.endpointDescription")}
          >
            <div className="flex flex-col items-end gap-1">
              <div className="flex items-center gap-2">
                <Input
                  type="url"
                  value={draftEndpoint}
                  placeholder={DEFAULT_HF_ENDPOINT}
                  disabled={locked}
                  aria-label={t("settings.general.hub.endpoint")}
                  onChange={(event) => setDraftEndpoint(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && endpointChanged) {
                      saveEndpoint();
                    }
                  }}
                  className="h-8 w-60"
                />
                <Button
                  variant="outline"
                  size="sm"
                  disabled={locked || !endpointChanged}
                  onClick={saveEndpoint}
                >
                  {saving ? t("common.saving") : t("common.save")}
                </Button>
              </div>
              {errorNote}
            </div>
          </SettingsRow>
          <SettingsRow
            label={t("settings.general.hub.datasetsServer")}
            description={t("settings.general.hub.datasetsServerDescription")}
          >
            <Switch
              checked={settings?.datasetsServerFollowsEndpoint ?? false}
              disabled={locked || !settings?.hfEndpoint}
              onCheckedChange={(checked) =>
                settings &&
                void save({
                  ...settings,
                  datasetsServerFollowsEndpoint: checked,
                })
              }
            />
          </SettingsRow>
        </>
      ) : null}
    </SettingsSection>
  );
}
