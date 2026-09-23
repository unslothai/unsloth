// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useIsAccountOwner } from "@/features/auth";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
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

// The page's CSP is set when it is served, so browsing picks up a new endpoint on reload.
// Module scope, so the notice outlives this section unmounting until that reload.
let reloadPending = false;

export function HubSettingsSection() {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const [settings, setSettings] = useState<HubSettings | null>(null);
  const [draftEndpoint, setDraftEndpoint] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadNeeded, setReloadNeeded] = useState(reloadPending);

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
      // Either one can move the datasets server off the origins the page allows.
      if (
        settings &&
        (saved.hfEndpoint !== settings.hfEndpoint ||
          saved.datasetsServerFollowsEndpoint !==
            settings.datasetsServerFollowsEndpoint)
      ) {
        reloadPending = true;
        setReloadNeeded(true);
      }
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
      const saved = await updateHubSource(source);
      // A page served under ModelScope does not allow a custom endpoint's origins.
      if (
        settings?.activeSource === "modelscope" &&
        saved.activeSource === "huggingface" &&
        saved.hfEndpoint
      ) {
        reloadPending = true;
        setReloadNeeded(true);
      }
      setSettings(saved);
    } catch {
      setError(t("settings.general.hub.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

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
        </div>
      </SettingsRow>
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
          {settings?.source === "modelscope" ? (
            <span className="max-w-[300px] text-right text-xs text-muted-foreground">
              {t("settings.general.hub.endpointInactive")}
            </span>
          ) : null}
          {reloadNeeded && isTauri ? (
            // The desktop webview's CSP is built at launch from the environment.
            <span className="max-w-[300px] text-right text-xs text-muted-foreground">
              {t("settings.general.hub.desktopBrowsing")}
            </span>
          ) : reloadNeeded ? (
            <span className="flex max-w-[300px] items-center justify-end gap-2 text-right text-xs text-muted-foreground">
              {t("settings.general.hub.reloadNeeded")}
              <Button
                variant="outline"
                size="sm"
                onClick={() => window.location.reload()}
              >
                {t("settings.general.hub.reload")}
              </Button>
            </span>
          ) : null}
          {error ? (
            <span className="max-w-[300px] text-right text-xs text-destructive">
              {error}
            </span>
          ) : null}
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
            void save({ ...settings, datasetsServerFollowsEndpoint: checked })
          }
        />
      </SettingsRow>
    </SettingsSection>
  );
}
