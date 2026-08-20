


import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type OpenAIAutoSwitchSettings,
  type OpenAIAutoSwitchUpdate,
  loadOpenAIAutoSwitchSettings,
  updateOpenAIAutoSwitchSettings,
} from "../api/openai-auto-switch";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

// Mirrors MIN_AUTO_UNLOAD_IDLE_SECONDS in the backend settings store.
const MIN_IDLE_SECONDS = 60;

// Its own row so the section keeps one job per control: this TTL has no enable
// toggle in front of it, unlike the chat one above.
function MediaIdleUnloadRow({
  draftSeconds,
  onDraftChange,
  onSave,
  settings,
  isSaving,
  error,
}: {
  draftSeconds: string;
  onDraftChange: (value: string) => void;
  onSave: () => void;
  settings: OpenAIAutoSwitchSettings | null;
  isSaving: boolean;
  error: string | null;
}) {
  const t = useT();
  const disabled = !settings || isSaving;
  // A saved TTL that cannot run because residency is vetoing it; the number alone
  // would not say so. "Only unload models loaded by the API" is per-model now, so
  // it spares individual pipelines rather than holding the whole TTL off.
  const paused =
    settings !== null &&
    settings.mediaAutoUnloadIdleSeconds > 0 &&
    !settings.mediaIdleUnloadActive;
  return (
    <SettingsRow
      label={t("settings.general.modelAutoSwitch.mediaIdleUnload")}
      description={t(
        "settings.general.modelAutoSwitch.mediaIdleUnloadDescription",
      )}
    >
      <div className="flex flex-col items-end gap-1">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5">
            <Input
              type="number"
              min={0}
              step={1}
              value={draftSeconds}
              aria-label={t(
                "settings.general.modelAutoSwitch.mediaIdleSecondsAriaLabel",
              )}
              disabled={disabled}
              onChange={(event) => onDraftChange(event.target.value)}
              className="h-8 w-24"
            />
            <span className="text-xs font-medium text-muted-foreground">s</span>
          </div>
          <Button
            variant="outline"
            size="sm"
            disabled={disabled}
            onClick={onSave}
          >
            {isSaving ? t("common.saving") : t("common.save")}
          </Button>
        </div>
        {error ? (
          <span className="max-w-[260px] text-right text-xs text-destructive">
            {error}
          </span>
        ) : paused ? (
          <span className="max-w-[260px] text-right text-xs text-muted-foreground">
            {t("settings.general.modelAutoSwitch.mediaIdlePaused")}
          </span>
        ) : null}
      </div>
    </SettingsRow>
  );
}

export function ModelAutoSwitchSection() {
  const t = useT();
  const [settings, setSettings] = useState<OpenAIAutoSwitchSettings | null>(
    null,
  );
  const [draftIdleSeconds, setDraftIdleSeconds] = useState("0");
  const [draftMediaIdleSeconds, setDraftMediaIdleSeconds] = useState("0");
  const [error, setError] = useState<string | null>(null);
  // The media row validates its own input, so its message belongs beside it.
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadOpenAIAutoSwitchSettings()
      .then((loaded) => {
        if (cancelled) return;
        setSettings(loaded);
        setDraftIdleSeconds(String(loaded.autoUnloadIdleSeconds));
        setDraftMediaIdleSeconds(String(loaded.mediaAutoUnloadIdleSeconds));
        setError(null);
      })
      .catch((loadError) => {
        if (cancelled) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : t("settings.general.modelAutoSwitch.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  // Parse an idle-seconds draft: 0 (off) or >= MIN_IDLE_SECONDS; else null.
  const parseIdleSeconds = (draft: string): number | null => {
    if (!draft.trim()) {
      return null;
    }
    const parsed = Number(draft);
    if (!Number.isInteger(parsed)) {
      return null;
    }
    return parsed === 0 || parsed >= MIN_IDLE_SECONDS ? parsed : null;
  };

  // syncDraft only for a write the chat idle-seconds input owns; every other row
  // leaves that draft alone so a save elsewhere cannot discard what is typed there.
  const persist = async (
    update: OpenAIAutoSwitchUpdate,
    { syncDraft = false }: { syncDraft?: boolean } = {},
  ) => {
    setIsSaving(true);
    setError(null);
    try {
      const saved = await updateOpenAIAutoSwitchSettings(update);
      setSettings(saved);
      if (update.mediaAutoUnloadIdleSeconds !== undefined) {
        setDraftMediaIdleSeconds(String(saved.mediaAutoUnloadIdleSeconds));
      }
      if (syncDraft) {
        setDraftIdleSeconds(String(saved.autoUnloadIdleSeconds));
      }
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.general.modelAutoSwitch.saveError"),
      );
    } finally {
      setIsSaving(false);
    }
  };

  // Idle-unload is tied to auto-switch (the freed model reloads via the swap).
  // Toggling off preserves the saved seconds rather than zeroing them — the
  // backend gates unloading on the enabled flag, so it never unloads while off.
  // Enabling commits the drafted value, falling back to the last saved one so
  // it can never get stuck.
  const handleToggle = (enabled: boolean) => {
    const savedIdleSeconds = settings?.autoUnloadIdleSeconds ?? 0;
    if (!enabled) {
      void persist({ enabled: false, autoUnloadIdleSeconds: savedIdleSeconds });
      return;
    }
    void persist(
      {
        enabled: true,
        autoUnloadIdleSeconds:
          parseIdleSeconds(draftIdleSeconds) ?? savedIdleSeconds,
      },
      { syncDraft: true },
    );
  };

  const handleSaveIdle = () => {
    const idleSeconds = parseIdleSeconds(draftIdleSeconds);
    if (idleSeconds === null) {
      setError(t("settings.general.modelAutoSwitch.idleError"));
      return;
    }
    void persist(
      { enabled: true, autoUnloadIdleSeconds: idleSeconds },
      { syncDraft: true },
    );
  };

  // The image/video TTL is its own setting, so it saves on its own: no enable
  // toggle gates it, and the chat seconds are left untouched.
  const handleSaveMediaIdle = () => {
    if (!settings) return;
    const mediaIdleSeconds = parseIdleSeconds(draftMediaIdleSeconds);
    setMediaError(
      mediaIdleSeconds === null
        ? t("settings.general.modelAutoSwitch.idleError")
        : null,
    );
    if (mediaIdleSeconds === null) return;
    void persist({
      enabled: settings.enabled,
      mediaAutoUnloadIdleSeconds: mediaIdleSeconds,
    });
  };

  const handleKeepKvToggle = (keepKv: boolean) => {
    if (!settings) return;
    void persist({ enabled: settings.enabled, autoUnloadKeepKv: keepKv });
  };

  const handleAutoDownloadToggle = (autoDownload: boolean) => {
    if (!settings) return;
    void persist({
      enabled: settings.enabled,
      autoDownloadModel: autoDownload,
    });
  };

  // Its own setting, so it saves alone: the chat toggle above is left untouched.
  const handleMediaAutoSwitchToggle = (mediaAutoSwitch: boolean) => {
    if (!settings) return;
    void persist({
      enabled: settings.enabled,
      mediaAutoSwitchModel: mediaAutoSwitch,
    });
  };

  const handleApiOnlyToggle = (apiOnly: boolean) => {
    if (!settings) return;
    void persist({ enabled: settings.enabled, autoUnloadApiOnly: apiOnly });
  };

  return (
    <SettingsSection title={t("settings.general.modelAutoSwitch.sectionTitle")}>
      <SettingsRow
        label={t("settings.general.modelAutoSwitch.enable")}
        description={t("settings.general.modelAutoSwitch.enableDescription")}
      >
        <Switch
          checked={settings?.enabled ?? false}
          disabled={!settings || isSaving}
          onCheckedChange={handleToggle}
        />
      </SettingsRow>
      <SettingsRow
        label={t("settings.general.modelAutoSwitch.autoDownload")}
        description={t(
          "settings.general.modelAutoSwitch.autoDownloadDescription",
        )}
      >
        <Switch
          checked={settings?.autoDownloadModel ?? false}
          disabled={!settings?.enabled || isSaving}
          onCheckedChange={handleAutoDownloadToggle}
        />
      </SettingsRow>
      <SettingsRow
        label={t("settings.general.modelAutoSwitch.idleUnload")}
        description={t(
          "settings.general.modelAutoSwitch.idleUnloadDescription",
        )}
      >
        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <Input
                type="number"
                min={0}
                step={1}
                value={draftIdleSeconds}
                aria-label={t(
                  "settings.general.modelAutoSwitch.idleSecondsAriaLabel",
                )}
                disabled={!settings?.enabled || isSaving}
                onChange={(event) => setDraftIdleSeconds(event.target.value)}
                className="h-8 w-24"
              />
              <span className="text-xs font-medium text-muted-foreground">
                s
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              disabled={!settings?.enabled || isSaving}
              onClick={handleSaveIdle}
            >
              {isSaving ? t("common.saving") : t("common.save")}
            </Button>
          </div>
          {error ? (
            <span className="max-w-[260px] text-right text-xs text-destructive">
              {error}
            </span>
          ) : settings && !settings.enabled && settings.idleUnloadActive ? (
            <span className="max-w-[260px] text-right text-xs text-muted-foreground">
              {t("settings.general.modelAutoSwitch.idleActiveViaEnv")}
            </span>
          ) : settings && !settings.enabled ? (
            <span className="max-w-[260px] text-right text-xs text-muted-foreground">
              {t("settings.general.modelAutoSwitch.idleNeedsEnable")}
            </span>
          ) : null}
        </div>
      </SettingsRow>
      <SettingsRow
        label={t("settings.general.modelAutoSwitch.mediaEnable")}
        description={t(
          "settings.general.modelAutoSwitch.mediaEnableDescription",
        )}
      >
        <Switch
          checked={settings?.mediaAutoSwitchModel ?? false}
          disabled={!settings || isSaving}
          onCheckedChange={handleMediaAutoSwitchToggle}
        />
      </SettingsRow>
      <MediaIdleUnloadRow
        draftSeconds={draftMediaIdleSeconds}
        onDraftChange={setDraftMediaIdleSeconds}
        onSave={handleSaveMediaIdle}
        settings={settings}
        isSaving={isSaving}
        error={mediaError}
      />
      {settings?.idleUnloadActive ? (
        <SettingsRow
          label={t("settings.general.modelAutoSwitch.keepKv")}
          description={t("settings.general.modelAutoSwitch.keepKvDescription")}
        >
          <Switch
            checked={settings.autoUnloadKeepKv}
            disabled={isSaving}
            onCheckedChange={handleKeepKvToggle}
          />
        </SettingsRow>
      ) : null}
      {/* Also whenever a media TTL is saved: this switch decides which media models that
          TTL may free, so it has to be reachable without re-enabling chat unloading first. */}
      {settings &&
      (settings.idleUnloadActive || settings.mediaAutoUnloadIdleSeconds > 0) ? (
        <SettingsRow
          label={t("settings.general.modelAutoSwitch.apiOnly")}
          description={t("settings.general.modelAutoSwitch.apiOnlyDescription")}
        >
          <Switch
            checked={settings.autoUnloadApiOnly}
            disabled={isSaving}
            onCheckedChange={handleApiOnlyToggle}
          />
        </SettingsRow>
      ) : null}
    </SettingsSection>
  );
}
