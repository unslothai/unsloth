// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings section for system-wide voice typing. Rendered only in the desktop app
// on platforms that can arm a global shortcut (see voice_typing_info).

import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  chordFromEvent,
  formatShortcut,
  isModifierCode,
  queryMicrophonePermission,
  requestVoiceTypingMicrophoneAccess,
  requestVoiceTypingModelDownload,
  type MicrophonePermission,
  type VoiceTypingMode,
  useVoiceTypingStore,
} from "@/features/voice-typing";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { useVoiceSettingsStore } from "../stores/voice-settings-store";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

interface VoiceTypingInfo {
  os: string;
  shortcutSupported: boolean;
  holdSupported: boolean;
  pasteSupported: boolean;
}

export function VoiceTypingSettingsSection() {
  const t = useT();
  const enabled = useVoiceTypingStore((state) => state.enabled);
  const setEnabled = useVoiceTypingStore((state) => state.setEnabled);
  const shortcut = useVoiceTypingStore((state) => state.shortcut);
  const setShortcut = useVoiceTypingStore((state) => state.setShortcut);
  const mode = useVoiceTypingStore((state) => state.mode);
  const setMode = useVoiceTypingStore((state) => state.setMode);
  const hudEnabled = useVoiceTypingStore((state) => state.hudEnabled);
  const setHudEnabled = useVoiceTypingStore((state) => state.setHudEnabled);
  const soundEnabled = useVoiceTypingStore((state) => state.soundEnabled);
  const setSoundEnabled = useVoiceTypingStore((state) => state.setSoundEnabled);
  const appendTrailingSpace = useVoiceTypingStore(
    (state) => state.appendTrailingSpace,
  );
  const setAppendTrailingSpace = useVoiceTypingStore(
    (state) => state.setAppendTrailingSpace,
  );
  const keepOnClipboard = useVoiceTypingStore((state) => state.keepOnClipboard);
  const setKeepOnClipboard = useVoiceTypingStore(
    (state) => state.setKeepOnClipboard,
  );
  const readiness = useVoiceTypingStore((state) => state.readiness);
  const preloadModel = useVoiceTypingStore((state) => state.preloadModel);
  const setPreloadModel = useVoiceTypingStore((state) => state.setPreloadModel);
  const autoStopSilence = useVoiceTypingStore((state) => state.autoStopSilence);
  const setAutoStopSilence = useVoiceTypingStore(
    (state) => state.setAutoStopSilence,
  );
  const pasteAutomatically = useVoiceTypingStore(
    (state) => state.pasteAutomatically,
  );
  const setPasteAutomatically = useVoiceTypingStore(
    (state) => state.setPasteAutomatically,
  );
  const sttModel = useVoiceSettingsStore((state) => state.sttModel);

  const [info, setInfo] = useState<VoiceTypingInfo | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [needsModifier, setNeedsModifier] = useState(false);
  const [micPermission, setMicPermission] =
    useState<MicrophonePermission>("unknown");

  useEffect(() => {
    if (!isTauri) return;
    let live = true;
    void invoke<VoiceTypingInfo>("voice_typing_info")
      .then((value) => {
        if (live) setInfo(value);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  // The microphone row follows the webview's own permission state, which can
  // change while the dialog is open (the user leaves to click Allow and returns).
  useEffect(() => {
    let live = true;
    const refresh = () => {
      void queryMicrophonePermission().then((permission) => {
        if (live) setMicPermission(permission);
      });
    };
    refresh();
    window.addEventListener("focus", refresh);
    return () => {
      live = false;
      window.removeEventListener("focus", refresh);
    };
  }, []);

  // While the capture box is armed, the next chord becomes the shortcut. Capture
  // runs before the app's own shortcut handling so nothing else reacts.
  useEffect(() => {
    if (!capturing) return;
    const onKeyDown = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (event.key === "Escape") {
        setCapturing(false);
        setNeedsModifier(false);
        return;
      }
      const chord = chordFromEvent(event);
      if (!chord) {
        // Modifier-only presses keep waiting; a bare key explains the rule.
        if (!isModifierCode(event.code)) setNeedsModifier(true);
        return;
      }
      setShortcut(chord);
      setCapturing(false);
      setNeedsModifier(false);
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [capturing, setShortcut]);

  // Browser dev sessions and platforms without a shortcut layer never see this.
  if (!isTauri || !info?.shortcutSupported) return null;

  const modeDescription =
    mode === "hold"
      ? t("settings.voice.voiceTyping.modeHoldDescription")
      : mode === "toggle"
        ? t("settings.voice.voiceTyping.modeToggleDescription")
        : t("settings.voice.voiceTyping.modeHybridDescription");

  const modelStatusText =
    readiness === "ready"
      ? t("settings.voice.voiceTyping.modelReady")
      : readiness === "loading"
        ? t("settings.voice.voiceTyping.modelLoading")
        : readiness === "will-load"
          ? t("settings.voice.voiceTyping.modelWillLoad")
          : readiness === "needs-download"
            ? t("settings.voice.voiceTyping.modelNeedsDownload")
            : readiness === "engine-missing"
              ? t("settings.voice.voiceTyping.modelEngineMissing")
              : readiness === "custom"
                ? t("settings.voice.voiceTyping.modelCustom")
                : t("settings.voice.voiceTyping.modelUnknown");

  return (
    <SettingsSection
      title={t("settings.voice.voiceTyping.sectionTitle")}
      description={t("settings.voice.voiceTyping.sectionDescription")}
    >
      <SettingsRow
        label={t("settings.voice.voiceTyping.enableLabel")}
        description={t("settings.voice.voiceTyping.enableDescription")}
      >
        <Switch
          checked={enabled}
          onCheckedChange={setEnabled}
          data-testid="voice-typing-enable"
          aria-label={t("settings.voice.voiceTyping.enableLabel")}
        />
      </SettingsRow>
      {enabled ? (
        <p
          className="-mt-1 pb-3 text-xs text-muted-foreground"
          data-testid="voice-typing-active-hint"
        >
          {t("settings.voice.voiceTyping.activeHint", {
            shortcut: formatShortcut(shortcut),
          })}
        </p>
      ) : null}

      {enabled ? (
        <SettingsRow
          label={t("settings.voice.voiceTyping.modelLabel")}
          description={sttModel}
        >
          <div className="flex items-center gap-2">
            <span
              className="text-xs text-muted-foreground"
              data-testid="voice-typing-model-status"
            >
              {modelStatusText}
            </span>
            {readiness === "needs-download" ? (
              <Button
                variant="outline"
                size="sm"
                data-testid="voice-typing-model-download"
                onClick={requestVoiceTypingModelDownload}
              >
                {t("settings.voice.voiceTyping.modelDownloadAction")}
              </Button>
            ) : null}
          </div>
        </SettingsRow>
      ) : null}

      {enabled ? (
        <SettingsRow
          label={t("settings.voice.dictation.microphoneLabel")}
          description={
            micPermission === "denied"
              ? t("settings.voice.dictation.micAccessBlockedDesktop")
              : t("settings.voice.dictation.microphoneDescription")
          }
        >
          <div className="flex items-center gap-2">
            <span
              className="text-xs text-muted-foreground"
              data-testid="voice-typing-mic-status"
            >
              {micPermission === "granted"
                ? t("settings.voice.voiceTyping.micAccessGranted")
                : micPermission === "denied"
                  ? t("settings.voice.voiceTyping.micAccessBlockedStatus")
                  : t("settings.voice.voiceTyping.micAccessNeeded")}
            </span>
            {micPermission !== "granted" ? (
              <Button
                variant="outline"
                size="sm"
                data-testid="voice-typing-mic-allow"
                onClick={() => {
                  void requestVoiceTypingMicrophoneAccess().then(() => {
                    void queryMicrophonePermission().then(setMicPermission);
                  });
                }}
              >
                {t("settings.voice.dictation.allowMicrophone")}
              </Button>
            ) : null}
          </div>
        </SettingsRow>
      ) : null}

      <SettingsRow
        label={t("settings.voice.voiceTyping.shortcutLabel")}
        description={t("settings.voice.voiceTyping.shortcutHint")}
      >
        <Button
          variant="outline"
          size="sm"
          data-testid="voice-typing-shortcut"
          className="min-w-44 justify-center font-mono text-xs"
          aria-label={t("settings.voice.voiceTyping.shortcutLabel")}
          onClick={() => {
            setNeedsModifier(false);
            setCapturing((value) => !value);
          }}
        >
          {capturing
            ? t("settings.voice.voiceTyping.shortcutListening")
            : formatShortcut(shortcut)}
        </Button>
      </SettingsRow>
      {needsModifier ? (
        <p className="pb-2 text-xs text-destructive">
          {t("settings.voice.voiceTyping.shortcutNeedsModifier")}
        </p>
      ) : null}

      <SettingsRow
        label={t("settings.voice.voiceTyping.modeLabel")}
        description={modeDescription}
      >
        <Select
          value={mode}
          onValueChange={(value) => setMode(value as VoiceTypingMode)}
        >
          <SelectTrigger
            size="sm"
            className="w-44"
            data-testid="voice-typing-mode"
            aria-label={t("settings.voice.voiceTyping.modeLabel")}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="hybrid">
              {t("settings.voice.voiceTyping.modeHybrid")}
            </SelectItem>
            <SelectItem value="hold">
              {t("settings.voice.voiceTyping.modeHold")}
            </SelectItem>
            <SelectItem value="toggle">
              {t("settings.voice.voiceTyping.modeToggle")}
            </SelectItem>
          </SelectContent>
        </Select>
      </SettingsRow>

      <SettingsRow
        label={t("settings.voice.voiceTyping.hudLabel")}
        description={t("settings.voice.voiceTyping.hudDescription")}
      >
        <Switch
          checked={hudEnabled}
          onCheckedChange={setHudEnabled}
          data-testid="voice-typing-hud"
          aria-label={t("settings.voice.voiceTyping.hudLabel")}
        />
      </SettingsRow>

      <SettingsRow
        label={t("settings.voice.voiceTyping.soundsLabel")}
        description={t("settings.voice.voiceTyping.soundsDescription")}
      >
        <Switch
          checked={soundEnabled}
          onCheckedChange={setSoundEnabled}
          data-testid="voice-typing-sounds"
          aria-label={t("settings.voice.voiceTyping.soundsLabel")}
        />
      </SettingsRow>

      <SettingsRow
        label={t("settings.voice.voiceTyping.trailingSpaceLabel")}
        description={t("settings.voice.voiceTyping.trailingSpaceDescription")}
      >
        <Switch
          checked={appendTrailingSpace}
          onCheckedChange={setAppendTrailingSpace}
          data-testid="voice-typing-trailing-space"
          aria-label={t("settings.voice.voiceTyping.trailingSpaceLabel")}
        />
      </SettingsRow>

      <SettingsRow
        label={t("settings.voice.voiceTyping.clipboardLabel")}
        description={t("settings.voice.voiceTyping.clipboardDescription")}
      >
        <Switch
          checked={keepOnClipboard}
          onCheckedChange={setKeepOnClipboard}
          data-testid="voice-typing-keep-clipboard"
          aria-label={t("settings.voice.voiceTyping.clipboardLabel")}
        />
      </SettingsRow>

      <SettingsRow
        label={t("settings.voice.voiceTyping.pasteLabel")}
        description={t("settings.voice.voiceTyping.pasteDescription")}
      >
        <Switch
          checked={pasteAutomatically}
          onCheckedChange={setPasteAutomatically}
          data-testid="voice-typing-paste"
          aria-label={t("settings.voice.voiceTyping.pasteLabel")}
        />
      </SettingsRow>

      <SettingsRow
        label={t("settings.voice.voiceTyping.preloadLabel")}
        description={t("settings.voice.voiceTyping.preloadDescription")}
      >
        <Switch
          checked={preloadModel}
          onCheckedChange={setPreloadModel}
          data-testid="voice-typing-preload"
          aria-label={t("settings.voice.voiceTyping.preloadLabel")}
        />
      </SettingsRow>

      <SettingsRow
        label={t("settings.voice.voiceTyping.autoStopLabel")}
        description={t("settings.voice.voiceTyping.autoStopDescription")}
      >
        <Switch
          checked={autoStopSilence}
          onCheckedChange={setAutoStopSilence}
          data-testid="voice-typing-auto-stop"
          aria-label={t("settings.voice.voiceTyping.autoStopLabel")}
        />
      </SettingsRow>

      <p className="pt-1 text-xs text-muted-foreground">
        {t("settings.voice.voiceTyping.engineNote")}
      </p>
    </SettingsSection>
  );
}
