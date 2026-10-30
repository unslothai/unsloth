// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings section for system-wide voice typing. Rendered only in the desktop app
// on platforms that can arm a global shortcut (see voice_typing_info).

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useMicrophonePermissionStore } from "@/features/microphone";
import {
  DEFAULT_VOICE_TYPING_SHORTCUT,
  MIC_WARM_ALWAYS,
  VOICE_TYPING_HUD_POSITIONS,
  VOICE_TYPING_PASTE_METHODS,
  VOICE_TYPING_SOUND_THEMES,
  type VoiceTypingHudPosition,
  type VoiceTypingMode,
  type VoiceTypingPasteMethod,
  type VoiceTypingSoundTheme,
  chordFromEvent,
  formatShortcut,
  isConflictingShortcut,
  isModifierCode,
  previewVoiceTypingSound,
  requestVoiceTypingModelDownload,
  useVoiceTypingStore,
} from "@/features/voice-typing";
import { type TranslationKey, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { Volume02Icon } from "@/lib/volume-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { invoke } from "@tauri-apps/api/core";
import { useEffect, useRef, useState } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsGroupDivider, SettingsSection } from "./settings-section";

interface VoiceTypingInfo {
  os: string;
  shortcutSupported: boolean;
  holdSupported: boolean;
  pasteSupported: boolean;
}

const SOUND_THEME_LABEL: Record<VoiceTypingSoundTheme, TranslationKey> = {
  chime: "settings.voice.voiceTyping.soundsThemeChime",
  marimba: "settings.voice.voiceTyping.soundsThemeMarimba",
  pop: "settings.voice.voiceTyping.soundsThemePop",
  soft: "settings.voice.voiceTyping.soundsThemeSoft",
};

const PASTE_METHOD_LABEL: Record<VoiceTypingPasteMethod, TranslationKey> = {
  ctrl_v: "settings.voice.voiceTyping.pasteMethodCtrlV",
  ctrl_shift_v: "settings.voice.voiceTyping.pasteMethodCtrlShiftV",
  shift_insert: "settings.voice.voiceTyping.pasteMethodShiftInsert",
  direct: "settings.voice.voiceTyping.pasteMethodDirect",
};

const HUD_POSITION_LABEL: Record<VoiceTypingHudPosition, TranslationKey> = {
  bottom: "settings.voice.voiceTyping.hudPositionBottom",
  top: "settings.voice.voiceTyping.hudPositionTop",
};

export function VoiceTypingSettingsSection() {
  const t = useT();
  const enabled = useVoiceTypingStore((state) => state.enabled);
  const setEnabled = useVoiceTypingStore((state) => state.setEnabled);
  const shortcut = useVoiceTypingStore((state) => state.shortcut);
  const setShortcut = useVoiceTypingStore((state) => state.setShortcut);
  const mode = useVoiceTypingStore((state) => state.mode);
  const setMode = useVoiceTypingStore((state) => state.setMode);
  const soundEnabled = useVoiceTypingStore((state) => state.soundEnabled);
  const setSoundEnabled = useVoiceTypingStore((state) => state.setSoundEnabled);
  const soundTheme = useVoiceTypingStore((state) => state.soundTheme);
  const setSoundTheme = useVoiceTypingStore((state) => state.setSoundTheme);
  const soundVolume = useVoiceTypingStore((state) => state.soundVolume);
  const setSoundVolume = useVoiceTypingStore((state) => state.setSoundVolume);
  const appendTrailingSpace = useVoiceTypingStore(
    (state) => state.appendTrailingSpace,
  );
  const setAppendTrailingSpace = useVoiceTypingStore(
    (state) => state.setAppendTrailingSpace,
  );
  const copyLastTranscript = useVoiceTypingStore(
    (state) => state.copyLastTranscript,
  );
  const setCopyLastTranscript = useVoiceTypingStore(
    (state) => state.setCopyLastTranscript,
  );
  const preloadModel = useVoiceTypingStore((state) => state.preloadModel);
  const setPreloadModel = useVoiceTypingStore((state) => state.setPreloadModel);
  const autoStopSilence = useVoiceTypingStore((state) => state.autoStopSilence);
  const setAutoStopSilence = useVoiceTypingStore(
    (state) => state.setAutoStopSilence,
  );
  const autoStopDelayMs = useVoiceTypingStore((state) => state.autoStopDelayMs);
  const setAutoStopDelayMs = useVoiceTypingStore(
    (state) => state.setAutoStopDelayMs,
  );
  const micWarmSeconds = useVoiceTypingStore((state) => state.micWarmSeconds);
  const setMicWarmSeconds = useVoiceTypingStore(
    (state) => state.setMicWarmSeconds,
  );
  const removeFillerWords = useVoiceTypingStore(
    (state) => state.removeFillerWords,
  );
  const setRemoveFillerWords = useVoiceTypingStore(
    (state) => state.setRemoveFillerWords,
  );
  const spokenPunctuation = useVoiceTypingStore(
    (state) => state.spokenPunctuation,
  );
  const setSpokenPunctuation = useVoiceTypingStore(
    (state) => state.setSpokenPunctuation,
  );
  const submitAfterInsert = useVoiceTypingStore(
    (state) => state.submitAfterInsert,
  );
  const setSubmitAfterInsert = useVoiceTypingStore(
    (state) => state.setSubmitAfterInsert,
  );
  const pasteAutomatically = useVoiceTypingStore(
    (state) => state.pasteAutomatically,
  );
  const setPasteAutomatically = useVoiceTypingStore(
    (state) => state.setPasteAutomatically,
  );
  const pasteMethod = useVoiceTypingStore((state) => state.pasteMethod);
  const setPasteMethod = useVoiceTypingStore((state) => state.setPasteMethod);
  const hudPosition = useVoiceTypingStore((state) => state.hudPosition);
  const setHudPosition = useVoiceTypingStore((state) => state.setHudPosition);
  const muteWhileRecording = useVoiceTypingStore(
    (state) => state.muteWhileRecording,
  );
  const setMuteWhileRecording = useVoiceTypingStore(
    (state) => state.setMuteWhileRecording,
  );
  const readiness = useVoiceTypingStore((state) => state.readiness);
  const lastTranscript = useVoiceTypingStore((state) => state.lastTranscript);

  const micPermission = useMicrophonePermissionStore(
    (state) => state.permission,
  );
  const loadMicPermission = useMicrophonePermissionStore((state) => state.load);
  const requestMicPermission = useMicrophonePermissionStore(
    (state) => state.request,
  );
  const askAgainMicPermission = useMicrophonePermissionStore(
    (state) => state.askAgain,
  );
  const resetMicPermission = useMicrophonePermissionStore(
    (state) => state.reset,
  );

  const [info, setInfo] = useState<VoiceTypingInfo | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [needsModifier, setNeedsModifier] = useState(false);
  const shortcutButtonRef = useRef<HTMLButtonElement | null>(null);

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

  useEffect(() => {
    void loadMicPermission();
  }, [loadMicPermission]);

  // While the capture box is armed, suspend the native global shortcut poller and
  // capture the next chord. Clicking outside or blurring cancels capture cleanly.
  useEffect(() => {
    if (!capturing) return;
    void invoke("voice_typing_suspend_shortcut", { suspended: true }).catch(
      () => {},
    );
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
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (shortcutButtonRef.current && target && shortcutButtonRef.current.contains(target)) {
        return;
      }
      setCapturing(false);
      setNeedsModifier(false);
    };
    const onWindowBlur = () => {
      setCapturing(false);
      setNeedsModifier(false);
    };
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("blur", onWindowBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("blur", onWindowBlur);
      void invoke("voice_typing_suspend_shortcut", { suspended: false }).catch(
        () => {},
      );
    };
  }, [capturing, setShortcut]);

  // Browser dev sessions and platforms without a shortcut layer never see this.
  if (!isTauri || !info?.shortcutSupported) return null;

  const modeDescription =
    mode === "hold"
      ? t("settings.voice.voiceTyping.modeHoldDescription")
      : mode === "toggle"
        ? t("settings.voice.voiceTyping.modeToggleDescription")
        : t("settings.voice.voiceTyping.modeHybridDescription");

  const micStatusText =
    micPermission === "granted"
      ? t("settings.voice.voiceTyping.micAccessGranted")
      : micPermission === "denied"
        ? t("settings.voice.voiceTyping.micAccessBlockedStatus")
        : t("settings.voice.voiceTyping.micAccessNeeded");
  const micStatusDescription =
    micPermission === "granted"
      ? t("settings.voice.voiceTyping.micAccessGrantedDescription")
      : micPermission === "denied"
        ? t("settings.voice.voiceTyping.micAccessBlockedDescription")
        : t("settings.voice.voiceTyping.micAccessNeededDescription");

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
                : t("settings.voice.voiceTyping.modelChecking");

  const shortcutHasConflict = isConflictingShortcut(shortcut);

  return (
    <SettingsSection title={t("settings.voice.voiceTyping.sectionTitle")}>
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
        <>
          <div
            className="-mt-1 flex flex-wrap items-center justify-between gap-2 pb-3 text-xs text-muted-foreground"
            data-testid="voice-typing-active-hint"
          >
            <span>
              {t("settings.voice.voiceTyping.activeHint", {
                shortcut: formatShortcut(shortcut),
              })}
            </span>
            <span
              className="inline-flex items-center gap-1.5 rounded-full bg-muted/60 px-2.5 py-0.5 text-ui-11 font-medium text-foreground/80"
              data-testid="voice-typing-model-status"
            >
              <span
                className={`size-1.5 rounded-full ${
                  readiness === "ready" || readiness === "custom"
                    ? "bg-emerald-500"
                    : readiness === "loading" || readiness === "will-load"
                      ? "bg-amber-500"
                      : readiness === "needs-download" ||
                          readiness === "engine-missing"
                        ? "bg-destructive"
                        : "bg-muted-foreground/50"
                }`}
              />
              {modelStatusText}
              {readiness === "needs-download" ? (
                <button
                  type="button"
                  className="ml-1 font-semibold text-primary underline underline-offset-2 hover:opacity-80"
                  onClick={() => requestVoiceTypingModelDownload()}
                >
                  {t("settings.voice.voiceTyping.downloadAction")}
                </button>
              ) : null}
            </span>
          </div>

          {lastTranscript ? (
            <div className="mb-3 flex items-center justify-between gap-3 rounded-lg border border-border/60 bg-muted/30 px-3 py-2">
              <div className="min-w-0 flex-1">
                <p className="text-ui-11 font-medium text-muted-foreground">
                  {t("settings.voice.voiceTyping.lastTranscriptLabel")}
                </p>
                <p className="truncate text-xs text-foreground">
                  {lastTranscript}
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="shrink-0 gap-1.5"
                data-testid="voice-typing-copy-last"
                onClick={() => {
                  void copyToClipboard(lastTranscript).then((ok) => {
                    if (ok) {
                      toast.success(
                        t("settings.voice.voiceTyping.copiedNotice"),
                      );
                    }
                  });
                }}
              >
                <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
                {t("settings.voice.voiceTyping.copyLastAction")}
              </Button>
            </div>
          ) : null}

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.shortcutLabel")}
            description={t("settings.voice.voiceTyping.shortcutHint")}
          >
            <div className="flex items-center gap-2">
              {shortcut !== DEFAULT_VOICE_TYPING_SHORTCUT ? (
                <Button
                  variant="ghost"
                  size="sm"
                  data-testid="voice-typing-shortcut-reset"
                  onClick={() => {
                    setCapturing(false);
                    setNeedsModifier(false);
                    setShortcut(DEFAULT_VOICE_TYPING_SHORTCUT);
                  }}
                >
                  {t("settings.voice.voiceTyping.shortcutReset")}
                </Button>
              ) : null}
              <Button
                ref={shortcutButtonRef}
                variant={capturing ? "secondary" : "outline"}
                size="sm"
                data-testid="voice-typing-shortcut"
                className="w-56 justify-center font-mono text-xs"
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
            </div>
          </SettingsRow>
          {needsModifier ? (
            <p className="pb-2 text-xs text-destructive">
              {t("settings.voice.voiceTyping.shortcutNeedsModifier")}
            </p>
          ) : shortcutHasConflict ? (
            <p className="pb-2 text-xs text-amber-600 dark:text-amber-400">
              {t("settings.voice.voiceTyping.shortcutConflictHint")}
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
                className="w-56"
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
            label={t("settings.voice.voiceTyping.hudPositionLabel")}
            description={t("settings.voice.voiceTyping.hudPositionDescription")}
          >
            <Select
              value={hudPosition}
              onValueChange={(value) =>
                setHudPosition(value as VoiceTypingHudPosition)
              }
            >
              <SelectTrigger
                size="sm"
                className="w-56"
                data-testid="voice-typing-hud-position"
                aria-label={t("settings.voice.voiceTyping.hudPositionLabel")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {VOICE_TYPING_HUD_POSITIONS.map((pos) => (
                  <SelectItem key={pos} value={pos}>
                    {t(HUD_POSITION_LABEL[pos])}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </SettingsRow>

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.micAccessLabel")}
            description={micStatusDescription}
          >
            <div className="flex items-center gap-2">
              <span
                className="flex items-center gap-1.5 text-xs text-muted-foreground"
                data-testid="voice-typing-mic-status"
              >
                <span
                  className={`size-1.5 rounded-full ${
                    micPermission === "granted"
                      ? "bg-emerald-500"
                      : micPermission === "denied"
                        ? "bg-destructive"
                        : "bg-muted-foreground/50"
                  }`}
                />
                {micStatusText}
              </span>
              {micPermission === "granted" ? (
                <Button
                  variant="ghost"
                  size="sm"
                  data-testid="voice-typing-mic-reset"
                  onClick={() => {
                    void resetMicPermission();
                  }}
                >
                  {t("settings.voice.voiceTyping.micAccessReset")}
                </Button>
              ) : micPermission === "denied" ? (
                <Button
                  variant="outline"
                  size="sm"
                  data-testid="voice-typing-mic-ask-again"
                  onClick={() => {
                    void askAgainMicPermission("settings");
                  }}
                >
                  {t("settings.voice.voiceTyping.micAccessAskAgain")}
                </Button>
              ) : (
                <Button
                  variant="outline"
                  size="sm"
                  data-testid="voice-typing-mic-allow"
                  onClick={() => {
                    void requestMicPermission("settings");
                  }}
                >
                  {t("settings.voice.dictation.allowMicrophone")}
                </Button>
              )}
            </div>
          </SettingsRow>

          <SettingsRow
            label={t("settings.voice.voiceTyping.micWarmLabel")}
            description={
              micWarmSeconds === MIC_WARM_ALWAYS
                ? t("settings.voice.voiceTyping.micWarmAlwaysDescription")
                : t("settings.voice.voiceTyping.micWarmDescription")
            }
          >
            <Select
              value={String(micWarmSeconds)}
              onValueChange={(value) => setMicWarmSeconds(Number(value))}
            >
              <SelectTrigger
                size="sm"
                className="w-56"
                data-testid="voice-typing-mic-warm"
                aria-label={t("settings.voice.voiceTyping.micWarmLabel")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">
                  {t("settings.voice.voiceTyping.micWarmOff")}
                </SelectItem>
                <SelectItem value="15">
                  {t("settings.voice.voiceTyping.micWarmSeconds", {
                    seconds: 15,
                  })}
                </SelectItem>
                <SelectItem value="30">
                  {t("settings.voice.voiceTyping.micWarmSeconds", {
                    seconds: 30,
                  })}
                </SelectItem>
                <SelectItem value="60">
                  {t("settings.voice.voiceTyping.micWarmSeconds", {
                    seconds: 60,
                  })}
                </SelectItem>
                <SelectItem value={String(MIC_WARM_ALWAYS)}>
                  {t("settings.voice.voiceTyping.micWarmAlways")}
                </SelectItem>
              </SelectContent>
            </Select>
          </SettingsRow>

          <SettingsRow
            label={t("settings.voice.voiceTyping.muteAudioLabel")}
            description={t("settings.voice.voiceTyping.muteAudioDescription")}
          >
            <Switch
              checked={muteWhileRecording}
              onCheckedChange={setMuteWhileRecording}
              data-testid="voice-typing-mute-audio"
              aria-label={t("settings.voice.voiceTyping.muteAudioLabel")}
            />
          </SettingsRow>

          <SettingsGroupDivider />

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

          {pasteAutomatically ? (
            <>
              <SettingsRow
                label={t("settings.voice.voiceTyping.pasteMethodLabel")}
                description={t(
                  "settings.voice.voiceTyping.pasteMethodDescription",
                )}
              >
                <Select
                  value={pasteMethod}
                  onValueChange={(value) =>
                    setPasteMethod(value as VoiceTypingPasteMethod)
                  }
                >
                  <SelectTrigger
                    size="sm"
                    className="w-56"
                    data-testid="voice-typing-paste-method"
                    aria-label={t(
                      "settings.voice.voiceTyping.pasteMethodLabel",
                    )}
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VOICE_TYPING_PASTE_METHODS.map((method) => (
                      <SelectItem key={method} value={method}>
                        {t(PASTE_METHOD_LABEL[method])}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingsRow>

              <SettingsRow
                label={t("settings.voice.voiceTyping.submitLabel")}
                description={t("settings.voice.voiceTyping.submitDescription")}
              >
                <Switch
                  checked={submitAfterInsert}
                  onCheckedChange={setSubmitAfterInsert}
                  data-testid="voice-typing-submit"
                  aria-label={t("settings.voice.voiceTyping.submitLabel")}
                />
              </SettingsRow>
            </>
          ) : null}

          <SettingsRow
            label={t("settings.voice.voiceTyping.trailingSpaceLabel")}
            description={t(
              "settings.voice.voiceTyping.trailingSpaceDescription",
            )}
          >
            <Switch
              checked={appendTrailingSpace}
              onCheckedChange={setAppendTrailingSpace}
              data-testid="voice-typing-trailing-space"
              aria-label={t("settings.voice.voiceTyping.trailingSpaceLabel")}
            />
          </SettingsRow>

          {pasteAutomatically ? (
            <SettingsRow
              label={t("settings.voice.voiceTyping.copyTranscriptLabel")}
              description={t(
                "settings.voice.voiceTyping.copyTranscriptDescription",
              )}
            >
              <Switch
                checked={copyLastTranscript}
                onCheckedChange={setCopyLastTranscript}
                data-testid="voice-typing-copy-transcript"
                aria-label={t("settings.voice.voiceTyping.copyTranscriptLabel")}
              />
            </SettingsRow>
          ) : null}

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.fillerWordsLabel")}
            description={t("settings.voice.voiceTyping.fillerWordsDescription")}
          >
            <Switch
              checked={removeFillerWords}
              onCheckedChange={setRemoveFillerWords}
              data-testid="voice-typing-filler-words"
              aria-label={t("settings.voice.voiceTyping.fillerWordsLabel")}
            />
          </SettingsRow>

          <SettingsRow
            label={t("settings.voice.voiceTyping.spokenPunctuationLabel")}
            description={t(
              "settings.voice.voiceTyping.spokenPunctuationDescription",
            )}
          >
            <Switch
              checked={spokenPunctuation}
              onCheckedChange={setSpokenPunctuation}
              data-testid="voice-typing-spoken-punctuation"
              aria-label={t(
                "settings.voice.voiceTyping.spokenPunctuationLabel",
              )}
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

          {autoStopSilence ? (
            <SettingsRow
              label={t("settings.voice.voiceTyping.autoStopDelayLabel")}
              description={t(
                "settings.voice.voiceTyping.autoStopDelayDescription",
              )}
            >
              <Select
                value={String(autoStopDelayMs)}
                onValueChange={(value) => setAutoStopDelayMs(Number(value))}
              >
                <SelectTrigger
                  size="sm"
                  className="w-56"
                  data-testid="voice-typing-auto-stop-delay"
                  aria-label={t(
                    "settings.voice.voiceTyping.autoStopDelayLabel",
                  )}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1500">
                    {t("settings.voice.voiceTyping.seconds", {
                      seconds: "1.5",
                    })}
                  </SelectItem>
                  <SelectItem value="2500">
                    {t("settings.voice.voiceTyping.seconds", {
                      seconds: "2.5",
                    })}
                  </SelectItem>
                  <SelectItem value="4000">
                    {t("settings.voice.voiceTyping.seconds", { seconds: 4 })}
                  </SelectItem>
                  <SelectItem value="6000">
                    {t("settings.voice.voiceTyping.seconds", { seconds: 6 })}
                  </SelectItem>
                </SelectContent>
              </Select>
            </SettingsRow>
          ) : null}

          <SettingsGroupDivider />

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

          {soundEnabled ? (
            <>
              <SettingsRow
                label={t("settings.voice.voiceTyping.soundsThemeLabel")}
                description={t(
                  "settings.voice.voiceTyping.soundsThemeDescription",
                )}
              >
                <div className="flex items-center gap-2">
                  <Select
                    value={soundTheme}
                    onValueChange={(value) =>
                      setSoundTheme(value as VoiceTypingSoundTheme)
                    }
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-44"
                      data-testid="voice-typing-sound-theme"
                      aria-label={t(
                        "settings.voice.voiceTyping.soundsThemeLabel",
                      )}
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {VOICE_TYPING_SOUND_THEMES.map((theme) => (
                        <SelectItem key={theme} value={theme}>
                          {t(SOUND_THEME_LABEL[theme])}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="sm"
                    data-testid="voice-typing-sound-preview"
                    aria-label={t("settings.voice.voiceTyping.soundsPreview")}
                    onClick={() =>
                      previewVoiceTypingSound(soundTheme, soundVolume)
                    }
                  >
                    <HugeiconsIcon icon={Volume02Icon} className="size-3.5" />
                  </Button>
                </div>
              </SettingsRow>

              <SettingsRow
                label={t("settings.voice.voiceTyping.soundsVolumeLabel")}
                description={t(
                  "settings.voice.voiceTyping.soundsVolumeDescription",
                )}
              >
                <div className="flex items-center gap-3">
                  <Slider
                    value={[soundVolume]}
                    min={0}
                    max={1}
                    step={0.05}
                    onValueChange={([value]) => {
                      if (value !== undefined) setSoundVolume(value);
                    }}
                    className="w-44"
                    aria-label={t(
                      "settings.voice.voiceTyping.soundsVolumeLabel",
                    )}
                  />
                  <span className="w-9 text-right text-xs tabular-nums text-muted-foreground">
                    {Math.round(soundVolume * 100)}%
                  </span>
                </div>
              </SettingsRow>
            </>
          ) : null}

          <SettingsGroupDivider />

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

          <p className="pt-1 text-xs text-muted-foreground">
            {t("settings.voice.voiceTyping.engineNote")}
          </p>
        </>
      ) : null}
    </SettingsSection>
  );
}
