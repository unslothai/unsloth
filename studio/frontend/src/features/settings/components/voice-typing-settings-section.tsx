// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings section for system-wide voice typing. Rendered only in the desktop app
// on platforms that can arm a global shortcut (see voice_typing_info).

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  buildExternalModelId,
  customProviderDisplayName,
  useChatRuntimeStore,
  useExternalProvidersStore,
} from "@/features/chat";
import { useHubInventory } from "@/features/hub";
import {
  DEFAULT_LLM_POST_PROCESS_PROMPT,
  DEFAULT_VOICE_TYPING_SHORTCUT,
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
  useVoiceTypingStore,
} from "@/features/voice-typing";
import { type TranslationKey, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Volume02Icon } from "@/lib/volume-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { invoke } from "@tauri-apps/api/core";
import { useEffect, useMemo, useRef, useState } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

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
  const grammarFixerEnabled = useVoiceTypingStore(
    (state) => state.grammarFixerEnabled,
  );
  const setGrammarFixerEnabled = useVoiceTypingStore(
    (state) => state.setGrammarFixerEnabled,
  );
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
  const autoStopSilence = useVoiceTypingStore((state) => state.autoStopSilence);
  const setAutoStopSilence = useVoiceTypingStore(
    (state) => state.setAutoStopSilence,
  );
  const autoStopDelayMs = useVoiceTypingStore((state) => state.autoStopDelayMs);
  const setAutoStopDelayMs = useVoiceTypingStore(
    (state) => state.setAutoStopDelayMs,
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
  const llmPostProcessEnabled = useVoiceTypingStore(
    (state) => state.llmPostProcessEnabled,
  );
  const setLlmPostProcessEnabled = useVoiceTypingStore(
    (state) => state.setLlmPostProcessEnabled,
  );
  const llmThinkingEnabled = useVoiceTypingStore(
    (state) => state.llmThinkingEnabled,
  );
  const setLlmThinkingEnabled = useVoiceTypingStore(
    (state) => state.setLlmThinkingEnabled,
  );
  const llmPostProcessModel = useVoiceTypingStore(
    (state) => state.llmPostProcessModel,
  );
  const setLlmPostProcessModel = useVoiceTypingStore(
    (state) => state.setLlmPostProcessModel,
  );
  const llmPostProcessPrompt = useVoiceTypingStore(
    (state) => state.llmPostProcessPrompt,
  );
  const setLlmPostProcessPrompt = useVoiceTypingStore(
    (state) => state.setLlmPostProcessPrompt,
  );
  const resetLlmPostProcessPrompt = useVoiceTypingStore(
    (state) => state.resetLlmPostProcessPrompt,
  );

  const [showAdvanced, setShowAdvanced] = useState(false);

  const residentCheckpoint = useChatRuntimeStore(
    (state) => state.residentCheckpoint,
  );
  const checkpoint = useChatRuntimeStore((state) => state.params.checkpoint);
  const externalProviders = useExternalProvidersStore(
    (state) => state.providers,
  );
  const connectionsEnabled = useExternalProvidersStore(
    (state) => state.connectionsEnabled,
  );

  const activeChatModelLabel = useMemo(() => {
    return (residentCheckpoint || checkpoint || "").trim();
  }, [residentCheckpoint, checkpoint]);

  const hubInventory = useHubInventory();

  const llmModelOptions = useMemo(() => {
    const local: { id: string; label: string }[] = [];
    const seen = new Set<string>();

    for (const r of hubInventory.cachedRows ?? []) {
      if (r.liveDownload || r.isAudio) continue;
      const id = r.loadId || r.repoId;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      local.push({ id, label: r.repoId || id });
    }

    for (const r of hubInventory.localRows ?? []) {
      if (r.isAudio || !r.capabilities?.canChat) continue;
      const id = r.modelId || r.path;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      local.push({ id, label: r.name || r.modelId || id });
    }

    const external: { id: string; label: string; group: string }[] = [];
    if (connectionsEnabled) {
      for (const provider of externalProviders) {
        if (provider.decisionsOnly) continue;
        const groupLabel =
          provider.name || customProviderDisplayName(provider.providerType);
        for (const modelId of provider.models ?? []) {
          if (!modelId) continue;
          external.push({
            id: buildExternalModelId(provider.id, modelId),
            label: modelId,
            group: groupLabel,
          });
        }
      }
    }
    return { local, external };
  }, [hubInventory.cachedRows, hubInventory.localRows, externalProviders, connectionsEnabled]);

  const [info, setInfo] = useState<VoiceTypingInfo | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [needsModifier, setNeedsModifier] = useState(false);
  const [draftVolume, setDraftVolume] = useState<number | null>(null);
  const shortcutButtonRef = useRef<HTMLButtonElement | null>(null);
  const effectiveVolume = draftVolume ?? soundVolume;

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

  const shortcutHasConflict = isConflictingShortcut(shortcut);
  const promptModified =
    llmPostProcessPrompt.trim() !== DEFAULT_LLM_POST_PROCESS_PROMPT.trim();

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
            label="Advanced Grammar Fixer"
            description="Ultra-fast on-device correction for casing, repeated words, a/an articles, contractions, and punctuation"
          >
            <Switch
              checked={grammarFixerEnabled}
              onCheckedChange={setGrammarFixerEnabled}
              data-testid="voice-typing-grammar-fixer"
              aria-label="Advanced Grammar Fixer"
            />
          </SettingsRow>

          <SettingsRow
            label={t("settings.voice.voiceTyping.llmPostProcessLabel")}
            description={t(
              "settings.voice.voiceTyping.llmPostProcessDescription",
            )}
          >
            <Switch
              checked={llmPostProcessEnabled}
              onCheckedChange={setLlmPostProcessEnabled}
              data-testid="voice-typing-llm-post-process"
              aria-label={t("settings.voice.voiceTyping.llmPostProcessLabel")}
            />
          </SettingsRow>

          {llmPostProcessEnabled ? (
            <>
              <SettingsRow
                label={t("settings.voice.voiceTyping.llmModelLabel")}
                description={t(
                  "settings.voice.voiceTyping.llmModelDescription",
                )}
              >
                <Select
                  value={llmPostProcessModel || "__active__"}
                  onValueChange={(value) =>
                    setLlmPostProcessModel(value === "__active__" ? "" : value)
                  }
                >
                  <SelectTrigger
                    size="sm"
                    className="w-56"
                    data-testid="voice-typing-llm-model"
                    aria-label={t("settings.voice.voiceTyping.llmModelLabel")}
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__active__">
                      {activeChatModelLabel
                        ? `${t("settings.voice.voiceTyping.llmModelActive")} (${activeChatModelLabel})`
                        : t("settings.voice.voiceTyping.llmModelActive")}
                    </SelectItem>
                    {llmModelOptions.local.length > 0 ? (
                      <SelectGroup>
                        <SelectLabel>
                          {t("settings.voice.voiceTyping.llmModelLocalGroup")}
                        </SelectLabel>
                        {llmModelOptions.local.map((m) => (
                          <SelectItem key={m.id} value={m.id}>
                            {m.label}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    ) : (
                      <SelectItem value="__no_local__" disabled className="text-xs text-muted-foreground italic">
                        No downloaded models found on this device
                      </SelectItem>
                    )}
                    {llmModelOptions.external.length > 0 ? (
                      <SelectGroup>
                        <SelectLabel>
                          {t(
                            "settings.voice.voiceTyping.llmModelConnectedGroup",
                          )}
                        </SelectLabel>
                        {llmModelOptions.external.map((m) => (
                          <SelectItem key={m.id} value={m.id}>
                            {m.group}: {m.label}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    ) : null}
                  </SelectContent>
                </Select>
              </SettingsRow>

              <SettingsRow
                label="Model Thinking"
                description="Allow model reasoning tokens (disabled by default for lightning-fast dictation)"
              >
                <Switch
                  checked={llmThinkingEnabled}
                  onCheckedChange={setLlmThinkingEnabled}
                  data-testid="voice-typing-llm-thinking"
                  aria-label="Model Thinking"
                />
              </SettingsRow>

              <div className="flex flex-col gap-2 py-2.5">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <div className="text-sm font-medium text-foreground">
                      {t("settings.voice.voiceTyping.llmPromptLabel")}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {t("settings.voice.voiceTyping.llmPromptDescription")}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={!promptModified}
                    data-testid="voice-typing-llm-prompt-reset"
                    onClick={resetLlmPostProcessPrompt}
                  >
                    {t("settings.voice.voiceTyping.llmPromptReset")}
                  </Button>
                </div>
                <Textarea
                  value={llmPostProcessPrompt}
                  onChange={(event) =>
                    setLlmPostProcessPrompt(event.target.value)
                  }
                  rows={6}
                  className="font-mono text-xs leading-relaxed"
                  data-testid="voice-typing-llm-prompt"
                  aria-label={t("settings.voice.voiceTyping.llmPromptLabel")}
                />
              </div>
            </>
          ) : null}

          {/* Collapsible Advanced Settings */}
          <Collapsible
            open={showAdvanced}
            onOpenChange={setShowAdvanced}
            className="w-full pt-2"
          >
            <div className="flex items-center justify-between border-t border-border/40 py-2.5">
              <span className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
                More / Advanced Settings
              </span>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="h-7 text-xs">
                  {showAdvanced ? "Hide Advanced" : "Show Advanced"}
                  <HugeiconsIcon
                    icon={ChevronDownStandardIcon}
                    className={`ml-1 size-3.5 transition-transform duration-200 ${
                      showAdvanced ? "rotate-180" : ""
                    }`}
                  />
                </Button>
              </CollapsibleTrigger>
            </div>

            <CollapsibleContent className="flex flex-col gap-0">
              <SettingsRow
                label={t("settings.voice.voiceTyping.hudPositionLabel")}
                description={t(
                  "settings.voice.voiceTyping.hudPositionDescription",
                )}
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
                    aria-label={t(
                      "settings.voice.voiceTyping.hudPositionLabel",
                    )}
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

              <SettingsRow
                label={t("settings.voice.voiceTyping.muteAudioLabel")}
                description={t(
                  "settings.voice.voiceTyping.muteAudioDescription",
                )}
              >
                <Switch
                  checked={muteWhileRecording}
                  onCheckedChange={setMuteWhileRecording}
                  data-testid="voice-typing-mute-audio"
                  aria-label={t("settings.voice.voiceTyping.muteAudioLabel")}
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
                    description={t(
                      "settings.voice.voiceTyping.submitDescription",
                    )}
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
                  aria-label={t(
                    "settings.voice.voiceTyping.trailingSpaceLabel",
                  )}
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
                    aria-label={t(
                      "settings.voice.voiceTyping.copyTranscriptLabel",
                    )}
                  />
                </SettingsRow>
              ) : null}

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
                description={t(
                  "settings.voice.voiceTyping.autoStopDescription",
                )}
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
                        {t("settings.voice.voiceTyping.seconds", {
                          seconds: 4,
                        })}
                      </SelectItem>
                      <SelectItem value="6000">
                        {t("settings.voice.voiceTyping.seconds", {
                          seconds: 6,
                        })}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </SettingsRow>
              ) : null}

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
                            <SelectItem
                              key={theme}
                              value={theme}
                              className="group flex w-full items-center justify-between pr-2"
                              onMouseEnter={() =>
                                previewVoiceTypingSound(theme, effectiveVolume)
                              }
                            >
                              <span>{t(SOUND_THEME_LABEL[theme])}</span>
                              <span
                                role="button"
                                tabIndex={0}
                                className="ml-2 inline-flex size-5 shrink-0 items-center justify-center rounded text-muted-foreground transition-opacity hover:text-foreground opacity-60 hover:opacity-100"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  e.preventDefault();
                                  previewVoiceTypingSound(
                                    theme,
                                    effectiveVolume,
                                  );
                                }}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter" || e.key === " ") {
                                    e.stopPropagation();
                                    e.preventDefault();
                                    previewVoiceTypingSound(
                                      theme,
                                      effectiveVolume,
                                    );
                                  }
                                }}
                                title="Play sound preview"
                              >
                                <HugeiconsIcon
                                  icon={Volume02Icon}
                                  className="size-3"
                                />
                              </span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Button
                        variant="outline"
                        size="icon-sm"
                        className="size-8 aspect-square shrink-0 rounded-md p-0"
                        data-testid="voice-typing-sound-preview"
                        aria-label={t(
                          "settings.voice.voiceTyping.soundsPreview",
                        )}
                        onClick={() =>
                          previewVoiceTypingSound(soundTheme, effectiveVolume)
                        }
                      >
                        <HugeiconsIcon
                          icon={Volume02Icon}
                          className="size-3.5"
                        />
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
                        value={[effectiveVolume]}
                        min={0}
                        max={1}
                        step={0.05}
                        onValueChange={([value]) => {
                          if (value !== undefined) {
                            const snapped = Math.round(value * 20) / 20;
                            setDraftVolume(snapped);
                          }
                        }}
                        onValueCommit={([value]) => {
                          if (value !== undefined) {
                            const snapped = Math.round(value * 20) / 20;
                            setSoundVolume(snapped);
                          }
                          setDraftVolume(null);
                        }}
                        className="w-44"
                        aria-label={t(
                          "settings.voice.voiceTyping.soundsVolumeLabel",
                        )}
                      />
                      <span className="w-9 text-right text-xs tabular-nums text-muted-foreground">
                        {Math.round(effectiveVolume * 100)}%
                      </span>
                    </div>
                  </SettingsRow>
                </>
              ) : null}
            </CollapsibleContent>
          </Collapsible>
        </>
      ) : null}
    </SettingsSection>
  );
}
