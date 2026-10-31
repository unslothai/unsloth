// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings section for system-wide voice typing. Rendered only in the desktop app
// on platforms that can arm a global shortcut (see voice_typing_info).

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  buildExternalModelId,
  customProviderDisplayName,
  useChatRuntimeStore,
  useExternalProvidersStore,
} from "@/features/chat";
import { useHubInventory } from "@/features/hub";
import { resolveModelCatalogEntry } from "@/features/chat/model-catalog";
import {
  DEFAULT_LLM_POST_PROCESS_PROMPT,
  DEFAULT_VOICE_TYPING_SHORTCUT,
  VOICE_TYPING_HUD_POSITIONS,
  VOICE_TYPING_IDLE_STOP_OPTIONS,
  VOICE_TYPING_PASTE_METHODS,
  VOICE_TYPING_SOUND_THEMES,
  VOICE_TYPING_SUBMIT_KEYS,
  type VoiceTypingHudPosition,
  type VoiceTypingMode,
  type VoiceTypingPasteMethod,
  type VoiceTypingSoundTheme,
  type VoiceTypingSubmitKey,
  chordFromEvent,
  formatShortcut,
  isConflictingShortcut,
  isModifierCode,
  previewVoiceTypingSound,
  useVoiceTypingStore,
} from "@/features/voice-typing";
import { type TranslationKey, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import {
  CheckmarkCircle02Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { invoke } from "@tauri-apps/api/core";
import { useEffect, useMemo, useRef, useState } from "react";
import { PickerRow, PickerTrigger } from "./picker";
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

const SUBMIT_KEY_LABEL: Record<VoiceTypingSubmitKey, TranslationKey> = {
  none: "settings.voice.voiceTyping.submitKeyNone",
  enter: "settings.voice.voiceTyping.submitKeyEnter",
  ctrl_enter: "settings.voice.voiceTyping.submitKeyCtrlEnter",
  shift_enter: "settings.voice.voiceTyping.submitKeyShiftEnter",
};

const HUD_POSITION_LABEL: Record<VoiceTypingHudPosition, TranslationKey> = {
  bottom: "settings.voice.voiceTyping.hudPositionBottom",
  top: "settings.voice.voiceTyping.hudPositionTop",
};

/** Model families that are not chat LLMs, excluded from transcript refining. */
const NON_LLM_MODEL =
  /(whisper|transcribe|tts|dall-e|gpt-image|sora|veo\d|imagen|flux|stable-diffusion|embedding|embed-|rerank|moderation|audio|speech|realtime)/i;

interface ModelOption {
  id: string;
  label: string;
  group?: string;
}

/**
 * Refining-model picker. Reuses the dictation picker's trigger and row styles so
 * both model pickers look and behave identically, and only ever lists chat LLMs.
 */
function RefiningModelPicker({
  value,
  onChange,
  activeLabel,
  local,
  external,
}: {
  value: string;
  onChange: (value: string) => void;
  activeLabel: string;
  local: ModelOption[];
  external: ModelOption[];
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const matches = (option: ModelOption) =>
    !needle || option.label.toLowerCase().includes(needle);
  const filteredLocal = local.filter(matches);
  const filteredExternal = external.filter(matches);
  const activeOptionLabel = `${t("settings.voice.voiceTyping.llmModelActive")}${
    activeLabel ? ` (${activeLabel})` : ""
  }`;
  const selected = value
    ? (local.find((option) => option.id === value) ??
      external.find((option) => option.id === value))
    : null;
  const selectedLabel = value ? (selected?.label ?? value) : activeOptionLabel;
  const totalOptions = local.length + external.length;

  const choose = (next: string) => {
    onChange(next);
    setOpen(false);
    setQuery("");
  };

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setQuery("");
      }}
    >
      <PopoverTrigger asChild={true}>
        <PickerTrigger
          className="w-56"
          data-testid="voice-typing-llm-model"
          aria-label={t("settings.voice.voiceTyping.llmModelLabel")}
        >
          <span className="truncate" title={selectedLabel}>
            {selectedLabel}
          </span>
        </PickerTrigger>
      </PopoverTrigger>
      <PopoverContent align="start" sideOffset={4} className="w-72 gap-0 p-0">
        {totalOptions > 6 ? (
          <div className="relative p-1.5 pb-0.5">
            <HugeiconsIcon
              icon={Search01Icon}
              strokeWidth={2}
              className="text-muted-foreground pointer-events-none absolute top-[calc(50%+2px)] left-4 size-3.5 -translate-y-1/2"
            />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t("settings.voice.voiceTyping.llmModelSearch")}
              aria-label={t("settings.voice.voiceTyping.llmModelSearch")}
              className="h-8 pl-8 text-sm"
              autoFocus={true}
            />
          </div>
        ) : null}
        <div className="max-h-64 overflow-y-auto p-1">
          <PickerRow
            selected={!value}
            onClick={() => choose("")}
            className="text-xs"
          >
            <span className="truncate">{activeOptionLabel}</span>
            {!value ? (
              <HugeiconsIcon
                icon={CheckmarkCircle02Icon}
                className="size-3.5 shrink-0 text-primary"
              />
            ) : null}
          </PickerRow>

          {filteredLocal.length > 0 ? (
            <>
              <div className="px-3 pt-2 pb-1 text-ui-10 font-semibold tracking-wide text-muted-foreground uppercase">
                {t("settings.voice.voiceTyping.llmModelLocalGroup")}
              </div>
              {filteredLocal.map((option) => (
                <PickerRow
                  key={option.id}
                  selected={option.id === value}
                  onClick={() => choose(option.id)}
                >
                  <span className="truncate" title={option.label}>
                    {option.label}
                  </span>
                  {option.id === value ? (
                    <HugeiconsIcon
                      icon={CheckmarkCircle02Icon}
                      className="size-3.5 shrink-0 text-primary"
                    />
                  ) : null}
                </PickerRow>
              ))}
            </>
          ) : null}

          {filteredExternal.length > 0 ? (
            <>
              <div className="px-3 pt-2 pb-1 text-ui-10 font-semibold tracking-wide text-muted-foreground uppercase">
                {t("settings.voice.voiceTyping.llmModelConnectedGroup")}
              </div>
              {filteredExternal.map((option) => (
                <PickerRow
                  key={option.id}
                  selected={option.id === value}
                  onClick={() => choose(option.id)}
                >
                  <span className="min-w-0 truncate" title={option.label}>
                    <span className="text-muted-foreground">
                      {option.group}:{" "}
                    </span>
                    {option.label}
                  </span>
                  {option.id === value ? (
                    <HugeiconsIcon
                      icon={CheckmarkCircle02Icon}
                      className="size-3.5 shrink-0 text-primary"
                    />
                  ) : null}
                </PickerRow>
              ))}
            </>
          ) : null}

          {totalOptions === 0 ? (
            <div className="px-3 py-3 text-xs text-muted-foreground">
              {t("settings.voice.voiceTyping.llmModelNoLocal")}
            </div>
          ) : filteredLocal.length === 0 &&
            filteredExternal.length === 0 &&
            needle ? (
            <div className="px-3 py-3 text-xs text-muted-foreground">
              {t("settings.voice.voiceTyping.llmModelNoMatch")}
            </div>
          ) : null}
        </div>
      </PopoverContent>
    </Popover>
  );
}

type SoundThemeChoice = VoiceTypingSoundTheme | "off";

/**
 * One control for feedback sounds: Off plus each theme. Resting the pointer on
 * a theme for a second previews it; the active choice gets a tick. There is no
 * separate enable switch, preview button or volume control.
 */
function SoundThemePicker({
  value,
  onChange,
}: {
  value: SoundThemeChoice;
  onChange: (choice: SoundThemeChoice) => void;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const hoverTimer = useRef<number | null>(null);

  const clearHover = () => {
    if (hoverTimer.current !== null) {
      window.clearTimeout(hoverTimer.current);
      hoverTimer.current = null;
    }
  };

  useEffect(() => clearHover, []);

  const choices: SoundThemeChoice[] = ["off", ...VOICE_TYPING_SOUND_THEMES];

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) clearHover();
      }}
    >
      <PopoverTrigger asChild={true}>
        <PickerTrigger
          className="w-56"
          data-testid="voice-typing-sound-theme"
          aria-label={t("settings.voice.voiceTyping.soundsThemeLabel")}
        >
          <span className="truncate">
            {value === "off"
              ? t("settings.voice.voiceTyping.soundsOff")
              : t(SOUND_THEME_LABEL[value])}
          </span>
        </PickerTrigger>
      </PopoverTrigger>
      <PopoverContent align="start" sideOffset={4} className="w-56 p-1">
        {choices.map((choice) => (
          <PickerRow
            key={choice}
            selected={choice === value}
            className="py-2 text-sm"
            onClick={() => {
              onChange(choice);
              setOpen(false);
            }}
            onMouseEnter={() => {
              clearHover();
              if (choice === "off") return;
              // Preview only after the pointer rests on the row: scrubbing the
              // menu should not fire a burst of tones.
              hoverTimer.current = window.setTimeout(() => {
                hoverTimer.current = null;
                previewVoiceTypingSound(choice, 1);
              }, 1000);
            }}
            onMouseLeave={clearHover}
            onFocus={() => {
              if (choice !== "off") previewVoiceTypingSound(choice, 1);
            }}
          >
            <span className="truncate">
              {choice === "off"
                ? t("settings.voice.voiceTyping.soundsOff")
                : t(SOUND_THEME_LABEL[choice])}
            </span>
            {choice === value ? (
              <HugeiconsIcon
                icon={CheckmarkCircle02Icon}
                className="size-3.5 shrink-0 text-primary"
              />
            ) : null}
          </PickerRow>
        ))}
      </PopoverContent>
    </Popover>
  );
}

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
  const idleStopMs = useVoiceTypingStore((state) => state.idleStopMs);
  const setIdleStopMs = useVoiceTypingStore((state) => state.setIdleStopMs);
  const spokenPunctuation = useVoiceTypingStore(
    (state) => state.spokenPunctuation,
  );
  const setSpokenPunctuation = useVoiceTypingStore(
    (state) => state.setSpokenPunctuation,
  );
  const insertSubmitKey = useVoiceTypingStore(
    (state) => state.insertSubmitKey,
  );
  const setInsertSubmitKey = useVoiceTypingStore(
    (state) => state.setInsertSubmitKey,
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
      if (
        r.liveDownload ||
        r.partial ||
        r.companion ||
        r.audioType ||
        r.audioWorkflows?.length ||
        !r.capabilities?.canChat ||
        NON_LLM_MODEL.test(`${r.repoId} ${r.loadId}`)
      ) {
        continue;
      }
      const id = r.loadId || r.repoId;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      local.push({ id, label: r.repoId || id });
    }

    for (const r of hubInventory.localRows ?? []) {
      if (
        r.audioType ||
        !r.capabilities?.canChat ||
        NON_LLM_MODEL.test(`${r.repoId ?? ""} ${r.modelId ?? ""}`)
      ) {
        continue;
      }
      const id = r.modelId || r.path;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      local.push({ id, label: r.displayName || r.title || r.modelId || id });
    }

    const external: { id: string; label: string; group: string }[] = [];
    if (connectionsEnabled) {
      for (const provider of externalProviders) {
        if (provider.decisionsOnly) continue;
        const groupLabel =
          provider.name || customProviderDisplayName(provider.providerType);
        for (const modelId of provider.models ?? []) {
          if (!modelId) continue;
          // Only chat LLMs: skip audio/video/image/embedding endpoints.
          const catalog = resolveModelCatalogEntry(provider.providerType, modelId);
          if (catalog?.inputModalities && !catalog.inputModalities.includes("text")) {
            continue;
          }
          if (NON_LLM_MODEL.test(modelId)) continue;
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
            label={t("settings.voice.voiceTyping.grammarFixerLabel")}
            description={t(
              "settings.voice.voiceTyping.grammarFixerDescription",
            )}
          >
            <Switch
              checked={grammarFixerEnabled}
              onCheckedChange={setGrammarFixerEnabled}
              data-testid="voice-typing-grammar-fixer"
              aria-label={t("settings.voice.voiceTyping.grammarFixerLabel")}
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
                  !llmPostProcessModel && !activeChatModelLabel
                    ? "settings.voice.voiceTyping.llmModelDescriptionNoActive"
                    : "settings.voice.voiceTyping.llmModelDescription",
                )}
              >
                <RefiningModelPicker
                  value={llmPostProcessModel}
                  onChange={setLlmPostProcessModel}
                  activeLabel={activeChatModelLabel}
                  local={llmModelOptions.local}
                  external={llmModelOptions.external}
                />
              </SettingsRow>

              <SettingsRow
                label={t("settings.voice.voiceTyping.llmThinkingLabel")}
                description={t(
                  "settings.voice.voiceTyping.llmThinkingDescription",
                )}
              >
                <Switch
                  checked={llmThinkingEnabled}
                  onCheckedChange={setLlmThinkingEnabled}
                  data-testid="voice-typing-llm-thinking"
                  aria-label={t("settings.voice.voiceTyping.llmThinkingLabel")}
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
                    <Select
                      value={insertSubmitKey}
                      onValueChange={(value) =>
                        setInsertSubmitKey(value as VoiceTypingSubmitKey)
                      }
                    >
                      <SelectTrigger
                        size="sm"
                        className="w-56"
                        data-testid="voice-typing-submit"
                        aria-label={t(
                          "settings.voice.voiceTyping.submitLabel",
                        )}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {VOICE_TYPING_SUBMIT_KEYS.map((key) => (
                          <SelectItem key={key} value={key}>
                            {t(SUBMIT_KEY_LABEL[key])}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
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
                label={t("settings.voice.voiceTyping.idleStopLabel")}
                description={t(
                  "settings.voice.voiceTyping.idleStopDescription",
                )}
              >
                <Select
                  value={String(idleStopMs)}
                  onValueChange={(value) => setIdleStopMs(Number(value))}
                >
                  <SelectTrigger
                    size="sm"
                    className="w-56"
                    data-testid="voice-typing-idle-stop"
                    aria-label={t(
                      "settings.voice.voiceTyping.idleStopLabel",
                    )}
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VOICE_TYPING_IDLE_STOP_OPTIONS.map((ms) => (
                      <SelectItem key={ms} value={String(ms)}>
                        {ms === 0
                          ? t("settings.voice.voiceTyping.idleStopNever")
                          : ms < 60_000
                            ? t("settings.voice.voiceTyping.idleStopSeconds", {
                                seconds: ms / 1000,
                              })
                            : t("settings.voice.voiceTyping.idleStopMinutes", {
                                minutes: ms / 60_000,
                              })}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingsRow>

              <SettingsRow
                label={t("settings.voice.voiceTyping.soundsThemeLabel")}
                description={t(
                  "settings.voice.voiceTyping.soundsThemeDescription",
                )}
              >
                <SoundThemePicker
                  value={soundEnabled ? soundTheme : "off"}
                  onChange={(choice) => {
                    if (choice === "off") {
                      setSoundEnabled(false);
                      return;
                    }
                    setSoundTheme(choice);
                    setSoundEnabled(true);
                  }}
                />
              </SettingsRow>
        </>
      ) : null}
    </SettingsSection>
  );
}
