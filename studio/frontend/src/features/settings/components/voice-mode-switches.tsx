// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Spinner } from "@/components/ui/spinner";
import { setVoiceDecision, useVoiceDecision } from "@/features/chat";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, useId, useState } from "react";
import { VoiceSoundControls } from "@/components/assistant-ui/voice-sound-controls";
import { useVoicePreset } from "@/components/assistant-ui/voice-model-selector";
import { voiceSpeakerLabel } from "@/features/chat/voice/use-voice-model-spec";
import { useAudioVoicesStore } from "@/features/audio/stores/audio-voices-store";
import { useChatRuntimeStore } from "@/features/chat";
import { isChatModelVoice } from "@/features/chat/voice/voice-models";
import { audioModelLabel } from "../../audio/audio-workspace-utils";
import {
  TURN_JUDGE_BAR,
  VOICE_CUSTOM_INSTRUCTION_MAX,
  VOICE_REPLY_INSTRUCTIONS,
  VOICE_TURN_ENDINGS,
  type VoiceReplyInstruction,
  type VoiceRolloutSetting,
  type VoiceTurnEnding,
} from "../stores/stt-model-catalog";
import { useVoiceSettingsStore } from "../stores/voice-settings-store";
import { SettingsRow } from "./settings-row";

function RolloutRow({ setting }: { setting: VoiceRolloutSetting }) {
  const t = useT();
  const on = useVoiceSettingsStore((s) => s[setting]);
  const setVoiceRollout = useVoiceSettingsStore((s) => s.setVoiceRollout);
  const label = t(`settings.voice.voiceMode.switches.${setting}.label`);
  return (
    <SettingsRow label={label} description={t(`settings.voice.voiceMode.switches.${setting}.description`)}>
      <Switch
        checked={on}
        onCheckedChange={(value) => setVoiceRollout(setting, value)}
        aria-label={label}
        data-testid={`voice-switch-${setting}`}
      />
    </SettingsRow>
  );
}

function TurnEndingRow() {
  const t = useT();
  const ending = useVoiceSettingsStore((s) => s.turnEnding);
  const setEnding = useVoiceSettingsStore((s) => s.setTurnEnding);
  const turnJudgeBar = useVoiceSettingsStore((s) => s.turnJudgeBar);
  const setTurnJudgeBar = useVoiceSettingsStore((s) => s.setTurnJudgeBar);
  const label = t("settings.voice.voiceMode.turnEnding.label");
  const barLabel = t("settings.voice.voiceMode.turnEnding.judgeBar.label");
  return (
    <>
      <SettingsRow
        label={label}
        description={t(`settings.voice.voiceMode.turnEnding.options.${ending}.description`)}
      >
        <Select value={ending} onValueChange={(value) => setEnding(value as VoiceTurnEnding)}>
          <SelectTrigger aria-label={label} className="w-56" data-testid="voice-turn-ending">
            <SelectValue />
          </SelectTrigger>
          <SelectContent align="end">
            {VOICE_TURN_ENDINGS.map((option) => (
              <SelectItem key={option} value={option}>
                {t(`settings.voice.voiceMode.turnEnding.options.${option}.label`)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SettingsRow>
      {ending === "smartTurnJudge" ? (
        <SettingsRow label={barLabel} description={t("settings.voice.voiceMode.turnEnding.judgeBar.description")}>
          <div className="flex w-56 items-center gap-3">
            <Slider
              value={[turnJudgeBar]}
              min={TURN_JUDGE_BAR.min}
              max={TURN_JUDGE_BAR.max}
              step={TURN_JUDGE_BAR.step}
              onValueChange={(values: number[]) => setTurnJudgeBar(values[0])}
              aria-label={barLabel}
              data-testid="voice-turn-judge-bar"
            />
            <span className="w-10 text-right text-sm tabular-nums text-muted-foreground">
              {Math.round(turnJudgeBar * 100)}%
            </span>
          </div>
        </SettingsRow>
      ) : null}
    </>
  );
}

const NO_DECISION_MODEL = "__none__";

// The decision model voice mode asks; installation-wide, so only the owner can change it.
function DecisionModelRow() {
  const t = useT();
  const { status, state, switched, failed } = useVoiceDecision(null);
  const [saving, setSaving] = useState(false);
  const label = t("settings.voice.voiceMode.decisionModel.label");
  // No pick of your own ("None") still decides with the launch default or the built-in classifiers when the server
  // has them: the option says which, instead of promising voice mode runs without one.
  const fallback = status?.launch_default ?? null;
  const current = !status || status.source !== "setting" ? NO_DECISION_MODEL : (status.current ?? NO_DECISION_MODEL);
  const noneLabel = fallback
    ? (status?.models.find((model) => model.name === fallback)?.label ?? fallback)
    : status?.builtin_configured
      ? t("voicePanel.builtinClassifiers")
      : t("settings.voice.voiceMode.decisionModel.none");
  const decidesWithoutPick = Boolean(fallback || status?.builtin_configured);
  const pick = (value: string) => {
    setSaving(true);
    setVoiceDecision(value === NO_DECISION_MODEL ? null : value)
      .then(switched)
      .catch((error: unknown) =>
        toast.error(t("voicePanel.decideSwitchFailed"), {
          description: error instanceof Error ? error.message : undefined,
        }),
      )
      .finally(() => setSaving(false));
  };
  const description = failed
    ? t("voicePanel.decideOwnerOnly")
    : state === "error" && status?.error
      ? status.error
      : t(
          current === NO_DECISION_MODEL && !decidesWithoutPick
            ? "settings.voice.voiceMode.decisionModel.noneDescription"
            : "settings.voice.voiceMode.decisionModel.description",
        );
  return (
    <SettingsRow label={label} description={description}>
      <div className="flex items-center gap-2">
        {state === "loading" || saving ? (
          <Spinner className="size-3.5" aria-label={t("settings.voice.voiceMode.decisionModel.loading")} />
        ) : null}
        <Select value={current} onValueChange={pick} disabled={failed || !status || saving}>
          <SelectTrigger aria-label={label} className="w-56" data-testid="voice-decision-model">
            <SelectValue />
          </SelectTrigger>
          <SelectContent align="end">
            {/* Picked and also the launch default: clearing the pick changes nothing, so no second row says the same. */}
            {status?.source === "setting" && fallback !== null && status.current === fallback ? null : (
              <SelectItem value={NO_DECISION_MODEL}>{noneLabel}</SelectItem>
            )}
            {/* The launch default is the None row only while nothing is picked; picked, it is its own row. */}
            {(status?.models ?? []).filter((model) => status?.source === "setting" || model.name !== fallback).map((model) => (
              <SelectItem key={model.name} value={model.name} disabled={!model.available}>
                {model.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </SettingsRow>
  );
}

function ReplyInstructionRow() {
  const t = useT();
  const choice = useVoiceSettingsStore((s) => s.replyInstruction);
  const setChoice = useVoiceSettingsStore((s) => s.setReplyInstruction);
  const custom = useVoiceSettingsStore((s) => s.replyInstructionCustom);
  const setCustom = useVoiceSettingsStore((s) => s.setReplyInstructionCustom);
  const label = t("settings.voice.voiceMode.replyInstruction.label");
  const customLabel = t("settings.voice.voiceMode.replyInstruction.custom.label");
  const customId = useId();
  return (
    <>
      <SettingsRow
        label={label}
        description={t(`settings.voice.voiceMode.replyInstruction.options.${choice}.description`)}
      >
        <Select value={choice} onValueChange={(value) => setChoice(value as VoiceReplyInstruction)}>
          <SelectTrigger aria-label={label} className="w-56" data-testid="voice-reply-instruction">
            <SelectValue />
          </SelectTrigger>
          <SelectContent align="end">
            {VOICE_REPLY_INSTRUCTIONS.map((option) => (
              <SelectItem key={option} value={option}>
                {t(`settings.voice.voiceMode.replyInstruction.options.${option}.label`)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SettingsRow>
      {choice === "custom" ? (
        <div className="grid gap-1.5 pb-2">
          <label htmlFor={customId} className="text-sm font-medium">
            {customLabel}
          </label>
          <Textarea
            id={customId}
            rows={4}
            className="resize-y text-sm"
            placeholder={t("settings.voice.voiceMode.replyInstruction.custom.placeholder")}
            value={custom}
            maxLength={VOICE_CUSTOM_INSTRUCTION_MAX}
            onChange={(event) => setCustom(event.target.value)}
            data-testid="voice-reply-instruction-custom"
          />
          <p className="text-xs text-muted-foreground" aria-live="polite">
            {custom.trim()
              ? t("settings.voice.voiceMode.replyInstruction.custom.hint", {
                  n: custom.length,
                  max: VOICE_CUSTOM_INSTRUCTION_MAX,
                })
              : t("settings.voice.voiceMode.replyInstruction.custom.empty")}
          </p>
        </div>
      ) : null}
    </>
  );
}

const MODEL_DEFAULT_VOICE = "__default__";
const SAVED_PREFIX = "saved:";
const SPEAKER_PREFIX = "speaker:";

// The voice mode speech model's voice and options, set here or in the voice panel before it loads.
function SpeakingVoiceRow() {
  const t = useT();
  const modelId = useChatRuntimeStore((s) => s.selectedVoiceModelId);
  const loaded = useChatRuntimeStore((s) => s.voiceSlotLoaded);
  const savedVoiceId = useChatRuntimeStore((s) => s.selectedVoiceId);
  const speaker = useChatRuntimeStore((s) => s.selectedVoiceSpeaker);
  const savedVoices = useAudioVoicesStore((s) => s.voices);
  const preset = useVoicePreset(modelId, loaded);
  const label = t("settings.voice.voiceMode.voice.label");
  useEffect(() => {
    if (preset.canClone) void useAudioVoicesStore.getState().refresh();
  }, [preset.canClone]);
  if (!modelId) return null;
  const chatVoice = isChatModelVoice(modelId);
  // Named as the voice panel names it (no "-GGUF"), so both places read the same.
  const modelName = chatVoice
    ? t("voicePanel.chatModelVoice")
    : audioModelLabel(modelId).replace(/-GGUF$/i, "");
  const speakers = preset.speakers;
  const builtins = speakers && speakers.values.length > 1 ? speakers.values : [];
  const saved = preset.canClone ? savedVoices : [];
  const heard =
    savedVoiceId && saved.some((v) => v.id === savedVoiceId)
      ? `${SAVED_PREFIX}${savedVoiceId}`
      : speaker && builtins.includes(speaker)
        ? `${SPEAKER_PREFIX}${speaker}`
        : MODEL_DEFAULT_VOICE;
  const defaultName =
    speakers?.fallback && builtins.includes(speakers.fallback)
      ? voiceSpeakerLabel(speakers.fallback, speakers.family, t)
      : null;
  const choose = (value: string) => {
    const store = useChatRuntimeStore.getState();
    if (value.startsWith(SAVED_PREFIX)) {
      store.setSelectedVoiceId(value.slice(SAVED_PREFIX.length));
    } else {
      store.setSelectedVoiceId("");
      store.setSelectedVoiceSpeaker(value.startsWith(SPEAKER_PREFIX) ? value.slice(SPEAKER_PREFIX.length) : "");
    }
  };
  const hasChoices = builtins.length > 0 || saved.length > 0;
  return (
    <SettingsRow
      label={label}
      description={t("settings.voice.voiceMode.voice.description", { model: modelName })}
      below={
        chatVoice ? null : (
          // Nested under the voice it tunes, so it does not read as Read aloud's speed above.
          <div className="w-full">
            <VoiceSoundControls
              modelId={modelId}
              spec={preset.spec}
              className="max-w-[calc(30rem*var(--ui-space-scale,1))] border-l-2 border-border/60 pl-4 pr-0"
            />
          </div>
        )
      }
    >
      {hasChoices ? (
        <Select value={heard} onValueChange={choose}>
          <SelectTrigger aria-label={label} className="min-w-56 max-w-72" size="sm" data-testid="voice-speaking-voice">
            <SelectValue />
          </SelectTrigger>
          <SelectContent align="end" className="max-h-[min(--spacing(72),var(--radix-select-content-available-height))]">
            <SelectItem value={MODEL_DEFAULT_VOICE}>
              {defaultName
                ? t("settings.voice.voiceMode.voice.modelDefaultNamed", { name: defaultName })
                : t("voicePanel.modelsOwnVoice")}
            </SelectItem>
            {builtins
              .filter((choice) => choice !== speakers?.fallback)
              .map((choice) => (
                <SelectItem key={choice} value={`${SPEAKER_PREFIX}${choice}`}>
                  {voiceSpeakerLabel(choice, speakers?.family ?? null, t)}
                </SelectItem>
              ))}
            {saved.map((voice) => (
              <SelectItem key={voice.id} value={`${SAVED_PREFIX}${voice.id}`}>
                {voice.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : preset.spec.state === "loading" ? (
        <Spinner className="size-4 text-muted-foreground" />
      ) : (
        <span className="text-sm text-muted-foreground">{preset.label ?? t("voicePanel.modelsOwnVoice")}</span>
      )}
    </SettingsRow>
  );
}

/** Each voice mode behaviour, set or switched off here. */
export function VoiceModeSwitches() {
  return (
    <>
      <SpeakingVoiceRow />
      <RolloutRow setting="clauseFirstChunk" />
      <TurnEndingRow />
      <DecisionModelRow />
      <ReplyInstructionRow />
      <RolloutRow setting="speechNormalize" />
      <RolloutRow setting="heardCut" />
    </>
  );
}
