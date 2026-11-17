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
import { useId, useState } from "react";
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

/** Each voice mode behaviour, set or switched off here. */
export function VoiceModeSwitches() {
  return (
    <>
      <RolloutRow setting="clauseFirstChunk" />
      <TurnEndingRow />
      <DecisionModelRow />
      <ReplyInstructionRow />
      <RolloutRow setting="speechNormalize" />
      <RolloutRow setting="heardCut" />
    </>
  );
}
