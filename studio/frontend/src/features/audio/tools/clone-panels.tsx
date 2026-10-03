// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/features/chat";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import type { ReactNode } from "react";
import { INDEX_TTS2_EMOTIONS } from "../clone-policy";
import { AudioSourceInput } from "../components/audio-source-input";
import { Field } from "../components/field";
import {
  type CosyVoiceMode,
  type CosyVoiceModeValue,
  type EmotionMode,
  type EmotionValue,
  type ExpressivenessValue,
  F5_DIALECTS,
  type SpeedDialectValue,
  type TimbreOnlyValue,
  chatterboxExpressivenessLogic,
  cosyVoiceModeLogic,
  f5SpeedDialectLogic,
  indexTts2EmotionLogic,
  qwen3TimbreLogic,
} from "./panel-logic";
import type { AudioToolPanel } from "./types";

export function PanelSection({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <section className="grid gap-3" aria-label={title}>
      <div className="grid gap-0.5">
        <h3 className="text-ui-13 font-medium text-foreground">{title}</h3>
        {hint ? (
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            {hint}
          </p>
        ) : null}
      </div>
      {children}
    </section>
  );
}

const qwen3TimbrePanel: AudioToolPanel<TimbreOnlyValue> = {
  ...qwen3TimbreLogic,
  Component: ({ value, onChange, disabled }) => (
    <div className="grid gap-1.5">
      <label
        htmlFor="clone-timbre-only"
        className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
      >
        Timbre only
        <Switch
          id="clone-timbre-only"
          checked={value.timbreOnly}
          disabled={disabled}
          onCheckedChange={(timbreOnly) => onChange({ timbreOnly })}
        />
      </label>
      <p className="text-ui-11p5 leading-snug text-muted-foreground">
        Copies the sound of the voice without the clip's words, so no transcript
        is needed. Usually a little less close to the original.
      </p>
    </div>
  ),
};

const indexTts2EmotionPanel: AudioToolPanel<EmotionValue> = {
  ...indexTts2EmotionLogic,
  Component: ({ value, onChange, disabled }) => (
    <PanelSection
      title="Emotion"
      hint="Optional. Leave it empty for the reference's own delivery."
    >
      <PillTabs
        ariaLabel="Emotion source"
        value={value.mode}
        onValueChange={(mode) =>
          onChange({ ...value, mode: mode as EmotionMode })
        }
        disabled={disabled}
        fit={true}
        compact={true}
        className="[&>button]:px-3"
        tabs={[
          { value: "text", label: "From text" },
          { value: "audio", label: "From audio" },
          { value: "mixer", label: "Mixer" },
        ]}
      />
      {value.mode === "text" ? (
        <Field
          label="Describe the emotion"
          htmlFor="clone-emotion-text"
          hint="A few words, such as “excited and a little nervous”."
        >
          <Textarea
            id="clone-emotion-text"
            value={value.text}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...value, text: event.target.value })
            }
            className="min-h-16"
          />
        </Field>
      ) : value.mode === "audio" ? (
        <AudioSourceInput
          id="clone-emotion-audio"
          label="Emotion clip"
          hint="Its mood is copied, not its voice."
          value={value.source}
          onChange={(source) => onChange({ ...value, source })}
          disabled={disabled}
          allowSavedVoice={false}
        />
      ) : (
        <div className="grid gap-2">
          {INDEX_TTS2_EMOTIONS.map((emotion, index) => (
            <ParamSlider
              key={emotion.key}
              label={emotion.label}
              value={value.vector[index] ?? 0}
              min={0}
              max={1}
              step={0.05}
              disabled={disabled}
              inline={true}
              onChange={(next) => {
                const vector = INDEX_TTS2_EMOTIONS.map((_, i) =>
                  i === index ? next : (value.vector[i] ?? 0),
                );
                onChange({ ...value, vector });
              }}
            />
          ))}
        </div>
      )}
      <ParamSlider
        label="Strength"
        value={value.alpha}
        min={0}
        max={1}
        step={0.05}
        disabled={disabled}
        info="How strongly the emotion colours the voice."
        onChange={(alpha) => onChange({ ...value, alpha })}
      />
    </PanelSection>
  ),
};

const chatterboxExpressivenessPanel: AudioToolPanel<ExpressivenessValue> = {
  ...chatterboxExpressivenessLogic,
  Component: ({ value, onChange, disabled }) => (
    <PanelSection
      title="Expressiveness"
      hint="The first run with a new voice or setting takes 5 to 15 s while the model studies it."
    >
      <ParamSlider
        label="Intensity"
        value={value.exaggeration}
        min={0}
        max={2}
        step={0.05}
        disabled={disabled}
        info="Higher is more dramatic; 0.5 is natural."
        onChange={(exaggeration) => onChange({ ...value, exaggeration })}
      />
      <ParamSlider
        label="Guidance"
        value={value.guidance}
        min={0}
        max={5}
        step={0.05}
        disabled={disabled}
        info="How closely the delivery follows the reference. Lower suits fast speakers."
        onChange={(guidance) => onChange({ ...value, guidance })}
      />
    </PanelSection>
  ),
};

const cosyVoiceModePanel: AudioToolPanel<CosyVoiceModeValue> = {
  ...cosyVoiceModeLogic,
  Component: ({ value, onChange, disabled }) => (
    <PanelSection
      title="Mode"
      hint={
        value.mode === "zero_shot"
          ? "Same language: speaks in the clip's language, using what's said in it."
          : value.mode === "cross_lingual"
            ? "Cross-lingual: speaks another language in the same voice. No transcript needed."
            : "Instruct: the same voice, delivered the way you describe."
      }
    >
      <PillTabs
        ariaLabel="Mode"
        value={value.mode}
        onValueChange={(mode) =>
          onChange({ ...value, mode: mode as CosyVoiceMode })
        }
        disabled={disabled}
        fit={true}
        compact={true}
        className="[&>button]:px-3"
        tabs={[
          { value: "zero_shot", label: "Same language" },
          { value: "cross_lingual", label: "Cross-lingual" },
          { value: "instruct", label: "Instruct" },
        ]}
      />
      {value.mode === "instruct" ? (
        <Field label="Instruction" htmlFor="clone-instruction">
          <Textarea
            id="clone-instruction"
            value={value.instruction}
            disabled={disabled}
            placeholder="Speak slowly, in a calm and warm tone."
            onChange={(event) =>
              onChange({ ...value, instruction: event.target.value })
            }
            className="min-h-16"
          />
        </Field>
      ) : null}
    </PanelSection>
  ),
};

const f5SpeedDialectPanel: AudioToolPanel<SpeedDialectValue> = {
  ...f5SpeedDialectLogic,
  Component: ({ value, onChange, disabled, specs }) => (
    <PanelSection title="Speed and dialect">
      <ParamSlider
        label="Speed"
        value={value.speed}
        min={0.5}
        max={2}
        step={0.05}
        disabled={disabled}
        displayValue={`${value.speed.toFixed(2)}×`}
        onChange={(speed) => onChange({ ...value, speed })}
      />
      {specs.some((spec) => spec.name === "dialect") ? (
        <div className="grid gap-1.5">
          <label
            htmlFor="clone-dialect"
            className="text-ui-13 font-medium text-foreground"
          >
            Dialect
          </label>
          <Select
            value={value.dialect}
            onValueChange={(dialect) => onChange({ ...value, dialect })}
            disabled={disabled}
          >
            <SelectTrigger id="clone-dialect" size="sm" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {F5_DIALECTS.map((dialect) => (
                <SelectItem key={dialect.value} value={dialect.value}>
                  {dialect.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      ) : null}
    </PanelSection>
  ),
};

export const CLONE_TOOL_PANELS = [
  qwen3TimbrePanel,
  indexTts2EmotionPanel,
  chatterboxExpressivenessPanel,
  cosyVoiceModePanel,
  f5SpeedDialectPanel,
] as const;
