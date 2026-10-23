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
import { Field } from "../components/field";
import { PanelSection } from "./clone-panels";
import {
  ACE_STEP_BPM,
  ACE_STEP_KEYS,
  ACE_STEP_TIME_SIGNATURES,
  type AceStepMusicalValue,
  STABLE_AUDIO_SAMPLERS,
  type StableAudioSamplerValue,
  YUE_COT_LABELS,
  type YueCompositionValue,
  type YueCot,
  aceStepMusicalLogic,
  stableAudioSamplerLogic,
  stepsRange,
  yueCompositionLogic,
} from "./music-panel-logic";
import type { AudioToolPanel } from "./types";

/** Radix forbids an empty item value. */
const AUTO = "auto";

function AutoSelect({
  id,
  value,
  onChange,
  options,
  disabled,
}: {
  id: string;
  value: string;
  onChange: (next: string) => void;
  options: readonly { value: string; label: string }[];
  disabled: boolean;
}) {
  return (
    <Select
      value={value || AUTO}
      onValueChange={(next) => onChange(next === AUTO ? "" : next)}
      disabled={disabled}
    >
      <SelectTrigger id={id} size="sm" className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={AUTO}>Auto</SelectItem>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export const aceStepMusicalPanel: AudioToolPanel<AceStepMusicalValue> = {
  ...aceStepMusicalLogic,
  Component: ({ value, onChange, disabled }) => (
    <PanelSection
      title="Musical controls"
      hint="Optional. Anything left on Auto is chosen by the model."
    >
      <div className="grid gap-1.5">
        <label
          htmlFor="music-bpm-auto"
          className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
        >
          Set the tempo
          <Switch
            id="music-bpm-auto"
            checked={value.bpm !== null}
            disabled={disabled}
            onCheckedChange={(on) =>
              onChange({ ...value, bpm: on ? ACE_STEP_BPM.default : null })
            }
          />
        </label>
        {value.bpm !== null ? (
          <ParamSlider
            label="BPM"
            value={value.bpm}
            min={ACE_STEP_BPM.min}
            max={ACE_STEP_BPM.max}
            step={1}
            valueSize={4}
            disabled={disabled}
            onChange={(bpm) => onChange({ ...value, bpm })}
          />
        ) : null}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Key" htmlFor="music-key">
          <AutoSelect
            id="music-key"
            value={value.keyscale}
            onChange={(keyscale) => onChange({ ...value, keyscale })}
            options={ACE_STEP_KEYS.map((key) => ({ value: key, label: key }))}
            disabled={disabled}
          />
        </Field>
        <Field label="Time signature" htmlFor="music-time-signature">
          <AutoSelect
            id="music-time-signature"
            value={value.timesignature}
            onChange={(timesignature) => onChange({ ...value, timesignature })}
            options={ACE_STEP_TIME_SIGNATURES}
            disabled={disabled}
          />
        </Field>
      </div>
      <Field
        label="Keep out"
        htmlFor="music-avoid"
        hint="Sounds or styles to avoid, such as “distorted guitar, shouting”."
      >
        <Textarea
          id="music-avoid"
          value={value.avoid}
          disabled={disabled}
          onChange={(event) =>
            onChange({ ...value, avoid: event.target.value })
          }
          className="min-h-12"
        />
      </Field>
      <Field label="Sampler" htmlFor="music-ace-sampler">
        <AutoSelect
          id="music-ace-sampler"
          value={value.sampler}
          onChange={(sampler) => onChange({ ...value, sampler })}
          options={[
            { value: "euler", label: "Euler (faster)" },
            { value: "heun", label: "Heun (smoother, slower)" },
          ]}
          disabled={disabled}
        />
      </Field>
    </PanelSection>
  ),
};

export const yueCompositionPanel: AudioToolPanel<YueCompositionValue> = {
  ...yueCompositionLogic,
  Component: ({ value, onChange, disabled }) => (
    <PanelSection title="Planning" hint={YUE_COT_LABELS[value.cot].hint}>
      <PillTabs
        ariaLabel="Planning"
        value={value.cot}
        onValueChange={(cot) => onChange({ cot: cot as YueCot })}
        disabled={disabled}
        fit={true}
        compact={true}
        className="[&>button]:px-3"
        tabs={(Object.keys(YUE_COT_LABELS) as YueCot[]).map((cot) => ({
          value: cot,
          label: YUE_COT_LABELS[cot].label,
        }))}
      />
    </PanelSection>
  ),
};

export const stableAudioSamplerPanel: AudioToolPanel<StableAudioSamplerValue> =
  {
    ...stableAudioSamplerLogic,
    Component: ({ value, onChange, disabled, specs }) => {
      const steps = stepsRange(specs);
      return (
        <PanelSection
          title="Sampler"
          hint="Optional. More steps can sound cleaner and take longer."
        >
          <Field label="Method" htmlFor="music-sa-sampler">
            <AutoSelect
              id="music-sa-sampler"
              value={value.sampler}
              onChange={(sampler) => onChange({ ...value, sampler })}
              options={STABLE_AUDIO_SAMPLERS}
              disabled={disabled}
            />
          </Field>
          <ParamSlider
            label="Steps"
            value={value.steps ?? steps.default}
            min={steps.min}
            max={steps.max}
            step={1}
            valueSize={4}
            disabled={disabled}
            onChange={(next) => onChange({ ...value, steps: next })}
          />
        </PanelSection>
      );
    },
  };

export const MUSIC_TOOL_PANELS = [
  aceStepMusicalPanel,
  yueCompositionPanel,
  stableAudioSamplerPanel,
] as const;
