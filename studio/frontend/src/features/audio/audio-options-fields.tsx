// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ParamSlider } from "@/features/chat";
import {
  type AudioOptionSpec,
  type AudioOptionValue,
  type AudioOptionValues,
  audioOptionDisplayValue,
  audioOptionFloatStep,
  audioOptionLabel,
  audioVoiceLabel,
  coerceAudioOptionValue,
} from "./audio-options";

function OptionHint({ text }: { text?: string | null }) {
  return text ? (
    <p className="text-ui-11p5 leading-snug text-muted-foreground">{text}</p>
  ) : null;
}

/** One control per declared option, drawn from the type alone: a switch, a slider when the
 *  range is bounded (else a number box), a select, or a text box. */
export function AudioOptionFields({
  specs,
  values,
  onChange,
  disabled,
  family,
}: {
  specs: readonly AudioOptionSpec[];
  values: AudioOptionValues;
  /** `undefined` clears the option back to the model's default. */
  onChange: (name: string, value: AudioOptionValue | undefined) => void;
  disabled?: boolean;
  family?: string | null;
}) {
  return (
    <>
      {specs.map((spec) => {
        const id = `audio-option-${spec.name.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
        const label = `${audioOptionLabel(spec.name)}${spec.required ? " *" : ""}`;
        const value = audioOptionDisplayValue(spec, values);
        if (spec.type === "bool") {
          return (
            <div key={spec.name} className="grid gap-1.5">
              <label
                htmlFor={id}
                className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
              >
                {label}
                <Switch
                  id={id}
                  checked={value === true}
                  disabled={disabled}
                  onCheckedChange={(checked) => onChange(spec.name, checked)}
                />
              </label>
              <OptionHint text={spec.description} />
            </div>
          );
        }
        if (
          (spec.type === "int" || spec.type === "float") &&
          spec.min != null &&
          spec.max != null &&
          spec.max > spec.min
        ) {
          const step = spec.type === "int" ? 1 : audioOptionFloatStep(spec.min, spec.max);
          return (
            <ParamSlider
              key={spec.name}
              label={label}
              value={typeof value === "number" ? value : spec.min}
              min={spec.min}
              max={spec.max}
              step={step}
              disabled={disabled}
              info={spec.description ?? undefined}
              onChange={(next) =>
                onChange(spec.name, coerceAudioOptionValue(spec, next))
              }
            />
          );
        }
        if (spec.type === "enum") {
          return (
            <div key={spec.name} className="grid gap-1.5">
              <label className="text-ui-13 font-medium text-foreground" htmlFor={id}>
                {label}
              </label>
              <Select
                value={typeof value === "string" ? value : undefined}
                onValueChange={(next) => onChange(spec.name, next)}
                disabled={disabled}
              >
                <SelectTrigger id={id} size="sm" className="w-full">
                  <SelectValue placeholder="Model default" />
                </SelectTrigger>
                <SelectContent>
                  {(spec.values ?? []).map((choice) => (
                    <SelectItem key={choice} value={choice}>
                      {spec.name === "voice" ? audioVoiceLabel(choice, family) : choice}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <OptionHint text={spec.description} />
            </div>
          );
        }
        const numeric = spec.type === "int" || spec.type === "float";
        return (
          <div key={spec.name} className="grid gap-1.5">
            <label className="text-ui-13 font-medium text-foreground" htmlFor={id}>
              {label}
            </label>
            <Input
              id={id}
              type={numeric ? "number" : "text"}
              inputMode={numeric ? "decimal" : undefined}
              step={spec.type === "int" ? 1 : numeric ? "any" : undefined}
              min={spec.min ?? undefined}
              max={spec.max ?? undefined}
              // As typed: clamping mid-entry breaks multi-digit input. The request clamps.
              value={
                values[spec.name] === undefined ? "" : String(values[spec.name])
              }
              placeholder={
                spec.default != null ? String(spec.default) : "Model default"
              }
              disabled={disabled}
              onChange={(event) => {
                const raw = event.target.value;
                if (raw === "") {
                  onChange(spec.name, undefined);
                } else if (!numeric) {
                  onChange(spec.name, raw);
                } else if (Number.isFinite(Number(raw))) {
                  onChange(spec.name, Number(raw));
                }
              }}
            />
            <OptionHint text={spec.description} />
          </div>
        );
      })}
    </>
  );
}
