// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import type { AudioOptionSpec } from "../audio-options";
import { AudioOptionFields } from "../audio-options-fields";
import { formatVibeVoiceScript } from "../clone-policy";
import { VoicePicker } from "../components/voice-picker";
import {
  SUPERTONIC_LANGUAGES,
  type SpeakVoiceValue,
  speakVoiceLogic,
  supertonicLanguageLogic,
  usesSavedVoice,
  vibeVoiceDialogueLogic,
} from "./panel-logic";
import type { AudioModelContext, AudioToolPanel } from "./types";

/** The model's built-in voice list, unless a saved voice is in use. */
const builtinVoiceSpecs = (
  specs: readonly AudioOptionSpec[],
  value: SpeakVoiceValue,
  ctx: AudioModelContext,
) =>
  usesSavedVoice(value, ctx)
    ? []
    : specs.filter((spec) => spec.name === ctx.voiceOption);

const speakVoicePanel: AudioToolPanel<SpeakVoiceValue> = {
  ...speakVoiceLogic,
  Component: ({ value, onChange, disabled, specs, ctx }) => (
    <div className="grid gap-2">
      {ctx.audioWorkflows?.includes("clone") ? (
        <>
          {/* The voice list carries its own label. */}
          {builtinVoiceSpecs(specs, value, ctx).length === 0 ? (
            <span className="text-ui-13 font-medium text-foreground">
              Voice
            </span>
          ) : null}
          <PillTabs
            ariaLabel="Voice"
            value={value.source}
            onValueChange={(source) =>
              onChange({
                ...value,
                source: source === "saved" ? "saved" : "builtin",
              })
            }
            disabled={disabled}
            fit={true}
            compact={true}
            className="[&>button]:px-3"
            tabs={[
              { value: "builtin", label: "Built-in" },
              { value: "saved", label: "Saved" },
            ]}
          />
        </>
      ) : null}
      {usesSavedVoice(value, ctx) ? (
        <VoicePicker
          selectedId={value.voiceId}
          disabled={disabled}
          onSelect={(voice) => onChange({ ...value, voiceId: voice.id })}
          onDeselect={() => onChange({ ...value, voiceId: null })}
        />
      ) : ctx.voiceOption && builtinVoiceSpecs(specs, value, ctx).length > 0 ? (
        <AudioOptionFields
          specs={builtinVoiceSpecs(specs, value, ctx)}
          values={
            value.builtin === null ? {} : { [ctx.voiceOption]: value.builtin }
          }
          onChange={(_name, builtin) =>
            onChange({ ...value, builtin: builtin ?? null })
          }
          disabled={disabled}
          family={ctx.audioFamily}
        />
      ) : (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          The model's own voice, shaped by a voice description when you give
          one.
        </p>
      )}
    </div>
  ),
};

const vibeVoiceDialoguePanel: AudioToolPanel<null> = {
  ...vibeVoiceDialogueLogic,
  Component: ({ core }) => {
    const text = core?.text.trim() ?? "";
    const script = formatVibeVoiceScript(text);
    return (
      <div className="grid gap-1.5">
        <span className="text-ui-13 font-medium text-foreground">Dialogue</span>
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          This model reads a script with one line per speaker, such as “Speaker
          1: Hello.” Plain text is read by Speaker 1.
        </p>
        {text && script !== text ? (
          <pre className="max-h-24 overflow-auto whitespace-pre-wrap rounded-lg bg-muted px-3 py-2 font-mono text-ui-11p5 text-muted-foreground">
            {script}
          </pre>
        ) : null}
      </div>
    );
  },
};

const LANGUAGE_NAMES = new Intl.DisplayNames(["en"], { type: "language" });
// Radix Select cannot hold an empty value.
const DEFAULT_LANGUAGE = "__default__";

const supertonicLanguagePanel: AudioToolPanel<string> = {
  ...supertonicLanguageLogic,
  Component: ({ value, onChange, disabled }) => (
    <div className="grid gap-1.5">
      <label
        className="text-ui-13 font-medium text-foreground"
        htmlFor="supertonic-language"
      >
        Language
      </label>
      <Select
        value={value || DEFAULT_LANGUAGE}
        onValueChange={(next) =>
          onChange(next === DEFAULT_LANGUAGE ? "" : next)
        }
        disabled={disabled}
      >
        <SelectTrigger id="supertonic-language" size="sm" className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={DEFAULT_LANGUAGE}>English (default)</SelectItem>
          {SUPERTONIC_LANGUAGES.filter((code) => code !== "en").map((code) => (
            <SelectItem key={code} value={code}>
              {LANGUAGE_NAMES.of(code) ?? code}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-ui-11p5 leading-snug text-muted-foreground">
        The language of the text, so it is pronounced right.
      </p>
    </div>
  ),
};

export const SPEAK_TOOL_PANELS = [
  speakVoicePanel,
  vibeVoiceDialoguePanel,
  supertonicLanguagePanel,
] as const;
