// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Speak's model tools beyond the instruction fields: a saved voice for models that also clone,
// and VibeVoice's speaker script.

import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { formatVibeVoiceScript } from "../clone-policy";
import { VoicePicker } from "../components/voice-picker";
import {
  type SpeakVoiceValue,
  speakVoiceLogic,
  vibeVoiceDialogueLogic,
} from "./panel-logic";
import type { AudioToolPanel } from "./types";

/** Models that both speak and clone (VoxCPM2) can read the text in a saved voice. */
export const speakVoicePanel: AudioToolPanel<SpeakVoiceValue> = {
  ...speakVoiceLogic,
  Component: ({ value, onChange, disabled }) => (
    <div className="grid gap-2">
      <span className="text-ui-13 font-medium text-foreground">Voice</span>
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
      {value.source === "saved" ? (
        <VoicePicker
          selectedId={value.voiceId}
          disabled={disabled}
          onSelect={(voice) => onChange({ ...value, voiceId: voice.id })}
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

/** VibeVoice reads `Speaker N:` lines; plain text is sent as the first speaker's. */
export const vibeVoiceDialoguePanel: AudioToolPanel<null> = {
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
          <pre className="max-h-24 overflow-auto whitespace-pre-wrap rounded-xl bg-muted px-3 py-2 font-mono text-ui-11p5 text-muted-foreground">
            {script}
          </pre>
        ) : null}
      </div>
    );
  },
};

export const SPEAK_TOOL_PANELS = [
  speakVoicePanel,
  vibeVoiceDialoguePanel,
] as const;
