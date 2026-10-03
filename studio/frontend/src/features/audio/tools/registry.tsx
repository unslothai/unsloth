// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NativeAudioInstructionsKind } from "../audio-page-policy";
import {
  InstructionsField,
  MossLanguageField,
} from "../components/instructions-fields";
import type { AudioWorkflowId } from "../workflows";
import { CLONE_TOOL_PANELS } from "./clone-panels";
import { instructionsKindFor, panelApplies } from "./select";
import { SEPARATE_TOOL_PANELS } from "./separate-panels";
import { SPEAK_TOOL_PANELS } from "./speak-panels";
import type {
  AnyAudioToolPanel,
  AudioModelContext,
  AudioToolPanel,
} from "./types";

export interface InstructionsValue {
  instructions: string;
  language: string;
}

export const MUSIC_DESCRIPTION_REQUIRED =
  "Add a music description. This model needs one beside the lyrics.";

const VOICE_DESCRIPTION_REQUIRED =
  "Describe the voice. This model needs a voice description.";

function needsVoiceDescription(ctx: AudioModelContext): boolean {
  return ctx.requiredInputs?.includes("instruct") === true;
}

function instructionsPanel(
  id: string,
  kind: NativeAudioInstructionsKind,
  title: string,
  workflows: readonly AudioWorkflowId[],
): AudioToolPanel<InstructionsValue> {
  return {
    id,
    families: [],
    workflows,
    title,
    // Request fields, not spec options, so Advanced keeps every option.
    claims: [],
    appliesTo: (ctx) => instructionsKindFor(ctx) === kind,
    initial: () => ({ instructions: "", language: "" }),
    Component: ({ value, onChange, ctx }) => (
      <>
        <InstructionsField
          instructionsKind={kind}
          musicNeedsDescription={ctx.musicNeedsDescription}
          voiceRequired={kind === "voice" && needsVoiceDescription(ctx)}
          audioInstructions={value.instructions}
          setAudioInstructions={(instructions) =>
            onChange({ ...value, instructions })
          }
        />
        {kind === "style" ? (
          <MossLanguageField
            audioLanguage={value.language}
            setAudioLanguage={(language) => onChange({ ...value, language })}
          />
        ) : null}
      </>
    ),
    toRequest: (value) => {
      const instructions = value.instructions.trim();
      const language = value.language.trim();
      return {
        ...(instructions ? { instructions } : {}),
        ...(kind === "style" && language ? { language } : {}),
      };
    },
    validate: (value, _core, ctx) =>
      kind === "music" && ctx.musicNeedsDescription && !value.instructions.trim()
        ? MUSIC_DESCRIPTION_REQUIRED
        : kind === "voice" &&
            needsVoiceDescription(ctx) &&
            !value.instructions.trim()
          ? VOICE_DESCRIPTION_REQUIRED
          : null,
  };
}

const INSTRUCTION_PANELS: readonly AudioToolPanel<InstructionsValue>[] =
  [
    // Qwen3-TTS VoiceDesign and CustomVoice and VoxCPM2 read it; other GGUF speech models ignore it.
    instructionsPanel("voice-design", "voice", "Voice design", ["speak"]),
    instructionsPanel("higgs-scene", "scene", "Scene", ["speak"]),
    instructionsPanel("moss-style", "style", "Style", ["speak"]),
    // MiniMax Music 3 and YuE2 require it; other music models fall back to the lyrics.
    instructionsPanel("music-description", "music", "Description", ["music"]),
  ];

const INSTRUCTION_PANEL_IDS: ReadonlySet<string> = new Set(
  INSTRUCTION_PANELS.map((panel) => panel.id),
);

export function isInstructionPanel(id: string): boolean {
  return INSTRUCTION_PANEL_IDS.has(id);
}

export const AUDIO_TOOL_PANELS: readonly AnyAudioToolPanel[] = [
  SPEAK_TOOL_PANELS[0],
  ...INSTRUCTION_PANELS,
  ...SPEAK_TOOL_PANELS.slice(1),
  ...CLONE_TOOL_PANELS,
  ...SEPARATE_TOOL_PANELS,
];

export function audioToolPanelsFor(
  workflow: AudioWorkflowId,
  ctx: AudioModelContext,
): readonly AnyAudioToolPanel[] {
  return AUDIO_TOOL_PANELS.filter((panel) =>
    panelApplies(panel, workflow, ctx),
  );
}
