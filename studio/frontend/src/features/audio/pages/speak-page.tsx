// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentProps } from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/features/chat";
import { Field } from "../components/field";
import type { SpeakConnection } from "../hooks/use-speak-connection";
import { RecommendedModelPicks } from "../components/recommended-model-picks";
import { TtsOutput, TtsRailFields } from "./tts-workspace";

export function SpeakRail(
  props: Omit<ComponentProps<typeof TtsRailFields>, "musicGeneration">,
) {
  return <TtsRailFields {...props} musicGeneration={false} />;
}

export function SpeakConnectionRail({
  connection,
  prompt,
  setPrompt,
  instructions,
  setInstructions,
  setVoice,
  setSpeed,
  disabled,
}: {
  connection: SpeakConnection;
  prompt: string;
  setPrompt: (value: string) => void;
  instructions: string;
  setInstructions: (value: string) => void;
  setVoice: (value: string) => void;
  setSpeed: (value: number) => void;
  disabled: boolean;
}) {
  return (
    <>
      <Field
        label="Text"
        htmlFor="audio-prompt"
        hint={`What ${connection.providerName} should say. The clip lands in the gallery.`}
      >
        <Textarea
          data-type-to-activate="prompt"
          id="audio-prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Type the sentence to speak…"
          className="min-h-28"
        />
      </Field>
      <Field
        label="Voice"
        htmlFor="audio-connection-voice"
        hint="A voice the connection offers. Empty uses alloy."
      >
        <Input
          id="audio-connection-voice"
          value={connection.voice}
          onChange={(event) => setVoice(event.target.value)}
          placeholder="alloy"
        />
      </Field>
      <Field
        label="Style instructions"
        htmlFor="audio-instructions"
        hint="Optional. Sent to models that take instructions; others ignore them."
      >
        <Textarea
          id="audio-instructions"
          value={instructions}
          onChange={(event) => setInstructions(event.target.value)}
          placeholder="Warm, measured delivery with a calm conversational tone…"
          className="min-h-24"
        />
      </Field>
      <ParamSlider
        label="Speed"
        value={connection.speed}
        min={0.25}
        max={4}
        step={0.05}
        onChange={setSpeed}
        disabled={disabled}
      />
    </>
  );
}

export function SpeakOutput(
  {
    modelReady,
    recommendedModels = [],
    onPickModel,
    ...props
  }: Omit<ComponentProps<typeof TtsOutput>, "emptyText" | "emptyActions"> & {
    modelReady: boolean;
    recommendedModels?: readonly { id: string; name: string }[];
    onPickModel?: (id: string) => void;
  },
) {
  return (
    <TtsOutput
      {...props}
      emptyActions={
        modelReady ? null : (
          <RecommendedModelPicks models={recommendedModels} onPick={onPickModel} />
        )
      }
      emptyTitle="Generated speech will appear here"
      emptyText={
        modelReady
          ? "Type a sentence and press Generate."
          : "Load a speech model, type a sentence, and press Generate. For a conversation between several voices, load VibeVoice and write one line per speaker, like \"Speaker 1: Hi.\""
      }
    />
  );
}
