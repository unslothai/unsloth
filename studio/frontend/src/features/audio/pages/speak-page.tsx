// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentProps } from "react";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { isMusicGenerationModel, macTtsCatalogChoiceIsRunnable } from "../catalog";
import { TtsOutput, TtsRailFields } from "./tts-workspace";

export function speakPageModels(
  models: ModelOption[],
  isMac: boolean,
): ModelOption[] {
  return models.filter(
    (model) =>
      !isMusicGenerationModel(model.id, model.audioType) &&
      (!isMac || macTtsCatalogChoiceIsRunnable(model.id)),
  );
}

export function SpeakRail(
  props: Omit<ComponentProps<typeof TtsRailFields>, "musicGeneration">,
) {
  return <TtsRailFields {...props} musicGeneration={false} />;
}

export function SpeakOutput(
  {
    modelReady,
    ...props
  }: Omit<ComponentProps<typeof TtsOutput>, "emptyText"> & {
    modelReady: boolean;
  },
) {
  return (
    <TtsOutput
      {...props}
      emptyText={
        modelReady
          ? "Generated speech lands here. Type a sentence and press Generate."
          : "Generated speech lands here. Load a TTS model, type a sentence, and press Generate."
      }
    />
  );
}
