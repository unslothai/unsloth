// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentProps } from "react";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { isMusicGenerationModel, macTtsCatalogChoiceIsRunnable } from "../catalog";
import { TtsOutput, TtsRailFields } from "./tts-workspace";

export function musicPageModels(
  models: ModelOption[],
  isMac: boolean,
): ModelOption[] {
  return models.filter(
    (model) =>
      isMusicGenerationModel(model.id, model.audioType) &&
      (!isMac || macTtsCatalogChoiceIsRunnable(model.id)),
  );
}

export function MusicRail(
  props: Omit<ComponentProps<typeof TtsRailFields>, "musicGeneration">,
) {
  return <TtsRailFields {...props} musicGeneration={true} />;
}

export function MusicOutput(
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
          ? "Generated music lands here. Write lyrics or a description, then press Generate."
          : "Generated music lands here. Load a music model, write lyrics or a description, and press Generate."
      }
    />
  );
}
