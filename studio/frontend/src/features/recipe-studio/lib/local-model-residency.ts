// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getInferenceStatus,
  validateModel,
} from "@/features/chat/api/chat-api";
import { isOllamaModelId } from "@/features/hub/lib/model-identity";

export type LocalModelSelection = {
  target: string;
  ggufVariant: string;
  aliases: string[];
  requestedContextLength?: number | null;
  isMlx?: boolean;
};

function isDirectGgufTarget(target: string): boolean {
  return target.toLowerCase().endsWith(".gguf");
}

function localSelectionMatchesActive(input: {
  target: string;
  ggufVariant: string;
  activeModel: string | null | undefined;
  activeVariant: string;
}): boolean {
  const { target, ggufVariant, activeModel, activeVariant } = input;
  if (!activeModel || activeModel.toLowerCase() !== target.toLowerCase()) {
    return false;
  }
  return (
    activeVariant === ggufVariant ||
    (isDirectGgufTarget(target) && !ggufVariant)
  );
}

/** A context request, read only where it pins: llama.cpp echoes n_ctx while Auto, MLX does not. */
export function contextIntent(
  value: number | null | undefined,
  isMlx: boolean | null | undefined,
): number | null {
  return isMlx && typeof value === "number" && value > 0 ? value : null;
}

/** One blob has three spellings and can carry two tags, so a disagreement is put to the server. */
async function localSelectionMatchesResident(input: {
  target: string;
  ggufVariant: string;
  activeModel: string | null | undefined;
  activeVariant: string;
}): Promise<boolean> {
  if (localSelectionMatchesActive(input)) {
    return true;
  }
  if (!isOllamaModelId(input.target)) {
    return false;
  }
  try {
    const validated = await validateModel({
      // biome-ignore lint/style/useNamingConvention: api schema
      model_path: input.target,
      // biome-ignore lint/style/useNamingConvention: api schema
      hf_token: null,
    });
    return validated.resident === true;
  } catch {
    return false;
  }
}

export async function isLocalModelAlreadyLoaded(
  selection: LocalModelSelection,
): Promise<boolean> {
  const { target, ggufVariant, requestedContextLength } = selection;
  try {
    const status = await getInferenceStatus();
    if (
      !(await localSelectionMatchesResident({
        target,
        ggufVariant,
        activeModel: status.model_identifier ?? status.active_model,
        activeVariant: status.gguf_variant?.trim() ?? "",
      }))
    ) {
      return false;
    }
    // A different context intent is a different load: asking for nothing inherits no pin.
    const residentIsMlx = status.is_mlx ?? false;
    return (
      contextIntent(requestedContextLength, residentIsMlx) ===
      contextIntent(status.requested_context_length, residentIsMlx)
    );
  } catch {
    return false;
  }
}
