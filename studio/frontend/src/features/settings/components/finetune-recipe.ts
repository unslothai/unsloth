// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings Data tab glue: turn chat history into a fine-tuning JSONL, stage
// it as a Data Recipe seed upload, and open a new recipe on that file.

import { buildFineTuneJsonl } from "@/features/chat";
import { saveRecipe } from "@/features/data-recipes/data/recipes-db";
import { createEmptyRecipePayload } from "@/features/recipe-studio";
import { inspectSeedUpload } from "@/features/recipe-studio/api";
import { toast } from "@/lib/toast";

/** btoa cannot handle code points above latin-1, so encode UTF-8 bytes. */
function base64FromString(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** Builds the JSONL, uploads it as a local recipe seed, and saves a new
 *  recipe whose seed block points at the file. Returns the recipe id, or
 *  null when there is nothing to export. */
export async function createFineTuneRecipeFromChats(): Promise<string | null> {
  const { lines, conversations } = await buildFineTuneJsonl();
  if (conversations === 0) {
    toast.info("No chats with a user and assistant exchange to export.");
    return null;
  }

  const dateLabel = new Date().toISOString().slice(0, 10);
  const filename = `chat-finetune-${dateLabel}.jsonl`;
  const inspected = await inspectSeedUpload({
    filename,
    // biome-ignore lint/style/useNamingConvention: api schema
    content_base64: base64FromString(lines.join("\n")),
    // biome-ignore lint/style/useNamingConvention: api schema
    preview_size: 10,
  });

  const payload = createEmptyRecipePayload();
  payload.recipe.seed_config = {
    source: {
      // biome-ignore lint/style/useNamingConvention: api schema
      seed_type: "local",
      path: inspected.resolved_path,
    },
    // biome-ignore lint/style/useNamingConvention: api schema
    sampling_strategy: "ordered",
    // biome-ignore lint/style/useNamingConvention: api schema
    selection_strategy: null,
  };
  payload.ui.nodes = [{ id: "seed", x: 0, y: 0, width: 400 }];
  payload.ui.seed_source_type = "local";
  payload.ui.seed_columns = inspected.columns;
  payload.ui.seed_preview_rows = inspected.preview_rows ?? [];
  payload.ui.local_file_name = filename;

  const record = await saveRecipe({
    name: `Chat fine-tuning ${dateLabel}`,
    payload,
  });
  return record.id;
}
