// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One conversation to save, and the title its source is listed under. */
export interface ProjectSourcePlan {
  readonly id: string;
  readonly title: string;
}

/** The threads behind a sidebar row, as far as naming their sources needs. */
export interface ProjectSourceThread {
  readonly id: string;
  readonly modelId?: string;
  readonly modelType?: string;
}

/** "unsloth/Qwen3-8B-GGUF:Q4_K_M" reads as "Qwen3-8B-GGUF" in a filename. */
function modelLabel(thread: ProjectSourceThread): string | undefined {
  const label = thread.modelId?.split("/").pop()?.split(":")[0]?.trim();
  return label || undefined;
}

/** Which half of a compare this is. listStoredChatThreads sorts by updatedAt, so arrival order is
 *  whichever half answered last: naming by index alone would swap the two names between saves.
 *  modelType carries the pane, so only a thread missing it falls back to position. The LoRA
 *  compare gets the compare header's words, since "base" beats "1" in a filename. */
function paneLabel(thread: ProjectSourceThread, index: number): string {
  if (thread.modelType === "base") return "base";
  if (thread.modelType === "lora") return "fine-tuned";
  if (thread.modelType === "model1") return "1";
  if (thread.modelType === "model2") return "2";
  return String(index + 1);
}

/** What a "Save to project sources" click uploads. A compare pair is two models answering the same
 *  prompt and both halves carry the row's single title, so name each after its model: the sources
 *  panel lists a document by filename and nothing else would tell the two apart. */
export function planChatItemSources(
  item: { id: string; title: string; type: string },
  threads: readonly ProjectSourceThread[],
): ProjectSourcePlan[] {
  if (item.type === "single") return [{ id: item.id, title: item.title }];
  if (threads.length <= 1) {
    return threads.map((thread) => ({ id: thread.id, title: item.title }));
  }
  const named = threads.map((thread, index) => ({
    thread,
    label: modelLabel(thread) ?? paneLabel(thread, index),
    side: paneLabel(thread, index),
  }));
  const uses = new Map<string, number>();
  for (const { label } of named) uses.set(label, (uses.get(label) ?? 0) + 1);
  return named.map(({ thread, label, side }) => {
    // Only a colliding half carries a side, so two different models keep the plain "<title> - <model>" name.
    const suffix = (uses.get(label) ?? 0) > 1 ? ` - ${side}` : "";
    return { id: thread.id, title: `${item.title} - ${label}${suffix}` };
  });
}
