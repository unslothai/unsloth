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

/**
 * What tells two halves apart once their model labels do not. The LoRA compare
 * runs one checkpoint with the adapter off and on, so both threads record the
 * same modelId and only modelType differs; the same happens when the two panes
 * of a general compare answered on the same checkpoint. Name the halves as the
 * compare header does ("Base Model" / "Fine-tuned"), else by position.
 */
function sideLabel(thread: ProjectSourceThread, index: number): string {
  if (thread.modelType === "base") return "base";
  if (thread.modelType === "lora") return "fine-tuned";
  return String(index + 1);
}

/**
 * What a "Save to project sources" click uploads. A compare pair is two models
 * answering the same prompt and both halves carry the row's single title, so
 * name each after its model: the sources panel lists a document by filename and
 * nothing else would tell the two apart.
 */
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
    index,
    label: modelLabel(thread) ?? String(index + 1),
  }));
  const uses = new Map<string, number>();
  for (const { label } of named) uses.set(label, (uses.get(label) ?? 0) + 1);
  return named.map(({ thread, index, label }) => {
    // Only a colliding half carries a side, so two different models keep the
    // plain "<title> - <model>" name.
    const side =
      (uses.get(label) ?? 0) > 1 ? ` - ${sideLabel(thread, index)}` : "";
    return { id: thread.id, title: `${item.title} - ${label}${side}` };
  });
}
