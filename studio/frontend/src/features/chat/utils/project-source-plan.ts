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
}

/** "unsloth/Qwen3-8B-GGUF:Q4_K_M" reads as "Qwen3-8B-GGUF" in a filename. */
function modelLabel(thread: ProjectSourceThread): string | undefined {
  const label = thread.modelId?.split("/").pop()?.split(":")[0]?.trim();
  return label || undefined;
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
  return threads.map((thread, index) => ({
    id: thread.id,
    title: `${item.title} - ${modelLabel(thread) ?? index + 1}`,
  }));
}
