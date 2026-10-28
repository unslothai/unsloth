// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ProjectSourcePlan {
  readonly id: string;
  readonly title: string;
}

export interface ProjectSourceThread {
  readonly id: string;
  readonly modelId?: string;
  readonly modelType?: string;
}

function modelLabel(thread: ProjectSourceThread): string | undefined {
  const label = thread.modelId?.split("/").pop()?.split(":")[0]?.trim();
  return label || undefined;
}

/** listStoredChatThreads sorts by updatedAt, so name by modelType, not index. */
function paneLabel(thread: ProjectSourceThread, index: number): string {
  if (thread.modelType === "base") return "base";
  if (thread.modelType === "lora") return "fine-tuned";
  if (thread.modelType === "model1") return "1";
  if (thread.modelType === "model2") return "2";
  return String(index + 1);
}

/** Compare halves share one title, so name each source after its model. */
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
    const suffix = (uses.get(label) ?? 0) > 1 ? ` - ${side}` : "";
    return { id: thread.id, title: `${item.title} - ${label}${suffix}` };
  });
}
