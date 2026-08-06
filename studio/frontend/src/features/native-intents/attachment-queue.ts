


import type { NativeIntent } from "./types";

export type PendingNativeAttachments = Record<string, NativeIntent[]>;

export function enqueueNativeAttachments(
  pending: PendingNativeAttachments,
  targetKey: string,
  intents: NativeIntent[],
): PendingNativeAttachments {
  const attachments = intents.filter((intent) => intent.kind === "attachment");
  if (attachments.length === 0) {
    return pending;
  }
  return {
    ...pending,
    [targetKey]: [...(pending[targetKey] ?? []), ...attachments],
  };
}

export function dequeueNativeAttachments(
  pending: PendingNativeAttachments,
  targetKey: string,
): [NativeIntent[], PendingNativeAttachments] {
  const attachments = pending[targetKey] ?? [];
  if (attachments.length === 0) {
    return [[], pending];
  }
  const remaining = { ...pending };
  delete remaining[targetKey];
  return [attachments, remaining];
}
