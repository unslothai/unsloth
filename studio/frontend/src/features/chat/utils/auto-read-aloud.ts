// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readIncompleteInfo } from "./continuation.ts";

export interface AutoReadMessage {
  readonly id: string;
  readonly role: string;
  readonly metadata?: unknown;
  readonly status?: { readonly type: string };
  readonly content: readonly {
    readonly type: string;
    readonly text?: string;
  }[];
}

export type AutoReadDecision =
  | { kind: "wait" }
  | { kind: "skip" }
  | { kind: "speak"; messageId: string };

export function decideAutoRead(
  messages: readonly AutoReadMessage[],
): AutoReadDecision {
  const last = messages.at(-1);
  if (!last || last.role !== "assistant") return { kind: "skip" };
  if (last.status?.type === "running") return { kind: "wait" };
  if (last.status?.type !== "complete") return { kind: "skip" };
  // Cut-off turns can still settle as "complete".
  if (readIncompleteInfo(last.metadata)) return { kind: "skip" };
  const hasText = last.content.some(
    (part) => part.type === "text" && !!part.text?.trim(),
  );
  return hasText ? { kind: "speak", messageId: last.id } : { kind: "skip" };
}
