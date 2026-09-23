// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type TurnMessage = { readonly id: string; readonly role: string };

export interface ThreadTurns {
  // openerIds[n - 1] is the user message that opens turn n
  readonly openerIds: readonly string[];
  // compared by value so the navigator skips renders while a reply streams
  readonly signature: string;
  // turn per message index, 0 before the first user message
  readonly turnAt: Int32Array;
}

// keyed on array identity: selectors run on every store write, so one o(n) pass is shared per revision
const turnsByMessages = new WeakMap<object, ThreadTurns>();

export function threadTurns(messages: readonly TurnMessage[]): ThreadTurns {
  const known = turnsByMessages.get(messages);
  if (known) {
    return known;
  }
  const openerIds: string[] = [];
  const turnAt = new Int32Array(messages.length);
  for (let index = 0; index < messages.length; index++) {
    const message = messages[index];
    if (message.role === "user") {
      openerIds.push(message.id);
    }
    turnAt[index] = openerIds.length;
  }
  const turns = { openerIds, signature: openerIds.join("\n"), turnAt };
  turnsByMessages.set(messages, turns);
  return turns;
}

export function turnNumberAt(
  messages: readonly TurnMessage[],
  index: number,
): number {
  return threadTurns(messages).turnAt[index] ?? 0;
}

export function turnOpenerIdAt(
  messages: readonly TurnMessage[],
  index: number,
): string | undefined {
  const turns = threadTurns(messages);
  const turn = turns.turnAt[index] ?? 0;
  return turn > 0 ? turns.openerIds[turn - 1] : undefined;
}
