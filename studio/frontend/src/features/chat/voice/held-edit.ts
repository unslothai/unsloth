// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A held fragment the user clicked into. The turn judge parks an unfinished
 * transcript in the composer and sends it on its own later; a click on the box
 * means they are about to edit it, so from then on nothing is auto-sent: more
 * speech is appended and the turn goes when they send it. Sending, or clearing
 * the box, hands the next turn back to the loop.
 */
export type HeldEditDeps = {
  /** The composer's text, read only when a decision is due, never per keystroke. */
  readBox: () => string;
  onChange?: (owned: boolean) => void;
  trace?: (event: string, data: Record<string, unknown>) => void;
};

export type HeldEdit = {
  /** A click on the composer while `held` is parked there. True when it took the turn. */
  claim: (held: string) => boolean;
  /** Whether the loop may send a turn by itself now. An emptied box ends the edit. */
  autoSend: () => boolean;
  /** The user sent the turn. */
  sent: () => void;
  /** Voice off or a thread switch. */
  reset: () => void;
  readonly owned: boolean;
};

export function createHeldEdit(deps: HeldEditDeps): HeldEdit {
  let owned = false;
  const set = (next: boolean, why: string, data: Record<string, unknown> = {}) => {
    if (owned === next) return;
    owned = next;
    deps.trace?.(next ? "hold.userEdit" : "hold.userEditEnd", { why, ...data });
    deps.onChange?.(next);
  };
  return {
    claim: (held) => {
      if (owned || !held.trim()) return false;
      set(true, "click", { text: held });
      return true;
    },
    autoSend: () => {
      if (owned && !deps.readBox().trim()) set(false, "cleared");
      return !owned;
    },
    sent: () => set(false, "sent"),
    reset: () => set(false, "reset"),
    get owned() {
      return owned;
    },
  };
}
