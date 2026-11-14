// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type HeldEditDeps = {
  /** The composer's text, read only when a decision is due, never per keystroke. */
  readBox: () => string;
  onChange?: (owned: boolean) => void;
  trace?: (event: string, data: Record<string, unknown>) => void;
};

export type HeldEdit = {
  claim: (held: string) => boolean;
  autoSend: () => boolean;
  sent: () => void;
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
