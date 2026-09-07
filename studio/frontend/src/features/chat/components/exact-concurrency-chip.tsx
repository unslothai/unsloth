// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { exactConcurrencyChip } from "../lib/exact-concurrency";

/**
 * Says, next to the model name, that this model's answers do not depend on what else is
 * decoding beside them. Renders nothing in the ordinary `off` case, so it costs the
 * header no room until there is something to report.
 *
 * The label/title pair lives in a plain `.ts` next door; the test runner strips types but
 * does not transform JSX, so wording asserted here would be untestable.
 */
export function ExactConcurrencyChip() {
  const state = useChatRuntimeStore((s) => s.loadedExactConcurrency);
  const chip = exactConcurrencyChip(state);
  if (!chip) return null;
  return (
    <span
      // The full sentence is the title; a chip this small can only carry the word.
      title={chip.title}
      data-testid="exact-concurrency-chip"
      data-exact-concurrency={state}
      className="pointer-events-auto shrink-0 self-center rounded-full border border-border/60 px-2 py-0.5 text-ui-10 font-medium tracking-[0.08em] text-muted-foreground/80"
    >
      {chip.label}
    </span>
  );
}
