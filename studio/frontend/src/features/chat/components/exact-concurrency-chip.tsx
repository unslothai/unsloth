// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { exactConcurrencyChip } from "../lib/exact-concurrency";

/** Says this model's answers do not depend on concurrent decoding. Renders nothing for `off`. */
export function ExactConcurrencyChip() {
  const state = useChatRuntimeStore((s) => s.loadedExactConcurrency);
  // A first turn files its flag under "__default" until the conversation persists.
  const recomputed = useChatRuntimeStore(
    (s) =>
      s.preemptRecomputedByThreadId[s.activeThreadId || "__default"] === true,
  );
  const chip = exactConcurrencyChip(state, { recomputed });
  if (!chip) return null;
  return (
    <span
      title={chip.title}
      data-testid="exact-concurrency-chip"
      data-exact-concurrency={state}
      data-exact-recomputed={recomputed ? "true" : undefined}
      className="pointer-events-auto shrink-0 self-center rounded-full border border-border/60 px-2 py-0.5 text-ui-10 font-medium tracking-[0.08em] text-muted-foreground/80"
    >
      {chip.label}
    </span>
  );
}
