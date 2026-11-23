// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  compactMarkPercent,
  deriveContextUsageBar,
} from "../src/features/chat/lib/context-usage-bar-state.ts";
import {
  apiCompactionRequestFields,
  cloudCompactionTrigger,
  normalizeAutoCompactTokens,
  parseAutoCompactTokens,
} from "../src/features/chat/utils/auto-compaction.ts";
import {
  COMPACTION_NOTICE_LABEL,
  compactionNoticeKind,
} from "../src/features/chat/utils/context-truncation.ts";

import { readSrc } from "./helpers/kit.ts";

test("a connection's Auto-compact after replaces the 75% default and lifts the native cap", () => {
  assert.deepEqual(
    apiCompactionRequestFields({
      autoCompactEnabled: true,
      contextLength: 1_000_000,
      autoCompactTokens: 400_000,
    }),
    {
      context_overflow: "truncate_oldest",
      compaction_threshold: 400_000,
      compaction_uncapped: true,
      context_window: 1_000_000,
    },
  );
  // Never past the model's window.
  assert.equal(
    apiCompactionRequestFields({
      autoCompactEnabled: true,
      contextLength: 200_000,
      autoCompactTokens: 400_000,
    }).compaction_threshold,
    200_000,
  );
  // A connection with no catalogued window still carries its own trigger.
  assert.deepEqual(
    apiCompactionRequestFields({
      autoCompactEnabled: true,
      contextLength: null,
      autoCompactTokens: 6_000,
    }),
    {
      context_overflow: "truncate_oldest",
      compaction_threshold: 6_000,
      compaction_uncapped: true,
    },
  );
});

test("/compact on a cloud model asks for a compaction even with auto-compact off", () => {
  assert.deepEqual(
    apiCompactionRequestFields({
      autoCompactEnabled: false,
      contextLength: 200_000,
      compactNow: true,
    }),
    {
      context_overflow: "truncate_oldest",
      compaction_threshold: 150_000,
      context_window: 200_000,
      compact_context: true,
    },
  );
  assert.deepEqual(
    apiCompactionRequestFields({ autoCompactEnabled: false, contextLength: null, compactNow: true }),
    { context_overflow: "truncate_oldest", compact_context: true },
  );
});

test("the Auto-compact after field takes whole tokens within the server's range", () => {
  assert.equal(parseAutoCompactTokens(""), null);
  assert.equal(parseAutoCompactTokens("  "), null);
  assert.equal(parseAutoCompactTokens("300,000"), 300_000);
  assert.equal(parseAutoCompactTokens("1000"), 1_000);
  assert.equal(parseAutoCompactTokens("1200000"), 1_200_000);
  for (const bad of ["999", "1200001", "12.5", "lots", "-5"]) {
    assert.throws(() => parseAutoCompactTokens(bad), /Auto-compact after/);
  }
  assert.equal(normalizeAutoCompactTokens(300_000), 300_000);
  for (const bad of [999, 1.5, "300000", null, undefined]) {
    assert.equal(normalizeAutoCompactTokens(bad), undefined);
  }
  assert.equal(normalizeAutoCompactTokens(2_000_001), 1_200_000);
});

test("the meter's trigger mirrors what each provider really does", () => {
  // Generic connections compact at three quarters of the window.
  assert.equal(
    cloudCompactionTrigger({ providerType: "openrouter", contextLength: 400_000 }),
    300_000,
  );
  // Native compaction stops at 200K unless the connection says otherwise.
  for (const providerType of ["anthropic", "openai"]) {
    assert.equal(cloudCompactionTrigger({ providerType, contextLength: 1_000_000 }), 200_000);
    assert.equal(
      cloudCompactionTrigger({ providerType, contextLength: 1_000_000, autoCompactTokens: 600_000 }),
      600_000,
    );
  }
  // Claude never compacts below 50K.
  assert.equal(
    cloudCompactionTrigger({ providerType: "anthropic", contextLength: 200_000, autoCompactTokens: 5_000 }),
    50_000,
  );
  assert.equal(
    cloudCompactionTrigger({ providerType: "openai", contextLength: 200_000, autoCompactTokens: 5_000 }),
    5_000,
  );
  assert.equal(cloudCompactionTrigger({ providerType: "custom", contextLength: null }), null);
  assert.equal(
    cloudCompactionTrigger({ providerType: "custom", contextLength: null, autoCompactTokens: 6_000 }),
    6_000,
  );
});

test("the context meter marks where a cloud chat compacts", () => {
  assert.equal(compactMarkPercent(150_000, 200_000), 75);
  assert.equal(compactMarkPercent(200_000, 200_000), null);
  assert.equal(compactMarkPercent(150_000, null), null);
  assert.equal(compactMarkPercent(null, 200_000), null);
  const state = deriveContextUsageBar({
    used: 190_000,
    total: 200_000,
    cloud: true,
    compactAt: 150_000,
  });
  assert.equal(state?.compactMarkPercent, 75);
  // The window belongs to the provider: no "raise Context Length" advice.
  assert.equal(state?.advice, "none");
  assert.equal(deriveContextUsageBar({ used: 190_000, total: 200_000 })?.advice, "stops-at-limit");
  assert.equal(deriveContextUsageBar({ used: 10, total: 200 })?.compactMarkPercent, null);

  const bar = readSrc("features/chat/components/context-usage-bar.tsx");
  assert.match(bar, /data-slot="compaction-mark"/);
  assert.match(bar, /Auto-compacts at/);
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /total=\{isExternalModel \? cloudContextWindow : loadedContextLength\}/);
  assert.match(page, /compactAt=\{cloudCompactAt\}/);
  assert.match(page, /isExternalModel && autoCompactEnabled && activeExternalProviderType !== "openai_codex"/);
});

test("the thread says whether history was summarized or had to be dropped", () => {
  const fit = { dropped_messages: 4, fits: true };
  assert.equal(compactionNoticeKind(fit), "compacted");
  assert.equal(
    compactionNoticeKind({ ...fit, summarized: true, summary_source: "unsloth" }),
    "summarized",
  );
  assert.equal(compactionNoticeKind({ ...fit, summarized: true }), "summarized-by-provider");
  assert.equal(compactionNoticeKind({ ...fit, summary_failed: true }), "dropped-after-failed-summary");
  assert.equal(compactionNoticeKind({ ...fit, summary_failed: false }), "dropped");
  assert.notEqual(COMPACTION_NOTICE_LABEL.summarized, COMPACTION_NOTICE_LABEL.dropped);
  assert.equal(COMPACTION_NOTICE_LABEL.compacted, "Context compacted");

  const divider = readSrc("components/assistant-ui/compaction-divider.tsx");
  assert.match(divider, /label=\{COMPACTION_NOTICE_LABEL\[kind\]\}/);
  assert.match(divider, /data-compaction-kind=\{kind\}/);
  // The divider is the record; only history dropped without a summary also raises a toast.
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /if \(\s*\(noticeKind === "dropped" \|\| noticeKind === "dropped-after-failed-summary"\) &&/,
  );
  assert.match(adapter, /toast\.warning\(COMPACTION_NOTICE_LABEL\[noticeKind\]/);
  assert.doesNotMatch(adapter, /This conversation was compacted/);
});

test("the connection form stores Auto-compact after with the connection", () => {
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  assert.match(dialog, /id="provider-auto-compact-tokens"/);
  assert.match(dialog, /parseAutoCompactTokens\(autoCompactTokensDraft\)/);
  assert.match(dialog, /autoCompactTokens: updated\.auto_compact_tokens \?\? undefined/);
  const api = readSrc("features/chat/api/providers-api.ts");
  assert.equal(api.match(/auto_compact_tokens: payload\.autoCompactTokens/g)?.length, 2);
  const sync = readSrc("features/chat/sync-external-providers.ts");
  assert.match(sync, /autoCompactTokens: config\.auto_compact_tokens \?\? undefined/);
});

test("the composer offers no Compact chip: the header meter shows the context", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.doesNotMatch(thread, /CompactOnReturnChip|compact-on-return/);
  assert.match(thread, /<AuiIf[\s\S]{0,200}?>\s*<ComposerPrimitive\.Send asChild=\{true\}>/);
});
