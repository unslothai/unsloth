// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Speed and context stats setting: settled (default) vs realtime, and the throttle realtime rides on.

import assert from "node:assert/strict";
import test from "node:test";

import type { LiveGenerationProgress } from "../src/features/chat/stores/chat-runtime-store.ts";
import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
const {
  createTrailingThrottle,
  GENERATION_STATS_UPDATES,
  LIVE_STATS_INTERVAL_MS,
  liveContextUsage,
  normaliseGenerationStatsUpdates,
} = await import("../src/features/chat/lib/live-generation-stats.ts");
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

const LOCALES = [
  "ar",
  "cs",
  "de",
  "en",
  "es-419",
  "es",
  "fa",
  "fr",
  "he",
  "hi",
  "id",
  "it",
  "ja",
  "ko",
  "nl",
  "pl",
  "pt-br",
  "ru",
  "sv",
  "th",
  "tr",
  "uk",
  "vi",
  "zh-CN",
  "zh-TW",
];

function fakeClock() {
  let now = 0;
  let timers: { at: number; run: () => void; id: number }[] = [];
  let nextId = 1;
  return {
    clock: {
      now: () => now,
      setTimeout: (run: () => void, ms: number) => {
        const id = nextId++;
        timers.push({ at: now + ms, run, id });
        return id;
      },
      clearTimeout: (id: unknown) => {
        timers = timers.filter((timer) => timer.id !== id);
      },
    },
    advance(ms: number) {
      now += ms;
      for (const timer of timers.filter((t) => t.at <= now)) {
        timers = timers.filter((t) => t !== timer);
        timer.run();
      }
    },
    pending: () => timers.length,
  };
}

const live = (
  fields: Partial<LiveGenerationProgress>,
): LiveGenerationProgress => ({
  prefill: null,
  prefillShown: false,
  decode: null,
  ...fields,
});
const prefill = (
  processed: number,
  total: number | null,
  cached: number | null = null,
) => ({
  processed,
  total,
  cached,
  tokPerS: null,
  elapsedMs: null,
  fit: null,
});
const decode = (tokens: number) => ({
  tokens,
  tokPerS: 42,
  draftN: null,
  draftAccepted: null,
});

test("the setting defaults to settled, and only an exact realtime survives a reload", () => {
  assert.deepEqual(GENERATION_STATS_UPDATES, ["settled", "realtime"]);
  assert.equal(
    useChatPreferencesStore.getInitialState().generationStatsUpdates,
    "settled",
  );
  const merge = useChatPreferencesStore.persist.getOptions().merge;
  assert.ok(merge);
  const current = useChatPreferencesStore.getInitialState();
  assert.equal(
    merge({ generationStatsUpdates: "realtime" }, current)
      .generationStatsUpdates,
    "realtime",
  );
  assert.equal(
    merge({ generationStatsUpdates: "settled" }, current)
      .generationStatsUpdates,
    "settled",
  );
  // Records written before the setting, or by a future value, fall back to today's behaviour.
  assert.equal(merge({}, current).generationStatsUpdates, "settled");
  assert.equal(
    merge({ generationStatsUpdates: "fast" }, current).generationStatsUpdates,
    "settled",
  );
  assert.equal(normaliseGenerationStatsUpdates(undefined), "settled");

  useChatPreferencesStore.getState().setGenerationStatsUpdates("realtime");
  assert.equal(
    useChatPreferencesStore.getState().generationStatsUpdates,
    "realtime",
  );
  useChatPreferencesStore.getState().setGenerationStatsUpdates("settled");
  assert.equal(
    useChatPreferencesStore.getState().generationStatsUpdates,
    "settled",
  );
});

test("realtime renders at most once per interval however fast frames arrive, and the last frame lands", () => {
  assert.ok(LIVE_STATS_INTERVAL_MS >= 100 && LIVE_STATS_INTERVAL_MS <= 250);
  const { clock, advance, pending } = fakeClock();
  let runs = 0;
  const throttle = createTrailingThrottle(200, () => runs++, clock);
  throttle.schedule();
  assert.equal(runs, 1, "the first frame shows at once");
  // One frame per token at 1,000 tok/s for 199 ms: one trailing render, not 199.
  for (let i = 0; i < 199; i++) {
    advance(1);
    throttle.schedule();
  }
  assert.equal(runs, 1);
  advance(1);
  assert.equal(runs, 2, "the burst lands once the interval is up");
  assert.equal(pending(), 0);
  advance(500);
  throttle.schedule();
  assert.equal(runs, 3);
  throttle.schedule();
  throttle.cancel();
  advance(1000);
  assert.equal(runs, 3, "an unmounted meter schedules no render");
});

test("realtime context counts the prompt as it is read, then the prompt plus the reply so far", () => {
  assert.equal(liveContextUsage(undefined), null);
  assert.deepEqual(
    liveContextUsage(live({ prefill: prefill(2048, 5316, 1024) })),
    {
      used: 2048,
      promptTokens: 2048,
      completionTokens: 0,
      cached: 1024,
    },
  );
  assert.deepEqual(
    liveContextUsage(
      live({ prefill: prefill(5316, 5316), decode: decode(300) }),
    ),
    {
      used: 5616,
      promptTokens: 5316,
      completionTokens: 300,
      cached: 0,
    },
  );
  // No prompt total, a compaction in progress, or the pass already settled: keep the settled count.
  assert.equal(
    liveContextUsage(
      live({ prefill: prefill(10, null), decode: decode(3) }),
    ),
    null,
  );
  assert.equal(liveContextUsage(live({ decode: decode(3) })), null);
  assert.equal(
    liveContextUsage(
      live({ prefill: prefill(10, 20), compacting: true }),
    ),
    null,
  );
  assert.equal(
    liveContextUsage(
      live({ prefill: prefill(20, 20), decode: decode(3), settled: true }),
    ),
    null,
  );
});

test("settled mode keeps today's paths: no live subscription in the meter, the line only with details", () => {
  const bar = readSrc("features/chat/components/live-context-usage-bar.tsx");
  const gate = bar.slice(
    bar.indexOf("export const LiveContextUsageBar"),
    bar.indexOf("const RealtimeContextUsageBar"),
  );
  assert.match(
    gate,
    /realtime \? \(\s*<RealtimeContextUsageBar [^>]*\/>\s*\) : \(\s*<ContextUsageBar \{\.\.\.props\} \/>/,
  );
  assert.doesNotMatch(
    gate,
    /useThrottledLiveGeneration|liveGenerationByThreadId|useChatRuntimeStore/,
  );
  const realtime = bar.slice(bar.indexOf("const RealtimeContextUsageBar"));
  assert.match(realtime, /useThrottledLiveGeneration\(liveThreadKey\)/);
  assert.doesNotMatch(
    realtime,
    /liveGenerationByThreadId/,
    "realtime reads live figures only through the throttle",
  );

  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(
    page,
    /<LiveContextUsageBar\s+(?:\/\/[^\n]*\s+)?liveThreadKey=\{activeThreadId \?\? "__default"\}/,
  );
  assert.doesNotMatch(page, /<ContextUsageBar\b/);

  const line = readSrc("components/assistant-ui/live-generation-line.tsx");
  assert.match(
    line,
    /return realtime \? <ThrottledDecodeFigures \/> : <LiveDecodeFigures \/>;/,
  );
  const direct = line.slice(
    line.indexOf("const LiveDecodeFigures"),
    line.indexOf("const ThrottledDecodeFigures"),
  );
  assert.match(
    direct,
    /useChatRuntimeStore\(/,
    "settled keeps the direct per-frame subscription",
  );
  const throttled = line.slice(
    line.indexOf("const ThrottledDecodeFigures"),
    line.indexOf("const DecodeFigures"),
  );
  assert.match(throttled, /useThrottledLiveGeneration\(/);
  assert.doesNotMatch(throttled, /useChatRuntimeStore\(/);

  const hook = readSrc("features/chat/hooks/use-throttled-live-generation.ts");
  assert.match(hook, /createTrailingThrottle\(LIVE_STATS_INTERVAL_MS/);
  assert.match(hook, /useChatRuntimeStore\.subscribe\(/);
  assert.match(hook, /throttle\.cancel\(\)/);
});

test("the toggle row sits in Chat defaults only, is searchable, and is translated in every locale", async () => {
  const row = readSrc(
    "features/settings/components/generation-stats-updates-row.tsx",
  );
  assert.match(row, /GENERATION_STATS_UPDATES\.map\(\(mode\) =>/);
  assert.match(row, /onClick=\{\(\) => setGenerationStatsUpdates\(mode\)\}/);
  const chat = readSrc("features/settings/tabs/chat-tab.tsx");
  const defaults = chat.slice(
    chat.indexOf(
      '<SettingsSection title={t("settings.general.chatDefaults")}>',
    ),
    chat.indexOf(
      '<SettingsSection title={t("settings.chat.groups.conversations.title")}>',
    ),
  );
  assert.match(defaults, /<GenerationStatsUpdatesRow \/>/);
  assert.equal(
    chat.match(/<GenerationStatsUpdatesRow \/>/g)?.length,
    1,
    "only once on the Chat tab",
  );
  assert.doesNotMatch(
    chat,
    /generationStatsUpdates/,
    "the Chat tab renders the shared row, not its own copy",
  );
  const system = readSrc("features/settings/tabs/resources-tab.tsx");
  assert.doesNotMatch(system, /GenerationStatsUpdatesRow/, "not on System");
  const search = readSrc("features/settings/settings-search.ts");
  assert.match(
    search,
    /"settings\.chat\.generationStatsUpdates": "settings\.chat\.generationStatsUpdatesKeywords"/,
  );
  const index = (tab: string) =>
    search.slice(
      search.indexOf(`  ${tab}: [`),
      search.indexOf("],", search.indexOf(`  ${tab}: [`)),
    );
  assert.match(index("chat"), /"settings\.chat\.generationStatsUpdates"/);
  assert.doesNotMatch(
    index("resources"),
    /"settings\.chat\.generationStatsUpdates"/,
  );

  const keys = [
    "generationStatsUpdates",
    "generationStatsUpdatesDescription",
    "generationStatsUpdatesSettled",
    "generationStatsUpdatesRealtime",
    "generationStatsUpdatesKeywords",
  ];
  for (const locale of LOCALES) {
    const module = (await import(`../src/i18n/locales/${locale}.ts`)) as Record<
      string,
      unknown
    >;
    const catalog = Object.values(module)[0] as {
      settings: { chat: Record<string, unknown> };
    };
    for (const key of keys) {
      const value = catalog.settings.chat[key];
      assert.ok(
        typeof value === "string" && value.length > 0,
        `${locale} settings.chat.${key}`,
      );
    }
  }
});

test("the toggle's tooltip explains both modes and warns that Realtime may cost performance", async () => {
  const row = readSrc(
    "features/settings/components/generation-stats-updates-row.tsx",
  );
  assert.match(
    row,
    /hint=\{t\("settings\.chat\.generationStatsUpdatesHint"\)\}/,
  );
  const { en } = (await import("../src/i18n/locales/en.ts")) as {
    en: { settings: { chat: { generationStatsUpdatesHint: string } } };
  };
  const hint = en.settings.chat.generationStatsUpdatesHint;
  assert.match(hint, /lag/);
  assert.match(hint, /performance/);
  for (const locale of LOCALES) {
    const module = (await import(`../src/i18n/locales/${locale}.ts`)) as Record<
      string,
      unknown
    >;
    const catalog = Object.values(module)[0] as {
      settings: { chat: Record<string, unknown> };
    };
    assert.ok(
      typeof catalog.settings.chat.generationStatsUpdatesHint === "string",
      `${locale} hint`,
    );
  }
});
