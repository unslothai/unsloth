// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { AudioTextRunRequest } from "../src/features/audio/audio-run-request.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { buildAudioRunBody } = await import(
  "../src/features/audio/audio-run-request.ts"
);
const { newMusicGroupId, runMusicTakes } = await import(
  "../src/features/audio/music/music-policy.ts"
);

const request: AudioTextRunRequest = {
  workflow: "music",
  text: "lofi",
  options: { seed: 7 },
  music: { mode: "song", lyrics: "la", duration_s: 30, variations: 3 },
};
const GROUP = "a".repeat(32);

/** Runs the takes with a fake server whose take n returns seed 1000 + n. */
function takes({
  count = 3,
  how = "sequential" as "batch" | "sequential" | null,
  failOn = 0,
  stopAfter = 0,
}) {
  const sent: AudioTextRunRequest[] = [];
  const shown: number[] = [];
  const statuses: string[] = [];
  const controller = new AbortController();
  const done = runMusicTakes({
    request,
    count,
    how,
    signal: controller.signal,
    groupId: GROUP,
    run: async (take) => {
      sent.push(take);
      if (sent.length === failOn) throw new Error("take failed");
      return { clips: [{ seed: 1000 + sent.length }] };
    },
    onTake: async (response) => {
      shown.push(response.clips[0].seed);
      if (shown.length === stopAfter) controller.abort();
    },
    onStatus: (status) => statuses.push(status),
  });
  return { done, sent, shown, statuses };
}

test("sequential takes run one request each and reuse take 1's seed", async () => {
  const h = takes({});
  await h.done;
  assert.deepEqual(h.statuses, [
    "Take 1 of 3…",
    "Take 2 of 3…",
    "Take 3 of 3…",
  ]);
  assert.deepEqual(
    h.sent.map((t) => [t.take, t.group_id, t.music?.variations, t.seed]),
    [
      [1, GROUP, 1, undefined],
      [2, GROUP, 1, 1001],
      [3, GROUP, 1, 1001],
    ],
  );
  assert.deepEqual(h.sent[0].options, { seed: 7 });
  assert.deepEqual(h.shown, [1001, 1002, 1003]);
  const body = buildAudioRunBody(h.sent[1]);
  assert.deepEqual(
    [body.take, body.group_id, body.seed, body.variations],
    [2, GROUP, 1001, undefined],
  );
});

test("an inline take 1 (history could not keep it) still hands its seed on", async () => {
  const sent: AudioTextRunRequest[] = [];
  await runMusicTakes({
    request,
    count: 2,
    how: "sequential",
    run: async (take) => {
      sent.push(take);
      return { clips: [], audio: { seed: 1001 } };
    },
    onTake: async () => {},
    onStatus: () => {},
  });
  assert.deepEqual(sent.map((t) => t.seed), [undefined, 1001]);
});

test("Stop after take 2 keeps two takes; a failed take 2 keeps take 1 and rejects", async () => {
  const stopped = takes({ stopAfter: 2 });
  await stopped.done;
  assert.deepEqual([stopped.sent.length, stopped.shown], [2, [1001, 1002]]);
  const failed = takes({ failOn: 2 });
  await assert.rejects(failed.done, /take failed/);
  assert.deepEqual(failed.shown, [1001]);
});

test("batch and single takes stay one request", async () => {
  for (const [how, count] of [
    ["batch", 3],
    ["sequential", 1],
    [null, 2],
  ] as const) {
    const h = takes({ how, count });
    await h.done;
    assert.deepEqual([h.sent.length, h.statuses], [1, []]);
    assert.equal(h.sent[0], request);
  }
  assert.match(newMusicGroupId(), /^[0-9a-f]{32}$/);
  assert.doesNotMatch(
    readSrc("features/audio/music/music-policy.ts"),
    /randomUUID\(/,
  );
  const speak = buildAudioRunBody({
    workflow: "speak",
    text: "hi",
    take: 2,
    group_id: GROUP,
  });
  assert.equal(speak.take, undefined);
});

test("the music hook drives the takes and shows status text only", () => {
  const hook = readSrc("features/audio/hooks/use-music-generation.ts");
  assert.match(
    hook,
    /await runMusicTakes\(\{[\s\S]*how: rule\.variations\?\.how \?\? null,[\s\S]*onStatus: setRunStatus/,
  );
  assert.match(hook, /reportAudioFailure\(error\)/);
  assert.doesNotMatch(hook, /<Progress/);
});
