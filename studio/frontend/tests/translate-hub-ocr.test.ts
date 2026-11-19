// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { quantOnDisk } from "../src/features/translate/quant-on-disk.ts";

import { readText } from "./helpers/kit.ts";

type Row = { quant: string; downloaded?: boolean; partial?: boolean };

function lister(onDisk: { local: Row[] | null; remote: Row[] }) {
  const calls = { local: 0, remote: 0 };
  const list = async (localOnly: boolean) => {
    if (localOnly) {
      calls.local++;
      // The backend answers 404 for a repo it has no cached listing of.
      if (onDisk.local === null) throw new Error("404");
      return onDisk.local;
    }
    calls.remote++;
    return onDisk.remote;
  };
  return { calls, list };
}

test("a quant complete on disk is never listed on the Hub", async () => {
  const { calls, list } = lister({
    local: [{ quant: "Q4_K_M", downloaded: true }],
    remote: [],
  });
  assert.equal(await quantOnDisk("q4_k_m", list, { remote: true }), true);
  assert.deepEqual(calls, { local: 1, remote: 0 });
});

test("a missing or partial quant is asked of the Hub once, then only of the disk", async () => {
  for (const local of [null, [{ quant: "Q4_K_M", partial: true }]]) {
    const { calls, list } = lister({
      local,
      remote: [{ quant: "Q4_K_M", downloaded: false }],
    });
    assert.equal(await quantOnDisk("q4_k_m", list, { remote: true }), false);
    assert.deepEqual(calls, { local: 1, remote: 1 });
    const again = await quantOnDisk("q4_k_m", list, { remote: false });
    assert.equal(again, local === null ? null : false);
    assert.deepEqual(calls, { local: 2, remote: 1 });
  }
});

test("five translations cost no Hub listing once each model's state is known", async () => {
  const disk = { "1.8B": true, "7B": false };
  const listers = {
    "1.8B": lister({
      local: [{ quant: "Q4_K_M", downloaded: true }],
      remote: [],
    }),
    "7B": lister({
      local: null,
      remote: [{ quant: "Q4_K_M", downloaded: false }],
    }),
  };
  const known: Record<string, boolean | undefined> = {};
  // The page's check: on open and after every run.
  for (let pause = 0; pause <= 5; pause++) {
    for (const id of ["1.8B", "7B"] as const) {
      const onDisk = await quantOnDisk("q4_k_m", listers[id].list, {
        remote: known[id] !== false,
      });
      if (onDisk !== null) known[id] = onDisk;
    }
  }
  assert.deepEqual(known, disk);
  assert.deepEqual(listers["1.8B"].calls, { local: 6, remote: 0 });
  assert.deepEqual(listers["7B"].calls, { local: 6, remote: 1 });
});

test("a model downloaded since the last check reads as downloaded from the disk", async () => {
  const state = {
    local: null as Row[] | null,
    remote: [{ quant: "Q4_K_M", downloaded: false }],
  };
  const { calls, list } = lister(state);
  assert.equal(await quantOnDisk("q4_k_m", list, { remote: true }), false);
  state.local = [{ quant: "Q4_K_M", downloaded: true }];
  assert.equal(await quantOnDisk("q4_k_m", list, { remote: false }), true);
  assert.deepEqual(calls, { local: 2, remote: 1 });
});

test("the page re-checks a model known missing from the disk alone", () => {
  const page = readText("../src/features/translate/translate-page.tsx");
  assert.match(
    page,
    /translateModelDownloaded\(model, abort\.signal, \{\s*remote: downloadedRef\.current\[model\.id\] !== false,/,
  );
  const runner = readText("../src/features/translate/translate-runner.ts");
  assert.match(
    runner,
    /listGgufVariants\(model\.repo, undefined, \{ signal, localOnly \}\)/,
  );
});

test("only Translate sends its character limit to the document reader", () => {
  const page = readText("../src/features/translate/translate-page.tsx");
  assert.match(
    page,
    /extractDocumentText\(\s*file,\s*undefined,\s*TRANSLATE_SOURCE_MAX_CHARS,\s*\)/,
  );
  const api = readText("../src/features/rag/api/rag-api.ts");
  assert.match(
    api,
    /if \(maxChars !== undefined\) form\.append\("max_chars", String\(maxChars\)\);/,
  );
  const chat = readText("../src/features/chat/server-document.ts");
  assert.match(chat, /extractDocumentText\(file, sameAccount\)/);
});
