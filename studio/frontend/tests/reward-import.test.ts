// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as RewardsApi from "../src/features/training/api/rewards-api.ts";
import * as rewardImport from "../src/features/training/lib/reward-import.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { strToU8, zipSync, Zip, ZipPassThrough } from "fflate";
import {
  checkRewardImportNames,
  MAX_REWARD_BYTES,
  readRewardZip,
  RewardImportError,
} from "../src/features/training/lib/reward-import.ts";

const markdown =
  "---\nname: answer\nkind: rule\ndescription: תשובה נכונה\n---\ntype: length\nmax_chars: 100\nscore:\n  under: 1\n  over: -1\n";
const hasCode = (code: RewardImportError["code"]) => (error: unknown) =>
  error instanceof RewardImportError && error.code === code;

test("reads the exact format produced by multi-reward export without changing UTF-8 or newlines", () => {
  const second = markdown.replace("name: answer", "name: second");
  const bytes = zipSync({
    "answer/REWARD.md": strToU8(markdown),
    "second/REWARD.md": strToU8(second),
  });
  assert.deepEqual(readRewardZip(bytes), [
    { path: "answer/REWARD.md", markdown },
    { path: "second/REWARD.md", markdown: second },
  ]);
});

test("ignores unrelated files and macOS archive metadata", () => {
  assert.deepEqual(
    readRewardZip(
      zipSync({
        "REWARD.md": strToU8(markdown),
        "readme.txt": strToU8("not a reward"),
        "__MACOSX/answer/REWARD.md": strToU8("metadata"),
      }),
    ),
    [{ path: "REWARD.md", markdown }],
  );
});

test("refuses empty ZIPs and unrelated contents", () => {
  const archives: Record<string, Uint8Array>[] = [
    {},
    { "readme.txt": strToU8("hello") },
  ];
  for (const entries of archives) {
    assert.throws(
      () => readRewardZip(zipSync(entries)),
      hasCode("importZipEmpty"),
    );
  }
});

test("refuses corrupted ZIPs and invalid UTF-8", () => {
  const valid = zipSync({ "answer/REWARD.md": strToU8(markdown) });
  assert.throws(
    () => readRewardZip(valid.subarray(0, 50)),
    hasCode("importInvalidZip"),
  );
  assert.throws(
    () => readRewardZip(zipSync({ "REWARD.md": new Uint8Array([0xc3, 0x28]) })),
    hasCode("importInvalidZip"),
  );
});

test("enforces the backend manifest limit on decompressed bytes", () => {
  const atLimit = new Uint8Array(MAX_REWARD_BYTES).fill(65);
  assert.equal(
    readRewardZip(zipSync({ "REWARD.md": atLimit }))[0].markdown.length,
    MAX_REWARD_BYTES,
  );
  assert.throws(
    () =>
      readRewardZip(
        zipSync({ "REWARD.md": new Uint8Array(MAX_REWARD_BYTES + 1) }),
      ),
    hasCode("importTooLarge"),
  );
});

function streamingZip(entries: [string, Uint8Array][]): Uint8Array {
  const chunks: Uint8Array[] = [];
  const zip = new Zip((error, data) => {
    if (error) throw error;
    chunks.push(data);
  });
  for (const [name, bytes] of entries) {
    const entry = new ZipPassThrough(name);
    zip.add(entry);
    entry.push(bytes, true);
  }
  zip.end();
  return Buffer.concat(chunks);
}

test("bounds streaming archives whose uncompressed size is not declared", () => {
  assert.deepEqual(
    readRewardZip(streamingZip([["REWARD.md", strToU8(markdown)]])),
    [{ path: "REWARD.md", markdown }],
  );
  assert.throws(
    () =>
      readRewardZip(
        streamingZip([["REWARD.md", new Uint8Array(MAX_REWARD_BYTES + 1)]]),
      ),
    hasCode("importTooLarge"),
  );
});

test("refuses repeated archive paths rather than silently replacing a file", () => {
  const bytes = streamingZip([
    ["REWARD.md", strToU8(markdown)],
    ["REWARD.md", strToU8(markdown)],
  ]);
  assert.throws(() => readRewardZip(bytes), hasCode("importDuplicate"));
});

test("caps archives at the backend library limit", () => {
  const bytes = zipSync(
    Object.fromEntries(
      Array.from({ length: 501 }, (_, i) => [
        `r${i}/REWARD.md`,
        strToU8(markdown),
      ]),
    ),
  );
  assert.throws(() => readRewardZip(bytes), hasCode("importTooLarge"));
});

test("rejects duplicate canonical reward names even when overwrite is enabled", () => {
  for (const overwrite of [false, true]) {
    assert.throws(
      () => checkRewardImportNames(["answer", "answer"], [], overwrite),
      hasCode("importDuplicate"),
    );
  }
});

test("requires explicit replacement for library conflicts before any file is saved", () => {
  assert.throws(
    () => checkRewardImportNames(["new-reward", "answer"], ["answer"], false),
    hasCode("importConflict"),
  );
  assert.doesNotThrow(() =>
    checkRewardImportNames(["new-reward", "answer"], ["answer"], true),
  );
  assert.doesNotThrow(() =>
    checkRewardImportNames(["new-reward"], ["answer"], false),
  );
});

function importClient(
  options: {
    invalid?: string;
    existing?: string[];
    failSave?: string;
  } = {},
) {
  const writes: { markdown: string; overwrite: boolean }[] = [];
  const previews: string[] = [];
  const api = loadWithStubs<typeof RewardsApi>(
    new URL("../src/features/training/api/rewards-api.ts", import.meta.url),
    {
      "../lib/reward-import": rewardImport,
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          const body = init?.body ? JSON.parse(String(init.body)) : null;
          if (path.endsWith("/preview")) {
            const text = body.rewards[0].markdown;
            previews.push(text);
            if (text === options.invalid)
              return Response.json({ detail: "Invalid rule" }, { status: 400 });
            return Response.json({
              scores: [{ name: text.toLowerCase(), score: 0, weighted: 0 }],
              total: 0,
            });
          }
          if (init?.method === "POST") {
            writes.push(body);
            if (body.markdown === options.failSave)
              return Response.json({ detail: "Disk full" }, { status: 500 });
            return Response.json({
              name: body.markdown.toLowerCase(),
              valid: true,
            });
          }
          return Response.json(
            (options.existing ?? []).map((name) => ({ name })),
          );
        },
      },
    },
  );
  return { api, writes, previews };
}

const importFiles = ["first", "second", "third"].map((name) => ({
  path: `${name}/REWARD.md`,
  markdown: name,
}));

test("preflight validates the entire set before persisting any rewards", async () => {
  const c = importClient({ invalid: "third" });
  const result = await c.api.importRewardFiles(importFiles);
  assert.deepEqual(c.previews, ["first", "second", "third"]);
  assert.deepEqual(c.writes, []);
  assert.deepEqual(result.imported, []);
  assert.deepEqual(result.remaining, importFiles);
  assert.match(String(result.error), /third\/REWARD.md: Invalid rule/);
});

test("duplicate canonical names and existing names abort before any writes", async () => {
  const duplicate = importClient();
  const result = await duplicate.api.importRewardFiles(
    [importFiles[0], { path: "another/REWARD.md", markdown: "FIRST" }],
    true,
  );
  assert.ok(hasCode("importDuplicate")(result.error));
  assert.deepEqual(duplicate.writes, []);
  const conflict = importClient({ existing: ["third"] });
  const conflicted = await conflict.api.importRewardFiles(importFiles);
  assert.ok(hasCode("importConflict")(conflicted.error));
  assert.deepEqual(conflict.writes, []);
});

test("explicit overwrite imports every validated file", async () => {
  const c = importClient({ existing: ["second"] });
  const result = await c.api.importRewardFiles(importFiles, true);
  assert.equal(result.error, null);
  assert.deepEqual(result.remaining, []);
  assert.deepEqual(
    result.imported.map((r) => r.name),
    ["first", "second", "third"],
  );
  assert.deepEqual(
    c.writes,
    importFiles.map((file) => ({ markdown: file.markdown, overwrite: true })),
  );
});

test("partial write failure retains successes and retries only unsaved rewards", async () => {
  const c = importClient({ failSave: "second" });
  const result = await c.api.importRewardFiles(importFiles);
  assert.deepEqual(
    result.imported.map((r) => r.name),
    ["first"],
  );
  assert.deepEqual(result.remaining, importFiles.slice(1));
  assert.match(String(result.error), /second\/REWARD.md: Disk full/);
  assert.equal(c.writes.length, 2);
  const retry = importClient({ existing: ["first"] });
  const retried = await retry.api.importRewardFiles(result.remaining);
  assert.equal(retried.error, null);
  assert.deepEqual(
    retry.writes.map((w) => w.markdown),
    ["second", "third"],
  );
});
