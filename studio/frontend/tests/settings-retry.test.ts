// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the settings queue keeps after a failed PUT.
//
// The case that matters is the one that used to break the tab: /api/chat/settings
// is extra="forbid" and rejects the whole body on one bad field, so requeueing a
// permanently-rejected patch made every later save carry it and fail too. Verified
// against a live Unsloth in Chromium, Firefox and WebKit before the fix.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ChatSettingsRequestError,
  isTerminalSettingsRejection,
  isUnderKeepaliveBudget,
  rejectedSettingKeys,
  retryablePatchAfterFailure,
} from "../src/features/chat/utils/settings-retry.ts";

const extraForbidden = (...fields: string[]) =>
  new ChatSettingsRequestError(
    "Request failed (400)",
    400,
    fields.map((field) => ({
      type: "extra_forbidden",
      loc: [field],
      msg: "Extra inputs are not permitted",
    })),
  );

test("a 4xx is terminal and a 5xx is not", () => {
  for (const status of [400, 401, 403, 404, 409, 422]) {
    assert.equal(
      isTerminalSettingsRejection(
        new ChatSettingsRequestError("x", status, null),
      ),
      true,
      `${status}`,
    );
  }
  for (const status of [408, 429, 500, 502, 503, 504]) {
    assert.equal(
      isTerminalSettingsRejection(
        new ChatSettingsRequestError("x", status, null),
      ),
      false,
      `${status}`,
    );
  }
});

test("a network failure is not a rejection", () => {
  assert.equal(isTerminalSettingsRejection(new TypeError("Failed to fetch")), false);
  assert.equal(isTerminalSettingsRejection(undefined), false);
});

test("the rejected fields are read out of the pydantic detail", () => {
  assert.deepEqual(
    rejectedSettingKeys(extraForbidden("ragTopK", "toolsEnabled").detail).sort(),
    ["ragTopK", "toolsEnabled"],
  );
});

test("a nested failure names the whole setting", () => {
  const detail = [{ type: "float_parsing", loc: ["inferenceParams", "temperature"] }];
  assert.deepEqual(rejectedSettingKeys(detail), ["inferenceParams"]);
});

test("a detail that is not a pydantic list names nothing", () => {
  assert.deepEqual(rejectedSettingKeys("Cannot apply partial settings patch"), []);
  assert.deepEqual(rejectedSettingKeys(null), []);
  assert.deepEqual(rejectedSettingKeys([{ loc: [] }, { loc: [7] }, 3]), []);
});

test("a transient failure keeps the whole patch", () => {
  const patch = { ragTopK: 11, toolsEnabled: true };
  const result = retryablePatchAfterFailure(patch, new TypeError("offline"));
  assert.deepEqual(result.patch, patch);
  assert.deepEqual(result.dropped, []);
  assert.equal(result.progressed, false);
});

test("a rejected field is dropped and the rest of the patch survives", () => {
  const patch = { ragTopK: 11, toolsEnabled: true, permissionMode: "ask" };
  const result = retryablePatchAfterFailure(patch, extraForbidden("ragTopK"));
  assert.deepEqual(result.patch, { toolsEnabled: true, permissionMode: "ask" });
  assert.deepEqual(result.dropped, ["ragTopK"]);
  // Strictly smaller, so the caller may reschedule without looping forever.
  assert.equal(result.progressed, true);
});

test("an unattributable rejection drops the patch rather than guessing", () => {
  const error = new ChatSettingsRequestError("Request failed (400)", 400, "nope");
  const result = retryablePatchAfterFailure({ ragTopK: 11 }, error);
  assert.deepEqual(result.patch, {});
  assert.deepEqual(result.dropped, ["ragTopK"]);
  assert.equal(result.progressed, false);
});

test("a rejection naming a field the patch does not hold drops nothing extra", () => {
  const result = retryablePatchAfterFailure(
    { ragTopK: 11 },
    extraForbidden("somethingElse"),
  );
  assert.deepEqual(result.patch, {});
  assert.deepEqual(result.dropped, ["ragTopK"]);
});

test("the whole-patch rejection of an old backend leaves nothing pending", () => {
  // A new bundle against a rolled-back backend: every mirrored field is refused,
  // the supported ones are not, and the retry carries only those.
  const patch = {
    autoTitle: true,
    inferenceParams: { temperature: 0.7 },
    toolsEnabled: true,
    permissionMode: "ask",
    ragTopK: 11,
  };
  const result = retryablePatchAfterFailure(
    patch,
    extraForbidden("toolsEnabled", "permissionMode", "ragTopK"),
  );
  assert.deepEqual(result.patch, {
    autoTitle: true,
    inferenceParams: { temperature: 0.7 },
  });
  assert.equal(result.progressed, true);
});

test("retrying is bounded: each round strictly shrinks the patch", () => {
  let patch: Record<string, unknown> = {
    a: 1, b: 2, c: 3, autoTitle: true,
  };
  const rounds: number[] = [];
  for (const field of ["a", "b", "c"]) {
    const result = retryablePatchAfterFailure(patch, extraForbidden(field));
    patch = result.patch as Record<string, unknown>;
    rounds.push(Object.keys(patch).length);
  }
  assert.deepEqual(rounds, [3, 2, 1]);
  assert.deepEqual(patch, { autoTitle: true });
});

// Fetch caps all in-flight keepalive bodies at 64 KiB and a valid
// researchWebsitePolicy (2000 domains x 253 characters) is far past that, so a patch
// sent with keepalive over the budget fails immediately -- measured as "Failed to
// fetch" in Chromium, "NetworkError" in Firefox and "Load failed" in WebKit.

test("the keepalive budget is decided by bytes, not by string length", () => {
  assert.equal(isUnderKeepaliveBudget(JSON.stringify({ ragTopK: 11 })), true);
  assert.equal(isUnderKeepaliveBudget("x".repeat(20 * 1024)), true);
  assert.equal(isUnderKeepaliveBudget("x".repeat(61 * 1024)), false);
  // A four-byte character is one UTF-16 pair: a length-only check would call this
  // 44 KiB and send it, and the request would be refused by the browser.
  assert.equal(isUnderKeepaliveBudget("\u{1f600}".repeat(22 * 1024)), false);
});
