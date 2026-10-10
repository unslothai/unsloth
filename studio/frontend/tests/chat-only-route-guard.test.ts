// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Evaluates the lifted guard from __root.tsx (not importable here) instead of pattern-matching it.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const src = await readSrcAsync("app/routes/__root.tsx");

function lift(pattern: RegExp, what: string): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in __root.tsx`);
  return found[0];
}

// The predicate signatures are stripped by hand; a changed signature fails to parse below.
const declarations = [
  lift(/const CHAT_ONLY_ALLOWED = new Set\(\[[\s\S]*?\n\]\);/, "CHAT_ONLY_ALLOWED"),
  lift(/const SELF_GATED_WHILE_UNKNOWN = \[[^\]]*\];/, "SELF_GATED_WHILE_UNKNOWN"),
  lift(/function waitsOutUnknownVerdict\([\s\S]*?\n\}/, "waitsOutUnknownVerdict"),
  lift(/function isChatOnlyAllowed\([\s\S]*?\n\}/, "isChatOnlyAllowed"),
]
  .join("\n")
  .replaceAll("(pathname: string): boolean", "(pathname)");

const guard = /if \(\s*isChatOnly\(\) &&([\s\S]*?)\)\s*\{\s*throw redirect/.exec(src);
assert.ok(guard, "could not find the chat-only redirect in beforeLoad");
const condition = `isChatOnly() &&${guard[1]}`
  .replaceAll("isChatOnly()", "chatOnly")
  .replaceAll("location.pathname", "pathname");
assert.ok(!condition.includes("location."), "the guard reads a location this test cannot set");
assert.ok(!condition.includes("isChatOnly()"), "the guard reads a verdict this test cannot set");

const redirectsToChat = new Function(
  "pathname",
  "chatOnly",
  "unmeasured",
  `${declarations}\nreturn Boolean(${condition});`,
) as (pathname: string, chatOnly: boolean, unmeasured: boolean) => boolean;

const measuredChatOnly = (pathname: string) => redirectsToChat(pathname, true, false);

test("a measured chat-only host reaches /video and its own explanation", () => {
  assert.equal(
    measuredChatOnly("/video"),
    false,
    "a direct link or a reload at /video bounces to /chat, so the no-GPU, no-PyTorch and " +
      "macOS explanations on VideoPage are unreachable on every host that has one",
  );
  assert.equal(
    measuredChatOnly("/video/anything"),
    false,
    "only the exact path is allowed through, so a child route still bounces",
  );
});

// The Train page has no capability message, so /studio still redirects on chat-only hosts.
test("a measured chat-only host is still redirected off /studio", () => {
  assert.equal(measuredChatOnly("/studio"), true);
  assert.equal(measuredChatOnly("/studio/runs"), true);
});

test("an unmeasured verdict still lets both pages wait it out", () => {
  for (const path of ["/studio", "/studio/runs", "/video"]) {
    assert.equal(
      redirectsToChat(path, true, true),
      false,
      `${path} is redirected on the pre-measurement guess, which is one-way`,
    );
  }
});

test("the pages that self-gate are unaffected, and everything else still redirects", () => {
  for (const path of ["/chat", "/export", "/images", "/api-monitor", "/data-recipes"]) {
    assert.equal(measuredChatOnly(path), false, `${path} no longer survives the guard`);
  }
  for (const path of ["/settings", "/videos", "/videoish"]) {
    assert.equal(measuredChatOnly(path), true, `${path} slipped through the chat-only guard`);
  }
});

test("a host that is not chat-only is never redirected", () => {
  for (const path of ["/studio", "/video", "/settings"]) {
    assert.equal(redirectsToChat(path, false, false), false);
  }
});
