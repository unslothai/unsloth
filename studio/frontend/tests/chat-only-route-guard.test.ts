// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The chat-only route guard in __root.tsx, run rather than pattern-matched. Its own tests read
// the file as text (the module is .tsx and pulls in the whole app, so it is not importable
// here), which cannot answer the question that matters: given a host and a path, does the guard
// redirect? Lift the constants, the two predicates and the beforeLoad condition out of the
// source and evaluate them.
//
// The case this was written for: a measured chat-only host, meaning a CPU-only box, a Mac
// without usable MLX, or one with no PyTorch. `unmeasured` is false there, so the wait-it-out
// list does not apply, and /video used to fall through to the redirect. VideoPage carries the
// backend's own explanation for exactly those hosts (video_capability reports
// pytorch_not_installed / no_accelerator / macos_unsupported), so bouncing to /chat made the
// message unreachable in the only cases it exists for.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const src = await readFile(new URL("../src/app/routes/__root.tsx", import.meta.url), "utf8");

function lift(pattern: RegExp, what: string): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in __root.tsx`);
  return found[0];
}

// The only annotations in the lifted code are the two identical predicate signatures. Dropping
// them by hand keeps this a plain-node test; a changed signature fails loudly below, when the
// evaluated source refuses to parse.
const declarations = [
  lift(/const CHAT_ONLY_ALLOWED = new Set\(\[[\s\S]*?\n\]\);/, "CHAT_ONLY_ALLOWED"),
  lift(/const SELF_GATED_WHILE_UNKNOWN = \[[^\]]*\];/, "SELF_GATED_WHILE_UNKNOWN"),
  lift(/function waitsOutUnknownVerdict\([\s\S]*?\n\}/, "waitsOutUnknownVerdict"),
  lift(/function isChatOnlyAllowed\([\s\S]*?\n\}/, "isChatOnlyAllowed"),
]
  .join("\n")
  .replaceAll("(pathname: string): boolean", "(pathname)");

// The guard itself, with its two inputs made into parameters: the store call becomes the
// host verdict, the router's location becomes the path under test.
const guard = /if \(\s*isChatOnly\(\) &&([\s\S]*?)\)\s*\{\s*throw redirect/.exec(src);
assert.ok(guard, "could not find the chat-only redirect in beforeLoad");
const condition = `isChatOnly() &&${guard[1]}`
  .replaceAll("isChatOnly()", "chatOnly")
  .replaceAll("location.pathname", "pathname");
// Both inputs have to have been substituted, or the evaluated guard is not the shipped one.
assert.ok(!condition.includes("location."), "the guard reads a location this test cannot set");
assert.ok(!condition.includes("isChatOnly()"), "the guard reads a verdict this test cannot set");

const redirectsToChat = new Function(
  "pathname",
  "chatOnly",
  "unmeasured",
  `${declarations}\nreturn Boolean(${condition});`,
) as (pathname: string, chatOnly: boolean, unmeasured: boolean) => boolean;

// A host that has answered: chat_only true, nothing left to wait for.
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

// The Train page has no equivalent message: it is the training wizard or nothing, and
// StudioPage navigates to /chat itself the moment it reads a measured chat-only verdict. Being
// allowed in would buy a flash of the wizard and a lazy chunk before the same exit. The reason
// is on the sidebar row's tooltip instead. Keep the redirect; this pins the asymmetry as chosen.
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
  // Allowed for the same reason /video now is: each explains itself instead of vanishing.
  for (const path of ["/chat", "/export", "/images", "/api-monitor", "/data-recipes"]) {
    assert.equal(measuredChatOnly(path), false, `${path} no longer survives the guard`);
  }
  // Nothing was widened past the paths that opt in.
  for (const path of ["/settings", "/videos", "/videoish"]) {
    assert.equal(measuredChatOnly(path), true, `${path} slipped through the chat-only guard`);
  }
});

test("a host that is not chat-only is never redirected", () => {
  for (const path of ["/studio", "/video", "/settings"]) {
    assert.equal(redirectsToChat(path, false, false), false);
  }
});
