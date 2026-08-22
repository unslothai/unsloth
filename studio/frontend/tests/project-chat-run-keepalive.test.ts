// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8908: a chat started inside a project ran under ProjectLanding's own
// ChatRuntimeProvider, so leaving the project view mid-response unmounted the
// runtime, useLocalRuntime's cleanup called detach(), and the backend cancelled
// on the disconnect. The fix is structural and invisible at runtime until
// somebody navigates mid-run, so the shape is pinned here: one provider above
// the project/single switch, carrying no key, with compare a sibling rather
// than a replacement.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const page = readFileSync(
  new URL("../src/features/chat/chat-page.tsx", import.meta.url),
  "utf8",
);
const provider = readFileSync(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
  "utf8",
);

/** The source of one component, from its declaration to the next one. */
function componentSource(source: string, declaration: string): string {
  const start = source.indexOf(declaration);
  assert.notEqual(start, -1, `${declaration} not found`);
  // Both closers: a plain declaration ends at a column-0 "}", a memo() wrapper
  // at "});". Anything nested is indented, so neither matches early.
  const ends = ["\n}\n", "\n});\n"]
    .map((closer) => source.indexOf(closer, start))
    .filter((index) => index !== -1);
  assert.notEqual(ends.length, 0, `end of ${declaration} not found`);
  return source.slice(start, Math.min(...ends));
}

test("one runtime provider sits above the project/single switch", () => {
  const mounts = page.match(/<ChatRuntimeProvider/g) ?? [];
  assert.equal(
    mounts.length,
    2,
    "one shared provider plus ComparePane's own; a third means a view built its own again",
  );

  // ComparePane must keep its own: two panes need two runtimes, and
  // useRemoteThreadListRuntime throws when providers nest.
  const comparePane = componentSource(page, "function ComparePane({");
  assert.equal(
    (comparePane.match(/<ChatRuntimeProvider/g) ?? []).length,
    1,
    "ComparePane owns exactly one",
  );

  // The two views the bug was about must not build one, or switching between
  // them remounts the runtime and cuts the run off again.
  for (const declaration of [
    "const SingleContent = memo(function SingleContent({",
    "function ProjectLanding({",
  ]) {
    assert.equal(
      componentSource(page, declaration).includes("<ChatRuntimeProvider"),
      false,
      `${declaration} must render under the shared provider, not build one`,
    );
  }
});

test("the shared provider is never keyed", () => {
  // A key on it is indistinguishable from remounting it: keying by project
  // (the state before the fix), by thread, or by nonce all restore #8908.
  const openingTags = page.match(/<ChatRuntimeProvider[\s\S]*?\n\s*>/g) ?? [];
  assert.equal(openingTags.length, 2);
  for (const tag of openingTags) {
    assert.equal(tag.includes("key="), false, `keyed provider: ${tag}`);
  }

  // The project's new-chat nonce is owned by ChatPage, not ProjectLanding:
  // ProjectLanding lives under the provider, so a nonce held there would be
  // one the provider could not see.
  assert.match(
    page,
    /const \[projectNewThreadNonce, setProjectNewThreadNonce\] = useState\(/,
  );
});

test("compare hides the shared provider instead of unmounting it", () => {
  // Rendering CompareContent in the provider's place unmounts it, which is the
  // same detach()/cancel path as #8908 -- reachable now that a run survives the
  // navigation that precedes opening a compare chat.
  assert.match(page, /const baseBackgrounded = view\.mode === "compare";/);
  assert.match(page, /inert=\{baseBackgrounded \|\| undefined\}/);
  assert.match(page, /\{view\.mode === "compare" \? \(\s*<CompareContent/);
  assert.equal(
    /\) : \(\s*<CompareContent/.test(page),
    false,
    "compare must be a sibling of the provider, not its alternative",
  );

  // Hidden, it must not drive the shared single-chat state the compare view is
  // using: the active thread, the context bar, the thread-scoped settings.
  assert.match(page, /backgrounded=\{baseBackgrounded\}/);
  for (const gated of [
    /<ActiveThreadSync\s+enabled=\{[\s\S]*?!backgrounded\s*\}/,
    /<ThreadScopedSettingsSync\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ActiveBranchRegistrar\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ThreadContextUsageRecount\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ThreadNewChatSwitch[\s\S]*?nonce=\{newThreadNonce\}[\s\S]*?paused=\{backgrounded\}[\s\S]*?\/>/,
  ]) {
    assert.match(provider, gated);
  }
});

test("compare preserves a materialized project chat", () => {
  const landing = componentSource(page, "function ProjectLanding({");
  assert.match(landing, /const wasActiveRef = useRef\(active\);/);
  assert.match(
    landing,
    /const resumed = active && !wasActiveRef\.current;\s*wasActiveRef\.current = active;\s*if \(!active\) \{\s*return;\s*\}/,
  );
  assert.match(
    landing,
    /if \(resumed && pendingNewThreadId\) \{\s*useChatRuntimeStore\.getState\(\)\.setActiveThreadId\(pendingNewThreadId\);\s*return;/,
  );
});

test("a staged attachment does not follow the user into the next view", () => {
  // switchToNewThread() reuses the uninitialized new thread rather than minting
  // one, and the composer belongs to that thread. With the provider shared, that
  // is literally the same composer across a project switch, so an unsent
  // attachment would be filed into a chat under the next project.
  assert.match(
    provider,
    /const switchState = newThreadSwitchStateRef\.current;\s*if \(switchState\.activeNonce === nonce\) \{\s*return;\s*\}/,
  );
  assert.match(
    provider,
    /const clearAfterSwitch =[\s\S]{0,160}?switchState\.activeNonce === null;/,
  );
  const switchSource = componentSource(
    provider,
    "function ThreadNewChatSwitch({",
  );
  assert.equal(
    switchSource.includes("useRef<NewThreadSwitchState>"),
    false,
    "the nonce guard must outlive ThreadNewChatSwitch mounts",
  );
  const runtimeProvider = componentSource(
    provider,
    "export function ChatRuntimeProvider({",
  );
  assert.match(
    runtimeProvider,
    /const newThreadSwitchStateRef = useRef<NewThreadSwitchState>\(\{\s*activeNonce: null,\s*hasSwitched: false,\s*\}\);/,
  );
  assert.match(
    runtimeProvider,
    /if \(!initialThreadId && !newThreadNonce\) \{\s*newThreadSwitchStateRef\.current\.hasSwitched = true;\s*\}/,
  );
  assert.match(
    runtimeProvider,
    /<ThreadNewChatSwitch[\s\S]*?newThreadSwitchStateRef=\{newThreadSwitchStateRef\}[\s\S]*?\/>/,
  );
  const savedThreadSwitch = componentSource(
    provider,
    "function ThreadAutoSwitch({",
  );
  assert.match(
    savedThreadSwitch,
    /newThreadSwitchStateRef\.current\.activeNonce = null;/,
  );
  assert.match(
    runtimeProvider,
    /<ThreadAutoSwitch[\s\S]*?newThreadSwitchStateRef=\{newThreadSwitchStateRef\}[\s\S]*?\/>/,
  );
});

test("the outgoing thread id is captured before the provider blanks it", () => {
  // ThreadNewChatSwitch is an earlier sibling of the provider's children, so on
  // an already-mounted provider its effect reaches setActiveThreadId(null)
  // before ProjectLanding's effects run. Read in an effect, the guard below
  // would compare against null forever and Back into a project would swap the
  // landing for the chat the user just left.
  assert.match(
    page,
    /const \[initialActiveThreadId\] = useState\(\s*\(\) => useChatRuntimeStore\.getState\(\)\.activeThreadId,\s*\);/,
  );
  assert.match(
    page,
    /if \(\s*activeThreadId === initialActiveThreadId \|\|\s*activeThreadId === pendingNewThreadId\s*\) \{/,
  );
  assert.equal(
    page.includes("initialActiveThreadRef"),
    false,
    "no effect-assigned ref left behind",
  );
});
