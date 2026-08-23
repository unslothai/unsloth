// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8908: a chat started inside a project ran under ProjectLanding's own
// ChatRuntimeProvider, so leaving the project view mid-response unmounted the runtime,
// useLocalRuntime's cleanup called detach(), and the backend cancelled on the disconnect.
// The fix is structural and invisible until somebody navigates mid-run, so the shape is
// pinned here: one provider above the project/single switch, no key, compare a sibling
// rather than a replacement.

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
  // Both closers: a plain declaration ends at a column-0 "}", a memo() wrapper at "});".
  // Anything nested is indented, so neither matches early.
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

  // ComparePane keeps its own: two panes need two runtimes, and
  // useRemoteThreadListRuntime throws when providers nest.
  const comparePane = componentSource(page, "function ComparePane({");
  assert.equal(
    (comparePane.match(/<ChatRuntimeProvider/g) ?? []).length,
    1,
    "ComparePane owns exactly one",
  );

  // The two views the bug was about must not build one, or switching between them
  // remounts the runtime and cuts the run off again.
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
  // A key is indistinguishable from remounting: by project (the state before the fix),
  // by thread, or by nonce all restore #8908.
  const openingTags = page.match(/<ChatRuntimeProvider[\s\S]*?\n\s*>/g) ?? [];
  assert.equal(openingTags.length, 2);
  for (const tag of openingTags) {
    assert.equal(tag.includes("key="), false, `keyed provider: ${tag}`);
  }

  // The project's new-chat nonce is owned by ChatPage, not ProjectLanding: the latter
  // lives under the provider, so a nonce held there would be invisible to it.
  assert.match(
    page,
    /const \[projectNewThreadNonce, setProjectNewThreadNonce\] = useState\(/,
  );
});

test("compare hides the shared provider instead of unmounting it", () => {
  // Rendering CompareContent in the provider's place unmounts it: the same detach()/cancel
  // path as #8908, reachable now that a run survives the navigation before a compare open.
  assert.match(page, /const baseBackgrounded = view\.mode === "compare";/);
  assert.match(page, /inert=\{baseBackgrounded \|\| undefined\}/);
  assert.match(page, /\{view\.mode === "compare" \? \(\s*<CompareContent/);
  assert.equal(
    /\) : \(\s*<CompareContent/.test(page),
    false,
    "compare must be a sibling of the provider, not its alternative",
  );

  // Hidden, it must not drive the shared single-chat state the compare view is using.
  assert.match(page, /backgrounded=\{baseBackgrounded\}/);
  for (const gated of [
    /<ActiveThreadSync\s+enabled=\{[\s\S]*?!backgrounded\s*\}/,
    /<ThreadScopedSettingsSync\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ActiveBranchRegistrar\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ThreadContextUsageRecount\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ThreadNewChatSwitch[\s\S]*?nonce=\{newThreadNonce\}[\s\S]*?paused=\{backgrounded\}[\s\S]*?\/>/,
    // The saved-thread switch stands down too: its first effect calls
    // requestTemporaryPromptQueueStop(), which names every temporary queue on the page
    // rather than this provider's, so a hidden pane reaching it would stop a queue the
    // compare view owns. syncActiveThreadId is a dependency, so the effect re-runs on
    // every compare open and close.
    /<ThreadAutoSwitch[\s\S]*?paused=\{backgrounded\}[\s\S]*?\/>/,
  ]) {
    assert.match(provider, gated);
  }

  const savedThreadSwitch = componentSource(
    provider,
    "function ThreadAutoSwitch({",
  );
  assert.match(savedThreadSwitch, /if \(isLoading \|\| paused\) \{/);
});

test("a switch that never opens releases its nonce", () => {
  // switchToNewThread() marks the nonce served before it resolves, so a rejection would
  // leave the guard reading it as handled and the same New Chat could never be retried in
  // place. Both arms are handled, so the rejection is never unhandled either.
  const switchSource = componentSource(
    provider,
    "function ThreadNewChatSwitch({",
  );
  // Either arm of the returning-nonce choice: a nonce coming back to a chat it already
  // owns reopens that thread instead of minting, and both go through the same handled
  // .then(ok, err) pair. What this pins is that neither is left unhandled.
  assert.match(
    switchSource,
    /void Promise\.resolve\([\s\S]*?aui\.threads\(\)\.switchToNewThread\(\),?\s*\)\.then\(/,
  );
  assert.match(
    switchSource,
    /returningToOwnChat && recorded\s*\?\s*aui\.threads\(\)\.switchToThread\(recorded\)/,
  );
  // Keyed by attempt as well as nonce: leaving for a saved chat releases the nonce, so
  // two switches for one nonce can overlap and the older must not release the newer's
  // thread.
  assert.match(
    switchSource,
    /if \(\s*switchStateNow\.attempt === attempt &&\s*switchStateNow\.activeNonce === nonce\s*\) \{\s*switchStateNow\.activeNonce = null;\s*\}/,
  );
  assert.match(switchSource, /const attempt = switchState\.attempt \+ 1;/);
  // clearAttachments() removes each staged file through the attachment adapter, so the
  // promise needs handling and not just the synchronous call.
  assert.match(
    switchSource,
    /void Promise\.resolve\(aui\.composer\(\)\.clearAttachments\(\)\)\.catch\(\s*\(\) => undefined,\s*\);/,
  );
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
  // switchToNewThread() reuses the uninitialized new thread rather than minting one, and
  // the composer belongs to that thread. With the provider shared it is the same composer
  // across a project switch, so an unsent attachment would land in the next project's chat.
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
    /const newThreadSwitchStateRef = useRef<NewThreadSwitchState>\(\{\s*activeNonce: null,\s*hasSwitched: false,\s*attempt: 0,\s*pendingSavedThreadIds: \[\],\s*nonceThread: null,\s*\}\);/,
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
  // ThreadNewChatSwitch is an earlier sibling, so on an already-mounted provider its
  // effect reaches setActiveThreadId(null) before ProjectLanding's effects run. Read in
  // an effect, the guard below would compare against null forever and Back into a project
  // would swap the landing for the chat the user just left.
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
