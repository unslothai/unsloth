// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One shared provider above the project/single switch: unmounting it detaches the runtime and
// the backend cancels the run.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const page = readSrc("features/chat/chat-page.tsx");
const provider = readSrc("features/chat/runtime-provider.tsx");

function componentSource(source: string, declaration: string): string {
  const start = source.indexOf(declaration);
  assert.notEqual(start, -1, `${declaration} not found`);
  // Plain declarations end at column-0 "}", memo() wrappers at "});".
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

  // ComparePane keeps its own: useRemoteThreadListRuntime throws when providers nest.
  const comparePane = componentSource(page, "function ComparePane({");
  assert.equal(
    (comparePane.match(/<ChatRuntimeProvider/g) ?? []).length,
    1,
    "ComparePane owns exactly one",
  );

  // Building one per view would remount the runtime and cancel the run on switch.
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
  // Any key on the provider is equivalent to remounting it.
  const openingTags = page.match(/<ChatRuntimeProvider[\s\S]*?\n\s*>/g) ?? [];
  assert.equal(openingTags.length, 2);
  for (const tag of openingTags) {
    assert.equal(tag.includes("key="), false, `keyed provider: ${tag}`);
  }

  // The nonce lives in ChatPage, since ProjectLanding is under the provider.
  assert.match(
    page,
    /const \[projectNewThreadNonce, setProjectNewThreadNonce\] = useState\(/,
  );
});

test("compare hides the shared provider instead of unmounting it", () => {
  // Rendering CompareContent in the provider's place would unmount it and cancel the run.
  assert.match(page, /const baseBackgrounded = view\.mode === "compare";/);
  assert.match(page, /inert=\{baseBackgrounded \|\| undefined\}/);
  assert.match(page, /\{view\.mode === "compare" \? \(\s*<CompareContent/);
  assert.equal(
    /\) : \(\s*<CompareContent/.test(page),
    false,
    "compare must be a sibling of the provider, not its alternative",
  );

  assert.match(page, /backgrounded=\{baseBackgrounded\}/);
  for (const gated of [
    /<ActiveThreadSync\s+enabled=\{[\s\S]*?!backgrounded\s*\}/,
    /<ThreadScopedSettingsSync\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ActiveBranchRegistrar\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ThreadContextUsageRecount\s+enabled=\{[^}]*!backgrounded\s*\}/,
    /<ThreadNewChatSwitch[\s\S]*?nonce=\{newThreadNonce\}[\s\S]*?paused=\{backgrounded\}[\s\S]*?\/>/,
    // requestTemporaryPromptQueueStop() targets every queue on the page, so a hidden pane must pause.
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
  // The nonce is marked served before resolving, so a rejection must be handled to allow retry.
  const switchSource = componentSource(
    provider,
    "function ThreadNewChatSwitch({",
  );
  assert.match(
    switchSource,
    /void Promise\.resolve\([\s\S]*?aui\.threads\(\)\.switchToNewThread\(\),?\s*\)\.then\(/,
  );
  assert.match(
    switchSource,
    /returningToOwnChat && recorded\s*\?\s*aui\.threads\(\)\.switchToThread\(recorded\)/,
  );
  // Keyed by attempt too: overlapping switches for one nonce must not release each other's thread.
  assert.match(
    switchSource,
    /if \(\s*switchStateNow\.attempt === attempt &&\s*switchStateNow\.activeNonce === nonce\s*\) \{\s*switchStateNow\.activeNonce = null;\s*\}/,
  );
  assert.match(switchSource, /const attempt = switchState\.attempt \+ 1;/);
  // clearAttachments() is async per file, so its promise needs handling.
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
    /if \(resumed && pendingNewThreadId\) \{[\s\S]*?useChatRuntimeStore\.getState\(\)\.setActiveThreadId\(pendingNewThreadId\);\s*return;/,
  );
});

test("a staged attachment does not follow the user into the next view", () => {
  // The shared composer survives project switches, so unsent attachments would leak across.
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
    /const newThreadSwitchStateRef = useRef<NewThreadSwitchState>\(\{\s*activeNonce: null,\s*hasSwitched: false,\s*attempt: 0,\s*pendingSavedThreadIds: \[\],\s*nonceThread: null,\s*landedAttempt: 0,\s*\}\);/,
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
  // ThreadNewChatSwitch's effect nulls the active id before ProjectLanding's effects run.
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

test("a nonce only owns a thread its own switch opened", () => {
  const switchSource = componentSource(provider, "function ThreadNewChatSwitch(");
  // The previous chat's claim is retired while on screen, so current thread is not the arrival.
  assert.match(
    switchSource,
    /if \(mainThreadId && switchState\.landedAttempt === switchState\.attempt\) \{\s*switchState\.nonceThread = \{ nonce, threadId: mainThreadId \};/,
  );
  assert.match(
    switchSource,
    /if \(switchStateNow\.attempt === attempt\) \{\s*switchStateNow\.landedAttempt = attempt;/,
  );
});

test("the remembered thread is looked up defensively", () => {
  const switchSource = componentSource(provider, "function ThreadNewChatSwitch(");
  // getItemById throws for a dropped id, and there is no error boundary above this effect.
  assert.match(
    switchSource,
    /try \{\s*recordedRemoteId = runtimeThreads\?\.threads\s*\.getItemById\(recorded\)\s*\.getState\(\)\?\.remoteId;\s*\} catch \{/,
  );
});

test("every active-thread publication stands down while backgrounded", () => {
  // An ungated hidden pane would publish itself as active and leak into compare exports.
  const publications = provider.match(/setActiveThreadId\(/g) ?? [];
  assert.ok(publications.length >= 6, "expected the publications to still be here");
  assert.match(
    provider,
    /!backgroundedRef\?\.current &&\s*!switchInFlight\s*\) \{[\s\S]*?store\.setActiveThreadId\(remoteId\);/,
  );
  assert.match(
    provider,
    /!backgroundedRef\.current &&\s*!switchInFlight\s*\) \{[\s\S]*?store\.setActiveThreadId\(remoteId\);/,
  );
  // Refs avoid memo deps: a new hook identity would rebuild the runtime.
  assert.match(
    provider,
    /createRuntimeHook\(\s*modelType,\s*pairId,\s*initialThreadId,\s*onInitialHistoryReady,\s*backgroundedRef,\s*newThreadSwitchStateRef,\s*\),\s*\[initialThreadId, modelType, onInitialHistoryReady, pairId\],/,
  );
});

test("a publication landing mid-switch does not reclaim the view", () => {
  // mainThreadId is still the outgoing thread until the async switch resolves.
  const guards =
    provider.match(
      /const switchInFlight[\s\S]{0,240}?switchState\.landedAttempt !== switchState\.attempt/g,
    ) ?? [];
  assert.equal(guards.length, 2, "both publications need the same stand-down");
  for (const guard of guards) {
    assert.match(guard, /activeNonce !== null/);
  }
  assert.match(
    provider,
    /<ThreadBackendAutosave[\s\S]*?newThreadSwitchStateRef=\{newThreadSwitchStateRef\}[\s\S]*?\/>/,
  );
});

test("the landing does not restore a chat that was deleted while it was away", () => {
  // Nothing else clears the retained id, so a chat deleted during compare would reappear.
  const restore = page.slice(
    page.indexOf("const resumed = active && !wasActiveRef.current;"),
  );
  const guard = restore.slice(0, restore.indexOf("// Leaving a created chat"));
  assert.match(guard, /if \(!isChatThreadDeleted\(pendingNewThreadId\)\) \{/);
  assert.ok(
    guard.indexOf("isChatThreadDeleted(pendingNewThreadId)") <
      guard.indexOf("setActiveThreadId(pendingNewThreadId)"),
    "the check has to come before the restore it guards",
  );
  // Fall through so the rotate below leaves a fresh thread.
  assert.doesNotMatch(guard, /isChatThreadDeleted\(pendingNewThreadId\)\) \{\s*return;/);
  assert.match(page, /import \{ isChatThreadDeleted \} from "\.\/utils\/chat-thread-tombstones";/);
});

test("no restore path puts a deleted chat back on screen", () => {
  // Deletes tombstone storage rather than runtime.threads.delete(), so all three restore paths
  // need the guard.
  const restore = componentSource(provider, "function NonceThreadResumeRestore({");
  assert.match(restore, /if \(isChatThreadDeleted\(remoteId\)\) \{\s*return;\s*\}/);
  assert.ok(
    restore.indexOf("isChatThreadDeleted(remoteId)") <
      restore.indexOf("setActiveThreadId(mainThreadId)"),
    "the check has to come before the publication it guards",
  );
  assert.match(
    provider,
    /const returningToOwnChat = Boolean\(\s*recorded && recordedRemoteId && !isChatThreadDeleted\(recordedRemoteId\),\s*\);/,
  );
});

test("a delayed first send keeps the creation inputs it was sent under", () => {
  // initialize() reads the latest projectId after attachment extraction, so the send must be stamped.
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /claimThreadCreation\([\s\S]*?\);\s*aui\.composer\(\)\.send\(\);/,
    "the stamp has to be taken before send() starts awaiting, not after",
  );
  // The prompt queue never passes the composer, so it stamps from its start-time capture.
  assert.match(
    thread,
    /if \(initializingFreshThread\) \{\s*claimThreadCreation\(\[state\.id, state\.remoteId\], \{\s*projectId: projectIdAtQueueStart,\s*incognito: incognitoAtQueueStart,/,
  );
  // Per send: switchToNewThread() can reuse the same blank thread across views.
  assert.doesNotMatch(provider, /claimThreadCreation\(/);
  // Every field: ChatPage clears `incognito` on entering a project.
  assert.match(provider, /const claim = readThreadCreationClaim\(threadId\);/);
  for (const field of [
    /const incognitoAtInit = claim \? claim\.incognito : runtimeStateAtInit\.incognito;/,
    /const modelIdAtInit = claim\s*\? claim\.modelId/,
    /const createdAtInit = claim \? claim\.createdAt : Date\.now\(\);/,
    /const projectIdAtInit = claim \? claim\.projectId : projectId;/,
  ]) {
    assert.match(provider, field);
  }
  // A claim of null/false must win; `??` would treat it as no claim.
  assert.doesNotMatch(provider, /claim\?\.(projectId|incognito|modelId|createdAt) \?\?/);

  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /const creationClaim = unstable_threadId\s*\? readThreadCreationClaim\(unstable_threadId\)\s*: undefined;\s*const composerProjectIdAtSend = creationClaim\s*\? creationClaim\.projectId/,
  );
  // No ordering between the two readers, so the claim must outlive initialize().
  assert.doesNotMatch(provider, /releaseThreadCreationClaim/);
  const claimModule = readSrc("features/chat/utils/chat-thread-creation-claim.ts");
  assert.doesNotMatch(claimModule, /export function releaseThreadCreationClaim/);
});

test("compare lists its threads once before it waits on any run", () => {
  const globalWaits =
    page.match(
      /const anyRunning = useChatRuntimeStore\(\s*\(s\) => Object\.keys\(s\.runningByThreadId\)\.length > 0,\s*\);/g,
    ) ?? [];
  assert.equal(globalWaits.length, 1, "General Compare waits on every pane run");
  const localWaits =
    page.match(
      /const anyRunning = useChatRuntimeStore\(\s*\(s\) => Object\.keys\(s\.localRunByThreadId\)\.length > 0,\s*\);/g,
    ) ?? [];
  assert.equal(localWaits.length, 1, "LoRA Compare waits only on local runs");
  const gates =
    page.match(
      /if \(\(?anyRunning(?: \|\| comparing\))? && listedPairRef\.current === pairId\) return;\s*listedPairRef\.current = pairId;/g,
    ) ?? [];
  assert.equal(gates.length, 2, "both variants must exempt their first list");
  assert.equal(
    (page.match(/\}, \[pairId, anyRunning(?:, comparing)?\]\);/g) ?? []).length,
    2,
    "the settle edge is what re-lists; without it a fresh pair never learns its ids",
  );
});
