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
    /if \(resumed && pendingNewThreadId\) \{[\s\S]*?useChatRuntimeStore\.getState\(\)\.setActiveThreadId\(pendingNewThreadId\);\s*return;/,
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

test("a nonce only owns a thread its own switch opened", () => {
  const switchSource = componentSource(provider, "function ThreadNewChatSwitch(");
  // The record is the thing the reopen runs on. Taking it from whatever thread happens to
  // be current recorded the chat the user came FROM, because that chat's own claim was
  // retired while it was on screen and an unclaimed current thread looks like an arrival.
  assert.match(
    switchSource,
    /if \(mainThreadId && switchState\.landedAttempt === switchState\.attempt\) \{\s*switchState\.nonceThread = \{ nonce, threadId: mainThreadId \};/,
  );
  // ...and the only thing that sets landedAttempt is that attempt's own switch resolving.
  assert.match(
    switchSource,
    /if \(switchStateNow\.attempt === attempt\) \{\s*switchStateNow\.landedAttempt = attempt;/,
  );
});

test("the remembered thread is looked up defensively", () => {
  const switchSource = componentSource(provider, "function ThreadNewChatSwitch(");
  // getItemById throws for an id the store has dropped rather than returning undefined, so
  // the optional chain is not the guard it looks like, and an effect that throws with no
  // error boundary above it takes the app down. The id here is remembered across every view
  // switch, which is exactly the kind that can go stale.
  assert.match(
    switchSource,
    /try \{\s*recordedRemoteId = runtimeThreads\?\.threads\s*\.getItemById\(recorded\)\s*\.getState\(\)\?\.remoteId;\s*\} catch \{/,
  );
});

test("every active-thread publication stands down while backgrounded", () => {
  // ThreadBackendAutosave's publication is gated, and so is the history adapter's sibling
  // inside append() -- which is the one a background run reaches when its assistant message
  // is persisted during compare. Ungated, a hidden pane names itself active and compare's
  // exportThreadIds ([model1, model2, activeThreadId]) picks up the unrelated base chat.
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
  // The refs reach the adapter without joining the memo's dependencies: a new runtime-hook
  // identity would rebuild the runtime, and not rebuilding it is the whole point.
  assert.match(
    provider,
    /createRuntimeHook\(\s*modelType,\s*pairId,\s*initialThreadId,\s*onInitialHistoryReady,\s*backgroundedRef,\s*newThreadSwitchStateRef,\s*\),\s*\[initialThreadId, modelType, onInitialHistoryReady, pairId\],/,
  );
});

test("a publication landing mid-switch does not reclaim the view", () => {
  // switchToNewThread() is async, so mainThreadId is still the OUTGOING thread until it
  // resolves, and both publications read it as proof that "this pane is on screen". A write
  // landing in that gap republishes the chat the user just left, and the project view (opened
  // with no active thread) renders it inside the new project. attempt !== landedAttempt is
  // that window.
  const guards =
    provider.match(
      /const switchInFlight[\s\S]{0,240}?switchState\.landedAttempt !== switchState\.attempt/g,
    ) ?? [];
  assert.equal(guards.length, 2, "both publications need the same stand-down");
  for (const guard of guards) {
    assert.match(guard, /activeNonce !== null/);
  }
  // Both read the SAME ref the switch components mutate, so the two never disagree.
  assert.match(
    provider,
    /<ThreadBackendAutosave[\s\S]*?newThreadSwitchStateRef=\{newThreadSwitchStateRef\}[\s\S]*?\/>/,
  );
});

test("the landing does not restore a chat that was deleted while it was away", () => {
  // Retaining the chat across the detour is what this PR added; at the merge base the landing
  // unmounted and there was nothing to restore. Nothing else clears the retained id: it is
  // component state, the sidebar deletes globally without reaching it, and compare does not
  // hold the chat as the store's active id. So a chat deleted during compare came back on
  // screen with a composer pointed at it.
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
  // Falling through rather than returning: the rotate below is what leaves the landing
  // with a fresh thread to send into instead of a dangling retained id.
  assert.doesNotMatch(guard, /isChatThreadDeleted\(pendingNewThreadId\)\) \{\s*return;/);
  assert.match(page, /import \{ isChatThreadDeleted \} from "\.\/utils\/chat-thread-tombstones";/);
});

test("no restore path puts a deleted chat back on screen", () => {
  // Three places put a retained id back: ProjectLanding's resume effect (pinned above),
  // NonceThreadResumeRestore, and the remembered-nonce reopen. All three ask the RUNTIME
  // whether it still knows the thread, and Unsloth deletes by tombstoning storage rather than
  // calling runtime.threads.delete(), so all three still answer yes after a delete. Guarding
  // one fixes nothing.
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
  // The adapter is rebuilt every render and handed to the core via __internal_setOptions, so
  // initialize() reads the provider's LATEST projectId. send() awaits every incomplete
  // attachment before handleSend and Unsloth's PDF/DOCX/text adapters extract there, so with
  // the provider surviving a project switch a document send materializes in whichever project
  // is on screen by then.
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    thread,
    /claimThreadCreation\([\s\S]*?\);\s*aui\.composer\(\)\.send\(\);/,
    "the stamp has to be taken before send() starts awaiting, not after",
  );
  // The prompt queue initializes its own fresh thread and never passes the composer, so it
  // has to stamp too, from what it captured when the queue started.
  assert.match(
    thread,
    /if \(initializingFreshThread\) \{\s*claimThreadCreation\(\[state\.id, state\.remoteId\], \{\s*projectId: projectIdAtQueueStart,\s*incognito: incognitoAtQueueStart,/,
  );
  // Per SEND, not per thread creation: switchToNewThread() reuses an untouched blank
  // thread, so the same local id can be the current new thread in two views in a row.
  assert.doesNotMatch(provider, /claimThreadCreation\(/);
  // Every field, not just the project. ChatPage's view effect clears `incognito` on the way
  // into a project, so a document sent from a Temporary Chat was persisted as a normal one.
  assert.match(provider, /const claim = readThreadCreationClaim\(threadId\);/);
  for (const field of [
    /const incognitoAtInit = claim \? claim\.incognito : runtimeStateAtInit\.incognito;/,
    /const modelIdAtInit = claim\s*\? claim\.modelId/,
    /const createdAtInit = claim \? claim\.createdAt : Date\.now\(\);/,
    /const projectIdAtInit = claim \? claim\.projectId : projectId;/,
  ]) {
    assert.match(provider, field);
  }
  // A claim OF null/false must win over what the store holds now; `claim?.x ?? store` would
  // read it as no claim at all.
  assert.doesNotMatch(provider, /claim\?\.(projectId|incognito|modelId|createdAt) \?\?/);

  // The RUN resolves its project separately from the record write, and takes the run's
  // instructions, RAG sources and sandbox from it. It reads the same stamp.
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    adapter,
    /const creationClaim = unstable_threadId\s*\? readThreadCreationClaim\(unstable_threadId\)\s*: undefined;\s*const composerProjectIdAtSend = creationClaim\s*\? creationClaim\.projectId/,
  );
  // ...and the claim has to outlive initialize(), because there is no ordering guarantee
  // between the two readers. Consuming it on the first read starves the second.
  assert.doesNotMatch(provider, /releaseThreadCreationClaim/);
  const claimModule = readFileSync(
    new URL("../src/features/chat/utils/chat-thread-creation-claim.ts", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(claimModule, /export function releaseThreadCreationClaim/);
});

test("compare lists its threads once before it waits on any run", () => {
  // Two constraints one boolean cannot carry. The wait has to be GLOBAL, because a first
  // compare run starts before either thread exists and files its handles under "__default";
  // and it has to exist at all, because these ids feed ComparePane's `initialThreadId`, so
  // learning them mid-run points ThreadAutoSwitch at a generating thread. But the shared
  // provider keeps a base chat's run alive across the switch into compare, so a global wait
  // on the FIRST list left an existing compare on blank runtimes with no later edge to
  // recover on. That list has nothing to clobber, so only re-lists wait.
  const waits =
    page.match(
      /const anyRunning = useChatRuntimeStore\(\s*\(s\) => Object\.keys\(s\.runningByThreadId\)\.length > 0,\s*\);/g,
    ) ?? [];
  assert.equal(waits.length, 2, "both compare variants share the global wait");
  const gates =
    page.match(
      /if \(anyRunning && listedPairRef\.current === pairId\) return;\s*listedPairRef\.current = pairId;/g,
    ) ?? [];
  assert.equal(gates.length, 2, "both variants must exempt their first list");
  assert.equal(
    (page.match(/\}, \[pairId, anyRunning\]\);/g) ?? []).length,
    2,
    "the settle edge is what re-lists; without it a fresh pair never learns its ids",
  );
});
