// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every thread row has carried modelId since long before this notice, and each send now
// moves it to the model that send ran on, so a chat knows the model it last used. What was
// missing was offering it back. The rules below are the ones that keep the offer from becoming a
// nuisance: it never loads anything on its own, it stays quiet whenever it could not be
// honoured, and a model still loaded alongside is switched to without asking.

import assert from "node:assert/strict";
import test from "node:test";

import {
  atDefaultUiScale,
  installLocalStorageFake,
  readText,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();
const {
  chatModelIsResident,
  chatModelLoadedTarget,
  chatModelSelectableId,
  chatModelSwitchMeta,
  createChatModelHistoryReader,
  resolveChatModelSwitchTarget,
  stepChatModelAutoSwitch,
} = await import("../src/features/chat/components/chat-model-notice-switch.ts");
const { modelDisplayName } = await import(
  "../src/features/hub/lib/model-identity.ts"
);
const { chatLocalModelOptions } = await import(
  "../src/features/chat/local-model-options.ts"
);
const {
  DEFAULT_PER_MODEL_CONFIG,
  resolveInitialConfig,
  resolveResidentInitialConfig,
  savePerModelConfig,
} = await import("../src/features/model-picker/model-config/per-model-config.ts");
const { shouldPersistRunModel } = await import(
  "../src/features/chat/utils/queued-chat-run-settings.ts"
);
const { wantsDownloadManagerStaging } = await import(
  "../src/features/chat/utils/model-download-staging.ts"
);

const notice = atDefaultUiScale(readText("../src/features/chat/components/chat-model-notice.tsx"));
const page = atDefaultUiScale(readText("../src/features/chat/chat-page.tsx"));
const runtimeProvider = atDefaultUiScale(readText("../src/features/chat/runtime-provider.tsx"));
const runtimeHook = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
const adapter = atDefaultUiScale(readText("../src/features/chat/api/chat-adapter.ts"));
const chatApi = atDefaultUiScale(readText("../src/features/chat/api/chat-api.ts"));
const types = atDefaultUiScale(readText("../src/features/chat/types.ts"));
const thread = atDefaultUiScale(readText("../src/components/assistant-ui/thread.tsx"));
const researchPanel = readText(
  "../src/features/chat/components/research-activity-panel.tsx",
);
const switchSource = readText(
  "../src/features/chat/components/chat-model-notice-switch.ts",
);

function switchBackSelection(
  target: { modelId: string; ggufVariant?: string | null },
  rows: Parameters<typeof chatModelSwitchMeta>[1] = [],
) {
  const meta = chatModelSwitchMeta(target, rows);
  return {
    id: target.modelId,
    source: meta?.source,
    isLora: meta?.isLora,
    ggufVariant: meta?.ggufVariant,
    isDownloaded: meta?.isDownloaded,
    isGguf: meta?.isGguf,
  };
}

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  assert.ok(start !== -1, `not found: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

// Source assertions: the notice reaches external-providers, which does not resolve in a
// bare node test. The sibling thread-scoped suites do the same for the same reason.

test("the notice never loads a model on its own", () => {
  // Opening a chat must not evict what is resident: a local load is multi-gigabyte.
  assert.doesNotMatch(notice, /loadModel|setCheckpoint/);
  // A model that is not loaded is only ever switched to from the button.
  assert.match(notice, /onClick=\{\(\) => onSwitch\(switchTarget\)\}/);
  // The one automatic switch is to a slot that is already loaded.
  const body = notice.slice(notice.indexOf("export function ChatModelNotice"));
  assert.match(body, /chatModelLoadedTarget\(createdModel, loadedModels\)/);
  assert.match(
    body,
    /onAutoSwitch\(\s*switchTo,\s*\(\) => autoSwitchRef\.current === state && createdKeyRef\.current === key,\s*\);/,
  );
});

test("a chat goes back to its model when that model is still loaded", () => {
  const slots = [
    { id: "unsloth/Model-A-GGUF", quant: "Q4_K_M", checkpoint: "unsloth/Model-A-GGUF" },
    { id: "Model-B", quant: null, checkpoint: "/models/Model-B" },
  ];
  assert.deepEqual(
    chatModelLoadedTarget(
      { modelId: "unsloth/Model-A-GGUF", ggufVariant: "Q4_K_M" },
      slots,
    ),
    { modelId: "unsloth/Model-A-GGUF", ggufVariant: "Q4_K_M" },
  );
  // A local model is loaded by its path, so the slot's checkpoint is what gets selected.
  assert.deepEqual(
    chatModelLoadedTarget({ modelId: "/models/Model-B", ggufVariant: null }, slots),
    { modelId: "/models/Model-B", ggufVariant: null },
  );
  // A chat saved without a variant takes the quant the slot is serving.
  assert.deepEqual(
    chatModelLoadedTarget({ modelId: "unsloth/Model-A-GGUF" }, slots),
    { modelId: "unsloth/Model-A-GGUF", ggufVariant: "Q4_K_M" },
  );
});

test("a chat is not switched to a loaded slot that would need another load", () => {
  const slots = [
    { id: "unsloth/Model-A-GGUF", quant: "Q4_K_M", checkpoint: "unsloth/Model-A-GGUF" },
  ];
  // Another quant of the same repo is a different load.
  assert.equal(
    chatModelLoadedTarget(
      { modelId: "unsloth/Model-A-GGUF", ggufVariant: "Q8_0" },
      slots,
    ),
    null,
  );
  // A slot whose quant is not known yet cannot vouch for the saved variant.
  assert.equal(
    chatModelLoadedTarget(
      { modelId: "unsloth/Model-A-GGUF", ggufVariant: "Q4_K_M" },
      [{ id: "unsloth/Model-A-GGUF", checkpoint: "unsloth/Model-A-GGUF" }],
    ),
    null,
  );
  // Nor a chat saved without a variant: the slot may serve another quant until its lookup lands.
  assert.equal(
    chatModelLoadedTarget(
      { modelId: "unsloth/Model-A-GGUF" },
      [{ id: "unsloth/Model-A-GGUF", checkpoint: "unsloth/Model-A-GGUF" }],
    ),
    null,
  );
  assert.equal(
    chatModelLoadedTarget({ modelId: "unsloth/Model-C" }, slots),
    null,
  );
  assert.equal(chatModelLoadedTarget({ modelId: "unsloth/Model-A-GGUF" }, []), null);
});

test("the automatic switch runs once per chat open", () => {
  const modelA = { modelId: "unsloth/Model-A-GGUF", ggufVariant: "Q4_K_M" };
  const open = (threadId: string | undefined) => ({
    threadId,
    active: true,
    resident: false,
    loadedTarget: modelA as typeof modelA | null,
    modelLoading: false,
  });
  // Opening chat A while Model B is selected switches to A's loaded slot, once.
  let step = stepChatModelAutoSwitch({ settled: false }, open("a"));
  assert.deepEqual(step.switchTo, modelA);
  step = stepChatModelAutoSwitch(step.state, open("a"));
  assert.equal(step.switchTo, null);
  // Going to chat B and back to A switches again.
  step = stepChatModelAutoSwitch(step.state, { ...open("b"), loadedTarget: null });
  step = stepChatModelAutoSwitch(step.state, open("a"));
  assert.deepEqual(step.switchTo, modelA);

  // Opened already on its model: a pick made afterwards inside the chat is not undone.
  step = stepChatModelAutoSwitch({ settled: false }, {
    ...open("a"),
    resident: true,
    loadedTarget: null,
  });
  step = stepChatModelAutoSwitch(step.state, open("a"));
  assert.equal(step.switchTo, null);

  // Waits for the saved model and the loaded slots to be known, then switches.
  step = stepChatModelAutoSwitch({ settled: false }, { ...open("a"), loadedTarget: null });
  assert.equal(step.state.settled, false);
  step = stepChatModelAutoSwitch(step.state, open("a"));
  assert.deepEqual(step.switchTo, modelA);

  // A load already in flight wins, and is not undone when it lands.
  step = stepChatModelAutoSwitch({ settled: false }, { ...open("a"), modelLoading: true });
  assert.equal(step.switchTo, null);
  step = stepChatModelAutoSwitch(step.state, open("a"));
  assert.equal(step.switchTo, null);

  // Leaving the chat route and coming back to the same chat is a new open, so it switches again.
  step = stepChatModelAutoSwitch({ settled: false }, open("a"));
  step = stepChatModelAutoSwitch(step.state, { ...open("a"), active: false });
  step = stepChatModelAutoSwitch(step.state, open("a"));
  assert.deepEqual(step.switchTo, modelA);

  // Off the chat route, or on an unsent New Chat, nothing happens.
  assert.equal(
    stepChatModelAutoSwitch({ settled: false }, { ...open("a"), active: false }).switchTo,
    null,
  );
  assert.equal(stepChatModelAutoSwitch({ settled: false }, open(undefined)).switchTo, null);

  const body = notice.slice(notice.indexOf("export function ChatModelNotice"));
  assert.match(
    body,
    /onAutoSwitch\(\s*switchTo,\s*\(\) => autoSwitchRef\.current === state && createdKeyRef\.current === key,\s*\);/,
  );
  // Any loaded slot counts, Keep models loaded on or off: the switch only ever adopts it.
  assert.match(body, /createdModel && !resident\s*\?\s*chatModelLoadedTarget/);
  assert.doesNotMatch(body, /keepModelsLoaded/);
  // A cross-tab revision carries no thread, so the saved model is read again rather than kept stale.
  // Only the plain cross-tab Event re-reads (streamed chunks are CustomEvents), and a slower read
  // never lands over a newer answer.
  assert.match(notice, /if \(event instanceof CustomEvent\) return;\s*reader\.invalidate\(\);\s*const generation = \+\+readGeneration;/);
  assert.match(notice, /if \(updatedThread\.id === threadId\) readGeneration \+= 1;/);
  assert.match(
    notice,
    /getStoredChatThread\(threadId\)\s*\.then\(\(thread\) => \{\s*if \(thread && generation === readGeneration\) reader\.applyUpdate\(thread\);/,
  );
  // Leaving a chat drops its read: a change made while unsubscribed must not settle the next open.
  assert.match(
    notice,
    /window\.removeEventListener\(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated\);\s*(\/\/[^\n]*\n\s*)?setRead\(null\);/,
  );
  // Unmounting ends the open, so a pending switch's check turns false.
  assert.match(body, /useEffect\(\s*\(\) => \(\) => \{\s*autoSwitchRef\.current = \{ settled: false \};\s*\},\s*\[\],\s*\);/);
  assert.match(
    page,
    /loadedModels=\{loadedModels\}\s*onSwitch=\{handleSwitchBackToChatModel\}\s*onAutoSwitch=\{handleAutoSwitchToChatModel\}/,
  );
});

test("the automatic switch only adopts a slot, it never reaches /load", () => {
  const auto = slice(page, "const handleAutoSwitchToChatModel", "\n  );\n");
  assert.match(auto, /isDownloaded: true,\s*adoptOnly: true,/);
  // Any pick of the user's after dispatch, staged or no-op, cancels the pending adoption.
  assert.match(auto, /stillWanted: \(\) => stillWanted\(\) && userPickCountRef\.current === picks,/);
  assert.match(page, /if \(!meta\?\.adoptOnly\) userPickCountRef\.current \+= 1;\s*if \(!value\) return;/);
  assert.match(page, /adoptOnly: meta\?\.adoptOnly,\s*stillWanted: meta\?\.stillWanted,\s*\};\s*await stageOrLoad\(selection\);/);
  // Bails after the adopt check and before the lease that every /load holds.
  const select = slice(runtimeHook, "const selectModel = useCallback", "let lifecycleLease");
  const adopted = select.lastIndexOf("void refreshContextUsage({ afterModelLoad: true });");
  const bail = select.indexOf("if (adoptOnly) {");
  assert.ok(adopted !== -1 && bail > adopted, "adoptOnly bails after the adopt path");
  assert.match(select.slice(bail), /restorePreviousConfig\(\);[\s\S]*?return;/);
  // Never displaces a load (before taking an intent, and in the replacement loop that would cancel it),
  // and never lands on a chat that is no longer open.
  const guard = /if \(adoptOnly && \(loadInFlight\(\) \|\| !stillWanted\(\)\)\) \{\s*restorePreviousConfig\(\);\s*return;/g;
  assert.equal(select.match(guard)?.length, 2);
  assert.ok(
    select.indexOf("if (adoptOnly && (loadInFlight() || !stillWanted()))") <
      select.indexOf("const loadIntentId = ++modelSelectionIntentEpoch;"),
  );
  assert.match(select, /if \(adoptOnly && \(inFlightLoad \|\| activeRun\)\) \{\s*restorePreviousConfig\(\);\s*return;/);
  // Downloads nothing, so it never opens the Hub token prompt.
  assert.match(select, /const mayReachHub =\s*!adoptOnly &&/);
  // NPU picks always reach /load, and an external pick is never a loaded slot.
  assert.match(
    page,
    /if \(meta\?\.adoptOnly && \(isNpuModelId\(value\) \|\| isExternalModelId\(value\)\)\) \{\s*return;\s*\}/,
  );
  assert.ok(
    page.indexOf("if (meta?.adoptOnly && (isNpuModelId(value)") < page.indexOf("void loadNpuModel(value"),
  );
  assert.ok(
    select.indexOf("if (adoptOnly && (inFlightLoad || activeRun))") <
      select.indexOf("await cancelLoadRun(activeRun, true)"),
  );
});

test("the notice stays quiet when it has nothing to offer", () => {
  const body = notice.slice(notice.indexOf("export function ChatModelNotice"));
  // no stamp, already on the pick (including snapshot vs repo id), or a model that has since gone away
  assert.match(body, /if \(!createdModel \|\| resident\) return null;/);
  assert.match(
    body,
    /chatModelIsResident\(createdModel, checkpoint, activeGgufVariant\)/,
  );
  assert.match(
    body,
    /chatModelSelectableId\(\s*createdModel\.modelId,\s*selectableModelIds,?\s*\)/,
  );
});

test("switching chats does not show the outgoing chat's model", () => {
  // The answer is keyed by chat id so a stale async value can never render for another chat.
  const hook = notice.slice(
    notice.indexOf("export function useChatCreatedModel"),
    notice.indexOf("type ChatModelNoticeProps"),
  );
  assert.match(
    hook,
    /return read && read\.threadId === threadId \? read\.model : null;/,
  );
  assert.match(hook, /detail\?\.thread/);
  assert.match(hook, /reader\.applyUpdate\(updatedThread\)/);
  assert.match(hook, /reader\.applyInitial\(thread\)/);
  assert.match(hook, /reader\.dispose\(\)/);
  assert.match(
    hook,
    /addEventListener\(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated\)/,
  );
  assert.match(
    hook,
    /removeEventListener\(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated\)/,
  );
  assert.match(
    chatApi,
    /notifyChatHistoryUpdated\(\{ thread: savedThread \}\)/,
  );
  assert.match(chatApi, /notifyChatHistoryUpdated\(\{ thread \}\)/);
});

test("a thread update wins over an older initial model read", () => {
  const seen: unknown[] = [];
  const reader = createChatModelHistoryReader("thread-1", (model) => {
    seen.push(model);
  });
  reader.applyUpdate({
    id: "thread-1",
    modelId: "resolved-model",
    modelGgufVariant: "Q6_K",
  });
  reader.applyInitial({
    id: "thread-1",
    modelId: "interim-model",
    modelGgufVariant: "Q8_0",
  });
  reader.applyUpdate({ id: "another-thread", modelId: "unrelated-model" });
  assert.deepEqual(seen, [{ modelId: "resolved-model", ggufVariant: "Q6_K" }]);
  reader.dispose();
  reader.applyUpdate({ id: "thread-1", modelId: "late-model" });
  assert.equal(seen.length, 1);
});

test("the notice is wired to the picker's own handler, not a private path", () => {
  // The switch back must reuse handleCheckpointChange so confirmations and checks do not drift.
  assert.match(page, /onSwitch=\{handleSwitchBackToChatModel\}/);
  const wrapper = slice(
    page,
    "const handleSwitchBackToChatModel",
    "const inventoryRefreshStartedRef",
  );
  assert.match(wrapper, /handleCheckpointChange\(/);
  assert.match(wrapper, /chatModelSwitchMeta\(target, loraModels\)/);
  assert.match(page, /selectableModelIds=\{selectableModelIds\}/);
  assert.match(page, /view\.mode === "single" && \(\s*<ChatModelNotice/);
});

test("only models that can actually be selected are offered", () => {
  const set = page.slice(page.indexOf("const selectableModelIds = useMemo"));
  for (const source of ["models", "loraModels", "externalModels"]) {
    assert.ok(
      set.slice(0, 400).includes(`...${source}.map`),
      `${source} is missing from the selectable set`,
    );
  }
});

const HUB = "/home/u/.cache/huggingface/hub";
const REPO_ID = "unsloth/Repo-GGUF";
const SNAPSHOT_A = `${HUB}/models--unsloth--Repo-GGUF/snapshots/aaa1111`;
const SNAPSHOT_B = `${HUB}/models--unsloth--Repo-GGUF/snapshots/bbb2222`;

test("Switch Back resolves a snapshot-path chat to the live picker row", () => {
  assert.equal(
    chatModelSelectableId(SNAPSHOT_A, new Set([SNAPSHOT_A, REPO_ID])),
    SNAPSHOT_A,
  );
  assert.equal(chatModelSelectableId(SNAPSHOT_A, new Set([REPO_ID])), REPO_ID);
  assert.equal(
    chatModelSelectableId(SNAPSHOT_A, new Set([SNAPSHOT_B])),
    SNAPSHOT_B,
  );
  assert.equal(
    chatModelSelectableId(SNAPSHOT_A, new Set(["unsloth/Other-GGUF"])),
    null,
  );
  assert.equal(
    chatModelSelectableId(
      SNAPSHOT_A,
      new Set([`${HUB}/models--unsloth--Other-GGUF/snapshots/ccc3333`]),
    ),
    null,
  );
  assert.equal(
    chatModelSelectableId(
      "/srv/models/a/Repo-Q4_K_M.gguf",
      new Set(["Repo-Q4_K_M"]),
    ),
    null,
  );
});

test("a snapshot-path chat is already on its repo-id checkpoint", () => {
  const created = { modelId: SNAPSHOT_A, ggufVariant: "Q4_K_M" };
  assert.equal(chatModelIsResident(created, REPO_ID, "Q4_K_M"), true);
  assert.equal(chatModelIsResident(created, SNAPSHOT_B, "Q4_K_M"), true);
  assert.equal(
    chatModelIsResident(
      { modelId: REPO_ID, ggufVariant: "Q4_K_M" },
      SNAPSHOT_A,
      "Q4_K_M",
    ),
    true,
  );
  assert.equal(chatModelIsResident(created, REPO_ID, "Q8_0"), false);
  assert.equal(
    chatModelIsResident(created, "unsloth/Other-GGUF", "Q4_K_M"),
    false,
  );
});

test("external and ollama-manifest ids never alias across case", () => {
  for (const [upper, lower] of [
    [
      `external::vendor::${encodeURIComponent("Vendor/Qwen3.8-27B")}`,
      `external::vendor::${encodeURIComponent("vendor/qwen3.8-27b")}`,
    ],
    [
      `ollama-manifest:${encodeURIComponent("/home/u/.ollama/manifests/Llama")}`,
      `ollama-manifest:${encodeURIComponent("/home/u/.ollama/manifests/llama")}`,
    ],
  ]) {
    assert.equal(chatModelIsResident({ modelId: upper }, upper, null), true);
    assert.equal(chatModelIsResident({ modelId: upper }, lower, null), false);
    assert.equal(chatModelSelectableId(upper, new Set([lower])), null);
    assert.equal(chatModelSelectableId(upper, new Set([upper])), upper);
  }
});

test("a snapshot-path chat is labelled by its repo name, not the revision sha", () => {
  assert.equal(modelDisplayName(SNAPSHOT_A), "Repo-GGUF");
  assert.equal(modelDisplayName("org/model.gguf"), "model.gguf");
  const body = notice.slice(notice.indexOf("export function ChatModelNotice"));
  assert.match(body, /externalModelLabel\(createdModel\.modelId\)/);
  assert.match(body, /modelDisplayName\(createdModel\.modelId\)/);
});

test("the original-model control sits beside the picker in the chat header", () => {
  const header = slice(page, "chat-header-fade", "{/* One provider shared");
  assert.match(header, /<ModelSelector[\s\S]*<ChatModelNotice/);
  assert.match(
    header,
    /\{view\.mode === "single" && \(\s*<ChatModelNotice\s+active=\{active\}/,
  );
  // Hide the portaled UI after the hook runs, keeping the saved model and subscription off-route.
  assert.match(
    notice,
    /const createdModel = useChatCreatedModel\(threadId\);[\s\S]*?if \(!active\) return null;/,
  );
  // A chat change must dismiss a portaled popover along with its old model details.
  assert.match(notice, /<Popover key=\{threadId\}>/);
});

test("the conversation and research panel reserve only the chat header", () => {
  // The original-model control lives inside the header, so it needs no extra row.
  for (const source of [page, notice, thread, researchPanel]) {
    assert.doesNotMatch(source, /--studio-chat-notice-height/);
  }
  assert.match(
    thread,
    /\[--thread-header-offset:calc\(var\(--studio-content-top-inset,0px\)-var\(--studio-chat-scroller-top,0px\)\+var\(--studio-chat-header-height,48px\)\)\] mt-\[var\(--studio-chat-scroller-top,0px\)\] pt-\[var\(--thread-header-offset\)\]/,
  );
  const fade = slice(page, "chat-header-fade", '"');
  assert.match(
    fade,
    /top-\[calc\(var\(--studio-content-top-inset,0px\)\+var\(--studio-chat-header-height,48px\)\)\]/,
  );
  const style = slice(researchPanel, 'variant === "panel"', ": undefined");
  assert.match(
    style,
    /marginTop:\s*"calc\(var\(--studio-content-top-inset, 0px\) \+ var\(--studio-chat-header-height, 48px\)\)"/,
  );
  assert.match(
    style,
    /height:\s*"calc\(100% - var\(--studio-content-top-inset, 0px\) - var\(--studio-chat-header-height, 48px\)\)"/,
  );
});

test("a chat started as New Chat gets the notice once its row exists", () => {
  // No thread id in ?new=<nonce>; the store's id is this chat's only after the blank, hence the latch.
  const gate = slice(
    page,
    "const newChatBlankedRef",
    "const newChatThreadId =",
  );
  assert.match(
    gate,
    /activeThreadId === null \|\| isAssistantLocalThreadId\(activeThreadId\)/,
  );
  assert.match(gate, /newChatBlankedRef\.current = search\.new;/);

  const derived = slice(page, "const newChatThreadId =", "\n  const");
  assert.match(derived, /newChatBlankedRef\.current === search\.new/);
  assert.match(derived, /\? persistedActiveThreadId/);
  assert.match(derived, /: null/);

  const notice = slice(page, "<ChatModelNotice", "/>");
  assert.match(
    notice,
    /threadId=\{view\.threadId \?\? newChatThreadId \?\? undefined\}/,
  );
});

test("a chat thread keeps the GGUF variant it started on", () => {
  assert.match(types, /modelGgufVariant\?: string \| null;/);
  const creator = slice(
    runtimeProvider,
    "export async function ensureThreadRecord",
    "function createStudioDbAdapter",
  );
  assert.match(creator, /modelGgufVariantAtInit/);
  assert.match(creator, /modelGgufVariant: modelGgufVariantAtInit/);
  const initialize = slice(
    runtimeProvider,
    "initialize(threadId",
    "async rename",
  );
  assert.match(initialize, /runtimeStateAtInit\.activeGgufVariant/);
  assert.match(initialize, /modelGgufVariant: modelGgufVariantAtInit/);
});

test("a send records the model and GGUF variant it ran on", () => {
  const persistence = slice(
    adapter,
    "const persistRunModel",
    "if (queuedRunSettings)",
  );
  assert.match(persistence, /modelGgufVariant: string \| null/);
  assert.match(persistence, /modelGgufVariant,/);
  assert.match(persistence, /shouldPersistRunModel\(storedThread, modelId, modelGgufVariant\)/);
  // A failed write only loses the offer on reopen.
  assert.match(persistence, /\} catch \{/);
  assert.match(adapter, /activeGgufVariant: state\.activeGgufVariant/);
  assert.match(adapter, /activeGgufVariant: status\.gguf_variant \?\? null/);
  assert.match(
    adapter,
    /queuedEmptyModelRuntime !== null[\s\S]{0,100}queuedEmptyModelRuntime\.activeGgufVariant[\s\S]{0,80}liveRuntime\.activeGgufVariant/,
  );
  // Chat, audio and research runs record it once past the gates that refuse a send, and never
  // hold the send up waiting on it.
  assert.doesNotMatch(adapter, /await persistRunModel\(/);
  assert.equal(
    adapter.match(
      /releaseCurrentPreStreamRun\(\);\s*(?:\/\/[^\n]*\s*)?void persistRunModel\(params\.checkpoint, runtime\.activeGgufVariant\);\s*runtime\.setThreadRunning\(threadKey, true/g,
    )?.length,
    3,
  );
  const queuedSettings = readText(
    "../src/features/chat/utils/queued-chat-run-settings.ts",
  );
  assert.match(queuedSettings, /"activeGgufVariant"/);
  assert.match(
    thread,
    /modelGgufVariant: runSettingsAtQueueStart\.activeGgufVariant/,
  );
});

test("a chat remembers the model it last sent to", () => {
  // A queued send on a row written without a model fills it in.
  assert.equal(shouldPersistRunModel({ modelId: "" }, "unsloth/Qwen3-8B", null), true);
  // Switching models mid-chat moves the chat to the new one.
  assert.equal(shouldPersistRunModel({ modelId: "unsloth/Qwen3-8B" }, "unsloth/gemma-3-4b-it", null), true);
  assert.equal(
    shouldPersistRunModel(
      { modelId: "unsloth/Qwen3-8B-GGUF", modelGgufVariant: "Q4_K_M" },
      "unsloth/Qwen3-8B-GGUF",
      "Q8_0",
    ),
    true,
  );
  // Same model again writes nothing, so repeated sends are a no-op.
  assert.equal(shouldPersistRunModel({ modelId: "unsloth/Qwen3-8B" }, "unsloth/Qwen3-8B", null), false);
  assert.equal(
    shouldPersistRunModel(
      { modelId: "unsloth/Qwen3-8B-GGUF", modelGgufVariant: "Q4_K_M" },
      "unsloth/Qwen3-8B-GGUF",
      "Q4_K_M",
    ),
    false,
  );
  // An unknown variant never clears the saved one for the same model.
  assert.equal(
    shouldPersistRunModel(
      { modelId: "unsloth/Qwen3-8B-GGUF", modelGgufVariant: "Q4_K_M" },
      "unsloth/Qwen3-8B-GGUF",
      null,
    ),
    false,
  );
  // Compare panes keep the model they were opened with, and no row or no model means no write.
  assert.equal(shouldPersistRunModel({ modelId: "a", pairId: "pair-1" }, "b", null), false);
  assert.equal(shouldPersistRunModel(undefined, "b", null), false);
  assert.equal(shouldPersistRunModel({ modelId: "a" }, "", null), false);
});

// Switch back must carry the picker's metadata, or a local GGUF loads down the transformers path.

test("switching back to a single-file local GGUF still loads as a GGUF", () => {
  const meta = chatModelSwitchMeta({ modelId: "/models/qwen3-4b-q4.gguf" }, [
    {
      id: "/models/qwen3-4b-q4.gguf",
      name: "qwen3-4b-q4",
      source: "local",
      isGguf: true,
      isDirectGguf: true,
    },
  ]);
  assert.deepEqual(meta, {
    source: "local",
    isLora: false,
    isDownloaded: true,
    isGguf: true,
  });
});

test("a single .gguf file carries its format out of the local inventory", () => {
  const [option] = chatLocalModelOptions([
    {
      id: "/models/qwen3-4b-q4.gguf",
      display_name: "qwen3-4b-q4",
      path: "/models/qwen3-4b-q4.GGUF",
      source: "custom",
      model_format: "gguf",
    },
  ]);
  assert.equal(option.isGguf, true);
});

test("a saved GGUF variant is carried back exactly", () => {
  store.clear();
  const modelId = "/models/Qwen3-4B-GGUF";
  assert.deepEqual(
    chatModelSwitchMeta({ modelId, ggufVariant: "Q6_K" }, [
      {
        id: modelId,
        name: "Qwen3-4B-GGUF",
        source: "local",
        isGguf: true,
      },
    ]),
    {
      source: "local",
      isLora: false,
      isDownloaded: true,
      isGguf: true,
      ggufVariant: "Q6_K",
    },
  );
  assert.deepEqual(
    chatModelSwitchMeta(
      { modelId: "unsloth/Qwen3-4B-GGUF", ggufVariant: "Q8_0" },
      [],
    ),
    { ggufVariant: "Q8_0" },
  );
});

test("switching back to a hub GGUF loads it instead of staging a download", () => {
  store.clear();
  // Hub picks without isDownloaded go through the download manager, which can silently do nothing.
  const selection = switchBackSelection({
    modelId: "unsloth/Qwen3-4B-GGUF",
    ggufVariant: "Q8_0",
  });
  assert.equal(wantsDownloadManagerStaging(selection), false);
  assert.equal(selection.ggufVariant, "Q8_0");
  assert.equal(
    wantsDownloadManagerStaging({
      id: "unsloth/Qwen3-4B-GGUF",
      source: "hub",
      ggufVariant: "Q8_0",
    }),
    true,
  );
  const built = slice(
    page,
    "const selection = {",
    "await stageOrLoad(selection);",
  );
  for (const field of [
    "source: meta?.source",
    "isLora: meta?.isLora",
    "ggufVariant: meta?.ggufVariant",
    "isGguf: meta?.isGguf",
  ]) {
    assert.ok(built.includes(field), `selection lost ${field}`);
  }
  assert.match(
    built,
    /isDownloaded: meta\?\.isDownloaded \|\| isSameLoadedModel/,
  );
  assert.match(
    page,
    /const wantManagerStaging = wantsDownloadManagerStaging\(selection\);/,
  );

  const stage = slice(
    page,
    "const stageOrLoad = useCallback",
    "if (store.modelLoading)",
  );
  assert.match(
    stage,
    /const wantManagerStaging = wantsDownloadManagerStaging\(selection\);[\s\S]*if \(wantManagerStaging\) \{[\s\S]*dismissStartToastsForModelSelection\(\)/,
  );
});

test("a history update that leaves the model alone is not re-emitted", () => {
  const seen: unknown[] = [];
  const reader = createChatModelHistoryReader("thread-1", (model) => {
    seen.push(model);
  });
  reader.applyInitial({
    id: "thread-1",
    modelId: "model-a",
    modelGgufVariant: "Q6_K",
  });
  reader.applyUpdate({
    id: "thread-1",
    modelId: "model-a",
    modelGgufVariant: "Q6_K",
  });
  reader.applyUpdate({
    id: "thread-1",
    modelId: "model-a",
    modelGgufVariant: "Q6_K",
  });
  assert.equal(seen.length, 1);
  reader.applyUpdate({
    id: "thread-1",
    modelId: "model-a",
    modelGgufVariant: "Q8_0",
  });
  assert.equal(seen.length, 2);
  reader.dispose();
});

test("a legacy GGUF directory recovers its sole saved variant and context", () => {
  store.clear();
  const modelId = String.raw`N:\AI Models\Qwen\Qwen3.6-40B-Q6_K`;
  assert.ok(
    savePerModelConfig(modelId, "Q6_K", {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 32768,
    }),
  );
  const [option] = chatLocalModelOptions([
    {
      id: modelId,
      display_name: "Qwen3.6-40B-Q6_K",
      path: modelId,
      source: "custom",
      model_format: "gguf",
    },
  ]);
  assert.equal(option.isDirectGguf, undefined);
  const target = resolveChatModelSwitchTarget({ modelId });
  assert.equal(target.ggufVariant?.toLowerCase(), "q6_k");
  const meta = chatModelSwitchMeta(target, [option]);
  assert.equal(meta?.isGguf, true);
  assert.equal(meta?.ggufVariant?.toLowerCase(), "q6_k");
  assert.equal(
    resolveInitialConfig(modelId, target.ggufVariant ?? undefined).config
      .customContextLength,
    32768,
  );
});

test("a legacy GGUF directory does not guess between saved variants", () => {
  store.clear();
  const modelId = "/models/Qwen3-4B-GGUF";
  for (const variant of ["Q4_K_M", "Q8_0"]) {
    assert.ok(
      savePerModelConfig(modelId, variant, {
        ...DEFAULT_PER_MODEL_CONFIG,
        customContextLength: 32768,
      }),
    );
  }
  const [option] = chatLocalModelOptions([
    {
      id: modelId,
      display_name: "Qwen3-4B-GGUF",
      path: modelId,
      source: "models_dir",
      model_format: "gguf",
    },
  ]);
  const meta = chatModelSwitchMeta({ modelId }, [option]);
  assert.equal(meta?.isGguf, false);
  assert.equal(meta?.ggufVariant, undefined);
  assert.equal(meta?.config, undefined);
});

test("a legacy GGUF directory does not infer a quant from config alone", () => {
  store.clear();
  const modelId = "/models/Qwen3-4B-GGUF";
  assert.ok(
    savePerModelConfig(modelId, "Q6_K", {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 32768,
    }),
  );
  assert.equal(
    resolveChatModelSwitchTarget({ modelId }).ggufVariant,
    undefined,
  );
});

test("the switch back leaves the remembered config to stageOrLoad", () => {
  store.clear();
  const modelId = "/models/qwen3-4b-q4.gguf";
  assert.ok(
    savePerModelConfig(modelId, null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 32768,
    }),
  );
  const selection = switchBackSelection({ modelId }, [
    {
      id: modelId,
      name: "qwen3-4b-q4",
      source: "local",
      isGguf: true,
      isDirectGguf: true,
    },
  ]);
  assert.doesNotMatch(
    slice(switchSource, "import type {", "export type ChatModelSwitchTarget"),
    /resolveInitialConfig/,
  );
  const stage = slice(
    page,
    "const stageOrLoad = useCallback",
    "useRepoDownload(",
  );
  assert.match(
    stage,
    /selection\.config \?\? rememberedConfigFor\(selection\)/,
  );
  const remembered = slice(
    page,
    "const rememberedConfigFor = useCallback",
    "const isExternalModel",
  );
  assert.match(
    remembered,
    /resolveResidentInitialConfig\(\s*selection\.id,\s*selection\.ggufVariant,?\s*\)/,
  );
  assert.equal(
    resolveResidentInitialConfig(selection.id, selection.ggufVariant).config
      .customContextLength,
    32768,
  );
});

test("switch back recovers a repo-keyed context through a snapshot path", () => {
  store.clear();
  const snapshotPath =
    "/home/u/.cache/huggingface/hub/models--unsloth--Repo-GGUF/snapshots/2f1c9ab";
  const repoId = "unsloth/Repo-GGUF";
  assert.ok(
    savePerModelConfig(repoId, "Q4_K_M", {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 32768,
    }),
  );
  assert.equal(resolveInitialConfig(snapshotPath, "Q4_K_M").remembered, false);
  assert.equal(
    resolveResidentInitialConfig(snapshotPath, "Q4_K_M").config
      .customContextLength,
    32768,
  );
  const remembered = slice(
    page,
    "const rememberedConfigFor = useCallback",
    "const isExternalModel",
  );
  assert.match(
    remembered,
    /resolveResidentInitialConfig\(\s*selection\.id,\s*selection\.ggufVariant,?\s*\)/,
  );
});

test("switching back to a fine-tuned row mirrors the picker's own metadata", () => {
  const adapters = [
    { id: "run-1", name: "run-1", source: "training" as const },
    {
      id: "run-2",
      name: "run-2",
      source: "exported" as const,
      exportType: "merged" as const,
    },
  ];
  assert.deepEqual(chatModelSwitchMeta({ modelId: "run-1" }, adapters), {
    source: "lora",
    isLora: true,
    isDownloaded: true,
    isGguf: false,
  });
  assert.deepEqual(chatModelSwitchMeta({ modelId: "run-2" }, adapters), {
    source: "exported",
    isLora: false,
    isDownloaded: true,
    isGguf: false,
  });
});

test("a Hub or external id is still switched on the id alone", () => {
  assert.equal(
    chatModelSwitchMeta({ modelId: "unsloth/Qwen3-4B-GGUF" }, []),
    undefined,
  );
  assert.equal(
    chatModelSwitchMeta({ modelId: "openai:gpt-5-mini" }, []),
    undefined,
  );
});

test("another tab's change leaves no model until a fresh read lands", () => {
  const seen: unknown[] = [];
  const reader = createChatModelHistoryReader("t1", (model) => seen.push(model));
  reader.applyInitial({ id: "t1", modelId: "unsloth/A" } as never);
  reader.invalidate();
  // The opening read that was still in flight no longer answers.
  reader.applyInitial({ id: "t1", modelId: "unsloth/A" } as never);
  // The fresh read lands even when it names the same model.
  reader.applyUpdate({ id: "t1", modelId: "unsloth/A" } as never);
  assert.deepEqual(
    seen.map((m) => (m as { modelId?: string } | null)?.modelId ?? null),
    ["unsloth/A", null, "unsloth/A"],
  );
});
