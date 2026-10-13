import assert from "node:assert/strict";
import { register } from "node:module";
import test, { type TestContext } from "node:test";
import type * as chatRuntime from "../src/features/chat/stores/chat-runtime-store.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: storage } = installLocalStorageFake();
register("./thread-sampling-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);
type RuntimeModule = typeof chatRuntime;
const runtimeUrl = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;
const THREAD = "tool-selection-chat";
const MODEL = "unsloth/tool-capable-model";
const TOOL_KEYS = [
  "toolsEnabled",
  "codeToolsEnabled",
  "imageToolsEnabled",
  "webFetchToolsEnabled",
  "mcpEnabledForChat",
  "ragEnabled",
] as const;

async function boot(t: TestContext): Promise<RuntimeModule> {
  storage.clear();
  storage.set("unsloth_chat_settings_imported_to_studio_db", "true");
  settingsHttp.settings = { rememberParamsPerModel: false };
  settingsHttp.puts.length = 0;
  threadRows.reset();
  const runtime = (await import(
    `${runtimeUrl}?scenario=${encodeURIComponent(t.name)}`
  )) as RuntimeModule;
  t.after(async () => {
    await runtime.awaitThreadScopedSettingsWrite(THREAD);
    await runtime.flushPendingChatSettings();
    await runtime.awaitStartedThreadScopedSettingsWrites();
  });
  const store = runtime.useChatRuntimeStore;
  await store.getState().hydratePersistedSettings();
  store.getState().setCheckpoint(MODEL, null);
  store.setState({ supportsTools: true });
  return runtime;
}

for (const scope of ["defaults", "chat", "pending chat"] as const) {
  for (const search of [false, true]) {
    for (const code of [false, true]) {
      test(`${scope} remembers Search=${search} and Code=${code} across unload and reload`, async (t) => {
        const runtime = await boot(t);
        const store = runtime.useChatRuntimeStore;
        const original = { toolsEnabled: !search, codeToolsEnabled: !code };
        if (scope !== "defaults") {
          store.getState().setActiveThreadId(THREAD);
          if (scope === "pending chat") {
            runtime.beginThreadScopedPairing(THREAD);
          } else {
            store.getState().applyThreadScopedSettings(THREAD, original);
          }
        }
        store.getState().setToolsEnabled(search);
        store.getState().setCodeToolsEnabled(code);
        store.getState().setImageToolsEnabled(search);
        store.getState().setWebFetchToolsEnabled(code);
        store.getState().setMcpEnabledForChat(search);
        store.getState().setRagEnabled(code);
        const expected = Object.fromEntries(
          TOOL_KEYS.map((key) => [key, store.getState()[key]]),
        );

        store.getState().clearCheckpoint();
        assert.equal(store.getState().params.checkpoint, "");
        assert.equal(store.getState().supportsTools, false);
        for (const key of TOOL_KEYS) {
          assert.equal(store.getState()[key], expected[key], key);
        }

        if (scope === "pending chat") {
          store.getState().applyThreadScopedSettings(THREAD, original);
        }
        await runtime.awaitThreadScopedSettingsWrite(THREAD);
        await runtime.flushPendingChatSettings();
        if (scope !== "defaults") {
          for (const key of TOOL_KEYS) {
            assert.equal(
              threadRows.rows.get(THREAD)?.[key],
              expected[key],
              key,
            );
          }
          assert.equal(storage.get("unsloth_chat_tools_enabled"), undefined);
          assert.equal(
            storage.get("unsloth_chat_code_tools_enabled"),
            undefined,
          );
        }

        store.getState().setCheckpoint(MODEL, null);
        store.setState({
          supportsTools: true,
          ...runtime.resolveToolsEnabledOnLoad(true),
        });
        for (const key of TOOL_KEYS) {
          assert.equal(store.getState()[key], expected[key], key);
        }
        assert.deepEqual(runtime.resolveToolsEnabledOnLoad(false), {
          toolsEnabled: false,
          codeToolsEnabled: false,
        });
      });
    }
  }
}

test("unloading does not overwrite a settled chat or its defaults", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.getState().setActiveThreadId(THREAD);
  store.getState().applyThreadScopedSettings(THREAD, {
    toolsEnabled: true,
    codeToolsEnabled: true,
  });
  await runtime.flushPendingChatSettings();
  const writesBeforeUnload = settingsHttp.puts.length;

  store.getState().clearCheckpoint();
  await runtime.awaitThreadScopedSettingsWrite(THREAD);
  await runtime.flushPendingChatSettings();
  assert.equal(threadRows.writes.length, 0);
  assert.equal(settingsHttp.puts.length, writesBeforeUnload);

  store.getState().setPermissionMode("ask");
  await runtime.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(threadRows.rows.get(THREAD)?.toolsEnabled, true);
  assert.equal(threadRows.rows.get(THREAD)?.codeToolsEnabled, true);

  store.getState().setActiveThreadId(null);
  store.getState().applyThreadScopedSettings(null, null);
  assert.equal(store.getState().toolsEnabled, false);
  assert.equal(store.getState().codeToolsEnabled, false);
  store.getState().setActiveThreadId(THREAD);
  store
    .getState()
    .applyThreadScopedSettings(THREAD, threadRows.rows.get(THREAD)!);
  assert.equal(store.getState().toolsEnabled, true);
  assert.equal(store.getState().codeToolsEnabled, true);
});

test("unloading preserves armed research without changing tool exclusivity", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.getState().setToolsEnabled(true);
  store.getState().setCodeToolsEnabled(true);
  store.getState().setDeepResearchEnabled(true);
  store.getState().clearCheckpoint();
  assert.equal(store.getState().deepResearchEnabled, true);
  assert.equal(store.getState().toolsEnabled, false);
  assert.equal(store.getState().codeToolsEnabled, false);
  assert.equal(storage.get("unsloth_chat_deep_research_enabled"), "true");

  store.getState().setCodeToolsEnabled(true);
  assert.equal(store.getState().deepResearchEnabled, false);
});
