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
const { useExternalProvidersStore } = await import(
  "../src/features/chat/stores/external-providers-store.ts"
);
const { setProviderModelCapabilities } = await import(
  "../src/features/chat/external-providers.ts"
);
type RuntimeModule = typeof chatRuntime;
const runtimeUrl = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;
const THREAD = "tool-selection-chat";
const MODEL = "unsloth/tool-capable-model";
const LOCAL_TOOL_KEYS = [
  "toolsEnabled",
  "codeToolsEnabled",
  "mcpEnabledForChat",
  "ragEnabled",
] as const;
const TOOL_KEYS = [
  ...LOCAL_TOOL_KEYS,
  "imageToolsEnabled",
  "webFetchToolsEnabled",
] as const;

async function boot(t: TestContext, hydrate = true): Promise<RuntimeModule> {
  storage.clear();
  storage.set("unsloth_chat_settings_imported_to_studio_db", "true");
  settingsHttp.settings = { rememberParamsPerModel: false };
  settingsHttp.puts.length = 0;
  threadRows.reset();
  useExternalProvidersStore.setState({ providers: [] });
  const runtime = (await import(
    `${runtimeUrl}?scenario=${encodeURIComponent(t.name)}`
  )) as RuntimeModule;
  t.after(async () => {
    await runtime.awaitThreadScopedSettingsWrite(THREAD);
    await runtime.flushPendingChatSettings();
    await runtime.awaitStartedThreadScopedSettingsWrites();
  });
  const store = runtime.useChatRuntimeStore;
  if (hydrate) await store.getState().hydratePersistedSettings();
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
        for (const key of LOCAL_TOOL_KEYS) {
          assert.equal(store.getState()[key], expected[key], key);
        }
        assert.deepEqual(runtime.resolveToolsEnabledOnLoad(false), {
          toolsEnabled: false,
          codeToolsEnabled: false,
          deepResearchEnabled: false,
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

for (const scope of ["defaults", "chat", "pending chat"] as const) {
  test(`${scope} disables Research for an incompatible local model and restores its preference`, async (t) => {
    const runtime = await boot(t);
    const store = runtime.useChatRuntimeStore;
    if (scope !== "defaults") {
      store.getState().setActiveThreadId(THREAD);
      if (scope === "pending chat") {
        runtime.beginThreadScopedPairing(THREAD);
      } else {
        store
          .getState()
          .applyThreadScopedSettings(THREAD, { deepResearchEnabled: false });
      }
    }
    store.getState().setDeepResearchEnabled(true);
    store.getState().clearCheckpoint();
    store.getState().setCheckpoint("unsloth/no-tools", null);
    store.setState({
      supportsTools: false,
      ...runtime.resolveToolsEnabledOnLoad(false),
    });
    assert.equal(store.getState().deepResearchEnabled, false);
    store.getState().setDeepResearchEnabled(true);
    assert.equal(store.getState().deepResearchEnabled, false);

    if (scope === "pending chat") {
      store
        .getState()
        .applyThreadScopedSettings(THREAD, { deepResearchEnabled: false });
    }
    store.getState().setPermissionMode("ask");
    await runtime.awaitThreadScopedSettingsWrite(THREAD);
    assert.equal(store.getState().deepResearchEnabled, false);
    if (scope !== "defaults") {
      assert.equal(threadRows.rows.get(THREAD)?.deepResearchEnabled, true);
      store
        .getState()
        .applyThreadScopedSettings(THREAD, threadRows.rows.get(THREAD)!);
      assert.equal(store.getState().deepResearchEnabled, false);
    } else {
      assert.equal(storage.get("unsloth_chat_deep_research_enabled"), "true");
    }

    store.getState().clearCheckpoint();
    store.getState().setCheckpoint(MODEL, null);
    store.setState({
      supportsTools: true,
      ...runtime.resolveToolsEnabledOnLoad(true),
    });
    assert.equal(store.getState().deepResearchEnabled, true);
    store.getState().setDeepResearchEnabled(false);
    store.setState(runtime.resolveToolsEnabledOnLoad(true));
    assert.equal(store.getState().deepResearchEnabled, false);
  });
}

test("late settings hydration cannot arm Research on an incompatible local model", async (t) => {
  const runtime = await boot(t, false);
  const store = runtime.useChatRuntimeStore;
  settingsHttp.settings = { deepResearchEnabled: true };
  store.setState({
    supportsTools: false,
    ...runtime.resolveToolsEnabledOnLoad(false),
  });
  await store.getState().hydratePersistedSettings();
  assert.equal(store.getState().deepResearchEnabled, false);
  assert.equal(storage.get("unsloth_chat_deep_research_enabled"), "true");
  store.setState({
    supportsTools: true,
    ...runtime.resolveToolsEnabledOnLoad(true),
  });
  assert.equal(store.getState().deepResearchEnabled, true);
});

test("suppressed Research remains scoped to its chat", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.getState().setActiveThreadId(THREAD);
  store
    .getState()
    .applyThreadScopedSettings(THREAD, { deepResearchEnabled: true });
  store.setState({
    supportsTools: false,
    ...runtime.resolveToolsEnabledOnLoad(false),
  });
  store.getState().setActiveThreadId("other-chat");
  store
    .getState()
    .applyThreadScopedSettings("other-chat", { deepResearchEnabled: false });
  store.setState({
    supportsTools: true,
    ...runtime.resolveToolsEnabledOnLoad(true),
  });
  assert.equal(store.getState().deepResearchEnabled, false);
  store.getState().setActiveThreadId(THREAD);
  store
    .getState()
    .applyThreadScopedSettings(THREAD, { deepResearchEnabled: true });
  assert.equal(store.getState().deepResearchEnabled, true);
  store.getState().setToolsEnabled(true);
  store.setState(runtime.resolveToolsEnabledOnLoad(true));
  assert.equal(store.getState().toolsEnabled, true);
  assert.equal(store.getState().deepResearchEnabled, false);
});

const CLAMPED_TOOLS = [
  ["toolsEnabled", "setToolsEnabled"],
  ["codeToolsEnabled", "setCodeToolsEnabled"],
  ["imageToolsEnabled", "setImageToolsEnabled"],
  ["webFetchToolsEnabled", "setWebFetchToolsEnabled"],
] as const;

for (const [key, setter] of CLAMPED_TOOLS) {
  for (const scope of ["pending chat", "pending save", "saved chat"] as const) {
    test(`${key} survives capability suppression with a ${scope}`, async (t) => {
      const runtime = await boot(t);
      const store = runtime.useChatRuntimeStore;
      store.getState().setActiveThreadId(THREAD);
      if (scope === "pending chat") {
        runtime.beginThreadScopedPairing(THREAD);
      } else {
        store.getState().applyThreadScopedSettings(THREAD, { [key]: false });
      }
      store.getState()[setter](true);
      if (scope === "saved chat") {
        await runtime.awaitThreadScopedSettingsWrite(THREAD);
      }
      store.getState().clearCheckpoint();
      store.getState().setCheckpoint("unsloth/no-tools", null);
      store.setState({ [key]: false, supportsTools: false });
      if (scope === "pending chat") {
        store.getState().applyThreadScopedSettings(THREAD, { [key]: false });
      }
      store.getState().clearCheckpoint();
      store.getState().setPermissionMode("ask");
      await runtime.awaitThreadScopedSettingsWrite(THREAD);
      assert.equal(threadRows.rows.get(THREAD)?.[key], true);
      assert.equal(runtime.threadScopedOverride(key), true);

      store.getState()[setter](false);
      await runtime.awaitThreadScopedSettingsWrite(THREAD);
      assert.equal(threadRows.rows.get(THREAD)?.[key], false);
    });
  }

  test(`${key} keeps the default when a chat opens under an incompatible model`, async (t) => {
    const runtime = await boot(t);
    const store = runtime.useChatRuntimeStore;
    store.getState()[setter](true);
    store.setState({ [key]: false, supportsTools: false });
    store.getState().setActiveThreadId(THREAD);
    store.getState().applyThreadScopedSettings(THREAD, null);
    await runtime.awaitThreadScopedSettingsWrite(THREAD);
    assert.equal(threadRows.rows.get(THREAD)?.[key], true);
  });
}

test("Research can be selected while model capabilities are still loading", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.getState().setCheckpoint("unsloth/no-tools", null);
  store.setState({ supportsTools: false });
  const lease = store.getState().beginModelLoading("loading");
  assert.ok(lease !== null);
  store.getState().setDeepResearchEnabled(true);
  store.getState().setCheckpoint(MODEL, null);
  store.setState({ supportsTools: true, ...runtime.resolveToolsEnabledOnLoad(true) });
  store.getState().endModelLoading(lease);
  assert.equal(store.getState().deepResearchEnabled, true);
});

test("Research hydration keeps the global default separate from a held chat edit", async (t) => {
  const runtime = await boot(t, false);
  const store = runtime.useChatRuntimeStore;
  settingsHttp.settings = { deepResearchEnabled: true };
  store.setState({ supportsTools: false });
  store.getState().setActiveThreadId(THREAD);
  runtime.beginThreadScopedPairing(THREAD);
  store.getState().setDeepResearchEnabled(false);
  await store.getState().hydratePersistedSettings();
  store.getState().applyThreadScopedSettings(THREAD, null);
  await runtime.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(threadRows.rows.get(THREAD)?.deepResearchEnabled, false);
  assert.equal(runtime.threadScopedDefault("deepResearchEnabled"), true);
  store.getState().setActiveThreadId(null);
  store.getState().applyThreadScopedSettings(null, null);
  store.setState({ supportsTools: true, ...runtime.resolveToolsEnabledOnLoad(true) });
  assert.equal(store.getState().deepResearchEnabled, true);
});

for (const scope of ["defaults", "chat"] as const) {
  test(`${scope} restores Research across incompatible and compatible external models`, async (t) => {
    const runtime = await boot(t);
    const store = runtime.useChatRuntimeStore;
    useExternalProvidersStore.getState().setProviders([{
      id: "tools-provider",
      providerType: "vllm",
      name: "Tool capability fixture",
      baseUrl: "http://localhost:8000/v1",
      models: ["capable", "unsupported"],
      createdAt: 0,
      updatedAt: 0,
    }]);
    setProviderModelCapabilities("vllm", {
      capable: { studio_tools: true },
      unsupported: { studio_tools: false },
    });
    if (scope === "chat") {
      store.getState().setActiveThreadId(THREAD);
      store.getState().applyThreadScopedSettings(THREAD, { deepResearchEnabled: false });
    }
    store.getState().setDeepResearchEnabled(true);
    store.getState().setCheckpoint("external::tools-provider::unsupported", null);
    assert.equal(store.getState().deepResearchEnabled, false);
    store.getState().setPermissionMode("ask");
    await runtime.awaitThreadScopedSettingsWrite(THREAD);
    assert.equal(
      scope === "chat"
        ? threadRows.rows.get(THREAD)?.deepResearchEnabled
        : storage.get("unsloth_chat_deep_research_enabled") === "true",
      true,
    );
    store.getState().setCheckpoint("external::tools-provider::capable", null);
    assert.equal(store.getState().deepResearchEnabled, true);
    if (scope === "chat") {
      store.getState().applyThreadScopedSettings(THREAD, threadRows.rows.get(THREAD)!);
      assert.equal(store.getState().deepResearchEnabled, true);
    }
    store.getState().setDeepResearchEnabled(false);
    store.getState().setCheckpoint("external::tools-provider::unsupported", null);
    store.getState().setCheckpoint("external::tools-provider::capable", null);
    assert.equal(store.getState().deepResearchEnabled, false);
  });
}

for (const scope of ["defaults", "chat", "pending chat"] as const) {
  test(`${scope} restores Research when switching from an incompatible local model to Codex`, async (t) => {
    const runtime = await boot(t);
    const store = runtime.useChatRuntimeStore;
    useExternalProvidersStore.getState().setProviders([{
      id: "codex",
      providerType: "openai_codex",
      name: "Codex fixture",
      baseUrl: "https://chatgpt.com/backend-api/codex",
      models: ["gpt-5.4"],
      createdAt: 0,
      updatedAt: 0,
    }]);
    setProviderModelCapabilities("openai_codex", {
      "gpt-5.4": { studio_tools: true },
    });
    if (scope !== "defaults") {
      store.getState().setActiveThreadId(THREAD);
      if (scope === "pending chat") {
        runtime.beginThreadScopedPairing(THREAD);
      } else {
        store.getState().applyThreadScopedSettings(THREAD, { deepResearchEnabled: false });
      }
    }
    store.getState().setDeepResearchEnabled(true);
    store.getState().clearCheckpoint();
    store.getState().setCheckpoint("unsloth/no-tools", null);
    store.setState({ supportsTools: false, ...runtime.resolveToolsEnabledOnLoad(false) });
    assert.equal(store.getState().deepResearchEnabled, false);
    store.getState().setCheckpoint("external::codex::gpt-5.4", null);
    assert.equal(store.getState().deepResearchEnabled, true);
    if (scope === "pending chat") {
      store.getState().applyThreadScopedSettings(THREAD, { deepResearchEnabled: false });
      assert.equal(store.getState().deepResearchEnabled, true);
    }
  });
}

test("an unresolved external provider preserves Research until its capabilities arrive", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.getState().setDeepResearchEnabled(true);
  store.getState().setCheckpoint("external::late-provider::research-model", null);
  assert.equal(store.getState().deepResearchEnabled, false);
  assert.equal(storage.get("unsloth_chat_deep_research_enabled"), "true");
  store.getState().setDeepResearchEnabled(true);
  assert.equal(store.getState().deepResearchEnabled, false);
  useExternalProvidersStore.getState().setProviders([{
    id: "late-provider",
    providerType: "vllm",
    name: "Late provider fixture",
    baseUrl: "http://localhost:8000/v1",
    models: ["research-model"],
    createdAt: 0,
    updatedAt: 0,
  }]);
  setProviderModelCapabilities("vllm", { "research-model": { studio_tools: true } });
  store.setState({ supportsTools: true, ...runtime.resolveToolsEnabledOnLoad(true) });
  assert.equal(store.getState().deepResearchEnabled, true);
});

test("a temporary chat keeps Research off when another tab stores it on", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  useExternalProvidersStore.getState().setProviders([{
    id: "codex",
    providerType: "openai_codex",
    name: "Codex fixture",
    baseUrl: "https://chatgpt.com/backend-api/codex",
    models: ["gpt-5.4"],
    createdAt: 0,
    updatedAt: 0,
  }]);
  setProviderModelCapabilities("openai_codex", {
    "gpt-5.4": { studio_tools: true },
  });
  store.getState().setIncognito(true);
  storage.set("unsloth_chat_deep_research_enabled", "true");
  assert.equal(runtime.resolveToolsEnabledOnLoad(true).deepResearchEnabled, false);
  store.getState().setCheckpoint("external::codex::gpt-5.4", null);
  assert.equal(store.getState().deepResearchEnabled, false);
  store.getState().clearCheckpoint();
  assert.equal(store.getState().deepResearchEnabled, false);
});

test("Kimi keeps Search off and Thinking on when a chat opens without stored pills", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.getState().setToolsEnabled(true);
  assert.equal(storage.get("unsloth_chat_tools_enabled"), "true");
  useExternalProvidersStore.getState().setProviders([{
    id: "kimi",
    providerType: "kimi",
    name: "Kimi fixture",
    baseUrl: "https://api.moonshot.ai/v1",
    models: ["kimi-k2.5"],
    createdAt: 0,
    updatedAt: 0,
  }]);
  store.getState().setCheckpoint("external::kimi::kimi-k2.5", null);
  store.setState({
    supportsTools: false,
    supportsBuiltinWebSearch: true,
    toolsEnabled: false,
    reasoningEnabled: true,
  });
  store.getState().setActiveThreadId(THREAD);
  store.getState().applyThreadScopedSettings(THREAD, null);
  assert.equal(store.getState().toolsEnabled, false);
  assert.equal(store.getState().reasoningEnabled, true);
  await runtime.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(threadRows.rows.get(THREAD)?.toolsEnabled, false);
  assert.equal(threadRows.rows.get(THREAD)?.reasoningEnabled, true);
  store.getState().setActiveThreadId(null);
  store.getState().applyThreadScopedSettings(null, null);
  assert.equal(store.getState().toolsEnabled, false);
  assert.equal(store.getState().reasoningEnabled, true);
  assert.equal(storage.get("unsloth_chat_tools_enabled"), "true");
});

test("unloading drops a Search pill the provider lit without a stored preference", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.setState({ toolsEnabled: true, codeToolsEnabled: true });
  assert.equal(storage.get("unsloth_chat_tools_enabled"), undefined);
  store.getState().clearCheckpoint();
  assert.equal(store.getState().toolsEnabled, false);
  assert.equal(store.getState().codeToolsEnabled, false);
  store.getState().setToolsEnabled(true);
  store.getState().clearCheckpoint();
  assert.equal(store.getState().toolsEnabled, true);
});

test("a held edit committed on leave cannot re-arm Research on an incompatible local model", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.getState().setDeepResearchEnabled(true);
  store.getState().clearCheckpoint();
  store.getState().setCheckpoint("unsloth/no-tools", null);
  store.setState({ supportsTools: false, ...runtime.resolveToolsEnabledOnLoad(false) });
  assert.equal(store.getState().deepResearchEnabled, false);
  store.getState().setActiveThreadId(THREAD);
  runtime.beginThreadScopedPairing(THREAD);
  store.getState().setMcpEnabledForChat(true);
  store.getState().setActiveThreadId(null);
  const commit = runtime.commitHeldThreadScopedEditsToTheirThread();
  assert.equal(store.getState().deepResearchEnabled, false);
  assert.equal(store.getState().mcpEnabledForChat, false);
  await commit;
  assert.equal(threadRows.rows.get(THREAD)?.deepResearchEnabled, false);
  assert.equal(threadRows.rows.get(THREAD)?.mcpEnabledForChat, true);
  assert.equal(storage.get("unsloth_chat_deep_research_enabled"), "true");
  store.getState().clearCheckpoint();
  store.getState().setCheckpoint(MODEL, null);
  store.setState({ supportsTools: true, ...runtime.resolveToolsEnabledOnLoad(true) });
  assert.equal(store.getState().deepResearchEnabled, true);
});

test("a chat's stored Search and Code stay off in the composer under a model without tools", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  store.getState().clearCheckpoint();
  store.getState().setCheckpoint("unsloth/no-tools", null);
  store.setState({ supportsTools: false, ...runtime.resolveToolsEnabledOnLoad(false) });
  store.getState().setActiveThreadId(THREAD);
  store.getState().applyThreadScopedSettings(THREAD, {
    toolsEnabled: true,
    codeToolsEnabled: true,
  });
  assert.equal(store.getState().toolsEnabled, false);
  assert.equal(store.getState().codeToolsEnabled, false);
  assert.equal(runtime.threadScopedOverride("toolsEnabled"), true);
  store.getState().setPermissionMode("ask");
  await runtime.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(threadRows.rows.get(THREAD)?.toolsEnabled, true);
  assert.equal(threadRows.rows.get(THREAD)?.codeToolsEnabled, true);
  store.getState().clearCheckpoint();
  assert.equal(store.getState().toolsEnabled, true);
  store.getState().setCheckpoint(MODEL, null);
  store.setState({ supportsTools: true, ...runtime.resolveToolsEnabledOnLoad(true) });
  assert.equal(store.getState().toolsEnabled, true);
  assert.equal(store.getState().codeToolsEnabled, true);
});

for (const pills of [
  { toolsEnabled: true, codeToolsEnabled: true },
  { deepResearchEnabled: true },
] as const) {
  test(`a snapshot written while a model loads keeps ${Object.keys(pills).join(" and ")} on the chat`, async (t) => {
    const runtime = await boot(t);
    const store = runtime.useChatRuntimeStore;
    store.getState().clearCheckpoint();
    store.getState().setCheckpoint("unsloth/no-tools", null);
    store.setState({ supportsTools: false, ...runtime.resolveToolsEnabledOnLoad(false) });
    store.getState().setActiveThreadId(THREAD);
    store.getState().applyThreadScopedSettings(THREAD, pills);
    for (const key of Object.keys(pills) as (keyof typeof pills)[]) {
      assert.equal(store.getState()[key], false, key);
    }
    const lease = store.getState().beginModelLoading("loading");
    assert.ok(lease !== null);
    store.getState().setPermissionMode("ask");
    await runtime.awaitThreadScopedSettingsWrite(THREAD);
    store.setState({ supportsTools: false, ...runtime.resolveToolsEnabledOnLoad(false) });
    store.getState().setPermissionMode("auto");
    await runtime.awaitThreadScopedSettingsWrite(THREAD);
    for (const key of Object.keys(pills) as (keyof typeof pills)[]) {
      assert.equal(threadRows.rows.get(THREAD)?.[key], true, key);
      assert.equal(runtime.threadScopedOverride(key), true, key);
    }
    store.getState().setCheckpoint(MODEL, null);
    store.setState({ supportsTools: true, ...runtime.resolveToolsEnabledOnLoad(true) });
    store.getState().endModelLoading(lease);
    for (const key of Object.keys(pills) as (keyof typeof pills)[]) {
      assert.equal(store.getState()[key], true, key);
    }
  });
}

for (const leave of [false, true]) {
  test(`Kimi keeps a Thinking click over a held Search click when the chat ${leave ? "is left" : "opens"}`, async (t) => {
    const runtime = await boot(t);
    const store = runtime.useChatRuntimeStore;
    useExternalProvidersStore.getState().setProviders([{
      id: "kimi",
      providerType: "kimi",
      name: "Kimi fixture",
      baseUrl: "https://api.moonshot.ai/v1",
      models: ["kimi-k2.5"],
      createdAt: 0,
      updatedAt: 0,
    }]);
    store.getState().setCheckpoint("external::kimi::kimi-k2.5", null);
    store.setState({
      supportsTools: false,
      supportsBuiltinWebSearch: true,
      toolsEnabled: false,
      reasoningEnabled: false,
    });
    store.getState().setActiveThreadId(THREAD);
    runtime.beginThreadScopedPairing(THREAD);
    store.getState().setToolsEnabled(true);
    store.getState().setReasoningEnabled(true);
    store.getState().setToolsEnabled(false, { persist: false });
    if (leave) {
      store.getState().setActiveThreadId(null);
      await runtime.commitHeldThreadScopedEditsToTheirThread();
    } else {
      store.getState().applyThreadScopedSettings(THREAD, null);
      assert.equal(store.getState().toolsEnabled, false);
      assert.equal(store.getState().reasoningEnabled, true);
      await runtime.awaitThreadScopedSettingsWrite(THREAD);
    }
    assert.equal(threadRows.rows.get(THREAD)?.toolsEnabled, false);
    assert.equal(threadRows.rows.get(THREAD)?.reasoningEnabled, true);
  });
}

for (const saved of [false, true]) {
  test(`Kimi's Thinking click after ${saved ? "a saved" : "an unsaved"} Search click leaves Search ${saved ? "on" : "off"} for the next model`, async (t) => {
    const runtime = await boot(t);
    const store = runtime.useChatRuntimeStore;
    useExternalProvidersStore.getState().setProviders([{
      id: "kimi",
      providerType: "kimi",
      name: "Kimi fixture",
      baseUrl: "https://api.moonshot.ai/v1",
      models: ["kimi-k2.5"],
      createdAt: 0,
      updatedAt: 0,
    }]);
    store.getState().setCheckpoint("external::kimi::kimi-k2.5", null);
    store.setState({
      supportsTools: false,
      supportsBuiltinWebSearch: true,
      toolsEnabled: false,
      reasoningEnabled: true,
    });
    store.getState().setActiveThreadId(THREAD);
    store.getState().applyThreadScopedSettings(THREAD, {
      toolsEnabled: false,
      reasoningEnabled: true,
    });
    store.getState().setToolsEnabled(true);
    store.getState().setReasoningEnabled(false, { persist: false });
    if (saved) await runtime.awaitThreadScopedSettingsWrite(THREAD);
    store.getState().setReasoningEnabled(true);
    store.getState().setToolsEnabled(false, { persist: false });
    assert.equal(runtime.threadScopedOverride("toolsEnabled"), saved);
    store.getState().clearCheckpoint();
    assert.equal(store.getState().toolsEnabled, saved);
    await runtime.awaitThreadScopedSettingsWrite(THREAD);
    assert.equal(threadRows.rows.get(THREAD)?.toolsEnabled, saved);
  });
}

test("enabling Search on a chat whose stored Research is clamped off clears the stored Research", async (t) => {
  const runtime = await boot(t);
  const store = runtime.useChatRuntimeStore;
  useExternalProvidersStore.getState().setProviders([{
    id: "search-only",
    providerType: "vllm",
    name: "Search-only fixture",
    baseUrl: "http://localhost:8000/v1",
    models: ["no-studio-tools"],
    createdAt: 0,
    updatedAt: 0,
  }]);
  setProviderModelCapabilities("vllm", { "no-studio-tools": { studio_tools: false } });
  store.getState().setCheckpoint("external::search-only::no-studio-tools", null);
  store.setState({ supportsTools: false, supportsBuiltinWebSearch: true });
  store.getState().setActiveThreadId(THREAD);
  store.getState().applyThreadScopedSettings(THREAD, { deepResearchEnabled: true });
  assert.equal(store.getState().deepResearchEnabled, false);
  assert.equal(runtime.threadScopedOverride("deepResearchEnabled"), true);
  store.getState().setToolsEnabled(true);
  await runtime.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(threadRows.rows.get(THREAD)?.toolsEnabled, true);
  assert.equal(threadRows.rows.get(THREAD)?.deepResearchEnabled, false);
  assert.equal(runtime.threadScopedOverride("deepResearchEnabled"), false);
});
