// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The reply notification is opt in and only fires while the window is in the background.
// runtime-provider.tsx and chat-tab.tsx are .tsx, which node's type stripping cannot import,
// so their wiring is pinned from source.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// Same "?bust=N" resolver as model-load-native-notifications.test.ts: api-base reads `isTauri`
// at evaluation, so each case re-evaluates the module graph under its own globals.
register("./helpers/notification-resolver.mjs", import.meta.url);

const MODULE = new URL(
  "../src/features/chat/utils/reply-finished-notification.ts",
  import.meta.url,
).href;
const STORE = new URL(
  "../src/features/chat/stores/chat-preferences-store.ts",
  import.meta.url,
).href;

type Env = {
  tauri: boolean;
  enabled: boolean;
  hidden?: boolean;
  focused?: boolean;
  permission?: "absent" | "default" | "granted" | "denied";
  answer?: "granted" | "denied";
  pluginGranted?: boolean;
};

type Control = {
  sent: { title: string; body?: string }[];
  granted: boolean;
  mode: string;
  requests: number;
};

let generation = 0;

function memoryStorage(): Storage {
  const items = new Map<string, string>();
  return {
    get length() {
      return items.size;
    },
    clear: () => items.clear(),
    getItem: (key) => items.get(key) ?? null,
    key: (index) => [...items.keys()][index] ?? null,
    removeItem: (key) => void items.delete(key),
    setItem: (key, value) => void items.set(key, String(value)),
  };
}

function define(name: string, value: unknown) {
  Object.defineProperty(globalThis, name, {
    value,
    configurable: true,
    writable: true,
  });
}

async function load(env: Env) {
  const {
    tauri,
    enabled,
    hidden = true,
    focused = !hidden,
    permission = "absent",
    answer = "granted",
    pluginGranted = false,
  } = env;
  const created: { title: string; options?: NotificationOptions }[] = [];
  const storage = memoryStorage();
  define("localStorage", storage);
  const windowStub: Record<string, unknown> = {
    location: { protocol: tauri ? "tauri:" : "http:" },
    localStorage: storage,
    focus() {},
  };
  if (tauri) {
    windowStub.__TAURI_INTERNALS__ = {};
  }
  if (permission !== "absent") {
    const api = function (title: string, options?: NotificationOptions) {
      created.push({ title, options });
      return { close() {}, onclick: null };
    } as unknown as {
      permission: string;
      requestPermission: () => Promise<string>;
    };
    api.permission = permission;
    api.requestPermission = async () => {
      api.permission = answer;
      return answer;
    };
    windowStub.Notification = api;
  }
  define("window", windowStub);
  define("document", {
    visibilityState: hidden ? "hidden" : "visible",
    hasFocus: () => focused,
  });

  const control = ((globalThis as Record<string, unknown>)
    .__TAURI_NOTIFICATION_STUB__ ??= {}) as Control;
  control.sent = [];
  control.granted = pluginGranted;
  control.mode = "ok";
  control.requests = 0;

  generation += 1;
  const store = await import(`${STORE}?bust=${generation}`);
  store.useChatPreferencesStore.setState({ notifyWhenReplyFinishes: enabled });
  const mod = (await import(`${MODULE}?bust=${generation}`)) as {
    notifyReplyFinished: () => void;
    requestReplyFinishedPermission: () => Promise<boolean>;
  };
  return { mod, control, created };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test("off by default, including installs that never saw the setting", async () => {
  const { useChatPreferencesStore } = await import(
    `${STORE}?bust=default`
  );
  assert.equal(
    useChatPreferencesStore.getInitialState().notifyWhenReplyFinishes,
    false,
  );
  assert.match(
    readSrc("features/chat/stores/chat-preferences-store.ts"),
    /notifyWhenReplyFinishes: saved\?\.notifyWhenReplyFinishes === true/,
  );
});

test("a finished reply notifies nothing while the setting is off", async () => {
  for (const tauri of [true, false]) {
    const { mod, control, created } = await load({
      tauri,
      enabled: false,
      permission: "granted",
      pluginGranted: true,
    });
    mod.notifyReplyFinished();
    await settle();
    assert.equal(control.sent.length, 0);
    assert.equal(created.length, 0);
  }
});

test("Desktop sends one native notification per reply finished in the background", async () => {
  const { mod, control } = await load({
    tauri: true,
    enabled: true,
    pluginGranted: true,
  });
  mod.notifyReplyFinished();
  await settle();
  mod.notifyReplyFinished();
  await settle();
  assert.equal(control.sent.length, 2);
  assert.equal(control.sent[0].title, "Reply finished");
  assert.equal(control.requests, 0, "a finished reply never prompts");
});

test("an unfocused but visible window still counts as the background", async () => {
  const { mod, control } = await load({
    tauri: true,
    enabled: true,
    hidden: false,
    focused: false,
    pluginGranted: true,
  });
  mod.notifyReplyFinished();
  await settle();
  assert.equal(control.sent.length, 1);
});

test("nothing fires while the user is looking at the window", async () => {
  for (const tauri of [true, false]) {
    const { mod, control, created } = await load({
      tauri,
      enabled: true,
      hidden: false,
      focused: true,
      permission: "granted",
      pluginGranted: true,
    });
    mod.notifyReplyFinished();
    await settle();
    assert.equal(control.sent.length, 0);
    assert.equal(created.length, 0);
  }
});

test("the browser notifies only with permission already granted", async () => {
  const granted = await load({
    tauri: false,
    enabled: true,
    permission: "granted",
  });
  granted.mod.notifyReplyFinished();
  assert.equal(granted.created.length, 1);
  assert.equal(granted.created[0].title, "Reply finished");
  assert.equal(granted.created[0].options?.tag, "unsloth-reply-finished");

  for (const permission of ["default", "denied", "absent"] as const) {
    const env = await load({ tauri: false, enabled: true, permission });
    env.mod.notifyReplyFinished();
    assert.equal(env.created.length, 0, permission);
  }
});

test("turning the switch on asks for permission and reports the answer", async () => {
  const asked = await load({
    tauri: false,
    enabled: true,
    permission: "default",
    answer: "granted",
  });
  assert.equal(await asked.mod.requestReplyFinishedPermission(), true);

  const refused = await load({
    tauri: false,
    enabled: true,
    permission: "default",
    answer: "denied",
  });
  assert.equal(await refused.mod.requestReplyFinishedPermission(), false);

  for (const permission of ["denied", "absent"] as const) {
    const env = await load({ tauri: false, enabled: true, permission });
    assert.equal(await env.mod.requestReplyFinishedPermission(), false);
  }

  const desktop = await load({
    tauri: true,
    enabled: true,
    pluginGranted: true,
  });
  assert.equal(await desktop.mod.requestReplyFinishedPermission(), true);
});

test("the run adapter notifies once after the stream ends, never on a stop", () => {
  const source = readSrc("features/chat/runtime-provider.tsx");
  const start = source.indexOf("function createPersistedRunAdapter(");
  assert.notEqual(start, -1, "createPersistedRunAdapter is gone");
  const body = source.slice(start, source.indexOf("\nfunction ", start + 1));
  assert.match(
    body,
    /yield\* result;\n\s*\} else \{\n\s*yield await result;\n\s*\}\n\s*if \(!options\.abortSignal\?\.aborted\) \{\n\s*notifyReplyFinished\(\);/,
  );
  assert.equal(body.match(/notifyReplyFinished\(\)/g)?.length, 1);
});

test("the setting is a searchable switch in Chat settings that reverts when refused", async () => {
  const tab = readSrc("features/settings/tabs/chat-tab.tsx");
  const at = tab.indexOf('label={t("settings.chat.replyNotification")}');
  assert.notEqual(at, -1);
  const row = tab.slice(at, tab.indexOf("</SettingsRow>", at));
  assert.match(row, /checked=\{notifyWhenReplyFinishes\}/);
  assert.match(row, /requestReplyFinishedPermission\(\)/);
  assert.match(row, /setNotifyWhenReplyFinishes\(false\)/);
  const { SETTINGS_SEARCH_INDEX } = await import(
    new URL("../src/features/settings/settings-search.ts", import.meta.url)
      .href
  );
  assert.ok(
    SETTINGS_SEARCH_INDEX.chat.includes("settings.chat.replyNotification"),
  );
});
