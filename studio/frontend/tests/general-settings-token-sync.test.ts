// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

const tabUrl = new URL(
  "../src/features/settings/tabs/general-tab.tsx",
  import.meta.url,
);

// Every update-banner preference the tab can import, read off the module itself: #12893 added
// the audio.cpp banner to the tab after this test pinned three names, and every case failed on
// the missing export. Setters are no-ops and readers say the banner is hidden.
const updateBannerPrefStub = Object.fromEntries(
  [
    ...readFileSync(
      new URL("../src/hooks/use-llama-update-pref.ts", import.meta.url),
      "utf8",
    ).matchAll(/^export function (\w+)/gm),
  ].map(([, name]) => [name, name.startsWith("set") ? () => undefined : () => false]),
);

type EffectSlot = {
  cleanup?: () => void;
  deps?: unknown[];
  next?: () => undefined | (() => void);
};

function sameDeps(left: unknown[] | undefined, right: unknown[] | undefined) {
  return (
    left !== undefined &&
    right !== undefined &&
    left.length === right.length &&
    left.every((value, index) => Object.is(value, right[index]))
  );
}

function nodes(node: unknown): StubElement[] {
  if (Array.isArray(node)) {
    return node.flatMap(nodes);
  }
  if (!node || typeof node !== "object" || !("props" in node)) {
    return [];
  }
  const element = node as StubElement;
  return [element, ...nodes(element.props.children)];
}

function generalTab(initialToken: string) {
  const states: unknown[] = [];
  const refs: Array<{ current: unknown }> = [];
  const effects: EffectSlot[] = [];
  const writes: string[] = [];
  let stateCursor = 0;
  let refCursor = 0;
  let effectCursor = 0;
  const translate = (key: string) => key;
  const runtime = {
    hfToken: initialToken,
    setHfToken(token: string) {
      writes.push(token);
      runtime.hfToken = token;
      credential.isPersisting = true;
      credential.persistenceError = null;
    },
  };
  const credential: {
    isPersisting: boolean;
    persistenceError: string | null;
  } = { isPersisting: false, persistenceError: null };
  const componentStub = new Proxy(
    {},
    { get: (_target, property) => String(property) },
  );
  const noop = () => undefined;
  const never = () => new Promise<never>(() => undefined);
  const api = loadWithStubs<{ GeneralTab: () => StubElement }>(tabUrl, {
    "react/jsx-runtime": stubJsxRuntime(),
    react: {
      useState(initial: unknown) {
        const index = stateCursor++;
        if (!(index in states)) {
          states[index] = initial;
        }
        return [
          states[index],
          (value: unknown) => {
            states[index] =
              typeof value === "function"
                ? (value as (current: unknown) => unknown)(states[index])
                : value;
          },
        ];
      },
      useRef(initial: unknown) {
        const index = refCursor++;
        refs[index] ??= { current: initial };
        return refs[index];
      },
      useEffect(effect: () => undefined | (() => void), deps?: unknown[]) {
        const index = effectCursor++;
        const slot = effects[index] ?? {};
        if (!sameDeps(slot.deps, deps)) {
          slot.next = effect;
          slot.deps = deps;
        }
        effects[index] = slot;
      },
    },
    "@/components/ui/button": componentStub,
    "@/components/ui/dialog": componentStub,
    "@/components/ui/input": { Input: "Input" },
    "@/components/ui/switch": componentStub,
    "@/features/chat": {
      useChatRuntimeStore: Object.assign(
        (selector: (state: typeof runtime) => unknown) => selector(runtime),
        { getState: () => runtime },
      ),
    },
    "@/features/chat/stores/sidebar-organization-keys": {
      SIDEBAR_ORGANIZATION_STORAGE_KEY: "sidebar-order",
    },
    "@/features/loaded-models": {
      LOADED_MODELS_PREFERENCE_KEYS: {},
      setShowLoadedModels: noop,
      useShowLoadedModels: () => false,
    },
    "@/features/hub": {
      TRANSPORT_MODE_STORAGE_KEY: "transport",
      useHfTokenStore: Object.assign(
        (selector: (state: typeof credential) => unknown) =>
          selector(credential),
        { getState: () => credential },
      ),
    },
    "@/features/training": {
      emitTrainingRunsChanged: noop,
      TRAINING_UI_PREFERENCE_KEYS: [],
    },
    "@/hooks/use-llama-update-pref": updateBannerPrefStub,
    "@/hooks": {
      useHfTokenValidation: () => ({
        isValid: false,
        isChecking: false,
        error: null,
      }),
    },
    "@/i18n": { LOCALE_STORAGE_KEY: "locale", useT: () => translate },
    "@/lib/api-base": { isTauri: false },
    "@/lib/spellcheck": {
      SPELLCHECK_STORAGE_KEY: "unsloth_spellcheck",
      setSpellCheck: noop,
      useSpellCheck: () => true,
    },
    "@/lib/toast": { toast: { success: noop, error: noop } },
    "@/lib/utils": {
      cn: (...values: unknown[]) => values.filter(Boolean).join(" "),
    },
    "lucide-react": componentStub,
    "../api/helper-precache": {
      loadHelperPrecacheSettings: never,
      updateHelperPrecacheSettings: never,
    },
    "../api/preview-sharing": {
      loadPreviewSharing: never,
      rotatePreviewLinks: never,
      updatePreviewSharing: never,
    },
    "../api/upload-limit": {
      DEFAULT_UPLOAD_LIMIT_MB: 100,
      loadUploadLimitSettings: never,
      updateUploadLimitSettings: never,
    },
    "../api/close-to-tray": {
      loadCloseToTray: noop,
      updateCloseToTray: noop,
    },
    "../api/launch-at-login": {
      loadLaunchAtLogin: noop,
      updateLaunchAtLogin: noop,
    },
    "@/features/auth": { useIsAccountOwner: () => false },
    "@/features/library": {
      LIBRARY_CHATS_PREFS_STORAGE_KEY: "library-chats",
      LIBRARY_SETTINGS_STORAGE_KEY: "library-settings",
      LIBRARY_VIEW_STORAGE_KEY: "library-view",
    },
    "../components/change-password-dialog": componentStub,
    "../components/desktop-repair-control": componentStub,
    "../components/desktop-update-control": componentStub,
    "../components/documents-rag-section": componentStub,
    "../components/language-select": componentStub,
    "../components/download-transport-row": componentStub,
    "../components/hub-settings-section": componentStub,
    "../components/settings-row": componentStub,
    "../components/settings-section": componentStub,
    "../components/studio-version-section": componentStub,
    "../hooks/use-desktop-boolean-setting": {
      useDesktopBooleanSetting: () => ({
        error: null,
        saving: false,
        supported: false,
        update: noop,
        value: false,
      }),
    },
    "../stores/keyboard-shortcuts-store": {
      KEYBOARD_SHORTCUTS_STORAGE_KEY: "shortcuts",
    },
    "../stores/interface-scale-store": {
      INTERFACE_SCALE_STORAGE_KEY: "scale",
    },
    "../stores/settings-panel-prefs-store": {
      SETTINGS_PANEL_PREFS_STORAGE_KEY: "panel-prefs",
    },
    "@/features/chat/utils/project-attachment-target": {
      CHAT_PROJECT_ATTACHMENT_TARGET_KEY: "attachment-target",
    },
  });

  const flushEffects = () => {
    for (const slot of effects) {
      if (!slot.next) {
        continue;
      }
      slot.cleanup?.();
      const effect = slot.next;
      slot.next = undefined;
      slot.cleanup = effect() ?? undefined;
    }
  };
  const render = (runEffects = true) => {
    stateCursor = 0;
    refCursor = 0;
    effectCursor = 0;
    const tree = api.GeneralTab();
    if (runEffects) {
      flushEffects();
    }
    return tree;
  };
  return {
    render,
    writes,
    setRemoteToken(token: string) {
      runtime.hfToken = token;
    },
    settleSave(token: string, error: string | null = null) {
      runtime.hfToken = token;
      credential.isPersisting = false;
      credential.persistenceError = error;
    },
    unmount() {
      for (const slot of effects) {
        slot.cleanup?.();
      }
    },
  };
}

function tokenInput(tree: StubElement) {
  const input = nodes(tree).find((node) => node.type === "Input");
  assert.ok(input);
  return input;
}

test("an untouched General Settings tab adopts a token changed in another tab", () => {
  const tab = generalTab("hf_old");
  tab.render();
  tab.setRemoteToken("hf_new");
  tab.render();
  const tree = tab.render();

  assert.equal(tokenInput(tree).props.value, "hf_new");
  tab.unmount();
  assert.deepEqual(tab.writes, []);
});

test("blur before the remote sync effect cannot commit an untouched stale draft", () => {
  const tab = generalTab("hf_old");
  tab.render();
  tab.setRemoteToken("hf_new");
  const tree = tab.render(false);

  (tokenInput(tree).props.onBlur as () => void)();
  tab.unmount();
  assert.deepEqual(tab.writes, []);
});

test("a locally edited token remains authoritative across a remote update", () => {
  const tab = generalTab("hf_old");
  const input = tokenInput(tab.render());
  (input.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: " hf_local " },
  });
  tab.render();
  tab.setRemoteToken("hf_remote");
  const tree = tab.render();

  assert.equal(tokenInput(tree).props.value, " hf_local ");
  tab.unmount();
  assert.deepEqual(tab.writes, ["hf_local"]);
});

test("a blurred edit relinquishes the draft to later remote updates", () => {
  const tab = generalTab("hf_old");
  let input = tokenInput(tab.render());
  (input.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: "hf_local" },
  });
  input = tokenInput(tab.render());
  (input.props.onBlur as () => void)();
  tab.render();
  tab.settleSave("hf_local");
  tab.render();

  tab.setRemoteToken("hf_remote");
  tab.render();
  const tree = tab.render();

  assert.equal(tokenInput(tree).props.value, "hf_remote");
  tab.unmount();
  assert.deepEqual(tab.writes, ["hf_local"]);
});

test("a settled save adopts the newer token reconciled from another tab", () => {
  const tab = generalTab("hf_old");
  let input = tokenInput(tab.render());
  (input.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: "hf_local" },
  });
  input = tokenInput(tab.render());
  (input.props.onBlur as () => void)();
  tab.render();

  tab.settleSave("hf_remote");
  tab.render();
  const tree = tab.render();

  assert.equal(tokenInput(tree).props.value, "hf_remote");
  tab.unmount();
  assert.deepEqual(tab.writes, ["hf_local"]);
});

test("closing before the settled-save effect cannot repost the submitted draft", () => {
  const tab = generalTab("hf_old");
  let input = tokenInput(tab.render());
  (input.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: "hf_local" },
  });
  input = tokenInput(tab.render());
  (input.props.onBlur as () => void)();
  tab.render();

  tab.settleSave("hf_remote");
  tab.render(false);
  tab.unmount();

  assert.deepEqual(tab.writes, ["hf_local"]);
});

test("typing after submit keeps the newer draft through save reconciliation", () => {
  const tab = generalTab("hf_old");
  let input = tokenInput(tab.render());
  (input.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: "hf_first" },
  });
  input = tokenInput(tab.render());
  (input.props.onBlur as () => void)();
  input = tokenInput(tab.render());
  (input.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: "hf_second" },
  });

  tab.settleSave("hf_remote");
  const tree = tab.render();

  assert.equal(tokenInput(tree).props.value, "hf_second");
  tab.unmount();
  assert.deepEqual(tab.writes, ["hf_first", "hf_second"]);
});

test("a failed save retains the edited draft and retries it on close", () => {
  const tab = generalTab("hf_old");
  let input = tokenInput(tab.render());
  (input.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: "hf_local" },
  });
  input = tokenInput(tab.render());
  (input.props.onBlur as () => void)();
  tab.render();

  tab.settleSave("hf_old", "Could not save the token.");
  const tree = tab.render();

  assert.equal(tokenInput(tree).props.value, "hf_local");
  tab.unmount();
  assert.deepEqual(tab.writes, ["hf_local", "hf_local"]);
});

test("a remote update after failure cannot masquerade as the failed save", () => {
  const tab = generalTab("hf_old");
  let input = tokenInput(tab.render());
  (input.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: "hf_local" },
  });
  input = tokenInput(tab.render());
  (input.props.onBlur as () => void)();
  tab.render();
  tab.settleSave("hf_old", "Could not save the token.");
  tab.render();

  tab.settleSave("hf_remote");
  const tree = tab.render();

  assert.equal(tokenInput(tree).props.value, "hf_local");
  tab.unmount();
  assert.deepEqual(tab.writes, ["hf_local", "hf_local"]);
});
