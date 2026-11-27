// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";

import type * as SourcesModule from "../src/components/assistant-ui/sources.tsx";
import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { store: storage } = installLocalStorageFake();
registerBundlerResolver();
const { useChatPreferencesStore: preferences } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);
const STORAGE_KEY = "unsloth_chat_preferences";

beforeEach(() => {
  preferences.setState(preferences.getInitialState(), true);
  storage.clear();
});

test("link previews remain enabled for new users", () => {
  assert.equal(preferences.getInitialState().showLinkPreviews, true);
});

test("older preferences gain enabled previews without resetting saved choices", async () => {
  storage.set(
    STORAGE_KEY,
    JSON.stringify({
      version: 1,
      state: {
        showResponseModel: true,
        autoScrollWhileGenerating: false,
        sendShortcut: "mod-enter",
      },
    }),
  );
  await preferences.persist.rehydrate();
  const state = preferences.getState();
  assert.equal(state.showLinkPreviews, true);
  assert.equal(state.showResponseModel, true);
  assert.equal(state.autoScrollWhileGenerating, false);
  assert.equal(state.sendShortcut, "mod-enter");
});

test("both preview choices survive reloading the persisted preferences", async () => {
  for (const enabled of [false, true]) {
    preferences.getState().setShowLinkPreviews(enabled);
    const saved = storage.get(STORAGE_KEY)!;
    assert.equal(JSON.parse(saved).state.showLinkPreviews, enabled);
    preferences.setState(preferences.getInitialState(), true);
    storage.set(STORAGE_KEY, saved);
    await preferences.persist.rehydrate();
    assert.equal(preferences.getState().showLinkPreviews, enabled);
  }
});

test("setting the same preview choice repeatedly preserves the stored state", () => {
  for (const enabled of [false, true]) {
    preferences.getState().setShowLinkPreviews(enabled);
    const saved = storage.get(STORAGE_KEY);
    preferences.getState().setShowLinkPreviews(enabled);
    assert.equal(storage.get(STORAGE_KEY), saved);
    assert.equal(preferences.getState().showLinkPreviews, enabled);
  }
});

const passthrough = ({ children }: { children?: React.ReactNode }) =>
  React.createElement(React.Fragment, null, children);
let showLinkPreviews = true;
const { SourcesGroup } = loadWithStubs<typeof SourcesModule>(
  new URL("../src/components/assistant-ui/sources.tsx", import.meta.url),
  {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/lib/open-link": { openLink: () => false },
    "@assistant-ui/react": {
      useAuiState: (select: (state: unknown) => unknown) =>
        select({ message: { content: [] } }),
    },
    "@/components/assistant-ui/message-derived": {
      memoOnArray: (select: unknown) => select,
    },
    "@/features/chat/stores/chat-preferences-store": {
      useChatPreferencesStore: (
        select: (state: { showLinkPreviews: boolean }) => unknown,
      ) => select({ showLinkPreviews }),
    },
    "@/lib/utils": {
      cn: (...classes: unknown[]) => classes.filter(Boolean).join(" "),
    },
    "./badge": { Badge: passthrough, badgeVariants: () => "" },
    "@/components/ui/hover-card": {
      HoverCard: ({ children }: { children?: React.ReactNode }) =>
        React.createElement("div", { "data-preview-root": true }, children),
      HoverCardTrigger: passthrough,
      HoverCardContent: ({ children }: { children?: React.ReactNode }) =>
        React.createElement("aside", { "data-preview-content": true }, children),
    },
  },
);

const SOURCE = {
  id: "source-1",
  url: "https://example.com/reference",
  title: "Reference article",
  description: "Details shown in the hover preview.",
};

function renderSources(enabled: boolean): string {
  showLinkPreviews = enabled;
  return renderToStaticMarkup(
    React.createElement(SourcesGroup, {
      sources: [SOURCE],
      allowRemoteIcons: false,
    }),
  );
}

test("disabling previews keeps the source link without mounting a hover card", () => {
  const html = renderSources(false);
  assert.match(html, /href="https:\/\/example\.com\/reference"/);
  assert.match(html, /Reference article/);
  assert.doesNotMatch(html, /data-preview-root|data-preview-content/);
  assert.doesNotMatch(html, /Details shown in the hover preview/);
});

test("enabling previews restores the card around the same source link", () => {
  const html = renderSources(true);
  assert.match(html, /href="https:\/\/example\.com\/reference"/);
  assert.match(html, /data-preview-root="true"/);
  assert.match(html, /data-preview-content="true"/);
  assert.match(html, /Details shown in the hover preview/);
});
