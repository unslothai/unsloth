// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import * as jsx from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { useAppearanceCustomStore } from "../src/features/settings/stores/appearance-custom-store.ts";

const { OwnerAvatar } = loadWithStubs<{
  OwnerAvatar: React.ComponentType<{ owner: string; remote?: boolean }>;
}>(new URL("../src/features/hub/catalog/owner-avatar.tsx", import.meta.url), {
  react: React,
  "react/jsx-runtime": jsx,
  "@/lib/utils": { cn: (...values: unknown[]) => values.filter(Boolean).join(" ") },
  "../lib/avatar-theme": { ownerPaletteColor: () => "#123456" },
  "../lib/hf-owner-avatar": { useHfOwnerAvatar: () => "https://example.com/author.png" },
  "../lib/provider-logos": { resolveOwnerProviderLogo: () => null },
});

test("global mascot preference preserves model-author photos and Unsloth branding", () => {
  for (const showMascots of [false, true]) {
    useAppearanceCustomStore.getState().patch({ showMascots });
    const author = renderToStaticMarkup(React.createElement(OwnerAvatar, { owner: "author" }));
    assert.match(author, /<img[^>]+src="https:\/\/example.com\/author.png"/);
    const brand = renderToStaticMarkup(React.createElement(OwnerAvatar, { owner: "unsloth", remote: false }));
    assert.match(brand, /<img[^>]+src="\/rounded.png"/);
  }
});
