// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("Chat with files keeps its cards 12px under the header", () => {
  assert.match(
    readSrc("index.css"),
    /\.unsloth-files-panel \{[^}]*@apply relative mb-2 flex w-full flex-col gap-3 overflow-hidden rounded-2xl px-4 pt-2 pb-4;/,
  );
});
