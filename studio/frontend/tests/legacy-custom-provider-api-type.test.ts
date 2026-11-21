// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("only Custom connections expose API type selection", () => {
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  assert.match(dialog, /function shouldShowProviderApiType\(providerType: string\): boolean \{\s*return providerType === LEGACY_CUSTOM_PROVIDER_TYPE;\s*\}/);
  assert.match(dialog, /\{shouldShowProviderApiType\(providerType\) \? \(/);
});
