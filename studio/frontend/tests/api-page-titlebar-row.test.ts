// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the API page's model picker sits in the titlebar row, where Chat's does", () => {
  const page = readSrc("features/api-monitor/api-monitor-page.tsx");
  const root = readSrc("app/routes/__root.tsx");
  assert.match(
    root,
    /const isChatLike = isChatRoute \|\| isImagesRoute \|\| isVideoRoute \|\| isAudioRoute;/,
  );
  assert.match(root, /--studio-non-chat-content-top-inset/);
  assert.match(
    page,
    /!isMobileShell &&\s*"-mt-\[calc\(var\(--studio-non-chat-content-top-inset,var\(--studio-content-top-inset,0px\)\)-var\(--studio-content-top-inset,0px\)\)\]"/,
  );
  assert.match(page, /"pointer-events-none sticky top-0 z-40 flex/);
  assert.match(
    page,
    /data-tour="api-toolbar" className="pointer-events-auto flex/,
  );
  const chat = readSrc("features/chat/chat-page.tsx");
  assert.match(
    chat,
    /"pointer-events-none relative z-40 flex h-\[calc\(48px\*var\(--ui-space-scale,1\)\)\] shrink-0 items-start/,
  );
});
