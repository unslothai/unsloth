// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PAGES = [
  ["images", "features/images/images-page.tsx"],
  ["video", "features/video/video-page.tsx"],
] as const;

for (const [name, path] of PAGES) {
  test(`the ${name} pill eject reports the unload like the chat pill`, () => {
    const page = readSrc(path);
    assert.match(page, /onEject=\{status\?\.loaded \? handleEject : undefined\}/);
    const start = page.indexOf("const handleEject = useCallback");
    assert.notEqual(start, -1, "handleEject is not declared");
    const body = page.slice(start, page.indexOf("}, [handleUnload]);", start));
    const guard = body.indexOf('toast.info("Wait for the model to finish unloading.")');
    const loading = body.indexOf('toast.loading("Unloading model")');
    const unload = body.indexOf("await handleUnload()");
    assert.ok(guard !== -1 && guard < loading, "a second eject is not turned away before the toast");
    assert.ok(loading !== -1 && loading < unload, "the toast does not open before the unload starts");
    assert.match(body, /toast\.success\("Model unloaded", \{ id: toastId/);
    assert.match(body, /toast\.dismiss\(toastId\)/);
  });
}
