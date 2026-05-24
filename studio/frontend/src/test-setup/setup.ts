// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom does not implement ResizeObserver; the HTML preview iframe uses one
// inside its srcDoc but the parent component also references it indirectly
// through DOM listeners. A no-op shim is enough for tests.
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver =
    ResizeObserverStub as unknown as typeof ResizeObserver;
}

// jsdom's URL.createObjectURL is not implemented and throws by default.
// HtmlPreview now loads its document through a blob: URL so the iframe gets
// an opaque origin and escapes the host Studio CSP. The shim below is enough
// for the renderer tests, which only assert the resulting src starts with
// "blob:" and never actually fetch the URL.
let __blobCounter = 0;
if (typeof URL.createObjectURL !== "function") {
  URL.createObjectURL = ((_blob: Blob) =>
    `blob:jsdom/${++__blobCounter}`) as typeof URL.createObjectURL;
}
if (typeof URL.revokeObjectURL !== "function") {
  URL.revokeObjectURL = (() => undefined) as typeof URL.revokeObjectURL;
}

afterEach(() => {
  cleanup();
});
