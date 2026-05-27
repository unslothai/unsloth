// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom shims: ResizeObserver + URL.createObjectURL.
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver =
    ResizeObserverStub as unknown as typeof ResizeObserver;
}

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
