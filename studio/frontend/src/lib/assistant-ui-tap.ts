// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `@assistant-ui/tap` as the app and assistant-ui load it (a vite alias): upstream, except `tapResources`,
// which skips children whose element and context did not change (./assistant-ui-tap-resources.ts).
export * from "../../node_modules/@assistant-ui/tap/dist/index.js";
export { tapResources, withContextProvider } from "./assistant-ui-tap-resources";
