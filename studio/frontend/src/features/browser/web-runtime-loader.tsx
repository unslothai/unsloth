// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense } from "react";
import { currentEntry, useBrowserStore } from "./store";

const Runtime = lazy(() => import("./web-runtime").then((module) => ({ default: module.WebRuntime })));

export function WebRuntimeLoader() {
  const hasWebTab = useBrowserStore((state) => state.tabs.some((tab) => currentEntry(tab).kind === "web"));
  return hasWebTab ? <Suspense fallback={null}><Runtime /></Suspense> : null;
}
