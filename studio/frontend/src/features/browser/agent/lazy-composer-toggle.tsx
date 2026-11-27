// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense, type ComponentProps } from "react";
import type { BrowserAgentPill as Pill } from "./composer-toggle";
import { useBrowserAgentStore } from "./agent-store";

// Off the startup bundle: the item loads with the + menu, the pill only once browser use is on.
const MenuItem = lazy(() =>
  import("./composer-toggle").then((module) => ({ default: module.BrowserAgentMenuItem })),
);
const PillComponent = lazy(() =>
  import("./composer-toggle").then((module) => ({ default: module.BrowserAgentPill })),
);

export function BrowserAgentMenuItem() {
  return (
    <Suspense fallback={null}>
      <MenuItem />
    </Suspense>
  );
}

export function BrowserAgentPill(props: ComponentProps<typeof Pill>) {
  const enabled = useBrowserAgentStore((state) => state.enabled);
  if (!enabled) return null;
  return (
    <Suspense fallback={null}>
      <PillComponent {...props} />
    </Suspense>
  );
}
