// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ComponentProps, Suspense, lazy } from "react";
import type { ComputerUsePill as Pill } from "./composer-toggle";
import { useComputerUseStore } from "./store";

// Off the startup bundle: the item loads with the + menu, the pill only once computer use is on.
const MenuItem = lazy(() =>
  import("./composer-toggle").then((module) => ({ default: module.ComputerUseMenuItem })),
);
const PillComponent = lazy(() =>
  import("./composer-toggle").then((module) => ({ default: module.ComputerUsePill })),
);

export function ComputerUseMenuItem() {
  return (
    <Suspense fallback={null}>
      <MenuItem />
    </Suspense>
  );
}

export function ComputerUsePill(props: ComponentProps<typeof Pill>) {
  const enabled = useComputerUseStore((state) => state.enabled);
  if (!enabled) return null;
  return (
    <Suspense fallback={null}>
      <PillComponent {...props} />
    </Suspense>
  );
}
