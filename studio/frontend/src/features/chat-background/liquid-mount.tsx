// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Suspense, lazy } from "react";
import { useMaterialStore } from "./material";

// The refraction library loads only once a Liquid Glass material and glass menus are on.
const Layers = lazy(() =>
  import("./liquid-glass").then((module) => ({
    default: module.GlassMenuLayers,
  })),
);

export function LiquidGlassMount() {
  const material = useMaterialStore((s) => s.material);
  const menus = useMaterialStore((s) => s.menus);
  if (!menus || (material !== "liquid" && material !== "liquid-3d"))
    return null;
  return (
    <Suspense fallback={null}>
      <Layers active={true} />
    </Suspense>
  );
}
