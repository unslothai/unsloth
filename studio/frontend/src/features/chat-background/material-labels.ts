// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type { WindowMaterial } from "./model";

export const MATERIAL_LABELS: Record<WindowMaterial, TranslationKey> = {
  solid: "chatBackground.materialSolid",
  transparent: "chatBackground.materialTransparent",
  frosted: "chatBackground.materialFrosted",
  blurred: "chatBackground.materialBlurred",
  liquid: "chatBackground.materialLiquid",
  "liquid-3d": "chatBackground.materialLiquid3d",
  macos: "chatBackground.materialMacos",
};
