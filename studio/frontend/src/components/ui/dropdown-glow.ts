// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";

export type DropdownGlow = "card" | "picker";

export const DropdownGlowContext = createContext<DropdownGlow | null>(null);

const CARD_SURFACE = /(?:^|\s)(?:dialog-soft-surface|settings-surface)(?:\s|$)/;

export function dialogDropdownGlow(
  className: string | undefined,
  inherited: DropdownGlow | null,
): DropdownGlow | null {
  if (className === undefined) return inherited;
  return CARD_SURFACE.test(className) || className.includes("dark:bg-card")
    ? "card"
    : inherited;
}

export function useDropdownGlow(): DropdownGlow | undefined {
  return useContext(DropdownGlowContext) ?? undefined;
}
