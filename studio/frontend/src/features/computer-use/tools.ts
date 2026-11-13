// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { computerUseReady, useComputerUseStore } from "./store";
import { COMPUTER_USE_TOOL } from "./summary";

/** The computer tool for this turn: none unless the composer switch is on and the server offers it. */
export function computerUseToolsForTurn(): string[] {
  const { enabled, status } = useComputerUseStore.getState();
  return enabled && computerUseReady(status) ? [COMPUTER_USE_TOOL] : [];
}
