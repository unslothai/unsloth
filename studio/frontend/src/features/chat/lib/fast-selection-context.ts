// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext } from "react";

/** The route supplies the same callback used by the model picker. */
export const FastModelSelectionContext = createContext<
  ((checkpoint: string) => void) | null
>(null);
