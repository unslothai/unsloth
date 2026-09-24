// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext } from "react";

/** The page's selection, keyed `item:<id>` or `folder:<id>` like the list rows. */
export const CardSelectionContext = createContext<{
  selection: ReadonlySet<string>;
  toggle: (key: string) => void;
} | null>(null);
