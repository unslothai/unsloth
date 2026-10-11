// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext } from "react";

/**
 * "all": answers, every fence coloured, a growing one included.
 * "settled": thinking traces. A fence is coloured once its closing delimiter lands, when it nears
 * the viewport, and never while it grows: colouring the growing block is what doubled a trace's
 * main-thread cost.
 */
export type CodeHighlightMode = "all" | "settled";

export const CodeHighlightModeContext = createContext<CodeHighlightMode>("all");
