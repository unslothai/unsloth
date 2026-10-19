// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext } from "react";

// "all": answers, coloured on the main thread while they grow. "settled": thinking traces,
// coloured on the worker (a growing fence only while on screen).
export type CodeHighlightMode = "all" | "settled";

export const CodeHighlightModeContext = createContext<CodeHighlightMode>("all");
