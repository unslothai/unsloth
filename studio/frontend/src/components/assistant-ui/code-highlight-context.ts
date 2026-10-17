// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext } from "react";

// Answers keep syntax highlighting; Thinking code stays plain while streaming and when complete.
export type CodeHighlightMode = "all" | "plain";

export const CodeHighlightModeContext = createContext<CodeHighlightMode>("all");
