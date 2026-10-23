// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createCodePlugin } from "./code-plugin";

// Replaces the `code` export of @streamdown/code, whose token cache never evicts. One instance so
// these surfaces share a single set of Shiki highlighters, as they did with that export.
export const codePlugin = createCodePlugin();
