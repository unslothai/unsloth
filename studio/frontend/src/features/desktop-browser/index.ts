// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { BrowserSplit } from "./browser-split";
export { shouldCloseBrowser, type BrowserContext } from "./browser-session";
export {
  desktopBrowserToolsOn,
  useDesktopBrowserStore,
  type BrowserApprovalChoice,
} from "./browser-store";
export {
  parseBrowserClientRequest,
  runBrowserClientRequest,
} from "./agent/agent-executor";
export {
  resultSummary,
  siteOf,
  supersedeBrowserSnapshots,
} from "./agent/agent-format";
export { browserToolsForTurn } from "@/lib/browser-tool-names";
