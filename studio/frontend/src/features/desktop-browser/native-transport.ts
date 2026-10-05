// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { invoke } from "@tauri-apps/api/core";
import type { BrowserTransport } from "./browser-session";

export const nativeBrowserTransport: BrowserTransport = {
  open: (sessionId, url, rect) =>
    invoke("desktop_browser_open", { sessionId, url, rect }),
  close: (sessionId) => invoke("desktop_browser_close", { sessionId }),
  navigate: (sessionId, url) =>
    invoke("desktop_browser_navigate", { sessionId, url }),
  action: (sessionId, action) =>
    invoke("desktop_browser_action", { sessionId, action }),
  setBounds: (sessionId, revision, rect, visible) =>
    invoke("desktop_browser_set_bounds", {
      sessionId,
      revision,
      rect,
      visible,
    }),
  snapshot: (sessionId) => invoke("desktop_browser_snapshot", { sessionId }),
  agent: (sessionId, op, args) =>
    invoke("desktop_browser_agent", { sessionId, op, args }),
};
