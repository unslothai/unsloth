// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { openUrlInBrowserPanel } from "@/features/browser";
import { openLink } from "@/lib/open-link";

export const MCP_ENGINES_DOCS = "https://unsloth.ai/docs/basics/mcp/mcp-engines";

/** Opens the app's install steps in Unsloth's browser, else the system browser. `close` first, or the modal hides the panel. */
export function openInstallGuide(section: string, close?: () => void): void {
  const url = `${MCP_ENGINES_DOCS}#${section}`;
  close?.();
  if (!openUrlInBrowserPanel(url)) openLink(url);
}
