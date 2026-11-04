// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function browserScopeContains(section: HTMLElement | null, tabId: string | null, node: Node): boolean {
  if (section?.contains(node)) return true;
  if (!tabId) return false;
  const element = node instanceof Element ? node : node.parentElement;
  const host = element?.closest<HTMLElement>("[data-browser-scope-tab]");
  return !!host && host.dataset.browserScopeTab === tabId && !host.closest("[inert], [hidden]");
}
