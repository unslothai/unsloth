// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ObserverTarget = { id: string; chat: string };
type AgentTarget = { shared?: boolean; tabId: string | null };

export function observedTarget<T extends ObserverTarget>(
  viewerChat: string,
  selection: { chat: string; tabId: string } | null,
  tabs: readonly T[],
  agents: Record<string, AgentTarget>,
): T | undefined {
  if (selection?.chat !== viewerChat) return undefined;
  return tabs.find((tab) => tab.id === selection.tabId && tab.chat !== viewerChat &&
    agents[tab.chat]?.tabId === tab.id);
}

export type ObservationViewport = { scrollX: number; scrollY: number; viewportWidth: number; viewportHeight: number };

export function observationDocument(html: string, nonce: string, viewport: ObservationViewport): string {
  if (!/^[a-f0-9]{32}$/.test(nonce)) throw new Error("Invalid preview nonce");
  const x = Number.isFinite(viewport.scrollX) ? viewport.scrollX : 0;
  const y = Number.isFinite(viewport.scrollY) ? viewport.scrollY : 0;
  const policy = `default-src 'none'; img-src data:; style-src 'unsafe-inline'; font-src data:; script-src 'nonce-${nonce}'; form-action 'none'; base-uri 'none'`;
  const restore = `document.documentElement.inert=true;addEventListener('load',()=>scrollTo({left:${x},top:${y},behavior:'instant'}),{once:true});`;
  return `<!doctype html><meta http-equiv="Content-Security-Policy" content="${policy}"><style>html{scroll-behavior:auto!important}</style><script nonce="${nonce}">${restore}</script>${html.replace(/^\s*<!doctype[^>]*>/i, "")}`;
}
