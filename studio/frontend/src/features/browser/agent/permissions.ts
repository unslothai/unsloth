// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Settings > Browser > Agent permissions: what the browser agent may do, by default and per site. */

/** `ask` by default means a site the user didn't name and no page linked to; on a site, every visit from elsewhere. */
export type BrowsingRule = "allow" | "ask" | "block";
/** Sign-ins, form posts and buttons that buy, send or delete. Password and payment fields stay off-limits either way. */
export type ActionsRule = "ask" | "allow";
export type DownloadsRule = "ask" | "allow" | "block";
export type SiteRule = { browsing: BrowsingRule; actions: ActionsRule; downloads: DownloadsRule };
export type AgentPermissions = { defaults: SiteRule; sites: Record<string, SiteRule> };
export type HistoryAccess = "ask" | "allow" | "never";

export const DEFAULT_RULE: SiteRule = { browsing: "ask", actions: "ask", downloads: "ask" };
export const DEFAULT_PERMISSIONS: AgentPermissions = { defaults: DEFAULT_RULE, sites: {} };
export const MAX_SITE_RULES = 200;

const HOST_RE = /^(\*\.)?([a-z0-9-]+\.)*[a-z0-9-]+$/;

/** "https://www.Example.com/a" -> "example.com"; "*.example.com" stays a subdomain pattern. Null if it isn't a host. */
export function normalizeSitePattern(input: string): string | null {
  let value = input.trim().toLowerCase();
  if (!value) return null;
  const wildcard = value.startsWith("*.");
  if (wildcard) value = value.slice(2);
  try {
    value = new URL(/^[a-z][a-z0-9+.-]*:\/\//.test(value) ? value : `https://${value}`).hostname;
  } catch {
    return null;
  }
  value = value.replace(/^www\./, "").replace(/\.$/, "");
  const pattern = wildcard ? `*.${value}` : value;
  return value.includes(".") && value.length <= 253 && HOST_RE.test(pattern) ? pattern : null;
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.toLowerCase().replace(/^www\./, "");
  } catch {
    return "";
  }
}

/** "example.com" covers the site and its subdomains, "*.example.com" only the subdomains. */
export function patternMatches(pattern: string, host: string): boolean {
  if (pattern.startsWith("*.")) return host.endsWith(pattern.slice(1));
  return host === pattern || host.endsWith(`.${pattern}`);
}

/** The most specific site rule for `url`, else the default; `site` is the pattern that applied. */
export function ruleFor(url: string, permissions: AgentPermissions): { rule: SiteRule; site: string | null } {
  const host = hostOf(url);
  let best: string | null = null;
  for (const pattern of Object.keys(permissions.sites)) {
    if (host && patternMatches(pattern, host) && (!best || pattern.replace("*.", "").length > best.replace("*.", "").length)) {
      best = pattern;
    }
  }
  const site = best ? permissions.sites[best] : undefined;
  return site ? { rule: site, site: best } : { rule: permissions.defaults, site: null };
}

const BROWSING = new Set(["allow", "ask", "block"]);
const ACTIONS = new Set(["ask", "allow"]);
const DOWNLOADS = new Set(["ask", "allow", "block"]);

/** Persisted values, checked: an unknown value falls back to the default rather than to allow. */
export function checkedRule(value: unknown, fallback: SiteRule = DEFAULT_RULE): SiteRule {
  const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  return {
    browsing: BROWSING.has(raw.browsing as string) ? (raw.browsing as BrowsingRule) : fallback.browsing,
    actions: ACTIONS.has(raw.actions as string) ? (raw.actions as ActionsRule) : fallback.actions,
    downloads: DOWNLOADS.has(raw.downloads as string) ? (raw.downloads as DownloadsRule) : fallback.downloads,
  };
}

export function checkedPermissions(value: unknown): AgentPermissions {
  const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const sites: Record<string, SiteRule> = {};
  const listed = raw.sites && typeof raw.sites === "object" ? Object.entries(raw.sites as Record<string, unknown>) : [];
  for (const [pattern, rule] of listed.slice(0, MAX_SITE_RULES)) {
    const normalized = normalizeSitePattern(pattern);
    if (normalized) sites[normalized] = checkedRule(rule);
  }
  return { defaults: checkedRule(raw.defaults), sites };
}
