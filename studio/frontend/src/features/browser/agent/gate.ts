// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the browser agent may do on its own, ask the user about first, or never do. Decided
 *  here because only the panel knows what a ref on the live page is. A page can be written to
 *  talk the model into anything, so the rules hold whatever the model says it wants. */

import type { ElementInfo } from "./page-agent";

/** `prompt` is for the model; `ask`, `host` and `name` let the panel say it in the reader's language. */
export type Question = { ask: "open" | "signIn" | "submit" | "click"; host: string; name: string; detail: string };

export type Gate = { kind: "allow" } | ({ kind: "ask"; prompt: string } & Question) | { kind: "block"; reason: string };

const ALLOW: Gate = { kind: "allow" };

export type SitePolicy = { allowedDomains: string[]; blockedDomains: string[] };

export function hostOfUrl(url: string): string {
  try {
    return new URL(url).hostname.toLowerCase().replace(/^www\./, "");
  } catch {
    return "";
  }
}

const matches = (host: string, domain: string) => {
  const bare = domain.toLowerCase().replace(/^\*?\./, "").replace(/^www\./, "");
  return host === bare || host.endsWith(`.${bare}`);
};

/** The request's website policy, applied to every page the agent opens or submits to. */
export function policyBlocks(url: string, policy: SitePolicy): string | null {
  const host = hostOfUrl(url);
  if (!host) return "that is not a web address";
  if (policy.blockedDomains.some((domain) => matches(host, domain))) return `the website policy blocks ${host}`;
  if (policy.allowedDomains.length && !policy.allowedDomains.some((domain) => matches(host, domain))) {
    return `the website policy only allows ${policy.allowedDomains.join(", ")}`;
  }
  return null;
}

export type NavigationContext = {
  policy: SitePolicy;
  knownHosts: ReadonlySet<string>;
  /** Links on the page the agent last read: following one carries nothing the model made up. */
  pageLinks: ReadonlySet<string>;
  userText: string;
};

function mentioned(host: string, text: string): boolean {
  const lower = text.toLowerCase();
  // example.com in "look at example.com", or the site's name ("search wikipedia").
  if (lower.includes(host)) return true;
  const name = host.split(".").slice(-2, -1)[0];
  return Boolean(name && name.length >= 4 && new RegExp(`\\b${name.replace(/[^a-z0-9-]/g, "")}\\b`).test(lower));
}

/** A new site needs the user's OK unless they named it or the page links to it: a made-up address
 *  is how a hijacked agent would carry what it read to someone else. */
export function gateNavigation(url: string, context: NavigationContext): Gate {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return { kind: "block", reason: "that is not a web address" };
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    return { kind: "block", reason: "only http and https pages can be opened" };
  }
  const blocked = policyBlocks(url, context.policy);
  if (blocked) return { kind: "block", reason: blocked };
  const host = hostOfUrl(url);
  if (context.knownHosts.has(host) || context.pageLinks.has(parsed.href) || mentioned(host, context.userText)) return ALLOW;
  return { kind: "ask", ask: "open", host, name: "", prompt: `Open ${host}?`, detail: parsed.href };
}

// Fields whose contents are the user's to type: the agent never fills them.
const SECRET_AUTOCOMPLETE = /\b(current-password|new-password|one-time-code|cc-[a-z-]+)\b/;
const SECRET_NAME = /\b(password|passcode|card ?number|cvc|cvv|security code|ssn|social security)\b/i;

// What a button does when its words say it spends, sends, publishes, signs in or deletes.
const CONSEQUENTIAL =
  /\b(buy|purchase|order|checkout|check out|pay|payment|donate|subscribe|send|post|publish|tweet|reply|comment|share|delete|remove|erase|transfer|book|reserve|sign ?up|register|log ?in|sign ?in|authori[sz]e|allow|accept|agree|confirm|install|unsubscribe|cancel)\b/i;

const CONSEQUENTIAL_LINK = /\b(buy|purchase|order now|checkout|check out|pay|donate|delete|unsubscribe|subscribe)\b/i;

export type ElementAction = "click" | "type" | "select" | "submit";

/** `submit`: the action will submit the element's form (a submit button, or Enter in a field). */
export function gateElement(action: ElementAction, info: ElementInfo, pageUrl: string, policy: SitePolicy): Gate {
  const host = hostOfUrl(pageUrl) || "this site";
  if (info.role === "file-input" || info.type === "file") {
    return { kind: "block", reason: "uploading files is not something the browser agent can do; ask the user to do it" };
  }
  const secret =
    info.role === "password" ||
    info.type === "password" ||
    SECRET_AUTOCOMPLETE.test(info.autocomplete) ||
    (info.tag !== "a" && info.tag !== "button" && SECRET_NAME.test(info.name));
  if (action === "type" && secret) {
    return {
      kind: "block",
      reason: "that is a password or payment field; ask the user to fill it in themselves in the browser panel",
    };
  }
  const submits = action === "submit" || (action === "click" && info.submits);
  if (submits && info.formAction) {
    const blocked = policyBlocks(info.formAction, policy);
    if (blocked) return { kind: "block", reason: blocked };
  }
  if (submits && info.formHasPassword) {
    return { kind: "ask", ask: "signIn", host, name: "", prompt: `Sign in on ${host}?`, detail: describe(info) };
  }
  if (submits && info.formMethod === "post") {
    return { kind: "ask", ask: "submit", host, name: "", prompt: `Submit this form on ${host}?`, detail: describe(info) };
  }
  // A link mostly just opens a page ("Sign in" goes to the sign-in form), so fewer words count there.
  const consequential = (info.role === "link" ? CONSEQUENTIAL_LINK : CONSEQUENTIAL).test(info.name);
  if ((action === "click" || submits) && consequential) {
    return { kind: "ask", ask: "click", host, name: clip(info.name), prompt: `Click “${clip(info.name)}” on ${host}?`, detail: describe(info) };
  }
  return ALLOW;
}

const clip = (value: string) => (value.length > 60 ? `${value.slice(0, 59)}…` : value);

function describe(info: ElementInfo): string {
  const what = `${info.role}${info.name ? ` “${clip(info.name)}”` : ""}`;
  if (info.formAction && (info.submits || info.formMethod === "post")) {
    return `${what}, sending the form to ${info.formAction}`;
  }
  return info.href ? `${what} → ${info.href}` : what;
}
