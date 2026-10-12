// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the browser agent may do on its own, ask the user about first, or never do. Decided
 *  here because only the panel knows what a ref on the live page is. A page can be written to
 *  talk the model into anything, so the rules hold whatever the model says it wants. */

import type { ElementInfo } from "./page-agent";
import { type AgentPermissions, patternMatches, ruleFor } from "./permissions";

/** `prompt` is for the model; `ask`, `host` and `name` let the panel say it in the reader's language. */
export type Question = {
  ask: "open" | "signIn" | "submit" | "click" | "history";
  host: string;
  name: string;
  detail: string;
};

/** `linked`: allowed only because the page links there, so its site isn't approved for addresses the model makes up. */
export type Gate =
  | { kind: "allow"; linked?: true }
  | ({ kind: "ask"; prompt: string } & Question)
  | { kind: "block"; reason: string };

const ALLOW: Gate = { kind: "allow" };
const LINKED: Gate = { kind: "allow", linked: true };

export type SitePolicy = { allowedDomains: string[]; blockedDomains: string[] };

export function hostOfUrl(url: string): string {
  try {
    // "evil.com." is evil.com to DNS, so it meets evil.com's rules.
    return new URL(url).hostname.toLowerCase().replace(/\.+$/, "").replace(/^www\./, "");
  } catch {
    return "";
  }
}

// As web_access_policy._matches_domain: whole hostnames, so www.example.com covers only itself.
const matches = (host: string, domain: string) => {
  const bare = domain.toLowerCase().replace(/^\*?\./, "").replace(/\.+$/, "");
  return host === bare || host.endsWith(`.${bare}`);
};

function policyHost(url: string): string {
  try {
    return new URL(url).hostname.toLowerCase().replace(/\.+$/, "");
  } catch {
    return "";
  }
}

/** The request's website policy, applied to every page the agent opens or submits to. */
export function policyBlocks(url: string, policy: SitePolicy): string | null {
  const host = policyHost(url);
  if (!host) return "that is not a web address";
  if (policy.blockedDomains.some((domain) => matches(host, domain))) return `the website policy blocks ${host}`;
  if (policy.allowedDomains.length && !policy.allowedDomains.some((domain) => matches(host, domain))) {
    return `the website policy only allows ${policy.allowedDomains.join(", ")}`;
  }
  return null;
}

/** The website policy, then the user's Agent permissions: why the agent may not be on `url`, or null. */
export function navigationBlocks(url: string, policy: SitePolicy, permissions: AgentPermissions): string | null {
  const blocked = policyBlocks(url, policy);
  if (blocked) return blocked;
  const { rule, site } = ruleFor(url, permissions);
  if (rule.browsing !== "block") return null;
  return site
    ? `the user blocked ${site} in the agent's site permissions`
    : "the user only lets the agent open the sites they allowed in Settings";
}

export type NavigationContext = {
  policy: SitePolicy;
  permissions: AgentPermissions;
  /** The page the agent is on: moving within a site needs no new OK. */
  currentUrl: string;
  knownHosts: ReadonlySet<string>;
  /** Links on the page the agent last read: following one carries nothing the model made up. */
  pageLinks: ReadonlySet<string>;
  userText: string;
};

// Hosts that give anyone a subdomain (from the public suffix list's private section).
const SHARED_HOSTS = [
  "github.io", "gitlab.io", "pages.dev", "workers.dev", "vercel.app", "netlify.app", "herokuapp.com",
  "web.app", "firebaseapp.com", "appspot.com", "azurewebsites.net", "cloudfront.net", "amazonaws.com",
  "blogspot.com", "wordpress.com", "tumblr.com", "substack.com", "notion.site", "glitch.me", "repl.co",
  "onrender.com", "fly.dev", "ngrok.io", "ngrok-free.app", "trycloudflare.com", "surge.sh", "neocities.org",
  "hf.space", "streamlit.app", "deno.dev", "googleusercontent.com", "sharepoint.com", "myshopify.com",
];

const NAMED_TLDS = new Set(["com", "org", "net", "edu", "gov"]);

function mentioned(host: string, text: string): boolean {
  const lower = text.toLowerCase();
  // example.com in "look at example.com", as a whole name (not e.com inside it).
  const escaped = host.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // A dot after it only as the sentence's end: example.com.evil.test names another host.
  if (new RegExp(`(^|[^a-z0-9.-])(www\\.)?${escaped}($|[^a-z0-9.-]|\\.(?![a-z0-9-]))`).test(lower)) return true;
  // The site's name ("search wikipedia" for wikipedia.org), but not on a shared host, where each
  // subdomain is someone else's: "github" must not approve collector.github.io.
  // Only on the big generic TLDs: "wikipedia" must not approve a made-up wikipedia.xyz.
  if (SHARED_HOSTS.some((suffix) => host.endsWith(`.${suffix}`))) return false;
  const labels = host.split(".");
  // Only the site itself: a brand's subdomains can carry anyone's data (docs.google.com/forms,
  // script.google.com), so "google" approves google.com and nothing under it.
  if (labels.length !== 2) return false;
  const [name, tld] = labels;
  if (!NAMED_TLDS.has(tld ?? "")) return false;
  // A word of its own, not a label inside some other hostname.
  const word = name?.replace(/[^a-z0-9-]/g, "");
  return Boolean(word && word.length >= 4 && new RegExp(`(^|[^a-z0-9.-])${word}($|[^a-z0-9.-]|\\.(?![a-z0-9-]))`).test(lower));
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
  const blocked = navigationBlocks(url, context.policy, context.permissions);
  if (blocked) return { kind: "block", reason: blocked };
  const host = hostOfUrl(url);
  const ask: Gate = { kind: "ask", ask: "open", host, name: "", prompt: `Open ${host}?`, detail: parsed.href };
  const { rule, site } = ruleFor(url, context.permissions);
  if (rule.browsing === "allow") return ALLOW;
  if (site) {
    const here = hostOfUrl(context.currentUrl);
    return here && patternMatches(site, here) ? ALLOW : ask;
  }
  if (context.knownHosts.has(host) || mentioned(host, context.userText)) return ALLOW;
  if (context.pageLinks.has(parsed.href)) return LINKED;
  return ask;
}

// Fields whose contents are the user's to type: the agent never fills them.
const SECRET_AUTOCOMPLETE = /\b(current-password|new-password|one-time-code|cc-[a-z-]+)\b/;
const SECRET_NAME = /\b(password|passcode|card ?number|cvc|cvv|security code|ssn|social security)\b/i;

// What a button does when its words say it spends, sends, publishes, signs in or deletes.
const CONSEQUENTIAL =
  /\b(buy|purchase|order|checkout|check out|pay|payment|donate|subscribe|send|post|publish|tweet|reply|comment|share|delete|remove|erase|transfer|book|reserve|sign ?up|register|log ?in|sign ?in|authori[sz]e|allow|accept|agree|confirm|install|unsubscribe|cancel)\b/i;

const CONSEQUENTIAL_LINK = /\b(buy|purchase|order now|checkout|check out|pay|donate|delete|unsubscribe|subscribe)\b/i;

// The same verbs in the languages Studio ships in (src/i18n/locales). Whole words where words are
// spaced; anywhere in Chinese, Japanese and Korean, where a verb sits inside a longer run.
const SPACED_WORDS = [
  // es, pt-br
  "comprar", "compra", "pagar", "pague", "eliminar", "borrar", "excluir", "apagar", "enviar", "transferir", "suscribirse", "assinar", "confirmar", "publicar", "finalizar compra",
  // fr
  "acheter", "payer", "commander", "supprimer", "envoyer", "virement", "transférer", "s'abonner", "confirmer", "publier", "se connecter", "s'inscrire",
  // de
  "kaufen", "bezahlen", "bestellen", "löschen", "entfernen", "senden", "überweisen", "abonnieren", "bestätigen", "veröffentlichen", "anmelden", "registrieren",
  // it
  "acquista", "paga", "ordina", "elimina", "invia", "trasferisci", "iscriviti", "abbonati", "conferma", "pubblica", "accedi",
  // sv
  "köp", "betala", "beställ", "radera", "ta bort", "skicka", "överför", "prenumerera", "bekräfta", "publicera", "logga in",
  // ru
  "купить", "оплатить", "заказать", "удалить", "отправить", "перевести", "подписаться", "подтвердить", "опубликовать", "войти",
  // ar, he, hi
  "شراء", "ادفع", "دفع", "حذف", "إرسال", "تحويل", "اشتراك", "تأكيد", "نشر",
  "קנה", "רכישה", "תשלום", "מחק", "שלח", "העבר", "הירשם", "אשר", "פרסם", "התחבר",
  "खरीदें", "भुगतान", "हटाएं", "भेजें", "सदस्यता", "पुष्टि", "प्रकाशित",
];
const UNSPACED_WORDS = [
  "购买", "支付", "付款", "下单", "删除", "发送", "转账", "订阅", "确认", "发布", "登录", "注册",
  "購入", "支払", "注文", "削除", "送信", "送金", "振込", "購読", "投稿", "ログイン",
  "구매", "결제", "주문", "삭제", "전송", "송금", "이체", "구독", "게시", "로그인",
];
const LOCALIZED = new RegExp(
  `(?<![\\p{L}\\p{M}])(?:${SPACED_WORDS.join("|")})(?![\\p{L}\\p{M}])|${UNSPACED_WORDS.join("|")}`,
  "iu",
);

export type ElementAction = "click" | "type" | "select" | "submit";

/** `submit`: the action will submit the element's form (a submit button, or Enter in a field). */
export function gateElement(
  action: ElementAction,
  info: ElementInfo,
  pageUrl: string,
  policy: SitePolicy,
  permissions: AgentPermissions,
): Gate {
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
    const blocked = navigationBlocks(info.formAction, policy, permissions);
    if (blocked) return { kind: "block", reason: blocked };
  }
  if (ruleFor(pageUrl, permissions).rule.actions === "allow") return ALLOW;
  if (submits && info.formHasPassword) {
    return { kind: "ask", ask: "signIn", host, name: "", prompt: `Sign in on ${host}?`, detail: describe(info) };
  }
  if (submits && info.formMethod === "post") {
    return { kind: "ask", ask: "submit", host, name: "", prompt: `Submit this form on ${host}?`, detail: describe(info) };
  }
  // A link mostly just opens a page ("Sign in" goes to the sign-in form), so fewer words count there.
  const consequential = (info.role === "link" ? CONSEQUENTIAL_LINK : CONSEQUENTIAL).test(info.name) || LOCALIZED.test(info.name);
  // Selecting a checkbox, radio or switch is a click on it.
  // A select is a click: on a checkbox, its name; on a dropdown, the option chosen (the executor
  // passes it as the name), since a page can act on the change event.
  const clicks = action === "click" || submits || action === "select";
  if (clicks && consequential) {
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
