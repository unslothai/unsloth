// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The model's hands in a page: reads it as text with refs, and clicks, types and scrolls by ref.
 *  Shipped to the page as source (`pageAgentSource`), so `installPageAgent` must stay
 *  self-contained: no imports and nothing from outside its own body. */

export type AgentCommand =
  | { op: "snapshot"; fullPage?: boolean; ref?: string }
  | { op: "find"; query: string }
  | { op: "describe"; ref: string }
  | { op: "click"; ref: string; directLinks: boolean }
  | { op: "type"; ref: string; text: string; submit: boolean }
  | { op: "select"; ref: string; value: string }
  | { op: "scroll"; direction: "up" | "down"; ref?: string }
  | { op: "press"; key: string }
  | { op: "status" };

/** What a ref is, so the panel can decide whether acting on it needs the user's OK. */
export type ElementInfo = {
  role: string;
  name: string;
  tag: string;
  type: string;
  href: string;
  autocomplete: string;
  formMethod: string;
  formAction: string;
  formHasPassword: boolean;
  submits: boolean;
};

export type AgentAnswer = {
  ok: boolean;
  error?: string;
  text?: string;
  links?: string[];
  note?: string;
  navigate?: string;
  info?: ElementInfo;
  quietMs?: number;
  title?: string;
};

export function installPageAgent(): void {
  const KEY = Symbol.for("unsloth.agent");
  const host = window as unknown as Record<symbol, unknown>;
  if (host[KEY]) return;

  const VIEW_CHARS = 8000;
  const PAGE_CHARS = 24000;
  const MAX_LINKS = 400;
  const SKIP = new Set(["script", "style", "noscript", "template", "head", "meta", "link", "title"]);
  const ROLES = new Set([
    "button",
    "link",
    "checkbox",
    "radio",
    "switch",
    "tab",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "combobox",
    "textbox",
    "searchbox",
    "slider",
    "spinbutton",
    "treeitem",
  ]);
  // Mirrors gate.ts, which can't be imported here (this function ships as source).
  const SECRET = /\b(current-password|new-password|one-time-code|cc-[a-z-]+)\b/i;
  const SECRET_NAME = /\b(password|passcode|card ?number|cvc|cvv|security code|ssn|social security)\b/i;
  const TEXT_INPUTS = new Set(["text", "search", "email", "url", "tel", "number", "password", ""]);
  const BUTTON_INPUTS = new Set(["button", "submit", "reset", "image"]);
  const NOT_TEXT = new Set([...BUTTON_INPUTS, "checkbox", "radio", "range", "file", "color", "hidden"]);

  const refs = new WeakMap<Element, number>();
  const byRef = new Map<number, WeakRef<Element>>();
  let nextRef = 1;
  let lastMutation = performance.now();
  try {
    new MutationObserver(() => {
      lastMutation = performance.now();
    }).observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
  } catch {
    // A page without a document element yet: settling then reads as quiet.
  }

  let lastRef = Number.MAX_SAFE_INTEGER;
  const refOf = (el: Element): string => {
    let id = refs.get(el);
    if (id === undefined) {
      // Past its range a number would belong to the next page's refs.
      if (nextRef > lastRef) throw new Error("this page has too many elements to number; use browser_find.");
      id = nextRef++;
      refs.set(el, id);
      byRef.set(id, new WeakRef(el));
    }
    return `r${id}`;
  };
  const lookup = (ref: string): Element => {
    const el = byRef.get(Number(String(ref).replace(/^r/i, "")))?.deref();
    if (!el || !el.isConnected) throw new Error(`${ref} is no longer on the page; call browser_snapshot for fresh refs.`);
    return el;
  };

  const clean = (value: string | null | undefined, max = 80): string => {
    const text = (value ?? "").replace(/\s+/g, " ").trim();
    return text.length > max ? `${text.slice(0, max - 1)}…` : text;
  };
  const quoted = (value: string) => JSON.stringify(value);

  const hidden = (el: Element, style: CSSStyleDeclaration): boolean =>
    el.getAttribute("aria-hidden") === "true" ||
    (el as HTMLElement).hidden === true ||
    style.display === "none" ||
    style.visibility === "hidden" ||
    style.visibility === "collapse" ||
    style.opacity === "0";
  const sized = (rect: DOMRect) => rect.width > 1 && rect.height > 1;
  const onScreen = (rect: DOMRect) =>
    rect.bottom > 0 && rect.right > 0 && rect.top < window.innerHeight && rect.left < window.innerWidth;

  const roleOf = (el: Element): string | null => {
    const explicit = el.getAttribute("role")?.trim().split(/\s+/)[0];
    if (explicit && ROLES.has(explicit)) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === "a" && el.hasAttribute("href")) return "link";
    if (tag === "button" || tag === "summary") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "input") {
      const type = ((el as HTMLInputElement).type || "text").toLowerCase();
      if (type === "hidden") return null;
      if (type === "button" || type === "submit" || type === "reset" || type === "image") return "button";
      if (type === "checkbox" || type === "radio") return type;
      if (type === "range") return "slider";
      if (type === "search") return "searchbox";
      if (type === "file") return "file-input";
      if (type === "password") return "password";
      return "textbox";
    }
    if ((el as HTMLElement).isContentEditable && el.hasAttribute("contenteditable")) return "textbox";
    return null;
  };

  // A control's words, minus what the page hides inside it: hidden text must not name a button.
  const visibleText = (root: Element): string => {
    const words: string[] = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      let shown = true;
      for (let el = node.parentElement; el && shown; el = el === root ? null : el.parentElement) {
        const style = getComputedStyle(el);
        if (hidden(el, style)) shown = false;
        // display: contents draws no box of its own.
        else if (el !== root && style.display !== "contents" && !sized(el.getBoundingClientRect())) shown = false;
      }
      if (shown) words.push(node.nodeValue ?? "");
    }
    return words.join(" ");
  };

  const nameOf = (el: Element): string => {
    const aria = el.getAttribute("aria-label");
    if (aria?.trim()) return clean(aria);
    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const text = labelledBy
        .split(/\s+/)
        .map((id) => document.getElementById(id)?.textContent ?? "")
        .join(" ");
      if (text.trim()) return clean(text);
    }
    const field = el as HTMLInputElement;
    if (field.labels?.length) {
      const text = Array.from(field.labels)
        .map((label) => visibleText(label))
        .join(" ");
      if (text.trim()) return clean(text);
    }
    const tag = el.tagName.toLowerCase();
    if (tag === "input" && ["button", "submit", "reset"].includes(field.type) && field.value) return clean(field.value);
    if (tag === "input" || tag === "textarea") {
      const hint = el.getAttribute("placeholder") || el.getAttribute("title") || el.getAttribute("name");
      return clean(hint);
    }
    if (tag === "select") return clean(el.getAttribute("title") || el.getAttribute("name"));
    const text = visibleText(el);
    if (text.trim()) return clean(text);
    // An icon link: its picture's alt, or a title on it or inside it (an upvote arrow).
    const img = el.querySelector("img[alt]");
    const titled = el.querySelector("[title], [aria-label]");
    return clean(
      img?.getAttribute("alt") ||
        el.getAttribute("title") ||
        titled?.getAttribute("aria-label") ||
        titled?.getAttribute("title"),
    );
  };

  // The proxied frame's own address is the shell's; its <base> is the page's.
  const pageUrl = (): URL => new URL(document.baseURI || location.href);

  /** Same-site links as a path: most of a page's links are, and the origin is in the header. */
  const shortHref = (href: string): string => {
    try {
      const url = new URL(href, document.baseURI);
      return clean(url.origin === pageUrl().origin ? url.pathname + url.search + url.hash : url.href, 120);
    } catch {
      return clean(href, 120);
    }
  };

  const isSecret = (el: Element, role: string | null): boolean =>
    role === "password" ||
    SECRET.test(el.getAttribute("autocomplete") ?? "") ||
    Boolean((el as HTMLInputElement).value && SECRET_NAME.test(nameOf(el)));

  // Inline styles the mask replaced, put back after the screenshot.
  const masked = new Map<HTMLElement, string | null>();
  /** Hides what the user typed into secret fields while a screenshot is taken. */
  const mask = (on: boolean): Record<string, unknown> => {
    for (const [el, style] of masked) {
      el.style.cssText = style ?? "";
      if (style === null) el.removeAttribute("style");
    }
    masked.clear();
    if (!on) return { ok: true };
    for (const el of Array.from(document.querySelectorAll<HTMLInputElement>("input, textarea"))) {
      if (!el.value || !isSecret(el, roleOf(el))) continue;
      masked.set(el, el.getAttribute("style"));
      el.style.setProperty("color", "transparent", "important");
      el.style.setProperty("-webkit-text-fill-color", "transparent", "important");
      el.style.setProperty("text-shadow", "none", "important");
    }
    return { ok: true };
  };

  const stateOf = (el: Element, role: string): string[] => {
    const parts: string[] = [];
    const tag = el.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea") {
      const field = el as HTMLInputElement;
      if (role === "checkbox" || role === "radio") parts.push(field.checked ? "checked" : "unchecked");
      // The same fields the panel never types into: what the user filled in stays theirs.
      else if (isSecret(el, role)) {
        parts.push(field.value ? "filled" : "empty");
      }
      else if (role !== "button" && role !== "file-input" && field.value) parts.push(`value=${quoted(clean(field.value, 60))}`);
    } else if (tag === "select") {
      const select = el as HTMLSelectElement;
      const chosen = select.selectedOptions[0];
      if (chosen) parts.push(`selected=${quoted(clean(chosen.text, 40))}`);
      const options = Array.from(select.options).map((option) => clean(option.text, 30));
      if (options.length) {
        parts.push(`options: ${options.slice(0, 8).join(" | ")}${options.length > 8 ? ` (+${options.length - 8} more)` : ""}`);
      }
    } else if (role === "textbox" && (el as HTMLElement).isContentEditable) {
      const value = clean((el as HTMLElement).innerText, 60);
      if (value) parts.push(`value=${quoted(value)}`);
    } else if (role === "link") {
      const href = el.getAttribute("href") ?? "";
      if (href && !href.startsWith("javascript:")) parts.push(`-> ${shortHref(href)}`);
    }
    const checked = el.getAttribute("aria-checked");
    if (checked && tag !== "input") parts.push(checked === "true" ? "checked" : "unchecked");
    const expanded = el.getAttribute("aria-expanded");
    if (expanded) parts.push(expanded === "true" ? "expanded" : "collapsed");
    if (el.getAttribute("aria-selected") === "true" || el.getAttribute("aria-current")) parts.push("current");
    if (el.getAttribute("aria-pressed") === "true") parts.push("pressed");
    if ((el as HTMLButtonElement).disabled || el.getAttribute("aria-disabled") === "true") parts.push("disabled");
    if ((el as HTMLInputElement).required) parts.push("required");
    return parts;
  };

  const elementLine = (el: Element, role: string): string => {
    const name = nameOf(el);
    return [`[${refOf(el)}] ${role}`, name ? quoted(name) : "", ...stateOf(el, role)].filter(Boolean).join(" ");
  };

  const BLOCK = /^(block|flex|grid|table|list-item|table-row|table-cell|flow-root)/;

  const read = (root: Element, wholePage: boolean, budget: number): { text: string; links: string[]; cut: boolean } => {
    const lines: string[] = [];
    const links: string[] = [];
    let used = 0;
    let cut = false;
    let buffer = "";
    let bufferShown = false;
    const push = (line: string) => {
      if (cut) return;
      if (used + line.length + 1 > budget) {
        cut = true;
        return;
      }
      lines.push(line);
      used += line.length + 1;
    };
    const flush = () => {
      const text = clean(buffer, 400);
      if (text && bufferShown && text.length > 1) push(text);
      buffer = "";
      bufferShown = false;
    };
    const walk = (el: Element) => {
      if (cut) return;
      const tag = el.tagName.toLowerCase();
      if (SKIP.has(tag)) return;
      const style = getComputedStyle(el);
      if (hidden(el, style)) return;
      const rect = el.getBoundingClientRect();
      const shown = wholePage || onScreen(rect);
      const role = roleOf(el);
      if (role) {
        if (sized(rect) || role === "checkbox" || role === "radio") {
          flush();
          if (shown) {
            push(elementLine(el, role));
            if (role === "link" && links.length < MAX_LINKS) {
              const href = (el as HTMLAnchorElement).href;
              if (href) links.push(href);
            }
          }
        }
        // Its words are its name, already out; a custom combobox wraps the field that matters.
        if (role !== "combobox" || tag === "select") return;
      }
      if (tag === "iframe" || tag === "canvas" || tag === "video" || tag === "img") {
        if (shown && sized(rect)) {
          const label = tag === "img" ? clean(el.getAttribute("alt"), 80) : "";
          if (tag !== "img" || label) {
            flush();
            push(`${tag === "img" ? "image" : tag}${label ? ` ${quoted(label)}` : ""}`);
          }
        }
        return;
      }
      const heading = /^h([1-6])$/.exec(tag);
      if (heading) {
        flush();
        const text = clean((el as HTMLElement).innerText, 200);
        if (shown && text) push(`${"#".repeat(Number(heading[1]))} ${text}`);
        // Its links and buttons still get refs; its words are already out.
        for (const child of Array.from(el.querySelectorAll("a[href], button, input, [role]"))) walk(child);
        return;
      }
      const block = BLOCK.test(style.display);
      if (block) flush();
      // A label's words are already its field's name.
      const labelled = tag === "label" && Boolean((el as HTMLLabelElement).control);
      const children = el.shadowRoot ? [...Array.from(el.shadowRoot.childNodes), ...Array.from(el.childNodes)] : Array.from(el.childNodes);
      for (const node of children) {
        if (cut) return;
        if (node.nodeType === Node.TEXT_NODE) {
          const value = node.nodeValue ?? "";
          if (!labelled) {
            buffer += value;
            if (value.trim() && shown && sized(rect)) bufferShown = true;
          }
        } else if (node.nodeType === Node.ELEMENT_NODE) {
          walk(node as Element);
        }
      }
      if (block) flush();
    };
    walk(root);
    flush();
    return { text: lines.join("\n"), links, cut };
  };

  const scroller = (): Element => document.scrollingElement ?? document.documentElement;

  const position = (): string => {
    const page = scroller();
    const height = Math.max(page.scrollHeight, document.body?.scrollHeight ?? 0);
    if (height <= window.innerHeight + 4) return "";
    const top = Math.round(page.scrollTop || window.scrollY);
    const bottom = Math.min(height, top + window.innerHeight);
    return `Showing ${top}-${bottom}px of ${height}px; scroll for more.`;
  };

  const snapshot = (fullPage: boolean, ref?: string): Record<string, unknown> => {
    const root = ref ? lookup(ref) : (document.body ?? document.documentElement);
    const { text, links, cut } = read(root, fullPage || Boolean(ref), fullPage || ref ? PAGE_CHARS : VIEW_CHARS);
    const header = !ref && !fullPage ? position() : "";
    const tail = cut ? "\n… (more below: use browser_scroll, browser_find, or a snapshot ref)" : "";
    const dialog = document.querySelector("dialog[open], [role=dialog][aria-modal=true], [role=alertdialog]");
    const notes = [header, dialog ? "A dialog is open." : ""].filter(Boolean).join(" ");
    return { ok: true, text: `${notes ? `${notes}\n` : ""}${text || "(nothing readable here)"}${tail}`, links };
  };

  const find = (query: string): Record<string, unknown> => {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    const hits: string[] = [];
    const links: string[] = [];
    // The snapshot's rules too, so hidden text can't reach the model through a search instead.
    const visible = (el: Element) => {
      if (el.closest("[aria-hidden=true], [hidden]") || !sized(el.getBoundingClientRect())) return false;
      const check = (el as Element & { checkVisibility?: (options?: object) => boolean }).checkVisibility;
      if (check) return check.call(el, { checkOpacity: true, checkVisibilityCSS: true });
      for (let node: Element | null = el; node; node = node.parentElement) {
        if (hidden(node, getComputedStyle(node))) return false;
      }
      return true;
    };
    const all = Array.from(document.querySelectorAll("body *")).slice(0, 30000);
    for (const el of all) {
      if (hits.length >= 20) break;
      const tag = el.tagName.toLowerCase();
      if (SKIP.has(tag)) continue;
      const role = roleOf(el);
      let haystack = "";
      if (role) haystack = `${role} ${nameOf(el)} ${stateOf(el, role).join(" ")}`;
      else {
        const own = Array.from(el.childNodes)
          .filter((node) => node.nodeType === Node.TEXT_NODE)
          .map((node) => node.nodeValue ?? "")
          .join(" ");
        if (!own.trim()) continue;
        haystack = own;
      }
      const lower = haystack.toLowerCase();
      if (!terms.every((term) => lower.includes(term))) continue;
      if (!visible(el)) continue;
      const rect = el.getBoundingClientRect();
      const where = rect.bottom < 0 ? " (above)" : rect.top > window.innerHeight ? " (below)" : "";
      if (role) {
        hits.push(`${elementLine(el, role)}${where}`);
        const href = (el as HTMLAnchorElement).href;
        if (role === "link" && href) links.push(href);
      } else {
        hits.push(`[${refOf(el)}] text ${quoted(clean(haystack, 160))}${where}`);
      }
    }
    return { ok: true, text: hits.length ? hits.join("\n") : `Nothing on this page matches ${quoted(query)}.`, links };
  };

  /** No ref: the focused element, which a key press would act on. */
  const describe = (ref: string): Record<string, unknown> => {
    const el = ref ? lookup(ref) : (document.activeElement ?? document.body);
    const field = el as HTMLInputElement;
    const form = (field.form ?? el.closest("form")) as HTMLFormElement | null;
    const anchor = el.closest("a[href]") as HTMLAnchorElement | null;
    const tag = el.tagName.toLowerCase();
    const type = (field.type || "").toLowerCase();
    const isSubmitter = (node: Element) => {
      const kind = (node.getAttribute("type") ?? "").toLowerCase();
      return node.tagName === "BUTTON" ? kind === "" || kind === "submit" : node.tagName === "INPUT" && (kind === "submit" || kind === "image");
    };
    const submits = Boolean(form) && isSubmitter(el);
    // What actually submits: this button, or for Enter in a field the form's default button.
    // Its formaction/formmethod override the form's, as browsers and the proxied frame honor.
    const submitter = submits ? el : Array.from(form?.elements ?? []).find(isSubmitter);
    const method = submitter?.getAttribute("formmethod") || form?.getAttribute("method") || "get";
    const action = submitter?.hasAttribute("formaction")
      ? new URL(submitter.getAttribute("formaction") ?? "", document.baseURI).href
      : form?.action || pageUrl().href;
    const info = {
      role: roleOf(el) ?? (anchor ? "link" : "element"),
      name: nameOf(el) || nameOf(anchor ?? el),
      tag,
      type,
      href: anchor?.href ?? "",
      autocomplete: (el.getAttribute("autocomplete") ?? "").toLowerCase(),
      formMethod: form ? method.toLowerCase() : "get",
      formAction: form ? action : "",
      // form.elements, not descendants: a field joins a form from anywhere with form="id".
      formHasPassword: Array.from(form?.elements ?? []).some((node) => (node as HTMLInputElement).type === "password"),
      submits,
    };
    return { ok: true, info };
  };

  const covering = (el: Element): Element | null => {
    const rect = el.getBoundingClientRect();
    const top = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    if (!top || top === el || el.contains(top) || top.contains(el)) return null;
    if ((top as HTMLLabelElement).control === el) return null;
    // Only an overlay counts: a styled box over a real checkbox is how many sites draw them.
    for (let node: Element | null = top; node && node !== document.body; node = node.parentElement) {
      if (node.contains(el)) return null;
      const position = getComputedStyle(node).position;
      if (position === "fixed" || position === "sticky") return node;
    }
    return null;
  };

  const mouse = (el: Element, type: string) => {
    const rect = el.getBoundingClientRect();
    const init = {
      bubbles: true,
      cancelable: true,
      composed: true,
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
      button: 0,
      view: window,
    };
    const event = type.startsWith("pointer")
      ? new PointerEvent(type, { ...init, pointerId: 1, pointerType: "mouse", isPrimary: true })
      : new MouseEvent(type, init);
    el.dispatchEvent(event);
  };

  const click = (ref: string, directLinks: boolean): Record<string, unknown> => {
    const el = lookup(ref);
    el.scrollIntoView({ block: "center", inline: "center" });
    const anchor = el.closest("a[href]") as HTMLAnchorElement | null;
    const href = anchor?.href ?? "";
    if (anchor && /^https?:/i.test(href)) {
      const target = new URL(href);
      const hashOnly = target.hash && target.href.split("#")[0] === pageUrl().href.split("#")[0];
      if (!hashOnly && (directLinks || anchor.target === "_blank")) return { ok: true, navigate: href };
    }
    const over = covering(el);
    if (over) {
      // No page words here: errors reach the model outside the untrusted-content fence.
      return { ok: false, error: `${ref} is covered by an overlay; snapshot to find its close button, or scroll first.` };
    }
    mouse(el, "pointerover");
    mouse(el, "pointerdown");
    mouse(el, "mousedown");
    (el as HTMLElement).focus?.({ preventScroll: true });
    mouse(el, "pointerup");
    mouse(el, "mouseup");
    if (el instanceof HTMLElement) el.click();
    else mouse(el, "click");
    return { ok: true };
  };

  const setValue = (el: HTMLInputElement | HTMLTextAreaElement, value: string) => {
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    // The prototype's setter, so frameworks that wrap the element's own setter see the change.
    Object.getOwnPropertyDescriptor(proto, "value")?.set?.call(el, value);
    el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, inputType: "insertText", data: value }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const key = (target: Element, name: string): boolean => {
    const codes: Record<string, string> = { " ": "Space", Space: "Space" };
    const keyName = name === "Space" ? " " : name;
    const init = { key: keyName, code: codes[name] ?? name, bubbles: true, cancelable: true, composed: true };
    const down = target.dispatchEvent(new KeyboardEvent("keydown", init));
    if (down && keyName === "Enter") target.dispatchEvent(new KeyboardEvent("keypress", { ...init, charCode: 13, keyCode: 13 }));
    target.dispatchEvent(new KeyboardEvent("keyup", init));
    return down;
  };

  const press = (name: string, on?: Element): Record<string, unknown> => {
    const active = on ?? (document.activeElement && document.activeElement !== document.body ? document.activeElement : null);
    const target = active ?? document.body ?? document.documentElement;
    const proceed = key(target, name);
    if (!proceed) return { ok: true };
    const tag = target.tagName.toLowerCase();
    if (name === "Enter") {
      const field = target as HTMLInputElement;
      const inputType = tag === "input" ? (field.type || "text").toLowerCase() : "";
      // Only a text field sends its form on Enter, the case the panel gates as a submit.
      if (tag === "input" && field.form && !NOT_TEXT.has(inputType)) {
        field.form.requestSubmit();
        return { ok: true, note: "Submitted the form." };
      }
      if (tag === "button" || tag === "a" || BUTTON_INPUTS.has(inputType)) (target as HTMLElement).click();
    } else if (name === "Space" && (tag === "button" || (tag === "input" && ["checkbox", "radio"].includes((target as HTMLInputElement).type)))) {
      (target as HTMLElement).click();
    } else if (name === "Backspace" && (tag === "input" || tag === "textarea")) {
      const field = target as HTMLInputElement;
      setValue(field, field.value.slice(0, -1));
    } else if (name === "Tab") {
      const focusable = Array.from(
        document.querySelectorAll<HTMLElement>("a[href], button, input, select, textarea, [tabindex]:not([tabindex='-1'])"),
      ).filter((el) => !(el as HTMLButtonElement).disabled && el.getClientRects().length > 0);
      const next = focusable[(focusable.indexOf(target as HTMLElement) + 1) % Math.max(1, focusable.length)];
      next?.focus();
    } else if (!active) {
      const page = scroller();
      const step = window.innerHeight * 0.85;
      if (name === "PageDown" || name === "Space") page.scrollBy(0, step);
      else if (name === "PageUp") page.scrollBy(0, -step);
      else if (name === "Home") page.scrollTo(0, 0);
      else if (name === "End") page.scrollTo(0, page.scrollHeight);
      else if (name === "ArrowDown") page.scrollBy(0, 80);
      else if (name === "ArrowUp") page.scrollBy(0, -80);
    }
    return { ok: true };
  };

  const type = (ref: string, text: string, submit: boolean): Record<string, unknown> => {
    const el = lookup(ref);
    el.scrollIntoView({ block: "center" });
    (el as HTMLElement).focus?.({ preventScroll: true });
    const tag = el.tagName.toLowerCase();
    if (tag === "textarea" || (tag === "input" && TEXT_INPUTS.has((el as HTMLInputElement).type.toLowerCase()))) {
      setValue(el as HTMLInputElement, text);
    } else if ((el as HTMLElement).isContentEditable) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(el);
      selection?.removeAllRanges();
      selection?.addRange(range);
      if (!document.execCommand("insertText", false, text)) el.textContent = text;
      el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
    } else {
      return { ok: false, error: `${ref} is not a text field.` };
    }
    return submit ? press("Enter", el) : { ok: true };
  };

  const choose = (ref: string, value: string): Record<string, unknown> => {
    const el = lookup(ref);
    const wanted = value.trim().toLowerCase();
    if (el instanceof HTMLSelectElement) {
      const option = Array.from(el.options).find(
        (candidate) => candidate.text.trim().toLowerCase() === wanted || candidate.value.toLowerCase() === wanted,
      ) ?? Array.from(el.options).find((candidate) => candidate.text.toLowerCase().includes(wanted));
      if (!option) return { ok: false, error: `${ref} has no option ${quoted(value)}.` };
      el.value = option.value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return { ok: true, note: `Selected ${quoted(clean(option.text, 60))}.` };
    }
    const role = roleOf(el);
    if (role === "checkbox" || role === "radio" || role === "switch") {
      const on = ["true", "on", "yes", "checked", "1", "check"].includes(wanted);
      const current = el instanceof HTMLInputElement ? el.checked : el.getAttribute("aria-checked") === "true";
      if (current !== on) return click(ref, false);
      return { ok: true, note: `Already ${on ? "checked" : "unchecked"}.` };
    }
    return { ok: false, error: `${ref} is not a dropdown or checkbox; click it, then click the option.` };
  };

  const scrollPage = (direction: string, ref?: string): Record<string, unknown> => {
    if (ref) {
      lookup(ref).scrollIntoView({ block: "center" });
      return { ok: true };
    }
    const sign = direction === "up" ? -1 : 1;
    let target: Element = scroller();
    if (target.scrollHeight <= target.clientHeight + 1) {
      // The page itself doesn't scroll; an app usually scrolls a panel under the middle of the screen.
      for (let node = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2); node; node = node.parentElement) {
        const overflow = getComputedStyle(node).overflowY;
        if ((overflow === "auto" || overflow === "scroll") && node.scrollHeight > node.clientHeight + 1) {
          target = node;
          break;
        }
      }
    }
    const before = target.scrollTop;
    target.scrollBy(0, sign * Math.max(200, target.clientHeight * 0.85));
    if (target.scrollTop === before) return { ok: true, note: `Already at the ${direction === "up" ? "top" : "bottom"}.` };
    return { ok: true };
  };

  const run = (command: Record<string, unknown>): Record<string, unknown> => {
    // Each page numbers its refs from where the panel says, so a ref kept from the last page
    // misses rather than landing on whatever has that number here.
    if (nextRef === 1 && typeof command.base === "number" && Number.isSafeInteger(command.base) && command.base > 0) {
      nextRef = command.base;
      if (typeof command.span === "number" && Number.isSafeInteger(command.span) && command.span > 0) {
        lastRef = command.base + command.span - 1;
      }
    }
    try {
      const ref = typeof command.ref === "string" ? command.ref : "";
      switch (command.op) {
        case "status":
          return { ok: true, quietMs: Math.round(performance.now() - lastMutation), title: document.title };
        case "snapshot":
          return snapshot(command.fullPage === true, ref || undefined);
        case "find":
          return find(String(command.query ?? ""));
        case "describe":
          return describe(ref);
        case "click":
          return click(ref, command.directLinks === true);
        case "type":
          return type(ref, String(command.text ?? ""), command.submit === true);
        case "select":
          return choose(ref, String(command.value ?? ""));
        case "scroll":
          return scrollPage(String(command.direction ?? "down"), ref || undefined);
        case "press":
          return press(String(command.key ?? ""));
        case "mask":
          return mask(command.on === true);
        default:
          return { ok: false, error: "unknown command" };
      }
    } catch (error) {
      return { ok: false, error: clean(error instanceof Error ? error.message : String(error), 300) };
    }
  };

  // Not writable or configurable: once installed, a page script can't swap it out.
  Object.defineProperty(host, KEY, { value: { run }, enumerable: false });
}

export function pageAgentSource(): string {
  return `(${installPageAgent.toString()})();`;
}
