// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The model's hands in a page: reads it as text with refs, and clicks, types and scrolls by ref.
 *  Shipped to the page as source (`pageAgentSource`), so `installPageAgent` must stay
 *  self-contained: no imports and nothing from outside its own body. */

/** `x` and `y` are CSS pixels of the visible page, as a pointer event's clientX and clientY. */
export type AgentCommand =
  | { op: "snapshot"; fullPage?: boolean; ref?: string }
  | { op: "find"; query: string }
  | { op: "describe"; ref: string; x?: number; y?: number }
  | { op: "click"; ref: string; directLinks: boolean; x?: number; y?: number; expect?: string }
  | { op: "locate"; ref: string; scroll: boolean }
  | { op: "pointer"; phase: "down" | "move" | "up" | "cancel"; x: number; y: number; expect?: string }
  | { op: "type"; ref: string; text: string; submit: boolean }
  | { op: "select"; ref: string; value: string }
  | { op: "scroll"; direction: "up" | "down"; ref?: string }
  | { op: "press"; key: string }
  | { op: "evaluate"; id: string; doc: string; script: string }
  | { op: "evaluated"; id: string }
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
  /** Whether one of those password fields holds a value (the user's to type); never the value. */
  formPasswordFilled?: boolean;
  /** For a dropdown described with a value: the label of the option that value selects. */
  option?: string;
  /** For Enter in a field: the label of the button that will submit its form. */
  submitter?: string;
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
  /** Requests the page started recently and is still waiting on. */
  requests?: number;
  title?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  scrollX?: number;
  scrollY?: number;
  scrolls?: number;
  pending?: boolean;
  hit?: string;
  sig?: string;
  doc?: string;
};

/** `evaluate`: native views splice the script in past the CSP as `run`; the proxied frame compiles it. */
export type PageAgent = {
  run: (command: Record<string, unknown>) => Record<string, unknown>;
  evaluate: (id: string, doc: unknown, run: () => unknown) => Record<string, unknown>;
  owns: (doc: unknown) => boolean;
};

/** Returns the agent it built, so the proxied frame keeps its own reference rather than trusting
 *  KEY, which a page script can define before the agent arrives. */
export function installPageAgent(): PageAgent {
  const KEY = Symbol.for("unsloth.agent");
  const host = window as unknown as Record<symbol, unknown>;

  const VIEW_CHARS = 8000;
  const PAGE_CHARS = 24000;
  const MAX_LINKS = 400;
  const SKIP = new Set(["script", "style", "noscript", "template", "head", "meta", "link", "title", "unsloth-agent-cursor"]);
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
  // Date and time fields are textboxes to the model too; they take their ISO-like value as typed.
  const TEXT_INPUTS = new Set([
    "text", "search", "email", "url", "tel", "number", "password", "",
    "date", "time", "datetime-local", "month", "week",
  ]);
  const DATE_FORMATS: Record<string, string> = {
    date: "2026-03-14", time: "09:30", "datetime-local": "2026-03-14T09:30", month: "2026-03", week: "2026-W11",
  };
  const BUTTON_INPUTS = new Set(["button", "submit", "reset", "image"]);
  const NOT_TEXT = new Set([...BUTTON_INPUTS, "checkbox", "radio", "range", "file", "color", "hidden"]);
  const EDITABLE_TAGS = new Set(["input", "textarea"]);
  const CARET_KEYS = new Set(["ArrowLeft", "ArrowRight", "Home", "End"]);

  class AgentError extends Error {}
  const refs = new WeakMap<Element, number>();
  const byRef = new Map<number, WeakRef<Element>>();
  let nextRef = 1;
  let lastMutation = performance.now();
  let scrolls = 0;
  addEventListener("scroll", () => scrolls++, { capture: true, passive: true });
  try {
    new MutationObserver((records) => {
      // The agent's cursor going in or out (or into a dialog) is not the page changing.
      const own = (node: Node) => node.nodeName === "UNSLOTH-AGENT-CURSOR";
      if (records.every((r) => r.type === "childList" && [...r.addedNodes].every(own) && [...r.removedNodes].every(own))) return;
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
      if (nextRef > lastRef) throw new AgentError("this page has too many elements to number; use browser_find.");
      id = nextRef++;
      refs.set(el, id);
      byRef.set(id, new WeakRef(el));
    }
    return `r${id}`;
  };
  const lookup = (ref: string): Element => {
    const el = byRef.get(Number(String(ref).replace(/^r/i, "")))?.deref();
    if (!el || !el.isConnected) throw new AgentError(`${ref} is no longer on the page; call browser_snapshot for fresh refs.`);
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
      // Text under 2px is for the model, not the user, as in the snapshot.
      let shown = !node.parentElement || Number.parseFloat(getComputedStyle(node.parentElement).fontSize) >= 2;
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
      // Read as the user sees it: a hidden or 1px element must not name the control.
      const text = labelledBy
        .split(/\s+/)
        .map((id) => {
          const label = document.getElementById(id);
          if (!label) return "";
          if (getComputedStyle(label).display !== "contents" && !sized(label.getBoundingClientRect())) return "";
          return visibleText(label);
        })
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
    if (tag === "input" && field.type === "image" && el.getAttribute("alt")) return clean(el.getAttribute("alt"));
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

  // The input type, not the role: role="textbox" on a password field must not show its value.
  const isSecret = (el: Element, role: string | null): boolean =>
    role === "password" ||
    (el.tagName.toLowerCase() === "input" && (el as HTMLInputElement).type.toLowerCase() === "password") ||
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
    cursor.mask(on);
    if (!on) return { ok: true };
    const fields: HTMLInputElement[] = [];
    const visit = (root: Document | ShadowRoot) => {
      fields.push(...Array.from(root.querySelectorAll<HTMLInputElement>("input, textarea, [contenteditable]")));
      for (const host of Array.from(root.querySelectorAll("*"))) if (host.shadowRoot) visit(host.shadowRoot);
    };
    visit(document);
    // A frame's fields (card and code widgets often live in one) can't be read from here: hide it.
    for (const frame of Array.from(document.querySelectorAll<HTMLElement>("iframe, frame"))) {
      masked.set(frame, frame.getAttribute("style"));
      frame.style.setProperty("visibility", "hidden", "important");
    }
    for (const el of fields) {
      const secret = el.isContentEditable
        ? Boolean(el.innerText.trim()) && (SECRET.test(el.getAttribute("autocomplete") ?? "") || SECRET_NAME.test(nameOf(el)))
        : Boolean(el.value) && isSecret(el, roleOf(el));
      if (!secret) continue;
      masked.set(el, el.getAttribute("style"));
      el.style.setProperty("color", "transparent", "important");
      el.style.setProperty("-webkit-text-fill-color", "transparent", "important");
      el.style.setProperty("text-shadow", "none", "important");
    }
    return { ok: true, ...view() };
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
      const secret = SECRET.test(el.getAttribute("autocomplete") ?? "") || SECRET_NAME.test(nameOf(el));
      if (secret) parts.push(value ? "filled" : "empty");
      else if (value) parts.push(`value=${quoted(value)}`);
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
        const text = clean(visibleText(el), 200);
        if (shown && text) push(`${"#".repeat(Number(heading[1]))} ${text}`);
        // Its links and buttons still get refs; its words are already out.
        for (const child of Array.from(el.querySelectorAll("a[href], button, input, [role]"))) walk(child);
        return;
      }
      const block = BLOCK.test(style.display);
      if (block) flush();
      // A label's words are already its field's name.
      const labelled = tag === "label" && Boolean((el as HTMLLabelElement).control);
      // Text too small to read is there for the model, not the user. Only this element's own text:
      // a font-size:0 parent whose children reset it is a common inline-block layout trick.
      const unreadable = Number.parseFloat(style.fontSize) < 2;
      const children = el.shadowRoot ? [...Array.from(el.shadowRoot.childNodes), ...Array.from(el.childNodes)] : Array.from(el.childNodes);
      for (const node of children) {
        if (cut) return;
        if (node.nodeType === Node.TEXT_NODE) {
          const value = node.nodeValue ?? "";
          if (unreadable) buffer += " ";
          else if (!labelled) {
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
        if (!own.trim() || Number.parseFloat(getComputedStyle(el).fontSize) < 2) continue;
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

  // The normalized type: a button with an invalid type attribute is a submit button.
  const isSubmitter = (node: Element) =>
    node instanceof HTMLButtonElement
      ? node.type === "submit"
      : node instanceof HTMLInputElement && (node.type === "submit" || node.type === "image");
  /** The button Enter in a field submits with, as a browser picks it: the form's first submit button
   *  in tree order (form.elements leaves out <input type=image>). */
  const defaultButton = (form: HTMLFormElement | null) =>
    form
      ? (Array.from(document.querySelectorAll("button, input")).find(
          (node) => (node as HTMLInputElement).form === form && isSubmitter(node),
        ) as HTMLButtonElement | HTMLInputElement | undefined)
      : undefined;

  type Point = { x: number; y: number };
  const pointIn = (command: Record<string, unknown>): Point | null =>
    typeof command.x === "number" && typeof command.y === "number" && Number.isFinite(command.x) && Number.isFinite(command.y)
      ? { x: command.x, y: command.y }
      : null;

  // The agent's pointer (Settings > Browser). Everything it moves lives in a closed shadow root, which
  // no MutationObserver on the document sees, so the page and the settle wait only see it inserted
  // once; it moves by transform (compositor only) and pointer-events: none keeps it out of hit tests.
  // CSSOM and createElementNS only: a page's CSP or Trusted Types can block <style> and innerHTML.
  const cursor = (() => {
    const SVG = "http://www.w3.org/2000/svg";
    const IDLE_MS = 4000;
    let on = false;
    // Reduce motion: the cursor jumps and fades instead of gliding, and presses get no ring.
    let still = false;
    // Studio's accent as "r,g,b"; only digits parsed from rgb() ever reach a style.
    let rgb = "59,130,246";
    let parts: { host: HTMLElement; mover: HTMLElement; arrow: SVGSVGElement; ring: HTMLElement; rgb: string } | null = null;
    let shown = false;
    let masking = false;
    let idle = 0;
    // Only the host needs !important (page CSS can reach it); inside the closed root it would also
    // outrank the ring's animation.
    const styled = <E extends Element>(el: E, css: Record<string, string>, important = false): E => {
      for (const [name, value] of Object.entries(css)) {
        (el as unknown as HTMLElement).style.setProperty(name, value, important ? "important" : "");
      }
      return el;
    };
    // The glow is on the arrow alone: under a shared filter the ring's pulse would re-run it every frame.
    const tint = () => {
      if (!parts || parts.rgb === rgb) return;
      parts.rgb = rgb;
      parts.arrow.style.filter = `drop-shadow(0 0 3px rgba(${rgb},.95)) drop-shadow(0 0 7px rgba(${rgb},.55))`;
      parts.ring.style.borderColor = `rgba(${rgb},.9)`;
    };
    const build = () => {
      if (parts?.host.isConnected) return parts;
      remove();
      const host = styled(document.createElement("unsloth-agent-cursor"), {
        all: "initial", display: "block", position: "fixed", left: "0", top: "0", width: "0", height: "0",
        overflow: "visible", "pointer-events": "none", "z-index": "2147483647", contain: "layout style size",
      }, true);
      host.setAttribute("aria-hidden", "true");
      const root = host.attachShadow({ mode: "closed" });
      const mover = styled(document.createElement("div"), {
        position: "absolute", left: "0", top: "0", opacity: "0", "will-change": "transform, opacity",
        transition: "opacity 160ms ease-out",
      });
      const ring = styled(document.createElement("div"), {
        position: "absolute", left: "-14px", top: "-14px", width: "28px", height: "28px", "box-sizing": "border-box",
        "border-radius": "50%", border: "2px solid", opacity: "0",
      });
      // Hugeicons cursor-02, its corners rounded further by a round-joined stroke of the fill (white
      // outline drawn wider underneath), then scaled back to size about its tip at (3, 3). The rounded
      // tip ends up 1.25px short of (0, 0), hence the offset.
      const arrow = styled(document.createElementNS(SVG, "svg"), {
        position: "absolute", left: "-1.25px", top: "-1.25px", width: "24px", height: "24px", overflow: "visible",
      });
      arrow.setAttribute("viewBox", "0 0 24 24");
      const shape = document.createElementNS(SVG, "g");
      shape.setAttribute("transform", "translate(3 3) scale(.837) translate(-3 -3)");
      for (const [color, width] of [["#fff", "6.5"], ["#0b0b0f", "3.5"]]) {
        const path = document.createElementNS(SVG, "path");
        path.setAttribute(
          "d",
          "M9.80282 4.62973L15.8364 6.99069C19.3164 8.35243 21.0564 9.03329 20.9987 10.1133C20.941 11.1934 19.1251 11.6886 15.4933 12.6791C14.412 12.974 13.8713 13.1215 13.4964 13.4963C13.1215 13.8712 12.9741 14.4119 12.6791 15.4933C11.6887 19.125 11.1934 20.9409 10.1134 20.9986C9.03335 21.0563 8.35249 19.3163 6.99075 15.8363L4.62979 9.80276C3.20411 6.15934 2.49127 4.33764 3.41448 3.41442C4.3377 2.49121 6.15941 3.20405 9.80282 4.62973Z",
        );
        path.setAttribute("fill", color);
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", width);
        path.setAttribute("stroke-linejoin", "round");
        shape.append(path);
      }
      arrow.append(shape);
      mover.append(ring, arrow);
      root.append(mover);
      parts = { host, mover, arrow, ring, rgb: "" };
      shown = false;
      return parts;
    };
    // A modal dialog, an open popover or a fullscreen element paints above every z-index (and fullscreen
    // shows only its own subtree), so the cursor goes into the topmost one. Not the top layer itself:
    // a popover's ::backdrop could take the user's clicks.
    const layer = (): Element => {
      let top: Element | null = document.fullscreenElement;
      // Queried apart: a selector this engine doesn't know would void the other.
      for (const selector of ["dialog:modal", ":popover-open"]) {
        try {
          const open = document.querySelectorAll(selector);
          if (open.length > 0) top = open[open.length - 1];
        } catch {
          // Unknown selector.
        }
      }
      return top && !/^(video|img|canvas|iframe|input|textarea|select)$/i.test(top.tagName) ? top : document.documentElement;
    };
    function remove() {
      window.clearTimeout(idle);
      parts?.host.remove();
      parts = null;
      shown = false;
    }
    return {
      get on() {
        return on;
      },
      /** From every command, so turning it off in Settings clears it at the next call. */
      configure(command: Record<string, unknown>) {
        if (typeof command.cursor === "boolean" && command.cursor !== on) {
          on = command.cursor;
          if (!on) remove();
        }
        const color = typeof command.cursorColor === "string" ? /^rgb\((\d{1,3}), (\d{1,3}), (\d{1,3})\)$/.exec(command.cursorColor) : null;
        if (color) rgb = color.slice(1).map((part) => Math.min(255, Number(part))).join(",");
        if (typeof command.cursorStill === "boolean") still = command.cursorStill;
      },
      /** Glides to `at` (CSS px of the viewport); `press` adds the ring. Drag steps come every frame,
       *  so they follow closely (`glideMs`); a jump to a new target eases. */
      show(at: Point, press: boolean, glideMs = 180) {
        if (!on) return;
        const c = build();
        tint();
        const parent = layer();
        if (c.host.parentNode !== parent) {
          parent.append(c.host);
          shown = false;
        }
        // Appearing after a rest: at the target, not gliding in from where it faded, and without the
        // forced layout a transition reset would need.
        const glide = shown && !still;
        c.mover.style.transition = glide
          ? `transform ${glideMs}ms cubic-bezier(.2,.7,.3,1), opacity 160ms ease-out`
          : "opacity 160ms ease-out";
        c.mover.style.transform = `translate3d(${Math.round(at.x)}px, ${Math.round(at.y)}px, 0)`;
        c.mover.style.opacity = masking ? "0" : "1";
        shown = true;
        if (press && !still && typeof c.ring.animate === "function") {
          c.ring.animate(
            [
              { transform: "scale(.35)", opacity: 0.9 },
              { transform: "scale(1.5)", opacity: 0 },
            ],
            { duration: 420, delay: glide ? glideMs : 0, easing: "ease-out" },
          );
        }
        window.clearTimeout(idle);
        idle = window.setTimeout(() => {
          shown = false;
          if (parts) parts.mover.style.opacity = "0";
        }, IDLE_MS);
      },
      /** Out of the model's screenshots: it would read the pointer as part of the page. */
      mask(hide: boolean) {
        masking = hide;
        if (!parts) return;
        parts.mover.style.transition = "none";
        parts.mover.style.opacity = !hide && shown ? "1" : "0";
      },
    };
  })();
  /** Where an action will land, taken before it runs: a click can remove its own target. Scrolled as
   *  the action itself scrolls (so its own scroll is then a no-op); a dropdown, which isn't scrolled, as it is. */
  const aimAt = (command: Record<string, unknown>): Point | null => {
    const op = command.op;
    if (op !== "click" && op !== "type" && op !== "select" && op !== "pointer") return null;
    const at = pointIn(command);
    if (at || op === "pointer") return at;
    const el = byRef.get(Number(String(command.ref ?? "").replace(/^r/i, "")))?.deref();
    if (!el?.isConnected) return null;
    if (op === "click") el.scrollIntoView({ block: "center", inline: "center" });
    else if (op === "type") el.scrollIntoView({ block: "center" });
    const rect = el.getBoundingClientRect();
    return sized(rect) ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 } : null;
  };

  const elementAt = ({ x, y }: Point): Element => {
    const outside = "that point is outside the visible page; take a screenshot to see what is there.";
    if (x < 0 || y < 0 || x >= window.innerWidth || y >= window.innerHeight) throw new AgentError(outside);
    let el = document.elementFromPoint(x, y);
    while (el?.shadowRoot) {
      const inner = el.shadowRoot.elementFromPoint(x, y);
      if (!inner || inner === el) break;
      el = inner;
    }
    if (!el) throw new AgentError(outside);
    if (el instanceof HTMLIFrameElement || el.tagName.toLowerCase() === "frame") {
      throw new AgentError("that point is inside an embedded frame, which the browser tools can't reach.");
    }
    return el;
  };

  const describe = (ref: string, value?: unknown, point?: Point | null): Record<string, unknown> => {
    const found = point ? elementAt(point) : ref ? lookup(ref) : (document.activeElement ?? document.body);
    // A click on text inside a button or label reaches that control (a label's being its field or
    // button), which may submit.
    // A point lands on whatever is drawn, often an icon inside a control: gate the control.
    const control = point ? (lineage(found).find((node) => roleOf(node) !== null) ?? found) : found;
    const holder = control.closest("button, label") ?? control;
    const el = holder instanceof HTMLLabelElement && holder.control ? holder.control : holder;
    const field = el as HTMLInputElement;
    const form = (field.form ?? el.closest("form")) as HTMLFormElement | null;
    // Through shadow roots, as linkOut finds it, so the link is gated before a native click follows it.
    const anchor = lineage(el).find((node): node is HTMLAnchorElement => node instanceof HTMLAnchorElement && node.hasAttribute("href")) ?? null;
    const tag = el.tagName.toLowerCase();
    const type = (field.type || "").toLowerCase();
    const submits = Boolean(form) && isSubmitter(el);
    // What actually submits: this button, or for Enter in a field the form's default button.
    // Its formaction/formmethod override the form's, as browsers and the proxied frame honor.
    const submitter = submits ? el : defaultButton(form);
    const method = submitter?.getAttribute("formmethod") || form?.getAttribute("method") || "get";
    const action = submitter?.hasAttribute("formaction")
      ? new URL(submitter.getAttribute("formaction") ?? "", document.baseURI).href
      : form?.action || pageUrl().href;
    // form.elements, not descendants: a field joins a form from anywhere with form="id".
    const passwords = Array.from(form?.elements ?? []).filter(
      (node): node is HTMLInputElement => (node as HTMLInputElement).type === "password",
    );
    const info = {
      role: roleOf(el) ?? (anchor ? "link" : "element"),
      name: nameOf(el) || nameOf(anchor ?? el),
      tag,
      type,
      href: anchor?.href ?? "",
      autocomplete: (el.getAttribute("autocomplete") ?? "").toLowerCase(),
      formMethod: form ? method.toLowerCase() : "get",
      formAction: form ? action : "",
      formHasPassword: passwords.length > 0,
      formPasswordFilled: passwords.some((field) => field.value !== ""),
      submits,
      // What choose() will pick, so the panel gates that option's label, not the model's words.
      ...(el instanceof HTMLSelectElement && typeof value === "string"
        ? { option: clean(findOption(el, value)?.text ?? "", 200) }
        : {}),
      ...(!submits && submitter ? { submitter: clean(nameOf(submitter), 200) } : {}),
    };
    return point ? { ok: true, info, hit: refOf(found), sig: JSON.stringify(info), doc: refOf(document.documentElement) } : { ok: true, info };
  };

  const stillAt = (at: Point, expect: unknown, sig?: unknown): Element => {
    const el = elementAt(at);
    if (typeof expect === "string" && expect && refOf(el) !== expect) {
      throw new AgentError("the page changed under that point; take a screenshot to see it again.");
    }
    // And what it is: an href or form can change on the same node between the check and the act.
    if (typeof sig === "string" && sig && describe("", undefined, at).sig !== sig) {
      throw new AgentError("the page changed under that point; take a screenshot to see it again.");
    }
    return el;
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

  const mouse = (el: Element, type: string, at?: Point, buttons = 0, relatedTarget: Element | null = null): boolean => {
    const rect = el.getBoundingClientRect();
    const init = {
      bubbles: !/(enter|leave)$/.test(type),
      cancelable: true,
      composed: true,
      clientX: at ? at.x : rect.left + rect.width / 2,
      clientY: at ? at.y : rect.top + rect.height / 2,
      button: 0,
      buttons,
      relatedTarget,
      view: window,
    };
    const event = type.startsWith("pointer")
      ? new PointerEvent(type, { ...init, pointerId: 1, pointerType: "mouse", isPrimary: true })
      : new MouseEvent(type, init);
    return el.dispatchEvent(event);
  };

  const linkOut = (el: Element, directLinks: boolean): string | null => {
    const anchor = lineage(el).find((node): node is HTMLAnchorElement => node instanceof HTMLAnchorElement && node.hasAttribute("href"));
    const href = anchor?.href ?? "";
    if (!anchor || !/^https?:/i.test(href)) return null;
    const target = new URL(href);
    const hashOnly = target.hash && target.href.split("#")[0] === pageUrl().href.split("#")[0];
    return !hashOnly && (directLinks || anchor.target === "_blank") ? href : null;
  };

  const parentOf = (node: Element): Element | null => {
    if (node.parentElement) return node.parentElement;
    const root = node.getRootNode();
    return root instanceof ShadowRoot ? root.host : null;
  };
  const lineage = (el: Element | null): Element[] => {
    const chain: Element[] = [];
    for (let node = el; node; node = parentOf(node)) chain.push(node);
    return chain;
  };

  let hovered: Element | null = null;
  const hoverTo = (el: Element, at: Point, buttons = 0) => {
    if (hovered !== el) {
      const old = hovered?.isConnected ? hovered : null;
      const before = lineage(old);
      const after = lineage(el);
      if (old) {
        mouse(old, "pointerout", at, buttons, el);
        mouse(old, "mouseout", at, buttons, el);
        for (const node of before.filter((node) => !after.includes(node))) {
          mouse(node, "pointerleave", at, buttons, el);
          mouse(node, "mouseleave", at, buttons, el);
        }
      }
      mouse(el, "pointerover", at, buttons, old);
      mouse(el, "mouseover", at, buttons, old);
      for (const node of after.filter((node) => !before.includes(node)).reverse()) {
        mouse(node, "pointerenter", at, buttons, old);
        mouse(node, "mouseenter", at, buttons, old);
      }
      hovered = el;
    }
    mouse(el, "pointermove", at, buttons);
    mouse(el, "mousemove", at, buttons);
  };

  const FOCUSABLE = "a[href], button, input, select, textarea, summary, [tabindex], [contenteditable]";
  const pressAt = (el: Element, at: Point, expect?: unknown, sig?: unknown) => {
    hoverTo(el, at);
    // A hover handler can open an overlay or rewire the target: the press waits for a fresh check.
    if (expect && stillAt(at, expect, sig) !== el) {
      throw new AgentError("the page changed under that point; take a screenshot to see it again.");
    }
    mouse(el, "pointerdown", at, 1);
    if (mouse(el, "mousedown", at, 1)) (el.closest(FOCUSABLE) as HTMLElement | null)?.focus?.({ preventScroll: true });
  };

  const clickAt = (at: Point, directLinks: boolean, expect: unknown, sig: unknown): Record<string, unknown> => {
    const el = stillAt(at, expect, sig);
    // As a real click: a disabled control takes none, and a dispatched one would still reach its listeners.
    if (el.closest(":disabled")) return { ok: false, error: `${refOf(el)} is disabled or read-only.` };
    const navigate = linkOut(el, directLinks);
    if (navigate) return { ok: true, navigate };
    pressAt(el, at, expect, sig);
    mouse(el, "pointerup", at);
    mouse(el, "mouseup", at);
    mouse(el, "click", at);
    return { ok: true };
  };

  const locate = (ref: string, scroll: boolean): Record<string, unknown> => {
    const el = lookup(ref);
    const centre = () => {
      const rect = el.getBoundingClientRect();
      return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    };
    const inView = ({ x, y }: Point) => x >= 0 && y >= 0 && x < window.innerWidth && y < window.innerHeight;
    if (scroll && !inView(centre())) el.scrollIntoView({ block: "nearest", inline: "nearest" });
    const at = centre();
    if (!inView(at)) return { ok: true };
    if (!lineage(elementAt(at)).includes(el)) {
      return { ok: false, error: `${ref} is covered by an overlay; snapshot to find its close button, or scroll first.` };
    }
    return { ok: true, ...at };
  };

  // `html`: the source is draggable, so a browser would send drag events, not mouse moves.
  let drag: { html: { source: Element; data: DataTransfer; over: Element | null; accepted: boolean } | null } | null = null;
  const dragEvent = (el: Element, type: string, at: Point, data: DataTransfer): boolean =>
    el.dispatchEvent(
      new DragEvent(type, { bubbles: true, cancelable: true, composed: true, clientX: at.x, clientY: at.y, dataTransfer: data, view: window }),
    );

  const pointer = (phase: unknown, at: Point, expect: unknown, sig: unknown): Record<string, unknown> => {
    // The press and the release are held to the elements the user was asked about; moves are not.
    const el = phase === "down" || phase === "up" ? stillAt(at, expect, sig) : elementAt(at);
    if (phase === "down") {
      if (el.closest(":disabled")) return { ok: false, error: `${refOf(el)} is disabled or read-only.` };
      pressAt(el, at, expect, sig);
      drag = { html: null };
      const source = lineage(el).find((node) => node.getAttribute("draggable") === "true");
      if (source && typeof DataTransfer === "function" && typeof DragEvent === "function") {
        const data = new DataTransfer();
        if (dragEvent(source, "dragstart", at, data)) drag.html = { source, data, over: null, accepted: false };
      }
      return { ok: true };
    }
    if (phase === "move") {
      const html = drag?.html;
      if (!html) {
        hoverTo(el, at, drag ? 1 : 0);
        return { ok: true };
      }
      dragEvent(html.source, "drag", at, html.data);
      if (html.over !== el) {
        if (html.over) dragEvent(html.over, "dragleave", at, html.data);
        dragEvent(el, "dragenter", at, html.data);
        html.over = el;
      }
      html.accepted = !dragEvent(el, "dragover", at, html.data);
      return { ok: true };
    }
    const html = drag?.html;
    drag = null;
    if (phase === "cancel") {
      key(document.activeElement ?? document.body, "Escape");
      if (html) dragEvent(html.source, "dragend", at, html.data);
      else {
        hoverTo(el, at, 1);
        mouse(el, "pointerup", at);
        mouse(el, "mouseup", at);
      }
      return { ok: true };
    }
    if (html) {
      if (html.over !== el) {
        if (html.over) dragEvent(html.over, "dragleave", at, html.data);
        dragEvent(el, "dragenter", at, html.data);
        html.over = el;
      }
      html.accepted = !dragEvent(el, "dragover", at, html.data);
      // Drop zone handlers can rewire the target: the drop waits for a fresh check, as the press does.
      if (expect && stillAt(at, expect, sig) !== el) {
        dragEvent(html.source, "dragend", at, html.data);
        throw new AgentError("the page changed under that point; take a screenshot to see it again.");
      }
      if (html.accepted) dragEvent(el, "drop", at, html.data);
      dragEvent(html.source, "dragend", at, html.data);
      return html.accepted ? { ok: true } : { ok: true, note: "Nothing at the end point took the drop." };
    }
    hoverTo(el, at, 1);
    if (expect && stillAt(at, expect, sig) !== el) {
      throw new AgentError("the page changed under that point; take a screenshot to see it again.");
    }
    mouse(el, "pointerup", at);
    mouse(el, "mouseup", at);
    return { ok: true };
  };

  const SCRIPT_CHARS = 20000;
  const scripts = new Map<string, { done: boolean; text: string }>();
  const thrown = (error: unknown): string => {
    try {
      return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    } catch {
      return "an error";
    }
  };
  const shown = (value: unknown): string => {
    if (value === undefined) return "undefined (use return to give back a value)";
    try {
      const json = JSON.stringify(value, (_key, item) =>
        item instanceof Element ? `<${item.tagName.toLowerCase()}>` : typeof item === "bigint" ? String(item) : item,
      );
      if (json !== undefined) return json;
    } catch {
    }
    try {
      return String(value);
    } catch {
      return "a value that can't be shown";
    }
  };
  /** `doc`: the document the user approved the script on (`status` gives it); any other refuses. */
  const evaluate = (id: string, doc: unknown, start: () => unknown): Record<string, unknown> => {
    if (typeof doc !== "string" || refOf(document.documentElement) !== doc) {
      return { ok: false, error: "the page changed while the user was asked, so the script did not run; take a snapshot." };
    }
    const oldest = scripts.keys().next().value;
    if (scripts.size >= 20 && oldest !== undefined) scripts.delete(oldest);
    const entry = { done: false, text: "" };
    scripts.set(String(id), entry);
    const finish = (text: string) => {
      entry.done = true;
      entry.text = text.length > SCRIPT_CHARS ? `${text.slice(0, SCRIPT_CHARS)}… (cut)` : text;
    };
    try {
      Promise.resolve(start()).then(
        (value) => finish(`Returned: ${shown(value)}`),
        (error) => finish(`Threw ${thrown(error)}`),
      );
    } catch (error) {
      finish(`Threw ${thrown(error)}`);
    }
    return { ok: true, pending: true };
  };
  const compileAndEvaluate = (id: string, doc: unknown, script: string): Record<string, unknown> => {
    let start: () => unknown;
    try {
      const AsyncFunction = Object.getPrototypeOf(async () => undefined).constructor as new (body: string) => () => unknown;
      start = new AsyncFunction(script);
    } catch (error) {
      start = () => Promise.reject(error);
    }
    return evaluate(id, doc, start);
  };
  const evaluated = (id: string): Record<string, unknown> => {
    const entry = scripts.get(id);
    if (!entry) return { ok: false, error: "the script's page went away before it finished; take a snapshot." };
    if (!entry.done) return { ok: true, pending: true };
    scripts.delete(id);
    return { ok: true, text: entry.text };
  };

  const click = (ref: string, directLinks: boolean): Record<string, unknown> => {
    const el = lookup(ref);
    el.scrollIntoView({ block: "center", inline: "center" });
    const over = covering(el);
    if (over) {
      // No page words here: errors reach the model outside the untrusted-content fence.
      return { ok: false, error: `${ref} is covered by an overlay; snapshot to find its close button, or scroll first.` };
    }
    const navigate = linkOut(el, directLinks);
    if (navigate) return { ok: true, navigate };
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

  // The agent works only in its own tab: a form or link aimed at a new tab (target, formtarget)
  // would be dropped by the panel, which opens tabs only on a real user click, while the model hears
  // it went. The target is read after the event, by the browser or by the proxy shell's deferred
  // setTimeout, so it points here for the action and is put back on a later timer.
  const inOwnTab = <T>(act: () => T): T => {
    const restore: Array<() => void> = [];
    const point = (el: unknown, name: string) => {
      if (!(el instanceof Element) || !el.hasAttribute(name)) return;
      const old = el.getAttribute(name) ?? "";
      el.setAttribute(name, "_self");
      restore.push(() => el.setAttribute(name, old));
    };
    const onSubmit = (event: Event) => {
      point((event as SubmitEvent).submitter, "formtarget");
      point(event.target, "target");
    };
    const onClick = (event: Event) => {
      const link = event.composedPath().find((node) => node instanceof Element && node.matches("a[href], area[href]"));
      point(link, "target");
    };
    window.addEventListener("submit", onSubmit, true);
    window.addEventListener("click", onClick, true);
    try {
      return act();
    } finally {
      window.removeEventListener("submit", onSubmit, true);
      window.removeEventListener("click", onClick, true);
      if (restore.length) setTimeout(() => restore.forEach((undo) => undo()));
    }
  };

  const press = (name: string, on?: Element): Record<string, unknown> => {
    const active = on ?? (document.activeElement && document.activeElement !== document.body ? document.activeElement : null);
    const target = active ?? document.body ?? document.documentElement;
    // WebKit acts on a synthetic keypress (implicit submit, a button's click), and so may the page's
    // own key handler: acting again would send the form or click the button twice.
    let acted = false;
    let submitted = false;
    const mark = (event: Event) => {
      acted = true;
      if (event.type === "submit") submitted = true;
    };
    window.addEventListener("submit", mark, true);
    window.addEventListener("click", mark, true);
    let proceed: boolean;
    try {
      proceed = key(target, name);
    } finally {
      window.removeEventListener("submit", mark, true);
      window.removeEventListener("click", mark, true);
    }
    if (!proceed || acted) return submitted ? { ok: true, note: "Submitted the form." } : { ok: true };
    const tag = target.tagName.toLowerCase();
    if (name === "Enter") {
      const field = target as HTMLInputElement;
      const inputType = tag === "input" ? (field.type || "text").toLowerCase() : "";
      // Only a text field sends its form on Enter, the case the panel gates as a submit.
      if (tag === "input" && field.form && !NOT_TEXT.has(inputType)) {
        // With the button describe() gated, so its formaction and formmethod apply here too.
        field.form.requestSubmit(defaultButton(field.form) ?? null);
        return { ok: true, note: "Submitted the form." };
      }
      if (tag === "button" || tag === "a" || BUTTON_INPUTS.has(inputType)) (target as HTMLElement).click();
    } else if (name === "Space" && (tag === "button" || (tag === "input" && ["checkbox", "radio"].includes((target as HTMLInputElement).type)))) {
      (target as HTMLElement).click();
    } else if ((name === "Backspace" || name === "Delete") && (tag === "input" || tag === "textarea")) {
      const field = target as HTMLInputElement;
      const end = field.selectionEnd ?? field.value.length;
      const start = field.selectionStart ?? end;
      // A selection goes either way; otherwise one character before (Backspace) or after (Delete).
      const [from, to] = start !== end ? [start, end] : name === "Backspace" ? [Math.max(0, start - 1), end] : [start, Math.min(field.value.length, end + 1)];
      setValue(field, field.value.slice(0, from) + field.value.slice(to));
      field.setSelectionRange?.(from, from);
    } else if ((name === "Backspace" || name === "Delete") && (target as HTMLElement).isContentEditable) {
      document.execCommand(name === "Backspace" ? "delete" : "forwardDelete");
    } else if (CARET_KEYS.has(name) && (tag === "input" || tag === "textarea")) {
      // A synthetic key doesn't move the caret, so move it as the key would on one line.
      const field = target as HTMLInputElement;
      const at = field.selectionStart ?? 0;
      const to = name === "Home" ? 0 : name === "End" ? field.value.length : Math.max(0, Math.min(field.value.length, at + (name === "ArrowLeft" ? -1 : 1)));
      try {
        field.setSelectionRange(to, to);
      } catch {
        return { ok: false, error: "that key does nothing in this field; click or type instead." };
      }
    } else if (active && (EDITABLE_TAGS.has(tag) || (target as HTMLElement).isContentEditable) && name !== "Escape" && name !== "Tab") {
      // Not a field key we can act out: say so rather than report a change that didn't happen.
      return { ok: false, error: "that key does nothing in this field; click or type instead." };
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
    const field = el as HTMLInputElement;
    if ((tag === "textarea" || tag === "input") && (field.matches(":disabled") || field.readOnly)) {
      return { ok: false, error: `${ref} is disabled or read-only.` };
    }
    if (tag === "textarea" || (tag === "input" && TEXT_INPUTS.has(field.type.toLowerCase()))) {
      // A date or time field drops a value it can't parse: tried on a detached field of the same
      // type first, so a bad value never blanks the page's field or fires its change handlers.
      const format = DATE_FORMATS[field.type.toLowerCase()];
      if (format && text) {
        const probe = document.createElement("input");
        probe.type = field.type;
        probe.value = text;
        if (!probe.value) return { ok: false, error: `${ref} takes a value like ${format}.` };
      }
      setValue(field, text);
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

  const findOption = (select: HTMLSelectElement, value: string): HTMLOptionElement | undefined => {
    const wanted = value.trim().toLowerCase();
    return (
      Array.from(select.options).find(
        (candidate) => candidate.text.trim().toLowerCase() === wanted || candidate.value.toLowerCase() === wanted,
      ) ?? Array.from(select.options).find((candidate) => candidate.text.toLowerCase().includes(wanted))
    );
  };

  const choose = (ref: string, value: string): Record<string, unknown> => {
    const el = lookup(ref);
    if (el instanceof HTMLSelectElement) {
      // :disabled, not .disabled: a disabled fieldset disables what is inside it.
      if (el.matches(":disabled")) return { ok: false, error: `${ref} is disabled or read-only.` };
      const option = findOption(el, value);
      if (!option) return { ok: false, error: `${ref} has no such option; snapshot it to see its options.` };
      el.value = option.value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return { ok: true, note: `Selected ${quoted(clean(option.text, 60))}.` };
    }
    const role = roleOf(el);
    if (role === "checkbox" || role === "radio" || role === "switch") {
      const on = ["true", "on", "yes", "checked", "1", "check"].includes(value.trim().toLowerCase());
      const current = el instanceof HTMLInputElement ? el.checked : el.getAttribute("aria-checked") === "true";
      if (current && !on && role === "radio") {
        return { ok: false, error: `${ref} is a radio button; select another option in its group instead.` };
      }
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

  const dispatch = (command: Record<string, unknown>): Record<string, unknown> => {
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
        case "status": {
          // Fetch/XHR the proxied shell counts as open since `since` (routes/browser_network.py).
          const pending = host[Symbol.for("unsloth.browser.pending")];
          const since = typeof command.since === "number" ? command.since : Number.POSITIVE_INFINITY;
          return {
            ok: true,
            quietMs: Math.round(performance.now() - lastMutation),
            // Absent where nothing counts them (native views), so a wait can't take that for "none open".
            ...(typeof pending === "function" ? { requests: Number(pending(since)) || 0 } : {}),
            title: document.title,
            ...view(),
          };
        }
        case "snapshot":
          return snapshot(command.fullPage === true, ref || undefined);
        case "find":
          return find(String(command.query ?? ""));
        case "describe":
          return describe(ref, command.value, pointIn(command));
        case "click": {
          const at = pointIn(command);
          return inOwnTab(() =>
            at ? clickAt(at, command.directLinks === true, command.expect, command.sig) : click(ref, command.directLinks === true),
          );
        }
        case "locate":
          return locate(ref, command.scroll === true);
        case "pointer": {
          const at = pointIn(command);
          return at ? inOwnTab(() => pointer(command.phase, at, command.expect, command.sig)) : { ok: false, error: "unknown command" };
        }
        case "evaluate":
          return compileAndEvaluate(String(command.id ?? ""), command.doc, String(command.script ?? ""));
        case "evaluated":
          return evaluated(String(command.id ?? ""));
        case "type":
          return inOwnTab(() => type(ref, String(command.text ?? ""), command.submit === true));
        case "select":
          return inOwnTab(() => choose(ref, String(command.value ?? "")));
        case "scroll":
          return scrollPage(String(command.direction ?? "down"), ref || undefined);
        case "press":
          return inOwnTab(() => press(String(command.key ?? "")));
        case "mask":
          return mask(command.on === true);
        default:
          return { ok: false, error: "unknown command" };
      }
    } catch (error) {
      // Only the agent's own words: a page script can throw anything, and errors reach the model unfenced.
      if (error instanceof AgentError) return { ok: false, error: clean(error.message, 300) };
      return { ok: false, error: "the page's own script failed during that action; take a fresh snapshot." };
    }
  };

  const run = (command: Record<string, unknown>): Record<string, unknown> => {
    cursor.configure(command);
    let aim: Point | null = null;
    try {
      aim = cursor.on ? aimAt(command) : null;
    } catch {
      // Only a picture: the action still runs.
    }
    const answer = dispatch(command);
    if (aim && answer.ok === true) {
      try {
        const move = command.op === "pointer" && command.phase === "move";
        cursor.show(aim, command.op !== "pointer" || command.phase === "down", move ? 40 : 180);
      } catch {
        // Only a picture: the action already happened.
      }
    }
    return answer;
  };

  // For native views, which look it up on every call. A page that defines KEY first still wins
  // there: the agent shares the page's world, so its answers are untrusted.
  function view() {
    return {
      width: window.innerWidth,
      height: window.innerHeight,
      scrollX: Math.round(window.scrollX),
      scrollY: Math.round(window.scrollY),
      scrolls,
      doc: refOf(document.documentElement),
    };
  }
  const owns = (doc: unknown): boolean => typeof doc === "string" && refOf(document.documentElement) === doc;
  const agent: PageAgent = { run, evaluate, owns };
  if (!Object.prototype.hasOwnProperty.call(host, KEY)) Object.defineProperty(host, KEY, { value: agent, enumerable: false });
  return agent;
}

export function pageAgentSource(): string {
  return `(${installPageAgent.toString()})();`;
}
