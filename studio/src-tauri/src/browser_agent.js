// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

(function (NAME) {
  "use strict";
  const win = window;
  try {
    if (win.top !== win) return;
  } catch (_) {
    return;
  }
  if (Object.prototype.hasOwnProperty.call(win, NAME)) return;

  // natives are captured before page scripts run and called uncurried, so page tampering cannot break or observe them
  const FnProto = Function.prototype;
  const uncurry = FnProto.bind.bind(FnProto.call);
  const { defineProperty, getOwnPropertyDescriptor: getOwnDesc, create: createObject, freeze, keys: objectKeys } = Object;
  const hasOwn = uncurry(Object.prototype.hasOwnProperty);
  const { stringify: jsonStringify, parse: jsonParse } = JSON;
  const { max, min, round, floor, ceil, abs } = Math;
  const isArray = Array.isArray;
  const Str = String;
  const parseNum = parseFloat;
  const URLCtor = URL;
  const WeakSetCtor = WeakSet;
  const WeakRefCtor = win.WeakRef;
  // a new id means a new document, which proves a navigation replaced the page; the URL changes before the load starts
  const DOC_ID = Math.random().toString(36).slice(2);

  const method = (proto, name) => (proto && typeof proto[name] === "function" ? uncurry(proto[name]) : null);
  const accessor = (proto, name, kind) => {
    const d = proto && getOwnDesc(proto, name);
    return d && d[kind] ? uncurry(d[kind]) : null;
  };
  function makeMap(Ctor) {
    const get = uncurry(Ctor.prototype.get);
    const set = uncurry(Ctor.prototype.set);
    return () => {
      const m = new Ctor();
      return { get: (k) => get(m, k), set: (k, v) => set(m, k, v) };
    };
  }
  const newWeakMap = makeMap(WeakMap);
  const newMap = makeMap(Map);
  const wsAdd = uncurry(WeakSetCtor.prototype.add);
  const wsHas = uncurry(WeakSetCtor.prototype.has);
  const deref = WeakRefCtor ? uncurry(WeakRefCtor.prototype.deref) : null;

  const EP = win.Element.prototype;
  const HEP = win.HTMLElement.prototype;
  const DP = win.Document.prototype;
  const IP = win.HTMLInputElement.prototype;
  const TAP = win.HTMLTextAreaElement.prototype;
  const PerfP = win.Performance.prototype;
  const N = {
    rect: method(EP, "getBoundingClientRect"),
    rects: method(EP, "getClientRects"),
    getAttr: method(EP, "getAttribute"),
    hasAttr: method(EP, "hasAttribute"),
    setAttr: method(EP, "setAttribute"),
    matches: method(EP, "matches"),
    checkVisibility: method(EP, "checkVisibility"),
    closest: method(EP, "closest"),
    qsa: method(EP, "querySelectorAll"),
    docQsa: method(DP, "querySelectorAll"),
    scrollIntoView: method(EP, "scrollIntoView"),
    elScrollBy: method(EP, "scrollBy"),
    shadowRoot: accessor(EP, "shadowRoot", "get"),
    attachShadow: method(EP, "attachShadow"),
    remove: method(EP, "remove"),
    append: method(win.Node.prototype, "appendChild"),
    getRootNode: method(win.Node.prototype, "getRootNode"),
    rootFromPoint: method(win.ShadowRoot.prototype, "elementFromPoint"),
    // read through the prototypes: a page can shadow these on its own nodes and fake containment
    parentNode: accessor(win.Node.prototype, "parentNode", "get"),
    nodeType: accessor(win.Node.prototype, "nodeType", "get"),
    elSlot: accessor(EP, "assignedSlot", "get"),
    textSlot: accessor(win.Text.prototype, "assignedSlot", "get"),
    shadowHost: accessor(win.ShadowRoot.prototype, "host", "get"),
    focus: method(HEP, "focus"),
    click: method(HEP, "click"),
    showPopover: method(HEP, "showPopover"),
    hidePopover: method(HEP, "hidePopover"),
    dispatch: method(win.EventTarget.prototype, "dispatchEvent"),
    inputGet: accessor(IP, "value", "get"),
    inputSet: accessor(IP, "value", "set"),
    textareaGet: accessor(TAP, "value", "get"),
    textareaSet: accessor(TAP, "value", "set"),
    inputSelStart: accessor(IP, "selectionStart", "get"),
    inputSelEnd: accessor(IP, "selectionEnd", "get"),
    inputSetSel: method(IP, "setSelectionRange"),
    textareaSelStart: accessor(TAP, "selectionStart", "get"),
    textareaSelEnd: accessor(TAP, "selectionEnd", "get"),
    textareaSetSel: method(TAP, "setSelectionRange"),
    optionSelectedSet: accessor(win.HTMLOptionElement.prototype, "selected", "set"),
    elementFromPoint: method(DP, "elementFromPoint"),
    execCommand: method(DP, "execCommand"),
    getSelection: method(DP, "getSelection"),
    createElement: method(DP, "createElement"),
    getElementById: method(DP, "getElementById"),
    fragGetElementById: method(win.DocumentFragment.prototype, "getElementById"),
    hasFocus: method(DP, "hasFocus"),
    assignedNodes: method(win.HTMLSlotElement && win.HTMLSlotElement.prototype, "assignedNodes"),
    requestSubmit: method(win.HTMLFormElement.prototype, "requestSubmit"),
    formSubmit: method(win.HTMLFormElement.prototype, "submit"),
    dialogClose: method(win.HTMLDialogElement && win.HTMLDialogElement.prototype, "close"),
    computedStyle: uncurry(win.getComputedStyle),
    setTimeout: uncurry(win.setTimeout),
    winScrollBy: uncurry(win.scrollBy),
    perfNow: method(PerfP, "now"),
    entriesByType: method(PerfP, "getEntriesByType"),
    moObserve: method(win.MutationObserver.prototype, "observe"),
    listen: method(win.EventTarget.prototype, "addEventListener"),
    unlisten: method(win.EventTarget.prototype, "removeEventListener"),
    evTarget: accessor(win.Event.prototype, "target", "get"),
    evType: accessor(win.Event.prototype, "type", "get"),
    evTrusted: (e) => e.isTrusted,
    mouseX: accessor(win.MouseEvent.prototype, "clientX", "get"),
    mouseY: accessor(win.MouseEvent.prototype, "clientY", "get"),
    contains: method(win.Node.prototype, "contains"),
  };
  const Ev = {
    Event: win.Event,
    Mouse: win.MouseEvent,
    Pointer: win.PointerEvent || win.MouseEvent,
    Keyboard: win.KeyboardEvent,
    Input: win.InputEvent || win.Event,
    Wheel: win.WheelEvent,
  };
  const doc = win.document;
  const loc = win.location;
  const perf = win.performance;
  const now = () => N.perfNow(perf);
  const installedAt = now();

  const ownNodes = new WeakSetCtor();
  const isOwn = (n) => wsHas(ownNodes, n);
  let lastMutation = installedAt;
  let mutationCount = 0;
  let resourceCount = 0;
  let observedResources = 0;
  let lastResource = installedAt;

  function ownRecord(r) {
    if (isOwn(r.target)) return true;
    if (r.type !== "childList") return false;
    let any = false;
    for (const list of [r.addedNodes, r.removedNodes]) {
      for (let i = 0; i < list.length; i++) {
        if (!isOwn(list[i])) return false;
        any = true;
      }
    }
    return any;
  }
  try {
    const mo = new win.MutationObserver((records) => {
      for (let i = 0; i < records.length; i++) {
        if (ownRecord(records[i])) continue;
        lastMutation = now();
        mutationCount++;
        return;
      }
    });
    N.moObserve(mo, doc, { subtree: true, childList: true, attributes: true, characterData: true });
  } catch (_) {
    /* settle detection degrades to time since install */
  }
  // the resource timeline buffer stops growing at 250 entries, so an observer keeps counting past that
  try {
    new win.PerformanceObserver((list) => {
      const entries = list.getEntries();
      observedResources += entries.length;
      for (let i = 0; i < entries.length; i++) noteResource(entries[i]);
    }).observe({ type: "resource", buffered: true });
  } catch (_) {
    /* polling getEntriesByType still works */
  }
  function noteResource(e) {
    const end = e.responseEnd || e.startTime + e.duration || 0;
    if (end > lastResource) lastResource = end;
  }
  function pollResources() {
    const entries = attempt(() => N.entriesByType(perf, "resource"), []);
    const n = max(entries.length, observedResources);
    if (n <= resourceCount) return;
    for (let i = resourceCount; i < entries.length; i++) noteResource(entries[i]);
    if (n > entries.length) lastResource = max(lastResource, now() - 50);
    resourceCount = n;
  }

  const WS = /\s+/g;
  const WS_SPLIT = /\s+/;
  const CITATION = /^\[[^\]]{1,6}\]$/;
  const collapse = (s) => (s ? Str(s).replace(WS, " ").trim() : "");
  // rust's json decoder rejects a lone surrogate, which a page's own text can carry
  const wellFormed = (s) => s.replace(/[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/g, "\ufffd");
  // a cut index is moved off the middle of a surrogate pair because a lone half fails JSON decoding
  const unsplit = (s, i) => (i > 0 && i < s.length && (s.charCodeAt(i - 1) & 0xfc00) === 0xd800 ? i - 1 : i);
  const clip = (s, n) => (s.length > n ? s.slice(0, unsplit(s, n - 1)).trimEnd() + "…" : s);
  const quote = (s) => jsonStringify(s);
  const attr = (el, name) => N.getAttr(el, name);
  const styleOf = (el) => N.computedStyle(win, el);
  const viewOf = (el) => (el.ownerDocument && el.ownerDocument.defaultView) || win;
  const fail = (code, message) => ({ agentError: true, code, message });
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  function attempt(fn, fallback) {
    try {
      return fn();
    } catch (_) {
      return fallback;
    }
  }
  function alnumCount(s) {
    const m = s.match(/[\p{L}\p{N}]/gu);
    return m ? m.length : 0;
  }
  // WebKit logs a console error per contentDocument read on a cross-origin frame, so plainly foreign frames are skipped
  function frameDoc(frame) {
    return attempt(() => {
      const src = attr(frame, "src");
      if (src && !/^(about:|javascript:|data:|blob:)/i.test(src.trim())) {
        if (new URLCtor(src, frame.baseURI).origin !== frame.ownerDocument.location.origin) return null;
      }
      return frame.contentDocument;
    }, null);
  }
  function frameElementOf(d) {
    return attempt(() => {
      const v = d.defaultView;
      return v && v !== win ? v.frameElement : null;
    }, null);
  }
  function parentOf(n) {
    const kind = N.nodeType(n);
    const slot = kind === 1 ? N.elSlot(n) : kind === 3 && N.textSlot ? N.textSlot(n) : null;
    if (slot) return slot;
    const p = N.parentNode(n);
    if (!p) return null;
    const parentKind = N.nodeType(p);
    if (parentKind === 11) return attempt(() => N.shadowHost(p), null);
    return parentKind === 9 ? frameElementOf(p) : p;
  }
  function containsComposed(anc, n) {
    for (let x = n; x; x = parentOf(x)) if (x === anc) return true;
    return false;
  }
  // flat-tree children: open shadow roots replace light children, slots yield assigned nodes, closed roots are hidden
  function forEachChild(node, fn) {
    if (node.nodeType === 1) {
      const sr = N.shadowRoot(node);
      if (sr) {
        for (let c = sr.firstChild; c; c = c.nextSibling) fn(c);
        return;
      }
      if (node.localName === "slot" && N.assignedNodes) {
        const nodes = N.assignedNodes(node, { flatten: true });
        if (nodes.length) {
          for (let i = 0; i < nodes.length; i++) fn(nodes[i]);
          return;
        }
      }
    }
    for (let c = node.firstChild; c; c = c.nextSibling) fn(c);
  }
  function deepActive() {
    let a = doc.activeElement;
    for (let guard = 0; a && guard < 32; guard++) {
      const sr = N.shadowRoot(a);
      const d = sr ? null : a.localName === "iframe" || a.localName === "frame" ? frameDoc(a) : null;
      const inner = sr ? sr.activeElement : d && d.activeElement !== d.body ? d.activeElement : null;
      if (!inner) break;
      a = inner;
    }
    return a || null;
  }
  function frameOffset(d) {
    let x = 0;
    let y = 0;
    for (let guard = 0; d && d !== doc && guard < 16; guard++) {
      const fe = frameElementOf(d);
      if (!fe) break;
      const r = N.rect(fe);
      const cs = styleOf(fe);
      x += r.left + fe.clientLeft + (parseNum(cs.paddingLeft) || 0);
      y += r.top + fe.clientTop + (parseNum(cs.paddingTop) || 0);
      d = fe.ownerDocument;
    }
    return { x, y };
  }
  function boxOf(r, ox, oy) {
    return { l: r.left + ox, t: r.top + oy, r: r.right + ox, b: r.bottom + oy, w: r.width, h: r.height };
  }
  function topRect(el) {
    const o = el.ownerDocument === doc ? { x: 0, y: 0 } : frameOffset(el.ownerDocument);
    return boxOf(N.rect(el), o.x, o.y);
  }

  const wordSet = (words) => {
    const o = createObject(null);
    for (const w of words.split(" ")) o[w] = 1;
    return o;
  };
  const INPUT_ROLES = {
    button: "button", submit: "button", reset: "button", image: "button", file: "button", color: "button",
    checkbox: "checkbox", radio: "radio", range: "slider", number: "spinbutton", search: "searchbox",
  };
  const TAG_ROLES = {
    button: "button", textarea: "textbox", h1: "heading", h2: "heading", h3: "heading", h4: "heading", h5: "heading",
    h6: "heading", nav: "navigation", main: "main", aside: "complementary", form: "form", search: "search",
    dialog: "dialog", option: "option", iframe: "iframe", frame: "iframe",
  };
  const TEXT_INPUTS = wordSet("text search email url tel password number");
  const VALUE_INPUTS = wordSet("date time datetime-local month week color range");
  const CTL_ROLES = wordSet(
    "button link checkbox radio switch tab menuitem menuitemcheckbox menuitemradio option combobox textbox searchbox " +
      "slider spinbutton treeitem gridcell listbox",
  );
  const NAME_FROM_CONTENT = wordSet(
    "button link checkbox radio switch tab menuitem menuitemcheckbox menuitemradio option treeitem heading cell gridcell " +
      "columnheader rowheader tooltip clickable img",
  );
  const LANDMARKS = {
    banner: "banner", navigation: "navigation", main: "main", complementary: "complementary", contentinfo: "contentinfo",
    form: "form", search: "search", region: "region", dialog: "dialog", alertdialog: "dialog",
  };
  const SECTIONING = "article,aside,main,nav,section,[role=article],[role=complementary],[role=main],[role=navigation],[role=region]";
  const SECTIONING_TAGS = wordSet("article aside main nav section");
  const SKIP_TAGS = wordSet("script style noscript template head meta link title base datalist param source track");
  const INLINE_TAGS = wordSet(
    "a abbr b bdi bdo cite code data dfn em font i kbd mark q s samp small span strong sub sup time u var wbr label img",
  );

  const inputType = (el) => Str(el.type || "text").toLowerCase();
  function isEditingHost(el) {
    if (!el.isContentEditable) return false;
    const p = el.parentElement;
    return !(p && p.isContentEditable);
  }
  function roleOf(el, sectioned) {
    const explicit = attr(el, "role");
    if (explicit) {
      const r = explicit.trim().split(WS_SPLIT)[0].toLowerCase();
      if (r && r !== "none" && r !== "presentation" && r !== "generic") return r;
    }
    const t = el.localName;
    if (TAG_ROLES[t]) return TAG_ROLES[t];
    switch (t) {
      case "a":
      case "area":
        return N.hasAttr(el, "href") ? "link" : "";
      case "input": {
        const ty = inputType(el);
        if (ty === "hidden") return "";
        if (N.hasAttr(el, "list") && TEXT_INPUTS[ty]) return "combobox";
        return INPUT_ROLES[ty] || "textbox";
      }
      case "select":
        return el.multiple || el.size > 1 ? "listbox" : "combobox";
      case "summary":
        return el.parentElement && el.parentElement.localName === "details" ? "button" : "";
      case "header":
      case "footer": {
        const scoped = sectioned !== undefined ? sectioned : !!(el.parentElement && N.closest(el.parentElement, SECTIONING));
        return scoped ? "" : t === "header" ? "banner" : "contentinfo";
      }
      case "section":
        return N.hasAttr(el, "aria-label") || N.hasAttr(el, "aria-labelledby") ? "region" : "";
      case "img":
        return attr(el, "alt") === "" ? "" : "img";
    }
    return N.hasAttr(el, "contenteditable") && isEditingHost(el) ? "textbox" : "";
  }
  function headingLevel(el) {
    const m = /^h([1-6])$/.exec(el.localName);
    const l = m ? +m[1] : parseInt(attr(el, "aria-level"), 10);
    return l > 0 ? l : 2;
  }
  function hiddenForName(n) {
    if (N.hasAttr(n, "hidden") || attr(n, "aria-hidden") === "true") return true;
    const cs = styleOf(n);
    return cs.display === "none" || cs.visibility === "hidden" || cs.visibility === "collapse";
  }
  function byId(root, id) {
    if (root && root.nodeType === 9) return N.getElementById(root, id);
    return root && root.nodeType === 11 && N.fragGetElementById ? N.fragGetElementById(root, id) : null;
  }
  function svgTitle(svg) {
    for (let c = svg.firstElementChild; c; c = c.nextElementSibling) if (c.localName === "title") return collapse(c.textContent);
    return "";
  }
  const optionLabel = (o) => collapse(o.label || o.text);
  function selectedLabel(sel) {
    const o = sel.options && sel.options[sel.selectedIndex];
    return o ? optionLabel(o) : "";
  }
  // accname 2F/2G subset: skips hidden descendants, uses embedded control values, and spaces block boundaries
  function textAlt(root, includeHidden, exclude, skip) {
    let out = "";
    const rec = (n) => {
      if (out.length > 600) return;
      if (n.nodeType === 3) {
        out += n.data;
        return;
      }
      if (n.nodeType !== 1 || n === exclude || (skip && wsHas(skip, n)) || isOwn(n)) return;
      const t = n.localName;
      if (SKIP_TAGS[t]) return;
      if (n !== root) {
        if (!includeHidden && hiddenForName(n)) return;
        const al = collapse(attr(n, "aria-label"));
        let v = al;
        if (!al && (t === "img" || t === "area")) v = attr(n, "alt") || "";
        else if (!al && t === "input") v = /^(checkbox|radio|password|hidden)$/.test(inputType(n)) ? "" : N.inputGet(n);
        else if (!al && t === "select") v = selectedLabel(n);
        else if (!al && t === "textarea") v = N.textareaGet(n);
        if (al || /^(img|area|input|select|textarea)$/.test(t)) {
          out += " " + v + " ";
          return;
        }
      }
      if (t === "svg" || t === "br") {
        out += " " + (t === "svg" ? svgTitle(n) : "") + " ";
        return;
      }
      const block = !INLINE_TAGS[t];
      const before = out.length;
      if (block) out += " ";
      forEachChild(n, rec);
      // an empty icon element keeps its tooltip (accname step 2I), unless it was emptied by `skip`
      if (n !== root && !skip && !out.slice(before).trim()) out += " " + (attr(n, "title") || "") + " ";
      if (block) out += " ";
    };
    rec(root);
    return collapse(out);
  }
  function labelledBy(el) {
    const ids = attr(el, "aria-labelledby");
    if (!ids) return "";
    const root = N.getRootNode(el);
    const parts = [];
    for (const id of ids.trim().split(WS_SPLIT)) {
      const ref = byId(root, id) || (root !== el.ownerDocument ? byId(el.ownerDocument, id) : null);
      const s = ref ? collapse(attr(ref, "aria-label")) || textAlt(ref, true) : "";
      if (s) parts.push(s);
    }
    return collapse(parts.join(" "));
  }
  function labelText(el) {
    const labels = attempt(() => el.labels, null);
    if (!labels || !labels.length) return "";
    const parts = [];
    for (let i = 0; i < labels.length; i++) parts.push(textAlt(labels[i], false, el));
    return collapse(parts.join(" "));
  }
  // `from === "placeholder"` means the placeholder already served as the name and must not be printed again
  function nameInfo(el, role) {
    const info = (name, from) => ({ name, from: name ? from : "" });
    let s = labelledBy(el) || collapse(attr(el, "aria-label"));
    if (s) return info(s, "label");
    const t = el.localName;
    if (t === "input" || t === "textarea" || t === "select") {
      const ty = t === "input" ? inputType(el) : "";
      if (ty === "button" || ty === "submit" || ty === "reset") {
        return info(collapse(N.inputGet(el)) || { submit: "Submit", reset: "Reset" }[ty] || "", "value");
      }
      if (ty === "image") return info(collapse(attr(el, "alt")) || collapse(N.inputGet(el)) || "Submit", "alt");
      s = labelText(el) || collapse(attr(el, "title"));
      if (s) return info(s, "label");
      return info(collapse(attr(el, "placeholder")) || collapse(attr(el, "aria-placeholder")), "placeholder");
    }
    if (t === "img" || t === "area") {
      s = collapse(attr(el, "alt"));
      if (s) return info(s, "alt");
    } else if (t === "fieldset" || t === "table" || t === "figure") {
      const cap = el.querySelector({ fieldset: "legend", table: "caption", figure: "figcaption" }[t]);
      s = cap ? textAlt(cap) : "";
      if (s) return info(s, "content");
    } else if (t === "label") {
      return info(textAlt(el, false, el.control || undefined), "content");
    }
    if (!role || NAME_FROM_CONTENT[role]) {
      s = textAlt(el);
      if (s) return info(s, "content");
    }
    s = collapse(attr(el, "title"));
    if (s) return info(s, "title");
    return info(collapse(attr(el, "aria-placeholder")) || collapse(attr(el, "data-placeholder")), "placeholder");
  }
  const accName = (el, role) => nameInfo(el, role === undefined ? roleOf(el) : role).name;
  const ICON_WORDS = new RegExp(
    "(close|menu|hamburger|search|cart|basket|bag|account|profile|user|login|logout|sign-?in|delete|trash|remove|edit|" +
      "pencil|more|kebab|ellipsis|settings|gear|cog|next|prev|previous|back|forward|play|pause|share|like|heart|favorite|" +
      "star|filter|sort|plus|minus|add|expand|collapse|home|bell|notification|help|info|download|upload|copy|refresh|" +
      "reload|send|mic|camera|attach|emoji|calendar|chevron|arrow|zoom|fullscreen|mute|volume)",
    "i",
  );
  // nameless icon buttons are common; a class/id/icon hint helps a small model without posing as a real label
  function iconHint(el) {
    const kids = N.qsa(el, "svg, use, i, span, img");
    for (let i = -1; i < kids.length && i < 4; i++) {
      const n = i < 0 ? el : kids[i];
      const words = ["id", "class", "data-icon", "data-testid", "name", "href", "xlink:href"].map((a) => attr(n, a)).join(" ");
      const m = ICON_WORDS.exec(words);
      if (m) return m[1].toLowerCase();
    }
    return "";
  }

  const refOfEl = newWeakMap();
  const elOfRef = newMap();
  let refCounter = 0;
  function refFor(el) {
    let r = refOfEl.get(el);
    if (!r) {
      r = "e" + ++refCounter;
      refOfEl.set(el, r);
      elOfRef.set(r, WeakRefCtor ? new WeakRefCtor(el) : el);
    }
    return r;
  }
  function attached(el) {
    if (!el.isConnected) return false;
    let d = el.ownerDocument;
    for (let guard = 0; d !== doc && guard < 16; guard++) {
      const fe = frameElementOf(d);
      if (!fe || !fe.isConnected || frameDoc(fe) !== d) return false;
      d = fe.ownerDocument;
    }
    return d === doc;
  }
  function resolve(ref) {
    if (typeof ref !== "string" || !/^e\d+$/.test(ref.trim())) {
      throw fail("bad_args", `"${ref}" is not a valid ref; refs look like e12 and come from the latest snapshot.`);
    }
    ref = ref.trim();
    const held = elOfRef.get(ref);
    const el = held ? (deref ? deref(held) : held) : null;
    if (!el || !attached(el)) throw fail("stale_ref", `Element ${ref} is no longer on the page. Use the new snapshot.`);
    return el;
  }

  const controlOf = (el) => (el.localName === "label" && el.control ? el.control : el);
  function describe(el) {
    const target = controlOf(el);
    const role = roleOf(target);
    const name = accName(target, role) || (target !== el ? accName(el, "") : "");
    const label = role || (styleOf(target).cursor === "pointer" ? "clickable" : target.localName);
    return name ? `${label} ${quote(clip(name, 80))}` : label;
  }
  function isPassword(el) {
    if (el.localName !== "input") return false;
    const ty = inputType(el);
    return ty === "password" || (ty === "text" && /(^|\s)(current|new)-password(\s|$)/.test(attr(el, "autocomplete") || ""));
  }
  function isDisabled(el) {
    return attr(el, "aria-disabled") === "true" || attempt(() => N.matches(el, ":disabled"), !!el.disabled);
  }

  // two clip rects: hard (overflow hidden/clip; outside is invisible) and view (listing window; outside needs a scroll)
  const INF = 1e9;
  const intersect = (a, b) => ({ l: max(a.l, b.l), t: max(a.t, b.t), r: min(a.r, b.r), b: min(a.b, b.b) });
  function placeOf(r, hard, view) {
    if (r.w <= 1 || r.h <= 1 || r.r <= hard.l || r.l >= hard.r || r.b <= hard.t || r.t >= hard.b) return "hidden";
    return r.b <= view.t ? "above" : r.t >= view.b ? "below" : "in";
  }
  function controlHidden(c) {
    const cs = styleOf(c);
    if (cs.display === "none" || cs.visibility !== "visible" || cs.opacity === "0") return true;
    const r = N.rect(c);
    return r.width <= 1 || r.height <= 1;
  }
  // a styled checkbox's real input is often on top at opacity 0 and still takes the click; labelled ones use the label
  function ghostControl(el, tag, cs) {
    if (tag !== "input" || cs.pointerEvents === "none" || !/^(checkbox|radio|file)$/.test(inputType(el))) return false;
    if (el.labels && el.labels.length) return false;
    const r = N.rect(el);
    return r.width >= 8 && r.height >= 8;
  }
  function cheapHasName(el) {
    return !!(collapse(attr(el, "aria-label")) || collapse(el.textContent) || attr(el, "title") || N.qsa(el, "img[alt],svg").length);
  }

  function collect(mode) {
    const vw = win.innerWidth;
    const vh = win.innerHeight;
    const se = doc.scrollingElement || doc.documentElement;
    const all = mode === "find";
    const viewport = { l: 0, t: 0, r: vw, b: vh };
    const sx = win.scrollX;
    const sy = win.scrollY;
    const rootHard = { l: -sx, t: -sy, r: max(se ? se.scrollWidth : vw, vw) - sx, b: max(se ? se.scrollHeight : vh, vh) - sy };
    const window_ = { l: -INF, t: -0.5 * vh, r: INF, b: 1.5 * vh };
    const rootView = all ? { l: -INF, t: -INF, r: INF, b: INF } : mode === "marks" ? viewport : window_;
    const st = { items: [], dialogs: [], mode, all, active: deepActive(), wantText: mode !== "marks", labelIds: createObject(null) };
    const labelled = N.docQsa(doc, "[aria-labelledby]");
    for (let i = 0; i < labelled.length; i++) {
      for (const id of attr(labelled[i], "aria-labelledby").trim().split(WS_SPLIT)) st.labelIds[id] = 1;
    }
    const ctx = {
      group: null, hard: rootHard, view: rootView, absHard: rootHard, absView: rootView, fixedHard: viewport,
      fixedView: all ? viewport : intersect(viewport, rootView), ox: 0, oy: 0, owner: null, quiet: false, ctl: null,
      heading: null, sectioned: false, depth: 0,
    };
    if (doc.documentElement) visit(doc.documentElement, ctx, st, "auto");
    return st;
  }
  const newGroup = (ctx, el, label) => (ctx.depth >= 3 ? null : { label, el, parent: ctx.group, name: null });

  // prose joins the nearest block's run, a nested block ends it; text in controls and labels is quiet (not own words)
  function appendText(owner, data, quiet, st) {
    if (owner.dead) return;
    if (!owner.run) {
      if (!data.trim()) return;
      if (owner.place === undefined) {
        owner.rect = boxOf(N.rect(owner.el), owner.ox, owner.oy);
        owner.place = placeOf(owner.rect, owner.hard, owner.view);
      }
      if (owner.place !== "in") {
        owner.dead = true;
        return;
      }
      owner.run = { kind: "text", el: owner.el, parts: [], own: 0, group: owner.group, rect: owner.rect, ctl: owner.ctl, place: "in" };
      st.items.push(owner.run);
    }
    owner.run.parts.push({ t: data, own: !quiet });
    if (!quiet) owner.run.own += alnumCount(data);
  }

  function visit(el, ctx, st, parentCursor) {
    const tag = el.localName;
    if (SKIP_TAGS[tag] || isOwn(el) || N.hasAttr(el, "inert") || attr(el, "aria-hidden") === "true") return;
    const cs = styleOf(el);
    const display = cs.display;
    if (display === "none" || cs.contentVisibility === "hidden") return;
    if (cs.opacity === "0" && !ghostControl(el, tag, cs)) return;
    const shown = cs.visibility === "visible";
    const pos = cs.position;
    const hard = pos === "fixed" ? ctx.fixedHard : pos === "absolute" ? ctx.absHard : ctx.hard;
    const view = pos === "fixed" ? ctx.fixedView : pos === "absolute" ? ctx.absView : ctx.view;
    const role = roleOf(el, ctx.sectioned);
    const cursor = cs.cursor;
    const inline = display === "inline" || display === "contents";
    let rect = null;
    const getRect = () => {
      if (!rect) rect = boxOf(N.rect(display === "contents" && el.firstElementChild ? el.firstElementChild : el), ctx.ox, ctx.oy);
      return rect;
    };
    if (tag === "iframe" || tag === "frame") {
      if (shown) visitFrame(el, ctx, st, hard, view, getRect());
      return;
    }

    // 1 is a real control, 2 is clickable by cursor/onclick only and is dropped later if it wraps real controls
    let ctlKind = 0;
    let proxy = null;
    if (role && CTL_ROLES[role]) {
      ctlKind = role === "gridcell" ? +N.hasAttr(el, "tabindex") : role === "listbox" ? +(tag === "select") : 1;
    } else if (tag === "label") {
      const c = el.control;
      if (c && c.localName === "input" && controlHidden(c)) {
        ctlKind = 1;
        proxy = c;
      }
    } else if (tag !== "html" && tag !== "body" && !ctx.ctl) {
      if ((cursor === "pointer" && parentCursor !== "pointer") || N.hasAttr(el, "onclick")) ctlKind = 2;
    }
    let item = null;
    if (ctlKind && shown && (!N.checkVisibility || N.checkVisibility(el))) {
      const p = placeOf(getRect(), hard, view);
      if (p !== "hidden" && !(ctlKind === 2 && p !== "in" && !cheapHasName(el))) {
        const r = proxy ? roleOf(proxy) : ctlKind === 2 && !CTL_ROLES[role] ? role || "clickable" : role;
        item = {
          kind: "ctl", el, proxy, role: r, rect: getRect(), place: p, group: ctx.group, tentative: ctlKind === 2,
          idx: st.items.length, heading: ctx.heading,
        };
        st.items.push(item);
        if (ctx.heading) ctx.heading.ctls.push(item);
      }
    }
    let headingItem = null;
    if (role === "heading" && shown && !ctx.ctl && !item && st.wantText && placeOf(getRect(), hard, view) === "in") {
      headingItem = { kind: "heading", el, level: headingLevel(el), group: ctx.group, rect: getRect(), place: "in", ctls: [] };
      st.items.push(headingItem);
    }
    let group = null;
    if (LANDMARKS[role] && !ctx.ctl && (role !== "region" || N.hasAttr(el, "aria-label") || N.hasAttr(el, "aria-labelledby"))) {
      group = newGroup(ctx, el, LANDMARKS[role]);
      if (group && (role === "dialog" || role === "alertdialog")) st.dialogs.push(group);
    }

    const c = {
      group: group || ctx.group, hard, view, absHard: ctx.absHard, absView: ctx.absView, fixedHard: ctx.fixedHard,
      fixedView: ctx.fixedView, ox: ctx.ox, oy: ctx.oy, owner: ctx.owner, ctl: item ? el : ctx.ctl,
      quiet: ctx.quiet || !!item || (tag === "label" && !!el.control) || !!(el.id && st.labelIds[el.id] && !group),
      heading: headingItem || ctx.heading, sectioned: ctx.sectioned || !!SECTIONING_TAGS[tag], depth: group ? ctx.depth + 1 : ctx.depth,
    };
    const ovx = cs.overflowX;
    const ovy = cs.overflowY;
    if (!inline && tag !== "html" && tag !== "body" && (ovx !== "visible" || ovy !== "visible")) {
      const r = getRect();
      const xh = ovx === "hidden" || ovx === "clip";
      const yh = ovy === "hidden" || ovy === "clip";
      const xs = !xh && ovx !== "visible";
      const ys = !yh && ovy !== "visible";
      // a scroller's content is reachable across its whole scroll range, even past what the document can scroll to
      const sl = r.l - el.scrollLeft;
      const stp = r.t - el.scrollTop;
      c.hard = {
        l: xh ? max(hard.l, r.l) : xs ? sl : hard.l,
        r: xh ? min(hard.r, r.r) : xs ? sl + el.scrollWidth : hard.r,
        t: yh ? max(hard.t, r.t) : ys ? stp : hard.t,
        b: yh ? min(hard.b, r.b) : ys ? stp + el.scrollHeight : hard.b,
      };
      const grow = ys && st.mode === "snapshot" ? 0.5 * r.h : 0;
      if (yh || ys) c.view = { l: view.l, r: view.r, t: max(view.t, r.t - grow), b: min(view.b, r.b + grow) };
    }
    if (pos !== "static" || cs.transform !== "none") {
      c.absHard = c.hard;
      c.absView = c.view;
    }
    if (!inline) {
      if (ctx.owner) ctx.owner.run = null;
      c.owner = {
        el, run: null, place: undefined, rect: null, hard: c.hard, view: c.view, ox: c.ox, oy: c.oy, group: c.group,
        ctl: c.ctl, dead: !st.wantText || (!st.all && (c.quiet || !!c.heading)),
      };
    }
    // a closed <details> hides all but its summary via a UA content-visibility slot, leaving styles and rects normal
    const closedDetails = tag === "details" && !el.open;
    if (!/^(select|textarea|input|svg|canvas)$/.test(tag)) {
      forEachChild(el, (n) => {
        if (closedDetails && n.localName !== "summary") return;
        if (n.nodeType === 1) visit(n, c, st, cursor);
        else if (n.nodeType === 3 && shown && c.owner) appendText(c.owner, n.data, c.quiet, st);
      });
    }
    if (!inline && ctx.owner) ctx.owner.run = null;
    if (item && item.tentative) {
      for (let i = item.idx + 1; i < st.items.length && !item.dead; i++) if (st.items[i].kind === "ctl") item.dead = true;
    }
  }

  function visitFrame(el, ctx, st, hard, view, r) {
    const p = placeOf(r, hard, view);
    if (p === "hidden") return;
    const name =
      collapse(attr(el, "title")) || collapse(attr(el, "aria-label")) || collapse(attr(el, "name")) ||
      attempt(() => new URLCtor(el.src).host, "");
    const d = frameDoc(el);
    if (!d || !d.documentElement) {
      if (p === "in") st.items.push({ kind: "frame", el, name, group: ctx.group, rect: r, place: p });
      return;
    }
    const cs = styleOf(el);
    const ox = r.l + el.clientLeft + (parseNum(cs.paddingLeft) || 0);
    const oy = r.t + el.clientTop + (parseNum(cs.paddingTop) || 0);
    const box = { l: ox, t: oy, r: ox + el.clientWidth, b: oy + el.clientHeight };
    const grow = st.mode === "snapshot" ? 0.5 * el.clientHeight : 0;
    const inner = intersect(view, { l: -INF, t: box.t - grow, r: INF, b: box.b + grow });
    const group = newGroup(ctx, el, "iframe");
    if (group) group.name = name;
    if (ctx.owner) ctx.owner.run = null;
    const fh = intersect(hard, box);
    visit(d.documentElement, {
      group: group || ctx.group, hard: fh, view: inner, absHard: fh, absView: inner, fixedHard: intersect(ctx.fixedHard, box),
      fixedView: intersect(ctx.fixedView, box), ox, oy, owner: null, quiet: false, ctl: null, heading: null,
      sectioned: false, depth: group ? ctx.depth + 1 : ctx.depth,
    }, st, "auto");
  }

  const isStrictModal = (el) => attempt(() => N.matches(el, ":modal"), false);
  const isModalDialog = (el) => isStrictModal(el) || attr(el, "aria-modal") === "true";
  function inGroup(item, g) {
    for (let x = item.group; x; x = x.parent) if (x === g) return true;
    return false;
  }
  const inViewport = (r) => r.b > 0 && r.t < win.innerHeight && r.r > 0 && r.l < win.innerWidth;
  // elementFromPoint stops at a shadow host, so step into open shadow roots for the real target
  function deepHit(d, x, y) {
    let hit = N.elementFromPoint(d, x, y);
    for (let i = 0; hit && i < 32; i++) {
      const root = N.shadowRoot(hit);
      const inner = root && N.rootFromPoint ? N.rootFromPoint(root, x, y) : null;
      if (!inner || inner === hit) break;
      hit = inner;
    }
    return hit;
  }
  function covered(item) {
    const el = item.el;
    const r = item.rect;
    const l = max(r.l, 0);
    const t = max(r.t, 0);
    const rr = min(r.r, win.innerWidth);
    const b = min(r.b, win.innerHeight);
    if (rr <= l || b <= t) return false;
    const off = el.ownerDocument === doc ? { x: 0, y: 0 } : frameOffset(el.ownerDocument);
    const hit = deepHit(el.ownerDocument, (l + rr) / 2 - off.x, (t + b) / 2 - off.y);
    return !!hit && !containsComposed(el, hit) && !(item.proxy && containsComposed(item.proxy, hit));
  }
  // a declared modal, or a dialog over most of the page, goes first and covered elements drop (clicks hit the overlay)
  function pickModal(st) {
    for (let i = st.dialogs.length - 1; i >= 0; i--) {
      const g = st.dialogs[i];
      const page = [];
      let hasItems = false;
      for (const it of st.items) {
        if (it.kind !== "ctl" || it.place === "hidden") continue;
        if (inGroup(it, g)) hasItems = true;
        else if (!it.dead && it.place === "in" && inViewport(it.rect)) page.push(it);
      }
      if (!hasItems) continue;
      const declared = isModalDialog(g.el);
      const step = max(1, ceil(page.length / 24));
      let sampled = 0;
      let hits = 0;
      for (let j = 0; j < page.length; j += step, sampled++) if (covered(page[j])) hits++;
      const blocked = isStrictModal(g.el) || (sampled ? hits / sampled >= 0.6 : declared);
      if (declared || blocked) return { group: g, blocked };
    }
    return null;
  }

  let activeEl = null;
  function stateText(it, nameFrom) {
    const el = it.proxy || it.el;
    const t = el.localName;
    const ty = t === "input" ? inputType(el) : "";
    const role = it.role;
    const out = [];
    const value = (v) => v && out.push("value=" + quote(clip(collapse(v), 60)));
    if ((ty === "checkbox" || ty === "radio") && (el.indeterminate || el.checked)) out.push(el.indeterminate ? "mixed" : "checked");
    const ac = attr(el, "aria-checked");
    if (ac === "true" || ac === "mixed") out.push(ac === "true" ? "checked" : "mixed");
    if (attr(el, "aria-pressed") === "true") out.push("pressed");
    if (attr(el, "aria-selected") === "true") out.push("selected");
    let ex = attr(el, "aria-expanded");
    if (ex === null && t === "summary" && el.parentElement && el.parentElement.localName === "details") ex = Str(el.parentElement.open);
    if (ex === "true" || ex === "false") out.push(ex === "true" ? "expanded" : "collapsed");
    if (isDisabled(el)) out.push("disabled");
    if (el.required || attr(el, "aria-required") === "true") out.push("required");
    if (((t === "input" || t === "textarea") && el.readOnly) || attr(el, "aria-readonly") === "true") out.push("readonly");
    if (isPassword(el)) {
      out.push(N.inputGet(el) ? "(password, filled)" : "(password)");
    } else if (isCardField(el)) {
      out.push(valueOf(el) ? "(card details, filled)" : "(card details)");
    } else if (ty === "file") {
      if (el.files && el.files.length) out.push(`files=${el.files.length}`);
    } else if (ty && !/^(checkbox|radio|button|submit|reset|image)$/.test(ty)) {
      value(N.inputGet(el));
    } else if (t === "textarea") {
      out.push("multiline");
      value(N.textareaGet(el));
    } else if (t === "select") {
      const opts = el.options;
      const picked = [];
      for (let i = 0; i < opts.length; i++) {
        if (opts[i].selected && (el.multiple || i === el.selectedIndex)) picked.push(optionLabel(opts[i]));
      }
      value(picked.join(", "));
      let list = "";
      if (el === activeEl || ex === "true") {
        const shown = [];
        for (let i = 0; i < opts.length && i < 8; i++) shown.push(quote(clip(optionLabel(opts[i]), 30)));
        list = ": " + shown.join(", ") + (opts.length > 8 ? ", …" : "");
      }
      out.push(`options=${opts.length}${list}`);
    } else if (role === "slider" || role === "spinbutton" || role === "progressbar") {
      value(attr(el, "aria-valuetext") || attr(el, "aria-valuenow"));
    } else if ((role === "textbox" || role === "searchbox") && isEditingHost(el)) {
      value(el.textContent);
    } else if (role === "combobox" && nameFrom !== "content") {
      value(el.textContent);
    }
    if (nameFrom !== "placeholder" && !((t === "input" || t === "textarea") && valueOf(el))) {
      const ph = collapse(attr(el, "placeholder") || attr(el, "aria-placeholder"));
      if (ph) out.push("placeholder=" + quote(clip(ph, 60)));
    }
    if (el === activeEl && t !== "body") out.push("focused");
    return out.length ? " " + out.join(" ") : "";
  }
  function valueOf(el) {
    return el.localName === "input" ? N.inputGet(el) : el.localName === "textarea" ? N.textareaGet(el) : "";
  }
  function ctlLine(it) {
    const el = it.proxy || it.el;
    let info = nameInfo(el, it.role === "clickable" ? "" : it.role);
    if (!info.name && it.proxy) info = { name: accName(it.el, ""), from: "content" };
    let s = it.role || el.localName;
    if (info.name) s += " " + quote(clip(info.name, 80));
    else if (!it.tentative) {
      const field = el.localName === "input" || el.localName === "textarea" ? attr(el, "name") || attr(el, "id") : "";
      const href = !field && it.role === "link" && el.localName === "a" ? absUrl(el) : "";
      const hint = !field && !href ? iconHint(el) : "";
      if (field) s += ` (field: ${clip(field, 30)})`;
      else if (href) s += ` (url: ${clip(href.replace(/^https?:\/\/(www\.)?/, ""), 60)})`;
      else if (hint) s += ` (icon: ${hint})`;
    }
    it.name = info.name;
    return s + stateText(it, info.from);
  }
  // a run opening with a link (title-link cards, feed items) drops it; a mostly-link run keeps only its own words
  function runText(run) {
    if (run.own < 2) return "";
    const parts = run.parts;
    let start = 0;
    while (start < parts.length && (!parts[start].own || !parts[start].t.trim())) start++;
    let full = "";
    for (let i = start; i < parts.length; i++) full += parts[i].t;
    full = collapse(full);
    if (run.own * 3 >= alnumCount(full)) return full;
    const own = [];
    for (let i = start; i < parts.length; i++) if (parts[i].own && alnumCount(parts[i].t) >= 2) own.push(parts[i].t);
    return collapse(own.join(" "));
  }
  function groupHeader(g, modal) {
    if (g.name === null) g.name = labelledBy(g.el) || collapse(attr(g.el, "aria-label")) || collapse(attr(g.el, "title"));
    const name = g.name ? " " + quote(clip(g.name, 60)) : "";
    return modal ? `Modal dialog${name}${modal.blocked ? " (page behind it is blocked)" : ""}:` : `${g.label}${name}:`;
  }
  function scrollInfo() {
    const se = doc.scrollingElement || doc.documentElement;
    const vh = win.innerHeight;
    const H = max(se ? se.scrollHeight : vh, vh);
    const y = round(win.scrollY);
    const tail = (below) => (below > 0.05 ? below.toFixed(1) + " screens below" : "at bottom");
    if (H > vh + 2 && windowScrolls("y")) return `Scroll: ${y}–${y + vh} of ${H}px (${tail((H - y - vh) / vh)})`;
    const box = dominantScroller("y");
    if (!box) return "Scroll: whole page visible";
    const top = round(box.scrollTop);
    const h = box.clientHeight;
    const rest = tail((box.scrollHeight - top - h) / max(1, h));
    return `Scroll: page fits; main scroll area ${top}–${top + h} of ${box.scrollHeight}px (${rest})`;
  }

  // a per-row repeated control gets the text of its largest ancestor with no other copy, minus action buttons
  function rowContext(lines) {
    const groups = newMap();
    const keys = [];
    for (const ln of lines) {
      if (ln.kind !== "ctl") continue;
      const key = (ln.e.it.role || "") + "\u0000" + (ln.e.it.name || ln.body);
      const g = groups.get(key);
      if (g) g.push(ln);
      else {
        groups.set(key, [ln]);
        keys.push(key);
      }
    }
    const actions = new WeakSetCtor();
    for (const ln of lines) {
      if (ln.kind === "ctl" && ln.e.it.role !== "link") wsAdd(actions, ln.e.it.el);
    }
    for (const key of keys) {
      const dupes = groups.get(key);
      if (dupes.length > 1) for (const ln of dupes) wsAdd(actions, ln.e.it.el);
    }
    for (const key of keys) {
      const dupes = groups.get(key);
      if (dupes.length < 2) continue;
      const holds = newMap();
      for (const ln of dupes) {
        for (let a = parentOf(ln.e.it.el); a && a.nodeType === 1; a = parentOf(a)) holds.set(a, (holds.get(a) || 0) + 1);
      }
      for (const ln of dupes) {
        const el = ln.e.it.el;
        let row = null;
        for (let a = parentOf(el); a && a.nodeType === 1 && holds.get(a) === 1; a = parentOf(a)) row = a;
        const text = row ? clip(collapse(textAlt(row, false, el, actions).replace(/\(\s*\)|\[\s*\]/g, "")), 40) : "";
        if (text && text !== ln.e.it.name) ln.body += ` (in ${quote(text)})`;
      }
    }
  }

  let snapshots = 0;
  let prevSeen = new WeakSetCtor();
  let retakeBase = { count: 0, seen: prevSeen };
  function buildSnapshot(maxChars, retake) {
    // a smaller retake of the snapshot just discarded marks new elements against the one before it
    const base = retake ? retakeBase : { count: snapshots, seen: prevSeen };
    const st = collect("snapshot");
    activeEl = st.active;
    const modal = pickModal(st);
    const seen = new WeakSetCtor();
    const entries = [];
    let above = 0;
    let below = 0;
    let behind = 0;
    for (const it of st.items) {
      if (it.dead) continue;
      if (it.kind === "ctl") wsAdd(seen, it.el);
      const inModal = !!modal && inGroup(it, modal.group);
      if (modal && !inModal && (modal.blocked || (it.place === "in" && covered(it)))) {
        if (modal.blocked && it.kind === "ctl") behind++;
        continue;
      }
      if (it.place === "above" || it.place === "below") {
        if (it.kind === "ctl") it.place === "above" ? above++ : below++;
      } else if (it.place === "in") entries.push({ it, modal: inModal });
    }

    const lines = [];
    const isNew = base.count > 0;
    for (const e of entries) {
      const it = e.it;
      if (it.kind === "ctl") {
        const body = ctlLine(it);
        // citation markers like [12] are noise in a page overview
        if ((it.tentative && !it.name) || (it.role === "link" && CITATION.test(it.name))) continue;
        lines.push({ e, kind: "ctl", star: isNew && !wsHas(base.seen, it.el) ? "*" : "", body });
      } else if (it.kind === "heading") {
        const name = textAlt(it.el);
        if (!name) continue;
        const live = it.ctls.filter((x) => !x.dead);
        if (live.length === 1 && accName(live[0].proxy || live[0].el, live[0].role).toLowerCase() === name.toLowerCase()) {
          it.merged = live[0];
          continue;
        }
        lines.push({ e, kind: "heading", text: `heading ${quote(clip(name, 100))} (h${it.level})` });
      } else if (it.kind === "text") {
        const tx = runText(it);
        if (!tx) continue;
        const prev = lines[lines.length - 1];
        // sibling cells and spans (table rows, flex rows) read better as one line
        if (prev && prev.kind === "text" && prev.e.it.group === it.group && prev.raw.length + tx.length + 3 <= 100) {
          const pe = prev.e.it.el;
          if (pe === it.el || parentOf(pe) === parentOf(it.el)) {
            prev.raw += " · " + tx;
            prev.text = `text ${quote(prev.raw)}`;
            continue;
          }
        }
        const raw = clip(tx, 100);
        lines.push({ e, kind: "text", raw, text: `text ${quote(raw)}` });
      } else if (it.kind === "frame") {
        lines.push({ e, kind: "frame", text: `iframe ${it.name ? quote(clip(it.name, 60)) + " " : ""}(cross-origin, not readable)` });
      }
    }
    rowContext(lines);
    // refs go only to lines that survive the cap; until then the widest ref keeps every length estimate an upper bound
    let fresh = 0;
    for (const ln of lines) {
      if (ln.kind !== "ctl") continue;
      const it = ln.e.it;
      if (it.heading && it.heading.merged === it) ln.body += ` (h${it.heading.level})`;
      if (!refOfEl.get(it.el)) fresh++;
    }
    const widest = "e" + (refCounter + fresh);
    const ctlText = (ln, ref) => (ln.text = `${ln.star}[${ref}] ${ln.body}`);
    for (const ln of lines) if (ln.kind === "ctl") ctlText(ln, refOfEl.get(ln.e.it.el) || widest);

    const head = [`Page: ${collapse(doc.title) || "(untitled)"}`, `URL: ${loc.href}`, scrollInfo()];
    const keep = lines.map(() => true);
    let trimmed = 0;
    const render = () => {
      const out = head.slice();
      if (above && !modal) out.push(`(${plural(above, "more interactive element")} above; scroll up or use browser_find)`);
      const printed = new WeakSetCtor();
      const emit = (ln) => {
        const chain = [];
        for (let g = ln.e.it.group; g && g !== (ln.e.modal ? modal.group : null); g = g.parent) chain.unshift(g);
        let depth = 0;
        if (ln.e.modal) {
          if (!wsHas(printed, modal.group)) out.push(groupHeader(modal.group, modal));
          wsAdd(printed, modal.group);
          depth = 1;
        }
        for (const g of chain) {
          if (!wsHas(printed, g)) out.push("  ".repeat(depth) + groupHeader(g, null));
          wsAdd(printed, g);
          depth++;
        }
        out.push("  ".repeat(depth) + ln.text);
      };
      lines.forEach((ln, i) => keep[i] && ln.e.modal && emit(ln));
      if (modal && modal.blocked && behind) {
        out.push(`(the page behind the dialog has ${plural(behind, "more interactive element")}; dismiss the dialog to reach them)`);
      }
      lines.forEach((ln, i) => keep[i] && !ln.e.modal && emit(ln));
      const more = below + trimmed;
      if (more && !(modal && modal.blocked)) out.push(`(${plural(more, "more interactive element")} below; scroll or use browser_find)`);
      return out.join("\n");
    };
    let text = render();
    if (text.length > maxChars) {
      // trim prose first, farthest from the viewport first, then whole lines from the bottom, page before modal
      const vh = win.innerHeight;
      const dist = (i) => {
        const r = lines[i].e.it.rect;
        return r.b < 0 ? -r.b : r.t > vh ? r.t - vh : 0;
      };
      const prose = [];
      lines.forEach((ln, i) => ln.kind === "text" && prose.push(i));
      prose.sort((a, b) => dist(b) - dist(a) || b - a);
      let over = text.length - maxChars;
      for (let k = 0; k < prose.length && over > 0; k++) {
        keep[prose[k]] = false;
        over -= lines[prose[k]].text.length + 1;
      }
      text = render();
      const order = [];
      for (let i = lines.length - 1; i >= 0; i--) if (!lines[i].e.modal) order.push(i);
      for (let i = lines.length - 1; i >= 0; i--) if (lines[i].e.modal) order.push(i);
      for (let k = 0; text.length > maxChars && k < order.length; ) {
        for (let need = text.length - maxChars + 60; need > 0 && k < order.length; ) {
          const i = order[k++];
          if (!keep[i]) continue;
          keep[i] = false;
          if (lines[i].kind === "ctl") trimmed++;
          need -= lines[i].text.length + 3;
        }
        text = render();
      }
    }
    let refCount = 0;
    lines.forEach((ln, i) => {
      if (keep[i] && ln.kind === "ctl") {
        ctlText(ln, refFor(ln.e.it.el));
        refCount++;
      }
    });
    text = render();
    if (text.length > maxChars) text = text.slice(0, unsplit(text, maxChars - 1)) + "…";
    retakeBase = base;
    snapshots = base.count + 1;
    prevSeen = seen;
    activeEl = null;
    return { text, url: loc.href, title: doc.title, refCount, docId: DOC_ID };
  }

  function find(args) {
    const query = collapse(args.query).toLowerCase();
    if (!query) throw fail("bad_args", "find needs a non-empty query.");
    const limit = clampInt(args.limit, 20, 1, 200);
    const tokens = query.split(" ");
    const hit = (s) => {
      const h = s.toLowerCase();
      return h.indexOf(query) >= 0 || (tokens.length > 1 && tokens.every((t) => h.indexOf(t) >= 0));
    };
    const st = collect("find");
    activeEl = st.active;
    const matches = [];
    const matchedCtl = new WeakSetCtor();
    for (const it of st.items) {
      if (it.dead) continue;
      if (it.kind === "ctl") {
        const el = it.proxy || it.el;
        // cheap prefilter so names are computed only for plausible matches
        const quick = [el.textContent, valueOf(el)].concat(["aria-label", "title", "placeholder", "alt"].map((a) => attr(el, a))).join(" ");
        const formish = /^(input|select|textarea)$/.test(el.localName) || N.hasAttr(el, "aria-labelledby");
        if (!formish && !hit(quick) && !N.qsa(el, "img[alt],svg").length) continue;
        const line = ctlLine(it);
        if ((it.tentative && !it.name) || !hit(it.name + " " + line)) continue;
        wsAdd(matchedCtl, it.el);
        matches.push({ ctl: true, line: `[${refFor(it.el)}] ${line}` });
      } else if (it.kind === "heading") {
        const name = textAlt(it.el);
        if (name && hit(name)) matches.push({ ctl: false, line: `heading ${quote(clip(name, 100))} (h${it.level})` });
      } else if (it.kind === "text" && (it.own >= 2 || it.ctl)) {
        let full = "";
        for (const p of it.parts) full += p.t;
        full = collapse(full);
        if (!full || !hit(full) || (it.ctl && wsHas(matchedCtl, it.ctl))) continue;
        let at = full.toLowerCase().indexOf(query);
        if (at < 0) at = full.toLowerCase().indexOf(tokens[0]);
        const from = unsplit(full, max(0, at - 40));
        const to = unsplit(full, min(full.length, at + query.length + 40));
        let line = `text: ${quote((from > 0 ? "…" : "") + full.slice(from, to).trim() + (to < full.length ? "…" : ""))}`;
        if (it.ctl && it.ctl.isConnected) line += ` (inside [${refFor(it.ctl)}] ${describe(it.ctl)})`;
        matches.push({ ctl: false, line });
      }
    }
    activeEl = null;
    if (!matches.length) return { text: `No matches for ${quote(collapse(args.query))} on this page.`, count: 0, docId: DOC_ID };
    // interactive matches win the limited slots, and output stays in page order
    let room = limit;
    for (const m of matches) if (m.ctl && room > 0) (m.keep = true), room--;
    for (const m of matches) if (!m.ctl && room > 0) (m.keep = true), room--;
    const out = matches.filter((m) => m.keep).map((m) => m.line);
    if (matches.length > out.length) out.push(`(${matches.length - out.length} more matches; refine the query)`);
    return { text: out.join("\n"), count: matches.length, docId: DOC_ID };
  }

  const PURCHASE_RE = /\b(?:buy|purchase|pay(?:ment)?|order|place (?:your |my |the )?order|check ?out|subscribe|donate)\b/i;
  const NOT_PURCHASE_RE = /\border (?:history|status|details|tracking|summary)\b/i;
  const DELETE_RE = /\b(?:delete|remove|cancel (?:my |your |the )?subscription|close (?:my |your |the )?account|deactivate|erase)\b/i;
  const SEND_RE = /\b(?:send|post|publish|tweet|reply|comment|submit (?:message|reply|comment|review|post))\b/i;
  const CARD_FIELD_RE = /card.?num|cc.?num|credit.?card|cvv|cvc|csc|security.?code|card.?exp|exp(?:iry|iration).?date/i;
  function isCardField(el) {
    if (el.localName !== "input" && el.localName !== "select") return false;
    if (/(^|\s)cc-/.test((attr(el, "autocomplete") || "").toLowerCase())) return true;
    return CARD_FIELD_RE.test(`${attr(el, "name") || ""} ${attr(el, "id") || ""} ${attr(el, "data-testid") || ""}`);
  }
  // the form, or for form-less SPA logins the nearest ancestor holding several inputs
  function formScope(el) {
    const f = el.form || N.closest(el, "form");
    if (f) return f;
    let p = el.parentElement;
    for (let i = 0; p && i < 6; i++, p = p.parentElement) if (N.qsa(p, "input").length > 1) return p;
    return null;
  }
  function isActivator(el, role) {
    const t = el.localName;
    if (t === "button") return true;
    if (t === "input") return /^(submit|button|image)$/.test(inputType(el));
    return /^(button|link|menuitem|clickable)$/.test(role) || (t === "a" && N.hasAttr(el, "href"));
  }
  function sensitivity(el) {
    el = controlOf(el);
    if (isPassword(el)) return "password";
    const role = roleOf(el) || (styleOf(el).cursor === "pointer" ? "clickable" : "");
    const activator = isActivator(el, role);
    const textual = el.localName === "input" ? !!TEXT_INPUTS[inputType(el)] : el.localName === "textarea";
    if (activator || textual) {
      const scope = formScope(el);
      const inputs = scope ? N.qsa(scope, "input") : [];
      let filled = false;
      let card = false;
      for (let i = 0; i < inputs.length; i++) {
        if (isPassword(inputs[i]) && N.inputGet(inputs[i])) filled = true;
        if (isCardField(inputs[i])) card = true;
      }
      if (filled) return "credentials";
      if (card && scope.localName === "form") return "payment";
    }
    if (isCardField(el)) return "payment";
    if (!activator) return null;
    const name = accName(el, role === "clickable" ? "" : role);
    if (PURCHASE_RE.test(name) && !NOT_PURCHASE_RE.test(name)) return "purchase";
    if (DELETE_RE.test(name)) return "delete";
    return SEND_RE.test(name) ? "send" : null;
  }

  let overlay = null;
  const OVERLAY_CSS =
    ".m{position:fixed;box-sizing:border-box;border:2px solid var(--c);border-radius:3px}" +
    ".t{position:fixed;font:700 11px/14px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#fff;" +
    "background:var(--c);padding:0 3px;border-radius:3px;white-space:nowrap}" +
    ".h{position:fixed;box-sizing:border-box;border:3px solid #f59e0b;border-radius:4px;box-shadow:0 0 0 3px rgba(245,158,11,.35)}";
  const HOST_CSS =
    "all:initial!important;position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important;" +
    "margin:0!important;padding:0!important;border:0!important;background:transparent!important;overflow:visible!important;" +
    "pointer-events:none!important;z-index:2147483647!important;display:block!important;" +
    "max-width:none!important;max-height:none!important";
  const MARK_COLORS = ["#e11d48", "#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2"];
  function ensureOverlay() {
    if (overlay && overlay.host.isConnected) return overlay;
    const host = N.createElement(doc, "unsloth-agent-overlay");
    wsAdd(ownNodes, host);
    N.setAttr(host, "style", HOST_CSS);
    // the top layer keeps the marks above open modal <dialog>s
    if (N.showPopover) N.setAttr(host, "popover", "manual");
    const root = N.attachShadow(host, { mode: "closed" });
    const style = N.createElement(doc, "style");
    style.textContent = OVERLAY_CSS;
    const marks = N.createElement(doc, "div");
    const flashes = N.createElement(doc, "div");
    for (const n of [style, marks, flashes]) N.append(root, n);
    N.append(doc.documentElement, host);
    if (N.showPopover) attempt(() => N.showPopover(host));
    overlay = { host, marks, flashes, flashCount: 0 };
    return overlay;
  }
  function dropOverlayIfIdle() {
    if (!overlay || overlay.marks.firstChild || overlay.flashCount > 0) return;
    N.remove(overlay.host);
    overlay = null;
  }
  function box(cls, r, color) {
    const d = N.createElement(doc, "div");
    d.className = cls;
    d.style.cssText = `left:${r.l}px;top:${r.t}px;width:${max(0, r.r - r.l)}px;height:${max(0, r.b - r.t)}px;--c:${color}`;
    return d;
  }
  function flash(el, ms) {
    const r = topRect(el);
    if (r.w <= 0 && r.h <= 0) return;
    const ov = ensureOverlay();
    const d = box("h", { l: r.l - 3, t: r.t - 3, r: r.r + 3, b: r.b + 3 });
    N.append(ov.flashes, d);
    ov.flashCount++;
    N.setTimeout(win, () => {
      N.remove(d);
      if (overlay !== ov) return;
      ov.flashCount--;
      dropOverlayIfIdle();
    }, clampInt(ms, 700, 50, 10000));
  }
  function marks(args) {
    if (!args.on) {
      if (overlay) overlay.marks.textContent = "";
      dropOverlayIfIdle();
      return { count: 0 };
    }
    const st = collect("marks");
    const modal = pickModal(st);
    const ov = ensureOverlay();
    ov.marks.textContent = "";
    const vw = win.innerWidth;
    const vh = win.innerHeight;
    let count = 0;
    for (const it of st.items) {
      if (it.kind !== "ctl" || it.dead || it.place !== "in") continue;
      if (modal && !inGroup(it, modal.group) && (modal.blocked || covered(it))) continue;
      if (it.tentative && !accName(it.el, "")) continue;
      const r = it.rect;
      const vis = { l: max(r.l, 0), t: max(r.t, 0), r: min(r.r, vw), b: min(r.b, vh) };
      if (vis.r <= vis.l || vis.b <= vis.t) continue;
      const color = MARK_COLORS[count++ % MARK_COLORS.length];
      const tag = box("t", { l: min(vis.l, vw - 40), t: vis.t >= 15 ? vis.t - 15 : vis.t, r: 0, b: 0 }, color);
      tag.style.width = tag.style.height = "";
      tag.textContent = refFor(it.el);
      N.append(ov.marks, box("m", vis, color));
      N.append(ov.marks, tag);
    }
    if (!count) dropOverlayIfIdle();
    return { count, docId: DOC_ID };
  }

  function eventInit(view, extra) {
    const init = { bubbles: true, cancelable: true, composed: true, view };
    for (const k in extra) init[k] = extra[k];
    return init;
  }
  function construct(Ctor, type, init) {
    try {
      return new Ctor(type, init);
    } catch (_) {
      // some engines reject a cross-realm `view`, and the event works without it
      init.view = null;
      return new Ctor(type, init);
    }
  }
  const fire = (target, Ctor, type, init) => N.dispatch(target, construct(Ctor, type, init));
  function focusable(el) {
    const t = el.localName;
    if (el.tabIndex >= 0 || N.hasAttr(el, "tabindex")) return true;
    if (t === "a" || t === "area") return N.hasAttr(el, "href");
    return /^(button|input|select|textarea|summary|iframe)$/.test(t) || isEditingHost(el);
  }
  function focusEl(el) {
    attempt(() => {
      if (el instanceof viewOf(el).HTMLElement) N.focus(el, { preventScroll: true });
      else if (typeof el.focus === "function") el.focus({ preventScroll: true });
    });
  }
  // what a real mousedown does to focus: the nearest focusable ancestor gets it, or the active element loses it
  function focusFor(target) {
    for (let x = target; x; x = parentOf(x)) {
      if (x.nodeType === 1 && focusable(x) && !x.disabled) {
        if (deepActive() !== x) focusEl(x);
        return;
      }
    }
    const a = deepActive();
    if (a && a !== a.ownerDocument.body && typeof a.blur === "function") a.blur();
  }
  function bringIntoView(el) {
    const r = topRect(el);
    if (r.t >= 0 && r.l >= 0 && r.b <= win.innerHeight && r.r <= win.innerWidth) return;
    try {
      N.scrollIntoView(el, { block: "center", inline: "nearest", behavior: "instant" });
    } catch (_) {
      N.scrollIntoView(el, { block: "center", inline: "nearest" });
    }
  }
  function centerOf(el) {
    const r = N.rect(el);
    const v = viewOf(el);
    const l = max(r.left, 0);
    const t = max(r.top, 0);
    const rr = min(r.right, v.innerWidth);
    const b = min(r.bottom, v.innerHeight);
    return rr > l && b > t ? { x: (l + rr) / 2, y: (t + b) / 2 } : { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }
  // returns what a real click at the element's center would have hit instead, if anything
  function mouseSequence(el) {
    const view = viewOf(el);
    const c = centerOf(el);
    const hit = deepHit(el.ownerDocument, c.x, c.y);
    const target = hit && containsComposed(el, hit) ? hit : el;
    const base = { clientX: c.x, clientY: c.y, screenX: c.x + (view.screenX || 0), screenY: c.y + (view.screenY || 0), button: 0 };
    const send = (Ctor, type, buttons, bubbles, extra) => {
      const init = eventInit(view, base);
      init.buttons = buttons;
      for (const k in extra) init[k] = extra[k];
      if (!bubbles) init.bubbles = init.cancelable = false;
      return fire(target, Ctor, type, init);
    };
    const pointer = { pointerId: 1, pointerType: "mouse", isPrimary: true, width: 1, height: 1 };
    const ptr = (type, buttons, bubbles) => send(Ev.Pointer, type, buttons, bubbles, { ...pointer, pressure: buttons ? 0.5 : 0 });
    const mouse = (type, buttons, bubbles, detail) => send(Ev.Mouse, type, buttons, bubbles, { detail });
    ptr("pointerover", 0, true);
    ptr("pointerenter", 0, false);
    mouse("mouseover", 0, true, 0);
    mouse("mouseenter", 0, false, 0);
    ptr("pointermove", 0, true);
    mouse("mousemove", 0, true, 0);
    // a cancelled pointerdown suppresses the compatibility mouse events but never the click, matching real input
    const pd = ptr("pointerdown", 1, true);
    const md = pd ? mouse("mousedown", 1, true, 1) : true;
    if (md) focusFor(target);
    ptr("pointerup", 0, true);
    if (pd) mouse("mouseup", 0, true, 1);
    mouse("click", 0, true, 1);
    const viaLabel = el.localName === "label" && el.control && hit && containsComposed(el.control, hit);
    return hit && !containsComposed(el, hit) && !viaLabel ? hit : null;
  }
  function ensureActionable(el, ref) {
    if (!N.rects(el).length && styleOf(el).display !== "contents") {
      throw fail("not_visible", `Element ${ref} is not visible right now (hidden or collapsed). Take a new snapshot.`);
    }
  }
  function click(args) {
    const el = resolve(args.ref);
    const t = controlOf(el).localName;
    if (t === "select" || t === "option") {
      throw fail("use_select", `${args.ref} is a dropdown (<select>); use browser_select with the option's label instead of clicking it.`);
    }
    ensureActionable(el, args.ref);
    if (isDisabled(el)) throw fail("disabled", `${args.ref} (${describe(el)}) is disabled; it cannot be clicked right now.`);
    const description = describe(el);
    bringIntoView(el);
    flash(el, 700);
    const occluder = mouseSequence(el);
    return occluder ? { description, occludedBy: describe(occluder) } : { description };
  }

  const KEY_CODES = {
    Enter: 13, Escape: 27, Tab: 9, Backspace: 8, Delete: 46, ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40,
    PageUp: 33, PageDown: 34, Home: 36, End: 35, Shift: 16, Control: 17, Alt: 18, Meta: 91, Insert: 45,
  };
  const PUNCT_CODES = {
    ".": ["Period", 190], ",": ["Comma", 188], "/": ["Slash", 191], ";": ["Semicolon", 186], "'": ["Quote", 222],
    "[": ["BracketLeft", 219], "]": ["BracketRight", 221], "\\": ["Backslash", 220], "-": ["Minus", 189],
    "=": ["Equal", 187], "`": ["Backquote", 192],
  };
  function keyInfo(key) {
    if (key === " ") return { key, code: "Space", keyCode: 32, charCode: 32, printable: true };
    if (KEY_CODES[key]) return { key, code: key, keyCode: KEY_CODES[key], charCode: key === "Enter" ? 13 : 0, printable: false };
    if (/^F([1-9]|1[0-2])$/.test(key)) return { key, code: key, keyCode: 111 + +key.slice(1), charCode: 0, printable: false };
    let code = "";
    let keyCode = 0;
    if (/^[a-z0-9]$/i.test(key)) {
      code = (/\d/.test(key) ? "Digit" : "Key") + key.toUpperCase();
      keyCode = key.toUpperCase().charCodeAt(0);
    } else if (PUNCT_CODES[key]) [code, keyCode] = PUNCT_CODES[key];
    return { key, code, keyCode, charCode: key.codePointAt(0), printable: true };
  }
  function keyEvent(target, type, k, mods) {
    const init = eventInit(viewOf(target), { key: k.key, code: k.code, location: 0, repeat: false });
    for (const m of ["ctrlKey", "shiftKey", "altKey", "metaKey"]) init[m] = !!mods[m];
    const ev = construct(Ev.Keyboard, type, init);
    // constructed KeyboardEvents report keyCode/which/charCode as 0; own properties fix that, leaving prototypes alone
    const kc = type === "keypress" ? k.charCode : k.keyCode;
    defineProperty(ev, "keyCode", { get: () => kc });
    defineProperty(ev, "which", { get: () => kc });
    defineProperty(ev, "charCode", { get: () => (type === "keypress" ? k.charCode : 0) });
    return N.dispatch(target, ev);
  }
  function editableKind(el) {
    const t = el.localName;
    if (t === "textarea") return "textarea";
    if (t === "input") {
      const ty = inputType(el);
      return TEXT_INPUTS[ty] ? "input" : VALUE_INPUTS[ty] ? "value" : "";
    }
    return el.isContentEditable ? "rich" : "";
  }
  // labels, and wrappers such as div[role=combobox] or custom elements hosting one field, type into that field
  function editTarget(el) {
    el = controlOf(el);
    if (editableKind(el)) return el;
    const inner = [];
    const rec = (n) => {
      if (inner.length > 1 || n.nodeType !== 1) return;
      if (n !== el && editableKind(n) && n.localName !== "body") inner.push(n);
      else forEachChild(n, rec);
    };
    rec(el);
    return inner.length === 1 ? inner[0] : null;
  }
  const getVal = (el) => (el.localName === "textarea" ? N.textareaGet(el) : N.inputGet(el));
  // the prototype setter bypasses value trackers that frameworks (React) install on the instance, so they see the edit
  const setVal = (el, v) => (el.localName === "textarea" ? N.textareaSet(el, v) : N.inputSet(el, v));
  function getSel(el) {
    const ta = el.localName === "textarea";
    return attempt(() => {
      const s = ta ? N.textareaSelStart(el) : N.inputSelStart(el);
      const e = ta ? N.textareaSelEnd(el) : N.inputSelEnd(el);
      return typeof s === "number" && typeof e === "number" ? [s, e] : null;
    }, null);
  }
  const setSel = (el, a, b) => attempt(() => (el.localName === "textarea" ? N.textareaSetSel(el, a, b) : N.inputSetSel(el, a, b)));
  const inputEvent = (el, type, inputType, data) =>
    fire(el, Ev.Input, type, eventInit(viewOf(el), { inputType, data, cancelable: type === "beforeinput" }));
  function insertIntoField(el, str, kind) {
    if (!inputEvent(el, "beforeinput", kind, str)) return;
    const v = getVal(el);
    const sel = getSel(el) || [v.length, v.length];
    let ins = str;
    if (el.maxLength > 0) ins = ins.slice(0, max(0, el.maxLength - (v.length - (sel[1] - sel[0]))));
    if (!ins) return;
    setVal(el, v.slice(0, sel[0]) + ins + v.slice(sel[1]));
    setSel(el, sel[0] + ins.length, sel[0] + ins.length);
    inputEvent(el, "input", kind, str);
  }
  function deleteInField(el, forward) {
    const v = getVal(el);
    let sel = getSel(el) || [v.length, v.length];
    if (sel[0] === sel[1]) sel = forward ? [sel[0], min(v.length, sel[0] + 1)] : [max(0, sel[0] - 1), sel[0]];
    const kind = forward ? "deleteContentForward" : "deleteContentBackward";
    if (sel[0] === sel[1] || !inputEvent(el, "beforeinput", kind, null)) return;
    setVal(el, v.slice(0, sel[0]) + v.slice(sel[1]));
    setSel(el, sel[0], sel[0]);
    inputEvent(el, "input", kind, null);
  }
  function richInsert(el, str) {
    const d = el.ownerDocument;
    if (attempt(() => N.execCommand(d, "insertText", false, str), false)) return;
    if (!inputEvent(el, "beforeinput", "insertText", str)) return;
    const sel = N.getSelection(d);
    const node = d.createTextNode(str);
    if (sel && sel.rangeCount) {
      const range = sel.getRangeAt(0);
      range.deleteContents();
      range.insertNode(node);
      range.setStartAfter(node);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);
    } else el.appendChild(node);
    inputEvent(el, "input", "insertText", str);
  }
  function selectContents(el, toEnd) {
    const d = el.ownerDocument;
    const sel = N.getSelection(d);
    if (!sel) return;
    const range = d.createRange();
    range.selectNodeContents(el);
    if (toEnd) range.collapse(false);
    sel.removeAllRanges();
    sel.addRange(range);
  }
  function typeChars(el, kind, text) {
    const perChar = text.length <= 64;
    for (const ch of perChar ? text : [text]) {
      // no key events for a line break: enter can send a message and shift+enter can run a notebook cell
      if (ch === "\n") {
        if (kind === "rich") richInsert(el, ch);
        else if (kind !== "input") insertIntoField(el, ch, "insertLineBreak");
        continue;
      }
      const k = perChar ? keyInfo(ch) : null;
      const mods = { shiftKey: !!k && /^[A-Z]$/.test(ch) };
      let ok = !k || keyEvent(el, "keydown", k, mods);
      if (k && ok) ok = keyEvent(el, "keypress", k, mods);
      if (ok && kind === "rich") richInsert(el, ch);
      else if (ok) insertIntoField(el, ch, "insertText");
      if (k) keyEvent(el, "keyup", k, mods);
    }
  }
  function type(args) {
    const el0 = resolve(args.ref);
    const text = args.text === undefined || args.text === null ? "" : Str(args.text);
    const el = editTarget(el0);
    if (!el) {
      const hint = el0.localName === "select" ? " Use browser_select for dropdowns." : " Use browser_click for buttons, links and checkboxes.";
      throw fail("not_editable", `${args.ref} (${describe(el0)}) is not a text field.${hint}`);
    }
    if (isPassword(el)) {
      throw fail("password_field", `${args.ref} is a password field. Passwords must be typed by the user; ask them to take over (browser_handoff).`);
    }
    if (isCardField(el)) {
      throw fail("password_field", `${args.ref} takes card details. Those must be typed by the user; ask them to take over (browser_handoff).`);
    }
    if (isDisabled(el) || el.readOnly) throw fail("not_editable", `${args.ref} (${describe(el)}) is disabled or read-only.`);
    ensureActionable(el, args.ref);
    bringIntoView(el);
    flash(el, 700);
    focusEl(el);
    const kind = editableKind(el);
    const clear = args.clear !== false;
    if (kind === "rich") {
      selectContents(el, !clear);
      const cleared = () => attempt(() => N.execCommand(el.ownerDocument, "delete", false, null), false);
      if (clear && collapse(el.textContent) && !cleared()) el.textContent = "";
      if (text) typeChars(el, kind, text);
    } else if (kind === "value") {
      setVal(el, text);
      inputEvent(el, "input", "insertReplacementText", text);
    } else {
      const cur = getVal(el);
      if (clear && cur) {
        setSel(el, 0, cur.length);
        if (inputEvent(el, "beforeinput", "deleteContentBackward", null)) {
          setVal(el, "");
          inputEvent(el, "input", "deleteContentBackward", null);
        }
      } else if (!clear) setSel(el, cur.length, cur.length);
      if (text) typeChars(el, kind, text);
    }
    if (kind !== "rich") fire(el, Ev.Event, "change", { bubbles: true });
    const out = { description: describe(el) };
    if (args.submit) out.submitted = pressEnter(el);
    out.value = kind === "rich" ? clip(collapse(el.textContent), 200) : getVal(el);
    return out;
  }
  // the first submit button the form owns, in tree order; form.elements alone would miss image buttons by spec
  function defaultButton(form) {
    const isSubmit = (b) => {
      const t = b.localName;
      if (t === "button") return Str(b.type || "submit").toLowerCase() === "submit";
      const ty = t === "input" ? inputType(b) : "";
      return ty === "submit" || ty === "image";
    };
    const inside = N.qsa(form, "button, input");
    for (let i = 0; i < inside.length; i++) if (inside[i].form === form && isSubmit(inside[i])) return inside[i];
    const els = form.elements;
    for (let i = 0; i < els.length; i++) if (isSubmit(els[i])) return els[i];
    return null;
  }
  // implicit submission clicks the form's default button (so the submitter and its handlers run), else requestSubmit()
  function implicitSubmit(form) {
    const b = defaultButton(form);
    if (b) {
      if (isDisabled(b)) return false;
      N.click(b);
      return true;
    }
    if (N.requestSubmit) N.requestSubmit(form);
    else if (fire(form, Ev.Event, "submit", { bubbles: true, cancelable: true })) N.formSubmit(form);
    return true;
  }
  function pressEnter(el) {
    const k = keyInfo("Enter");
    const down = keyEvent(el, "keydown", k, {});
    let submitted = false;
    if (down) {
      keyEvent(el, "keypress", k, {});
      const kind = editableKind(el);
      const form = el.form || (kind ? N.closest(el, "form") : null);
      if ((kind === "input" || kind === "value") && form) submitted = implicitSubmit(form);
    }
    keyEvent(el, "keyup", k, {});
    return submitted;
  }

  const normalize = (s) => collapse(s).toLowerCase();
  const alnumOnly = (s) => normalize(s).replace(/[^\p{L}\p{N}]+/gu, "");
  // exact label, exact value, case-insensitive, prefix, substring, punctuation-insensitive; enabled options win ties
  function pickOption(cands, wanted) {
    const w = Str(wanted);
    const wn = normalize(w);
    const wa = alnumOnly(w);
    const tests = [
      (c) => c.label === w,
      (c) => c.value !== undefined && c.value === w,
      (c) => normalize(c.label) === wn,
      (c) => c.value !== undefined && normalize(c.value) === wn,
      (c) => wn && normalize(c.label).indexOf(wn) === 0,
      (c) => wn && normalize(c.label).indexOf(wn) >= 0,
      (c) => wa && alnumOnly(c.label) === wa,
      (c) => wa && alnumOnly(c.label).indexOf(wa) >= 0,
    ];
    for (const test of tests) {
      const found = cands.filter(test);
      if (found.length) return found.find((c) => !c.disabled) || found[0];
    }
    return null;
  }
  function noOption(ref, cands, wanted) {
    const names = cands.slice(0, 30).map((c) => c.label).join(", ");
    const more = cands.length > 30 ? ` (+${cands.length - 30} more)` : "";
    return fail("no_option", `No option matching ${quote(Str(wanted))} in ${ref}. Available: ${names}${more}`);
  }
  function choose(cands, wanted, ref) {
    const c = pickOption(cands, wanted);
    if (!c) throw noOption(ref, cands, wanted);
    if (c.disabled) throw fail("no_option", `Option ${quote(c.label)} in ${ref} is disabled.`);
    return c;
  }
  function select(args) {
    const wanted = args.option;
    if (wanted === undefined || wanted === null || (typeof wanted !== "string" && !isArray(wanted))) {
      throw fail("bad_args", "select needs `option`: the visible label (or value) of the option to choose.");
    }
    let el = controlOf(resolve(args.ref));
    if (el.localName !== "select") {
      const inner = N.qsa(el, "select");
      if (inner.length === 1) el = inner[0];
    }
    if (el.localName === "select") return selectNative(el, args.ref, isArray(wanted) ? wanted : [wanted]);
    return selectAria(el, args.ref, isArray(wanted) ? wanted[0] : wanted);
  }
  function selectNative(el, ref, wanted) {
    if (isDisabled(el)) throw fail("disabled", `${ref} (${describe(el)}) is disabled.`);
    const cands = [];
    for (let i = 0; i < el.options.length; i++) {
      const o = el.options[i];
      const group = o.parentElement && o.parentElement.localName === "optgroup" ? o.parentElement : null;
      cands.push({ o, label: optionLabel(o), value: o.value, disabled: o.disabled || !!(group && group.disabled) });
    }
    const picks = wanted.map((w) => choose(cands, w, ref));
    bringIntoView(el);
    flash(el, 700);
    focusEl(el);
    if (el.multiple) for (const c of cands) N.optionSelectedSet(c.o, false);
    for (const c of picks) N.optionSelectedSet(c.o, true);
    fire(el, Ev.Event, "input", { bubbles: true, composed: true });
    fire(el, Ev.Event, "change", { bubbles: true });
    return { description: describe(el), selected: picks.map((c) => c.label).join(", ") };
  }
  function ariaOptions(el) {
    const scopes = [];
    const ids = `${attr(el, "aria-controls") || ""} ${attr(el, "aria-owns") || ""}`.trim();
    const root = N.getRootNode(el);
    for (const id of ids ? ids.split(WS_SPLIT) : []) {
      const s = byId(root, id) || byId(el.ownerDocument, id);
      if (s) scopes.push(s);
    }
    scopes.push(el);
    const out = [];
    const seen = new WeakSetCtor();
    const take = (o) => {
      if (wsHas(seen, o) || !N.rects(o).length || styleOf(o).visibility !== "visible") return;
      wsAdd(seen, o);
      out.push({ o, label: accName(o, "option"), disabled: attr(o, "aria-disabled") === "true" });
    };
    for (const s of scopes) {
      if (attr(s, "role") === "option") take(s);
      N.qsa(s, '[role="option"]').forEach(take);
    }
    if (!out.length) N.docQsa(el.ownerDocument, '[role="option"]').forEach(take);
    return out;
  }
  // options that render only after the opening click (a microtask commit) are unreachable here, hence options_pending
  function selectAria(el, ref, wanted) {
    const role = roleOf(el);
    if (role !== "combobox" && role !== "listbox" && role !== "button" && !attr(el, "aria-haspopup")) {
      throw fail("not_selectable", `${ref} (${describe(el)}) is not a dropdown or list; use browser_click instead.`);
    }
    bringIntoView(el);
    let cands = ariaOptions(el);
    let opened = false;
    if (!cands.length && role !== "listbox") {
      flash(el, 700);
      mouseSequence(el);
      opened = true;
      cands = ariaOptions(el);
    }
    if (!cands.length) {
      if (opened) throw fail("options_pending", `Opened ${ref} but its options have not appeared yet; call browser_select again (or take a snapshot).`);
      throw fail("no_option", `${ref} (${describe(el)}) shows no options.`);
    }
    const c = choose(cands, wanted, ref);
    bringIntoView(c.o);
    flash(c.o, 700);
    mouseSequence(c.o);
    return { description: describe(el), selected: c.label };
  }

  const KEY_ALIASES = {
    enter: "Enter", return: "Enter", escape: "Escape", esc: "Escape", tab: "Tab", backspace: "Backspace", delete: "Delete",
    del: "Delete", arrowup: "ArrowUp", up: "ArrowUp", arrowdown: "ArrowDown", down: "ArrowDown", arrowleft: "ArrowLeft",
    left: "ArrowLeft", arrowright: "ArrowRight", right: "ArrowRight", pageup: "PageUp", pagedown: "PageDown", home: "Home",
    end: "End", space: " ", spacebar: " ", insert: "Insert",
  };
  const MODIFIERS = {
    control: ["ctrlKey", "Control"], ctrl: ["ctrlKey", "Control"], meta: ["metaKey", "Meta"], cmd: ["metaKey", "Meta"],
    command: ["metaKey", "Meta"], super: ["metaKey", "Meta"], shift: ["shiftKey", "Shift"], alt: ["altKey", "Alt"],
    option: ["altKey", "Alt"],
  };
  function parseKey(spec) {
    if (typeof spec !== "string" || !spec) throw fail("bad_args", "press needs `key`, e.g. Enter, Escape, Tab or Control+a.");
    // "Control++" names the plus key itself.
    const parts = spec === "+" ? ["+"] : spec.endsWith("++") ? spec.slice(0, -2).split("+").concat(["+"]) : spec.split("+");
    const mods = { ctrlKey: false, shiftKey: false, altKey: false, metaKey: false };
    const modKeys = [];
    for (const p of parts.slice(0, -1)) {
      const m = MODIFIERS[p.trim().toLowerCase()];
      if (!m) throw fail("bad_args", `Unknown modifier "${p}" in "${spec}". Use Control, Shift, Alt or Meta.`);
      mods[m[0]] = true;
      modKeys.push(m[1]);
    }
    let key = parts[parts.length - 1];
    if (KEY_ALIASES[key.trim().toLowerCase()]) key = KEY_ALIASES[key.trim().toLowerCase()];
    else if (/^f([1-9]|1[0-2])$/i.test(key)) key = key.toUpperCase();
    else if ([...key].length !== 1) {
      throw fail("bad_args", `Unknown key "${spec}". Use names like Enter, Escape, Tab, ArrowDown, PageDown, Home, or a single character.`);
    }
    if (mods.shiftKey && /^[a-z]$/.test(key)) key = key.toUpperCase();
    return { k: keyInfo(key), mods, modKeys };
  }
  function tabbables(scope) {
    const out = [];
    const rec = (el) => {
      const t = el.localName;
      if (SKIP_TAGS[t] || isOwn(el) || N.hasAttr(el, "inert")) return;
      const cs = styleOf(el);
      if (cs.display === "none") return;
      if (t === "iframe" || t === "frame") {
        const d = frameDoc(el);
        if (d && d.documentElement) rec(d.documentElement);
        return;
      }
      if (cs.visibility === "visible" && el.tabIndex >= 0 && !el.disabled && !(t === "input" && inputType(el) === "hidden")) {
        const r = N.rect(el);
        const closed = N.closest(el, "details:not([open])");
        if ((r.width > 0 || r.height > 0) && (!closed || (t === "summary" && el.parentElement === closed))) out.push(el);
      }
      if (t !== "select" && t !== "textarea") forEachChild(el, (n) => n.nodeType === 1 && rec(n));
    };
    rec(scope);
    // only the checked radio of a group (or its first member) is a tab stop
    const groups = newMap();
    const res = [];
    for (const el of out) {
      if (el.localName === "input" && inputType(el) === "radio" && el.name) {
        const key = el.name + (el.form ? "\u0000f" : "");
        const g = groups.get(key);
        if (g) {
          if (el.checked && !g.el.checked) (res[g.i] = el), (g.el = el);
          continue;
        }
        groups.set(key, { el, i: res.length });
      }
      res.push(el);
    }
    return res.filter((e) => e.tabIndex > 0).sort((a, b) => a.tabIndex - b.tabIndex).concat(res.filter((e) => e.tabIndex === 0));
  }
  function findModalEl() {
    let found = null;
    const ds = N.docQsa(doc, 'dialog[open], [aria-modal="true"]');
    for (let i = 0; i < ds.length; i++) if (isModalDialog(ds[i]) && N.rects(ds[i]).length) found = ds[i];
    return found;
  }
  function moveFocus(backward) {
    const list = tabbables(findModalEl() || doc.documentElement);
    if (!list.length) return;
    const i = list.indexOf(deepActive());
    const next = i < 0 ? list[backward ? list.length - 1 : 0] : list[(i + (backward ? -1 : 1) + list.length) % list.length];
    focusEl(next);
    // tabbing into a text field selects its contents, as browsers do
    const kind = editableKind(next);
    if (deepActive() === next && kind && kind !== "rich") setSel(next, 0, getVal(next).length);
  }
  function closeTopLayer() {
    const ds = N.docQsa(doc, "dialog[open]");
    for (let i = ds.length - 1; i >= 0; i--) {
      if (!attempt(() => N.matches(ds[i], ":modal"), true)) continue;
      if (fire(ds[i], Ev.Event, "cancel", { cancelable: true })) N.dialogClose(ds[i]);
      return;
    }
    attempt(() => {
      const ps = N.docQsa(doc, ":popover-open");
      for (let i = ps.length - 1; i >= 0; i--) {
        if (!isOwn(ps[i]) && attr(ps[i], "popover") !== "manual") return N.hidePopover(ps[i]);
      }
    });
  }
  // key presses reach the same fields typing does, so secrets are refused here too
  const isSecretField = (el) => !!el && (isPassword(el) || isCardField(el));
  function press(args) {
    const { k, mods, modKeys } = parseKey(args.key);
    const target = deepActive() || doc.body || doc.documentElement;
    if (isSecretField(target) && (k.printable || k.key === "Backspace" || k.key === "Delete")) {
      throw fail("password_field", "The focused field takes a password or card number. Those must be typed by the user; ask them to take over (browser_handoff).");
    }
    for (const m of modKeys) keyEvent(target, "keydown", keyInfo(m), mods);
    const down = keyEvent(target, "keydown", k, mods);
    const plain = !mods.ctrlKey && !mods.metaKey && !mods.altKey;
    const ok = down && (!k.printable || !plain || keyEvent(target, "keypress", k, mods));
    if (ok) defaultKeyAction(target, k, mods, plain);
    keyEvent(target, "keyup", k, mods);
    for (const m of modKeys.slice().reverse()) keyEvent(target, "keyup", keyInfo(m), mods);
    const a = deepActive();
    return { key: args.key, prevented: !down, focused: a && a !== doc.body ? describe(a) : null };
  }
  // default actions that synthetic key events cannot trigger on their own
  function defaultKeyAction(target, k, mods, plain) {
    const key = k.key;
    const kind = editableKind(target);
    if (key === "Tab") return moveFocus(mods.shiftKey);
    if (key === "Escape") return closeTopLayer();
    if ((mods.ctrlKey || mods.metaKey) && key.toLowerCase() === "a") {
      if (kind === "rich") selectContents(target, false);
      else if (kind) setSel(target, 0, getVal(target).length);
      return;
    }
    if (!plain) return;
    const doc_ = target.ownerDocument;
    if (kind === "input" || kind === "textarea" || kind === "rich") {
      if (key === "Enter") {
        if (kind === "textarea") insertIntoField(target, "\n", "insertLineBreak");
        else if (kind === "rich") attempt(() => N.execCommand(doc_, mods.shiftKey ? "insertLineBreak" : "insertParagraph", false, null));
        else if (target.form) implicitSubmit(target.form);
      } else if (key === "Backspace" || key === "Delete") {
        if (kind === "rich") attempt(() => N.execCommand(doc_, key === "Delete" ? "forwardDelete" : "delete", false, null));
        else deleteInField(target, key === "Delete");
      } else if (k.printable) {
        if (kind === "rich") richInsert(target, key);
        else insertIntoField(target, key, "insertText");
      } else if ((key === "Home" || key === "End") && kind !== "rich") {
        const n = key === "Home" ? 0 : getVal(target).length;
        setSel(target, n, n);
      }
      return;
    }
    const t = target.localName;
    if (t === "select" && (key === "ArrowDown" || key === "ArrowUp")) {
      const o = target.options[target.selectedIndex + (key === "ArrowDown" ? 1 : -1)];
      if (o) {
        N.optionSelectedSet(o, true);
        fire(target, Ev.Event, "input", { bubbles: true, composed: true });
        fire(target, Ev.Event, "change", { bubbles: true });
      }
      return;
    }
    const role = roleOf(target);
    const toggle = t === "input" && /^(checkbox|radio)$/.test(inputType(target));
    const activatable = toggle || /^(button|summary)$/.test(t) || (t === "a" && N.hasAttr(target, "href")) ||
      (t === "input" && /^(button|submit|reset|image)$/.test(inputType(target))) ||
      /^(button|link|menuitem|menuitemcheckbox|menuitemradio|tab|option|checkbox|switch|radio)$/.test(role);
    const html = target instanceof viewOf(target).HTMLElement;
    if (activatable && html && ((key === "Enter" && !toggle) || (key === " " && role !== "link"))) return N.click(target);
    const page = 0.875 * win.innerHeight;
    const dy = { PageDown: page, PageUp: -page, " ": mods.shiftKey ? -page : page, ArrowDown: 40, ArrowUp: -40, Home: -1e7, End: 1e7 }[key];
    if (dy) doScroll(scrollTargetFor(target, "y"), 0, dy);
    else if (key === "ArrowLeft" || key === "ArrowRight") doScroll(scrollTargetFor(target, "x"), key === "ArrowLeft" ? -40 : 40, 0);
  }

  function scrollableIn(el, axis) {
    const o = axis === "y" ? styleOf(el).overflowY : styleOf(el).overflowX;
    if (o !== "auto" && o !== "scroll" && o !== "overlay") return false;
    return axis === "y" ? el.scrollHeight > el.clientHeight + 1 : el.scrollWidth > el.clientWidth + 1;
  }
  function windowScrolls(axis, d) {
    d = d || doc;
    const se = d.scrollingElement || d.documentElement;
    const v = d.defaultView;
    if (!se || !(axis === "y" ? se.scrollHeight > v.innerHeight + 1 : se.scrollWidth > v.innerWidth + 1)) return false;
    const prop = axis === "y" ? "overflowY" : "overflowX";
    const root = styleOf(d.documentElement)[prop];
    const body = d.body ? styleOf(d.body)[prop] : "";
    const hidden = (o) => o === "hidden" || o === "clip";
    return !(hidden(root) || (hidden(body) && root === "visible"));
  }
  let scrollerCache = null;
  // many apps scroll an inner full-height element, not the window, so the largest visible scroller stands in for it
  function dominantScroller(axis) {
    const c = scrollerCache;
    if (c && c.m === mutationCount && c.axis === axis && (!c.el || c.el.isConnected)) return c.el;
    let best = null;
    let bestArea = 0;
    const vw = win.innerWidth;
    const vh = win.innerHeight;
    const all = N.docQsa(doc, "body *");
    for (let i = 0; i < all.length; i++) {
      const el = all[i];
      if (axis === "y" ? el.scrollHeight <= el.clientHeight + 1 : el.scrollWidth <= el.clientWidth + 1) continue;
      if (el.clientHeight < 50 || !scrollableIn(el, axis)) continue;
      const r = N.rect(el);
      const area = max(0, min(r.right, vw) - max(r.left, 0)) * max(0, min(r.bottom, vh) - max(r.top, 0));
      if (area > bestArea) (bestArea = area), (best = el);
    }
    if (bestArea < vw * vh * 0.15) best = null;
    scrollerCache = { m: mutationCount, axis, el: best };
    return best;
  }
  function scrollTargetFor(el, axis) {
    for (let x = el; x; x = parentOf(x)) {
      if (x.nodeType !== 1) continue;
      if (x.localName === "html" || x.localName === "body") {
        if (windowScrolls(axis, x.ownerDocument)) return { win: x.ownerDocument.defaultView, d: x.ownerDocument };
      } else if (scrollableIn(x, axis)) return { el: x };
    }
    return pageScroller(axis);
  }
  function pageScroller(axis) {
    const dom = windowScrolls(axis) ? null : dominantScroller(axis);
    return dom ? { el: dom } : { win, d: doc };
  }
  function scrollMetrics(t) {
    const el = t.el;
    if (el) return { x: el.scrollLeft, y: el.scrollTop, h: el.scrollHeight, vh: el.clientHeight, w: el.scrollWidth, vw: el.clientWidth };
    const se = t.d.scrollingElement || t.d.documentElement;
    const v = t.win;
    return { x: v.scrollX, y: v.scrollY, h: max(se.scrollHeight, v.innerHeight), vh: v.innerHeight, w: se.scrollWidth, vw: v.innerWidth };
  }
  function doScroll(t, dx, dy) {
    try {
      const opts = { left: dx, top: dy, behavior: "instant" };
      if (t.el) N.elScrollBy(t.el, opts);
      else N.winScrollBy(t.win, opts);
    } catch (_) {
      if (!t.el) return N.winScrollBy(t.win, dx, dy);
      t.el.scrollLeft += dx;
      t.el.scrollTop += dy;
    }
  }
  function scroll(args) {
    const dir = Str(args.direction || "down").toLowerCase();
    if (!/^(up|down|left|right)$/.test(dir)) throw fail("bad_args", 'scroll direction must be "up", "down", "left" or "right".');
    const axis = dir === "up" || dir === "down" ? "y" : "x";
    const amount = typeof args.amount === "number" && args.amount > 0 ? min(args.amount, 20) : 0.8;
    const t = args.ref ? scrollTargetFor(resolve(args.ref), axis) : pageScroller(axis);
    const before = scrollMetrics(t);
    const sign = dir === "up" || dir === "left" ? -1 : 1;
    const dist = sign * amount * (axis === "y" ? before.vh : before.vw);
    doScroll(t, axis === "x" ? dist : 0, axis === "y" ? dist : 0);
    let after = scrollMetrics(t);
    const atEdge = axis === "y" && (sign < 0 ? before.y <= 0 : before.y + before.vh >= before.h - 1);
    // custom scrollers (carousels, full-page sliders) only react to wheel input
    const wheel = after.x === before.x && after.y === before.y && !atEdge && !!Ev.Wheel;
    if (wheel) {
      const c = t.el ? centerOf(t.el) : { x: win.innerWidth / 2, y: win.innerHeight / 2 };
      const target = N.elementFromPoint(doc, c.x, c.y) || t.el || doc.body;
      const delta = { deltaX: axis === "x" ? dist : 0, deltaY: axis === "y" ? dist : 0, deltaMode: 0 };
      fire(target, Ev.Wheel, "wheel", eventInit(win, { clientX: c.x, clientY: c.y, ...delta }));
      after = scrollMetrics(t);
    }
    const out = {
      scrollY: round(after.y), scrollHeight: round(after.h), viewportHeight: round(after.vh),
      atTop: after.y <= 1, atBottom: after.y + after.vh >= after.h - 2, target: t.el ? describe(t.el) : "page",
    };
    if (axis === "x") (out.scrollX = round(after.x)), (out.scrollWidth = round(after.w));
    if (wheel) out.wheel = true;
    return out;
  }

  const READ_SKIP_TAGS = wordSet("nav aside footer form button select input textarea iframe frame svg canvas video audio object embed dialog menu");
  const READ_SKIP_ROLES = wordSet("navigation complementary contentinfo banner search dialog alertdialog menu menubar toolbar tablist");
  const NOISE = /(^|[\s_-])(share|sharing|social|newsletter|cookie|advert|ads|sponsor|promo|breadcrumbs?|skip-?link|mw-editsection|navbox|noprint|metadata|sr-only|visually-hidden)([\s_-]|$)/i;
  const BLOCK_TAGS = /^(p|div|section|article|main|header|ul|ol|li|table|pre|blockquote|h[1-6]|hr|dl|dt|dd|figure|figcaption|details|summary|address)$/;
  const BLOCK_DISPLAY = /^(block|flex|grid|list-item|table|flow-root|table-row-group|table-header-group|table-footer-group|table-row|table-caption)/;
  const BR = "\u0001";
  const textLength = (el) => (el ? collapse(el.textContent).length : 0);
  function readSkip(el, root) {
    if (el === root) return false;
    const t = el.localName;
    if (SKIP_TAGS[t] || READ_SKIP_TAGS[t] || isOwn(el) || N.hasAttr(el, "hidden") || attr(el, "aria-hidden") === "true") return true;
    const r = attr(el, "role");
    if ((r && READ_SKIP_ROLES[r.trim().split(WS_SPLIT)[0]]) || (t === "header" && roleOf(el) === "banner")) return true;
    const cls = `${attr(el, "class") || ""} ${attr(el, "id") || ""}`;
    if (cls.length > 1 && NOISE.test(cls)) return true;
    const cs = styleOf(el);
    if (cs.display === "none" || cs.visibility === "hidden") return true;
    if (cs.position !== "absolute") return false;
    const box = N.rect(el);
    return box.width <= 1 || box.height <= 1;
  }
  function readRoot() {
    const body = doc.body;
    if (!body) return doc.documentElement;
    const total = textLength(body);
    const pick = (sel) => {
      let best = null;
      let bestLen = 0;
      N.docQsa(doc, sel).forEach((el) => {
        const len = N.rects(el).length ? textLength(el) : 0;
        if (len > bestLen) (best = el), (bestLen = len);
      });
      return bestLen >= max(200, total * 0.25) ? best : null;
    };
    let root = pick("main, [role=main]") || pick("article") || body;
    // descend while a single child carries most of the text
    for (let guard = 0; guard < 40; guard++) {
      const len = textLength(root);
      let best = null;
      let bestLen = 0;
      for (let c = root.firstElementChild; c; c = c.nextElementSibling) {
        if (READ_SKIP_TAGS[c.localName] || SKIP_TAGS[c.localName]) continue;
        const l = textLength(c);
        if (l > bestLen) (best = c), (bestLen = l);
      }
      if (!best || bestLen < len * 0.75 || /^(p|ul|ol|table|pre|blockquote|dl|h[1-6]|li)$/.test(best.localName)) break;
      root = best;
    }
    return root;
  }
  function absUrl(a) {
    const raw = attr(a, "href") || "";
    if (!raw || raw[0] === "#" || /^javascript:/i.test(raw)) return "";
    return attempt(() => new URLCtor(raw, a.baseURI).href, "");
  }
  const isBlock = (el) => BLOCK_TAGS.test(el.localName) || (!INLINE_TAGS[el.localName] && BLOCK_DISPLAY.test(styleOf(el).display));
  const cleanInline = (s) => s.replace(/[ \t\r\n\f ]+/g, " ").split(BR).map((x) => x.trim()).join("\n").trim();
  function mdInline(el, root) {
    let out = "";
    forEachChild(el, (n) => {
      if (n.nodeType === 3) out += n.data;
      else if (n.nodeType === 1 && !readSkip(n, root)) out += mdNode(n, root);
    });
    return out;
  }
  function mdNode(n, root) {
    const t = n.localName;
    if (t === "br") return BR;
    if (t === "img") {
      const alt = collapse(attr(n, "alt"));
      return alt ? ` [image: ${alt}] ` : "";
    }
    if (t === "a") {
      const text = collapse(mdInline(n, root).split(BR).join(" "));
      if (!text || CITATION.test(text)) return "";
      const href = absUrl(n);
      return href ? `[${text}](${href})` : text;
    }
    if ((t === "code" || t === "kbd" || t === "samp") && !N.closest(n, "pre")) {
      const text = collapse(n.textContent);
      return text ? "`" + text + "`" : "";
    }
    if (t === "sup" && CITATION.test(collapse(n.textContent))) return "";
    const block = isBlock(n);
    return (block ? " " : "") + mdInline(n, root) + (block ? " " : "");
  }
  function mdBlocks(el, out, root) {
    let inline = "";
    const flush = () => {
      const t = cleanInline(inline);
      if (t) out.push(t);
      inline = "";
    };
    forEachChild(el, (n) => {
      if (n.nodeType === 3) inline += n.data;
      else if (n.nodeType !== 1 || readSkip(n, root)) return;
      else if (!isBlock(n)) inline += mdNode(n, root);
      else {
        flush();
        mdBlock(n, out, root);
      }
    });
    flush();
  }
  function mdBlock(n, out, root) {
    const t = n.localName;
    const sub = () => {
      const s = [];
      mdBlocks(n, s, root);
      return s;
    };
    const push = (s) => s && out.push(s);
    const line = () => cleanInline(mdInline(n, root)).replace(/\n/g, " ");
    const h = /^h([1-6])$/.exec(t);
    if (h) push(line() && "#".repeat(+h[1]) + " " + line());
    else if (t === "ul" || t === "ol") push(mdList(n, root, t === "ol"));
    else if (t === "pre") {
      const code = n.textContent.replace(/\n+$/, "");
      if (!code.trim()) return;
      const c = n.querySelector("code");
      const m = /(?:^|\s)(?:language|lang)-([\w+#-]+)/.exec(`${attr(n, "class") || ""} ${c ? attr(c, "class") || "" : ""}`);
      const fence = code.indexOf("```") >= 0 ? "~~~" : "```";
      out.push(`${fence}${m ? m[1] : ""}\n${code}\n${fence}`);
    } else if (t === "table") push(mdTable(n, root));
    else if (t === "blockquote") {
      const body = sub();
      if (body.length) out.push(body.join("\n\n").split("\n").map((l) => "> " + l).join("\n"));
    } else if (t === "hr") out.push("---");
    else if (t === "li") {
      const body = sub();
      if (body.length) out.push("- " + body.join("\n").split("\n").join("\n  "));
    } else if (t === "dt") push(line() && `**${line()}**`);
    else if (t === "img") push(collapse(attr(n, "alt")) && `[image: ${collapse(attr(n, "alt"))}]`);
    else mdBlocks(n, out, root);
  }
  function mdList(list, root, ordered) {
    const lines = [];
    let i = ordered ? parseInt(attr(list, "start"), 10) || 1 : 1;
    forEachChild(list, (li) => {
      if (li.nodeType !== 1 || readSkip(li, root)) return;
      if (li.localName === "ul" || li.localName === "ol") {
        const nested = mdList(li, root, li.localName === "ol");
        if (nested) lines.push(nested.split("\n").map((l) => "  " + l).join("\n"));
        return;
      }
      if (li.localName !== "li") return;
      const sub = [];
      mdBlocks(li, sub, root);
      if (!sub.length) return;
      const marker = ordered ? `${i++}. ` : "- ";
      lines.push(marker + sub.join("\n").split("\n").join("\n" + " ".repeat(marker.length)));
    });
    return lines.join("\n");
  }
  function mdTable(table, root) {
    const rows = table.rows;
    if (!rows || !rows.length) return "";
    // layout tables (nested tables, block content) read as plain blocks
    if (N.qsa(table, "table").length || N.qsa(table, "td p, td ul, td ol, td div div, td h1, td h2, td h3").length > 2) {
      const sub = [];
      mdBlocks(table, sub, root);
      return sub.join("\n\n");
    }
    const grid = [];
    let cols = 0;
    for (let r = 0; r < rows.length; r++) {
      const row = rows[r];
      if (N.closest(row, "table") !== table || readSkip(row, root)) continue;
      const cells = [];
      for (let c = 0; c < row.cells.length; c++) {
        const sub = [];
        mdBlocks(row.cells[c], sub, root);
        cells.push(sub.join(" ").replace(/\n+/g, " ").replace(/\|/g, "\\|").trim());
      }
      if (!cells.some((x) => x)) continue;
      cols = max(cols, cells.length);
      grid.push(cells);
    }
    if (!grid.length) return "";
    const cap = table.caption ? cleanInline(mdInline(table.caption, root)) : "";
    let md;
    if (cols <= 8) {
      const row = (cells) => "| " + cells.concat(Array(cols - cells.length).fill("")).join(" | ") + " |";
      md = [row(grid[0]), "|" + " --- |".repeat(cols)].concat(grid.slice(1).map(row)).join("\n");
    } else {
      md = grid.map((cells) => "- " + cells.filter((x) => x).join(" · ")).join("\n");
    }
    return cap ? cap + "\n\n" + md : md;
  }
  let readCache = null;
  function readMarkdown() {
    if (readCache && readCache.m === mutationCount && readCache.url === loc.href) return readCache.text;
    const root = readRoot();
    const blocks = [];
    // a deep root (the article body) can miss the page title, so it is kept
    if (root !== doc.body) {
      const h1 = N.docQsa(doc, "h1");
      for (let i = 0; i < h1.length; i++) {
        if (!N.rects(h1[i]).length) continue;
        if (!root.contains(h1[i]) && collapse(h1[i].textContent)) blocks.push("# " + collapse(h1[i].textContent));
        break;
      }
    }
    mdBlocks(root, blocks, root);
    const text = blocks.join("\n\n").replace(/\n{3,}/g, "\n\n").trim();
    readCache = { m: mutationCount, url: loc.href, text };
    return text;
  }
  function read(args) {
    const text = readMarkdown();
    const total = text.length;
    const offset = unsplit(text, clampInt(args.offset, 0, 0, total));
    const size = clampInt(args.maxChars, 8000, 200, 100000);
    let end = unsplit(text, min(total, offset + size));
    // stop at a paragraph or line boundary near the window end; nextOffset tells where the next page starts
    if (end < total) {
      const slice = text.slice(offset, end);
      const floor_ = round(size * 0.75);
      let cut = slice.lastIndexOf("\n\n");
      if (cut < floor_) cut = slice.lastIndexOf("\n");
      if (cut < floor_) cut = slice.lastIndexOf(" ");
      if (cut >= floor_) end = offset + cut + 1;
    }
    return { text: text.slice(offset, end), url: loc.href, title: doc.title, offset, nextOffset: end < total ? end : null, total };
  }

  function inspect(args) {
    const el = resolve(args.ref);
    const target = controlOf(el);
    const role = roleOf(target) || (styleOf(target).cursor === "pointer" ? "clickable" : "");
    const kind = editableKind(target);
    const out = {
      description: describe(el), role, tag: target.localName, inputType: target.localName === "input" ? inputType(target) : null,
      sensitive: sensitivity(el), name: accName(target, role === "clickable" ? "" : role), disabled: isDisabled(target),
      secret: isSecretField(target),
    };
    if (target.localName === "a" && N.hasAttr(target, "href")) out.href = absUrl(target) || attr(target, "href");
    if (kind && !isPassword(target)) out.value = kind === "rich" ? clip(collapse(target.textContent), 200) : getVal(target);
    return out;
  }
  function locate(args) {
    const el = resolve(args.ref);
    bringIntoView(el);
    const r = topRect(el);
    const vw = win.innerWidth;
    const vh = win.innerHeight;
    const onScreen = r.r > 0 && r.b > 0 && r.l < vw && r.t < vh && r.w > 0 && r.h > 0;
    const visible = onScreen && styleOf(el).visibility === "visible" && !!N.rects(el).length;
    const cx = (max(r.l, 0) + min(r.r, vw)) / 2;
    const cy = (max(r.t, 0) + min(r.b, vh)) / 2;
    let occludedBy = null;
    if (visible) {
      const off = el.ownerDocument === doc ? { x: 0, y: 0 } : frameOffset(el.ownerDocument);
      const hit = deepHit(el.ownerDocument, cx - off.x, cy - off.y);
      const viaLabel =
        hit && ((el.localName === "label" && el.control && containsComposed(el.control, hit)) || (hit.localName === "label" && hit.control === el));
      if (hit && !containsComposed(el, hit) && !viaLabel) occludedBy = describe(hit);
    }
    const r2 = (v) => round(v * 100) / 100;
    return {
      rect: { x: r2(r.l), y: r2(r.t), width: r2(r.w), height: r2(r.h) },
      center: { x: r2(onScreen ? cx : r.l + r.w / 2), y: r2(onScreen ? cy : r.t + r.h / 2) },
      visible,
      occludedBy,
      // native input aims at the top document; a frame's own listeners would not see it land
      inFrame: el.ownerDocument !== doc,
    };
  }
  // what Enter or Space on the focused (or ref'd) element would activate, so the caller can ask before it happens
  function activation(args) {
    const target = args.ref ? controlOf(resolve(args.ref)) : deepActive();
    if (!target || target === doc.body) return { sensitive: null, description: null };
    const key = args.key ? parseKey(args.key).k.key : "Enter";
    const kind = editableKind(target);
    if (key === "Enter" && (kind === "input" || kind === "value")) {
      const form = target.form || N.closest(target, "form");
      const button = form ? defaultButton(form) : null;
      return {
        sensitive: sensitivity(target) || (button ? sensitivity(button) : null),
        description: describe(button || target),
      };
    }
    if (key === "Enter" || key === " ") return { sensitive: sensitivity(target), description: describe(target) };
    return { sensitive: isSecretField(target) ? "password" : null, description: describe(target) };
  }
  // watches where trusted clicks land for the native click path; it lives here so the page cannot fake what it reports
  let clickProbe = null;
  function probe(args) {
    const x = typeof args.x === "number" ? args.x : NaN;
    const y = typeof args.y === "number" ? args.y : NaN;
    if (args.step === "arm") {
      if (clickProbe) clickProbe.stop();
      // judged against the element meant, not the node at the point, which hover can restyle or swap before the press
      const target = Number.isFinite(x) && Number.isFinite(y) && args.ref ? resolve(args.ref) : null;
      if (!target) return { armed: false };
      const within = (g) =>
        containsComposed(target, g) || (target.localName === "label" && !!target.control && containsComposed(target.control, g));
      const seen = { down: false, up: false, hit: "", tag: "" };
      const on = (e) => {
        const g = N.evTarget(e);
        if (!N.evTrusted(e) || !g || abs(N.mouseX(e) - x) > 2 || abs(N.mouseY(e) - y) > 2) return;
        const type = N.evType(e);
        if (type === "mousedown") seen.down = true;
        else if (type === "mouseup") seen.up = true;
        else if (!seen.hit) {
          seen.hit = within(g) ? "on" : containsComposed(g, target) ? "ancestor" : "other";
          seen.tag = Str(g.localName || "").replace(/[^a-z0-9-]/g, "");
        }
      };
      const kinds = ["mousedown", "mouseup", "click"];
      for (const t of kinds) N.listen(win, t, on, true);
      clickProbe = { seen, stop: () => { for (const t of kinds) N.unlisten(win, t, on, true); clickProbe = null; } };
      return { armed: true, docId: DOC_ID };
    }
    const seen = clickProbe ? clickProbe.seen : null;
    if (args.step === "disarm" && clickProbe) clickProbe.stop();
    const out = seen ? { down: seen.down, up: seen.up, hit: seen.hit, tag: seen.tag } : { down: false, up: false, hit: "", tag: "" };
    out.docId = DOC_ID;
    return out;
  }
  function highlight(args) {
    const el = resolve(args.ref);
    flash(el, args.ms);
    return { description: describe(el) };
  }
  function status() {
    pollResources();
    const t = now();
    return {
      docId: DOC_ID, readyState: doc.readyState, url: loc.href, title: doc.title, msSinceMutation: round(t - lastMutation), resourceCount,
      msSinceResource: round(max(0, t - lastResource)), hasFocus: attempt(() => N.hasFocus(doc), false),
    };
  }
  function clampInt(v, dflt, lo, hi) {
    const n = typeof v === "number" ? v : typeof v === "string" && v.trim() ? +v : NaN;
    return Number.isFinite(n) ? max(lo, min(hi, floor(n))) : dflt;
  }

  const OPS = createObject(null);
  OPS.snapshot = (args) => buildSnapshot(clampInt(args.maxChars, 6000, 200, 200000), args.retake === true);
  OPS.read = read;
  OPS.find = find;
  OPS.inspect = inspect;
  OPS.locate = locate;
  OPS.click = click;
  OPS.type = type;
  OPS.select = select;
  OPS.press = press;
  OPS.scroll = scroll;
  OPS.status = status;
  OPS.marks = marks;
  OPS.highlight = highlight;
  OPS.activation = activation;
  OPS.probe = probe;

  // JSON.stringify would call a toJSON that a page defines on Object.prototype, so objects are serialized by hand
  function serialize(value, depth) {
    if (value === null || value === undefined) return "null";
    const t = typeof value;
    if (t === "string") return jsonStringify(wellFormed(value));
    if (t === "boolean") return jsonStringify(value);
    if (t === "number") return Number.isFinite(value) ? jsonStringify(value) : "null";
    if (t !== "object" || depth > 12) return "null";
    if (isArray(value)) {
      let out = "[";
      for (let i = 0; i < value.length; i++) out += (i ? "," : "") + serialize(value[i], depth + 1);
      return out + "]";
    }
    let out = "{";
    let first = true;
    // an index loop: for...of runs the page's Array iterator, which it can replace
    const keys = objectKeys(value);
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      const v = value[k];
      if (v === undefined || typeof v === "function") continue;
      out += (first ? "" : ",") + jsonStringify(k) + ":" + serialize(v, depth + 1);
      first = false;
    }
    return out + "}";
  }

  function run(requestJson) {
    let res;
    try {
      const req = typeof requestJson === "string" ? jsonParse(requestJson) : requestJson;
      const op = req && typeof req.op === "string" ? req.op : "";
      const args = req && req.args && typeof req.args === "object" ? req.args : {};
      if (!op || !hasOwn(OPS, op)) {
        res = { ok: false, code: "bad_op", error: `Unknown browser op ${quote(op)}.` };
      } else {
        const value = OPS[op](args);
        // no prototype, so a setter the page puts on Object.prototype cannot swallow a field
        res = { __proto__: null, ok: true };
        for (const k in value) if (hasOwn(value, k)) res[k] = value[k];
      }
    } catch (e) {
      const msg = e && e.agentError ? e.message : attempt(() => Str((e && e.message) || e), "unknown error");
      res = e && e.agentError ? { ok: false, code: e.code, error: msg } : { ok: false, error: `The page agent failed: ${clip(msg, 300)}` };
    }
    return attempt(() => serialize(res, 0), '{"ok":false,"error":"The page agent produced an unserializable result."}');
  }

  defineProperty(win, NAME, { value: freeze({ run }), writable: false, enumerable: false, configurable: false });
})
