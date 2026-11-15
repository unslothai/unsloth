// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shortcut availability is asked on every press and every menu re-render. Each check used to be a
// document query, a whole-tree walk that on a long chat cost tens of ms per press. The elements the
// checks look for now register themselves, so a check reads those plus what sits outside the app
// root, and never walks the root's subtree.

import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  COMPOSER_INPUT_SELECTOR,
  isSurfaceBackgrounded,
  isSurfaceInForeground,
  surfaceElements,
} = await import("../src/features/settings/hooks/use-shortcut.ts");
const { isFindScopeBackgrounded, MODAL_BACKDROP_SELECTOR } = await import(
  "../src/features/find-in-page/lib/find-backgrounded.ts"
);
const {
  FIND_SCOPE_SURFACE,
  hasTrackedSurface,
  MODAL_SURFACE,
  registerSurface,
  SIDEBAR_SURFACE,
  trustSurfaceRegistry,
} = await import("../src/lib/surface-registry.ts");

const OPEN_DIALOG_SELECTOR = /const OPEN_DIALOG_SELECTOR =\n\s*'([^']+)'/.exec(
  readSrc("components/ui/sidebar.tsx"),
)?.[1] as string;

// Enough of the DOM for these selectors: attribute, :not(attribute), class, id; comma lists.
type Simple = {
  not: boolean;
  attr?: string;
  value?: string;
  cls?: string;
  id?: string;
};
function parseCompound(text: string): Simple[] {
  const out: Simple[] = [];
  const re =
    /(:not\()?(?:\[([\w-]+)(?:="([^"]*)")?\]|\.([\w-]+)|#([\w-]+))\)?/g;
  for (const m of text.matchAll(re)) {
    out.push({
      not: Boolean(m[1]),
      attr: m[2],
      value: m[3],
      cls: m[4],
      id: m[5],
    });
  }
  return out;
}

let walks = 0;
let fakeDocument: FakeDocument;

class FakeElement {
  parentElement: FakeElement | null = null;
  children: FakeElement[] = [];
  attrs = new Map<string, string>();
  constructor(attrs: Record<string, string> = {}) {
    for (const [k, v] of Object.entries(attrs)) this.attrs.set(k, v);
  }
  append(...kids: FakeElement[]): this {
    for (const kid of kids) {
      kid.parentElement = this;
      this.children.push(kid);
    }
    return this;
  }
  remove(): void {
    const parent = this.parentElement;
    if (!parent) return;
    parent.children = parent.children.filter((kid) => kid !== this);
    this.parentElement = null;
  }
  get isConnected(): boolean {
    let node: FakeElement | null = this;
    while (node.parentElement) node = node.parentElement;
    return node === fakeDocument.documentElement;
  }
  matches(selector: string): boolean {
    return selector.split(",").some((part) =>
      parseCompound(part.trim()).every((s) => {
        let hit: boolean;
        if (s.cls)
          hit = (this.attrs.get("class") ?? "").split(" ").includes(s.cls);
        else if (s.id) hit = this.attrs.get("id") === s.id;
        else
          hit =
            this.attrs.has(s.attr as string) &&
            (s.value === undefined ||
              this.attrs.get(s.attr as string) === s.value);
        return s.not ? !hit : hit;
      }),
    );
  }
  closest(selector: string): FakeElement | null {
    for (let node: FakeElement | null = this; node; node = node.parentElement) {
      if (node.matches(selector)) return node;
    }
    return null;
  }
  *descendants(): Generator<FakeElement> {
    for (const kid of this.children) {
      yield kid;
      yield* kid.descendants();
    }
  }
  querySelectorAll(selector: string): FakeElement[] {
    const out: FakeElement[] = [];
    for (const node of this.descendants()) {
      walks += 1;
      if (node.matches(selector)) out.push(node);
    }
    return out;
  }
  querySelector(selector: string): FakeElement | null {
    for (const node of this.descendants()) {
      walks += 1;
      if (node.matches(selector)) return node;
    }
    return null;
  }
}

class FakeDocument {
  documentElement = new FakeElement();
  body = new FakeElement();
  documentQueries = 0;
  constructor() {
    this.documentElement.append(this.body);
  }
  querySelectorAll(selector: string): FakeElement[] {
    this.documentQueries += 1;
    return [
      this.documentElement,
      ...this.documentElement.querySelectorAll(selector),
    ].filter((node) => node.matches(selector));
  }
  querySelector(selector: string): FakeElement | null {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

/** A long chat: thousands of rows inside the root, plus the surfaces the checks look for. */
function longChat() {
  fakeDocument = new FakeDocument();
  const root = new FakeElement({ id: "root" });
  const sidebar = new FakeElement({ "data-slot": "sidebar" });
  const scope = new FakeElement({ "data-find-scope": "" });
  const composer = new FakeElement({
    class: "aui-composer-input unsloth-composer-input",
  });
  const thread = new FakeElement();
  for (let i = 0; i < 3000; i += 1)
    thread.append(new FakeElement({ class: "row" }));
  scope.append(thread, composer);
  root.append(sidebar, scope);
  fakeDocument.body.append(root);
  const drops = [
    registerSurface(SIDEBAR_SURFACE, sidebar as unknown as Element),
    registerSurface(FIND_SCOPE_SURFACE, scope as unknown as Element),
    registerSurface(COMPOSER_INPUT_SELECTOR, composer as unknown as Element),
  ];
  return {
    root,
    sidebar,
    scope,
    composer,
    drop: () => {
      for (const d of drops) d();
    },
  };
}

/** Radix portals a modal to <body> and aria-hides the root. */
function openModal(
  root: FakeElement,
  kind: "dialog" | "alert-dialog" | "sheet" = "dialog",
) {
  const portal = new FakeElement();
  const overlay = new FakeElement({
    "data-slot": `${kind}-overlay`,
    "data-state": "open",
  });
  const content = new FakeElement({
    "data-slot": `${kind}-content`,
    "data-state": "open",
  });
  portal.append(overlay, content);
  fakeDocument.body.append(portal);
  root.attrs.set("aria-hidden", "true");
  const drops = [
    registerSurface(MODAL_SURFACE, overlay as unknown as Element),
    registerSurface(MODAL_SURFACE, content as unknown as Element),
  ];
  return {
    overlay,
    content,
    close: () => {
      overlay.attrs.set("data-state", "closed");
      content.attrs.set("data-state", "closed");
      root.attrs.delete("aria-hidden");
    },
    unmount: () => {
      for (const d of drops) d();
      portal.remove();
    },
  };
}

/** Every answer the shortcut checks give, read from the live (fake) document. */
function viaDocument() {
  const q = (s: string) =>
    fakeDocument.querySelectorAll(s) as unknown as Element[];
  const backgrounded = (s: string) => {
    const found = q(s);
    return (
      found.length > 0 &&
      found.every((el) => el.closest('[aria-hidden="true"], [inert]') !== null)
    );
  };
  return {
    findBackgrounded:
      q(MODAL_BACKDROP_SELECTOR).length > 0 || backgrounded(FIND_SCOPE_SURFACE),
    openDialog: q(OPEN_DIALOG_SELECTOR).length > 0,
    composerForeground: q(COMPOSER_INPUT_SELECTOR).some(
      (el) => !el.closest('[aria-hidden="true"], [inert]'),
    ),
    sidebarBackgrounded: backgrounded(SIDEBAR_SURFACE),
    sidebarCount: q(SIDEBAR_SURFACE).length,
  };
}

function viaChecks() {
  return {
    findBackgrounded: isFindScopeBackgrounded(),
    openDialog: hasTrackedSurface(MODAL_SURFACE, OPEN_DIALOG_SELECTOR),
    composerForeground: isSurfaceInForeground(COMPOSER_INPUT_SELECTOR),
    sidebarBackgrounded: isSurfaceBackgrounded(SIDEBAR_SURFACE),
    sidebarCount: surfaceElements(SIDEBAR_SURFACE).length,
  };
}

function withDocument(run: () => void) {
  const saved = (globalThis as { document?: unknown }).document;
  try {
    run();
  } finally {
    (globalThis as { document?: unknown }).document = saved;
    trustSurfaceRegistry(null);
  }
}

test("checks on a long chat answer without a single document query or root walk", () => {
  assert.ok(
    OPEN_DIALOG_SELECTOR,
    "sidebar.tsx still names the open dialog selector",
  );
  withDocument(() => {
    const chat = longChat();
    (globalThis as { document?: unknown }).document = fakeDocument;
    trustSurfaceRegistry(chat.root as unknown as Element);

    const check = (label: string) => {
      const expected = viaDocument();
      walks = 0;
      fakeDocument.documentQueries = 0;
      assert.deepEqual(viaChecks(), expected, label);
      assert.equal(
        fakeDocument.documentQueries,
        0,
        `${label}: a document query`,
      );
      // Only what is outside the root may be walked: portals, a handful of nodes.
      assert.ok(walks < 20, `${label}: walked ${walks} nodes`);
    };
    check("idle");
    for (const kind of ["dialog", "alert-dialog", "sheet"] as const) {
      const modal = openModal(chat.root, kind);
      check(`${kind} open`);
      modal.close();
      check(`${kind} closing (still mounted, data-state closed)`);
      modal.unmount();
      check(`${kind} gone`);
    }

    // A third-party portal (a fullscreen table) cannot register; it is outside the root, so read.
    const fullscreen = new FakeElement({ "aria-modal": "true" });
    fakeDocument.body.append(fullscreen);
    check("unregistered aria-modal portal");
    assert.equal(isFindScopeBackgrounded(), true);
    fullscreen.remove();

    // A detached element left registered (an unmount mid-handover) is not counted.
    chat.sidebar.remove();
    check("sidebar unmounted");
    assert.equal(surfaceElements(SIDEBAR_SURFACE).length, 0);
    chat.drop();
  });
});

test("without the app entry trusting it, every check still reads the document", () => {
  withDocument(() => {
    const chat = longChat();
    (globalThis as { document?: unknown }).document = fakeDocument;
    const modal = openModal(chat.root);
    fakeDocument.documentQueries = 0;
    assert.deepEqual(viaChecks(), viaDocument());
    assert.ok(fakeDocument.documentQueries > 0);
    modal.unmount();
    chat.drop();
  });
});

test("registration is refcounted and its unregister idempotent", () => {
  withDocument(() => {
    const chat = longChat();
    (globalThis as { document?: unknown }).document = fakeDocument;
    trustSurfaceRegistry(chat.root as unknown as Element);
    const extra = registerSurface(
      SIDEBAR_SURFACE,
      chat.sidebar as unknown as Element,
    );
    extra();
    extra();
    assert.equal(
      surfaceElements(SIDEBAR_SURFACE).length,
      1,
      "the first registration holds",
    );
    chat.drop();
    assert.equal(surfaceElements(SIDEBAR_SURFACE).length, 0);
  });
});

// The registry is only as good as its producers: every element in the app a check can match must
// register through useSurfaceRef, or a check would miss it inside the root.
test("every producer of a tracked selector registers itself", () => {
  const srcDir = fileURLToPath(new URL("../src", import.meta.url));
  const files: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (/\.tsx$/.test(name)) files.push(path);
    }
  };
  walk(srcDir);
  const producers = [
    // Not after "[": that is a selector reading them, not an element.
    /(?<!\[)data-slot="(?:alert-)?dialog-(?:overlay|content)"/,
    /(?<!\[)data-slot="sheet-overlay"/,
    /(?<!\[)data-slot="sidebar"/,
    /(?<!\[)aria-modal(?:=\{true\}|="true")/,
    /(?<!\[)data-blocking-screen=""/,
    /className="[^"]*\baui-composer-input\b/,
    /\[FIND_SCOPE_ATTRIBUTE\]: ""/,
  ];
  let seen = 0;
  for (const path of files) {
    const lines = readFileSync(path, "utf8").split("\n");
    const refs = new Set(
      [...lines.join("\n").matchAll(/const (\w+) = useSurfaceRef\b/g)].map(
        (m) => m[1],
      ),
    );
    lines.forEach((line, i) => {
      if (/^\s*(\/\/|\*)/.test(line) || !producers.some((re) => re.test(line)))
        return;
      seen += 1;
      // From the element's opening "<Tag" to just past the attribute: a ref may sit either side.
      let start = i;
      while (start > 0 && !/^\s*<[A-Za-z][\w.]*(\s|$)/.test(lines[start]))
        start -= 1;
      const tag = lines.slice(start, i + 10).join("\n");
      const registered = [...tag.matchAll(/\bref=\{(\w+)\}/g)].some((m) =>
        refs.has(m[1]),
      );
      assert.ok(
        registered,
        `${path}:${i + 1} renders a tracked surface without useSurfaceRef`,
      );
    });
  }
  assert.ok(
    seen >= 12,
    `found only ${seen} producers; the scan lost track of them`,
  );
});
