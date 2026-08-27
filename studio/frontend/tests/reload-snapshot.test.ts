// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const script = readFileSync(
  new URL("../public/reload-snapshot.js", import.meta.url),
  "utf8",
);
const indexHtml = readFileSync(
  new URL("../index.html", import.meta.url),
  "utf8",
);
const indexCss = readFileSync(
  new URL("../src/index.css", import.meta.url),
  "utf8",
);
const rootRouteSource = readFileSync(
  new URL("../src/app/routes/__root.tsx", import.meta.url),
  "utf8",
);
const runtimeProviderSource = readFileSync(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
  "utf8",
);
const chatPageSource = readFileSync(
  new URL("../src/features/chat/chat-page.tsx", import.meta.url),
  "utf8",
);
const sharedComposerSource = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);
const imageDropzoneSource = readFileSync(
  new URL("../src/components/image-dropzone.tsx", import.meta.url),
  "utf8",
);
const attachmentPreviewSource = readFileSync(
  new URL(
    "../src/components/assistant-ui/attachment-preview.tsx",
    import.meta.url,
  ),
  "utf8",
);

const attachmentSource = readFileSync(
  new URL("../src/components/assistant-ui/attachment.tsx", import.meta.url),
  "utf8",
);
const imagesPageSource = readFileSync(
  new URL("../src/features/images/images-page.tsx", import.meta.url),
  "utf8",
);
const videoPageSource = readFileSync(
  new URL("../src/features/video/video-page.tsx", import.meta.url),
  "utf8",
);
const audioPageSource = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);
const hubPageSource = readFileSync(
  new URL("../src/features/hub/hub-page.tsx", import.meta.url),
  "utf8",
);
const referencePickerSource = readFileSync(
  new URL("../src/features/video/reference-picker.tsx", import.meta.url),
  "utf8",
);

const unstructuredDropZoneSource = readFileSync(
  new URL(
    "../src/features/recipe-studio/dialogs/seed/unstructured-drop-zone.tsx",
    import.meta.url,
  ),
  "utf8",
);

const projectsPageSource = readFileSync(
  new URL("../src/features/chat/projects-page.tsx", import.meta.url),
  "utf8",
);

const seedDialogSource = readFileSync(
  new URL(
    "../src/features/recipe-studio/dialogs/seed/seed-dialog.tsx",
    import.meta.url,
  ),
  "utf8",
);

const projectSourceDropzoneSource = readFileSync(
  new URL(
    "../src/features/rag/components/project-source-dropzone.tsx",
    import.meta.url,
  ),
  "utf8",
);
const dataRecipesPageSource = readFileSync(
  new URL(
    "../src/features/data-recipes/pages/data-recipes-page.tsx",
    import.meta.url,
  ),
  "utf8",
);
const editRecipePageSource = readFileSync(
  new URL(
    "../src/features/data-recipes/pages/edit-recipe-page.tsx",
    import.meta.url,
  ),
  "utf8",
);
const exportPageSource = readFileSync(
  new URL("../src/features/export/export-page.tsx", import.meta.url),
  "utf8",
);
const studioPageSource = readFileSync(
  new URL("../src/features/studio/studio-page.tsx", import.meta.url),
  "utf8",
);
const apiMonitorPageSource = readFileSync(
  new URL("../src/features/api-monitor/api-monitor-page.tsx", import.meta.url),
  "utf8",
);
const authFormSource = readFileSync(
  new URL("../src/features/auth/components/auth-form.tsx", import.meta.url),
  "utf8",
);

type Listener = (event: Record<string, unknown>) => void;

/** A node of the fake #root subtree the capture walks. */
interface ElementSpec {
  tag: string;
  /** Live DOM state React writes as a property rather than an attribute. */
  value?: string;
  checked?: boolean;
  selected?: boolean;
  type?: string;
  autocomplete?: string;
  attributes?: Record<string, string>;
  display?: string;
  visibility?: string;
  overflowX?: string;
  overflowY?: string;
  scrollTop?: number;
  scrollLeft?: number;
  scrollHeight?: number;
  scrollWidth?: number;
  clientHeight?: number;
  clientWidth?: number;
  currentSrc?: string;
  naturalWidth?: number;
  naturalHeight?: number;
  videoWidth?: number;
  videoHeight?: number;
  width?: number;
  height?: number;
  /** [top, right, bottom, left]. Ignored for a `display: contents` box. */
  rect?: [number, number, number, number];
  text?: string;
  children?: ElementSpec[];
}

interface ShadowStub {
  mode: string;
  children: Array<Record<string, unknown>>;
  appendChild(child: Record<string, unknown>): Record<string, unknown>;
}

interface StubElement {
  tagName: string;
  spec: ElementSpec;
  value: string;
  checked: boolean;
  selected: boolean;
  type: string;
  autocomplete: string;
  scrollTop: number;
  scrollLeft: number;
  scrollHeight: number;
  scrollWidth: number;
  clientHeight: number;
  clientWidth: number;
  currentSrc: string;
  naturalWidth: number;
  naturalHeight: number;
  videoWidth: number;
  videoHeight: number;
  width: number;
  height: number;
  textContent: string;
  attributeOverrides: Record<string, string>;
  hasAttribute(name: string): boolean;
  getAttribute(name: string): string | null;
  setAttribute(name: string, value: string): void;
  children: StubElement[];
  parent: StubElement | null;
  readonly parentElement: StubElement | null;
  readonly nextElementSibling: StubElement | null;
  attributes: Array<{ name: string }>;
  closest(selector: string): StubElement | null;
  getBoundingClientRect(): {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  remove(): void;
  replaceChildren(): void;
  removeAttribute(name: string): void;
  querySelectorAll(selector: string): StubElement[];
  cloneNode(deep: boolean): StubElement;
  readonly firstElementChild: StubElement | undefined;
  readonly innerHTML: string;
  readonly outerHTML: string;
}

function createElement(spec: ElementSpec, parent: StubElement | null = null) {
  const element = {
    tagName: spec.tag.toUpperCase(),
    spec,
    value: spec.value ?? "",
    checked: spec.checked ?? false,
    selected: spec.selected ?? false,
    type: spec.type ?? "text",
    autocomplete: spec.autocomplete ?? "",
    scrollTop: spec.scrollTop ?? 0,
    scrollLeft: spec.scrollLeft ?? 0,
    scrollHeight: spec.scrollHeight ?? spec.clientHeight ?? 900,
    scrollWidth: spec.scrollWidth ?? spec.clientWidth ?? 1440,
    clientHeight: spec.clientHeight ?? 900,
    clientWidth: spec.clientWidth ?? 1440,
    currentSrc: spec.currentSrc ?? spec.attributes?.src ?? "",
    naturalWidth: spec.naturalWidth ?? 0,
    naturalHeight: spec.naturalHeight ?? 0,
    videoWidth: spec.videoWidth ?? 0,
    videoHeight: spec.videoHeight ?? 0,
    width: spec.width ?? 0,
    height: spec.height ?? 0,
    textContent: spec.text ?? "",
    children: [] as StubElement[],
    parent,
    attributeOverrides: {
      ...(spec.attributes ?? {}),
      ...(spec.autocomplete ? { autocomplete: spec.autocomplete } : {}),
    } as Record<string, string>,
    get attributes() {
      return Object.keys(element.attributeOverrides).map((name) => ({ name }));
    },
    hasAttribute(name: string) {
      return Object.hasOwn(element.attributeOverrides, name);
    },
    getAttribute(name: string) {
      return element.attributeOverrides[name] ?? null;
    },
    closest(selector: string) {
      let node: StubElement | null = element;
      while (node) {
        if (node.tagName.toLowerCase() === selector) return node;
        node = node.parent;
      }
      return null;
    },
    getBoundingClientRect() {
      // Chromium, Firefox and WebKit all report an empty rectangle for a box
      // that is not laid out, whatever its children cover.
      const [top, right, bottom, left] =
        spec.display === "contents"
          ? [0, 0, 0, 0]
          : (spec.rect ?? [0, 1440, 900, 0]);
      return { top, right, bottom, left };
    },
    remove() {
      if (!element.parent) return;
      element.parent.children = element.parent.children.filter(
        (child) => child !== element,
      );
      element.parent = null;
    },
    replaceChildren() {
      for (const child of element.children) child.parent = null;
      element.children = [];
      element.textContent = "";
    },
    setAttribute(name: string, value: string) {
      element.attributeOverrides[name] = value;
    },
    removeAttribute(name: string) {
      delete element.attributeOverrides[name];
    },
    get parentElement() {
      return element.parent;
    },
    get nextElementSibling() {
      if (!element.parent) return null;
      const index = element.parent.children.indexOf(element);
      return element.parent.children[index + 1] ?? null;
    },
    querySelectorAll(selector: string) {
      const wanted = selector.split(",").map((name) => name.trim());
      const found: StubElement[] = [];
      const walk = (node: StubElement) => {
        for (const child of node.children) {
          if (
            selector === "*" ||
            wanted.includes(child.tagName.toLowerCase()) ||
            wanted.some(
              (name) =>
                name.startsWith("[") &&
                name.endsWith("]") &&
                child.hasAttribute(name.slice(1, -1)),
            )
          ) {
            found.push(child);
          }
          walk(child);
        }
      };
      walk(element);
      return found;
    },
    cloneNode() {
      return createElement(spec);
    },
    get firstElementChild() {
      return element.children[0];
    },
    get innerHTML() {
      return element.children
        .map((child) => child.outerHTML)
        .concat(element.textContent)
        .join("");
    },
    get outerHTML() {
      const attributes = Object.entries(element.attributeOverrides)
        .map(([name, value]) => ` ${name}="${value}"`)
        .join("");
      return `<${spec.tag}${attributes}>${element.innerHTML}</${spec.tag}>`;
    },
  } as StubElement;
  element.children = (spec.children ?? []).map((child) =>
    createElement(child, element),
  );
  return element;
}

/** Enough of CSSStyleDeclaration for the inline custom properties on <html>. */
type StubStyle = {
  length: number;
  getPropertyValue(name: string): string;
  setProperty(name: string, value: string): void;
  removeProperty(name: string): void;
} & Record<number, string>;

function createStyle(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial));
  const style = {
    getPropertyValue: (name: string) => values.get(name) ?? "",
    setProperty: (name: string, value: string) => {
      values.set(name, value);
      index();
    },
    removeProperty: (name: string) => {
      values.delete(name);
      index();
    },
    get length() {
      return values.size;
    },
  } as StubStyle;
  const index = () => {
    [...values.keys()].forEach((name, position) => {
      style[position] = name;
    });
  };
  index();
  return style;
}

function createEnvironment(options: {
  navigationType: "navigate" | "reload";
  storage?: Map<string, string>;
  rootHtml?: string;
  rootTree?: ElementSpec;
  bodyPortals?: ElementSpec[];
  viewport?: { width: number; height: number };
  htmlVariables?: Record<string, string>;
  htmlAttributes?: Record<string, string>;
  styleSheets?: string[];
  frameDataBody?: string;
  inlineStyleSheets?: string[];
  trackScrollRestores?: boolean;
  localStorage?: Map<string, string>;
  supportsPageSwap?: boolean;
}) {
  const storage = options.storage ?? new Map<string, string>();
  const listeners = new Map<string, Listener[]>();
  const animationFrames: Array<() => void> = [];
  const fontFaces: Array<{ family: string; source: string }> = [];
  const appended: Array<{ innerHTML: string; removed: boolean }> = [];
  let scrollRestoreCalls = 0;
  const bodyTree = options.rootTree
    ? createElement({
        tag: "body",
        children: [
          {
            ...options.rootTree,
            attributes: { ...options.rootTree.attributes, id: "root" },
          },
          ...(options.bodyPortals ?? []),
        ],
      })
    : null;
  const clone = {
    innerHTML: `<div id="root">${options.rootHtml ?? "<main>Chat is ready</main>"}</div>`,
    querySelectorAll: () => [],
  };
  const root = bodyTree?.children[0] ?? {
    firstElementChild: {},
    querySelectorAll: () => [],
  };
  const body = bodyTree ?? {
    cloneNode: () => clone,
    querySelectorAll: () => [],
  };

  const window = {
    addEventListener(name: string, listener: Listener) {
      const current = listeners.get(name) ?? [];
      current.push(listener);
      listeners.set(name, current);
    },
  };
  if (options.supportsPageSwap) {
    Object.assign(window, { onpageswap: null });
  }
  const htmlAttributes = new Map(Object.entries(options.htmlAttributes ?? {}));
  const documentElement = {
    style: createStyle(options.htmlVariables),
    hasAttribute: (name: string) => htmlAttributes.has(name),
    getAttribute: (name: string) => htmlAttributes.get(name) ?? null,
    setAttribute: (name: string, value: string) => {
      htmlAttributes.set(name, value);
    },
    removeAttribute: (name: string) => {
      htmlAttributes.delete(name);
    },
    appendChild(element: { innerHTML: string; removed: boolean }) {
      appended.push(element);
    },
  };
  const document = {
    body,
    documentElement,
    fonts: {
      add(face: { family: string; source: string }) {
        fontFaces.push(face);
      },
    },
    getElementById: () => root,
    querySelectorAll: (selector: string) => {
      if (selector === 'link[rel="stylesheet"]') {
        return (options.styleSheets ?? []).map((href) => ({ href }));
      }
      if (selector === "style[data-vite-dev-id]") {
        return (options.inlineStyleSheets ?? []).map((textContent) => ({
          textContent,
        }));
      }
      return [];
    },
    createElement: (tag?: string) => {
      if (tag === "canvas") {
        return {
          width: 0,
          height: 0,
          getContext: () => ({ drawImage() {} }),
          toDataURL: () =>
            "data:image/webp;base64," +
            (options.frameDataBody ?? "retained-frame"),
        };
      }
      const element = {
        tagName: (tag ?? "div").toUpperCase(),
        className: "",
        attributes: {} as Record<string, string>,
        style: {
          values: {} as Record<string, string>,
          setProperty(name: string, value: string) {
            element.style.values[name] = value;
          },
        },
        rel: "",
        href: "",
        onload: null as (() => void) | null,
        onerror: null as (() => void) | null,
        inert: false,
        innerHTML: "",
        textContent: "",
        children: [] as Record<string, unknown>[],
        removed: false,
        shadow: null as ShadowStub | null,
        attachShadow(init: { mode: string }) {
          const shadow: ShadowStub = {
            mode: init.mode,
            children: [],
            appendChild(child: Record<string, unknown>) {
              shadow.children.push(child);
              return child;
            },
          };
          element.shadow = shadow;
          return shadow;
        },
        setAttribute(name: string, value: string) {
          element.attributes[name] = value;
        },
        getAttribute(name: string) {
          return element.attributes[name] ?? null;
        },
        querySelectorAll(selector?: string) {
          if (
            tag === "body" &&
            options.trackScrollRestores &&
            selector ===
              "[data-reload-scroll-top], [data-reload-scroll-left]"
          ) {
            scrollRestoreCalls += 1;
            return [
              {
                scrollTop: 0,
                scrollLeft: 0,
                getAttribute(name: string) {
                  return name === "data-reload-scroll-top" ? "600" : null;
                },
              },
            ];
          }
          return [];
        },
        appendChild(child: Record<string, unknown>) {
          element.children.push(child);
          return child;
        },
        remove() {
          element.removed = true;
        },
      };
      return element;
    },
  };

  vm.runInNewContext(script, {
    Array,
    Date,
    JSON,
    Object,
    clearTimeout() {
      // Timer bookkeeping is outside this lifecycle test.
    },
    document,
    getComputedStyle: (element: StubElement) =>
      // <html> resolves to its own declaration block, which is where the
      // design tokens the copy has to carry are read from.
      (element as unknown) === documentElement
        ? documentElement.style
        : {
            display: element.spec?.display ?? "block",
            visibility: element.spec?.visibility ?? "visible",
            overflowX: element.spec?.overflowX ?? "visible",
            overflowY: element.spec?.overflowY ?? "visible",
          },
    innerHeight: options.viewport?.height ?? 900,
    innerWidth: options.viewport?.width ?? 1440,
    location: { pathname: "/chat", search: "" },
    localStorage: {
      getItem: (key: string) => options.localStorage?.get(key) ?? null,
    },
    FontFace: class {
      family: string;
      source: string;
      constructor(family: string, source: string) {
        this.family = family;
        this.source = source;
      }
      load() {
        return Promise.resolve(this);
      }
    },
    performance: {
      getEntriesByType: () => [{ type: options.navigationType }],
    },
    requestAnimationFrame(callback: () => void) {
      animationFrames.push(callback);
    },
    sessionStorage: {
      getItem: (key: string) => storage.get(key) ?? null,
      removeItem: (key: string) => storage.delete(key),
      setItem: (key: string, value: string) => storage.set(key, value),
    },
    setTimeout: () => 1,
    window,
  });

  return {
    storage,
    fontFaces,
    appended,
    htmlVariables: documentElement.style,
    htmlAttributes,
    get scrollRestoreCalls() {
      return scrollRestoreCalls;
    },
    /** The host element of the retained shell, if one was restored. */
    get shell() {
      const host = appended[0] as unknown as
        | {
            shadow: ShadowStub | null;
            removed: boolean;
            style: Record<string, unknown>;
          }
        | undefined;
      if (!host?.shadow) return null;
      const children = host.shadow.children;
      const root = children[children.length - 1];
      const body = (
        root?.children as Record<string, unknown>[] | undefined
      )?.find((child) => child.tagName === "BODY");
      return {
        host,
        hostStyle: host.style,
        mode: host.shadow.mode,
        stylesheets: children
          .filter((child) => child.rel === "stylesheet")
          .map((child) => child.href as string),
        inlineStyles: children
          .filter((child) => child.tagName === "STYLE")
          .map((child) => child.textContent as string),
        html: (body?.innerHTML ?? root?.innerHTML ?? "") as string,
        bodyTag: (body?.tagName ?? "") as string,
        rootClass: (root?.className ?? "") as string,
        rootTag: (root?.tagName ?? "") as string,
        rootAttributes: (root?.attributes ?? {}) as Record<string, string>,
        rootTokens: ((
          root?.style as
            | { values?: Record<string, string> }
            | undefined
        )?.values ?? {}) as Record<string, string>,
        rootVisibility: (root?.style as { visibility?: string } | undefined)
          ?.visibility,
      };
    },
    dispatch(name: string, event: Record<string, unknown> = {}) {
      for (const listener of listeners.get(name) ?? []) {
        listener(event);
      }
    },
    loadStyleSheets() {
      const host = appended[0] as unknown as
        | { shadow: ShadowStub | null }
        | undefined;
      for (const child of host?.shadow?.children ?? []) {
        if (child.rel === "stylesheet") {
          (child.onload as (() => void) | null)?.();
        }
      }
    },
    runAnimationFrame() {
      const callbacks = animationFrames.splice(0);
      for (const callback of callbacks) {
        callback();
      }
    },
  };
}

function storedSnapshot(storage: Map<string, string>) {
  const raw = storage.get("unsloth.reload-snapshot.v1");
  assert.ok(raw, "expected a stored snapshot");
  return JSON.parse(raw) as {
    html: string;
    styles?: string[];
    inlineStyles?: string[];
    rootClass?: string;
    tokens?: Record<string, string>;
    appearance?: {
      variables: Record<string, string>;
      attributes: Record<string, string>;
    };
  };
}

test("carries the rendered shell through a reload until the new shell is ready", () => {
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Existing chat</main>",
    styleSheets: ["/assets/index-abc123.css"],
    supportsPageSwap: true,
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.equal(incoming.appended.length, 1);
  assert.equal(
    incoming.shell?.html,
    '<div id="root"><main>Existing chat</main></div>',
  );
  assert.equal(incoming.storage.size, 0);

  incoming.dispatch("unsloth:app-shell-ready");
  assert.equal(incoming.appended[0].removed, false);
  incoming.runAnimationFrame();
  assert.equal(incoming.appended[0].removed, false);
  incoming.runAnimationFrame();
  assert.equal(incoming.appended[0].removed, true);
});

test("captures reloads through WebKit's pagehide fallback", () => {
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>WebKit chat</main>",
    styleSheets: ["/assets/index-abc123.css"],
  });
  outgoing.dispatch("pagehide", { persisted: false });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.match(incoming.shell?.html ?? "", /WebKit chat/);
});

test("does not run the pagehide fallback when pageswap is available", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Chromium chat</main>",
    styleSheets: ["/assets/index-abc123.css"],
    supportsPageSwap: true,
  });
  environment.dispatch("pagehide", { persisted: false });
  assert.equal(environment.storage.size, 0);
});

test("carries the retained shell through consecutive reloads", () => {
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Stable chat</main>",
    styleSheets: ["/assets/index-abc123.css"],
    supportsPageSwap: true,
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const firstReload = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
    rootHtml: "<main>Still loading</main>",
    supportsPageSwap: true,
  });
  firstReload.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });
  firstReload.dispatch("pagehide", { persisted: false });

  const secondReload = createEnvironment({
    navigationType: "reload",
    storage: firstReload.storage,
  });
  assert.match(secondReload.shell?.html ?? "", /Stable chat/);
  assert.doesNotMatch(secondReload.shell?.html ?? "", /Still loading/);
});

test("restores from a parser-blocking head script before the first body paint", () => {
  const scriptPosition = indexHtml.indexOf(
    '<script src="/reload-snapshot.js">',
  );
  assert.ok(scriptPosition > 0);
  assert.ok(scriptPosition < indexHtml.indexOf("</head>"));
});

test("does not retain the shell for a non-reload navigation", () => {
  const environment = createEnvironment({ navigationType: "navigate" });
  environment.dispatch("pageswap", {
    activation: { navigationType: "push" },
  });
  assert.equal(environment.storage.size, 0);
});

test("never persists a Temporary Chat shell", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Sensitive temporary transcript</main>",
    htmlAttributes: { "data-reload-snapshot-private": "" },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });
  assert.equal(environment.storage.size, 0);
});

test("mirrors Temporary Chat privacy and lets history completion retire chat shells", () => {
  assert.match(
    rootRouteSource,
    /toggleAttribute\(\s*"data-reload-snapshot-private",\s*incognito,/,
  );
  assert.equal(
    [...rootRouteSource.matchAll(/<ReloadSnapshotReady \/>/g)].length,
    1,
    "chat must not use the generic commit-time readiness marker",
  );
  assert.match(
    runtimeProviderSource,
    /async load\(\) \{[\s\S]*?const completeLoad =[\s\S]*?unsloth:app-shell-ready[\s\S]*?await loadGenerationOverlaySnapshot\([\s\S]*?listStoredChatMessages[\s\S]*?return completeLoad/,
  );
  assert.match(
    runtimeProviderSource,
    /!reloadReadyThreadId \|\| loadedThreadId === reloadReadyThreadId/,
  );
  // The rejection arm is a .then(onFulfilled, onRejected) pair since #8908, which needs
  // both outcomes: that PR retires the switch's claim on either. What this guards is
  // unchanged -- a failed switchToThread still releases the retained shell -- and the
  // call must stay AHEAD of that PR's staleness guard, since a superseded attempt is
  // still an attempt that ended and the shell would otherwise wait for ever.
  assert.match(
    runtimeProviderSource,
    /switchToThread\(threadId\)[\s\S]*?\.then\([\s\S]*?onSwitchFailed\?\.\(\)/,
  );
  const rejectionArm = runtimeProviderSource.slice(
    runtimeProviderSource.indexOf("onSwitchFailed?.()"),
  );
  assert.ok(
    rejectionArm.indexOf("onSwitchFailed?.()") <
      rejectionArm.indexOf("attempt !== attemptAtStart"),
    "onSwitchFailed must fire before the staleness guard can return early",
  );
  assert.match(
    runtimeProviderSource,
    /const signalFailedInitialSwitchReady = useCallback[\s\S]*?onInitialHistoryReady\(\)[\s\S]*?unsloth:app-shell-ready[\s\S]*?onSwitchFailed=\{signalFailedInitialSwitchReady\}/,
  );
  assert.match(
    runtimeProviderSource,
    /createRuntimeHook\([\s\S]*?modelType,[\s\S]*?pairId,[\s\S]*?initialThreadId,[\s\S]*?onInitialHistoryReady/,
  );
  // A compare pane reports readiness through onInitialHistoryReady, and its
  // runtime bootstraps on an empty thread first, so this branch must check the
  // thread or both panes go ready while their conversations still load.
  assert.match(
    runtimeProviderSource,
    /const completeLoad =[\s\S]*?if \(onInitialHistoryReady\) \{\s*if \(loadedTheRequestedThread\) onInitialHistoryReady\(\);/,
  );
  assert.match(
    runtimeProviderSource,
    /const signalFailedInitialSwitchReady = useCallback[\s\S]*?if \(onInitialHistoryReady\) \{\s*onInitialHistoryReady\(\);/,
  );
  assert.match(
    chatPageSource,
    /state\.panes\.add\(pane\)[\s\S]*?state\.panes\.size < 2[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(
    chatPageSource,
    /\.finally\(\(\) => \{\s*if \(isActive\) setThreadsSettled\(true\)/,
  );
  assert.match(
    chatPageSource,
    /if \(!threadsSettled\) return;\s*if \(!baseThreadId\) markInitialHistoryReady\("base"\);\s*if \(!loraThreadId\) markInitialHistoryReady\("lora"\);/,
  );
  assert.match(
    chatPageSource,
    /if \(!threadsSettled\) return;\s*if \(!model1ThreadId\) markInitialHistoryReady\("model1"\);\s*if \(!model2ThreadId\) markInitialHistoryReady\("model2"\);/,
  );
  assert.match(
    chatPageSource,
    /function useCompareVariant\(pairId: string\)[\s\S]*?residentCheckpoint === undefined\) return null;[\s\S]*?s\.loras\.some[\s\S]*?s\.models\.some\(\(model\) => model\.id === cp && model\.isLora\)[\s\S]*?variant: CompareVariant \| null;[\s\S]*?thread\.modelType === "model1" \|\| thread\.modelType === "model2"[\s\S]*?thread\.modelType === "base" \|\| thread\.modelType === "lora"/,
  );
  assert.match(
    chatPageSource,
    /const compareVariant = useCompareVariant\(pairId\);\s*if \(compareVariant === null\) return <><\/>;\s*return compareVariant === "lora" \?/,
  );
  assert.match(
    chatPageSource,
    /for \(let attempt = 0; attempt < 2; attempt \+= 1\)[\s\S]*?catch \(error\) \{\s*if \(!isActive\) return;\s*if \(!isExpectedBackgroundChatStorageError\(error\)\) throw error;/,
  );
  const compareVariantSource = chatPageSource.slice(
    chatPageSource.indexOf("function useCompareVariant"),
    chatPageSource.indexOf("const CompareContent"),
  );
  assert.doesNotMatch(compareVariantSource, /catch \(error\)[\s\S]*?setResolution/);
  assert.match(
    compareVariantSource,
    /if \(resolution\?\.pairId !== pairId\) return null;\s*return resolution\.variant \?\? \(modelsError \? "general" : null\);/,
  );
  const loraCompareSource = chatPageSource.slice(
    chatPageSource.indexOf("const LoraCompareContent"),
    chatPageSource.indexOf("function GeneralCompareHeader"),
  );
  assert.doesNotMatch(loraCompareSource, /modelType === "model[12]"/);
  const generalCompareSource = chatPageSource.slice(
    chatPageSource.indexOf("const GeneralCompareContent"),
    chatPageSource.indexOf("const PROJECT_CHAT_EXPORT_OPTIONS"),
  );
  assert.doesNotMatch(generalCompareSource, /modelType === "(?:base|lora)"/);
  assert.match(
    chatPageSource,
    /const previewsReady = items\.every[\s\S]*?!dataLoaded \|\|[\s\S]*?!runtimeReady \|\|[\s\S]*?!previewsReady \|\|[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(
    chatPageSource,
    /dataLoaded=\{currentProjectItemsLoaded && !projectsLoading\}/,
  );
});

test("media pages retire the shell only after gallery and preview hydration", () => {
  assert.doesNotMatch(
    rootRouteSource,
    /<RouteBoundary>\s*<(?:ImagesPage|VideoPage|AudioPage)/,
  );
  assert.match(
    imagesPageSource,
    /Promise\.all\(\[\s*refreshStatus\(\)[\s\S]*?await loadGallery\(\)[\s\S]*?await ensureSrc\(initialSelection\)[\s\S]*?onInitialReady\?\.\(\)/,
  );
  assert.match(
    videoPageSource,
    /Promise\.all\(\[\s*refreshStatus\(\)[\s\S]*?await loadGallery\(\)[\s\S]*?await ensureSrc\(initialSelection\)[\s\S]*?onInitialReady\?\.\(\)/,
  );
  assert.match(
    audioPageSource,
    /Promise\.all\(\[\s*refreshGallery\(\),\s*refreshStatus\(\),\s*refreshSttStatus\(\)[\s\S]*?await ensureClipSrc\(initialSelection\)[\s\S]*?onInitialReady\?\.\(\)/,
  );
  assert.match(
    imagesPageSource,
    /if \(initialReadySent\.current\) \{\s*void refreshStatus\(\)/,
  );
  assert.match(
    videoPageSource,
    /if \(initialReadySent\.current\) \{\s*void refreshStatus\(\)/,
  );
  assert.match(
    audioPageSource,
    /if \(initialReadySent\.current\) \{\s*void refreshStatus\(\);\s*void refreshSttStatus\(\);\s*void refreshGallery\(\)/,
  );
  assert.match(
    audioPageSource,
    /mode === "transcribe"[\s\S]*?data-reload-snapshot-sensitive=\{\s*transcript \|\| transcribedName \? "" : undefined/,
  );
});

test("data-backed routes own reload readiness until hydration settles", () => {
  assert.match(
    rootRouteSource,
    /const routeOwnsReloadReadiness =\s*pathname === "\/hub" \|\|\s*pathname === "\/projects"/,
  );
  assert.match(
    rootRouteSource,
    /readyWhenCommitted=\{!routeOwnsReloadReadiness\}/,
  );
  assert.match(
    hubPageSource,
    /!initialResidentStatusSettled[\s\S]*?\(isDiscoverTab \? isLoading : !inventorySettled\)[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(
    projectsPageSource,
    /if \(!hasLoaded \|\| reloadReadySent\.current\) \{\s*return;\s*\}[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(
    rootRouteSource,
    /pathname === "\/data-recipes" \|\|\s*pathname\.startsWith\("\/data-recipes\/"\)/,
  );
  assert.match(
    dataRecipesPageSource,
    /if \(!ready \|\| reloadReadySent\.current\) \{\s*return;\s*\}[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(
    editRecipePageSource,
    /if \(loadState\.status === "loading" \|\| reloadReadySent\.current\) \{\s*return;\s*\}[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(rootRouteSource, /pathname === "\/export"/);
  assert.match(
    exportPageSource,
    /loadingCheckpoints \|\|[\s\S]*?isLoadingLocalModels \|\|[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(rootRouteSource, /pathname === "\/studio"/);
  assert.match(
    studioPageSource,
    /capabilitiesUnknown \|\|[\s\S]*?!hasHydratedRuntime \|\|[\s\S]*?isHydratingRuntime \|\|[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(rootRouteSource, /pathname === "\/api-monitor"/);
  assert.match(
    apiMonitorPageSource,
    /if \(loading \|\| reloadReadySent\.current\) \{\s*return;\s*\}[\s\S]*?unsloth:app-shell-ready/,
  );
  assert.match(rootRouteSource, /pathname === "\/login"/);
  assert.match(rootRouteSource, /pathname === "\/change-password"/);
  assert.match(
    authFormSource,
    /if \(statusLoading \|\| reloadReadySent\.current\) return;[\s\S]*?unsloth:app-shell-ready/,
  );
});

test("captures Vite's injected development stylesheet", () => {
  const css = ".reload-snapshot-shell { color: rgb(1 2 3); }";
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Development shell</main>",
    inlineStyleSheets: [css],
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const snapshot = storedSnapshot(outgoing.storage);
  assert.deepEqual(snapshot.styles, []);
  assert.deepEqual(snapshot.inlineStyles, [css]);

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.deepEqual(incoming.shell?.inlineStyles, [css]);
});

test("positions the retained host before development CSS arrives", () => {
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Development shell</main>",
    inlineStyleSheets: [".reload-snapshot-shell { display: block; }"],
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.equal(incoming.shell?.hostStyle.position, "fixed");
  assert.equal(incoming.shell?.hostStyle.inset, "0");
  assert.equal(incoming.shell?.hostStyle.zIndex, "2147483647");
  assert.equal(incoming.shell?.hostStyle.pointerEvents, "none");
  assert.equal(incoming.shell?.hostStyle.background, "var(--background)");
});

test("restores scroll offsets again after linked styles load", () => {
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Scrolled shell</main>",
    styleSheets: ["/assets/index-abc123.css"],
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
    trackScrollRestores: true,
  });
  assert.equal(incoming.scrollRestoreCalls, 1);
  incoming.runAnimationFrame();
  assert.equal(incoming.scrollRestoreCalls, 2);
  incoming.loadStyleSheets();
  assert.equal(incoming.scrollRestoreCalls, 3);
  incoming.loadStyleSheets();
  assert.equal(incoming.scrollRestoreCalls, 3);
});

test("keeps what a display:contents wrapper renders, drops what is offscreen", () => {
  // A `display: contents` box is not laid out, so its rectangle is empty even
  // while its children fill the viewport: /studio wraps its whole page in one.
  const environment = createEnvironment({
    navigationType: "navigate",
    rootTree: {
      tag: "div",
      rect: [0, 1440, 900, 0],
      children: [
        {
          tag: "div",
          display: "contents",
          children: [
            { tag: "main", rect: [0, 1440, 900, 0], text: "Unsloth is ready" },
          ],
        },
        { tag: "aside", rect: [-400, 1440, -100, 0], text: "Scrolled past" },
        { tag: "footer", display: "none", text: "Collapsed" },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /Unsloth is ready/);
  assert.doesNotMatch(html, /Scrolled past/);
  assert.doesNotMatch(html, /Collapsed/);
});

test("carries the appearance customization so the shell paints in its own colors", () => {
  // theme-boot.js resolves mode and palette only; the rest lands in a React
  // effect, which is well after the restored shell has painted.
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Existing chat</main>",
    styleSheets: ["/assets/index-abc123.css"],
    htmlVariables: {
      "--background": "#241b2f",
      "--font-sans": '"Inter"',
      "--studio-sidebar-live-width": "317px",
    },
    htmlAttributes: {
      "data-contrast-adjust": "",
      "data-panel-resizing": "true",
    },
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { appearance } = storedSnapshot(outgoing.storage);
  assert.deepEqual(appearance?.variables, {
    "--background": "#241b2f",
    "--font-sans": '"Inter"',
  });
  assert.deepEqual(appearance?.attributes, { "data-contrast-adjust": "" });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.equal(
    incoming.htmlVariables.getPropertyValue("--background"),
    "#241b2f",
  );
  assert.equal(incoming.htmlAttributes.get("data-contrast-adjust"), "");
  // Transient state (a panel mid-drag) is not appearance and must not be replayed.
  assert.equal(incoming.htmlAttributes.has("data-panel-resizing"), false);
  assert.equal(
    incoming.htmlVariables.getPropertyValue("--studio-sidebar-live-width"),
    "",
  );
});

test("adapts root-scoped palette state to the snapshot shell", () => {
  assert.match(
    indexCss,
    /\.reload-snapshot-shell:not\(\[data-palette\]\)[\s\S]*?\.palette-card\[data-palette-value="standard"\]/,
  );
  assert.match(
    indexCss,
    /\.reload-snapshot-shell\[data-palette="classic"\][\s\S]*?\.palette-card\[data-palette-value="classic"\]/,
  );
  assert.match(
    indexCss,
    /\.reload-snapshot-shell\[data-palette="minimal"\][\s\S]*?\.palette-card\[data-palette-value="minimal"\]/,
  );
});

test("loads selected imported fonts before revealing the shell", async () => {
  const persistedAppearance = new Map([
    [
      "unsloth_appearance_customization",
      JSON.stringify({
        state: {
          customization: {
            uiFont: "Studio Sans",
            headingFont: "Bundled Heading",
            chatFont: null,
            codeFont: null,
            importedFonts: [
              {
                name: "Studio Sans",
                dataUrl: "data:font/woff2;base64,QUJD",
              },
              {
                name: "Unused Font",
                dataUrl: "data:font/woff2;base64,REVG",
              },
            ],
          },
        },
        version: 7,
      }),
    ],
  ]);
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Existing chat</main>",
    styleSheets: ["/assets/index-abc123.css"],
    localStorage: persistedAppearance,
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
    localStorage: persistedAppearance,
  });
  assert.equal(incoming.shell?.rootVisibility, "hidden");
  assert.deepEqual(
    incoming.fontFaces.map((face) => ({
      family: face.family,
      source: face.source,
    })),
    [
      {
        family: "Studio Sans",
        source: "url(data:font/woff2;base64,QUJD)",
      },
    ],
  );
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(incoming.shell?.rootVisibility, "visible");
});

test("drops the retained shell's animations instead of pausing them", () => {
  // Not reachable from the script tests: a paused animation holds its FIRST
  // keyframe, so `animate-in fade-in` entrances would render at opacity 0 for
  // as long as the shell is up. Only the stylesheet can express this.
  const start = indexCss.indexOf(".reload-snapshot-shell *,");
  assert.ok(start > 0);
  const rule = indexCss.slice(start, indexCss.indexOf("}", start));
  assert.doesNotMatch(rule, /animation-play-state:\s*paused/);
  assert.match(rule, /animation:\s*none\s*!important/);
});

test("leaves the retained shell click-through", () => {
  // A full-viewport overlay that takes pointer events swallows every click for
  // as long as it is up, which is up to the five-second fail-open.
  const start = indexCss.indexOf(".reload-snapshot {");
  assert.ok(start > 0);
  const rule = indexCss.slice(start, indexCss.indexOf("}", start));
  assert.match(rule, /pointer-events:\s*none/);
});

test("builds the retained shell inside a closed shadow tree", () => {
  // The copy duplicates the live markup, so leaving it in the page tree makes
  // every `textarea[aria-label=...]` style query ambiguous while it is up.
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Existing chat</main>",
    styleSheets: ["/assets/index-abc123.css", "/assets/chat-def456.css"],
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const stored = storedSnapshot(outgoing.storage);
  assert.deepEqual(stored.styles, [
    "/assets/index-abc123.css",
    "/assets/chat-def456.css",
  ]);

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.equal(incoming.shell?.mode, "closed");
  assert.equal(
    incoming.shell?.html,
    '<div id="root"><main>Existing chat</main></div>',
  );
  // Selectors do not cross the boundary, so the copy needs its own styles.
  assert.deepEqual(incoming.shell?.stylesheets, [
    "/assets/index-abc123.css",
    "/assets/chat-def456.css",
  ]);
  // The shell's own rules live in that stylesheet and hang off this marker.
  assert.match(incoming.shell?.rootClass ?? "", /^reload-snapshot-shell /);
});

test("roots the copy in an html element so html-anchored rules still match", () => {
  // 80-odd rules in the app stylesheet are anchored on `html`, light/dark
  // theming above all, and a selector cannot reach across the shadow boundary
  // to the real document element.
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Existing chat</main>",
    styleSheets: ["/assets/index-abc123.css"],
    htmlAttributes: { "data-palette": "classic", "data-contrast-adjust": "" },
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.equal(incoming.shell?.rootTag, "HTML");
  assert.equal(incoming.shell?.bodyTag, "BODY");
  assert.deepEqual(incoming.shell?.rootAttributes, {
    "data-palette": "classic",
    "data-contrast-adjust": "",
  });
});

test("freezes the design tokens onto the copy's own root", () => {
  // `:root` matches a document's root element only, so tokens declared there
  // never reach a shadow tree; the copy carries the resolved set instead.
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Existing chat</main>",
    styleSheets: ["/assets/index-abc123.css"],
    htmlVariables: { "--font-heading": '"Hellix"', "--tracking-normal": "0em" },
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });
  assert.deepEqual(storedSnapshot(outgoing.storage).tokens, {
    "--font-heading": '"Hellix"',
    "--tracking-normal": "0em",
  });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.equal(incoming.shell?.rootTokens["--font-heading"], '"Hellix"');
  assert.equal(incoming.shell?.rootTokens["--tracking-normal"], "0em");
});

test("carries live form state, except what sensitive fields hide", () => {
  // React writes value/checked as DOM properties; cloneNode copies attributes,
  // so a typed composer would come back empty. Secret fields can be revealed as
  // plain text, but their values must stay behind regardless of presentation.
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      rect: [0, 1440, 900, 0],
      children: [
        {
          tag: "textarea",
          rect: [0, 1440, 200, 0],
          value: "half-written prompt",
        },
        {
          tag: "input",
          rect: [0, 1440, 240, 0],
          type: "search",
          value: "gemma",
        },
        {
          tag: "input",
          rect: [0, 1440, 280, 0],
          type: "checkbox",
          checked: true,
        },
        {
          tag: "input",
          rect: [0, 1440, 320, 0],
          type: "password",
          value: "hunter2",
          attributes: { value: "hunter2" },
        },
        {
          tag: "input",
          rect: [0, 1440, 360, 0],
          type: "text",
          autocomplete: "current-password",
          value: "revealed-password",
          attributes: { value: "revealed-password" },
        },
        {
          tag: "input",
          rect: [0, 1440, 400, 0],
          type: "file",
          value: "C:\\fakepath\\private-dataset.jsonl",
          attributes: { value: "C:\\fakepath\\private-dataset.jsonl" },
        },
        {
          tag: "input",
          rect: [0, 1440, 440, 0],
          type: "text",
          value: "revealed-api-key",
          attributes: {
            value: "revealed-api-key",
            "data-reload-snapshot-sensitive": "",
          },
        },
        {
          tag: "textarea",
          rect: [0, 1440, 480, 0],
          value: "Authorization: Bearer secret-header",
          text: "Authorization: Bearer stale-header",
          attributes: { "data-reload-snapshot-sensitive": "" },
        },
        {
          tag: "code",
          rect: [0, 1440, 520, 0],
          text: "Authorization: Bearer rendered-api-key",
          attributes: { "data-reload-snapshot-sensitive": "" },
        },
        {
          // A filename row repeats the value in its tooltip and in the
          // accessible name beside it, so clearing text alone still ships it.
          tag: "span",
          rect: [0, 1440, 560, 0],
          text: "rendered-local-filename.csv",
          attributes: {
            "data-reload-snapshot-sensitive": "",
            title: "rendered-local-filename.csv",
            "aria-label": "Remove rendered-local-filename.csv",
            placeholder: "rendered-local-filename.csv",
          },
        },
        {
          tag: "div",
          rect: [0, 1440, 600, 0],
          attributes: { "data-reload-snapshot-sensitive": "" },
          children: [
            {
              tag: "img",
              rect: [530, 100, 590, 40],
              naturalWidth: 1024,
              naturalHeight: 768,
              attributes: {
                src: "blob:https://studio.test/pending-local-attachment",
                alt: "pending-local-attachment",
              },
            },
          ],
        },
        {
          tag: "div",
          rect: [0, 1440, 700, 0],
          attributes: { "data-reload-snapshot-sensitive": "" },
          children: [
            {
              tag: "img",
              rect: [630, 100, 690, 40],
              naturalWidth: 1024,
              naturalHeight: 768,
              attributes: {
                src: "data:image/png;base64,private-local-picker",
                alt: "Source",
              },
            },
          ],
        },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /half-written prompt/);
  assert.match(html, /value="gemma"/);
  assert.match(html, /checked=""/);
  assert.doesNotMatch(html, /hunter2/);
  assert.doesNotMatch(html, /revealed-password/);
  assert.doesNotMatch(html, /fakepath|private-dataset/);
  assert.doesNotMatch(html, /revealed-api-key/);
  assert.doesNotMatch(html, /secret-header|stale-header/);
  assert.doesNotMatch(html, /rendered-api-key/);
  assert.doesNotMatch(
    html,
    /pending-local-attachment|private-local-picker|retained-frame/,
  );
  assert.doesNotMatch(html, /rendered-local-filename/);
  assert.match(
    sharedComposerSource,
    /function PendingImageThumb[\s\S]*?data-reload-snapshot-sensitive/,
  );
  assert.match(
    imageDropzoneSource,
    /if \(value\)[\s\S]*?data-reload-snapshot-sensitive/,
  );
  // A picked file's NAME renders outside the file input in several places that
  // neither type=file nor a marked ancestor reaches: the dialogs portal out.
  assert.match(
    referencePickerSource,
    /if \(value\)[\s\S]*?data-reload-snapshot-sensitive[\s\S]*?value\.name/,
  );
  assert.match(
    projectsPageSource,
    /data-reload-snapshot-sensitive[\s\S]*?importFile\?\.name/,
  );
  assert.match(
    unstructuredDropZoneSource,
    /data-reload-snapshot-sensitive[\s\S]*?entry\.name/,
  );
  assert.match(
    seedDialogSource,
    /data-reload-snapshot-sensitive[\s\S]*?localFile\?\.name/,
  );
  // The monitor renders prompt/reply text in three places, not just the
  // expanded payload: every row's excerpt, and a lifecycle row's error text.
  assert.match(
    apiMonitorPageSource,
    /function PayloadBlock[\s\S]*?data-reload-snapshot-sensitive/,
  );
  assert.match(
    apiMonitorPageSource,
    /data-reload-snapshot-sensitive[\s\S]*?\{entry\.error\}/,
  );
  assert.match(
    apiMonitorPageSource,
    /data-reload-snapshot-sensitive[\s\S]*?\{preview\}/,
  );
  assert.match(
    sharedComposerSource,
    /data-reload-snapshot-sensitive[\s\S]*?pendingAudio\.name/,
  );
  // Both carriers: the tooltip on the name, and the accessible name on the
  // remove button, a sibling no ancestor marker would reach.
  assert.match(
    projectSourceDropzoneSource,
    /data-reload-snapshot-sensitive[\s\S]*?title=\{entry\.name\}/,
  );
  assert.match(
    projectSourceDropzoneSource,
    /data-reload-snapshot-sensitive\s*\n\s*aria-label=\{`Remove \$\{entry\.name\}`\}/,
  );
  assert.match(
    attachmentSource,
    /export const ComposerAttachments[\s\S]*?data-reload-snapshot-sensitive/,
  );
  // The dialog lives in attachment-preview.tsx; attachment.tsx passes the flag.
  assert.match(attachmentSource, /<AttachmentPreviewDialog redactFromReload=/);
  assert.match(
    attachmentPreviewSource,
    /AttachmentPreviewDialog[\s\S]*?redactFromReload/,
  );
  for (const dialog of [
    "AttachmentImageDialog",
    "AttachmentTextDialog",
    "AttachmentAudioDialog",
  ]) {
    assert.match(
      attachmentPreviewSource,
      new RegExp(
        `${dialog}[\\s\\S]*?data-reload-snapshot-sensitive=\\{redactFromReload \\? "" : undefined\\}`,
      ),
      `${dialog} must redact its portaled content`,
    );
  }
});

test("keeps native select options that paint the closed control label", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      children: [
        {
          tag: "select",
          rect: [100, 420, 140, 200],
          children: [
            {
              tag: "option",
              rect: [0, 0, 0, 0],
              text: "GPU 0",
            },
            {
              tag: "optgroup",
              rect: [0, 0, 0, 0],
              children: [
                {
                  tag: "option",
                  rect: [0, 0, 0, 0],
                  text: "GPU 1",
                  selected: true,
                },
              ],
            },
          ],
        },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /<option>GPU 0<\/option>/);
  assert.match(html, /<optgroup><option selected="">GPU 1<\/option><\/optgroup>/);
});

test("preserves IDs that the isolated copy references internally", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      children: [
        {
          tag: "svg",
          children: [
            {
              tag: "linearGradient",
              attributes: { id: "history-fill" },
            },
            {
              tag: "path",
              attributes: { fill: "url(#history-fill)" },
            },
          ],
        },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /id="history-fill"/);
  assert.match(html, /fill="url\(#history-fill\)"/);
});

test("carries visible body portals along with the app root", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      children: [{ tag: "main", text: "Settings page" }],
    },
    bodyPortals: [
      {
        tag: "div",
        rect: [100, 1100, 700, 300],
        attributes: { role: "dialog" },
        text: "Confirm deletion",
      },
    ],
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /Settings page/);
  assert.match(html, /role="dialog"/);
  assert.match(html, /Confirm deletion/);
});

test("keeps scroll-container geometry with a bounded visible slice", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      children: [
        {
          tag: "section",
          overflowY: "auto",
          clientHeight: 300,
          scrollHeight: 1200,
          scrollTop: 600,
          children: [
            {
              tag: "p",
              rect: [-500, 900, -450, 100],
              text: "Earlier message keeps the scroll geometry",
            },
            {
              tag: "p",
              rect: [300, 900, 350, 100],
              text: "Visible message",
            },
          ],
        },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /data-reload-scroll-top="600"/);
  assert.doesNotMatch(html, /Earlier message keeps the scroll geometry/);
  assert.match(html, /data-reload-spacer/);
  assert.match(html, /Visible message/);
});

test("coalesces a transcript larger than the snapshot cap", () => {
  const earlierMessages = Array.from({ length: 1100 }, (_, index) => ({
    tag: "article",
    rect: [-5000 + index * 4, 900, -4996 + index * 4, 100] as [
      number,
      number,
      number,
      number,
    ],
    text: "Earlier message " + "x".repeat(4096),
  }));
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      children: [
        {
          tag: "section",
          overflowY: "auto",
          clientHeight: 300,
          scrollHeight: 6000,
          scrollTop: 5000,
          children: [
            ...earlierMessages,
            {
              tag: "article",
              rect: [300, 900, 350, 100],
              text: "Visible message survives",
            },
          ],
        },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /Visible message survives/);
  assert.equal((html.match(/data-reload-spacer/g) ?? []).length, 1);
  assert.ok(html.length < 3 * 1024 * 1024);
});

test("materializes visible ephemeral media and drops protected URLs", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      children: [
        {
          tag: "img",
          rect: [100, 900, 700, 100],
          naturalWidth: 1024,
          naturalHeight: 768,
          attributes: { src: "blob:https://studio.test/generated-image" },
        },
        {
          tag: "audio",
          rect: [720, 900, 780, 100],
          attributes: {
            controls: "",
            src: "blob:https://studio.test/generated-audio",
          },
        },
        {
          tag: "video",
          rect: [100, 1300, 700, 950],
          videoWidth: 1280,
          videoHeight: 720,
          attributes: {
            src:
              "/api/inference/video/gallery/clip/file-signed?token=clip.123.secret",
          },
        },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /src="data:image\/webp;base64,retained-frame"/);
  assert.match(html, /<audio controls="">/);
  assert.match(html, /<video poster="data:image\/webp;base64,retained-frame">/);
  assert.doesNotMatch(html, /blob:/);
  assert.doesNotMatch(html, /clip\.123\.secret|[?&]token=/);
});

test("keeps trusted chart CSS while stripping every other style", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      children: [
        {
          tag: "style",
          rect: [0, 0, 0, 0],
          attributes: { "data-reload-snapshot-style": "" },
          text: "[data-chart=loss] { --color-displayLoss: #16a34a; }",
        },
        {
          tag: "style",
          rect: [0, 0, 0, 0],
          text: ".untrusted { display: block; }",
        },
        {
          tag: "svg",
          children: [{ tag: "path", attributes: { "data-chart": "loss" } }],
        },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /--color-displayLoss: #16a34a/);
  assert.doesNotMatch(html, /\.untrusted/);
});

test("keeps the shell when materialized media alone passes the cap", () => {
  // capturePixels rasterizes at devicePixelRatio, so the same page costs 4x on
  // a 2x display. Discarding the whole snapshot there puts back the blank
  // flash this exists to remove; drop the pixels and keep the layout instead.
  const tiles = Array.from({ length: 6 }, (_, index) => ({
    tag: "img",
    rect: [10 + index, 400, 200 + index, 200] as [number, number, number, number],
    naturalWidth: 1400,
    naturalHeight: 1000,
    attributes: { src: "blob:https://studio/tile-" + index },
  }));
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    frameDataBody: "A".repeat(600 * 1024),
    rootTree: {
      tag: "div",
      children: [
        { tag: "h1", rect: [10, 60, 400, 20], text: "Images" },
        ...tiles,
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(html, /Images/, "the shell must survive the overflow");
  assert.ok(html.length < 3 * 1024 * 1024);
  assert.equal(
    (html.match(/data:image/g) ?? []).length,
    0,
    "the pixels that caused the overflow must be the thing dropped",
  );
});

test("materializes visible canvas pixels into the retained shell", () => {
  const environment = createEnvironment({
    navigationType: "navigate",
    styleSheets: ["/assets/index-abc123.css"],
    rootTree: {
      tag: "div",
      children: [
        {
          tag: "canvas",
          rect: [100, 900, 700, 100],
          width: 1024,
          height: 768,
          attributes: { class: "absolute inset-0", style: "opacity:0.8" },
        },
      ],
    },
  });
  environment.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const { html } = storedSnapshot(environment.storage);
  assert.match(
    html,
    /style="opacity:0\.8;background-image:url\(data:image\/webp;base64,retained-frame\);background-size:100% 100%;background-repeat:no-repeat;"/,
  );
});

test("skips the shell when the snapshot carries no stylesheets", () => {
  // Without them the copy paints unstyled inside the shadow tree, which reads
  // worse than the blank interval this replaces.
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Existing chat</main>",
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });
  assert.deepEqual(storedSnapshot(outgoing.storage).styles, []);

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.equal(incoming.appended.length, 0);
});
