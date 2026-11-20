// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Capture is the only DOM-aware part of the quote source path. These tests use
// a minimal DOM to prove that block boundaries produced by the renderer become
// separators in the stored anchor, while single-block and non-separator text
// differences keep the captured selection.

import assert from "node:assert/strict";
import test, { after } from "node:test";

import {
  buildTextIndex,
  type FindElementLike,
} from "../src/features/find-in-page/lib/find-text-index.ts";
import { resolveQuoteAnchor } from "../src/components/assistant-ui/message-quote-model.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const [{ captureQuoteAnchor }, { QUOTE_PRUNE_SELECTOR }] = await Promise.all([
  import("../src/components/assistant-ui/message-quote-capture.ts"),
  import("../src/components/assistant-ui/message-selection-model.ts"),
]);

type FakeNode = FakeText | FakeElement | FakeFragment;

class FakeText {
  readonly nodeType = 3;
  readonly data: string;
  readonly nodeValue: string;
  parentElement: FakeElement | null = null;
  readonly isConnected = true;

  constructor(data: string) {
    this.data = data;
    this.nodeValue = data;
  }

  get textContent(): string {
    return this.data;
  }
}

class FakeFragment {
  readonly nodeType = 11;
  readonly childNodes: FakeNode[];
  private readonly overrideText: string | null;

  constructor(content: FakeNode[] | string) {
    if (typeof content === "string") {
      this.childNodes = [];
      this.overrideText = content;
    } else {
      this.childNodes = content;
      this.overrideText = null;
    }
  }

  get textContent(): string {
    return this.overrideText ??
      this.childNodes.map((node) => node.textContent).join("");
  }
}

class FakeElement {
  readonly nodeType = 1;
  readonly childNodes: FakeNode[] = [];
  readonly attrs = new Map<string, string>();
  parentElement: FakeElement | null = null;
  readonly isConnected = true;
  readonly tagName: string;

  constructor(tagName: string) {
    this.tagName = tagName;
  }

  append(...nodes: FakeNode[]): void {
    for (const node of nodes) {
      this.childNodes.push(node);
      if (node instanceof FakeElement || node instanceof FakeText) {
        node.parentElement = this;
      }
    }
  }

  remove(): void {
    if (this.parentElement !== null) {
      this.parentElement.childNodes.splice(
        this.parentElement.childNodes.indexOf(this),
        1,
      );
      this.parentElement = null;
    }
  }

  setAttribute(name: string, value: string): void {
    this.attrs.set(name, value);
  }

  getAttribute(name: string): string | null {
    return this.attrs.get(name) ?? null;
  }

  matches(selector: string): boolean {
    return selector.split(",").some((part) => {
      const value = part.trim();
      if (value === "button") return this.tagName === "BUTTON";
      if (value === "svg") return this.tagName === "SVG";
      if (value === "[aria-hidden='true']") {
        return this.getAttribute("aria-hidden") === "true";
      }
      const dataStreamdown = /^\[data-streamdown='([^']+)'\]$/.exec(value);
      if (dataStreamdown !== null) {
        return this.getAttribute("data-streamdown") === dataStreamdown[1];
      }
      return value === ".aui-md-table-actions" &&
        (this.getAttribute("class") ?? "")
          .split(/\s+/)
          .includes("aui-md-table-actions");
    });
  }

  checkVisibility(): boolean {
    return true;
  }

  querySelectorAll(selector: string): FakeNode[] {
    const found: FakeNode[] = [];
    const walk = (node: FakeNode): void => {
      if (node instanceof FakeElement) {
        if (node.matches(selector)) found.push(node);
        for (const child of node.childNodes) walk(child);
      } else if (node instanceof FakeFragment) {
        for (const child of node.childNodes) walk(child);
      }
    };
    for (const child of this.childNodes) walk(child);
    return found;
  }

  cloneDeep(): FakeElement {
    const clone = new FakeElement(this.tagName);
    for (const [name, value] of this.attrs) {
      clone.setAttribute(name, value);
    }
    for (const child of this.childNodes) {
      clone.append(
        child instanceof FakeElement
          ? child.cloneDeep()
          : child instanceof FakeText
            ? new FakeText(child.data)
            : child,
      );
    }
    return clone;
  }

  get textContent(): string {
    return this.childNodes.map((node) => node.textContent).join("");
  }
}

class FakeRange {
  readonly collapsed = false;
  readonly startContainer: FakeText;
  readonly startOffset: number;
  readonly endContainer: FakeText;
  readonly endOffset: number;
  private readonly captured: string;
  private readonly cloneRoot: FakeElement | null;

  constructor(
    startContainer: FakeText,
    startOffset: number,
    endContainer: FakeText,
    endOffset: number,
    captured: string,
    cloneRoot: FakeElement | null = null,
  ) {
    this.startContainer = startContainer;
    this.startOffset = startOffset;
    this.endContainer = endContainer;
    this.endOffset = endOffset;
    this.captured = captured;
    this.cloneRoot = cloneRoot;
  }

  cloneContents(): FakeFragment {
    return this.cloneRoot === null
      ? new FakeFragment(this.captured)
      : new FakeFragment([this.cloneRoot.cloneDeep()]);
  }
}

const previousDocument = (globalThis as { document?: unknown }).document;
(globalThis as { document?: unknown }).document = {
  createElement: (tagName: string) => new FakeElement(tagName.toUpperCase()),
};

after(() => {
  (globalThis as { document?: unknown }).document = previousDocument;
});

function element(tagName: string, ...children: FakeNode[]): FakeElement {
  const node = new FakeElement(tagName.toUpperCase());
  node.append(...children);
  return node;
}

function text(data: string): FakeText {
  return new FakeText(data);
}

function captured(
  content: FakeElement,
  start: FakeText,
  end: FakeText,
  capturedText: string,
  endOffset = end.data.length,
  startOffset = 0,
  cloneContent: boolean | FakeElement = false,
) {
  const range = new FakeRange(
    start,
    startOffset,
    end,
    endOffset,
    capturedText,
    cloneContent === true ? content : cloneContent || null,
  );
  const anchor = captureQuoteAnchor({
    messageId: "message-1",
    turn: 2,
    markdown: capturedText,
    range: range as unknown as Range,
    content: content as unknown as HTMLElement,
  });
  const index = buildTextIndex(
    content as unknown as FindElementLike,
    [],
    QUOTE_PRUNE_SELECTOR,
  );
  assert.ok(anchor);
  return { anchor, index, resolved: resolveQuoteAnchor(index, anchor) };
}

function codeBlock(language = "Python", code = "const alpha = 1;") {
  const languageText = text(language);
  const header = element("div", element("span", languageText));
  header.setAttribute("data-streamdown", "code-block-header");
  const actions = element("div", element("button"));
  actions.setAttribute("data-code-block-actions", "");
  const codeText = text(code);
  const body = element(
    "div",
    element("pre", element("code", element("span", codeText))),
  );
  body.setAttribute("data-streamdown", "code-block-body");
  const block = element("div", header, actions, body);
  block.setAttribute("data-streamdown", "code-block");
  return { block, languageText, codeText };
}

test("multi-paragraph selections store the indexed separator", () => {
  const first = text("alpha one");
  const second = text("beta two");
  const content = element("div", element("p", first), element("p", second));
  const { anchor, resolved } = captured(
    content,
    first,
    second,
    "alpha onebeta two",
  );

  assert.equal(anchor.text, "alpha one\nbeta two");
  assert.equal(anchor.start, 0);
  assert.equal(anchor.end, 18);
  assert.deepEqual(resolved, { start: 0, end: 18, status: "exact" });
});

test("br boundaries store the indexed separator", () => {
  const first = text("line one");
  const second = text("line two");
  const content = element(
    "div",
    first,
    element("br"),
    second,
  );
  const { anchor, resolved } = captured(
    content,
    first,
    second,
    "line oneline two",
  );

  assert.equal(anchor.text, "line one\nline two");
  assert.deepEqual(resolved, {
    start: 0,
    end: anchor.text.length,
    status: "exact",
  });
});

test("multiple list items store each indexed separator", () => {
  const first = text("one");
  const second = text("two");
  const third = text("three");
  const content = element(
    "ul",
    element("li", first),
    element("li", second),
    element("li", third),
  );
  const { anchor, resolved } = captured(
    content,
    first,
    third,
    "onetwothree",
  );

  assert.equal(anchor.text, "one\ntwo\nthree");
  assert.deepEqual(resolved, {
    start: 0,
    end: anchor.text.length,
    status: "exact",
  });
});

test("heading-to-paragraph boundaries store the indexed separator", () => {
  const heading = text("Heading");
  const body = text("Body");
  const content = element(
    "div",
    element("h2", heading),
    element("p", body),
  );
  const { anchor, resolved } = captured(
    content,
    heading,
    body,
    "HeadingBody",
  );

  assert.equal(anchor.text, "Heading\nBody");
  assert.deepEqual(resolved, {
    start: 0,
    end: anchor.text.length,
    status: "exact",
  });
});

test("single-block selections preserve the captured text", () => {
  const passage = text("alpha one");
  const content = element("div", element("p", passage));
  const { anchor, resolved } = captured(
    content,
    passage,
    passage,
    "alpha one",
  );

  assert.equal(anchor.text, "alpha one");
  assert.deepEqual(resolved, {
    start: 0,
    end: 9,
    status: "exact",
  });
});

test("CRLF is normalized, but genuinely different spacing is not collapsed", () => {
  const first = text("alpha one");
  const second = text("beta two");
  const content = element("div", element("p", first), element("p", second));

  const normalized = captured(
    content,
    first,
    second,
    "alpha\u00a0one\r\nbeta two",
  );
  assert.equal(normalized.anchor.text, "alpha one\nbeta two");

  const extra = captured(
    content,
    first,
    second,
    "alpha  onebeta two",
  );
  assert.equal(extra.anchor.text, "alpha  onebeta two");
  assert.equal(extra.resolved, null);
});

test("paragraph-to-code selection excludes the language header and resolves", () => {
  const paragraph = text("alpha beta alpha beta gamma");
  const { block, codeText } = codeBlock();
  const content = element("div", element("p", paragraph), block);

  const { anchor, index, resolved } = captured(
    content,
    paragraph,
    codeText,
    "alpha beta alpha beta gammaconst alpha = 1;",
    codeText.data.length,
    0,
    true,
  );

  assert.equal(
    anchor.text,
    "alpha beta alpha beta gamma\nconst alpha = 1;",
  );
  assert.equal(index.text.includes("python"), false);
  assert.match(anchor.text, /const alpha = 1;$/);
  assert.deepEqual(resolved, {
    start: 0,
    end: anchor.text.length,
    status: "exact",
  });
});

test("ordinary index still includes the language header", () => {
  const paragraph = text("alpha beta alpha beta gamma");
  const { block } = codeBlock();
  const content = element("div", element("p", paragraph), block);
  const index = buildTextIndex(content as unknown as FindElementLike);

  assert.equal(index.text.includes("python"), true);
  assert.ok(
    index.segments.some((segment) => segment.node.data === "Python"),
  );
});

test("code-only selection resolves at the filtered code offset", () => {
  const paragraph = text("alpha beta alpha beta gamma");
  const { block, codeText } = codeBlock();
  const content = element("div", element("p", paragraph), block);

  const { anchor, resolved } = captured(
    content,
    codeText,
    codeText,
    "const alpha = 1;",
    codeText.data.length,
    0,
    block,
  );

  assert.equal(anchor.text, "const alpha = 1;");
  assert.deepEqual(resolved, {
    start: 28,
    end: 44,
    status: "exact",
  });
});

test("header-only selection fails safely without an anchor", () => {
  const paragraph = text("alpha beta alpha beta gamma");
  const { block, languageText } = codeBlock();
  const content = element("div", element("p", paragraph), block);
  const range = new FakeRange(
    languageText,
    0,
    languageText,
    languageText.data.length,
    "Python",
    content,
  );

  const anchor = captureQuoteAnchor({
    messageId: "message-1",
    turn: 2,
    markdown: "Python",
    range: range as unknown as Range,
    content: content as unknown as HTMLElement,
  });

  assert.equal(anchor, null);
});
