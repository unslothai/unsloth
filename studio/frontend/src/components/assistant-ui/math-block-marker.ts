// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * Marks the block holding inline maths so index.css can apply content-visibility. KaTeX positions make
 * WebKitGTK scroll walk every layer; inline .katex cannot take size containment, so hoist to the block.
 * Runs inside the maths plugin's rehype pass: a rehypePlugins prop would drop Streamdown's sanitizer, and
 * mdast classes get sanitized. The class is emitted even when the feature is off so DOM stays identical.
 */

interface HastProperties {
  className?: unknown;
  [key: string]: unknown;
}

interface HastNode {
  type: string;
  tagName?: string;
  properties?: HastProperties;
  children?: HastNode[];
}

/** The marker. Read by `index.css`, and by nothing else. */
export const MATH_BLOCK_CLASS = "aui-math-block";

/* Only un-numbered displays: style containment scopes katexEqnNo, renumbering (1)(1)(1) on Chromium. */
export const MATH_DISPLAY_CLASS = "aui-math-display";

const EQN_NUM_CLASSES = new Set(["eqn-num", "mml-eqn-num"]);

/** What survives `rehype-sanitize` on a maths `code` element. */
const MATH_CLASS = "language-math";

/* Containable blocks prose with inline maths can sit in; `pre` holds display maths only. */
const BLOCK_TAGS = new Set([
  "p",
  "div",
  "blockquote",
  "dd",
  "figcaption",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
]);

/*
 * Size containment does not apply to table internals, and style containment on li hides its list
 * marker, so maths in table cells and list items is abandoned rather than hoisted.
 */
const UNCONTAINABLE_TAGS = new Set([
  "li",
  "ol",
  "ul",
  "td",
  "th",
  "table",
  "thead",
  "tbody",
  "tfoot",
  "tr",
]);

/* Bounded walk: an unbounded one over a malformed tree is worse than marking nothing. */
const MAX_HOPS = 12;

const classListOf = (node: HastNode): string[] => {
  const raw = node.properties?.className;
  if (Array.isArray(raw)) return raw.map(String);
  if (typeof raw === "string") return raw.split(/\s+/).filter(Boolean);
  return [];
};

const addClass = (node: HastNode): void => {
  const properties: HastProperties = node.properties ?? {};
  node.properties = properties;
  const current = classListOf(node);
  if (current.includes(MATH_BLOCK_CLASS)) return;
  properties.className = [...current, MATH_BLOCK_CLASS];
};

const isInlineMath = (node: HastNode, parent: HastNode | undefined): boolean =>
  node.type === "element" &&
  node.tagName === "code" &&
  classListOf(node).includes(MATH_CLASS) &&
  parent?.tagName !== "pre";

/** Marks the nearest containable ancestor in `stack`; null for tables, list items or > MAX_HOPS. */
export const markNearestBlock = (stack: HastNode[]): HastNode | null => {
  let hops = 0;
  for (let i = stack.length - 1; i >= 0 && hops < MAX_HOPS; i -= 1, hops += 1) {
    const candidate = stack[i];
    const tagName = candidate.tagName ?? "";
    if (UNCONTAINABLE_TAGS.has(tagName)) return null;
    if (!BLOCK_TAGS.has(tagName)) continue;
    /* Streamdown makes a list item's p inline, and li cannot be contained, so abandon; checked by
     * tests/math-block-marker.test.ts against the installed Streamdown build. */
    if (tagName === "p" && i > 0 && stack[i - 1].tagName === "li") return null;
    addClass(candidate);
    return candidate;
  }
  return null;
};

/** Mark every block that holds inline maths; returns the count, for testing without a DOM. */
export const markMathBlocks = (tree: HastNode): number => {
  const stack: HastNode[] = [];
  let marked = 0;

  const visit = (node: HastNode, parent: HastNode | undefined): void => {
    if (isInlineMath(node, parent)) {
      if (markNearestBlock(stack)) marked += 1;
      return;
    }
    const children = node.children;
    if (!children || children.length === 0) return;
    const isElement = node.type === "element";
    if (isElement) stack.push(node);
    for (const child of children) visit(child, node);
    if (isElement) stack.pop();
  };

  visit(tree, undefined);
  return marked;
};

type Transformer = (tree: HastNode, file: unknown) => unknown;
type Attacher = (
  this: unknown,
  ...options: unknown[]
) => Transformer | undefined;

/** Does this subtree carry a KaTeX equation number? */
const hasEquationNumber = (node: HastNode): boolean => {
  if (node.type === "element" && classListOf(node).some((c) => EQN_NUM_CLASSES.has(c))) {
    return true;
  }
  for (const child of node.children ?? []) {
    if (hasEquationNumber(child)) return true;
  }
  return false;
};

/**
 * Runs after KaTeX: marks un-numbered displays, and unmarks blocks holding a numbered display
 * (markMathBlocks ran before .eqn-num existed). Returns counts, for testing without a browser.
 */
export const guardEquationNumbers = (tree: HastNode): { marked: number; unmarked: number } => {
  let marked = 0;
  let unmarked = 0;
  const visit = (node: HastNode): void => {
    if (node.type === "element") {
      const classes = classListOf(node);
      if (classes.includes("katex-display") && !hasEquationNumber(node)) {
        const properties: HastProperties = node.properties ?? {};
        node.properties = properties;
        if (!classes.includes(MATH_DISPLAY_CLASS)) {
          properties.className = [...classes, MATH_DISPLAY_CLASS];
          marked += 1;
        }
      }
      if (classes.includes(MATH_BLOCK_CLASS) && hasEquationNumber(node)) {
        const properties: HastProperties = node.properties ?? {};
        node.properties = properties;
        properties.className = classes.filter((c) => c !== MATH_BLOCK_CLASS);
        unmarked += 1;
      }
    }
    for (const child of node.children ?? []) visit(child);
  };
  visit(tree);
  return { marked, unmarked };
};

// Returns ONE attacher: Streamdown appends it as one entry, and an array would read as a tuple.
export const withMathBlockMarker = (mathRehypePlugin: unknown): Attacher => {
  const [attacher, options] = (
    Array.isArray(mathRehypePlugin) ? mathRehypePlugin : [mathRehypePlugin]
  ) as [Attacher, unknown?];
  return function markThenRenderMaths(this: unknown) {
    const renderMaths = attacher.call(this, options);
    return (tree: HastNode, file: unknown) => {
      markMathBlocks(tree);
      const rendered = renderMaths ? renderMaths(tree, file) : undefined;
      // After KaTeX, because equation numbers do not exist until it has run.
      guardEquationNumbers(tree);
      return rendered;
    };
  };
};
