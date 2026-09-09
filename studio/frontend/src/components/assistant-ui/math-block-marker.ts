// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * MARK THE BLOCK THAT HOLDS INLINE MATHS, so a stylesheet has something it can contain.
 *
 * WHY THIS EXISTS. On WebKitGTK, which is what Studio and Desktop render through on Linux,
 * `RenderLayerScrollableArea::scrollTo` reaches
 * `RenderLayer::recursiveUpdateLayerPositionsAfterScroll`, which recurses over every descendant
 * RenderLayer once per scroll EVENT with no dirty-bit pruning and exactly one early out,
 * `!m_hasVisibleDescendant && !m_hasVisibleContent`. `RenderBoxModelObject::requiresLayer()` is
 * true for `isPositioned()`, and `isPositioned()` is `position != static`, so `position: relative`
 * alone buys a layer. KaTeX sets `position: relative` on `.katex`, `.base` and `.vlist`, and
 * `position: absolute` on `.katex-mathml` and `.vlist > span`. A 500K-character maths-bearing
 * thread measures 21,713 positioned boxes, about 20,700 of them under KaTeX, and 287.6 ms of
 * blocked main thread to move that scroller by ONE PIXEL.
 *
 * `content-visibility: auto` reaches that same early out for content the user cannot see: a
 * skipped subtree is `isSkippedContent()` and `RenderLayer::computeHasVisibleContent()` returns
 * false for it, so the walk stops at the skipped root instead of descending.
 *
 * WHY THE MATHS ROOT ITSELF IS NOT THE PLACE TO PUT IT. `content-visibility` needs SIZE
 * CONTAINMENT to take effect, and size containment does not apply to a non-atomic inline-level box
 * (css-contain-2 #containment-size; WebKit implements the check in
 * `Style::ContainmentChecker::shouldApplySizeContainment`). Shipped `katex.css` gives `.katex` no
 * `display`, so inline maths computes to `display: inline` and the declaration is silently inert
 * on it. That was measured rather than reasoned about: a rule naming `.katex` had the engine act
 * on 0 of 146 sampled inline roots. In the same corpus only about 4,919 of the positioned boxes
 * under maths sit under DISPLAY maths against about 15,291 under INLINE maths, so a rule that can
 * only reach display maths leaves three quarters of the cost behind, and measured as such: minus
 * 27 percent on the mean and plus 13 percent on p50, under the bar either way.
 *
 * SO THE DECLARATION IS HOISTED TO THE BLOCK THAT CONTAINS THE MATHS. For inline maths that block
 * is the paragraph, list item or heading it sits in, and nothing in CSS can select "an element
 * that contains maths" except `:has()`, which is itself the measured owner of the 500K scroll cost
 * on Chromium. So the renderer marks it, here.
 *
 * DISPLAY MATHS NEEDS NO MARKER and deliberately gets none. `remark-math` emits it as
 * `<pre><code class="language-math math-display">`, `rehype-katex` replaces that whole `pre` with
 * `<span class="katex-display">`, and `katex.css` gives that span `display: block`. It is already
 * a block-level element with a stable class, so `index.css` names it directly. Marking it here
 * would put a class on a node that is about to be thrown away.
 *
 * WHERE THIS RUNS, AND WHY IT IS NOT A `remarkPlugins` OR `rehypePlugins` PROP. Both were tried
 * and both are wrong here:
 *
 *   - Passing `rehypePlugins` to `<Streamdown>` replaces its default list, and Streamdown only
 *     installs the `allowedTags` sanitizer schema when that list is still the default one
 *     (`rehypePlugins === defaultRehypePlugins`). A pipeline that does not carry the merge silently
 *     drops the sanitize pass this renderer depends on -- a security regression to buy a class name;
 *     the one now passed for data images carries it (`lib/markdown-data-images.ts`).
 *   - Marking on the mdast side, through `data.hProperties`, lands the class BEFORE
 *     `rehype-sanitize`, whose `defaultSchema` permits `className` on `a`, `code`, `h2`, `li`,
 *     `ol`, `section` and `ul` and NOWHERE ELSE. A class on a `<p>` would be stripped, and
 *     stripped silently, which is exactly the shape of a change that measures as doing nothing.
 *
 * So the marker is composed onto the MATHS plugin's own rehype pass, which Streamdown appends
 * after `raw`, `sanitize` and `harden`. It is the only hook in this pipeline that runs on hast,
 * after the sanitizer, without displacing anything.
 *
 * WHAT IT SEES THERE. The sanitizer strips `math-inline` and `math-display` for the same reason it
 * would strip ours: `defaultSchema` allows only `['className', /^language-./]` on `code`. So by
 * this point both kinds of maths carry `language-math` and nothing else, and inline is told from
 * display exactly the way `rehype-katex` itself tells them apart one plugin later: display maths
 * is a `code` whose parent is a `pre`.
 *
 * THE CLASS IS EMITTED WHETHER OR NOT THE FEATURE IS ON. It is inert without the stylesheet rule,
 * and emitting it unconditionally is what lets a measurement compare a window with the feature on
 * against a window with it off WITHOUT the two windows holding different DOM. An arm that mutates
 * the DOM inside its own measured window is billing itself for someone else's work.
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

/*
 * The class the stylesheet contains DISPLAY maths by, added AFTER KaTeX has run and only to
 * displays that carry no equation number. `katex.css` resets `katexEqnNo` on `body` and increments
 * it in `.katex .eqn-num::before`, and style containment, which `content-visibility: auto` applies
 * unconditionally, scopes that increment to each contained display. Measured on Chromium: three
 * numbered displays read (1) (2) (3) with the feature off and (1) (1) (1) with it on. The same
 * fixture on WebKitGTK 2.50.4 does NOT reproduce it, 0 differing pixels, so this is engine
 * dependent and Chromium is the engine most of the web UI runs in.
 *
 * A `:has()` rule would express this in CSS alone and is exactly what must not be used: two
 * `:has()` rules were the measured owner of the whole 500K scroll cost on Chromium, fixed in #9669.
 * So the renderer decides, and the stylesheet names a plain class.
 */
export const MATH_DISPLAY_CLASS = "aui-math-display";

/** KaTeX's two equation-number markers, from `katex.css`. */
const EQN_NUM_CLASSES = new Set(["eqn-num", "mml-eqn-num"]);

/** What survives `rehype-sanitize` on a maths `code` element. See the header. */
const MATH_CLASS = "language-math";

/*
 * Elements that can take size containment AND that a run of prose carrying inline maths can be
 * sitting in. `pre` is absent because a `code.language-math` inside one is DISPLAY maths, which
 * this module does not touch.
 */
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
 * Size containment does NOT apply to internal table elements (css-contain-2 #containment-size), so
 * a `td` or `th` cannot be the target, and neither can anything above it short of the whole table,
 * which would be far too coarse a thing to collapse. Maths inside a table cell is therefore
 * ABANDONED rather than hoisted past. Recorded as a known limit rather than hidden: the corpus
 * this was measured on has no maths in tables.
 *
 * LIST ITEMS ARE HERE FOR A DIFFERENT AND MEASURED REASON. `content-visibility: auto` applies
 * STYLE containment, which scopes the automatic `list-item` counter, and a contained `li` can then
 * no longer resolve `counter(list-item)` for its own `::marker`. The observed result is not a
 * renumbering, it is that the number DISAPPEARS: photographed on WebKitGTK 2.50.4 with a five item
 * ordered list where only items 2 and 4 carried the class, the marker column differed by 61 and 59
 * pixels at full channel swing on exactly those two items and by 0 pixels on the other three, so
 * the list read 1, nothing, 3, nothing, 5. Siblings still observe the increment; what breaks is the
 * contained item's own marker.
 *
 * An earlier revision marked the `li` DELIBERATELY, because Streamdown gives a list item's
 * paragraph `[&>p]:inline` and an inline box cannot take size containment, which leaves the `li` as
 * the only containable ancestor. There is no third option there, so maths inside a list item is
 * abandoned like maths inside a table cell. `ol` and `ul` are listed too, so the walk cannot hoist
 * PAST an item and contain the whole list, which would lose every marker in it instead of one.
 *
 * The cost of this exemption was censused rather than assumed: of the 595 marked blocks in the 500K
 * corpus, 595 are `p` and 0 are `li`, so it gives up nothing that the +92% measurement was
 * counting. A thread that does put maths in lists loses the optimisation for those items and keeps
 * its numbering, which is the right way round.
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

/*
 * The same bound the measurement harness used when it walked the rendered DOM for the equivalent
 * arm. Maths buried more than twelve elements deep inside inline wrappers is not a shape this
 * renderer produces, and an unbounded walk over a malformed tree is worse than marking nothing.
 */
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

/** Inline maths is a maths `code` element whose parent is not a `pre`. Display maths is the rest. */
const isInlineMath = (node: HastNode, parent: HastNode | undefined): boolean =>
  node.type === "element" &&
  node.tagName === "code" &&
  classListOf(node).includes(MATH_CLASS) &&
  parent?.tagName !== "pre";

/**
 * Walk up `stack` (outermost first) and mark the nearest element that can take containment.
 *
 * Returns the element marked, or `null` when there is nothing markable, which happens for maths
 * inside a table cell, maths deeper than `MAX_HOPS` inside inline wrappers, and maths with no
 * block ancestor at all.
 */
export const markNearestBlock = (stack: HastNode[]): HastNode | null => {
  let hops = 0;
  for (let i = stack.length - 1; i >= 0 && hops < MAX_HOPS; i -= 1, hops += 1) {
    const candidate = stack[i];
    const tagName = candidate.tagName ?? "";
    if (UNCONTAINABLE_TAGS.has(tagName)) return null;
    if (!BLOCK_TAGS.has(tagName)) continue;
    /*
     * Streamdown renders a list item with `[&>p]:inline`, so the paragraph a list item wraps its
     * text in computes to `display: inline` and cannot take size containment. Neither can the list
     * item, for the counter reason recorded on `UNCONTAINABLE_TAGS`, so there is nothing markable
     * here and the maths is abandoned rather than hoisted to something that would break numbering.
     * `tests/math-block-marker.test.ts` reads the `[&>p]:inline` class out of the installed
     * Streamdown build, so the day Streamdown stops doing it this stops being justified loudly
     * rather than quietly.
     */
    if (tagName === "p" && i > 0 && stack[i - 1].tagName === "li") return null;
    addClass(candidate);
    return candidate;
  }
  return null;
};

/**
 * Mark every block that holds inline maths. Returns how many blocks were marked, which is what
 * makes the transform testable without a DOM and without a browser.
 */
export const markMathBlocks = (tree: HastNode): number => {
  const stack: HastNode[] = [];
  let marked = 0;

  const visit = (node: HastNode, parent: HastNode | undefined): void => {
    if (isInlineMath(node, parent)) {
      // A maths root's own subtree holds nothing else of interest.
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

/**
 * Compose the marker in front of the maths renderer, preserving whatever options the maths plugin
 * was configured with.
 *
 * Takes the `rehypePlugin` off a Streamdown maths plugin, which is either an attacher or an
 * `[attacher, options]` tuple, and returns a single attacher that marks the tree and then hands it
 * to the original. Returning ONE attacher matters: Streamdown appends this value to its rehype
 * list as a single entry, where an array would be read as an `[attacher, options]` tuple.
 */
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
 * Run AFTER KaTeX. Two jobs, both about the equation counter:
 *
 *   1. Give every `.katex-display` WITHOUT an equation number the class the stylesheet contains by.
 *      A numbered one is left alone and simply does not take containment.
 *   2. Take `MATH_BLOCK_CLASS` back off any block that turned out to contain a numbered display.
 *      `markMathBlocks` runs before KaTeX, when a display is still
 *      `<pre><code class="language-math">` and no `.eqn-num` exists to see, so a blockquote or div
 *      holding both inline maths and a numbered display would otherwise scope the counter from
 *      above and break the numbering just as effectively.
 *
 * Returns how many displays were marked and how many blocks were unmarked, which is what makes
 * this testable without a browser.
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
