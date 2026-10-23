// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { posix } from "node:path";
import postcss, { type AtRule, type Container, type Root, type Rule } from "postcss";
import selectorParser, { type Node as SelectorNode, type Selector } from "postcss-selector-parser";
import type { Plugin } from "vite";

const VIEWPORT_FEATURE =
  /\b(?:(?:min|max)-)?(?:device-)?(?:width|height|aspect-ratio)\b|\borientation\b/i;
const GATED_AT_RULES = new Set([
  "media",
  "supports",
  "layer",
  "container",
  "starting-style",
]);
const LEGACY_PSEUDO_ELEMENTS = new Set([":before", ":after", ":first-line", ":first-letter"]);
const ATTRIBUTE_PREFIX = "data-mq-";
const BOOT_SCRIPT_NAME = "viewport-media.js";

export type ViewportQuery = { attribute: string; query: string };

type Gate = { attribute: string; present: boolean };

function isPseudoElement(node: SelectorNode): boolean {
  return (
    node.type === "pseudo" &&
    (node.value.startsWith("::") || LEGACY_PSEUDO_ELEMENTS.has(node.value.toLowerCase()))
  );
}

function gateComplex(selector: Selector, root: string): string {
  const nodes = [...selector.nodes];
  while (nodes[0]?.type === "comment" || (nodes[0]?.type === "combinator" && !nodes[0].value.trim())) {
    nodes.shift();
  }
  const first = nodes[0];
  const last = nodes[nodes.length - 1];
  if (!first || !last) throw new Error(`empty selector in "${String(selector)}"`);
  first.spaces.before = "";
  last.spaces.after = "";
  const cut = nodes.findIndex((node) => node.type === "combinator" || isPseudoElement(node));
  const left = nodes.slice(0, cut === -1 ? nodes.length : cut).join("");
  const right = cut === -1 ? "" : nodes.slice(cut).join("");
  return `:where(${root}) ${left}${right},${left}:where(${root})${right}`;
}

export function gateSelector(selector: string, root: string): string {
  const list = selectorParser().astSync(selector, { lossless: true }).nodes;
  if (list.map(String).join(",") !== selector) {
    throw new Error(`selector list "${selector}" does not round-trip`);
  }
  return list.map((complex) => gateComplex(complex, root)).join(",");
}

function gateRoot(gates: readonly Gate[]): string {
  return `html${gates
    .map(({ attribute, present }) => (present ? `[${attribute}]` : `:not([${attribute}])`))
    .join("")}`;
}

function attributeFor(query: string): string {
  const slug = query
    .toLowerCase()
    .replace(/>=/g, "-gte-")
    .replace(/<=/g, "-lte-")
    .replace(/>/g, "-gt-")
    .replace(/</g, "-lt-")
    .replace(/=/g, "-eq-")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${ATTRIBUTE_PREFIX}${slug}`;
}

export function viewportGate(condition: string): { query: string; present: boolean } | null {
  const trimmed = condition.trim();
  if (!VIEWPORT_FEATURE.test(trimmed)) return null;
  const negated = /^not\s+all\s+and\s+(\([^()]*\))$/i.exec(trimmed);
  if (negated) return { query: negated[1], present: false };
  return /^\([^()]*\)$/.test(trimmed) ? { query: trimmed, present: true } : null;
}

function gatesEverything(block: AtRule): boolean {
  let gated = true;
  block.walkAtRules((rule) => {
    if (GATED_AT_RULES.has(rule.name.toLowerCase())) return;
    gated = false;
    return false;
  });
  return gated;
}

function gateRule(rule: Rule, root: string): void {
  rule.selector = gateSelector(rule.raws.selector?.raw ?? rule.selector, root);
  delete rule.raws.selector;
}

function printCopy(block: AtRule): AtRule {
  return postcss.atRule({
    name: "media",
    params: "print",
    raws: { before: "", afterName: " ", between: "", after: "" },
    nodes: [block.clone({ raws: { ...block.raws, before: "" } })],
  });
}

function rewrite(container: Container, gates: readonly Gate[], queries: Map<string, string>): void {
  for (const node of [...(container.nodes ?? [])]) {
    if (node.type === "rule") {
      if (gates.length > 0) gateRule(node, gateRoot(gates));
      continue;
    }
    if (node.type !== "atrule" || !node.nodes) continue;
    const name = node.name.toLowerCase();
    const gate = name === "media" ? viewportGate(node.params) : null;
    if (!gate || !gatesEverything(node)) {
      if (GATED_AT_RULES.has(name)) rewrite(node, gates, queries);
      continue;
    }
    const found = gates.length === 0 ? new Map(queries) : queries;
    const attribute = found.get(gate.query) ?? attributeFor(gate.query);
    found.set(gate.query, attribute);
    const screen = node.clone({ params: "screen", raws: { ...node.raws, params: undefined } });
    if (gates.length > 0) {
      rewrite(screen, [...gates, { attribute, present: gate.present }], found);
      node.replaceWith(screen);
      continue;
    }
    try {
      rewrite(screen, [{ attribute, present: gate.present }], found);
    } catch {
      continue;
    }
    found.forEach((value, query) => queries.set(query, value));
    node.replaceWith(screen, printCopy(node));
  }
}

function gateStylesheet(css: string, from?: string): { root: Root; queries: ViewportQuery[] } {
  const queries = new Map<string, string>();
  const root = postcss.parse(css, { from });
  rewrite(root, [], queries);
  return { root, queries: [...queries].map(([query, attribute]) => ({ attribute, query })) };
}

export function gateViewportMedia(css: string): { css: string; queries: ViewportQuery[] } {
  const { root, queries } = gateStylesheet(css);
  return { css: queries.length === 0 ? css : root.toString(), queries };
}

export function viewportBootScript(queries: readonly ViewportQuery[]): string {
  // Escaped beyond JSON so the data stays safe if this script is ever inlined into HTML.
  const entries = JSON.stringify(queries.map(({ attribute, query }) => [attribute, query]))
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
  return `(function(){var root=document.documentElement;var entries=${entries};entries.forEach(function(entry){var list=window.matchMedia(entry[1]);var apply=function(){root.toggleAttribute(entry[0],list.matches)};apply();list.addEventListener("change",apply)});window.addEventListener("resize",function(){entries.forEach(function(entry){root.toggleAttribute(entry[0],window.matchMedia(entry[1]).matches)})})})();\n`;
}

export function viewportMediaGate(): Plugin[] {
  const queries = new Map<string, ViewportQuery>();
  let base = "/";
  let sourcemap = false;
  return [
    {
      name: "viewport-media-gate",
      apply: "build",
      configResolved(config) {
        sourcemap = Boolean(config.build.sourcemap);
      },
      transform: {
        filter: { id: { include: /\.css(?:$|\?)/, exclude: /[?&](?:inline|raw|url)\b/ }, code: "@media" },
        handler(code, id) {
          let gated: { root: Root; queries: ViewportQuery[] };
          try {
            gated = gateStylesheet(code, id);
          } catch (error) {
            this.warn(`left ${id} ungated: ${error instanceof Error ? error.message : String(error)}`);
            return null;
          }
          if (gated.queries.length === 0) return null;
          for (const query of gated.queries) queries.set(query.query, query);
          if (!sourcemap) return { code: gated.root.toString(), map: null };
          const result = gated.root.toResult({ to: id, map: { inline: false, annotation: false, sourcesContent: true } });
          return { code: result.css, map: result.map.toString() };
        },
      },
    },
    {
      name: "viewport-media-boot",
      apply: "build",
      enforce: "post",
      configResolved(config) {
        base = config.base;
      },
      generateBundle(_options, bundle) {
        if (queries.size === 0) return;
        const sorted = [...queries.values()].sort((a, b) => a.attribute.localeCompare(b.attribute));
        const fileName = this.getFileName(
          this.emitFile({ type: "asset", name: BOOT_SCRIPT_NAME, source: viewportBootScript(sorted) }),
        );
        for (const asset of Object.values(bundle)) {
          if (asset.type !== "asset" || !asset.fileName.endsWith(".html")) continue;
          const html = String(asset.source);
          const head = html.indexOf("<head");
          const script = head === -1 ? -1 : html.indexOf("<script", head);
          const at = script === -1 ? html.indexOf("</head>") : script;
          const src = base === "./" || base === ""
            ? `./${posix.relative(posix.dirname(asset.fileName), fileName)}`
            : `${base}${fileName}`;
          if (at !== -1) asset.source = `${html.slice(0, at)}<script src="${src.replace(/&/g, "&amp;").replace(/"/g, "&quot;")}"></script>\n    ${html.slice(at)}`;
        }
      },
    },
  ];
}
