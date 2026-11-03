// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const JINJA_REF_RE = /{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}/g;
const JINJA_EXPR_RE = /{{\s*([^{}]+?)\s*}}/g;
const SIMPLE_JINJA_EXPR_RE = /^[a-zA-Z_][a-zA-Z0-9_.]*$/;
const PLAIN_JINJA_EXPR_RE = /^[a-zA-Z0-9_.\s-]+$/;
const NESTED_REFERENCE_ROOTS = new Set(["user"]);

function isValidNestedReference(expr: string, validSet: Set<string>): boolean {
  if (!expr.includes(".")) {
    return false;
  }
  const parts = expr.split(".").map((part) => part.trim()).filter(Boolean);
  if (parts.length < 2) {
    return false;
  }
  const root = parts[0];
  return validSet.has(root) && NESTED_REFERENCE_ROOTS.has(root);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function extractRefs(template: string): string[] {
  if (!template) {
    return [];
  }
  const refs = new Set<string>();
  for (const match of template.matchAll(JINJA_REF_RE)) {
    if (match[1]) {
      refs.add(match[1]);
    }
  }
  return Array.from(refs);
}

export function findInvalidJinjaReferences(
  template: string,
  validReferences: string[],
): string[] {
  if (!template) {
    return [];
  }
  const validSet = new Set(
    validReferences.map((name) => name.trim()).filter(Boolean),
  );
  const invalid = new Set<string>();

  for (const match of template.matchAll(JINJA_EXPR_RE)) {
    const expr = (match[1] ?? "").trim();
    if (!expr) {
      continue;
    }
    if (SIMPLE_JINJA_EXPR_RE.test(expr)) {
      if (!validSet.has(expr) && !isValidNestedReference(expr, validSet)) {
        invalid.add(expr);
      }
      continue;
    }
    if (PLAIN_JINJA_EXPR_RE.test(expr)) {
      invalid.add(expr);
    }
  }

  return Array.from(invalid);
}

export function replaceRef(
  template: string,
  from: string,
  to: string,
): string {
  if (!template || from === to) {
    return template;
  }
  const pattern = new RegExp(`{{\\s*${escapeRegExp(from)}\\s*}}`, "g");
  const renamed = template.replace(pattern, `{{ ${to} }}`);
  // Inside any other tag (`{{ ticket.title }}`, `{% if ticket.priority %}`) the name is renamed
  // where it is a variable root.
  return renamed.replace(/{{[\s\S]*?}}|{%[\s\S]*?%}/g, (tag) =>
    renameRoot(tag, from, to),
  );
}

/** Rename a variable in a Jinja expression (a tag's inside, or a sampler condition such as
 * `topic == 'science'`) where it is a root: outside string literals, not after a dot or inside a
 * longer name. A filter (`| length`), test (`is defined`) or keyword argument (`length=10`) of the
 * same name is not the column. */
export function renameRoot(expression: string, from: string, to: string): string {
  if (!expression || !from || from === to) return expression;
  const root = new RegExp(
    `(?<![\\w.])(?<!\\|\\s*)(?<!\\bis\\s+(?:not\\s+)?)${escapeRegExp(from)}(?!\\w)(?!\\s*=(?!=))`,
    "g",
  );
  return expression
    .split(/('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")/)
    .map((part, i) => (i % 2 ? part : part.replace(root, to)))
    .join("");
}

export function removeRef(template: string, ref: string): string {
  if (!template) {
    return template;
  }
  const escaped = escapeRegExp(ref);
  const fullLine = new RegExp(`^\\s*{{\\s*${escaped}\\s*}}\\s*$`);
  const inline = new RegExp(`{{\\s*${escaped}\\s*}}`, "g");
  const next = template
    .split("\n")
    .flatMap((line) => {
      if (fullLine.test(line)) {
        return [];
      }
      return [line.replace(inline, "").replace(/\s+$/g, "")];
    })
    .join("\n");
  return next;
}
