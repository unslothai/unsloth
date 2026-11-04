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
  const parts = expr
    .split(".")
    .map((part) => part.trim())
    .filter(Boolean);
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

const JINJA_TAG_RE = /\{\{([\s\S]*?)\}\}|\{%([\s\S]*?)%\}/g;
const QUOTED_RE = /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/g;
const LOOP_RE = /^for\s+(.+?)\s+in\b([\s\S]*)$/;
const SET_RE = /^set\s+([^=]+?)\s*=([\s\S]*)$/;

export type JinjaScope = { name: string; from: number; to: number };

/** Where a template's own variables are in scope: a `{% for %}` loop's up to its `{% endfor %}`, a
 * `{% set %}`'s from there on, each starting after its own tag (whose right side reads the scope
 * before it). Inside them the name is that variable, not a column. */
export function jinjaScopes(text: string): JinjaScope[] {
  const scopes: JinjaScope[] = [];
  const open: { names: string[]; from: number }[] = [];
  const ident = /[A-Za-z_][A-Za-z0-9_]*/g;
  for (const block of text.matchAll(JINJA_TAG_RE)) {
    if (block[2] === undefined) continue;
    const tag = block[2].replace(QUOTED_RE, '""').trim();
    const at = block.index ?? 0;
    const after = at + block[0].length;
    const loop = LOOP_RE.exec(tag);
    if (loop) open.push({ names: loop[1].match(ident) ?? [], from: after });
    else if (/^endfor\b/.test(tag)) {
      const scope = open.pop();
      for (const name of scope?.names ?? [])
        scopes.push({ name, from: scope?.from ?? 0, to: at });
    } else {
      const set = SET_RE.exec(tag);
      for (const name of set?.[1].match(ident) ?? [])
        scopes.push({ name, from: after, to: text.length });
    }
  }
  // An unclosed loop runs to the end.
  for (const scope of open)
    for (const name of scope.names)
      scopes.push({ name, from: scope.from, to: text.length });
  return scopes;
}

/** Whether ``name`` is the template's own variable at ``at``, not a column. */
export function boundAt(
  scopes: JinjaScope[],
  name: string,
  at: number,
): boolean {
  return scopes.some((s) => s.name === name && s.from <= at && at <= s.to);
}

/** The part of a `{% %}` tag that reads: a loop's iterable and a `set`'s right side, not the names
 * they bind (``head`` is what comes before it). */
export function tagReadPart(body: string): { head: string; read: string } {
  const loop = LOOP_RE.exec(body.trim());
  const set = loop ? null : SET_RE.exec(body.trim());
  const read = loop?.[2] ?? set?.[2];
  if (read === undefined) return { head: "", read: body };
  const cut = body.lastIndexOf(read);
  return { head: body.slice(0, cut), read: body.slice(cut) };
}

export function replaceRef(template: string, from: string, to: string): string {
  if (!template || from === to) {
    return template;
  }
  const scopes = jinjaScopes(template);
  const exact = new RegExp(`^\\s*${escapeRegExp(from)}\\s*$`);
  // In every tag the name is renamed where it is a variable root (`{{ ticket.title }}`,
  // `{% if ticket.priority %}`), except where the template binds it itself (`{% set ticket = ... %}`).
  return template.replace(
    JINJA_TAG_RE,
    (tag, output, statement, at: number) => {
      if (boundAt(scopes, from, at)) return tag;
      if (output !== undefined)
        return exact.test(output)
          ? `{{ ${to} }}`
          : `{{${renameRoot(output, from, to)}}}`;
      const { head, read } = tagReadPart(statement);
      return `{%${head}${renameRoot(read, from, to)}%}`;
    },
  );
}

/** Rename a variable in a Jinja expression (a tag's inside, or a sampler condition such as
 * `topic == 'science'`) where it is a root: outside string literals, not after a dot (spaced or not) or inside a
 * longer name. A filter (`| length`), test (`is defined`) or keyword argument (`length=10`) of the
 * same name is not the column. */
export function renameRoot(
  expression: string,
  from: string,
  to: string,
): string {
  if (!expression || !from || from === to) return expression;
  const root = new RegExp(
    `(?<![\\w.])(?<!\\.\\s*)(?<!\\|\\s*)(?<!\\bis\\s+(?:not\\s+)?)${escapeRegExp(from)}(?!\\w)(?!\\s*=(?!=))`,
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
