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
const NESTED_RE = /\([^()]*\)|\[[^[\]]*\]|\{[^{}]*\}/g;

/** Text with every bracketed part blanked, so only its top-level commas and `=`s are left. */
function topLevel(text: string): string {
  let out = text;
  for (let prev = ""; prev !== out; ) {
    prev = out;
    out = out.replace(NESTED_RE, (m) => " ".repeat(m.length));
  }
  return out;
}

/** The parameter names in a macro's or call block's `(...)` list: `(topic, n=1)` binds topic and n. */
function paramNames(list: string): string[] {
  return topLevel(list)
    .split(",")
    .map((p) => /^\s*([A-Za-z_][A-Za-z0-9_]*)/.exec(p)?.[1])
    .filter((n): n is string => Boolean(n));
}

/** The names a block tag binds for its body: `with a = 1, b = 2`, `macro show(topic)`, `call(user) ...`. */
function blockNames(
  tag: string,
): { kind: string; names: string[]; declares?: string } | null {
  const word = /^(with|macro|call)\b\s*([\s\S]*)$/.exec(tag);
  if (!word) return null;
  const [, kind, rest] = word;
  if (kind === "with")
    return {
      kind,
      names: topLevel(rest)
        .split(",")
        .map((p) => /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)/.exec(p)?.[1])
        .filter((n): n is string => Boolean(n)),
    };
  // `macro show(topic)` lists its parameters after the name, `call(user) list(...)` right away.
  const open =
    kind === "macro" ? rest.indexOf("(") : rest.startsWith("(") ? 0 : -1;
  // A macro's own name is bound where it is declared, as a `set` is.
  const declares =
    kind === "macro" ? /^([A-Za-z_][A-Za-z0-9_]*)/.exec(rest)?.[1] : undefined;
  if (open < 0) return { kind, names: [], declares };
  const close = topLevel(rest.slice(open + 1)).indexOf(")");
  return {
    kind,
    declares,
    names: paramNames(
      rest.slice(open + 1, close < 0 ? undefined : open + 1 + close),
    ),
  };
}

/** Where a macro header's parameter defaults are (`macro show(value=topic)` reads topic), as
 * [start, end) offsets into ``body``. The macro's name and parameter names read nothing. */
function macroDefaults(body: string): [number, number][] {
  const open = body.indexOf("(");
  if (open < 0) return [];
  const list = topLevel(body.slice(open + 1));
  const close = list.indexOf(")");
  const spans: [number, number][] = [];
  let start = open + 1;
  for (const param of (close < 0 ? list : list.slice(0, close)).split(",")) {
    const eq = /=(?!=)/.exec(param);
    if (eq) spans.push([start + eq.index + 1, start + param.length]);
    start += param.length + 1;
  }
  return spans;
}

const RAW_RE =
  /\{%[-+]?\s*raw\s*[-+]?%\}[\s\S]*?(?:\{%[-+]?\s*endraw\s*[-+]?%\}|$)/g;

/** Where a template's `{% raw %}...{% endraw %}` blocks are, as [start, end) offsets: their insides
 * are literal text, not tags. An unclosed one runs to the end, as Jinja reads it. */
export function rawSpans(text: string): [number, number][] {
  return [...text.matchAll(RAW_RE)].map((m) => [
    m.index ?? 0,
    (m.index ?? 0) + m[0].length,
  ]);
}

/** Whether offset ``at`` is inside a raw block (after its `{% raw %}` tag starts). */
export function inRaw(spans: [number, number][], at: number): boolean {
  return spans.some(([a, b]) => a < at && at < b);
}

export type JinjaScope = { name: string; from: number; to: number };

/** Where a template's own variables are in scope: a `{% for %}` loop's up to its `{% endfor %}` (or
 * its own `{% else %}`), a `with`'s, macro's or call block's names up to its end tag, a `{% set %}`'s
 * from there on, each starting after its own tag (whose right side reads the scope before it).
 * Inside them the name is that variable, not a column. */
export function jinjaScopes(text: string): JinjaScope[] {
  const scopes: JinjaScope[] = [];
  // A block's own names, and the `set`s made inside it: both end at its end tag.
  const open: {
    kind: string;
    names: string[];
    from: number;
    sets: { name: string; from: number }[];
  }[] = [];
  // Open `for`, `if`, `with`, `macro` and `call` blocks, innermost last, to tell a loop's own
  // `{% else %}` from an `if`'s and match each end tag.
  const blocks: string[] = [];
  const ident = /[A-Za-z_][A-Za-z0-9_]*/g;
  const close = (scope: (typeof open)[number], to: number) => {
    for (const name of scope.names) scopes.push({ name, from: scope.from, to });
    for (const set of scope.sets) scopes.push({ ...set, to });
  };
  const raw = rawSpans(text);
  for (const block of text.matchAll(JINJA_TAG_RE)) {
    if (block[2] === undefined || inRaw(raw, block.index ?? 0)) continue;
    const tag = unmarked(block[2].replace(QUOTED_RE, '""'));
    const at = block.index ?? 0;
    const after = at + block[0].length;
    const loop = LOOP_RE.exec(tag);
    const bound = loop ? null : blockNames(tag);
    const end = /^end(for|with|macro|call)\b/.exec(tag)?.[1];
    if (loop || bound) {
      const kind = bound?.kind ?? "for";
      const names = loop ? (loop[1].match(ident) ?? []) : (bound?.names ?? []);
      const outer = open[open.length - 1];
      if (bound?.declares) {
        if (outer) outer.sets.push({ name: bound.declares, from: after });
        else
          scopes.push({ name: bound.declares, from: after, to: text.length });
      }
      open.push({ kind, names, from: after, sets: [] });
      blocks.push(kind);
    } else if (end) {
      if (blocks[blocks.length - 1] === end) blocks.pop();
      if (open[open.length - 1]?.kind === end) close(open.pop()!, at);
    } else if (/^if\b/.test(tag)) blocks.push("if");
    else if (/^endif\b/.test(tag)) {
      if (blocks[blocks.length - 1] === "if") blocks.pop();
    } else if (/^else$/.test(tag) && blocks[blocks.length - 1] === "for") {
      // A loop's `{% else %}` runs when it had no items, in the enclosing scope: its names end here.
      const scope = open[open.length - 1];
      if (scope) {
        close(scope, at);
        scope.names = [];
        scope.sets = [];
        scope.from = after;
      }
    } else {
      const set = SET_RE.exec(tag);
      const inBlock = open[open.length - 1];
      // A `set` inside an `if` (within its own scope block) may not run, so a read after it can still be
      // the column: it binds nothing here, and a rename renames it with the reads (replaceRef).
      if (blocks[blocks.length - 1] === "if") continue;
      // `{% set ns.topic = ... %}` sets an attribute of `ns`: it binds no variable named topic.
      const targets = (set?.[1] ?? "")
        .split(",")
        .map((t) => t.trim())
        .filter((t) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(t));
      for (const name of targets) {
        if (inBlock) inBlock.sets.push({ name, from: after });
        else scopes.push({ name, from: after, to: text.length });
      }
    }
  }
  // An unclosed block runs to the end.
  for (const scope of open) close(scope, text.length);
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

/** A tag's inside without Jinja's whitespace control (`{%- endfor %}`, `{% for x in y -%}`). */
function unmarked(body: string): string {
  return body
    .trim()
    .replace(/^[-+]\s*/, "")
    .replace(/\s*[-+]$/, "");
}

/** The parts of a `{% %}` tag that read: a loop's iterable and a `set`'s right side, not the names
 * they bind (``head`` comes before them). A loop's filter (`for x in items if x > 0`) is ``filter``:
 * its own variables are bound there (``bound``). A macro's header and a call block's parameter
 * list name its own parameters: they read nothing. */
export function tagReadPart(body: string): {
  head: string;
  read: string;
  filter: string;
  bound: string[];
} {
  const header = /^(macro\b|call\s*\()/.exec(unmarked(body));
  if (header?.[1] === "macro")
    return {
      head: body,
      read: macroDefaults(body)
        .map(([a, b]) => body.slice(a, b))
        .join(" , "),
      filter: "",
      bound: [],
    };
  if (header) {
    // `call(user) list_users(users)`: the call after the parameter list reads.
    const start = body.indexOf("(");
    const close = topLevel(body.slice(start + 1)).indexOf(")");
    if (close >= 0) {
      const cut = start + 2 + close;
      return {
        head: body.slice(0, cut),
        read: body.slice(cut),
        filter: "",
        bound: [],
      };
    }
  }
  const loop = LOOP_RE.exec(unmarked(body));
  const set = loop ? null : SET_RE.exec(unmarked(body));
  const read = loop?.[2] ?? set?.[2];
  if (read === undefined)
    return { head: "", read: body, filter: "", bound: [] };
  const cut = body.lastIndexOf(read);
  const rest = body.slice(cut);
  // The filter starts at a top-level `if` (outside quotes); `recursive` ends a loop header too.
  const masked = rest.replace(QUOTED_RE, (q) => " ".repeat(q.length));
  const split = loop ? masked.search(/\s(if|recursive)\b/) : -1;
  return {
    head: body.slice(0, cut),
    read: split < 0 ? rest : rest.slice(0, split),
    filter: split < 0 ? "" : rest.slice(split),
    bound: loop ? (loop[1].match(/[A-Za-z_][A-Za-z0-9_]*/g) ?? []) : [],
  };
}

export function replaceRef(template: string, from: string, to: string): string {
  if (!template || from === to) {
    return template;
  }
  const scopes = jinjaScopes(template);
  const raw = rawSpans(template);
  const exact = new RegExp(`^\\s*${escapeRegExp(from)}\\s*$`);
  // In every tag the name is renamed where it is a variable root (`{{ ticket.title }}`,
  // `{% if ticket.priority %}`), except where the template binds it itself (`{% set ticket = ... %}`).
  return template.replace(
    JINJA_TAG_RE,
    (tag, output, statement, at: number) => {
      if (boundAt(scopes, from, at) || inRaw(raw, at)) return tag;
      if (output !== undefined)
        return exact.test(output)
          ? `{{ ${to} }}`
          : `{{${renameRoot(output, from, to)}}}`;
      if (/^macro\b/.test(unmarked(statement))) {
        // Only the parameter defaults read; rename them right to left so offsets hold.
        let next = statement;
        for (const [a, b] of macroDefaults(statement).reverse())
          next =
            next.slice(0, a) +
            renameRoot(next.slice(a, b), from, to) +
            next.slice(b);
        return `{%${next}%}`;
      }
      const { head, read, filter, bound } = tagReadPart(statement);
      // A conditional `set` of the name binds nothing (jinjaScopes): it is renamed with the reads.
      const target = new RegExp(`(?<![\\w.])${escapeRegExp(from)}(?![\\w])`);
      if (
        SET_RE.test(unmarked(statement)) &&
        !boundAt(scopes, from, at + tag.length)
      )
        return `{%${head.replace(target, to)}${renameRoot(read, from, to)}${filter}%}`;
      const kept = bound.includes(from) ? filter : renameRoot(filter, from, to);
      return `{%${head}${renameRoot(read, from, to)}${kept}%}`;
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
