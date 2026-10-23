// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Flow expression language. Same semantics as studio/backend/core/flows/expr.py;
// both run studio/backend/tests/fixtures/flows_expr_cases.json.

import { EXPR_OPS, type Expr } from "./types";

const MAX_STEPS = 10_000;
const MAX_DEPTH = 64;
const MAX_DICE = 100;
const MAX_SIDES = 1000;
const DICE_RE = /^\s*(\d*)\s*d\s*(\d+)\s*(?:([+-])\s*(\d+))?\s*$/i;
const OPS = new Set<string>(EXPR_OPS);

export class ExprError extends Error {}

export interface Rng {
  random(): number;
}

/** mulberry32; `state` is a uint32 that can be saved and restored. */
export class Prng implements Rng {
  state: number;
  constructor(seed = 0) {
    this.state = seed >>> 0;
  }
  random(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    const a = this.state;
    let t = Math.imul(a ^ (a >>> 15), 1 | a) >>> 0;
    t = ((t + (Math.imul(t ^ (t >>> 7), 61 | t) >>> 0)) >>> 0) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  randint(lo: number, hi: number): number {
    if (hi < lo) [lo, hi] = [hi, lo];
    return lo + Math.floor(this.random() * (hi - lo + 1));
  }
}

/** FNV-1a 32 over the parts joined by "|". */
export function seedFrom(...parts: unknown[]): number {
  let h = 0x811c9dc5;
  const bytes = new TextEncoder().encode(parts.map(fmt).join("|"));
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function randint(rng: Rng, lo: number, hi: number): number {
  if (hi < lo) [lo, hi] = [hi, lo];
  return lo + Math.floor(rng.random() * (hi - lo + 1));
}

export function rollDice(
  spec: string,
  rng: Rng,
): { total: number; rolls: number[] } {
  const m = DICE_RE.exec(String(spec ?? ""));
  if (!m) throw new ExprError(`bad dice "${spec}": use a form like 2d6+1`);
  const count = Number(m[1] || 1);
  const sides = Number(m[2]);
  if (count < 1 || count > MAX_DICE || sides < 1 || sides > MAX_SIDES) {
    throw new ExprError(`dice "${spec}" out of range`);
  }
  const rolls = Array.from({ length: count }, () => randint(rng, 1, sides));
  const bonus = Number(m[4] || 0) * (m[3] === "-" ? -1 : 1);
  return { total: rolls.reduce((s, r) => s + r, 0) + bonus, rolls };
}

const isNum = (x: unknown): x is number => typeof x === "number";
const isObj = (x: unknown): x is Record<string, unknown> =>
  typeof x === "object" && x !== null && !Array.isArray(x);

function norm(x: number): number {
  if (!Number.isFinite(x)) throw new ExprError("number out of range");
  return x === 0 ? 0 : x;
}

export function fmt(x: unknown): string {
  if (x === null || x === undefined) return "";
  if (typeof x === "boolean") return x ? "true" : "false";
  if (isNum(x)) return String(norm(x));
  if (typeof x === "string") return x;
  return JSON.stringify(x);
}

export function truthy(x: unknown): boolean {
  if (x === null || x === undefined || x === false) return false;
  if (isNum(x)) return x !== 0;
  if (typeof x === "string" || Array.isArray(x)) return x.length > 0;
  if (isObj(x)) return Object.keys(x).length > 0;
  return Boolean(x);
}

function eq(a: unknown, b: unknown): boolean {
  if (typeof a === "boolean" || typeof b === "boolean") return a === b;
  if (isNum(a) && isNum(b)) return a === b;
  if ((a === null || a === undefined) && (b === null || b === undefined))
    return true;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length)
      return false;
    return a.every((x, i) => eq(x, b[i]));
  }
  if (isObj(a) || isObj(b)) {
    if (!isObj(a) || !isObj(b)) return false;
    const ka = Object.keys(a);
    const kb = new Set(Object.keys(b));
    if (ka.length !== kb.size || !ka.every((k) => kb.has(k))) return false;
    return ka.every((k) => eq(a[k], b[k]));
  }
  return a === b;
}

export function lookup(vars: Record<string, unknown>, path: string): unknown {
  let cur: unknown = vars;
  for (const part of String(path).split(".")) {
    if (isObj(cur)) {
      cur = Object.hasOwn(cur, part) ? cur[part] : null;
    } else if (Array.isArray(cur) && /^-?\d+$/.test(part)) {
      let i = Number(part);
      if (i < 0) i += cur.length;
      cur = i >= 0 && i < cur.length ? cur[i] : null;
    } else {
      return null;
    }
    if (cur === null || cur === undefined) return null;
  }
  return cur;
}

function num(x: unknown, op: string): number {
  if (!isNum(x)) throw new ExprError(`${op} needs numbers, got '${fmt(x)}'`);
  return x;
}

function strLen(s: string): number {
  return Array.from(s).length;
}

class Evaluator {
  steps = 0;
  private vars: Record<string, unknown>;
  private rng: Rng | null;
  constructor(vars: Record<string, unknown>, rng: Rng | null) {
    this.vars = vars;
    this.rng = rng;
  }

  needRng(): Rng {
    if (!this.rng) this.rng = new Prng(0);
    return this.rng;
  }

  ev(e: unknown, depth = 0): unknown {
    this.steps += 1;
    if (this.steps > MAX_STEPS) throw new ExprError("expression too long");
    if (depth > MAX_DEPTH) throw new ExprError("expression too deep");
    if (e === null || e === undefined) return null;
    if (typeof e === "boolean" || typeof e === "string") return e;
    if (isNum(e)) return norm(e);
    if (Array.isArray(e)) throw new ExprError('use {"list": [...]} for a list');
    if (!isObj(e)) throw new ExprError("bad expression");
    if ("var" in e) {
      if (typeof e.var !== "string")
        throw new ExprError("var must be a string");
      return lookup(this.vars, e.var);
    }
    if ("list" in e) {
      if (!Array.isArray(e.list)) throw new ExprError("list must be an array");
      return e.list.map((x) => this.ev(x, depth + 1));
    }
    const op = e.op;
    const args = e.args ?? [];
    if (typeof op !== "string" || !OPS.has(op))
      throw new ExprError(`unknown op '${String(op)}'`);
    if (!Array.isArray(args)) throw new ExprError("args must be an array");
    return this.apply(op, args, depth + 1);
  }

  apply(op: string, raw: unknown[], depth: number): unknown {
    const ev = (x: unknown) => this.ev(x, depth);
    const arity = (n: number) => {
      if (raw.length !== n)
        throw new ExprError(`${op} takes ${n} argument${n !== 1 ? "s" : ""}`);
    };
    if (op === "and") {
      for (const a of raw) if (!truthy(ev(a))) return false;
      return true;
    }
    if (op === "or") {
      for (const a of raw) if (truthy(ev(a))) return true;
      return false;
    }
    if (op === "if") {
      arity(3);
      return truthy(ev(raw[0])) ? ev(raw[1]) : ev(raw[2]);
    }
    if (op === "rand") {
      arity(0);
      return this.needRng().random();
    }
    const args = raw.map(ev);
    switch (op) {
      case "not":
        arity(1);
        return !truthy(args[0]);
      case "==":
      case "!=": {
        arity(2);
        const r = eq(args[0], args[1]);
        return op === "==" ? r : !r;
      }
      case "<":
      case "<=":
      case ">":
      case ">=": {
        arity(2);
        const [a, b] = args;
        const ok =
          (isNum(a) && isNum(b)) ||
          (typeof a === "string" && typeof b === "string");
        if (!ok)
          throw new ExprError(`${op} compares two numbers or two strings`);
        const x = a as number | string;
        const y = b as number | string;
        return op === "<"
          ? x < y
          : op === "<="
            ? x <= y
            : op === ">"
              ? x > y
              : x >= y;
      }
      case "+": {
        if (args.length === 0) return 0;
        if (args.every((a) => typeof a === "string"))
          return (args as string[]).join("");
        if (args.every((a) => Array.isArray(a)))
          return (args as unknown[][]).flat(1);
        let total = 0;
        for (const a of args) total += num(a, op);
        return norm(total);
      }
      case "-":
        if (args.length === 1) return norm(-num(args[0], op));
        arity(2);
        return norm(num(args[0], op) - num(args[1], op));
      case "*": {
        let total = 1;
        for (const a of args) total *= num(a, op);
        return norm(total);
      }
      case "/":
      case "%": {
        arity(2);
        const a = num(args[0], op);
        const b = num(args[1], op);
        if (b === 0) throw new ExprError("division by zero");
        return norm(op === "/" ? a / b : a - b * Math.floor(a / b));
      }
      case "min":
      case "max": {
        const vals =
          args.length === 1 && Array.isArray(args[0])
            ? (args[0] as unknown[])
            : args;
        if (vals.length === 0) throw new ExprError(`${op} of nothing`);
        const nums = vals.map((v) => num(v, op));
        return norm(op === "min" ? Math.min(...nums) : Math.max(...nums));
      }
      case "abs":
        arity(1);
        return norm(Math.abs(num(args[0], op)));
      case "round": {
        if (args.length !== 1 && args.length !== 2)
          throw new ExprError("round takes 1 or 2 arguments");
        const digits = args.length === 2 ? Math.trunc(num(args[1], op)) : 0;
        const f = 10 ** digits;
        return norm(Math.floor(num(args[0], op) * f + 0.5) / f);
      }
      case "len": {
        arity(1);
        const a = args[0];
        if (a === null || a === undefined) return 0;
        if (typeof a === "string") return strLen(a);
        if (Array.isArray(a)) return a.length;
        if (isObj(a)) return Object.keys(a).length;
        throw new ExprError("len needs text, a list or an object");
      }
      case "contains": {
        arity(2);
        const [a, b] = args;
        if (Array.isArray(a)) return a.some((x) => eq(x, b));
        if (typeof a === "string")
          return typeof b === "string" && a.includes(b);
        if (isObj(a)) return typeof b === "string" && Object.hasOwn(a, b);
        return false;
      }
      case "concat":
        return args.map(fmt).join("");
      case "lower":
      case "upper": {
        arity(1);
        const s = fmt(args[0]);
        return op === "lower" ? s.toLowerCase() : s.toUpperCase();
      }
      case "randint": {
        arity(2);
        return randint(
          this.needRng(),
          Math.trunc(num(args[0], op)),
          Math.trunc(num(args[1], op)),
        );
      }
      case "dice":
        arity(1);
        return rollDice(String(args[0]), this.needRng()).total;
    }
    throw new ExprError(`unknown op '${op}'`);
  }
}

export function evalExpr(
  expr: Expr | unknown,
  vars: Record<string, unknown> = {},
  rng: Rng | null = null,
): unknown {
  return new Evaluator(vars, rng).ev(expr);
}

/** Evaluate every `{var: expr}` against the vars as they were before; returns the updated copy. */
export function applySets(
  sets: Record<string, Expr> | null | undefined,
  vars: Record<string, unknown>,
  rng: Rng | null = null,
): Record<string, unknown> {
  const out = { ...vars };
  for (const [name, e] of Object.entries(sets ?? {}))
    out[name] = evalExpr(e, vars, rng);
  return out;
}
