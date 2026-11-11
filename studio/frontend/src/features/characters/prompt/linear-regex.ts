// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Lorebook regex keys come from imported cards and run on every send, and a JS RegExp can backtrack
 * for minutes. This compiles a key to a Thompson NFA, whose search is linear in the text. Each
 * character class or escape is still checked by a one-character RegExp, so its meaning matches JS.
 * Backreferences and lookaround have no linear form: such a key compiles to null and is skipped.
 */

type Inst =
  | { op: "char"; test: (ch: string) => boolean }
  | { op: "assert"; kind: "^" | "$" | "b" | "B" }
  | { op: "split"; x: number; y: number }
  | { op: "jmp"; x: number }
  | { op: "match" };

type Node =
  | { t: "char"; test: (ch: string) => boolean }
  | { t: "assert"; kind: "^" | "$" | "b" | "B" }
  | { t: "seq"; items: Node[] }
  | { t: "alt"; options: Node[] }
  | { t: "repeat"; node: Node; min: number; max: number };

const MAX_PROGRAM = 256;
// Nodes visited while compiling: a repeat of something empty, (?:a{0}){100}, emits nothing yet still costs work.
const MAX_COMPILE_STEPS = 4096;
const MAX_REPEAT = 100;

class Unsupported extends Error {}

function charTest(source: string, flags: string): (ch: string) => boolean {
  const re = new RegExp(`^(?:${source})$`, flags);
  return (ch) => re.test(ch);
}

function parse(pattern: string, flags: string): Node {
  const charFlags = flags.replace(/[gmy]/g, "");
  const unicode = flags.includes("u");
  let pos = 0;

  function escapeAt(): string {
    // pattern[pos] is the backslash; returns the escape's source and moves past it.
    const start = pos;
    pos += 1;
    const ch = pattern[pos];
    if (ch === undefined) throw new Unsupported();
    if (/[1-9]/.test(ch) || ch === "k") throw new Unsupported();
    if ((ch === "p" || ch === "P") && pattern[pos + 1] === "{") {
      const end = pattern.indexOf("}", pos);
      if (end < 0) throw new Unsupported();
      pos = end + 1;
    } else if (ch === "x") pos += 3;
    else if (ch === "u" && pattern[pos + 1] === "{") {
      const end = pattern.indexOf("}", pos);
      if (end < 0) throw new Unsupported();
      pos = end + 1;
    } else if (ch === "u") pos += 5;
    else if (ch === "c") pos += 2;
    else pos += 1;
    return pattern.slice(start, pos);
  }

  function atom(): Node {
    const ch = pattern[pos];
    if (ch === "(") {
      pos += 1;
      if (pattern[pos] === "?") {
        if (pattern[pos + 1] === ":") pos += 2;
        else if (pattern[pos + 1] === "<" && pattern[pos + 2] !== "=" && pattern[pos + 2] !== "!") {
          const end = pattern.indexOf(">", pos);
          if (end < 0) throw new Unsupported();
          pos = end + 1;
        } else throw new Unsupported();
      }
      const inner = alternation();
      if (pattern[pos] !== ")") throw new Unsupported();
      pos += 1;
      return inner;
    }
    if (ch === "[") {
      const start = pos;
      pos += 1;
      if (pattern[pos] === "^") pos += 1;
      while (pos < pattern.length && pattern[pos] !== "]") {
        if (pattern[pos] === "\\") escapeAt();
        else pos += 1;
      }
      if (pattern[pos] !== "]") throw new Unsupported();
      pos += 1;
      return { t: "char", test: charTest(pattern.slice(start, pos), charFlags) };
    }
    if (ch === "\\") {
      const next = pattern[pos + 1];
      if (next === "b" || next === "B") {
        pos += 2;
        return { t: "assert", kind: next };
      }
      return { t: "char", test: charTest(escapeAt(), charFlags) };
    }
    if (ch === "^" || ch === "$") {
      pos += 1;
      return { t: "assert", kind: ch };
    }
    if (ch === ".") {
      pos += 1;
      return { t: "char", test: charTest(".", charFlags) };
    }
    // In u mode a literal is a whole code point, as the text is read.
    const literal = unicode ? String.fromCodePoint(pattern.codePointAt(pos)!) : ch;
    pos += literal.length;
    return { t: "char", test: charTest(literal.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&"), charFlags) };
  }

  function quantifier(): { min: number; max: number } | null {
    const ch = pattern[pos];
    let range: { min: number; max: number } | null = null;
    if (ch === "*") range = { min: 0, max: Infinity };
    else if (ch === "+") range = { min: 1, max: Infinity };
    else if (ch === "?") range = { min: 0, max: 1 };
    else if (ch === "{") {
      const match = /^\{(\d+)(,(\d*))?\}/.exec(pattern.slice(pos));
      if (!match) return null;
      const min = Number(match[1]);
      const max = match[2] === undefined ? min : match[3] === "" ? Infinity : Number(match[3]);
      if (min > MAX_REPEAT || (max !== Infinity && max > MAX_REPEAT) || max < min) throw new Unsupported();
      pos += match[0].length - 1;
      range = { min, max };
    }
    if (!range) return null;
    pos += 1;
    if (pattern[pos] === "?") pos += 1;
    return range;
  }

  function sequence(): Node {
    const items: Node[] = [];
    while (pos < pattern.length && pattern[pos] !== "|" && pattern[pos] !== ")") {
      const node = atom();
      const range = quantifier();
      if (range && node.t === "assert") throw new Unsupported();
      items.push(range ? { t: "repeat", node, ...range } : node);
    }
    return { t: "seq", items };
  }

  function alternation(): Node {
    const options = [sequence()];
    while (pattern[pos] === "|") {
      pos += 1;
      options.push(sequence());
    }
    return options.length === 1 ? options[0] : { t: "alt", options };
  }

  const root = alternation();
  if (pos !== pattern.length) throw new Unsupported();
  return root;
}

function compile(root: Node): Inst[] {
  const program: Inst[] = [];
  const emit = (inst: Inst) => {
    if (program.length >= MAX_PROGRAM) throw new Unsupported();
    program.push(inst);
    return program.length - 1;
  };
  let steps = 0;
  const walk = (node: Node): void => {
    if (++steps > MAX_COMPILE_STEPS) throw new Unsupported();
    switch (node.t) {
      case "char":
        emit({ op: "char", test: node.test });
        return;
      case "assert":
        emit({ op: "assert", kind: node.kind });
        return;
      case "seq":
        node.items.forEach(walk);
        return;
      case "alt": {
        const jumps: number[] = [];
        node.options.forEach((option, index) => {
          if (index < node.options.length - 1) {
            const split = emit({ op: "split", x: 0, y: 0 });
            (program[split] as { x: number }).x = split + 1;
            walk(option);
            jumps.push(emit({ op: "jmp", x: 0 }));
            (program[split] as { y: number }).y = program.length;
          } else walk(option);
        });
        for (const jump of jumps) (program[jump] as { x: number }).x = program.length;
        return;
      }
      case "repeat": {
        for (let i = 0; i < node.min; i += 1) walk(node.node);
        if (node.max === Infinity) {
          const split = emit({ op: "split", x: 0, y: 0 });
          walk(node.node);
          emit({ op: "jmp", x: split });
          program[split] = { op: "split", x: split + 1, y: program.length };
          return;
        }
        const splits: number[] = [];
        for (let i = node.min; i < node.max; i += 1) {
          splits.push(emit({ op: "split", x: 0, y: 0 }));
          walk(node.node);
        }
        for (const split of splits) program[split] = { op: "split", x: split + 1, y: program.length };
        return;
      }
    }
  };
  walk(root);
  emit({ op: "match" });
  return program;
}

const WORD = /[A-Za-z0-9_]/;
const LINE_END = /[\n\r\u2028\u2029]/;

/** A tester for a lorebook regex key that runs in linear time, or null when the key needs backtracking. */
export function linearRegex(source: string, flags: string): ((text: string) => boolean) | null {
  let program: Inst[];
  try {
    program = compile(parse(source, flags));
  } catch {
    return null;
  }
  const multiline = flags.includes("m");
  const sticky = flags.includes("y");
  const unicode = flags.includes("u");
  return (text) => {
    const chars = unicode ? Array.from(text) : text.split("");
    const holds = (kind: "^" | "$" | "b" | "B", at: number) => {
      const before = chars[at - 1];
      const after = chars[at];
      if (kind === "^") return at === 0 || (multiline && LINE_END.test(before));
      if (kind === "$") return at === chars.length || (multiline && LINE_END.test(after));
      const edge = WORD.test(before ?? "") !== WORD.test(after ?? "");
      return kind === "b" ? edge : !edge;
    };
    let mark = new Int32Array(program.length).fill(-1);
    let current: number[] = [];
    let next: number[] = [];
    // Follows jumps and zero-width steps from pc; true once a thread reaches match.
    const add = (list: number[], start: number, at: number, step: number): boolean => {
      const stack = [start];
      while (stack.length) {
        const pc = stack.pop()!;
        if (mark[pc] === step) continue;
        mark[pc] = step;
        const inst = program[pc];
        if (inst.op === "match") return true;
        if (inst.op === "jmp") stack.push(inst.x);
        else if (inst.op === "split") stack.push(inst.y, inst.x);
        else if (inst.op === "assert") {
          if (holds(inst.kind, at)) stack.push(pc + 1);
        } else list.push(pc);
      }
      return false;
    };
    for (let at = 0; at <= chars.length; at += 1) {
      // Search: a new thread starts at every position, or only at the start for a sticky key.
      if ((!sticky || at === 0) && add(current, 0, at, at)) return true;
      if (at === chars.length) break;
      next = [];
      const ch = chars[at];
      for (const pc of current) {
        const inst = program[pc];
        if (inst.op === "char" && inst.test(ch) && add(next, pc + 1, at + 1, at + 1)) return true;
      }
      current = next;
    }
    mark = new Int32Array(0);
    return false;
  };
}
