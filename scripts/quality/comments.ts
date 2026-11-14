// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Comment rules shared by the commit gate and the repo-wide fixer.

const COMMENT = /^\s*\/\/\s?(.*)$/;
const DIRECTIVE =
  /^(?:eslint|biome|@ts-|SPDX-|Copyright|prettier-|shellcheck|noinspection|sourceMappingURL|istanbul|c8|v8)/;
const BULLET = /^(?:[-*+\u2022]|\d+[.)])\s/;
const SENTENCE_END = /[.!?:;)\]}>,"'`]$/;
const CONTINUES = /^[a-z(]/;
const URL_LIKE = /^\w+:\/\//;

export function commentBody(line) {
  const match = COMMENT.exec(line);
  return match ? match[1].trim() : null;
}

export function isDirective(body) {
  return body !== null && DIRECTIVE.test(body);
}

export function indentation(line) {
  return line.length - line.trimStart().length;
}

/** Join a wrapped line comment with the next line when the pair fits. */
export function joinWrappedComments(text, { width = 96 } = {}) {
  const lines = text.split(/\r?\n/);
  const out = [];
  let joined = 0;

  for (let index = 0; index < lines.length; index += 1) {
    const current = lines[index];
    const next = lines[index + 1];
    const first = commentBody(current);
    const second = next === undefined ? null : commentBody(next);

    const joinable =
      first !== null &&
      second !== null &&
      first.length > 0 &&
      second.length > 0 &&
      indentation(current) === indentation(next) &&
      !isDirective(first) &&
      !isDirective(second) &&
      !BULLET.test(second) &&
      !URL_LIKE.test(second) &&
      !SENTENCE_END.test(first) &&
      CONTINUES.test(second) &&
      `${current.trimEnd()} ${second}`.length <= width;

    if (joinable) {
      out.push(`${current.trimEnd()} ${second}`);
      index += 1;
      joined += 1;
    } else {
      out.push(current);
    }
  }

  return { text: out.join("\n"), joined };
}

/** New files open with the license header and one short `//` summary. */
export function hasFileSummary(lines) {
  const header = lines.slice(0, 6);
  const hasLicense =
    header.some((line) => line.includes("SPDX-License-Identifier:")) &&
    header.some((line) => line.trimStart().startsWith("// Copyright"));
  if (!hasLicense) return false;
  const licenseEnd = Math.max(
    header.findIndex((line) => line.includes("SPDX-License-Identifier:")),
    header.findIndex((line) => line.trimStart().startsWith("// Copyright")),
  );
  return header.slice(licenseEnd + 1).some((line) => {
    const body = commentBody(line);
    return body !== null && !isDirective(body) && body.length > 0;
  });
}

export function splitLines(text) {
  return text.split(/\r?\n/);
}
