// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Re-creates src/vendor/<package>/ from the pinned npm tarballs: checks each tarball's integrity,
// copies the built ESM + types + LICENSE, and points the bare imports between the packages at the
// sibling folders, so Vite, tsc and `node --test` resolve them without an npm dependency.
// Usage: node scripts/vendor-prosemirror.mjs

import { execFileSync } from "node:child_process";
import {
  cpSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const VENDOR = fileURLToPath(new URL("../src/vendor/", import.meta.url));

export const PACKAGES = [
  [
    "orderedmap",
    "2.1.1",
    "sha512-TvAWxi0nDe1j/rtMcWcIj94+Ffe6n7zhow33h40SKxmsmozs6dz/e+EajymfoFcHd7sxNn8yHM8839uixMOV6g==",
    "dist/",
  ],
  [
    "rope-sequence",
    "1.3.4",
    "sha512-UT5EDe2cu2E/6O4igUr5PSFs23nvvukicWHx6GnOPlHAiiYbzNuCRQCuiUdHJQcqKalLKlrYJnjY0ySGsXNQXQ==",
    "dist/",
  ],
  [
    "w3c-keyname",
    "2.2.8",
    "sha512-dpojBhNsCNN7T82Tm7k26A6G9ML3NkhDsnw9n/eoxSRlVBB4CEtIQ/KTCLI2Fwf3ataSXRhYFkQi3SlnFwPvPQ==",
    "",
  ],
  [
    "prosemirror-model",
    "1.25.12",
    "sha512-Ue2gTmXMa7EhpLNhC7J+h4+ykD8ha12K6rrZFFKKJHBForfIStw5gJ6Zrf1mqqAa9NmpmEB4wp9bKmA4eUVjWg==",
    "dist/",
  ],
  [
    "prosemirror-transform",
    "1.12.4",
    "sha512-2/WuPum3pn+O/Vi2vzpi8210ICUe2kIPXdLYjQ09Pv5DtT5bcKWMJhAKPBwvpPjpp5iOYUVwVEr0gOM1BY1Mgw==",
    "dist/",
  ],
  [
    "prosemirror-state",
    "1.4.4",
    "sha512-6jiYHH2CIGbCfnxdHbXZ12gySFY/fz/ulZE333G6bPqIZ4F+TXo9ifiR86nAHpWnfoNjOb3o5ESi7J8Uz1jXHw==",
    "dist/",
  ],
  [
    "prosemirror-view",
    "1.42.9",
    "sha512-bKbfzeKZOu2H2JQU2TaJmYu6EBcor3VnGIR4jR31cbEt1ztUnfUB6bKGV1+fJzOJdrVWR+tRU0208zM3oQx4wA==",
    "dist/",
  ],
  [
    "prosemirror-keymap",
    "1.2.3",
    "sha512-4HucRlpiLd1IPQQXNqeo81BGtkY8Ai5smHhKW9jjPKRc2wQIxksg7Hl1tTI2IfT2B/LgX6bfYvXxEpJl7aKYKw==",
    "dist/",
  ],
  [
    "prosemirror-history",
    "1.5.1",
    "sha512-onlwnqKvgtFg6srZOmDOAhlAy1FIHQpOExORybX0S+DvJepE5tPALrB9iGet/2sqqqrelSnB10OoiOJqAvu2DA==",
    "dist/",
  ],
  [
    "prosemirror-commands",
    "1.7.2",
    "sha512-q6Q6szxqdu9Xd6EcdKsqXghu5nQdZTpB4Q9yd04WRc7/jt763e/rT60Owh0L1GYY+T46o5rD+9lEN36dZS43tw==",
    "dist/",
  ],
  [
    "prosemirror-inputrules",
    "1.5.1",
    "sha512-7wj4uMjKaXWAQ1CDgxNzNtR9AlsuwzHfdFH1ygEHA2KHF2DOEaXl1CJfNPAKCg9qNEh4rum975QLaCiQPyY6Fw==",
    "dist/",
  ],
];

const names = new Set(PACKAGES.map(([name]) => name));
const relink = (source) =>
  source.replace(/from '([\w-]+)'/g, (whole, name) =>
    names.has(name) ? `from '../${name}/index.js'` : whole,
  );

const scratch = mkdtempSync(join(tmpdir(), "vendor-prosemirror-"));
try {
  for (const [name, version, integrity, dir] of PACKAGES) {
    const [packed] = JSON.parse(
      execFileSync(
        "npm",
        ["pack", `${name}@${version}`, "--json", "--pack-destination", scratch],
        { encoding: "utf8" },
      ),
    );
    if (packed.integrity !== integrity)
      throw new Error(`${name}@${version}: integrity ${packed.integrity}`);
    const unpacked = join(scratch, name);
    rmSync(unpacked, { recursive: true, force: true });
    execFileSync("mkdir", ["-p", unpacked]);
    execFileSync("tar", [
      "xzf",
      join(scratch, packed.filename),
      "-C",
      unpacked,
    ]);
    const from = join(unpacked, "package");
    const to = join(VENDOR, name);
    rmSync(to, { recursive: true, force: true });
    execFileSync("mkdir", ["-p", to]);
    cpSync(join(from, "LICENSE"), join(to, "LICENSE"));
    const header = `// ${name} ${version} from npm (${integrity}), MIT: see LICENSE. Imports point at the sibling vendored packages.\n`;
    for (const file of ["index.js", "index.d.ts"]) {
      writeFileSync(
        join(to, file),
        header + relink(readFileSync(join(from, dir, file), "utf8")),
      );
    }
    if (name === "prosemirror-view")
      cpSync(
        join(from, "style", "prosemirror.css"),
        join(to, "prosemirror.css"),
      );
    console.log(`${name}@${version}`);
  }
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
