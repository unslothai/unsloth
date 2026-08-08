// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// NVIDIA QA P0-1: the managed Windows xFormers was built for PyTorch 2.10.0+cu128 and
// Python 3.10.11 while the app ran cu130 and Python 3.13.2, so its CUDA extensions never
// loaded. The About tab showed a version string, which a mismatched wheel reports exactly
// as happily as a working one -- so "installed", "imports" and "runs" have to survive the
// parse as three separate answers, and a backend that predates the field has to parse to
// null rather than to a false all-clear.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { parseAcceleratorReport, hasDeadAccelerator } = await import(
  "../src/hooks/accelerator-report.ts"
);

const BROKEN_XFORMERS = {
  python_version: "3.13.2",
  torch_version: "2.10.0+cu130",
  torch_cuda: "13.0",
  probed: true,
  packages: {
    bitsandbytes: {
      version: "0.48.0",
      installed: true,
      imports: true,
      runs: null,
      reason: null,
    },
    xformers: {
      version: "0.0.34",
      installed: true,
      imports: true,
      runs: false,
      reason: "xformers was built for torch 2.10.0+cu128 ...",
      built_for: {
        torch: "2.10.0+cu128",
        cuda: "12.8",
        hip: null,
        python: "3.10.11",
      },
    },
    flash_attn: {
      version: null,
      installed: false,
      imports: false,
      runs: null,
      reason: null,
    },
  },
  degraded: ["xformers"],
};

test("an installed-but-dead package survives the parse as exactly that", () => {
  const report = parseAcceleratorReport(BROKEN_XFORMERS);
  assert.ok(report);
  const xformers = report.packages.find((p) => p.name === "xformers");
  assert.ok(xformers);
  // The three answers stay separate: collapsing them is how a version string alone
  // came to stand in for "working".
  assert.equal(xformers.installed, true);
  assert.equal(xformers.imports, true);
  assert.equal(xformers.runs, false);
  assert.equal(xformers.builtFor?.torch, "2.10.0+cu128");
  assert.equal(xformers.builtFor?.python, "3.10.11");
});

test("packages come back in a fixed order regardless of key order", () => {
  const report = parseAcceleratorReport(BROKEN_XFORMERS);
  assert.deepEqual(
    report?.packages.map((p) => p.name),
    ["xformers", "flash_attn", "bitsandbytes"],
  );
});

test("an unknown package is kept, and sorted after the known ones", () => {
  // A backend that grows a package must not need a frontend release to show it, and it
  // must not displace the ones the UI knows how to label. The single-package version of
  // this test asserted an ordering it never exercised.
  const report = parseAcceleratorReport({
    packages: {
      somethingnew: { version: "1.0", installed: true, imports: true },
      bitsandbytes: { version: "0.48.0", installed: true, imports: true },
      xformers: { version: "0.0.34", installed: true, imports: true },
    },
    degraded: [],
  });
  assert.deepEqual(
    report?.packages.map((p) => p.name),
    ["xformers", "bitsandbytes", "somethingnew"],
  );
});

test("a ROCm build reports hip, which the CUDA-only parse dropped", () => {
  const report = parseAcceleratorReport({
    packages: {
      xformers: {
        version: "0.0.33",
        installed: true,
        imports: true,
        runs: false,
        built_for: {
          torch: null,
          cuda: null,
          hip: "6.4.43483",
          python: "3.10.11",
        },
      },
    },
    degraded: ["xformers"],
  });
  assert.equal(report?.packages[0].builtFor?.hip, "6.4.43483");
  assert.equal(report?.packages[0].builtFor?.cuda, null);
});

test("a missing runs is null, not false", () => {
  // torchao and bitsandbytes have no separate kernel-load step, so `runs` is unknown
  // rather than broken. Coercing it to false would light the banner on every machine.
  const report = parseAcceleratorReport(BROKEN_XFORMERS);
  const bnb = report?.packages.find((p) => p.name === "bitsandbytes");
  assert.equal(bnb?.runs, null);
});

test("an older backend that sends no report parses to null", () => {
  // Not to an empty healthy report: "we cannot tell" must not render as "all fine".
  assert.equal(parseAcceleratorReport(undefined), null);
  assert.equal(parseAcceleratorReport(null), null);
});

test("probed defaults to true and only an explicit false turns it off", () => {
  assert.equal(parseAcceleratorReport({ packages: {} })?.probed, true);
  assert.equal(
    parseAcceleratorReport({ packages: {}, probed: false })?.probed,
    false,
  );
});

test("the banner fires only on a degraded package", () => {
  assert.equal(
    hasDeadAccelerator(parseAcceleratorReport(BROKEN_XFORMERS)),
    true,
  );
  assert.equal(
    hasDeadAccelerator(
      parseAcceleratorReport({ ...BROKEN_XFORMERS, degraded: [] }),
    ),
    false,
  );
  // No report at all (older backend, or a fetch without include_details) must not
  // produce a banner claiming something is broken.
  assert.equal(hasDeadAccelerator(null), false);
});
