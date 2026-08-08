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

const {
  parseAcceleratorReport,
  hasDeadAccelerator,
  acceleratorHealth,
  acceleratorShowsReason,
} = await import("../src/hooks/accelerator-report.ts");

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

test("a package the backend chose not to probe reads as unknown, not broken", () => {
  // The probe set is per device: a ROCm host probes flash_attn and skips bitsandbytes, an
  // Intel host does the reverse. Reading the report-wide flag onto every row rendered the
  // skipped ones as "Not loading" as soon as one other row was probed -- a red badge for a
  // package that is fine, and one the banner does not even list, since it is not degraded.
  const report = parseAcceleratorReport({
    probed: true,
    packages: {
      xformers: {
        version: "0.0.34",
        installed: true,
        probed: true,
        imports: true,
        runs: true,
      },
      bitsandbytes: {
        version: "0.48.0",
        installed: true,
        probed: false,
        imports: false,
        runs: null,
        reason: "not used on this device",
      },
    },
    degraded: [],
  });
  assert.ok(report);
  const [xformers, bnb] = report.packages;
  assert.equal(acceleratorHealth(xformers, report.probed), "working");
  assert.equal(acceleratorHealth(bnb, report.probed), "unknown");
});

test("a backend too old to answer per package falls back to the report flag", () => {
  const older = parseAcceleratorReport({
    probed: true,
    packages: {
      xformers: { version: "0.0.34", installed: true, imports: true, runs: true },
    },
    degraded: [],
  });
  assert.ok(older);
  assert.equal(older.packages[0].probed, null);
  assert.equal(acceleratorHealth(older.packages[0], older.probed), "working");
  assert.equal(acceleratorHealth(older.packages[0], false), "unknown");
});

test("not installed stays not installed, probed or not", () => {
  const report = parseAcceleratorReport(BROKEN_XFORMERS);
  const flash = report?.packages.find((p) => p.name === "flash_attn");
  assert.ok(flash);
  assert.equal(acceleratorHealth(flash, true), "absent");
  // And a probed package that is dead is still dead.
  const xformers = report?.packages.find((p) => p.name === "xformers");
  assert.ok(xformers);
  assert.equal(acceleratorHealth({ ...xformers, probed: true }, true), "broken");
});

test("a probe that ran but could not decide is unknown, not working", () => {
  // `runs === null` from a probe that DID run means the kernel question could not be
  // answered: an xformers layout with no recognised load record, a missing bitsandbytes
  // checker, a torch with no dispatcher table, a flash-attn on a card no prebuilt wheel
  // covers. It rendered as Working -- the false all-clear this report exists to remove,
  // and one nothing else corrects, since those packages are not in `degraded` either.
  const report = parseAcceleratorReport({
    probed: true,
    packages: {
      flash_attn: {
        version: "2.8.3",
        installed: true,
        probed: true,
        imports: true,
        runs: null,
        reason: "no prebuilt wheel covers compute capability 12.0",
      },
      xformers: {
        version: "0.0.34",
        installed: true,
        probed: true,
        imports: true,
        runs: true,
      },
    },
    degraded: [],
  });
  assert.ok(report);
  const [xformers, flash] = report.packages;
  assert.equal(acceleratorHealth(xformers, report.probed), "working");
  assert.equal(acceleratorHealth(flash, report.probed), "unknown");
  // A failed import is still broken, whatever runs says.
  assert.equal(
    acceleratorHealth({ ...flash, imports: false }, report.probed),
    "broken",
  );
});


test("an unknown result keeps the reason the backend sent with it", () => {
  // Most unknowns are deliberate and carry an explanation -- flash-attn imported with no
  // kernel launched, xformers registering an op whose image may be missing, torchao with no
  // native operator. The row showed "Not checked" and discarded all of it, so a skipped
  // native extension looked identical to a probe that never ran.
  assert.equal(acceleratorShowsReason("unknown", "no kernel was launched"), true);
  assert.equal(acceleratorShowsReason("broken", null), true);
  // A reasonless unknown really is "not checked"; there is nothing to say.
  assert.equal(acceleratorShowsReason("unknown", null), false);
  assert.equal(acceleratorShowsReason("unknown", ""), false);
  assert.equal(acceleratorShowsReason("working", "ignored"), false);
  assert.equal(acceleratorShowsReason("absent", "ignored"), false);
});
