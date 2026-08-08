// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Parsing for the accelerator-health block of `GET /api/system/hardware?include_details=true`.
 *
 * Split out of use-hardware-info so it stays importable without react or the auth chain,
 * which is what lets it be unit tested directly.
 */

/** What an xformers wheel records in its own cpp_lib.json. */
export interface AcceleratorBuild {
  torch: string | null;
  cuda: string | null;
  // ROCm wheels record hip and leave cuda null; without this a ROCm xformers renders no
  // build detail at all.
  hip: string | null;
  python: string | null;
}

/**
 * One optimized-kernel package. `installed` and `imports` and `runs` are three separate
 * questions on purpose: an xformers built for a different torch is installed, imports,
 * reports a version -- and has no working kernels (NVIDIA QA P0-1). `runs` is null where
 * the question does not apply (absent, unprobed, or no separate kernel-load step).
 */
export interface AcceleratorPackage {
  name: string;
  version: string | null;
  installed: boolean;
  /**
   * Whether the probe child actually ran for THIS package. Null from a backend too old to
   * say, which is the only case the report-wide flag is allowed to answer for: the probe
   * set is per device, so a package the backend deliberately skipped is unknown, not dead.
   */
  probed: boolean | null;
  imports: boolean;
  runs: boolean | null;
  reason: string | null;
  builtFor: AcceleratorBuild | null;
}

export interface AcceleratorReport {
  pythonVersion: string | null;
  torchVersion: string | null;
  torchCuda: string | null;
  /** False when UNSLOTH_SKIP_ACCELERATOR_PROBE is set: unknown, not healthy. */
  probed: boolean;
  packages: AcceleratorPackage[];
  /** Import names of packages that are installed but cannot load. */
  degraded: string[];
}

// Fixed order so the About table does not reshuffle between reads, and so the most
// consequential package (the one in the P0) is first.
const ACCELERATOR_ORDER = ["xformers", "flash_attn", "torchao", "bitsandbytes"];

function parseAcceleratorBuild(raw: unknown): AcceleratorBuild | null {
  if (typeof raw !== "object" || raw === null) return null;
  const build = raw as Record<string, unknown>;
  return {
    torch: (build.torch as string) ?? null,
    cuda: (build.cuda as string) ?? null,
    hip: (build.hip as string) ?? null,
    python: (build.python as string) ?? null,
  };
}

export function parseAcceleratorReport(raw: unknown): AcceleratorReport | null {
  if (typeof raw !== "object" || raw === null) return null;
  const report = raw as Record<string, unknown>;
  const rawPackages = (report.packages ?? {}) as Record<
    string,
    Record<string, unknown>
  >;
  // Unknown names go last rather than being dropped: a backend that grows a package
  // should not need a frontend release to show it.
  const names = Object.keys(rawPackages).sort((a, b) => {
    const ia = ACCELERATOR_ORDER.indexOf(a);
    const ib = ACCELERATOR_ORDER.indexOf(b);
    return (
      (ia < 0 ? ACCELERATOR_ORDER.length : ia) -
      (ib < 0 ? ACCELERATOR_ORDER.length : ib)
    );
  });
  return {
    pythonVersion: (report.python_version as string) ?? null,
    torchVersion: (report.torch_version as string) ?? null,
    torchCuda: (report.torch_cuda as string) ?? null,
    probed: report.probed !== false,
    packages: names.map((name) => {
      const entry = rawPackages[name] ?? {};
      return {
        name,
        version: (entry.version as string) ?? null,
        installed: entry.installed === true,
        probed: typeof entry.probed === "boolean" ? entry.probed : null,
        imports: entry.imports === true,
        runs: typeof entry.runs === "boolean" ? entry.runs : null,
        reason: (entry.reason as string) ?? null,
        builtFor: parseAcceleratorBuild(entry.built_for),
      };
    }),
    degraded: Array.isArray(report.degraded)
      ? (report.degraded as string[])
      : [],
  };
}

export type Health = "working" | "broken" | "absent" | "unknown";

/**
 * Three questions collapse to one badge. `runs === false` and "imports but does not run"
 * are the same thing to a user, and both mean the kernels are dead.
 *
 * `reportProbed` is only the fallback for a backend too old to answer per package. The
 * probe set is per device -- a ROCm host never probes bitsandbytes -- so reading the
 * report-wide flag onto every row renders a deliberately skipped package as "Not loading"
 * the moment any other package is probed, and it is not in `degraded` either.
 */
export function acceleratorHealth(
  pkg: AcceleratorPackage,
  reportProbed: boolean,
): Health {
  if (!pkg.installed) return "absent";
  if (!(pkg.probed ?? reportProbed)) return "unknown";
  if (!pkg.imports || pkg.runs === false) return "broken";
  return "working";
}

/** True when something is installed and cannot load -- the case worth a banner. */
export function hasDeadAccelerator(report: AcceleratorReport | null): boolean {
  return (report?.degraded.length ?? 0) > 0;
}
