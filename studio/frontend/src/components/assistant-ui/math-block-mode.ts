// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * Kept free of JSX and import.meta so node --experimental-strip-types tests can load it.
 * Unset resolves to the ship default (on); an unrecognised value deliberately resolves to "off".
 */
export type MathBlockMode = "off" | "contain";

/* Find-in-page is handled by gateOnEngine; list numbering by math-block-marker.ts. */
export const SHIP_DEFAULT: MathBlockMode = "contain";

/*
 * WebKit before Safari 26 cannot find skipped content-visibility content (webkit.org/b/283846).
 * Anchor positioning shipped in the same release, so it is the proxy; WebKitGTK freezes its UA.
 */
export const FIND_IN_PAGE_PROBE = "anchor-name: --unsloth-probe";

/** An explicit runtime override wins over the probe; a build flag does not. */
export const gateOnEngine = (
  mode: MathBlockMode,
  engineFindsSkippedContent: boolean,
  forcedByRuntime: boolean,
): MathBlockMode =>
  mode !== "contain" || engineFindsSkippedContent || forcedByRuntime ? mode : "off";

export const isRuntimeForced = (runtime: unknown): boolean =>
  runtime === true || runtime === "1" || runtime === "contain";

/**
 * @param runtime  `__UNSLOTH_MATH_BLOCK_CONTAINMENT__`: string, boolean (console form) or absent.
 * @param build    `VITE_UNSLOTH_MATH_BLOCK_CONTAINMENT`, `""` when never set.
 */
export const resolveMathBlockMode = (
  runtime: unknown,
  build: string,
): MathBlockMode => {
  const raw =
    typeof runtime === "string"
      ? runtime
      : runtime === true
        ? "contain"
        : runtime === false
          ? "off"
          : build;
  return raw === "1" || raw === "contain"
    ? "contain"
    : raw === ""
      ? SHIP_DEFAULT
      : "off";
};

/** On documentElement so it is reachable before any thread mounts and flips without a render. */
export const MATH_BLOCK_CONTAINMENT_ATTRIBUTE = "data-math-block-containment";
export const MATH_BLOCK_CONTAINMENT_ON = "on";

/* Redefine the global as an accessor so assigning it from devtools after load reapplies. */
export const installOverrideWatcher = (
  scope: Record<string, unknown>,
  apply: () => MathBlockMode,
): boolean => {
  try {
    let held = scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__;
    Object.defineProperty(scope, "__UNSLOTH_MATH_BLOCK_CONTAINMENT__", {
      configurable: true,
      enumerable: true,
      get: () => held,
      set: (next: unknown) => {
        held = next;
        apply();
      },
    });
    return true;
  } catch {
    // A hostile global must not fail startup: this runs before the first render.
    return false;
  }
};
