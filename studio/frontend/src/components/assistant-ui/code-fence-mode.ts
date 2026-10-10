// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * Fence mode, decided in one pure .ts so node --experimental-strip-types tests can run it.
 * "tokenize" is a measurement arm, reachable only by that exact string, never by default.
 */
export type FenceMode = "off" | "defer" | "tokenize" | "window";

/** Moving this line is the whole of "turn deferral on by default"; every override still works. */
export const SHIP_DEFAULT: FenceMode = "defer";

/** Unset takes SHIP_DEFAULT; an unrecognised non-empty value degrades to `off`, never the default. */
export const resolveFenceMode = (
  runtime: unknown,
  build: string,
): FenceMode => {
  const raw =
    typeof runtime === "string"
      ? runtime
      : runtime === true
        ? "defer"
        : runtime === false
          ? "off"
          : build;
  return raw === "1" || raw === "defer"
    ? "defer"
    : raw === "tokenize"
      ? "tokenize"
      : raw === "window"
        ? "window"
        : raw === ""
          ? SHIP_DEFAULT
          : "off";
};
