// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  gateOnEngine,
  isRuntimeForced,
  type MathBlockMode,
} from "./math-block-mode.ts";

export type ContainmentMode = MathBlockMode;

export type ContainmentFlag = {
  attribute: string;
  on: string;
  global: string;
  shipDefault: ContainmentMode;
  resolve: (runtime: unknown, build: string) => ContainmentMode;
  mode: (
    runtime: unknown,
    build: string,
    engineFindsSkippedContent: boolean,
  ) => ContainmentMode;
  installWatcher: (
    scope: Record<string, unknown>,
    apply: () => ContainmentMode,
  ) => boolean;
};

export function defineContainmentFlag({
  attribute,
  global,
  shipDefault = "contain",
}: {
  attribute: string;
  global: string;
  shipDefault?: ContainmentMode;
}): ContainmentFlag {
  const resolve = (runtime: unknown, build: string): ContainmentMode => {
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
        ? shipDefault
        : "off";
  };
  return {
    attribute,
    on: "on",
    global,
    shipDefault,
    resolve,
    mode: (runtime, build, engineFindsSkippedContent) =>
      gateOnEngine(
        resolve(runtime, build),
        engineFindsSkippedContent,
        isRuntimeForced(runtime),
      ),
    installWatcher: (scope, apply) => {
      try {
        let held = scope[global];
        Object.defineProperty(scope, global, {
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
        return false;
      }
    },
  };
}
