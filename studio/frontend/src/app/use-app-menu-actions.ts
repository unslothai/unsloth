// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useKeyboardShortcutsStore } from "@/features/settings";
import { isMacPlatform } from "@/features/settings/lib/keyboard-shortcuts";
import { isTauri } from "@/lib/api-base";
import { useEffect, useMemo, useRef } from "react";
import { type AppMenuAction, menuAccelerators } from "./app-menu-chords";

export type { AppMenuAction } from "./app-menu-chords";

const hasAppMenus = isTauri && isMacPlatform();

// Serialized, so the unmount disable never lands before the sync it follows.
let syncQueue: Promise<unknown> = Promise.resolve();
function sync(
  enabled: string[],
  accelerators: Partial<Record<AppMenuAction, string | null>> = {},
): void {
  syncQueue = syncQueue
    .then(() => import("@tauri-apps/api/core"))
    .then(({ invoke }) => invoke("set_app_menu_actions", { enabled, accelerators }))
    .catch(() => undefined);
}

/** Enables only items with a handler; all disabled while `ready` is false. */
export function useAppMenuActions(
  handlers: Record<AppMenuAction, (() => void) | null>,
  ready: boolean,
): void {
  const latest = useRef<Partial<typeof handlers>>(handlers);
  useEffect(() => {
    latest.current = ready ? handlers : {};
  });
  const enabled = ready
    ? (Object.keys(handlers) as AppMenuAction[]).filter((action) => handlers[action]).join(",")
    : "";
  const overrides = useKeyboardShortcutsStore((s) => s.overrides);
  const accelerators = useMemo(
    () => (hasAppMenus ? JSON.stringify(menuAccelerators(overrides)) : "{}"),
    [overrides],
  );

  useEffect(() => {
    if (!hasAppMenus) return;
    sync(enabled ? enabled.split(",") : [], JSON.parse(accelerators));
  }, [enabled, accelerators]);

  useEffect(() => {
    if (!hasAppMenus) return;
    return () => sync([]);
  }, []);

  useEffect(() => {
    if (!hasAppMenus) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void import("@tauri-apps/api/event")
      .then(({ listen }) =>
        listen<AppMenuAction>("app-menu-action", ({ payload }) =>
          latest.current[payload]?.(),
        ),
      )
      .then((cleanup) => {
        if (disposed) cleanup();
        else unlisten = cleanup;
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);
}
