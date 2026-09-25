// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useKeyboardShortcutsStore } from "@/features/settings";
import { isTauri } from "@/lib/api-base";
import { useEffect, useRef } from "react";
import { type AppMenuAction, menuAccelerators } from "./app-menu-chords";

export type { AppMenuAction } from "./app-menu-chords";

/** Run app menu actions, and enable in the menu only those with a handler.
 *  A null handler leaves its item disabled, and so does every item while `ready` is false (the
 *  desktop app's install, startup or recovery screen). */
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
  // Joined so the effect re-runs only when a chord really changes.
  const accelerators = useKeyboardShortcutsStore((s) =>
    JSON.stringify(menuAccelerators(s.overrides)),
  );

  useEffect(() => {
    if (!isTauri) return;
    const sync = (actions: string[]) =>
      void import("@tauri-apps/api/core")
        .then(({ invoke }) =>
          invoke("set_app_menu_actions", {
            enabled: actions,
            accelerators: JSON.parse(accelerators),
          }),
        )
        .catch(() => undefined);
    sync(enabled ? enabled.split(",") : []);
    return () => sync([]);
  }, [enabled, accelerators]);

  useEffect(() => {
    if (!isTauri) return;
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
