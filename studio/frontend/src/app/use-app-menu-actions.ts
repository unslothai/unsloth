// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { useEffect, useRef } from "react";

/** Actions the desktop File and View menus send (src-tauri/src/app_menu.rs). */
export type AppMenuAction =
  | "new-chat"
  | "new-temporary-chat"
  | "open-folder"
  | "toggle-sidebar"
  | "find"
  | "previous-chat"
  | "next-chat"
  | "back"
  | "forward"
  | "zoom-in"
  | "zoom-out"
  | "actual-size";

/** Run app menu actions, and enable in the menu only those with a handler.
 *  A null handler leaves its item disabled. */
export function useAppMenuActions(
  handlers: Record<AppMenuAction, (() => void) | null>,
): void {
  const latest = useRef(handlers);
  useEffect(() => {
    latest.current = handlers;
  });
  const enabled = (Object.keys(handlers) as AppMenuAction[])
    .filter((action) => handlers[action])
    .join(",");

  useEffect(() => {
    if (!isTauri) return;
    const sync = (actions: string[]) =>
      void import("@tauri-apps/api/core")
        .then(({ invoke }) => invoke("set_app_menu_actions", { enabled: actions }))
        .catch(() => undefined);
    sync(enabled ? enabled.split(",") : []);
    return () => sync([]);
  }, [enabled]);

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
