// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { useEffect, useRef, useState } from "react";
import {
  type NativeDropTargetHandlers,
  registerNativeDropTarget,
} from "./native-drop-targets";

/**
 * Claim native drops landing on the returned ref's element. Without this a drop
 * anywhere on the window goes to the chat-wide handler, which is how a file
 * dropped on a dialog's own drop zone ended up attached to the chat behind it.
 */
export function useNativeDropTarget(
  options: NativeDropTargetHandlers & { enabled?: boolean },
): (element: HTMLElement | null) => void {
  const { enabled = true } = options;
  const [element, setElement] = useState<HTMLElement | null>(null);
  const latest = useRef(options);
  latest.current = options;

  useEffect(() => {
    if (!isTauri || !enabled || !element) return;
    return registerNativeDropTarget(element, {
      onDrop: (paths) => latest.current.onDrop(paths),
      onDragOver: (over) => latest.current.onDragOver?.(over),
    });
  }, [element, enabled]);

  return setElement;
}
