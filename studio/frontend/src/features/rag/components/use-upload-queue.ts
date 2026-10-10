// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Holds files added while an upload is running and starts them once it finishes, so a second
 * drop or pick waits its turn instead of being refused. `key` names the destination (a chat, a
 * project): files queued for one are dropped, with a toast, if the surface moves to another.
 */
export function useUploadQueue<T>(
  run: (items: T[]) => void,
  busy: boolean,
  key: string | null,
): { enqueue: (items: T[]) => void; queued: number } {
  const queue = useRef<{ key: string | null; items: T[] }>({ key, items: [] });
  const [queued, setQueued] = useState(0);
  const latestRun = useRef(run);
  latestRun.current = run;
  const latestKey = useRef(key);
  latestKey.current = key;

  const enqueue = useCallback(
    (items: T[]) => {
      if (items.length === 0) return;
      // A folder walk resolves after the fact: the surface may show another destination by now,
      // and latestRun would upload there.
      if (key !== latestKey.current) {
        toast.info(
          items.length === 1
            ? "A dropped file was not added"
            : `${items.length} dropped files were not added`,
          { description: "You moved away before they finished reading." },
        );
        return;
      }
      if (!busy && queue.current.items.length === 0) {
        latestRun.current(items);
        return;
      }
      queue.current = { key, items: [...queue.current.items, ...items] };
      setQueued(queue.current.items.length);
      toast.info(
        items.length === 1 ? "Queued 1 file" : `Queued ${items.length} files`,
        { description: "They upload when the current upload finishes." },
      );
    },
    [busy, key],
  );

  useEffect(() => {
    const pending = queue.current;
    if (pending.items.length === 0) return;
    if (pending.key !== key) {
      queue.current = { key, items: [] };
      setQueued(0);
      toast.info(
        pending.items.length === 1
          ? "A queued file was not added"
          : `${pending.items.length} queued files were not added`,
        { description: "You moved away before the earlier upload finished." },
      );
      return;
    }
    if (busy) return;
    queue.current = { key, items: [] };
    setQueued(0);
    latestRun.current(pending.items);
  }, [busy, key]);

  return { enqueue, queued };
}
