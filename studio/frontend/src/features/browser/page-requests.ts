// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PageResponse } from "./api";
import type { PageRequest } from "./frame-message";

// A few at once, as browsers keep per host, then a bounded queue.
const MAX_RUNNING = 6;
const MAX_QUEUED = 256;

type Job = { id: string; request: PageRequest; controller: AbortController };

/** One page's requests to its own site (routes/browser_network.py), made by `send` for the page at
 *  `pageUrl()`. Each gets one `reply`, null when it failed; one the page aborted gets none. */
export function pageRequests(
  send: (page: string, request: PageRequest, signal: AbortSignal) => Promise<PageResponse>,
  reply: (id: string, answer: PageResponse | null) => void,
  pageUrl: () => string | null,
) {
  const open = new Map<string, AbortController>();
  const queue: Job[] = [];
  let running = 0;
  const next = () => {
    while (running < MAX_RUNNING && queue.length) {
      const job = queue.shift() as Job;
      if (job.controller.signal.aborted) continue;
      const page = pageUrl();
      if (!page) {
        open.delete(job.id);
        reply(job.id, null);
        continue;
      }
      running++;
      send(page, job.request, job.controller.signal)
        .catch(() => null)
        .then((answer) => {
          running--;
          if (open.get(job.id) === job.controller) open.delete(job.id);
          if (!job.controller.signal.aborted) reply(job.id, answer);
          next();
        });
    }
  };
  return {
    /** `request` null is one the panel won't make: it fails at once. */
    request(id: string, request: PageRequest | null) {
      if (!request || queue.length >= MAX_QUEUED) {
        reply(id, null);
        return;
      }
      const controller = new AbortController();
      open.get(id)?.abort();
      open.set(id, controller);
      queue.push({ id, request, controller });
      next();
    },
    abort(id: string) {
      open.get(id)?.abort();
      open.delete(id);
    },
    /** The page is gone: end all it asked for, queued or running. */
    stop() {
      for (const controller of open.values()) controller.abort();
      open.clear();
      queue.length = 0;
    },
  };
}
