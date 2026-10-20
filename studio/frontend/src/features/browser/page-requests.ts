// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PageResponse } from "./api";
import type { PageRequest } from "./frame-message";

// Each one holds a connection to Studio until the site answers, and browsers keep 6 per host over
// HTTP/1.1; shared by every page so some stay free for Studio itself. Then a bounded queue per page.
const MAX_RUNNING = 3;
const MAX_QUEUED = 256;
let running = 0;
// Pages with requests waiting, in turn order. A page that starts one goes to the back, so one busy
// page can't take every slot another page is waiting for.
const turns = new Set<() => boolean>();

function schedule() {
  for (const startOne of turns) {
    if (running >= MAX_RUNNING) return;
    turns.delete(startOne);
    if (startOne()) turns.add(startOne);
  }
}

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
  /** Starts this page's next request; whether more are waiting. */
  const startOne = () => {
    while (queue.length) {
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
          schedule();
        });
      break;
    }
    return queue.length > 0;
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
      turns.add(startOne);
      schedule();
    },
    abort(id: string) {
      const controller = open.get(id);
      controller?.abort();
      open.delete(id);
      // A queued one leaves the queue, so aborted requests don't fill its bound.
      const at = queue.findIndex((job) => job.controller === controller);
      if (at >= 0) queue.splice(at, 1);
    },
    /** The page is gone: end all it asked for, queued or running. */
    stop() {
      for (const controller of open.values()) controller.abort();
      open.clear();
      queue.length = 0;
      turns.delete(startOne);
    },
  };
}
