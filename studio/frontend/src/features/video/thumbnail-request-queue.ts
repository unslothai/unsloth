// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Backend poster generation opens and decodes a video, so visible cards must not fan out into an
 * unbounded burst. This queue is shared by the active and archived galleries. */
export const VIDEO_THUMBNAIL_CONCURRENCY = 3;

interface WaitingTask {
  start: () => void;
}

export class ThumbnailRequestQueue {
  private readonly limit: number;
  private readonly waiting: WaitingTask[] = [];
  private activeCount = 0;

  constructor(limit = VIDEO_THUMBNAIL_CONCURRENCY) {
    this.limit = Math.max(1, Math.floor(limit));
  }

  run<T>(task: () => Promise<T>): Promise<T> {
    const result = new Promise<T>((resolve, reject) => {
      this.waiting.push({
        start: () => {
          Promise.resolve()
            .then(task)
            .then(resolve, reject)
            .finally(() => {
              this.activeCount -= 1;
              this.pump();
            });
        },
      });
    });
    this.pump();
    return result;
  }

  get active(): number {
    return this.activeCount;
  }

  get pending(): number {
    return this.waiting.length;
  }

  private pump(): void {
    while (this.activeCount < this.limit) {
      const next = this.waiting.shift();
      if (!next) {
        return;
      }
      this.activeCount += 1;
      next.start();
    }
  }
}

export const videoThumbnailQueue = new ThumbnailRequestQueue();
