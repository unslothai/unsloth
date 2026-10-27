// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ActiveModelDownload } from "./api.ts";

export function isTrainingDownloadVariant(
  variant: string | null | undefined,
): boolean {
  return /^@training-[0-9a-f]{32}$/.test(variant ?? "");
}

/** Only real server-owned jobs are adopted; a selected model alone is no download. */
export function trainingDownloads(downloads: readonly ActiveModelDownload[]) {
  return downloads.filter(
    (job) =>
      job.owner === "training" &&
      isTrainingDownloadVariant(job.variant) &&
      Boolean(job.repo_id?.trim()) &&
      Number.isSafeInteger(job.generation) &&
      (job.state === "running" || job.state === "cancelling"),
  );
}

/** Drop late responses after a session switch or global-view teardown. */
export class TrainingDownloadDiscovery {
  private epoch = 0;
  private disposed = false;
  private abort = new AbortController();
  private ports: {
    fetch: (signal: AbortSignal) => Promise<readonly ActiveModelDownload[]>;
    session: () => string;
    adopt: (job: ActiveModelDownload) => void;
  };

  constructor(ports: TrainingDownloadDiscovery["ports"]) {
    this.ports = ports;
  }

  async poll(): Promise<boolean> {
    const epoch = this.epoch;
    const session = this.ports.session();
    const current = () =>
      !this.disposed &&
      epoch === this.epoch &&
      session === this.ports.session();
    try {
      const jobs = await this.ports.fetch(this.abort.signal);
      if (current())
        for (const job of trainingDownloads(jobs)) this.ports.adopt(job);
    } catch {
      /* Transient failures retain regular manager rows and their own polling. */
    }
    return current();
  }

  reset(): void {
    this.epoch += 1;
    this.abort.abort();
    this.abort = new AbortController();
  }

  dispose(): void {
    this.disposed = true;
    this.reset();
  }
}
