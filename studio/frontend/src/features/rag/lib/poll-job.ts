// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type IndexJob, terminalJobStatus } from "../types/rag";

/** Waiting for one ingestion job to settle.
 *
 * Lives here rather than in rag-api so the node:test runner can reach it: that
 * module imports the auth barrel, which pulls in an image asset the runner
 * cannot load. The fetch is a parameter for the same reason.
 */

export const JOB_POLL_MS = 700;
// Long enough for a large re-index, bounded so a worker that dies without
// writing a terminal status cannot hang the caller forever.
export const JOB_WAIT_TIMEOUT_MS = 5 * 60_000;

/** Poll until a job reaches a terminal status, and return it.
 *
 * For a caller that must know the outcome before reporting success -- saving an
 * edited source, where the reply only means the re-index started. Polling rather
 * than the SSE stream because this waits for one short-lived job and needs no
 * progress; `useRagDocuments` keeps the streaming path for the list.
 *
 * A finished-and-failed job is returned, not thrown: that is an outcome the
 * caller reports, distinct from the wait itself breaking. Only a job that never
 * settles throws, so a dead worker surfaces as an error rather than a spinner
 * that never ends.
 */
export async function pollJobUntilTerminal(
  fetchJob: (jobId: string) => Promise<IndexJob>,
  jobId: string,
  {
    timeoutMs = JOB_WAIT_TIMEOUT_MS,
    pollMs = JOB_POLL_MS,
  }: { timeoutMs?: number; pollMs?: number } = {},
): Promise<IndexJob> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = await fetchJob(jobId);
    if (terminalJobStatus(job.status)) return job;
    if (Date.now() >= deadline) {
      throw new Error("Timed out waiting for indexing to finish");
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
}
