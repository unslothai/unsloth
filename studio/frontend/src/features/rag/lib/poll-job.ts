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
// How long a job may go without visibly advancing before it is treated as dead. This
// bounds *silence*, not total duration: embedding a near-limit source on a slow CPU can
// take far longer than this and is not a failure, so the budget is renewed every time the
// job's stage or progress moves. Only a worker that stops saying anything trips it.
//
// This relies on the backend reporting progress *within* a long stage, not just on entering
// it: ingestion._run publishes one update per embedding batch for exactly that reason. A
// stage that reported itself once and then went quiet for its whole duration would look
// identical to a dead worker from here, however long the budget.
export const JOB_WAIT_TIMEOUT_MS = 5 * 60_000;

/** Poll until a job reaches a terminal status, and return it.
 *
 * For a caller that must know the outcome before reporting success -- saving an
 * edited source, where the reply only means the re-index started. Polling rather
 * than the SSE stream because this waits for one short-lived job and needs no
 * progress; `useRagDocuments` keeps the streaming path for the list.
 *
 * A finished-and-failed job is returned, not thrown: that is an outcome the
 * caller reports, distinct from the wait itself breaking. Only a job that stops
 * advancing throws, so a dead worker surfaces as an error rather than a spinner
 * that never ends -- while a live slow one is waited out. Converting elapsed time
 * into failure would report a save as lost and offer a retry that can only 409
 * against the job still holding the claim, and the edit would then land anyway.
 */
export async function pollJobUntilTerminal(
  fetchJob: (jobId: string) => Promise<IndexJob>,
  jobId: string,
  {
    timeoutMs = JOB_WAIT_TIMEOUT_MS,
    pollMs = JOB_POLL_MS,
  }: { timeoutMs?: number; pollMs?: number } = {},
): Promise<IndexJob> {
  let deadline = Date.now() + timeoutMs;
  let lastSeen: string | null = null;
  for (;;) {
    const job = await fetchJob(jobId);
    if (terminalJobStatus(job.status)) return job;
    // A live worker moves through stages and reports progress as it goes. Any change is
    // proof it is still there, so the budget starts again from that moment.
    const seen = `${job.status} ${job.stage ?? ""} ${job.progress ?? ""}`;
    if (seen !== lastSeen) {
      lastSeen = seen;
      deadline = Date.now() + timeoutMs;
    }
    if (Date.now() >= deadline) {
      throw new Error("Timed out waiting for indexing to finish");
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
}
