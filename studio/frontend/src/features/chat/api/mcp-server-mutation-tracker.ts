// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const pendingMcpServerMutations = new Set<Promise<void>>();
const mutationSettlementListeners = new Set<(epoch: number) => void>();
let mcpServerMutationEpoch = 0;

export function getMcpServerMutationEpoch(): number {
  return mcpServerMutationEpoch;
}

export function subscribeToMcpServerMutationSettlements(
  listener: (epoch: number) => void,
): () => void {
  mutationSettlementListeners.add(listener);
  return () => mutationSettlementListeners.delete(listener);
}

export function trackMcpServerMutation<T>(mutation: Promise<T>): Promise<T> {
  // The settlement promise always fulfills, so background waiters never add unhandled rejections.
  mcpServerMutationEpoch += 1;
  const settlement = mutation.then(
    () => undefined,
    () => undefined,
  );
  pendingMcpServerMutations.add(settlement);
  void settlement.then(() => {
    pendingMcpServerMutations.delete(settlement);
    mcpServerMutationEpoch += 1;
    const settledEpoch = mcpServerMutationEpoch;
    for (const listener of [...mutationSettlementListeners]) {
      try {
        listener(settledEpoch);
      } catch {
        // Observers must not change caller promise semantics or block siblings.
      }
    }
  });
  return mutation;
}

export async function readMcpServerMutationSnapshot<T>(
  read: () => Promise<T>,
): Promise<T> {
  // Retry only when a mutation starts or settles during the read, not while one is pending.
  while (true) {
    const epochBeforeRead = mcpServerMutationEpoch;
    try {
      const result = await read();
      if (mcpServerMutationEpoch === epochBeforeRead) return result;
    } catch (error) {
      if (mcpServerMutationEpoch === epochBeforeRead) throw error;
    }
  }
}

export async function waitForPendingMcpServerMutations(): Promise<void> {
  // Re-snapshot until empty so a mutation starting mid-settlement is not missed.
  while (pendingMcpServerMutations.size > 0) {
    await Promise.all([...pendingMcpServerMutations]);
  }
}

export async function readAfterPendingMcpServerMutations<T>(
  read: () => Promise<T>,
): Promise<T> {
  while (true) {
    await waitForPendingMcpServerMutations();
    const epochBeforeRead = mcpServerMutationEpoch;
    let result: T;
    try {
      result = await read();
    } catch (error) {
      await waitForPendingMcpServerMutations();
      if (mcpServerMutationEpoch !== epochBeforeRead) continue;
      throw error;
    }
    await waitForPendingMcpServerMutations();
    if (mcpServerMutationEpoch === epochBeforeRead) return result;
  }
}
