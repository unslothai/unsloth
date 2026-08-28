// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const pendingMcpServerMutations = new Set<Promise<void>>();
const mutationSettlementListeners = new Set<(epoch: number) => void>();
let mcpServerMutationEpoch = 0;

export function subscribeToMcpServerMutationSettlements(
  listener: (epoch: number) => void,
): () => void {
  mutationSettlementListeners.add(listener);
  return () => mutationSettlementListeners.delete(listener);
}

export function trackMcpServerMutation<T>(mutation: Promise<T>): Promise<T> {
  // The returned operation keeps its original result/rejection for its caller.
  // The internal settlement promise always fulfills so background waiters never
  // create an additional unhandled rejection when a component unmounts.
  mcpServerMutationEpoch += 1;
  const settlement = mutation.then(
    () => undefined,
    () => undefined,
  );
  pendingMcpServerMutations.add(settlement);
  void settlement.then(() => {
    pendingMcpServerMutations.delete(settlement);
    mcpServerMutationEpoch += 1;
    if (pendingMcpServerMutations.size !== 0) return;

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

export async function waitForPendingMcpServerMutations(): Promise<void> {
  // A mutation may begin while an earlier batch is settling. Re-snapshot until
  // the module-level set is empty so an open-time refresh cannot miss it.
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
