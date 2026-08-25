// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export class ThreadRecordWriteCoordinator {
  private readonly pending = new Map<string, Set<Promise<unknown>>>();
  private readonly unconfirmed = new Map<string, Set<symbol>>();
  private readonly blockedError: (threadId: string) => Error;
  private readonly confirmsAbsence: (error: unknown) => boolean;
  private admissionClosed = false;

  constructor(
    blockedError: (threadId: string) => Error,
    confirmsAbsence: (error: unknown) => boolean,
  ) {
    this.blockedError = blockedError;
    this.confirmsAbsence = confirmsAbsence;
  }

  observe<T>(threadId: string, work: Promise<T>): Promise<T> {
    const workForThread = this.pending.get(threadId) ?? new Set();
    workForThread.add(work);
    this.pending.set(threadId, workForThread);

    const forget = () => {
      const current = this.pending.get(threadId);
      current?.delete(work);
      if (current?.size === 0) {
        this.pending.delete(threadId);
      }
    };
    work.then(forget, forget);
    return work;
  }

  write<T>(threadId: string, start: () => Promise<T>): Promise<T> {
    if (this.admissionClosed) {
      return Promise.reject(this.blockedError(threadId));
    }

    const operation = Symbol(threadId);
    const unconfirmedForThread = this.unconfirmed.get(threadId) ?? new Set();
    unconfirmedForThread.add(operation);
    this.unconfirmed.set(threadId, unconfirmedForThread);
    const work = Promise.resolve().then(start);
    const confirm = () => {
      const current = this.unconfirmed.get(threadId);
      current?.delete(operation);
      if (current?.size === 0) {
        this.unconfirmed.delete(threadId);
      }
    };
    work.then(confirm, (error: unknown) => {
      if (this.confirmsAbsence(error)) {
        confirm();
      }
    });
    return this.observe(threadId, work);
  }

  async settleCurrent(threadId: string): Promise<void> {
    const current = this.pending.get(threadId);
    if (current) {
      await Promise.allSettled(Array.from(current));
    }
  }

  hasPending(threadId: string): boolean {
    return this.pending.has(threadId);
  }

  idsRequiringFence(): string[] {
    return Array.from(
      new Set([...this.pending.keys(), ...this.unconfirmed.keys()]),
    );
  }

  confirmFinalState(threadIds: Iterable<string>): void {
    for (const threadId of threadIds) {
      this.unconfirmed.delete(threadId);
    }
  }

  closeAdmission(): () => void {
    this.admissionClosed = true;
    return () => {
      this.admissionClosed = false;
    };
  }
}
