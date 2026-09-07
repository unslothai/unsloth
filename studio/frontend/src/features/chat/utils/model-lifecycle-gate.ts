export type ModelLifecycleLease = number;

/** Exclusive ownership for the singleton local-model lifecycle. The store mirrors this gate into
 *  `modelLoading` for UI consumers, while the lease prevents one async caller from clearing
 *  another caller's loading state. */
export class ModelLifecycleGate {
  private activeLease: ModelLifecycleLease | null = null;
  private nextLease = 1;

  tryAcquire(): ModelLifecycleLease | null {
    if (this.activeLease !== null) {
      return null;
    }
    const lease = this.nextLease++;
    this.activeLease = lease;
    return lease;
  }

  release(lease: ModelLifecycleLease): boolean {
    if (this.activeLease !== lease) {
      return false;
    }
    this.activeLease = null;
    return true;
  }
}

export const chatModelLifecycleGate = new ModelLifecycleGate();
