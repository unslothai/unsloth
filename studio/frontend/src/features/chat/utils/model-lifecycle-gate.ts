export type ModelLifecycleLease = number;
export type ModelLifecyclePhase = "preparing" | "loading" | "unloading";

/** Exclusive ownership for the singleton local-model lifecycle. The store mirrors this gate into
 *  `modelLoading` for UI consumers, while the lease prevents one async caller from clearing
 *  another caller's loading state. */
export class ModelLifecycleGate {
  private activeLease: ModelLifecycleLease | null = null;
  private phase: ModelLifecyclePhase | null = null;
  private nextLease = 1;

  tryAcquire(
    phase: ModelLifecyclePhase = "preparing",
  ): ModelLifecycleLease | null {
    if (this.activeLease !== null) {
      return null;
    }
    const lease = this.nextLease++;
    this.activeLease = lease;
    this.phase = phase;
    return lease;
  }

  canQueue(): boolean {
    return this.activeLease === null || this.phase === "loading";
  }

  markLoading(lease: ModelLifecycleLease): boolean {
    if (this.activeLease !== lease || this.phase !== "preparing") {
      return false;
    }
    this.phase = "loading";
    return true;
  }

  markFailed(lease: ModelLifecycleLease): boolean {
    if (this.activeLease !== lease || this.phase !== "loading") return false;
    this.phase = "preparing";
    return true;
  }

  release(lease: ModelLifecycleLease): boolean {
    if (this.activeLease !== lease) {
      return false;
    }
    this.activeLease = null;
    this.phase = null;
    return true;
  }
}

export const chatModelLifecycleGate = new ModelLifecycleGate();
