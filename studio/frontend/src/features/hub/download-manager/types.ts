


import type { TransportMode } from "./constants";

export interface TransportConflictInfo {
  previous: TransportMode;
  next: TransportMode;
  resumable: boolean;
}
