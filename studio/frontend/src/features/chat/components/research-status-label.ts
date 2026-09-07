// SPDX-License-Identifier: AGPL-3.0-only
import type { ResearchRunStatus } from "../types/research";

export function researchStatusLabel(status: ResearchRunStatus): string {
  switch (status) {
    case "planning":
      return "Planning";
    case "awaiting_approval":
      return "Review plan";
    case "queued":
      return "Queued";
    case "running":
      return "Researching";
    case "paused":
      return "Paused";
    case "cancelling":
      return "Stopping";
    case "cancelled":
      return "Cancelled";
    case "completed":
      return "Complete";
    case "failed":
      return "Failed";
  }
}
