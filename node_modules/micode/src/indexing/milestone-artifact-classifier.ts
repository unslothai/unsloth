export const MILESTONE_ARTIFACT_TYPES = {
  FEATURE: "feature",
  DECISION: "decision",
  SESSION: "session",
} as const;

export type MilestoneArtifactType = (typeof MILESTONE_ARTIFACT_TYPES)[keyof typeof MILESTONE_ARTIFACT_TYPES];

const FEATURE_HINTS = ["requirement", "implementation", "capability", "scope", "spec"];
const DECISION_HINTS = ["decision", "decided", "trade-off", "rationale", "chosen"];
const SESSION_HINTS = ["meeting", "status", "discussion", "notes", "update"];

const matchesAny = (content: string, hints: string[]) => hints.some((hint) => content.includes(hint));

export function classifyMilestoneArtifact(content: string): MilestoneArtifactType {
  const normalized = content.toLowerCase();
  const isFeature = matchesAny(normalized, FEATURE_HINTS);
  const isDecision = matchesAny(normalized, DECISION_HINTS);
  const isSession = matchesAny(normalized, SESSION_HINTS);

  if (isFeature) return MILESTONE_ARTIFACT_TYPES.FEATURE;
  if (isDecision) return MILESTONE_ARTIFACT_TYPES.DECISION;
  if (isSession) return MILESTONE_ARTIFACT_TYPES.SESSION;

  return MILESTONE_ARTIFACT_TYPES.SESSION;
}
