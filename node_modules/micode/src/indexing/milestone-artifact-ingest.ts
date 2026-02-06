import { type ArtifactIndex, getArtifactIndex } from "../tools/artifact-index";
import { log } from "../utils/logger";
import {
  classifyMilestoneArtifact,
  MILESTONE_ARTIFACT_TYPES,
  type MilestoneArtifactType,
} from "./milestone-artifact-classifier";

export interface MilestoneArtifactInput {
  id: string;
  milestoneId: string;
  sourceSessionId?: string;
  createdAt?: string;
  tags?: string[];
  payload: string;
}

export async function ingestMilestoneArtifact(
  input: MilestoneArtifactInput,
  index?: ArtifactIndex,
  classifier: (content: string) => MilestoneArtifactType = classifyMilestoneArtifact,
): Promise<void> {
  const artifactIndex = index ?? (await getArtifactIndex());
  let artifactType: MilestoneArtifactType;

  try {
    artifactType = classifier(input.payload);
  } catch (error) {
    log.error("milestone-ingest", "Failed to classify milestone artifact, defaulting to session", error);
    artifactType = MILESTONE_ARTIFACT_TYPES.SESSION;
  }

  await artifactIndex.indexMilestoneArtifact({
    id: input.id,
    milestoneId: input.milestoneId,
    artifactType,
    sourceSessionId: input.sourceSessionId,
    createdAt: input.createdAt,
    tags: input.tags,
    payload: input.payload,
  });
}
