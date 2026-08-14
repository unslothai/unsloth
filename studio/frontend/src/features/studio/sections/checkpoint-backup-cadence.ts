export const BACKUP_INTERVAL_OPTIONS = [1, 2, 3, 5] as const;
export const MAX_BACKUP_INTERVAL_CHECKPOINTS = 1000;
export const MAX_REMOTE_CHECKPOINTS = 20;

export function effectiveBackupSteps(
  saveSteps: number,
  intervalCheckpoints: number,
): number {
  return saveSteps * intervalCheckpoints;
}

export function backupIntervalError(value: number): string | null {
  return Number.isInteger(value) &&
    value >= 1 &&
    value <= MAX_BACKUP_INTERVAL_CHECKPOINTS
    ? null
    : "Enter at least 1 checkpoint.";
}

export function remoteCheckpointRetentionError(value: number): string | null {
  return Number.isInteger(value) &&
    value >= 1 &&
    value <= MAX_REMOTE_CHECKPOINTS
    ? null
    : `Enter a whole number from 1 to ${MAX_REMOTE_CHECKPOINTS}.`;
}
