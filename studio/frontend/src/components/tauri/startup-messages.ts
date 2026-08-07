export const INITIAL_STARTUP_MESSAGE = "Starting Unsloth...";
export const MODELS_STARTUP_MESSAGE = "Loading application services...";
export const SERVER_STARTUP_MESSAGE = "Starting local server...";
export const STATUS_MESSAGE_ROTATION_MS = 5_000;

export interface InstallProgressMessage {
  title: string;
  subtitle: string;
}

const INSTALL_PROGRESS_MESSAGES: readonly InstallProgressMessage[] = [
  {
    title: "Preparing your installation...",
    subtitle: "Checking what Unsloth needs on this computer.",
  },
  {
    title: "Setting up your workspace...",
    subtitle: "Creating a private environment for Unsloth.",
  },
  {
    title: "Installing required components...",
    subtitle: "This can take a few minutes.",
  },
  {
    title: "Getting local AI tools ready...",
    subtitle: "Setup will continue automatically.",
  },
  {
    title: "Setup is still working...",
    subtitle: "Some components take longer to configure.",
  },
];

function normalizedRotationIndex(index: number, length: number): number {
  const wholeIndex = Number.isFinite(index) ? Math.trunc(index) : 0;
  return ((wholeIndex % length) + length) % length;
}

export function installProgressMessage(
  currentStepIndex: number,
  rotationIndex = 0,
): InstallProgressMessage {
  const phaseOffset =
    currentStepIndex < 2 ? 0 : currentStepIndex < 4 ? 1 : currentStepIndex < 6 ? 2 : 3;
  return INSTALL_PROGRESS_MESSAGES[
    normalizedRotationIndex(phaseOffset + rotationIndex, INSTALL_PROGRESS_MESSAGES.length)
  ];
}

export type StartupMessage =
  | typeof INITIAL_STARTUP_MESSAGE
  | typeof MODELS_STARTUP_MESSAGE
  | typeof SERVER_STARTUP_MESSAGE;

const STARTUP_WAITING_MESSAGES = [
  "Preparing local services...",
  "Getting your workspace ready...",
  "Still getting things ready...",
] as const;

export function startupWaitingMessage(
  phaseMessage: StartupMessage,
  rotationIndex: number,
): string {
  if (rotationIndex === 0) return phaseMessage;
  return STARTUP_WAITING_MESSAGES[
    normalizedRotationIndex(rotationIndex - 1, STARTUP_WAITING_MESSAGES.length)
  ];
}

export function startupMessageFromLog(
  current: StartupMessage,
  line: string,
): StartupMessage {
  const normalized = line.trim();
  if (normalized === "- Starting server...") {
    return SERVER_STARTUP_MESSAGE;
  }
  if (
    current === INITIAL_STARTUP_MESSAGE &&
    normalized === "- loading PyTorch, Unsloth and Transformers..."
  ) {
    return MODELS_STARTUP_MESSAGE;
  }
  return current;
}
