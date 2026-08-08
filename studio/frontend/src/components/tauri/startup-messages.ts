export const INITIAL_STARTUP_MESSAGE = "Starting Unsloth...";
export const MODELS_STARTUP_MESSAGE = "Loading models...";
export const SERVER_STARTUP_MESSAGE = "Nearly done...";
export const STATUS_MESSAGE_ROTATION_MS = 5_000;

export interface InstallProgressMessage {
  title: string;
  subtitle: string;
}

const INSTALL_PHASE_MESSAGES: readonly InstallProgressMessage[] = [
  {
    title: "Preparing your workspace...",
    subtitle: "Checking what this computer needs.",
  },
  {
    title: "Downloading required components...",
    subtitle: "Creating your local environment.",
  },
  {
    title: "Installing Unsloth...",
    subtitle: "Setting up local AI tools...",
  },
  {
    title: "Finishing setup...",
    subtitle: "Getting everything ready.",
  },
];

const INSTALL_WAITING_SUBTITLES = [
  "Setup will continue automatically.",
  "This can take a few minutes.",
  "Some components take longer to configure.",
] as const;

function normalizedRotationIndex(index: number, length: number): number {
  const wholeIndex = Number.isFinite(index) ? Math.trunc(index) : 0;
  return ((wholeIndex % length) + length) % length;
}

export function installProgressMessage(
  currentStepIndex: number,
  rotationIndex = 0,
): InstallProgressMessage {
  const phaseIndex =
    currentStepIndex < 2 ? 0 : currentStepIndex < 4 ? 1 : currentStepIndex < 6 ? 2 : 3;
  const phaseMessage = INSTALL_PHASE_MESSAGES[phaseIndex];
  if (rotationIndex === 0) return phaseMessage;

  return {
    title: phaseMessage.title,
    subtitle:
      INSTALL_WAITING_SUBTITLES[
        normalizedRotationIndex(rotationIndex - 1, INSTALL_WAITING_SUBTITLES.length)
      ],
  };
}

export type StartupMessage =
  | typeof INITIAL_STARTUP_MESSAGE
  | typeof MODELS_STARTUP_MESSAGE
  | typeof SERVER_STARTUP_MESSAGE;

const STARTUP_WAITING_MESSAGES = [
  INITIAL_STARTUP_MESSAGE,
  "Loading projects...",
] as const;

export function startupWaitingMessage(
  phaseMessage: StartupMessage,
  rotationIndex: number,
): string {
  if (phaseMessage !== INITIAL_STARTUP_MESSAGE) return phaseMessage;
  return STARTUP_WAITING_MESSAGES[
    normalizedRotationIndex(rotationIndex, STARTUP_WAITING_MESSAGES.length)
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
