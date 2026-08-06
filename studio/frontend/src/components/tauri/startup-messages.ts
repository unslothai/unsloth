export const INITIAL_STARTUP_MESSAGE = "Loading Unsloth...";
export const MODELS_STARTUP_MESSAGE = "Loading models...";
export const SERVER_STARTUP_MESSAGE = "Starting server...";
export const SERVER_START_FALLBACK_MS = 3_000;

export interface InstallProgressMessage {
  title: string;
  subtitle: string;
}

const GETTING_READY_MESSAGE: InstallProgressMessage = {
  title: "Getting things ready...",
  subtitle: "This won’t take long.",
};

export function installProgressMessage(
  currentStepIndex: number,
): InstallProgressMessage {
  if (currentStepIndex < 2) return GETTING_READY_MESSAGE;
  if (currentStepIndex < 4) {
    return {
      title: "Preparing your workspace...",
      subtitle: "Setting up everything Unsloth needs.",
    };
  }
  if (currentStepIndex < 6) {
    return {
      title: "Installing Unsloth...",
      subtitle: "Your private AI workspace is taking shape.",
    };
  }
  return {
    title: "Nearly done...",
    subtitle: "Finishing setup.",
  };
}

export type StartupMessage =
  | typeof INITIAL_STARTUP_MESSAGE
  | typeof MODELS_STARTUP_MESSAGE
  | typeof SERVER_STARTUP_MESSAGE;

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
