import type { TranslationKey } from "@/i18n";

export const INITIAL_STARTUP_MESSAGE = "desktopStartup.starting";
export const MODELS_STARTUP_MESSAGE = "desktopStartup.loadingModels";
export const SERVER_STARTUP_MESSAGE = "desktopStartup.nearlyDone";
export const UPDATE_STARTUP_MESSAGE = "desktopStartup.finishingUpdate";
export const STATUS_MESSAGE_ROTATION_MS = 5_000;

export interface InstallProgressMessage {
  title: TranslationKey;
  subtitle: TranslationKey;
}

const INSTALL_PHASE_MESSAGES: readonly InstallProgressMessage[] = [
  {
    title: "desktopStartup.install.preparing",
    subtitle: "desktopStartup.install.preparingDetail",
  },
  {
    title: "desktopStartup.install.downloading",
    subtitle: "desktopStartup.install.downloadingDetail",
  },
  {
    title: "desktopStartup.install.installing",
    subtitle: "desktopStartup.install.installingDetail",
  },
  {
    title: "desktopStartup.install.finishing",
    subtitle: "desktopStartup.install.finishingDetail",
  },
];

const INSTALL_WAITING_SUBTITLES = [
  "desktopStartup.install.continuesAutomatically",
  "desktopStartup.install.fewMinutes",
  "desktopStartup.install.someTakeLonger",
] as const satisfies readonly TranslationKey[];

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
  | typeof SERVER_STARTUP_MESSAGE
  | typeof UPDATE_STARTUP_MESSAGE;

const STARTUP_WAITING_MESSAGES = [
  INITIAL_STARTUP_MESSAGE,
  "desktopStartup.loadingProjects",
] as const satisfies readonly TranslationKey[];

export function startupWaitingMessage(
  phaseMessage: StartupMessage,
  rotationIndex: number,
): TranslationKey {
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
