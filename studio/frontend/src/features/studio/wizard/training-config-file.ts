


const INVALID_FILENAME_CHARACTERS = '<>:"/\\|?*';
const PATH_SEPARATOR_PATTERN = /[\\/]/;
const TIMESTAMP_SEPARATOR_PATTERN = /[:T]/g;
const TRAILING_WINDOWS_FILENAME_PATTERN = /[. ]+$/g;
const WINDOWS_RESERVED_NAME =
  /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/i;
const FILENAME_SEGMENT_MAX_BYTES = 64;
export const MAX_TRAINING_CONFIG_BYTES = 1024 * 1024;
export const TRAINING_CONFIG_TOO_LARGE_ERROR_KEY =
  "studio.training.configTooLarge" as const;

export class TrainingConfigFileError extends Error {
  readonly translationKey = TRAINING_CONFIG_TOO_LARGE_ERROR_KEY;

  constructor() {
    super(TRAINING_CONFIG_TOO_LARGE_ERROR_KEY);
    this.name = "TrainingConfigFileError";
  }
}

export async function readBrowserTrainingConfig(
  file: Pick<File, "size" | "text">,
): Promise<string> {
  if (file.size > MAX_TRAINING_CONFIG_BYTES) {
    throw new TrainingConfigFileError();
  }
  return await file.text();
}

function replaceInvalidFilenameCharacters(value: string): string {
  let result = "";
  let replacing = false;
  for (const character of value) {
    const invalid =
      character.charCodeAt(0) <= 31 ||
      INVALID_FILENAME_CHARACTERS.includes(character);
    if (invalid) {
      if (!replacing) {
        result += "-";
      }
      replacing = true;
      continue;
    }
    result += character;
    replacing = false;
  }
  return result;
}

function truncateUtf8(value: string, maxBytes: number): string {
  let bytes = 0;
  let result = "";
  for (const character of value) {
    const codePoint = character.codePointAt(0) ?? 0;
    const characterBytes =
      codePoint <= 0x7f
        ? 1
        : codePoint <= 0x7ff
          ? 2
          : codePoint <= 0xffff
            ? 3
            : 4;
    if (bytes + characterBytes > maxBytes) {
      break;
    }
    result += character;
    bytes += characterBytes;
  }
  return result;
}

function filenameSegment(value: string | null, fallback: string): string {
  const basename =
    value?.split(PATH_SEPARATOR_PATTERN).filter(Boolean).pop()?.trim() ?? "";
  const sanitized = truncateUtf8(
    replaceInvalidFilenameCharacters(basename),
    FILENAME_SEGMENT_MAX_BYTES,
  ).replace(TRAILING_WINDOWS_FILENAME_PATTERN, "");
  if (!sanitized || WINDOWS_RESERVED_NAME.test(sanitized)) {
    return fallback;
  }
  return sanitized;
}

export function trainingConfigFilename({
  model,
  method,
  dataset,
  now = new Date(),
}: {
  model: string | null;
  method: string | null;
  dataset: string | null;
  now?: Date;
}): string {
  const timestamp = now
    .toISOString()
    .replace(TIMESTAMP_SEPARATOR_PATTERN, "-")
    .slice(0, 19);
  return `${[
    filenameSegment(model, "model"),
    filenameSegment(method, "qlora"),
    filenameSegment(dataset, "dataset"),
    timestamp,
  ].join("_")}.yaml`;
}
