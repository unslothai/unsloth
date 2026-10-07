// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Audio API examples, built without React so the node test runner can check every one.

import {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_REPO,
  type AudioCppWorkflow,
  audioCppModelFor,
  audioCppWorkflowsFor,
  isAudioCppFolderId,
  isCloneAndConvertFamilyId,
  isCloneOnlyFamilyId,
  isConvertOnlyFamilyId,
  isSpeakAndCloneFamilyId,
} from "../../audio/audio-cpp-catalog";
import { EDIT_ADAPTERS } from "../../audio/edit-adapters";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import type { AudioApiModel } from "../api/openai-model-catalog";
import { psSingle, shSingle } from "./agent-command";
import { keylessBaseEligible } from "./keyless-example-eligibility";

export type AudioApiTab = "speak" | "clone" | "transcribe" | "workflows";
export type AudioRunWorkflow = "separate" | "convert" | "music" | "edit";
export type AudioApiLang = "curl" | "python" | "javascript";
export type AudioApiOs = "unix" | "windows";
/** One model per example: the tabs, with Workflows split by the run it shows. */
export type AudioApiExample =
  | Exclude<AudioApiTab, "workflows">
  | AudioRunWorkflow;

export const AUDIO_API_TABS: readonly AudioApiTab[] = [
  "speak",
  "clone",
  "transcribe",
  "workflows",
];
export const AUDIO_RUN_WORKFLOWS: readonly AudioRunWorkflow[] = [
  "separate",
  "convert",
  "music",
  "edit",
];

const KEY_PLACEHOLDER = "sk-unsloth-YOUR_KEY";
// The OpenAI SDKs require some api_key; the keyless dummy the chat examples use.
const KEYLESS_KEY_PLACEHOLDER = "not-needed";

/** The key the examples print: the revealed one, else whichever placeholder the server admits. */
export function audioApiKey(
  apiKey: string | null | undefined,
  server: {
    base: string;
    tunnel: boolean;
    scope: KeylessApiAccessScope;
    exposure: KeylessApiAccessExposure | null;
  },
): string {
  if (apiKey) return apiKey;
  // Keyless access admits the audio routes only at its "everything else" scope.
  const keyless =
    !server.tunnel &&
    server.scope === "full" &&
    keylessBaseEligible(server.base, server.scope, server.exposure);
  return keyless ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER;
}

// The chat examples store a variant ("pythonTools"); its language carries over.
export function langFromStored(stored: string | null): AudioApiLang {
  if (stored?.startsWith("python")) return "python";
  if (stored?.startsWith("javascript")) return "javascript";
  return "curl";
}

/** What to store when a language is picked here: nothing when the chat card's variant already has it. */
export function audioLangToStore(
  stored: string | null,
  lang: AudioApiLang,
): string | null {
  return stored && langFromStored(stored) === lang ? null : lang;
}

/** Where an Audio page's "Use via API" lands. */
export function audioApiExampleFor(workflow: AudioCppWorkflow): {
  tab: AudioApiTab;
  run: AudioRunWorkflow | null;
} {
  return workflow === "speak" ||
    workflow === "clone" ||
    workflow === "transcribe"
    ? { tab: workflow, run: null }
    : { tab: "workflows", run: workflow };
}

const folder = (name: string) => `${AUDIO_CPP_REPO}/${name}`;

/** Shown when nothing downloaded fits, so the example still has the right shape. */
export const AUDIO_API_PLACEHOLDER_MODELS: Record<AudioApiExample, string> = {
  speak: folder("Kokoro-82M-GGUF"),
  clone: folder("VoxCPM2-GGUF"),
  transcribe: folder("Parakeet-TDT-0.6B-v3-GGUF"),
  separate: folder("HTDemucs-GGUF"),
  convert: folder("Chatterbox-GGUF"),
  music: folder("ACE-Step1.5-GGUF"),
  edit: folder("DotTTS-Edit-GGUF"),
};

/**
 * Whether an Audio page's model can run the example. The page can hand over whatever is
 * resident, a chat model included, so once /v1/models answers it must list the model; until
 * then only a catalog model that runs the workflow counts. Separation models are not listed.
 */
export function audioApiModelFits(
  models: readonly AudioApiModel[] | null,
  id: string,
  example: AudioApiExample,
): boolean {
  if (models === null || example === "separate") {
    const known = audioCppModelFor(id);
    return !!known && audioCppWorkflowsFor(known).includes(example);
  }
  const listed = models.find((model) => model.id === id);
  return !!listed && canRun(listed, example);
}

// /v1/models gives a task, not workflows, so the audio.cpp catalog says what each model does.
// Models it does not know fall back to the family hints the Audio pages use.
function canRun(model: AudioApiModel, example: AudioApiExample): boolean {
  if (example === "transcribe") {
    return model.task === "automatic-speech-recognition";
  }
  // Separation models are not listed in /v1/models yet.
  if (model.task !== "text-to-speech" || example === "separate") return false;
  const known = audioCppModelFor(model.id);
  if (known) return audioCppWorkflowsFor(known).includes(example);
  switch (example) {
    case "speak":
      return !isCloneOnlyFamilyId(model.id) && !isConvertOnlyFamilyId(model.id);
    case "clone":
      return isCloneOnlyFamilyId(model.id) || isSpeakAndCloneFamilyId(model.id);
    case "convert":
      return (
        isCloneAndConvertFamilyId(model.id) || isConvertOnlyFamilyId(model.id)
      );
    default:
      // Every edit and music model is in the catalog.
      return false;
  }
}

const CATALOG_ORDER = new Map(
  AUDIO_CPP_MODELS.map((model, index) => [model.id.toLowerCase(), index]),
);

/**
 * The model an example names, or null when none fits. A loaded one first, then the Audio
 * pages' own order (their default picks lead it), then repos the catalog does not know.
 */
export function pickAudioApiModel(
  models: readonly AudioApiModel[],
  example: AudioApiExample,
): string | null {
  const rank = (model: AudioApiModel) => [
    model.loaded ? 0 : 1,
    // The Transcribe example shows timestamps when a downloaded model returns them.
    example === "transcribe" && !transcribeWithTimestamps(model.id) ? 1 : 0,
    CATALOG_ORDER.get(model.id.toLowerCase()) ?? AUDIO_CPP_MODELS.length,
  ];
  const fits = models
    .filter((model) => canRun(model, example))
    .map((model) => ({ id: model.id, rank: rank(model) }));
  fits.sort(
    (a, b) =>
      a.rank[0] - b.rank[0] || a.rank[1] - b.rank[1] || a.rank[2] - b.rank[2],
  );
  return fits[0]?.id ?? null;
}

// Mirrors ALWAYS_TIMESTAMPED in studio/backend/core/inference/stt_details.py. Qwen3-ASR times
// its words only once its aligner is downloaded, and other engines answer timestamps with a 400.
const TIMESTAMPED_FAMILY_HINT =
  /parakeet[-_]?tdt|moss[-_]?transcribe[-_]?diarize|vibevoice[-_]?asr|kroko/i;

export function transcribeWithTimestamps(model: string): boolean {
  return isAudioCppFolderId(model) && TIMESTAMPED_FAMILY_HINT.test(model);
}

export interface AudioApiSnippetInput {
  base: string;
  apiKey: string;
  model: string;
  lang: AudioApiLang;
  os: AudioApiOs;
}

const SPEAK_TEXT = "Hello from Unsloth.";
const CLONE_TEXT = "This is my voice, speaking text I never recorded.";
const CLONE_TRANSCRIPT = "The words spoken in me.wav.";
const EDIT_FROM = "The quick brown fox jumps over the lazy dog.";
const EDIT_TO = "The quick red fox jumps over the lazy dog.";

// An uploaded input's id, written in each language's own way.
type Ref = { upload: string };
type Json = string | number | boolean | Ref | Json[] | { [key: string]: Json };

function isRef(value: Json): value is Ref {
  return typeof value === "object" && value !== null && "upload" in value;
}

type Style = "json" | "python" | "jq" | "powershell";

function refExpr(name: string, style: Style): string {
  switch (style) {
    case "python":
      return `${name}["id"]`;
    case "jq":
      return `$${name}`;
    case "powershell":
      return `"$($${name}.id)"`;
    default:
      return `${name}.id`;
  }
}

function scalar(value: string | number | boolean, style: Style): string {
  if (typeof value === "boolean" && style === "python") {
    return value ? "True" : "False";
  }
  const text = JSON.stringify(value);
  // An expandable here-string reads ` and $ as escapes.
  return style === "powershell" ? text.replace(/[`$]/g, "`$&") : text;
}

/** Pretty JSON that is also a Python dict, a JS object and a jq program. */
function render(value: Json, style: Style, indent = ""): string {
  if (isRef(value)) return refExpr(value.upload, style);
  if (typeof value !== "object") return scalar(value, style);
  // Python's own four spaces; two elsewhere, like the chat examples.
  const inner = `${indent}${style === "python" ? "    " : "  "}`;
  const entries = Array.isArray(value)
    ? value.map((item) => `${inner}${render(item, style, inner)}`)
    : Object.entries(value).map(
        ([key, item]) =>
          `${inner}${JSON.stringify(key)}: ${render(item, style, inner)}`,
      );
  const [open, close] = Array.isArray(value) ? ["[", "]"] : ["{", "}"];
  return entries.length
    ? `${open}\n${entries.join(",\n")}\n${indent}${close}`
    : `${open}${close}`;
}

const j = (value: string) => JSON.stringify(value);

// Mirrors the rvc family in audio_cpp_models.py: it converts only to its built-in voices
// and refuses a target recording.
const BUILTIN_VOICE_CONVERTER = /(^|[-_ /.])rvc([-_ /.]|$)/i;
// Mirrors the music specs there: Stable Audio's SFX build and ControlFoley make no songs.
const SFX_ONLY_MUSIC = /(^|[-_ /.])sfx([-_ /.]|$)|controlfoley/i;

function editFamily(model: string): string {
  if (/firered/i.test(model)) return "firered_audio";
  if (/vevo[-_]?2/i.test(model)) return "vevo2";
  return "dots_tts";
}

interface RunPlan {
  uploads: { name: string; file: string }[];
  body: { [key: string]: Json };
}

function runPlan(workflow: AudioRunWorkflow, model: string): RunPlan {
  switch (workflow) {
    case "separate":
      return {
        uploads: [{ name: "source", file: "song.wav" }],
        body: {
          workflow,
          model,
          inputs: { source: { input_id: { upload: "source" } } },
        },
      };
    case "convert":
      if (BUILTIN_VOICE_CONVERTER.test(model)) {
        return {
          uploads: [{ name: "source", file: "speech.wav" }],
          body: {
            workflow,
            model,
            inputs: { source: { input_id: { upload: "source" } } },
            convert: { voice: "default" },
          },
        };
      }
      return {
        uploads: [
          { name: "source", file: "speech.wav" },
          { name: "target", file: "voice.wav" },
        ],
        body: {
          workflow,
          model,
          inputs: {
            source: { input_id: { upload: "source" } },
            target: { input_id: { upload: "target" } },
          },
        },
      };
    case "music":
      if (SFX_ONLY_MUSIC.test(model)) {
        return {
          uploads: [],
          body: {
            workflow,
            model,
            mode: "sfx",
            text: "Rain on a tin roof with distant thunder",
          },
        };
      }
      return {
        uploads: [],
        body: {
          workflow,
          model,
          mode: "song",
          text: "Upbeat synth pop with a catchy chorus",
          lyrics:
            "[verse]\nCity lights are calling me\n[chorus]\nWe dance until the morning",
        },
      };
    default: {
      // The same edit part the Edit page sends for this family.
      const edit = Object.fromEntries(
        Object.entries(
          EDIT_ADAPTERS[editFamily(model)].buildEdit({
            transcript: EDIT_FROM,
            edited: EDIT_TO,
            mode: "words",
          }),
        ),
      );
      return {
        uploads: [{ name: "source", file: "speech.wav" }],
        body: {
          workflow,
          model,
          text: EDIT_TO,
          inputs: {
            source: { input_id: { upload: "source" } },
            reference_text: EDIT_FROM,
          },
          edit,
        },
      };
    }
  }
}

function bashPreamble(base: string, apiKey: string): string {
  return `BASE='${shSingle(base)}'\nKEY='${shSingle(apiKey)}'\n`;
}

function powershellPreamble(base: string, apiKey: string): string {
  return `$base = '${psSingle(base)}'\n$key = '${psSingle(apiKey)}'\n`;
}

function bashUpload(name: string, file: string): string {
  return `${name.toUpperCase()}=$(curl -s "$BASE/v1/audio/inputs?name=${file}" \\
  -H "Authorization: Bearer $KEY" \\
  --data-binary @${file} | jq -r .id)`;
}

function powershellUpload(name: string, file: string): string {
  return `$${name} = curl.exe -s "$base/v1/audio/inputs?name=${file}" \`
  -H "Authorization: Bearer $key" \`
  --data-binary "@${file}" | ConvertFrom-Json`;
}

/** A JSON body for curl: a plain literal, or built by jq when it carries upload ids. */
function bashBody(body: { [key: string]: Json }, refs: string[]): string {
  if (refs.length === 0) return `'${shSingle(render(body, "json", "  "))}'`;
  const args = refs
    .map((ref) => `--arg ${ref} "$${ref.toUpperCase()}"`)
    .join(" ");
  return `"$(jq -n ${args} '${shSingle(render(body, "jq", "  "))}')"`;
}

function powershellBody(body: { [key: string]: Json }): string {
  return `$body = @"\n${render(body, "powershell")}\n"@
Set-Content -Path body.json -Value $body -Encoding ascii`;
}

interface ClientNeeds {
  /** The OpenAI SDK client, for speech and transcriptions. */
  sdk: boolean;
  /** Studio routes the SDK has no method for: uploads, saved voices, workflow runs. */
  studio: boolean;
  upload: boolean;
  /** A workflow run answers only when it finishes, which can take longer than 5 minutes. */
  longRuns: boolean;
}

const SDK_ONLY: ClientNeeds = {
  sdk: true,
  studio: false,
  upload: false,
  longRuns: false,
};
const SDK_AND_UPLOADS: ClientNeeds = {
  sdk: true,
  studio: true,
  upload: true,
  longRuns: false,
};

function pythonClient(
  base: string,
  apiKey: string,
  { sdk, studio, upload }: ClientNeeds,
): string {
  const imports = [
    ...(studio ? ["import httpx"] : []),
    ...(sdk ? ["from openai import OpenAI"] : []),
  ];
  let out = `${imports.join("\n")}

BASE = ${j(base)}
KEY = ${j(apiKey)}
`;
  if (sdk) {
    out += `
client = OpenAI(base_url=f"{BASE}/v1", api_key=KEY)`;
  }
  if (studio) {
    out += `${sdk ? "\n" : ""}
# Uploads, saved voices and workflow runs are Studio routes the OpenAI SDK has no method for.
studio = httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {KEY}"}, timeout=None)


def post(path, **kwargs):
    response = studio.post(path, **kwargs)
    response.raise_for_status()
    return response.json()`;
  }
  if (upload) {
    out += `


def upload(file):
    with open(file, "rb") as audio:
        return post("/v1/audio/inputs", params={"name": file}, content=audio.read())`;
  }
  return out;
}

function javascriptClient(
  base: string,
  apiKey: string,
  { sdk, studio, upload, longRuns }: ClientNeeds,
): string {
  let out = `import fs from "node:fs";
${sdk ? 'import OpenAI from "openai";\n' : ""}${longRuns ? 'import { Agent, setGlobalDispatcher } from "undici";\n' : ""}
const BASE = ${j(base)};
const KEY = ${j(apiKey)};`;
  if (longRuns) {
    out += `

// fetch stops waiting for a response after 5 minutes; a long run takes longer than that.
setGlobalDispatcher(new Agent({ headersTimeout: 0, bodyTimeout: 0 }));`;
  }
  if (sdk) {
    out += `

const client = new OpenAI({ baseURL: BASE + "/v1", apiKey: KEY });`;
  }
  if (studio) {
    out += `

// Uploads, saved voices and workflow runs are Studio routes the OpenAI SDK has no method for.
async function studio(path, init = {}) {
  const response = await fetch(BASE + path, {
    ...init,
    headers: { Authorization: "Bearer " + KEY, ...init.headers },
  });
  if (!response.ok) throw new Error(await response.text());
  return response;
}`;
  }
  if (upload) {
    out += `

async function upload(file) {
  const response = await studio("/v1/audio/inputs?name=" + file, {
    method: "POST",
    body: fs.readFileSync(file),
  });
  return response.json();
}`;
  }
  return out;
}

function speakSnippet({
  base,
  apiKey,
  model,
  lang,
  os,
}: AudioApiSnippetInput): string {
  const body = {
    model,
    input: SPEAK_TEXT,
    voice: "alloy",
    response_format: "mp3",
  };
  const voiceNote =
    "voice: a saved voice id or one of the model's own speakers. Other names, like alloy, keep its default voice.";
  if (lang === "python") {
    return `${pythonClient(base, apiKey, SDK_ONLY)}

# ${voiceNote}
with client.audio.speech.with_streaming_response.create(
    model=${j(model)},
    voice="alloy",
    input=${j(SPEAK_TEXT)},
    response_format="mp3",
) as speech:
    speech.stream_to_file("speech.mp3")`;
  }
  if (lang === "javascript") {
    return `${javascriptClient(base, apiKey, SDK_ONLY)}

// ${voiceNote}
const speech = await client.audio.speech.create({
  model: ${j(model)},
  voice: "alloy",
  input: ${j(SPEAK_TEXT)},
  response_format: "mp3",
});
fs.writeFileSync("speech.mp3", Buffer.from(await speech.arrayBuffer()));`;
  }
  if (os === "windows") {
    return `${powershellPreamble(base, apiKey)}# ${voiceNote}
${powershellBody(body)}
curl.exe "$base/v1/audio/speech" \`
  -H "Authorization: Bearer $key" \`
  -H "Content-Type: application/json" \`
  -d "@body.json" -o speech.mp3`;
  }
  return `${bashPreamble(base, apiKey)}# ${voiceNote}
curl "$BASE/v1/audio/speech" \\
  -H "Authorization: Bearer $KEY" \\
  -H "Content-Type: application/json" \\
  -d ${bashBody(body, [])} \\
  -o speech.mp3`;
}

function cloneSnippet({
  base,
  apiKey,
  model,
  lang,
  os,
}: AudioApiSnippetInput): string {
  const voice: { [key: string]: Json } = {
    name: "Me",
    source: { input_id: { upload: "recording" } },
    transcript: CLONE_TRANSCRIPT,
  };
  const speech: { [key: string]: Json } = {
    model,
    input: CLONE_TEXT,
    voice: { upload: "voice" },
    response_format: "mp3",
  };
  const steps =
    "Upload a short recording of the voice, save it as a voice, then speak in it.";
  const transcriptNote =
    "transcript: what me.wav says. Some models need it to clone.";
  if (lang === "python") {
    return `${pythonClient(base, apiKey, SDK_AND_UPLOADS)}


# ${steps}
recording = upload("me.wav")
# ${transcriptNote}
voice = post("/v1/audio/voices", json=${render(voice, "python")})

with client.audio.speech.with_streaming_response.create(
    model=${j(model)},
    voice=voice["id"],
    input=${j(CLONE_TEXT)},
    response_format="mp3",
) as speech:
    speech.stream_to_file("clone.mp3")`;
  }
  if (lang === "javascript") {
    return `${javascriptClient(base, apiKey, SDK_AND_UPLOADS)}

// ${steps}
const recording = await upload("me.wav");
// ${transcriptNote}
const saved = await studio("/v1/audio/voices", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(${render(voice, "json").replace(/\n/g, "\n  ")}),
});
const voice = await saved.json();

const speech = await client.audio.speech.create({
  model: ${j(model)},
  voice: voice.id,
  input: ${j(CLONE_TEXT)},
  response_format: "mp3",
});
fs.writeFileSync("clone.mp3", Buffer.from(await speech.arrayBuffer()));`;
  }
  if (os === "windows") {
    return `${powershellPreamble(base, apiKey)}# ${steps}
${powershellUpload("recording", "me.wav")}
# ${transcriptNote}
${powershellBody(voice)}
$voice = curl.exe -s "$base/v1/audio/voices" \`
  -H "Authorization: Bearer $key" \`
  -H "Content-Type: application/json" \`
  -d "@body.json" | ConvertFrom-Json
${powershellBody(speech)}
curl.exe "$base/v1/audio/speech" \`
  -H "Authorization: Bearer $key" \`
  -H "Content-Type: application/json" \`
  -d "@body.json" -o clone.mp3`;
  }
  return `${bashPreamble(base, apiKey)}# ${steps}
${bashUpload("recording", "me.wav")}
# ${transcriptNote}
VOICE=$(curl -s "$BASE/v1/audio/voices" \\
  -H "Authorization: Bearer $KEY" \\
  -H "Content-Type: application/json" \\
  -d ${bashBody(voice, ["recording"])} | jq -r .id)
curl "$BASE/v1/audio/speech" \\
  -H "Authorization: Bearer $KEY" \\
  -H "Content-Type: application/json" \\
  -d ${bashBody(speech, ["voice"])} \\
  -o clone.mp3`;
}

function transcribeSnippet({
  base,
  apiKey,
  model,
  lang,
  os,
}: AudioApiSnippetInput): string {
  const timed = transcribeWithTimestamps(model);
  if (lang === "python") {
    const options = timed
      ? `
    response_format="verbose_json",
    timestamp_granularities=["segment"],
    language="en",`
      : "";
    const output = timed
      ? `for segment in transcript.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")`
      : "print(transcript.text)";
    return `${pythonClient(base, apiKey, SDK_ONLY)}

with open("speech.wav", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model=${j(model)},
        file=audio,${options.replace(/\n {4}/g, "\n        ")}
    )
${output}`;
  }
  if (lang === "javascript") {
    const options = timed
      ? `
  response_format: "verbose_json",
  timestamp_granularities: ["segment"],
  language: "en",`
      : "";
    const output = timed
      ? `for (const segment of transcript.segments ?? []) {
  console.log(\`[\${segment.start.toFixed(1)}s - \${segment.end.toFixed(1)}s] \${segment.text}\`);
}`
      : "console.log(transcript.text);";
    return `${javascriptClient(base, apiKey, SDK_ONLY)}

const transcript = await client.audio.transcriptions.create({
  model: ${j(model)},
  file: fs.createReadStream("speech.wav"),${options}
});
${output}`;
  }
  const fields = [
    "file=@speech.wav",
    `model=${model}`,
    ...(timed
      ? [
          "response_format=verbose_json",
          "timestamp_granularities[]=segment",
          "language=en",
        ]
      : []),
  ];
  if (os === "windows") {
    return `${powershellPreamble(base, apiKey)}curl.exe "$base/v1/audio/transcriptions" \`
  -H "Authorization: Bearer $key" \`
${fields.map((field) => `  -F '${psSingle(field)}'`).join(" `\n")}`;
  }
  return `${bashPreamble(base, apiKey)}curl "$BASE/v1/audio/transcriptions" \\
  -H "Authorization: Bearer $KEY" \\
${fields.map((field) => `  -F '${shSingle(field)}'`).join(" \\\n")}`;
}

function runSnippet(
  workflow: AudioRunWorkflow,
  { base, apiKey, model, lang, os }: AudioApiSnippetInput,
): string {
  const { uploads, body } = runPlan(workflow, model);
  const refs = uploads.map((item) => item.name);
  const needs = {
    sdk: false,
    studio: true,
    upload: uploads.length > 0,
    longRuns: true,
  };
  const save =
    workflow === "separate"
      ? "Saves each stem as <clip id>.wav."
      : "Saves each clip the run made as <clip id>.wav.";
  if (lang === "python") {
    return `${pythonClient(base, apiKey, needs)}


${uploads.map((item) => `${item.name} = upload(${j(item.file)})`).join("\n")}${uploads.length ? "\n" : ""}result = post("/v1/audio/run", json=${render(body, "python")})

# ${save}
for clip in result["clips"]:
    audio = studio.get(clip["url"])
    audio.raise_for_status()
    with open(f"{clip['id']}.wav", "wb") as out:
        out.write(audio.content)`;
  }
  if (lang === "javascript") {
    return `${javascriptClient(base, apiKey, needs)}

${uploads.map((item) => `const ${item.name} = await upload(${j(item.file)});`).join("\n")}${uploads.length ? "\n" : ""}const run = await studio("/v1/audio/run", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(${render(body, "json").replace(/\n/g, "\n  ")}),
});
const result = await run.json();

// ${save}
for (const clip of result.clips) {
  const audio = await studio(clip.url);
  fs.writeFileSync(clip.id + ".wav", Buffer.from(await audio.arrayBuffer()));
}`;
  }
  if (os === "windows") {
    return `${powershellPreamble(base, apiKey)}${uploads.map((item) => `${powershellUpload(item.name, item.file)}\n`).join("")}${powershellBody(body)}
$result = curl.exe -s "$base/v1/audio/run" \`
  -H "Authorization: Bearer $key" \`
  -H "Content-Type: application/json" \`
  -d "@body.json" | ConvertFrom-Json
# ${save}
foreach ($clip in $result.clips) {
  curl.exe -s "$base$($clip.url)" -H "Authorization: Bearer $key" -o "$($clip.id).wav"
}`;
  }
  return `${bashPreamble(base, apiKey)}${uploads.map((item) => `${bashUpload(item.name, item.file)}\n`).join("")}# ${save}
curl -s "$BASE/v1/audio/run" \\
  -H "Authorization: Bearer $KEY" \\
  -H "Content-Type: application/json" \\
  -d ${bashBody(body, refs)} \\
  | jq -r '.clips[] | .id + " " + .url' \\
  | while read -r id url; do
      curl -s "$BASE$url" -H "Authorization: Bearer $KEY" -o "$id.wav"
    done`;
}

export function buildAudioApiSnippet(
  example: AudioApiExample,
  input: AudioApiSnippetInput,
): string {
  switch (example) {
    case "speak":
      return speakSnippet(input);
    case "clone":
      return cloneSnippet(input);
    case "transcribe":
      return transcribeSnippet(input);
    default:
      return runSnippet(example, input);
  }
}
