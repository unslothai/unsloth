// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const S = await import(
  "../src/features/settings/components/audio-api-snippets.ts"
);
type Model = Parameters<typeof S.pickAudioApiModel>[0][number];

const BASE = "http://127.0.0.1:8888";
const KEY = "sk-unsloth-abc";
const tts = (id: string, loaded = false): Model => ({
  id,
  loaded,
  task: "text-to-speech",
});
const asr = (id: string, loaded = false): Model => ({
  id,
  loaded,
  task: "automatic-speech-recognition",
});
const cpp = (name: string) => `audio-cpp/audio.cpp-gguf/${name}`;

const PATHS: Record<string, string[]> = {
  speak: ["/v1/audio/speech"],
  clone: ["/v1/audio/inputs", "/v1/audio/voices", "/v1/audio/speech"],
  transcribe: ["/v1/audio/transcriptions"],
  separate: ["/v1/audio/inputs", "/v1/audio/run"],
  convert: ["/v1/audio/inputs", "/v1/audio/run"],
  music: ["/v1/audio/run"],
  edit: ["/v1/audio/inputs", "/v1/audio/run"],
};

const EXAMPLES = Object.keys(
  PATHS,
) as (keyof typeof S.AUDIO_API_PLACEHOLDER_MODELS)[];
const VARIANTS = [
  { lang: "curl", os: "unix" },
  { lang: "curl", os: "windows" },
  { lang: "python", os: "unix" },
  { lang: "javascript", os: "unix" },
] as const;

test("every example names its routes, the server, the key and the model", () => {
  for (const example of EXAMPLES) {
    for (const { lang, os } of VARIANTS) {
      const model = S.AUDIO_API_PLACEHOLDER_MODELS[example];
      const code = S.buildAudioApiSnippet(example, {
        base: BASE,
        apiKey: KEY,
        model,
        lang,
        os,
      });
      const where = `${example} ${lang} ${os}`;
      for (const path of PATHS[example]) {
        // The SDKs call speech and transcriptions by method, not path.
        const call =
          lang === "curl"
            ? path
            : path.replace(
                /^\/v1\/audio\/(speech|transcriptions)$/,
                "client.audio.$1",
              );
        assert.ok(code.includes(call), `${where}: ${call}`);
      }
      assert.ok(code.includes(BASE), `${where}: base`);
      assert.ok(code.includes(KEY), `${where}: key`);
      assert.ok(code.includes(model), `${where}: model`);
    }
  }
});

test("Windows curl runs curl.exe with backtick continuations, bash with backslashes", () => {
  const input = { base: BASE, apiKey: KEY, model: cpp("HTDemucs-GGUF") };
  const win = S.buildAudioApiSnippet("separate", {
    ...input,
    lang: "curl",
    os: "windows",
  });
  assert.match(win, /curl\.exe /);
  assert.match(win, / `\n/);
  assert.ok(!/ \\\n/.test(win));
  assert.match(win, /"input_id": "\$\(\$source\.id\)"/);
  const unix = S.buildAudioApiSnippet("separate", {
    ...input,
    lang: "curl",
    os: "unix",
  });
  assert.match(unix, / \\\n/);
  assert.ok(!unix.includes("curl.exe"));
  assert.match(unix, /jq -n --arg source "\$SOURCE"/);
});

test("model ids and text are escaped for each language", () => {
  const model = `odd"name'$x`;
  const input = { base: BASE, apiKey: KEY, model };
  const python = S.buildAudioApiSnippet("speak", {
    ...input,
    lang: "python",
    os: "unix",
  });
  assert.ok(python.includes(`model=${JSON.stringify(model)}`));
  const js = S.buildAudioApiSnippet("speak", {
    ...input,
    lang: "javascript",
    os: "unix",
  });
  assert.ok(js.includes(`model: ${JSON.stringify(model)}`));
  // bash: the whole body is one single-quoted word.
  const bash = S.buildAudioApiSnippet("speak", {
    ...input,
    lang: "curl",
    os: "unix",
  });
  const quoted = bash.slice(
    bash.indexOf("-d '") + 4,
    bash.indexOf("' \\\n  -o"),
  );
  assert.equal(JSON.parse(quoted.replace(/'\\''/g, "'")).model, model);
  // PowerShell: an expandable here-string, so $ and ` are escaped.
  const ps = S.buildAudioApiSnippet("speak", {
    ...input,
    lang: "curl",
    os: "windows",
  });
  assert.ok(ps.includes('"model": "odd\\"name\'`$x"'));
});

test("the Edit body follows the model's family", () => {
  const input = {
    base: BASE,
    apiKey: KEY,
    lang: "python",
    os: "unix",
  } as const;
  const dots = S.buildAudioApiSnippet("edit", {
    ...input,
    model: cpp("DotTTS-Edit-GGUF"),
  });
  assert.ok(dots.includes('<sub targ=\\"red\\">brown</sub>'));
  const firered = S.buildAudioApiSnippet("edit", {
    ...input,
    model: cpp("FireRedAudio-GGUF"),
  });
  assert.ok(firered.includes("Replace 'brown' with 'red'."));
  const vevo = S.buildAudioApiSnippet("edit", {
    ...input,
    model: cpp("Vevo2-GGUF"),
  });
  assert.ok(!vevo.includes("markup") && !vevo.includes("instructions"));
});

test("timestamps are asked for only where the model returns them", () => {
  assert.ok(S.transcribeWithTimestamps(cpp("Parakeet-TDT-0.6B-v3-GGUF")));
  assert.ok(S.transcribeWithTimestamps(cpp("MOSS-Transcribe-Diarize-GGUF")));
  // Needs its aligner first.
  assert.ok(!S.transcribeWithTimestamps(cpp("Qwen3-ASR-0.6B-GGUF")));
  // Other engines answer timestamp_granularities with a 400.
  assert.ok(!S.transcribeWithTimestamps("openai/whisper-large-v3-turbo"));
  const input = { base: BASE, apiKey: KEY, lang: "curl", os: "unix" } as const;
  const timed = S.buildAudioApiSnippet("transcribe", {
    ...input,
    model: cpp("Parakeet-TDT-0.6B-v3-GGUF"),
  });
  assert.ok(timed.includes("timestamp_granularities[]=segment"));
  const plain = S.buildAudioApiSnippet("transcribe", {
    ...input,
    model: "openai/whisper-large-v3-turbo",
  });
  assert.ok(
    !plain.includes("timestamp_granularities") &&
      !plain.includes("verbose_json"),
  );
});

test("each example picks a downloaded model that can run it, loaded first", () => {
  const models = [
    tts(cpp("ACE-Step1.5-GGUF")),
    tts(cpp("Qwen3-TTS-12Hz-0.6B-Base-GGUF")),
    tts(cpp("Kokoro-82M-GGUF")),
    tts(cpp("VoxCPM2-GGUF"), true),
    tts(cpp("RVC-GGUF")),
    tts(cpp("DotTTS-Edit-GGUF")),
    asr("openai/whisper-small"),
    asr(cpp("Parakeet-TDT-0.6B-v3-GGUF"), true),
  ];
  assert.equal(S.pickAudioApiModel(models, "speak"), cpp("VoxCPM2-GGUF"));
  assert.equal(S.pickAudioApiModel(models, "clone"), cpp("VoxCPM2-GGUF"));
  assert.equal(S.pickAudioApiModel(models, "convert"), cpp("RVC-GGUF"));
  assert.equal(S.pickAudioApiModel(models, "music"), cpp("ACE-Step1.5-GGUF"));
  assert.equal(S.pickAudioApiModel(models, "edit"), cpp("DotTTS-Edit-GGUF"));
  assert.equal(
    S.pickAudioApiModel(models, "transcribe"),
    cpp("Parakeet-TDT-0.6B-v3-GGUF"),
  );
  // Unknown folders come after the catalog's, whatever their name sorts as.
  assert.equal(
    S.pickAudioApiModel(
      [
        tts(cpp("Breeze-TTS-2-GGUF")),
        tts(cpp("Kokoro-82M-GGUF")),
        tts(cpp("Chatterbox-Turbo-GGUF")),
      ],
      "speak",
    ),
    cpp("Kokoro-82M-GGUF"),
  );
  assert.equal(
    S.pickAudioApiModel(
      [asr(cpp("Qwen3-ASR-0.6B-GGUF")), asr(cpp("Parakeet-TDT-0.6B-v3-GGUF"))],
      "transcribe",
    ),
    cpp("Parakeet-TDT-0.6B-v3-GGUF"),
  );
  // /v1/models lists no separation models.
  assert.equal(S.pickAudioApiModel(models, "separate"), null);
});

test("music and clone-only models never speak, and an unknown repo uses the family hints", () => {
  const models = [
    tts(cpp("ACE-Step1.5-GGUF")),
    tts(cpp("Qwen3-TTS-12Hz-0.6B-Base-GGUF")),
  ];
  assert.equal(S.pickAudioApiModel(models, "speak"), null);
  assert.equal(
    S.pickAudioApiModel([tts("someone/IndexTTS2-GGUF")], "clone"),
    "someone/IndexTTS2-GGUF",
  );
  assert.equal(
    S.pickAudioApiModel([tts("someone/IndexTTS2-GGUF")], "speak"),
    null,
  );
  assert.equal(
    S.pickAudioApiModel([tts("unsloth/orpheus-3b-0.1-ft")], "speak"),
    "unsloth/orpheus-3b-0.1-ft",
  );
  assert.equal(S.pickAudioApiModel([], "transcribe"), null);
});

test("each Audio page opens the example for its workflow", () => {
  assert.deepEqual(S.audioApiExampleFor("speak"), { tab: "speak", run: null });
  assert.deepEqual(S.audioApiExampleFor("transcribe"), {
    tab: "transcribe",
    run: null,
  });
  assert.deepEqual(S.audioApiExampleFor("separate"), {
    tab: "workflows",
    run: "separate",
  });
  assert.deepEqual(S.audioApiExampleFor("edit"), {
    tab: "workflows",
    run: "edit",
  });
});
