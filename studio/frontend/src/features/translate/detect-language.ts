// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A guess at a draft's language: a code from TRANSLATE_LANGUAGES, or null when unsure.

const SCRIPTS: readonly [string, RegExp][] = [
  ["kana", /[぀-ヿ]/g],
  ["hangul", /[가-힯ᄀ-ᇿ]/g],
  ["han", /[一-鿿㐀-䶿]/g],
  ["thai", /[฀-๿]/g],
  ["hebrew", /[֐-׿]/g],
  ["khmer", /[ក-៿]/g],
  ["myanmar", /[က-႟]/g],
  ["tibetan", /[ༀ-࿿]/g],
  ["tamil", /[஀-௿]/g],
  ["telugu", /[ఀ-౿]/g],
  ["gujarati", /[઀-૿]/g],
  ["bengali", /[ঀ-৿]/g],
  ["devanagari", /[ऀ-ॿ]/g],
  ["arabic", /[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]/g],
  ["cyrillic", /[Ѐ-ӿ]/g],
  ["latin", /[a-zÀ-ɏḀ-ỿ]/gi],
];

const ONE_LANGUAGE: Record<string, string> = {
  kana: "ja",
  hangul: "ko",
  thai: "th",
  hebrew: "he",
  khmer: "km",
  myanmar: "my",
  tibetan: "bo",
  tamil: "ta",
  telugu: "te",
  gujarati: "gu",
  bengali: "bn",
};

const count = (text: string, pattern: RegExp) =>
  (text.match(pattern) ?? []).length;

const SIMPLIFIED = /[们这说对时会过还没为国学发经东车马门见问间关]/g;
const TRADITIONAL = /[們這說對時會過還沒為國學發經東車馬門見問間關]/g;
const CANTONESE = /[嘅咗喺哋冇嘢佢啲嚟噉咁乜唔]/g;

function han(text: string): string {
  if (count(text, CANTONESE) >= 2) return "yue";
  return count(text, TRADITIONAL) > count(text, SIMPLIFIED) ? "zh-Hant" : "zh";
}

function devanagari(text: string): string {
  const marathi = count(text, /आहे|आणि|नाही|ळ|होते|आहेत|मी |तुम्ही/g);
  const hindi = count(text, /है|और|नहीं|हैं|था|मैं|आप |क्या/g);
  return marathi > hindi ? "mr" : "hi";
}

function arabic(text: string): string {
  if (count(text, /[ەۇۆۈېڭ]/g) >= 2) return "ug";
  const urdu = count(text, /[ںےٹڈڑھ]|ہے|کے|میں|اور/g);
  const persian = count(text, /[پچژگ]|است|که |این |را |می‌/g);
  if (urdu >= 2 && urdu >= persian) return "ur";
  if (persian >= 2) return "fa";
  return "ar";
}

function cyrillic(text: string): string {
  const lower = text.toLowerCase();
  if (count(lower, /[әғқңұһ]/g) >= 2) return "kk";
  if (count(lower, /[їєґ]|і/g) >= 2) return "uk";
  if (count(lower, /[өү]|байна|нь |юм |гэж|бол /g) >= 2) return "mn";
  return "ru";
}

const LATIN: readonly [string, RegExp, readonly string[]][] = [
  [
    "vi",
    /[ơưăđạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]/g,
    [
      "và",
      "của",
      "là",
      "không",
      "có",
      "một",
      "những",
      "được",
      "trong",
      "tôi",
      "bạn",
      "này",
    ],
  ],
  [
    "tr",
    /[ığş]/g,
    [
      "ve",
      "bir",
      "bu",
      "için",
      "çok",
      "değil",
      "ama",
      "ben",
      "sen",
      "ile",
      "gibi",
      "var",
    ],
  ],
  [
    "pl",
    /[łąęńśźż]/g,
    [
      "nie",
      "się",
      "jest",
      "to",
      "na",
      "że",
      "jak",
      "ale",
      "czy",
      "dla",
      "był",
      "tak",
    ],
  ],
  [
    "cs",
    /[řěůťďň]/g,
    [
      "je",
      "se",
      "na",
      "že",
      "to",
      "jsem",
      "ale",
      "jako",
      "pro",
      "tak",
      "by",
      "jsou",
    ],
  ],
  [
    "de",
    /[ßäöü]/g,
    [
      "der",
      "die",
      "und",
      "das",
      "ist",
      "nicht",
      "ich",
      "ein",
      "eine",
      "mit",
      "sie",
      "auf",
      "für",
      "zu",
      "den",
    ],
  ],
  [
    "pt",
    /[ãõ]/g,
    [
      "não",
      "que",
      "uma",
      "para",
      "com",
      "você",
      "os",
      "do",
      "da",
      "em",
      "é",
      "mas",
      "isso",
      "muito",
      "um",
    ],
  ],
  [
    "es",
    /[ñ¿¡]/g,
    [
      "que",
      "el",
      "los",
      "las",
      "una",
      "para",
      "con",
      "por",
      "es",
      "pero",
      "muy",
      "está",
      "como",
      "más",
      "del",
      "y",
    ],
  ],
  [
    "fr",
    /[èêœùû]/g,
    [
      "le",
      "les",
      "des",
      "est",
      "une",
      "et",
      "pas",
      "pour",
      "que",
      "dans",
      "je",
      "vous",
      "nous",
      "avec",
      "sur",
      "ce",
    ],
  ],
  [
    "it",
    /[ìò]/g,
    [
      "il",
      "che",
      "non",
      "di",
      "per",
      "una",
      "sono",
      "con",
      "della",
      "gli",
      "più",
      "anche",
      "è",
      "ma",
      "mi",
    ],
  ],
  [
    "nl",
    /ĳ/g,
    [
      "de",
      "het",
      "een",
      "en",
      "van",
      "niet",
      "dat",
      "ik",
      "is",
      "op",
      "voor",
      "met",
      "zijn",
      "maar",
      "je",
    ],
  ],
  [
    "id",
    /$^/g,
    [
      "yang",
      "dan",
      "tidak",
      "dengan",
      "untuk",
      "ini",
      "itu",
      "adalah",
      "saya",
      "akan",
      "dari",
      "bisa",
      "sudah",
      "karena",
    ],
  ],
  [
    "ms",
    /$^/g,
    [
      "yang",
      "dan",
      "tidak",
      "dengan",
      "untuk",
      "ini",
      "itu",
      "ialah",
      "saya",
      "akan",
      "dari",
      "boleh",
      "awak",
      "kerana",
    ],
  ],
  [
    "tl",
    /$^/g,
    [
      "ang",
      "ng",
      "sa",
      "mga",
      "ako",
      "ka",
      "hindi",
      "ito",
      "ay",
      "na",
      "po",
      "ko",
      "siya",
      "kung",
    ],
  ],
  [
    "en",
    /$^/g,
    [
      "the",
      "and",
      "is",
      "are",
      "to",
      "of",
      "in",
      "that",
      "it",
      "you",
      "for",
      "with",
      "was",
      "this",
      "not",
      "have",
    ],
  ],
];

function latin(text: string): string | null {
  const words = text.toLowerCase().match(/[\p{L}'’]+/gu) ?? [];
  if (words.length === 0) return null;
  let best: string | null = null;
  let top = 0;
  let second = 0;
  for (const [code, marks, common] of LATIN) {
    const known = new Set(common);
    const score =
      words.filter((word) => known.has(word)).length +
      Math.min(3, count(text.toLowerCase(), marks)) * 1.5;
    if (score > top) {
      second = top;
      top = score;
      best = code;
    } else if (score > second) {
      second = score;
    }
  }
  return top >= 2 && top > second ? best : null;
}

const SAMPLE_CHARS = 2000;

export function detectLanguage(source: string): string | null {
  const text = source.slice(0, SAMPLE_CHARS);
  let script: string | null = null;
  let most = 0;
  let letters = 0;
  for (const [name, pattern] of SCRIPTS) {
    const found = count(text, pattern);
    letters += found;
    const weight = name === "kana" && found > 0 ? found + 1000 : found;
    if (weight > most) {
      most = weight;
      script = name;
    }
  }
  if (!script || letters < 2) return null;
  if (script in ONE_LANGUAGE) return ONE_LANGUAGE[script];
  if (script === "han") return han(text);
  if (script === "devanagari") return devanagari(text);
  if (script === "arabic") return arabic(text);
  if (script === "cyrillic") return letters >= 4 ? cyrillic(text) : null;
  return latin(text);
}
