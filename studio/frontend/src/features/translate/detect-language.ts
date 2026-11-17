// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Names a draft's language only when its letters settle it: a code from TRANSLATE_LANGUAGES, or
// null. Languages that share the Latin alphabet are never guessed.

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

const words = (text: string) => text.trim().split(/\s+/).length;

const SIMPLIFIED =
  /[们这说对时会过还没为国学发经东车马门见问间关谢请语话让给样后来个么吗现实点电开长觉师岁书认识爱机观应该听读写买卖钱饭馆气边远进运动业产报纸华汉号码银线网医药乐欢谁岛湾龙]/g;
const TRADITIONAL =
  /[們這說對時會過還沒為國學發經東車馬門見問間關謝請語話讓給樣後來個麼嗎現實點電開長覺師歲書認識愛機觀應該聽讀寫買賣錢飯館氣邊遠進運動業產報紙華漢號碼銀線網醫藥樂歡誰島灣龍]/g;
const CANTONESE = /[嘅咗喺哋冇嘢佢啲嚟噉咁乜唔]/g;

function han(text: string): string {
  if (count(text, CANTONESE) >= 2) return "yue";
  return count(text, TRADITIONAL) > count(text, SIMPLIFIED) ? "zh-Hant" : "zh";
}

function devanagari(text: string): string | null {
  const marathi = count(text, /आहे|आणि|नाही|ळ|होते|आहेत|आहात|तुम्ही/g);
  const hindi = count(text, /है|और|नहीं|हैं|था|मैं|आप |क्या/g);
  if (marathi > hindi) return "mr";
  if (hindi > marathi) return "hi";
  return words(text) >= 3 ? "hi" : null;
}

function arabic(text: string): string | null {
  if (count(text, /[ەۇۆۈېڭ]/g) >= 2) return "ug";
  if (count(text, /[ںےٹڈڑہھ]/g) >= 2) return "ur";
  if (count(text, /[پچژگ]/g) >= 1) return "fa";
  if (count(text, /[ةىأإؤئيك]/g) >= 1) return "ar";
  return null;
}

function cyrillic(text: string): string | null {
  const lower = text.toLowerCase();
  if (count(lower, /[әғқңұһ]/g) >= 1) return "kk";
  if (count(lower, /[өү]/g) >= 1) return "mn";
  if (count(lower, /[іїєґ]/g) >= 1) return "uk";
  if (count(lower, /[ыэъё]/g) >= 1 || words(text) >= 5) return "ru";
  return null;
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
  if (script === "cyrillic") return cyrillic(text);
  return null;
}
