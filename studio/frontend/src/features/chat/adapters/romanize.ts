// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Best-effort romanisation for dictation. When the user picks the
// "Romanized" language option, non-Latin transcripts are transliterated into
// Latin letters instead of being left in their native script. This is a
// rule-based pass (no model download): it handles Urdu/Arabic, Hindi and
// Cyrillic/Greek well enough for dictation, and leaves scripts it cannot
// faithfully romanise (CJK, for example) untouched.

const URDU_WORDS: Record<string, string> = {
  آپ: "aap",
  میں: "main",
  مجھے: "mujhe",
  ہم: "hum",
  تم: "tum",
  وہ: "woh",
  یہ: "yeh",
  ہے: "hai",
  ہیں: "hain",
  تھا: "tha",
  تھی: "thi",
  تھے: "the",
  کیا: "kya",
  کیسے: "kaise",
  کیوں: "kyun",
  کہاں: "kahan",
  کب: "kab",
  کون: "kaun",
  نہیں: "nahi",
  ہاں: "haan",
  شکریہ: "shukriya",
  مہربانی: "meharbani",
  السلام: "assalam",
  علیکم: "alaikum",
  خدا: "khuda",
  حافظ: "hafiz",
  صبح: "subah",
  بخیر: "bakhair",
  شب: "shab",
  میرا: "mera",
  میری: "meri",
  میرے: "mere",
  نام: "naam",
  حال: "haal",
  ٹھیک: "theek",
  بہت: "bohat",
  اچھا: "achha",
  اچھی: "achhi",
  اچھے: "achhe",
  کرو: "karo",
  کر: "kar",
  کریں: "karein",
  رہا: "raha",
  رہی: "rahi",
  رہے: "rahe",
  گیا: "gaya",
  گئی: "gayi",
  گئے: "gaye",
  آیا: "aaya",
  آئی: "aayi",
  آئے: "aaye",
  دو: "do",
  تین: "teen",
  چار: "chaar",
  پانچ: "paanch",
  چھ: "chhe",
  سات: "saat",
  آٹھ: "aath",
  نو: "nau",
  دس: "das",
  ایک: "ek",
  سو: "sau",
  ہزار: "hazaar",
  لاکھ: "lakh",
  کروڑ: "crore",
  دن: "din",
  رات: "raat",
  آج: "aaj",
  کل: "kal",
  شام: "shaam",
  دوپہر: "dopahar",
  گھر: "ghar",
  کام: "kaam",
  پانی: "paani",
  کھانا: "khana",
  کتاب: "kitaab",
  دوست: "dost",
  بھائی: "bhai",
  بہن: "behan",
  ماں: "maan",
  باپ: "baap",
  بیٹا: "beta",
  بیٹی: "beti",
  شادی: "shaadi",
  محبت: "mohabbat",
  زندگی: "zindagi",
  دنیا: "duniya",
  وقت: "waqt",
  بات: "baat",
  لوگ: "log",
  یہاں: "yahan",
  وہاں: "wahan",
  ابھی: "abhi",
  پھر: "phir",
  بھی: "bhi",
  تو: "to",
  اور: "aur",
  لیکن: "lekin",
  کیونکہ: "kyunke",
  اگر: "agar",
  جب: "jab",
  تب: "tab",
  ساتھ: "saath",
  بغیر: "baghair",
  لیے: "liye",
  لئے: "liye",
  طرح: "tarah",
  لیں: "lein",
  دیں: "dein",
  جائیں: "jayein",
  آئیں: "aayein",
  دیکھیں: "dekhein",
  سنیں: "sunein",
  بولیں: "bolein",
  لکھیں: "likhein",
  پڑھیں: "parhein",
};

const URDU_CHARS: Record<string, string> = {
  ء: "",
  آ: "aa",
  ا: "a",
  ب: "b",
  پ: "p",
  ت: "t",
  ٹ: "t",
  ث: "s",
  ج: "j",
  چ: "ch",
  ح: "h",
  خ: "kh",
  د: "d",
  ڈ: "d",
  ذ: "z",
  ر: "r",
  ڑ: "r",
  ز: "z",
  ژ: "zh",
  س: "s",
  ش: "sh",
  ص: "s",
  ض: "z",
  ط: "t",
  ظ: "z",
  ع: "a",
  غ: "gh",
  ف: "f",
  ق: "q",
  ک: "k",
  گ: "g",
  ل: "l",
  م: "m",
  ن: "n",
  ں: "n",
  ہ: "h",
  ھ: "h",
  ی: "i",
  ے: "e",
  ۓ: "e",
  و: "o",
  ؤ: "o",
  ئ: "i",
  أ: "a",
  إ: "i",
  ة: "a",
  ى: "a",
};

const ARABIC_CHARS: Record<string, string> = {
  ...URDU_CHARS,
  ث: "th",
  ذ: "dh",
  ش: "sh",
  ص: "s",
  ض: "d",
  ط: "t",
  ظ: "z",
  ع: "a",
  غ: "gh",
  ق: "q",
  ك: "k",
  و: "w",
  ي: "y",
  ة: "a",
};

const CYRILLIC_CHARS: Record<string, string> = {
  а: "a",
  б: "b",
  в: "v",
  г: "g",
  д: "d",
  е: "e",
  ё: "yo",
  ж: "zh",
  з: "z",
  и: "i",
  й: "y",
  к: "k",
  л: "l",
  м: "m",
  н: "n",
  о: "o",
  п: "p",
  р: "r",
  с: "s",
  т: "t",
  у: "u",
  ф: "f",
  х: "kh",
  ц: "ts",
  ч: "ch",
  ш: "sh",
  щ: "shch",
  ъ: "",
  ы: "y",
  ь: "",
  э: "e",
  ю: "yu",
  я: "ya",
};

const GREEK_CHARS: Record<string, string> = {
  α: "a",
  β: "v",
  γ: "g",
  δ: "d",
  ε: "e",
  ζ: "z",
  η: "i",
  θ: "th",
  ι: "i",
  κ: "k",
  λ: "l",
  μ: "m",
  ν: "n",
  ξ: "x",
  ο: "o",
  π: "p",
  ρ: "r",
  σ: "s",
  ς: "s",
  τ: "t",
  υ: "y",
  φ: "f",
  χ: "ch",
  ψ: "ps",
  ω: "o",
};

const DEVANAGARI_VOWELS: Record<string, string> = {
  अ: "a",
  आ: "aa",
  इ: "i",
  ई: "ee",
  उ: "u",
  ऊ: "oo",
  ऋ: "ri",
  ए: "e",
  ऐ: "ai",
  ओ: "o",
  औ: "au",
};

const DEVANAGARI_MARKS: Record<string, string> = {
  "ा": "a",
  "ि": "i",
  "ी": "ee",
  "ु": "u",
  "ू": "oo",
  "ृ": "ri",
  "े": "e",
  "ै": "ai",
  "ो": "o",
  "ौ": "au",
  "ं": "n",
  "ँ": "n",
  "ः": "h",
};

const DEVANAGARI_CONSONANTS: Record<string, string> = {
  क: "k",
  ख: "kh",
  ग: "g",
  घ: "gh",
  ङ: "ng",
  च: "ch",
  छ: "chh",
  ज: "j",
  झ: "jh",
  ञ: "n",
  ट: "t",
  ठ: "th",
  ड: "d",
  ढ: "dh",
  ण: "n",
  त: "t",
  थ: "th",
  द: "d",
  ध: "dh",
  न: "n",
  प: "p",
  फ: "ph",
  ब: "b",
  भ: "bh",
  म: "m",
  य: "y",
  र: "r",
  ल: "l",
  व: "v",
  श: "sh",
  ष: "sh",
  स: "s",
  ह: "h",
  ळ: "l",
  क़: "q",
  ख़: "kh",
  ग़: "gh",
  ज़: "z",
  ड़: "r",
  ढ़: "rh",
  फ़: "f",
};

const ARABIC_DIACRITICS = /[\u064B-\u065F\u0670\u06D6-\u06ED]/u;

function isDevanagari(ch: string): boolean {
  const code = ch.codePointAt(0) ?? 0;
  return code >= 0x0900 && code <= 0x097f;
}

function romanizeDevanagari(text: string): string {
  const chars = Array.from(text);
  let out = "";
  let word = "";
  const flush = () => {
    if (!word) return;
    // Hindi's inherent final "a" is usually silent: "aap", not "aapa".
    out += /[a-z]$/.test(word) && word.endsWith("a") ? word.slice(0, -1) : word;
    word = "";
  };
  for (let index = 0; index < chars.length; index += 1) {
    const ch = chars[index];
    if (!isDevanagari(ch)) {
      flush();
      out += ch;
      continue;
    }
    if (ch === "्") {
      // Virama removes the inherent vowel; the consonant is already emitted.
      continue;
    }
    const vowel = DEVANAGARI_VOWELS[ch];
    if (vowel) {
      word += vowel;
      continue;
    }
    const mark = DEVANAGARI_MARKS[ch];
    if (mark) {
      word += mark;
      continue;
    }
    const consonant = DEVANAGARI_CONSONANTS[ch];
    if (consonant) {
      word += consonant;
      const next = chars[index + 1];
      if (next && (DEVANAGARI_MARKS[next] || next === "्")) {
        // The following mark supplies the vowel.
        continue;
      }
      word += "a";
      continue;
    }
    word += ch;
  }
  flush();
  return out;
}

function romanizeArabicScript(text: string): string {
  const urdu = /[ٹڈڑںےۓھ]/u.test(text);
  const map = urdu ? URDU_CHARS : ARABIC_CHARS;
  const words = text.split(/(\s+)/);
  return words
    .map((part) => {
      if (!part || /^\s+$/.test(part)) return part;
      const core = /^([^\p{L}]*)([\p{L}]+)([^\p{L}]*)$/u.exec(part);
      if (core) {
        const dictionary = URDU_WORDS[core[2]];
        if (dictionary) return `${core[1]}${dictionary}${core[3]}`;
      }
      let out = "";
      for (const ch of part) {
        if (ARABIC_DIACRITICS.test(ch)) continue;
        out += map[ch] ?? ch;
      }
      return out;
    })
    .join("")
    .replaceAll("؟", "?")
    .replaceAll("،", ",")
    .replaceAll("؛", ";");
}

function romanizeAlphabet(
  text: string,
  map: Record<string, string>,
): string {
  return Array.from(text)
    .map((ch) => {
      const lower = ch.toLowerCase();
      const mapped = map[lower];
      if (!mapped) return ch;
      if (ch === lower) return mapped;
      return mapped.charAt(0).toUpperCase() + mapped.slice(1);
    })
    .join("");
}

/**
 * Transliterate non-Latin dictation into Latin letters. Latin text passes
 * through unchanged; scripts without a rule set are returned as-is.
 */
export function romanizeToLatin(text: string): string {
  if (!text) return text;
  if (!/[\u0400-\u04FF\u0370-\u03FF\u0600-\u06FF\u0900-\u097F]/u.test(text)) {
    return text;
  }
  if (/[\u0600-\u06FF]/u.test(text)) return romanizeArabicScript(text);
  if (/[\u0900-\u097F]/u.test(text)) {
    return romanizeDevanagari(text);
  }
  if (/[\u0400-\u04FF]/u.test(text)) {
    return romanizeAlphabet(text, CYRILLIC_CHARS);
  }
  return romanizeAlphabet(text, GREEK_CHARS);
}
