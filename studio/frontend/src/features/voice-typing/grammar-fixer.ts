// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Instant, offline grammar and spelling fixer for dictated text.
//
// A speech model already spells and punctuates well. What is left is the long
// tail of small mistakes that a rule-based linter catches reliably: wrong
// articles, repeated words, missing apostrophes, sentence capitalisation,
// "should of", their/there and your/you're slips and spacing around
// punctuation. The rule set is modelled on the lint categories of Harper
// (a/an, repeated words, sentence capitalisation, spell-check of common
// contractions, confused words) but is a few kilobytes of plain TypeScript, so
// it adds nothing measurable to the app size, runs in well under a
// millisecond and needs no model, no WASM and no download.
//
// Every rule is deliberately high precision: when a fix could be wrong, it is
// left out. Text that is not English is never touched.

type Rule = readonly [RegExp, string | ((...match: string[]) => string)];

const MASK_OPEN = "\uE000";
const MASK_CLOSE = "\uE001";

/** URLs, e-mails, paths, file names, handles and identifiers must never be edited. */
const PROTECTED =
  /https?:\/\/\S+|\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b|\S*[\\/_#@]\S*|\b\w+(?:\.\w+)*\.(?:com|org|net|io|dev|ai|app|js|ts|tsx|py|rs|md|json|txt|html|css|yml|yaml|toml|exe|sh)\b|\b\w*\d\w*\b/g;

function mask(text: string): { text: string; tokens: string[] } {
  const tokens: string[] = [];
  const masked = text.replace(PROTECTED, (token) => {
    tokens.push(token);
    return `${MASK_OPEN}${tokens.length - 1}${MASK_CLOSE}`;
  });
  return { text: masked, tokens };
}

function unmask(text: string, tokens: string[]): string {
  return text.replace(
    new RegExp(`${MASK_OPEN}(\\d+)${MASK_CLOSE}`, "g"),
    (_m, index: string) => tokens[Number(index)] ?? "",
  );
}

function matchCase(source: string, replacement: string): string {
  if (source.length > 1 && source === source.toUpperCase()) {
    return replacement.toUpperCase();
  }
  if (source[0] && source[0] === source[0].toUpperCase()) {
    return replacement[0].toUpperCase() + replacement.slice(1);
  }
  return replacement;
}

// ---------------------------------------------------------------------------
// Contractions and common misspellings
// ---------------------------------------------------------------------------

const CONTRACTIONS: Record<string, string> = {
  dont: "don't",
  doesnt: "doesn't",
  didnt: "didn't",
  cant: "can't",
  wont: "won't",
  isnt: "isn't",
  arent: "aren't",
  wasnt: "wasn't",
  werent: "weren't",
  couldnt: "couldn't",
  wouldnt: "wouldn't",
  shouldnt: "shouldn't",
  hasnt: "hasn't",
  havent: "haven't",
  hadnt: "hadn't",
  mustnt: "mustn't",
  im: "I'm",
  ive: "I've",
  youre: "you're",
  theyre: "they're",
  youve: "you've",
  theyve: "they've",
  weve: "we've",
  thats: "that's",
  whats: "what's",
  heres: "here's",
  theres: "there's",
  wouldve: "would've",
  couldve: "could've",
  shouldve: "should've",
};

const MISSPELLINGS: Record<string, string> = {
  alot: "a lot",
  irregardless: "regardless",
  noone: "no one",
  incase: "in case",
  infact: "in fact",
  atleast: "at least",
  eachother: "each other",
  aswell: "as well",
  alright: "alright",
  definately: "definitely",
  definitly: "definitely",
  seperate: "separate",
  seperately: "separately",
  occured: "occurred",
  occurence: "occurrence",
  recieve: "receive",
  recieved: "received",
  untill: "until",
  wich: "which",
  becuase: "because",
  beacuse: "because",
  tommorow: "tomorrow",
  tomorow: "tomorrow",
  accross: "across",
  adress: "address",
  beleive: "believe",
  calender: "calendar",
  concious: "conscious",
  existance: "existence",
  goverment: "government",
  immediatly: "immediately",
  independant: "independent",
  neccessary: "necessary",
  necesary: "necessary",
  occassion: "occasion",
  persistant: "persistent",
  publically: "publicly",
  refered: "referred",
  relevent: "relevant",
  succesful: "successful",
  successfull: "successful",
  tounge: "tongue",
  truely: "truly",
  wierd: "weird",
  thier: "their",
  teh: "the",
  adn: "and",
  taht: "that",
  woudl: "would",
  coudl: "could",
  shoudl: "should",
};

/** Words that are brands or acronyms and keep one fixed spelling. */
const PROPER_TERMS: Record<string, string> = {
  github: "GitHub",
  gitlab: "GitLab",
  youtube: "YouTube",
  javascript: "JavaScript",
  typescript: "TypeScript",
  openai: "OpenAI",
  chatgpt: "ChatGPT",
  iphone: "iPhone",
  ipad: "iPad",
  macos: "macOS",
  ios: "iOS",
  linkedin: "LinkedIn",
  nvidia: "NVIDIA",
  pytorch: "PyTorch",
  llm: "LLM",
  llms: "LLMs",
  gpu: "GPU",
  gpus: "GPUs",
  cpu: "CPU",
  cpus: "CPUs",
  api: "API",
  apis: "APIs",
  ai: "AI",
  json: "JSON",
  html: "HTML",
  css: "CSS",
  sql: "SQL",
  url: "URL",
  urls: "URLs",
  usb: "USB",
  pdf: "PDF",
  ui: "UI",
  ux: "UX",
  vram: "VRAM",
  ssd: "SSD",
  wifi: "Wi-Fi",
  monday: "Monday",
  tuesday: "Tuesday",
  wednesday: "Wednesday",
  thursday: "Thursday",
  friday: "Friday",
  saturday: "Saturday",
  sunday: "Sunday",
  january: "January",
  february: "February",
  april: "April",
  june: "June",
  july: "July",
  august: "August",
  september: "September",
  october: "October",
  november: "November",
  december: "December",
};

// ---------------------------------------------------------------------------
// Confused words, agreement and idioms
// ---------------------------------------------------------------------------

const PHRASE_RULES: readonly Rule[] = [
  [/\b(should|would|could|must|might) of\b/gi, (_m, verb) => `${verb} have`],
  [/\byour (welcome)\b/gi, (_m, w) => `you're ${w}`],
  [
    /\byour (going|not|a|an|the|being|doing|right|wrong|welcome|gonna|able|supposed|about|really|very|so|too|also|just|still|always|never|already)\b/gi,
    (_m, w) => `you're ${w}`,
  ],
  [
    /\b(their) (is|are|was|were|isn't|aren't|wasn't|weren't|has|have)\b/gi,
    (_m, _t, verb) => `there ${verb}`,
  ],
  [/\bthere (own)\b/gi, (_m, w) => `their ${w}`],
  [/\bthey're (own)\b/gi, (_m, w) => `their ${w}`],
  [
    /\bits (a|an|been|not|going|just|very|too|really|also|only|now|because|important|okay|ok|time|true|possible|impossible|easy|hard|fine|good|bad|better|worse|late|early|working|done|clear|obvious)\b/gi,
    (_m, w) => `it's ${w}`,
  ],
  [
    /\b(more|less|better|worse|rather|other|fewer|greater|smaller|larger|higher|lower|faster|slower|bigger|easier|harder|older|younger|longer|shorter) then\b/gi,
    (_m, w) => `${w} than`,
  ],
  [/\bto (much|many)\b/gi, (_m, w) => `too ${w}`],
  [/\ban other\b/gi, (m) => matchCase(m, "another")],
  [/\bfor all intensive purposes\b/gi, () => "for all intents and purposes"],
  [/\bcould care less\b/gi, () => "couldn't care less"],
  [/\bcan able to\b/gi, () => "can"],
  [/\bmore (better|easier|faster|bigger|smaller)\b/gi, (_m, w) => w],
  [/\b(he|she|it) don't\b/gi, (_m, w) => `${w} doesn't`],
  [/\b(they|we|you) was\b/gi, (_m, w) => `${w} were`],
  [/\b(I|we|they|you) (has)\b/g, (_m, w) => `${w} have`],
  [
    /\bthere is (many|several|multiple|few|a few|a lot of|two|three|four|five|six|seven|eight|nine|ten)\b/gi,
    (_m, w) => `there are ${w}`,
  ],
  [/\bi\s+could\s+of\b/gi, () => "I could have"],
  [/\balot of\b/gi, () => "a lot of"],
  [/\b(a) (hour|honest|honor|honour|heir|herb)\b/gi, (_m, _a, w) => `an ${w}`],
];

// ---------------------------------------------------------------------------
// a / an
// ---------------------------------------------------------------------------

const AN_EXCEPTIONS = /^(hour|honest|honou?r|heir|herb|hono)/i;
const A_EXCEPTIONS =
  /^(uni([^nmd]|$)|use|usu|uti|uro|eu|one|once|ubi|ura|uk\b|uran|ute)/i;
const VOWEL_SOUND_LETTERS = new Set("AEIOFHLMNRSX");

function startsWithVowelSound(word: string): boolean {
  const clean = word.replace(/^[^A-Za-z]+/, "");
  if (!clean) return false;
  if (clean.length <= 5 && clean === clean.toUpperCase() && /[A-Z]{2,}/.test(clean)) {
    // Acronyms are read letter by letter: "an HTML", "a GPU", "an LLM".
    return VOWEL_SOUND_LETTERS.has(clean[0]);
  }
  if (AN_EXCEPTIONS.test(clean)) return true;
  if (A_EXCEPTIONS.test(clean)) return false;
  return /^[aeiou]/i.test(clean);
}

function fixArticles(text: string): string {
  return text.replace(
    /\b(a|an)(\s+)([A-Za-z][A-Za-z'-]*)/gi,
    (match, article: string, space: string, word: string) => {
      // "A" alone as a letter or list marker, or "a" followed by a masked token.
      const wantsAn = startsWithVowelSound(word);
      const isAn = article.toLowerCase() === "an";
      if (wantsAn === isAn) return match;
      const fixed = wantsAn ? "an" : "a";
      return `${matchCase(article, fixed)}${space}${word}`;
    },
  );
}

// ---------------------------------------------------------------------------
// Repeated words
// ---------------------------------------------------------------------------

/** Legitimate doubled words and sound effects. */
const ALLOWED_REPEATS = new Set([
  "had",
  "that",
  "is",
  "bye",
  "no",
  "so",
  "yeah",
  "very",
  "really",
  "ha",
  "blah",
  "knock",
  "tick",
  "boom",
  "well",
  "far",
  "long",
  "much",
  "many",
  "yes",
  "oh",
  "hey",
  "hi",
  "go",
  "now",
  "ok",
  "okay",
  "please",
  "sorry",
  "thank",
  "thanks",
  "nice",
  "good",
  "more",
  "bit",
  "day",
  "by",
  "one",
]);

function fixRepeatedWords(text: string): string {
  let current = text;
  for (let pass = 0; pass < 3; pass += 1) {
    const next = collapseRepeats(current);
    if (next === current) break;
    current = next;
  }
  return current;
}

function collapseRepeats(text: string): string {
  return text.replace(
    /\b([A-Za-z']+)(,?\s+)(\1)\b(?![-'])/gi,
    (match, first: string) => {
      if (ALLOWED_REPEATS.has(first.toLowerCase())) return match;
      return first;
    },
  );
}

// ---------------------------------------------------------------------------
// Spacing, punctuation and capitalisation
// ---------------------------------------------------------------------------

function fixSpacing(text: string): string {
  return text
    .replace(/[ \t]{2,}/g, " ")
    .replace(/ +([,.;:!?])(?!\.)/g, "$1")
    .replace(/,{2,}/g, ",")
    .replace(/(?<![.])\.\.(?![.])/g, ".")
    .replace(/,\s*\./g, ".")
    .replace(/([a-z]{2,})([,;:])(?=[A-Za-z])/g, "$1$2 ")
    .replace(/([a-z]{2,})([.!?])(?=[A-Z][a-z])/g, "$1$2 ")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")");
}

function capitalizeSentences(text: string): string {
  return text.replace(
    /(^|[.!?\u2026]\s+|\n+)(["'\u201c\u2018([]*)(\p{Ll})/gu,
    (_m, prefix: string, opening: string, letter: string) =>
      `${prefix}${opening}${letter.toUpperCase()}`,
  );
}

function fixStandaloneI(text: string): string {
  return text
    .replace(/\bi(?='(?:m|ll|ve|d)\b)/g, "I")
    .replace(/\bi\b(?!['.\-@/])/g, "I");
}

function fixWordMaps(text: string): string {
  return text.replace(/\b[A-Za-z]+\b/g, (word) => {
    const lower = word.toLowerCase();
    const contraction = CONTRACTIONS[lower];
    if (contraction && lower !== "im") {
      return word === word.toUpperCase() && word.length > 3
        ? contraction.toUpperCase()
        : matchCase(word, contraction);
    }
    // "im" is only a contraction when lower-case; "IM" and "Im" stay as typed.
    if (lower === "im" && word === "im") return "I'm";
    const spelling = MISSPELLINGS[lower];
    if (spelling && spelling !== lower) return matchCase(word, spelling);
    const proper = PROPER_TERMS[lower];
    if (proper) return proper;
    return word;
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** True when the text is plausibly English: the fixer never edits other languages. */
function isEnglishLike(text: string, language: string): boolean {
  const base = language.trim().toLowerCase().split(/[-_]/, 1)[0] ?? "";
  if (base && base !== "en" && base !== "auto") return false;
  const letters = text.match(/\p{L}/gu);
  if (!letters || letters.length === 0) return false;
  const ascii = letters.filter((ch) => ch.charCodeAt(0) < 128).length;
  return ascii / letters.length >= 0.9;
}

export interface GrammarFixOptions {
  /** Dictation language, e.g. "en" or "en-US". Empty means automatic. */
  language?: string;
}

/**
 * Fix the common grammar, spelling, spacing and capitalisation mistakes in
 * dictated English text. Never throws and returns the input on any failure.
 */
export function fixGrammar(
  input: string,
  options: GrammarFixOptions = {},
): string {
  if (!input || !input.trim()) return input;
  if (!isEnglishLike(input, options.language ?? "")) return input;
  try {
    const masked = mask(input);
    let text = masked.text;
    text = fixSpacing(text);
    text = fixStandaloneI(text);
    text = fixWordMaps(text);
    for (const [pattern, replacement] of PHRASE_RULES) {
      text = text.replace(pattern, replacement as never);
    }
    text = fixRepeatedWords(text);
    text = fixArticles(text);
    text = fixStandaloneI(text);
    text = capitalizeSentences(text);
    return unmask(text, masked.tokens);
  } catch {
    return input;
  }
}
