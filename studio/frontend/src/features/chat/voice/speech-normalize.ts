// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure. Holds only the forms the Kokoro listening sample got wrong (all confirmed by ear): money,
// thousands separators, decimals, clock times, fractions, ISO dates, "No.", Roman numerals, phone
// numbers and GGUF. Each rewrites to words the voice reads correctly.

type Spoken = string | ((match: string, ...groups: (string | undefined)[]) => string);

const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES = ["", " thousand", " million", " billion", " trillion"];
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
  "October", "November", "December"];
const DENOMINATORS = ["", "", "half", "third", "quarter", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"];
const CURRENCY: Record<string, readonly [one: string, many: string, minorOne: string, minorMany: string]> = {
  $: ["dollar", "dollars", "cent", "cents"],
  "€": ["euro", "euros", "cent", "cents"],
  "£": ["pound", "pounds", "penny", "pence"],
  "¥": ["yen", "yen", "", ""],
};
const MONEY_SCALE: Record<string, string> = { K: "thousand", M: "million", B: "billion", T: "trillion" };
const ROMAN: Record<string, number> = { I: 1, V: 5, X: 10, L: 50 };

function below1000(n: number): string {
  const words: string[] = [];
  if (n >= 100) words.push(`${ONES[Math.floor(n / 100)]} hundred`);
  const rest = n % 100;
  if (rest >= 20) words.push(TENS[Math.floor(rest / 10)] + (rest % 10 ? ` ${ONES[rest % 10]}` : ""));
  else if (rest > 0 || words.length === 0) words.push(ONES[rest]);
  return words.join(" ");
}

function cardinal(digits: string): string {
  const n = Number(digits);
  if (n === 0) return "zero";
  const groups: string[] = [];
  for (let rest = n, scale = 0; rest > 0; rest = Math.floor(rest / 1000), scale++) {
    if (rest % 1000) groups.unshift(below1000(rest % 1000) + SCALES[scale]);
  }
  return groups.join(" ");
}

function ordinal(digits: string): string {
  const words = cardinal(digits).split(" ");
  const last = words.pop()!;
  const irregular: Record<string, string> = { one: "first", two: "second", three: "third", five: "fifth",
    eight: "eighth", nine: "ninth", twelve: "twelfth" };
  words.push(irregular[last] ?? (last.endsWith("y") ? `${last.slice(0, -1)}ieth` : `${last}th`));
  return words.join(" ");
}

function year(digits: string): string {
  const n = Number(digits);
  if (n >= 2000 && n < 2010) return cardinal(digits);
  if (n < 1100 || n >= 2100) return cardinal(digits);
  const rest = n % 100;
  const head = cardinal(String(Math.floor(n / 100)));
  if (rest === 0) return `${head} hundred`;
  return `${head} ${rest < 10 ? `oh ${ONES[rest]}` : cardinal(String(rest))}`;
}

const spellDigits = (digits: string) => [...digits].map((d) => ONES[Number(d)]).join(" ");
const decimal = (whole: string, fraction: string) => `${cardinal(whole)} point ${spellDigits(fraction)}`;
// Longer than this the digits are a code or an id, which the voice reads well enough as written.
const plain = (digits: string) => digits.replace(/,/g, "").length <= 15;

function romanValue(text: string): number | null {
  let total = 0;
  for (let i = 0; i < text.length; i++) {
    const value = ROMAN[text[i]];
    const next = ROMAN[text[i + 1]] ?? 0;
    total += value < next ? -value : value;
  }
  // Only the canonical spelling counts, so "IIII" or "VX" stay words.
  const canonical = [["L", 50], ["XL", 40], ["X", 10], ["IX", 9], ["V", 5], ["IV", 4], ["I", 1]] as const;
  let rebuilt = "";
  for (let rest = total, k = 0; rest > 0; ) {
    const [sym, val] = canonical[k];
    if (rest >= val) { rebuilt += sym; rest -= val; } else k++;
  }
  return rebuilt === text ? total : null;
}

function money(match: string, symbol = "$", written = "0", fraction?: string, letter?: string, word?: string): string {
  if (!plain(written)) return match;
  const whole = written.replace(/,/g, "");
  const [one, many, minorOne, minorMany] = CURRENCY[symbol];
  const scale = letter ? MONEY_SCALE[letter] : word;
  const amount = fraction ? decimal(whole, fraction) : cardinal(whole);
  if (scale) return `${amount} ${scale} ${many}`;
  if (!fraction || fraction.length !== 2 || !minorMany) return `${amount} ${amount === "one" ? one : many}`;
  const cents = Number(fraction);
  const minor = cents ? `${cardinal(fraction)} ${cents === 1 ? minorOne : minorMany}` : "";
  if (Number(whole) === 0) return minor || `zero ${many}`;
  const main = `${cardinal(whole)} ${whole === "1" ? one : many}`;
  // "five dollars twenty-five", but "one dollar and one cent": a lone small number reads as part of the dollars.
  if (!cents) return main;
  return cents < 10 ? `${main} and ${minor}` : `${main} ${cardinal(fraction)}`;
}

function clock(match: string, hour = "", minute = "", meridiem?: string): string {
  const h = Number(hour);
  if (h === 0 && !meridiem) return match;
  const said = h === 0 ? "twelve" : cardinal(hour);
  const suffix = meridiem ? ` ${meridiem.toLowerCase()} m` : "";
  if (minute === "00") return said + (suffix || (h <= 12 ? " o'clock" : " hundred"));
  return `${said} ${minute[0] === "0" ? `oh ${ONES[Number(minute[1])]}` : cardinal(minute)}${suffix}`;
}

// Keyed by the language part of the voice settings' BCP 47 tag ("en" for "en-US"). Order matters:
// a phone number or a date is read before its digits could match a plainer rule.
const RULES: Readonly<Record<string, readonly (readonly [RegExp, Spoken])[]>> = {
  en: [
    [/(?<![\w+])(?:\+(\d{1,3})[\s.-]?)?(?:\((\d{3})\)|\b(\d{3}))[\s.-]?(\d{3})[\s.-](\d{4})\b/g,
      (_m, country, area, bare, mid, last) =>
        [country && `plus ${spellDigits(country)}`, spellDigits(area ?? bare ?? ""), spellDigits(mid ?? ""), spellDigits(last ?? "")]
          .filter(Boolean).join(", ")],
    [/\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b/g,
      (_m, y = "", mo = "", d = "") => `${MONTHS[Number(mo) - 1]} ${ordinal(d)}, ${year(y)}`],
    [/([$€£¥])\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?:\s?([KMBT])(?![\w-])|\s(thousand|million|billion|trillion)\b)?/g,
      (m, symbol, whole, fraction, letter, word) => money(m, symbol, whole, fraction, letter, word)],
    [/\b([01]?\d|2[0-3]):([0-5]\d)(?:\s?([AaPp])\.?[Mm](?![A-Za-z])(?:\.(?=\s+[a-z]))?)?/g,
      (m, hour, minute, meridiem) => clock(m, hour, minute, meridiem)],
    // A comma before a space ends the number ("1,000, 2,000"); one before a digit is a malformed group.
    [/(?<![\d,.])(\d{1,3}(?:,\d{3})+)(?:\.(\d+))?(?!\d|,\d)/g,
      (m, whole = "", fraction) => (plain(whole) ? (fraction ? decimal(whole.replace(/,/g, ""), fraction) : cardinal(whole.replace(/,/g, ""))) : m)],
    [/(?<![\d.\w])(\d+)\.(\d+)(?!\.?\d)(?:(\s?%)|([KMBT])(?![\w-]))?/g,
      (m, whole = "", fraction = "", percent, letter) =>
        (plain(whole) ? decimal(whole, fraction) + (percent ? " percent" : letter ? ` ${MONEY_SCALE[letter]}` : "") : m)],
    [/\b24\/7\b/g, "twenty four seven"],
    [/(?<![\d/])([1-9])\/([2-9]|10)(?![\d/])/g,
      (m, top = "", bottom = "") => {
        const n = Number(top), d = Number(bottom);
        if (n >= d) return m;
        const name = DENOMINATORS[d];
        return `${ONES[n]} ${n === 1 ? name : name === "half" ? "halves" : `${name}s`}`;
      }],
    [/\b([Nn])o\.\s?(?=\d)/g, (_m, n) => `${n === "N" ? "N" : "n"}umber `],
    [/\b(War|Chapter|Part|Volume|Book|Act|Phase|Season|Episode|Bowl|Mark|Gen)\s([IVXL]{1,7})\b/g,
      (m, word, numeral = "", at, all) => {
        // A numeral is never followed by an apostrophe ("Book I'd"); after the name Mark, a lone I before a
        // lowercase word is the pronoun ("Mark I think"). "Chapter I of" stays a numeral.
        const after = String(all).slice(Number(at) + m.length);
        if (numeral === "I" && (/^['’]/.test(after) || (word === "Mark" && /^\s+[a-z]/.test(after)))) return m;
        const v = romanValue(numeral);
        return v ? `${word} ${cardinal(String(v))}` : m;
      }],
    [/\b([A-Z][a-z]+)\s([IVXL]{2,7})\b/g,
      (m, name, numeral = "") => {
        const v = /^X+L$/.test(numeral) ? null : romanValue(numeral);
        return v ? `${name} the ${ordinal(String(v))}` : m;
      }],
    [/\bGGUF\b/g, "G G U F"],
  ],
};

export function normalizeForSpeech(text: string, locale: string): string {
  const key = locale.split("-")[0].toLowerCase();
  const rules = Object.hasOwn(RULES, key) ? RULES[key] : [];
  return rules.reduce(
    (out, [pattern, spoken]) => (typeof spoken === "string" ? out.replace(pattern, spoken) : out.replace(pattern, spoken)),
    text,
  );
}
