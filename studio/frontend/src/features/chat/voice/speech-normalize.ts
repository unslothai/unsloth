// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
const UNITS: Record<string, readonly [one: string, many: string]> = {
  km: ["kilometer", "kilometers"], kg: ["kilogram", "kilograms"], cm: ["centimeter", "centimeters"],
  mm: ["millimeter", "millimeters"], mg: ["milligram", "milligrams"], mi: ["mile", "miles"],
  mph: ["mile per hour", "miles per hour"], kph: ["kilometer per hour", "kilometers per hour"],
  lb: ["pound", "pounds"], lbs: ["pound", "pounds"], oz: ["ounce", "ounces"], ft: ["foot", "feet"],
  ml: ["milliliter", "milliliters"], mL: ["milliliter", "milliliters"], GB: ["gigabyte", "gigabytes"],
  MB: ["megabyte", "megabytes"], TB: ["terabyte", "terabytes"], GHz: ["gigahertz", "gigahertz"],
  MHz: ["megahertz", "megahertz"], kW: ["kilowatt", "kilowatts"],
};
const DAYS: Record<string, string> = { Mon: "Monday", Tue: "Tuesday", Tues: "Tuesday", Wed: "Wednesday",
  Thu: "Thursday", Thur: "Thursday", Thurs: "Thursday", Fri: "Friday", Sat: "Saturday", Sun: "Sunday" };
const MONTH_ABBR: Record<string, string> = { Jan: "January", Feb: "February", Mar: "March", Apr: "April",
  Jun: "June", Jul: "July", Aug: "August", Sep: "September", Sept: "September", Oct: "October",
  Nov: "November", Dec: "December" };
const MONTH_WORD = "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\\.?|January|February|March|April|June|July|August|September|October|November|December";
const unit = (n: string, pair: readonly [string, string]) => `${n} ${n === "1" ? pair[0] : pair[1]}`;
const per = (n: string, one: string, many: string) => `${n} ${n === "1" ? one : many}`;

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

const IRREGULAR_ORDINALS: Readonly<Record<string, string>> = { one: "first", two: "second", three: "third",
  five: "fifth", eight: "eighth", nine: "ninth", twelve: "twelfth" };

function ordinal(digits: string): string {
  const words = cardinal(digits).split(" ");
  const last = words.pop()!;
  words.push(IRREGULAR_ORDINALS[last] ?? (last.endsWith("y") ? `${last.slice(0, -1)}ieth` : `${last}th`));
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
// Over 15 digits: codes or ids, read as written.
const plain = (digits: string) => digits.replace(/,/g, "").length <= 15;

// Largest first: the greedy rebuild needs it.
const ROMAN_CANONICAL = [["L", 50], ["XL", 40], ["X", 10], ["IX", 9], ["V", 5], ["IV", 4], ["I", 1]] as const;

function romanValue(text: string): number | null {
  let total = 0;
  for (let i = 0; i < text.length; i++) {
    const value = ROMAN[text[i]];
    const next = ROMAN[text[i + 1]] ?? 0;
    total += value < next ? -value : value;
  }
  let rebuilt = "";
  for (let rest = total, k = 0; rest > 0; ) {
    const [sym, val] = ROMAN_CANONICAL[k];
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
  // Under 10 cents needs "and", or the cents read as dollars.
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

// Rule order matters.
const RULES: Readonly<Record<string, readonly (readonly [RegExp, Spoken])[]>> = {
  en: [
    // Durations first, or "30m" reads as meters.
    [/\b(\d{1,3})\s?(?:h|hrs?)\s?(\d{1,2})\s?m(?:in|ins)?\b/g,
      (_m, h = "", m = "") => `${per(h, "hour", "hours")} and ${per(m, "minute", "minutes")}`],
    [/\b(\d{1,3})\s?(?:h|hrs?)\b(?!\.\d|\w)/g, (_m, h = "") => per(h, "hour", "hours")],
    [/\b(\d{1,4})\s?mins?\b/g, (_m, m = "") => per(m, "minute", "minutes")],
    [/\b(\d{1,4})\s?secs?\b/g, (_m, s = "") => per(s, "second", "seconds")],
    [/(\d(?:[\d,]*\d)?(?:\.\d+)?)\s?(km|kg|cm|mm|mg|mi|mph|kph|lbs?|oz|ft|ml|mL|GB|MB|TB|GHz|MHz|kW)(?![\w/])/g,
      (_m, n = "", u = "") => unit(n, UNITS[u])],
    [/(?<![\d.])(\d+(?:\.\d+)?)\s?°\s?([CF])\b/g,
      (_m, n = "", scale) => `${n} ${n === "1" ? "degree" : "degrees"} ${scale === "C" ? "Celsius" : "Fahrenheit"}`],
    // "Tue" reads as "two": expand it only beside a date.
    [new RegExp(`\\b(Mon|Tues?|Wed|Thu(?:rs?)?|Fri|Sat|Sun)\\.?(?=,?\\s+(?:(?:${MONTH_WORD})\\b|\\d))`, "g"),
      (_m, d = "") => DAYS[d]],
    [/\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?\s+([1-9]|[12]\d|3[01])\b(?![:.,]?\d)(?=$|[,.;:!?)]|\s+(?:\d{4}|at|to|and|through|until|from|or)\b)/g,
      (_m, mo = "", d = "") => `${MONTH_ABBR[mo]} ${ordinal(d)}`],
    [/\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+([1-9]|[12]\d|3[01])\b(?![:.,]?\d)(?=$|[,.;:!?)]|\s+(?:\d{4}|at|to|and|through|until|from|or)\b)/g,
      (_m, mo = "", d = "") => `${mo} ${ordinal(d)}`],
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
        // "Book I'd" and "Mark I think" keep the "I".
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
