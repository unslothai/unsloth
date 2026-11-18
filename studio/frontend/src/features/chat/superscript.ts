// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The rule web fetch applies to <sup> (studio/backend/core/inference/_html_to_md.py), for attachments.
export const SUPERSCRIPT_BASE_CHARS = 128;

const PLAIN_SUFFIXES = new Set(["st", "nd", "rd", "th", "tm", "sm", "mc", "md", "(tm)", "(sm)", "(r)", "(c)", "mr", "m.r."]);
const SECOND_SUFFIXES = new Set(["d", "de", "ds", "des"]);
const DIGIT_ORDINAL_SUFFIXES = new Set(
  "e es er ers re res ère ères ème èmes eme emes nd nde bis ter quater quinquies am pm a.m. p.m. ndes nds º ª o a".split(" "),
);
const SUPERIOR_ABBREVIATIONS = new Map(
  Object.entries({
    M: ["me", "mes", "lle", "lles", "gr", "e", "r", "rs", "s"],
    D: ["r", "rs", "re", "res"],
    P: ["r", "rs", "re", "res"],
    n: ["o", "os"],
    N: ["o", "os"],
    S: ["t", "te", "ts", "tes", "r"],
    J: ["r"],
    C: ["ie", "ies"],
  }),
);
const CURRENCY_CODES = new Set(
  ("AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL BSD BTN " +
    "BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD " +
    "FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY " +
    "KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP " +
    "MRU MUR MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR " +
    "RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP SVC STN SYP SZL THB TJS TMT TND " +
    "TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VED VES VND VUV WST XAF XCD XCG XOF XPF YER ZAR ZMW " +
    "ZWG ZWL").split(" "),
);
const THREE_DECIMAL_CURRENCIES = new Set(["BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"]);
const ZERO_DECIMAL_CURRENCIES = new Set("BIF CLP DJF GNF ISK JPY KMF KRW PYG RWF UGX VND VUV XAF XOF XPF".split(" "));
const FOOTNOTE_CLASSES = new Set(["reference", "footnote", "footnote-ref", "noteref", "fn", "cite"]);
const FOOTNOTE_CLASS_PREFIXES = ["footnote", "noteref", "cite", "citation", "endnote", "fnref", "reference"];

const CITATION_LIST_RE = /^\d+(?:\s*[–,]\s*\d+)+$/;
const THOUSANDS_RE = /^\d{1,3}(?:,\d{3})+$/;
const ROMAN_NUMERAL_TAIL_RE = /(?<!\p{L})[IVXLCDM]+$/u;
const LAST_WORD_RE = /(?<!\p{L})\p{L}+$/u;
const BARE_EXPONENT_RE = /^[-+−]?(?:\d+(?:[.,]\d+)?|\p{L})$/u;
const PRICE_TAIL_RE = /(\S)\s?\d(?:[\d,.'’]|[ \u00a0\u202f]\d)*$/u;
const CODE_PRICE_TAIL_RE = /\b([A-Z]{3})[ \u00a0\u202f]?\d(?:[\d,.'’]|[ \u00a0\u202f]\d)*$/;
const ALNUM_RE = /[\p{L}\p{N}]/u;
const DIGIT_RE = /\p{Nd}/u;

function afterPrice(base: string): number {
  const context = base.slice(-40);
  const sign = PRICE_TAIL_RE.exec(context)?.[1];
  if (sign && /\p{Sc}/u.test(sign)) return "₫₲₩".includes(sign) ? 0 : 2;
  const code = CODE_PRICE_TAIL_RE.exec(context)?.[1];
  if (!code || !CURRENCY_CODES.has(code) || ZERO_DECIMAL_CURRENCIES.has(code)) return 0;
  return THREE_DECIMAL_CURRENCIES.has(code) ? 3 : 2;
}

export function isNoteReference(className: string | null, role: string | null): boolean {
  return (
    (className ?? "")
      .toLowerCase()
      .split(/\s+/)
      .some(
        (token) =>
          FOOTNOTE_CLASSES.has(token) ||
          token.split(/[-_]/).some((part) => part === "fn" || FOOTNOTE_CLASS_PREFIXES.some((prefix) => part.startsWith(prefix))),
      ) || (role ?? "").toLowerCase().split(/\s+/).includes("doc-noteref")
  );
}

export function raiseSuperscript(before: string, raised: string): string {
  const base = before.slice(-SUPERSCRIPT_BASE_CHARS);
  const end = base.slice(-1);
  if (!end || !(ALNUM_RE.test(end) || ")]}|）］｝".includes(end))) return raised;
  const visible = raised.trim();
  const lower = visible.toLowerCase();
  const word = LAST_WORD_RE.exec(base)?.[0] ?? "";
  const price = DIGIT_RE.test(end) ? afterPrice(base) : 0;
  if (
    !visible ||
    visible.includes("\n") ||
    visible[0] === "[" ||
    (".,".includes(visible[0]) && price) ||
    !ALNUM_RE.test(visible) ||
    PLAIN_SUFFIXES.has(lower) ||
    (word.length > 1 && CITATION_LIST_RE.test(visible) && (visible.includes("–") || !THOUSANDS_RE.test(visible))) ||
    (end === "2" && SECOND_SUFFIXES.has(lower)) ||
    ((DIGIT_RE.test(end) || ROMAN_NUMERAL_TAIL_RE.test(base)) && DIGIT_ORDINAL_SUFFIXES.has(lower)) ||
    SUPERIOR_ABBREVIATIONS.get(word)?.includes(visible) ||
    (visible.length === price && /^\d+$/.test(visible))
  ) {
    return raised;
  }
  const bare =
    BARE_EXPONENT_RE.test(visible) ||
    (visible.startsWith("(") && visible.endsWith(")") && visible.split("(").length === 2);
  return (bare ? `^${visible}` : `^(${visible})`) + raised.slice(raised.trimEnd().length);
}
