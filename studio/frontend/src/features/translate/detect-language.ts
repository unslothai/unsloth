// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const SAMPLE_CHARS = 2000;

// Scripts that only one of the Translate languages uses.
const SINGLE_SCRIPT: readonly [RegExp, string][] = [
  [/\p{Script=Hangul}/gu, "ko"],
  [/\p{Script=Thai}/gu, "th"],
  [/\p{Script=Khmer}/gu, "km"],
  [/\p{Script=Myanmar}/gu, "my"],
  [/\p{Script=Tibetan}/gu, "bo"],
  [/\p{Script=Hebrew}/gu, "he"],
  [/\p{Script=Bengali}/gu, "bn"],
  [/\p{Script=Tamil}/gu, "ta"],
  [/\p{Script=Telugu}/gu, "te"],
  [/\p{Script=Gujarati}/gu, "gu"],
  [/\p{Script=Ethiopic}/gu, "am"],
];

const count = (text: string, pattern: RegExp) =>
  text.match(pattern)?.length ?? 0;
const countOf = (text: string, chars: string) =>
  count(text, new RegExp(`[${chars}]`, "gu"));
const tokensOf = (text: string) =>
  text.split(/[^\p{L}\p{M}]+/u).filter(Boolean);
const hits = (tokens: readonly string[], known: ReadonlySet<string>) =>
  tokens.filter((token) => known.has(token)).length;
const words = (list: string): ReadonlySet<string> => new Set(list.split(" "));

/** The best code, or null when it does not clearly beat the runner-up. */
function clearWinner(
  scores: readonly [string, number][],
  closeOk?: (a: string, b: string) => boolean,
): string | null {
  const [best, second] = [...scores].sort((a, b) => b[1] - a[1]);
  if (!best || best[1] < 2) return null;
  if (!second || best[1] >= second[1] * 1.5) return best[0];
  // Near-identical languages that only a few words tell apart.
  return closeOk?.(best[0], second[0]) && best[1] > second[1] ? best[0] : null;
}

// Same characters in the same order: each pair differs only by the Simplified / Traditional form.
const SIMPLIFIED =
  "这个们说来时国会对为发学过还没么吗见让经关东车门长问间开电话语书写买卖头气样亲爱听觉现应该种实点从无机进动业产员万与区处将并体华军网线级认识论设计证记议农场报声类务达运选际传统济战权历观两难题风飞马鱼鸟龙医药钱银铁错乐";
const TRADITIONAL =
  "這個們說來時國會對為發學過還沒麼嗎見讓經關東車門長問間開電話語書寫買賣頭氣樣親愛聽覺現應該種實點從無機進動業產員萬與區處將並體華軍網線級認識論設計證記議農場報聲類務達運選際傳統濟戰權歷觀兩難題風飛馬魚鳥龍醫藥錢銀鐵錯樂";

function chinese(text: string): string | null {
  if (countOf(text, "嘅咗唔冇佢哋嚟啲喺嘢嗰") > 0) return "yue";
  const traditional = countOf(text, TRADITIONAL);
  const simplified = countOf(text, SIMPLIFIED);
  if (traditional === simplified) return null;
  return traditional > simplified ? "zh-Hant" : "zh";
}

// Uyghur and Urdu letters first; Persian and Arabic differ in ye, kaf and ta marbuta.
function arabicScript(text: string): string | null {
  if (countOf(text, "ېۆۈۇۋە") > 0) return "ug";
  if (countOf(text, "ےٹڈڑںہ") > 0) return "ur";
  const persian = countOf(text, "پچژگیک");
  const arabic = countOf(text, "يكة");
  if (persian === arabic) return null;
  return persian > arabic ? "fa" : "ar";
}

const UKRAINIAN = words("що та це він вона як де дякую добре так від для");
const RUSSIAN = words("что это как где он она и спасибо хорошо да от для");
const MONGOLIAN = words("нь юм бол энэ вэ болон байна байгаа гэж юу бид тэр");

function cyrillic(text: string): string | null {
  if (countOf(text, "әғқңұһ") > 0) return "kk";
  // Kazakh shares ө and ү with Mongolian; і is Kazakh only of the two.
  if (countOf(text, "өү") > 0) return countOf(text, "і") > 0 ? "kk" : "mn";
  const tokens = tokensOf(text);
  if (hits(tokens, MONGOLIAN) > hits(tokens, RUSSIAN)) return "mn";
  const ukrainian = countOf(text, "іїєґ") + hits(tokens, UKRAINIAN);
  const russian = countOf(text, "ыъё") + hits(tokens, RUSSIAN);
  if (ukrainian === russian) return null;
  return ukrainian > russian ? "uk" : "ru";
}

const HINDI = words("है हैं और नहीं में था थे थी यह के का को से ने भी इस");
const MARATHI = words(
  "आहे आहेत आणि नाही मध्ये होते होता होती पण व या केले असे त्यांनी म्हणून आले",
);

function devanagari(text: string): string | null {
  const tokens = tokensOf(text);
  const marathi =
    hits(tokens, MARATHI) + countOf(text, "ळ") + text.split("च्या").length - 1;
  const hindi = hits(tokens, HINDI);
  if (marathi === hindi) return null;
  return marathi > hindi ? "mr" : "hi";
}

// Common function words, then letters only that language uses among these.
const LATIN: Record<string, [ReadonlySet<string>, string]> = {
  en: [
    words(
      "the and is are of to in it you that this for was with have not what my your be do can will at on we they he she a an from by or as which their has had were but hello thanks please",
    ),
    "",
  ],
  de: [
    words(
      "der die das und ist nicht ich sie es ein eine zu mit den auf wir du sind auch wie was dem von sich als im des dass für oder werden um wurde einer einem nach aus er über wenn einen bei wird haben können bitte hallo danke ja",
    ),
    "äß",
  ],
  nl: [
    words(
      "de het een en van is niet ik dat je op te zijn met voor wat maar er ook wij jij hoe die om of als ze door aan worden naar deze werd bij kunnen uit wordt dit hij heeft zich hebben dan tot veel dank jullie",
    ),
    "",
  ],
  es: [
    words(
      "el la los las de que y es en un una por con no para se del está muy pero como yo tú usted su al lo son sus más ser también este fue pueden entre sobre hola gracias qué cómo",
    ),
    "ñ¿¡",
  ],
  pt: [
    words(
      "o os as de que e é em um uma não do da para com está você muito mas como eu isso no na mais por ou são dos ao foi das ser sua seu ele pode à também foram tem até olá obrigado obrigada",
    ),
    "ãõ",
  ],
  fr: [
    words(
      "le la les de et est un une des je que pas en du il vous ce ne pour suis sont avec mais nous à dans qui plus sur au par ou été ont se être aux peut c l d qu bonjour merci oui",
    ),
    "œêîûùë",
  ],
  it: [
    words(
      "il la di che e è un una non per sono con del della mi lo gli ho io ma come questo molto i le si più ha da nel al dei delle essere alla anche se cui nella stato dal hanno tra sul ciao grazie",
    ),
    "",
  ],
  tr: [
    words(
      "bir ve bu da de için ne ile çok değil ben mi var olarak sen biz ama gibi daha en veya olan büyük ancak ya her diğer sonra şekilde bazı genellikle kadar tarafından hem ilk iyi yeni merhaba teşekkür",
    ),
    "ğşı",
  ],
  pl: [
    words(
      "i w z na się nie to jest że do co jak ale tak jestem mnie czy jego są od po przez lub dla oraz jako dzień dobry dziękuję",
    ),
    "łąęśźżń",
  ],
  cs: [
    words(
      "a je se na v to že s z do jsem není co jak ale tak jsou jsme mám kde si chci by být pro od po jako nebo také děkuji ahoj",
    ),
    "řěůčšžý",
  ],
  id: [
    words(
      "yang dan di ini itu dengan untuk tidak ada saya dari akan pada anda atau ke dalam lebih mereka adalah orang tahun oleh telah banyak sebagai seperti bisa kamu karena mau saja bahwa memiliki",
    ),
    "",
  ],
  ms: [
    words(
      "yang dan di ini itu dengan untuk tidak ada saya dari akan pada anda atau ke dalam lebih mereka adalah orang tahun oleh telah banyak sebagai seperti boleh awak kerana mahu sahaja bahawa mempunyai daripada ialah tak",
    ),
    "",
  ],
  tl: [
    words(
      "ang ng mga sa na ay hindi ko ako ka mo siya ito at kung lang po namin para may kanyang salamat kumusta",
    ),
    "",
  ],
  sw: [
    words(
      "kwa katika kuwa hii hiyo huyo kama lakini pia sana kutoka kwenye hata kila wake wao wetu yako yangu mimi wewe yeye sisi ninyi wapi lini nini habari asante karibu nataka kununua kituo treni tiketi",
    ),
    "",
  ],
  ha: [
    words(
      "domin kuma amma cikin daga tare wannan wancan yanzu sosai lokaci mutane gida wajen tashar jirgin kasa tikiti godiya sannu barka lafiya nawa yaya zuwa bayan kafin saboda",
    ),
    "ɓɗƙ",
  ],
  yo: [
    words(
      "ninu lati pẹlu ṣugbọn nitori gbogbo ọjọ ọkọ ibudo tikẹti fẹ nibo bawo daadaa ile awọn rẹ wọn yii naa dupẹ pupọ",
    ),
    "ẹọṣ",
  ],
  ig: [
    words(
      "nke maka site nwere chọrọ ịzụta tiketi ọdụ ụgbọ oloko ebee dị anyị unu nkeji daalụ kedu ọma ukwuu",
    ),
    "ịụṅ",
  ],
  zu: [
    words(
      "ukuthi lapho khona kodwa futhi ngoba phezu phansi phakathi kwaso lesi leyo mina wena yena thina nina bona ngifuna ukuthenga ithikithi isiteshi kuphi sawubona ngiyabonga kakhulu kuhle",
    ),
    "",
  ],
  af: [
    words(
      "ek jy ons hulle julle vir baie hierdie daardie stasie treinstasie kaartjie naaste dankie goeiedag asseblief hoekom watter graag koop",
    ),
    "",
  ],
  so: [
    words(
      "waxaan rabaa inaan iibsado tigidh saldhigga tareenka ugu dhow xaggee ayuu yaalaa mahadsanid aad iyo wanaagsan sidee tahay maanta halkan kaas",
    ),
    "",
  ],
};

const MALAY = new Set(["id", "ms"]);

function latin(text: string): string | null {
  if (count(text, /(?![ẹịọụ])[ơưđăạ-ỹ]/gu) >= 2) return "vi";
  // Elisions split too: French "l'homme" counts "l".
  const tokens = tokensOf(text);
  return clearWinner(
    Object.entries(LATIN).map(([code, [known, letters]]) => [
      code,
      hits(tokens, known) + (letters ? countOf(text, letters) : 0),
    ]),
    (a, b) => MALAY.has(a) && MALAY.has(b),
  );
}

/** A Translate language code for the text, or null when it is too short or mixed to tell. */
export function detectTranslateLanguage(text: string): string | null {
  const sample = text
    .slice(0, SAMPLE_CHARS)
    .normalize("NFC")
    .replace(/https?:\/\/\S+/gi, " ")
    .toLowerCase();
  // Marks too: the script patterns below count Devanagari vowel signs and the like.
  const letters = count(sample, /[\p{L}\p{M}]/gu);
  if (letters < 2) return null;
  const dominant = (pattern: RegExp) => count(sample, pattern) >= letters * 0.6;

  for (const [pattern, code] of SINGLE_SCRIPT) {
    if (dominant(pattern)) return code;
  }
  if (dominant(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/gu)) {
    return /[\p{Script=Hiragana}\p{Script=Katakana}]/u.test(sample)
      ? "ja"
      : chinese(sample);
  }
  if (dominant(/\p{Script=Arabic}/gu)) return arabicScript(sample);
  if (dominant(/\p{Script=Cyrillic}/gu)) return cyrillic(sample);
  if (dominant(/\p{Script=Devanagari}/gu)) return devanagari(sample);
  if (dominant(/\p{Script=Latin}/gu)) return latin(sample);
  return null;
}
