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
const words = (list: string): ReadonlySet<string> =>
  new Set(list.normalize("NFC").split(" "));

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

// Dutch and Afrikaans share these, so only the words the other lacks tell them apart.
const DUTCH_AFRIKAANS =
  "die en van is in te het dat om op met wat of aan ons kan maar ook nog al hier daar uit na tot was gaan kom toe dit waar hoe alle geen niks iets iemand mens goed moet word wanneer omdat haar jou onder net";

// Common function words, letters only that language uses among these, then a word shape for languages that glue words together.
const LATIN: Record<string, [ReadonlySet<string>, string, RegExp?]> = {
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
      `de een niet ik je zijn voor er wij jij als ze door worden naar deze werd bij kunnen wordt hij heeft zich hebben dan veel dank jullie ${DUTCH_AFRIKAANS}`,
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
  hmn: [
    words(
      "kuv koj nws peb nej lawv yog muaj tsis tau yuav xav nyob mus ua hais pom paub kawm thiab rau hauv txog lub tus cov tias vim zoo tsaug ntau lus hmoob neeg hnub tsev chaw qhov twg cas ntawm ntawd ib kev nrog puas thaum yeej kom tom qab ces yam txhua lawm sib tshiab",
    ),
    "",
  ],
  sw: [
    words(
      "na ya wa kwa ni za katika kuwa cha vya hii hiyo huo hizo pia lakini kama au ili kwamba sana tu bado kila wote yote zote mimi wewe yeye sisi ninyi wao yangu yako yake yetu yenu yao wangu wako wake zao langu lako lake nini gani wapi lini vipi nani je hapa kutoka hadi mpaka baada kabla kuna hakuna ndiyo hapana habari asante karibu sasa leo kesho jana mtu watu siku zaidi bila pamoja juu chini ndani nje kati hivyo yoyote",
    ),
    "",
    // Infinitives, then subject and tense prefixes.
    /^ku[a-z]{3,}a$|^(?:ni|tu|wa|m|ki|vi)(?:na|li|ta|me)[a-z]{2,}a$/,
  ],
  ha: [
    words(
      "da kuma cikin daga wannan waɗannan amma domin saboda shi ita mu ku sun za ne ce duk kowa yi ko ga wani wata wasu akwai babu haka kamar sai tun har bayan kafin yanzu gobe jiya yau mutane mutum abin inda lokacin zuwa game kan wajen sannu nagode godiya barka lafiya yaya suna yana tana muna kuna sosai kawai dai ba ya na a ta ina zai aka ake yan kowane kowace",
    ),
    "ɓɗƙƴ",
  ],
  yo: [
    words(
      "ni ti si àti ó wọ́n mo kí ní láti fún pẹ̀lú gbogbo ṣùgbọ́n nítorí tàbí jẹ́ kò yìí náà rẹ̀ mi wa wọn sì bí ń yóò máa kan ọ̀kan ènìyàn ẹni lórí sí inú èyí kankan ẹ dupẹ pupọ ọjọ",
    ),
    "ẹṣ",
  ],
  ig: [
    words(
      "na nke ya ka ha m ị ọ anyị unu ndị niile nile bụ dị nwere maka site ma mana ebe mgbe gị onye ihe bụla ga ike ikike ụfọdụ ahụ kwa nwee",
    ),
    "ịụṅ",
  ],
  zu: [
    words(
      "futhi ukuthi kodwa ngoba uma noma kakhulu lapho khona kanye yonke wonke bonke zonke konke lokho lokhu leli lesi labo lezi kule kulo kuze ngakho nje kahle manje namuhla yebo mina wena yena thina nina bona yini kanjani unjani ngubani sawubona ngiyabonga abantu umuntu izinto ukuba ngokuthi ngaphandle phakathi ngaphansi ngaphezu ukuze kwa akukho lami lakho lakhe wami yami sami bami zami",
    ),
    "",
    // Noun-class and verb prefixes.
    /^(?:ngi|nge|ngo|uku|oku|kwe|ezi|isi|izi|umu|ubu)[a-z]{3,}$/,
  ],
  af: [
    words(
      `${DUTCH_AFRIKAANS} nie ek jy hy sy hulle julle vir baie sal mekaar hul sê hoekom mense dankie asseblief hom n sodat teen elkeen kry dis my`,
    ),
    "ê",
  ],
  so: [
    words(
      "iyo ee ka ku u la ay waa oo ah ayaa wax waxaa waxay wuxuu kale kasta ama haddii laakiin sida markii inuu inay inaan aan uu loo lagu laga si dad dadka qof qofka kuwa mid hal aad badan wanaagsan maanta berri halkan halkaas sidee maxaa xagee haa maya nabad iska isku ugu dhammaan ahaan leh jira mahadsanid waxaan waxaad ayuu ayay ayaan iyada isaga anigu adiga waan",
    ),
    "",
    // Long vowels with x, q or dh.
    /^(?=[a-z]*(?:aa|ee|ii|oo|uu))[a-z]*(?:x|q|dh)[a-z]*$/,
  ],
};

const CLOSE = [new Set(["id", "ms"]), new Set(["nl", "af"])];

function latin(text: string): string | null {
  // Yoruba and Igbo write ẹ ị ọ ụ too, but none of Vietnamese's other letters nor its closed syllables.
  const vietnamese =
    count(text, /(?![ẹịọụ])[ơưđăạ-ỹ]/gu) +
    count(text, /[ẹịọụ](?:ch|c|ng|nh|p|t)(?![\p{L}\p{M}])/gu);
  if (vietnamese > 0 && vietnamese + countOf(text, "ẹịọụ") >= 2) return "vi";
  // Elisions split too: French "l'homme" counts "l".
  const tokens = tokensOf(text);
  return clearWinner(
    Object.entries(LATIN).map(([code, [known, letters, shape]]) => [
      code,
      hits(tokens, known) +
        (letters ? countOf(text, letters) : 0) +
        (shape ? tokens.filter((token) => shape.test(token)).length : 0),
    ]),
    (a, b) => CLOSE.some((pair) => pair.has(a) && pair.has(b)),
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
