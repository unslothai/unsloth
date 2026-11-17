// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { detectLanguage } from "../src/features/translate/detect-language.ts";
import { TRANSLATE_LANGUAGES } from "../src/features/translate/languages.ts";

const SAMPLES: Record<string, string> = {
  en: "The weather is lovely today and I am going for a walk with my friends.",
  zh: "今天天气很好，我们一起去公园散步，这对身体很有好处。",
  "zh-Hant": "今天天氣很好，我們一起去公園散步，這對身體很有好處。",
  es: "El tiempo está muy bueno hoy y voy a dar un paseo con mis amigos por el parque.",
  fr: "Il fait très beau aujourd'hui et je vais me promener avec mes amis dans le parc.",
  de: "Das Wetter ist heute sehr schön und ich gehe mit meinen Freunden im Park spazieren.",
  ja: "今日は天気がとても良いので、友達と公園を散歩します。",
  ko: "오늘은 날씨가 아주 좋아서 친구들과 공원에서 산책할 거예요.",
  pt: "O tempo está muito bom hoje e eu vou passear com os meus amigos, mas não é tarde.",
  it: "Oggi il tempo è molto bello e vado a fare una passeggiata con gli amici, che sono già qui.",
  ru: "Сегодня очень хорошая погода, и я иду гулять с друзьями в парк.",
  ar: "الطقس جميل جدا اليوم وسأذهب في نزهة مع أصدقائي في الحديقة.",
  hi: "आज मौसम बहुत अच्छा है और मैं अपने दोस्तों के साथ घूमने जा रहा हूँ।",
  tr: "Bugün hava çok güzel ve arkadaşlarımla parkta yürüyüşe çıkacağım, ama geç değil.",
  nl: "Het weer is vandaag erg mooi en ik ga met mijn vrienden wandelen in het park.",
  pl: "Dzisiaj jest bardzo ładna pogoda i idę na spacer z przyjaciółmi, ale nie jest późno.",
  uk: "Сьогодні дуже гарна погода, і я йду гуляти з друзями в парку біля річки.",
  cs: "Dnes je velmi hezké počasí a jdu se projít s přáteli do parku, protože to je dobré.",
  vi: "Hôm nay thời tiết rất đẹp và tôi sẽ đi dạo với bạn bè của tôi trong công viên.",
  th: "วันนี้อากาศดีมากและฉันจะไปเดินเล่นกับเพื่อนในสวนสาธารณะ",
  id: "Cuaca hari ini sangat bagus dan saya akan berjalan dengan teman karena sudah bisa pergi, itu adalah hal yang baik.",
  ms: "Cuaca hari ini sangat baik dan saya akan berjalan dengan kawan kerana awak boleh pergi, itu ialah perkara yang baik.",
  tl: "Maganda ang panahon ngayon at ako ay maglalakad kasama ang mga kaibigan ko sa parke.",
  he: "מזג האוויר יפה מאוד היום ואני הולך לטייל עם החברים שלי בפארק.",
  fa: "امروز هوا خیلی خوب است و من با دوستانم به پارک می‌روم که قدم بزنیم.",
  ur: "آج موسم بہت اچھا ہے اور میں اپنے دوستوں کے ساتھ پارک میں سیر کرنے جا رہا ہوں۔",
  bn: "আজ আবহাওয়া খুব ভালো এবং আমি আমার বন্ধুদের সাথে পার্কে হাঁটতে যাচ্ছি।",
  ta: "இன்று வானிலை மிகவும் நன்றாக உள்ளது, நான் நண்பர்களுடன் நடக்கப் போகிறேன்.",
  te: "ఈ రోజు వాతావరణం చాలా బాగుంది, నేను స్నేహితులతో నడకకు వెళ్తున్నాను.",
  mr: "आज हवामान खूप छान आहे आणि मी माझ्या मित्रांसोबत फिरायला जात आहे.",
  gu: "આજે હવામાન ખૂબ સરસ છે અને હું મારા મિત્રો સાથે ફરવા જઈ રહ્યો છું.",
  km: "ថ្ងៃនេះអាកាសធាតុល្អណាស់ ហើយខ្ញុំនឹងទៅដើរលេងជាមួយមិត្តភក្តិ។",
  my: "ဒီနေ့ ရာသီဥတု အရမ်းကောင်းပြီး သူငယ်ချင်းတွေနဲ့ လမ်းလျှောက်ထွက်မယ်။",
  yue: "今日天氣好好，我哋一齊去公園行吓，佢哋都嚟咗喇，唔使等。",
  bo: "དེ་རིང་གནམ་གཤིས་ཧ་ཅང་ཡག་པོ་འདུག ང་གྲོགས་པོ་དང་མཉམ་དུ་འགྲོ་གི་ཡིན།",
  kk: "Бүгін ауа райы өте жақсы, мен достарыммен саябаққа серуендеуге барамын, қазір ерте.",
  mn: "Өнөөдөр цаг агаар маш сайхан байна, би найзуудтайгаа цэцэрлэгт хүрээлэнд алхана гэж бодож байна.",
  ug: "بۈگۈن ھاۋا بەك ياخشى، مەن دوستلىرىم بىلەن باغچىغا سەيلە قىلغىلى بارىمەن.",
};

const LATIN = new Set([
  "en",
  "es",
  "fr",
  "de",
  "pt",
  "it",
  "tr",
  "nl",
  "pl",
  "cs",
  "vi",
  "id",
  "ms",
  "tl",
]);

test("every language Translate offers has a sample here", () => {
  assert.deepEqual(
    TRANSLATE_LANGUAGES.map((language) => language.code).sort(),
    Object.keys(SAMPLES).sort(),
  );
});

for (const [code, sample] of Object.entries(SAMPLES)) {
  if (LATIN.has(code)) {
    test(`a sentence in ${code} shares the Latin alphabet, so it is not guessed`, () => {
      assert.equal(detectLanguage(sample), null);
    });
  } else {
    test(`a sentence in ${code} is recognised`, () => {
      assert.equal(detectLanguage(sample), code);
    });
  }
}

const PHRASES: [string | null, string][] = [
  ["hi", "आप कैसे हैं?"],
  ["mr", "तुम्ही कसे आहात?"],
  ["ar", "كيف حالك؟"],
  ["fa", "حال شما چطور است؟"],
  ["ur", "آپ کیسے ہیں؟"],
  ["uk", "Як у вас справи сьогодні?"],
  ["ja", "ありがとう"],
  ["ko", "안녕하세요"],
  ["th", "สวัสดี"],
  ["he", "שלום"],
  ["bn", "ধন্যবাদ"],
  ["zh", "谢谢你"],
  ["zh-Hant", "謝謝你"],
  // The same word in two languages, or too little to tell: nothing is better than a wrong name.
  [null, "नमस्कार"],
  [null, "سلام"],
  [null, "Привет"],
  [null, "Добрий день"],
  [null, "Как дела?"],
  [null, "Hello"],
  [null, "Bom dia"],
  [null, "Please send me the report by Friday afternoon."],
];

for (const [code, phrase] of PHRASES) {
  test(`"${phrase}" gives ${code ?? "no guess"}`, () => {
    assert.equal(detectLanguage(phrase), code);
  });
}

test("too little text, or none, gives no guess", () => {
  for (const text of ["", "   ", "ok", "12345 !!!"]) {
    assert.equal(detectLanguage(text), null);
  }
});
