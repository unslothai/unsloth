// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { detectTranslateLanguage } from "../src/features/translate/detect-language.ts";
import { TRANSLATE_LANGUAGES } from "../src/features/translate/languages.ts";

const SENTENCES: Record<string, string[]> = {
  en: [
    "Where is the nearest train station? I want to buy a ticket.",
    "How are you?",
  ],
  zh: ["这个周末我们去公园散步，你觉得怎么样？", "我没有时间。"],
  "zh-Hant": ["這個週末我們去公園散步，你覺得怎麼樣？", "我沒有時間。"],
  es: [
    "¿Dónde está la estación de tren más cercana? Quiero comprar un billete.",
    "Hola, ¿cómo estás?",
  ],
  fr: [
    "Où est la gare la plus proche ? Je voudrais acheter un billet.",
    "Je ne sais pas, merci.",
  ],
  de: [
    "Wo ist der nächste Bahnhof? Ich möchte eine Fahrkarte kaufen.",
    "Danke, das ist gut.",
  ],
  ja: ["一番近い駅はどこですか？切符を買いたいです。", "ありがとう"],
  ko: ["가장 가까운 기차역이 어디예요? 표를 사고 싶어요.", "감사합니다"],
  pt: [
    "Onde fica a estação de trem mais próxima? Eu quero comprar uma passagem.",
    "Não sei, obrigado.",
  ],
  it: [
    "Dov'è la stazione più vicina? Io vorrei comprare un biglietto per la città di Roma.",
    "Ciao, grazie mille.",
  ],
  ru: [
    "Где находится ближайший вокзал? Я хочу купить билет.",
    "Спасибо, всё хорошо.",
  ],
  ar: [
    "أين تقع أقرب محطة قطار؟ أريد أن أشتري تذكرة.",
    "شكرا جزيلا لك على المساعدة",
  ],
  hi: [
    "सबसे नज़दीकी रेलवे स्टेशन कहाँ है? मुझे टिकट खरीदना है।",
    "यह बहुत अच्छा है और मैं खुश हूँ।",
  ],
  tr: [
    "En yakın tren istasyonu nerede? Bir bilet almak istiyorum.",
    "Çok teşekkür ederim.",
  ],
  nl: [
    "Waar is het dichtstbijzijnde treinstation? Ik wil een kaartje kopen.",
    "Dank je wel, dat is goed.",
  ],
  pl: [
    "Gdzie jest najbliższa stacja kolejowa? Chcę kupić bilet.",
    "Dziękuję bardzo, to jest dobre.",
  ],
  uk: [
    "Де знаходиться найближчий вокзал? Я хочу купити квиток.",
    "Дякую, все добре.",
  ],
  cs: [
    "Kde je nejbližší vlakové nádraží? Chci si koupit jízdenku.",
    "Děkuji, to je dobré.",
  ],
  vi: ["Ga tàu gần nhất ở đâu? Tôi muốn mua một vé.", "Cảm ơn bạn rất nhiều."],
  th: ["สถานีรถไฟที่ใกล้ที่สุดอยู่ที่ไหน ฉันอยากซื้อตั๋ว", "ขอบคุณ"],
  id: [
    "Di mana stasiun kereta terdekat? Saya mau membeli tiket karena saya harus pergi.",
    "Saya tidak bisa datang sekarang.",
  ],
  ms: [
    "Di mana stesen kereta api yang terdekat? Saya mahu membeli tiket kerana saya perlu pergi.",
    "Saya tak boleh datang sekarang.",
  ],
  tl: [
    "Nasaan ang pinakamalapit na istasyon ng tren? Gusto kong bumili ng tiket.",
    "Salamat po sa tulong mo.",
  ],
  he: ["איפה תחנת הרכבת הקרובה? אני רוצה לקנות כרטיס.", "תודה רבה"],
  fa: [
    "نزدیک‌ترین ایستگاه قطار کجاست؟ من می‌خواهم یک بلیط بخرم.",
    "خیلی ممنون از کمک شما",
  ],
  ur: [
    "سب سے قریبی ریلوے اسٹیشن کہاں ہے؟ میں ٹکٹ خریدنا چاہتا ہوں۔",
    "آپ کا بہت شکریہ",
  ],
  bn: ["সবচেয়ে কাছের রেলস্টেশন কোথায়? আমি একটি টিকিট কিনতে চাই।", "ধন্যবাদ"],
  ta: ["அருகிலுள்ள ரயில் நிலையம் எங்கே? நான் ஒரு டிக்கெட் வாங்க வேண்டும்.", "நன்றி"],
  te: ["దగ్గరలోని రైల్వే స్టేషన్ ఎక్కడ ఉంది? నేను టికెట్ కొనాలి.", "ధన్యవాదాలు"],
  mr: [
    "सर्वात जवळचे रेल्वे स्टेशन कुठे आहे? मला तिकीट विकत घ्यायचे आहे.",
    "हे खूप छान आहे आणि मी आनंदी आहे.",
  ],
  gu: ["સૌથી નજીકનું રેલવે સ્ટેશન ક્યાં છે? મારે ટિકિટ ખરીદવી છે.", "આભાર"],
  km: ["តើស្ថានីយ៍រថភ្លើងដែលនៅជិតបំផុតនៅឯណា? ខ្ញុំចង់ទិញសំបុត្រ។", "អរគុណ"],
  my: ["အနီးဆုံးရထားဘူတာက ဘယ်မှာလဲ။ လက်မှတ်ဝယ်ချင်ပါတယ်။", "ကျေးဇူးတင်ပါတယ်"],
  yue: ["佢哋今日唔嚟啦，你知唔知點解？", "我食咗飯喇，你呢？"],
  bo: ["རླངས་འཁོར་འབབ་ཚུགས་ག་པར་ཡོད། ང་ལ་ལག་འཁྱེར་ཉོ་དགོས།", "ཐུགས་རྗེ་ཆེ།"],
  kk: [
    "Ең жақын теміржол вокзалы қайда? Мен билет сатып алғым келеді.",
    "Рақмет, бәрі жақсы.",
    "Бүгін күн жылы",
  ],
  mn: [
    "Хамгийн ойрын галт тэрэгний буудал хаана байдаг вэ? Би тасалбар авмаар байна.",
    "Баярлалаа, бүх зүйл сайн байна.",
  ],
  ug: [
    "ئەڭ يېقىن پويىز ئىستانسىسى قەيەردە؟ مەن بىلەت سېتىۋالماقچىمەن.",
    "كۆپ رەھمەت سىزگە",
  ],
  am: [
    "በጣም የቅርቡ የባቡር ጣቢያ የት ነው? ትኬት መግዛት እፈልጋለሁ።",
    "አመሰግናለሁ",
  ],
  sw: [
    "Kituo cha treni cha karibu kiko wapi? Mimi nataka kununua tiketi kwa safari yangu.",
    "Habari yako, asante sana.",
  ],
  ha: [
    "Ina tashar jirgin ƙasa mafi kusa? Ina so in sayi tikiti domin tafiya zuwa gida.",
    "Sannu, barka da zuwa, mun gode sosai.",
  ],
  yo: [
    "Nibo ni ibudo ọkọ oju-irin ti o sunmọ julọ wa? Mo fẹ ra tikẹti kan lati lọ si ile.",
    "Ẹ ku ọjọ meta, a dupẹ pupọ.",
  ],
  ig: [
    "Ebee ka ọdụ ụgbọ oloko kacha nso dị? Achọrọ m ịzụta tiketi maka njem anyị.",
    "Kedu ka ị mere? Daalụ nke ukwuu.",
  ],
  zu: [
    "Kuphi isiteshi sesitimela esiseduze? Mina ngifuna ukuthenga ithikithi futhi ngihambe.",
    "Sawubona, ngiyabonga kakhulu.",
  ],
  af: [
    "Waar is die naaste treinstasie? Ek wil graag hierdie kaartjie vir ons koop.",
    "Goeiedag, baie dankie vir julle hulp.",
  ],
  so: [
    "Xaggee ayuu ku yaalaa saldhigga tareenka ugu dhow? Waxaan rabaa inaan iibsado tigidh.",
    "Aad iyo aad ayaad u mahadsanid, sidee tahay maanta?",
  ],
};

test("every Translate language is recognised from a sentence and a short phrase", () => {
  assert.deepEqual(
    Object.keys(SENTENCES).sort(),
    TRANSLATE_LANGUAGES.map((language) => language.code).sort(),
  );
  const wrong: string[] = [];
  for (const [code, sentences] of Object.entries(SENTENCES)) {
    for (const sentence of sentences) {
      const detected = detectTranslateLanguage(sentence);
      if (detected !== code) wrong.push(`${code} -> ${detected}: ${sentence}`);
    }
  }
  assert.deepEqual(wrong, []);
});

test("nothing is claimed for text too short or too mixed to tell", () => {
  for (const text of ["", "  ", "42", "a", "OK", "Hi", "https://unsloth.ai"])
    assert.equal(detectTranslateLanguage(text), null, text);
  assert.equal(detectTranslateLanguage("Merhaba dunya"), null);
  // No Simplified or Traditional only character: either is a guess.
  assert.equal(detectTranslateLanguage("你好"), null);
  // Indonesian and Malay share these words.
  assert.equal(detectTranslateLanguage("Saya ada di rumah dengan dia."), null);
});

test("decomposed accents and uppercase links read like any other text", () => {
  assert.equal(
    detectTranslateLanguage("Cảm ơn bạn rất nhiều.".normalize("NFD")),
    "vi",
  );
  assert.equal(
    detectTranslateLanguage(
      "HTTPS://EXAMPLE.COM/the-and-is-of Ich habe keine Zeit, danke.",
    ),
    "de",
  );
});

test("only the start of a long text is read", () => {
  const long = `${"Ich habe keine Zeit. ".repeat(200)}${"Hello there, how are you? ".repeat(400)}`;
  assert.equal(detectTranslateLanguage(long), "de");
});
