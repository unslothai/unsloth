// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type CustomizationIssue,
  DEFAULT_CUSTOMIZATION,
  fontFamilyFromFileName,
  sanitizeCustomization,
  sanitizeCustomizationWithIssues,
  sanitizeFontName,
} from "../src/features/settings/lib/appearance-customization.ts";

type Kind = CustomizationIssue["kind"];
const issue = (path: string, kind: Kind): CustomizationIssue => ({ path, kind });
const GOOD_FONT = "data:font/woff2;base64,QUJD";
const font = (name: string) => ({ name, dataUrl: GOOD_FONT });
const otf = (name: string, data = "QUFB") => ({
  name,
  dataUrl: `data:font/otf;base64,${data}`,
});

// Pinned output of the old implementation. Key order is part of the contract (isDefaultCustomization
// compares JSON), so the comparison is on the serialized string.
const GOLDEN: [string, unknown, string][] = [
  [
    "empty object",
    {},
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"elementOutline":null,"pointerCursors":false,"reduceMotion":"on","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "junk types",
    {
      colors: "red", uiFont: 42, headingFont: ["Inter"], chatWidth: 3, sentAttachments: null, codeFont: {},
      importedFonts: "nope", uiFontSize: "15", codeFontSize: Number.NaN, contrast: "70", composerOutline: true,
      pointerCursors: "yes", reduceMotion: 1, fontSmoothing: 0, codeLineNumbers: "true", codeWordWrap: 1,
      codeLigatures: null, sidebarMenu: 7, sidebarNav: {}, sidebarNavAuto: "projects",
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"elementOutline":null,"pointerCursors":false,"reduceMotion":"on","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":[]}`,
  ],
  [
    "out-of-range sizes and contrast",
    { uiFontSize: 99, codeFontSize: 1, contrast: 140, composerOutline: -20 },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":20,"codeFontSize":10,"contrast":100,"composerOutline":0,"elementOutline":null,"pointerCursors":false,"reduceMotion":"on","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "more than three imported fonts and a duplicate name",
    { importedFonts: [otf("A"), otf("B"), otf("A", "QkJC"), otf("C"), otf("D"), otf("E")] },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[{"name":"A","dataUrl":"data:font/otf;base64,QUFB"},{"name":"B","dataUrl":"data:font/otf;base64,QUFB"},{"name":"C","dataUrl":"data:font/otf;base64,QUFB"}],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"elementOutline":null,"pointerCursors":false,"reduceMotion":"on","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
];

for (const [name, input, json] of GOLDEN) {
  test(`sanitizeCustomization golden: ${name}`, () => {
    assert.equal(JSON.stringify(sanitizeCustomization(input)), json);
  });
}

const ISSUE_CASES: [string, unknown, CustomizationIssue[]][] = [
  ["DEFAULT_CUSTOMIZATION reports nothing", DEFAULT_CUSTOMIZATION, []],
  [
    "missing or null fields are the default and never reported",
    { uiFont: undefined, uiFontSize: null, importedFonts: null, sidebarNavAuto: null, colors: { light: null }, sidebarMenu: [{ id: "chat" }] },
    [],
  ],
  ["uppercase hex is normalized, not lost", { colors: { light: { accent: "#ABCDEF" } } }, []],
  ["invalid accent", { colors: { light: { accent: "red" } } }, [issue("colors.light.accent", "invalid")]],
  ["colors not an object", { colors: "red" }, [issue("colors", "invalid")]],
  ["light mode not an object", { colors: { light: "red" } }, [issue("colors.light", "invalid")]],
  [
    "bad hex and non-string colors",
    { colors: { dark: { background: "#GGGGGG", foreground: 12 } } },
    [issue("colors.dark.background", "invalid"), issue("colors.dark.foreground", "invalid")],
  ],
  ["CSS-breaking font name", { uiFont: 'Evil"; x{}' }, [issue("uiFont", "invalid")]],
  ["blank font name", { codeFont: "   " }, [issue("codeFont", "invalid")]],
  [
    "unknown enum values",
    { chatWidth: "huge", sentAttachments: "grid", reduceMotion: "always" },
    [issue("chatWidth", "invalid"), issue("sentAttachments", "invalid"), issue("reduceMotion", "invalid")],
  ],
  [
    "wrong types for sizes and booleans",
    { uiFontSize: "15", codeFontSize: Number.NaN, contrast: "70", pointerCursors: "yes", fontSmoothing: 0 },
    [
      issue("uiFontSize", "invalid"), issue("codeFontSize", "invalid"), issue("contrast", "invalid"),
      issue("pointerCursors", "invalid"), issue("fontSmoothing", "invalid"),
    ],
  ],
  [
    "wrong container types",
    { importedFonts: "nope", sidebarMenu: 7, sidebarNav: {}, sidebarNavAuto: "projects" },
    [issue("importedFonts", "invalid"), issue("sidebarMenu", "invalid"), issue("sidebarNav", "invalid"), issue("sidebarNavAuto", "invalid")],
  ],
  [
    "wrong sidebar flags",
    { sidebarMenu: [{ id: "api", visible: "no" }], sidebarNav: [{ id: "hub", pinned: "yes" }] },
    [issue("sidebarMenu[0].visible", "invalid"), issue("sidebarNav[0].pinned", "invalid")],
  ],
  [
    "out-of-range numbers are clamped",
    { uiFontSize: 99, codeFontSize: 1, contrast: 140, composerOutline: -20 },
    [issue("uiFontSize", "clamped"), issue("codeFontSize", "clamped"), issue("contrast", "clamped"), issue("composerOutline", "clamped")],
  ],
  ["a fraction that rounds is clamped", { uiFontSize: 14.6 }, [issue("uiFontSize", "clamped")]],
  [
    "imported fonts: bad entries dropped, bad name invalid",
    {
      importedFonts: [
        font("Good"),
        { name: "Svg", dataUrl: "data:image/svg+xml;base64,PHN2Zz4=" },
        { name: "", dataUrl: GOOD_FONT },
        null,
        5,
        { name: "Hi", dataUrl: 12 },
        { name: "Ok;{}", dataUrl: GOOD_FONT },
      ],
    },
    [
      issue("importedFonts[1]", "dropped"), issue("importedFonts[2]", "dropped"), issue("importedFonts[3]", "dropped"),
      issue("importedFonts[4]", "dropped"), issue("importedFonts[5]", "dropped"), issue("importedFonts[6].name", "invalid"),
    ],
  ],
  ["repeated font name", { importedFonts: [font("A"), font("A")] }, [issue("importedFonts[1]", "dropped")]],
  [
    "malformed font data URL",
    { importedFonts: [{ name: "Bad", dataUrl: "data:font/woff2;base64,AB!" }] },
    [issue("importedFonts[0]", "dropped")],
  ],
  [
    "sidebar menu: unknown and repeated ids",
    { sidebarMenu: [{ id: "api", visible: false }, { id: "api" }, { id: "bogus" }] },
    [issue("sidebarMenu[1]", "dropped"), issue("sidebarMenu[2]", "dropped")],
  ],
  [
    "sidebar nav: unknown and repeated ids",
    { sidebarNav: [{ id: "hub", pinned: false }, { id: "nope" }, { id: "hub" }] },
    [issue("sidebarNav[1]", "dropped"), issue("sidebarNav[2]", "dropped")],
  ],
  [
    "Projects rule listed twice, plus a row with no rule",
    { sidebarNavAuto: ["projects", "hub", "projects"] },
    [issue("sidebarNavAuto[1]", "dropped"), issue("sidebarNavAuto[2]", "dropped")],
  ],
];

// Issue order follows evaluation order, which is an implementation detail, so compare sorted serialized issues.
const normalized = (issues: CustomizationIssue[]) =>
  issues.map((entry) => JSON.stringify(entry)).sort();

for (const [name, input, expected] of ISSUE_CASES) {
  test(`issues: ${name}`, () => {
    assert.deepEqual(
      normalized(sanitizeCustomizationWithIssues(input).issues),
      normalized(expected),
    );
  });
}

test("a non-object payload is all defaults with no issues", () => {
  for (const raw of ["abc", 5, true, [], [{ uiFont: "Lato" }]]) {
    assert.deepEqual(sanitizeCustomizationWithIssues(raw), {
      value: DEFAULT_CUSTOMIZATION,
      issues: [],
      omittedIssues: 0,
    });
  }
});

test("the font cap keeps three and reports one dropped issue at the next index", () => {
  const result = sanitizeCustomizationWithIssues({
    importedFonts: ["A", "B", "C", "D", "E"].map(font),
  });
  assert.deepEqual(result.value.importedFonts.map((entry) => entry.name), ["A", "B", "C"]);
  assert.deepEqual(result.issues, [issue("importedFonts[3]", "dropped")]);
});

test("sanitizeCustomization is the value half of sanitizeCustomizationWithIssues", () => {
  for (const raw of [undefined, {}, { uiFontSize: 99, colors: "x" }]) {
    assert.deepEqual(sanitizeCustomization(raw), sanitizeCustomizationWithIssues(raw).value);
  }
});

test("sanitizeFontName keeps the cleaned name or returns null", () => {
  assert.equal(sanitizeFontName(' Fira "Code";  '), "Fira Code");
  assert.equal(sanitizeFontName("a\u0007b"), "ab");
  assert.equal(sanitizeFontName("   "), null);
  assert.equal(sanitizeFontName(7), null);
});

test("fontFamilyFromFileName gives the family a font file registers under", () => {
  assert.equal(fontFamilyFromFileName("Brand_Sans-Bold.woff2"), "Brand Sans Bold");
  assert.equal(fontFamilyFromFileName("My,Font`x.ttf"), "MyFontx");
});

test("issues stop at the cap and the rest are only counted", () => {
  const rows = Array.from({ length: 500 }, () => ({ id: "nope", pinned: true }));
  const { issues, omittedIssues } = sanitizeCustomizationWithIssues({ sidebarNav: rows });
  assert.equal(issues.length, 100);
  assert.equal(omittedIssues, 400);
});
