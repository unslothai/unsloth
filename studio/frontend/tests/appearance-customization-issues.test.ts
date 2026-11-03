// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type CustomizationIssue,
  fontFamilyFromFileName,
  sanitizeCustomizationWithIssues,
  sanitizeFontName,
} from "../src/features/settings/lib/appearance-customization.ts";
import * as store from "../src/features/settings/stores/appearance-custom-store.ts";
import { readSrc } from "./helpers/kit.ts";

const { DEFAULT_CUSTOMIZATION, sanitizeCustomization } = store;

// Raw payloads and the exact JSON sanitizeCustomization produced for each. Key order is part of the
// contract (isDefaultCustomization compares JSON), so the comparison is on the serialized string.
const GOLDEN_CASES: [string, unknown, string][] = [
  [
    "undefined",
    undefined,
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "empty object",
    {},
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "junk types",
    {
      colors: "red",
      uiFont: 42,
      headingFont: ["Inter"],
      chatWidth: 3,
      sentAttachments: null,
      codeFont: {},
      importedFonts: "nope",
      uiFontSize: "15",
      codeFontSize: Number.NaN,
      contrast: "70",
      composerOutline: true,
      pointerCursors: "yes",
      reduceMotion: 1,
      fontSmoothing: 0,
      codeLineNumbers: "true",
      codeWordWrap: 1,
      codeLigatures: null,
      sidebarMenu: 7,
      sidebarNav: {},
      sidebarNavAuto: "projects",
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":[]}`,
  ],
  [
    "out-of-range sizes and contrast",
    {
      uiFontSize: 99,
      codeFontSize: 1,
      contrast: 140,
      composerOutline: -20,
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":20,"codeFontSize":10,"contrast":100,"composerOutline":0,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "fractional sizes round",
    {
      uiFontSize: 14.6,
      codeFontSize: 19.4,
      contrast: 49.5,
      composerOutline: 33.5,
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":15,"codeFontSize":19,"contrast":50,"composerOutline":34,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "bad hex colors",
    {
      colors: {
        light: {
          accent: "red",
          background: "#GGGGGG",
          foreground: "#abc",
        },
        dark: {
          accent: 123,
          background: "#12345678",
          foreground: null,
        },
      },
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "uppercase hex is lowercased",
    {
      colors: {
        light: {
          accent: "#ABCDEF",
          background: "#FFFFFF",
          foreground: "#0A0B0C",
        },
      },
    },
    `{"colors":{"light":{"accent":"#abcdef","background":"#ffffff","foreground":"#0a0b0c"},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "font names with CSS-breaking characters",
    {
      uiFont: 'Evil"; x{}',
      headingFont: "Inter\\/(x)<y>",
      chatFont: "  Fira\tCode  ",
      codeFont: "a,b`c",
      importedFonts: [],
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":"Evil x","headingFont":"Interxy","chatFont":"FiraCode","chatWidth":"standard","sentAttachments":"auto","codeFont":"abc","importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "imported fonts: bad entries dropped",
    {
      importedFonts: [
        {
          name: "Good",
          dataUrl: "data:font/woff2;base64,QUJD",
        },
        {
          name: "Svg",
          dataUrl: "data:image/svg+xml;base64,PHN2Zz4=",
        },
        {
          name: "",
          dataUrl: "data:font/ttf;base64,QUJD",
        },
        null,
        5,
        {
          name: "Hi",
          dataUrl: 12,
        },
        {
          name: "Ok;{}",
          dataUrl: "data:application/x-font-ttf;base64,QUJD",
        },
      ],
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[{"name":"Good","dataUrl":"data:font/woff2;base64,QUJD"},{"name":"Ok","dataUrl":"data:application/x-font-ttf;base64,QUJD"}],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "more than three imported fonts and a duplicate name",
    {
      importedFonts: [
        {
          name: "A",
          dataUrl: "data:font/otf;base64,QUFB",
        },
        {
          name: "B",
          dataUrl: "data:font/otf;base64,QUFB",
        },
        {
          name: "A",
          dataUrl: "data:font/otf;base64,QkJC",
        },
        {
          name: "C",
          dataUrl: "data:font/otf;base64,QUFB",
        },
        {
          name: "D",
          dataUrl: "data:font/otf;base64,QUFB",
        },
        {
          name: "E",
          dataUrl: "data:font/otf;base64,QUFB",
        },
      ],
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[{"name":"A","dataUrl":"data:font/otf;base64,QUFB"},{"name":"B","dataUrl":"data:font/otf;base64,QUFB"},{"name":"C","dataUrl":"data:font/otf;base64,QUFB"}],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "unknown and duplicate sidebar ids",
    {
      sidebarMenu: [
        {
          id: "api",
          visible: false,
        },
        {
          id: "api",
          visible: true,
        },
        {
          id: "bogus",
          visible: true,
        },
        {
          id: "chat",
        },
        null,
      ],
      sidebarNav: [
        {
          id: "hub",
          pinned: false,
        },
        {
          id: "hub",
          pinned: true,
        },
        {
          id: "nope",
          pinned: true,
        },
        {
          id: "tasks",
        },
      ],
      sidebarNavAuto: ["projects", "hub", "projects", "nope"],
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":false},{"id":"chat","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":false},{"id":"tasks","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "bad enums",
    {
      chatWidth: "huge",
      sentAttachments: "grid",
      reduceMotion: "always",
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "shipped sidebar layout with no sidebarNavAuto",
    {
      sidebarNav: [
        {
          id: "projects",
          pinned: true,
        },
        {
          id: "hub",
          pinned: true,
        },
        {
          id: "images",
          pinned: true,
        },
        {
          id: "train",
          pinned: true,
        },
        {
          id: "video",
          pinned: false,
        },
        {
          id: "recipes",
          pinned: false,
        },
        {
          id: "export",
          pinned: false,
        },
      ],
    },
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"projects","pinned":true},{"id":"hub","pinned":true},{"id":"images","pinned":true},{"id":"train","pinned":true},{"id":"video","pinned":false},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"library","pinned":true},{"id":"audio","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
  [
    "a full valid customization",
    {
      colors: {
        light: {
          accent: "#3366ff",
          background: "#fafafa",
          foreground: "#111111",
        },
        dark: {
          accent: "#ff8800",
          background: null,
          foreground: "#eeeeee",
        },
      },
      uiFont: "Lato",
      headingFont: "Georgia",
      chatFont: "Merriweather",
      chatWidth: "wide",
      sentAttachments: "chips",
      codeFont: "Cascadia Code",
      importedFonts: [
        {
          name: "Brand",
          dataUrl: "data:font/woff2;base64,QUJDRA==",
        },
      ],
      uiFontSize: 17,
      codeFontSize: 13,
      contrast: 72,
      composerOutline: 40,
      pointerCursors: true,
      reduceMotion: "off",
      fontSmoothing: false,
      codeLineNumbers: true,
      codeWordWrap: true,
      codeLigatures: false,
      sidebarMenu: [
        {
          id: "chat",
          visible: true,
        },
        {
          id: "api",
          visible: false,
        },
      ],
      sidebarNav: [
        {
          id: "tasks",
          pinned: true,
        },
        {
          id: "hub",
          pinned: false,
        },
      ],
      sidebarNavAuto: [],
    },
    `{"colors":{"light":{"accent":"#3366ff","background":"#fafafa","foreground":"#111111"},"dark":{"accent":"#ff8800","background":null,"foreground":"#eeeeee"}},"uiFont":"Lato","headingFont":"Georgia","chatFont":"Merriweather","chatWidth":"wide","sentAttachments":"chips","codeFont":"Cascadia Code","importedFonts":[{"name":"Brand","dataUrl":"data:font/woff2;base64,QUJDRA=="}],"uiFontSize":17,"codeFontSize":13,"contrast":72,"composerOutline":40,"pointerCursors":true,"reduceMotion":"off","fontSmoothing":false,"codeLineNumbers":true,"codeWordWrap":true,"codeLigatures":false,"sidebarMenu":[{"id":"chat","visible":true},{"id":"api","visible":false},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"tasks","pinned":true},{"id":"hub","pinned":false},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false}],"sidebarNavAuto":[]}`,
  ],
  [
    "DEFAULT_CUSTOMIZATION used as input",
    DEFAULT_CUSTOMIZATION,
    `{"colors":{"light":{"accent":null,"background":null,"foreground":null},"dark":{"accent":null,"background":null,"foreground":null}},"uiFont":null,"headingFont":null,"chatFont":null,"chatWidth":"standard","sentAttachments":"auto","codeFont":null,"importedFonts":[],"uiFontSize":null,"codeFontSize":null,"contrast":50,"composerOutline":null,"pointerCursors":false,"reduceMotion":"system","fontSmoothing":true,"codeLineNumbers":false,"codeWordWrap":false,"codeLigatures":true,"sidebarMenu":[{"id":"api","visible":true},{"id":"darkMode","visible":true},{"id":"guidedTour","visible":true},{"id":"profile","visible":false},{"id":"appearance","visible":true},{"id":"resources","visible":false},{"id":"chat","visible":false},{"id":"connections","visible":false}],"sidebarNav":[{"id":"hub","pinned":true},{"id":"projects","pinned":true},{"id":"library","pinned":true},{"id":"images","pinned":true},{"id":"video","pinned":false},{"id":"audio","pinned":false},{"id":"train","pinned":true},{"id":"recipes","pinned":false},{"id":"export","pinned":false},{"id":"api","pinned":false},{"id":"tasks","pinned":false}],"sidebarNavAuto":["projects"]}`,
  ],
];

for (const [name, input, json] of GOLDEN_CASES) {
  test(`sanitizeCustomization golden: ${name}`, () => {
    assert.equal(JSON.stringify(sanitizeCustomization(input)), json);
  });
}

const GOOD_FONT = "data:font/woff2;base64,QUJD";

const VALID_CUSTOMIZATION = {
  colors: {
    light: { accent: "#3366ff", background: null, foreground: "#111111" },
    dark: { accent: null, background: "#101010", foreground: null },
  },
  uiFont: "Lato",
  chatWidth: "wide",
  importedFonts: [
    { name: "Brand", dataUrl: "data:font/woff2;base64,QUJDRA==" },
  ],
  uiFontSize: 17,
  codeFontSize: 13,
  contrast: 72,
  composerOutline: 40,
  reduceMotion: "off",
  sidebarMenu: [{ id: "chat", visible: true }],
  sidebarNav: [{ id: "tasks", pinned: true }],
  sidebarNavAuto: [],
};

function issuesOf(value: unknown): CustomizationIssue[] {
  return sanitizeCustomizationWithIssues(value).issues;
}

/** Issue order follows evaluation order, which is an implementation detail; these tests compare sets. */
function sorted(issues: CustomizationIssue[]): CustomizationIssue[] {
  return [...issues].sort((a, b) =>
    a.path < b.path ? -1 : a.path > b.path ? 1 : 0,
  );
}

test("the default and a fully valid customization report nothing", () => {
  assert.deepStrictEqual(issuesOf(DEFAULT_CUSTOMIZATION), []);
  assert.deepStrictEqual(issuesOf(VALID_CUSTOMIZATION), []);
  // Uppercase hex is normalized, which is not a loss.
  assert.deepStrictEqual(
    issuesOf({ colors: { light: { accent: "#ABCDEF" } } }),
    [],
  );
});

test("a missing or null field is the default and is never reported", () => {
  assert.deepStrictEqual(issuesOf({}), []);
  assert.deepStrictEqual(
    issuesOf({
      uiFont: undefined,
      uiFontSize: null,
      importedFonts: null,
      sidebarNavAuto: null,
      colors: { light: null },
    }),
    [],
  );
  assert.deepStrictEqual(issuesOf({ sidebarMenu: [{ id: "chat" }] }), []);
});

test("a non-object payload is all defaults with no issues", () => {
  for (const raw of ["abc", 5, true, [], [{ uiFont: "Lato" }]]) {
    const result = sanitizeCustomizationWithIssues(raw);
    assert.deepStrictEqual(result.value, DEFAULT_CUSTOMIZATION);
    assert.deepStrictEqual(result.issues, []);
  }
});

test("invalid: a wrong type or unknown value, reported at its path", () => {
  assert.deepStrictEqual(issuesOf({ colors: { light: { accent: "red" } } }), [
    { path: "colors.light.accent", kind: "invalid" },
  ]);
  assert.deepStrictEqual(
    sorted(
      issuesOf({ colors: { dark: { background: "#GGGGGG", foreground: 12 } } }),
    ),
    [
      { path: "colors.dark.background", kind: "invalid" },
      { path: "colors.dark.foreground", kind: "invalid" },
    ],
  );
  assert.deepStrictEqual(issuesOf({ colors: "red" }), [
    { path: "colors", kind: "invalid" },
  ]);
  assert.deepStrictEqual(issuesOf({ colors: { light: "red" } }), [
    { path: "colors.light", kind: "invalid" },
  ]);
  assert.deepStrictEqual(issuesOf({ uiFont: 'Evil"; x{}' }), [
    { path: "uiFont", kind: "invalid" },
  ]);
  assert.deepStrictEqual(issuesOf({ codeFont: "   " }), [
    { path: "codeFont", kind: "invalid" },
  ]);
  assert.deepStrictEqual(
    sorted(
      issuesOf({
        chatWidth: "huge",
        sentAttachments: "grid",
        reduceMotion: "always",
      }),
    ),
    [
      { path: "chatWidth", kind: "invalid" },
      { path: "reduceMotion", kind: "invalid" },
      { path: "sentAttachments", kind: "invalid" },
    ],
  );
  assert.deepStrictEqual(
    sorted(
      issuesOf({ uiFontSize: "15", codeFontSize: Number.NaN, contrast: "70" }),
    ),
    [
      { path: "codeFontSize", kind: "invalid" },
      { path: "contrast", kind: "invalid" },
      { path: "uiFontSize", kind: "invalid" },
    ],
  );
  assert.deepStrictEqual(
    sorted(issuesOf({ pointerCursors: "yes", fontSmoothing: 0 })),
    [
      { path: "fontSmoothing", kind: "invalid" },
      { path: "pointerCursors", kind: "invalid" },
    ],
  );
  assert.deepStrictEqual(
    sorted(issuesOf({ sidebarMenu: [{ id: "api", visible: "no" }] })),
    [{ path: "sidebarMenu[0].visible", kind: "invalid" }],
  );
  assert.deepStrictEqual(
    sorted(issuesOf({ sidebarNav: [{ id: "hub", pinned: "yes" }] })),
    [{ path: "sidebarNav[0].pinned", kind: "invalid" }],
  );
  assert.deepStrictEqual(
    sorted(
      issuesOf({
        importedFonts: "nope",
        sidebarMenu: 7,
        sidebarNav: {},
        sidebarNavAuto: "projects",
      }),
    ),
    [
      { path: "importedFonts", kind: "invalid" },
      { path: "sidebarMenu", kind: "invalid" },
      { path: "sidebarNav", kind: "invalid" },
      { path: "sidebarNavAuto", kind: "invalid" },
    ],
  );
});

test("clamped: a number outside its range, or rounded to another value", () => {
  const cases: [
    Record<string, unknown>,
    keyof typeof DEFAULT_CUSTOMIZATION,
    number,
  ][] = [
    [{ uiFontSize: 99 }, "uiFontSize", 20],
    [{ codeFontSize: 1 }, "codeFontSize", 10],
    [{ contrast: 140 }, "contrast", 100],
    [{ composerOutline: -20 }, "composerOutline", 0],
    [{ uiFontSize: 14.6 }, "uiFontSize", 15],
  ];
  for (const [input, path, expected] of cases) {
    const result = sanitizeCustomizationWithIssues(input);
    assert.deepStrictEqual(result.issues, [{ path, kind: "clamped" }], path);
    assert.equal(result.value[path], expected, path);
  }
  assert.deepStrictEqual(
    issuesOf({ uiFontSize: 15, contrast: 50, composerOutline: 0 }),
    [],
  );
});

test("dropped: an array entry that was removed, by index", () => {
  const result = sanitizeCustomizationWithIssues({
    importedFonts: [
      { name: "Good", dataUrl: GOOD_FONT },
      { name: "Svg", dataUrl: "data:image/svg+xml;base64,PHN2Zz4=" },
      { name: "", dataUrl: GOOD_FONT },
      null,
      5,
      { name: "Hi", dataUrl: 12 },
      { name: "Ok;{}", dataUrl: GOOD_FONT },
    ],
  });
  assert.deepStrictEqual(
    result.value.importedFonts.map((font) => font.name),
    ["Good", "Ok"],
  );
  assert.deepStrictEqual(sorted(result.issues), [
    { path: "importedFonts[1]", kind: "dropped" },
    { path: "importedFonts[2]", kind: "dropped" },
    { path: "importedFonts[3]", kind: "dropped" },
    { path: "importedFonts[4]", kind: "dropped" },
    { path: "importedFonts[5]", kind: "dropped" },
    { path: "importedFonts[6].name", kind: "invalid" },
  ]);
});

test("dropped: fonts over the cap, repeated names, and malformed data URLs", () => {
  const font = (name: string) => ({ name, dataUrl: GOOD_FONT });
  assert.deepStrictEqual(
    issuesOf({ importedFonts: ["A", "B", "C", "D", "E"].map(font) }),
    [
      { path: "importedFonts[3]", kind: "dropped" },
      { path: "importedFonts[4]", kind: "dropped" },
    ],
  );
  assert.deepStrictEqual(issuesOf({ importedFonts: [font("A"), font("A")] }), [
    { path: "importedFonts[1]", kind: "dropped" },
  ]);
  assert.deepStrictEqual(
    issuesOf({
      importedFonts: [{ name: "Bad", dataUrl: "data:font/woff2;base64,AB!" }],
    }),
    [{ path: "importedFonts[0]", kind: "dropped" }],
  );
});

test("dropped: sidebar rows with unknown or repeated ids, and Projects without its rule", () => {
  assert.deepStrictEqual(
    sorted(
      issuesOf({
        sidebarMenu: [
          { id: "api", visible: false },
          { id: "api" },
          { id: "bogus" },
        ],
      }),
    ),
    [
      { path: "sidebarMenu[1]", kind: "dropped" },
      { path: "sidebarMenu[2]", kind: "dropped" },
    ],
  );
  assert.deepStrictEqual(
    sorted(
      issuesOf({
        sidebarNav: [
          { id: "hub", pinned: false },
          { id: "nope" },
          { id: "hub" },
        ],
      }),
    ),
    [
      { path: "sidebarNav[1]", kind: "dropped" },
      { path: "sidebarNav[2]", kind: "dropped" },
    ],
  );
  assert.deepStrictEqual(
    issuesOf({ sidebarNavAuto: ["projects", "hub", "projects"] }),
    [
      { path: "sidebarNavAuto[1]", kind: "dropped" },
      { path: "sidebarNavAuto[2]", kind: "dropped" },
    ],
  );
});

test("sanitizeCustomization is the value half of sanitizeCustomizationWithIssues", () => {
  for (const raw of [
    undefined,
    {},
    VALID_CUSTOMIZATION,
    { uiFontSize: 99, colors: "x" },
  ]) {
    assert.deepStrictEqual(
      sanitizeCustomization(raw),
      sanitizeCustomizationWithIssues(raw).value,
    );
  }
  assert.equal(store.isDefaultCustomization(sanitizeCustomization({})), true);
});

test("sanitizeFontName keeps the cleaned name or returns null", () => {
  assert.equal(sanitizeFontName(' Fira "Code";  '), "Fira Code");
  assert.equal(sanitizeFontName("a\u0007b"), "ab");
  assert.equal(sanitizeFontName("   "), null);
  assert.equal(sanitizeFontName(7), null);
});

test("fontFamilyFromFileName gives the family a font file registers under", () => {
  assert.equal(
    fontFamilyFromFileName("Brand_Sans-Bold.woff2"),
    "Brand Sans Bold",
  );
  assert.equal(fontFamilyFromFileName("plain.otf"), "plain");
  assert.equal(fontFamilyFromFileName("noextension"), "noextension");
  assert.equal(fontFamilyFromFileName('Evil";x.ttf'), "Evilx");
  assert.equal(fontFamilyFromFileName("Name, Bold.ttf"), "Name Bold");
  assert.equal(fontFamilyFromFileName(".ttf"), "");
  assert.equal(fontFamilyFromFileName("__.woff"), "");
});

test("the lib module imports no packages and touches no DOM globals", () => {
  const source = readSrc("features/settings/lib/appearance-customization.ts");
  assert.doesNotMatch(source, /from\s+["'][^.]/);
  assert.doesNotMatch(source, /\b(window|document|globalThis)\b/);
});

test("the store still exports every value it exported before the move", () => {
  const exported = new Set(Object.keys(store));
  for (const name of [
    "DEFAULT_CUSTOMIZATION",
    "SIDEBAR_MENU_ITEM_IDS",
    "SIDEBAR_MENU_DEFAULT_VISIBLE",
    "SIDEBAR_NAV_ITEM_IDS",
    "SIDEBAR_NAV_AUTO_ITEM_IDS",
    "SIDEBAR_NAV_DEFAULT_PINNED",
    "sidebarNavRowPinned",
    "sidebarNavAutoAfterChoice",
    "MAX_IMPORTED_FONTS",
    "MAX_IMPORTED_FONT_NAME_LENGTH",
    "MAX_IMPORTED_FONT_DATA_URL_LENGTH",
    "MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH",
    "UI_FONT_SIZE_RANGE",
    "CODE_FONT_SIZE_RANGE",
    "isHexColor",
    "sanitizeCustomization",
    "migrateShippedSidebarNavDefault",
    "migrateAppearanceMenuDefault",
    "isDefaultCustomization",
    "CONTRAST_SURFACE_MIX_VAR",
    "useAppearanceCustomStore",
    "applyCustomizationToDocument",
    "prefersReducedMotion",
  ]) {
    assert.ok(exported.has(name), name);
  }
});
