// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DeepPartialMessageTree } from "../types";
import type { en } from "./en.ts";

export const hu = {
  linkMenu: {
    openInBrowser: "Megnyitás az Unsloth böngészőben",
    openExternal: "Megnyitás külső böngészőben",
    copyLink: "Hivatkozás másolása",
    saveLinkAs: "Hivatkozás mentése másként…",
    alwaysInBrowser: "Hivatkozások megnyitása mindig az Unsloth böngészőben",
    linkCopied: "A hivatkozás kimásolva",
    copyFailed: "Nem sikerült a másolás.",
    saveFailed: "Nem sikerült menteni a fájlt.",
    openFile: "Fájl megnyitása",
    openDefaultApp: "Megnyitás az alapértelmezett alkalmazásban",
    openWith: "Megnyitás ezzel",
    unslothBrowser: "Unsloth böngésző",
    newChat: "Új csevegés",
    browserTab: "Új böngészőlap",
    saveAs: "Mentés másként…",
    copyPath: "Elérési út másolása",
    copyContents: "Fájl tartalmának másolása",
    revealFinder: "Megjelenítés a Finderben",
    revealExplorer: "Megjelenítés az Intézőben",
    revealFolder: "Megjelenítés a mappában",
    pathCopied: "Az elérési út kimásolva",
    contentsCopied: "A tartalom kimásolva",
    openFailed: "Nem sikerült megnyitni: {name}.",
    revealFailed: "Nem sikerült megjeleníteni a mappájában: {name}.",
  },
  imageViewer: {
    title: "Kép",
    description:
      "A képek közötti váltáshoz használja a nyílbillentyűket, a nagyításhoz a plusz és mínusz billentyűt.",
    failed: "Nem sikerült betölteni ezt a képet.",
    downloadFailed: "Nem sikerült menteni a képet.",
    openSource: "Forrásoldal megnyitása",
    download: "Letöltés",
    close: "Bezárás",
    previous: "Előző kép",
    next: "Következő kép",
    zoomIn: "Nagyítás",
    zoomOut: "Kicsinyítés",
    fit: "Ablakhoz igazítás",
    copy: "Kép másolása",
    copied: "A kép kimásolva",
    copyFailed: "Nem sikerült a kép másolása.",
    removeFromMessage: "Eltávolítás az üzenetből",
  },
  browser: {
    drafts: {
      title: "Kiüríti a beírt szöveget tartalmazó oldalt?",
      description:
        "A csevegés gyors működése érdekében legfeljebb három beírt szöveget tartalmazó rejtett oldal maradhat futásban. A folytatással a rendszer kiüríti a következőket: {pages}, ami a nem mentett szöveg elvesztésével járhat. A Mégse gomb visszatér az előző laphoz.",
      confirm: "Kiürítés és folytatás",
    },
    agentSettings: {
      title: "Böngészőhasználat",
      docs: "Dokumentáció",
      docsLabel: "A böngészőhasználat dokumentációjának megnyitása",
      enable: "A modellek használhatják a böngészőt",
      enableDescription:
        "A modellek saját böngészőlapon nyitnak meg oldalakat, elolvassák azokat, és kattintanak, illetve gépelnek Ön helyett. Az üzenetmező + menüjében is elérhető.",
      history: "Böngészési előzmények",
      historyDescription:
        "Teljes hozzáférésnél nincs szükség jóváhagyásra; a Soha beállítás továbbra is tiltja a hozzáférést",
      alwaysAsk: "Mindig kérdezzen",
      alwaysAllow: "Mindig engedélyezze",
      never: "Soha",
      scripts: "Szkriptek futtatása",
      scriptsDescription:
        "Megadhatja, hogy a modellek futtathatnak-e szkripteket az oldalakon. Minden szkriptet láthat, mielőtt lefutna",
      cursor: "Kurzor megjelenítése",
      cursorDescription:
        "Mutató megjelenítése ott, ahol a modellek kattintanak, gépelnek és húznak az oldalon",
      askEveryTime: "Kérdezzen minden alkalommal",
      permissionsTitle: "Ügynökengedélyek",
      permissionsDescription:
        "Alapértelmezések és webhelyenkénti kivételek beállítása. A jelszó- és fizetési mezőket a rendszer soha nem tölti ki.",
      site: "Webhely vagy minta",
      sitePlaceholder: "example.com vagy *.example.com",
      default: "Alapértelmezett",
      browsing: "Böngészés",
      actions: "Műveletek",
      downloads: "Letöltések",
      unlabelled: "Címke nélküli kattintások",
      askNewSites: "Kérdezzen új webhelyeknél",
      requiresApproval: "Jóváhagyást igényel",
      blocked: "Letiltva",
      addSite: "Hozzáadás",
      add: "Hozzáadás",
      cancel: "Mégse",
      removeSite: "Eltávolítás: {site}",
      permissionsNote:
        "Csak a saját engedélyekkel rendelkező webhelyek jelennek meg itt. Műveletnek számítanak a bejelentkezések, az űrlapküldések, valamint a vásárlásra, küldésre vagy törlésre szolgáló gombok.",
      siteInvalid:
        "Adjon meg egy webhelyet, például example.com vagy *.example.com.",
    },
    agent: {
      working: "Az Unsloth ezen a lapon böngészik",
      waiting: "Az Unsloth a böngészőben várja a válaszát (csevegés: {chat})",
      otherChat: "egy másik csevegés",
      show: "Megjelenítés",
      stop: "Leállítás",
      allow: "Engedélyezés",
      deny: "Elutasítás",
      ask: {
        open: "Megnyithatja az Unsloth a következőt: {host}?",
        signIn: "Bejelentkezhet az Unsloth ide: {host}?",
        submit: "Elküldheti az Unsloth ezt az űrlapot? Webhely: {host}",
        click: "Rákattinthat az Unsloth erre: „{name}”? Webhely: {host}",
        history: "Olvashatja az Unsloth a böngészési előzményeit?",
        script: "Futtathatja az Unsloth ezt a szkriptet? Webhely: {host}",
        pointer:
          "Kattinthat az Unsloth címke nélküli elemekre, például vászonra? Webhely: {host}",
      },
      step: {
        open: "Megnyitás: {detail}",
        back: "Vissza",
        forward: "Előre",
        reload: "Az oldal újratöltése",
        read: "Az oldal olvasása",
        find: "Keresés: „{detail}”",
        click: "Kattintás: „{detail}”",
        type: "Gépelés ide: „{detail}”",
        select: "Kiválasztás: „{detail}”",
        press: "Billentyű lenyomása: {detail}",
        scroll: "Görgetés",
        screenshot: "Képernyőkép készítése",
        evaluate: "Szkript futtatása az oldalon",
        history: "A böngészési előzmények olvasása",
        move: "Az egér mozgatása",
        drag: "Húzás",
      },
    },
    title: "Böngésző",
    tabs: "Böngészőlapok",
    newTab: "Új lap",
    newTabFullView: "Új lap teljes nézetben",
    closeTab: "Lap bezárása",
    close: "Böngésző bezárása",
    show: "Böngésző megjelenítése",
    back: "Vissza",
    forward: "Előre",
    reload: "Újratöltés",
    more: "Továbbiak",
    addressPlaceholder: "Keresés vagy URL megadása",
    local: "Helyi",
    browserUse: {
      label: "Böngészőhasználat",
      description: "A modellek vezérelhetik a beépített böngészőt",
      unavailable:
        "Töltsön be eszközhívásra képes modellt, hogy használhassa a böngészőt.",
    },
    suggested: "Javasolt",
    suggestedMenu: {
      open: "Megnyitás",
      openInNewTab: "Megnyitás új lapon",
      remove: "Eltávolítás a javasoltak közül",
    },
    tabMenu: {
      duplicate: "Duplikálás",
      closeOthers: "Többi lap bezárása",
      pin: "Rögzítés az oldalsávon",
      unpin: "Rögzítés feloldása az oldalsávon",
      newTabRight: "Új lap jobbra",
      fork: "Elágaztatás",
      forkNewChat: "Új csevegés",
      forkTemporaryChat: "Ideiglenes csevegés",
      copyUrl: "URL másolása",
      openExternal: "Megnyitás külső böngészőben",
      rename: "Átnevezés",
      renameLabel: "Lap neve",
      mute: "Lap némítása",
      unmute: "Lap némításának feloldása",
      close: "Bezárás",
      closeRight: "Jobbra lévő lapok bezárása",
      pageOptions: "Oldalbeállítások",
    },
    siteInfo: {
      label: "Webhelyadatok",
      secure: "A kapcsolat biztonságos",
      insecure: "A kapcsolat nem biztonságos",
      back: "Vissza",
      close: "Bezárás",
      security: "Biztonság",
      secureTitle: "A kapcsolat biztonságos",
      secureDescription:
        "Az adatai, például a jelszavak és a kártyaszámok, bizalmasak maradnak, amikor elküldi őket erre a webhelyre.",
      insecureTitle: "A kapcsolat nem biztonságos",
      insecureDescription:
        "Az erre a webhelyre küldött adatokat, például a jelszavakat és a kártyaszámokat, mások is láthatják.",
      learnMore: "További információ",
      certificateValid: "A tanúsítvány érvényes",
    },
    tools: "Eszközök",
    recents: "Legutóbbiak",
    website: "Webhely",
    recentNewer: "Újabbak",
    recentOlder: "Régebbiek",
    openExternal: "Megnyitás a rendszer böngészőjében",
    copyLink: "Hivatkozás másolása",
    linkCopied: "A hivatkozás kimásolva",
    download: "Letöltés",
    settings: "Böngésző beállításai",
    cannotShowFile:
      "Ez a fájl itt nem jeleníthető meg. A megnyitáshoz töltse le.",
    fileGone: "Ez a fájl már nem érhető el. Nyissa meg újra a csevegésből.",
    settingsTitle: "Böngésző",
    linksTitle: "Hivatkozások és fájlok",
    addressBarTitle: "Címsor",
    openLinksSetting: "Hivatkozások megnyitása itt",
    openLinksSettingDescription:
      "Hol nyíljanak meg a csevegésben lévő hivatkozások: az alapértelmezett böngészőben vagy a beszélgetés melletti böngésző egyik lapján.",
    linkDestinationDefault: "Alapértelmezett böngésző",
    linkDestinationPanel: "Unsloth böngésző",
    openFilesSetting: "Fájlok megnyitása a böngészőben",
    openFilesSettingDescription:
      "A csatolt dokumentumok, például a PDF-, Office-, HTML- és szövegfájlok, előnézeti ablak helyett a beszélgetés melletti böngészőlapon nyílnak meg.",
    searchEngineSetting: "Keresőmotor",
    searchEngineSettingDescription:
      "Akkor használatos, ha a címsorba URL helyett szavakat ír be.",
    showFullUrlSetting: "Teljes URL megjelenítése",
    showFullUrlSettingDescription:
      "A lekérdezés és a töredék is megjelenik a címsorban, nem csak a webhely és az elérési út.",
    bookmarksTitle: "Könyvjelzők",
    bookmarksToolbarSetting: "Könyvjelzősáv",
    bookmarksToolbarSettingDescription:
      "Mikor jelenjenek meg a mentett oldalak a címsor alatt.",
    bookmarkEditorSettingDescription:
      "Mentéskor elnevezheti a könyvjelzőt, és kiválaszthatja a helyét.",
    bookmarksSettingDescription:
      "A böngészőben mentett oldalak megtekintése, szerkesztése és eltávolítása.",
    browsingDataTitle: "Böngészési adatok",
    historySetting: "Böngészési előzmények",
    historySettingDescription:
      "A böngészőben meglátogatott oldalak megtekintése és kezelése.",
    downloadsSetting: "Letöltési előzmények",
    downloadsSettingDescription:
      "A böngészőből letöltött fájlok megtekintése és kezelése.",
    pagesFromChat:
      "Ennek a böngészőben való megtekintéséhez nyisson meg egy csevegést.",
    tabsTitle: "Lapok",
    switchToNewTabsSetting: "Azonnali váltás az új lapokra",
    switchToNewTabsSettingDescription:
      "Ha ⌘ vagy Ctrl billentyűvel új lapon nyit meg egy hivatkozást, az a háttér helyett azonnal megjelenik.",
    switchToNewTabsNative:
      "Az asztali alkalmazásban az oldalak mindig előtérben nyitják meg az új lapokat.",
    defaultZoomSetting: "Alapértelmezett nagyítás",
    defaultZoomSettingDescription:
      "Mekkora méretben jelenjenek meg az oldalak megnyitáskor. Egy oldal nagyításának visszaállítása erre az értékre tér vissza.",
    newTabPageTitle: "Új lap oldal",
    showSuggestedSetting: "Javasolt webhelyek megjelenítése",
    showSuggestedSettingDescription:
      "Gyakran látogatott webhelyek parancsikonként egy új lapon.",
    showRecentsSetting: "Legutóbbi oldalak megjelenítése",
    showRecentsSettingDescription:
      "A nemrég meglátogatott oldalak listája egy új lapon.",
    hiddenSuggestionsSetting: "Eltávolított javaslatok",
    hiddenSuggestionsSettingDescription:
      "A Javasolt listáról eltávolított webhelyek: {count}. Állítsa vissza őket, hogy ismét megjelenjenek.",
    hiddenSuggestionsNone:
      "A Javasolt listáról eltávolított webhelyek itt állíthatók vissza.",
    restore: "Visszaállítás",
    importBookmarksSetting: "Importálás és exportálás",
    importBookmarksSettingDescription:
      "Könyvjelzők behozatala Chrome-ból, Safariból, Firefoxból vagy Edge-ből az általuk exportált HTML-fájllal, illetve a saját könyvjelzők mentése ilyen fájlként.",
    importBookmarks: "Importálás…",
    exportBookmarks: "Exportálás",
    importBookmarksDoneOne: "{count} könyvjelző importálva.",
    importBookmarksDoneMany: "{count} könyvjelző importálva.",
    importBookmarksNone: "A fájlban nincs új könyvjelző.",
    importBookmarksPartial:
      "{count} importálva, de további {leftOut} nem fért el: az Unsloth legfeljebb {max} könyvjelzőt tárol.",
    importBookmarksFailed:
      "Nem sikerült beolvasni a fájlt. Válasszon egy böngészőből exportált HTML-könyvjelzőfájlt.",
    downloadsTitle: "Letöltések",
    askWhereToSaveSetting: "Rákérdezés minden fájl mentési helyére",
    askWhereToSaveSettingDescription:
      "Mentési párbeszédpanel megjelenítése a böngészőben letöltött fájlokhoz.",
    askWhereToSaveUnsupported:
      "A webböngészője a saját letöltési mappájába menti a letöltéseket.",
    downloadLocationSetting: "Hely",
    downloadLocationDefault: "A rendszer Letöltések mappája",
    downloadLocationChange: "Módosítás",
    downloadLocationReset: "Visszaállítás",
    askBeforeDownloadingSetting: "Rákérdezés letöltés előtt",
    askBeforeDownloadingSettingDescription:
      "A webhelyekről származó fájlok megerősítése mentésük előtt.",
    downloadSitesSetting: "Webhelyek letöltési engedélyei",
    downloadSitesSettingDescription:
      "Webhelyek, amelyekről mindig engedélyezi vagy mindig letiltja a letöltést.",
    downloadSitesNone:
      "Ha a rákérdezéskor bejelöli a Megjegyzés a következő alkalomra lehetőséget, a webhely itt jelenik meg.",
    downloadSitesFor: "Letöltések innen: {host}",
    downloadSitesRemove: "Elfelejtés: {host}",
    downloadSiteAllow: "Mindig engedélyezze",
    downloadSiteBlock: "Mindig tiltsa le",
    downloadPrompt: {
      title: "Letölti ezt a fájlt?",
      description:
        "Egy webhely ({host}) menteni szeretné a számítógépére a következő fájlt: {name}.",
      dangerous:
        "Az ilyen fájlok programokat futtathatnak a számítógépén. Csak akkor töltse le, ha megbízik ebben a webhelyben: {host}.",
      remember: "Megjegyzés a következő alkalomra",
      cancel: "Mégse",
      download: "Letöltés",
      blocked: "Egy letöltés letiltva innen: {host}.",
      ready: "Menthető: {name}",
      save: "Mentés",
    },
    saveDownloadHistorySetting: "Letöltési előzmények megőrzése",
    saveDownloadHistorySettingDescription:
      "A böngészőből letöltött fájlok listázása a Letöltési előzmények között.",
    saveHistorySetting: "Böngészési előzmények mentése",
    saveHistorySettingDescription:
      "A meglátogatott oldalak megjegyzése az Előzmények, a Legutóbbiak és a Javasolt lista számára.",
    historyRetentionSetting: "Előzmények megőrzési ideje",
    historyRetentionSettingDescription:
      "Az ennél régebbi látogatások automatikusan törlődnek.",
    retention: {
      forever: "Törlésig",
      days90: "3 hónap",
      days30: "1 hónap",
      days7: "1 hét",
      days1: "1 nap",
    },
    annotationsTitle: "Megjegyzések",
    annotationScreenshotsSetting: "Képernyőképek a megjegyzésekhez",
    annotationScreenshotsSettingDescription:
      "Az oldalról készült, az Ön jelöléseit tartalmazó képernyőkép segít a látásmodelleknek megérteni a megjegyzéseit, de több környezetet használ.",
    annotationScreenshotsUnsupported:
      "Az asztali alkalmazás vagy egy Chromium-alapú böngésző, például Chrome vagy Edge szükséges hozzá.",
    annotationScreenshotsHint:
      "Webböngészőben minden alkalommal meg kell osztania a lapot, amikor megjegyzéseket küld.",
    annotationScreenshotsAlways: "Mindig legyen benne",
    annotationScreenshotsNever: "Soha ne legyen benne",
    clearDataSetting: "Böngészési adatok törlése",
    clearDataSettingDescription:
      "A böngészési előzmények, a letöltési előzmények és a gyorsítótárazott oldalak törlése. Az oldalak csak az Unsloth újratöltéséig őrzik meg a cookie-kat és a webhelyadatokat.",
    manage: "Kezelés",
    fullView: {
      enter: "Teljes nézet",
      exit: "Kilépés a teljes nézetből",
      minimize: "Csevegés kis méretre állítása",
      showChat: "Csevegés megjelenítése",
      showConversation: "Beszélgetés megjelenítése",
      hideConversation: "Beszélgetés elrejtése",
      newChat: "Új csevegés",
      options: "Csevegési beállítások",
      openChat: "Megnyitás teljes nézetben",
      moveLeft: "Áthelyezés a bal oldali panelre",
      moveRight: "Áthelyezés a jobb oldali panelre",
    },
    annotate: {
      countOne: "1 megjegyzés",
      countMany: "{count} megjegyzés",
      cancel: "Mégse",
      send: "Küldés",
      move: "Áthelyezés",
      placeholder: "Írjon le egy módosítást, vagy tegyen fel egy kérdést",
      areaQuote: "Az oldal egy területe",
      imageQuote: "Kép",
      page: "Kérdés ezzel az oldallal kapcsolatban",
      pagePlaceholder: "Tegyen fel egy kérdést ezzel kapcsolatban",
      save: "Megjegyzés mentése",
      dictate: "Diktálás",
      stopDictating: "Diktálás leállítása",
      dictateFailed: "Nem sikerült elindítani a diktálást",
    },
    menu: {
      find: "Keresés az oldalon",
      print: "Nyomtatás",
      printFailed: "Nem sikerült kinyomtatni az oldalt",
      screenshot: "Képernyőkép készítése",
      zoom: "Nagyítás",
      zoomIn: "Nagyítás",
      zoomOut: "Kicsinyítés",
      zoomReset: "Nagyítás visszaállítása",
      deviceToolbar: "Eszköz eszköztár megjelenítése",
      clearData: "Böngészési adatok törlése",
    },
    screenshot: {
      added: "A képernyőkép hozzáadva a csevegéshez",
      taken: "A képernyőkép elkészült",
      save: "Mentés",
      failed: "Nem sikerült képernyőképet készíteni",
      otherSurface: "Az oldalról készülő képernyőképhez ossza meg ezt a lapot",
    },
    device: {
      mobile: "Mobil",
      tablet: "Táblagép",
      close: "Eszköz eszköztár elrejtése",
    },
    downloads: {
      title: "Letöltések",
      inProgressLabel: "Letöltés folyamatban",
      inProgress: "Letöltés…",
      complete: "A letöltés befejeződött",
      failed: "A letöltés sikertelen",
      downloaded: "Letöltve",
      missing: "A fájlt áthelyezték vagy törölték",
      open: "Megnyitás",
      openFailed: "Nem sikerült megnyitni: {name}.",
      copyLink: "Letöltési hivatkozás másolása",
      showAll: "Összes megjelenítése",
      savePage: "Oldal mentése másként…",
    },
    file: {
      openIn: "Megnyitás itt",
      newChat: "Új csevegés",
      newBrowserTab: "Új böngészőlap",
      requestEdits: "Módosítások kérése",
      requestEditsPrompt: "{name} szerkesztése: ",
      preview: "Előnézet",
      source: "Forrás",
      viewMode: "Fájlnézet",
      runAgain: "Újrafuttatás",
      console: "Konzol",
      copy: "Tartalom másolása",
      copied: "Másolva",
      copyFailed: "Nem sikerült a fájl másolása.",
      copyPath: "Elérési út másolása",
      pathCopied: "Az elérési út kimásolva",
      openFailed: "Nem sikerült megnyitni: {name}.",
      wrap: "Sortörés",
      reactNeedsNode:
        "A React-előnézetekhez Node.js 20.19 vagy újabb szükséges. Ha hiányzik, telepítse, majd futtassa újra az Unsloth telepítőjét.",
      reactFailed:
        "Nem sikerült előkészíteni az előnézetet. Az újrapróbálkozáshoz futtassa újra.",
    },
    video: {
      play: "Lejátszás",
      pause: "Szünet",
      seek: "Tekerés",
      volume: "Hangerő",
      mute: "Némítás",
      unmute: "Némítás feloldása",
      more: "További lejátszási beállítások",
      speed: "Lejátszási sebesség",
      normalSpeed: "Normál",
      loop: "Ismétlés",
      pictureInPicture: "Kép a képben",
      fullscreen: "Teljes képernyő",
      exitFullscreen: "Kilépés a teljes képernyőből",
      copyFrame: "Képkocka másolása",
      frameCopied: "A képkocka kimásolva",
      copyFrameFailed: "Nem sikerült a képkocka másolása.",
      copyOptions: "Másolási lehetőségek",
      open: "Megnyitás",
      openOptions: "Megnyitási lehetőségek",
    },
    pages: {
      searchDownloads: "Keresés a letöltési előzményekben",
      clearAll: "Összes törlése",
      deleted: "Törölve",
      showInFinder: "Megjelenítés a Finderben",
      showInExplorer: "Megjelenítés az Intézőben",
      showInFolder: "Megjelenítés a mappában",
      revealFailed: "Nem sikerült megjeleníteni a mappájában: {name}.",
      history: "Előzmények",
      downloads: "Letöltések",
      bookmarks: "Könyvjelzők",
      search: "Keresés",
      today: "Ma",
      yesterday: "Tegnap",
      remove: "Eltávolítás",
      clearHistory: "Előzmények törlése",
      clearDownloads: "Letöltések törlése",
      noHistory: "A böngészőben meglátogatott oldalak itt jelennek meg.",
      showMore: "Továbbiak megjelenítése",
      noDownloads: "A böngészőből letöltött fájlok itt jelennek meg.",
      noBookmarks:
        "A címsorban lévő csillaggal könyvjelzővel ellátott oldalak itt jelennek meg.",
      noMatches: "Nincs a keresésnek megfelelő találat.",
      fromChat: "Csevegésből",
      searchHistory: "Keresés a böngészési előzményekben",
      filter: "Szűrés",
      allTime: "Teljes időszak",
      allTimeHistory: "Teljes előzmények",
      lastWeek: "Elmúlt 7 nap",
      lastMonth: "Elmúlt 30 nap",
      customDates: "Dátumok kiválasztása…",
      clearFilter: "Szűrő törlése",
      openPage: "Oldal megnyitása",
      removeFromHistory: "Eltávolítás az előzményekből",
      selected: "{count} kijelölve",
      cancel: "Mégse",
      select: "Kijelölés: {title}",
      collapse: "Összecsukás: {date}",
      expand: "Kibontás: {date}",
      pageActions: "Műveletek: {title}",
    },
    bookmarks: {
      bookmarkPage: "Oldal hozzáadása a könyvjelzőkhöz",
      add: "Könyvjelző hozzáadása",
      edit: "Könyvjelző szerkesztése",
      name: "Név",
      location: "Hely",
      toolbar: "Könyvjelzősáv",
      other: "Egyéb könyvjelzők",
      showEditor: "Szerkesztő megjelenítése mentéskor",
      cancel: "Mégse",
      remove: "Könyvjelző eltávolítása",
      delete: "Könyvjelző törlése",
      save: "Mentés",
      saved: "Mentve a könyvjelzők közé",
      removed: "A könyvjelző eltávolítva",
      undo: "Visszavonás",
      manage: "Könyvjelzők kezelése",
      more: "További könyvjelzők",
      toolbarEmpty:
        "Ha ide szeretne felvenni egy oldalt, jelölje meg könyvjelzővel a címsorban lévő csillaggal.",
      toolbarAlways: "Mindig látható",
      toolbarNewTab: "Csak új lapon látható",
      toolbarNever: "Soha nem látható",
    },
    clearData: {
      title: "Törli a böngészési adatokat?",
      description:
        "Ez törli a böngészési előzményeket, a letöltési előzményeket, a gyorsítótárazott oldalakat, valamint a webhelyek cookie-jait és adatait, így a webhelyek kijelentkeztetik Önt. A letöltött fájlok a számítógépén maradnak.",
      cancel: "Mégse",
      confirm: "Adatok törlése",
      done: "A böngészési adatok törölve",
    },
    error: {
      title: "Nem sikerült betölteni az oldalt",
      description:
        "A webhely ({host}) nem válaszolt, elutasította a kapcsolatot, vagy letiltotta a kérést.",
      retry: "Újrapróbálkozás",
      botCheckTitle: "Ez a webhely valódi böngészőt ellenőriz",
      botCheckDescription:
        "A webhely ({host}) azt kéri a látogatóktól, hogy igazolják, hogy emberek, amire az alkalmazáson belüli nézet itt nem képes. Nyissa meg inkább a böngészőjében.",
      botCheckDescriptionWeb:
        "A webhely ({host}) azt kéri a látogatóktól, hogy igazolják, hogy emberek, amire az alkalmazáson belüli nézet itt nem képes. Nyissa meg a böngészőjében, vagy ebben az alkalmazásban, amely meg tudja nyitni: {app}.",
      desktopApp: "Unsloth Desktop",
      resubmit:
        "Ez az oldal egy űrlapból származik. Az újratöltés ismét elküldi az űrlapot, ami megismételheti az általa végzett műveletet.",
      upload:
        "Ebben a nézetben a fájlfeltöltés nem működik. A küldéshez nyissa meg az oldalt a rendszer böngészőjében.",
      scriptNavigation:
        "Ez az oldal szkripttel próbált egy másik címre lépni, amit ez a nézet nem tud követni. Nyissa meg inkább a rendszer böngészőjében.",
      needsBrowser:
        "Ez az oldal olyan böngészőfunkciót próbált használni, amelyet ez a nézet letilt, ezért egyes részei nem működhetnek. Nyissa meg inkább a rendszer böngészőjében.",
    },
    native: {
      externalPrompt:
        "Egy webhely ({host}) meg szeretné nyitni a következőt: {url}",
      open: "Megnyitás",
      downloading: "Letöltés: {name}",
      downloaded: "Mentve a Letöltések mappába: {name}",
      downloadFailed: "Nem sikerült letölteni: {name}",
      notMarked:
        "A fájl mentése megtörtént, de nem sikerült internetről letöltöttként megjelölni, ezért a rendszer nem fog figyelmeztetni megnyitás előtt. Fájl: {name}",
      blocked:
        "Ez a cím nem nyitható meg a böngészőpanelen. Csak nyilvános webhelyek nyithatók meg.",
      blockedLocal:
        "Ez a helyi cím itt nem nyitható meg. A megnyitáshoz írja be a címsorba.",
      studioAddress:
        "Ez az Unsloth saját címe, ezért nem nyitható meg a böngészőpanelen.",
      clearDataSettingDescription:
        "A böngészési előzmények, a letöltési előzmények, a gyorsítótárazott oldalak, valamint az oldalak által tárolt cookie-k és webhelyadatok törlése, ami kijelentkezteti Önt a webhelyekről.",
      clearDataDescription:
        "Ez törli a böngészési előzményeket, a letöltési előzményeket, a gyorsítótárazott oldalakat, valamint a webhelyek cookie-jait és adatait, így a webhelyek kijelentkeztetik Önt. A letöltött fájlok a számítógépén maradnak.",
      clearDataFailed: "Nem sikerült törölni a webhelyadatokat. Próbálja újra.",
    },
  },
  managedEngines: {
    gpu: "GPU-k",
    gpuHelp: "Válassza ki a használni kívánt GPU-kat.",
    parallelism: "Több GPU-s mód",
    tensor: "Tenzorpárhuzamos",
    pipeline: "Folyamatpárhuzamos (pipeline)",
    data: "Replikák (adatpárhuzamos)",
    pipelineHelp:
      "A modell rétegeinek elosztása a GPU-k között. Minden GPU-nak el kell tudnia tárolni a saját rétegeit és gyorsítótárát.",
    dataHelp:
      "A kérések elosztása a GPU-k között. A sűrű modelleknek minden GPU-n el kell férniük; a MoE-modellek megoszthatják a szakértőket.",
    tensorParallel: "A rétegszámítások elosztása {count} GPU között.",
    inUse:
      "A motor javítása, visszaállítása vagy eltávolítása előtt ürítse ki a modellt.",
    title: "További motorok",
    description:
      "További motorok GPU-khoz (béta). Gyorsabban futtathatják a modelleket, különösen kötegelt következtetésnél. Itt telepítse őket, majd válassza ki a modell futtatási beállításaiban.",
    picker: "Következtetési motor",
    precision: "Pontosság",
    precisionAuto: "A modell alapértelmezése",
    precisionHelp:
      "A modell alapértelmezése megtartja a tárolt pontosságot. A többi lehetőség betöltéskor átalakítja a nem kvantált súlyokat.",
    kvCache: "KV-gyorsítótár",
    kvCacheFp8: "FP8 (fele memória)",
    kvCacheHelp:
      "Az FP8 8 biten tárolja a környezet gyorsítótárát, így ugyanabban a memóriában körülbelül kétszer akkora környezet fér el. Ha az ellenőrzőpontban nincsenek kalibrált skálák, a válaszok kissé pontatlanabbak lehetnek.",
    gpuMemory: "GPU-memória korlátja",
    gpuMemoryAuto: "Automatikus",
    gpuMemoryHelp:
      "A kijelölt GPU-k memóriájából a motor által legfeljebb használható rész. Az Automatikus beállítás a szabad memóriát használja; csökkentse, ha helyet szeretne hagyni más programoknak.",
    seed: "Kezdőérték (seed)",
    seedPlaceholder: "A motor alapértelmezése",
    seedHelp:
      "A motor véletlenszám-kezdőértékének beállítása. Az egyidejű csevegések továbbra is módosíthatják a mintavételezett válaszokat.",
    default: "Alapértelmezett",
    installRequired: "telepítés szükséges",
    ownerRequired:
      "Kérje meg az Unsloth tulajdonosát, hogy telepítse ezt a motort.",
    cancelInstall: "Telepítés megszakítása",
    installed: "Telepítve: {version}",
    repair: "Javítás",
    update: "Motor frissítése",
    install: "Motor telepítése",
    installing:
      "A motor telepítése folyamatban. Az előrehaladást a Letöltések között követheti.",
    remove: "Motor eltávolítása",
    rollback: "Előző telepítés visszaállítása",
    installTitle: "Telepítés: {engine}",
    installAndLoad: "Telepítés és betöltés",
    confirm:
      "Telepíti a következőt: {engine} {version}? Ez több GB letöltésével járhat.",
    confirmSized:
      "Telepíti a következőt: {engine} {version}? Letöltendő méret: kb. {size}.",
    wslSetup:
      "Windows rendszeren ez a motor ({engine}) az Unsloth által beállított privát WSL2-környezetben fut. Egy rendszergazdai engedélykérésre és esetleg újraindításra kell számítani.",
    wslReady:
      "Windows rendszeren ez a motor ({engine}) az Unsloth privát WSL2-környezetében fut.",
    wslRestart:
      "A WSL2 bekapcsolásának befejezéséhez indítsa újra a Windowst, majd kattintson ismét a Telepítés gombra.",
    background: "A telepítés a háttérben történik.",
    failed:
      "A motor telepítése sikertelen. Próbálja újra, vagy használja az alapértelmezett motort.",
    details: "Technikai részletek",
    cancelled: "A telepítés megszakítva. Újrapróbálkozhat.",
    scope:
      "Szöveges és képes csevegés NVIDIA GPU-kon. Az első betöltés tovább tarthat.",
  },
  sandboxSetup: {
    levelTitle: "Az OS-sandbox nem érhető el",
    levelDescription:
      "A Magas szint használatához telepítse, vagy maradjon az Alacsony szintnél (szoftveres sandbox).",
    useLow: "Alacsony szintű sandbox használata",
    checking: "A sandbox ellenőrzése ezen a számítógépen…",
    install: "Sandbox telepítése",
    windowsSetup: "Windows-sandbox beállítása",
    copyCommand: "Parancs másolása",
    copied: "A parancs kimásolva",
    copyFailed: "Nem sikerült a parancs másolása",
    copiedRunIt:
      "A parancs kimásolva. Futtassa egy terminálban azon a számítógépen, amelyen az Unsloth fut, majd váltson Magas szintre.",
    ownerOnly:
      "Csak a tulajdonos telepítheti, azon a számítógépen, amelyen az Unsloth fut.",
    windowsNote:
      "Minden újraindítás után rendszergazdai lépést igényel. Az MXC előzetes verzió, nem biztonsági határ.",
    running:
      "Beállítás folyamatban. Hagyja jóvá a jelszókérést vagy a rendszergazdai engedélykérést azon a számítógépen, amelyen az Unsloth fut…",
    succeeded: "Az OS-sandbox készen áll. A sandbox Magas szintje bekapcsolva.",
    stillUnavailable:
      "A beállítás befejeződött, de az OS-sandbox továbbra sem felel meg az ellenőrzésnek.",
    declined:
      "A jelszókérés vagy a rendszergazdai engedélykérés el lett utasítva.",
    failed: "A beállítás sikertelen.",
    startError: "Nem sikerült elindítani a sandbox beállítását",
    checkFailed: "Nem sikerült ellenőrizni a sandboxot ezen a számítógépen.",
    retry: "Újrapróbálkozás",
    keepsRunning:
      "Ennek bezárása nem állítja le a beállítást. Az eredmény a Beállítások > Biztonság részen jelenik meg.",
  },
  composerSettings: {
    title: "Üzenetmező",
    plainText: "Egyszerű szöveges üzenetmező",
    plainTextDescription:
      "A kód, a Markdown és a hivatkozások szó szerinti szövegként maradnak. Kapcsolja ki a formázott előnézet megjelenítéséhez.",
    showContext: "Környezetablak kihasználtsága",
    sendShortcut: "Küldés billentyűparancsa",
    sendMultiline: "{mod} + Enter többsoros promptoknál",
    sendAlways: "Mindig {mod} + Enter",
    sendEnterDescription: "Az Enter küld. A Shift + Enter új sort kezd.",
    sendMultilineDescription:
      "Az Enter elküldi az egysoros promptot. Ha a prompt egynél több sorból áll, az Enter új sort kezd, a {mod} + Enter pedig küld.",
    sendAlwaysDescription: "A {mod} + Enter küld. Az Enter új sort kezd.",
    followUp: "Folytató üzenetek viselkedése",
    followUpDescription:
      "Mi történjen, ha válasz közben küld üzenetet. Egyszeri ellenkező viselkedéshez nyomja meg ezt: {shortcut}.",
    followUpMultilineShortcut: "{shortcut} (többsoros promptban: {multiline})",
    queue: "Sorba állítás",
    steer: "Irányítás",
    steerDescription:
      "Az Irányítás leállítja az aktuális választ, és következőként elküldi a folytató üzenetet.",
    settings: "Üzenetmező beállításai",
    preview: "Formázott előnézet",
  },
  promptQueue: {
    loading: "Sorba állított promptok betöltése",
    listLabel: "Sorba állított promptok",
    regionLabel: "Promptsor, {current}/{total}",
    itemLabel: "Sorba állított prompt, {position}/{total}: {prompt}",
    reorderInstructions:
      "Az átrendezéshez húzza a fogantyút. Ha a fogantyú fókuszban van, a Fel vagy Le billentyűvel egy pozícióval mozgathatja, a Home vagy End billentyűvel pedig az elejére vagy a végére.",
    dragTooltip: "Húzza az átrendezéshez",
    reorderLabel: "Sorba állított prompt átrendezése, {position}/{total}",
    paused: "Szüneteltetve",
    steer: "Irányítás",
    steerTooltip:
      "A válasz megszakítása és ennek a promptnak a következőként való elküldése",
    steerLabel: "Irányítás a sorba állított prompttal, sorszám: {position}",
    removeTooltip: "Eltávolítás a sorból",
    removeLabel: "Sorba állított prompt eltávolítása, sorszám: {position}",
    moreTooltip: "További lehetőségek",
    moreLabel:
      "További lehetőségek a sorba állított prompthoz, sorszám: {position}",
    editItem: "Üzenet szerkesztése",
    copyItem: "Üzenet másolása",
    editLabel: "Sorba állított prompt szerkesztése, sorszám: {position}",
    cancel: "Mégse",
    save: "Mentés",
    turnOffQueueing: "Sorba állítás kikapcsolása",
    turnOnQueueing: "Sorba állítás bekapcsolása",
    resume: "Sor folytatása",
    queueButton: "Üzenet sorba állítása",
    steerButton: "Válasz irányítása",
    sendTooltip: "Üzenet küldése ({shortcut})",
    sendLabel: "Üzenet küldése",
    followUpTooltip: "{action} ({send}) · ellenkezőjéhez: {opposite}",
    announceUpdated: "A sorba állított prompt frissítve.",
    announceEditFailed:
      "Ez a prompt már nem szerkeszthető, mert a sor megváltozott.",
    announceRemoved: "A prompt eltávolítva a sorból.",
    announceSteered: "Ez a prompt következőként irányítja a választ.",
    announceSteerFailed:
      "Ez a prompt nem tudta irányítani a választ. Ellenőrizze a sort, és próbálja újra.",
    announceCopied: "A prompt kimásolva.",
    announceCopyFailed: "Nem sikerült a prompt másolása. Próbálja újra.",
    announceQueueingOn:
      "Az új folytató üzenetek az aktuális válasz után sorba állnak.",
    announceQueueingOff:
      "Az új folytató üzenetek az aktuális választ irányítják.",
    announceMoved: "A prompt áthelyezve erre a pozícióra: {position}/{total}.",
    announceMoveFailed:
      "A sor megváltozott, mielőtt a promptot áthelyezhette volna. Próbálja újra.",
    announceDragReset:
      "A sor megváltozott. A maradék promptok átrendezéséhez húzza újra.",
    editingHint: "Üzenet szerkesztése",
    queueingOffHint:
      "Az új üzenetek megszakítják a választ, és következőként futnak.",
    queueingOnHint: "Az új üzenetek kivárják a sorukat.",
    queueingHintShared: "A sor megmarad.",
  },
  rewind: {
    title: "Visszatekerés",
    confirmTitle: "Visszateker erre az üzenetre?",
    restoreLabel: "Visszaállítás",
    pickDescription:
      "Válassza ki az üzenetet, amelyre vissza szeretne tekerni.",
    pickEmpty: "Még nincs olyan üzenet, amelyre vissza lehetne tekerni.",
    attachmentOnly: "Csak mellékletek",
    modeBoth: "Beszélgetés és fájlok",
    modeConversation: "Csak a beszélgetés",
    modeFiles: "Csak a fájlok",
    checkFailed: "Nem sikerült ellenőrizni a fájlmódosításokat: {error}",
    summary:
      "Visszaállítandó fájlok: {restore}. Eltávolítandó: {remove}. Újra létrehozandó: {recreate}.",
    conflicts: "Azóta módosult, kimarad",
    untracked: "Nem állítható vissza",
    scopeNote:
      "A fájlok közé a kód is beletartozik. A programok, a csomagok és a csevegés mappáján kívüli fájlok nem változnak.",
    back: "Vissza",
    cancel: "Mégse",
    confirm: "Visszatekerés",
    waitForResponse: "Várja meg, amíg az aktuális válasz befejeződik",
    restoredOne: "1 fájl visszaállítva",
    restoredMany: "{count} fájl visszaállítva",
    nothingRestored: "Egyetlen fájlt sem kellett visszaállítani",
    skippedCount: "{count} kihagyva",
    filesNotRestored:
      "{count} módosult fájl nem állítható vissza, ezek változatlanok maradnak.",
    restoreFailed: "Nem sikerült visszaállítani a fájlokat",
    undo: "Visszavonás",
    undone: "A fájlok visszaállítása visszavonva",
    undoFailed: "Nem sikerült visszavonni a fájlok visszaállítását",
  },
  chatMenu: {
    more: "Csevegési beállítások",
    copy: "Másolás",
    compact: "Beszélgetés tömörítése",
    compactQueued: "A tömörítés a következő üzenetnél történik meg",
    compactEmpty: "Még nincs mit tömöríteni",
    compactUnsupported:
      "A tömörítéshez eszközhasználatot támogató GGUF- vagy MLX-modell szükséges",
  },
  halloween: {
    navLabel: "Csokit vagy csalunk…",
    title: "Csokit vagy csalunk?!",
    description:
      "Kotyvasztottunk néhány ijesztgetést az Unslothnak, október végéig itt maradnak.",
    chooseTheme: "Válasszon hátborzongató témát",
    spookyGreetings: "Kísérteties üdvözlések",
    spookyGreetingsDescription:
      "A lajhár egy kis ijesztéssel köszönti az új csevegéseket.",
    decorations: "Pókok és denevérek",
    decorationsDescription: "Pókhálók, és itt-ott egy pók vagy denevér.",
    surprise: "Csokit vagy csalunk!",
    surpriseDescription: "Hagyja, hogy a lajhár válasszon témát Önnek.",
    themesTitle: "Hátborzongató témák",
    back: "Vissza",
    footer:
      "Az Unsloth november 1-jén éjfélkor visszaáll a megszokott állapotába.",
    removeNow: "Eltávolítás most",
    hideFromSidebar: "Elrejtés az oldalsávról",
    greeting: {
      pumpkins: "A tökök kifaragva. Mi következik?",
      brewing: "Kezdődjön a kotyvasztás",
      trickOrTreat: "Csokit vagy csalunk?",
      manor: "Üdvözöljük a kastélyban",
      manorName: "Üdvözöljük a kastélyban, {name}",
      happyHalloween: "Boldog Halloweent",
      happyHalloweenName: "Boldog Halloweent, {name}",
      spirits: "A szellemek figyelnek…",
      mysteries: "Milyen rejtélyekre derítünk fényt ma éjjel?",
      conjure: "Mit varázsolhatok elő Önnek?",
    },
  },
  welcome: {
    goodToSeeYou: "Örülünk, hogy itt van",
    goodToSeeYouName: "Örülünk, hogy itt van, {name}",
    readyWhenYouAre: "Készen állunk, amikor Ön is",
    onYourMind: "Mi jár ma a fejében?",
    howCanIHelp: "Miben segíthetek?",
    goodMorning: "Jó reggelt",
    goodMorningName: "Jó reggelt, {name}",
    goodEvening: "Jó estét",
    goodEveningName: "Jó estét, {name}",
    tonight: "Mi a terv ma estére?",
    nightOwl: "Éjszakai bagoly üzemmód?",
    lateNightIdeas: "Késő esti ötletek?",
    upLate: "Egy ötlet miatt fent van még?",
    nightShift: "Kezdődik az éjszakai műszak",
    nightShiftName: "Kezdődik az éjszakai műszak, {name}",
    temporaryChat: "Ideiglenes csevegés",
    temporaryChatDescription:
      "Ez a csevegés nem jelenik meg az előzményekben, és nem kerül mentésre. Kilépéskor eltűnik.",
  },
  picker: {
    unifiedTightFit: "Szűkös hely",
    unifiedTightFitHint:
      "Lehet, hogy rövidebb környezettel töltődik be, vagy lassabban fut ezen a Macen.",
    onDevice: "Az eszközön",
    huggingFace: "Hugging Face",
    retry: "Újrapróbálkozás",
    loadMore: "Továbbiak betöltése",
    offlineTitle: "Nincs internetkapcsolat",
    offlineBody:
      "Váltson az Eszköz nézetre a gyorsítótárban lévő vagy helyi {noun} használatához.",
    offlineSwitchDevice: "Eszköz",
    searchAriaLabel: "Keresés: {noun}",
    modelSourceAriaLabel: "Modellforrás",
    hubSectionAriaLabel: "Hub szakasz",
    modelDropped: "Már nem érhető el",
    modelDroppedByProvider: "{provider} · már nem érhető el",
    modelDisabled: "Nincs engedélyezve",
    modelDisabledByProvider: "{provider} · nincs engedélyezve",
    multipleMatches:
      "Több egyező találat ({noun}). Válasszon egyet a listából.",
    rateLimitedTitle: "Elérte a Hugging Face lekérési korlátját",
    rateLimitedBody:
      "Várjon egy kicsit, majd próbálja újra a keresést ({noun}).",
    modelScope: "ModelScope",
    useModelScope: "A ModelScope használata",
    useModelScopeHint:
      "Keresés és letöltés helyette a ModelScope-ról. A Beállításokban visszaválthat.",
    useModelScopeFailed: "Nem sikerült átváltani a ModelScope-ra.",
    updateToken: "Token frissítése",
    hfToken: {
      label: "HF-token",
      saved: "Mentve",
      add: "Nincs megadva",
      savedAriaLabel: "Hugging Face-token mentve",
      addAriaLabel: "Hugging Face-token megadása",
      savedHint:
        "A token mentve. A hozzáférés ellenőrzése használatkor történik.",
      addHint:
        "Adjon meg egy tokent a privát és korlátozott hozzáférésű tárolók eléréséhez.",
    },
  },
  common: {
    duplicate: "Kettőzés",
    cancel: "Mégse",
    close: "Bezárás",
    delete: "Törlés",
    done: "Kész",
    error: "Hiba",
    export: "Exportálás",
    help: "Súgó",
    todayAt: "Ma, {time}",
    yesterdayAt: "Tegnap, {time}",
    loading: "Betöltés...",
    new: "Új",
    rename: "Átnevezés",
    save: "Mentés",
    saving: "Mentés...",
    search: "Keresés",
    shutdown: "Leállítás",
  },
  shell: {
    helpMenu: {
      documentation: "Dokumentáció",
      keyboardShortcuts: "Billentyűparancsok",
      whatsNew: "Újdonságok",
      troubleshooting: "Hibaelhárítás",
      systemStatus: "Rendszerállapot",
      sendFeedback: "Visszajelzés küldése",
      about: "Az Unsloth névjegye",
    },
    find: {
      label: "Keresés az oldalon",
      previous: "Előző találat",
      next: "Következő találat",
      close: "Keresés bezárása",
      truncated: "Ez az oldal túl hosszú a teljes kereséshez.",
      searchChat: "Keresés a csevegésben",
      searchBrowser: "Keresés a böngészőoldalon",
    },
    zoom: {
      label: "Nagyítás",
      zoomOut: "Kicsinyítés",
      zoomIn: "Nagyítás",
      reset: "Visszaállítás",
      announce: "Nagyítás: {percent}%",
    },
    beta: "BÉTA",
    brand: "unsloth",
    product: "Unsloth",
    accountMenu: "Fiókmenü: {name}",
    updateAvailable: "Frissítés érhető el",
    resize: {
      collapse: "Kattintson az összecsukáshoz",
      expand: "Kattintson a kibontáshoz",
      drag: "Húzza az átméretezéshez",
    },
    aria: {
      home: "Unsloth kezdőlap",
      closeSidebar: "Oldalsáv bezárása",
      openSidebar: "Oldalsáv megnyitása",
      resizeSidebar: "Oldalsáv átméretezése vagy összecsukása",
      resizeRunSettings: "Futtatási beállítások átméretezése vagy bezárása",
      openRunSettings: "Futtatási beállítások megnyitása",
      chatOptions: "Csevegési beállítások",
      runOptions: "Futtatási beállítások",
    },
    navigation: {
      newChat: "Új csevegés",
      returnToChat: "Vissza a csevegéshez",
      returnToChats: "Vissza a csevegésekhez ({count})",
      chatGenerating: "Generálás",
      chatAwaitingApproval: "Jóváhagyásra vár",
      chatUnread: "Olvasatlan",
      compare: "Összehasonlítás",
      search: "Keresés",
      projects: "Projektek",
      library: "Könyvtár",
      hub: "Modellközpont",
      train: "Tanítás",
      recipes: "Receptek",
      images: "Képek",
      video: "Videó",
      audio: "Hang",
      tasks: "Ütemezett",
      trainChecking: "A gép tanítási támogatásának ellenőrzése...",
      videoChecking: "A gép videótámogatásának ellenőrzése...",
      more: "Továbbiak",
      customizeSidebar: "Oldalsáv testreszabása",
      newBadge: "Új",
      export: "Exportálás",
      recents: "Legutóbbiak",
      noChatsYet: "Még nincs csevegés",
      noChats: "Nincs csevegés",
      allProjectsPinned: "Minden projekt rögzítve van",
      allProjectsFiled: "Minden projekt rögzítve van vagy szakaszba sorolva",
      noProjects: "Nincs projekt",
      showMore: "Több megjelenítése",
      showLess: "Kevesebb megjelenítése",
      settings: "Beállítások",
      api: "API",
      lightMode: "Világos mód",
      darkMode: "Sötét mód",
      guidedTour: "Bemutató",
      help: "Súgó",
      logOut: "Kijelentkezés",
      logOutNeedsPassword:
        "Kijelentkezés előtt állítson be jelszót a Beállítások > Távoli elérés és LAN lapon, hogy újra be tudjon jelentkezni.",
      logOutFailed:
        "Nem sikerült biztonságosan kijelentkezni. Próbálja újra egy kis idő múlva.",
      shutdown: "Leállítás",
    },
    commandPalette: {
      placeholder: "Írjon be egy parancsot vagy keresést...",
      noResults: "Nincs találat.",
      navigation: "Navigáció",
      actions: "Műveletek",
      chat: "Csevegés",
      searchChats: "Keresés a csevegésekben...",
    },
    search: {
      placeholder: "Keresés",
      tabs: {
        all: "Összes",
        chats: "Csevegések",
        projects: "Projektek",
        files: "Fájlok",
        models: "Modellek",
      },
      recents: "Legutóbbiak",
      actions: "Műveletek",
      newChat: "Új csevegés",
      newTemporaryChat: "Új ideiglenes csevegés",
      fineTune: "Modell finomhangolása",
      generateImage: "Kép generálása",
      generateVideo: "Videó generálása",
      untitledChat: "Névtelen csevegés",
      compare: "Összehasonlítás",
      loading: "Betöltés…",
      empty: {
        all: "Még nincs miben keresni.",
        chats: "Még nincs csevegés.",
        projects: "Még nincs projekt.",
        files: "Még nincs fájl a Könyvtárban.",
        models: "Még nincs letöltött modell.",
      },
      noMatches: "Nincs találat.",
      when: {
        today: "Ma",
        pastWeek: "Elmúlt hét",
        pastMonth: "Elmúlt hónap",
        older: "Régebbi",
      },
      footer: {
        close: "Bezárás",
        changeType: "Típus váltása",
        open: "Megnyitás",
      },
    },
    notFound: {
      title: "Az oldal nem található",
      description: "Nem létezik: {path}",
      backToChat: "Vissza a csevegéshez",
    },
    selection: {
      pinProjects: "Projektek rögzítése",
      unpinProjects: "Projektek rögzítésének feloldása",
      deleteProjects: "Projektek törlése",
      deleteProjectsTitle: "Projektek törlése",
      deleteProjectsDescription:
        "Törli a kijelölt projekteket ({count})? A csevegéseik véglegesen törlődnek.",
      deleteProjectsFilesDescription:
        "Minden projekt munkaterület-mappája törlődik a lemezről.",
      countSelected: "{count} kijelölve",
      pinChats: "Csevegések rögzítése",
      unpinChats: "Csevegések rögzítésének feloldása",
      archiveChats: "Csevegések archiválása",
      markUnread: "Megjelölés olvasatlanként",
      markRead: "Megjelölés olvasottként",
      deleteChats: "Csevegések törlése",
      deleteTitle: "Csevegések törlése",
      deleteDescription:
        "Törli a kijelölt csevegéseket ({count})? Ez a művelet nem vonható vissza.",
      deleteFilesDescription:
        "Minden csevegés saját sandboxmappája törlődik a lemezről. A projekten belül írt fájlok a projekt munkaterületén maradnak.",
      deleteFilesLabel: "Fájlok és sandboxmappa törlése",
      deleteChatFilesDescription:
        "A csevegés saját sandboxmappája törlődik a lemezről. A projekten belül írt fájlok a projekt munkaterületén maradnak.",
    },
    organize: {
      sidebarHeading: "Oldalsáv rendezése",
      byProject: "Projekt szerint",
      inOneList: "Egy listában",
      sortChatsBy: "Csevegések rendezése",
      projectsHome: "Projektek kezdőlapja",
      lastUpdated: "Utolsó frissítés",
      manualOrder: "Egyéni sorrend",
      switchedToManual: "Egyéni rendezés: a sorok húzással átrendezhetők",
      organizeChats: "Csevegések rendezése",
      organizeProjects: "Projektek rendezése",
      sortPinnedChats: "Rögzített csevegések rendezése",
      show: "Megjelenítés",
      newSection: "Új szakasz",
    },
    sections: {
      createTitle: "Új szakasz",
      createDescription:
        "Csoportosítsa a csevegéseket és projekteket tetszése szerint",
      namePlaceholder: "Szakasz neve",
      create: "Szakasz létrehozása",
      renameTitle: "Szakasz átnevezése",
      renameDescription: "Adjon új nevet ennek a szakasznak",
      edit: "Szerkesztés",
      remove: "Szakasz eltávolítása",
      markAllRead: "Az összes megjelölése olvasottként",
      moveTo: "Áthelyezés ide",
      section: "Szakasz",
      sectionsHeading: "Szakaszok",
      removeFromProject: "Eltávolítás a projektből",
      newSection: "Új szakasz",
      removeFromSection: "Eltávolítás a szakaszból",
      removeFrom: "Eltávolítás innen: {name}",
      empty: "Húzzon ide csevegéseket vagy projekteket",
      sectionOptions: "Szakaszbeállítások",
      newChatInSection: "Új csevegés itt: {name}",
      deleted: "Szakasz törölve: „{name}”",
      undo: "Visszavonás",
    },
    dialog: {
      deleteChat: {
        title: "Csevegés törlése",
        description: "Biztosan törli ezt a csevegést: „{name}”?",
      },
      deleteRun: {
        title: "Tanítási futtatás törlése",
        description: "Biztosan törli ezt a futtatást: „{name}”?",
      },
      renameChat: {
        title: "Csevegés átnevezése",
        placeholder: "Csevegés címe",
      },
      renameRun: {
        title: "Futtatás átnevezése",
        placeholder: "Futtatás neve",
      },
    },
    toast: {
      cannotDeleteRunningRun:
        "Folyamatban lévő tanítási futtatás nem törölhető",
      failedToDeleteChat: "Nem sikerült törölni a csevegést",
      failedToDeleteRun: "Nem sikerült törölni a futtatást",
      failedToRenameChat: "Nem sikerült átnevezni a csevegést",
      failedToRenameRun: "Nem sikerült átnevezni a futtatást",
    },
  },
  settings: {
    accounts: {
      signsInAs: "Bejelentkezési név: {username}",
      signsInAsOwner: "Bejelentkezési név: {alias} vagy {username}",
      signInName: "Az Ön bejelentkezési neve",
      signInNameDescription:
        "Ezt a nevet kell beírnia a bejelentkezéshez, ha a telepítésnek több fiókja van. Az unsloth név mindig működik. Hagyja üresen, ha csak az unsloth nevet kívánja használni.",
      signInNameSave: "Név mentése",
      title: "Fiókok",
      description:
        "Privát Unsloth-fiókok létrehozása. Az új felhasználók egyszer használható beállítási kóddal jelentkeznek be, majd jelszót választanak.",
      username: "Felhasználónév",
      create: "Fiók létrehozása",
      createDescription:
        "Ossza meg a beállítási kódot, hogy a felhasználó maga választhasson jelszót.",
      actionsFor: "Kezelés: {username}",
      actions: "Műveletek",
      search: "Fiókok keresése",
      noResults: "Nincs egyező fiók",
      created: "Létrehozva",
      status: "Állapot",
      loginHint:
        "Jelentkezzen be ezzel a felhasználónévvel: {username}, jelszóként ezt a kódot használva, majd válasszon új jelszót.",
      privateAccount: "Privát fiók",
      modelAccess: "Modellhozzáférés",
      modelAccessTitle: "Modellhozzáférés: {username}",
      modelAccessDescription:
        "Válassza ki, mely Hugging Face-modelleket láthatja, töltheti le, töltheti be és használhatja ez a fiók az API-kulcsain keresztül.",
      modelAccessLabel: "Engedélyezett modellek",
      modelAccessHint:
        "Soronként egy modell, például unsloth/Qwen3-4B-GGUF, vagy unsloth/* egy teljes szervezethez. Hagyja üresen az összes modell engedélyezéséhez.",
      modelAccessSave: "Mentés",
      modelAccessLimited: "Korlátozott modellhozzáférés",
      empty: "Még nincs más fiók",
      setupCode: "Beállítókód",
      setupFor: "Beállítókód: {username}",
      shownOnce:
        "Másolja ki most ezt a kódot, és ossza meg a fiók tulajdonosával. Csak itt jelenik meg, és 60 percen belül egyszer használható fel.",
      expires: "Lejárat: {expiry}",
      copy: "Beállítókód másolása",
      copied: "Másolva",
      copyFailed:
        "Nem sikerült másolni. Jelölje ki és másolja a fenti beállítási kódot.",
      dismiss: "Kész",
      owner: "Telepítés tulajdonosa",
      active: "Engedélyezve",
      inactive: "Letiltva",
      regenerate: "Beállítókód újragenerálása",
      resetTitle: "Visszaállítja a felhasználó jelszavát? ({username})",
      resetDescription:
        "A beállítási kód újragenerálása lecseréli a felhasználó jelszavát, kijelentkezteti, és visszavonja az API-kulcsait (felhasználó: {username}). Adja át neki az új kódot, hogy ismét jelszót választhasson.",
      deactivate: "Letiltás",
      reactivate: "Engedélyezés",
      delete: "Fiók törlése",
      deleteTitle: "Törli ezt a fiókot: {username}?",
      deleteDescription:
        "Ez visszavonja a felhasználó munkameneteit, és megszakítja a munkáját (felhasználó: {username}). A csevegései, beállításai, hitelesítő adatai, feltöltései, adatkészletei, tanítási futtatásai, kimenetei, exportjai, galériái, sandboxai, projektjei és ideiglenes fájljai kivonásra kerülnek. A mappák átnevezve félrekerülnek, sosem törlődnek. Ha újra létrehozza ezt a felhasználónevet, egy új fiók indul ezen adatok nélkül.",
      cancel: "Mégse",
      retry: "Fiókok frissítése",
      failed: "A fiókkérés sikertelen.",
    },
    title: "Beállítások",
    dialog: {
      title: "Beállítások",
      description: "Az Unsloth beállításainak kezelése.",
      closeAriaLabel: "Beállítások bezárása",
      searchPlaceholder: "Keresés a beállításokban…",
      searchNoResults: "Nem található beállítás.",
      panelFailed: "Ezt a szakaszt nem sikerült betölteni.",
      panelReload: "Újratöltés",
    },
    library: {
      layoutSection: "Elrendezés",
      cardSize: "Kártyaméret",
      cardSizeDescription: "Hány kártya fér el egy sorban.",
      small: "Kicsi",
      medium: "Közepes",
      large: "Nagy",
      imageLayout: "Képkártyák",
      imageLayoutDescription:
        "A képek eredeti arányának megtartása, vagy egyforma négyzetekre vágásuk.",
      masonry: "Eredeti arány",
      square: "Négyzet",
      showCardDates: "Dátum megjelenítése a kártyákon",
      showCardDatesDescription:
        "Az egyes fájlok utolsó módosításának időpontja.",
      browsingSection: "Böngészés",
      startTab: "Kezdőlap",
      startTabDescription: "A lap, amelyen a Könyvtár megnyílik.",
      suggested: "Javasolt",
      all: "Összes",
      sort: "Rendezés",
      sortDescription:
        "A fájlok és mappák sorrendje. A Javasolt lap mindig a legújabbakat mutatja.",
      recent: "Legújabb",
      oldest: "Legrégebbi",
      name: "Név",
      suggestedLimit: "Javasolt elemek",
      suggestedLimitDescription:
        "Hány legutóbbi fájlt mutasson a Javasolt lap.",
      tabAuto: "Ha vannak fájljai",
      tabAlways: "Mindig",
      contentSection: "Tartalom",
      contentDescription:
        "Válassza ki, mit mutasson a Könyvtár. Az elrejtett fájlok nem törlődnek.",
      showChatAttachments: "Csevegésmellékletek",
      showChatAttachmentsDescription: "A csevegésekhez csatolt fájlok.",
      showChatToolFiles: "Csevegési eszközök fájljai",
      showChatToolFilesDescription:
        "A csevegések során kód és eszközök által létrehozott fájlok.",
      showGeneratedMedia: "Generált média",
      showGeneratedMediaDescription:
        "Képek, videók és hangok a Képek, Videó és Hang oldalakról.",
      showFineTunesDescription:
        "Az Unslothban tanított vagy exportált modellek.",
      deletingSection: "Törlés",
      confirmDelete: "Megerősítés törlés előtt",
      confirmDeleteDescription:
        "Rákérdezés a fájlok és mappák Könyvtárból való törlése előtt.",
      reset: "Könyvtárbeállítások visszaállítása",
      resetDescription:
        "A fenti beállítások visszaállítása az alapértelmezettre.",
      resetButton: "Visszaállítás",
      resetDone: "A Könyvtár beállításai visszaállítva",
      storageSection: "Tárhely",
      downloadsSection: "Képletöltések",
      imageInfoStamp: "Információs felirat",
      imageInfoStampDescription:
        "A modell, a dátum és a seed megjelenítése a Képek oldalról letöltött képek alján. A generálási beállítások mindenképpen a fájlba mentődnek.",
      storageDescription:
        "Nyisson meg egy kategóriát a fájljai méret szerint csökkenő sorrendben való megtekintéséhez, és törölje, amire már nincs szüksége.",
      storageUsed: "A Könyvtár által használt hely: {size}",
      storageDisk: "Szabad hely ezen a lemezen: {free} / {total}",
      storageEmpty: "Még semmi sincs tárolva a Könyvtárban.",
      storageHidden: "A Tartalom beállításaiban elrejtve: {size}",
      storageHiddenLabel: "Elrejtve",
      storageRefresh: "Frissítés",
      storageOpen: "Megnyitás",
      storageError: "Nem sikerült beolvasni a Könyvtár tárhelyadatait.",
      locationsSection: "Fájlok a lemezen",
      locationsDescription:
        "Hol tárolja az Unsloth az egyes fájltípusokat. A Könyvtár mappái csak címkék az Unslothon belül, nem mappák a lemezen. A finomhangolt modellek és az exportok a helyükön maradnak, mivel a tanítás és a csevegések elérési út alapján hivatkoznak rájuk.",
      locationUploads: "Könyvtárfeltöltések",
      locationExports: "Exportok",
      revealFailed: "Nem sikerült megnyitni a fájlkezelőt",
      locationReset: "Visszaállítás",
      locationMoveTitle: "Áthelyezés új mappába: {name}",
      locationMoveDescription:
        "Válasszon bármelyik mappát. Ha már vannak benne fájlok, az Unsloth saját mappát hoz létre benne. Minden, ami már itt van, szintén áthelyeződik.",
      locationMoveAction: "Áthelyezés ide",
      locationMoving: "Áthelyezés folyamatban: {name}…",
      locationMoved: "Áthelyezve: {name}",
      locationMoveFailed: "Nem sikerült áthelyezni: {name}",
      locationFree: "Szabad hely ezen a meghajtón: {free}",
      locationUnavailable:
        "Nem érhető el. Csatlakoztassa újra a meghajtót, vagy állítsa vissza a mappát.",
      locationResetUnavailableTitle: "Visszaállítja a következőt: {name}?",
      locationResetUnavailableDescription:
        "A meghajtója nincs csatlakoztatva. A fájlok azon a meghajtón maradnak, és nem jelennek meg a Könyvtárban, amíg vissza nem helyezi őket. Hely: {path}",
      locationLeftBehind:
        "Az itt lévő fájlok azon a meghajtón maradnak: {path}",
      dataStorage: "Könyvtár tárhelye",
      manageStorage: "Tárhely kezelése",
      categoryImages: "Képek",
      categoryVideos: "Videók",
      categoryFineTunes: "Finomhangolt modellek",
      itemCountOne: "1 elem",
      itemCount: "{count} elem",
      tabsSection: "Lapok",
      tabsDescription:
        "Válassza ki, mely lapokat mutassa a Könyvtár. A hivatkozásból megnyitott lap továbbra is megjelenik.",
      tabHidden: "Elrejtve",
      lastVisited: "Utoljára megtekintett",
    },
    tabs: {
      sandbox: "Biztonság",
      accounts: "Fiókok",
      general: "Általános",
      profile: "Profil",
      appearance: "Megjelenés",
      resources: "Rendszer",
      chat: "Csevegés",
      voice: "Hang",
      connections: "Kapcsolatok",
      data: "Adatok",
      apiKeys: "API",
      remoteLan: "Távoli elérés és LAN",
      agents: "Ügynökök",
      keyboardShortcuts: "Billentyűparancsok",
      debugging: "Naplók",
      about: "Névjegy",
    },
    keyboardShortcuts: {
      title: "Billentyűparancsok",
      intro:
        "Billentyűparancs módosításához jelölje ki a billentyűkombinációt, majd nyomja le az új billentyűket",
      searchPlaceholder: "Billentyűparancsok keresése…",
      keystrokePlaceholder: "Nyomja le a keresendő billentyűparancsot",
      searchByKeystrokes: "Keresés billentyűleütések alapján",
      searchByName: "Keresés név alapján",
      noResults: "Nincs a keresésnek megfelelő billentyűparancs.",
      unassigned: "nincs",
      recording: "Nyomja le a billentyűsorozatot",
      needsModifier:
        "Adja hozzá a ⌘, Ctrl vagy Alt billentyűt. Egy önálló billentyű elnyelné a gépelést.",
      conflict: "Egy másik billentyűparancs is használja",
      conflictShadowed:
        "Egy másik billentyűparancs is ezt a kombinációt használja, és az fut le helyette",
      edit: "Billentyűparancs módosítása",
      reset: "Alapértelmezett visszaállítása",
      resetAll: "Alapértelmezések visszaállítása",
      browserReserved:
        "Lehet, hogy a böngésző lefoglalja ezt a kombinációt. Az asztali alkalmazásban működik.",
      metaUnsupported:
        "A Windows billentyű nem használható billentyűparancsokban.",
      altGraphUnsupported:
        "Az AltGr kombinációk nem használhatók billentyűparancsként.",
      desktopOnly: "Csak asztali alkalmazásban",
      groups: {
        composer: "Üzenetmező",
        chat: "Csevegés",
        navigation: "Navigáció",
        app: "Alkalmazás",
      },
      actions: {
        findInPage: {
          label: "Keresés az oldalon",
          description: "Keresés az oldal szövegében",
        },
        newBrowserTab: {
          label: "Új böngészőlap",
          description: "Új lap megnyitása a csevegés melletti böngészőben",
        },
        toggleBrowserFullView: {
          label: "Böngésző teljes nézetének be- és kikapcsolása",
          description:
            "A böngésző megjelenítése a teljes ablakban, a csevegés fölötte lebeg",
        },
        openSettings: {
          label: "Beállítások megnyitása",
          description: "A beállítások párbeszédpanel megnyitása",
        },
        openKeyboardShortcuts: {
          label: "Billentyűparancsok",
          description: "A billentyűparancsok listájának megnyitása",
        },
        openCommandPalette: {
          label: "Parancspaletta",
          description: "A parancspaletta megnyitása",
        },
        searchChats: {
          label: "Keresés a csevegésekben",
          description: "A csevegéskereső párbeszédpanel megnyitása",
        },
        openMcpServers: {
          label: "MCP-kiszolgálók",
          description: "MCP-kiszolgálók beállítása ehhez a csevegéshez",
        },
        logOut: {
          label: "Kijelentkezés",
          description: "Kijelentkezés az Unslothból",
        },
        approveToolRequest: {
          label: "Kérés jóváhagyása",
          description: "A jóváhagyásra váró eszközhívás engedélyezése",
        },
        declineToolRequest: {
          label: "Kérés elutasítása",
          description: "A jóváhagyásra váró eszközhívás elutasítása",
        },
        newChat: {
          label: "Új csevegés",
          description: "Új csevegés indítása",
        },
        newTemporaryChat: {
          label: "Új ideiglenes csevegés",
          description: "Az előzményekbe nem mentett csevegés indítása",
        },
        newStandaloneChat: {
          label: "Új önálló csevegés",
          description: "Új csevegés indítása projekten kívül",
        },
        archiveChat: {
          label: "Csevegés archiválása",
          description:
            "A kijelölt csevegések vagy az aktuális csevegés archiválása",
        },
        markChatUnread: {
          label: "Megjelölés olvasatlanként",
          description:
            "A kijelölt csevegések vagy az aktuális csevegés megjelölése olvasatlanként",
        },
        togglePinChat: {
          label: "Rögzítés be- és kikapcsolása",
          description:
            "A kijelölt csevegések vagy az aktuális csevegés rögzítése, illetve a rögzítés feloldása",
        },
        selectAllChats: {
          label: "Összes csevegés kijelölése",
          description: "Az oldalsáv összes csevegésének kijelölése",
        },
        clearChatSelection: {
          label: "Kijelölés megszüntetése",
          description:
            "A kijelölt csevegések kijelölésének megszüntetése. Az Escape billentyű is ezt teszi",
        },
        deleteSelectedChats: {
          label: "Kijelölt csevegések törlése",
          description: "Az összes kijelölt csevegés törlése",
        },
        nextRecentlyViewedChat: {
          label: "Következő nemrég megtekintett csevegés",
          description: "Előre lépés a nemrég megtekintett csevegések között",
        },
        previousRecentlyViewedChat: {
          label: "Előző nemrég megtekintett csevegés",
          description: "Visszalépés a nemrég megtekintett csevegések között",
        },
        nextChat: {
          label: "Következő csevegés",
          description: "Váltás az oldalsáv következő csevegésére",
        },
        previousChat: {
          label: "Előző csevegés",
          description: "Váltás az oldalsáv előző csevegésére",
        },
        nextChatNeedingAttention: {
          label: "Következő figyelmet igénylő csevegés",
          description:
            "Váltás a következő generáló, várakozó vagy olvasatlan csevegésre",
        },
        clearAllUnreads: {
          label: "Összes olvasatlan törlése",
          description: "Az összes csevegés megjelölése olvasottként",
        },
        goToRecentChat1: {
          label: "Ugrás az 1. legutóbbi csevegésre",
          description: "A Legutóbbiak első csevegésének megnyitása",
        },
        goToRecentChat2: {
          label: "Ugrás a 2. legutóbbi csevegésre",
          description: "A Legutóbbiak második csevegésének megnyitása",
        },
        goToRecentChat3: {
          label: "Ugrás a 3. legutóbbi csevegésre",
          description: "A Legutóbbiak harmadik csevegésének megnyitása",
        },
        goToRecentChat4: {
          label: "Ugrás a 4. legutóbbi csevegésre",
          description: "A Legutóbbiak negyedik csevegésének megnyitása",
        },
        goToRecentChat5: {
          label: "Ugrás az 5. legutóbbi csevegésre",
          description: "A Legutóbbiak ötödik csevegésének megnyitása",
        },
        goToRecentChat6: {
          label: "Ugrás a 6. legutóbbi csevegésre",
          description: "A Legutóbbiak hatodik csevegésének megnyitása",
        },
        switchToChat: {
          label: "Váltás a Csevegésre",
          description: "Ugrás a csevegési munkaterületre",
        },
        switchToProjects: {
          label: "Váltás a Projektekre",
          description: "Ugrás a projektek munkaterületére",
        },
        switchToHub: {
          label: "Váltás a Modellközpontra",
          description: "Ugrás a modellközpontba",
        },
        switchToTrain: {
          label: "Váltás a Tanításra",
          description: "Ugrás a tanítási munkaterületre",
        },
        switchToRecipes: {
          label: "Váltás a Receptekre",
          description: "Ugrás a Data Recipes oldalra",
        },
        switchToImages: {
          label: "Váltás a Képekre",
          description: "Ugrás a képek munkaterületére",
        },
        switchToVideo: {
          label: "Váltás a Videóra",
          description: "Ugrás a videó munkaterületére",
        },
        switchToAudio: {
          label: "Váltás a Hangra",
          description: "Ugrás a hang munkaterületére",
        },
        switchToExport: {
          label: "Váltás az Exportálásra",
          description: "Ugrás az exportálási munkaterületre",
        },
        toggleSidebar: {
          label: "Oldalsáv be- és kikapcsolása",
          description: "Az oldalsáv megjelenítése vagy elrejtése",
        },
        toggleApiMonitor: {
          label: "API-tevékenység be- és kikapcsolása",
          description: "Az API-tevékenységfigyelő megjelenítése vagy elrejtése",
        },
        openModelPicker: {
          label: "Modellválasztó megnyitása",
          description: "A csevegés modelljének kiválasztása",
        },
        openProjectPicker: {
          label: "Projektválasztó megnyitása",
          description: "Váltás másik projektre a csevegés fejlécéből",
        },
        startDictation: {
          label: "Diktálás",
          description: "Diktálás indítása vagy leállítása az üzenetmezőbe",
        },
        attachFiles: {
          label: "Fotók és fájlok csatolása",
          description: "Melléklet hozzáadása az üzenetmezőhöz",
        },
        openComposerMenu: {
          label: "A + menü megnyitása",
          description:
            "Az üzenetmező eszközöket és mellékleteket tartalmazó + menüjének megnyitása",
        },
        sendMessage: {
          label: "Üzenet küldése",
          description: "Az üzenetmező tartalmának elküldése",
        },
        queueMessage: {
          label: "Üzenet sorba állítása",
          description: "A piszkozat a promptsor végére kerül",
        },
        steerMessage: {
          label: "Válasz irányítása",
          description:
            "Az aktuális válasz leállítása és a piszkozat elküldése következőként",
        },
        stopGeneration: {
          label: "Generálás leállítása",
          description: "Az éppen íródó válasz leállítása",
        },
        cycleReasoningEffort: {
          label: "Gondolkodási szint váltása",
          description: "Lépkedés a gondolkodási szintek között",
        },
        increaseReasoningEffort: {
          label: "Gondolkodási szint növelése",
          description: "A gondolkodási szint emelése egy fokozattal",
        },
        decreaseReasoningEffort: {
          label: "Gondolkodási szint csökkentése",
          description: "A gondolkodási szint csökkentése egy fokozattal",
        },
        renameChat: {
          label: "Csevegés átnevezése",
          description: "Az aktuális csevegés átnevezése",
        },
        forkChat: {
          label: "Csevegés elágaztatása",
          description: "Új csevegés elágaztatása a legutóbbi üzenettől",
        },
        copyChatAsMarkdown: {
          label: "Másolás Markdownként",
          description:
            "A teljes csevegés másolása a vágólapra Markdown formátumban",
        },
        copySessionId: {
          label: "Munkamenet-azonosító másolása",
          description: "A csevegés sandbox-munkamenet-azonosítójának másolása",
        },
      },
    },
    sandbox: {
      title: "Biztonság",
      description:
        "A Python- és Terminál-eszközhívások operációsrendszer-szintű sandboxban fussanak-e ezen a számítógépen.",
      docs: "Dokumentáció",
      docsLabel: "A sandbox dokumentációjának megnyitása",
      toolsSection: "Ez a számítógép",
      refresh: "Frissítés",
      python: "Python",
      terminal: "Terminál",
      osIsolation: "OS-sandbox ({backend})",
      softwareSafeguards: "Szoftveres védelem",
      noReason:
        "Itt még egyik OS-sandbox sem működik, ezért az Unsloth minden hívást egyenként ellenőriz.",
      runsInCmd:
        "A cmd.exe-ben fut, mivel a Git Bash nem indul el a Windows sandboxban.",
      loadError: "Nem sikerült betölteni a sandbox állapotát",
      saveError: "Nem sikerült menteni a sandbox beállítását",
      prepareError: "Nem sikerült előkészíteni ezt a számítógépet",
      unsupported:
        "Ez az Unsloth-kiszolgáló még nem jelenti a sandbox állapotát. Frissítse az Unslothot, hogy itt kezelhesse.",
      windowsSection: "Windows sandbox (előzetes verzió)",
      windowsDescription:
        "A Microsoft MXC sandboxa. A Windowsba beépített konténert használja, vagy egy tartalékmegoldást az ilyen konténer nélküli Windows-verziókon.",
      builtinLabel: "Beépített konténer",
      builtinDescription:
        "A Windows sandbox a Windows beépített konténerében fut, rendszergazdai lépés nélkül.",
      builtinInUse: "Használatban",
      optInLabel: "Tartalék sandbox használata",
      optInDescription:
        "Beépített konténer nélküli Windows-verziókhoz. Ekkor a Python és a Terminál csak a csevegés mappájában lévő fájlokat módosíthatja.",
      optInNotNeeded:
        "Ezen a számítógépen nincs rá szükség, mivel a beépített konténer működik.",
      disclosure:
        "A tartalékmegoldáshoz egyszeri rendszergazdai lépés (A számítógép előkészítése) szükséges, amelyet minden újraindítás után meg kell ismételni. Amíg egy eszköz fut, az Unsloth ideiglenes hozzáférést ad neki a szükséges mappákhoz. Az MXC még előzetes verzió, nem biztonsági határ.",
      lockedDacl: "Beállítva: UNSLOTH_MXC_ALLOW_DACL_FALLBACK",
      grantsLabel: "Gyorsabb eszközindítás",
      grantsDescription:
        "Megtartja az írásvédett hozzáférést az Unsloth saját futtatókörnyezeti mappáihoz a hívások között. Kikapcsolva minden hívás néhány másodperccel lassabban indul.",
      lockedGrants: "Beállítva: UNSLOTH_MXC_PERSISTENT_READ_GRANTS",
      memoryLabel: "Memóriakorlát",
      memoryDescription:
        "Az egy sandboxolt Python- vagy Terminál-hívás által lefoglalható legtöbb memória. A következő hívástól érvényes. Alapértelmezés: {defaultSize} GB.",
      memoryLocked: "Beállítva: UNSLOTH_STUDIO_SANDBOX_AS_GB",
      memoryInvalid: "Adjon meg egy egész számot {min} és {max} között.",
      restored: "Hozzáférés eltávolítva {count} mappából.",
      hostPrepLabel: "Rendszergazdai beállítás",
      prepPrepared: "Kész",
      prepNeeds: "Szükséges",
      prepNeedsAgain: "A legutóbbi újraindítás után ismét szükséges",
      prepUnknown: "Ismeretlen",
      prepareButton: "A számítógép előkészítése",
      preparing: "Várakozás a rendszergazdai engedélykérésre…",
      prepSucceeded: "A számítógép előkészítve.",
      prepDeclined: "A rendszergazdai engedélykérést elutasították.",
      prepFailed: "Az előkészítés sikertelen.",
      setupLabel: "OS-sandbox",
      setupCommandHint: "Futtassa ezt az Unslothot futtató számítógépen:",
      macosBuiltIn:
        "A macOS beépített sandboxszal rendelkezik, így nincs mit telepíteni.",
      setupKeywords:
        "telepítés beállítás install setup bubblewrap bwrap seatbelt mxc sudo pkexec apparmor uac",
      setupSucceeded: "Az OS-sandbox használatra kész.",
      runtimeMissing:
        "Még nincs telepítve. A telepítéshez nem kell rendszergazdai engedélykérés.",
      runtimeLabel: "Windows sandbox futtatókörnyezete",
      installRuntime: "Futtatókörnyezet telepítése",
      installingRuntime: "Telepítés…",
      installRuntimeFailed: "A futtatókörnyezet telepítése sikertelen.",
      installRuntimeError:
        "Nem sikerült elindítani a futtatókörnyezet telepítését.",
      unsupportedArch:
        "A Windows sandbox (MXC) csak x64 és ARM64 Windows rendszeren fut, ezért ezen a számítógépen nem érhető el.",
      builtinNeedsUpdate:
        "A Windows beépített konténeréhez legalább ez a build szükséges: {required}, ezen a számítógépen pedig ez fut: {current}. Telepítse a legújabb Windows-frissítést, hogy rendszergazdai lépés nélkül használhassa, vagy kapcsolja be az alábbi tartalékmegoldást.",
      unsupportedBuild:
        "A Windows sandboxhoz (MXC) Windows 11 24H2 (26100-as build) vagy újabb szükséges, ezért ezen a számítógépen nem érhető el.",
      permissionLabel: "Eszközengedélyek",
      levelRowLabel: "Sandbox szintje",
      levelPickerTitle: "Hogyan fusson a kód sandboxban?",
      levelOff: "Letiltva",
      levelOffShort: "A teljes hozzáférés kikapcsolja a sandboxot",
      levelHighDetail:
        "A Python és a Terminál az OS-sandboxban (bubblewrap, Seatbelt vagy MXC) fut, a szoftveres sandbox mellett. Működő OS-sandbox nélkül a kockázatos hívások előbb engedélyt kérnek.",
      levelLowDetail:
        "A Python és a Terminál csak szoftveres sandboxszal fut, az OS-sandboxon kívül. A kockázatos hívások előbb engedélyt kérnek, hacsak nem a Teljes hozzáférést választotta.",
      levelLabel: "Sandbox",
      levelHigh: "Magas",
      levelLow: "Alacsony",
      levelHighShort: "OS-szintű elkülönítést ad a Pythonhoz és a Terminálhoz",
      levelLowShort: "Csak szoftveres. A kockázatos kód előbb engedélyt kér",
      levelFullAccessNote:
        "A teljes hozzáférés kikapcsolja a sandboxot. Ez a beállítás akkor lép ismét érvénybe, ha másik szintet választ.",
      learnMore: "További információ",
      managedNote:
        "Az OS-sandboxot ennek az Unsloth-kiszolgálónak a tulajdonosa állítja be.",
    },
    debugging: {
      description:
        "Az Unsloth naplóinak olvasása az alkalmazás elhagyása nélkül.",
      logSection: "Naplófájl",
      source: "Naplófájl",
      sourceHint:
        "A modellfuttatók saját naplót írnak, így a sikertelen betöltés vagy generálás oka gyakran ott található, nem a kiszolgáló naplójában.",
      path: "Hely",
      pathCopy: "Elérési út másolása",
      currentSession: "Aktuális",
      statusLive: "Élő",
      statusPaused: "Szüneteltetve",
      statusStale: "Elavult",
      filterPlaceholder: "Sorok szűrése",
      lineCount: "{count} sor",
      filteredLineCount: "{shown} / {total} sor",
      wrapLines: "Sortörés",
      jumpToLatest: "Legújabb",
      resizeLog: "Húzza a napló átméretezéséhez",
      noMatches: "Egy sor sem felel meg a szűrőnek.",
      mode: "Mód",
      modeLive: "Élő",
      modeInterval: "3 másodpercenként",
      modeManual: "Kézi",
      refreshNow: "Frissítés most",
      privacyNote:
        "A hitelesítő adatok ebben a nézetben el vannak takarva. A lemezen lévő fájlban nincsenek eltakarva.",
      copyVisible: "Látható napló másolása",
      empty: "Még nincs naplóbejegyzés.",
      disabled:
        "A fájlba naplózás ki van kapcsolva (UNSLOTH_STUDIO_NO_FILE_LOG=1).",
      missing: "Nem található naplófájl.",
      unreadable: "A naplófájlt nem sikerült beolvasni.",
      viewLogs: "Naplók megtekintése",
      timeout:
        "A naplókérés túllépte az időkorlátot. Lehet, hogy a kiszolgáló nem érhető el.",
      droppedNotice:
        "Néhány sor kimaradt: a napló gyorsabban íródott, mint ahogy be lehetett olvasni.",
      morePending:
        "További sorok beolvasása folyamatban; a következő frissítéskor jelennek meg.",
      staleSession:
        "A fájlba naplózás ki van kapcsolva, ezért ez egy korábbi munkamenet, és nem frissül.",
      downloadAllLogs: "Összes napló letöltése (.zip)",
      downloadingAllLogs: "Naplók csomagolása...",
      exportMaskedNote:
        "Az exportált fájlokban a hitelesítő adatok el vannak takarva. A nagyon nagy naplókból csak a legutóbbi sorok maradnak meg, és egyes naplók teljesen kimaradhatnak; lásd az archívumban az EXPORT_WARNINGS.txt fájlt.",
      downloadedTo: "Mentve ide: {path}",
      downloadedToBrowser: "A letöltés elindult.",
      showInFolder: "Megjelenítés a mappában",
      openLogsFolder: "Naplómappa megnyitása",
      openLogsFolderFailed: "Nem sikerült megnyitni a naplómappát.",
      exportFailed: "Nem sikerült letölteni a naplókat.",
      exportTooOld:
        "A futó Unsloth-háttérrendszer túl régi a naplók exportálásához. Frissítse a háttérrendszert, és indítsa újra.",
      exportForbidden:
        "Az összes napló letöltéséhez bejelentkezett Unsloth-munkamenet szükséges. Az API-kulcs nem elegendő.",
      keywords:
        "hibakeresés napló naplók hiba hibák összeomlás veremkiírás hibaelhárítás diagnosztika sikertelen debug log logs error crash traceback stack trace troubleshoot diagnostics failed",
    },
    voice: {
      title: "Hang",
      description: "Mikrofon, diktálás, beszédfelismerés és felolvasás",
      dictation: {
        audioUploadTitle: "Diktálás felvétellel",
        audioUploadDescription:
          "A böngésző biztonságos kapcsolatot (HTTPS) követel meg ahhoz, hogy közvetlenül az Unslothban rögzíthessen. Ezen a HTTP-kapcsolaton használja inkább a telefonja hangrögzítőjét, vagy válasszon egy mentett felvételt. Az Unsloth-kiszolgáló átírja a hangot, és hozzáadja a szöveget az üzenethez.",
        audioUploadChooseFile: "Felvétel kiválasztása",
        audioUploadRecord: "Hangfelvétel",
        audioUploadIphoneHint:
          "iPhone-on mentsen egy felvételt a Diktafon vagy egy másik hangrögzítő alkalmazásból, majd válassza ki itt.",
        audioUploadServerModelNote:
          "Ez a felvétel a fent látható helyi modellt használja az Unsloth-kiszolgálón, akkor is, ha a normál diktálás más motort használ.",
        audioUploadAutomatic: "Automatikus",
        audioUploadModelReady: "Készen áll ezen az Unsloth-kiszolgálón",
        audioUploadRetryTitle: "Nem sikerült átírni: {file}",
        audioUploadRetry: "Átírás újrapróbálása",
        audioUploadChooseModel:
          "Válasszon helyi beszédfelismerő modellt a Hang beállításokban.",
        audioUploadEmpty: "A kiválasztott felvétel üres.",
        audioUploadTooLarge: "Válasszon ennél kisebb felvételt: {size}.",
        audioUploadVideoUnsupported: "Hangfelvételt válasszon, ne videót.",
        audioUploadNoSpeech: "A modell nem hallott beszédet a felvételen.",
        audioUploadFailed: "A felvételt nem sikerült átírni.",
        audioUploadCancel: "Átírás megszakítása",
        audioUploadTranscribing: "Hang átírása…",
        sectionTitle: "Diktálás",
        engineLabel: "Diktálómotor",
        engineBrowser: "Böngésző",
        engineBrowserDescription:
          "A hangot a böngésző beszédszolgáltatásával írja át. STT-modell használatához válassza a „Helyi átírás” lehetőséget.",
        engineModel: "Helyi átírás",
        engineModelDescription:
          "Helyileg futtat egy beszédfelismerő (STT) modellt, és internet nélkül is működik. Letöltés, betöltés, majd tétlenség után kiürítés.",
        engineCustom: "Egyéni végpont",
        engineCustomDescription:
          "A rögzített hangot egy OpenAI-kompatibilis STT-kiszolgálónak küldi a Kapcsolatok közül.",
        connectionLabel: "Kapcsolat",
        connectionDescription:
          "OpenAI-kompatibilis kiszolgálót és opcionális API-kulcsot a Kapcsolatok lapon adhat hozzá.",
        connectionPlaceholder: "Válasszon kapcsolatot",
        connectionEmpty: "Nincs elérhető kapcsolat",
        customModelLabel: "Modell",
        customModelDescription:
          "A /v1/audio/transcriptions végpontnak küldött modellnév.",
        sttModelLabel: "Beszédfelismerő modell",
        sttModelDescription:
          "Válasszon vagy keressen helyileg futtatandó STT-modellt.",
        sttQuantLabel: "Kvantálás",
        sttQuantDescription:
          "Válasszon vagy keressen helyileg futtatandó STT-modellt, majd annak kvantálását.",
        sttDeviceLabel: "Betöltés ide",
        sttDeviceAuto: "GPU, ha elérhető",
        sttDeviceCpu: "CPU RAM",
        sttDeviceAutoDescription:
          "A GPU használata, ha van, egyébként a CPU-é.",
        sttDeviceCpuDescription:
          "A modell a rendszer RAM-jában marad. Az átírás lassabb, de nem használ GPU-memóriát.",
        sttModelSearchPlaceholder: "Bármely modell keresése a HF-en",
        sttModelSearching: "Keresés a Hugging Face-en…",
        sttModelValidating: "Whisper-kompatibilitás ellenőrzése…",
        sttModelNoResults: "Nem található Whisper-modell",
        sttModelInvalid: "Ez a tároló nem használható diktáláshoz",
        sttModelFailed: "Nem sikerült betölteni az STT-modellt",
        sttModelUnsupported: "Ez a böngésző nem támogatja a hangrögzítést",
        sttChecking: "Ellenőrzés…",
        sttOnDemand: "Letöltve",
        sttLoadingModel: "Modell betöltése…",
        sttReady: "Betöltve: {device}",
        sttLoaded: "Betöltve",
        sttUnavailable:
          "Nincs telepítve ezen a kiszolgálón. A helyi diktálás engedélyezéséhez futtassa ezt: `unsloth studio update`.",
        sttRetry: "Újrapróbálkozás",
        sttDownloadChecking: "Letöltési állapot ellenőrzése…",
        sttNotDownloaded: "Nincs letöltve",
        sttDownloadStatusFailed:
          "Nem sikerült ellenőrizni a letöltési állapotot",
        sttDownload: "Letöltés",
        sttDownloadConfirmTitle: "Letölti a következő modellt: {model}?",
        sttDownloadConfirmBody:
          "A helyi diktálás teljesen internet nélkül fut, de előbb szüksége van erre a beszédmodellre: {model}. Kb. {size}, amely egyszer töltődik le a Hugging Face-gyorsítótárba.",
        sttDownloadConfirmBodyUnsized:
          "A helyi diktálás teljesen internet nélkül fut, de előbb szüksége van erre a beszédmodellre: {model}. Egyszer töltődik le a Hugging Face-gyorsítótárba.",
        sttOpenVoiceSettings: "Hangbeállítások megnyitása",
        sttDownloadStarted: "Letöltés: {model}",
        sttDownloading: "Letöltés… {progress}%",
        sttCancelDownload: "Mégse",
        sttCancellingDownload: "Megszakítás…",
        sttCancelDownloadFailed: "Nem sikerült megszakítani a letöltést",
        sttDownloadComplete: "A beszédfelismerő modell letöltve",
        sttModelReady: "Diktálásra kész: {model}",
        sttRecommended: "Ajánlott",
        sttDownloadFailed: "Nem sikerült letölteni a beszédfelismerő modellt",
        sttLoad: "Betöltés",
        sttUnload: "Kiürítés",
        sttUnloading: "Kiürítés…",
        microphoneLabel: "Mikrofon",
        microphoneFallbackName: "Mikrofon {index}",
        microphoneDescription: "Diktáláshoz használatos",
        microphoneFallbackHint:
          "Diktáláshoz használatos. Ha a böngésző beszédmotorja nem tudja használni ezt az eszközt, a rendszer alapértelmezett eszköze lép a helyébe",
        microphoneGrantDescription:
          "Engedélyezze a mikrofonhozzáférést az eszköznevek megjelenítéséhez",
        allowMicrophone: "Mikrofon engedélyezése",
        micAccessBlocked:
          "A mikrofonhozzáférés le van tiltva. Engedélyezze a mikrofonhozzáférést ehhez az Unsloth-oldalhoz, majd próbálja újra.",
        micAccessBlockedDesktop:
          "A mikrofonhozzáférés le van tiltva. Próbálja újra, és válassza az Engedélyezés lehetőséget, vagy engedélyezze a mikrofont a rendszer adatvédelmi beállításaiban.",
        micAccessUnsupported:
          "Ebben a böngészőben vagy környezetben a mikrofonhozzáférés nem támogatott.",
        systemDefault: "Rendszer alapértelmezése",
        savedMicDisconnected: "Mentett mikrofon (nincs csatlakoztatva)",
        languageLabel: "Diktálás nyelve",
        languageDescription: "A felismerendő nyelv",
        languageAuto: "Automatikus (a böngésző nyelve)",
        languageAutoDetect: "Automatikus (nyelvfelismerés)",
      },
      dictionary: {
        sectionTitle: "Diktálási szótár",
        sectionDescription:
          "Megadhatja, hogyan írja le a diktálás az egyes szavakat vagy kifejezéseket",
        manageLabel: "Egyéni írásmódok",
        manage: "Kezelés",
        backToVoice: "Vissza a Hang beállításokhoz",
        addEntry: "Bejegyzés hozzáadása",
        newEntryAria: "Új szótárbejegyzés",
        entryPlaceholder: "Kovács Anna",
        entryAria: "Szótárbejegyzés {index}",
        removeEntryAria: "Szótárbejegyzés eltávolítása: {index}",
      },
      recents: {
        sectionTitle: "Diktálási előzmények",
        sectionDescription:
          "Minden diktálás ide mentődik, így a szöveg visszaállítható",
        manageLabel: "Diktálási előzmények",
        manage: "Kezelés",
        pageDescription:
          "Minden diktálás mentésre kerül. Megtekintheti, másolhatja vagy törölheti őket, illetve megnyithatja a csevegést, amelyben egy diktálást használt.",
        searchPlaceholder: "Diktálások keresése",
        sortLabel: "Diktálások rendezése",
        sortNewest: "Legújabb",
        sortOldest: "Legrégebbi",
        sortAlpha: "A–Z",
        noMatches: "Egy diktálás sem felel meg a keresésnek",
        detailTitle: "Mentett diktálás",
        backToVoice: "Vissza a Hang beállításokhoz",
        backToRecents: "Vissza a legutóbbi diktálásokhoz",
        view: "Teljes diktálás megtekintése",
        empty: "Még nincs diktálás",
        dictationColumn: "Diktálás",
        dateColumn: "Létrehozás dátuma",
        copy: "Diktálás másolása",
        copied: "Vágólapra másolva",
        copyFailed: "Nem sikerült a vágólapra másolni",
        delete: "Diktálás törlése",
        deleteTitle: "Diktálás törlése",
        deleteDescription:
          "Törli ezt a mentett diktálást? Ez a művelet nem vonható vissza.",
        deleteLinkedDescription:
          "Törli ezt a mentett diktálást? A csevegést is törölheti, amelyben használta. Ez a művelet nem vonható vissza.",
        deleteWithChat: "Csevegés és diktálás törlése",
        deleteWithChatFailed: "Nem sikerült törölni a csevegést",
        clear: "Előzmények törlése",
        clearTitle: "Diktálási előzmények törlése",
        clearDescription:
          "Törli az összes mentett diktálást? Ez a művelet nem vonható vissza.",
        clearConfirm: "Összes törlése",
        showMore: "Több megjelenítése ({count})",
        openChat: "Csevegés megnyitása",
      },
      readAloud: {
        sectionTitle: "Felolvasás",
        buttonLabel: "Felolvasás gomb",
        buttonDescription: "Megjelenítés az asszisztens válaszain",
        autoLabel: "Válaszok automatikus felolvasása",
        autoDescription: "Minden válasz felolvasása, amint elkészül",
        engineLabel: "TTS-motor",
        engineSystemDescription: "Az eszköz beépített hangjai",
        engineStudioDescription:
          "A Hang › Szövegfelolvasás oldalon betöltött modellel beszél",
        engineSystem: "Rendszerhangok",
        engineStudio: "Unsloth-modell",
        engineCustom: "Egyéni végpont",
        engineCustomDescription:
          "OpenAI-kompatibilis TTS-kiszolgáló a kapcsolatai közül (pl. Kokoro)",
        connectionLabel: "Kapcsolat",
        connectionDescription:
          "OpenAI-kompatibilis kiszolgálót a Kapcsolatok lapon adhat hozzá",
        connectionPlaceholder: "Válasszon kapcsolatot",
        customModelLabel: "Modell",
        customVoiceDescription:
          "A végpont által várt hangnév; alapértelmezés: alloy",
        modelLabel: "TTS-modell",
        modelDescription:
          "Nincs betöltve. Töltsön be egyet a Hang › Szövegfelolvasás oldalon.",
        modelLoaded: "Betöltve: {model}",
        keepBothLabel: "A csevegési modell betöltve tartása",
        keepBothDescription:
          "A Több modell betöltve tartása beállítás ki van kapcsolva, ezért egy beszédmodell betöltése kiüríti a csevegési modellt.",
        keepBothAction: "Mindkettő betöltve tartása",
        openAudioAction: "Hang megnyitása",
        voiceLabel: "Hang",
        voiceDescription: "Az eszköz legjobb hangjai",
        studioVoiceDescription:
          "A Hang oldalon mentett hangokhoz klónozó modell szükséges",
        studioVoiceDefault: "A modell saját hangja",
        studioVoiceSaved: "Mentett hang",
        speedLabel: "Sebesség",
        pitchLabel: "Hangmagasság",
        volumeLabel: "Hangerő",
        previewLabel: "Hang előnézete",
        previewDescription: "Rövid minta lejátszása",
        previewFailed: "A TTS-előnézet sikertelen",
        previewAction: "Előnézet",
        preparingAction: "Generálás…",
        stopAction: "Leállítás",
        ttsLabel: "Szövegfelolvasás",
        notSupported: "Ez a böngésző nem támogatja",
      },
    },
    general: {
      title: "Általános",
      description: "Az Unsloth általános beállításai.",
      account: "Fiók",
      huggingFaceToken: "Hugging Face-token",
      huggingFaceTokenDescription:
        "Korlátozott hozzáférésű modellek betöltéséhez és összetevők feltöltéséhez.",
      hideToken: "Token elrejtése",
      showToken: "Token megjelenítése",
      clearToken: "Törlés",
      checkingToken: "Token ellenőrzése...",
      tokenValidated: "Token ellenőrizve",
      password: "Jelszó",
      passwordDescription: "Az Unsloth-fiók jelszavának módosítása.",
      passwordDialog: {
        trigger: "Jelszó módosítása",
        title: "Jelszó módosítása",
        description:
          "Adja meg a jelenlegi jelszavát, és válasszon újat (legalább {minLength} karakter).",
        setTrigger: "Távoli jelszó beállítása",
        setTitle: "Távoli jelszó beállítása",
        setDescription:
          "Válassza ki a jelszót, amellyel a távoli böngészők unsloth néven bejelentkeznek (legalább {minLength} karakter). Az Unsloth asztali alkalmazás továbbra is automatikusan bejelentkezik.",
        setSubmit: "Jelszó beállítása",
        setting: "Beállítás...",
        setDone: "Jelszó beállítva.",
        currentPassword: "Jelenlegi jelszó",
        newPassword: "Új jelszó",
        confirmPassword: "Új jelszó megerősítése",
        currentTooShort:
          "A jelenlegi jelszónak legalább {minLength} karakterből kell állnia.",
        newTooShort:
          "Az új jelszónak legalább {minLength} karakterből kell állnia.",
        newHasSpaces: "Az új jelszó nem tartalmazhat szóközt.",
        mismatch: "A jelszavak nem egyeznek.",
        samePassword: "Az új jelszónak különböznie kell a jelenlegi jelszótól.",
        update: "Jelszó frissítése",
        updating: "Frissítés...",
        updated: "Jelszó frissítve.",
        updateFailed: "A jelszó frissítése sikertelen.",
      },
      chatDefaults: "Csevegés alapértelmezései",
      autoTitleNewChats: "Új csevegések automatikus elnevezése",
      autoTitleNewChatsDescription: "Cím generálása az első üzenetből.",
      helperLlm: {
        sectionTitle: "Segéd-LLM",
        preloadOnStartup: "Segéd-LLM előzetes letöltése indításkor",
        preloadOnStartupDescription:
          "Az AI Assist segédmodell letöltése a háttérben indításkor. Alapértelmezés szerint ki van kapcsolva; az AI Assist szükség esetén így is letöltheti.",
        disabledByEnv:
          "Letiltva a háttérrendszer környezetében beállított UNSLOTH_HELPER_MODEL_DISABLE által.",
        loadError: "Nem sikerült betölteni a Segéd-LLM beállításait.",
        saveError: "Nem sikerült menteni a Segéd-LLM beállításait.",
      },
      developerMode: {
        sectionTitle: "Fejlesztői mód",
        videoCpu: "Videógenerálás engedélyezése CPU-n (kísérleti)",
        videoCpuDescription:
          "Lehetővé teszi a Videó oldal futtatását támogatott GPU nélküli gépen. Alapértelmezés szerint ki van kapcsolva; nem támogatott és nagyon lassú.",
        notApplicable:
          "Ezen a gépen nincs hatása: a videó vagy már a GPU-n fut, vagy olyasmi akadályozza, amit a CPU nem tud megkerülni.",
        forcedOnByEnv:
          "Bekapcsolva a háttérrendszer környezetében beállított UNSLOTH_VIDEO_ALLOW_CPU által.",
        forcedOffByEnv:
          "Kikapcsolva a háttérrendszer környezetében beállított UNSLOTH_VIDEO_ALLOW_CPU által.",
        confirmTitle: "Engedélyezi a CPU-s videógenerálást?",
        confirmWarning:
          "A CPU-s videógenerálás órákig tarthat, nagy mennyiségű RAM-ot foglalhat, és lefagyaszthatja a rendszert. Csak saját felelősségére folytassa.",
        ramBelowMinimum:
          "A gépben {host} GB RAM van, ami kevesebb, mint a legkisebb videómodell által CPU-n igényelt {min} GB. A betöltés valószínűleg sikertelen lesz, vagy lefagyasztja a rendszert.",
        ramOk:
          "A gépben {host} GB RAM van. A legkisebb videómodellnek CPU-n körülbelül {min} GB kell; a nagyobb modelleknek jóval több.",
        ramUnknown:
          "A legkisebb videómodellnek CPU-n körülbelül {min} GB RAM kell; a nagyobb modelleknek jóval több.",
        performanceEstimate:
          "Zajtalanítási lépésenként percekre számítson: egy 5 képkockás, 832x480-as klip a Wan2.1 1.3B modellel 3 lépésben körülbelül 4 percig tartott 16 CPU-szálon, egy teljes hosszúságú klip alapértelmezett beállításokkal pedig órákig tarthat.",
        confirmAction: "Engedélyezés mégis",
        loadError: "Nem sikerült betölteni a Fejlesztői mód beállításait.",
        saveError: "Nem sikerült menteni a Fejlesztői mód beállításait.",
      },
      modelAutoSwitch: {
        sectionTitle: "Automatikus modellváltás (OpenAI API)",
        enable: "Modellváltás kérés alapján",
        enableDescription:
          "Az API-kérésben megnevezett letöltött modell betöltése a kiszolgálás előtt. Alapértelmezés szerint ki van kapcsolva.",
        autoDownload: "Hiányzó modellek letöltése",
        autoDownloadDescription:
          "Az API-kérésben megnevezett, még le nem töltött GGUF letöltése. Így bárki, akinek API-kulcsa van, lemezterületet és sávszélességet használhat.",
        idleUnload: "Automatikus kiürítés tétlenség esetén",
        idleUnloadDescription:
          "A VRAM felszabadítása ennyi másodpercnyi tétlenség után. 0 esetén betöltve marad, a minimum 60.",
        idleSecondsAriaLabel:
          "Automatikus kiürítés tétlenség esetén (másodperc)",
        mediaEnable: "Kép- és videómodell váltása kérés alapján",
        mediaEnableDescription:
          "Az API-kérésben megnevezett letöltött kép- vagy videómodell betöltése a generálás előtt. Független a csevegési modell váltásától. Alapértelmezés szerint ki van kapcsolva.",
        mediaIdleUnload:
          "Automatikus kiürítés tétlenség esetén képhez és videóhoz",
        mediaIdleUnloadDescription:
          "A VRAM felszabadítása a kép- és videómodellek kiürítésével ennyi másodpercnyi tétlenség után. Független a csevegési modell váltásától. 0 esetén betöltve maradnak, a minimum 60.",
        mediaIdleSecondsAriaLabel:
          "Kép és videó automatikus kiürítése tétlenség esetén (másodperc)",
        mediaIdlePaused:
          "Szünetel, amíg a Modell megtartása a GPU-memóriában beállítás be van kapcsolva.",
        idleNeedsEnable:
          "Előbb kapcsolja be a Modellváltás kérés alapján beállítást.",
        idleActiveViaEnv: "Aktív az UNSLOTH_MODEL_IDLE_TTL révén.",
        loadError:
          "Nem sikerült betölteni az automatikus modellváltás beállításait.",
        saveError:
          "Nem sikerült menteni az automatikus modellváltás beállításait.",
        idleError:
          "Adjon meg 0-t a modell betöltve tartásához, vagy legalább 60 másodpercet.",
        keepKv: "Csevegési környezet megőrzése kiürítéskor",
        keepKvDescription:
          "A KV-gyorsítótár mentése tétlenség miatti kiürítés előtt, vagy mielőtt egy kép- vagy videómodell átveszi a GPU-t, így a folytatott csevegéseknek nem kell újra beolvasniuk az előzményeket. Legfeljebb 10 GB a lemezen.",
        keepKvKeywords:
          "kv gyorsítótár környezet csere folytatás kép videó cache context swap resume image video",
        apiOnly: "Csak az API által betöltött modellek kiürítése",
        apiOnlyDescription:
          "A tétlenség miatti kiürítés a memóriában hagyja az Unslothból betöltött modellt, és csak az API-kérés által betöltötteket szabadítja fel.",
      },
      modelAliases: {
        sectionTitle: "Modellaliasok (OpenAI API)",
        description:
          "Állandó nevek API-kliensekhez, például fast-chat vagy code. Az aliast megnevező kérést az a modell szolgálja ki, amelyre az alias mutat, így a csatlakoztatott alkalmazások modellcsere után is működnek. Kapcsolja be a Modellváltás kérés alapján beállítást, hogy az alias be tudja tölteni a modelljét.",
        namePlaceholder: "fast-chat",
        nameAriaLabel: "Alias neve",
        modelAriaLabel: "A modell, amelyre az alias mutat",
        modelPlaceholder: "unsloth/Qwen3-4B-GGUF:Q4_K_M",
        removeAriaLabel: "Alias eltávolítása",
        add: "Alias hozzáadása",
        loadError: "Nem sikerült betölteni a modellaliasokat.",
        saveError: "Nem sikerült menteni a modellaliasokat.",
      },
      previewSharing: {
        sectionTitle: "Előnézet megosztása",
        enableLabel: "Nyilvános előnézeti hivatkozások",
        enableDescription:
          "Bárki, akinek aláírt hivatkozása van, bejelentkezés nélkül csevegést folytathat egy kész modellel. Kikapcsolva a nyilvános előnézeti felület elérhetetlenné válik; a megosztott hivatkozások nem működnek tovább.",
        loadError:
          "Nem sikerült betölteni az előnézet-megosztási beállításokat.",
        saveError: "Nem sikerült menteni az előnézet-megosztási beállításokat.",
        revokeLabel: "Összes előnézeti hivatkozás visszavonása",
        revokeDescription:
          "Az aláíró titkos kulcs cseréje, így minden eddig megosztott hivatkozás működésképtelenné válik. Az újonnan másolt hivatkozások működnek.",
        revokeAction: "Hivatkozások visszavonása",
        revoking: "Visszavonás...",
        revokeConfirmTitle: "Visszavonja az összes előnézeti hivatkozást?",
        revokeConfirmDescription:
          "Minden megosztott előnézeti hivatkozás azonnal működésképtelenné válik. Ez a művelet nem vonható vissza.",
        revokeConfirmAction: "Összes hivatkozás visszavonása",
        revoked: "Az összes előnézeti hivatkozás visszavonva",
        revokeError: "Nem sikerült visszavonni az előnézeti hivatkozásokat",
      },
      managedProviderUrls: {
        sectionTitle: "Felügyelt fiókok",
        enableLabel: "Helyi és hálózati kapcsolatok",
        enableDescription:
          "A felügyelt fiókok kapcsolatai helyi vagy hálózati címekre mutathatnak, például egy ezen a számítógépen vagy a helyi hálózaton futó Ollama- vagy llama.cpp-kiszolgálóra. Alapértelmezés szerint ki van kapcsolva, mert így ezek a fiókok elérhetik a számítógépén és a hálózatán futó szolgáltatásokat.",
        lockedByEnvironment:
          "Beállítva ezen a kiszolgálón: UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1, amely minden fióknál elutasítja a privát címeket.",
        loadError:
          "Nem sikerült betölteni a felügyelt fiókok kapcsolatbeállításait.",
        saveError:
          "Nem sikerült menteni a felügyelt fiókok kapcsolatbeállításait.",
      },
      permissions: {
        sectionTitle: "Engedélyek",
        names: {
          ask: "Jóváhagyás kérése",
          auto: "Jóváhagyás helyettem",
          off: "Automatikus futtatás",
          full: "Teljes hozzáférés",
        },
        details: {
          ask: "Minden eszközhívás előtt rákérdez, beleértve a terminál- és Python-kódot, a webes kereséseket, a fájlszerkesztéseket és az MCP-eszközöket. A külső szolgáltató által futtatott eszközök nem szünetelnek.",
          auto: "A rutinszerű eszközhívásokat önállóan futtatja, és csak akkor kérdez rá, ha egy művelet kockázatosnak tűnik, például hitelesítő adatok olvasása, jogosultságok emelése vagy romboló parancsok futtatása esetén.",
          off: "Minden eszközhívást rákérdezés nélkül futtat. A Python- és terminálkód továbbra is a sandboxban fut, amely korlátozza, mely fájlokat érhetik el a számítógépén. Működő OS-sandbox nélkül a kockázatos hívások előbb engedélyt kérnek.",
          full: "Minden eszközhívást rákérdezés nélkül futtat, és kikapcsolja a kód sandboxát, így a Python- és terminálkód bármely fájlt olvashat és módosíthat, amelyhez a fiókja hozzáfér.",
        },
      },
      notifications: {
        sectionTitle: "Értesítések",
        showLoadedModels: "Betöltött modellek kártyája",
        showLoadedModelsDescription:
          "Sarokkártya az összes betöltött modellel, egy kattintással kiüríthetők.",
        updateReminders: "Frissítési emlékeztetők",
        updateRemindersDescription:
          "Milyen gyakran ajánljon fel új verziót, ha elérhető.",
        llamaHint: "Csevegési és GGUF-modellek",
        audioHint: "Beszéd, zene és átírás",
        frequency: {
          always: "Amikor elérhető",
          daily: "Naponta egyszer",
          weekly: "Hetente egyszer",
          biweekly: "Kéthetente",
          monthly: "Havonta egyszer",
          off: "Ki",
        },
      },
      startup: {
        sectionTitle: "Indítás",
        launchAtLogin: "Az Unsloth futtatása bejelentkezéskor",
        launchAtLoginDescription:
          "Az Unsloth elindítása a háttérben bejelentkezéskor. A menüsoron vagy a rendszertálcán marad, amíg meg nem nyitja.",
        closeToTray: "Bezárás a tálcára",
        closeToTrayDescription:
          "Az Unsloth és a kiszolgálója a háttérben tovább fut, amikor bezárja a főablakot.",
        closeToTraySaveError:
          "Nem sikerült frissíteni a tálcára zárás beállítását.",
        menuBarIcon: "Megjelenítés a menüsoron",
        menuBarIconDescription:
          "Az Unsloth ikonjának megjelenítése a macOS menüsorán. Elrejtve az Unsloth helyette a Dockban marad, akkor is, ha bejelentkezéskor indul.",
        menuBarIconSaveError:
          "Nem sikerült frissíteni a menüsor-ikon beállítását.",
        loadError:
          "Nem sikerült betölteni a bejelentkezéskori indítás beállítását.",
        saveError:
          "Nem sikerült frissíteni a bejelentkezéskori indítás beállítását.",
      },
      hub: {
        source: "Modellforrás",
        sourceDescription:
          "Honnan keres és tölt le a modellközpont. Válassza a ModelScope-ot, ha a Hugging Face blokkolva van vagy lassú a hálózatán.",
        sourceFallback:
          "A ModelScope nem indítható el, ezért a Hugging Face van használatban. Ellenőrizze az Unsloth naplóit.",
        autoSourceTitle: "A modellforrás ModelScope-ra váltott",
        autoSourceDescription:
          "A Hugging Face az Ön régiójában gyakran lassú vagy blokkolt, ezért az Unsloth mostantól a ModelScope-ról tölti le a modelleket.",
        autoSourceAction: "Modellforrás-beállítások megnyitása",
        sectionTitle: "Modellközpont",
        endpoint: "Hugging Face-végpont",
        endpointDescription:
          "Honnan töltődnek le a modellek és az adatkészletek. Hagyja üresen a huggingface.co használatához, vagy adjon meg egy tükörkiszolgálót, például: https://hf-mirror.com.",
        datasetsServer: "Használat az adatkészlet-előnézetekhez",
        datasetsServerDescription:
          "Az adatkészlet-előnézeti kérések (HF_DATASETS_SERVER) is erre a végpontra kerülnek. Csak akkor kapcsolja be, ha a tükörkiszolgáló ezeket is kiszolgálja.",
        invalidEndpoint:
          "Adjon meg egy http(s) URL-t bejelentkezési adatok és lekérdezés nélkül. Sima http csak helyi címhez használható.",
        saveFailed: "Nem sikerült menteni a modellközpont beállításait.",
        loadFailed: "Nem sikerült betölteni a modellközpont beállításait.",
      },
      downloads: {
        sectionTitle: "Letöltések",
        transport: "Letöltési átvitel",
        transportDescription:
          "Hogyan érkeznek a modell- és adatkészletfájlok a Hugging Face-ről. A HTTPS ott folytatja, ahol abbamaradt; a Xet első letöltéskor gyakran gyorsabb, de megszakítás után elölről kezdi a fájlt.",
        transportHint:
          "A HTTPS sima TLS: minden hálózat, proxy és VPN engedi, a megszakított vagy félbeszakadt átvitel a lemezen már meglévő bájtoktól folytatódik, és a memóriahasználat egyenletes marad. A Xet deduplikált darabokat tölt le, így egy már meglévővel adatokat megosztó tároló sokkal gyorsabban megérkezhet, de ehhez hf_xet szükséges, több RAM-ot használ, és megszakításkor az éppen átvitt fájl elvész. Az Automatikus mód ehhez a géphez igazodik: figyelembe veszi a RAM-ot és azt, hogy a Xet elakadt-e itt, és visszavált HTTPS-re.",
        https: "HTTPS",
        xet: "Xet",
        auto: "Automatikus",
        httpsHint:
          "Szabványos TLS. Megszakítás után folytatható, bármely hálózaton működik, egyenletes memóriahasználat.",
        transportDescriptionNoResume:
          "Hogyan érkeznek a modell- és adatkészletfájlok a Hugging Face-ről. Ezen a telepítésen egyik átviteli mód sem tudja folytatni a letöltést, így a megszakított letöltés elölről kezdődik; a Xet első letöltéskor gyakran gyorsabb.",
        httpsHintNoResume:
          "Szabványos TLS. Bármely hálózaton működik, egyenletes memóriahasználat. Ez a telepítés nem tudja folytatni a megszakított letöltést.",
        xetHint:
          "Deduplikált, darabolt átvitel. Új letöltésnél gyakran gyorsabb, megszakításkor elölről kezdi a fájlt, több memóriát igényel.",
        autoHint:
          "Gépenként választ, és HTTPS-re vált, ha a Xet itt elakad vagy hibát jelez.",
        autoCurrently:
          "Az Automatikus mód ezen a gépen ezt használja: {transport}.",
        xetMissing: "A Xet nem érhető el, mert a hf_xet nincs telepítve.",
      },
      uploads: {
        sectionTitle: "Feltöltések",
        maxUploadSize: "Tanítási adatkészlet feltöltési korlátja",
        maxUploadSizeDescription: "Alapértelmezés: {defaultSize} MB.",
      },
      rag: {
        sectionTitle: "Dokumentumok és RAG",
        embeddingModel: "Beágyazási modell",
        embeddingModelShort:
          "Vektorokká alakítja a dokumentumokat, hogy a csevegés kereshessen bennük.",
        embeddingModelDescription:
          "A dokumentumok indexeléséhez és kereséséhez használt Hugging Face-modell vagy helyi elérési út. Alapértelmezés: {defaultModel}.",
        searchPlaceholder: "Bármely modell keresése a HF-en",
        reindexWarning:
          "Csak az újonnan indexelt dokumentumokra vonatkozik. A modell módosítása után ágyazza be újra a meglévőket.",
        reembed: {
          label: "Dokumentumok újbóli beágyazása",
          staleOne:
            "1 dokumentum másik beágyazási modellel lett indexelve, ezért a szemantikus keresés kihagyja.",
          staleMany:
            "{count} dokumentum másik beágyazási modellel lett indexelve, ezért a szemantikus keresés kihagyja őket.",
          staleArchived:
            "A korábbi csevegési előzmények másik beágyazási modellel lettek indexelve, ezért a szemantikus keresés kihagyja őket.",
          running: "Újbóli beágyazás: {done} / {total}",
          action: "Újbóli beágyazás",
          stop: "Leállítás",
          startFailed: "Nem sikerült elindítani az újbóli beágyazást",
          stopped: "Az újbóli beágyazás leállítva",
          finished: "A dokumentumok újra beágyazva",
          finishedWithFailures:
            "{count} dokumentumot nem sikerült újra beágyazni",
        },
        emptyError:
          "Adjon meg egy Hugging Face-modellazonosítót vagy helyi elérési utat.",
        loadError: "Nem sikerült betölteni a beágyazási modell beállítását.",
        saveError: "Nem sikerült menteni a beágyazási modellt.",
        saved: "A beágyazási modell mentve.",
        saveAnyway: "Mentés mindenképp",
        recommended: "Ajánlott",
        onDevice: "Az eszközön",
        searching: "Keresés a Hugging Face-en…",
        checking: "Ellenőrzés…",
        noResults: "Nem található beágyazási modell",
        download: "Letöltés",
        unload: "Kiürítés",
        unloadFailed: "Nem sikerült kiüríteni a beágyazási modellt",
        ejected: "A beágyazási modell kiürítve",
        ejectModel: "Modell kiürítése",
        onDeviceHint:
          "Letöltve erre a számítógépre. Csak dokumentumok indexelésekor vagy keresésekor töltődik be a memóriába, addig nem használ RAM-ot és VRAM-ot.",
        loadedHint:
          "Jelenleg a memóriában van, RAM-ot vagy VRAM-ot használ. A kiürítéshez vigye az egeret a modellválasztó fölé, és kattintson a piros ikonra.",
        notLoaded: "Nincs betöltve",
        notLoadedHint:
          "Jelenleg nincs a memóriában. A dokumentumok következő indexelésekor vagy keresésekor magától betöltődik, ezután a választóból kiürítheti.",
        downloadingStatus: "Letöltés…",
        notDownloaded: "Nincs letöltve",
        notDownloadedSized: "Nincs letöltve · {size}",
        loaded: "Betöltve",
        downloading: "Letöltés: {model}",
        downloadingDescription:
          "Az előrehaladás a letöltések panelen látható. Az indexelés a megérkezése után használja.",
        downloadFailed: "Nem sikerült elindítani a letöltést",
        downloadConflict: "Folytassa ezt a letöltést a modellközpontból",
        downloadBusy: "A letöltés már folyamatban van",
        pin: "Rögzítés a Fájlok menüben",
        unpin: "Rögzítés feloldása a Fájlok menüben",
        menuChip: "Modell",
        moreModels: "További modellek",
        pinMoreModels: "További modellek rögzítése",
        defaultTag: "Alapértelmezett",
        localModel: "Helyi modell",
        switched: "Beágyazási modell beállítva: {model}",
        switchedNeedsDownload: "Még nincs letöltve: {model}",
        switchedNeedsDownloadDescription:
          "Új fájlok indexelése előtt töltse le a Beállításokban.",
        switchedUncheckedDescription:
          "Nem sikerült ellenőrizni, hogy le van-e töltve. Új fájlok indexelése előtt ellenőrizze a Beállításokban.",
        openSettings: "Beállítások megnyitása",
        switchFailed: "Nem sikerült váltani a beágyazási modellt",
        back: "Vissza a forrásokhoz",
        menuSubtitle: "A fájlkeresést működteti",
        device: "Beágyazási eszköz",
        deviceDescription:
          "Hol fut a beágyazási modell, minden modell esetén. A GPU gyorsabban indexeli a nagy dokumentumkészleteket, de az Unsloth újraindításáig lefoglal valamennyi VRAM-ot.",
        deviceAuto: "Automatikus",
        deviceCpu: "CPU",
        deviceGpu: "GPU",
        deviceLocked: "Beállította: {name}.",
        deviceSaveError: "Nem sikerült módosítani a beágyazási eszközt",
      },
      storage: {
        sectionTitle: "Tárhely",
        modelsFolder: "Modellek mappája",
        modelsFolderDescription: "A letöltött modellek tárolási helye.",
        openAction: "Megnyitás",
        copyAction: "Elérési út másolása",
        copied: "Elérési út kimásolva",
        openError: "Nem sikerült megnyitni a mappát",
        copyError: "Nem sikerült másolni az elérési utat",
      },
      fullInstall: {
        label: "Telepítés típusa",
        description:
          "Csak csevegés (GGUF). A teljes telepítés tanítással, exportálással, videóval, safetensors-modellekkel és további képgenerálási funkciókkal bővül.",
        action: "Frissítés teljes telepítésre",
        confirmTitle: "Frissít teljes telepítésre?",
        confirmDescription:
          "Ez a Csak csevegés telepítés GGUF-modelleket futtat. A teljes telepítés a gép GPU-jához illő PyTorch-ot ad hozzá, amely lehetővé teszi a tanítást, az exportálást, a videót, a safetensors-modelleket és további képgenerálási funkciókat. Több gigabájtot tölt le (legalább 5 GB lemezterület szükséges), és újraindítja a kiszolgálót. A csevegések, modellek és beállítások megmaradnak. Ez több percig is eltarthat.",
        confirmAction: "Frissítés",
        keywords:
          "full install upgrade install type chat only gguf add training train training fine-tune pytorch torch export video image generation safetensors teljes telepítés frissítés telepítés típusa csak csevegés tanítás finomhangolás exportálás videó képgenerálás",
      },
      repairInstall: {
        label: "Telepítés javítása",
        description:
          "Újrafuttatja a telepítőt a felügyelt környezeten. Akkor használja, ha a GPU nem észlelhető, vagy az alkalmazás nem indul el.",
        action: "Telepítés javítása",
        confirmTitle: "Kijavítja ezt a telepítést?",
        confirmDescription:
          "Leállítja a kiszolgálót, és újrafuttatja a telepítőt, amely a gép GPU-jához a legújabb támogatott PyTorch-kiadást telepíti újra. A csevegések és beállítások megmaradnak. Ez több percig is eltarthat.",
        confirmDescriptionChatOnly:
          "Leállítja a kiszolgálót, és újrafuttatja a telepítőt. Ez egy Csak csevegés telepítés, ezért a telepítő egyben teljes, PyTorch-ot tartalmazó telepítésre is frissíti, összesen több gigabájt méretben. A csevegések és beállítások megmaradnak. Ez több percig is eltarthat.",
        confirmAction: "Javítás most",
      },
      resetPreferences: {
        sectionTitle: "Veszélyes zóna",
        label: "Minden helyi beállítás visszaállítása",
        description:
          "Törli a csak helyben tárolt beállításokat. A csevegések, az API-hozzáférés és az adatbázisban tárolt beállítások megmaradnak.",
        action: "Beállítások visszaállítása",
        confirmTitle: "Visszaállítja az összes helyi beállítást?",
        confirmDescription:
          "Törli a csak helyben tárolt beállításokat, és újratölti az Unslothot. A csevegések, az API-hozzáférés és az adatbázisban tárolt beállítások megmaradnak.",
        confirmAction: "Visszaállítás és újratöltés",
      },
    },
    profile: {
      title: "Profil",
      description: "Hogyan jelenik meg a profilja az Unslothban.",
      changePicture: "Profilkép módosítása",
      uploadPhoto: "Fénykép feltöltése",
      removePhoto: "Eltávolítás",
      pictureOptions: "Profilkép beállításai",
      displayName: "Megjelenített név",
      nickname: "Hogyan szólítsa az Unsloth?",
      nicknamePlaceholder: "Becenév",
      nicknameSaved: "A preferált név mentve",
      avatarShape: "Profilkép alakja",
      avatarShapeCircle: "Kör",
      avatarShapeRounded: "Lekerekített",
      greetingSloth: "Üdvözlés",
      greetingSlothDescription: "Üdvözlés megjelenítése az új csevegésekben.",
      chooseSloth: "Vagy válasszon egy lajhárt",
      noPicture: "Nincs profilkép",
      noneLabel: "Nincs",
      nameSaved: "A profilnév mentve",
      namePersistErrorTitle: "Nem sikerült tartósan menteni a profilnevet",
      namePersistErrorDescription:
        "A név ebben a munkamenetben frissült, de újratöltés után elveszhet.",
      photoUpdated: "A profilfénykép frissítve",
      photoPersistErrorTitle: "Nem sikerült tartósan menteni a profilfényképet",
      photoPersistErrorDescription:
        "A fénykép ebben a munkamenetben frissült, de újratöltés után elveszhet.",
      photoUpdateErrorTitle: "Nem sikerült frissíteni a profilfényképet",
      imageUseError: "Ez a kép nem használható.",
      stats: {
        title: "Statisztikái",
        subtitle:
          "Az alábbiak mind a saját előzményeiből vannak számolva. Semmit nem gyűjtünk, és semmi nem kerül elküldésre az Unslothnak.",
        retry: "Újrapróbálkozás",
        exportCsv: "CSV exportálása",
        exportError: "Nem sikerült exportálni a tokenhasználatot",
        privacyNote:
          "A statisztikák az Unsloth-telepítésben tárolt helyi csevegési, API-használati és tanítási előzményekből készülnek. Az API-promptokat, -válaszokat és -kulcsokat a statisztikákhoz soha nem tároljuk. Semmi nem kerül elküldésre az Unslothnak vagy harmadik félnek.",
        emptyChats:
          "Még nincs csevegés vagy API-használat. Kezdjen egy beszélgetést, vagy küldjön hitelesített helyi API-kérést, és a statisztikái itt jelennek meg.",
        lifetimeTokens: "Összes token eddig",
        peakTokens: "Csúcsnap",
        longestChat: "Leghosszabb csevegés",
        currentStreak: "Jelenlegi sorozat",
        longestStreak: "Leghosszabb sorozat",
        activityTitle: "Tokenaktivitás",
        activityDescription: {
          daily: "{total} · kezdete: {since}",
          weekly: "Legforgalmasabb hét: {total} · kezdete: {since}",
          cumulative: "Összesen {total} · kezdete: {since}",
        },
        mode: {
          daily: "Napi",
          weekly: "Heti",
          cumulative: "Halmozott",
        },
        cellTooltip: "{date} · {tokens}, {messages}",
        emptyDayTooltip: "{date} · Nincs aktivitás",
        apiRequests: "API-n keresztül: {count}",
        weekTooltip: "Hét kezdete: {date} · {tokens}",
        cumulativeTooltip:
          "A hét végéig, amely ekkor kezdődött: {date} · összesen {tokens}",
        less: "Kevesebb",
        more: "Több",
        insightsTitle: "Aktivitási áttekintés",
        totalChats: "Összes csevegés",
        totalMessages: "Összes üzenet",
        tokensIn: "Elküldött tokenek",
        tokensOut: "Generált tokenek",
        totalTokens: "Összes token",
        studioChatTokens: "Unsloth Chat-tokenek",
        apiTokens: "API-tokenek",
        cachedTokens: "Gyorsítótárazott tokenek",
        cachedValue: "{tokens} (a bemenet {percent}%-a)",
        avgTokensPerChat: "Átlagos token csevegésenként",
        timeInChat: "Csevegéssel töltött idő",
        activeDays: "Aktív napok",
        toolCalls: "Eszközhívások",
        attachments: "Csatolt fájlok",
        avgSpeed: "Átlagos sebesség",
        bestSpeed: "Leggyorsabb válasz",
        firstToken: "Átlagos idő az első tokenig",
        tokensPerSecond: "{value} tok/s",
        topModelsTitle: "Leggyakrabban használt modellek",
        topModelsDescription: "A váltott tokenek száma szerint rangsorolva",
        modelSummary: "{tokens} · {messages}",
        noModels: "Még nincs rögzített modellhasználat.",
        trainingTitle: "Tanítás",
        trainingDescription:
          "Finomhangolási futtatások ebből a munkaterületből",
        trainingRuns: "Futtatások",
        trainingCompleted: "Befejezve",
        trainingSteps: "Lépések",
        trainingTokens: "Betanított tokenek",
        trainingTime: "Tanítási idő",
        bestLoss: "Legjobb veszteség",
        runSteps: "{steps}",
        runLoss: "veszteség: {loss}",
      },
    },
    appearance: {
      title: "Megjelenés",
      description: "Hogyan néz ki az Unsloth ezen az eszközön.",
      gpuRendering: {
        sectionTitle: "Megjelenítés",
        label: "GPU-s megjelenítés",
        description:
          "Az Unsloth megjelenítése a grafikus kártyával. Az Automatikus mód ott használja, ahol ismerten működik. A módosítások az Unsloth újraindítása után lépnek érvénybe.",
        auto: "Automatikus",
        on: "Be",
        off: "Ki",
        oldWebkitWarning:
          "WebKitGTK {version} Wayland alatt: a GPU-s megjelenítés indításkor összeomolhat NVIDIA kártyán tört méretezéssel (WebKit 280210-es hiba). Ilyenkor az Unsloth magától visszavált CPU-s megjelenítésre. WebKitGTK 2.50.4 alatt nem hozott mérhető gyorsulást (mindkét esetben 8 fps), ezért itt a Be beállítás nem ajánlott.",
        oldDriverWarning:
          "NVIDIA-illesztőprogram {version} X11 alatt: az Unsloth kikapcsolva tartja a GPU-s megjelenítést, amíg az illesztőprogram nem 610-es vagy újabb. A régebbi illesztőprogramok képkockánként szivárogtatnak egy fájlleírót, és az ablak perceken belül leáll a rajzolással.",
        fallbackNote:
          "A GPU-s megjelenítés indításkor összeomlott ezen a rendszeren, ezért az Unsloth CPU-s megjelenítést használ.",
        tryAgain: "GPU újbóli kipróbálása",
        envOverride:
          "Környezeti változó állította be: {name}. Ennek a beállításnak nincs hatása, amíg el nem távolítja.",
        macNote:
          "A macOS mindig a GPU-val jelenít meg. A WebKit nem kínál támogatott kapcsolót a kikapcsolásához.",
        restartPending:
          "A módosítás érvénybe léptetéséhez indítsa újra az Unslothot.",
        restartNow: "Újraindítás most",
        loadError: "Nem sikerült betölteni a GPU-s megjelenítés beállítását.",
        saveError: "Nem sikerült menteni a GPU-s megjelenítés beállítását.",
        restartError: "Nem sikerült újraindítani az Unslothot.",
      },
      theme: {
        title: "Vizuális stílus",
        label: "Mód",
        description: "Világos, sötét, vagy a rendszer követése.",
        system: "Rendszer",
        light: "Világos",
        dark: "Sötét",
      },
      palette: {
        label: "Téma",
        description:
          "Az Unsloth-szerte használt színek, világos és sötét módban.",
        standard: "Normál",
        classic: "Klasszikus",
        minimal: "Minimalista",
        moreThemes: "További témák",
        halloween: "Halloween",
      },
      trickOrTreat: {
        label: "Csokit vagy csalunk",
        description:
          "Halloweeni témák és kísérteties üdvözlések november 1-ig.",
      },
      custom: {
        chatWidth: {
          label: "Csevegés szélessége",
          description: "Az üzenetek és a szerkesztőmező szélessége.",
          standard: "Normál",
          wide: "Széles",
          full: "Teljes szélesség",
        },
        sentAttachments: {
          label: "Mellékletek az elküldött üzenetekben",
          description:
            "A Normál minden fájlt a típusával együtt mutat, a Tömör soronként többet jelenít meg. Az Automatikus hat fájl felett Tömörre vált.",
          auto: "Automatikus",
          list: "Normál",
          chips: "Tömör",
        },
        reset: "Visszaállítás",
        resetAll: "Testreszabás visszaállítása",
        preferencesTitle: "Beállítások",
        colors: {
          accent: "Kiemelőszín",
          background: "Háttér",
          foreground: "Előtér",
        },
        fontDefault: "Alapértelmezett",
        fontBundledGroup: "Beépített",
        fontImportedGroup: "Importált",
        fontDeviceGroup: "Ezen az eszközön",
        fontFolderGroup: "Mappából",
        fontDeviceLoading: "Eszközbetűtípusok keresése…",
        fontSearch: "Betűtípusok keresése…",
        fontNoResults: "Nem található betűtípus.",
        colorPicker: {
          hue: "Színárnyalat",
          hex: "Hexadecimális szín",
          eyedropper: "Szín kiválasztása a képernyőről",
        },
        uiFont: {
          label: "Felület betűtípusa",
        },
        headingFont: {
          label: "Címsor betűtípusa",
        },
        chatFont: {
          label: "Csevegés betűtípusa",
        },
        codeFont: {
          label: "Kód betűtípusa",
        },
        importFont: {
          upload: "Feltöltés",
          scanFolder: "Mappa kiválasztása",
          alreadyAvailable:
            "Ez a betűtípus már elérhető, ezért a meglévő példányt használja az alkalmazás.",
          folderNoFonts: "Nem található betűtípusfájl ebben a mappában.",
          remove: "Eltávolítás",
          errorInvalidType:
            "Nem támogatott fájltípus. Használjon .woff2, .woff, .ttf vagy .otf fájlt.",
          errorTooLarge: "A betűtípusfájl túl nagy (legfeljebb 1,5 MB).",
          errorLimit: "Legfeljebb 3 betűtípust importálhat.",
          errorStorageFull:
            "Nincs elég helyi tárhely ehhez a betűtípushoz. Előbb távolítson el egy importált betűtípust.",
          errorFailed: "Nem sikerült betölteni ezt a betűtípusfájlt.",
        },
        interfaceScale: {
          label: "Felület méretezése",
          description: "Az összes felületelem átméretezése.",
        },
        uiFontSize: {
          label: "Felület betűmérete",
          description: "Az Unsloth felületéhez használt alapméret módosítása.",
        },
        codeFontSize: {
          label: "Kód betűmérete",
          description: "A kódhoz használt alapméret módosítása.",
        },
        fontSmoothing: {
          label: "Betűsimítás",
          description: "Simított betűélsimítás használata.",
        },
        contrast: {
          label: "Kontraszt",
        },
        reduceMotion: {
          label: "Mozgás csökkentése",
          description:
            "Animációk csökkentése vagy a rendszerbeállítás követése.",
          system: "Rendszer",
          on: "Be",
          off: "Ki",
        },
        pointerCursors: {
          label: "Mutató kurzor használata",
          description:
            "A kurzor mutatóra változik az interaktív elemek fölött.",
        },
      },
      language: {
        title: "Nyelv",
        label: "Megjelenítési nyelv",
        description: "Az Unsloth által használt nyelv.",
        autoDetect: "Automatikus felismerés",
        search: "Nyelvek keresése",
        noResults: "Nincs egyező nyelv",
        spellCheck: "Helyesírás-ellenőrzés",
        spellCheckDescription: "A hibásan írt szavak aláhúzása gépelés közben.",
      },
      layout: {
        title: "Elrendezés",
        compactSidebar: "Oldalsáv rögzítése alapértelmezés szerint",
        compactSidebarDescription:
          "Az oldalsáv kibontva marad ahelyett, hogy ikonokká csukódna össze.",
      },
      sidebarNav: {
        title: "Oldalsáv-navigáció",
        description:
          "Az oldalsáv lapjainak rögzítése és átrendezése. A nem rögzített lapok a Továbbiak menübe kerülnek.",
        dragToReorder: "Húzza az átrendezéshez",
        pinToSidebar: "Rögzítés az oldalsávon: {name}",
        moreHolds: "Továbbiak ({count})",
      },
      sidebarMenu: {
        title: "Profilmenü",
        description: "A neve alatti parancsikonok kiválasztása és átrendezése.",
        darkModeToggle: "Sötét mód kapcsoló",
        dragToReorder: "Húzza az átrendezéshez",
      },
    },
    resources: {
      title: "Rendszer és motor",
      description:
        "Az Unsloth-kiszolgáló hardverének és tárhelyének figyelése.",
      liveUpdates: "Élő frissítések",
      floatingWindow: "Lebegő ablak",
      showFloatingWindow: "Lebegő ablak megjelenítése",
      showFloatingWindowDescription:
        "A CPU-, RAM- és GPU-használat a képernyőn marad az Unsloth fölött.",
      liveMonitor: {
        title: "Élő figyelő",
        apiTitle: "API-figyelő",
        apiDescription:
          "Lássa valós időben, mit küldenek más alkalmazások a kiszolgáló API-jának.",
        summary: "Élő kérések, hibák és tokenhasználat",
        status: "{active} aktív · {recent} legutóbbi · {model}",
        noModelLoaded: "nincs betöltött modell",
        autoOpen: "A lebegő figyelő automatikusan megnyílik",
        autoOpenDescription:
          "Egy kis panelt nyit meg, amikor API-forgalom érkezik.",
        cpu: "CPU",
        ram: "RAM",
        disk: "Lemez",
        modelsDisk: "Modellek lemeze",
        vram: "VRAM",
        cpuCores: "{logical} logikai / {physical} fizikai mag",
        currentLoad: "Jelenlegi terhelés",
        free: "Szabad: {value}",
        noGpu: "Nincs látható GPU",
        gpuUnusable: "A GPU nem használható",
        gpuUnusableDetail: "Észlelve, de a PyTorch nem tudja használni",
        detailedView: "Részletes nézet",
        detailedViewDescription:
          "A CPU terhelésének, valamint minden GPU terhelésének, hőmérsékletének és teljesítményfelvételének megjelenítése a lebegő ablakban.",
        cpuLoad: "CPU-terhelés",
        gpuLoad: "GPU-terhelés",
        temperature: "Hőmérséklet",
        power: "Teljesítmény",
      },
      gpu: {
        title: "GPU-eszközök",
        memory: "GPU-memória",
        sharedWithSystemRam: "Megosztva a rendszer-RAM-mal",
        estimatedAvailable: "Becsült szabad: {value}",
        sharedEstimatedAvailable:
          "Megosztott rendszer-RAM: becsült szabad: {value}",
        ggufInference: "GGUF-modell memóriája",
        unavailable: "nem érhető el",
        detecting: "GPU-k keresése...",
        unreadable: "Nem sikerült beolvasni a kiszolgáló hardverét.",
        noGpu:
          "Nem észlelhető látható GPU. Fent a csak CPU-s erőforrások láthatók.",
        noUsableGpu: "Ezen a gépen nincs a PyTorch által használható GPU.",
        mismatchCpuBuild:
          "A PyTorch csak CPU-s változat ({version}), ezért az alábbi GPU-k nem használhatók. A GPU-támogatás visszaállításához javítsa a telepítést.",
        mismatchUnavailable:
          "A PyTorch ({version}) nem tudja inicializálni az alábbi GPU-kat, ezért azok nem használhatók. Ellenőrizze a GPU-illesztőprogramot, vagy javítsa a telepítést.",
        driverIdleEvict:
          "Az AMD-illesztőprogram ({version}) ismert hibája miatt a Windows lefagyhat, ha egy AMD GPU tétlen, leggyakrabban több GPU esetén. Frissítsen Adrenalin 26.9.2-es vagy újabb verzióra.",
        driverIdleEvictDetails: "Részletek",
        dismissNotice: "Elvetés",
        unusableDevice: "nem használható",
        unknownDevice: "Ismeretlen GPU",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "Használt: {value}",
        free: "Szabad: {value}",
        total: "Összesen: {value}",
        usedOf: "Használt: {used} / {total}",
        segmentUsed: "GPU által használt",
        segmentSystem: "Rendszer által használt",
        segmentAvailable: "Szabad",
      },
      llamaBackend: {
        title: "GGUF-következtetési motor",
        label: "Számítási háttérrendszer",
        description:
          "A llama.cpp által a GGUF-modellek futtatásához használt háttérrendszer.",
        runningOn: "A llama.cpp jelenleg ezen fut: {backend}.",
        hint: "Telepíti az ehhez a háttérrendszerhez készült llama.cpp-változatot, és a frissítések során is megtartja. Akkor hasznos, ha az automatikus választás összeomlik, vagy a GPU-illesztőprogram nem támogatja. Csak azok a háttérrendszerek jelennek meg, amelyekhez van változat erre a gépre; a tanítást nem érinti.",
        autoWith: "Automatikus ({backend})",
        apply: "Alkalmaz",
        applying: "Telepítés...",
        applyHint:
          "Letölti az új változatot, és újraindítja a llama.cpp-t. A betöltött modell kiürül.",
        applyHintWithSize:
          "Letölt {size} adatot, és újraindítja a llama.cpp-t. A betöltött modell kiürül.",
        switchedTo: "A llama.cpp mostantól ezen fut: {backend}.",
        switchFailed: "Nem sikerült módosítani a llama.cpp háttérrendszerét.",
        switchInterrupted: "A váltás a befejezés előtt megszakadt.",
        envLocked:
          "Az UNSLOTH_LLAMA_CPP_BACKEND környezeti változó felülbírálja ezt a beállítást, értéke: {backend}.",
        customPath: {
          label: "Egyéni llama.cpp-mappa",
          description: "Saját llama-server-build használata.",
          hint: "Válassza ki a llama-servert tartalmazó llama.cpp-mappát, vagy egy buildet, amelyben a build/bin alatt található. Az Unsloth az egyéni futtatókörnyezetet használja a GGUF-csevegéshez, a beágyazásokhoz és a támogatott hangmodellekhez. A környezeti változók továbbra is elsőbbséget élveznek.",
          automatic: "Automatikus (beépített)",
          bundled:
            "Az Unsloth által telepített llama.cpp-futtatókörnyezetet használja.",
          active:
            "A következő modellbetöltéskor már az egyéni llama-server fut.",
          environmentManaged: "Környezeti változó kezeli: {variable}.",
          missingBinary:
            "A llama-server már nem érhető el ebben a mappában. Válasszon másik mappát, vagy használja a beépített futtatókörnyezetet.",
          reloadRequired:
            "A kiválasztott llama-server használatához töltse be újra a modellt.",
          change: "Módosítás",
          saving: "Mentés...",
          useBundled: "Beépített használata",
          chooseTitle: "llama.cpp-mappa kiválasztása",
          chooseAction: "Ennek a mappának a használata",
          saved: "A llama.cpp-mappa frissítve",
          saveError: "Nem sikerült frissíteni a llama.cpp-mappát",
        },
        backends: {
          auto: "Automatikus",
          cpu: "CPU",
          cuda: "CUDA",
          rocm: "ROCm",
          vulkan: "Vulkan",
          metal: "Metal",
        },
        unsupported: {
          notInstalled:
            "Nem található felügyelt llama.cpp-telepítés, így nincs mit váltani.",
          localLink:
            "A llama.cpp egy Ön által csatolt helyi könyvtár, ezért az Unsloth nem cseréli le.",
          sourceBuild:
            "Ez a llama.cpp forráskódból készült, ezért a háttérrendszere innen nem váltható.",
          customPath:
            "Egyéni llama.cpp-mappa van kiválasztva. A számítási háttérrendszert annak buildje határozza meg.",
          unresolved:
            "Nem sikerült ellenőrizni az elérhető háttérrendszereket. Ellenőrizze a kapcsolatot, és próbálja újra.",
          updateChecksDisabled:
            "A frissítések keresése le van tiltva (UNSLOTH_DISABLE_UPDATE_CHECK=1), ezért az elérhető háttérrendszerek lekérdezése elmarad.",
        },
        llamaBackendKeywords:
          "llama.cpp backend gguf inference cuda rocm hip vulkan metal cpu gpu accelerator prebuilt switch engine custom path folder directory llama-server executable binary háttérrendszer következtetés gyorsító motor egyéni mappa elérési út",
      },
      modelMemory: {
        title: "Modellmemória",
        keepResident: "Modell megtartása a GPU-memóriában",
        keepResidentDescription: "A promptok között a VRAM-ban marad.",
        keepResidentHint:
          "Amíg a modell betöltve marad, a súlyok nem kerülnek vissza a rendszer-RAM-ba. Kikapcsolja a tétlenség miatti automatikus kiürítést, és ha a súlyok a gazdagép RAM-jában vannak (egységes memória vagy részleges GPU-áthelyezés), a --mlock kapcsolót is átadja, hogy az operációs rendszer ne lapozhassa ki őket, és ne kelljen újra feltölteni őket a következő promptnál.",
        noRamReserve: "Ne foglaljon rendszer-RAM-ot a modellnek",
        noRamReserveDescription:
          "A modellsúlyok által foglalt gazdagép-RAM csökkentése.",
        noRamReserveHint:
          "A támogatott Windows-buildeken kihagyja a memórialeképezéses betöltést, ha a modell teljesen a GPU-ra van áthelyezve, így a lapjai nem maradnak a memóriában. Egyébként megtartja a memórialeképezéses betöltést. A szükséges CPU-pufferek továbbra is használhatnak RAM-ot. Elhagyja a --no-mmap és a --mlock kapcsolót.",
        multiModel: "Több modell betöltve tartása",
        openSystemSettings: "Megnyitás a Rendszer lapon",
        multiModelOff: "Ki",
        multiModelFit: "Ha elférnek",
        multiModelAlways: "Mindig",
        multiModelDescription:
          "Egy csevegő-, beszéd-, kép- vagy videómodell betöltése kiüríti a többit.",
        multiModelDescriptionFit:
          "Egy modell betöltésekor a többi megmarad, ha elférnek. Ha nem, először a nem használtakat üríti ki, majd az aktívat.",
        multiModelDescriptionAlways:
          "Egy modell betöltése soha nem üríti ki a többit. Az el nem férő modell nem töltődik be. A tétlenség miatti automatikus kiürítés és a tanítás továbbra is felszabadít memóriát.",
        multiModelHint:
          "A csevegő-, beszéd-, kép- és videómodellekre vonatkozik. Minden betöltött csevegőmodell azokra a kérésekre válaszol, amelyek megnevezik. Több GPU esetén az Unsloth az új modellt olyan GPU-ra próbálja helyezni, amelyet más modell nem használ. A „Ha elférnek” mód körülbelül 1 GB-ot hagy szabadon. A többi mellé betöltött modell az általuk szabadon hagyott memóriát kapja, ezért rövidebb környezettel indulhat. A „Ha elférnek” és a „Mindig” mód soha nem üríti ki a másik GPU-n lévő modellt.",
        allowOversizedLoad: "Túlméretes modellek betöltésének engedélyezése",
        allowOversizedLoadDescription:
          "Kép- és videómodellek betöltése akkor is, ha a memóriaellenőrzés szerint nem férnek el.",
        allowOversizedLoadHint:
          "Ott érvényes, ahol a CPU és a GPU közös memóriát használ (Apple Silicon, integrált GPU-k). Az Unsloth elutasítja a betöltést, ha a modellsúlyokra vonatkozó becslése nagyobb, mint az operációs rendszer számára fenntartott hely után megmaradó memória. Ha ez be van kapcsolva, ezek a betöltések mégis megtörténnek. A becslés mindkét irányban tévedhet: a túlméretes betöltés meghiúsulhat, elfogyaszthatja az összes memóriát, erős lapozást okozhat, a számítógép lefagyhat, vagy az Unsloth leállhat. Alapértelmezés szerint ki van kapcsolva.",
        mediaIdleUnloadDescription:
          "VRAM felszabadítása a kép- és videómodellek kiürítésével ennyi másodperc tétlenség után. 0 esetén betöltve maradnak, a minimum 60.",
        mlockVetoed:
          "A --mlock kikapcsolva marad: a modell rögzítése a teljes modellnek foglalna le RAM-ot. A tétlenség miatti automatikus kiürítés továbbra is le van tiltva.",
        mlockNotApplicable:
          "Teljesen a GPU-n van: nincs mit zárolni a rendszer-RAM-ban. A tétlenség miatti automatikus kiürítés kikapcsolva marad.",
        memlockCapped:
          "Ez a rendszer a zárolt memóriát ennyiben korlátozza: {limit}. Egy nagyobb modell nem rögzíthető teljesen; emelje meg a korlátot az ulimit -l paranccsal.",
        reloadRequired:
          "Az új memóriakapcsolók alkalmazásához töltse be újra a modellt.",
        loadError: "Nem sikerült betölteni a modellmemória beállításait",
        saveError: "Nem sikerült menteni a modellmemória beállításait",
        modelMemoryKeywords:
          "mlock memlock ulimit vram gpu memory ram resident residency pin pinned page lock locked keep loaded unload idle mmap no-mmap load-mode swap paging memória rögzítés zárolás betöltve kiürítés tétlen lapozás",
        multiModelKeywords:
          "multiple several models together alongside swap fit unload evict vram gpu image video chat speech tts voice read aloud always idle több modell egyszerre kiürítés kép videó csevegés beszéd hang felolvasás mindig tétlen",
        mediaIdleUnloadKeywords:
          "image video diffusion idle unload ttl timeout seconds free vram gpu memory kép videó tétlen kiürítés időkorlát másodperc memória",
      },
      storage: {
        title: "Tárhely",
        systemDisk: "Rendszerlemez",
        systemUsed: "Használt a rendszerlemezen: {used} / {total}",
        modelsDiskSummary:
          "A modellek mappájának meghajtója: használt {used} / {total}, szabad: {free}",
        segmentModels: "Modellek",
        segmentCaches: "Gyorsítótárak",
        segmentOther: "Egyéb",
        segmentFree: "Szabad",
        refreshAction: "Frissítés",
        modelsDisk: "Modellek lemeze",
        diskUsage: "Használt: {used} / {total}",
        diskFree: "Szabad: {free}",
        modelsFolder: "Modellek mappája",
        modelsFolderDescription: "A letöltött modellek tárolási helye.",
        modelsFolderHint:
          "A letöltött modellek tárolási helye. Módosítsa, ha a modelleket nem a rendszermeghajtón szeretné tartani. Csak az új letöltésekre vonatkozik. A meglévő modellek a helyükön maradnak.",
        modelsFolderKeywords:
          "models folder directory path location download downloads cache storage disk drive move relocate hugging face modellek mappa elérési út hely letöltés gyorsítótár tárhely lemez meghajtó áthelyezés",
        futureDownloads: "Csak új letöltések",
        environmentManaged: "Környezeti változó kezeli: {variable}.",
        locationFree: "Szabad: {free}",
        openAction: "Megnyitás",
        copyAction: "Elérési út másolása",
        changeAction: "Módosítás",
        resetAction: "Alapértelmezett használata",
        chooseTitle: "Modellek letöltési helyének kiválasztása",
        chooseAction: "Használat a jövőbeli letöltésekhez",
        cacheSaved: "A modellek letöltési helye frissítve",
        cacheSaveError: "Nem sikerült frissíteni a modellek letöltési helyét",
        cachePickerError: "Nem sikerült megnyitni a mappaválasztót",
        copied: "Elérési út kimásolva",
        openError: "Nem sikerült megnyitni a mappát",
        copyError: "Nem sikerült másolni az elérési utat",
        caches: {
          label: "Gyorsítótárfájlok",
          description:
            "Gyorsítótárakban: {size}, ebből most törölhető: {reclaimable}.",
          hint: "Csomagletöltések, lefordított kernelek és átviteli gyorsítótárak, amelyeket az Unsloth szükség esetén újraépít. A letöltött modellek, projektek, csevegések, beállítások és a Hugging Face-token itt soha nem törlődnek.",
          keywords:
            "cache caches purge prune clear clean free space disk uv pip npm bun triton inductor cuda numba matplotlib vllm compiled xet temporary gyorsítótár törlés tisztítás szabad hely lemez lefordított ideiglenes",
          measuring: "Gyorsítótárak méretének mérése...",
          measureFailed: "Nem sikerült megmérni a gyorsítótárakat",
          empty: "Nem találhatók gyorsítótárfájlok.",
          detailsAction: "Részletek",
          recheckAction: "Újraellenőrzés",
          hideDetailsAction: "Részletek elrejtése",
          showFolders: "Mappák megjelenítése",
          hideFolders: "Mappák elrejtése",
          previousLocations:
            "A korábbi helyeket is tartalmazza. A fő Törlés gomb a jelenlegi gyorsítótármappára vonatkozik. A modellek gombjai csak a kiválasztott példányt törlik.",
          clearModelDescription:
            "Ez csak ezt a gyorsítótárazott példányt távolítja el. Szükség esetén újra letöltődik.",
          clearAction: "Gyorsítótárak törlése",
          clearOneAction: "Törlés",
          clearingAction: "Törlés...",
          confirmTitle: "Törli a gyorsítótárazott fájlokat?",
          confirmDescription:
            "Ez körülbelül ennyi helyet szabadít fel: {size}.",
          confirmOneTitle: "Törli a következőt: {name}?",
          safety:
            "Az Unsloth a gyorsítótárat a következő alkalommal újraépíti, amikor szüksége van rá. A letöltött modellek, projektek, csevegések, beállítások és a Hugging Face-token érintetlenek maradnak.",
          hubCost:
            "Ez a modellgyorsítótár. Törlése után ezek a modellek a következő használatkor újra letöltődnek.",
          datasetsCost:
            "Törlése után ezek az adatkészletek a következő használatkor újra letöltődnek.",
          blocked: "Nincs törölve: {reason}",
          cleared: "Felszabadítva: {size}",
          partial: "Néhány gyorsítótárfájlt nem sikerült eltávolítani",
          clearFailed: "Nem sikerült törölni a gyorsítótárakat",
          names: {
            uv: "uv csomaggyorsítótár",
            pip: "pip letöltési gyorsítótár",
            npm: "npm csomaggyorsítótár",
            bun: "Bun csomaggyorsítótár",
            torchInductor: "Torch Inductor fordítási gyorsítótár",
            torchExtensions: "Torch-bővítmények buildjei",
            triton: "Triton kernelgyorsítótár",
            cuda: "CUDA kernelgyorsítótár",
            numba: "Numba fordítási gyorsítótár",
            matplotlib: "Matplotlib betűtípus-gyorsítótár",
            vllm: "vLLM gyorsítótár",
            unslothCompiled: "Unsloth lefordított moduljai",
            hfXet: "Hugging Face átviteli gyorsítótár",
            hfAssets: "Hugging Face erőforrás-gyorsítótár",
            hfDatasets: "Hugging Face adatkészlet-gyorsítótár",
            hfHub: "Hugging Face modellgyorsítótár",
          },
        },
        lowDisk: {
          title: "Fogy a lemezterület",
          criticalTitle: "A lemezterület kritikusan kevés",
          description:
            "Szabad: {free} / {total}. A gyorsítótárak törlésével helyet szabadíthat fel.",
          action: "Gyorsítótárak áttekintése",
        },
      },
      environment: {
        title: "Környezet",
        backend: "Háttérrendszer",
        python: "Python",
        torch: "Torch",
        transformers: "Transformers",
        mlxLm: "mlx-lm",
        mlxVlm: "mlx-vlm",
        uptime: "Üzemidő",
        processMemory: "Folyamat memóriája",
        notInstalled: "Nincs telepítve",
        unknown: "Ismeretlen",
        vramWithShared: "{vram} VRAM + {shared} megosztott",
      },
    },
    agents: {
      title: "Ügynökök",
      description:
        "Kódolóügynökök, például a Claude Code és a Codex csatlakoztatása egy helyi modellhez az unsloth start paranccsal.",
      agentKeywords:
        "vs code vscode copilot pi mistral vibe deepseek harness hermes openclaw opencode claude codex ügynök kódolóügynök",
      vscodeNote:
        "A VS Code kivétel: ez a parancs hozzáadja a kiszolgált modellt a VS Code Copilot Chat modelllistájához (chatLanguageModels.json).",
      intro:
        "összekapcsolja a Claude Code, Codex, DeepSeek Harness, Hermes, OpenClaw, OpenCode és más ügynököket egy, az Unsloth által helyben kiszolgált modellel, teljesen offline. OpenAI-kompatibilis kiszolgálót futtat, és soha nem módosítja az ügynök konfigurációs fájljait.",
      readDocs: "Dokumentáció megnyitása",
      copy: "Másolás",
      copied: "Másolva",
      commandBuilder: "Parancsösszeállító",
      commandShell: "Parancsértelmező minden parancshoz",
      agent: "Kódolóügynök",
      model: "Modell",
      searchModels: "GGUF-modellek keresése...",
      noModels: "Nincs egyező GGUF-modell.",
      showingModels:
        "Találatok: {shown} / {total}. Gépeljen tovább a lista szűkítéséhez.",
      quantization: "Kvantálás",
      loadingQuantizations: "Kvantált változatok betöltése...",
      noQuantizations: "Nincs külön kvantált változat",
      recommended: "Ajánlott",
      downloaded: "Letöltve",
      quantizationLoadError:
        "Nem sikerült az összes kvantált változatot betölteni. A parancs az elérhető modellértéket használja.",
      generatedCommand: "Létrehozott parancs",
      docs: "Dokumentáció",
      docsLabel: "Az unsloth start dokumentációjának megnyitása",
      agentDocs: "Beállítási dokumentáció megnyitása: {agent}",
      copyGeneratedCommand: "Létrehozott parancs másolása",
      automaticSettingsNote:
        "Ha nem adott meg kapcsolókat, az Unsloth automatikusan alkalmazza a modellhez ajánlott beállításokat.",
      configurationNote:
        "Bármely konfigurációt módosíthat is. Lásd lejjebb, vagy a",
      configurationDocs: "dokumentációt",
      configurationFlagsSuffix: "a kapcsolókhoz.",
      modelNote:
        "A Codexhez llama-server által kiszolgált GGUF-modell szükséges. Más ügynökök transformers-alapú modelleket is használhatnak; a --model elhagyásával az Unslothban már betöltött modellt használják.",
      subagent: {
        title: "Helyi modell használata alügynökként",
        description:
          "Az ügynök ({agent}) megtartja jelenlegi modelljét, a kiválasztott feladatokat pedig erre a helyi Unsloth-modellre bízza.",
        setupCommand: "Beállítási parancs",
        copySetupCommand: "Alügynök beállítási parancsának másolása",
        usagePrompt: "Ezután írja be ide ({agent}):",
        copyUsagePrompt: "Alügynök használati promptjának másolása",
        defaultPrompt:
          "Indíts el egy helyi ügynököt ennek a függvénynek a megvalósításához.",
        opencodePrompt: "@unsloth keresd meg ennek a teszthibának az okát",
      },
      quickstart: {
        title: "Parancs összeállítása",
        description:
          "Ügynök indítása az Unslothban jelenleg betöltött modellel. Előbb töltsön be egy modellt, majd cserélje a claude-ot bármelyik alább felsorolt támogatott ügynökre.",
        noneDetected: "Nem található támogatott ügynök-CLI a PATH-on.",
        installed: "Telepítve",
      },
      supportedAgents: {
        title: "Támogatott ügynökök",
        description: "Minden ügynök a saját parancsával indul:",
        requiresGguf: "GGUF-modell szükséges",
      },
      models: {
        title: "Modell kiválasztása",
        description:
          "A --model kapcsolóval választhat modellt és kvantálást, a --context-length kapcsolóval pedig beállíthatja az ablakot. Használjon kvantálási utótagot vagy explicit --gguf-variant kapcsolót.",
        suffixLabel: "Kvantálási utótaggal",
        variantLabel: "Explicit változatkapcsolóval",
      },
      options: {
        title: "Gyakori beállítások",
        description:
          "Először az Unsloth kapcsolói kerülnek feldolgozásra; amit nem ismer fel, azt változatlanul továbbadja az ügynöknek.",
        model:
          "Modell kiválasztása. --model nélkül az unsloth start az Unslothban jelenleg betöltött modellt használja, és hibát jelez, ha nincs betöltött modell.",
        contextLength:
          "A kért környezethossz beállítása (alias: --max-seq-length).",
        ggufVariant: "A GGUF kvantálási változat kiválasztása.",
        loadIn4bit:
          "4 bites betöltés be- vagy kikapcsolása Hugging Face-modellekhez.",
        tensorParallel:
          "Tenzorpárhuzamosítás be- vagy kikapcsolása több GPU-n.",
        serve: "Az automatikus helyi kiszolgáló engedélyezése vagy letiltása.",
        launch:
          "Az ügynök indítása, vagy csak a parancs és a környezet kiírása.",
        persist:
          "Az Unsloth által kezelt ügynöktárhely megtartása a futtatások között.",
        asSubagent:
          "A szülőügynök megtartja jelenlegi modelljét, az Unsloth pedig helyi alügynökként regisztrálódik (Claude Code, Codex, OpenCode és Pi).",
        apiKey:
          "Az Unsloth API-kulcs megadása (vagy állítsa be az UNSLOTH_API_KEY változót).",
        reasoning:
          "Érvelés használata a csevegésben: be, ki vagy automatikus. Az automatikus a modell csevegősablonját követi, ami általában bekapcsolt állapotot jelent.",
        reasoningEffort:
          "A modell csevegősablonjának átadott érvelési erőfeszítés, pl. medium. A szintek modellenként eltérnek, ezért olyat használjon, amelyet az adott modell elfogad. Beállítás nélkül a sablon saját értéke marad.",
        yolo: "A jóváhagyási kérések kihagyása. Csak megbízható környezetben használja.",
      },
      remote: {
        title: "Kapcsolódás távoli Unslothhoz",
        description:
          "Az unsloth start máshol futó Unsloth-példányra irányításához az indítás előtt állítsa be ezeket (vagy adja át közvetlenül a --api-key kapcsolót):",
      },
      passthrough: {
        title: "Argumentumok átadása az ügynöknek",
        description:
          "Az Unsloth kapcsolói utáni argumentumok magához az ügynökhöz kerülnek, így a natív parancsok, például a resume, továbbra is működnek:",
      },
      dryRun: {
        title: "Előnézet indítás nélkül",
        description:
          "Adja hozzá a --no-launch kapcsolót, hogy az ügynök indítása helyett a környezet és a parancs íródjon ki. Ha a --model meg van adva, a modell feloldása és betöltése ettől még megtörténhet.",
      },
    },
    chat: {
      groups: {
        conversations: {
          title: "Beszélgetések",
        },
        files: {
          title: "Fájlok és beillesztés",
        },
        display: {
          title: "Megjelenítés",
        },
        composer: {
          title: "Üzenetmező",
        },
        menu: {
          title: "Csevegőmenü",
        },
        advanced: {
          title: "Speciális",
        },
        contextTitle: "Környezet",
      },
      library: {
        label: "Csevegések kezelése",
        description:
          "Csevegések, projektek és szakaszok keresése, rendezése és rendszerezése.",
        action: "Könyvtár megnyitása",
      },
      projectAttachmentsHint:
        "Ez a beállítás minden csevegés mellékletmenüjében felülbírálható.",
      rememberParamsPerModelHint:
        "Kikapcsolva minden modell ugyanazokat a beállításokat használja.",
      autoCompactHint:
        "Helyi GGUF- és MLX-csevegések: a kiszorított körök indexelésre kerülnek, így a modell visszakeresheti őket, visszaállításkor pedig a beférő állandó utasításokat szó szerint idézi vissza, a középső helyett a legrégebbieket és a legújabbakat megtartva. Az archiváláshoz mentett csevegés és vektorindex szükséges; ezek nélkül a régebbi körök elvesznek. A beállított környezethosszt használja, nem a szabad VRAM-ot. API-modellek: ahol lehet, a szolgáltató összefoglalja a régebbi köröket, egyébként a legrégebbi körök elvesznek, amikor a csevegés megközelíti a modell közzétett környezetablakát.",
      pastedTextShortDescription:
        "A legalább {count} karakteres beillesztések .txt mellékletté válnak.",
      pastedTextOffDescription:
        "A beillesztett szöveg mindig az üzenetmezőben marad.",
      title: "Csevegés",
      description: "A csevegés működésének testreszabása ezen az eszközön.",
      modelSelection: {
        title: "Modellválasztás",
        expandQuantizations: "Kvantált változatok kibontása",
        expandQuantizationsDescription:
          "A GGUF kvantált változatok megjelenítése az „Az eszközön” nézetben az egyes modellek megnyitása nélkül.",
        showAllQuantizations: "Összes kvantált változat",
        showAllQuantizationsDescription:
          "A le nem töltött kvantált változatok is megjelennek az „Az eszközön” nézetben.",
        showMemoryBar: "VRAM-használati sáv",
        showMemoryBarDescription:
          "A modellsúlyok, a környezet és a spekulatív dekódolás becsült VRAM-igényének megjelenítése.",
        fitOnDeviceOnly: "Csak az elférő modellek megjelenítése",
        fitOnDeviceOnlyDescription:
          "Az eszköz GPU-memóriájához és RAM-jához túl nagy modellek elrejtése a modellválasztóban és a Modellközpontban. A letöltött modellek láthatók maradnak.",
      },
      menu: {
        title: "Csevegőmenü",
        description:
          "Elemek rögzítése a csevegés + menüjében, és átrendezésük húzással. A többi a „Továbbiak” alá kerül.",
        deepResearch: "Mélykutatás",
        browserUse: "Böngészőhasználat",
        chatWithFiles: "Csevegés fájlokkal (RAG)",
        mcp: "MCP",
        skills: "Készségek",
        savedPrompts: "Mentett promptok",
        compareChat: "Csevegés összehasonlítása",
        exportChat: "Csevegés exportálása",
        reset: "Alapértelmezés visszaállítása",
        addFiles: "Fényképek és fájlok hozzáadása",
        webSearch: "Webes keresés",
        tools: "Eszközök",
        more: "Továbbiak",
      },
      pastedTextThreshold: "Hosszú beillesztések tömörítése",
      pastedTextThresholdDescription:
        "Az üzenetmezőbe történő közvetlen beillesztéshez nyomja meg a következőt: {shortcut}.",
      pastedTextThresholdOff: "Ki",
      autoScroll: "Görgetés generálás közben",
      autoScrollDescription:
        "Az automatikus görgetés látótérben tartja az új szöveget. A kézi mód helyben marad, így az elejétől olvashat.",
      autoScrollAuto: "Automatikus görgetés",
      autoScrollManual: "Kézi",
      autoScrollKeywords:
        "scroll autoscroll auto-scroll follow stick bottom jump streaming generating viewport lock hold auto manual görgetés automatikus követés alja generálás kézi",
      replyNotification: "Értesítés a válasz elkészültekor",
      replyNotificationDescription:
        "Rendszerértesítés megjelenítése, amikor egy válasz elkészül, miközben az Unsloth kis méretre van állítva vagy a háttérben fut.",
      replyNotificationBlocked:
        "Az értesítések le vannak tiltva. Engedélyezze őket az Unsloth számára a böngésző vagy a rendszer beállításaiban, majd próbálja újra.",
      replyNotificationKeywords:
        "notify notification alert done finished complete ready response reply background minimized desktop sound értesítés figyelmeztetés kész válasz háttér hang",
      showResponseModel: "Válaszadó modell",
      showResponseModelDescription:
        "A modell adatainak megjelenítése az asszisztens válaszaiban.",
      showReplyTimes: "Válaszidők",
      showReplyTimesDescription:
        "Az egyes válaszok elkészülési idejének megjelenítése a válasz alatt.",
      showReplyTimesKeywords:
        "timestamp time clock date hour when finished arrived reply response message időbélyeg idő óra dátum válasz üzenet",
      showGenerationDetails: "Generálási részletek",
      showGenerationDetailsDescription:
        "A prompt- és válaszsebesség élő megjelenítése tokenszámokkal, valamint annak, hogy hová töltődtek be a modell rétegei.",
      showGenerationDetailsKeywords:
        "verbose developer stats speed tok/s prefill decode layers offload vram experts részletes fejlesztői statisztika sebesség rétegek áthelyezés szakértők",
      inlineEditResponse: "Válasz szerkesztése a válaszokon",
      inlineEditResponseDescription:
        "A Válasz szerkesztése gomb minden válaszon látható a Továbbiak menü helyett.",
      composerAtBottom: "Üzenetmező alul",
      composerAtBottomDescription:
        "Az új csevegések az oldal közepe helyett az alján lévő üzenetmezővel indulnak.",
      projectAttachments: "Fájlok megosztása a projekten belül",
      projectAttachmentsDescription:
        "Az új csevegésmellékletek a projekt minden csevegésében elérhetők lesznek.",
      rememberParamsPerModel: "Beállítások megjegyzése modellenként",
      rememberParamsPerModelDescription:
        "Minden modellnél visszaállítja a legutóbb használt promptot, hőmérsékletet és egyéb beállításokat.",
      autoCompact: "Hosszú csevegések automatikus tömörítése",
      autoCompactDescription:
        "Amikor egy csevegés megtölti a környezetét, a régebbi körök kereshető archívumba kerülnek.",
      autoCompactKeywords:
        "compaction compact auto-compact context window truncate rolling checkpoint headroom archive retrieval recall rag search tömörítés környezetablak csonkítás archívum visszakeresés keresés",
      keepRunningWhenTabClosed: "Futás folytatása a lap bezárása után",
      keepRunningWhenTabClosedDescription:
        "A válaszok, a webes keresés és a Mélykutatás a lap bezárása után is folytatódnak a kiszolgálón. Kapcsolja ki, ha inkább le szeretné állítani őket.",
      keepRunningWhenTabClosedHint:
        "A tovább futó futtatás folytatásához nyissa meg újra a csevegést bármely eszközön. Kikapcsolva a futtatás az utolsó lapjának bezárása után körülbelül egy percen belül leáll, így egy újratöltés nem állítja le. A beállítás módosítása után indított futtatásokra vonatkozik. Az ideiglenes csevegések, az API-modellektől vagy mellékletekre érkező válaszok, valamint a böngészőpanelt használó körök mindig leállnak a lapjukkal együtt.",
      keepRunningWhenTabClosedKeywords:
        "background tab close closed leave keep running continue stop cancel deep research web search resume reconnect device phone háttér lap bezárás futás folytatás leállítás mélykutatás webes keresés újracsatlakozás eszköz telefon",
      firstTokenLabel: "Első token várakozási korlátja",
      firstTokenDescription:
        "Mennyi ideig várhatnak a helyi GGUF-modellek promptfeldolgozási előrehaladás nélkül az első token előtt. Minden fiók új várakozásaira vonatkozik. Korlátlan esetén addig vár, amíg le nem állítja.",
      firstTokenDefault: "20 perc (alapértelmezett)",
      firstTokenCustom: "Egyéni",
      firstTokenUnlimited: "Korlátlan",
      firstTokenMinutes: "Első token várakozási ideje percben",
      firstTokenRange: "Perc (1–10 080)",
      firstTokenSave: "Mentés",
      firstTokenSaving: "Mentés…",
      firstTokenLoadError:
        "Nem sikerült betölteni az első token várakozási korlátját.",
      firstTokenSaveError:
        "Nem sikerült menteni az első token várakozási korlátját.",
      firstTokenKeywords:
        "timeout first token prefill slow CPU overnight unlimited waiting limit időkorlát első token lassú éjszaka korlátlan várakozás",
      toolResultLimit: "Eszközkimenet hossza",
      toolResultLimitDescription:
        "Egy Python-, Terminál- vagy MCP-eszközeredményből a modell által beolvasott karakterek száma. Alapértelmezés: {count}.",
      toolResultLimitHint:
        "A hosszabb Python- és Terminál-kimenet fájlba kerül, amelyet a modell részletekben olvashat be. A magasabb korlát az Unslothban futó modellekre vonatkozik, a kis környezetablakok így is kevesebbet kapnak. Az API-modellek az alapértelmezést használják, hacsak nem csökkenti.",
      toolResultLimitDefault: "{count} (alapértelmezett)",
      toolResultLimitLocked: "Az UNSLOTH_TOOL_RESULT_MAX_CHARS állítja be",
      toolResultLimitLoadError:
        "Nem sikerült betölteni az eszközkimenet hosszát.",
      toolResultLimitSaveError:
        "Nem sikerült menteni az eszközkimenet hosszát.",
      toolResultLimitKeywords:
        "tool output result truncate truncated truncation limit cap characters chars length terminal python mcp eszköz kimenet eredmény csonkítás korlát karakterek hossz terminál",
      visibility: {
        collapsed: "Összecsukva",
        auto: "Kibontva futás közben",
        expanded: "Mindig kibontva",
      },
      visibilityKeywords:
        "collapse collapsed expand expanded open closed reasoning thinking tool calls tool activity fold group streaming összecsukás kibontás érvelés gondolkodás eszközhívások csoportosítás",
      thinking: {
        visibility: "Gondolkodás",
        visibilityDescription:
          "Hogyan nyílik meg az érvelés. Bármely blokkot továbbra is saját maga kibonthat vagy összecsukhat.",
      },
      memory: {
        title: "Memória",
        description:
          "Jegyzetek, amelyeket minden csevegés beolvas, minden modellnél, legyen az helyi vagy felhőalapú. A projektekben lévő csevegések a projekt AGENTS.md fájlját is beolvassák.",
        enable: "Memória használata a csevegésekben",
        enableDescription:
          "Ezek a jegyzetek minden csevegéssel elküldésre kerülnek.",
        locked: "Ezen a telepítésen az UNSLOTH_STUDIO_AGENTS_MD kikapcsolta.",
        editor: "Memóriajegyzetek",
        placeholder:
          "Például:\n- Pythonban írok, és uv-t használok.\n- Legyenek rövidek a válaszok.",
        hint: "Mentési hely: {path}. Ezt a fájlt bármely szövegszerkesztőben is szerkesztheti.",
        size: "{used} / {max} KB",
        save: "Mentés",
        saved: "Memória mentve",
        loadError: "Nem sikerült betölteni a memóriát.",
        saveError: "Nem sikerült menteni a memóriát.",
        keywords:
          "memory remember personalization personalisation profile preferences notes facts custom instructions about me agents.md claude.md memória megjegyzés személyre szabás profil beállítások jegyzetek egyéni utasítások rólam",
      },
      currentDate: {
        label: "A mai dátum közlése a modellel",
        description:
          "A mai dátum hozzáadása, hogy a keresés és a kutatás friss forrásokat találjon.",
        loadError: "Nem sikerült betölteni az aktuális dátum beállításait",
        saveError: "Nem sikerült frissíteni az aktuális dátum beállításait",
      },
      tools: {
        visibility: "Eszközhívások",
        visibilityDescription:
          "Hogyan nyílik meg az eszközök tevékenysége. Bármely hívást továbbra is saját maga kibonthat vagy összecsukhat.",
        foldIntoThinking: "Eszközhívások csoportosítása a Gondolkodás alatt",
        foldIntoThinkingDescription:
          "Egy kör eszközhívásai a Gondolkodás blokkon belül jelennek meg, nem külön sorokban.",
        foldIntoThinkingBlocked:
          "Nem érhető el, amíg az eszközhívások Mindig kibontva értékre vannak állítva, mert ez külön sorokban tartja őket.",
      },
      webSearch: {
        title: "Webes keresés",
        images: "Képek a webes keresésből",
        imagesDescription: "Képek belefoglalása a keresési eredményekbe.",
      },
      artifacts: {
        title: "HTML-oldalak",
        collapseHtmlBlocks: "HTML-blokkok összecsukása",
        collapseHtmlBlocksDescription:
          "A teljes HTML-dokumentumok a csevegésben kód helyett a böngészőben megnyíló kártyaként jelennek meg.",
        allowNetworkAccess: "Hálózati hozzáférés engedélyezése HTML-oldalaknak",
        allowNetworkAccessDescription:
          "A csevegésből származó HTML-oldalak külső szkripteket, stílusokat, betűtípusokat és médiát tölthetnek be. Offline előnézethez kapcsolja ki.",
        blockedBanner: "{count} külső erőforrás letiltva innen: {hosts}.",
        blockedBannerPlural: "{count} külső erőforrás letiltva innen: {hosts}.",
        blockedBannerAction: "Engedélyezés ennél az oldalnál",
        blockedTitle:
          "A hálózati hozzáférés ki van kapcsolva a HTML-oldalaknál",
        blockedHint:
          "Ha szeretné, hogy a HTML-oldalak külső erőforrásokat tölthessenek be, kapcsolja be ezt a beállítást a Beállítások → Böngésző helyen: „{setting}”, vagy engedélyezze csak ennél az oldalnál.",
        blockedSettingsAction: "Beállítások megnyitása",
        blockedDismiss: "Elvetés",
        errorTitle: "Hiba történt ezen az oldalon",
        errorTitlePlural: "{count} hiba történt ezen az oldalon",
        errorHint:
          "A Javítás a modellel a hibát az üzenetmezőbe helyezi. Semmi nem kerül elküldésre, amíg el nem küldi.",
        errorBannerAction: "Javítás a modellel",
        errorConsoleAction: "Konzol megnyitása",
        errorConsoleHideAction: "Konzol elrejtése",
        errorLocation: "{line}. sor, {column}. oszlop",
        errorLine: "{line}. sor",
        consoleTitle: "Konzol",
        reloadCanvas: "Újrafuttatás",
        consoleMessageCount: "{count} üzenet",
        consoleMessageCountPlural: "{count} üzenet",
        consoleClear: "Konzol törlése",
        consoleClose: "Konzol bezárása",
        consoleEmpty: "Még nincs konzolkimenet.",
        consoleCapped:
          "Csak az utolsó {count} bejegyzés marad meg; a korábbi kimenet elveszett.",
      },
      data: "Adatok",
      exportHistory: "Csevegési előzmények exportálása",
      exportHistoryDescription:
        "Az összes csevegés és üzenet letöltése JSON formátumban.",
      exportAction: "Exportálás",
      exportingAction: "Exportálás...",
      exportConversations: "Legutóbbiak és projektek exportálása",
      exportConversationsDescription:
        "A Legutóbbiak, vagy a Legutóbbiak a projektcsevegésekkel együtt letölthetők Training JSONL, CSV, ShareGPT JSONL vagy Markdown formátumban, egyesítve vagy csevegésenként. Az üzenetenkénti JSONL csevegésenként érhető el.",
      exportConversationsAction: "Exportálás",
      exportScopeRecents: "Legutóbbiak",
      exportScopeAll: "Legutóbbiak + projektek",
      exportCombinedSuffix: "(egyesítve)",
      exportPerChatSuffix: "(csevegésenként)",
      importChats: "Csevegések importálása",
      importChatsDescription:
        "JSON-, JSONL-, NDJSON-, CSV- vagy Markdown-fájlok importálása a Legutóbbiak közé.",
      importChatsAction: "Importálás",
      importNoConversations: "A fájlban nem található beszélgetés.",
      importedOneChat: "1 beszélgetés importálva a Legutóbbiak közé.",
      importedChatCount: "{count} beszélgetés importálva a Legutóbbiak közé.",
      importingChats: "Csevegések importálása: eddig {count} ({percent}%)...",
      importedChatCountPartial:
        "{count} beszélgetés importálva a Legutóbbiak közé; {failed} nem menthető.",
      importFailed: "Az importálás nem sikerült.",
      importFromSource: "Importálás innen: {source}",
      importFromSourceDescription:
        "Beszélgetések átmásolása az Unslothba innen: {source}, a hozzájuk tartozó projekt szerint csoportosítva.",
      importingAction: "Importálás...",
      importedSourceOneChat: "1 beszélgetés importálva innen: {source}.",
      importedSourceChatCount:
        "{count} beszélgetés importálva innen: {source}.",
      sourceUpdated: "{count} új üzenet hozzáadva innen: {source}.",
      sourceUpToDate: "A beszélgetések már naprakészek. Forrás: {source}.",
      importedSourcePartial:
        "Néhány beszélgetést nem sikerült importálni. Forrás: {source}.",
      clearHistory: "Csevegési előzmények törlése",
      clearHistoryDescription:
        "A csevegési előzmények törlése erről az eszközről.",
      clearAction: "Törlés",
      clearAllChats: "Összes csevegés törlése",
      clearAllChatsDescription:
        "Az eszközön lévő összes csevegés végleges törlése.",
      noChatsToClear: "Nincs törölhető csevegés.",
      clearOneChatDescription:
        "Az eszközön lévő egyetlen csevegés végleges törlése.",
      clearChatCountDescription:
        "Az eszközön lévő összes csevegés ({count}) végleges törlése.",
      clearChatsAction: "Csevegések törlése",
      clearOneChatTitle: "Töröl 1 csevegést?",
      clearChatsTitle: "Töröl {count} csevegést?",
      clearChatsConfirmDescription:
        "Véglegesen törli az eszközön lévő összes csevegést. Ez a művelet nem vonható vissza.",
      clearingAction: "Törlés...",
      clearOneChatAction: "1 csevegés törlése",
      clearChatCountAction: "{count} csevegés törlése",
      clearedAllChats: "Az összes csevegés törölve",
      clearedOneChat: "1 csevegés törölve",
      clearedChatCount: "{count} csevegés törölve",
      someChatsCouldNotBeCleared: "Néhány csevegést nem sikerült törölni",
      chatsClearedRemainOne:
        "{clearedCount} csevegés törölve; 1 csevegés megmaradt. Próbálja újra.",
      chatsClearedRemain:
        "{clearedCount} csevegés törölve; {remainingCount} csevegés megmaradt. Próbálja újra.",
      oneChatClearedRemain:
        "1 csevegés törölve; {remainingCount} csevegés megmaradt. Próbálja újra.",
      oneChatClearedRemainOne:
        "1 csevegés törölve; 1 csevegés megmaradt. Próbálja újra.",
      storageClearFailedOne:
        "Nem sikerült törölni a tárhelyet; 1 csevegés megmaradhatott. Próbálja újra.",
      storageClearFailed:
        "Nem sikerült törölni a tárhelyet; {count} csevegés megmaradhatott. Próbálja újra.",
      failedToClearChats: "Nem sikerült törölni a csevegéseket",
    },
    data: {
      library: {
        defaultSort: "Alapértelmezett",
        updated: "Frissítve",
        created: "Létrehozva",
        oldest: "Legrégebbi elöl",
        alphabetical: "Betűrendben",
        allChats: "Összes csevegés",
        singleChats: "Egyedi csevegések",
        compareChats: "Összehasonlító csevegések",
        allProjects: "Összes projekt",
        noProject: "Nincs projekt",
        unavailableProject: "Nem elérhető projekt",
        filterSort: "Szűrés és rendezés",
        sort: "Rendezés",
        type: "Típus",
        sortBy: "Rendezés módja",
        filterProject: "Szűrés projekt szerint",
        searchProjects: "Projektek keresése",
        noProjects: "Nem található projekt.",
        untitled: "Névtelen",
        oneChat: "1 csevegés",
        chatCount: "{count} csevegés",
        searchChats: "Csevegések vagy projektek keresése",
        searchArchivedChats: "Archivált csevegések vagy projektek keresése",
        searchImages: "Archivált képek keresése",
        searchVideos: "Archivált videók keresése",
        searchAudio: "Archivált hangfájlok keresése",
        unarchive: "Archiválás visszavonása",
        unarchiveAll: "Összes archiválásának visszavonása",
        unarchiveAllInProject:
          "Projekt összes elemének visszaállítása az archívumból",
        deleteAllInProject: "Projekt összes elemének törlése",
        groupActions: "Csoportműveletek: {group}",
        unarchiveResults: "Találatok visszaállítása az archívumból",
        deleteResults: "Találatok törlése",
        noChats: "Nincs a keresésnek megfelelő csevegés.",
        noArchivedChats: "Nincsenek archivált csevegések.",
        noArchivedMatches: "Nincs a keresésnek megfelelő archivált csevegés.",
        noMedia: "Nincsenek archivált elemek.",
        noMediaMatches: "Nincs a keresésnek megfelelő archivált elem.",
        itemCount: "Elemek: {count}",
        incompleteSearch:
          "A keresés nem teljes. Próbálja újra betölteni a hátralévő elemeket.",
        searchingRemaining:
          "Keresés a hátralévő elemek között ({count} betöltve)...",
        noLoadedMatches: "Nincs találat az eddig betöltött elemek között.",
        deleteItem: "Törlés: {title}",
        unarchiveItem: "Archiválás visszavonása: {title}",
        deleteItemsTitle: "Archivált elemek törlése ({count})",
        unarchiveItemsTitle: "Elemek visszaállítása az archívumból ({count})",
        deleteFilesWarning:
          "Ezek a fájlok véglegesen törlődnek. Ez a művelet nem vonható vissza.",
        restoreWarning: "Ezek az elemek visszakerülnek a galériába.",
        working: "Folyamatban...",
        loadFailed: "Nem sikerült betölteni az archivált elemeket",
        loadMoreFailed: "Nem sikerült további archivált elemeket betölteni",
        restoreFailed:
          "Nem sikerült visszaállítani a kijelölt elemeket az archívumból",
        deleteFailed: "Nem sikerült törölni a kijelölt elemeket",
        projectsFailed: "Nem sikerült betölteni az archivált projekteket",
        pageStalled: "Az archívum oldala nem lépett tovább. Próbálja újra.",
        pageChanged: "Az archívum betöltés közben megváltozott. Próbálja újra.",
        selectAll: "Összes látható csevegés kijelölése",
        selectItem: "Kijelölés: „{title}”",
        selectedChats: "Kijelölt csevegések: {count}",
        move: "Áthelyezés",
        pin: "Rögzítés",
        unpin: "Rögzítés feloldása",
        archive: "Archiválás",
        deletedChats: "Törölt csevegések: {count}",
        restoredChats: "Archívumból visszaállított csevegések: {count}",
        pinnedChats: "Rögzített csevegések: {count}",
        unpinnedChats: "Rögzítésből feloldott csevegések: {count}",
        movedChatsToProject:
          "Áthelyezett csevegések ({project} projektbe): {count}",
        movedChatsToRecents:
          "A Legutóbbiak közé áthelyezett csevegések: {count}",
        moveFailed: "Nem sikerült áthelyezni a csevegéseket",
        deleteChatsTitle: "Csevegések törlése ({count})",
        deleteChatsWarning:
          "Törli a kijelölt csevegéseket ({count})? Ez a művelet nem vonható vissza.",
        deleteArchivedTitle: "{count} archivált csevegés törlése",
        deleteArchivedWarning:
          "Törli a kijelölt archivált csevegéseket ({count})? Ez a művelet nem vonható vissza.",
        deleting: "Törlés...",
      },
      title: "Adatok",
      backToData: "Vissza az Adatokhoz",
      exportFailed: "Nem sikerült exportálni a csevegéseket",
      description:
        "Az eszközön tárolt csevegési előzmények és feltöltött fájlok kezelése.",
      manageFiles: {
        label: "Fájlok kezelése",
        description:
          "Képek, videók, hangfájlok és egyéb fájlok keresése, rendezése és rendszerezése.",
        action: "Könyvtár megnyitása",
      },
      archivedChats: "Archivált csevegések",
      archivedChatsDescription:
        "Az archivált csevegések megtekintése és kezelése.",
      archivedImages: "Archivált képek",
      archivedImagesDescription: "Az archivált képek megtekintése és kezelése.",
      archivedVideos: "Archivált videók",
      archivedVideosDescription:
        "Az archivált videók megtekintése és kezelése.",
      archivedAudio: "Archivált hangfájlok",
      archivedAudioDescription:
        "Az archivált hangfelvételek megtekintése és kezelése.",
      manageAction: "Kezelés",
      exportArchivedChats: "Exportálás",
      exportingArchivedChats: "Exportálás...",
      exportedOneArchivedChat: "1 archivált csevegés exportálva",
      exportedArchivedChatCount: "{count} archivált csevegés exportálva",
      noArchivedChatsToExport: "Nincs exportálható archivált csevegés.",
      failedToExportArchivedChats:
        "Nem sikerült exportálni az archivált csevegéseket",
      archiveAllChats: "Összes csevegés archiválása",
      archiveAllChatsDescription:
        "A Legutóbbiak és a Projektek összes csevegésének áthelyezése az archívumba.",
      noChatsToArchive: "Nincs archiválható csevegés.",
      archiveAllAction: "Összes archiválása",
      archivingAction: "Archiválás...",
      archiveAllChatsTitle: "Archiválja az összes csevegést?",
      archiveAllChatsConfirmDescription:
        "Az eszközön lévő összes csevegést áthelyezi az archívumba. Az archivált csevegések elérhetők maradnak, és bármikor visszaállíthatók az archívumból.",
      archivedAllChats: "Az összes csevegés archiválva",
      archivedOneChat: "1 csevegés archiválva",
      archivedChatCount: "{count} csevegés archiválva",
      failedToArchiveChats: "Nem sikerült archiválni a csevegéseket",
      confirmBeforeDeleting: "Megerősítés törlés előtt",
      confirmBeforeDeletingDescription:
        "Megerősítés kérése egy csevegés törlése előtt. Kapcsolja ki az azonnali törléshez.",
      alwaysDeleteFiles: "Fájlok törlése mindig",
      alwaysDeleteFilesDescription:
        "Egy csevegés törlése a saját sandboxmappáját is eltávolítja a lemezről. A projekten belül írt fájlok a projekt munkaterületén maradnak.",
      chatsSection: "Csevegések és archívumok",
      transferSection: "Importálás és exportálás",
      deletionSection: "Törlés",
      archives: "Archívumok",
      archiveChatsLabel: "Csevegések",
      archiveVideosLabel: "Videók",
      exportFormatsSummary:
        "A Legutóbbiak és a projektcsevegések exportálása a kívánt formátumban.",
      trainingSummary: "Csevegéseiből finomhangolási adatkészletet készíthet.",
      confirmDeletionSummary:
        "Megerősítés kérése az egyes csevegések törlése előtt.",
      sandboxFiles: "Csevegések sandboxfájljai",
      sandboxFilesDescription: "Alapértelmezett művelet csevegés törlésekor.",
      keepSandboxFiles: "Sandboxfájlok megtartása",
      deleteSandboxFiles: "Sandboxfájlok törlése",
      projectFilesKept:
        "A sandboxok az egyes csevegésekhez létrehozott mappák. A projektmunkaterületeken mentett fájlok megmaradnak.",
      deleteAllAction: "Összes törlése",
      deleteChatsOnly: "Csak a csevegések törlése…",
      deleteChatsAndSandboxes: "Csevegések és sandboxok törlése…",
      deleteSandboxFilesDescription:
        "Minden csevegés sandboxmappáját és a benne lévő fájlokat is törli. A projektmunkaterületeken lévő fájlok megmaradnak.",
      filesSection: "Fájlok",
      uploadedFiles: "Feltöltött fájlok",
      uploadedFilesDescription:
        "A csevegésekbe, projektekbe és tudásbázisokba feltöltött fájlok megtekintése és kezelése.",
      fineTuneExport: "Csevegések használata tanítási adatként",
      fineTuneExportDescription:
        "Finomhangolási JSONL-adatkészlet létrehozása a csevegéseiből. Töltse be a Tanítás lapon, finomítsa a Receptekben, vagy exportálja.",
      fineTuneExportAction: "JSONL exportálása",
      fineTuneRunAction: "Futtatás",
      fineTuneExportingAction: "Exportálás...",
      fineTuneOpenRecipesAction: "Megnyitás a Receptekben",
      fineTuneOpeningRecipesAction: "Megnyitás...",
      fineTuneTrainAction: "Betöltés a Tanítás lapon",
      fineTuneTrainingAction: "Betöltés...",
      fineTuneExportFailed: "Nem sikerült exportálni a tanítási adatokat",
      fineTuneRecipeFailed:
        "Nem sikerült megnyitni a csevegéseket a Receptekben",
      fineTuneTrainFailed:
        "Nem sikerült betölteni az adatkészletet a Tanítás lapon",
    },
    connections: {
      title: "Kapcsolatok",
      description: "Szolgáltatók és külső kapcsolatok kezelése.",
    },
    remoteLan: {
      title: "Távoli elérés és LAN",
      description:
        "Az Unsloth elérése más eszközeiről, a helyi hálózaton vagy egy ideiglenes nyilvános URL-en keresztül.",
    },
    apiKeys: {
      title: "API",
      description:
        "Az Unsloth elérése az OpenAI- és Anthropic-kompatibilis API-n keresztül.",
      readDocs: "API-dokumentáció megnyitása",
      noAccess: "Még nincs API-hozzáférés.",
      accessTokens: "Hozzáférési tokenek",
      quickstartTitle: "Gyors kezdés",
      columnName: "Név",
      columnToken: "Token",
      columnCreated: "Létrehozva",
      columnLastUsed: "Utoljára használva",
      columnExpires: "Lejárat",
      loadError: "Nem sikerült betölteni az API-hozzáférést.",
      createError: "Nem sikerült létrehozni a hozzáférési tokent.",
      revokeError: "Nem sikerült visszavonni a hozzáférési tokent.",
      never: "Soha",
      tokenNamePlaceholder: "Token neve",
      newAccessTokenName: "Új hozzáférési token neve",
      createToken: "Token létrehozása",
      neverExpiresHint: "Soha nem jár le",
      expiresInDaysHint: "{n} nap múlva lejár",
      createTokenHint: "Token létrehozása ezzel a névvel: „{name}”",
      defaultTokenName: "Új token {n}",
      creating: "Létrehozás...",
      newTokenCreated: "Új hozzáférési token létrehozva",
      accessTokenCopied: "Hozzáférési token kimásolva",
      copyAccessToken: "Hozzáférési token másolása",
      copyNow: "Másolja most – később nem jelenik meg újra.",
      usageExamples: "Csevegési API",
      usageNoModel:
        "Töltsön be vagy töltsön le egy modellt a futtatható példák megtekintéséhez. Ennek a kiszolgálónak még nincs megnevezhető modellje.",
      usageTools: "Eszközök",
      exampleLanguage: "Nyelv",
      exampleRequest: "Kérés",
      examplePlatform: "Platform",
      requestPlain: "Alap",
      requestTools: "Eszközökkel",
      requestAdvanced: "Speciális mintavételezés",
      exampleKind: "Példa",
      osUnix: "Linux / macOS / WSL",
      osWindows: "Windows",
      secureHttps: "Biztonságos HTTPS",
      secureHttpsHint:
        "A 0.0.0.0 port továbbra is bárhonnan elérhető. A teljes biztonság érdekében indítsa az Unslothot a --secure kapcsolóval, így csak ez a HTTPS-hivatkozás lesz elérhető.",
      copyTunnelUrl: "Alagút URL-címének másolása",
      copySnippet: "Kódrészlet másolása",
      copy: "Másolás",
      copied: "Másolva",
      codingAgents: "Kódolóügynökök",
      codingAgentsHint:
        "Kódolóügynök indítása ehhez a kiszolgálóhoz. A betöltött modellt használja; helyi kiszolgáló esetén automatikusan létrejön egy API-kulcs, távoli kiszolgálónál a parancs tartalmazza azt.",
      openAgentsTab: "Megnyitás az Ügynökök lapon",
      codingAgentsSwap:
        "A claude helyett a codex, openclaw, opencode vagy hermes is használható.",
      relativeNever: "soha",
      relativeJustNow: "az imént",
      expired: "lejárt",
      expiredStatus: "Lejárt",
      today: "ma",
      actionsFor: "Műveletek: {name}",
      copyPrefix: "Előtag másolása",
      copyFailed: "Nem sikerült másolni",
      revokeToken: "Token visszavonása",
      regenerateToken: "Token újragenerálása",
      regenerateTitle: "Újragenerálja ezt a hozzáférési tokent: „{name}”?",
      regenerateDescription:
        "Egy azonos nevű és lejárati idejű új token lép ennek a helyére. A jelenlegi tokent használó alkalmazások azonnal elveszítik a hozzáférést.",
      regenerateAction: "Újragenerálás",
      regenerating: "Újragenerálás...",
      regenerateError: "Nem sikerült újragenerálni a hozzáférési tokent.",
      revokeTitle: "Visszavonja ezt a hozzáférési tokent: „{name}”?",
      revokeDescription:
        "Az ezt a tokent használó alkalmazások azonnal elveszítik a hozzáférést. Ez a művelet nem vonható vissza.",
      revokeAction: "Visszavonás: „{name}”",
      revoking: "Visszavonás...",
      audioApi: {
        title: "Hang-API",
        description:
          "Beszéd, hangklónozás, átírás és a Hang oldal munkafolyamatai.",
        speak: "Beszéd",
        clone: "Klónozás",
        transcribe: "Átírás",
        workflows: "Munkafolyamatok",
        separate: "Szétválasztás",
        convert: "Átalakítás",
        music: "Zene",
        edit: "Szerkesztés",
        placeholderModel:
          "Egyik letöltött modell sem tudja még futtatni ezt a példát, ezért a megjelenített modell csak helyettesítő. Először töltsön le egyet a Hang oldalon.",
        autoSwitchOff:
          "A modell csak akkor töltődik be név alapján, ha az Automatikus modellváltás be van kapcsolva. Kapcsolja be, vagy először töltse be a modellt a hozzá tartozó Hang oldalon.",
        useViaApi: "Használat API-n keresztül",
      },
      mediaApi: {
        title: "Kép- és videó-API",
        description:
          "Képek és videók generálása saját kódból az OpenAI-kompatibilis végpontokkal.",
        images: "Képek",
        video: "Videó",
        placeholderModel:
          "A megjelenített modell csak példa. Töltsön be egy modellt a Képek vagy a Videó oldalon, és nyissa meg ezt az ottani API gombbal a kitöltéshez.",
        autoSwitchOff:
          "A kép- és videómodell automatikus váltása ki van kapcsolva, ezért a kérések az oldalon betöltött modellen futnak, függetlenül attól, hogy melyik modellt nevezik meg.",
      },
      trainingApi: {
        title: "Tanítási API",
        description:
          "Modellek finomhangolása saját kódból az OpenAI-kompatibilis finomhangolási végpontokkal: töltsön fel egy JSONL-fájlt, hozzon létre egy feladatot, kövesse az eseményeit, majd csevegjen a betanított modellel.",
        placeholder:
          "A megjelenített modell és beállítások csak példák. Nyissa meg ezt a Tanítás oldal API gombjával, hogy a jelenlegi beállításai kerüljenek bele.",
        hubDataset:
          "A Tanítás oldal Hugging Face- vagy S3-adatkészletet használ. Az API feltöltött fájlokon tanít, ezért mentse az adatkészletet JSONL formátumban, és azt töltse fel.",
        batching:
          "A sorban váró, egymással kompatibilis feladatok automatikusan együtt tanulnak.",
        chatNote:
          "A betanított modellel név alapján csak bekapcsolt Automatikus modellváltás mellett lehet csevegni; ellenkező esetben először töltse be a Csevegés oldalon.",
        cancelTitle: "Feladat megszakítása",
        cancelDescription:
          "Leállít egy sorban álló vagy futó feladatot. Nem része a fenti példának.",
      },
      decisionApi: {
        title: "Decision API",
        docs: "Dokumentáció",
        docsLabel: "A Decision API dokumentációjának megnyitása",
        description:
          "Igen/nem, feleletválasztós és pontozásos kérdések megválaszolása szövegekről egy ezen a gépen futó modellel vagy a Kapcsolatok közül választott döntési modellel. A TypeSafe SDK-val is működik.",
        enable: "Kérések kiszolgálása",
        enableDescription:
          "Kiszolgálja a /v1/systemone végpontot. Bekapcsoláskor letöltődik a modell.",
        enableRemoteDescription: "Kiszolgálja a /v1/systemone végpontot.",
        lockedByEnv: "Beállítva ezzel: {name}.",
        model: "Modell",
        thisMachine: "Ez a gép",
        modelMultilingual: "Többnyelvű",
        modelEnglish: "Angol",
        modelTypedDecisions: "Típusos döntések",
        modelClefFlash: "Clef-flash",
        modelClef: "Clef",
        recommended: "Ajánlott",
        device: "Futtatás helye",
        deviceDescription:
          "A GPU gyorsabban válaszol, és a modell kiürítésekor felszabadítja a memóriáját.",
        deviceCpu: "CPU",
        deviceGpu: "GPU",
        backend: "Futtatókörnyezet",
        backendAuto: "Automatikus",
        backendDescription:
          "Automatikus módban a Clef a llama.cpp-n keresztül fut, ha a modellnek van GGUF-változata, egyébként a PyTorch-ra vált. A llama.cpp képeket is be tud olvasni.",
        backendDescriptionMlx:
          "Automatikus módban a Clef Apple Siliconon MLX-en keresztül fut, és a llama.cpp-n keresztül, ha a modellnek csak a GGUF-változata van betöltve vagy letöltve. Az MLX csak Clef-modelleknél olvas be képeket. A llama.cpp képeket is be tud olvasni.",
        backendStatus: "Futtatókörnyezet: {backend}",
        backendNone: "nem érhető el",
        mediaImages: "Szöveget és képeket olvas be.",
        mediaText: "Csak szöveget olvas be.",
        checking: "Ellenőrzés…",
        notDownloaded: "Nincs letöltve · {size}",
        downloading: "Letöltés…",
        downloaded: "Letöltve · az első kéréskor töltődik be",
        ready: "Kész · az első kéréskor töltődik be",
        installing: "Telepítés…",
        loading: "Betöltés…",
        loadedOn: "Betöltve, eszköz: {device}",
        download: "Letöltés",
        downloadConfirmTitle: "Letölti a Laya {model} modellt?",
        downloadConfirmTitleModel: "Letölti ezt a modellt: {model}?",
        downloadConfirmBody:
          "A Decision API-nak szüksége van erre a modellre a kérések megválaszolásához. Mérete körülbelül {size}, egyszer töltődik le a Hugging Face-gyorsítótárba.",
        unload: "Kiürítés",
        downloadBusy: "Egy Decision API-modell letöltése már folyamatban van.",
        downloadFailed: "Nem sikerült elindítani a letöltést.",
        saveFailed: "Nem sikerült menteni a Decision API beállítását.",
        loadError: "Nem sikerült betölteni a Decision API beállításait.",
        sendsTo: "A kérések címzettje: {provider}.",
        connectionMissing:
          "Ez a kapcsolat megszűnt, vagy nincsenek döntési modelljei. Válasszon másik modellt.",
        addConnection:
          "Felhőben futó döntési modell használatához adjon hozzá egyet itt: ",
        connectionsLink: "Kapcsolatok",
        addConnectionSuffix: ".",
      },
      mcp: {
        title: "Ügynöki hozzáférés (MCP)",
        description:
          "Engedélyezze, hogy a kódolóügynökök (például a Claude Code és a Codex) MCP-n keresztül használják az Unslothot. Az ügynökök egy erről az oldalról származó Unsloth API-kulccsal jelentkeznek be.",
        enable: "Ügynökkapcsolatok engedélyezése",
        enableDescription:
          "Kiszolgálja a /mcp/ végpontot az Unsloth API-kulcsot tartalmazó kérések számára.",
        lockedByEnv: "Beállítva ezzel: {name}.",
        loadError: "Nem sikerült betölteni az ügynöki hozzáférés beállításait.",
        saveError: "Nem sikerült menteni az ügynöki hozzáférés beállítását.",
        agent: "Ügynök",
        exportKeyHint:
          "Az ügynök indítása előtt állítsa be ezt a változót egy erről az oldalról származó Unsloth API-kulcsra: {name}.",
        configFileHint: "Adja hozzá ezt a következő fájlhoz: {path}.",
        keywords:
          "mcp model context protocol agents claude codex tools ügynökök eszközök",
      },
    },
    about: {
      title: "Névjegy",
      description:
        "Dokumentáció, kiadási megjegyzések, visszajelzés és buildinformációk.",
      studioVersion: "Unsloth verziója",
      packageVersion: "Csomag verziója",
      desktopAppVersion: "Asztali alkalmazás verziója",
      desktopAppVersionUnavailable: "Nem érhető el",
      llamaCppVersion: "llama.cpp verziója",
      audioCppVersion: "audio.cpp verziója",
      mlxVersion: "MLX verziója",
      whisperCppVersion: "whisper.cpp verziója",
      engines: {
        title: "Motorok",
        description:
          "Az Unsloth által a modellek futtatásához használt futtatókörnyezetek. A frissítések a háttérben települnek, újraindítás nélkül.",
        updatesLabel: "Motorfrissítések",
        notInstalled: "Nincs telepítve",
        available: "Elérhető: {version}",
        upToDate: "Naprakész",
        updatesAvailable: "Frissítések érhetők el",
        checking: "Frissítések keresése...",
        checkFailed:
          "Nem sikerült frissítéseket keresni. Próbálja újra később.",
        unsupported:
          "Ennek a telepítésnek a motorjait az Unslothon kívül kezelik.",
        checkForUpdates: "Frissítések keresése",
        update: "Frissítés",
        updating: "Frissítés...",
        updated: "A motorok frissítve.",
        updateFailed: "A frissítés nem sikerült: {error}",
      },
      hardware: "Hardver",
      gpu: "GPU",
      gpuIntegrated: "integrált",
      cuda: "CUDA",
      rocm: "ROCm",
      xpu: "XPU",
      updates: "Frissítés",
      help: "Súgó",
      documentation: "Dokumentáció",
      releaseNotes: "Kiadási megjegyzések",
      whatsNew: "Újdonságok",
      feedback: "Visszajelzés",
      reportIssue: "Hiba bejelentése",
      license: {
        sectionTitle: "Licenc",
        studioLabel: "Unsloth",
        studioLicense: "AGPL-3.0",
        studioDescription: "Nyílt forráskódú, GNU AGPL v3.0 licenc alatt.",
        libraryLabel: "Unsloth Core",
        libraryLicense: "Apache-2.0",
        libraryDescription: "Apache 2.0 licenc alatt.",
      },
      dangerZone: "Veszélyes zóna",
      shutDownStudio: "Unsloth leállítása",
      shutDownStudioDescription:
        "Leállítja az Unsloth-kiszolgálót, és befejezi a munkamenetet.",
      shutDown: "Leállítás",
      update: {
        title: "Unsloth frissítése",
        commandText: "Szöveg: {label}",
        copied: "Másolva",
        copyCommand: "Parancs másolása",
        commandCopied: "Másolva: {label}",
        copyNamedCommand: "Másolás: {label}",
        checkingInstall: "Az Unsloth telepítési módjának ellenőrzése...",
        installIntro: "Az Unsloth telepítése vagy frissítése:",
        localUpdateHeading: "Helyi frissítés",
        installCommandUnix: "macOS/Linux telepítési parancs",
        installCommandWindows: "Windows telepítési parancs",
        localInstallDetected:
          "Helyi telepítés észlelve. Az eredeti checkoutból frissítsen, nehogy a PyPI-verzió felülírja.",
        pullThenUpdate:
          "Töltse le a legújabb módosításokat, majd futtassa a helyi telepítőt:",
        gitPullCommand: "git pull parancs",
        localInstallerCommand: "helyi telepítő parancsa",
        sourceInstallDetected:
          "Forrásból vagy VCS-csomagból történt telepítés észlelve. Telepítse újra az eredeti helyi elérési útról vagy Git URL-címről.",
        repoCheckoutFallback:
          "Ha még megvan a tároló checkoutja, futtassa onnan a helyi telepítőt:",
        restartAfterUpdate: "A frissítés után indítsa újra az Unslothot.",
        desktopManaged:
          "Az asztali alkalmazás automatikusan keresi az új verziókat. Itt is bármikor kereshet frissítést vagy frissíthet.",
        desktopReady: "Asztali alkalmazás frissítései",
        desktopReadyDescription:
          "Ellenőrzi, hogy elérhető-e újabb asztalialkalmazás-verzió.",
        desktopChecking: "Frissítések keresése",
        desktopCheckingDescription: "Ez általában néhány másodpercig tart.",
        desktopAvailable:
          "Elérhető az asztali alkalmazás új verziója: {version}",
        desktopAvailableDescription:
          "Frissítsen most; az asztali alkalmazás a folyamat végén újraindul.",
        desktopExternalServer:
          "Az alkalmazás egy már futó Unsloth-kiszolgálóhoz csatlakozott, és azt nem tudja frissíteni. Állítsa le azt a kiszolgálót, majd a frissítéshez lépjen ki az asztali alkalmazásból, és nyissa meg újra.",
        desktopManualInstall:
          "A legújabb Linux-csomag telepítéséhez nyissa meg a kiadási oldalt.",
        desktopCheckFailed: "Nem sikerült frissítéseket keresni",
        desktopCheckFailedDescription:
          "Ellenőrizze a kapcsolatot, és próbálja újra.",
        desktopCurrent: "Az asztali alkalmazás naprakész",
        desktopCurrentDescription:
          "Az Unsloth továbbra is automatikusan keresi a frissítéseket.",
        checkForUpdates: "Frissítések keresése",
        checkAgain: "Újbóli ellenőrzés",
        retryCheck: "Újrapróbálkozás",
        checking: "Ellenőrzés...",
        updateNow: "Frissítés most",
        openReleasePage: "Kiadási oldal megnyitása",
        unknownInstall:
          "Nem sikerült megállapítani, hogyan lett telepítve az Unsloth. Telepítővel vagy PyPI-ból végzett telepítés esetén használja a fenti parancsokat.",
        localCheckout:
          "Helyi checkoutból végzett telepítés esetén futtassa a helyi telepítőt abból a checkoutból:",
        docs: "Telepítési dokumentáció:",
        docsInstall: "Telepítés",
        docsUpdating: "Frissítés",
        docsMac: "Mac",
        docsWindows: "Windows",
        docsLinux: "Linux",
        docsDocker: "Docker",
        docsAmd: "AMD",
      },
    },
  },
  studio: {
    routeTitle: "Tanítás",
    wizard: {
      modelTitle: "Modell",
      modelDescription: "Modell és tanítási módszer kiválasztása",
      datasetTitle: "Adatkészlet",
      datasetDescription: "Tanítóadatok kiválasztása vagy feltöltése",
      paramsTitle: "Paraméterek",
      paramsDescription: "Tanítási paraméterek konfigurálása",
      configTitle: "Konfiguráció",
      configDescription: "Konfigurációk mentése és betöltése",
      modelLabel: "Modell",
      modelTooltip: "A finomhangolni kívánt alapmodell.",
      methodLabel: "Módszer",
      methodTooltip:
        "A modell tanításának módja. A LoRA és a QLoRA az összes súly helyett kis adaptereket frissít.",
      trainAsLabel: "Tanítás típusa",
      trainAsTooltip:
        "A nyelvi modell szöveget ír. A döntési modell a megadott lehetőségek közül választ egyet, valószínűséggel együtt, és a Decision API-ban fut.",
      trainAsLanguage: "Nyelvi modell",
      trainAsDecision: "Döntési modell",
      checkpointLabel: "Ellenőrzőpont",
      checkpointTooltip:
        "A finomhangolandó Laya-ellenőrzőpont. A többnyelvű változat a legtöbb adatkészlethez megfelel.",
      datasetLabel: "Adatkészlet",
      datasetTooltip: "A modell finomhangolásához használt tanítóadatok.",
      hfTokenDescription:
        "Korlátozott hozzáférésű vagy privát modellekhez és adatkészletekhez szükséges.",
      uploadLocalLabel: "Helyi fájl feltöltése",
      sourceBrowse: "Tallózás",
      releaseToUpload: "Engedje el a feltöltéshez",
      loadYaml: "YAML betöltése",
      saveYaml: "YAML mentése",
      resetDefaults: "Alapértékek visszaállítása",
      cachedModelGoneTitle: "A gyorsítótárazott modell már nem érhető el",
      cachedModelGoneDescription:
        "A modell fájljai már nincsenek ezen az eszközön, ezért a tanítás újra letölti őket.",
      cachedDatasetGoneTitle:
        "A gyorsítótárazott adatkészlet már nem érhető el",
      cachedDatasetGoneDescription:
        "Az adatkészlet fájljai már nincsenek ezen az eszközön, ezért a tanítás újra letölti őket.",
    },
    preview: {
      title: "Futtatás előnézete",
      ready: "Kész",
      notReady: "Nem kész",
      modelPending: "Modell függőben",
      datasetPending: "Adatkészlet függőben",
      method: "Módszer",
      length: "Hossz",
      stepZero: "{count} lépés",
      step: "{count} lépés",
      stepTwo: "{count} lépés",
      stepFew: "{count} lépés",
      stepMany: "{count} lépés",
      steps: "{count} lépés",
      epochZero: "{count} epoch",
      epoch: "{count} epoch",
      epochTwo: "{count} epoch",
      epochFew: "{count} epoch",
      epochMany: "{count} epoch",
      epochs: "{count} epoch",
      batch: "Köteg",
      context: "Környezet",
      lr: "Tanulási ráta",
      hardware: "Hardver",
      vram: "VRAM",
      noGpu: "Nem található GPU",
      hfToken: "HF-token",
      saved: "Mentve",
      notSet: "Nincs beállítva",
      files: "Fájlok",
      model: "Modell",
      dataset: "Adatkészlet",
      downloadsOnStart: "Indításkor letöltődik",
      continuesOnStart: "Indításkor folytatódik",
      noticeModelDownload:
        "Ez a modell még nincs ezen az eszközön. A tanítás automatikusan letölti.",
      noticeModelPartial:
        "A tanítás a betöltés előtt befejezi a modell részleges letöltését.",
      noticeDatasetDownload:
        "Ez az adatkészlet még nincs ezen az eszközön. A tanítás automatikusan letölti.",
      noticeDatasetPartial:
        "A tanítás a beolvasás előtt befejezi az adatkészlet részleges letöltését.",
      noticeTransformersUpgrade:
        "Egyik telepített transformers-verzió sem támogatja még ezt az architektúrát. A futtatás indításakor felajánljuk, hogy előbb telepítse a következő transformers-verziót: {version}.",
      noticeSixteenBitOnly:
        "Ez az architektúra 16 bites LoRA-val tanul: a 4 bites mód nem érhető el hozzá, ezért a futtatás jóval több VRAM-ot igényel, mint a QLoRA.",
      noticeInstallSwitchesSixteenBit:
        "Ha a modell saját kódjának megtartása helyett ezt a kiadást telepíti, a futtatás 16 bites LoRA-ra vált, amely jóval több VRAM-ot igényel, mint a QLoRA.",
      advancedSettings: "Speciális beállítások",
      defaultAdvancedSettings: "Alapértékek",
      nonDefaultAdvancedSettings: "{count} nem alapértelmezett",
    },
    datasetPicker: {
      noun: "adatkészlet",
      selectDataset: "Adatkészlet kiválasztása",
      hubPlaceholder: "Keresés a Hugging Face adatkészletei között...",
      devicePlaceholder: "Keresés a helyi adatkészletek között...",
      useAsHubDataset: "Használat Hugging Face-adatkészletként",
      hfCacheLabel: "HF-gyorsítótár",
      scanningLocal: "Adatkészletek keresése ezen az eszközön…",
      couldntScan: "Nem sikerült átvizsgálni a helyi adatkészleteket",
      someLocationsUnscanned:
        "Néhány adatkészlet-helyet nem sikerült átvizsgálni.",
      noLocalDatasets:
        "Még nincs semmi ezen az eszközön. Töltsön le egy adatkészletet a Hubról, készítsen egyet a Receptekben, vagy töltsön fel egy fájlt.",
      openDataRecipes: "Data Recipes megnyitása",
      searchingHub: "Keresés a Hugging Face-en…",
      noDatasetsFound: "Nem található adatkészlet.",
      tokenRejectedTitle: "A Hugging Face elutasította a tokent",
      tokenRejectedBody:
        "Frissítse a tokent a Beállítások → Általános helyen, majd próbálja újra.",
      hubUnreachable: "Nem sikerült elérni a Hugging Face-t",
      cantUseDataset: "Az adatkészlet nem használható",
      reasonInvalidHubId:
        "Adjon meg érvényes Hugging Face-adatkészletazonosítót: repo vagy owner/repo, egymást követő pontok vagy kötőjelek és .git utótag nélkül (részenként legfeljebb 96 karakter).",
      sourceRecipe: "Recept",
      sourceUpload: "Feltöltés",
      sourceLocal: "Helyi",
    },
    modelPicker: {
      noun: "modell",
      selectModel: "Modell kiválasztása",
      hubPlaceholder:
        "Keressen, vagy illesszen be egy Hugging Face-azonosítót...",
      devicePlaceholder:
        "Keressen a helyi modellek között, vagy illesszen be egy mappa elérési útját...",
      useAsHubModel: "Használat Hugging Face-modellként",
      useAsLocalPath: "Használat helyi elérési útként",
      hfCacheLabel: "HF-gyorsítótár",
      scanningLocal: "Helyi modellek keresése…",
      couldntScan: "Nem sikerült átvizsgálni a helyi modelleket",
      someLocationsUnscanned: "Néhány helyi helyet nem sikerült átvizsgálni.",
      noLocalModels: "Nem található helyi modell.",
      noLocalModelsHint:
        "Illesszen be fent egy mappa elérési útját, vagy váltson a Hugging Face-re.",
      searchingHub: "Keresés a Hugging Face-en…",
      noModelsFound: "Nem található modell.",
      tokenRejectedTitle: "A Hugging Face elutasította a tokent",
      tokenRejectedBody:
        "Frissítse a tokent a Beállítások → Általános helyen, majd próbálja újra.",
      tokenRejectedAnonymousBody:
        "A nyilvános modellek token nélkül is megjelennek. A privát és korlátozott hozzáférésű modellek eléréséhez frissítse a tokent a Beállítások → Általános helyen.",
      hubUnreachable: "Nem sikerült elérni a Hugging Face-t",
      cantUseModel: "A modell nem használható tanításhoz",
      reasonTypeMismatch:
        "Ez a modell nem felel meg az előző lépésben kiválasztott tanítási típusnak.",
      reasonEmptyId:
        "Adjon meg egy modellazonosítót vagy helyi modellelérési utat.",
      reasonGguf: "GGUF-modellek nem használhatók tanításhoz.",
      reasonAdapter:
        "Az adapterkimenetek nem használhatók tanítási alapmodellként.",
      reasonNotTrainable: "Ez az eszközön lévő modell nem tanítható.",
      reasonUnsupportedFormat: "Ez a modellformátum nem támogatott tanításhoz.",
      reasonInvalidHubId:
        "Adjon meg érvényes Hugging Face-modellazonosítót: repo vagy owner/repo, egymást követő pontok vagy kötőjelek és .git utótag nélkül (részenként legfeljebb 96 karakter).",
      sourceModelsFolder: "Modellek mappa",
      sourceHfCache: "HF-gyorsítótár",
      sourceLmStudio: "LM Studio",
      sourceOllama: "Ollama",
      sourceHermes: "Hermes",
      sourceCustomFolder: "Egyéni mappa",
      sourceLocalModel: "Helyi modell",
      vramOomBadge: "OOM",
      vramTightBadge: "Szűkös",
      vramNeeds: "Kb. {est}GB VRAM szükséges (GPU: {total}GiB)",
      vramTight: "~{est}GB VRAM (szűkös, összesen: {total}GiB)",
      vramApprox: "~{est}GB VRAM",
    },
    methods: {
      qlora: {
        label: "QLoRA",
        hint: "4 bites kvantálás. A legkevesebb VRAM, a leggyorsabb indulás.",
        note: "4 bites",
      },
      lora: {
        label: "LoRA",
        hint: "16 bites adapterek. Kiegyensúlyozott minőség és memóriaigény.",
        note: "16 bites",
      },
      full: {
        label: "Teljes finomhangolás",
        hint: "Az összes súlyt tanítja. A legjobb minőség, a legtöbb VRAM-ot igényli.",
        note: "fp16",
      },
      cpt: {
        label: "Folytatott előtanítás",
        hint: "Folytatott előtanítás új szakterületekhez vagy nyelvekhez.",
        note: "folytatott",
      },
    },
    subtitles: {
      configure: "Konfigurálás és a tanítás indítása",
      trainingInProgress: "Tanítás folyamatban",
      viewPastRuns: "Korábbi tanítási futtatások megtekintése",
      viewingPastRun: "Korábbi futtatás megtekintése",
    },
    tabs: {
      configure: "Konfigurálás",
      currentRun: "Jelenlegi futtatás",
      history: "Előzmények",
    },
    goToLibrary: "Ugrás a könyvtárba",
    imageTraining: "Képtanítás",
    goToImageTraining: "Kép-LoRA-k tanítása",
    loadingRuntime: "Tanítási futtatókörnyezet betöltése...",
    checkingSupport: "A gép tanítási támogatásának ellenőrzése...",
    backToHistory: "Vissza az előzményekhez",
    dataset: {
      selectors: {
        subset: "Részhalmaz",
        subsetTooltip:
          "Válassza ki, hogy az adatkészlet melyik részhalmazát (konfigurációját) használja.",
        trainSplit: "Tanítási felosztás",
        trainSplitTooltip: "Válassza ki a tanításhoz használt felosztást.",
        evaluationSplit: "Kiértékelési felosztás",
        evaluationSplitTooltip:
          "Válassza ki a kiértékeléshez használt felosztást. „Nincs” esetén a tanítás közben nem történik kiértékelés.",
        selectSubset: "Részhalmaz kiválasztása...",
        selectSplit: "Felosztás kiválasztása...",
        none: "Nincs",
        loading:
          "Az adatkészlet konfigurációinak és felosztásainak betöltése...",
        manualTitle: "Adatkészlet-beállítások kézi megadása",
        manualDescription:
          "Adja meg a használni kívánt Hugging Face-konfiguráció és -felosztás pontos nevét.",
        manualSubsetPlaceholder: "Konfiguráció neve (nem kötelező)",
        manualRequired: "Tanítási felosztás megadása kötelező.",
        manualTooLong: "Legfeljebb 128 karaktert használjon.",
        manualInvalid: "Ez az érték nem támogatott karaktereket tartalmaz.",
      },
      sourceAriaLabel: "Adatkészlet forrása",
      localDataset: "Helyi adatkészlet",
      localDatasetRows: " / {count} sor",
      huggingFaceDataset: "Hugging Face-adatkészlet",
      localDatasetMetadata: "Helyi adatkészlet metaadatai",
      dataRecipeOutput: "Data Recipe-kimenet.",
      rows: "Sorok",
      columns: "Oszlopok",
      batches: "Kötegek",
      updated: "Frissítve",
      evalDataset: "Kiértékelési adatkészlet",
      uploading: "Feltöltés...",
      uploadEvalFile: "Kiértékelési fájl feltöltése",
      fileTooLarge: "A fájl túl nagy",
      fileTooLargeDescription:
        "Fájl: {file}, mérete: {size}. A tanítási feltöltések legfeljebb ekkora fájlt támogatnak: {limit}.",
      documentRedirect: {
        title: "Ezt a fájlt előbb át kell alakítani",
        genericFile: "Ez a fájl",
        description:
          "A következő fájl forrásanyag, nem tanításra kész adatkészlet: {file}. A Data Recipes segítségével alakítsa a dokumentumot adatkészletté, majd térjen vissza ide a finomhangoláshoz.",
        nextStepTitle: "Javasolt következő lépés",
        nextStepDescription:
          "Nyissa meg a Learning Recipes részt, és kezdjen egy dokumentumalapú recepttel, például a PDF grounded QA recepttel.",
        openAction: "Learning Recipes megnyitása",
      },
      evalDatasetDescription:
        "Nem kötelező. Ha nincs megadva, a tanítóadatok egy kis része lesz leválasztva.",
      advanced: "Speciális",
      targetFormat: "Célformátum",
      targetFormatTooltip:
        "A tanítóadatok formátuma. Az automatikus felismerés a legtöbb adatkészletnél működik.",
      streamingInfoAriaLabel: "Adatkészlet-streamelési információk",
      streaming: {
        label: "Streamelés engedélyezése",
        description:
          "A Hugging Face szöveges adatkészleteinek streamelése letöltés helyett.",
        unavailable: "A streamelés nem érhető el. Engedélyezéséhez:",
        completionsUnavailable:
          "Nem érhető el, amíg az adatkészlet-streamelés engedélyezve van.",
        blockers: {
          source:
            "Használjon Hugging Face-adatkészletet (ne helyi feltöltést vagy S3-forrást).",
          maxSteps:
            "Állítsa a Maximális lépésszámot 0-nál nagyobbra – a streamelt adatkészletek hossza nem ismert.",
          trainOnCompletions:
            "Kapcsolja ki a „Csak asszisztensi válaszok” beállítást.",
          evalSplit:
            "Válasszon külön kiértékelési felosztást – a kiértékelés be van kapcsolva, de nincs megadva külön kiértékelési felosztás.",
          visionModel: "A látásmodellek nem támogatják a streamelést.",
          audioModel: "A hangmodellek nem támogatják a streamelést.",
          embeddingModel:
            "A beágyazási modellek nem támogatják a streamelést (a tanításhoz a teljes adatkészlet szükséges).",
          imageDataset:
            "Ez az adatkészlet képeket tartalmaz, ezért nem streamelhető.",
          audioDataset:
            "Ez az adatkészlet hangot tartalmaz, ezért nem streamelhető.",
          appleSilicon:
            "A streamelés Apple Siliconon (MLX) még nem támogatott.",
        },
        options: {
          trainOnCompletions: "csak asszisztensi válaszok",
          evaluation: "kiértékelés (külön kiértékelési felosztás szükséges)",
        },
        notifications: {
          turnedOffMaxSteps:
            "Streamelés kikapcsolva: a streameléshez 0-nál nagyobb rögzített Maximális lépésszám szükséges.",
          adjusted:
            "A beállítások a streameléshez igazítva. Letiltott, nem kompatibilis beállítások: {options}.",
          needsMaxSteps:
            "A streameléshez rögzített Maximális lépésszám szükséges (a streamelt adatkészletek hossza nem ismert). Előbb állítsa a Maximális lépésszámot 0-nál nagyobbra.",
          enabledAdjusted:
            "Streamelés engedélyezve. Letiltott, nem kompatibilis beállítások: {options}.",
          disabledForDetectedModality:
            "A streamelés kikapcsolva, mert a kép- és hangadatkészletekhez teljes letöltés szükséges. Ellenőrizze a beállítást, majd indítsa újra.",
        },
      },
      auto: "Automatikus",
      rawText: "Nyers szöveg",
      trainSplitStart: "Tanítási felosztás kezdete",
      trainSplitStartTooltip:
        "Csak a tanítási felosztás egy részén tanít, a megadott kezdő sorindextől (a sort is beleértve, 0-tól számozva). Hagyja üresen, ha az első sortól kíván kezdeni.",
      trainSplitEnd: "Tanítási felosztás vége",
      trainSplitEndTooltip:
        "A tanítási felosztásból még felhasznált utolsó sor indexe (a sort is beleértve, 0-tól számozva). Például ha a Kezdetet 0-ra, a Véget 99-re állítja, a tanítás az első 100 soron fut. Hagyja üresen az összes hátralévő sor használatához.",
      endPlaceholder: "Vége",
      clear: "Törlés",
      dropFileOrClick: "Húzzon ide egy fájlt, vagy kattintson a feltöltéshez",
      uploadDetails: "Feltöltés részletei",
      uploadDetailsTooltip:
        "Fájlonként legfeljebb {limit}. A PDF-, DOCX- és TXT-fájlok nem tanításra kész adatkészletek, ezért előbb alakítsa át őket a Learning Recipes részben.",
      viewDataset: "Adatkészlet megtekintése",
      previewTokens: "Tokenek előnézete",
      tokenPreview: {
        title: "Token-előnézet",
        description:
          "Így tokenizálja ezeket az adatokat a tanításhoz a következő modell: {model}. Csak a tokenizáló töltődik le.",
        datasetTab: "Adatkészlet-minták",
        textTab: "Saját szöveg",
        trained: "Tanított",
        masked: "Maszkolt",
        completionsOnly: "Csak asszisztensi válaszok",
        allTokens: "Minden token tanítva",
        responseMarker: "Válaszjelölő",
        maskingNotApplied:
          "A csevegésjelölők nem találhatók, ezért minden token tanítva lesz.",
        loading:
          "A tokenizáló betöltése. Az első előnézet kicsit tovább tart...",
        shuffle: "Minták keverése",
        showIds: "Tokenazonosítók megjelenítése",
        copyIds: "Azonosítók másolása",
        copied: "Tokenazonosítók kimásolva",
        sample: "Minta {index}",
        stats: "{tokens} token, ebből tanítva: {trained} ({percent}%)",
        truncated: "A környezethossz miatt {full} tokenből {max} maradt meg.",
        noTrained:
          "Ebben a sorban egyetlen token sincs tanítva, ezért a tanítás kihagyja.",
        hiddenTokens: "További {count} token nincs megjelenítve",
        multimodal:
          "A kép- és hangsorok itt nem jelennek meg. A tokenizálót a saját szövegén továbbra is kipróbálhatja.",
        textPlaceholder:
          "Írjon be vagy illesszen be szöveget a tokenjei megtekintéséhez...",
        textHint:
          "A „Csak asszisztensi válaszok” beállítással csak az asszisztensjelölő utáni szöveg lesz tanítva.",
        insertExample: "Csevegési példa beszúrása",
        vocab: "Szókincs: {count} token",
        retry: "Újrapróbálkozás",
        packing:
          "A csomagolás be van kapcsolva: a futtatás a sorokat környezethosszúságú szekvenciákká fűzi össze, így a sorok vágási pontja eltérhet ettől a soronkénti nézettől.",
      },
      uploadFailed: "A feltöltés nem sikerült",
      unknownError: "Ismeretlen hiba",
      unsupportedFileType: "Nem támogatott fájltípus",
      uploadOneFileType: "Töltsön fel egy fájlt (típus: {types}).",
      datasetUploaded: "Adatkészlet feltöltve",
      evalDatasetUploaded: "Kiértékelési adatkészlet feltöltve",
      uploadOneFileAtATime: "Egyszerre csak egy fájlt töltsön fel",
      uploadSingleFileDescription:
        "A tanítási adatkészlet feltöltése egyetlen fájlt fogad el.",
      previewLoadingHuggingFace:
        "Az adatkészlet előnézetének lekérése a Hugging Face-ről...",
      previewLoading: "Előnézet betöltése...",
      mappingRequirements: {
        audioAndText: "hang és szöveg",
        imageAndText: "kép és szöveg",
        instructionAndOutput: "utasítás és kimenet",
        humanAndGpt: "ember és GPT",
        userAndAssistant: "felhasználó és asszisztens",
      },
      mappingStatus: {
        heuristicTitle: "Heurisztikusan felismert hozzárendelés",
        readyTitle: "A hozzárendelés kész",
        requiredTitle: "Adatkészlet oszlopainak hozzárendelése",
        heuristicDescription:
          "Az alábbi oszlop-hozzárendelést heurisztikák alapján automatikusan felismertük. Ellenőrizze, és szükség esetén módosítsa az oszlopfejlécek legördülő listáival, vagy használja az AI Assist funkciót egy pontosabb hozzárendeléshez.",
        readyDescription:
          "Minden rendben. Az adatkészletet automatikusan átalakítjuk.",
        requiredDescription:
          "Rendeljen szerepköröket az oszlopokhoz a fejlécek legördülő listáival. Legalább ezeket rendelje hozzá: {required}.",
      },
      s3: {
        title: "S3-konfiguráció",
        description:
          ".parquet, .json, .jsonl vagy .csv adatkészletek betöltése az Amazon S3-ból",
        bucket: "Gyűjtő (bucket) neve",
        bucketPlaceholder: "my-training-data-bucket",
        region: "AWS-régió",
        regionPlaceholder: "us-east-1",
        prefix: "Elérési út előtagja",
        prefixPlaceholder: "datasets/whisper/",
        accessKeyId: "Hozzáférési kulcs azonosítója",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Titkos hozzáférési kulcs",
        secretAccessKeyPlaceholder: "Az AWS titkos hozzáférési kulcsa",
        useIamRole: "IAM-szerepkör használata",
      },
    },
    params: {
      mode: {
        simple: "Egyszerű",
        advanced: "Speciális",
        ariaLabel: "Paraméter mód",
      },
      projectName: "Projekt neve",
      optional: "Nem kötelező",
      projectNameDescription:
        "A tanítási kimeneti mappák neveiben, az exportálási alapértékekben és az előzményekben jelenik meg.",
      loraSettings: "LoRA-beállítások",
      trainingHyperparameters: "Tanítási hiperparaméterek",
      drafterSettings: "Spekulatív dekódolási vázlatmodell",
      drafterTrain: "Spekulatív dekódolási vázlatmodell tanítása",
      drafterTrainTooltip:
        "A finomhangolás után egy kis vázlatfejet tanít, így a következtetés gyorsabban fut a llama.cpp-ben és a vLLM-ben; a kimenetek nem változnak.",
      drafterChecking: "A modell ellenőrzése...",
      drafterUnavailable: "Ehhez a modellhez nem érhető el: {reason}",
      drafterRetrainMtp: "Újratanítja a modell MTP-fejét",
      drafterEagle3:
        "EAGLE-3 vázlatmodellt tanít (ennek a modellnek nincs MTP-feje)",
      drafterDataSource: "Tanítóadatok",
      drafterDataSourceDataset: "Az adatkészlet válaszai",
      drafterDataSourceRegenerate: "A modell saját válaszai (lassabb, jobb)",
      drafterMaxExamples: "Példák",
      drafterEpochs: "Epochok",
      drafterLearningRate: "Tanulási ráta",
      drafterLearningRateDefault: "Alapértelmezett",
      drafterDraftSteps: "Vázlattokenek",
      maxSteps: "Maximális lépésszám",
      epochs: "Epochok",
      useMaxSteps: "Maximális lépésszám használata",
      useEpochs: "Epochok használata",
      maxStepsTooltip:
        "Az optimalizálási lépések teljes számának felülbírálása.",
      epochsTooltip: "Az adatkészlet teljes bejárásainak száma.",
      contextLength: "Környezethossz",
      contextLengthTooltip: "A tanítási mintánkénti tokenek maximális száma.",
      customContextLength: "Egyéni érték megadása",
      learningRate: "Tanulási ráta",
      learningRateTooltip:
        "A súlyfrissítések lépésmérete. Kisebb értékkel a tanítás lassabb, de stabilabb.",
      learningRateDescription:
        "Ajánlott: LoRA esetén 2e-4, CPT esetén 5e-5, teljes finomhangolásnál 2e-5",
      embeddingLearningRate: "Beágyazási tanulási ráta",
      embeddingLearningRateTooltip:
        "Csak akkor használatos, ha a CPT az embed_tokens réteget tanítja. A beágyazások könnyebben destabilizálódnak, mint a LoRA-súlyok, ezért általában kisebb tanulási rátát igényelnek. Hagyja üresen az lr/10 használatához; a szokásos működő tartomány 2–10-szer kisebb a fő tanulási rátánál. Csak akkor növelje, ha a szókincs vagy a szakterületi tokenek adaptációja túl lassú.",
      rank: "Rang",
      rankTooltip:
        "Az alacsony rangú mátrixok dimenziója. Nagyobb érték = nagyobb kapacitás.",
      alpha: "Alfa",
      alphaTooltip:
        "A LoRA-frissítések skálázási tényezője. Általában a rang kétszerese.",
      dropout: "Dropout",
      dropoutTooltip:
        "A LoRA-rétegek dropout-valószínűsége a túlilleszkedés csökkentésére.",
      visionLayers: "Látásrétegek",
      languageLayers: "Nyelvi rétegek",
      attentionModules: "Figyelemmodulok",
      mlpModules: "MLP-modulok",
      targetModules: "Célmodulok",
      enableLora: "LoRA",
      trainWithLora: "Tanítás LoRA-val",
      stableRank: "Rangstabilizált LoRA",
      memoryEfficient: "SVD-vel inicializált LoRA",
      weightDecomposed: "LoRA szorzókkal",
      notSupportedAppleSilicon: "Apple Siliconon nem támogatott",
      doraNeedsVisionLayersOff:
        "A DoRA használatához kapcsolja ki a látásrétegek tanítását",
      optimization: "Optimalizálás",
      schedule: "Ütemezés",
      memory: "Memória",
      optimizer: "Optimalizáló",
      optimizerTooltip:
        "Optimalizálási algoritmus. A 8 bites változatok csökkentik a memóriahasználatot. Látásmodellekhez a Fused ajánlott.",
      optimizerTooltipMlx:
        "Optimalizálási algoritmus. Az alapértelmezett az AdamW. A Lion kevesebb memóriát használ, de általában kisebb tanulási rátát igényel.",
      lrScheduler: "Tanulásiráta-ütemező",
      lrSchedulerTooltip:
        "Hogyan változik a tanulási ráta a tanítás során. A lineáris egyenletesen csökken, a koszinusz görbe mentén.",
      optimizerOptions: {
        adamw8bit: "AdamW 8 bites",
        pagedAdamw8bit: "Paged AdamW 8 bites",
        adamwBnb8bit: "AdamW BNB 8 bites",
        pagedAdamw32bit: "Paged AdamW 32 bites",
        adamwTorch: "AdamW (PyTorch)",
        adamwTorchFused: "AdamW (PyTorch Fused)",
      },
      lrSchedulerOptions: {
        linear: "Lineáris",
        cosine: "Koszinusz",
      },
      batchSize: "Kötegméret",
      batchSizeTooltip:
        "Lépésenként feldolgozott minták száma. A nagyobb érték több VRAM-ot használ.",
      gradAccum: "Gradiensakkumulációs lépések",
      gradAccumTooltip: "Nagyobb kötegméretet szimulál további VRAM nélkül.",
      weightDecay: "Súlycsökkentés",
      weightDecayTooltip: "L2-regularizáció a túlilleszkedés megelőzésére.",
      warmupSteps: "Bemelegítő lépések",
      warmupStepsTooltip:
        "A stabilitás érdekében a tanítás elején fokozatosan növeli a tanulási rátát.",
      scheduleEpochsTooltip:
        "Az adatkészlet teljes bejárásainak száma. Állítsa 0-ra, ha a maximális lépésszám szerint szeretne futtatni.",
      saveSteps: "Mentési lépések",
      saveStepsTooltip: "Ellenőrzőpont mentése N lépésenként. 0 = letiltva.",
      evalSteps: "Kiértékelési lépések",
      evalStepsTooltip:
        "A kiértékelések közötti távolság az összes tanítási lépés arányában (0–1). A kiértékelés letiltásához állítsa 0-ra. Például 0,01 = kiértékelés a lépések minden 1%-ánál.",
      earlyStoppingPatience: "Korai leállítás türelme",
      earlyStoppingPatienceTooltip:
        "Leállítja a tanítást, ha ennyi egymást követő kiértékelés során nem csökken a kiértékelési veszteség, és a legjobb kiértékelés súlyait tartja meg. Kiértékelést igényel (0-nál nagyobb Kiértékelési lépések). 0 = letiltva.",
      seed: "Kezdőérték (seed)",
      seedTooltip: "Véletlenszám-kezdőérték a reprodukálhatósághoz.",
      offloadLayers: "Rétegek áthelyezése",
      offloadLayersTooltip:
        "A dekóderrétegeket a rendszer-RAM-ban tartja, és mindegyiket közvetlenül a futása előtt streameli a GPU-ra, így a VRAM-nál nagyobb modell is tanítható LoRA-val. Automatikus módban a lehető legkevesebb réteg kerül áthelyezésre; megadott szám esetén pontosan annyi. Gradiens-ellenőrzőpontozást igényel.",
      offloadOff: "Ki",
      offloadAuto: "Automatikus",
      offloadCount: "Darabszám",
      offloadVramBudget: "VRAM-keret (GiB)",
      offloadVramBudgetTooltip:
        "Az a legnagyobb VRAM-mennyiség, amelyet ez a futtatás használhat. Automatikus módban annyi réteg kerül áthelyezésre, hogy beleférjen, így két futtatás is osztozhat egy kártyán, mindkettő egy-egy részét használva. Üresen hagyva a teljes kártyát használja.",
      offloadWholeCard: "teljes kártya",
      offloadVramBudgetGpu: "GPU {index} VRAM-kerete (GiB)",
      prefetchDepth: "Előbetöltési mélység",
      prefetchDepthTooltip:
        "Hány áthelyezett réteg másolódik előre az éppen futó előtt. Automatikus módban 1-ről indul, és csak addig tart fenn mélyebb készletet, amíg az mérhetően csökkenti a másolásokra való várakozást.",
      offloadPanelTitle: "Áthelyezett rétegek",
      offloadPanelGpu: "GPU-n",
      offloadPanelHost: "Rendszer-RAM-ban",
      offloadPanelCopying: "Másolás",
      offloadPanelSwapped: "áthelyezve",
      offloadPanelStall: "várakozás a másolásokra",
      offloadPanelCopy: "másolás rétegenként",
      offloadPanelCompute: "számítás rétegenként",
      offloadPanelVram: "VRAM",
      offloadPanelDepth: "előbetöltési mélység",
      offloadPanelPinned: "rögzített",
      offloadPanelSweepNote:
        "A mozgó ablak a rétegek lekérésének sorrendjét mutatja, a láthatóság kedvéért lelassítva; a számok az utolsó lépésben mért értékek.",
      offloadPanelCard: "GPU {index}",
      offloadPanelCardUnknown: "Ismeretlen kártya",
      gradCheckpoint: "Gradiens-ellenőrzőpontozás",
      gradCheckpointTooltip:
        "Az aktivációk újraszámításával számítási időért cserébe memóriát takarít meg.",
      none: "Nincs",
      standard: "Normál",
      enablePacking: "Csomagolás engedélyezése",
      assistantCompletionsOnly: "Csak asszisztensi válaszok",
      readMore: "További információ",
    },
    training: {
      duplicateFailed: "Nem sikerült duplikálni a futtatást",
      duplicateDraftChanged:
        "A tanítási beállítások megváltoztak a betöltés közben. Próbálja újra a duplikálást.",
      duplicateNoModel:
        "Ennek a futtatásnak nincs mentett modellkonfigurációja.",
      startTraining: "Tanítás indítása",
      starting: "Indítás...",
      loadingModel: "Modell betöltése...",
      checkingDataset: "Adatkészlet ellenőrzése...",
      chooseModel: "Válasszon modellt",
      chooseDataset: "Válasszon adatkészletet",
      chooseModelAndDataset: "Válasszon modellt és adatkészletet",
      validation: {
        s3MultimodalUnsupported:
          "Az S3-adatkészletek látástanításhoz még nem támogatottak.",
        s3BucketRequired: "Előbb adja meg az S3-gyűjtő nevét.",
        s3CredentialsRequired:
          "Adja meg az S3 hozzáférési kulcsait, vagy engedélyezze az IAM-szerepkört.",
        modelRequired: "Előbb válasszon alapmodellt.",
        learningRatePositive: "Adjon meg nullánál nagyobb tanulási rátát.",
        embeddingLearningRateRange:
          "Adjon meg 0-nál nagyobb és 1-nél kisebb beágyazási tanulási rátát.",
        decisionColumnsMissing:
          "A döntési modellekhez state, questions és gold (vagy answers) oszlopok szükségesek. Hiányzik: {columns}.",
        decisionOwnerOnly:
          "Döntési modelleket csak az Unsloth tulajdonosa finomhangolhat.",
        hfDatasetRequired: "Előbb válasszon Hugging Face-adatkészletet.",
        hfDatasetSplitRequired:
          "Előbb válasszon ki vagy adjon meg egy tanítási felosztást.",
        localDatasetRequired: "Előbb válasszon helyi adatkészletet.",
        unsupportedDatasetSource: "Nem támogatott adatkészletforrás.",
      },
      startFailed: "Nem sikerült elindítani a tanítást",
      startUnconfirmed:
        "Az Unsloth nem tudta megerősíteni, hogy elindult-e a tanítás. Az állapot ellenőrzése a háttérben folyik.",
      stopFailed: "Nem sikerült leállítani a tanítást",
      trainingStillActiveTitle: "A tanítás még fut",
      stopBeforeConfig:
        "Előbb állítsa le a tanítást, majd térjen vissza a konfigurációhoz.",
      resumeFailed: "Nem sikerült folytatni a tanítást",
      resumeFailedTitle: "A tanítás nem folytatható",
      resumeUnavailable:
        "Csak a mentett ellenőrzőponttal rendelkező, leállított vagy hibával leállt futtatások folytathatók.",
      modelUnverified:
        "Nem sikerült ellenőrizni a modell beállításait. Ellenőrizze a kapcsolatot vagy a Hugging Face-tokent, majd próbálja újra.",
      legacyDatasetScriptUnsupported:
        "Ez a Hub-adatkészlet egy régi, egyéni szkriptre épül, és ebben a tanítási folyamatban nem támogatott.",
      hfModelAccessDenied:
        "A Hugging Face megtagadta a hozzáférést ehhez a modellhez. Adjon meg egy érvényes, a tárolóhoz hozzáféréssel rendelkező Hugging Face-tokent, fogadja el a szükséges hozzáférési feltételeket, majd próbálja újra.",
      hfModelVerificationRateLimited:
        "A Hugging Face-modell ellenőrzése sebességkorlátozás alatt áll. Próbálja újra hamarosan.",
      hfModelVerificationFailed:
        "Nem sikerült ellenőrizni a Hugging Face-modellt. Ellenőrizze a tároló azonosítóját és a hozzáférési tokent.",
      hfModelMetadataUnavailable:
        "A Hugging Face-modell metaadatai átmenetileg nem érhetők el. A tanítás indítása előtt próbálja újra.",
      datasetUnverified:
        "Nem sikerült ellenőrizni, hogy az adatkészlet kompatibilis-e ezzel a modellel. Ellenőrizze a kapcsolatot vagy a Hugging Face-tokent – a tanítás indításakor az ellenőrzés újra lefut.",
      setupChanged:
        "A tanítási beállítások megváltoztak az ellenőrzés közben. Tekintse át őket, és indítsa újra.",
      uploadConfigTooltip: "Mentett YAML-konfiguráció betöltése",
      saveConfigTooltip: "A jelenlegi konfiguráció letöltése YAML formátumban",
      resetConfigTooltip: "A modell alapértékeinek visszaállítása",
      configLoaded: "Konfiguráció betöltve",
      failedToLoadConfig: "Nem sikerült betölteni a konfigurációt",
      invalidYamlFile: "Érvénytelen YAML-fájl",
      configTooLarge: "A tanítási konfiguráció túl nagy (legfeljebb 1 MiB).",
      failedToReadFile: "Nem sikerült beolvasni a fájlt",
      failedToSaveConfig: "Nem sikerült menteni a konfigurációt",
      parametersReset: "A paraméterek visszaállítva a modell alapértékeire",
      audioIncompatible:
        "Ez a modell nem támogatja a hangot. Váltson hangot támogató modellre, vagy válasszon hang nélküli adatkészletet.",
      visionIncompatible:
        "A szöveges modell nem kompatibilis a multimodális adatkészlettel. Váltson látásmodellre, vagy válasszon csak szöveget tartalmazó adatkészletet.",
      cancelTitle: "Tanítás megszakítása",
      cancelDescription: "Megszakítja a jelenlegi tanítási futtatást?",
      continueAction: "Tanítás folytatása",
      cancelAction: "Tanítás megszakítása",
      stopTitle: "Tanítás leállítása",
      stopDescription:
        "Válassza ki, hogyan szeretné leállítani a jelenlegi tanítási futtatást. A Leállítás és mentés ellenőrzőpontot ír, amelyből később folytathatja; a mentés nélküli leállítás után a futtatás nem folytatható.",
      stopAction: "Leállítás",
      stopping: "Leállítás...",
      stopAndSave: "Leállítás és mentés",
      compareInChat: "Összehasonlítás a Csevegésben",
      exportModel: "Modell exportálása",
      milestone: "Mérföldkő",
      halfwayDone: "Félúton. A tanítás túljutott az 50%-on.",
      doneNextStep:
        "A tanítás kész. Következő lépés: az alapmodell és a finomhangolt modell kimeneteinek összehasonlítása.",
    },
    history: {
      title: "Előzmények",
      filesDeleted: "Fájlok törölve",
      deleteArtifactsLabel: "Az adapterfájlok törlése a lemezről is",
      deleteArtifactsDescription:
        "Eltávolítja a futtatás kimeneti mappáját, a mentett adapterekkel és ellenőrzőpontokkal együtt.",
      deleteArtifactsSharedNote:
        "Egy másik futtatás is ezt a kimeneti mappát használja. A fájlok megmaradnak, amíg az őket használó utolsó futtatás törlésre nem kerül.",
      artifactsKeptShared:
        "A futtatás törölve. Az adapterfájlok megmaradtak, mert egy másik futtatás is ugyanazt a mappát használja.",
      deleteArtifactsActiveError:
        "Ezeket a fájlokat a folyamatban lévő tanítási futtatás használja. A törlésük előtt állítsa le a tanítást.",
      deleteArtifactsFailed:
        "A futtatás törölve, de a fájljait nem sikerült eltávolítani.",
      deleteArtifactsRetainedError:
        "Az adapterfájlokat nem sikerült eltávolítani, ezért a tanítási futtatás megmaradt az előzményekben.",
      emptyDescription:
        "Még nincs tanítási futtatás. Indítsa el az első tanítási futtatást a Konfigurálás lapon.",
      loadError: "Nem sikerült betölteni a tanítási futtatásokat",
      deleteError: "Nem sikerült törölni a tanítási futtatást. Próbálja újra.",
      retry: "Újrapróbálkozás",
      loadMore: "Továbbiak betöltése",
      loading: "Betöltés...",
      loadingRun: "Tanítási futtatás betöltése...",
      runNotFound: "A futtatás nem található",
      deleteTitle: "Törli a tanítási futtatást?",
      deleteDescription:
        "Ez véglegesen törli ezt a tanítási futtatást és az összes metrikáját. Ez a művelet nem vonható vissza.",
      resumeTraining: "Tanítás folytatása",
      resuming: "Folytatás...",
      deleteRun: "Futtatás törlése",
      loss: "Veszteség",
      steps: "Lépések",
      lossTrendSparkline: "Veszteségtrend-minidiagram",
      relativeJustNow: "az imént",
      status: {
        completed: "Befejeződött",
        stopped: "Leállítva",
        error: "Hiba",
        running: "Fut",
        continued: "Folytatva",
      },
      message: {
        completed: "A tanítás befejeződött",
        stopped: "A tanítás leállítva",
        running: "Tanítás folyamatban",
        errored: "A tanítás hibával leállt",
      },
      copyPreviewLink: "Előnézeti hivatkozás másolása",
      previewLinkCopied: "Az előnézeti hivatkozás kimásolva",
      previewLinkCopyFailed: "Nem sikerült másolni a hivatkozást",
    },
    charts: {
      settings: "Diagrambeállítások",
      settingsDescription:
        "A diagramok megjelenítésének finomítása a tanítás futása közben.",
      openSettings: "Diagrambeállítások megnyitása",
      viewWindow: "Nézetablak",
      viewWindowDescription:
        "Csak a legutóbbi lépések vagy a teljes előzmény megjelenítése.",
      window: "Ablak",
      all: "Összes",
      trainingLoss: "Tanítási veszteség",
      trainingLossDescription: "Rávetített görbék és EMA-simítás szabályozása.",
      smoothing: "Simítás",
      smoothingDescription: "Jobbra húzva erősebb a simítás. `0` = nyers.",
      showRawLoss: "Nyers veszteség megjelenítése",
      showSmoothedLoss: "Simított veszteség megjelenítése",
      showAverageLine: "Átlagvonal megjelenítése",
      scaleAndCleanup: "Skála és tisztítás",
      linear: "Lineáris",
      log: "Logaritmikus",
      noClip: "Nincs vágás",
      clipP99: "Vágás p99-nél",
      clipP95: "Vágás p95-nél",
      lossAxis: "Veszteség tengely",
      gradientNormAxis: "Gradiensnorma tengely",
      learningRateAxis: "Tanulási ráta tengely",
      resetDefaults: "Alapértékek visszaállítása",
      loss: "Veszteség",
      smoothed: "Simított",
      evalLoss: "Kiértékelési veszteség",
      learningRate: "Tanulási ráta",
      lr: "LR",
      gradNorm: "Gradiensnorma",
      gradientNorm: "Gradiensnorma",
      step: "Lépés: {step}",
      averageValue: "átl. {value}",
      waitingForFirstEvaluationStep:
        "Várakozás az első kiértékelési lépésre...",
      evaluationNotConfigured: "A kiértékelés nincs beállítva",
      evalChartWillAppear: "A diagram az eval_steps elérésekor jelenik meg",
      setEvalDatasetAndSteps:
        "A kiértékelési veszteség követéséhez adja meg a kiértékelési adatkészletet és az eval_steps értéket",
    },
    progress: {
      title: "Tanítás előrehaladása",
      liveMetrics: "Élő tanítási metrikák",
      exportGguf: "Exportálás GGUF-ba",
      useInDecisionApi: "Használat a Decision API-ban",
      decisionApiEnabled:
        "A Decision API mostantól ezt a modellt szolgálja ki: {name}.",
      decisionApiFailed:
        "Nem sikerült a Decision API-t erre a modellre váltani.",
      openConfig: "Tanítási konfiguráció megnyitása",
      configLabel: "Tanítási konfiguráció",
      hyperparams: "Hiperparaméterek",
      epochs: "Epochok",
      batchSize: "Kötegméret",
      learningRate: "Tanulási ráta",
      optimizer: "Optimalizáló",
      maxSteps: "Lépések max. száma",
      contextLength: "Környezethossz",
      warmupSteps: "Bemelegítő lépések",
      rank: "Rang",
      alpha: "Alfa",
      dropout: "Dropout",
      variant: "Változat",
      epoch: "Epoch: {value}",
      percentComplete: "{percent}% kész",
      stepProgress: "Lépés: {current} / {total}",
      loss: "Veszteség",
      lr: "LR",
      gradNorm: "Gradiensnorma",
      project: "Projekt",
      model: "Modell",
      method: "Módszer",
      elapsed: "Eltelt idő: {value}",
      eta: "Becsült idő: {value}",
      stepsPerSecond: "{value} lépés/s",
      noStepsPerSecond: "-- lépés/s",
      tokens: "Tokenek: {value}",
      gpuMonitor: "GPU-figyelő",
      live: "Élő",
      utilization: "Kihasználtság",
      temperature: "Hőmérséklet",
      vram: "VRAM",
      power: "Teljesítmény",
      phase: {
        idle: "Tétlen",
        downloadingModel: "Modell letöltése",
        downloadingDataset: "Adatkészlet letöltése",
        loadingModel: "Modell betöltése",
        loadingDataset: "Adatkészlet betöltése",
        configuring: "Konfigurálás",
        training: "Tanítás",
        finalizing: "Modell mentése",
        trainingDrafter: "Vázlatmodell tanítása",
        completed: "Befejezve",
        error: "Hiba",
        stopped: "Leállítva",
      },
    },
    trainingStart: {
      ready: "Kész",
      downloading: "Letöltés",
      preparing: "Előkészítés",
      left: "Hátralévő idő: {eta}",
      downloaded: "Letöltve: {size}",
      terminalStart: "> az unsloth tanítás indul...",
      preparingResources: "> Modell és adatkészlet előkészítése...",
      gettingReady: "> Mindent előkészítünk a futtatáshoz...",
      resumingTraining: "Tanítás folytatása...",
      startingTraining: "tanítás indítása...",
      dataset: "Adatkészlet",
      datasetStreaming: "Adatkészlet: streamelés (nincs teljes letöltés)",
      modelWeights: "Modellsúlyok",
    },
  },
  modelMemory: {
    readout:
      "Súlyok: {model} + környezet: {context} = {total} / {budget} használható VRAM",
    readoutWithSpec:
      "Súlyok: {model} + KV: {kv} + MTP-vázlat: {spec} = {total} / {budget} használható VRAM",
    kvRate: "Lefoglalt KV, kb. {rate}/token",
    oomLikely: "A jelenlegi beállításokkal valószínű a memóriahiány (OOM)",
    tooLarge:
      "Nagyobb a VRAM-nál, ezért részben a CPU-ra lesz áthelyezve. Egy kisebb kvantált változat gyorsabban fut",
  },
  skills: {
    title: "Készségek",
    description:
      "A készségek a szokásos ügynökmappákból kerülnek felderítésre. Itt engedélyezheti őket, majd a csevegésben a @ beírásával hivatkozhat rájuk.",
    precedence:
      "A ~/.agents/skills elsőbbséget élvez a ~/.claude/skills mappával szemben.",
    refresh: "Frissítés",
    empty:
      "Nem található készség. Hozzon létre egy SKILL.md fájlt tartalmazó mappát a ~/.agents/skills vagy a ~/.claude/skills alatt, majd frissítsen.",
    sourceAgents: "Ügynökök",
    sourceClaude: "Claude",
    sourceBundled: "Beépített",
    shadowed: "Felülírva",
    invalid: "Érvénytelen",
    compatibility: "Kompatibilitás: {value}",
    shadowedBy:
      "Egy azonos nevű másik készség elsőbbséget élvez (forrás: {source}).",
    enable: "Engedélyezés: {name}",
    disable: "Letiltás: {name}",
    updateError: "Nem sikerült frissíteni a készséget",
    newSkill: "Új készség",
    import: "Importálás",
    delete: "Törlés: {name}",
    createDescription:
      "SKILL.md fájlként mentődik egy új mappába a ~/.agents/skills alatt, és létrejötte után azonnal engedélyezve lesz.",
    nameLabel: "Név",
    namePlaceholder: "pl. code-arithmetic",
    nameHint:
      "Kisbetűk, számok és egyszeres kötőjelek. Ez egyben a mappa neve is.",
    nameInvalid:
      "1–64 kisbetűt, számot vagy egyszeres kötőjelet használjon; nem kezdődhet és nem végződhet kötőjellel.",
    descriptionLabel: "Leírás",
    descriptionPlaceholder:
      "Akkor használja, ha a felhasználó számolást kér. Mindig a kódeszközzel számolja ki.",
    descriptionHint:
      "Ezt a sort látja a modell a készségkatalógusában. Ez alapján dönti el, mikor olvassa be a készséget.",
    instructionsLabel: "Utasítások",
    instructionsPlaceholder:
      "Mi a teendő, ha ez a készség alkalmazható, Markdown formátumban.",
    instructionsHint:
      "A SKILL.md törzse, amelyet a modell teljes egészében beolvas, amint kiválasztja a készséget.",
    create: "Készség létrehozása",
    save: "Módosítások mentése",
    created: "Létrehozva: {name}",
    saved: "Mentve: {name}",
    deleted: "Törölve: {name}",
    importedOne: "{count} készség hozzáadva",
    importedMany: "{count} készség hozzáadva",
    importPartial: "{added} / {total} készség hozzáadva",
    saveError: "Nem sikerült menteni a készséget",
    openError: "Nem sikerült megnyitni a készséget",
    deleteError: "Nem sikerült törölni a készséget",
    importError: "Nem sikerült importálni a készséget",
    exportError: "Nem sikerült exportálni a készséget",
    deleteTitle: "Készség törlése",
    deleteDescription:
      "Törli a(z) „{name}” készséget és a mappájának teljes tartalmát a ~/.agents/skills alatt? Ez a művelet nem vonható vissza.",
    search: "Készségek keresése név vagy leírás alapján…",
    noMatch: "Nincs a keresésnek megfelelő készség: „{query}”",
    clearSearch: "Keresés törlése",
    readOnlyClaude:
      "Csak olvasható: a ~/.claude/skills mappában található. A fájlt ott szerkessze.",
    readOnlyBundled: "Csak olvasható: az Unsloth része.",
    readOnlyLinked:
      "Csak olvasható: ez a bejegyzés egy hivatkozás. A fájlokat az eredeti helyükön szerkessze.",
    characters: "{count} karakter",
    unsaved: "nem mentett módosítások",
    revert: "Visszavonás",
    enabledLabel: "Engedélyezve",
    licenseLabel: "Licenc",
    compatibilityLabel: "Kompatibilitás",
    allowedToolsLabel: "Engedélyezett eszközök",
    sectionAgents: "Saját készségek",
    sectionAgentsHint:
      "A ~/.agents/skills mappában tárolva. Itt hozhatja létre, szerkesztheti és törölheti őket.",
    sectionClaude: "Claude-készségek",
    sectionClaudeHint:
      "A ~/.claude/skills mappában találhatók. Itt csak olvashatók; a fájlokat ott szerkessze.",
    sectionBundled: "Beépített",
    sectionBundledHint: "Az Unsloth része. Csak olvasható.",
    linked: "Hivatkozás",
    discardTitle: "Elveti a módosításokat?",
    discardDescription:
      "A következő elem nem mentett módosításai elvesznek: {name}.",
    discard: "Elvetés",
    mentions: "Készségek",
    manage: "Készségek kezelése",
    turnOff: "Készségek kikapcsolása",
    bulkActions: "Készségműveletek",
    enableAll: "Összes engedélyezése",
    disableAll: "Összes letiltása",
    resetAll: "Alapértelmezések visszaállítása",
    resetTitle: "Visszaállítja az összes készséget?",
    resetDescription:
      "Minden készség visszakerül a friss telepítés szerinti állapotba: a saját és a Claude-készségek bekapcsolva, a beépített készségek kikapcsolva. A be- és kikapcsolási döntései törlődnek.",
    reset: "Visszaállítás",
  },
  scheduledTasks: {
    title: "Ütemezett",
    description: "Promptok ütemezett futtatása bármelyik modelljével.",
    newTask: "Új feladat",
    upcoming: "Közelgő",
    columns: {
      repeats: "Ismétlődés",
      time: "Időpont",
      nextRun: "Következő futtatás",
    },
    monthlyDay: "Havonta, a hónap {day}. napján",
    filterBy: "Szűrés",
    filter: "Feladatok szűrése",
    active: "Aktív",
    noMatches: "Nincsenek ütemezett feladatok",
    search: "Ütemezett feladatok keresése",
    newResult: "Új eredmény",
    sentBy: "Ütemezett feladat küldte",
    chatMark: "Ütemezett feladatból származó csevegés",
    taskMissing: "Ez a feladat már nem létezik.",
    emptyTitle: "Még nincsenek ütemezett feladatok",
    emptyDescription:
      "Kérjen napi összefoglalót, heti emlékeztetőt vagy bármi mást, amit ütemezetten szeretne elvégeztetni.",
    createFirst: "Feladat létrehozása",
    loadFailed: "Nem sikerült betölteni az ütemezett feladatokat.",
    retry: "Újrapróbálkozás",
    paused: "Szüneteltetve",
    completed: "Befejezve",
    running: "Fut",
    lastRun: "Utolsó futtatás: {when}",
    lastRunFailed: "Az utolsó futtatás sikertelen ({when}): {reason}",
    openChat: "Csevegés megnyitása",
    runNow: "Futtatás most",
    pause: "Szüneteltetés",
    resume: "Folytatás",
    edit: "Szerkesztés",
    options: "Feladat beállításai",
    schedule: {
      once: "Egyszer: {date}, {time}",
      daily: "Minden nap, időpont: {time}",
      weekdays: "Minden munkanap, időpont: {time}",
      weekly: "Hetente ({days}), időpont: {time}",
      monthly: "Minden hónap {day}. napján, időpont: {time}",
    },
    form: {
      createTitle: "Feladat ütemezése",
      editTitle: "Ütemezett feladat szerkesztése",
      name: "Név",
      namePlaceholder: "Név (nem kötelező)",
      instructions: "Utasítások",
      instructionsHint:
        "A feladat minden futtatásakor elküldésre kerül a modellnek.",
      instructionsPlaceholder: "Utasítások…",
      repeat: "Ismétlés",
      once: "Egyszer",
      daily: "Naponta",
      weekdays: "Munkanapokon",
      weekly: "Hetente",
      monthly: "Havonta",
      date: "Dátum",
      time: "Időpont",
      days: "Napok",
      dayOfMonth: "A hónap napja",
      lastDayNote:
        "Azokban a hónapokban, amelyekben nincs ilyen nap, az utolsó napon fut.",
      advanced: "Speciális",
      notifications: "Értesítések",
      notifyAll: "Minden futtatás",
      notifyFailures: "Csak hibák",
      notifyNone: "Ki",
      archiveSuccessful: "Sikeres futtatások archiválása",
      project: "Projekt",
      projectNone: "Nincs",
      model: "Modell",
      modelLoaded: "Az éppen betöltött modell",
      modelOnDevice: "Ezen az eszközön",
      modelNote:
        "Ha a feladat futásakor ez a modell nincs betöltve, az Unsloth előbb rákérdez.",
      modelNoteAllowed:
        "Csak akkor tölti be ezt a modellt, amikor Ön nem használja az Unslothot, majd visszaállítja az előző állapotot.",
      modelEmpty:
        "Töltsön le egy modellt a Modellközpontban, vagy adjon hozzá egy kapcsolatot, hogy másikat választhasson.",
      modelMissing: "{model} (nincs ezen az eszközön)",
      allowLoad: "Automatikus váltás, amikor nem használom az Unslothot",
      allowLoadHint:
        "Amikor ez a feladat esedékes, és Ön 5 perce nem használta az Unslothot, kérdés nélkül betölti ezt a modellt, lefuttatja a feladatot, majd visszavált az előző modellre. Ha Ön éppen használja az Unslothot vagy tanítás fut, a futtatás kimarad, és a modellje érintetlen marad. Kikapcsolva az Unsloth a betöltés előtt rákérdez.",
      reasoning: "Gondolkodási szint",
      reasoningDefault: "Alapértelmezett",
      reasoningOff: "Ki",
      reasoningLowest: "Legalacsonyabb",
      reasoningHighest: "Legmagasabb",
      create: "Létrehozás",
    },
    deleteDialog: {
      title: "Törli az ütemezett feladatot?",
      description:
        "„{title}” nem fut többé. A már létrehozott csevegések megmaradnak.",
    },
    errors: {
      noModel:
        "Nem volt betöltve modell. Töltsön be egy modellt a Csevegésben, majd futtassa újra a feladatot.",
      modelNotLoaded:
        "A feladat modellje nem volt betöltve. Töltse be a Csevegésben, majd futtassa újra a feladatot.",
      modelLoadFailed:
        "Nem sikerült betölteni a feladat modelljét. Ellenőrizze, hogy még le van-e töltve a Modellközpontban.",
      modelBusy:
        "Kihagyva, mert Ön éppen használta az Unslothot, így a modellje érintetlen maradt.",
      connectionUnavailable:
        "A feladat kapcsolata hiányzik vagy ki van kapcsolva. Ellenőrizze a Kapcsolatok között.",
      timeout: "A modell túl sokáig nem válaszolt.",
      cancelled: "A futtatás leállt, mielőtt befejeződött volna.",
      interrupted:
        "Az Unsloth bezárult, mielőtt a futtatás befejeződött volna.",
      empty: "A modell üres választ adott.",
      unknown: "A futtatás sikertelen.",
      saveFailed: "Nem sikerült menteni a feladatot",
      updateFailed: "Nem sikerült frissíteni a feladatot",
      deleteFailed: "Nem sikerült törölni a feladatot",
      runFailed: "Nem sikerült elindítani a feladatot",
    },
    loadPrompt: {
      title: "Betölti a modellt? ({model})",
      description:
        "„{title}” a következő modellen fut, amely nincs betöltve: {model}. A futtatás a befejezésig betölti a modellt ({model}), majd visszaállítja az előző állapotot.",
      always:
        "Ne kérdezzen többet. Csak akkor váltson, amikor nem használom az Unslothot.",
      notNow: "Most nem",
      load: "Betöltés és futtatás",
    },
    toast: {
      finished: "Elkészült: „{title}”",
      failed: "Sikertelen: „{title}”",
      many: "{count} ütemezett feladat befejeződött",
      open: "Megnyitás",
      view: "Megtekintés",
    },
    notificationBody:
      "Az ütemezett feladat befejeződött. A válasz elolvasásához nyissa meg az Unslothot.",
  },
  library: {
    tabs: {
      ariaLabel: "Könyvtár szakaszai",
      suggested: "Javasolt",
      favorites: "Kedvencek",
      folders: "Mappák",
      images: "Képek",
      videos: "Videók",
      audio: "Hang",
      models: "Finomhangolt modellek",
      all: "Összes",
      chats: "Csevegések",
    },
    chats: {
      sections: {
        chats: "Csevegések",
        projects: "Projektek",
        archived: "Archivált",
        ariaLabel: "Csevegésszakaszok",
      },
      search: {
        chats: "Csevegések keresése",
        projects: "Projektek keresése",
        archived: "Archivált csevegések keresése",
        project: "Keresés itt: „{project}”",
        sections: "Szakaszok keresése",
      },
      toolbar: {
        show: "Megjelenítés",
        pinned: "Rögzített",
        forks: "Elágazások",
        comparisons: "Összehasonlítások",
        project: "Projekt",
        model: "Modell",
        groupBy: "Csoportosítás",
        groupNone: "Nincs csoportosítás",
        groupDate: "Dátum",
        groupProject: "Projekt",
        pinnedFirst: "Rögzítettek elöl",
        newChat: "Új csevegés",
        newProject: "Új projekt",
        sortChats: "Csevegések száma",
        withoutSection: "Nincs szakasz",
      },
      groups: {
        today: "Ma",
        yesterday: "Tegnap",
        week: "Előző 7 nap",
        month: "Előző 30 nap",
      },
      list: {
        created: "Létrehozva",
        lastActive: "Utoljára aktív",
        location: "Hely",
        lastModified: "Utolsó módosítás",
        contents: "Tartalom",
        oneMessage: "1 üzenet",
        messageCount: "{count} üzenet",
        showMore: "Több megjelenítése ({count})",
      },
      badges: {
        pinned: "Rögzítve",
        fork: "Elágazás",
        compare: "Összehasonlítás",
      },
      menu: {
        fork: "Csevegés elágaztatása",
        moveTo: "Áthelyezés projektbe",
        newChatInProject: "Új csevegés a projektben",
        edit: "Szerkesztés",
        deleteProject: "Projekt törlése",
        newChatInSection: "Új csevegés a szakaszban",
      },
      project: {
        oneProject: "1 projekt",
        projectCount: "{count} projekt",
      },
      empty: {
        chatsTitle: "Még nincsenek csevegések",
        chatsDescription:
          "A beszélgetései itt jelennek meg, a fájloktól elkülönítve.",
        archivedTitle: "Nincs archivált elem",
        archivedDescription: "Az archivált csevegések itt jelennek meg.",
        sectionsTitle: "Még nincsenek szakaszok",
        sectionsDescription:
          "A szakaszokkal tetszés szerint csoportosíthatja a csevegéseket és projekteket az oldalsávban.",
        sectionTitle: "Ebben a szakaszban még nincs semmi",
        sectionDescription:
          "Helyezzen ide csevegéseket a menüjükből, vagy indítson új csevegést ebben a szakaszban.",
      },
      dialog: {
        renameChat: "Csevegés átnevezése",
        deleteChatDescription:
          "Ez a csevegés véglegesen törlődik. Ez a művelet nem vonható vissza.",
        deleteProjectDescription:
          "A projekt és a csevegései ({count}) véglegesen törlődnek. Ez a művelet nem vonható vissza.",
        deleteFilesMany:
          "A csevegések által létrehozott fájlok és sandboxmappák eltávolítása is.",
        deleteProjectFilesLabel: "A projekt munkaterület-mappájának törlése",
        deleteProjectFiles:
          "A projekt munkaterület-mappája törlődik a lemezről.",
      },
      toast: {
        projectMoved: "Projekt áthelyezve: {project} → {section}",
        projectUnfiled:
          "Projekt eltávolítva a szakaszból: {project} ({section})",
        forkFailed: "Nem sikerült az elágaztatás",
        projectDeleted: "Törölve: „{name}”",
        projectDeleteFailed: "Nem sikerült törölni a projektet",
        removedFromSection: "Eltávolítva a szakaszból: {count}",
      },
      folder: {
        openChat: "Csevegés mappájának megnyitása",
        openChatTitle:
          "A mappa megnyitása, amelyet a csevegés eszközhívásai olvasnak és írnak",
        openProject: "Projektmappa megnyitása",
        openProjectTitle:
          "A mappa megnyitása, amelyet a projekt csevegései olvasnak és írnak",
        chatHint:
          "Egy csevegés fájlmappáját csak az asztali alkalmazás tudja megnyitni. Böngészőben töltse le a fájlt abból az eszközeredményből, amely létrehozta.",
        projectHint:
          "Egy projekt mappáját csak az asztali alkalmazás tudja megnyitni. Böngészőben töltse le a fájlokat azokból az eszközeredményekből, amelyek létrehozták őket.",
        manyFolders: "Ez a csevegés több mappába is írt.",
        manyFoldersDescription:
          "Egy áthelyezés előtt és után is futtatott eszközöket, ezért a mappát inkább egy eszközkártyáról nyissa meg.",
        chatFailed: "Nem sikerült megnyitni a csevegés mappáját.",
        projectFailed: "Nem sikerült megnyitni a projektmappát.",
        chatMissing: "Ennek a csevegésnek még nincs mappája.",
        projectMissing: "Ennek a projektnek még nincs mappája.",
        missingDescription:
          "Akkor jön létre, amikor egy eszköz először ír fájlt.",
        upload: "Feltöltés a csevegés mappájába",
        uploadTitle:
          "Fájlok hozzáadása a mappához, amelyet a csevegés eszközhívásai olvasnak és írnak",
        uploaded: "Hozzáadva a csevegés mappájához: {name}",
        uploadedMany: "{count} fájl hozzáadva a csevegés mappájához",
        uploadedDescription:
          "A csevegés eszközei most már megnyithatják őket. A beszélgetéshez nem lettek hozzáadva.",
        uploadExists: "Már szerepel a csevegés mappájában: {name}",
        uploadExistsMany: "{count} fájl már szerepel a csevegés mappájában",
        replace: "Csere",
        uploadFailed: "Nem sikerült feltölteni: {name}",
      },
    },
    empty: {
      suggestedTitle: "A könyvtár üres",
      suggestedDescription:
        "A csevegésekben feltöltött vagy létrehozott fájlok itt jelennek meg.",
      favoritesTitle: "Még nincsenek kedvencek",
      favoritesDescription:
        "Adjon fájlokat a kedvencekhez, hogy itt gyorsan megtalálja őket.",
      foldersTitle: "Hozza létre az első mappát",
      foldersDescription:
        "Hozzon létre mappákat a könyvtár elemeinek rendszerezéséhez.",
      imagesTitle: "Még nincsenek képek",
      imagesDescription: "A feltöltött vagy generált képek itt jelennek meg.",
      videosTitle: "Még nincsenek videók",
      videosDescription: "A feltöltött vagy generált videók itt jelennek meg.",
      audioTitle: "Még nincs hanganyag",
      audioDescription:
        "A Hang oldalon készített beszéd, zene és sávok itt jelennek meg.",
      modelsTitle: "Még nincsenek finomhangolt modellek",
      modelsDescription:
        "Az Unslothban tanított vagy exportált modellek itt jelennek meg.",
      createFolder: "Mappa létrehozása",
      generateImage: "Kép generálása",
      generateVideo: "Videó generálása",
      generateAudio: "Hang generálása",
      trainModel: "Modell tanítása",
      uploadFiles: "Fájlok feltöltése",
      dropHere: "vagy húzza ide a fájlokat",
      noMatchesTitle: "Nincs találat",
      noMatchesDescription:
        "Próbálkozzon másik kereséssel, vagy törölje a szűrőket.",
      loadErrorTitle: "Nem sikerült betölteni a Könyvtárat",
      loadErrorFallback: "Hiba történt.",
      tryAgain: "Próbálja újra",
    },
    sections: {
      items: "Elemek",
    },
    breadcrumb: "Navigációs útvonal",
    searchLibrary: "Keresés a könyvtárban",
    searchFolder: "Keresés a mappában",
    dropToUpload: "Engedje el a feltöltéshez",
    dropToUploadInto: "Engedje el a feltöltéshez ide: „{folder}”",
    itemCountOne: "{count} elem",
    itemCount: "{count} elem",
    selectItem: "Kijelölés: {name}",
    selection: {
      startChat: "Csevegés indítása",
      move: "Áthelyezés",
      selectAll: "Összes kijelölése",
      countOfTotal: "{count} / {total} kijelölve",
      clear: "Kijelölés megszüntetése",
    },
    menu: {
      moreActions: "További műveletek",
      chatAboutThis: "Csevegés erről",
      chatWithModel: "Csevegés ezzel a modellel",
      editImage: "Kép szerkesztése",
      addToFavorites: "Hozzáadás a Kedvencekhez",
      regenerateTitle: "Cím újragenerálása",
      removeFromFavorites: "Eltávolítás a Kedvencek közül",
      download: "Letöltés",
      addToFolder: "Hozzáadás mappához",
      newFolder: "Új mappa",
      noFolder: "Könyvtár (nincs mappa)",
      deleteFolder: "Mappa törlése",
    },
    reveal: {
      finder: "Megjelenítés a Finderben",
      explorer: "Megjelenítés az Intézőben",
      files: "Megjelenítés mappában",
    },
    toolbar: {
      filter: "Szűrés",
      source: "Forrás",
      fileType: "Fájltípus",
      clearFilters: "Szűrők törlése",
      workflow: "Létrehozás módja",
      uploaded: "Feltöltött",
      generated: "Generált",
      documents: "Dokumentumok",
      spreadsheets: "Táblázatok",
      presentations: "Bemutatók",
      pdfs: "PDF-fájlok",
      gridView: "Rácsnézet",
      listView: "Listanézet",
      grid: "Rács",
      list: "Lista",
      settings: "Könyvtár beállításai",
      sort: "Rendezés",
      sortDefault: "Alapértelmezett sorrend",
      sortName: "Név",
      sortModified: "Módosítva",
      sortSize: "Méret",
      sortAscending: "Növekvő",
      sortDescending: "Csökkenő",
    },
    audio: {
      play: "Lejátszás: {name}",
      pause: "Szüneteltetés: {name}",
      playFailed: "Nem sikerült lejátszani: {name}",
      fromRun: "Ugyanabból a futtatásból",
      downloadRun: "Összes letöltése",
    },
    create: {
      note: "Jegyzet",
      image: "Kép",
      video: "Videó",
      audio: "Hang",
      model: "Modell",
      folder: "Mappa",
      untitledNote: "Névtelen jegyzet",
    },
    list: {
      selectAll: "Összes kijelölése",
      name: "Név",
      lastActivity: "Utolsó tevékenység",
      favorite: "Kedvenc",
      modified: "Módosítva: {time}",
      opened: "Megnyitva: {time}",
      modifiedColumn: "Módosítva",
      justNow: "az imént",
    },
    size: {
      bytes: "{value} B",
      kilobytes: "{value} KB",
      megabytes: "{value} MB",
      gigabytes: "{value} GB",
      terabytes: "{value} TB",
    },
    modelKind: {
      lora: "LoRA",
      fullFineTune: "Teljes finomhangolás",
      loraExport: "LoRA-exportálás",
      mergedExport: "Egyesített exportálás",
      ggufExport: "GGUF-exportálás",
      model: "Modell",
    },
    dialog: {
      create: "Létrehozás",
      renameFolder: "Mappa átnevezése",
      renameFile: "Fájl átnevezése",
      deleteTitle: "Törli a következőt: „{name}”?",
      deleteManyTitle: "Töröl {count} elemet?",
      deleteManyDescription:
        "A fájlok az eredeti helyükről törlődnek. A törölt mappákban lévő elemek egy szinttel feljebb kerülnek.",
      deleteFolderIntoParent:
        "A teljes tartalom ide kerül át: „{folder}”. Egyetlen fájl sem törlődik.",
      deleteFolderIntoLibrary:
        "A teljes tartalom a Könyvtárba kerül át. Egyetlen fájl sem törlődik.",
      deleteUpload: "Ez véglegesen törli a fájlt.",
      deleteAttachment:
        "Abból a csevegésből is eltávolításra kerül, amelyhez csatolták.",
      deleteImage: "A Képek galériából is eltávolításra kerül.",
      deleteVideo: "A Videó galériából is eltávolításra kerül.",
      deleteAudio: "A Hang galériából is eltávolításra kerül.",
      deleteSandbox:
        "Abból a csevegésből is eltávolításra kerül, amely létrehozta.",
      deleteModel: "Ez véglegesen törli a modellt a lemezről.",
      unsavedTitle: "A módosítások nem lettek mentve",
      unsavedDescription:
        "{reason} Próbálja újra, vagy vesse el a módosításokat a fájl bezárásához.",
      keepEditing: "Szerkesztés folytatása",
      discardChanges: "Módosítások elvetése",
    },
    preview: {
      code: "Kód",
      preview: "Előnézet",
      fromChat: "Csevegésből",
      viewOriginalChat: "Eredeti csevegés megtekintése",
      viewInImages: "Megtekintés a Képekben",
      viewInVideo: "Megtekintés a Videóban",
      viewInAudio: "Megtekintés a Hang oldalon",
      viewOriginalProject: "Eredeti projekt megtekintése",
      viewTrainingRun: "Tanítási futtatás megtekintése",
      sheetTruncated: "A munkalap egy része látható",
      documentTruncated: "A dokumentum eleje látható",
      emptyDocument: "Ez a fájl üres.",
      cannotPreview:
        "Ez a fájl itt nem jeleníthető meg előnézetben. A megnyitáshoz töltse le.",
      noPreview: "Ehhez a fájltípushoz nincs előnézet.",
      tooLargeToPreview:
        "Ez a fájl túl nagy az itteni előnézethez. A megnyitáshoz töltse le.",
      startWriting: "Kezdjen írni…",
      type: "Típus",
      baseModel: "Alapmodell",
      size: "Méret",
      location: "Hely",
      unknown: "Ismeretlen",
      readOnlyUtf16:
        "Ez a fájl nem érvényes UTF-16 szöveg, ezért itt csak olvasásra nyílik meg.",
      readOnlyNotUtf8:
        "Ez a fájl nem UTF-8 szöveg, ezért itt csak olvasásra nyílik meg. Egyes karakterek esetleg nem jelennek meg helyesen.",
    },
    viewer: {
      scale: "Méretezés",
      fit: "Igazítás az ablakhoz",
      moreActionsImage: "További műveletek ezzel a képpel",
      moreActionsVideo: "További műveletek ezzel a videóval",
      moreActionsClip: "További műveletek ezzel a klippel",
      moreActionsFile: "További műveletek ezzel a fájllal",
      openImage: "Kép megnyitása",
      openImageNamed: "Kép megnyitása: {prompt}",
      openVideo: "Videó megnyitása",
      untitledImage: "Névtelen kép",
      untitledVideo: "Névtelen videó",
    },
    project: {
      label: "Projekt",
      newProject: "Új projekt",
      noProjects: "Még nincsenek projektek",
      alreadyIn: "Már a projektben van: {project}",
      addedTo: "Hozzáadva a projekthez: {project}",
      createAndAdd: "Létrehozás és hozzáadás",
      addFailedImage: "Nem sikerült hozzáadni a képet a projekthez",
      addFailedVideo: "Nem sikerült hozzáadni a videót a projekthez",
      addFailedClip: "Nem sikerült hozzáadni a klipet a projekthez",
      addFailedFile: "Nem sikerült hozzáadni a fájlt a projekthez",
      newProjectImage: "Kép hozzáadása új projekthez",
      newProjectVideo: "Videó hozzáadása új projekthez",
      newProjectClip: "Klip hozzáadása új projekthez",
      newProjectFile: "Fájl hozzáadása új projekthez",
    },
    toast: {
      missingItem: "Ez a fájl már nincs a Könyvtárban",
      moveFailed: "Nem sikerült az áthelyezés",
      movedToFolder: "Áthelyezve ide: {folder}",
      movedToLibrary: "Áthelyezve a Könyvtárba",
      folderFallback: "mappa",
      favoritesFailed: "Nem sikerült frissíteni a kedvenceket",
      addedToFavorites: "Hozzáadva a Kedvencekhez",
      removedFromFavorites: "Eltávolítva a Kedvencek közül",
      uploadingOne: "Feltöltés: {name}…",
      uploadingMany: "{count} fájl feltöltése…",
      uploadedOne: "Fájl feltöltve",
      uploadedMany: "{count} fájl feltöltve",
      uploadFailed: "A feltöltés sikertelen",
      uploadTooLarge: "Nagyobb mint 512 MB: {name}",
      readDropsFailed: "Nem sikerült beolvasni a behúzott fájlokat",
      createNoteFailed: "Nem sikerült létrehozni a jegyzetet",
      renameFailed: "Nem sikerült az átnevezés",
      createFolderFailed: "Nem sikerült létrehozni a mappát",
      signedOutBeforeFolder: "A mappa létrehozása előtt kijelentkezett.",
      signedOutBeforeSave: "A módosítás mentése előtt kijelentkezett.",
      signedOutBeforeUpload: "A feltöltés befejezése előtt kijelentkezett.",
      deleteFailed: "Néhány elemet nem sikerült törölni",
      saveFailed: "Nem sikerült a mentés",
      saveNoteFirst: "Előbb mentse a jegyzetet.",
      revealFailed: "Nem sikerült megnyitni a fájlkezelőt",
      downloadFailed: "Nem sikerült letölteni: {name}",
      downloadManyFailed: "Nem sikerült letölteni a fájlokat",
      downloadingMany: "{count} fájl letöltése",
      downloadingManyDescription:
        "Ha a böngésző rákérdez, engedélyezze, hogy ez a webhely több fájlt töltsön le.",
      preparingMany: "{count} fájl előkészítése…",
      zipFileName: "Könyvtár fájljai",
      nothingToChat: "Még nincs miről csevegni",
      emptyFolder: "Ebben a mappában nincsenek fájlok.",
      tooLargeOne: "Túl nagy a csatoláshoz: {name}",
      tooLargeMany: "Ezek a fájlok túl nagyok a csatoláshoz",
      attachedOne: "{count} fájl csatolva",
      attachedMany: "{count} fájl csatolva",
      skippedTooLarge: "{count} túl nagy a csatoláshoz.",
      skippedOverLimit:
        "További {count} meghaladja a(z) {limit} fájlos korlátot.",
      chatFilesWaiting:
        "{count} akkor lesz csatolva, amikor betölt egy modellt, amely olvasni tudja őket.",
      openFilesFailed: "Nem sikerült megnyitni a fájlokat",
      editImageFailed: "Nem sikerült megnyitni a képet szerkesztésre",
      speechModel: "Ez egy beszédmodell: {name}",
      speechModelDescription: "Válassza ki a Hang oldal modellmenüjéből.",
      noMediaLink:
        "A kiszolgáló nem adott vissza hivatkozást a fájl lejátszásához.",
      imageNotFound: "Nem található ez a kép",
      clipNotFound: "Nem található ez a klip",
      notFoundDescription: "Lehet, hogy archiválták vagy törölték.",
      imageTooLarge: "Ez a kép túl nagy a csatoláshoz",
      videoTooLarge: "Ez a videó túl nagy a csatoláshoz",
      attachmentLimit:
        "A csevegésmellékletek mérete legfeljebb {size} MB lehet.",
      attachingImage: "Kép csatolása…",
      attachingVideo: "Videó csatolása…",
      readImageFailed: "Nem sikerült beolvasni a képet ({status}).",
      readVideoFailed: "Nem sikerült beolvasni a videót ({status}).",
      openFileFailed: "Nem sikerült megnyitni a fájlt",
    },
  },
  exportDecision: {
    title: "Döntési modell",
    description:
      "A döntési modellek ({layout}) GGUF formátumba exportálhatók a llama.cpp döntési kiszolgálójához. Válasszon egy vagy több kvantálást.",
    adapterNote:
      "Ez az ellenőrzőpont LoRA-adaptereket tartalmaz; ezek a konvertálás előtt egyesítésre kerülnek.",
    notEligibleTitle: "A GGUF-exportálás nem érhető el",
    ggufOnly: "A döntési modellek csak GGUF formátumba exportálhatók",
    existing: "Már exportálva: {quantizations}",
    outputNote: "A GGUF-fájlok a futtatás mappájába kerülnek: {path}",
    methodLabel: "Döntési modell GGUF",
    outputLabel: "Döntési GGUF ({quantizations})",
  },
  decisions: {
    title: "Döntés kipróbálása",
    description: "Kérdezze a döntési modellt bármilyen szövegről.",
    tryIt: "Kipróbálás",
    tryItOff:
      "A kipróbáláshoz kapcsolja be a „Kérések kiszolgálása” lehetőséget.",
    typeNoul: "Igen / nem",
    typeChoice: "Választás",
    typeScore: "Pontszám",
    aboutNoul: "Igennel vagy nemmel válaszol, az igen valószínűségével együtt.",
    aboutChoice:
      "Egy lehetőséget választ egy listából, mindegyikhez valószínűséggel.",
    aboutScore:
      "Szintek skáláján értékeli az állapotot, a legalacsonyabbtól kezdve.",
    model: "Döntési modell",
    defaultModel: "Alapértelmezett ({model})",
    layaModel: "Laya {model}",
    thisMachine: "Ez a gép",
    modelReady: "Kész",
    modelLoading: "Betöltés",
    modelNotLoaded: "Az első futtatáskor töltődik be",
    connectionNotDefault:
      "Előbb állítsa be ezt Decision API-modellként a Beállítások > API alatt.",
    question: "Kérdés",
    questionHelp: "Amiről a modellnek döntenie kell.",
    result: "Eredmény",
    form: "Űrlap",
    json: "JSON",
    preview: "Előnézet",
    state: "Állapot",
    stateHelp: "A szöveg, amelyről dönteni kell. Egyszerű szöveg vagy JSON.",
    criteria: "Feltételek",
    criteriaNoulHelp: "Nem kötelező. Mi számít igennek és mi nemnek.",
    criteriaChoiceHelp: "A választható lehetőségek, mindegyik rövid leírással.",
    criteriaScoreHelp: "A skála szintjei, a legalacsonyabbtól kezdve.",
    yesWhen: "Igen, ha",
    noWhen: "Nem, ha",
    optionName: "Lehetőség",
    optionDescription: "Leírás",
    optionNamePlaceholder: "név",
    optionDescriptionPlaceholder: "Mikor kell választani",
    optionNameLabel: "{n}. lehetőség neve",
    optionDescriptionLabel: "{n}. lehetőség leírása",
    removeOption: "{n}. lehetőség eltávolítása",
    addOption: "Lehetőség hozzáadása",
    levelLabel: "{n}. szint",
    removeLevel: "{n}. szint eltávolítása",
    addLevel: "Szint hozzáadása",
    jsonHelp: "A kérés törzse. A Futtatás a beírt formában küldi el.",
    examples: "Példák",
    presetToolApproval: "Ügynöki eszközhívások jóváhagyása",
    presetPromptInjection: "Promptinjektálás észlelése",
    presetTicketRouting: "Támogatási jegyek továbbítása",
    presetModeration: "Üzenetek moderálása",
    presetIntent: "Szándék felismerése",
    presetGrading: "Válasz értékelése",
    run: "Döntés futtatása",
    running: "Fut",
    answer: "Válasz",
    yes: "Igen",
    no: "Nem",
    scoreDetail: "Pontszám: {score} (0–{max} skálán)",
    latency: "{ms} ms",
    inputTokens: "{count} bemeneti token",
    responseTitle: "Válasz",
    emptyTitle: "Még nincs eredmény",
    emptyBody:
      "Kattintson a Döntés futtatása gombra, vagy nyomja meg ezt: {shortcut}.",
    cantRun: "Nem sikerült a futtatás",
    failed: "Hiba: {status}",
    invalidJson: "A kérés törzse nem érvényes JSON.",
  },
  floatingPanels: {
    loadedModels: {
      title: "Betöltött modellek",
      countOne: "{count} modell betöltve",
      countMany: "{count} modell betöltve",
      pillLabel:
        "{count}. Részletek megjelenítése, vagy húzza az áthelyezéshez",
      dragToMove: "Húzza az áthelyezéshez",
      dragHandleLabel: "Húzza az áthelyezéshez",
      collapse: "Összecsukás",
      collapseLabel: "Betöltött modellek összecsukása",
      close: "Bezárás",
      closeLabel: "Betöltött modellek bezárása",
      backOnNextLoad: "A következő modellbetöltéskor újra megjelenik",
      ejectToFree: "Kiürítés a memória felszabadításához",
      ejectLabel: "Kiürítés: {model}",
      openTarget: "Megnyitás: {target}",
      rowLabel: "{model}. Megnyitás: {target}",
      loading: "Betöltés",
      ejecting: "Kiürítés",
      stillInMemory: "Még a memóriában van",
      alsoServing: "Kiszolgálást is végez",
      noneLoaded: "Nincs betöltött modell",
      noneLoadedDetail: "A betöltött modellek itt jelennek meg.",
      otherAccount: "Egy másik fiók használja",
      otherAccountDetail:
        "A memóriát egy Ön számára nem látható modell foglalja",
      toastStillResident:
        "„{model}” a kiürítés közben betöltődött, ezért még memóriát használ. A felszabadításhoz ürítse ki újra.",
      toastUnverified:
        "A kiürítés kérése elküldve, de a futtatókörnyezet nem erősítette meg: {model}. Kis idő múlva ellenőrizze a kártyát.",
      toastAlreadyFree: "Már nem volt betöltve: {model}.",
      toastReplaced:
        "Már nincs betöltve: {model}. A helyére „{resident}” került, és érintetlen maradt.",
      toastEjected: "Kiürítve: {model}",
      toastFailed: "Nem sikerült a kiürítés: {model}",
    },
    monitor: {
      dragHandleLabel:
        "Az élő figyelő áthelyezése. A nyílbillentyűk finoman mozgatják, a dupla kattintás visszaállítja.",
      reconnecting: "Újracsatlakozás...",
    },
  },
  projectsPage: {
    title: "Projektek",
    search: "Projektek keresése",
    importExport: "Projektek importálása / exportálása",
    importChatsAction: "Csevegések importálása…",
    exportAllProjects: "Összes projekt exportálása",
    exportProjectsAndRecents: "Projektek és legutóbbiak exportálása",
    combined: "Egyesítve",
    perChat: "Csevegésenként",
    newProject: "Új projekt",
    name: "Név",
    updated: "Frissítve",
    newestFirst: "Legújabb elöl",
    oldestFirst: "Legrégebbi elöl",
    today: "Ma",
    yesterday: "Tegnap",
    emptyDescription:
      "A projekt egy helyen tartja az összetartozó csevegéseket, közös utasításokkal és forrásokkal, amelyeket a benne lévő minden csevegés használhat.",
    createFirst: "Hozza létre az első projektjét",
    noMatches: "Nincs a keresésnek megfelelő projekt.",
    clearSearch: "Keresés törlése",
    showChats: "Csevegések megjelenítése",
    hideChats: "Csevegések elrejtése",
    pinProject: "Projekt rögzítése",
    unpinProject: "Projekt rögzítésének feloldása",
    newChatInProject: "Új csevegés a projektben",
    projectOptions: "Projekt beállításai",
    loadChatsFailed: "Nem sikerült betölteni a csevegéseket. Újrapróbálkozás",
    refreshChatsFailed:
      "Nem sikerült frissíteni a csevegéseket. Újrapróbálkozás",
    noChats: "Nincsenek csevegések",
    unread: "Olvasatlan",
    pinChat: "Csevegés rögzítése",
    unpinChat: "Csevegés rögzítésének feloldása",
    chatOptions: "Csevegés beállításai",
    exportAllChats: "Összes csevegés exportálása…",
    archive: "Archiválás",
    renameChat: "Csevegés átnevezése",
    chatName: "Csevegés neve",
    deleteChat: "Csevegés törlése",
    deleteChatDescription:
      "Biztosan törölni szeretné a következőt: {name}? Az üzenetei véglegesen törlődnek.",
    importChats: "Csevegések importálása",
    importDestination: "Válassza ki az importálás helyét: {name}",
    selectDestination: "Cél kiválasztása",
    recents: "Legutóbbiak",
    import: "Importálás",
    deleteProject: "Projekt törlése",
    deleteProjectDescription:
      "Biztosan törölni szeretné a következőt: {name}? A csevegései véglegesen törlődnek.",
    deleteProjectFiles: "A projekt munkaterület-mappája törlődik a lemezről.",
    deleteAll: "Összes törlése",
    projectFallback: "projekt",
    importFailed: "Az importálás sikertelen.",
    nothingToExport: "Nincsenek exportálható beszélgetések.",
    exportFailed: "Az exportálás sikertelen.",
    renameChatFailed: "Nem sikerült átnevezni a csevegést",
    archiveChatFailed: "Nem sikerült archiválni a csevegést",
    deleteChatFailed: "Nem sikerült törölni a csevegést",
    deleteProjectFailed: "Nem sikerült törölni a projektet",
  },
  auth: {
    welcomeBack: "Üdvözöljük újra",
    loginSubtitle: "Jelentkezzen be a jelszavával.",
    setupTitle: "Fiók beállítása",
    setupSubtitle: "Új jelszó létrehozása",
    login: "Bejelentkezés",
    changePassword: "Jelszó módosítása",
    alreadySetUp: "Már beállította a jelszót? ",
    backToLogin: "Vissza a bejelentkezéshez",
    username: "Felhasználónév",
    password: "Jelszó",
    currentPassword: "Jelenlegi jelszó",
    newPassword: "Új jelszó",
    confirmPassword: "Jelszó megerősítése",
    setupCodeHint:
      "Először jelentkezik be? Jelszóként illessze be a rendszergazdától kapott beállítási kódot.",
    ownerHint:
      "Ön a telepítés tulajdonosa? Jelentkezzen be unsloth néven, vagy a Beállítások > Fiókok alatt látható bejelentkezési névvel.",
    currentSetupCodeHint:
      "Jelenlegi jelszóként illessze be azt a beállítási kódot, amellyel az imént bejelentkezett.",
    minLength: "Legalább 8 karakter hosszúnak kell lennie.",
    ensureMatch: "Ügyeljen arra, hogy a jelszavak egyezzenek.",
    noSpaces: "Az új jelszó nem tartalmazhat szóközt.",
    currentTooShort:
      "A jelenlegi jelszónak legalább 8 karakter hosszúnak kell lennie.",
    setupInitFailed:
      "Nem sikerült elindítani a beállítást. Töltse be újra az oldalt, és próbálja újra.",
    newTooShort: "Az új jelszónak legalább 8 karakter hosszúnak kell lennie.",
    mismatch: "A jelszavak nem egyeznek.",
    mustDiffer: "Az új jelszónak különböznie kell a jelenlegi jelszótól.",
    pleaseWait: "Kérjük, várjon...",
    continueAsOwner: "Folytatás tulajdonosként",
    signInWithUsername:
      "Jelentkezzen be a felhasználónevével és a jelszavával.",
    bootstrapping:
      "A hitelesítés még az alapértelmezett rendszergazdai fiókot állítja be.",
    signInToChange:
      "A jelszó módosításához jelentkezzen be egyszer az előre megadott hitelesítő adatokkal.",
    alreadyUpdated:
      "A jelszó már frissítve lett. Használja a bejelentkezési képernyőt.",
    busy: "Az Unsloth foglalt. Próbálja újra egy kis idő múlva.",
    updateFailed: "Nem sikerült frissíteni a jelszót.",
    loadFailed: "Nem sikerült a betöltés.",
    failed: "A hitelesítés sikertelen.",
    shuttingDown:
      "Ez a példány leáll: elérhető volt a hálózaton, és az alapértelmezett jelszavát soha nem módosították.",
    shutdownCountdown:
      "Ez a példány elérhető a hálózaton, és még mindig az alapértelmezett jelszavát használja, ezért leáll ennyi idő múlva: {time}. Ha itt beállít egy jelszót, tovább fut.",
  },
  routes: {
    chat: "Csevegés",
    api: "API",
    dataRecipes: "Data Recipes",
    login: "Bejelentkezés",
    changePassword: "Jelszó módosítása",
  },
  remoteCode: {
    unsafeTitle: "Nem biztonságos fájlok észlelve",
    blockedTitle: "Egyéni kód letiltva",
    enableTitle: "Engedélyezi az egyéni kódot ehhez a modellhez?",
    thisModel: "Ez a modell",
    unsafeBody:
      "Modell: {model}. Olyan fájlokat tartalmaz, amelyeket a Hugging Face biztonsági vizsgálata nem biztonságosnak jelölt (például egy rosszindulatú pickle-fájl, amely a modell betöltésekor kódot futtatna). Nem tölthető be. A megjelölt fájlok le sem lettek töltve.",
    unsafeBodyFromProvider:
      "Modell: {model} ({provider}). Olyan fájlokat tartalmaz, amelyeket a Hugging Face biztonsági vizsgálata nem biztonságosnak jelölt (például egy rosszindulatú pickle-fájl, amely a modell betöltésekor kódot futtatna). Nem tölthető be. A megjelölt fájlok le sem lettek töltve.",
    customCode:
      "Modell: {model}. A tárháza egyéni Python-kódot deklarál. {note}",
    customCodeFromProvider:
      "Modell: {model} ({provider}). A tárháza egyéni Python-kódot deklarál. {note}",
    blockedNote:
      "Egy biztonsági vizsgálat KRITIKUS problémákat jelzett, ezért nem engedélyezhető.",
    reviewNote:
      "Tekintse át az alábbi biztonsági vizsgálatot. Csak akkor folytassa, ha megbízik a modell forrásában.",
    trustNote: "Csak akkor folytassa, ha megbízik a modell forrásában.",
    flaggedIntro:
      "Az automatikus ellenőrzőnk többek között a következő problémákat jelezte:",
    nothingFlagged:
      "Az automatikus ellenőrzőnk nem jelzett aggályos fájlt, de kérjük, ellenőrizze még egyszer.",
    enableAndContinue: "Engedélyezés és folytatás",
  },
  tour: {
    badge: "bemutató",
    defaultTitle: "Gyors bemutató",
    defaultBody: "Segítünk eligazodni.",
    skipTour: "Bemutató kihagyása",
    skip: "Kihagyás",
    back: "Vissza",
    next: "Tovább",
    tip: "Tipp: az `Esc` kihagyja. A bemutató alatt a kattintások le vannak tiltva, hogy nyugodtan olvashasson.",
    readMore: "További információ",
  },
  updateNotes: {
    openRelease: "Kiadás megnyitása",
    openChangelog: "Változásnapló megnyitása",
    retry: "Újrapróbálkozás",
    loadingChanges: "Új változások betöltése...",
    customLlamaRepo:
      "Ez a telepítés egyéni llama.cpp-tárházat követ, ezért a változásai nem hasonlíthatók össze.",
    noItemisedNotes:
      "Ez a build régebbi a tételes kiadási megjegyzéseknél, ezért a változásai nem hasonlíthatók össze.",
    compareFailed: "Nem sikerült összehasonlítani ezeket a kiadásokat.",
    llamaChangesLabel: "Új llama.cpp-változások ({from} → {to})",
    showingChanges: "{shown} / {total} változás látható",
    noCarriedChanges:
      "Ehhez a buildhez nincsenek felsorolva új átvett változások.",
    notesFor: "Kiadási megjegyzések: {release}",
    notesForVersion: "Kiadási megjegyzések, verzió: {version}",
    notesTruncated:
      "A megjegyzések csonkolva. Tekintse meg a teljes kiadási megjegyzéseket.",
    summaryLabel: "Kiadási megjegyzések összefoglalója",
    moreCount: "+{count} további",
    loadingNotes: "Kiadási megjegyzések betöltése...",
    loadFailed: "Nem sikerült betölteni a kiadási megjegyzéseket.",
    noNotesFor:
      "Még nincsenek közzétett kiadási megjegyzések ehhez: {release}.",
    noNotes: "Még nincsenek közzétett kiadási megjegyzések.",
  },
  desktopStartup: {
    checking: "Ellenőrzés...",
    installPrompt: "Az Unsloth telepítéséhez kattintson a Kezdés gombra.",
    getStarted: "Kezdés",
    chatOnly: "Csak csevegés (GGUF): kisebb telepítés, tanítás nélkül",
    installLocation: "Telepítés helye:",
    changeLocation: "Módosítás",
    useDefaultLocation: "Alapértelmezett használata",
    longPaths: {
      title: "A Windows engedélyt fog kérni",
      approve:
        "Hagyja jóvá egyszer a következőt: {tool}, hogy a lefordított kernelekhez bekapcsolódjanak a hosszú fájlelérési utak.",
      waiting: "Várakozás a Windows engedélyére...",
    },
    skip: "Kihagyás",
    continue: "Folytatás",
    install: {
      preparing: "A munkaterület előkészítése...",
      preparingDetail:
        "Annak ellenőrzése, mire van szüksége ennek a számítógépnek.",
      downloading: "A szükséges összetevők letöltése...",
      downloadingDetail: "A helyi környezet létrehozása.",
      installing: "Az Unsloth telepítése...",
      installingDetail: "Helyi MI-eszközök beállítása...",
      finishing: "A beállítás befejezése...",
      finishingDetail: "Minden előkészítése.",
      continuesAutomatically: "A beállítás automatikusan folytatódik.",
      fewMinutes: "Ez néhány percig is eltarthat.",
      someTakeLonger: "Egyes összetevők beállítása tovább tart.",
    },
    details: {
      showInstall: "Telepítési részletek megjelenítése",
      hideInstall: "Telepítési részletek elrejtése",
      showSetup: "Beállítási részletek megjelenítése",
      hideSetup: "Beállítási részletek elrejtése",
      showUpdate: "Frissítési részletek megjelenítése",
      hideUpdate: "Frissítési részletek elrejtése",
    },
    repair: {
      upgrading: "Frissítés a teljes telepítésre...",
      gettingReady: "Előkészítés...",
      upgradingDetail:
        "A PyTorch és a könyvtárai letöltése a gép GPU-jához. Ez több percig is eltarthat.",
      wontTakeLong: "Ez nem tart sokáig.",
      upgradeFailed: "Nem sikerült a frissítés a teljes telepítésre",
      updateFailed: "A frissítés sikertelen",
    },
    closing: "Az Unsloth Desktop bezárása...",
    closingDetail: "A háttérrendszer leállítása.",
    diskFull: "Nincs elég szabad lemezterület",
    setupFailed: "Hiba történt a beállítás során",
    tryAgain: "Próbálja újra",
    retry: "Újrapróbálkozás",
    elevation: {
      title: "Engedély szükséges",
      packages: "A következő rendszercsomagokat kell telepíteni:",
      allow: "Engedélyezés",
    },
    starting: "Az Unsloth indítása...",
    loadingProjects: "Projektek betöltése...",
    loadingModels: "Modellek betöltése...",
    nearlyDone: "Mindjárt kész...",
    finishingUpdate: "A frissítés befejezése...",
    serverStopped: "A kiszolgáló leállt",
    startServer: "Kiszolgáló indítása",
    somethingWrong: "Hiba történt",
    diagnostics: {
      copy: "Diagnosztika másolása",
      copying: "Másolás...",
      clipboardFailed:
        "Nem sikerült a vágólapra másolni. Jelölje ki és másolja az alábbi diagnosztikai adatokat.",
      copyFailed: "Nem sikerült a diagnosztika másolása: {error}",
    },
  },
  desktopUpdate: {
    updatingBackend: "A háttérrendszer frissítése...",
    downloadingApp: "Az alkalmazásfrissítés letöltése...",
    installing: "A frissítés telepítése...",
    failed: "A frissítés sikertelen",
    updating: "Frissítés...",
    updatingBackendDetail:
      "Ez néhány percig is eltarthat. Ne zárja be az alkalmazást.",
    percentDownloaded: "{percent}% letöltve",
    restartShortly: "Az alkalmazás hamarosan újraindul.",
    failedDetail: "Hiba történt a frissítés során.",
    downloadingFile: "Letöltés: {file}",
    downloadingAppLabel: "Az alkalmazásfrissítés letöltése",
    skipRestart: "Kihagyás és újraindítás",
    banner: {
      dismiss: "Az alkalmazásfrissítési értesítés elvetése",
      failedTitle: "Az alkalmazás frissítése sikertelen",
      newVersion: "Új Unsloth-verzió",
      backendRecovered:
        "A háttérrendszer helyreállt. A diagnosztikai adatok továbbra is elérhetők.",
      linuxManual:
        "A Linux-csomag telepítéséhez nyissa meg a GitHub kiadási oldalát",
      available: "Új alkalmazásfrissítés érhető el",
      showNotes: "Kiadási megjegyzések megjelenítése",
      hideNotes: "Kiadási megjegyzések elrejtése",
      copyDiagnostics: "Diagnosztika másolása",
      later: "Később",
      remindLater: "Emlékeztessen később",
      openReleasePage: "Kiadási oldal megnyitása",
      retry: "Frissítés újrapróbálása",
      update: "Frissítés",
    },
  },
  desktopShell: {
    navigation: "Oldalsáv és oldalnavigáció",
    collapseSidebar: "Oldalsáv összecsukása",
    expandSidebar: "Oldalsáv kibontása",
    goBack: "Vissza",
    goForward: "Előre",
    titlebar: "Ablak címsora",
    windowControls: "Ablakvezérlők",
    minimize: "Ablak kis méretre állítása",
    maximize: "Ablak teljes méretre állítása",
    restore: "Ablak előző méretének visszaállítása",
    close: "Ablak bezárása",
  },
  runtimeUpdate: {
    installAudio: "A hang-futtatókörnyezet telepítése",
    installingAudio: "A hangos futtatókörnyezet telepítése...",
    unknownVersion: "ismeretlen",
    failed: "A frissítés sikertelen: {error}",
    unknownError: "ismeretlen hiba",
    skipVersions: "Ezen verziók kihagyása",
    skipChange: "Ennek a változásnak a kihagyása",
    skipVersion: "Ennek a verziónak a kihagyása",
    skippedVersion: "Egyelőre kihagyva: {component} {version}",
    skippedUpdate: "Frissítés egyelőre kihagyva: {component}",
    skippedUpdates: "Frissítések egyelőre kihagyva: {components}",
    vulkanFaster:
      "A Vulkan több mint 10%-kal gyorsabb a ROCm-nél. Nincs szükség újraindításra",
    noRestart: "Nincs szükség újraindításra",
    downloadSize: "Letöltési méret: {size}",
    backendChange: "Háttérrendszer: {change}",
    updating: "Frissítés: {component}...",
    updatingLabel: "Frissítés: {component}",
    installingPhase: "Telepítés: {component} ({current} / {total})...",
    installingCount: "{count} frissítés telepítése...",
    bundleTitle: "{count} futtatókörnyezet-frissítés",
    newBackend: "Új háttérrendszer: {component}",
    newUpdate: "Új frissítés: {component}",
    hideUpdate:
      "Frissítés elrejtése az Unsloth újbóli megnyitásáig: {component}",
    hideUpdates:
      "Frissítések elrejtése az Unsloth újbóli megnyitásáig: {components}",
    whatsNew: "Újdonságok",
    hideWhatsNew: "Újdonságok elrejtése",
    later: "Később",
    update: "Frissítés",
    undo: "Visszavonás",
  },
} satisfies DeepPartialMessageTree<typeof en>;
