// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DeepPartialMessageTree } from "../types";
import type { en } from "./en.ts";

export const fil = {
  linkMenu: {
    openInBrowser: "Buksan sa Unsloth browser",
    openExternal: "Buksan sa external na browser",
    copyLink: "Kopyahin ang link",
    saveLinkAs: "I-save ang link bilang…",
    alwaysInBrowser: "Palaging buksan ang mga link sa Unsloth browser",
    linkCopied: "Nakopya ang link",
    copyFailed: "Hindi makopya.",
    saveFailed: "Hindi ma-save ang file.",
    openFile: "Buksan ang file",
    openDefaultApp: "Buksan sa default na app",
    openWith: "Buksan gamit ang",
    unslothBrowser: "Browser ng Unsloth",
    newChat: "Bagong chat",
    browserTab: "Bagong tab ng browser",
    saveAs: "I-save bilang…",
    copyPath: "Kopyahin ang path",
    copyContents: "Kopyahin ang nilalaman ng file",
    revealFinder: "Ipakita sa Finder",
    revealExplorer: "Ipakita sa Explorer",
    revealFolder: "Ipakita sa folder",
    pathCopied: "Nakopya ang path",
    contentsCopied: "Nakopya ang nilalaman",
    openFailed: "Hindi mabuksan ang {name}.",
    revealFailed: "Hindi maipakita ang {name} sa folder nito.",
  },
  imageViewer: {
    title: "Larawan",
    description:
      "Gamitin ang mga arrow key para lumipat sa pagitan ng mga larawan at ang plus o minus para mag-zoom.",
    failed: "Hindi ma-load ang larawang ito.",
    downloadFailed: "Hindi ma-save ang larawan.",
    openSource: "Buksan ang source page",
    download: "I-download",
    close: "Isara",
    previous: "Nakaraang larawan",
    next: "Susunod na larawan",
    zoomIn: "Mag-zoom in",
    zoomOut: "Mag-zoom out",
    fit: "Ipagkasya sa window",
    copy: "Kopyahin ang larawan",
    copied: "Nakopya ang larawan",
    copyFailed: "Hindi makopya ang larawan.",
    removeFromMessage: "Alisin sa mensahe",
  },
  browser: {
    drafts: {
      title: "I-unload ang page na may inilagay na text?",
      description:
        "Para manatiling mabilis tumugon ang chat, tatlong nakatagong page lang na may inilagay na text ang patuloy na tumatakbo. Kapag nagpatuloy ka, maa-unload ang {pages}, at maaaring mawala ang hindi na-save na text. Kapag Kanselahin, babalik ka sa nakaraang tab.",
      confirm: "I-unload at magpatuloy",
    },
    agentSettings: {
      title: "Paggamit ng browser",
      docs: "Docs",
      docsLabel: "Buksan ang docs ng paggamit ng browser",
      enable: "Hayaang gamitin ng mga model ang browser",
      enableDescription:
        "Nagbubukas ang mga model ng mga page sa sarili nilang tab ng browser, binabasa ang mga ito, at nagki-click at nagta-type para sa iyo. Nasa + menu rin ng composer.",
      history: "History ng pag-browse",
      historyDescription:
        "Nilalaktawan ng Buong access ang pag-apruba; hinaharangan pa rin ng Hindi kailanman ang access",
      alwaysAsk: "Palaging magtanong",
      alwaysAllow: "Palaging payagan",
      never: "Hindi kailanman",
      scripts: "Magpatakbo ng mga script",
      scriptsDescription:
        "Piliin kung puwedeng magpatakbo ang mga model ng mga script sa mga page. Makikita mo ang bawat script bago ito tumakbo",
      cursor: "Ipakita ang cursor",
      cursorDescription:
        "Magpakita ng pointer kung saan nagki-click, nagta-type at nagda-drag ang mga model sa page",
      askEveryTime: "Magtanong sa bawat pagkakataon",
      permissionsTitle: "Mga pahintulot ng agent",
      permissionsDescription:
        "Magtakda ng mga default at exception kada site. Hindi kailanman pinupunan ang mga field ng password at pagbabayad.",
      site: "Site o pattern",
      sitePlaceholder: "example.com o *.example.com",
      default: "Default",
      browsing: "Pag-browse",
      actions: "Mga aksyon",
      downloads: "Mga download",
      unlabelled: "Mga click na walang label",
      askNewSites: "Magtanong para sa mga bagong site",
      requiresApproval: "Kailangan ng pag-apruba",
      blocked: "Naka-block",
      addSite: "Idagdag",
      add: "Idagdag",
      cancel: "Kanselahin",
      removeSite: "Alisin ang {site}",
      permissionsNote:
        "Ang mga site lang na may sariling pahintulot ang lumalabas dito. Ang mga aksyon ay mga pag-sign in, pagsusumite ng form, at mga button na bumibili, nagpapadala, o nagde-delete.",
      siteInvalid: "Maglagay ng site gaya ng example.com o *.example.com.",
    },
    agent: {
      working: "Nagba-browse ang Unsloth sa tab na ito",
      waiting: "Kailangan ng Unsloth ang sagot mo sa browser para sa {chat}",
      otherChat: "ibang chat",
      show: "Ipakita",
      stop: "Ihinto",
      allow: "Payagan",
      deny: "Tanggihan",
      ask: {
        open: "Hayaang buksan ng Unsloth ang {host}?",
        signIn: "Hayaang mag-sign in ang Unsloth sa {host}?",
        submit: "Hayaang isumite ng Unsloth ang form na ito sa {host}?",
        click: "Hayaang i-click ng Unsloth ang “{name}” sa {host}?",
        history: "Hayaang basahin ng Unsloth ang iyong history ng pag-browse?",
        script: "Hayaang patakbuhin ng Unsloth ang script na ito sa {host}?",
        pointer:
          "Hayaang i-click ng Unsloth ang mga bahagi ng {host} na walang label, gaya ng canvas?",
      },
      step: {
        open: "Binubuksan ang {detail}",
        back: "Bumabalik",
        forward: "Sumusulong",
        reload: "Nire-reload ang page",
        read: "Binabasa ang page",
        find: "Hinahanap ang “{detail}”",
        click: "Kini-click ang “{detail}”",
        type: "Nagta-type sa “{detail}”",
        select: "Pinipili ang “{detail}”",
        press: "Pinipindot ang {detail}",
        scroll: "Nag-i-scroll",
        screenshot: "Kumukuha ng screenshot",
        evaluate: "Nagpapatakbo ng script sa page",
        history: "Binabasa ang iyong history ng pag-browse",
        move: "Iginagalaw ang mouse",
        drag: "Nagda-drag",
      },
    },
    title: "Browser",
    tabs: "Mga tab ng browser",
    newTab: "Bagong tab",
    newTabFullView: "Bagong tab sa full view",
    closeTab: "Isara ang tab",
    close: "Isara ang browser",
    show: "Ipakita ang browser",
    back: "Bumalik",
    forward: "Sumulong",
    reload: "I-reload",
    more: "Higit pa",
    addressPlaceholder: "Maghanap o maglagay ng URL",
    local: "Lokal",
    browserUse: {
      label: "Paggamit ng browser",
      description: "Hayaang kontrolin ng mga model ang built-in na browser",
      unavailable:
        "Mag-load ng model na kayang tumawag ng mga tool para magamit nito ang browser.",
    },
    suggested: "Iminumungkahi",
    suggestedMenu: {
      open: "Buksan",
      openInNewTab: "Buksan sa bagong tab",
      remove: "Alisin sa iminumungkahi",
    },
    tabMenu: {
      duplicate: "I-duplicate",
      closeOthers: "Isara ang ibang tab",
      pin: "I-pin sa sidebar",
      unpin: "I-unpin sa sidebar",
      newTabRight: "Bagong tab sa kanan",
      fork: "I-fork",
      forkNewChat: "Bagong chat",
      forkTemporaryChat: "Pansamantalang chat",
      copyUrl: "Kopyahin ang URL",
      openExternal: "Buksan sa external na browser",
      rename: "Palitan ang pangalan",
      renameLabel: "Pangalan ng tab",
      mute: "I-mute ang tab",
      unmute: "I-unmute ang tab",
      close: "Isara",
      closeRight: "Isara ang mga tab sa kanan",
      pageOptions: "Mga opsyon ng page",
    },
    siteInfo: {
      label: "Impormasyon ng site",
      secure: "Secure ang koneksyon",
      insecure: "Hindi secure ang koneksyon",
      back: "Bumalik",
      close: "Isara",
      security: "Seguridad",
      secureTitle: "Secure ang koneksyon",
      secureDescription:
        "Pribado ang iyong impormasyon, gaya ng mga password o numero ng card, kapag ipinadala ito sa site na ito.",
      insecureTitle: "Hindi secure ang koneksyon",
      insecureDescription:
        "Ang impormasyong ipinapadala mo sa site na ito, gaya ng mga password o numero ng card, ay maaaring makita ng iba.",
      learnMore: "Matuto pa",
      certificateValid: "Valid ang certificate",
    },
    tools: "Mga tool",
    recents: "Mga Kamakailan",
    website: "Website",
    recentNewer: "Mas bago",
    recentOlder: "Mas luma",
    openExternal: "Buksan sa browser ng system",
    copyLink: "Kopyahin ang link",
    linkCopied: "Nakopya ang link",
    download: "I-download",
    settings: "Mga Setting ng browser",
    cannotShowFile:
      "Hindi maipakita rito ang file na ito. I-download ito para mabuksan.",
    fileGone:
      "Hindi na available ang file na ito. Buksan itong muli mula sa chat.",
    settingsTitle: "Browser",
    linksTitle: "Mga link at file",
    addressBarTitle: "Address bar",
    openLinksSetting: "Buksan ang mga link sa",
    openLinksSettingDescription:
      "Kung saan bubukas ang mga link sa chat: ang iyong default na browser, o isang tab sa browser sa tabi ng pag-uusap.",
    linkDestinationDefault: "Default na browser",
    linkDestinationPanel: "Browser ng Unsloth",
    openFilesSetting: "Buksan ang mga file sa browser",
    openFilesSettingDescription:
      "Ang mga dokumentong ina-attach mo, gaya ng mga PDF, Office file, HTML at text, ay bubukas sa isang tab ng browser sa tabi ng pag-uusap sa halip na sa preview window.",
    searchEngineSetting: "Search engine",
    searchEngineSettingDescription:
      "Ginagamit kapag mga salita ang nailagay sa address bar sa halip na URL.",
    showFullUrlSetting: "Ipakita ang buong URL",
    showFullUrlSettingDescription:
      "Isama ang query at fragment sa address bar, hindi lang ang site at path.",
    bookmarksTitle: "Mga bookmark",
    bookmarksToolbarSetting: "Toolbar ng mga bookmark",
    bookmarksToolbarSettingDescription:
      "Kailan ipapakita ang iyong mga naka-save na page sa ilalim ng address bar.",
    bookmarkEditorSettingDescription:
      "Pangalanan ang bookmark at piliin ang lokasyon nito habang sine-save mo ito.",
    bookmarksSettingDescription:
      "Tingnan, i-edit at alisin ang mga page na naka-save sa browser.",
    browsingDataTitle: "Data ng pag-browse",
    historySetting: "History ng pag-browse",
    historySettingDescription:
      "Tingnan at pamahalaan ang mga page na binisita sa browser.",
    downloadsSetting: "History ng download",
    downloadsSettingDescription:
      "Tingnan at pamahalaan ang mga file na na-download mula sa browser.",
    pagesFromChat: "Magbukas ng chat para makita ito sa browser.",
    tabsTitle: "Mga tab",
    switchToNewTabsSetting: "Lumipat agad sa mga bagong tab",
    switchToNewTabsSettingDescription:
      "Kapag nagbukas ka ng link sa bagong tab gamit ang ⌘ o Ctrl, ipakita ito sa halip na buksan ito sa background.",
    switchToNewTabsNative:
      "Sa desktop app, palaging bumubukas sa harap ang mga bagong tab ng mga page.",
    defaultZoomSetting: "Default na zoom",
    defaultZoomSettingDescription:
      "Kung gaano kalaki lumalabas ang mga page kapag binuksan. Babalik dito ang zoom ng page kapag ni-reset ito.",
    newTabPageTitle: "Page ng bagong tab",
    showSuggestedSetting: "Ipakita ang mga iminumungkahing site",
    showSuggestedSettingDescription:
      "Mga site na madalas mong binibisita, bilang mga shortcut sa bagong tab.",
    showRecentsSetting: "Ipakita ang mga kamakailang page",
    showRecentsSettingDescription:
      "Mga page na binisita mo kamakailan, nakalista sa bagong tab.",
    hiddenSuggestionsSetting: "Mga inalis na mungkahi",
    hiddenSuggestionsSettingDescription:
      "Mga site na inalis mo sa Iminumungkahi: {count}. I-restore ang mga ito para lumabas ulit.",
    hiddenSuggestionsNone:
      "Maaaring i-restore dito ang mga site na inaalis mo sa Iminumungkahi.",
    restore: "I-restore",
    importBookmarksSetting: "I-import at i-export",
    importBookmarksSettingDescription:
      "Magdala ng mga bookmark mula sa Chrome, Safari, Firefox o Edge gamit ang HTML file na ine-export nila, o i-save ang sa iyo bilang ganoong file.",
    importBookmarks: "I-import…",
    exportBookmarks: "I-export",
    importBookmarksDoneOne: "Na-import ang {count} bookmark.",
    importBookmarksDoneMany: "Na-import ang {count} na bookmark.",
    importBookmarksNone: "Walang bagong bookmark sa file na iyon.",
    importBookmarksPartial:
      "Na-import ang {count}, pero hindi nagkasya ang {leftOut} pa: hanggang {max} bookmark lang ang itinatago ng Unsloth.",
    importBookmarksFailed:
      "Hindi mabasa ang file na iyon. Pumili ng bookmarks HTML file na na-export mula sa isang browser.",
    downloadsTitle: "Mga download",
    askWhereToSaveSetting: "Itanong kung saan ise-save ang bawat file",
    askWhereToSaveSettingDescription:
      "Magpakita ng save dialog para sa mga file na dina-download mo sa browser.",
    askWhereToSaveUnsupported:
      "Sine-save ng iyong web browser ang mga download sa sarili nitong downloads folder.",
    downloadLocationSetting: "Lokasyon",
    downloadLocationDefault: "Downloads folder ng system",
    downloadLocationChange: "Baguhin",
    downloadLocationReset: "I-reset",
    askBeforeDownloadingSetting: "Magtanong bago mag-download",
    askBeforeDownloadingSettingDescription:
      "Kumpirmahin ang mga file mula sa mga website bago ma-save ang mga ito.",
    downloadSitesSetting: "Mga pahintulot sa pag-download ng site",
    downloadSitesSettingDescription:
      "Mga site na pinili mong palaging payagan o i-block ang mga download.",
    downloadSitesNone:
      "Lagyan ng check ang Tandaan para sa susunod kapag tinanong, at lalabas dito ang site.",
    downloadSitesFor: "Mga download mula sa {host}",
    downloadSitesRemove: "Kalimutan ang {host}",
    downloadSiteAllow: "Palaging payagan",
    downloadSiteBlock: "Palaging i-block",
    downloadPrompt: {
      title: "I-download ang file na ito?",
      description: "Gustong i-save ng {host} ang {name} sa iyong computer.",
      dangerous:
        "Ang mga ganitong file ay maaaring magpatakbo ng mga program sa iyong computer. I-download lang ito kung pinagkakatiwalaan mo ang {host}.",
      remember: "Tandaan para sa susunod",
      cancel: "Kanselahin",
      download: "I-download",
      blocked: "Na-block ang isang download mula sa {host}.",
      ready: "Handa nang i-save ang {name}",
      save: "I-save",
    },
    saveDownloadHistorySetting: "Panatilihin ang history ng download",
    saveDownloadHistorySettingDescription:
      "Ilista ang mga file na dina-download mo mula sa browser sa ilalim ng History ng download.",
    saveHistorySetting: "I-save ang history ng pag-browse",
    saveHistorySettingDescription:
      "Tandaan ang mga page na binibisita mo, para sa History, Mga Kamakailan at Iminumungkahi.",
    historyRetentionSetting: "Panatilihin ang history nang",
    historyRetentionSettingDescription:
      "Awtomatikong dine-delete ang mga pagbisitang mas luma rito.",
    retention: {
      forever: "Hanggang i-clear",
      days90: "3 buwan",
      days30: "1 buwan",
      days7: "1 linggo",
      days1: "1 araw",
    },
    annotationsTitle: "Mga anotasyon",
    annotationScreenshotsSetting: "Mga screenshot ng anotasyon",
    annotationScreenshotsSettingDescription:
      "Ang screenshot ng page na may mga marka mo ay tumutulong sa mga vision model na maunawaan ang iyong mga komento, pero mas maraming context ang ginagamit nito.",
    annotationScreenshotsUnsupported:
      "Kailangan ang desktop app o isang browser na nakabatay sa Chromium, gaya ng Chrome o Edge.",
    annotationScreenshotsHint:
      "Sa isang web browser, hihilingin sa iyong i-share ang tab sa tuwing magpapadala ka ng mga anotasyon.",
    annotationScreenshotsAlways: "Palaging isama",
    annotationScreenshotsNever: "Huwag kailanman isama",
    clearDataSetting: "I-clear ang data ng pag-browse",
    clearDataSettingDescription:
      "I-clear ang history ng pag-browse, history ng download at mga naka-cache na page. Pinapanatili lang ng mga page ang cookies at data ng site hanggang ma-reload ang Unsloth.",
    manage: "Pamahalaan",
    fullView: {
      enter: "Pumasok sa full view",
      exit: "Lumabas sa full view",
      minimize: "I-minimize ang chat",
      showChat: "Ipakita ang chat",
      showConversation: "Ipakita ang pag-uusap",
      hideConversation: "Itago ang pag-uusap",
      newChat: "Bagong chat",
      options: "Mga opsyon ng chat",
      openChat: "Buksan sa full view",
      moveLeft: "Ilipat sa kaliwang pane",
      moveRight: "Ilipat sa kanang pane",
    },
    annotate: {
      countOne: "1 anotasyon",
      countMany: "{count} na anotasyon",
      cancel: "Kanselahin",
      send: "Ipadala",
      move: "Ilipat",
      placeholder: "Ilarawan ang pagbabago o magtanong",
      areaQuote: "Bahagi ng page",
      imageQuote: "Larawan",
      page: "Magtanong tungkol sa page na ito",
      pagePlaceholder: "Magtanong tungkol dito",
      save: "I-save ang komento",
      dictate: "Mag-dictate",
      stopDictating: "Ihinto ang pag-dictate",
      dictateFailed: "Hindi masimulan ang pag-dictate",
    },
    menu: {
      find: "Hanapin sa page",
      print: "I-print",
      printFailed: "Hindi ma-print ang page na ito",
      screenshot: "Kumuha ng screenshot",
      zoom: "Zoom",
      zoomIn: "Mag-zoom in",
      zoomOut: "Mag-zoom out",
      zoomReset: "I-reset ang zoom",
      deviceToolbar: "Ipakita ang toolbar ng device",
      clearData: "I-clear ang data ng pag-browse",
    },
    screenshot: {
      added: "Naidagdag ang screenshot sa chat",
      taken: "Nakuha ang screenshot",
      save: "I-save",
      failed: "Hindi makakuha ng screenshot",
      otherSurface:
        "I-share ang tab na ito para makakuha ng screenshot ng page",
    },
    device: {
      mobile: "Mobile",
      tablet: "Tablet",
      close: "Itago ang toolbar ng device",
    },
    downloads: {
      title: "Mga download",
      inProgressLabel: "Dina-download",
      inProgress: "Dina-download…",
      complete: "Tapos na ang download",
      failed: "Nabigo ang download",
      downloaded: "Na-download",
      missing: "Inilipat o na-delete ang file",
      open: "Buksan",
      openFailed: "Hindi mabuksan ang {name}.",
      copyLink: "Kopyahin ang link ng download",
      showAll: "Ipakita lahat",
      savePage: "I-save ang page bilang…",
    },
    file: {
      openIn: "Buksan sa",
      newChat: "Bagong chat",
      newBrowserTab: "Bagong tab ng browser",
      requestEdits: "Humiling ng mga pag-edit",
      requestEditsPrompt: "I-edit ang {name}: ",
      preview: "Preview",
      source: "Source",
      viewMode: "View ng file",
      runAgain: "Patakbuhin ulit",
      console: "Console",
      copy: "Kopyahin ang nilalaman",
      copied: "Nakopya",
      copyFailed: "Hindi makopya ang file.",
      copyPath: "Kopyahin ang path",
      pathCopied: "Nakopya ang path",
      openFailed: "Hindi mabuksan ang {name}.",
      wrap: "I-wrap ang mga linya",
      reactNeedsNode:
        "Kailangan ng mga React preview ang Node.js 20.19 o mas bago. I-install ito kung wala pa, pagkatapos ay patakbuhin ulit ang setup ng Unsloth.",
      reactFailed:
        "Hindi maihanda ang preview na ito. Patakbuhin ito ulit para subukang muli.",
    },
    video: {
      play: "I-play",
      pause: "I-pause",
      seek: "Mag-seek",
      volume: "Volume",
      mute: "I-mute",
      unmute: "I-unmute",
      more: "Higit pang opsyon sa pag-playback",
      speed: "Bilis ng pag-playback",
      normalSpeed: "Normal",
      loop: "I-loop",
      pictureInPicture: "Picture in picture",
      fullscreen: "Full screen",
      exitFullscreen: "Lumabas sa full screen",
      copyFrame: "Kopyahin ang frame",
      frameCopied: "Nakopya ang frame",
      copyFrameFailed: "Hindi makopya ang frame.",
      copyOptions: "Mga opsyon sa pagkopya",
      open: "Buksan",
      openOptions: "Mga opsyon sa pagbukas",
    },
    pages: {
      searchDownloads: "Maghanap sa history ng download",
      clearAll: "I-clear lahat",
      deleted: "Na-delete",
      showInFinder: "Ipakita sa Finder",
      showInExplorer: "Ipakita sa Explorer",
      showInFolder: "Ipakita sa folder",
      revealFailed: "Hindi maipakita ang {name} sa folder nito.",
      history: "History",
      downloads: "Mga download",
      bookmarks: "Mga bookmark",
      search: "Maghanap",
      today: "Ngayon",
      yesterday: "Kahapon",
      remove: "Alisin",
      clearHistory: "I-clear ang history",
      clearDownloads: "I-clear ang mga download",
      noHistory: "Lalabas dito ang mga page na binibisita mo sa browser.",
      showMore: "Magpakita pa",
      noDownloads:
        "Lalabas dito ang mga file na dina-download mo mula sa browser.",
      noBookmarks:
        "Lalabas dito ang mga page na bina-bookmark mo gamit ang star sa address bar.",
      noMatches: "Walang tumutugma sa iyong paghahanap.",
      fromChat: "Mula sa chat",
      searchHistory: "Maghanap sa history ng pag-browse",
      filter: "I-filter",
      allTime: "Lahat ng panahon",
      allTimeHistory: "History sa lahat ng panahon",
      lastWeek: "Huling 7 araw",
      lastMonth: "Huling 30 araw",
      customDates: "Pumili ng mga petsa…",
      clearFilter: "I-clear ang filter",
      openPage: "Buksan ang page",
      removeFromHistory: "Alisin sa history",
      selected: "{count} ang napili",
      cancel: "Kanselahin",
      select: "Piliin ang {title}",
      collapse: "I-collapse ang {date}",
      expand: "I-expand ang {date}",
      pageActions: "Mga aksyon para sa {title}",
    },
    bookmarks: {
      bookmarkPage: "I-bookmark ang page na ito",
      add: "Magdagdag ng bookmark",
      edit: "I-edit ang bookmark",
      name: "Pangalan",
      location: "Lokasyon",
      toolbar: "Toolbar ng mga bookmark",
      other: "Iba pang bookmark",
      showEditor: "Ipakita ang editor kapag nagse-save",
      cancel: "Kanselahin",
      remove: "Alisin ang bookmark",
      delete: "I-delete ang bookmark",
      save: "I-save",
      saved: "Na-save sa mga bookmark",
      removed: "Inalis ang bookmark",
      undo: "I-undo",
      manage: "Pamahalaan ang mga bookmark",
      more: "Higit pang bookmark",
      toolbarEmpty:
        "I-bookmark ang isang page gamit ang star sa address bar para idagdag ito rito.",
      toolbarAlways: "Palaging ipakita",
      toolbarNewTab: "Ipakita lang sa bagong tab",
      toolbarNever: "Huwag kailanman ipakita",
    },
    clearData: {
      title: "I-clear ang data ng pag-browse?",
      description:
        "Iki-clear nito ang iyong history ng pag-browse, history ng download, mga naka-cache na page, at ang cookies at data ng site, kaya masa-sign out ka sa mga site. Mananatili sa iyong computer ang mga na-download na file.",
      cancel: "Kanselahin",
      confirm: "I-clear ang data",
      done: "Na-clear ang data ng pag-browse",
    },
    error: {
      title: "Hindi ma-load ang page na ito",
      description:
        "Hindi tumugon ang {host}, tinanggihan nito ang koneksyon, o hinarangan nito ang request.",
      retry: "Subukan ulit",
      botCheckTitle:
        "Tinitingnan ng site na ito kung totoong browser ang gamit",
      botCheckDescription:
        "Hinihiling ng {host} sa mga bisita na patunayang tao sila, na hindi magagawa rito ng in-app na view. Buksan na lang ito sa iyong browser.",
      botCheckDescriptionWeb:
        "Hinihiling ng {host} sa mga bisita na patunayang tao sila, na hindi magagawa rito ng in-app na view. Buksan ito sa iyong browser, o sa {app}, na kayang magbukas nito.",
      desktopApp: "Unsloth Desktop",
      resubmit:
        "Galing sa isang form ang page na ito. Kapag ni-load ito ulit, ipapadala ulit ang form, na maaaring umulit sa ginawa nito.",
      upload:
        "Hindi gumagana ang pag-upload ng file sa view na ito. Buksan ang page sa browser ng iyong system para maipadala ito.",
      scriptNavigation:
        "Sinubukan ng page na ito na pumunta sa ibang address gamit ang script, na hindi masusundan ng view na ito. Buksan na lang ito sa browser ng iyong system.",
      needsBrowser:
        "Sinubukan ng page na ito na gumamit ng feature ng browser na hinaharangan ng view na ito, kaya maaaring hindi gumana ang ilang bahagi nito. Buksan na lang ito sa browser ng iyong system.",
    },
    native: {
      externalPrompt: "Gustong buksan ng {host} ang {url}",
      open: "Buksan",
      downloading: "Dina-download ang {name}",
      downloaded: "Na-save ang {name} sa Downloads",
      downloadFailed: "Hindi ma-download ang {name}",
      notMarked:
        "Na-save ang {name} pero hindi ito mamarkahan bilang na-download mula sa internet, kaya hindi ka babalaan ng iyong system bago ito buksan.",
      blocked:
        "Hindi mabubuksan ang address na ito sa browser panel. Mga pampublikong website lang ang binubuksan nito.",
      blockedLocal:
        "Hindi mabubuksan dito ang lokal na address na ito. I-type ito sa address bar para mabuksan.",
      studioAddress:
        "Ito ang sariling address ng Unsloth, kaya hindi ito mabubuksan sa browser panel.",
      clearDataSettingDescription:
        "I-clear ang history ng pag-browse, history ng download, mga naka-cache na page, at ang cookies at data ng site na itinatago ng mga page, kaya masa-sign out ka sa mga site.",
      clearDataDescription:
        "Iki-clear nito ang iyong history ng pag-browse, history ng download, mga naka-cache na page, at ang cookies at data ng site, kaya masa-sign out ka sa mga site. Mananatili sa iyong computer ang mga na-download na file.",
      clearDataFailed: "Hindi ma-clear ang data ng site. Subukan ulit.",
    },
  },
  managedEngines: {
    gpu: "Mga GPU",
    gpuHelp: "Piliin kung aling mga GPU ang gagamitin.",
    parallelism: "Mode ng multi-GPU",
    tensor: "Tensor parallel",
    pipeline: "Pipeline parallel",
    data: "Mga replica (data parallel)",
    pipelineHelp:
      "Hatiin ang mga layer ng model sa mga GPU. Dapat magkasya sa bawat GPU ang mga layer at cache nito.",
    dataHelp:
      "Ikalat ang mga request sa mga GPU. Dapat magkasya ang mga dense model sa bawat GPU; maaaring maghati ang mga MoE model sa mga expert.",
    tensorParallel: "Hatiin ang mga computation ng layer sa {count} na GPU.",
    inUse:
      "I-unload ang model bago i-repair, i-restore o alisin ang engine nito.",
    title: "Higit pang engine",
    description:
      "Mga karagdagang engine para sa mga GPU (Beta). Maaaring mas mabilis nilang patakbuhin ang mga model, lalo na sa batched inference. I-install dito, pagkatapos ay piliin sa mga setting ng pagpapatakbo ng model.",
    picker: "Engine ng inference",
    precision: "Precision",
    precisionAuto: "Default ng model",
    precisionHelp:
      "Pinapanatili ng Default ng model ang naka-store na precision. Kino-convert ng ibang opsyon ang mga unquantized na weight kapag nilo-load.",
    kvCache: "KV cache",
    kvCacheFp8: "FP8 (kalahati ng memory)",
    kvCacheHelp:
      "Iniimbak ng FP8 ang context cache sa 8 bit, kaya halos doble ang context na kasya sa parehong memory. Kung walang calibrated scale sa checkpoint, maaaring bahagyang hindi gaanong tumpak ang mga sagot.",
    gpuMemory: "Limitasyon ng memory ng GPU",
    gpuMemoryAuto: "Awtomatiko",
    gpuMemoryHelp:
      "Ang pinakamaraming bahagi ng bawat napiling GPU na maaaring gamitin ng engine. Ginagamit ng Awtomatiko ang bakante; babaan ito para mag-iwan ng puwang sa ibang program.",
    seed: "Seed",
    seedPlaceholder: "Default ng engine",
    seedHelp:
      "Itinatakda ang random seed ng engine. Maaari pa ring baguhin ng sabay-sabay na mga chat ang mga sampled na sagot.",
    default: "Default",
    installRequired: "kailangang i-install",
    ownerRequired:
      "Hilingin sa may-ari ng Unsloth na i-install ang engine na ito.",
    cancelInstall: "Kanselahin ang pag-install",
    installed: "Naka-install ang {version}",
    repair: "I-repair",
    update: "I-update ang engine",
    install: "I-install ang engine",
    installing:
      "Ini-install ang engine. Tingnan ang Mga download para sa progreso.",
    remove: "Alisin ang engine",
    rollback: "I-restore ang nakaraang installation",
    installTitle: "I-install ang {engine}",
    installAndLoad: "I-install at i-load",
    confirm:
      "I-install ang {engine} {version}? Maaari itong mag-download ng ilang GB.",
    confirmSized:
      "I-install ang {engine} {version}? Mga {size} ang ida-download.",
    wslSetup:
      "Sa Windows, tumatakbo ang {engine} sa isang pribadong WSL2 environment na sine-set up ng Unsloth. Asahan ang isang kahilingan ng pahintulot ng administrator at posibleng pag-restart.",
    wslReady:
      "Sa Windows, tumatakbo ang {engine} sa loob ng pribadong WSL2 environment ng Unsloth.",
    wslRestart:
      "I-restart ang Windows para matapos ang pag-on ng WSL2, pagkatapos ay i-click ulit ang I-install.",
    background: "Ini-install sa background.",
    failed:
      "Nabigo ang pag-install ng engine. Subukan ulit o gamitin ang default na engine.",
    details: "Mga teknikal na detalye",
    cancelled: "Kinansela ang pag-install. Puwede mong subukan ulit.",
    scope:
      "Text at image chat sa mga NVIDIA GPU. Maaaring mas matagal ang unang pag-load.",
  },
  sandboxSetup: {
    levelTitle: "Hindi available ang OS sandbox",
    levelDescription:
      "I-install ito para magamit ang Mataas, o manatili sa Mababa (software sandboxing).",
    useLow: "Gamitin ang Mababang sandbox",
    checking: "Tinitingnan ang sandbox sa computer na ito…",
    install: "I-install ang sandbox",
    windowsSetup: "I-set up ang Windows sandbox",
    copyCommand: "Kopyahin ang command",
    copied: "Nakopya ang command",
    copyFailed: "Hindi makopya ang command",
    copiedRunIt:
      "Nakopya ang command. Patakbuhin ito sa isang terminal sa computer na nagpapatakbo ng Unsloth, pagkatapos ay lumipat sa Mataas.",
    ownerOnly:
      "Ang may-ari lang ang makakapag-install nito, mula sa computer na nagpapatakbo ng Unsloth.",
    windowsNote:
      "Kailangan ng hakbang ng administrator pagkatapos ng bawat pag-restart. Preview ang MXC, hindi isang security boundary.",
    running:
      "Sine-set up. Ilagay ang password o aprubahan ang kahilingan ng pahintulot ng administrator sa computer na nagpapatakbo ng Unsloth…",
    succeeded: "Handa na ang OS sandbox. Naka-on ang Mataas na sandbox.",
    stillUnavailable:
      "Tapos na ang setup, pero hindi pa rin pumapasa sa check nito ang OS sandbox.",
    declined:
      "Tinanggihan ang kahilingan ng password o pahintulot ng administrator.",
    failed: "Nabigo ang setup.",
    startError: "Hindi nasimulan ang setup ng sandbox",
    checkFailed: "Hindi matingnan ang sandbox sa computer na ito.",
    retry: "Subukan ulit",
    keepsRunning:
      "Hindi nito ihihinto ang setup kapag isinara mo ito. Lalabas ang resulta nito sa Mga Setting > Seguridad.",
  },
  composerSettings: {
    title: "Composer",
    plainText: "Plain text na composer",
    plainTextDescription:
      "Panatilihin ang code, Markdown, at mga link bilang literal na text. I-off para magpakita ng naka-format na preview.",
    showContext: "Paggamit ng context window",
    sendShortcut: "Shortcut sa pagpapadala",
    sendMultiline: "{mod} + Enter para sa mga prompt na maraming linya",
    sendAlways: "Palaging {mod} + Enter",
    sendEnterDescription:
      "Nagpapadala ang Enter. Nagdadagdag ng bagong linya ang Shift + Enter.",
    sendMultilineDescription:
      "Nagpapadala ang Enter ng prompt na isang linya. Kapag higit sa isang linya na ito, nagdadagdag ng bagong linya ang Enter at nagpapadala ang {mod} + Enter.",
    sendAlwaysDescription:
      "Nagpapadala ang {mod} + Enter. Nagdadagdag ng bagong linya ang Enter.",
    followUp: "Gawi ng follow-up",
    followUpDescription:
      "Ang nangyayari kapag nagpadala ka habang may sagot. Pindutin ang {shortcut} para gawin ang kabaligtaran nang isang beses.",
    followUpMultilineShortcut:
      "{shortcut} ({multiline} sa prompt na maraming linya)",
    queue: "Ipila",
    steer: "Ibaling",
    steerDescription:
      "Inihihinto ng Ibaling ang kasalukuyang sagot at susunod na ipapadala ang iyong follow-up.",
    settings: "Mga Setting ng composer",
    preview: "Naka-format na preview",
  },
  promptQueue: {
    loading: "Nilo-load ang mga nakapilang prompt",
    listLabel: "Mga nakapilang prompt",
    regionLabel: "Pila ng prompt, {current} sa {total}",
    itemLabel: "Nakapilang prompt {position} sa {total}: {prompt}",
    reorderInstructions:
      "I-drag ang handle para ayusin ang pagkakasunod-sunod. Kapag naka-focus ang handle, gamitin ang Up o Down para lumipat nang isang posisyon, o ang Home o End para lumipat sa unahan o dulo.",
    dragTooltip: "I-drag para ayusin ang pagkakasunod-sunod",
    reorderLabel:
      "Ayusin ang pagkakasunod-sunod ng nakapilang prompt {position} sa {total}",
    paused: "Naka-pause",
    steer: "Ibaling",
    steerTooltip:
      "Putulin ang sagot at ipadala ang prompt na ito bilang susunod",
    steerLabel: "Ibaling gamit ang nakapilang prompt {position}",
    removeTooltip: "Alisin sa pila",
    removeLabel: "Alisin ang nakapilang prompt {position}",
    moreTooltip: "Higit pang opsyon",
    moreLabel: "Higit pang opsyon para sa nakapilang prompt {position}",
    editItem: "I-edit ang mensahe",
    copyItem: "Kopyahin ang mensahe",
    editLabel: "I-edit ang nakapilang prompt {position}",
    cancel: "Kanselahin",
    save: "I-save",
    turnOffQueueing: "I-off ang pagpila",
    turnOnQueueing: "I-on ang pagpila",
    resume: "Ituloy ang pila",
    queueButton: "Ipila ang mensahe",
    steerButton: "Ibaling ang sagot",
    sendTooltip: "Ipadala ang mensahe ({shortcut})",
    sendLabel: "Ipadala ang mensahe",
    followUpTooltip: "{action} ({send}) · {opposite} para sa kabaligtaran",
    announceUpdated: "Na-update ang nakapilang prompt.",
    announceEditFailed:
      "Hindi na maaaring i-edit ang prompt na ito dahil nagbago ang pila.",
    announceRemoved: "Inalis ang prompt sa pila.",
    announceSteered: "Ibabaling ng prompt na ito ang susunod na sagot.",
    announceSteerFailed:
      "Hindi naibaling ng prompt na ito ang sagot. Tingnan ang pila at subukan ulit.",
    announceCopied: "Nakopya ang prompt.",
    announceCopyFailed: "Hindi makopya ang prompt na ito. Subukan ulit.",
    announceQueueingOn:
      "Ipipila ang mga bagong follow-up pagkatapos ng kasalukuyang sagot.",
    announceQueueingOff:
      "Ibabaling ng mga bagong follow-up ang kasalukuyang sagot.",
    announceMoved: "Inilipat ang prompt sa posisyon {position} sa {total}.",
    announceMoveFailed:
      "Nagbago ang pila bago mailipat ang prompt na ito. Subukan ulit.",
    announceDragReset:
      "Nagbago ang pila. Mag-drag ulit para ayusin ang pagkakasunod-sunod ng natitirang mga prompt.",
    editingHint: "Ine-edit ang mensahe",
    queueingOffHint: "Pumuputol at susunod na tatakbo ang mga bagong mensahe.",
    queueingOnHint: "Naghihintay ng kanilang turno ang mga bagong mensahe.",
    queueingHintShared: "Pinapanatili ang pila.",
  },
  rewind: {
    title: "I-rewind",
    confirmTitle: "I-rewind sa mensaheng ito?",
    restoreLabel: "I-restore",
    pickDescription: "Pumili ng mensaheng pagre-rewind-an.",
    pickEmpty: "Wala pang mensaheng mapagre-rewind-an.",
    attachmentOnly: "Mga attachment lang",
    modeBoth: "Pag-uusap at mga file",
    modeConversation: "Pag-uusap lang",
    modeFiles: "Mga file lang",
    checkFailed: "Hindi matingnan ang mga pagbabago sa file: {error}",
    summary:
      "Mga file na ire-restore: {restore}. Aalisin: {remove}. Gagawin ulit: {recreate}.",
    conflicts: "Nagbago na mula noon, lalaktawan",
    untracked: "Hindi maaaring i-restore",
    scopeNote:
      "Kasama sa mga file ang code. Hindi binabago ang mga program, package at file sa labas ng folder ng chat na ito.",
    back: "Bumalik",
    cancel: "Kanselahin",
    confirm: "I-rewind",
    waitForResponse: "Hintaying matapos ang kasalukuyang sagot",
    restoredOne: "Na-restore ang 1 file",
    restoredMany: "Na-restore ang {count} na file",
    nothingRestored: "Walang file na kailangang i-restore",
    skippedCount: "{count} ang nilaktawan",
    filesNotRestored:
      "Hindi maaaring i-restore ang {count} na binagong file at mananatili ang mga ito sa kasalukuyang estado.",
    restoreFailed: "Hindi ma-restore ang mga file",
    undo: "I-undo",
    undone: "Na-undo ang pag-restore ng file",
    undoFailed: "Hindi ma-undo ang pag-restore ng file",
  },
  chatMenu: {
    more: "Mga opsyon ng chat",
    copy: "Kopyahin",
    compact: "I-compact ang pag-uusap",
    compactQueued: "Hiniling ang pag-compact para sa susunod na mensahe",
    compactEmpty: "Wala pang mako-compact",
    compactUnsupported:
      "Kailangan ng pag-compact ang GGUF o MLX model na may suporta sa tool",
  },
  halloween: {
    navLabel: "Trick or treat…",
    title: "Trick or Treat?!",
    description:
      "Nagtitimpla kami ng ilang katatakutan para sa Unsloth, narito hanggang matapos ang Oktubre.",
    chooseTheme: "Pumili ng nakakatakot na tema",
    spookyGreetings: "Nakakakilabot na pagbati",
    spookyGreetingsDescription:
      "Ginugulat ng sloth ang mga bagong chat sa pagbati nito.",
    decorations: "Mga gagamba at paniki",
    decorationsDescription: "Mga sapot, at paminsan-minsang gagamba o paniki.",
    surprise: "Trick or treat!",
    surpriseDescription: "Hayaang pumili ang sloth ng tema para sa iyo.",
    themesTitle: "Mga nakakatakot na tema",
    back: "Bumalik",
    footer: "Babalik sa normal ang Unsloth sa hatinggabi ng Nobyembre 1.",
    removeNow: "Alisin ngayon",
    hideFromSidebar: "Itago sa sidebar",
    greeting: {
      pumpkins: "Naukit na ang mga kalabasa. Ano ang susunod?",
      brewing: "Simulan na nating magtimpla",
      trickOrTreat: "Trick or treat?",
      manor: "Maligayang pagdating sa mansyon",
      manorName: "Maligayang pagdating sa mansyon, {name}",
      happyHalloween: "Maligayang Halloween",
      happyHalloweenName: "Maligayang Halloween, {name}",
      spirits: "Nakikinig ang mga espiritu…",
      mysteries: "Anong mga hiwaga ang ating tutuklasin ngayong gabi?",
      conjure: "Ano ang maaari kong ipatawag para sa iyo?",
    },
  },
  welcome: {
    goodToSeeYou: "Masaya akong makita ka",
    goodToSeeYouName: "Masaya akong makita ka, {name}",
    readyWhenYouAre: "Handa na ako kapag handa ka na",
    onYourMind: "Ano ang nasa isip mo ngayon?",
    howCanIHelp: "Paano kita matutulungan?",
    goodMorning: "Magandang umaga",
    goodMorningName: "Magandang umaga, {name}",
    goodEvening: "Magandang gabi",
    goodEveningName: "Magandang gabi, {name}",
    tonight: "Ano ang balak mo ngayong gabi?",
    nightOwl: "Puyat mode?",
    lateNightIdeas: "Mga ideya sa hatinggabi?",
    upLate: "Gising pa dahil may ideya?",
    nightShift: "Nagsisimula na ang night shift",
    nightShiftName: "Nagsisimula na ang night shift, {name}",
    temporaryChat: "Pansamantalang chat",
    temporaryChatDescription:
      "Hindi lalabas sa iyong history ang chat na ito at hindi ito sine-save. Mawawala ito kapag umalis ka.",
  },
  picker: {
    unifiedTightFit: "Sakto lang",
    unifiedTightFitHint:
      "Maaaring mag-load ito nang may mas maikling context o tumakbo nang mas mabagal sa Mac na ito.",
    onDevice: "Sa Device",
    huggingFace: "Hugging Face",
    retry: "Subukan ulit",
    loadMore: "Mag-load pa",
    offlineTitle: "Offline ka",
    offlineBody:
      "Lumipat sa Device para gamitin ang naka-cache o lokal na {noun}.",
    offlineSwitchDevice: "Device",
    searchAriaLabel: "Maghanap ng {noun}",
    modelSourceAriaLabel: "Pinagmulan ng model",
    hubSectionAriaLabel: "Seksyon ng hub",
    modelDropped: "Hindi na iniaalok",
    modelDroppedByProvider: "{provider} · hindi na iniaalok",
    modelDisabled: "Hindi naka-enable",
    modelDisabledByProvider: "{provider} · hindi naka-enable",
    multipleMatches:
      "Maraming tumutugmang {noun}. Pumili ng isa mula sa listahan.",
    rateLimitedTitle: "Naabot ang rate limit ng Hugging Face",
    rateLimitedBody:
      "Maghintay sandali, pagkatapos ay subukan ulit maghanap ng {noun}.",
    modelScope: "ModelScope",
    useModelScope: "Gamitin ang ModelScope",
    useModelScopeHint:
      "Sa ModelScope na lang maghanap at mag-download. Puwede kang bumalik sa Mga Setting.",
    useModelScopeFailed: "Hindi makalipat sa ModelScope.",
    updateToken: "I-update ang token",
    hfToken: {
      label: "HF token",
      saved: "Naka-save",
      add: "Hindi nakatakda",
      savedAriaLabel: "Naka-save ang token ng Hugging Face",
      addAriaLabel: "Itakda ang token ng Hugging Face",
      savedHint:
        "Naka-save ang token. Sinusuri ang access kapag ginamit mo ito.",
      addHint:
        "Magtakda ng token para ma-access ang mga pribado at gated na repository.",
    },
  },
  common: {
    duplicate: "I-duplicate",
    cancel: "Kanselahin",
    close: "Isara",
    delete: "I-delete",
    done: "Tapos na",
    error: "Error",
    export: "I-export",
    help: "Tulong",
    todayAt: "Ngayon, {time}",
    yesterdayAt: "Kahapon, {time}",
    loading: "Nilo-load...",
    new: "Bago",
    rename: "Palitan ang pangalan",
    save: "I-save",
    saving: "Sine-save...",
    search: "Maghanap",
    shutdown: "I-shut down",
  },
  shell: {
    helpMenu: {
      documentation: "Dokumentasyon",
      keyboardShortcuts: "Mga Keyboard Shortcut",
      whatsNew: "Ano ang Bago",
      troubleshooting: "Pag-troubleshoot",
      systemStatus: "Status ng System",
      sendFeedback: "Magpadala ng Feedback",
      about: "Tungkol sa Unsloth",
    },
    find: {
      label: "Hanapin sa page",
      previous: "Nakaraang tugma",
      next: "Susunod na tugma",
      close: "Isara ang paghahanap",
      truncated: "Masyadong mahaba ang page na ito para mahanap nang buo.",
      searchChat: "Maghanap sa chat",
      searchBrowser: "Maghanap sa page ng browser",
    },
    zoom: {
      label: "Zoom",
      zoomOut: "Mag-zoom out",
      zoomIn: "Mag-zoom in",
      reset: "I-reset",
      announce: "Zoom {percent}%",
    },
    beta: "BETA",
    brand: "unsloth",
    product: "Unsloth",
    accountMenu: "Menu ng account ni {name}",
    updateAvailable: "May available na update",
    resize: {
      collapse: "I-click para i-collapse",
      expand: "I-click para i-expand",
      drag: "I-drag para baguhin ang laki",
    },
    aria: {
      home: "Home ng Unsloth",
      closeSidebar: "Isara ang sidebar",
      openSidebar: "Buksan ang sidebar",
      resizeSidebar: "Baguhin ang laki o i-collapse ang sidebar",
      resizeRunSettings:
        "Baguhin ang laki o isara ang mga setting ng pagpapatakbo",
      openRunSettings: "Buksan ang mga setting ng pagpapatakbo",
      chatOptions: "Mga opsyon ng chat",
      runOptions: "Mga opsyon ng pagpapatakbo",
    },
    navigation: {
      newChat: "Bagong chat",
      returnToChat: "Bumalik sa Chat",
      returnToChats: "Bumalik sa {count} na Chat",
      chatGenerating: "Gumagawa ng sagot",
      chatAwaitingApproval: "Naghihintay ng pag-apruba",
      chatUnread: "Hindi pa nababasa",
      compare: "Ihambing",
      search: "Maghanap",
      projects: "Mga Proyekto",
      library: "Library",
      hub: "Model hub",
      train: "I-train",
      recipes: "Mga Recipe",
      images: "Mga Larawan",
      video: "Video",
      audio: "Audio",
      tasks: "Naka-iskedyul",
      trainChecking:
        "Tinitingnan kung sinusuportahan ng makinang ito ang training...",
      videoChecking:
        "Tinitingnan kung sinusuportahan ng makinang ito ang video...",
      more: "Higit pa",
      customizeSidebar: "I-customize ang sidebar",
      newBadge: "Bago",
      export: "I-export",
      recents: "Mga Kamakailan",
      noChatsYet: "Wala pang chat",
      noChats: "Walang chat",
      allProjectsPinned: "Naka-pin ang lahat ng proyekto",
      allProjectsFiled: "Naka-pin o nasa mga seksyon ang lahat ng proyekto",
      noProjects: "Walang proyekto",
      showMore: "Magpakita pa",
      showLess: "Magpakita ng mas kaunti",
      settings: "Mga Setting",
      api: "API",
      lightMode: "Maliwanag na Mode",
      darkMode: "Madilim na Mode",
      guidedTour: "Gabay na Tour",
      help: "Tulong",
      logOut: "Mag-sign out",
      logOutNeedsPassword:
        "Magtakda ng password sa Mga Setting > Remote at LAN bago mag-sign out, para makapag-sign in ka ulit.",
      logOutFailed: "Hindi ligtas na makapag-sign out. Subukan ulit maya-maya.",
      shutdown: "I-shut down",
    },
    commandPalette: {
      placeholder: "Mag-type ng command o maghanap...",
      noResults: "Walang nakitang resulta.",
      navigation: "Navigation",
      actions: "Mga aksyon",
      chat: "Chat",
      searchChats: "Maghanap ng mga chat...",
    },
    search: {
      placeholder: "Maghanap",
      tabs: {
        all: "Lahat",
        chats: "Mga Chat",
        projects: "Mga Proyekto",
        files: "Mga File",
        models: "Mga Model",
      },
      recents: "Mga Kamakailan",
      actions: "Mga aksyon",
      newChat: "Bagong chat",
      newTemporaryChat: "Bagong pansamantalang chat",
      fineTune: "Mag-fine-tune ng model",
      generateImage: "Gumawa ng larawan",
      generateVideo: "Gumawa ng video",
      untitledChat: "Chat na walang pamagat",
      compare: "Ihambing",
      loading: "Nilo-load…",
      empty: {
        all: "Wala pang mahahanap.",
        chats: "Wala pang chat.",
        projects: "Wala pang proyekto.",
        files: "Wala pang file sa iyong Library.",
        models: "Wala pang na-download na model.",
      },
      noMatches: "Walang resulta.",
      when: {
        today: "Ngayon",
        pastWeek: "Nakaraang linggo",
        pastMonth: "Nakaraang buwan",
        older: "Mas luma",
      },
      footer: {
        close: "Isara",
        changeType: "Baguhin ang uri",
        open: "Buksan",
      },
    },
    notFound: {
      title: "Hindi nahanap ang page",
      description: "Hindi umiiral ang {path}.",
      backToChat: "Bumalik sa chat",
    },
    selection: {
      pinProjects: "I-pin ang mga proyekto",
      unpinProjects: "I-unpin ang mga proyekto",
      deleteProjects: "I-delete ang mga proyekto",
      deleteProjectsTitle: "I-delete ang mga proyekto",
      deleteProjectsDescription:
        "I-delete ang {count} na proyekto? Permanenteng made-delete ang mga chat ng mga ito.",
      deleteProjectsFilesDescription:
        "Aalisin sa disk ang workspace folder ng bawat proyekto.",
      countSelected: "{count} ang napili",
      pinChats: "I-pin ang mga chat",
      unpinChats: "I-unpin ang mga chat",
      archiveChats: "I-archive ang mga chat",
      markUnread: "Markahan bilang hindi pa nababasa",
      markRead: "Markahan bilang nabasa na",
      deleteChats: "I-delete ang mga chat",
      deleteTitle: "I-delete ang mga chat",
      deleteDescription:
        "I-delete ang {count} na chat? Hindi na ito maaaring i-undo.",
      deleteFilesDescription:
        "Aalisin sa disk ang sariling sandbox folder ng bawat chat. Mananatili sa workspace ng proyekto ang mga file na isinulat nila sa loob ng isang proyekto.",
      deleteFilesLabel: "I-delete ang mga file at sandbox folder",
      deleteChatFilesDescription:
        "Aalisin sa disk ang sariling sandbox folder ng chat na ito. Mananatili sa workspace ng proyekto ang mga file na isinulat nito sa loob ng isang proyekto.",
    },
    organize: {
      sidebarHeading: "Ayusin ang sidebar",
      byProject: "Ayon sa proyekto",
      inOneList: "Sa iisang listahan",
      sortChatsBy: "Pagsunud-sunurin ang mga chat ayon sa",
      projectsHome: "Home ng mga proyekto",
      lastUpdated: "Huling na-update",
      manualOrder: "Manual na pagkakasunod-sunod",
      switchedToManual:
        "Manual na nakaayos: i-drag ang mga row para ayusin ang pagkakasunod-sunod",
      organizeChats: "Ayusin ang mga chat",
      organizeProjects: "Ayusin ang mga proyekto",
      sortPinnedChats: "Pagsunud-sunurin ang mga naka-pin na chat",
      show: "Ipakita",
      newSection: "Bagong seksyon",
    },
    sections: {
      createTitle: "Bagong seksyon",
      createDescription:
        "Pagpangkatin ang mga chat at proyekto ayon sa gusto mo",
      namePlaceholder: "Pangalan ng seksyon",
      create: "Gumawa ng seksyon",
      renameTitle: "Palitan ang pangalan ng seksyon",
      renameDescription: "Bigyan ng bagong pangalan ang seksyong ito",
      edit: "I-edit",
      remove: "Alisin ang seksyon",
      markAllRead: "Markahan lahat bilang nabasa na",
      moveTo: "Ilipat sa",
      section: "Seksyon",
      sectionsHeading: "Mga seksyon",
      removeFromProject: "Alisin sa proyekto",
      newSection: "Bagong seksyon",
      removeFromSection: "Alisin sa seksyon",
      removeFrom: "Alisin sa {name}",
      empty: "I-drag dito ang mga chat o proyekto",
      sectionOptions: "Mga opsyon ng seksyon",
      newChatInSection: "Bagong chat sa {name}",
      deleted: "Na-delete ang seksyong “{name}”",
      undo: "I-undo",
    },
    dialog: {
      deleteChat: {
        title: "I-delete ang chat",
        description:
          'Sigurado ka bang gusto mong i-delete ang chat na ito na "{name}"?',
      },
      deleteRun: {
        title: "I-delete ang training run",
        description:
          'Sigurado ka bang gusto mong i-delete ang run na ito na "{name}"?',
      },
      renameChat: {
        title: "Palitan ang pangalan ng chat",
        placeholder: "Pamagat ng chat",
      },
      renameRun: {
        title: "Palitan ang pangalan ng run",
        placeholder: "Pangalan ng run",
      },
    },
    toast: {
      cannotDeleteRunningRun:
        "Hindi maaaring i-delete ang tumatakbong training run",
      failedToDeleteChat: "Nabigo ang pag-delete ng chat",
      failedToDeleteRun: "Nabigo ang pag-delete ng run",
      failedToRenameChat: "Nabigo ang pagpapalit ng pangalan ng chat",
      failedToRenameRun: "Nabigo ang pagpapalit ng pangalan ng run",
    },
  },
  settings: {
    accounts: {
      signsInAs: "Nagsa-sign in bilang {username}",
      signsInAsOwner: "Nagsa-sign in bilang {alias} o {username}",
      signInName: "Ang iyong pangalan sa pag-sign in",
      signInNameDescription:
        "Ang pangalang tina-type mo para mag-sign in kapag may higit sa isang account ang installation na ito. Gumagana rin palagi ang unsloth. Iwanang walang laman para unsloth lang ang gamitin.",
      signInNameSave: "I-save ang pangalan",
      title: "Mga Account",
      description:
        "Gumawa ng mga pribadong Unsloth account. Nagsa-sign in ang mga bagong user gamit ang isang beses na setup code at pumipili sila ng password.",
      username: "Username",
      create: "Gumawa ng account",
      createDescription:
        "Mag-share ng setup code para makapili sila ng sarili nilang password.",
      actionsFor: "Pamahalaan si {username}",
      actions: "Mga aksyon",
      search: "Maghanap ng mga account",
      noResults: "Walang tumutugmang account",
      created: "Ginawa",
      status: "Status",
      loginHint:
        "Mag-sign in bilang {username} gamit ang code na ito bilang password, pagkatapos ay pumili ng bagong password.",
      privateAccount: "Pribadong account",
      modelAccess: "Access sa model",
      modelAccessTitle: "Access sa model para kay {username}",
      modelAccessDescription:
        "Piliin kung aling mga model sa Hugging Face ang maaaring makita, i-download, i-load at gamitin ng account na ito sa pamamagitan ng mga API key nito.",
      modelAccessLabel: "Mga pinapayagang model",
      modelAccessHint:
        "Isang model bawat linya, halimbawa unsloth/Qwen3-4B-GGUF, o unsloth/* para sa buong organisasyon. Iwanang walang laman para payagan ang bawat model.",
      modelAccessSave: "I-save",
      modelAccessLimited: "Limitadong access sa model",
      empty: "Wala pang ibang account",
      setupCode: "Setup code",
      setupFor: "Setup code para kay {username}",
      shownOnce:
        "Kopyahin ang code na ito ngayon at i-share ito sa may-ari ng account. Dito lang ito ipinapakita at isang beses lang itong magagamit sa loob ng 60 minuto.",
      expires: "Mag-e-expire {expiry}",
      copy: "Kopyahin ang setup code",
      copied: "Nakopya",
      copyFailed: "Hindi makopya. Piliin at kopyahin ang setup code sa itaas.",
      dismiss: "Tapos na",
      owner: "May-ari ng installation",
      active: "Naka-enable",
      inactive: "Naka-disable",
      regenerate: "Gumawa ulit ng setup code",
      resetTitle: "I-reset ang password ni {username}?",
      resetDescription:
        "Kapag gumawa ulit ng setup code, papalitan ang password ni {username}, masa-sign out siya at babawiin ang kanyang mga API key. Ibigay sa kanya ang bagong code para makapili siya ulit ng password.",
      deactivate: "I-disable",
      reactivate: "I-enable",
      delete: "I-delete ang account",
      deleteTitle: "I-delete si {username}?",
      deleteDescription:
        "Babawiin nito ang mga session ni {username} at kakanselahin ang kanyang ginagawa. Ire-retire ang kanyang mga chat, setting, credential, upload, dataset, training run, output, export, gallery, sandbox, proyekto at pansamantalang file. Pinapalitan ang pangalan ng mga directory at itinatabi, hindi kailanman dine-delete. Kapag ginawa ulit ang username na ito, magsisimula ang isang bagong account na wala ang alinman sa data na ito.",
      cancel: "Kanselahin",
      retry: "I-refresh ang mga account",
      failed: "Nabigo ang request sa account.",
    },
    title: "Mga Setting",
    dialog: {
      title: "Mga Setting",
      description: "Pamahalaan ang iyong mga kagustuhan sa Unsloth.",
      closeAriaLabel: "Isara ang Mga Setting",
      searchPlaceholder: "Maghanap sa Mga Setting…",
      searchNoResults: "Walang nahanap na setting.",
      panelFailed: "Hindi ma-load ang seksyong ito.",
      panelReload: "I-reload",
    },
    library: {
      layoutSection: "Layout",
      cardSize: "Laki ng card",
      cardSizeDescription: "Ilang card ang kasya sa isang row.",
      small: "Maliit",
      medium: "Katamtaman",
      large: "Malaki",
      imageLayout: "Mga image card",
      imageLayoutDescription:
        "Panatilihin ang hugis ng bawat larawan, o i-crop ang mga ito bilang pantay na parisukat.",
      masonry: "Orihinal na hugis",
      square: "Parisukat",
      showCardDates: "Ipakita ang mga petsa sa mga card",
      showCardDatesDescription:
        "Ipakita kung kailan huling nagbago ang bawat file.",
      browsingSection: "Pag-browse",
      startTab: "Buksan sa",
      startTabDescription: "Ang tab na unang bubukas sa Library.",
      suggested: "Iminumungkahi",
      all: "Lahat",
      sort: "Pagsunud-sunurin ayon sa",
      sortDescription:
        "Pagkakasunod-sunod ng mga file at folder. Palaging ipinapakita ng Iminumungkahi ang pinakabago.",
      recent: "Pinakabago",
      oldest: "Pinakaluma",
      name: "Pangalan",
      suggestedLimit: "Mga iminumungkahing item",
      suggestedLimitDescription:
        "Ilang kamakailang file ang ipinapakita ng Iminumungkahi.",
      tabAuto: "Kapag may mga file",
      tabAlways: "Palagi",
      contentSection: "Nilalaman",
      contentDescription:
        "Piliin kung ano ang ipinapakita ng Library. Hindi dine-delete ang mga nakatagong file.",
      showChatAttachments: "Mga attachment sa chat",
      showChatAttachmentsDescription: "Mga file na in-attach mo sa mga chat.",
      showChatToolFiles: "Mga file mula sa mga tool ng chat",
      showChatToolFilesDescription:
        "Mga file na ginawa ng code at mga tool habang nasa chat.",
      showGeneratedMedia: "Mga ginawang media",
      showGeneratedMediaDescription:
        "Mga larawan, video at audio mula sa mga page na Mga Larawan, Video at Audio.",
      showFineTunesDescription:
        "Mga model na tinrain o in-export mo sa Unsloth.",
      deletingSection: "Pag-delete",
      confirmDelete: "Kumpirmahin bago mag-delete",
      confirmDeleteDescription:
        "Magtanong bago mag-delete ng mga file at folder mula sa Library.",
      reset: "I-reset ang mga setting ng Library",
      resetDescription: "Ibalik sa default ang bawat opsyon sa itaas.",
      resetButton: "I-reset",
      resetDone: "Na-reset ang mga setting ng Library",
      storageSection: "Storage",
      downloadsSection: "Mga download ng larawan",
      imageInfoStamp: "Stamp ng impormasyon",
      imageInfoStampDescription:
        "Ipakita ang model, petsa at seed sa ibaba ng mga larawang dina-download mo mula sa page na Mga Larawan. Naka-save pa rin sa loob ng file ang mga setting ng pag-generate alinman ang piliin.",
      storageDescription:
        "Magbukas ng kategorya para makita ang mga file nito mula sa pinakamalaki, at i-delete ang hindi mo na kailangan.",
      storageUsed: "{size} ang ginagamit ng Library",
      storageDisk: "{free} ang bakante sa {total} sa disk na ito",
      storageEmpty: "Wala pang naka-store sa Library.",
      storageHidden: "{size} ang nakatago sa mga setting ng Nilalaman",
      storageHiddenLabel: "Nakatago",
      storageRefresh: "I-refresh",
      storageOpen: "Buksan",
      storageError: "Hindi mabasa ang storage ng Library.",
      locationsSection: "Mga file sa disk",
      locationsDescription:
        "Kung saan itinatago ng Unsloth ang bawat uri ng file. Ang mga folder sa Library ay mga label sa loob ng Unsloth, hindi mga folder sa disk. Nananatili sa kinalalagyan nila ang mga fine-tune at export, dahil tinutukoy sila ng training at mga chat ayon sa path.",
      locationUploads: "Mga upload sa Library",
      locationExports: "Mga export",
      revealFailed: "Hindi mabuksan ang file manager",
      locationReset: "I-reset",
      locationMoveTitle: "Ilipat ang {name} sa bagong folder",
      locationMoveDescription:
        "Pumili ng kahit anong folder. Kung may mga file na ito, gagawa ang Unsloth ng sarili nitong folder sa loob nito. Ililipat din ang lahat ng narito na.",
      locationMoveAction: "Ilipat dito",
      locationMoving: "Inililipat ang {name}…",
      locationMoved: "Nailipat ang {name}",
      locationMoveFailed: "Hindi mailipat ang {name}",
      locationFree: "{free} ang bakante sa drive na ito",
      locationUnavailable:
        "Hindi available. Ikonekta ulit ang drive nito, o i-reset ang folder.",
      locationResetUnavailableTitle: "I-reset ang {name}?",
      locationResetUnavailableDescription:
        "Hindi nakakonekta ang drive nito. Mananatili sa drive na iyon ang mga file sa {path} at hindi lalabas sa Library hangga't hindi mo ibinabalik ang mga ito.",
      locationLeftBehind: "Mananatili sa drive na iyon ang mga file sa {path}.",
      dataStorage: "Storage ng Library",
      manageStorage: "Pamahalaan ang storage",
      categoryImages: "Mga Larawan",
      categoryVideos: "Mga Video",
      categoryFineTunes: "Mga fine-tune",
      itemCountOne: "1 item",
      itemCount: "{count} na item",
      tabsSection: "Mga tab",
      tabsDescription:
        "Piliin kung aling mga tab ang ipinapakita ng Library. Lalabas pa rin ang tab na binuksan mo mula sa isang link.",
      tabHidden: "Nakatago",
      lastVisited: "Huling binisita",
    },
    tabs: {
      sandbox: "Seguridad",
      accounts: "Mga Account",
      general: "Pangkalahatan",
      profile: "Profile",
      appearance: "Hitsura",
      resources: "System",
      chat: "Chat",
      voice: "Boses",
      connections: "Mga Koneksyon",
      data: "Data",
      apiKeys: "API",
      remoteLan: "Remote at LAN",
      agents: "Mga Agent",
      keyboardShortcuts: "Mga Shortcut",
      debugging: "Mga Log",
      about: "Tungkol sa",
    },
    keyboardShortcuts: {
      title: "Mga keyboard shortcut",
      intro:
        "Para baguhin ang isang shortcut, piliin ang kombinasyon ng key, pagkatapos ay i-type ang mga bagong key",
      searchPlaceholder: "Maghanap ng mga shortcut…",
      keystrokePlaceholder: "Pindutin ang shortcut para maghanap",
      searchByKeystrokes: "Maghanap ayon sa mga keystroke",
      searchByName: "Maghanap ayon sa pangalan",
      noResults: "Walang shortcut na tumutugma sa paghahanap na iyon.",
      unassigned: "wala",
      recording: "Pindutin ang sunod-sunod na key",
      needsModifier:
        "Magdagdag ng ⌘, Ctrl o Alt. Lalamunin ng iisang key ang pagta-type.",
      conflict: "Ginagamit din ng ibang shortcut",
      conflictShadowed:
        "May ibang shortcut na may ganitong kombinasyon at iyon ang tatakbo",
      edit: "Baguhin ang shortcut",
      reset: "Ibalik ang default",
      resetAll: "Ibalik ang mga default",
      browserReserved:
        "Maaaring ireserba ng iyong browser ang kombinasyong ito para sa sarili nito. Gumagana ito sa desktop app.",
      metaUnsupported: "Hindi magagamit ang Windows key sa mga shortcut.",
      altGraphUnsupported:
        "Hindi magagamit bilang shortcut ang mga kombinasyong may AltGr.",
      desktopOnly: "Sa desktop app lang",
      groups: {
        composer: "Composer",
        chat: "Chat",
        navigation: "Navigation",
        app: "App",
      },
      actions: {
        findInPage: {
          label: "Hanapin sa page",
          description: "Hanapin ang text sa page na ito",
        },
        newBrowserTab: {
          label: "Bagong tab ng browser",
          description: "Magbukas ng bagong tab sa browser sa tabi ng chat",
        },
        toggleBrowserFullView: {
          label: "I-toggle ang full view ng browser",
          description:
            "Ipakita ang browser sa buong window, habang nakalutang sa ibabaw nito ang chat",
        },
        openSettings: {
          label: "Buksan ang Mga Setting",
          description: "Buksan ang dialog ng Mga Setting",
        },
        openKeyboardShortcuts: {
          label: "Mga keyboard shortcut",
          description: "Buksan ang listahang ito ng mga shortcut",
        },
        openCommandPalette: {
          label: "Command palette",
          description: "Buksan ang command palette",
        },
        searchChats: {
          label: "Maghanap ng mga chat",
          description: "Buksan ang dialog ng paghahanap ng chat",
        },
        openMcpServers: {
          label: "Mga MCP server",
          description: "I-configure ang mga MCP server para sa chat na ito",
        },
        logOut: {
          label: "Mag-sign out",
          description: "Mag-sign out sa Unsloth",
        },
        approveToolRequest: {
          label: "Aprubahan ang request",
          description: "Payagan ang tool call na naghihintay ng pag-apruba",
        },
        declineToolRequest: {
          label: "Tanggihan ang request",
          description: "Tanggihan ang tool call na naghihintay ng pag-apruba",
        },
        newChat: {
          label: "Bagong chat",
          description: "Magsimula ng bagong chat",
        },
        newTemporaryChat: {
          label: "Bagong pansamantalang chat",
          description: "Magsimula ng chat na hindi sine-save sa history",
        },
        newStandaloneChat: {
          label: "Bagong standalone na chat",
          description: "Magsimula ng bagong chat sa labas ng anumang proyekto",
        },
        archiveChat: {
          label: "I-archive ang chat",
          description: "I-archive ang mga napiling chat, o ang kasalukuyan",
        },
        markChatUnread: {
          label: "Markahan bilang hindi pa nababasa",
          description:
            "Markahan bilang hindi pa nababasa ang mga napiling chat, o ang kasalukuyan",
        },
        togglePinChat: {
          label: "I-toggle ang pin",
          description:
            "I-pin o i-unpin ang mga napiling chat, o ang kasalukuyan",
        },
        selectAllChats: {
          label: "Piliin lahat ng chat",
          description: "Piliin ang bawat chat sa sidebar",
        },
        clearChatSelection: {
          label: "I-clear ang pinili",
          description:
            "Alisin sa pagkakapili ang mga napiling chat. Ganoon din ang Escape",
        },
        deleteSelectedChats: {
          label: "I-delete ang mga napiling chat",
          description: "I-delete ang bawat napiling chat",
        },
        nextRecentlyViewedChat: {
          label: "Susunod na kamakailang tiningnang chat",
          description: "Umikot pasulong sa mga kamakailang tiningnang chat",
        },
        previousRecentlyViewedChat: {
          label: "Nakaraang kamakailang tiningnang chat",
          description: "Umikot pabalik sa mga kamakailang tiningnang chat",
        },
        nextChat: {
          label: "Susunod na chat",
          description: "Lumipat sa susunod na chat sa sidebar",
        },
        previousChat: {
          label: "Nakaraang chat",
          description: "Lumipat sa nakaraang chat sa sidebar",
        },
        nextChatNeedingAttention: {
          label: "Susunod na chat na nangangailangan ng pansin",
          description:
            "Lumipat sa susunod na chat na gumagawa ng sagot, nakapila o hindi pa nababasa",
        },
        clearAllUnreads: {
          label: "I-clear lahat ng hindi pa nababasa",
          description: "Markahan bilang nabasa na ang bawat chat",
        },
        goToRecentChat1: {
          label: "Pumunta sa kamakailang chat 1",
          description: "Buksan ang unang chat sa ilalim ng Mga Kamakailan",
        },
        goToRecentChat2: {
          label: "Pumunta sa kamakailang chat 2",
          description: "Buksan ang ikalawang chat sa ilalim ng Mga Kamakailan",
        },
        goToRecentChat3: {
          label: "Pumunta sa kamakailang chat 3",
          description: "Buksan ang ikatlong chat sa ilalim ng Mga Kamakailan",
        },
        goToRecentChat4: {
          label: "Pumunta sa kamakailang chat 4",
          description: "Buksan ang ikaapat na chat sa ilalim ng Mga Kamakailan",
        },
        goToRecentChat5: {
          label: "Pumunta sa kamakailang chat 5",
          description: "Buksan ang ikalimang chat sa ilalim ng Mga Kamakailan",
        },
        goToRecentChat6: {
          label: "Pumunta sa kamakailang chat 6",
          description: "Buksan ang ikaanim na chat sa ilalim ng Mga Kamakailan",
        },
        switchToChat: {
          label: "Lumipat sa Chat",
          description: "Pumunta sa workspace ng chat",
        },
        switchToProjects: {
          label: "Lumipat sa Mga Proyekto",
          description: "Pumunta sa workspace ng mga proyekto",
        },
        switchToHub: {
          label: "Lumipat sa Model hub",
          description: "Pumunta sa model hub",
        },
        switchToTrain: {
          label: "Lumipat sa I-train",
          description: "Pumunta sa workspace ng training",
        },
        switchToRecipes: {
          label: "Lumipat sa Mga Recipe",
          description: "Pumunta sa Data Recipes",
        },
        switchToImages: {
          label: "Lumipat sa Mga Larawan",
          description: "Pumunta sa workspace ng mga larawan",
        },
        switchToVideo: {
          label: "Lumipat sa Video",
          description: "Pumunta sa workspace ng video",
        },
        switchToAudio: {
          label: "Lumipat sa Audio",
          description: "Pumunta sa workspace ng audio",
        },
        switchToExport: {
          label: "Lumipat sa I-export",
          description: "Pumunta sa workspace ng pag-export",
        },
        toggleSidebar: {
          label: "I-toggle ang sidebar",
          description: "Ipakita o itago ang sidebar",
        },
        toggleApiMonitor: {
          label: "I-toggle ang aktibidad ng API",
          description: "Ipakita o itago ang monitor ng aktibidad ng API",
        },
        openModelPicker: {
          label: "Buksan ang picker ng model",
          description: "Piliin ang model para sa chat na ito",
        },
        openProjectPicker: {
          label: "Buksan ang picker ng proyekto",
          description: "Lumipat sa ibang proyekto mula sa header ng chat",
        },
        startDictation: {
          label: "Pag-dictate",
          description: "Simulan o ihinto ang pag-dictate sa composer",
        },
        attachFiles: {
          label: "Mag-attach ng mga larawan at file",
          description: "Magdagdag ng attachment sa composer",
        },
        openComposerMenu: {
          label: "Buksan ang + menu",
          description:
            "Buksan ang + menu ng composer para sa mga tool at attachment",
        },
        sendMessage: {
          label: "Ipadala ang mensahe",
          description: "Ipadala ang nasa composer",
        },
        queueMessage: {
          label: "Ipila ang mensahe",
          description: "Ipadala ang draft sa dulo ng pila ng prompt",
        },
        steerMessage: {
          label: "Ibaling ang sagot",
          description:
            "Ihinto ang kasalukuyang sagot at ipadala ang draft bilang susunod",
        },
        stopGeneration: {
          label: "Ihinto ang pag-generate",
          description: "Ihinto ang sagot na isinusulat",
        },
        cycleReasoningEffort: {
          label: "Umikot sa reasoning effort",
          description: "Isa-isahin ang mga antas ng reasoning effort",
        },
        increaseReasoningEffort: {
          label: "Taasan ang reasoning effort",
          description: "Itaas nang isang antas ang reasoning effort",
        },
        decreaseReasoningEffort: {
          label: "Babaan ang reasoning effort",
          description: "Ibaba nang isang antas ang reasoning effort",
        },
        renameChat: {
          label: "Palitan ang pangalan ng chat",
          description: "Palitan ang pangalan ng kasalukuyang chat",
        },
        forkChat: {
          label: "I-fork ang chat",
          description:
            "Gumawa ng bagong chat na sanga mula sa pinakabagong mensahe",
        },
        copyChatAsMarkdown: {
          label: "Kopyahin bilang Markdown",
          description: "Kopyahin ang buong chat sa clipboard bilang Markdown",
        },
        copySessionId: {
          label: "Kopyahin ang session id",
          description: "Kopyahin ang sandbox session id ng chat na ito",
        },
      },
    },
    sandbox: {
      title: "Seguridad",
      description:
        "Kung tatakbo ang mga tool call ng Python at Terminal sa loob ng OS sandbox sa computer na ito.",
      docs: "Docs",
      docsLabel: "Buksan ang docs ng sandbox",
      toolsSection: "Ang computer na ito",
      refresh: "I-refresh",
      python: "Python",
      terminal: "Terminal",
      osIsolation: "OS sandbox ({backend})",
      softwareSafeguards: "Mga software safeguard",
      noReason:
        "Wala pang OS sandbox na gumagana rito, kaya sinusuri na lang ng Unsloth ang bawat call.",
      runsInCmd:
        "Tumatakbo sa cmd.exe, dahil hindi makapagsimula ang Git Bash sa Windows sandbox.",
      loadError: "Hindi na-load ang status ng sandbox",
      saveError: "Hindi na-save ang setting ng sandbox",
      prepareError: "Hindi naihanda ang PC na ito",
      unsupported:
        "Hindi pa nag-uulat ng status ng sandbox ang Unsloth server na ito. I-update ang Unsloth para mapamahalaan ito rito.",
      windowsSection: "Windows sandbox (preview)",
      windowsDescription:
        "Ang MXC sandbox ng Microsoft. Ginagamit nito ang container na built-in sa Windows, o isang fallback sa mga bersyon ng Windows na wala nito.",
      builtinLabel: "Built-in na container",
      builtinDescription:
        "Tumatakbo ang Windows sandbox sa built-in na container ng Windows, nang walang hakbang ng administrator.",
      builtinInUse: "Ginagamit",
      optInLabel: "Gamitin ang fallback sandbox",
      optInDescription:
        "Para sa mga bersyon ng Windows na walang built-in na container. Sa ganitong paraan, ang mga file lang sa folder ng chat ang mababago ng Python at Terminal.",
      optInNotNeeded:
        "Hindi kailangan sa PC na ito, dahil gumagana ang built-in na container.",
      disclosure:
        'Kailangan ng fallback ang isang beses na hakbang ng administrator, ang "Ihanda ang PC na ito", na kailangang ulitin pagkatapos ng bawat pag-restart. Habang tumatakbo ang isang tool, binibigyan ito ng Unsloth ng pansamantalang access sa mga folder na kailangan nito. Preview pa rin ang MXC, hindi isang security boundary.',
      lockedDacl: "Itinakda ng UNSLOTH_MXC_ALLOW_DACL_FALLBACK",
      grantsLabel: "Mas mabilis na pagsisimula ng tool",
      grantsDescription:
        "Pinapanatili ang read-only na access sa sariling runtime folder ng Unsloth sa pagitan ng mga call. Kapag naka-off, ilang segundong mas mabagal magsimula ang bawat call.",
      lockedGrants: "Itinakda ng UNSLOTH_MXC_PERSISTENT_READ_GRANTS",
      memoryLabel: "Limitasyon ng memory",
      memoryDescription:
        "Pinakamaraming memory na maaaring ireserba ng isang sandboxed na Python o Terminal call. Ilalapat mula sa susunod na call. Ang default ay {defaultSize} GB.",
      memoryLocked: "Itinakda ng UNSLOTH_STUDIO_SANDBOX_AS_GB",
      memoryInvalid: "Maglagay ng buong numero mula {min} hanggang {max}.",
      restored: "Inalis ang access mula sa {count} folder.",
      hostPrepLabel: "Setup ng administrator",
      prepPrepared: "Tapos na",
      prepNeeds: "Kailangan",
      prepNeedsAgain: "Kailangan ulit pagkatapos ng huling pag-restart",
      prepUnknown: "Hindi alam",
      prepareButton: "Ihanda ang PC na ito",
      preparing:
        "Hinihintay ang pag-apruba sa kahilingan ng pahintulot ng administrator…",
      prepSucceeded: "Handa na ang PC na ito.",
      prepDeclined:
        "Tinanggihan ang kahilingan ng pahintulot ng administrator.",
      prepFailed: "Nabigo ang paghahanda.",
      setupLabel: "OS sandbox",
      setupCommandHint:
        "Patakbuhin ito sa computer na nagpapatakbo ng Unsloth:",
      macosBuiltIn:
        "May built-in na sandbox ang macOS, kaya walang kailangang i-install.",
      setupKeywords:
        "install setup bubblewrap bwrap seatbelt mxc sudo pkexec apparmor uac i-install",
      setupSucceeded: "Handa na ang OS sandbox.",
      runtimeMissing:
        "Hindi pa naka-install. Hindi kailangan ng pahintulot ng administrator para i-install ito.",
      runtimeLabel: "Runtime ng Windows sandbox",
      installRuntime: "I-install ang runtime",
      installingRuntime: "Ini-install…",
      installRuntimeFailed: "Nabigo ang pag-install ng runtime.",
      installRuntimeError: "Hindi masimulan ang pag-install ng runtime.",
      unsupportedArch:
        "Tumatakbo lang ang Windows sandbox (MXC) sa x64 at ARM64 na Windows, kaya hindi ito available sa PC na ito.",
      builtinNeedsUpdate:
        "Kailangan ng built-in na container ng Windows ang build {required} o mas bago, at {current} ang tumatakbo sa PC na ito. I-install ang pinakabagong update ng Windows para magamit ito nang walang hakbang ng administrator, o i-on ang fallback sa ibaba.",
      unsupportedBuild:
        "Kailangan ng Windows sandbox (MXC) ang Windows 11 24H2 (build 26100) o mas bago, kaya hindi ito available sa PC na ito.",
      permissionLabel: "Mga pahintulot ng tool",
      levelRowLabel: "Antas ng sandbox",
      levelPickerTitle: "Paano dapat i-sandbox ang code?",
      levelOff: "Naka-disable",
      levelOffShort: "Pinapatay ng Buong access ang sandbox",
      levelHighDetail:
        "Tumatakbo ang Python at Terminal sa loob ng OS sandbox (bubblewrap, Seatbelt o MXC) bukod pa sa software sandboxing. Kung walang gumaganang OS sandbox, magtatanong muna ang mga mapanganib na call.",
      levelLowDetail:
        "Tumatakbo ang Python at Terminal gamit lang ang software sandboxing, sa labas ng OS sandbox. Magtatanong muna ang mga mapanganib na call maliban kung pinili mo ang Buong access.",
      levelLabel: "Sandbox",
      levelHigh: "Mataas",
      levelLow: "Mababa",
      levelHighShort: "Nagdaragdag ng OS isolation sa Python at Terminal",
      levelLowShort: "Software lang. Magtatanong muna ang mapanganib na code",
      levelFullAccessNote:
        "Pinapatay ng Buong access ang sandbox. Ilalapat ulit ang setting na ito kapag pumili ka ng ibang antas.",
      learnMore: "Matuto pa",
      managedNote:
        "Ang may-ari ng Unsloth server na ito ang nagse-set up ng OS sandbox.",
    },
    debugging: {
      description: "Basahin ang mga log ng Unsloth nang hindi umaalis sa app.",
      logSection: "Log file",
      source: "Log file",
      sourceHint:
        "May sariling log ang mga model runner, kaya kadalasang doon naipapaliwanag ang nabigong pag-load o pag-generate sa halip na sa log ng server.",
      path: "Lokasyon",
      pathCopy: "Kopyahin ang path",
      currentSession: "Kasalukuyan",
      statusLive: "Live",
      statusPaused: "Naka-pause",
      statusStale: "Luma na",
      filterPlaceholder: "I-filter ang mga linya",
      lineCount: "{count} linya",
      filteredLineCount: "{shown} sa {total} linya",
      wrapLines: "I-wrap ang mga linya",
      jumpToLatest: "Pinakabago",
      resizeLog: "I-drag para baguhin ang laki ng log",
      noMatches: "Walang linyang tumutugma sa filter.",
      mode: "Mode",
      modeLive: "Live",
      modeInterval: "Bawat 3 segundo",
      modeManual: "Manual",
      refreshNow: "I-refresh ngayon",
      privacyNote:
        "Nakatago ang mga credential sa view na ito. Hindi nakatago ang mga ito sa file sa disk.",
      copyVisible: "Kopyahin ang nakikitang log",
      empty: "Wala pang nala-log.",
      disabled: "Naka-off ang pag-log sa file (UNSLOTH_STUDIO_NO_FILE_LOG=1).",
      missing: "Walang nakitang log file.",
      unreadable: "Hindi mabasa ang log file.",
      viewLogs: "Tingnan ang mga log",
      timeout:
        "Nag-time out ang request para sa log. Maaaring hindi maabot ang server.",
      droppedNotice:
        "May ilang linyang nilaktawan: mas mabilis isinulat ang log kaysa sa kayang basahin.",
      morePending:
        "May mga linya pang binabasa; darating ang mga ito sa susunod na pag-refresh.",
      staleSession:
        "Naka-off ang pag-log sa file, kaya mas naunang session ito at hindi mag-a-update.",
      downloadAllLogs: "I-download ang lahat ng log (.zip)",
      downloadingAllLogs: "Inihahanda ang mga log...",
      exportMaskedNote:
        "Nakatago ang mga credential sa mga na-export na file. Ang napakalaking log ay pinapanatili lang ang mga pinakabagong linya nito, at maaaring tuluyang hindi maisama ang ilang log; tingnan ang EXPORT_WARNINGS.txt sa archive.",
      downloadedTo: "Na-save sa {path}",
      downloadedToBrowser: "Nagsimula na ang pag-download.",
      showInFolder: "Ipakita sa folder",
      openLogsFolder: "Buksan ang folder ng mga log",
      openLogsFolderFailed: "Hindi mabuksan ang folder ng mga log.",
      exportFailed: "Hindi ma-download ang mga log.",
      exportTooOld:
        "Masyadong luma ang tumatakbong Unsloth backend para mag-export ng mga log. I-update ang backend na iyon at i-restart.",
      exportForbidden:
        "Kailangan ng naka-sign in na session ng Unsloth para i-download ang lahat ng log. Hindi sapat ang API key.",
      keywords:
        "debug debugging log logs error errors crash traceback stack trace troubleshoot diagnostics failed mga log nabigo pag-debug",
    },
    voice: {
      title: "Boses",
      description:
        "Mikropono, dictation, speech-to-text at pagbasa nang malakas",
      dictation: {
        audioUploadTitle: "Mag-dictate gamit ang recording",
        audioUploadDescription:
          "Kailangan ng iyong browser ng secure na koneksyon (HTTPS) para direktang mag-record sa Unsloth. Sa HTTP na koneksyong ito, gamitin ang recorder ng iyong phone o pumili na lang ng naka-save na recording. Ita-transcribe ng iyong Unsloth server ang audio at idaragdag ang text sa iyong mensahe.",
        audioUploadChooseFile: "Pumili ng recording",
        audioUploadRecord: "Mag-record ng audio",
        audioUploadIphoneHint:
          "Sa iPhone, mag-save ng recording mula sa Voice Memos o ibang recorder app, pagkatapos ay piliin ito rito.",
        audioUploadServerModelNote:
          "Ginagamit ng recording na ito ang local na model na ipinapakita sa itaas sa Unsloth server, kahit ibang engine ang ginagamit ng karaniwang dictation.",
        audioUploadAutomatic: "Awtomatiko",
        audioUploadModelReady: "Handa na sa Unsloth server na ito",
        audioUploadRetryTitle: "Hindi ma-transcribe ang {file}",
        audioUploadRetry: "Subukan ulit ang pag-transcribe",
        audioUploadChooseModel:
          "Pumili ng local na speech-recognition model sa mga setting ng Boses.",
        audioUploadEmpty: "Walang laman ang napiling recording.",
        audioUploadTooLarge: "Pumili ng recording na mas maliit sa {size}.",
        audioUploadVideoUnsupported: "Pumili ng audio recording, hindi video.",
        audioUploadNoSpeech:
          "Walang narinig na pananalita ang model sa recording na iyon.",
        audioUploadFailed: "Hindi ma-transcribe ang recording.",
        audioUploadCancel: "Kanselahin ang pag-transcribe",
        audioUploadTranscribing: "Tina-transcribe ang audio…",
        sectionTitle: "Dictation",
        engineLabel: "Engine ng dictation",
        engineBrowser: "Browser",
        engineBrowserDescription:
          "Nagta-transcribe ng audio gamit ang speech service ng iyong browser. Piliin ang 'Local na transcription' para gumamit ng STT model.",
        engineModel: "Local na transcription",
        engineModelDescription:
          "Nagpapatakbo ng speech-to-text (STT) model nang local at gumagana offline. I-download, i-load, pagkatapos ay ia-unload kapag matagal na hindi ginagamit.",
        engineCustom: "Custom na endpoint",
        engineCustomDescription:
          "Ipinapadala ang na-record na audio sa isang OpenAI-compatible na STT server mula sa iyong Mga Koneksyon.",
        connectionLabel: "Koneksyon",
        connectionDescription:
          "Magdagdag ng OpenAI-compatible na server at opsyonal na API key sa Mga Koneksyon.",
        connectionPlaceholder: "Pumili ng koneksyon",
        connectionEmpty: "Walang available na koneksyon",
        customModelLabel: "Model",
        customModelDescription:
          "Pangalan ng model na ipinapadala sa /v1/audio/transcriptions.",
        sttModelLabel: "Model ng speech recognition",
        sttModelDescription:
          "Pumili o maghanap ng STT model na patatakbuhin nang local.",
        sttQuantLabel: "Quantization",
        sttQuantDescription:
          "Pumili o maghanap ng STT model na patatakbuhin nang local, pagkatapos ay ang quantization nito.",
        sttDeviceLabel: "I-load sa",
        sttDeviceAuto: "GPU kapag available",
        sttDeviceCpu: "CPU RAM",
        sttDeviceAutoDescription:
          "Gamitin ang GPU kapag mayroon, at ang CPU kung wala.",
        sttDeviceCpuDescription:
          "Panatilihin ang model sa RAM ng system. Mas mabagal ang pag-transcribe, pero walang memory ng GPU na nagagamit.",
        sttModelSearchPlaceholder: "Maghanap ng kahit anong model sa HF",
        sttModelSearching: "Naghahanap sa Hugging Face…",
        sttModelValidating: "Sinusuri ang compatibility sa Whisper…",
        sttModelNoResults: "Walang nakitang Whisper model",
        sttModelInvalid:
          "Hindi magagamit ang repository na ito para sa dictation",
        sttModelFailed: "Hindi ma-load ang STT model",
        sttModelUnsupported:
          "Hindi sinusuportahan ang pag-record sa browser na ito",
        sttChecking: "Sinusuri…",
        sttOnDemand: "Na-download",
        sttLoadingModel: "Nilo-load ang model…",
        sttReady: "Na-load sa {device}",
        sttLoaded: "Na-load",
        sttUnavailable:
          "Hindi naka-install sa server na ito. Patakbuhin ang `unsloth studio update` para i-enable ang local na dictation.",
        sttRetry: "Subukan ulit",
        sttDownloadChecking: "Sinusuri ang status ng pag-download…",
        sttNotDownloaded: "Hindi pa na-download",
        sttDownloadStatusFailed: "Hindi masuri ang status ng pag-download",
        sttDownload: "I-download",
        sttDownloadConfirmTitle: "I-download ang {model}?",
        sttDownloadConfirmBody:
          "Ganap na offline tumatakbo ang local na dictation, pero kailangan muna nito ang speech model na {model}. Mga {size}, kukunin nang isang beses papunta sa iyong Hugging Face cache.",
        sttDownloadConfirmBodyUnsized:
          "Ganap na offline tumatakbo ang local na dictation, pero kailangan muna nito ang speech model na {model}. Kukunin ito nang isang beses papunta sa iyong Hugging Face cache.",
        sttOpenVoiceSettings: "Buksan ang mga setting ng Boses",
        sttDownloadStarted: "Dina-download ang {model}",
        sttDownloading: "Dina-download… {progress}%",
        sttCancelDownload: "Kanselahin",
        sttCancellingDownload: "Kinakansela…",
        sttCancelDownloadFailed: "Hindi makansela ang pag-download",
        sttDownloadComplete: "Na-download na ang model ng speech recognition",
        sttModelReady: "Handa na ang {model} para sa dictation",
        sttRecommended: "Inirerekomenda",
        sttDownloadFailed: "Hindi ma-download ang model ng speech recognition",
        sttLoad: "I-load",
        sttUnload: "I-unload",
        sttUnloading: "Ina-unload…",
        microphoneLabel: "Mikropono",
        microphoneFallbackName: "Mikropono {index}",
        microphoneDescription: "Ginagamit para sa dictation",
        microphoneFallbackHint:
          "Ginagamit para sa dictation. Babalik sa default ng system kung hindi magamit ng speech engine ng browser ang device na ito",
        microphoneGrantDescription:
          "Payagan ang access sa mic para maipakita ang mga pangalan ng device",
        allowMicrophone: "Payagan ang mikropono",
        micAccessBlocked:
          "Na-block ang access sa mikropono. Payagan ang access sa mikropono para sa page na ito ng Unsloth, pagkatapos ay subukan ulit.",
        micAccessBlockedDesktop:
          "Na-block ang access sa mikropono. Subukan ulit at piliin ang Payagan, o i-enable ang mikropono sa mga setting ng privacy ng iyong system.",
        micAccessUnsupported:
          "Hindi sinusuportahan ang access sa mikropono sa browser o context na ito.",
        systemDefault: "Default ng system",
        savedMicDisconnected: "Naka-save na mikropono (hindi nakakonekta)",
        languageLabel: "Wika ng dictation",
        languageDescription: "Wikang kikilalanin",
        languageAuto: "Auto (wika ng browser)",
        languageAutoDetect: "Auto (tukuyin ang wika)",
      },
      dictionary: {
        sectionTitle: "Diksyunaryo ng dictation",
        sectionDescription:
          "Itakda kung paano binabaybay ng dictation ang mga partikular na salita o parirala",
        manageLabel: "Mga custom na baybay",
        manage: "Pamahalaan",
        backToVoice: "Bumalik sa Boses",
        addEntry: "Magdagdag ng entry",
        newEntryAria: "Bagong entry sa diksyunaryo",
        entryPlaceholder: "Juan dela Cruz",
        entryAria: "Entry {index} sa diksyunaryo",
        removeEntryAria: "Alisin ang entry {index} sa diksyunaryo",
      },
      recents: {
        sectionTitle: "History ng dictation",
        sectionDescription:
          "Naka-save dito ang bawat dictation para mabawi mo ang text",
        manageLabel: "History ng dictation",
        manage: "Pamahalaan",
        pageDescription:
          "Naka-save ang bawat dictation. Tingnan, kopyahin, o i-delete ang mga ito, o buksan ang chat kung saan ginamit ang isang dictation.",
        searchPlaceholder: "Maghanap ng mga dictation",
        sortLabel: "Ayusin ang mga dictation",
        sortNewest: "Pinakabago",
        sortOldest: "Pinakaluma",
        sortAlpha: "A hanggang Z",
        noMatches: "Walang dictation na tumutugma sa iyong paghahanap",
        detailTitle: "Naka-save na dictation",
        backToVoice: "Bumalik sa Boses",
        backToRecents: "Bumalik sa mga kamakailang dictation",
        view: "Tingnan ang buong dictation",
        empty: "Wala pang dictation",
        dictationColumn: "Dictation",
        dateColumn: "Petsa ng paggawa",
        copy: "Kopyahin ang dictation",
        copied: "Nakopya sa clipboard",
        copyFailed: "Hindi makopya sa clipboard",
        delete: "I-delete ang dictation",
        deleteTitle: "I-delete ang dictation",
        deleteDescription:
          "I-delete ang naka-save na dictation na ito? Hindi na ito maaaring i-undo.",
        deleteLinkedDescription:
          "I-delete ang naka-save na dictation na ito? Maaari mo ring i-delete ang chat kung saan ito ginamit. Hindi na ito maaaring i-undo.",
        deleteWithChat: "I-delete ang chat at dictation",
        deleteWithChatFailed: "Hindi ma-delete ang chat",
        clear: "I-clear ang history",
        clearTitle: "I-clear ang history ng dictation",
        clearDescription:
          "I-delete ang lahat ng naka-save na dictation? Hindi na ito maaaring i-undo.",
        clearConfirm: "I-clear lahat",
        showMore: "Magpakita pa ({count})",
        openChat: "Buksan ang chat",
      },
      readAloud: {
        sectionTitle: "Basahin nang malakas",
        buttonLabel: "Button na Basahin nang malakas",
        buttonDescription: "Ipakita sa mga sagot ng assistant",
        autoLabel: "Awtomatikong basahin ang mga sagot",
        autoDescription:
          "Basahin nang malakas ang bawat sagot sa sandaling matapos ito",
        engineLabel: "TTS engine",
        engineSystemDescription: "Mga built-in na boses ng device",
        engineStudioDescription:
          "Nagsasalita gamit ang model na na-load sa Audio › Text to Speech",
        engineSystem: "Mga boses ng system",
        engineStudio: "Model ng Unsloth",
        engineCustom: "Custom na endpoint",
        engineCustomDescription:
          "Isang OpenAI-compatible na TTS server mula sa iyong mga koneksyon (hal. Kokoro)",
        connectionLabel: "Koneksyon",
        connectionDescription:
          "Magdagdag ng OpenAI-compatible na server sa tab na Mga Koneksyon",
        connectionPlaceholder: "Pumili ng koneksyon",
        customModelLabel: "Model",
        customVoiceDescription:
          "Pangalan ng boses na inaasahan ng endpoint; alloy ang default",
        modelLabel: "TTS model",
        modelDescription:
          "Walang naka-load. Mag-load ng isa sa Audio › Text to Speech.",
        modelLoaded: "Naka-load ang {model}",
        keepBothLabel: "Panatilihing naka-load ang chat model",
        keepBothDescription:
          "Naka-off ang Panatilihing naka-load ang maraming model, kaya ang pag-load ng speech model ay mag-a-unload ng chat model.",
        keepBothAction: "Panatilihing naka-load ang dalawa",
        openAudioAction: "Buksan ang Audio",
        voiceLabel: "Boses",
        voiceDescription: "Pinakamahusay na mga boses sa device na ito",
        studioVoiceDescription:
          "Kailangan ng cloning model ang mga boses na naka-save sa Audio",
        studioVoiceDefault: "Sariling boses ng model",
        studioVoiceSaved: "Naka-save na boses",
        speedLabel: "Bilis",
        pitchLabel: "Pitch",
        volumeLabel: "Volume",
        previewLabel: "I-preview ang boses",
        previewDescription: "Magpatugtog ng maikling sample",
        previewFailed: "Nabigo ang TTS preview",
        previewAction: "I-preview",
        preparingAction: "Gumagawa…",
        stopAction: "Ihinto",
        ttsLabel: "Text to speech",
        notSupported: "Hindi sinusuportahan sa browser na ito",
      },
    },
    general: {
      title: "Pangkalahatan",
      description: "Mga pangkalahatang kagustuhan para sa Unsloth.",
      account: "Account",
      huggingFaceToken: "Token ng Hugging Face",
      huggingFaceTokenDescription:
        "Ginagamit para mag-load ng mga gated model at mag-push ng mga artifact.",
      hideToken: "Itago ang token",
      showToken: "Ipakita ang token",
      clearToken: "I-clear",
      checkingToken: "Sinusuri ang token...",
      tokenValidated: "Na-validate ang token",
      password: "Password",
      passwordDescription:
        "Palitan ang password para sa Unsloth account na ito.",
      passwordDialog: {
        trigger: "Palitan ang password",
        title: "Palitan ang password",
        description:
          "Ilagay ang kasalukuyan mong password at pumili ng bago (hindi bababa sa {minLength} character).",
        setTrigger: "Magtakda ng remote password",
        setTitle: "Magtakda ng remote password",
        setDescription:
          "Piliin ang password na gagamitin ng mga remote na browser para mag-sign in bilang unsloth (hindi bababa sa {minLength} character). Patuloy na awtomatikong magsa-sign in ang Unsloth Desktop App.",
        setSubmit: "Itakda ang password",
        setting: "Itinatakda...",
        setDone: "Naitakda ang password.",
        currentPassword: "Kasalukuyang password",
        newPassword: "Bagong password",
        confirmPassword: "Kumpirmahin ang bagong password",
        currentTooShort:
          "Dapat hindi bababa sa {minLength} character ang kasalukuyang password.",
        newTooShort:
          "Dapat hindi bababa sa {minLength} character ang bagong password.",
        newHasSpaces: "Hindi maaaring may space ang bagong password.",
        mismatch: "Hindi magkatugma ang mga password.",
        samePassword:
          "Dapat iba ang bagong password sa kasalukuyan mong password.",
        update: "I-update ang password",
        updating: "Ina-update...",
        updated: "Na-update ang password.",
        updateFailed: "Nabigo ang pag-update ng password.",
      },
      chatDefaults: "Mga default ng chat",
      autoTitleNewChats: "Awtomatikong lagyan ng pamagat ang mga bagong chat",
      autoTitleNewChatsDescription: "Gumawa ng pamagat mula sa unang mensahe.",
      helperLlm: {
        sectionTitle: "Helper LLM",
        preloadOnStartup: "I-pre-cache ang Helper LLM sa startup",
        preloadOnStartupDescription:
          "I-download ang helper model ng AI Assist sa background sa startup. Naka-off bilang default; makukuha pa rin ito ng AI Assist kapag kailangan.",
        disabledByEnv:
          "Naka-disable dahil sa UNSLOTH_HELPER_MODEL_DISABLE sa environment ng backend.",
        loadError: "Nabigo ang pag-load ng mga setting ng Helper LLM.",
        saveError: "Nabigo ang pag-save ng mga setting ng Helper LLM.",
      },
      developerMode: {
        sectionTitle: "Developer mode",
        videoCpu: "Payagan ang pag-generate ng video sa CPU (experimental)",
        videoCpuDescription:
          "Pinapayagang tumakbo ang page na Video sa host na walang sinusuportahang GPU. Naka-off bilang default; hindi sinusuportahan at napakabagal.",
        notApplicable:
          "Walang epekto sa machine na ito: tumatakbo na ang video sa GPU nito, o may humaharang dito na hindi malulusutan ng CPU.",
        forcedOnByEnv:
          "Na-on dahil sa UNSLOTH_VIDEO_ALLOW_CPU sa environment ng backend.",
        forcedOffByEnv:
          "Na-off dahil sa UNSLOTH_VIDEO_ALLOW_CPU sa environment ng backend.",
        confirmTitle: "I-enable ang pag-generate ng video sa CPU?",
        confirmWarning:
          "Maaaring umabot nang ilang oras ang pag-generate ng video sa CPU, gumamit ng napakalaking RAM, at mag-freeze ng iyong system. Magpatuloy sa sarili mong panganib.",
        ramBelowMinimum:
          "May {host} GB na RAM ang machine na ito, mas mababa sa {min} GB na kailangan ng pinakamaliit na video model sa CPU. Malamang na mabigo ang pag-load o mag-freeze ang system.",
        ramOk:
          "May {host} GB na RAM ang machine na ito. Kailangan ng pinakamaliit na video model ang mga {min} GB sa CPU; mas marami pa ang kailangan ng mas malalaking model.",
        ramUnknown:
          "Kailangan ng pinakamaliit na video model ang mga {min} GB na RAM sa CPU; mas marami pa ang kailangan ng mas malalaking model.",
        performanceEstimate:
          "Asahan ang ilang minuto bawat denoising step: ang 5-frame na 832x480 na clip mula sa Wan2.1 1.3B sa 3 step ay inabot nang mga 4 na minuto sa 16 na CPU thread, at ang buong haba na clip sa mga default na setting ay maaaring umabot nang ilang oras.",
        confirmAction: "I-enable pa rin",
        loadError: "Nabigo ang pag-load ng mga setting ng Developer mode.",
        saveError: "Nabigo ang pag-save ng mga setting ng Developer mode.",
      },
      modelAutoSwitch: {
        sectionTitle: "Awtomatikong pagpapalit ng model (OpenAI API)",
        enable: "Palitan ang model ayon sa request",
        enableDescription:
          "I-load ang na-download na model na pinangalanan sa isang API request bago mag-serve. Naka-off bilang default.",
        autoDownload: "I-download ang mga nawawalang model",
        autoDownloadDescription:
          "Kunin ang GGUF na pinangalanan sa isang API request na hindi pa na-download. Magagamit na ng sinumang may API key ang disk at bandwidth.",
        idleUnload: "Awtomatikong pag-unload kapag idle",
        idleUnloadDescription:
          "Magbakante ng VRAM pagkatapos ng ganitong dami ng idle na segundo. Kapag 0, mananatili itong naka-load; ang minimum ay 60.",
        idleSecondsAriaLabel:
          "Segundo bago ang awtomatikong pag-unload kapag idle",
        mediaEnable: "Palitan ang image at video model ayon sa request",
        mediaEnableDescription:
          "I-load ang na-download na image o video model na pinangalanan sa isang API request bago mag-generate. Hiwalay sa pagpapalit ng chat model. Naka-off bilang default.",
        mediaIdleUnload:
          "Awtomatikong pag-unload kapag idle para sa image at video",
        mediaIdleUnloadDescription:
          "Magbakante ng VRAM sa pamamagitan ng pag-unload ng mga image at video model pagkatapos ng ganitong dami ng idle na segundo. Hiwalay sa pagpapalit ng chat model. Kapag 0, mananatili silang naka-load; ang minimum ay 60.",
        mediaIdleSecondsAriaLabel:
          "Segundo bago ang awtomatikong pag-unload kapag idle para sa image at video",
        mediaIdlePaused:
          "Naka-pause habang naka-on ang Panatilihin ang model sa memory ng GPU.",
        idleNeedsEnable: "I-on muna ang Palitan ang model ayon sa request.",
        idleActiveViaEnv: "Aktibo sa pamamagitan ng UNSLOTH_MODEL_IDLE_TTL.",
        loadError:
          "Nabigo ang pag-load ng mga setting ng awtomatikong pagpapalit ng model.",
        saveError:
          "Nabigo ang pag-save ng mga setting ng awtomatikong pagpapalit ng model.",
        idleError:
          "Ilagay ang 0 para panatilihing naka-load ang model, o hindi bababa sa 60 segundo.",
        keepKv: "Panatilihin ang context ng chat sa mga pag-unload",
        keepKvDescription:
          "I-save ang KV cache bago ang idle na pag-unload, o bago gamitin ng image o video model ang GPU, para hindi na muling basahin ang history ng mga itinuloy na chat. Hanggang 10 GB sa disk.",
        keepKvKeywords: "kv cache context swap resume image video ituloy",
        apiOnly: "I-unload lang ang mga model na na-load ng API",
        apiOnlyDescription:
          "Iniiwan ng idle na pag-unload sa memory ang model na na-load mo mula sa Unsloth, at binabakante lang ang mga na-load ng isang API request.",
      },
      modelAliases: {
        sectionTitle: "Mga alias ng model (OpenAI API)",
        description:
          "Mga matatag na pangalan para sa mga API client, gaya ng fast-chat o code. Ang request na nagpapangalan ng alias ay sine-serve ng model na itinuturo nito, kaya patuloy na gagana ang mga nakakonektang app kapag nagpalit ka ng model. I-on ang Palitan ang model ayon sa request para ma-load ng alias ang model nito.",
        namePlaceholder: "fast-chat",
        nameAriaLabel: "Pangalan ng alias",
        modelAriaLabel: "Model na itinuturo ng alias",
        modelPlaceholder: "unsloth/Qwen3-4B-GGUF:Q4_K_M",
        removeAriaLabel: "Alisin ang alias",
        add: "Magdagdag ng alias",
        loadError: "Nabigo ang pag-load ng mga alias ng model.",
        saveError: "Nabigo ang pag-save ng mga alias ng model.",
      },
      previewSharing: {
        sectionTitle: "Pagbabahagi ng preview",
        enableLabel: "Mga pampublikong preview link",
        enableDescription:
          "Payagan ang sinumang may signed link na makipag-chat sa tapos nang model, nang hindi kailangang mag-sign in. I-off para i-offline ang pampublikong preview; titigil sa paggana ang mga ibinahaging link.",
        loadError:
          "Nabigo ang pag-load ng mga setting ng pagbabahagi ng preview.",
        saveError:
          "Nabigo ang pag-save ng mga setting ng pagbabahagi ng preview.",
        revokeLabel: "Bawiin ang lahat ng preview link",
        revokeDescription:
          "Palitan ang signing secret para tumigil sa paggana ang bawat link na naibahagi mo. Patuloy na gagana ang mga bagong kinopyang link.",
        revokeAction: "Bawiin ang mga link",
        revoking: "Binabawi...",
        revokeConfirmTitle: "Bawiin ang lahat ng preview link?",
        revokeConfirmDescription:
          "Agad na titigil sa paggana ang bawat preview link na naibahagi mo. Hindi na ito maaaring i-undo.",
        revokeConfirmAction: "Bawiin ang lahat ng link",
        revoked: "Nabawi ang lahat ng preview link",
        revokeError: "Hindi mabawi ang mga preview link",
      },
      managedProviderUrls: {
        sectionTitle: "Mga pinamamahalaang account",
        enableLabel: "Mga local at network na koneksyon",
        enableDescription:
          "Payagan ang mga pinamamahalaang account na ituro ang kanilang mga koneksyon sa mga local o network address, gaya ng Ollama o llama.cpp server sa computer na ito o sa iyong LAN. Naka-off bilang default, dahil hinahayaan nito ang mga account na iyon na maabot ang mga serbisyong tumatakbo sa iyong computer at sa iyong network.",
        lockedByEnvironment:
          "Itinakda ng UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1 sa server na ito, na tumatanggi sa mga pribadong address para sa bawat account.",
        loadError:
          "Nabigo ang pag-load ng mga setting ng koneksyon ng pinamamahalaang account.",
        saveError:
          "Nabigo ang pag-save ng mga setting ng koneksyon ng pinamamahalaang account.",
      },
      permissions: {
        sectionTitle: "Mga pahintulot",
        names: {
          ask: "Humingi ng pag-apruba",
          auto: "Aprubahan para sa akin",
          off: "Awtomatikong patakbuhin",
          full: "Buong access",
        },
        details: {
          ask: "Nagtatanong bago ang bawat tool call, kasama ang terminal at Python code, mga paghahanap sa web, pag-edit ng file at mga MCP tool. Hindi ipinapause ang mga tool na pinapatakbo ng external na provider.",
          auto: "Kusang pinapatakbo ang mga karaniwang tool call at nagtatanong lang kapag mukhang mapanganib ang isang aksyon, gaya ng pagbasa ng mga credential, pagtaas ng mga pribilehiyo o pagpapatakbo ng mga mapanirang command.",
          off: "Pinapatakbo ang bawat tool call nang hindi nagtatanong. Tumatakbo pa rin sa sandbox ang Python at terminal code, na naglilimita kung aling mga file ang maaabot nila sa iyong computer. Kung walang gumaganang OS sandbox, magtatanong muna ang mga mapanganib na call.",
          full: "Pinapatakbo ang bawat tool call nang hindi nagtatanong at pinapatay ang code sandbox, kaya mababasa at mababago ng Python at terminal code ang anumang file na naa-access ng iyong account.",
        },
      },
      notifications: {
        sectionTitle: "Mga notification",
        showLoadedModels: "Card ng mga na-load na model",
        showLoadedModelsDescription:
          "Isang card sa sulok na may bawat na-load na model, para ma-unload ang isa sa isang click.",
        updateReminders: "Mga paalala sa update",
        updateRemindersDescription:
          "Gaano kadalas mag-alok ng bagong bersyon kapag handa na ito.",
        llamaHint: "Mga chat at GGUF model",
        audioHint: "Pananalita, musika at transcription",
        frequency: {
          always: "Tuwing available",
          daily: "Isang beses sa isang araw",
          weekly: "Isang beses sa isang linggo",
          biweekly: "Bawat dalawang linggo",
          monthly: "Isang beses sa isang buwan",
          off: "Naka-off",
        },
      },
      startup: {
        sectionTitle: "Startup",
        launchAtLogin: "Patakbuhin ang Unsloth sa pag-sign in",
        launchAtLoginDescription:
          "Simulan ang Unsloth sa background kapag nag-sign in ka. Mananatili ito sa menu bar o system tray hanggang buksan mo ito.",
        closeToTray: "Isara papunta sa system tray",
        closeToTrayDescription:
          "Panatilihing tumatakbo sa background ang Unsloth at ang server nito kapag isinara mo ang pangunahing window.",
        closeToTraySaveError:
          "Nabigo ang pag-update ng setting na isara papunta sa system tray.",
        menuBarIcon: "Ipakita sa menu bar",
        menuBarIconDescription:
          "Ipakita ang icon ng Unsloth sa menu bar ng macOS. Kapag nakatago, mananatili ang Unsloth sa Dock sa halip, kasama kapag nagsimula ito sa pag-sign in.",
        menuBarIconSaveError:
          "Nabigo ang pag-update ng setting ng icon sa menu bar.",
        loadError:
          "Nabigo ang pag-load ng setting na patakbuhin sa pag-sign in.",
        saveError:
          "Nabigo ang pag-update ng setting na patakbuhin sa pag-sign in.",
      },
      hub: {
        source: "Pinagmumulan ng model",
        sourceDescription:
          "Kung saan naghahanap at nagda-download ang model hub. Piliin ang ModelScope kung naka-block o mabagal ang Hugging Face sa iyong network.",
        sourceFallback:
          "Hindi masimulan ang ModelScope, kaya Hugging Face ang ginagamit. Tingnan ang mga log ng Unsloth.",
        autoSourceTitle: "Pinalitan ang pinagmumulan ng model sa ModelScope",
        autoSourceDescription:
          "Kadalasang mabagal o naka-block ang Hugging Face sa iyong rehiyon, kaya mula na sa ModelScope nagda-download ng mga model ang Unsloth.",
        autoSourceAction: "Buksan ang mga setting ng Pinagmumulan ng model",
        sectionTitle: "Model hub",
        endpoint: "Endpoint ng Hugging Face",
        endpointDescription:
          "Kung saan dina-download ang mga model at dataset. Iwanang blangko para sa huggingface.co, o maglagay ng mirror gaya ng https://hf-mirror.com.",
        datasetsServer: "Gamitin ito para sa mga preview ng dataset",
        datasetsServerDescription:
          "Ipadala rin ang mga request para sa preview ng dataset (HF_DATASETS_SERVER) sa endpoint na ito. I-on lang kung sine-serve ng mirror ang mga ito.",
        invalidEndpoint:
          "Maglagay ng http(s) URL na walang login o query. Gumagana lang ang plain http para sa local na address.",
        saveFailed: "Hindi ma-save ang mga setting ng hub.",
        loadFailed: "Hindi ma-load ang mga setting ng hub.",
      },
      downloads: {
        sectionTitle: "Mga download",
        transport: "Transport ng pag-download",
        transportDescription:
          "Paano dumarating ang mga file ng model at dataset mula sa Hugging Face. Itinutuloy ng HTTPS kung saan ito huminto; kadalasang mas mabilis ang Xet sa unang pag-download pero sinisimulan ulit ang file kapag kinansela mo.",
        transportHint:
          "Plain TLS ang HTTPS: pinapayagan ito ng bawat network, proxy at VPN, itinutuloy ang kinansela o naputol na transfer mula sa mga byte na nasa disk na, at nananatiling pantay ang paggamit ng memory. Kumukuha ang Xet ng mga deduplicated na chunk, kaya mas mabilis na darating ang repo na may kaparehong data sa isang mayroon ka na, pero kailangan nito ang hf_xet, gumagamit ng mas maraming RAM, at itinatapon ng pagkansela ang file na dina-download. Tinatanong ng Auto ang machine na ito: tinitimbang nito ang RAM at kung nagsa-stall ang Xet dito, at babalik sa HTTPS.",
        https: "HTTPS",
        xet: "Xet",
        auto: "Auto",
        httpsHint:
          "Karaniwang TLS. Itinutuloy pagkatapos ng pagkansela, gumagana sa anumang network, matatag ang paggamit ng memory.",
        transportDescriptionNoResume:
          "Paano dumarating ang mga file ng model at dataset mula sa Hugging Face. Walang transport na makakapagtuloy sa install na ito, kaya magsisimula ulit ang kinanselang download; kadalasang mas mabilis ang Xet sa unang pag-download.",
        httpsHintNoResume:
          "Karaniwang TLS. Gumagana sa anumang network, matatag ang paggamit ng memory. Hindi maitutuloy ng install na ito ang kinanselang download.",
        xetHint:
          "Transfer ng mga deduplicated na chunk. Kadalasang mas mabilis sa bagong pag-download, sinisimulan ulit ang file kapag kinansela mo, nangangailangan ng mas maraming memory.",
        autoHint:
          "Pumipili ayon sa machine at lumilipat sa HTTPS kung nagsa-stall o nabigo ang Xet dito.",
        autoCurrently: "Gumagamit ang Auto ng {transport} sa machine na ito.",
        xetMissing:
          "Hindi available ang Xet dahil hindi naka-install ang hf_xet.",
      },
      uploads: {
        sectionTitle: "Mga upload",
        maxUploadSize: "Limitasyon sa pag-upload ng training dataset",
        maxUploadSizeDescription: "Ang default ay {defaultSize} MB.",
      },
      rag: {
        sectionTitle: "Mga dokumento at RAG",
        embeddingModel: "Embedding model",
        embeddingModelShort:
          "Ginagawang vector ang iyong mga dokumento para mahanap ng chat ang mga ito.",
        embeddingModelDescription:
          "Hugging Face model o local na path na ginagamit para i-index at hanapin ang iyong mga dokumento. Ang default ay {defaultModel}.",
        searchPlaceholder: "Maghanap ng kahit anong model sa HF",
        reindexWarning:
          "Ang mga bagong na-index na dokumento lang ang naaapektuhan. I-re-embed ang mga dati nang dokumento pagkatapos palitan ang model.",
        reembed: {
          label: "I-re-embed ang mga dokumento",
          staleOne:
            "Na-index ang 1 dokumento gamit ang ibang embedding model, kaya nilalaktawan ito ng semantic search.",
          staleMany:
            "Na-index ang {count} dokumento gamit ang ibang embedding model, kaya nilalaktawan ang mga ito ng semantic search.",
          staleArchived:
            "Na-index ang naunang history ng chat gamit ang ibang embedding model, kaya nilalaktawan ito ng semantic search.",
          running: "Nire-re-embed: {done} sa {total}",
          action: "I-re-embed",
          stop: "Ihinto",
          startFailed: "Hindi masimulan ang pag-re-embed",
          stopped: "Inihinto ang pag-re-embed",
          finished: "Na-re-embed ang mga dokumento",
          finishedWithFailures: "Hindi ma-re-embed ang {count}",
        },
        emptyError: "Maglagay ng ID ng Hugging Face model o local na path.",
        loadError: "Nabigo ang pag-load ng setting ng embedding model.",
        saveError: "Nabigo ang pag-save ng embedding model.",
        saved: "Na-save ang embedding model.",
        saveAnyway: "I-save pa rin",
        recommended: "Inirerekomenda",
        onDevice: "Nasa device",
        searching: "Naghahanap sa Hugging Face…",
        checking: "Sinusuri…",
        noResults: "Walang nakitang embedding model",
        download: "I-download",
        unload: "I-eject",
        unloadFailed: "Hindi ma-eject ang embedding model",
        ejected: "Na-eject ang embedding model",
        ejectModel: "I-eject ang model",
        onDeviceHint:
          "Na-download sa computer na ito. Nilo-load lang ito sa memory habang ini-index o hinahanap ang mga dokumento, kaya hindi ito gumagamit ng RAM o VRAM hanggang doon.",
        loadedHint:
          "Nasa memory na ngayon, gumagamit ng RAM o VRAM. I-hover ang model picker at i-click ang pulang icon para i-eject ito.",
        notLoaded: "Hindi naka-load",
        notLoadedHint:
          "Wala sa memory sa ngayon. Kusa itong maglo-load sa susunod na pag-index o paghahanap ng mga dokumento, pagkatapos ay maaari mo itong i-eject mula sa picker.",
        downloadingStatus: "Dina-download…",
        notDownloaded: "Hindi pa na-download",
        notDownloadedSized: "Hindi pa na-download · {size}",
        loaded: "Naka-load",
        downloading: "Dina-download ang {model}",
        downloadingDescription:
          "Nasa panel ng mga download ang progreso. Gagamitin ito ng pag-index sa sandaling dumating ito.",
        downloadFailed: "Hindi masimulan ang pag-download",
        downloadConflict: "Ituloy ang download na ito mula sa Hub",
        downloadBusy: "Kasalukuyan nang dina-download",
        pin: "I-pin sa menu ng Mga File",
        unpin: "I-unpin sa menu ng Mga File",
        menuChip: "Model",
        moreModels: "Iba pang model",
        pinMoreModels: "Mag-pin ng iba pang model",
        defaultTag: "Default",
        localModel: "Local na model",
        switched: "Itinakda ang embedding model sa {model}",
        switchedNeedsDownload: "Hindi pa na-download ang {model}",
        switchedNeedsDownloadDescription:
          "I-download ito sa Mga Setting bago mag-index ng mga bagong file.",
        switchedUncheckedDescription:
          "Hindi masuri kung na-download na ito. Tingnan sa Mga Setting bago mag-index ng mga bagong file.",
        openSettings: "Buksan ang Mga Setting",
        switchFailed: "Hindi mapalitan ang embedding model",
        back: "Bumalik sa mga source",
        menuSubtitle: "Nagpapagana sa paghahanap ng file",
        device: "Device ng embedding",
        deviceDescription:
          "Kung saan tumatakbo ang embedding model, para sa bawat model. Mas mabilis mag-index ng malalaking set ng dokumento ang GPU pero may VRAM itong ginagamit hanggang mag-restart ang Unsloth.",
        deviceAuto: "Awtomatiko",
        deviceCpu: "CPU",
        deviceGpu: "GPU",
        deviceLocked: "Itinakda ng {name}.",
        deviceSaveError: "Hindi mapalitan ang device ng embedding",
      },
      storage: {
        sectionTitle: "Storage",
        modelsFolder: "Folder ng mga model",
        modelsFolderDescription:
          "Kung saan naka-store ang mga na-download na model.",
        openAction: "Buksan",
        copyAction: "Kopyahin ang path",
        copied: "Nakopya ang path",
        openError: "Hindi mabuksan ang folder",
        copyError: "Hindi makopya ang path",
      },
      fullInstall: {
        label: "Uri ng install",
        description:
          "Chat lang (GGUF). Nagdaragdag ang buong install ng training, pag-export, video, mga safetensors model at karagdagang feature sa pag-generate ng image.",
        action: "Mag-upgrade sa buong install",
        confirmTitle: "Mag-upgrade sa buong install?",
        confirmDescription:
          "Nagpapatakbo ng mga GGUF model ang install na Chat lang na ito. Nagdaragdag ang buong install ng PyTorch para sa GPU ng machine na ito, na nagpapagana ng training, pag-export, video, mga safetensors model at karagdagang feature sa pag-generate ng image. Magda-download ito ng ilang gigabyte (hindi bababa sa 5 GB na espasyo sa disk) at ire-restart ang server. Pananatilihin ang mga chat, model at setting. Maaari itong tumagal nang ilang minuto.",
        confirmAction: "Mag-upgrade",
        keywords:
          "full install upgrade install type chat only gguf add training train training fine-tune pytorch torch export video image generation safetensors buong install mag-upgrade i-export",
      },
      repairInstall: {
        label: "Ayusin ang installation",
        description:
          "Pinapatakbo ulit ang installer sa pinamamahalaang environment. Gamitin ito kung hindi natutukoy ang GPU o hindi magsisimula ang app.",
        action: "Ayusin ang install",
        confirmTitle: "Ayusin ang installation na ito?",
        confirmDescription:
          "Ihihinto ang server at patatakbuhin ulit ang installer, na mag-i-install ulit ng PyTorch para sa GPU ng machine na ito sa pinakabagong sinusuportahang release. Pananatilihin ang mga chat at setting. Maaari itong tumagal nang ilang minuto.",
        confirmDescriptionChatOnly:
          "Ihihinto ang server at patatakbuhin ulit ang installer. Isa itong install na Chat lang, kaya ia-upgrade din ito ng installer sa buong install na may PyTorch, ilang gigabyte sa kabuuan. Pananatilihin ang mga chat at setting. Maaari itong tumagal nang ilang minuto.",
        confirmAction: "Ayusin ngayon",
      },
      resetPreferences: {
        sectionTitle: "Mapanganib na bahagi",
        label: "I-reset ang lahat ng local na kagustuhan",
        description:
          "Kina-clear ang mga kagustuhang local lang. Pananatilihin ang mga chat, API access, at mga setting na naka-store sa DB.",
        action: "I-reset ang mga kagustuhan",
        confirmTitle: "I-reset ang lahat ng local na kagustuhan?",
        confirmDescription:
          "Kina-clear ang mga kagustuhang local lang at nire-reload ang Unsloth. Pananatilihin ang mga chat, API access, at mga setting na naka-store sa DB.",
        confirmAction: "I-reset at i-reload",
      },
    },
    profile: {
      title: "Profile",
      description: "Kung paano lumalabas ang iyong profile sa Unsloth.",
      changePicture: "Palitan ang larawan sa profile",
      uploadPhoto: "Mag-upload ng larawan",
      removePhoto: "Alisin",
      pictureOptions: "Mga opsyon sa larawan sa profile",
      displayName: "Pangalang ipinapakita",
      nickname: "Ano ang itatawag sa iyo ng Unsloth?",
      nicknamePlaceholder: "Palayaw",
      nicknameSaved: "Na-save ang gustong pangalan",
      avatarShape: "Hugis ng avatar",
      avatarShapeCircle: "Bilog",
      avatarShapeRounded: "Pabilog ang kanto",
      greetingSloth: "Pagbati",
      greetingSlothDescription: "Ipakita ang pagbati sa mga bagong chat.",
      chooseSloth: "O pumili ng sloth",
      noPicture: "Walang larawan sa profile",
      noneLabel: "Wala",
      nameSaved: "Na-save ang pangalan sa profile",
      namePersistErrorTitle:
        "Hindi ma-save nang permanente ang pangalan sa profile",
      namePersistErrorDescription:
        "Na-update ang pangalan para sa session na ito, pero maaaring hindi ito manatili pagkatapos mag-reload.",
      photoUpdated: "Na-update ang larawan sa profile",
      photoPersistErrorTitle:
        "Hindi ma-save nang permanente ang larawan sa profile",
      photoPersistErrorDescription:
        "Na-update ang larawan para sa session na ito, pero maaaring hindi ito manatili pagkatapos mag-reload.",
      photoUpdateErrorTitle: "Hindi ma-update ang larawan sa profile",
      imageUseError: "Hindi magamit ang image na ito.",
      stats: {
        title: "Iyong mga stat",
        subtitle:
          "Ang lahat ng nasa ibaba ay binilang mula sa sarili mong history. Walang kinokolekta o ipinapadala sa Unsloth.",
        retry: "Subukan ulit",
        exportCsv: "I-export ang CSV",
        exportError: "Hindi ma-export ang paggamit mo ng token",
        privacyNote:
          "Kinukuwenta ang mga stat mula sa local na chat, paggamit ng API, at history ng training na hawak ng iyong Unsloth install. Hindi kailanman sine-save para sa mga stat ang mga API prompt, sagot, at key. Walang ipinapadala sa Unsloth o sa anumang third party.",
        emptyChats:
          "Wala pang paggamit ng chat o API. Magsimula ng pag-uusap o gumawa ng authenticated na local API request at mapupuno rito ang iyong mga stat.",
        lifetimeTokens: "Mga token sa lahat ng panahon",
        peakTokens: "Pinakaabalang araw",
        longestChat: "Pinakamahabang chat",
        currentStreak: "Kasalukuyang streak",
        longestStreak: "Pinakamahabang streak",
        activityTitle: "Aktibidad ng token",
        activityDescription: {
          daily: "{total} mula noong {since}",
          weekly: "Pinakaabalang linggo {total} · mula noong {since}",
          cumulative: "{total} sa kabuuan mula noong {since}",
        },
        mode: {
          daily: "Araw-araw",
          weekly: "Lingguhan",
          cumulative: "Pinagsama-sama",
        },
        cellTooltip: "{date} · {tokens}, {messages}",
        emptyDayTooltip: "{date} · Walang aktibidad",
        apiRequests: "{count} sa pamamagitan ng API",
        weekTooltip: "Linggo ng {date} · {tokens}",
        cumulativeTooltip: "Hanggang linggo ng {date} · {tokens} sa kabuuan",
        less: "Mas kaunti",
        more: "Mas marami",
        insightsTitle: "Mga insight sa aktibidad",
        totalChats: "Kabuuang chat",
        totalMessages: "Kabuuang mensahe",
        tokensIn: "Mga token na ipinadala",
        tokensOut: "Mga token na na-generate",
        totalTokens: "Kabuuang token",
        studioChatTokens: "Mga token ng Unsloth Chat",
        apiTokens: "Mga API token",
        cachedTokens: "Mga naka-cache na token",
        cachedValue: "{tokens} ({percent}% ng input)",
        avgTokensPerChat: "Average na token bawat chat",
        timeInChat: "Oras sa chat",
        activeDays: "Mga aktibong araw",
        toolCalls: "Mga tool call",
        attachments: "Mga file na in-attach",
        avgSpeed: "Average na bilis",
        bestSpeed: "Pinakamabilis na sagot",
        firstToken: "Average na oras bago ang unang token",
        tokensPerSecond: "{value} tok/s",
        topModelsTitle: "Mga pinakaginagamit na model",
        topModelsDescription: "Inayos ayon sa mga token na napalitan",
        modelSummary: "{tokens} · {messages}",
        noModels: "Wala pang naitalang paggamit ng model.",
        trainingTitle: "Training",
        trainingDescription: "Mga fine-tuning run mula sa workspace na ito",
        trainingRuns: "Mga run",
        trainingCompleted: "Natapos",
        trainingSteps: "Mga step",
        trainingTokens: "Mga token na na-train",
        trainingTime: "Oras ng training",
        bestLoss: "Pinakamahusay na loss",
        runSteps: "{steps}",
        runLoss: "loss {loss}",
      },
    },
    appearance: {
      title: "Hitsura",
      description: "Kung paano lumalabas ang Unsloth sa device na ito.",
      gpuRendering: {
        sectionTitle: "Rendering",
        label: "Pag-render gamit ang GPU",
        description:
          "Iguhit ang Unsloth gamit ang iyong graphics card. Ginagamit ito ng Auto kung saan alam na gumagana ito. Ilalapat ang mga pagbabago pagkatapos mag-restart ang Unsloth.",
        auto: "Auto",
        on: "Naka-on",
        off: "Naka-off",
        oldWebkitWarning:
          "WebKitGTK {version} sa Wayland: maaaring mag-crash ang GPU rendering sa paglunsad sa NVIDIA na may fractional scaling (WebKit bug 280210). Kapag nangyari ito, kusang babalik ang Unsloth sa CPU rendering. Sa WebKitGTK 2.50.4, walang nasukat na pagbilis (8 fps sa alinman), kaya hindi inirerekomenda ang Naka-on dito.",
        oldDriverWarning:
          "NVIDIA driver {version} sa X11: pinapanatiling naka-off ng Unsloth ang GPU rendering hanggang 610 o mas bago ang driver. Nagli-leak ng isang file descriptor bawat frame ang mas lumang driver, at titigil sa pagguhit ang window sa loob ng ilang minuto.",
        fallbackNote:
          "Nag-crash ang GPU rendering sa paglunsad sa system na ito, kaya CPU rendering ang ginagamit ng Unsloth.",
        tryAgain: "Subukan ulit ang GPU",
        envOverride:
          "Itinakda ng environment variable na {name}. Walang epekto ang setting na ito hangga't hindi ito inaalis.",
        macNote:
          "Laging nagre-render gamit ang GPU ang macOS. Walang sinusuportahang switch ang WebKit para i-off ito.",
        restartPending: "I-restart ang Unsloth para mailapat.",
        restartNow: "I-restart ngayon",
        loadError: "Hindi ma-load ang setting ng GPU rendering.",
        saveError: "Hindi ma-save ang setting ng GPU rendering.",
        restartError: "Hindi ma-restart ang Unsloth.",
      },
      theme: {
        title: "Biswal na istilo",
        label: "Mode",
        description: "Maliwanag, madilim, o sundin ang iyong system.",
        system: "System",
        light: "Maliwanag",
        dark: "Madilim",
      },
      palette: {
        label: "Tema",
        description:
          "Mga kulay na ginagamit sa buong Unsloth, sa maliwanag at madilim na mode.",
        standard: "Standard",
        classic: "Classic",
        minimal: "Minimal",
        moreThemes: "Iba pang tema",
        halloween: "Halloween",
      },
      trickOrTreat: {
        label: "Trick or treat",
        description:
          "Mga tema ng Halloween at nakakatakot na pagbati, hanggang Nobyembre 1.",
      },
      custom: {
        chatWidth: {
          label: "Lapad ng chat",
          description: "Lapad ng mga mensahe at ng composer.",
          standard: "Standard",
          wide: "Malapad",
          full: "Buong lapad",
        },
        sentAttachments: {
          label: "Mga attachment sa mga naipadalang mensahe",
          description:
            "Ipinapakita ng Standard ang bawat file kasama ang uri nito, mas marami ang nagkakasya sa bawat linya sa Compact. Lumilipat ang Auto sa Compact kapag lampas sa anim na file.",
          auto: "Auto",
          list: "Standard",
          chips: "Compact",
        },
        reset: "I-reset",
        resetAll: "I-reset ang customization",
        preferencesTitle: "Mga kagustuhan",
        colors: {
          accent: "Accent",
          background: "Background",
          foreground: "Foreground",
        },
        fontDefault: "Default",
        fontBundledGroup: "Built-in",
        fontImportedGroup: "Na-import",
        fontDeviceGroup: "Sa device na ito",
        fontFolderGroup: "Mula sa folder",
        fontDeviceLoading: "Hinahanap ang mga font ng device…",
        fontSearch: "Maghanap ng mga font…",
        fontNoResults: "Walang nakitang font.",
        colorPicker: {
          hue: "Hue",
          hex: "Kulay na hex",
          eyedropper: "Pumili ng kulay mula sa screen",
        },
        uiFont: {
          label: "Font ng UI",
        },
        headingFont: {
          label: "Font ng heading",
        },
        chatFont: {
          label: "Font ng chat",
        },
        codeFont: {
          label: "Font ng code",
        },
        importFont: {
          upload: "I-upload",
          scanFolder: "Pumili ng folder",
          alreadyAvailable:
            "Available na ang font na ito, kaya ang kasalukuyang kopya ang ginagamit.",
          folderNoFonts: "Walang nakitang font file sa folder na iyon.",
          remove: "Alisin",
          errorInvalidType:
            "Hindi sinusuportahang uri ng file. Gumamit ng .woff2, .woff, .ttf, o .otf.",
          errorTooLarge: "Masyadong malaki ang font file (max 1.5 MB).",
          errorLimit: "Hanggang 3 font lang ang maaari mong i-import.",
          errorStorageFull:
            "Kulang ang local storage para sa font na ito. Mag-alis muna ng na-import na font.",
          errorFailed: "Hindi ma-load ang font file na ito.",
        },
        interfaceScale: {
          label: "Scale ng interface",
          description: "Baguhin ang laki ng lahat ng elemento ng interface.",
        },
        uiFontSize: {
          label: "Laki ng font ng UI",
          description:
            "Ayusin ang base na laki na ginagamit para sa UI ng Unsloth.",
        },
        codeFontSize: {
          label: "Laki ng font ng code",
          description: "Ayusin ang base na laki na ginagamit para sa code.",
        },
        fontSmoothing: {
          label: "Pagpapakinis ng font",
          description: "Gumamit ng pinakinis na font anti-aliasing.",
        },
        contrast: {
          label: "Contrast",
        },
        reduceMotion: {
          label: "Bawasan ang galaw",
          description: "Bawasan ang mga animation o sundin ang iyong system.",
          system: "System",
          on: "Naka-on",
          off: "Naka-off",
        },
        pointerCursors: {
          label: "Gumamit ng mga pointer cursor",
          description:
            "Gawing pointer ang cursor kapag naka-hover sa mga interactive na elemento.",
        },
      },
      language: {
        title: "Wika",
        label: "Wika ng display",
        description: "Ang wikang ginagamit ng Unsloth.",
        autoDetect: "Awtomatikong tukuyin",
        search: "Maghanap ng mga wika",
        noResults: "Walang tumutugmang wika",
        spellCheck: "Pag-check ng spelling",
        spellCheckDescription:
          "Salungguhitan ang mga maling baybay na salita habang nagta-type ka.",
      },
      layout: {
        title: "Layout",
        compactSidebar: "I-pin ang sidebar bilang default",
        compactSidebarDescription:
          "Panatilihing nakabukas ang sidebar sa halip na liitan sa mga icon.",
      },
      sidebarNav: {
        title: "Navigation ng sidebar",
        description:
          "I-pin at ayusin ulit ang mga tab ng sidebar. Mapupunta sa menu na Higit pa ang mga hindi naka-pin na tab.",
        dragToReorder: "I-drag para ayusin ulit",
        pinToSidebar: "I-pin ang {name} sa sidebar",
        moreHolds: "Higit pa ({count})",
      },
      sidebarMenu: {
        title: "Menu ng profile",
        description:
          "Pumili at ayusin ulit ang mga shortcut sa ilalim ng iyong pangalan.",
        darkModeToggle: "Toggle ng madilim na mode",
        dragToReorder: "I-drag para ayusin ulit",
      },
    },
    resources: {
      title: "System at Engine",
      description:
        "Subaybayan ang hardware at storage ng Unsloth server na ito.",
      liveUpdates: "Mga live na update",
      floatingWindow: "Lumulutang na window",
      showFloatingWindow: "Ipakita ang lumulutang na window",
      showFloatingWindowDescription:
        "Panatilihing nakikita sa screen ang paggamit ng CPU, RAM at GPU sa ibabaw ng Unsloth.",
      liveMonitor: {
        title: "Live na monitor",
        apiTitle: "Monitor ng API",
        apiDescription:
          "Tingnan kung ano ang ipinapadala ng ibang app sa API ng server na ito habang dumarating ito.",
        summary: "Mga live na request, error, at paggamit ng token",
        status: "{active} aktibo · {recent} kamakailan · {model}",
        noModelLoaded: "walang naka-load na model",
        autoOpen: "Awtomatikong bumubukas ang lumulutang na monitor",
        autoOpenDescription:
          "Nagbubukas ng maliit na panel kapag may dumating na API traffic.",
        cpu: "CPU",
        ram: "RAM",
        disk: "Disk",
        modelsDisk: "Disk ng mga model",
        vram: "VRAM",
        cpuCores: "{logical} logical / {physical} physical na core",
        currentLoad: "Kasalukuyang load",
        free: "{value} libre",
        noGpu: "Walang nakikitang GPU",
        gpuUnusable: "Hindi magamit ang GPU",
        gpuUnusableDetail: "Natukoy, pero hindi ito magamit ng PyTorch",
        detailedView: "Detalyadong view",
        detailedViewDescription:
          "Ipakita ang load ng CPU at ang load, temperatura at power ng bawat GPU sa lumulutang na window.",
        cpuLoad: "Load ng CPU",
        gpuLoad: "Load ng GPU",
        temperature: "Temperatura",
        power: "Power",
      },
      gpu: {
        title: "Mga GPU device",
        memory: "Memory ng GPU",
        sharedWithSystemRam: "Kahati ng RAM ng system",
        estimatedAvailable: "Tinatayang available: {value}",
        sharedEstimatedAvailable:
          "Kahating RAM ng system: tinatayang {value} ang available",
        ggufInference: "Memory ng GGUF model",
        unavailable: "hindi available",
        detecting: "Naghahanap ng mga GPU...",
        unreadable: "Hindi mabasa ang hardware ng server na ito.",
        noGpu:
          "Walang natukoy na nakikitang GPU. Ipinapakita sa itaas ang mga resource ng CPU lang.",
        noUsableGpu: "Walang GPU sa machine na ito na magagamit ng PyTorch.",
        mismatchCpuBuild:
          "CPU-only na build ang PyTorch ({version}), kaya hindi magagamit ang mga GPU sa ibaba. Ayusin ang installation para maibalik ang suporta sa GPU.",
        mismatchUnavailable:
          "Hindi ma-initialize ng PyTorch ({version}) ang mga GPU sa ibaba, kaya hindi magagamit ang mga ito. Suriin ang GPU driver, o ayusin ang installation.",
        driverIdleEvict:
          "May kilalang bug ang AMD driver {version} na maaaring mag-freeze ng Windows kapag idle ang isang AMD GPU, kadalasan kapag higit sa isa ang GPU. Mag-update sa Adrenalin 26.9.2 o mas bago.",
        driverIdleEvictDetails: "Mga detalye",
        dismissNotice: "I-dismiss",
        unusableDevice: "hindi magamit",
        unknownDevice: "Hindi kilalang GPU",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "{value} ang nagamit",
        free: "{value} libre",
        total: "{value} kabuuan",
        usedOf: "{used} ang nagamit sa {total}",
        segmentUsed: "Ginamit ng GPU",
        segmentSystem: "Ginamit ng system",
        segmentAvailable: "Available",
      },
      llamaBackend: {
        title: "Engine ng inference para sa GGUF",
        label: "Backend ng compute",
        description:
          "Ang backend na ginagamit ng llama.cpp para patakbuhin ang mga GGUF model.",
        runningOn: "Kasalukuyang tumatakbo ang llama.cpp sa {backend}.",
        hint: "Ini-install ang build ng llama.cpp para sa backend na ito at pinapanatili ito sa mga update. Kapaki-pakinabang kapag nagka-crash ang awtomatikong pagpili o hindi ito sinusuportahan ng iyong GPU driver. Ang mga backend lang na may build para sa machine na ito ang nakalista; hindi apektado ang training.",
        autoWith: "Awtomatiko ({backend})",
        apply: "Ilapat",
        applying: "Ini-install...",
        applyHint:
          "Dina-download ang bagong build at nire-restart ang llama.cpp. Ia-unload ang naka-load na model.",
        applyHintWithSize:
          "Magda-download ng {size} at ire-restart ang llama.cpp. Ia-unload ang naka-load na model.",
        switchedTo: "Tumatakbo na ngayon ang llama.cpp sa {backend}.",
        switchFailed: "Hindi mapalitan ang backend ng llama.cpp.",
        switchInterrupted: "Naantala ang pagpapalit bago ito natapos.",
        envLocked:
          "Itinakda sa {backend} ng environment variable na UNSLOTH_LLAMA_CPP_BACKEND, na nangingibabaw sa setting na ito.",
        customPath: {
          label: "Custom na folder ng llama.cpp",
          description: "Gamitin ang sarili mong build ng llama-server.",
          hint: "Piliin ang folder ng llama.cpp na naglalaman ng llama-server, o isang build kung saan ito nasa build/bin. Ginagamit ang custom na runtime para sa GGUF chat, mga embedding, at mga sinusuportahang voice model. Mas inuuna pa rin ang mga environment variable.",
          automatic: "Awtomatiko (kasama)",
          bundled:
            "Ginagamit ang runtime ng llama.cpp na in-install ng Unsloth.",
          active:
            "Gagamitin ang iyong custom na llama-server sa susunod na pag-load ng model.",
          environmentManaged:
            "Pinamamahalaan ng environment variable na {variable}.",
          missingBinary:
            "Wala na ang llama-server sa folder na ito. Pumili ng ibang folder o gamitin ang kasamang runtime.",
          reloadRequired:
            "I-reload ang model para magamit ang napiling llama-server.",
          change: "Palitan",
          saving: "Sine-save...",
          useBundled: "Gamitin ang kasama",
          chooseTitle: "Pumili ng folder ng llama.cpp",
          chooseAction: "Gamitin ang folder na ito",
          saved: "Na-update ang folder ng llama.cpp",
          saveError: "Hindi ma-update ang folder ng llama.cpp",
        },
        backends: {
          auto: "Awtomatiko",
          cpu: "CPU",
          cuda: "CUDA",
          rocm: "ROCm",
          vulkan: "Vulkan",
          metal: "Metal",
        },
        unsupported: {
          notInstalled:
            "Walang nakitang pinamamahalaang install ng llama.cpp, kaya walang backend na mapapalitan.",
          localLink:
            "Isang local na directory na ikaw mismo ang nag-link ang llama.cpp, kaya hindi ito papalitan ng Unsloth.",
          sourceBuild:
            "Na-build mula sa source ang llama.cpp na ito, kaya hindi mapapalitan dito ang backend nito.",
          customPath:
            "May napiling custom na folder ng llama.cpp. Ang build nito ang nagpapasya ng compute backend.",
          unresolved:
            "Hindi masuri ang mga available na backend. Suriin ang iyong koneksyon at subukan ulit.",
          updateChecksDisabled:
            "Naka-disable ang pagsuri ng update (UNSLOTH_DISABLE_UPDATE_CHECK=1), kaya hindi hinahanap ang mga available na backend.",
        },
        llamaBackendKeywords:
          "llama.cpp backend gguf inference cuda rocm hip vulkan metal cpu gpu accelerator prebuilt switch engine custom path folder directory llama-server executable binary palitan",
      },
      modelMemory: {
        title: "Memory ng model",
        keepResident: "Panatilihin ang model sa memory ng GPU",
        keepResidentDescription: "Manatili sa VRAM sa pagitan ng mga prompt.",
        keepResidentHint:
          "Huwag ibalik ang mga weight sa RAM ng system habang naka-load ang model. Pinapatay ang awtomatikong pag-unload kapag idle, at kapag nasa host RAM nga ang mga weight (unified memory, o bahagyang pag-offload sa GPU) ipinapasa rin nito ang --mlock para hindi ma-page out ng OS ang mga ito at muling i-upload sa susunod mong prompt.",
        noRamReserve: "Huwag magreserba ng RAM ng system para sa model",
        noRamReserveDescription:
          "Bawasan ang host RAM na hawak para sa mga weight ng model.",
        noRamReserveHint:
          "Nilalaktawan ang memory-mapped na pag-load sa mga sinusuportahang build ng Windows kapag ganap na na-offload sa GPU ang model, para hindi manatiling hawak ang mga page nito. Kung hindi, pinapanatili ang memory-mapped na pag-load. Maaari pa ring gumamit ng RAM ang mga kinakailangang CPU buffer. Inaalis ang --no-mmap at --mlock.",
        multiModel: "Panatilihing naka-load ang maraming model",
        openSystemSettings: "Buksan sa System",
        multiModelOff: "Naka-off",
        multiModelFit: "Kapag kasya",
        multiModelAlways: "Palagi",
        multiModelDescription:
          "Ang pag-load ng chat, speech, image o video model ay mag-a-unload ng iba pa.",
        multiModelDescriptionFit:
          "Pinapanatili ng pag-load ng model ang iba kapag kasya ang mga ito. Kapag hindi, ia-unload muna nito ang mga hindi ginagamit, pagkatapos ay ang aktibo.",
        multiModelDescriptionAlways:
          "Hindi kailanman nag-a-unload ng iba ang pag-load ng model. Hindi nilo-load ang model na hindi kasya. Nagbabakante pa rin ng memory ang awtomatikong pag-unload kapag idle at ang training.",
        multiModelHint:
          "Saklaw ang mga chat, speech, image at video model. Sinasagot ng bawat na-load na chat model ang mga request na nagpapangalan dito. Kapag maraming GPU, sinusubukan ng Unsloth na ilagay ang bagong model sa GPU na walang ibang model na gumagamit. Nag-iiwan ang “Kapag kasya” ng mga 1 GB na ekstra. Nakukuha ng model na na-load kasabay ng iba ang memory na iniwan nilang libre, kaya maaaring magsimula ito nang may mas maikling context. Hindi kailanman nag-a-unload ng model sa ibang GPU ang “Kapag kasya” at “Palagi”.",
        allowOversizedLoad: "Payagan ang pag-load ng sobrang laking model",
        allowOversizedLoadDescription:
          "I-load ang mga image at video model kahit sabihin ng pagsuri ng memory na hindi kakasya ang mga ito.",
        allowOversizedLoadHint:
          "Naaangkop kung saan iisang memory pool ang pinaghahatian ng CPU at GPU (Apple Silicon, mga integrated GPU). Tumatanggi ang Unsloth sa pag-load kapag mas malaki ang tantiya nito sa mga weight ng model kaysa sa memory na natitira pagkatapos magreserba ng espasyo para sa operating system. Kapag naka-on ito, susubukan pa rin ang mga pag-load na iyon. Maaaring mali ang tantiya sa alinmang direksyon: maaaring mabigo ang sobrang laking pag-load, ubusin ang lahat ng memory, magpa-swap nang husto o magpahinto sa pagtugon ng computer, o magpahinto sa Unsloth. Naka-off bilang default.",
        mediaIdleUnloadDescription:
          "Magbakante ng VRAM sa pamamagitan ng pag-unload ng mga image at video model pagkatapos ng ganitong dami ng idle na segundo. Pinapanatili silang naka-load ng 0, minimum na 60.",
        mlockVetoed:
          "Mananatiling naka-off ang --mlock: ang pag-pin sa model ay magrereserba ng RAM para sa kabuuan nito. Naka-disable pa rin ang awtomatikong pag-unload kapag idle.",
        mlockNotApplicable:
          "Ganap na nasa GPU: walang nasa RAM ng system na ila-lock. Mananatiling naka-off ang awtomatikong pag-unload kapag idle.",
        memlockCapped:
          "Nililimitahan ng system na ito ang naka-lock na memory sa {limit}. Hindi ganap na mapi-pin ang mas malaking model; itaas ang limitasyon gamit ang ulimit -l.",
        reloadRequired:
          "I-reload ang model para mailapat ang mga bagong memory flag.",
        loadError: "Nabigo ang pag-load ng mga setting ng memory ng model",
        saveError: "Nabigo ang pag-save ng mga setting ng memory ng model",
        modelMemoryKeywords:
          "mlock memlock ulimit vram gpu memory ram resident residency pin pinned page lock locked keep loaded unload idle mmap no-mmap load-mode swap paging i-unload naka-load",
        multiModelKeywords:
          "multiple several models together alongside swap fit unload evict vram gpu image video chat speech tts voice read aloud always idle maraming model boses palagi",
        mediaIdleUnloadKeywords:
          "image video diffusion idle unload ttl timeout seconds free vram gpu memory i-unload segundo",
      },
      storage: {
        title: "Storage",
        systemDisk: "Disk ng system",
        systemUsed: "{used} ang nagamit sa {total} sa disk ng system",
        modelsDiskSummary:
          "Drive ng folder ng mga model: {used} ang nagamit sa {total}, {free} libre",
        segmentModels: "Mga model",
        segmentCaches: "Mga cache",
        segmentOther: "Iba pa",
        segmentFree: "Libre",
        refreshAction: "I-refresh",
        modelsDisk: "Disk ng mga model",
        diskUsage: "{used} ang nagamit / {total}",
        diskFree: "{free} libre",
        modelsFolder: "Folder ng mga model",
        modelsFolderDescription:
          "Kung saan naka-store ang mga na-download na model.",
        modelsFolderHint:
          "Kung saan naka-store ang mga na-download na model. Palitan ito para hindi mapunta ang mga model sa iyong system drive. Ilalapat lang sa mga bagong download. Mananatili sa kinalalagyan nila ang mga model na mayroon ka na.",
        modelsFolderKeywords:
          "models folder directory path location download downloads cache storage disk drive move relocate hugging face lokasyon i-download",
        futureDownloads: "Mga bagong download lang",
        environmentManaged:
          "Pinamamahalaan ng environment variable na {variable}.",
        locationFree: "{free} libre",
        openAction: "Buksan",
        copyAction: "Kopyahin ang path",
        changeAction: "Palitan",
        resetAction: "Gamitin ang default",
        chooseTitle: "Pumili ng lokasyon ng pag-download ng model",
        chooseAction: "Gamitin para sa mga susunod na download",
        cacheSaved: "Na-update ang lokasyon ng pag-download ng model",
        cacheSaveError: "Hindi ma-update ang lokasyon ng pag-download ng model",
        cachePickerError: "Hindi mabuksan ang folder picker",
        copied: "Nakopya ang path",
        openError: "Hindi mabuksan ang folder",
        copyError: "Hindi makopya ang path",
        caches: {
          label: "Mga cache file",
          description:
            "{size} sa mga cache, kung saan {reclaimable} ang maaaring i-clear ngayon.",
          hint: "Mga download ng package, mga compiled kernel at mga transfer cache na binubuo ulit ng Unsloth kapag kailangan nito. Hindi kailanman kina-clear dito ang mga na-download na model, proyekto, chat, setting at ang iyong Hugging Face token.",
          keywords:
            "cache caches purge prune clear clean free space disk uv pip npm bun triton inductor cuda numba matplotlib vllm compiled xet temporary i-clear espasyo",
          measuring: "Sinusukat ang laki ng mga cache...",
          measureFailed: "Hindi masukat ang mga cache",
          empty: "Walang nakitang cache file.",
          detailsAction: "Mga detalye",
          recheckAction: "Suriin ulit",
          hideDetailsAction: "Itago ang mga detalye",
          showFolders: "Ipakita ang mga folder",
          hideFolders: "Itago ang mga folder",
          previousLocations:
            "Kasama ang mga dating lokasyon. Ang pangunahing button na I-clear ay nakakaapekto sa kasalukuyang folder ng cache. Ang mga button ng model ay kina-clear lang ang napiling kopya.",
          clearModelDescription:
            "Inaalis lang nito ang naka-cache na kopyang ito. Ida-download itong muli kapag kailangan.",
          clearAction: "I-clear ang mga cache",
          clearOneAction: "I-clear",
          clearingAction: "Kina-clear...",
          confirmTitle: "I-clear ang mga naka-cache na file?",
          confirmDescription: "Magbabakante ito ng mga {size}.",
          confirmOneTitle: "I-clear ang {name}?",
          safety:
            "Binubuo ulit ng Unsloth ang cache sa susunod na kailanganin nito. Hindi gagalawin ang mga na-download na model, proyekto, chat, setting at ang iyong Hugging Face token.",
          hubCost:
            "Ito ang cache ng model. Kapag na-clear ito, ida-download ulit ang mga model na iyon sa susunod mong paggamit sa mga ito.",
          datasetsCost:
            "Kapag na-clear ito, ida-download ulit ang mga dataset na iyon sa susunod mong paggamit sa mga ito.",
          blocked: "Hindi na-clear: {reason}",
          cleared: "Na-clear ang {size}",
          partial: "May ilang cache file na hindi maalis",
          clearFailed: "Hindi ma-clear ang mga cache",
          names: {
            uv: "Package cache ng uv",
            pip: "Download cache ng pip",
            npm: "Package cache ng npm",
            bun: "Package cache ng Bun",
            torchInductor: "Compile cache ng Torch Inductor",
            torchExtensions: "Mga build ng Torch extension",
            triton: "Kernel cache ng Triton",
            cuda: "Kernel cache ng CUDA",
            numba: "Compile cache ng Numba",
            matplotlib: "Font cache ng Matplotlib",
            vllm: "Cache ng vLLM",
            unslothCompiled: "Mga compiled module ng Unsloth",
            hfXet: "Transfer cache ng Hugging Face",
            hfAssets: "Asset cache ng Hugging Face",
            hfDatasets: "Dataset cache ng Hugging Face",
            hfHub: "Model cache ng Hugging Face",
          },
        },
        lowDisk: {
          title: "Paubos na ang espasyo sa disk",
          criticalTitle: "Kritikal na mababa ang espasyo sa disk",
          description:
            "{free} libre sa {total}. Makakapagbakante ng espasyo ang pag-clear ng mga cache.",
          action: "Suriin ang mga cache",
        },
      },
      environment: {
        title: "Environment",
        backend: "Backend",
        python: "Python",
        torch: "Torch",
        transformers: "Transformers",
        mlxLm: "mlx-lm",
        mlxVlm: "mlx-vlm",
        uptime: "Uptime",
        processMemory: "Memory ng proseso",
        notInstalled: "Hindi naka-install",
        unknown: "Hindi alam",
        vramWithShared: "{vram} VRAM + {shared} kahati",
      },
    },
    agents: {
      title: "Mga Agent",
      description:
        "Ikonekta ang mga coding agent gaya ng Claude Code at Codex sa isang local na model gamit ang unsloth start.",
      agentKeywords:
        "vs code vscode copilot pi mistral vibe deepseek harness hermes openclaw opencode claude codex",
      vscodeNote:
        "Eksepsiyon ang VS Code: idinadagdag ng command na ito ang sine-serve na model sa listahan ng model ng Copilot Chat ng VS Code (chatLanguageModels.json).",
      intro:
        "ay nagkokonekta ng Claude Code, Codex, DeepSeek Harness, Hermes, OpenClaw, OpenCode at iba pang agent sa isang model na sine-serve nang local ng Unsloth, ganap na offline. Nagpapatakbo ito ng OpenAI-compatible na server at hindi kailanman ginagalaw ang mga config file ng iyong agent.",
      readDocs: "Basahin ang docs",
      copy: "Kopyahin",
      copied: "Nakopya",
      commandBuilder: "Tagabuo ng command",
      commandShell: "Shell para sa lahat ng command",
      agent: "Coding agent",
      model: "Model",
      searchModels: "Maghanap ng mga GGUF model...",
      noModels: "Walang tumutugmang GGUF model.",
      showingModels:
        "Ipinapakita ang {shown} sa {total} tugma. Magpatuloy sa pag-type para paliitin ang listahan.",
      quantization: "Quantization",
      loadingQuantizations: "Nilo-load ang mga quantization...",
      noQuantizations: "Walang hiwalay na quantization",
      recommended: "Inirerekomenda",
      downloaded: "Na-download",
      quantizationLoadError:
        "Hindi ma-load ang lahat ng quantization. Gagamitin ng command ang available na value ng model.",
      generatedCommand: "Nabuong command",
      docs: "Docs",
      docsLabel: "Buksan ang docs ng unsloth start",
      agentDocs: "Buksan ang docs sa pag-set up ng {agent}",
      copyGeneratedCommand: "Kopyahin ang nabuong command",
      automaticSettingsNote:
        "Awtomatikong inilalapat ng Unsloth ang mga inirerekomendang setting ng model kung wala kang itinakdang flag.",
      configurationNote:
        "Maaari mo ring ayusin ang anumang configuration. Tingnan sa ibaba o ang",
      configurationDocs: "docs",
      configurationFlagsSuffix: "para sa mga flag.",
      modelNote:
        "Kailangan ng Codex ang GGUF model na sine-serve ng llama-server. Maaari ring gumamit ang ibang agent ng mga model na transformers ang backend; alisin ang --model para gamitin ang model na naka-load na sa Unsloth.",
      subagent: {
        title: "Gumamit ng local na model bilang subagent",
        description:
          "Panatilihin ang {agent} sa kasalukuyang model nito at ipasa ang mga piling gawain sa local na Unsloth model na ito.",
        setupCommand: "Command sa pag-set up",
        copySetupCommand: "Kopyahin ang command sa pag-set up ng subagent",
        usagePrompt: "Pagkatapos, sa {agent}, i-type ang:",
        copyUsagePrompt: "Kopyahin ang prompt sa paggamit ng subagent",
        defaultPrompt:
          "Gumawa ng local na agent para i-implement ang function na ito.",
        opencodePrompt:
          "@unsloth hanapin ang sanhi ng pagbagsak ng test na ito",
      },
      quickstart: {
        title: "Bumuo ng command",
        description:
          "Maglunsad ng agent laban sa model na kasalukuyang naka-load sa Unsloth. Mag-load muna ng model, pagkatapos ay palitan ang claude ng alinmang sinusuportahang agent sa ibaba.",
        noneDetected:
          "Walang nakitang sinusuportahang agent CLI sa iyong PATH.",
        installed: "Naka-install",
      },
      supportedAgents: {
        title: "Mga sinusuportahang agent",
        description:
          "Inilulunsad ang bawat agent gamit ang sarili nitong command:",
        requiresGguf: "Kailangan ng GGUF model",
      },
      models: {
        title: "Pagpili ng model",
        description:
          "Ipasa ang --model para pumili ng model at quantization, at ang --context-length para itakda ang window. Gumamit ng quantization suffix, o isang tahasang --gguf-variant flag.",
        suffixLabel: "May quantization suffix",
        variantLabel: "May tahasang variant flag",
      },
      options: {
        title: "Mga karaniwang opsyon",
        description:
          "Unang pina-parse ang mga flag ng Unsloth; ang anumang hindi nito nakikilala ay direktang ipinapasa sa agent.",
        model:
          "Pumili ng model. Kapag walang --model, gagamitin ng unsloth start ang model na kasalukuyang na-load sa Unsloth at mag-e-error ito kung walang na-load.",
        contextLength:
          "Itakda ang hinihiling na haba ng context (alias: --max-seq-length).",
        ggufVariant: "Piliin ang variant ng GGUF quantization.",
        loadIn4bit:
          "I-toggle ang 4-bit na pag-load para sa mga model ng Hugging Face.",
        tensorParallel: "I-toggle ang tensor-parallel sa maraming GPU.",
        serve: "I-enable o i-disable ang awtomatikong lokal na server.",
        launch: "Ilunsad ang agent, o i-print lang ang command at environment.",
        persist:
          "Panatilihin ang storage ng agent na pinamamahalaan ng Unsloth sa pagitan ng mga run.",
        asSubagent:
          "Panatilihin ang parent sa kasalukuyan nitong model at irehistro ang Unsloth bilang lokal na subagent (Claude Code, Codex, OpenCode, at Pi).",
        apiKey:
          "Ibigay ang iyong Unsloth API key (o itakda ang UNSLOTH_API_KEY).",
        reasoning:
          "Gamitin ang reasoning sa chat: on, off, o auto. Sinusunod ng auto ang chat template ng model, na kadalasang nangangahulugang on.",
        reasoningEffort:
          "Reasoning effort na ipinapasa sa chat template ng model, hal. medium. Nakadepende sa model ang mga level, kaya gumamit ng tinatanggap ng model na iyon. Kapag hindi itinakda, pinapanatili ang sariling level ng template.",
        yolo: "Laktawan ang mga prompt ng pag-apruba. Gamitin lang sa mga pinagkakatiwalaang environment.",
      },
      remote: {
        title: "Kumonekta sa remote na Unsloth",
        description:
          "Ituro ang unsloth start sa isang Unsloth instance na tumatakbo sa ibang lugar sa pamamagitan ng pagtakda ng mga ito bago maglunsad (o direktang ipasa ang --api-key):",
      },
      passthrough: {
        title: "Pagpapasa ng mga argument ng agent",
        description:
          "Ang mga argument pagkatapos ng mga flag ng Unsloth ay ipinapasa sa mismong agent, kaya gumagana pa rin ang mga native command tulad ng resume:",
      },
      dryRun: {
        title: "I-preview nang hindi inilulunsad",
        description:
          "Idagdag ang --no-launch para i-print ang environment at command sa halip na ilunsad ang agent. Kung nakatakda ang --model, maaari pa ring i-resolve at i-load ang model.",
      },
    },
    chat: {
      groups: {
        conversations: {
          title: "Mga pag-uusap",
        },
        files: {
          title: "Mga file at pag-paste",
        },
        display: {
          title: "Display",
        },
        composer: {
          title: "Kahon ng mensahe",
        },
        menu: {
          title: "Menu ng chat",
        },
        advanced: {
          title: "Advanced",
        },
        contextTitle: "Context",
      },
      library: {
        label: "Pamahalaan ang mga chat",
        description:
          "Maghanap, mag-sort, at mag-ayos ng mga chat, proyekto, at seksyon.",
        action: "Buksan ang Library",
      },
      projectAttachmentsHint:
        "I-override ang setting na ito mula sa menu ng attachment ng bawat chat.",
      rememberParamsPerModelHint:
        "Kapag naka-off, parehong mga setting ang gagamitin para sa bawat model.",
      autoCompactHint:
        "Mga lokal na GGUF at MLX chat: ini-index ang mga inalis na turn para mahanap at maibalik ng model, at sa pag-reset ay sinisipi nang salita por salita ang mga nakatayong tagubilin na kasya, inuuna ang pinakaluma at pinakabago kaysa sa nasa gitna. Kailangan ng naka-save na chat at ng vector index para sa pag-archive; kung wala ang mga ito, itinatapon ang mas lumang mga turn. Ginagamit ang haba ng context na itinakda mo, hindi ang available na VRAM. Mga API model: binubuod ng provider ang mas lumang mga turn kung kaya nito, kung hindi ay itinatapon ang pinakalumang mga turn habang papalapit ang chat sa nakalathalang context window ng model.",
      pastedTextShortDescription:
        "Ang mga paste na {count} character o higit pa ay nagiging .txt na attachment.",
      pastedTextOffDescription:
        "Palaging nananatili sa kahon ng mensahe ang na-paste na text.",
      title: "Chat",
      description: "I-customize kung paano gumagana ang chat sa device na ito.",
      modelSelection: {
        title: "Pagpili ng model",
        expandQuantizations: "I-expand ang mga quantization",
        expandQuantizationsDescription:
          "Ipakita ang mga GGUF quantization sa On Device nang hindi binubuksan ang bawat model.",
        showAllQuantizations: "Lahat ng quantization",
        showAllQuantizationsDescription:
          "Isama sa On Device ang mga quantization na hindi pa na-download.",
        showMemoryBar: "Bar ng paggamit ng VRAM",
        showMemoryBarDescription:
          "Ipakita ang tinatayang VRAM para sa mga weight ng model, context, at speculative decoding.",
        fitOnDeviceOnly: "Ipakita lang ang mga model na kasya",
        fitOnDeviceOnlyDescription:
          "Itago sa model picker at Hub ang mga model na masyadong malaki para sa memory ng GPU at RAM ng device na ito. Nananatiling nakikita ang mga na-download na model.",
      },
      menu: {
        title: "Menu ng chat",
        description:
          "I-pin ang mga item sa + menu ng chat at i-drag ang mga ito para ayusin ang pagkakasunod-sunod. Lilipat ang iba sa “Higit pa”.",
        deepResearch: "Deep research",
        browserUse: "Paggamit ng browser",
        chatWithFiles: "Makipag-chat sa mga file (RAG)",
        mcp: "MCP",
        skills: "Mga skill",
        savedPrompts: "Mga naka-save na prompt",
        compareChat: "Paghambingin ang chat",
        exportChat: "I-export ang chat",
        reset: "I-reset sa default",
        addFiles: "Magdagdag ng mga larawan at file",
        webSearch: "Paghahanap sa web",
        tools: "Mga tool",
        more: "Higit pa",
      },
      pastedTextThreshold: "Paikliin ang mahahabang paste",
      pastedTextThresholdDescription:
        "Pindutin ang {shortcut} para direktang mag-paste sa kahon ng mensahe.",
      pastedTextThresholdOff: "Naka-off",
      autoScroll: "Mag-scroll habang bumubuo",
      autoScrollDescription:
        "Pinapanatiling nakikita ng Auto-scroll ang bagong text. Hindi gumagalaw ang Manual para mabasa mo mula sa simula.",
      autoScrollAuto: "Auto-scroll",
      autoScrollManual: "Manual",
      autoScrollKeywords:
        "scroll autoscroll auto-scroll follow stick bottom jump streaming generating viewport lock hold auto manual mag-scroll sundan ibaba bumubuo",
      replyNotification: "Mag-notify kapag tapos na ang sagot",
      replyNotificationDescription:
        "Magpakita ng notification ng system kapag natapos ang isang sagot habang naka-minimize o nasa background ang Unsloth.",
      replyNotificationBlocked:
        "Naka-block ang mga notification. Payagan ang mga ito para sa Unsloth sa mga setting ng iyong browser o system, at subukan ulit.",
      replyNotificationKeywords:
        "notify notification alert done finished complete ready response reply background minimized desktop sound abiso tapos sagot tunog",
      showResponseModel: "Model ng sagot",
      showResponseModelDescription:
        "Ipakita ang mga detalye ng model sa mga sagot ng assistant.",
      showReplyTimes: "Oras ng sagot",
      showReplyTimesDescription:
        "Ipakita sa ibaba ng sagot kung kailan ito natapos.",
      showReplyTimesKeywords:
        "timestamp time clock date hour when finished arrived reply response message oras petsa sagot mensahe",
      showGenerationDetails: "Mga detalye ng pagbuo",
      showGenerationDetailsDescription:
        "Ipakita ang live na bilis ng prompt at sagot kasama ang bilang ng token, at kung saan na-load ang mga layer ng model.",
      showGenerationDetailsKeywords:
        "verbose developer stats speed tok/s prefill decode layers offload vram experts bilis",
      inlineEditResponse: "I-edit ang sagot sa mismong mga sagot",
      inlineEditResponseDescription:
        "Panatilihin ang I-edit ang sagot sa bawat sagot, sa halip na sa menu na Higit pa.",
      composerAtBottom: "Composer sa ibaba",
      composerAtBottomDescription:
        "Simulan ang mga bagong chat nang nasa ibaba ng page ang composer sa halip na sa gitna.",
      projectAttachments: "Ibahagi ang mga file sa buong proyekto",
      projectAttachmentsDescription:
        "Gawing available ang mga bagong attachment ng chat sa bawat chat sa proyekto.",
      rememberParamsPerModel: "Tandaan ang mga setting bawat model",
      rememberParamsPerModelDescription:
        "I-restore ang huling ginamit na prompt, temperature, at iba pang setting ng bawat model.",
      autoCompact: "Awtomatikong i-compact ang mahahabang chat",
      autoCompactDescription:
        "Lumilipat ang mas lumang mga turn sa isang archive na mahahanap kapag napuno ng chat ang context nito.",
      autoCompactKeywords:
        "compaction compact auto-compact context window truncate rolling checkpoint headroom archive retrieval recall rag search i-compact paghahanap",
      keepRunningWhenTabClosed: "Patuloy na tumakbo kahit isinara ang tab",
      keepRunningWhenTabClosedDescription:
        "Nagpapatuloy sa server ang mga sagot, paghahanap sa web, at Deep Research pagkatapos isara ang tab. I-off para ihinto na lang ang mga ito.",
      keepRunningWhenTabClosedHint:
        "Buksan ulit ang chat sa anumang device para ituloy ang run na nagpatuloy. Kapag naka-off, humihinto ang run sa loob ng humigit-kumulang isang minuto matapos isara ang huling tab nito, kaya hindi ito napapahinto ng pag-reload. Nalalapat sa mga run na sinimulan pagkatapos mong baguhin ito. Ang mga pansamantalang chat, mga sagot mula sa mga API model o sa mga attachment, at mga turn na gumagamit ng browser panel ay palaging humihinto kasabay ng kanilang tab.",
      keepRunningWhenTabClosedKeywords:
        "background tab close closed leave keep running continue stop cancel deep research web search resume reconnect device phone isara tumakbo ihinto",
      firstTokenLabel: "Limitasyon sa paghihintay ng unang token",
      firstTokenDescription:
        "Gaano katagal maaaring maghintay ang mga lokal na GGUF model nang walang progreso sa pagproseso ng prompt bago ang unang token. Nalalapat sa mga bagong paghihintay para sa lahat ng account. Ang Walang limitasyon ay patuloy na maghihintay hanggang ihinto mo.",
      firstTokenDefault: "20 min (default)",
      firstTokenCustom: "Custom",
      firstTokenUnlimited: "Walang limitasyon",
      firstTokenMinutes: "Paghihintay ng unang token sa minuto",
      firstTokenRange: "Minuto (1–10,080)",
      firstTokenSave: "I-save",
      firstTokenSaving: "Sine-save…",
      firstTokenLoadError:
        "Hindi ma-load ang limitasyon sa paghihintay ng unang token.",
      firstTokenSaveError:
        "Hindi ma-save ang limitasyon sa paghihintay ng unang token.",
      firstTokenKeywords:
        "timeout first token prefill slow CPU overnight unlimited waiting limit unang token mabagal paghihintay limitasyon",
      toolResultLimit: "Haba ng output ng tool",
      toolResultLimitDescription:
        "Bilang ng character ng isang resulta ng Python, Terminal o MCP tool na binabasa ng model. Ang default ay {count}.",
      toolResultLimitHint:
        "Ang mas mahabang output ng Python at Terminal ay sine-save sa isang file na mababasa ng model nang paisa-isang bahagi. Nalalapat ang mas mataas na limitasyon sa mga model na tumatakbo sa Unsloth, at mas kaunti pa rin ang nakukuha ng maliliit na context window. Pinapanatili ng mga API model ang default maliban kung ibababa mo ito.",
      toolResultLimitDefault: "{count} (default)",
      toolResultLimitLocked: "Itinakda ng UNSLOTH_TOOL_RESULT_MAX_CHARS",
      toolResultLimitLoadError: "Nabigong i-load ang haba ng output ng tool.",
      toolResultLimitSaveError: "Nabigong i-save ang haba ng output ng tool.",
      toolResultLimitKeywords:
        "tool output result truncate truncated truncation limit cap characters chars length terminal python mcp haba limitasyon",
      visibility: {
        collapsed: "Naka-collapse",
        auto: "I-expand habang tumatakbo",
        expanded: "Palaging naka-expand",
      },
      visibilityKeywords:
        "collapse collapsed expand expanded open closed reasoning thinking tool calls tool activity fold group streaming pag-iisip",
      thinking: {
        visibility: "Pag-iisip",
        visibilityDescription:
          "Kung paano bumubukas ang reasoning. Maaari mo pa ring i-expand o i-collapse ang anumang block nang ikaw mismo.",
      },
      memory: {
        title: "Memory",
        description:
          "Mga tala na binabasa ng bawat chat, para sa bawat model, lokal man o cloud. Binabasa rin ng mga chat sa isang proyekto ang AGENTS.md ng proyektong iyon.",
        enable: "Gamitin ang memory sa mga chat",
        enableDescription: "Ipadala ang mga talang ito kasama ng bawat chat.",
        locked:
          "Naka-off para sa install na ito dahil sa UNSLOTH_STUDIO_AGENTS_MD.",
        editor: "Mga tala sa memory",
        placeholder:
          "Halimbawa:\n- Nagsusulat ako ng Python at gumagamit ng uv.\n- Panatilihing maikli ang mga sagot.",
        hint: "Naka-save sa {path}. Maaari mo ring i-edit ang file na iyon sa anumang text editor.",
        size: "{used} / {max} KB",
        save: "I-save",
        saved: "Na-save ang memory",
        loadError: "Nabigong i-load ang memory.",
        saveError: "Nabigong i-save ang memory.",
        keywords:
          "memory remember personalization personalisation profile preferences notes facts custom instructions about me agents.md claude.md tandaan tala tagubilin",
      },
      currentDate: {
        label: "Sabihin sa model ang petsa ngayon",
        description:
          "Isama ang petsa ngayon para makahanap ang paghahanap at research ng mga bagong source.",
        loadError: "Nabigong i-load ang mga setting ng kasalukuyang petsa",
        saveError: "Nabigong i-update ang mga setting ng kasalukuyang petsa",
      },
      tools: {
        visibility: "Mga tool call",
        visibilityDescription:
          "Kung paano bumubukas ang aktibidad ng tool. Maaari mo pa ring i-expand o i-collapse ang anumang call nang ikaw mismo.",
        foldIntoThinking: "Igrupo ang mga tool call sa ilalim ng Pag-iisip",
        foldIntoThinkingDescription:
          "Ipakita ang mga tool call ng isang turn sa loob ng block nitong Pag-iisip sa halip na sa sarili nilang mga row.",
        foldIntoThinkingBlocked:
          "Hindi available habang nakatakda ang mga tool call sa Palaging naka-expand, na nagpapanatili sa kanila sa sarili nilang mga row.",
      },
      webSearch: {
        title: "Paghahanap sa web",
        images: "Mga larawan mula sa paghahanap sa web",
        imagesDescription:
          "Isama ang mga larawan sa mga resulta ng paghahanap.",
      },
      artifacts: {
        title: "Mga HTML page",
        collapseHtmlBlocks: "I-collapse ang mga HTML block",
        collapseHtmlBlocksDescription:
          "Ipakita ang buong HTML document sa chat bilang card na bumubukas sa browser, sa halip na bilang code.",
        allowNetworkAccess: "Payagan ang network access ng mga HTML page",
        allowNetworkAccessDescription:
          "Hayaang mag-load ang mga HTML page mula sa chat ng mga panlabas na script, style, font, at media. I-off para sa mga offline na preview.",
        blockedBanner:
          "Na-block ang {count} panlabas na resource mula sa {hosts}.",
        blockedBannerPlural:
          "Na-block ang {count} panlabas na resource mula sa {hosts}.",
        blockedBannerAction: "Payagan para sa page na ito",
        blockedTitle: "Naka-off ang network access para sa mga HTML page",
        blockedHint:
          "I-on ang “{setting}” sa Mga Setting → Browser para hayaang mag-load ang mga HTML page ng mga panlabas na resource, o payagan ito para lang sa page na ito.",
        blockedSettingsAction: "Buksan ang Mga Setting",
        blockedDismiss: "I-dismiss",
        errorTitle: "Nagkaroon ng error ang page na ito",
        errorTitlePlural: "Nagkaroon ng {count} error ang page na ito",
        errorHint:
          "Inilalagay ng Ayusin gamit ang model ang error sa kahon ng mensahe. Walang ipapadala hangga't hindi mo ito ipinapadala.",
        errorBannerAction: "Ayusin gamit ang model",
        errorConsoleAction: "Buksan ang console",
        errorConsoleHideAction: "Itago ang console",
        errorLocation: "linya {line}, column {column}",
        errorLine: "linya {line}",
        consoleTitle: "Console",
        reloadCanvas: "Patakbuhin ulit",
        consoleMessageCount: "{count} mensahe",
        consoleMessageCountPlural: "{count} mensahe",
        consoleClear: "I-clear ang console",
        consoleClose: "Isara ang console",
        consoleEmpty: "Wala pang output sa console.",
        consoleCapped:
          "Ang huling {count} entry lang ang itinatabi; itinapon ang naunang output.",
      },
      data: "Data",
      exportHistory: "I-export ang history ng chat",
      exportHistoryDescription:
        "I-download ang lahat ng chat at mensahe bilang JSON.",
      exportAction: "I-export",
      exportingAction: "Ine-export...",
      exportConversations: "I-export ang Mga Kamakailan at Mga Proyekto",
      exportConversationsDescription:
        "I-download ang Mga Kamakailan o Mga Kamakailan kasama ang mga chat ng proyekto bilang Training JSONL, CSV, ShareGPT JSONL, o Markdown, pinagsama o bawat chat. Available ang Message JSONL bawat chat.",
      exportConversationsAction: "I-export",
      exportScopeRecents: "Mga Kamakailan",
      exportScopeAll: "Mga Kamakailan + Mga Proyekto",
      exportCombinedSuffix: "(pinagsama)",
      exportPerChatSuffix: "(bawat chat)",
      importChats: "I-import ang mga chat",
      importChatsDescription:
        "I-import ang mga JSON, JSONL, NDJSON, CSV, o Markdown file sa Mga Kamakailan.",
      importChatsAction: "I-import",
      importNoConversations: "Walang nakitang pag-uusap sa file.",
      importedOneChat: "Na-import ang 1 pag-uusap sa Mga Kamakailan.",
      importedChatCount: "Na-import ang {count} pag-uusap sa Mga Kamakailan.",
      importingChats:
        "Ini-import ang mga chat: {count} sa ngayon ({percent}%)...",
      importedChatCountPartial:
        "Na-import ang {count} pag-uusap sa Mga Kamakailan; hindi ma-save ang {failed}.",
      importFailed: "Nabigo ang pag-import.",
      importFromSource: "I-import mula sa {source}",
      importFromSourceDescription:
        "Kopyahin ang iyong mga pag-uusap sa {source} papunta sa Unsloth, nakagrupo ayon sa proyektong kinabibilangan ng mga ito.",
      importingAction: "Ini-import...",
      importedSourceOneChat: "Na-import ang 1 pag-uusap mula sa {source}.",
      importedSourceChatCount:
        "Na-import ang {count} pag-uusap mula sa {source}.",
      sourceUpdated: "Nagdagdag ng {count} bagong mensahe mula sa {source}.",
      sourceUpToDate: "Napapanahon na ang mga pag-uusap sa {source}.",
      importedSourcePartial: "Hindi ma-import ang ilang pag-uusap sa {source}.",
      clearHistory: "I-clear ang history ng chat",
      clearHistoryDescription:
        "I-delete ang history ng chat mula sa device na ito.",
      clearAction: "I-clear",
      clearAllChats: "I-clear ang lahat ng chat",
      clearAllChatsDescription:
        "Permanenteng i-delete ang bawat chat sa device na ito.",
      noChatsToClear: "Walang chat na ic-clear.",
      clearOneChatDescription:
        "Permanenteng i-delete ang nag-iisang chat sa device na ito.",
      clearChatCountDescription:
        "Permanenteng i-delete ang lahat ng {count} chat sa device na ito.",
      clearChatsAction: "I-clear ang mga chat",
      clearOneChatTitle: "I-clear ang 1 chat?",
      clearChatsTitle: "I-clear ang {count} chat?",
      clearChatsConfirmDescription:
        "Permanenteng ide-delete ang bawat chat sa device na ito. Hindi na ito maaaring i-undo.",
      clearingAction: "Kini-clear...",
      clearOneChatAction: "I-clear ang 1 chat",
      clearChatCountAction: "I-clear ang {count} chat",
      clearedAllChats: "Na-clear ang lahat ng chat",
      clearedOneChat: "Na-clear ang 1 chat",
      clearedChatCount: "Na-clear ang {count} chat",
      someChatsCouldNotBeCleared: "Hindi ma-clear ang ilang chat",
      chatsClearedRemainOne:
        "Na-clear ang {clearedCount} chat; may natitirang 1 chat. Pakisubukan ulit.",
      chatsClearedRemain:
        "Na-clear ang {clearedCount} chat; may natitirang {remainingCount} chat. Pakisubukan ulit.",
      oneChatClearedRemain:
        "Na-clear ang 1 chat; may natitirang {remainingCount} chat. Pakisubukan ulit.",
      oneChatClearedRemainOne:
        "Na-clear ang 1 chat; may natitirang 1 chat. Pakisubukan ulit.",
      storageClearFailedOne:
        "Nabigo ang pag-clear ng storage; maaaring may natitirang 1 chat. Pakisubukan ulit.",
      storageClearFailed:
        "Nabigo ang pag-clear ng storage; maaaring may natitirang {count} chat. Pakisubukan ulit.",
      failedToClearChats: "Nabigong i-clear ang mga chat",
    },
    data: {
      library: {
        defaultSort: "Default",
        updated: "Na-update",
        created: "Ginawa",
        oldest: "Pinakaluma muna",
        alphabetical: "Ayon sa alpabeto",
        allChats: "Lahat ng chat",
        singleChats: "Mga solong chat",
        compareChats: "Mga compare chat",
        allProjects: "Lahat ng proyekto",
        noProject: "Walang proyekto",
        unavailableProject: "Hindi available na proyekto",
        filterSort: "I-filter at i-sort",
        sort: "I-sort",
        type: "Uri",
        sortBy: "I-sort ayon sa",
        filterProject: "I-filter ayon sa proyekto",
        searchProjects: "Maghanap ng mga proyekto",
        noProjects: "Walang nakitang proyekto.",
        untitled: "Walang pamagat",
        oneChat: "1 chat",
        chatCount: "{count} chat",
        searchChats: "Maghanap ng mga chat o proyekto",
        searchArchivedChats: "Maghanap ng mga naka-archive na chat o proyekto",
        searchImages: "Maghanap ng mga naka-archive na larawan",
        searchVideos: "Maghanap ng mga naka-archive na video",
        searchAudio: "Maghanap ng naka-archive na audio",
        unarchive: "I-unarchive",
        unarchiveAll: "I-unarchive lahat",
        unarchiveAllInProject: "I-unarchive lahat sa proyekto",
        deleteAllInProject: "I-delete lahat sa proyekto",
        groupActions: "Mga aksyon para sa {group}",
        unarchiveResults: "I-unarchive ang mga resulta",
        deleteResults: "I-delete ang mga resulta",
        noChats: "Walang chat na tumutugma sa iyong paghahanap.",
        noArchivedChats: "Walang naka-archive na chat.",
        noArchivedMatches:
          "Walang naka-archive na chat na tumutugma sa iyong paghahanap.",
        noMedia: "Walang naka-archive na item.",
        noMediaMatches:
          "Walang naka-archive na item na tumutugma sa iyong paghahanap.",
        itemCount: "Mga item: {count}",
        incompleteSearch:
          "Hindi kumpleto ang paghahanap. Subukan ulit na i-load ang natitirang mga item.",
        searchingRemaining:
          "Hinahanap ang natitirang mga item ({count} ang na-load)...",
        noLoadedMatches: "Walang tugma sa mga item na na-load sa ngayon.",
        deleteItem: "I-delete: {title}",
        unarchiveItem: "I-unarchive: {title}",
        deleteItemsTitle: "I-delete ang mga naka-archive na item ({count})",
        unarchiveItemsTitle: "I-unarchive ang mga item ({count})",
        deleteFilesWarning:
          "Permanenteng ide-delete ang mga file na ito. Hindi na ito maaaring i-undo.",
        restoreWarning: "Babalik sa gallery ang mga item na ito.",
        working: "Ginagawa...",
        loadFailed: "Nabigong i-load ang mga naka-archive na item",
        loadMoreFailed: "Nabigong mag-load ng higit pang naka-archive na item",
        restoreFailed: "Nabigong i-unarchive ang mga napiling item",
        deleteFailed: "Nabigong i-delete ang mga napiling item",
        projectsFailed: "Nabigong i-load ang mga naka-archive na proyekto",
        pageStalled: "Hindi umusad ang page ng archive. Subukan ulit.",
        pageChanged: "Nagbago ang archive habang nilo-load. Subukan ulit.",
        selectAll: "Piliin ang lahat ng nakikitang chat",
        selectItem: 'Piliin ang "{title}"',
        selectedChats: "Mga napiling chat: {count}",
        move: "Ilipat",
        pin: "I-pin",
        unpin: "I-unpin",
        archive: "I-archive",
        deletedChats: "Mga na-delete na chat: {count}",
        restoredChats: "Mga na-unarchive na chat: {count}",
        pinnedChats: "Mga na-pin na chat: {count}",
        unpinnedChats: "Mga na-unpin na chat: {count}",
        movedChatsToProject: "Mga chat na inilipat sa {project}: {count}",
        movedChatsToRecents: "Mga chat na inilipat sa Mga Kamakailan: {count}",
        moveFailed: "Nabigong ilipat ang mga chat",
        deleteChatsTitle: "I-delete ang mga chat ({count})",
        deleteChatsWarning:
          "I-delete ang mga napiling chat ({count})? Hindi na ito maaaring i-undo.",
        deleteArchivedTitle: "I-delete ang {count} naka-archive na chat",
        deleteArchivedWarning:
          "I-delete ang mga napiling naka-archive na chat ({count})? Hindi na ito maaaring i-undo.",
        deleting: "Dine-delete...",
      },
      title: "Data",
      backToData: "Bumalik sa Data",
      exportFailed: "Hindi ma-export ang mga chat",
      description:
        "Pamahalaan ang history ng chat at mga na-upload na file na naka-store sa device na ito.",
      manageFiles: {
        label: "Pamahalaan ang mga file",
        description:
          "Maghanap, mag-sort, at mag-ayos ng mga larawan, video, audio, at file.",
        action: "Buksan ang Library",
      },
      archivedChats: "Mga naka-archive na chat",
      archivedChatsDescription:
        "Tingnan at pamahalaan ang mga chat na na-archive mo.",
      archivedImages: "Mga naka-archive na larawan",
      archivedImagesDescription:
        "Tingnan at pamahalaan ang mga larawang na-archive mo.",
      archivedVideos: "Mga naka-archive na video",
      archivedVideosDescription:
        "Tingnan at pamahalaan ang mga video na na-archive mo.",
      archivedAudio: "Naka-archive na audio",
      archivedAudioDescription:
        "Tingnan at pamahalaan ang mga audio clip na na-archive mo.",
      manageAction: "Pamahalaan",
      exportArchivedChats: "I-export",
      exportingArchivedChats: "Ine-export...",
      exportedOneArchivedChat: "Na-export ang 1 naka-archive na chat",
      exportedArchivedChatCount: "Na-export ang {count} naka-archive na chat",
      noArchivedChatsToExport: "Walang naka-archive na chat na ie-export.",
      failedToExportArchivedChats:
        "Nabigong i-export ang mga naka-archive na chat",
      archiveAllChats: "I-archive ang lahat ng chat",
      archiveAllChatsDescription:
        "Ilipat sa archive ang bawat chat sa Mga Kamakailan at Mga Proyekto.",
      noChatsToArchive: "Walang chat na ia-archive.",
      archiveAllAction: "I-archive lahat",
      archivingAction: "Ina-archive...",
      archiveAllChatsTitle: "I-archive ang lahat ng chat?",
      archiveAllChatsConfirmDescription:
        "Ililipat sa archive ang bawat chat sa device na ito. Mananatiling available ang mga naka-archive na chat at maaaring i-unarchive anumang oras.",
      archivedAllChats: "Na-archive ang lahat ng chat",
      archivedOneChat: "Na-archive ang 1 chat",
      archivedChatCount: "Na-archive ang {count} chat",
      failedToArchiveChats: "Nabigong i-archive ang mga chat",
      confirmBeforeDeleting: "Kumpirmahin bago mag-delete",
      confirmBeforeDeletingDescription:
        "Humingi ng kumpirmasyon bago i-delete ang isang chat. I-off para agad na mag-delete.",
      alwaysDeleteFiles: "Palaging i-delete ang mga file",
      alwaysDeleteFilesDescription:
        "Ang pag-delete ng chat ay nag-aalis din ng sarili nitong sandbox folder mula sa disk. Nananatili sa workspace ng proyektong iyon ang mga file na isinulat nito sa loob ng isang proyekto.",
      chatsSection: "Mga chat at archive",
      transferSection: "Pag-import at pag-export",
      deletionSection: "Pag-delete",
      archives: "Mga archive",
      archiveChatsLabel: "Mga chat",
      archiveVideosLabel: "Mga video",
      exportFormatsSummary:
        "I-export ang Mga Kamakailan at mga chat ng proyekto sa format na gusto mo.",
      trainingSummary: "Gawing dataset para sa fine-tuning ang iyong mga chat.",
      confirmDeletionSummary: "Magtanong bago i-delete ang bawat chat.",
      sandboxFiles: "Mga sandbox file ng chat",
      sandboxFilesDescription: "Default na aksyon kapag nagde-delete ng chat.",
      keepSandboxFiles: "Panatilihin ang mga sandbox file",
      deleteSandboxFiles: "I-delete ang mga sandbox file",
      projectFilesKept:
        "Ang mga sandbox ay mga folder na ginawa para sa bawat chat. Pinapanatili ang mga file na naka-save sa mga workspace ng proyekto.",
      deleteAllAction: "I-delete lahat",
      deleteChatsOnly: "I-delete ang mga chat lang…",
      deleteChatsAndSandboxes: "I-delete ang mga chat at sandbox…",
      deleteSandboxFilesDescription:
        "I-delete din ang sandbox folder ng bawat chat at ang mga file sa loob nito. Pinapanatili ang mga file sa mga workspace ng proyekto.",
      filesSection: "Mga file",
      uploadedFiles: "Mga na-upload na file",
      uploadedFilesDescription:
        "Tingnan at pamahalaan ang mga file na na-upload sa mga chat, proyekto, at knowledge base.",
      fineTuneExport: "Gamitin ang mga chat bilang training data",
      fineTuneExportDescription:
        "Gumawa ng JSONL dataset para sa fine-tuning mula sa iyong mga chat. I-load ito sa Train, pinuhin sa Recipes, o i-export ito.",
      fineTuneExportAction: "I-export ang JSONL",
      fineTuneRunAction: "Patakbuhin",
      fineTuneExportingAction: "Ine-export...",
      fineTuneOpenRecipesAction: "Buksan sa Recipes",
      fineTuneOpeningRecipesAction: "Binubuksan...",
      fineTuneTrainAction: "I-load sa tab na Train",
      fineTuneTrainingAction: "Nilo-load...",
      fineTuneExportFailed: "Nabigong i-export ang training data",
      fineTuneRecipeFailed: "Nabigong buksan ang mga chat sa Recipes",
      fineTuneTrainFailed: "Nabigong i-load ang dataset sa tab na Train",
    },
    connections: {
      title: "Mga Koneksyon",
      description: "Pamahalaan ang mga provider at panlabas na koneksyon.",
    },
    remoteLan: {
      title: "Remote at LAN",
      description:
        "Abutin ang Unsloth na ito mula sa iba mo pang device, sa iyong lokal na network o sa pamamagitan ng pansamantalang pampublikong URL.",
    },
    apiKeys: {
      title: "API",
      description:
        "I-access ang Unsloth sa pamamagitan ng API na compatible sa OpenAI at Anthropic.",
      readDocs: "Basahin ang docs ng API",
      noAccess: "Wala pang API access.",
      accessTokens: "Mga access token",
      quickstartTitle: "Mabilisang simula",
      columnName: "Pangalan",
      columnToken: "Token",
      columnCreated: "Ginawa",
      columnLastUsed: "Huling ginamit",
      columnExpires: "Mag-e-expire",
      loadError: "Hindi ma-load ang API access.",
      createError: "Hindi makagawa ng access token.",
      revokeError: "Hindi mabawi ang access token.",
      never: "Hindi kailanman",
      tokenNamePlaceholder: "Pangalan ng token",
      newAccessTokenName: "Pangalan ng bagong access token",
      createToken: "Gumawa ng token",
      neverExpiresHint: "Hindi kailanman mag-e-expire",
      expiresInDaysHint: "Mag-e-expire sa loob ng {n} araw",
      createTokenHint: 'Gumawa ng token na may pangalang "{name}"',
      defaultTokenName: "Bagong Token {n}",
      creating: "Ginagawa...",
      newTokenCreated: "Nagawa ang bagong access token",
      accessTokenCopied: "Nakopya ang access token",
      copyAccessToken: "Kopyahin ang access token",
      copyNow: "Kopyahin ngayon - hindi na ito ipapakita ulit.",
      usageExamples: "Chat API",
      usageNoModel:
        "Mag-load o mag-download ng model para makita ang mga halimbawang mapapatakbo. Wala pang model na mapapangalanan ang server na ito.",
      usageTools: "Mga tool",
      exampleLanguage: "Wika",
      exampleRequest: "Request",
      examplePlatform: "Platform",
      requestPlain: "Basic",
      requestTools: "May mga tool",
      requestAdvanced: "Advanced na sampling",
      exampleKind: "Halimbawa",
      osUnix: "Linux / macOS / WSL",
      osWindows: "Windows",
      secureHttps: "Secure na HTTPS",
      secureHttpsHint:
        "Naaabot pa rin sa buong mundo ang port na 0.0.0.0. Para sa ganap na seguridad, ilunsad ang Unsloth gamit ang --secure para ang HTTPS link lang na ito ang ma-expose.",
      copyTunnelUrl: "Kopyahin ang URL ng tunnel",
      copySnippet: "Kopyahin ang snippet",
      copy: "Kopyahin",
      copied: "Nakopya",
      codingAgents: "Mga coding agent",
      codingAgentsHint:
        "Maglunsad ng coding agent laban sa server na ito. Ginagamit nito ang na-load na model; awtomatikong gumagawa ng API key ang lokal na server, at isinasama ito ng remote na server sa command.",
      openAgentsTab: "Buksan sa Agents",
      codingAgentsSwap:
        "Palitan ang claude ng codex, openclaw, opencode, o hermes.",
      relativeNever: "hindi kailanman",
      relativeJustNow: "ngayon lang",
      expired: "expired na",
      expiredStatus: "Expired na",
      today: "ngayong araw",
      actionsFor: "Mga aksyon para sa {name}",
      copyPrefix: "Kopyahin ang prefix",
      copyFailed: "Hindi makopya",
      revokeToken: "Bawiin ang token",
      regenerateToken: "I-regenerate ang token",
      regenerateTitle: 'I-regenerate ang access token na "{name}"?',
      regenerateDescription:
        "Papalitan ito ng bagong token na may parehong pangalan at tagal ng expiry. Agad na mawawalan ng access ang mga app na gumagamit ng kasalukuyang token.",
      regenerateAction: "I-regenerate",
      regenerating: "Nire-regenerate...",
      regenerateError: "Hindi ma-regenerate ang access token.",
      revokeTitle: 'Bawiin ang access token na "{name}"?',
      revokeDescription:
        "Agad na mawawalan ng access ang mga app na gumagamit ng token na ito. Hindi na ito maaaring i-undo.",
      revokeAction: 'Bawiin ang "{name}"',
      revoking: "Binabawi...",
      audioApi: {
        title: "Audio API",
        description:
          "Speech, voice cloning, transcription at ang mga workflow ng Audio page.",
        speak: "Magsalita",
        clone: "I-clone",
        transcribe: "I-transcribe",
        workflows: "Mga workflow",
        separate: "Paghiwalayin",
        convert: "I-convert",
        music: "Musika",
        edit: "I-edit",
        placeholderModel:
          "Wala pang na-download na model na makakapagpatakbo ng halimbawang ito, kaya pansamantala lang ang ipinapakitang model. Mag-download muna ng isa sa Audio page.",
        autoSwitchOff:
          "Nilo-load ang model ayon sa pangalan lang habang naka-on ang Model auto-switch. I-on ito, o i-load muna ang model sa Audio page nito.",
        useViaApi: "Gamitin sa pamamagitan ng API",
      },
      mediaApi: {
        title: "Larawan / Video API",
        description:
          "Bumuo ng mga larawan at video mula sa sarili mong code gamit ang mga endpoint na compatible sa OpenAI.",
        images: "Mga larawan",
        video: "Video",
        placeholderModel:
          "Halimbawa lang ang ipinapakitang model. Mag-load ng model sa Images o Video page at buksan ito mula sa API button nito para mapunan ito.",
        autoSwitchOff:
          "Naka-off ang Media auto-switch, kaya tumatakbo ang mga request sa model na na-load ng page, anuman ang pangalang nakasaad sa mga ito.",
      },
      trainingApi: {
        title: "Training API",
        description:
          "I-fine-tune ang mga model mula sa sarili mong code gamit ang mga fine-tuning endpoint na compatible sa OpenAI: mag-upload ng JSONL file, gumawa ng job, sundan ang mga event nito, at pagkatapos ay makipag-chat sa na-train na model.",
        placeholder:
          "Halimbawa lang ang ipinapakitang model at mga setting. Buksan ito mula sa API button sa Train page para mapunan ang kasalukuyan mong setup.",
        hubDataset:
          "Gumagamit ang iyong Train page ng dataset mula sa Hugging Face o S3. Nagte-train ang API sa mga na-upload na file, kaya i-save ang dataset bilang JSONL at i-upload iyon.",
        batching:
          "Awtomatikong sabay na nagte-train ang mga compatible na job na naghihintay sa queue.",
        chatNote:
          "Para makipag-chat sa na-train na model ayon sa pangalan, kailangang naka-on ang Model auto-switch, o i-load muna ito sa Chat page.",
        cancelTitle: "Kanselahin ang isang job",
        cancelDescription:
          "Inihihinto ang job na nasa queue o tumatakbo. Hindi bahagi ng halimbawa sa itaas.",
      },
      decisionApi: {
        title: "Decision API",
        docs: "Docs",
        docsLabel: "Buksan ang docs ng Decision API",
        description:
          "Sagutin ang mga tanong na oo/hindi, multiple choice at score tungkol sa text gamit ang model sa machine na ito o isang decision model mula sa Mga Koneksyon. Gumagana sa TypeSafe SDK.",
        enable: "Mag-serve ng mga request",
        enableDescription:
          "Sine-serve ang /v1/systemone. Ang pag-on dito ay magda-download ng model.",
        enableRemoteDescription: "Sine-serve ang /v1/systemone.",
        lockedByEnv: "Itinakda ng {name}.",
        model: "Model",
        thisMachine: "Ang machine na ito",
        modelMultilingual: "Multilingual",
        modelEnglish: "English",
        modelTypedDecisions: "Mga typed decision",
        modelClefFlash: "Clef-flash",
        modelClef: "Clef",
        recommended: "Inirerekomenda",
        device: "Patakbuhin sa",
        deviceDescription:
          "Mas mabilis sumagot ang GPU at pinapalaya nito ang memory nito kapag na-unload ang model.",
        deviceCpu: "CPU",
        deviceGpu: "GPU",
        backend: "Runtime",
        backendAuto: "Auto",
        backendDescription:
          "Sine-serve ng Auto ang Clef sa pamamagitan ng llama.cpp kapag may GGUF build ang model, at PyTorch ang ginagamit kung wala. Nakakabasa rin ng mga larawan ang llama.cpp.",
        backendDescriptionMlx:
          "Sine-serve ng Auto ang Clef sa pamamagitan ng MLX sa Apple Silicon at sa pamamagitan ng llama.cpp kapag ang GGUF build lang ng model ang na-load o na-download. Nakakabasa lang ng mga larawan ang MLX para sa mga Clef model. Nakakabasa rin ng mga larawan ang llama.cpp.",
        backendStatus: "Runtime: {backend}",
        backendNone: "hindi available",
        mediaImages: "Nagbabasa ng text at mga larawan.",
        mediaText: "Text lang ang binabasa.",
        checking: "Sinusuri…",
        notDownloaded: "Hindi pa na-download · {size}",
        downloading: "Dina-download…",
        downloaded: "Na-download · ilo-load sa unang request",
        ready: "Handa na · nilo-load sa unang request",
        installing: "Ini-install…",
        loading: "Nilo-load…",
        loadedOn: "Na-load sa {device}",
        download: "I-download",
        downloadConfirmTitle: "I-download ang Laya {model}?",
        downloadConfirmTitleModel: "I-download ang {model}?",
        downloadConfirmBody:
          "Kailangan ng Decision API ang model na ito para sumagot sa mga request. Humigit-kumulang {size}, kinukuha nang isang beses papunta sa iyong Hugging Face cache.",
        unload: "I-unload",
        downloadBusy: "May model ng Decision API na dina-download na.",
        downloadFailed: "Hindi masimulan ang pag-download.",
        saveFailed: "Hindi ma-save ang setting ng Decision API.",
        loadError: "Hindi ma-load ang mga setting ng Decision API.",
        sendsTo: "Ipinapadala ang mga request sa {provider}.",
        connectionMissing:
          "Wala na ang koneksyong ito o wala itong mga decision model. Pumili ng ibang model.",
        addConnection:
          "Para gumamit ng decision model na naka-host sa cloud, magdagdag ng isa sa ",
        connectionsLink: "Mga Koneksyon",
        addConnectionSuffix: ".",
      },
      mcp: {
        title: "Access ng agent (MCP)",
        description:
          "Hayaang gamitin ng mga coding agent tulad ng Claude Code at Codex ang Unsloth sa pamamagitan ng MCP. Nagsa-sign in ang mga agent gamit ang Unsloth API key mula sa page na ito.",
        enable: "Payagan ang mga koneksyon ng agent",
        enableDescription:
          "Sine-serve ang /mcp/ sa mga request na may dalang Unsloth API key.",
        lockedByEnv: "Itinakda ng {name}.",
        loadError: "Hindi ma-load ang mga setting ng access ng agent.",
        saveError: "Hindi ma-save ang setting ng access ng agent.",
        agent: "Agent",
        exportKeyHint:
          "Itakda ang {name} sa isang Unsloth API key mula sa page na ito bago mo simulan ang agent.",
        configFileHint: "Idagdag ito sa {path}.",
        keywords:
          "mcp model context protocol agents claude codex tools mga tool",
      },
    },
    about: {
      title: "Tungkol sa",
      description: "Docs, release notes, feedback, at impormasyon ng build.",
      studioVersion: "Bersyon ng Unsloth",
      packageVersion: "Bersyon ng Package",
      desktopAppVersion: "Bersyon ng Desktop App",
      desktopAppVersionUnavailable: "Hindi available",
      llamaCppVersion: "Bersyon ng llama.cpp",
      audioCppVersion: "Bersyon ng audio.cpp",
      mlxVersion: "Bersyon ng MLX",
      whisperCppVersion: "Bersyon ng whisper.cpp",
      engines: {
        title: "Mga Engine",
        description:
          "Ang mga runtime na ginagamit ng Unsloth para magpatakbo ng mga model. Ini-install ang mga update sa background, nang walang restart.",
        updatesLabel: "Mga update ng engine",
        notInstalled: "Hindi naka-install",
        available: "Available ang {version}",
        upToDate: "Napapanahon",
        updatesAvailable: "May mga available na update",
        checking: "Tumitingin ng mga update...",
        checkFailed: "Hindi makatingin ng mga update. Subukan ulit mamaya.",
        unsupported:
          "Pinamamahalaan sa labas ng Unsloth ang mga engine ng install na ito.",
        checkForUpdates: "Tumingin ng mga update",
        update: "I-update",
        updating: "Ina-update...",
        updated: "Na-update ang mga engine.",
        updateFailed: "Nabigo ang pag-update: {error}",
      },
      hardware: "Hardware",
      gpu: "GPU",
      gpuIntegrated: "integrated",
      cuda: "CUDA",
      rocm: "ROCm",
      xpu: "XPU",
      updates: "Update",
      help: "Tulong",
      documentation: "Dokumentasyon",
      releaseNotes: "Mga release note",
      whatsNew: "Ano ang bago",
      feedback: "Feedback",
      reportIssue: "Mag-ulat ng isyu",
      license: {
        sectionTitle: "Lisensya",
        studioLabel: "Unsloth",
        studioLicense: "AGPL-3.0",
        studioDescription: "Open source sa ilalim ng GNU AGPL v3.0.",
        libraryLabel: "Unsloth Core",
        libraryLicense: "Apache-2.0",
        libraryDescription: "Lisensyado sa ilalim ng Apache 2.0.",
      },
      dangerZone: "Mapanganib na bahagi",
      shutDownStudio: "I-shut down ang Unsloth",
      shutDownStudioDescription:
        "Inihihinto ang server ng Unsloth at tinatapos ang iyong session.",
      shutDown: "I-shut down",
      update: {
        title: "I-update ang Unsloth",
        commandText: "Text ng {label}",
        copied: "Nakopya",
        copyCommand: "Kopyahin ang command",
        commandCopied: "Nakopya ang {label}",
        copyNamedCommand: "Kopyahin ang {label}",
        checkingInstall: "Tinitingnan kung paano na-install ang Unsloth...",
        installIntro: "Para i-install o i-update ang Unsloth:",
        localUpdateHeading: "Lokal na update",
        installCommandUnix: "Command sa pag-install para sa macOS/Linux",
        installCommandWindows: "Command sa pag-install para sa Windows",
        localInstallDetected:
          "May na-detect na lokal na install. Mag-update mula sa orihinal mong checkout para hindi ito mapalitan ng PyPI.",
        pullThenUpdate:
          "I-pull ang pinakabagong mga pagbabago, at pagkatapos ay patakbuhin ang lokal na installer:",
        gitPullCommand: "command na git pull",
        localInstallerCommand: "command ng lokal na installer",
        sourceInstallDetected:
          "May na-detect na install mula sa source o VCS package. Mag-reinstall mula sa orihinal na lokal na path o Git URL.",
        repoCheckoutFallback:
          "Kung nasa iyo pa ang checkout ng repo, patakbuhin ang lokal na installer mula roon:",
        restartAfterUpdate: "I-restart ang Unsloth pagkatapos mag-update.",
        desktopManaged:
          "Awtomatikong tumitingin ang desktop app ng mga bagong bersyon ng app. Maaari ka ring tumingin o mag-update dito anumang oras.",
        desktopReady: "Mga update ng desktop app",
        desktopReadyDescription:
          "Tingnan kung may available na mas bagong bersyon ng desktop app.",
        desktopChecking: "Tumitingin ng mga update",
        desktopCheckingDescription: "Karaniwang ilang segundo lang ito.",
        desktopAvailable: "Available ang desktop app {version}",
        desktopAvailableDescription:
          "Mag-update ngayon at magre-restart ang desktop app kapag natapos ito.",
        desktopExternalServer:
          "Nakakonekta ang app sa isang server ng Unsloth na tumatakbo na at hindi nito ito ma-update. Ihinto ang server na iyon, pagkatapos ay umalis at buksan ulit ang desktop app para mag-update.",
        desktopManualInstall:
          "Buksan ang release page para i-install ang pinakabagong Linux package.",
        desktopCheckFailed: "Hindi makatingin ng mga update",
        desktopCheckFailedDescription:
          "Suriin ang iyong koneksyon at subukan ulit.",
        desktopCurrent: "Napapanahon ang desktop app",
        desktopCurrentDescription:
          "Patuloy na awtomatikong titingin ang Unsloth.",
        checkForUpdates: "Tumingin ng mga update",
        checkAgain: "Tumingin ulit",
        retryCheck: "Subukan ulit",
        checking: "Tumitingin...",
        updateNow: "Mag-update ngayon",
        openReleasePage: "Buksan ang release page",
        unknownInstall:
          "Hindi ma-detect kung paano na-install ang Unsloth. Para sa mga install gamit ang installer o PyPI, gamitin ang mga command sa itaas.",
        localCheckout:
          "Para sa mga install mula sa lokal na checkout, patakbuhin ang lokal na installer mula sa checkout na iyon:",
        docs: "Docs sa pag-install:",
        docsInstall: "Pag-install",
        docsUpdating: "Pag-update",
        docsMac: "Mac",
        docsWindows: "Windows",
        docsLinux: "Linux",
        docsDocker: "Docker",
        docsAmd: "AMD",
      },
    },
  },
  studio: {
    routeTitle: "Train",
    wizard: {
      modelTitle: "Model",
      modelDescription: "Pumili ng model at paraan ng training",
      datasetTitle: "Dataset",
      datasetDescription: "Pumili o mag-upload ng training data",
      paramsTitle: "Mga Parameter",
      paramsDescription: "I-configure ang mga parameter ng training",
      configTitle: "Configuration",
      configDescription: "I-save at i-load ang mga configuration",
      modelLabel: "Model",
      modelTooltip: "Ang base model na gusto mong i-fine-tune.",
      methodLabel: "Paraan",
      methodTooltip:
        "Kung paano tine-train ang model. Ina-update ng LoRA at QLoRA ang maliliit na adapter sa halip na bawat weight.",
      trainAsLabel: "I-train bilang",
      trainAsTooltip:
        "Ang language model ay sumusulat ng text. Ang decision model ay pumipili ng isa sa mga opsyong ibinigay mo, kasama ang probability, at tumatakbo sa Decision API.",
      trainAsLanguage: "Language model",
      trainAsDecision: "Decision model",
      checkpointLabel: "Checkpoint",
      checkpointTooltip:
        "Ang Laya checkpoint na ifa-fine-tune. Bagay ang Multilingual sa karamihan ng dataset.",
      datasetLabel: "Dataset",
      datasetTooltip:
        "Ang training data na ginagamit para i-fine-tune ang model.",
      hfTokenDescription:
        "Kailangan para sa mga gated o pribadong model at dataset.",
      uploadLocalLabel: "Mag-upload ng lokal na file",
      sourceBrowse: "Mag-browse",
      releaseToUpload: "Bitawan para mag-upload",
      loadYaml: "I-load ang YAML",
      saveYaml: "I-save ang YAML",
      resetDefaults: "I-reset sa mga default",
      cachedModelGoneTitle: "Hindi na available ang naka-cache na model",
      cachedModelGoneDescription:
        "Wala na sa device na ito ang mga file ng model, kaya ida-download ulit ng training ang mga ito.",
      cachedDatasetGoneTitle: "Hindi na available ang naka-cache na dataset",
      cachedDatasetGoneDescription:
        "Wala na sa device na ito ang mga file ng dataset, kaya ida-download ulit ng training ang mga ito.",
    },
    preview: {
      title: "Preview ng run",
      ready: "Handa na",
      notReady: "Hindi pa handa",
      modelPending: "Hinihintay ang model",
      datasetPending: "Hinihintay ang dataset",
      method: "Paraan",
      length: "Haba",
      stepZero: "{count} step",
      step: "{count} step",
      stepTwo: "{count} step",
      stepFew: "{count} step",
      stepMany: "{count} step",
      steps: "{count} step",
      epochZero: "{count} epoch",
      epoch: "{count} epoch",
      epochTwo: "{count} epoch",
      epochFew: "{count} epoch",
      epochMany: "{count} epoch",
      epochs: "{count} epoch",
      batch: "Batch",
      context: "Context",
      lr: "LR",
      hardware: "Hardware",
      vram: "VRAM",
      noGpu: "Walang na-detect na GPU",
      hfToken: "HF token",
      saved: "Naka-save",
      notSet: "Hindi nakatakda",
      files: "Mga file",
      model: "Model",
      dataset: "Dataset",
      downloadsOnStart: "Ida-download sa pagsisimula",
      continuesOnStart: "Itutuloy sa pagsisimula",
      noticeModelDownload:
        "Wala pa sa device na ito ang model na ito. Awtomatiko itong ida-download ng training.",
      noticeModelPartial:
        "Kukumpletuhin ng training ang bahagyang pag-download ng model bago ito i-load.",
      noticeDatasetDownload:
        "Wala pa sa device na ito ang dataset na ito. Awtomatiko itong ida-download ng training.",
      noticeDatasetPartial:
        "Kukumpletuhin ng training ang bahagyang pag-download ng dataset bago ito basahin.",
      noticeTransformersUpgrade:
        "Wala pang naka-install na transformers na sumusuporta sa architecture na ito. Sa pagsisimula ng run, aalukin kang i-install muna ang transformers {version}.",
      noticeSixteenBitOnly:
        "Nagte-train ang architecture na ito sa 16-bit LoRA: hindi available ang 4-bit para dito, kaya mas maraming VRAM ang kailangan ng run kaysa sa QLoRA.",
      noticeInstallSwitchesSixteenBit:
        "Ang pag-install sa release na iyon sa halip na panatilihin ang sariling code ng model ay maglilipat sa run na ito sa 16-bit LoRA, na nangangailangan ng mas maraming VRAM kaysa sa QLoRA.",
      advancedSettings: "Mga advanced na setting",
      defaultAdvancedSettings: "Mga default",
      nonDefaultAdvancedSettings: "{count} hindi default",
    },
    datasetPicker: {
      noun: "mga dataset",
      selectDataset: "Pumili ng dataset",
      hubPlaceholder: "Maghanap ng mga dataset sa Hugging Face...",
      devicePlaceholder: "Maghanap ng mga lokal na dataset...",
      useAsHubDataset: "Gamitin bilang Hugging Face dataset",
      hfCacheLabel: "HF cache",
      scanningLocal: "Ini-scan ang mga dataset sa device na ito…",
      couldntScan: "Hindi ma-scan ang mga lokal na dataset",
      someLocationsUnscanned: "Hindi ma-scan ang ilang lokasyon ng dataset.",
      noLocalDatasets:
        "Wala pang laman ang device na ito. Mag-download ng dataset mula sa Hub, gumawa ng isa sa Recipes, o mag-upload ng file.",
      openDataRecipes: "Buksan ang Data Recipes",
      searchingHub: "Naghahanap sa Hugging Face…",
      noDatasetsFound: "Walang nakitang dataset.",
      tokenRejectedTitle: "Tinanggihan ang token ng Hugging Face",
      tokenRejectedBody:
        "I-update ang iyong token sa Mga Setting → Pangkalahatan, at subukan ulit.",
      hubUnreachable: "Hindi maabot ang Hugging Face",
      cantUseDataset: "Hindi magamit ang dataset",
      reasonInvalidHubId:
        "Maglagay ng valid na ID ng dataset sa Hugging Face: repo o owner/repo, walang magkakasunod na tuldok o gitling at walang .git suffix (hanggang 96 na character bawat bahagi).",
      sourceRecipe: "Recipe",
      sourceUpload: "Upload",
      sourceLocal: "Lokal",
    },
    modelPicker: {
      noun: "mga model",
      selectModel: "Pumili ng model",
      hubPlaceholder: "Maghanap o mag-paste ng Hugging Face id...",
      devicePlaceholder:
        "Maghanap ng mga lokal na model o mag-paste ng path ng folder...",
      useAsHubModel: "Gamitin bilang Hugging Face model",
      useAsLocalPath: "Gamitin bilang lokal na path",
      hfCacheLabel: "HF cache",
      scanningLocal: "Ini-scan ang mga lokal na model…",
      couldntScan: "Hindi ma-scan ang mga lokal na model",
      someLocationsUnscanned: "Hindi ma-scan ang ilang lokal na lokasyon.",
      noLocalModels: "Walang nakitang lokal na model.",
      noLocalModelsHint:
        "Mag-paste ng path ng folder sa itaas o lumipat sa Hugging Face.",
      searchingHub: "Naghahanap sa Hugging Face…",
      noModelsFound: "Walang nakitang model.",
      tokenRejectedTitle: "Tinanggihan ang token ng Hugging Face",
      tokenRejectedBody:
        "I-update ang iyong token sa Mga Setting → Pangkalahatan, at subukan ulit.",
      tokenRejectedAnonymousBody:
        "Ipinapakita ang mga pampublikong model nang wala ito. I-update ang iyong token sa Mga Setting → Pangkalahatan para maabot ang mga pribado at gated na model.",
      hubUnreachable: "Hindi maabot ang Hugging Face",
      cantUseModel: "Hindi magamit ang model para sa training",
      reasonTypeMismatch:
        "Hindi tugma ang model na ito sa uri ng training na pinili sa nakaraang hakbang.",
      reasonEmptyId: "Maglagay ng model id o lokal na path ng model.",
      reasonGguf: "Hindi magagamit ang mga GGUF model para sa training.",
      reasonAdapter:
        "Hindi magagamit ang mga output na adapter bilang base model sa training.",
      reasonNotTrainable: "Hindi maaaring i-train ang on-device model na ito.",
      reasonUnsupportedFormat:
        "Hindi sinusuportahan para sa training ang format ng model na ito.",
      reasonInvalidHubId:
        "Maglagay ng valid na ID ng model sa Hugging Face: repo o owner/repo, walang magkakasunod na tuldok o gitling at walang .git suffix (hanggang 96 na character bawat bahagi).",
      sourceModelsFolder: "Folder ng mga model",
      sourceHfCache: "HF cache",
      sourceLmStudio: "LM Studio",
      sourceOllama: "Ollama",
      sourceHermes: "Hermes",
      sourceCustomFolder: "Custom na folder",
      sourceLocalModel: "Lokal na model",
      vramOomBadge: "OOM",
      vramTightBadge: "Sikip",
      vramNeeds: "Kailangan ng ~{est}GB VRAM (GPU: {total}GiB)",
      vramTight: "~{est}GB VRAM (sikip sa {total}GiB)",
      vramApprox: "~{est}GB VRAM",
    },
    methods: {
      qlora: {
        label: "QLoRA",
        hint: "4-bit quantization. Pinakamababang VRAM, pinakamabilis magsimula.",
        note: "4-bit",
      },
      lora: {
        label: "LoRA",
        hint: "16-bit na adapter. Balanseng kalidad at memory.",
        note: "16-bit",
      },
      full: {
        label: "Buong fine-tune",
        hint: "Tine-train ang lahat ng weight. Pinakamataas na kalidad, pinakamaraming VRAM ang kailangan.",
        note: "fp16",
      },
      cpt: {
        label: "Continued pretraining",
        hint: "Continued pretraining para sa mga bagong domain o wika.",
        note: "continued",
      },
    },
    subtitles: {
      configure: "I-configure at simulan ang training",
      trainingInProgress: "Kasalukuyang nagte-train",
      viewPastRuns: "Tingnan ang mga nakaraang training run",
      viewingPastRun: "Tinitingnan ang nakaraang run",
    },
    tabs: {
      configure: "I-configure",
      currentRun: "Kasalukuyang Run",
      history: "History",
    },
    goToLibrary: "Pumunta sa Library",
    imageTraining: "Training ng larawan",
    goToImageTraining: "Mag-train ng mga image LoRA",
    loadingRuntime: "Nilo-load ang training runtime...",
    checkingSupport:
      "Sinusuri ang machine na ito para sa suporta sa training...",
    backToHistory: "Bumalik sa history",
    dataset: {
      selectors: {
        subset: "Subset",
        subsetTooltip:
          "Piliin kung aling subset (config) ng dataset ang gagamitin.",
        trainSplit: "Split ng training",
        trainSplitTooltip:
          "Piliin kung aling split ang gagamitin para sa training.",
        evaluationSplit: "Split ng evaluation",
        evaluationSplitTooltip:
          'Piliin kung aling split ang gagamitin para sa evaluation. Ang "Wala" ay nangangahulugang walang evaluation habang nagte-train.',
        selectSubset: "Pumili ng subset...",
        selectSplit: "Pumili ng split...",
        none: "Wala",
        loading: "Nilo-load ang mga config at split ng dataset...",
        manualTitle: "Manu-manong ilagay ang mga opsyon ng dataset",
        manualDescription:
          "Ilagay ang eksaktong pangalan ng config at split ng Hugging Face na gagamitin.",
        manualSubsetPlaceholder: "Opsyonal na pangalan ng config",
        manualRequired: "Kailangan ng split para sa training.",
        manualTooLong: "Gumamit ng 128 character o mas kaunti.",
        manualInvalid:
          "May mga hindi sinusuportahang character ang value na ito.",
      },
      sourceAriaLabel: "Source ng dataset",
      localDataset: "Lokal na dataset",
      localDatasetRows: " / {count} row",
      huggingFaceDataset: "Dataset ng Hugging Face",
      localDatasetMetadata: "Metadata ng lokal na dataset",
      dataRecipeOutput: "Output ng Data Recipe.",
      rows: "Mga row",
      columns: "Mga column",
      batches: "Mga batch",
      updated: "Na-update",
      evalDataset: "Eval dataset",
      uploading: "Ina-upload...",
      uploadEvalFile: "Mag-upload ng eval file",
      fileTooLarge: "Masyadong malaki ang file",
      fileTooLargeDescription:
        "{size} ang {file}. Hanggang {limit} lang ang sinusuportahan ng mga upload para sa training.",
      documentRedirect: {
        title: "Kailangan munang i-convert ang file na ito",
        genericFile: "Ang file na ito",
        description:
          "Source material ang {file}, hindi dataset na handa nang i-train. Gamitin ang Data Recipes para gawing dataset ang document, at bumalik dito para mag-fine-tune.",
        nextStepTitle: "Pinakamainam na susunod na hakbang",
        nextStepDescription:
          "Buksan ang Learning Recipes at magsimula sa isang recipe na batay sa document tulad ng PDF grounded QA.",
        openAction: "Buksan ang Learning Recipes",
      },
      evalDatasetDescription:
        "Opsyonal. Kung walang ibinigay, hahatiin ang maliit na bahagi mula sa training data.",
      advanced: "Advanced",
      targetFormat: "Target na Format",
      targetFormatTooltip:
        "Format ng iyong training data. Gumagana ang Auto-detect para sa karamihan ng dataset.",
      streamingInfoAriaLabel: "Impormasyon sa streaming ng dataset",
      streaming: {
        label: "I-enable ang streaming",
        description:
          "I-stream ang mga text dataset ng Hugging Face sa halip na i-download ang mga ito.",
        unavailable: "Hindi available ang streaming. Para i-enable:",
        completionsUnavailable:
          "Hindi available habang naka-enable ang streaming ng dataset.",
        blockers: {
          source:
            "Gumamit ng dataset ng Hugging Face (hindi lokal na upload o S3 source).",
          maxSteps:
            "Itakda ang Max Steps > 0 — walang alam na haba ang mga streaming dataset.",
          trainOnCompletions: 'I-off ang "Mga completion ng assistant lang".',
          evalSplit:
            "Pumili ng hiwalay na eval split — naka-on ang evaluation pero walang nakatakdang hiwalay na eval split.",
          visionModel:
            "Hindi sinusuportahan ng mga vision model ang streaming.",
          audioModel: "Hindi sinusuportahan ng mga audio model ang streaming.",
          embeddingModel:
            "Hindi sinusuportahan ng mga embedding model ang streaming (kailangan ng training ang buong dataset).",
          imageDataset:
            "Mukhang mga larawan ang dataset na ito, na hindi maaaring i-stream.",
          audioDataset:
            "Mukhang audio ang dataset na ito, na hindi maaaring i-stream.",
          appleSilicon:
            "Hindi pa sinusuportahan ang streaming sa Apple Silicon (MLX).",
        },
        options: {
          trainOnCompletions: "mga completion ng assistant lang",
          evaluation: "evaluation (kailangan ng hiwalay na eval split)",
        },
        notifications: {
          turnedOffMaxSteps:
            "Na-off ang streaming: kailangan ng streaming ang nakapirming Max Steps > 0.",
          adjusted:
            "Inayos para sa streaming. Na-disable ang mga hindi compatible na opsyon: {options}.",
          needsMaxSteps:
            "Kailangan ng streaming ang nakapirming Max Steps (walang alam na haba ang mga streaming dataset). Itakda muna ang Max Steps > 0.",
          enabledAdjusted:
            "Na-enable ang streaming. Na-disable ang mga hindi compatible na opsyon: {options}.",
          disabledForDetectedModality:
            "Na-off ang streaming dahil kailangan ng buong pag-download ang mga image at audio dataset. Suriin ang setting, at magsimula ulit.",
        },
      },
      auto: "Auto",
      rawText: "Raw na Text",
      trainSplitStart: "Simula ng Train Split",
      trainSplitStartTooltip:
        "Mag-train lang sa isang bahagi ng iyong training split sa pamamagitan ng pagtukoy ng index ng panimulang row (kasama, nagsisimula sa 0). Iwanang blangko para magsimula sa unang row.",
      trainSplitEnd: "Dulo ng Train Split",
      trainSplitEndTooltip:
        "Index ng huling row na isasama mula sa training split (kasama, nagsisimula sa 0). Halimbawa, itakda ang Simula sa 0 at Dulo sa 99 para mag-train sa unang 100 row. Iwanang blangko para gamitin ang lahat ng natitirang row.",
      endPlaceholder: "Dulo",
      clear: "I-clear",
      dropFileOrClick: "I-drop ang file o i-click para mag-upload",
      uploadDetails: "Mga detalye ng upload",
      uploadDetailsTooltip:
        "Hanggang {limit} bawat file. Hindi handang i-train na dataset ang PDF, DOCX at TXT, kaya i-convert muna ang mga ito sa Learning Recipes.",
      viewDataset: "Tingnan ang dataset",
      previewTokens: "I-preview ang mga token",
      tokenPreview: {
        title: "Preview ng token",
        description:
          "Kung paano tino-tokenize ng {model} ang data na ito para sa training. Ang tokenizer lang ang ida-download.",
        datasetTab: "Mga sample ng dataset",
        textTab: "Ang iyong text",
        trained: "Tine-train",
        masked: "Naka-mask",
        completionsOnly: "Mga completion ng assistant lang",
        allTokens: "Tine-train ang lahat ng token",
        responseMarker: "Marker ng sagot",
        maskingNotApplied:
          "Hindi nakita ang mga chat marker, kaya tine-train ang bawat token.",
        loading:
          "Nilo-load ang tokenizer. Medyo mas matagal ang unang preview...",
        shuffle: "I-shuffle ang mga sample",
        showIds: "Ipakita ang mga token ID",
        copyIds: "Kopyahin ang mga ID",
        copied: "Nakopya ang mga token ID",
        sample: "Sample {index}",
        stats: "{tokens} token, {trained} ang tine-train ({percent}%)",
        truncated: "Pinutol sa {max} sa {full} token dahil sa haba ng context.",
        noTrained:
          "Walang token sa row na ito na tine-train, kaya nilalaktawan ito ng training.",
        hiddenTokens: "{count} pang token ang hindi ipinapakita",
        multimodal:
          "Hindi ipinapakita rito ang mga row ng larawan at audio. Maaari mo pa ring subukan ang tokenizer sa sarili mong text.",
        textPlaceholder:
          "Mag-type o mag-paste ng text para makita ang mga token nito...",
        textHint:
          'Kapag naka-on ang "Mga completion ng assistant lang", ang text lang pagkatapos ng assistant marker ang tine-train.',
        insertExample: "Maglagay ng halimbawang chat",
        vocab: "Vocabulary: {count} token",
        retry: "Subukan ulit",
        packing:
          "Naka-on ang packing: pinagsasama ng run ang mga row sa mga sequence na kasinghaba ng context, kaya maaaring iba ang bahaging pinuputol sa isang row kumpara sa view na ito na bawat row.",
      },
      uploadFailed: "Nabigo ang pag-upload",
      unknownError: "Hindi kilalang error",
      unsupportedFileType: "Hindi sinusuportahang uri ng file",
      uploadOneFileType: "Mag-upload ng isang {types} file.",
      datasetUploaded: "Na-upload ang dataset",
      evalDatasetUploaded: "Na-upload ang eval dataset",
      uploadOneFileAtATime: "Mag-upload ng isang file sa bawat pagkakataon",
      uploadSingleFileDescription:
        "Isang file lang ang tinatanggap ng pag-upload ng training dataset.",
      previewLoadingHuggingFace:
        "Kinukuha ang preview ng dataset mula sa Hugging Face...",
      previewLoading: "Nilo-load ang preview...",
      mappingRequirements: {
        audioAndText: "audio at text",
        imageAndText: "larawan at text",
        instructionAndOutput: "instruction at output",
        humanAndGpt: "human at GPT",
        userAndAssistant: "user at assistant",
      },
      mappingStatus: {
        heuristicTitle: "Mapping na na-detect gamit ang heuristics",
        readyTitle: "Handa na ang mapping",
        requiredTitle: "I-map ang mga column ng dataset",
        heuristicDescription:
          "Awtomatiko naming na-detect ang column mapping sa ibaba gamit ang heuristics. Pakisuri at ayusin gamit ang mga dropdown sa mga header ng column, o gamitin ang AI Assist para sa mas matalinong mapping.",
        readyDescription:
          "Mukhang maayos. Awtomatiko naming iko-convert ang dataset na ito.",
        requiredDescription:
          "Magtalaga ng mga role sa mga column gamit ang mga dropdown sa mga header. Sa pinakamababa, italaga ang {required}.",
      },
      s3: {
        title: "Configuration ng S3",
        description:
          "Mag-load ng mga .parquet, .json, .jsonl, o .csv na dataset mula sa Amazon S3",
        bucket: "Pangalan ng Bucket",
        bucketPlaceholder: "my-training-data-bucket",
        region: "Rehiyon ng AWS",
        regionPlaceholder: "us-east-1",
        prefix: "Prefix ng Path",
        prefixPlaceholder: "datasets/whisper/",
        accessKeyId: "Access Key ID",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Secret Access Key",
        secretAccessKeyPlaceholder: "Ang iyong AWS secret access key",
        useIamRole: "Gamitin ang IAM Role",
      },
    },
    params: {
      mode: {
        simple: "Simple",
        advanced: "Advanced",
        ariaLabel: "Mode ng parameter",
      },
      projectName: "Pangalan ng Proyekto",
      optional: "Opsyonal",
      projectNameDescription:
        "Ginagamit sa mga pangalan ng output folder ng training, mga default ng pag-export, at history.",
      loraSettings: "Mga Setting ng LoRA",
      trainingHyperparameters: "Mga Hyperparameter ng Training",
      drafterSettings: "Drafter para sa Speculative Decoding",
      drafterTrain: "Mag-train ng drafter para sa speculative decoding",
      drafterTrainTooltip:
        "Nagte-train ng maliit na draft head pagkatapos ng fine-tuning para mas mabilis tumakbo ang inference sa llama.cpp at vLLM; hindi nagbabago ang mga output.",
      drafterChecking: "Sinusuri ang model na ito...",
      drafterUnavailable: "Hindi available para sa model na ito: {reason}",
      drafterRetrainMtp: "Nire-retrain ang MTP head ng model na ito",
      drafterEagle3:
        "Nagte-train ng EAGLE-3 drafter (walang MTP head ang model na ito)",
      drafterDataSource: "Training data",
      drafterDataSourceDataset: "Mga sagot ng dataset",
      drafterDataSourceRegenerate:
        "Sariling mga sagot ng model (mas mabagal, mas mahusay)",
      drafterMaxExamples: "Mga halimbawa",
      drafterEpochs: "Mga epoch",
      drafterLearningRate: "Learning rate",
      drafterLearningRateDefault: "Default",
      drafterDraftSteps: "Mga draft token",
      maxSteps: "Max Steps",
      epochs: "Mga epoch",
      useMaxSteps: "Gamitin ang Max Steps",
      useEpochs: "Gamitin ang Epochs",
      maxStepsTooltip: "I-override ang kabuuang bilang ng optimizer step.",
      epochsTooltip: "Bilang ng buong pagdaan sa dataset.",
      contextLength: "Haba ng Context",
      contextLengthTooltip: "Maximum na bilang ng token bawat training sample.",
      customContextLength: "Maglagay ng custom na value",
      learningRate: "Learning Rate",
      learningRateTooltip:
        "Laki ng step para sa mga update ng weight. Mas mabagal pero mas matatag mag-train ang mas mababang value.",
      learningRateDescription:
        "Inirerekomenda: 2e-4 para sa LoRA, 5e-5 para sa CPT, 2e-5 para sa buong fine-tune",
      embeddingLearningRate: "Learning Rate ng Embedding",
      embeddingLearningRateTooltip:
        "Ginagamit lang kapag tine-train ng CPT ang embed_tokens. Mas madaling maging hindi matatag ang mga embedding kaysa sa mga LoRA weight, kaya karaniwang kailangan nila ng mas maliit na LR. Iwanang blangko para gamitin ang lr/10; ang karaniwang gumaganang saklaw ay 2x-10x na mas maliit kaysa sa pangunahing LR. Taasan lang ito kung masyadong mabagal ang adaptation ng vocabulary o domain token.",
      rank: "Rank",
      rankTooltip:
        "Dimensyon ng mga low-rank matrix. Mas mataas = mas malaking kapasidad.",
      alpha: "Alpha",
      alphaTooltip:
        "Scaling factor para sa mga update ng LoRA. Karaniwang 2x ng rank.",
      dropout: "Dropout",
      dropoutTooltip:
        "Probability ng dropout para sa mga LoRA layer para mabawasan ang overfitting.",
      visionLayers: "Mga vision layer",
      languageLayers: "Mga language layer",
      attentionModules: "Mga attention module",
      mlpModules: "Mga MLP module",
      targetModules: "Mga Target Module",
      enableLora: "LoRA",
      trainWithLora: "Mag-train gamit ang LoRA",
      stableRank: "LoRA na rank-stabilized",
      memoryEfficient: "LoRA na SVD-initialized",
      weightDecomposed: "LoRA na may mga multiplier",
      notSupportedAppleSilicon: "Hindi sinusuportahan sa Apple Silicon",
      doraNeedsVisionLayersOff:
        "I-off ang training ng vision layer para magamit ang DoRA",
      optimization: "Optimization",
      schedule: "Iskedyul",
      memory: "Memory",
      optimizer: "Optimizer",
      optimizerTooltip:
        "Algorithm ng optimization. Binabawasan ng mga 8-bit na variant ang paggamit ng memory. Inirerekomenda ang Fused para sa mga vision model.",
      optimizerTooltipMlx:
        "Algorithm ng optimization. AdamW ang default. Mas kaunting memory ang ginagamit ng Lion pero kadalasang kailangan nito ng mas mababang learning rate.",
      lrScheduler: "LR Scheduler",
      lrSchedulerTooltip:
        "Kung paano nagbabago ang learning rate sa buong training. Ang Linear ay bumababa nang tuloy-tuloy; ang cosine ay bumababa nang pakurba.",
      optimizerOptions: {
        adamw8bit: "AdamW 8-bit",
        pagedAdamw8bit: "Paged AdamW 8-bit",
        adamwBnb8bit: "AdamW BNB 8-bit",
        pagedAdamw32bit: "Paged AdamW 32-bit",
        adamwTorch: "AdamW (PyTorch)",
        adamwTorchFused: "AdamW (PyTorch Fused)",
      },
      lrSchedulerOptions: {
        linear: "Linear",
        cosine: "Cosine",
      },
      batchSize: "Batch size",
      batchSizeTooltip:
        "Mga sample na pinoproseso bawat step. Mas mataas, mas maraming VRAM ang gamit.",
      gradAccum: "Mga grad accum step",
      gradAccumTooltip:
        "Ginagaya ang mas malalaking batch size nang walang dagdag na VRAM.",
      weightDecay: "Weight decay",
      weightDecayTooltip: "L2 regularization para maiwasan ang overfitting.",
      warmupSteps: "Mga warmup step",
      warmupStepsTooltip:
        "Unti-unting itaas ang LR sa simula ng training para sa stability.",
      scheduleEpochsTooltip:
        "Bilang ng buong pagdaan sa dataset. Itakda sa 0 para patakbuhin ayon sa max steps.",
      saveSteps: "Mga save step",
      saveStepsTooltip:
        "Mag-save ng checkpoint bawat N step. 0 para i-disable.",
      evalSteps: "Mga eval step",
      evalStepsTooltip:
        "Bahagi ng kabuuang training step sa pagitan ng mga evaluation (0-1). Itakda sa 0 para i-disable ang evaluation. Hal. 0.01 = mag-evaluate bawat 1% ng mga step.",
      earlyStoppingPatience: "Early stopping patience",
      earlyStoppingPatienceTooltip:
        "Ihinto ang training pagkatapos ng ganito karaming sunod-sunod na evaluation nang walang mas mababang eval loss, at panatilihin ang mga weight mula sa pinakamahusay na evaluation. Kailangan ng evaluation (Eval Steps na higit sa 0). 0 para i-disable.",
      seed: "Seed",
      seedTooltip: "Random seed para sa reproducibility.",
      offloadLayers: "I-offload ang mga layer",
      offloadLayersTooltip:
        "Panatilihin ang mga decoder layer sa system RAM at i-stream ang bawat isa sa GPU bago ito tumakbo, para ma-train gamit ang LoRA ang model na mas malaki kaysa sa iyong VRAM. Ang Auto ay nag-o-offload ng pinakakaunting kasya; ang bilang ay nag-o-offload ng eksaktong ganoon karami. Kailangan ng gradient checkpointing.",
      offloadOff: "Naka-off",
      offloadAuto: "Auto",
      offloadCount: "Bilang",
      offloadVramBudget: "Budget ng VRAM (GiB)",
      offloadVramBudgetTooltip:
        "Ang pinakamaraming VRAM na puwedeng gamitin ng run na ito. Nag-o-offload ang Auto ng sapat na layer para magkasya sa ilalim nito, at puwedeng maghati ang dalawang run sa iisang card kung kukuha ang bawat isa ng bahagi nito. Kapag walang laman, gagamitin ang buong card.",
      offloadWholeCard: "buong card",
      offloadVramBudgetGpu: "VRAM budget ng GPU {index} (GiB)",
      prefetchDepth: "Lalim ng prefetch",
      prefetchDepthTooltip:
        "Ilan sa mga na-offload na layer ang kinokopya nang mas maaga sa tumatakbo. Nagsisimula ang Auto sa 1 at nagpapanatili lang ng mas malalim na pool habang nasusukat na binabawasan nito ang paghihintay sa pagkopya.",
      offloadPanelTitle: "Mga na-offload na layer",
      offloadPanelGpu: "Nasa GPU",
      offloadPanelHost: "Nasa system RAM",
      offloadPanelCopying: "Kinokopya",
      offloadPanelSwapped: "na-offload",
      offloadPanelStall: "naghihintay sa pagkopya",
      offloadPanelCopy: "pagkopya bawat layer",
      offloadPanelCompute: "compute bawat layer",
      offloadPanelVram: "VRAM",
      offloadPanelDepth: "lalim ng prefetch",
      offloadPanelPinned: "naka-pin",
      offloadPanelSweepNote:
        "Ipinapakita ng gumagalaw na window ang pagkakasunod-sunod ng pagkuha sa mga layer, pinabagal para makita; ang mga numero ay sinukat sa huling step.",
      offloadPanelCard: "GPU {index}",
      offloadPanelCardUnknown: "Hindi kilalang card",
      gradCheckpoint: "Grad checkpointing",
      gradCheckpointTooltip:
        "Ipagpalit ang compute para sa memory sa pamamagitan ng muling pag-compute ng mga activation.",
      none: "Wala",
      standard: "Standard",
      enablePacking: "I-enable ang packing",
      assistantCompletionsOnly: "Mga completion ng assistant lang",
      readMore: "Magbasa pa",
    },
    training: {
      duplicateFailed: "Hindi ma-duplicate ang run",
      duplicateDraftChanged:
        "Nagbago ang mga setting ng training habang nilo-load. Subukang i-duplicate ulit.",
      duplicateNoModel:
        "Walang naka-save na configuration ng model ang run na ito.",
      startTraining: "Simulan ang training",
      starting: "Sinisimulan...",
      loadingModel: "Nilo-load ang model...",
      checkingDataset: "Sinusuri ang dataset...",
      chooseModel: "Pumili ng model",
      chooseDataset: "Pumili ng dataset",
      chooseModelAndDataset: "Pumili ng model at dataset",
      validation: {
        s3MultimodalUnsupported:
          "Hindi pa sinusuportahan ang mga S3 dataset para sa vision training.",
        s3BucketRequired: "Maglagay muna ng pangalan ng S3 bucket.",
        s3CredentialsRequired:
          "Magbigay ng mga S3 access key o i-enable ang IAM role.",
        modelRequired: "Pumili muna ng base model.",
        learningRatePositive: "Maglagay ng learning rate na higit sa zero.",
        embeddingLearningRateRange:
          "Maglagay ng embedding learning rate na higit sa 0 at mas mababa sa 1.",
        decisionColumnsMissing:
          "Kailangan ng mga decision model ang mga column na state, questions at gold (o answers). Kulang: {columns}.",
        decisionOwnerOnly:
          "Ang may-ari lang ng Unsloth ang puwedeng mag-fine-tune ng mga decision model.",
        hfDatasetRequired: "Pumili muna ng Hugging Face dataset.",
        hfDatasetSplitRequired: "Pumili o maglagay muna ng train split.",
        localDatasetRequired: "Pumili muna ng lokal na dataset.",
        unsupportedDatasetSource:
          "Hindi sinusuportahang pinagmulan ng dataset.",
      },
      startFailed: "Hindi nasimulan ang training",
      startUnconfirmed:
        "Hindi makumpirma ng Unsloth kung nagsimula ang training. Sinusuri ang status sa background.",
      stopFailed: "Hindi naihinto ang training",
      trainingStillActiveTitle: "Aktibo pa ang training",
      stopBeforeConfig:
        "Ihinto muna ang training, pagkatapos ay bumalik sa configuration.",
      resumeFailed: "Hindi naituloy ang training",
      resumeFailedTitle: "Hindi maituloy ang training",
      resumeUnavailable:
        "Ang mga run lang na nahinto o nag-error na may naka-save na checkpoint ang puwedeng ituloy.",
      modelUnverified:
        "Hindi ma-verify ang mga setting ng model na ito. Suriin ang iyong koneksyon o Hugging Face token, pagkatapos ay subukan ulit.",
      legacyDatasetScriptUnsupported:
        "Umaasa ang Hub dataset na ito sa lumang custom script at hindi ito sinusuportahan sa training flow na ito.",
      hfModelAccessDenied:
        "Tinanggihan ng Hugging Face ang access sa model na ito. Magdagdag ng valid na Hugging Face token na may access sa repository at tanggapin ang anumang kinakailangang tuntunin sa access, pagkatapos ay subukan ulit.",
      hfModelVerificationRateLimited:
        "Nililimitahan ang dalas ng pag-verify ng Hugging Face model. Subukan ulit maya-maya.",
      hfModelVerificationFailed:
        "Hindi ma-verify ang Hugging Face model. Suriin ang repository ID at ang iyong access token.",
      hfModelMetadataUnavailable:
        "Pansamantalang hindi available ang metadata ng Hugging Face model. Subukan ulit bago simulan ang training.",
      datasetUnverified:
        "Hindi ma-verify kung compatible ang dataset sa model na ito. Suriin ang iyong koneksyon o Hugging Face token — susubukan ulit ang pagsusuri kapag sinimulan ang training.",
      setupChanged:
        "Nagbago ang setup ng training habang sinusuri ito. Suriin ito at magsimula ulit.",
      uploadConfigTooltip: "I-load ang naka-save na YAML config",
      saveConfigTooltip: "I-download ang kasalukuyang config bilang YAML",
      resetConfigTooltip: "I-reset sa mga default ng model",
      configLoaded: "Na-load ang config",
      failedToLoadConfig: "Hindi na-load ang config",
      invalidYamlFile: "Invalid na YAML file",
      configTooLarge: "Masyadong malaki ang training config (maximum 1 MiB).",
      failedToReadFile: "Hindi nabasa ang file",
      failedToSaveConfig: "Hindi na-save ang config",
      parametersReset: "Na-reset ang mga parameter sa mga default ng model",
      audioIncompatible:
        "Hindi sinusuportahan ng model na ito ang audio. Lumipat sa model na may kakayahang audio o pumili ng dataset na hindi audio.",
      visionIncompatible:
        "Hindi compatible ang text model sa multimodal na dataset. Lumipat sa vision model o pumili ng text-only na dataset.",
      cancelTitle: "Kanselahin ang training",
      cancelDescription:
        "Gusto mo bang kanselahin ang kasalukuyang training run?",
      continueAction: "Ituloy ang training",
      cancelAction: "Kanselahin ang training",
      stopTitle: "Ihinto ang training",
      stopDescription:
        "Piliin kung paano mo gustong ihinto ang kasalukuyang training run. Ang Ihinto at I-save ay nagsusulat ng checkpoint na puwede mong ituloy mamaya; hindi na maitutuloy kapag inihinto nang hindi nagse-save.",
      stopAction: "Ihinto",
      stopping: "Inihihinto...",
      stopAndSave: "Ihinto at I-save",
      compareInChat: "Ikumpara sa Chat",
      exportModel: "I-export ang model",
      milestone: "Milestone",
      halfwayDone: "Kalahati na. Lampas na sa 50% ang training.",
      doneNextStep:
        "Tapos na ang training. Susunod na hakbang: ikumpara ang mga output ng base at ng na-fine-tune.",
    },
    history: {
      title: "History",
      filesDeleted: "Na-delete ang mga file",
      deleteArtifactsLabel: "I-delete din ang mga adapter file sa disk",
      deleteArtifactsDescription:
        "Inaalis ang output folder ng run, kasama ang mga naka-save na adapter at checkpoint.",
      deleteArtifactsSharedNote:
        "May ibang run na gumagamit din ng output folder na ito. Pananatilihin ang mga file hanggang ma-delete ang huling run na gumagamit sa mga ito.",
      artifactsKeptShared:
        "Na-delete ang run. Pinanatili ang mga adapter file dahil may ibang run na gumagamit ng parehong folder.",
      deleteArtifactsActiveError:
        "Ginagamit ang mga file na ito ng kasalukuyang training run. Ihinto ang training bago i-delete ang mga ito.",
      deleteArtifactsFailed:
        "Na-delete ang run pero hindi naalis ang mga file nito.",
      deleteArtifactsRetainedError:
        "Hindi naalis ang mga adapter file, kaya pinanatili ang training run sa history.",
      emptyDescription:
        "Wala pang training run. Simulan ang iyong unang training run sa tab na Configure.",
      loadError: "Hindi na-load ang mga training run",
      deleteError: "Hindi na-delete ang training run. Pakisubukan ulit.",
      retry: "Subukan ulit",
      loadMore: "Mag-load pa",
      loading: "Nilo-load...",
      loadingRun: "Nilo-load ang training run...",
      runNotFound: "Hindi nahanap ang run",
      deleteTitle: "I-delete ang training run?",
      deleteDescription:
        "Permanenteng ide-delete nito ang training run na ito at ang lahat ng metric nito. Hindi na ito maaaring i-undo.",
      resumeTraining: "Ituloy ang training",
      resuming: "Itinutuloy...",
      deleteRun: "I-delete ang run",
      loss: "Loss",
      steps: "Mga step",
      lossTrendSparkline: "Sparkline ng trend ng loss",
      relativeJustNow: "ngayon lang",
      status: {
        completed: "Nakumpleto",
        stopped: "Nahinto",
        error: "Error",
        running: "Tumatakbo",
        continued: "Itinuloy",
      },
      message: {
        completed: "Nakumpleto ang training",
        stopped: "Nahinto ang training",
        running: "Kasalukuyang nagte-training",
        errored: "Nag-error ang training",
      },
      copyPreviewLink: "Kopyahin ang preview link",
      previewLinkCopied: "Nakopya ang preview link",
      previewLinkCopyFailed: "Hindi makopya ang link",
    },
    charts: {
      settings: "Mga Setting ng Chart",
      settingsDescription:
        "Isaayos ang presentasyon ng chart habang tuloy ang training.",
      openSettings: "Buksan ang mga setting ng chart",
      viewWindow: "Saklaw ng view",
      viewWindowDescription:
        "Ipakita lang ang mga pinakabagong step o ang buong history.",
      window: "Window",
      all: "Lahat",
      trainingLoss: "Training loss",
      trainingLossDescription: "Kontrolin ang mga overlay at EMA smoothing.",
      smoothing: "Smoothing",
      smoothingDescription:
        "Ilipat pakanan para sa mas maraming smoothing. `0` = raw.",
      showRawLoss: "Ipakita ang raw loss",
      showSmoothedLoss: "Ipakita ang smoothed loss",
      showAverageLine: "Ipakita ang linya ng average",
      scaleAndCleanup: "Scale at cleanup",
      linear: "Linear",
      log: "Log",
      noClip: "Walang clip",
      clipP99: "I-clip sa p99",
      clipP95: "I-clip sa p95",
      lossAxis: "Axis ng loss",
      gradientNormAxis: "Axis ng gradient norm",
      learningRateAxis: "Axis ng learning rate",
      resetDefaults: "I-reset sa mga default",
      loss: "Loss",
      smoothed: "Smoothed",
      evalLoss: "Eval loss",
      learningRate: "Learning rate",
      lr: "LR",
      gradNorm: "Grad norm",
      gradientNorm: "Gradient norm",
      step: "Step {step}",
      averageValue: "avg {value}",
      waitingForFirstEvaluationStep: "Hinihintay ang unang evaluation step...",
      evaluationNotConfigured: "Hindi naka-configure ang evaluation",
      evalChartWillAppear: "Lalabas ang chart kapag naabot na ang eval_steps",
      setEvalDatasetAndSteps:
        "Itakda ang eval dataset at eval_steps para masubaybayan ang eval loss",
    },
    progress: {
      title: "Progreso ng training",
      liveMetrics: "Live na metric ng training",
      exportGguf: "I-export sa GGUF",
      useInDecisionApi: "Gamitin sa Decision API",
      decisionApiEnabled: "Ang Decision API ay nagsisilbi na ngayon ng {name}.",
      decisionApiFailed: "Hindi mailipat ang Decision API sa model na ito.",
      openConfig: "Buksan ang training config",
      configLabel: "Training config",
      hyperparams: "Mga hyperparam",
      epochs: "Mga epoch",
      batchSize: "Batch size",
      learningRate: "Learning rate",
      optimizer: "Optimizer",
      maxSteps: "Max na step",
      contextLength: "Haba ng context",
      warmupSteps: "Mga warmup step",
      rank: "Rank",
      alpha: "Alpha",
      dropout: "Dropout",
      variant: "Variant",
      epoch: "Epoch {value}",
      percentComplete: "{percent}% kumpleto",
      stepProgress: "Step {current} / {total}",
      loss: "Loss",
      lr: "LR",
      gradNorm: "Grad norm",
      project: "Proyekto",
      model: "Model",
      method: "Paraan",
      elapsed: "Lumipas: {value}",
      eta: "ETA: {value}",
      stepsPerSecond: "{value} steps/s",
      noStepsPerSecond: "-- steps/s",
      tokens: "Mga token: {value}",
      gpuMonitor: "Monitor ng GPU",
      live: "Live",
      utilization: "Paggamit",
      temperature: "Temperatura",
      vram: "VRAM",
      power: "Power",
      phase: {
        idle: "Idle",
        downloadingModel: "Dina-download ang model",
        downloadingDataset: "Dina-download ang dataset",
        loadingModel: "Nilo-load ang model",
        loadingDataset: "Nilo-load ang dataset",
        configuring: "Kino-configure",
        training: "Nagte-training",
        finalizing: "Sine-save ang model",
        trainingDrafter: "Tine-train ang drafter",
        completed: "Nakumpleto",
        error: "Error",
        stopped: "Nahinto",
      },
    },
    trainingStart: {
      ready: "Handa na",
      downloading: "Dina-download",
      preparing: "Inihahanda",
      left: "{eta} na lang",
      downloaded: "{size} na-download",
      terminalStart: "> nagsisimula ang unsloth training...",
      preparingResources: "> Inihahanda ang model at dataset...",
      gettingReady: "> Inihahanda namin ang lahat para sa iyong run...",
      resumingTraining: "Itinutuloy ang training...",
      startingTraining: "sinisimulan ang training...",
      dataset: "Dataset",
      datasetStreaming: "Dataset: streaming (walang buong download)",
      modelWeights: "Mga weight ng model",
    },
  },
  modelMemory: {
    readout:
      "Weights {model} + context {context} = {total} sa {budget} na magagamit na VRAM",
    readoutWithSpec:
      "Weights {model} + KV {kv} + MTP draft {spec} = {total} sa {budget} na magagamit na VRAM",
    kvRate: "Nakareserba ang KV, ~{rate}/token",
    oomLikely: "Malamang mag-OOM sa kasalukuyang mga setting",
    tooLarge:
      "Mas malaki kaysa sa VRAM, mag-o-offload sa CPU. Mas mabilis tumakbo ang mas maliit na quantization",
  },
  skills: {
    title: "Mga Skill",
    description:
      "Natutuklasan ang mga skill mula sa iyong mga karaniwang agent folder. I-enable ang mga ito rito, pagkatapos ay i-type ang @ sa chat para banggitin ang isa.",
    precedence: "Mas inuuna ang ~/.agents/skills kaysa sa ~/.claude/skills.",
    refresh: "I-refresh",
    empty:
      "Walang nahanap na skill. Magdagdag ng SKILL.md folder sa ilalim ng ~/.agents/skills o ~/.claude/skills, pagkatapos ay mag-refresh.",
    sourceAgents: "Agents",
    sourceClaude: "Claude",
    sourceBundled: "Kasama",
    shadowed: "Natatabunan",
    invalid: "Invalid",
    compatibility: "Pagkakatugma: {value}",
    shadowedBy:
      "Mas inuuna ang isa pang {source} skill na may ganitong pangalan.",
    enable: "I-enable ang {name}",
    disable: "I-disable ang {name}",
    updateError: "Hindi ma-update ang skill",
    newSkill: "Bagong skill",
    import: "I-import",
    delete: "I-delete ang {name}",
    createDescription:
      "Ise-save bilang SKILL.md sa bagong folder sa ilalim ng ~/.agents/skills, at ie-enable sa sandaling umiral ito.",
    nameLabel: "Pangalan",
    namePlaceholder: "hal. code-arithmetic",
    nameHint:
      "Maliliit na titik, numero at iisang gitling. Ito rin ang pangalan ng folder.",
    nameInvalid:
      "Gumamit ng 1-64 na maliliit na titik, numero o iisang gitling, na hindi nagsisimula o nagtatapos sa gitling.",
    descriptionLabel: "Paglalarawan",
    descriptionPlaceholder:
      "Gamitin kapag humingi ang user ng arithmetic. Laging i-compute ito gamit ang code tool.",
    descriptionHint:
      "Ang linyang nakikita ng model sa skill catalog nito. Dito nito ibinabatay kung kailan babasahin ang skill.",
    instructionsLabel: "Mga tagubilin",
    instructionsPlaceholder:
      "Ano ang gagawin kapag naaangkop ang skill na ito, sa Markdown.",
    instructionsHint:
      "Ang body ng SKILL.md, binabasa nang buo kapag napili ng model ang skill.",
    create: "Gumawa ng skill",
    save: "I-save ang mga pagbabago",
    created: "Nagawa ang {name}",
    saved: "Na-save ang {name}",
    deleted: "Na-delete ang {name}",
    importedOne: "{count} skill ang naidagdag",
    importedMany: "{count} na skill ang naidagdag",
    importPartial: "Naidagdag ang {added} sa {total} na skill",
    saveError: "Hindi ma-save ang skill",
    openError: "Hindi mabuksan ang skill",
    deleteError: "Hindi ma-delete ang skill",
    importError: "Hindi ma-import ang skill",
    exportError: "Hindi ma-export ang skill",
    deleteTitle: "I-delete ang skill",
    deleteDescription:
      'I-delete ang "{name}" at ang lahat ng nasa folder nito sa ilalim ng ~/.agents/skills? Hindi na ito maaaring i-undo.',
    search: "Maghanap ng skill ayon sa pangalan o paglalarawan…",
    noMatch: 'Walang skill na tumutugma sa "{query}"',
    clearSearch: "I-clear ang paghahanap",
    readOnlyClaude:
      "Read-only: nahanap sa ~/.claude/skills. I-edit ang file doon.",
    readOnlyBundled: "Read-only: kasama sa Unsloth.",
    readOnlyLinked:
      "Read-only: link ang entry na ito. I-edit ang mga file kung nasaan ang mga ito.",
    characters: "{count} na character",
    unsaved: "mga hindi naka-save na pagbabago",
    revert: "Ibalik",
    enabledLabel: "Naka-enable",
    licenseLabel: "Lisensya",
    compatibilityLabel: "Compatibility",
    allowedToolsLabel: "Mga pinapayagang tool",
    sectionAgents: "Iyong mga skill",
    sectionAgentsHint:
      "Naka-save sa ~/.agents/skills. Gumawa, mag-edit at mag-delete ng mga ito rito.",
    sectionClaude: "Mga Claude skill",
    sectionClaudeHint:
      "Nahanap sa ~/.claude/skills. Read-only rito; i-edit ang mga file doon.",
    sectionBundled: "Kasama",
    sectionBundledHint: "Kasama sa Unsloth. Read-only.",
    linked: "Link",
    discardTitle: "I-discard ang mga pagbabago?",
    discardDescription:
      "Mawawala ang mga hindi naka-save na pagbabago sa {name}.",
    discard: "I-discard",
    mentions: "Mga Skill",
    manage: "Pamahalaan ang mga skill",
    turnOff: "I-off ang mga skill",
    bulkActions: "Mga aksyon sa skill",
    enableAll: "I-enable lahat",
    disableAll: "I-disable lahat",
    resetAll: "I-reset sa mga default",
    resetTitle: "I-reset ang lahat ng skill?",
    resetDescription:
      "Babalik ang bawat skill sa kung paano ito sa bagong install: naka-on ang iyong mga skill at mga Claude skill, naka-off ang mga kasamang skill. Mabubura ang iyong mga pinili na on at off.",
    reset: "I-reset",
  },
  scheduledTasks: {
    title: "Naka-iskedyul",
    description:
      "Magpatakbo ng mga prompt ayon sa iskedyul gamit ang alinman sa iyong mga model.",
    newTask: "Bagong task",
    upcoming: "Paparating",
    columns: {
      repeats: "Umuulit",
      time: "Oras",
      nextRun: "Susunod na run",
    },
    monthlyDay: "Buwanan, araw {day}",
    filterBy: "I-filter ayon sa",
    filter: "I-filter ang mga task",
    active: "Aktibo",
    noMatches: "Walang naka-iskedyul na task",
    search: "Maghanap ng naka-iskedyul na task",
    newResult: "Bagong resulta",
    sentBy: "Ipinadala ng naka-iskedyul na task",
    chatMark: "Chat mula sa naka-iskedyul na task",
    taskMissing: "Wala na ang task na ito.",
    emptyTitle: "Wala pang naka-iskedyul na task",
    emptyDescription:
      "Kumuha ng pang-araw-araw na buod, lingguhang paalala, o anumang gusto mong gawin ayon sa iskedyul.",
    createFirst: "Gumawa ng task",
    loadFailed: "Hindi ma-load ang mga naka-iskedyul na task.",
    retry: "Subukan ulit",
    paused: "Naka-pause",
    completed: "Nakumpleto",
    running: "Tumatakbo",
    lastRun: "Huling run {when}",
    lastRunFailed: "Nabigo ang huling run {when}: {reason}",
    openChat: "Buksan ang chat",
    runNow: "Patakbuhin ngayon",
    pause: "I-pause",
    resume: "Ituloy",
    edit: "I-edit",
    options: "Mga opsyon ng task",
    schedule: {
      once: "Isang beses, {date} nang {time}",
      daily: "Araw-araw nang {time}",
      weekdays: "Tuwing weekday nang {time}",
      weekly: "Tuwing {days} nang {time}",
      monthly: "Buwan-buwan sa araw {day} nang {time}",
    },
    form: {
      createTitle: "Mag-iskedyul ng task",
      editTitle: "I-edit ang naka-iskedyul na task",
      name: "Pangalan",
      namePlaceholder: "Pangalan (opsyonal)",
      instructions: "Mga tagubilin",
      instructionsHint: "Ipinapadala sa model tuwing tatakbo ang task.",
      instructionsPlaceholder: "Mga tagubilin…",
      repeat: "Ulitin",
      once: "Isang beses",
      daily: "Araw-araw",
      weekdays: "Mga weekday",
      weekly: "Lingguhan",
      monthly: "Buwanan",
      date: "Petsa",
      time: "Oras",
      days: "Mga araw",
      dayOfMonth: "Araw ng buwan",
      lastDayNote:
        "Ang mga buwang walang ganitong araw ay tatakbo sa huling araw ng mga ito.",
      advanced: "Advanced",
      notifications: "Mga notification",
      notifyAll: "Bawat run",
      notifyFailures: "Mga pagkabigo lang",
      notifyNone: "Naka-off",
      archiveSuccessful: "I-archive ang mga matagumpay na run",
      project: "Proyekto",
      projectNone: "Wala",
      model: "Model",
      modelLoaded: "Alinmang model ang naka-load",
      modelOnDevice: "Sa device na ito",
      modelNote:
        "Kung hindi naka-load ang model na ito kapag tumakbo ang task, magtatanong muna ang Unsloth.",
      modelNoteAllowed:
        "Ilo-load lang ang model na ito habang wala ka sa Unsloth, pagkatapos ay ibabalik ang dati mong gamit.",
      modelEmpty:
        "Mag-download ng model sa Model hub o magdagdag ng koneksyon para pumili ng iba.",
      modelMissing: "{model} (wala sa device na ito)",
      allowLoad: "Awtomatikong lumipat kapag wala ako",
      allowLoadHint:
        "Kapag oras na ng task na ito at hindi mo pa nagagamit ang Unsloth sa loob ng 5 minuto, ilo-load nito ang model na ito nang hindi nagtatanong, patatakbuhin ang task, pagkatapos ay babalik sa dati mong model. Kung ginagamit mo ang Unsloth o nagte-training ka sa oras na iyon, lalaktawan ang run at hindi gagalawin ang iyong model. Kapag naka-off, magtatanong muna ang Unsloth bago mag-load.",
      reasoning: "Lakas ng pag-iisip",
      reasoningDefault: "Default",
      reasoningOff: "Naka-off",
      reasoningLowest: "Pinakamababa",
      reasoningHighest: "Pinakamataas",
      create: "Gumawa",
    },
    deleteDialog: {
      title: "I-delete ang naka-iskedyul na task?",
      description:
        'Titigil sa pagtakbo ang "{title}". Pananatilihin ang mga chat na nagawa na nito.',
    },
    errors: {
      noModel:
        "Walang naka-load na model. Mag-load ng model sa Chat, pagkatapos ay patakbuhin ulit ang task.",
      modelNotLoaded:
        "Hindi na-load ang model ng task. I-load ito sa Chat, pagkatapos ay patakbuhin ulit ang task.",
      modelLoadFailed:
        "Hindi ma-load ang model ng task. Tiyaking naka-download pa rin ito sa Model hub.",
      modelBusy:
        "Nilaktawan habang ginagamit mo ang Unsloth, kaya hindi ginalaw ang iyong model.",
      connectionUnavailable:
        "Nawawala o naka-off ang koneksyon ng task. Suriin ito sa Mga Koneksyon.",
      timeout: "Masyadong natagalan ang model sa pagsagot.",
      cancelled: "Nahinto ang run bago ito natapos.",
      interrupted: "Nagsara ang Unsloth bago natapos ang run.",
      empty: "Walang laman ang sagot ng model.",
      unknown: "Nabigo ang run.",
      saveFailed: "Hindi ma-save ang task",
      updateFailed: "Hindi ma-update ang task",
      deleteFailed: "Hindi ma-delete ang task",
      runFailed: "Hindi masimulan ang task",
    },
    loadPrompt: {
      title: "I-load ang {model}?",
      description:
        'Tumatakbo ang "{title}" sa {model}, na hindi naka-load. Kapag pinatakbo ito, ilo-load ang {model} hanggang matapos ito, pagkatapos ay ibabalik ang dati mong gamit.',
      always:
        "Huwag nang magtanong ulit. Lumipat lang habang wala ako sa Unsloth.",
      notNow: "Hindi ngayon",
      load: "I-load at patakbuhin",
    },
    toast: {
      finished: 'Handa na ang "{title}"',
      failed: 'Nabigo ang "{title}"',
      many: "{count} naka-iskedyul na task ang natapos",
      open: "Buksan",
      view: "Tingnan",
    },
    notificationBody:
      "Natapos ang iyong naka-iskedyul na task. Buksan ang Unsloth para basahin ang sagot.",
  },
  library: {
    tabs: {
      ariaLabel: "Mga seksyon ng Library",
      suggested: "Iminumungkahi",
      favorites: "Mga Paborito",
      folders: "Mga Folder",
      images: "Mga Larawan",
      videos: "Mga Video",
      audio: "Audio",
      models: "Mga Fine-tune",
      all: "Lahat",
      chats: "Mga Chat",
    },
    chats: {
      sections: {
        chats: "Mga Chat",
        projects: "Mga Proyekto",
        archived: "Naka-archive",
        ariaLabel: "Mga seksyon ng chat",
      },
      search: {
        chats: "Maghanap ng chat",
        projects: "Maghanap ng proyekto",
        archived: "Maghanap ng naka-archive na chat",
        project: 'Maghanap sa "{project}"',
        sections: "Maghanap ng seksyon",
      },
      toolbar: {
        show: "Ipakita",
        pinned: "Naka-pin",
        forks: "Mga fork",
        comparisons: "Mga paghahambing",
        project: "Proyekto",
        model: "Model",
        groupBy: "I-grupo ayon sa",
        groupNone: "Walang pag-grupo",
        groupDate: "Petsa",
        groupProject: "Proyekto",
        pinnedFirst: "Una ang naka-pin",
        newChat: "Bagong chat",
        newProject: "Bagong proyekto",
        sortChats: "Bilang ng chat",
        withoutSection: "Walang seksyon",
      },
      groups: {
        today: "Ngayong araw",
        yesterday: "Kahapon",
        week: "Nakaraang 7 araw",
        month: "Nakaraang 30 araw",
      },
      list: {
        created: "Ginawa",
        lastActive: "Huling aktibo",
        location: "Lokasyon",
        lastModified: "Huling binago",
        contents: "Nilalaman",
        oneMessage: "1 mensahe",
        messageCount: "{count} na mensahe",
        showMore: "Magpakita pa ({count})",
      },
      badges: {
        pinned: "Naka-pin",
        fork: "Fork",
        compare: "Ihambing",
      },
      menu: {
        fork: "I-fork ang chat",
        moveTo: "Ilipat sa proyekto",
        newChatInProject: "Bagong chat sa proyekto",
        edit: "I-edit",
        deleteProject: "I-delete ang proyekto",
        newChatInSection: "Bagong chat sa seksyon",
      },
      project: {
        oneProject: "1 proyekto",
        projectCount: "{count} na proyekto",
      },
      empty: {
        chatsTitle: "Wala pang chat",
        chatsDescription:
          "Lumalabas dito ang iyong mga pag-uusap, nakahiwalay sa iyong mga file.",
        archivedTitle: "Walang naka-archive",
        archivedDescription: "Lumalabas dito ang mga chat na ina-archive mo.",
        sectionsTitle: "Wala pang seksyon",
        sectionsDescription:
          "Ginugrupo ng mga seksyon ang mga chat at proyekto sa sidebar ayon sa gusto mo.",
        sectionTitle: "Wala pang laman ang seksyong ito",
        sectionDescription:
          "Ilipat dito ang mga chat mula sa menu ng mga ito, o magsimula ng bagong chat sa seksyong ito.",
      },
      dialog: {
        renameChat: "Palitan ang pangalan ng chat",
        deleteChatDescription:
          "Permanenteng ide-delete ang chat na ito. Hindi na ito maaaring i-undo.",
        deleteProjectDescription:
          "Permanenteng ide-delete ang proyekto at ang mga chat nito ({count}). Hindi na ito maaaring i-undo.",
        deleteFilesMany:
          "Alisin din ang mga file at sandbox folder na ginawa ng mga chat na ito.",
        deleteProjectFilesLabel: "I-delete ang workspace folder ng proyekto",
        deleteProjectFiles: "Aalisin sa disk ang workspace folder ng proyekto.",
      },
      toast: {
        projectMoved: "Inilipat ang {project} sa {section}",
        projectUnfiled: "Inalis ang {project} sa {section}",
        forkFailed: "Hindi na-fork",
        projectDeleted: 'Na-delete ang "{name}"',
        projectDeleteFailed: "Hindi na-delete ang proyekto",
        removedFromSection: "Inalis sa seksyon: {count}",
      },
      folder: {
        openChat: "Buksan ang folder ng chat",
        openChatTitle:
          "Buksan ang folder na binabasa at sinusulatan ng mga tool call ng chat na ito",
        openProject: "Buksan ang folder ng proyekto",
        openProjectTitle:
          "Buksan ang folder na binabasa at sinusulatan ng mga chat ng proyektong ito",
        chatHint:
          "Ang desktop app lang ang makakapagbukas ng folder ng mga file ng chat. Sa browser, i-download ang file mula sa resulta ng tool na sumulat nito.",
        projectHint:
          "Ang desktop app lang ang makakapagbukas ng folder ng proyekto. Sa browser, i-download ang mga file mula sa mga resulta ng tool na sumulat ng mga ito.",
        manyFolders: "Sumulat ang chat na ito sa higit sa isang folder.",
        manyFoldersDescription:
          "Nagpatakbo ito ng mga tool bago at pagkatapos ng paglipat, kaya buksan na lang ang folder mula sa isang tool card.",
        chatFailed: "Hindi mabuksan ang folder ng chat.",
        projectFailed: "Hindi mabuksan ang folder ng proyekto.",
        chatMissing: "Wala pang folder ang chat na ito.",
        projectMissing: "Wala pang folder ang proyektong ito.",
        missingDescription:
          "Ginagawa ito sa unang beses na sumulat ng file ang isang tool.",
        upload: "I-upload sa folder ng chat",
        uploadTitle:
          "Magdagdag ng mga file sa folder na binabasa at sinusulatan ng mga tool call ng chat na ito",
        uploaded: "Naidagdag ang {name} sa folder ng chat",
        uploadedMany: "Naidagdag ang {count} na file sa folder ng chat",
        uploadedDescription:
          "Mabubuksan na ang mga ito ng mga tool sa chat na ito. Hindi idinadagdag ang mga ito sa pag-uusap.",
        uploadExists: "Nasa folder ng chat na ang {name}",
        uploadExistsMany: "Nasa folder ng chat na ang {count} na file",
        replace: "Palitan",
        uploadFailed: "Hindi ma-upload ang {name}",
      },
    },
    empty: {
      suggestedTitle: "Walang laman ang iyong library",
      suggestedDescription:
        "Lumalabas dito ang mga file na ina-upload o ginagawa mo sa mga chat.",
      favoritesTitle: "Wala pang paborito",
      favoritesDescription:
        "Magdagdag ng mga file sa iyong mga paborito para mabilis mong mahanap ang mga ito rito.",
      foldersTitle: "Gumawa ng iyong unang folder",
      foldersDescription:
        "Gumawa ng mga folder para ayusin ang mga item sa iyong library.",
      imagesTitle: "Wala pang larawan",
      imagesDescription:
        "Lumalabas dito ang mga larawang ina-upload o ginagawa mo.",
      videosTitle: "Wala pang video",
      videosDescription:
        "Lumalabas dito ang mga video na ina-upload o ginagawa mo.",
      audioTitle: "Wala pang audio",
      audioDescription:
        "Lumalabas dito ang mga speech, musika at stem na ginagawa mo sa pahina ng Audio.",
      modelsTitle: "Wala pang na-fine-tune na model",
      modelsDescription:
        "Lumalabas dito ang mga model na tine-train o ine-export mo sa Unsloth.",
      createFolder: "Gumawa ng folder",
      generateImage: "Gumawa ng larawan",
      generateVideo: "Gumawa ng video",
      generateAudio: "Gumawa ng audio",
      trainModel: "Mag-train ng model",
      uploadFiles: "Mag-upload ng mga file",
      dropHere: "o i-drop ang mga file dito",
      noMatchesTitle: "Walang tugma",
      noMatchesDescription:
        "Subukan ang ibang paghahanap o i-clear ang mga filter.",
      loadErrorTitle: "Hindi ma-load ang iyong Library",
      loadErrorFallback: "Nagkaroon ng problema.",
      tryAgain: "Subukan ulit",
    },
    sections: {
      items: "Mga item",
    },
    breadcrumb: "Breadcrumb",
    searchLibrary: "Maghanap sa library",
    searchFolder: "Maghanap sa folder",
    dropToUpload: "I-drop para i-upload",
    dropToUploadInto: 'I-drop para i-upload sa "{folder}"',
    itemCountOne: "{count} item",
    itemCount: "{count} na item",
    selectItem: "Piliin ang {name}",
    selection: {
      startChat: "Magsimula ng chat",
      move: "Ilipat",
      selectAll: "Piliin lahat",
      countOfTotal: "{count} sa {total} ang napili",
      clear: "I-clear ang pagpili",
    },
    menu: {
      moreActions: "Higit pang aksyon",
      chatAboutThis: "Makipag-chat tungkol dito",
      chatWithModel: "Makipag-chat sa model na ito",
      editImage: "I-edit ang larawan",
      addToFavorites: "Idagdag sa Mga Paborito",
      regenerateTitle: "Gawing muli ang pamagat",
      removeFromFavorites: "Alisin sa Mga Paborito",
      download: "I-download",
      addToFolder: "Idagdag sa folder",
      newFolder: "Bagong folder",
      noFolder: "Library (walang folder)",
      deleteFolder: "I-delete ang folder",
    },
    reveal: {
      finder: "Ipakita sa Finder",
      explorer: "Ipakita sa Explorer",
      files: "Ipakita sa Folder",
    },
    toolbar: {
      filter: "I-filter",
      source: "Pinagmulan",
      fileType: "Uri ng file",
      clearFilters: "I-clear ang mga filter",
      workflow: "Ginawa gamit ang",
      uploaded: "Na-upload",
      generated: "Ginawa",
      documents: "Mga Dokumento",
      spreadsheets: "Mga Spreadsheet",
      presentations: "Mga Presentation",
      pdfs: "Mga PDF",
      gridView: "Tingnan bilang grid",
      listView: "Tingnan bilang listahan",
      grid: "Grid",
      list: "Listahan",
      settings: "Mga setting ng Library",
      sort: "Ayusin",
      sortDefault: "Default na pagkakasunod-sunod",
      sortName: "Pangalan",
      sortModified: "Binago",
      sortSize: "Laki",
      sortAscending: "Pataas",
      sortDescending: "Pababa",
    },
    audio: {
      play: "I-play ang {name}",
      pause: "I-pause ang {name}",
      playFailed: "Hindi ma-play ang {name}",
      fromRun: "Mula sa parehong run",
      downloadRun: "I-download lahat",
    },
    create: {
      note: "Tala",
      image: "Larawan",
      video: "Video",
      audio: "Audio",
      model: "Model",
      folder: "Folder",
      untitledNote: "Walang pamagat na tala",
    },
    list: {
      selectAll: "Piliin lahat",
      name: "Pangalan",
      lastActivity: "Huling aktibidad",
      favorite: "Paborito",
      modified: "Binago {time}",
      opened: "Binuksan {time}",
      modifiedColumn: "Binago",
      justNow: "ngayon lang",
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
      fullFineTune: "Buong fine-tune",
      loraExport: "LoRA na export",
      mergedExport: "Merged na export",
      ggufExport: "GGUF na export",
      model: "Model",
    },
    dialog: {
      create: "Gumawa",
      renameFolder: "Palitan ang pangalan ng folder",
      renameFile: "Palitan ang pangalan ng file",
      deleteTitle: 'I-delete ang "{name}"?',
      deleteManyTitle: "I-delete ang {count} na item?",
      deleteManyDescription:
        "Ide-delete ang mga file mula sa kinaroroonan ng mga ito. Aakyat nang isang level ang anumang nasa loob ng na-delete na folder.",
      deleteFolderIntoParent:
        'Ililipat sa "{folder}" ang lahat ng nasa loob. Walang file na ide-delete.',
      deleteFolderIntoLibrary:
        "Ililipat sa iyong Library ang lahat ng nasa loob. Walang file na ide-delete.",
      deleteUpload: "Permanenteng ide-delete nito ang file.",
      deleteAttachment: "Aalisin din ito sa chat kung saan ito naka-attach.",
      deleteImage: "Aalisin din ito sa iyong gallery ng Mga Larawan.",
      deleteVideo: "Aalisin din ito sa iyong gallery ng Video.",
      deleteAudio: "Aalisin din ito sa iyong gallery ng Audio.",
      deleteSandbox: "Aalisin din ito sa chat na gumawa nito.",
      deleteModel: "Permanenteng ide-delete nito ang model sa disk.",
      unsavedTitle: "Hindi na-save ang iyong mga pagbabago",
      unsavedDescription:
        "{reason} Subukan ulit, o i-discard ang iyong mga pagbabago para isara ang file.",
      keepEditing: "Ituloy ang pag-edit",
      discardChanges: "I-discard ang mga pagbabago",
    },
    preview: {
      code: "Code",
      preview: "Preview",
      fromChat: "Mula sa chat",
      viewOriginalChat: "Tingnan ang orihinal na chat",
      viewInImages: "Tingnan sa Mga Larawan",
      viewInVideo: "Tingnan sa Video",
      viewInAudio: "Tingnan sa Audio",
      viewOriginalProject: "Tingnan ang orihinal na proyekto",
      viewTrainingRun: "Tingnan ang training run",
      sheetTruncated: "Ipinapakita ang bahagi ng sheet na ito",
      documentTruncated: "Ipinapakita ang simula ng dokumentong ito",
      emptyDocument: "Walang laman ang file na ito.",
      cannotPreview:
        "Hindi mapi-preview dito ang file na ito. I-download ito para buksan.",
      noPreview: "Walang preview para sa uri ng file na ito.",
      tooLargeToPreview:
        "Masyadong malaki ang file na ito para i-preview dito. I-download ito para buksan.",
      startWriting: "Magsimulang magsulat…",
      type: "Uri",
      baseModel: "Base model",
      size: "Laki",
      location: "Lokasyon",
      unknown: "Hindi alam",
      readOnlyUtf16:
        "Hindi valid na UTF-16 text ang file na ito, kaya read-only itong bubuksan dito.",
      readOnlyNotUtf8:
        "Hindi UTF-8 text ang file na ito, kaya read-only itong bubuksan dito. Maaaring hindi maipakita nang tama ang ilang character.",
    },
    viewer: {
      scale: "Scale",
      fit: "I-zoom para magkasya",
      moreActionsImage: "Higit pang aksyon para sa larawang ito",
      moreActionsVideo: "Higit pang aksyon para sa video na ito",
      moreActionsClip: "Higit pang aksyon para sa clip na ito",
      moreActionsFile: "Higit pang aksyon para sa file na ito",
      openImage: "Buksan ang larawan",
      openImageNamed: "Buksan ang larawan: {prompt}",
      openVideo: "Buksan ang video",
      untitledImage: "Walang pamagat na larawan",
      untitledVideo: "Walang pamagat na video",
    },
    project: {
      label: "Proyekto",
      newProject: "Bagong proyekto",
      noProjects: "Wala pang proyekto",
      alreadyIn: "Nasa {project} na",
      addedTo: "Naidagdag sa {project}",
      createAndAdd: "Gumawa at idagdag",
      addFailedImage: "Hindi naidagdag ang larawan sa proyekto",
      addFailedVideo: "Hindi naidagdag ang video sa proyekto",
      addFailedClip: "Hindi naidagdag ang clip sa proyekto",
      addFailedFile: "Hindi naidagdag ang file sa proyekto",
      newProjectImage: "Idagdag ang larawan sa bagong proyekto",
      newProjectVideo: "Idagdag ang video sa bagong proyekto",
      newProjectClip: "Idagdag ang clip sa bagong proyekto",
      newProjectFile: "Idagdag ang file sa bagong proyekto",
    },
    toast: {
      missingItem: "Wala na sa Library ang file na iyon",
      moveFailed: "Hindi mailipat",
      movedToFolder: "Inilipat sa {folder}",
      movedToLibrary: "Inilipat sa Library",
      folderFallback: "folder",
      favoritesFailed: "Hindi ma-update ang mga paborito",
      addedToFavorites: "Naidagdag sa Mga Paborito",
      removedFromFavorites: "Inalis sa Mga Paborito",
      uploadingOne: "Ina-upload ang {name}…",
      uploadingMany: "Ina-upload ang {count} na file…",
      uploadedOne: "Na-upload ang file",
      uploadedMany: "{count} na file ang na-upload",
      uploadFailed: "Nabigo ang pag-upload",
      uploadTooLarge: "Mas malaki sa 512 MB ang {name}.",
      readDropsFailed: "Hindi mabasa ang mga na-drop na file",
      createNoteFailed: "Hindi magawa ang tala",
      renameFailed: "Hindi mapalitan ang pangalan",
      createFolderFailed: "Hindi magawa ang folder",
      signedOutBeforeFolder: "Nag-sign out bago nagawa ang folder.",
      signedOutBeforeSave: "Nag-sign out bago na-save ang pagbabago.",
      signedOutBeforeUpload: "Nag-sign out bago natapos ang pag-upload.",
      deleteFailed: "Hindi ma-delete ang ilang item",
      saveFailed: "Hindi ma-save",
      saveNoteFirst: "I-save muna ang tala.",
      revealFailed: "Hindi mabuksan ang file manager",
      downloadFailed: "Hindi ma-download ang {name}",
      downloadManyFailed: "Hindi ma-download ang mga file",
      downloadingMany: "Dina-download ang {count} na file",
      downloadingManyDescription:
        "Kung magtanong ang iyong browser, payagan ang site na ito na mag-download ng maraming file.",
      preparingMany: "Inihahanda ang {count} na file…",
      zipFileName: "Mga file ng Library",
      nothingToChat: "Wala pang mapag-uusapan",
      emptyFolder: "Walang file ang folder na ito.",
      tooLargeOne: "Masyadong malaki ang {name} para i-attach",
      tooLargeMany: "Masyadong malaki ang mga file na ito para i-attach",
      attachedOne: "Na-attach ang {count} file",
      attachedMany: "Na-attach ang {count} na file",
      skippedTooLarge: "{count} ang masyadong malaki para i-attach.",
      skippedOverLimit: "{count} pa ang lampas sa limitasyong {limit} na file.",
      chatFilesWaiting:
        "Ia-attach ang {count} kapag na-load na ang model na nakakabasa sa mga ito.",
      openFilesFailed: "Hindi mabuksan ang mga file",
      editImageFailed: "Hindi mabuksan ang larawan para i-edit",
      speechModel: "Speech model ang {name}",
      speechModelDescription:
        "Piliin ito mula sa menu ng model sa pahina ng Audio.",
      noMediaLink:
        "Walang ibinalik na link ang server para i-play ang file na ito.",
      imageNotFound: "Hindi mahanap ang larawang ito",
      clipNotFound: "Hindi mahanap ang clip na ito",
      notFoundDescription: "Maaaring naka-archive o na-delete na ito.",
      imageTooLarge: "Masyadong malaki ang larawang ito para i-attach",
      videoTooLarge: "Masyadong malaki ang video na ito para i-attach",
      attachmentLimit: "Limitado sa {size} MB ang mga attachment sa chat.",
      attachingImage: "Ina-attach ang larawan…",
      attachingVideo: "Ina-attach ang video…",
      readImageFailed: "Hindi mabasa ang larawan ({status}).",
      readVideoFailed: "Hindi mabasa ang video ({status}).",
      openFileFailed: "Hindi mabuksan ang file",
    },
  },
  exportDecision: {
    title: "Decision model",
    description:
      "Ang mga {layout} decision model ay ine-export sa GGUF para sa decision server ng llama.cpp. Pumili ng isa o higit pang quantization.",
    adapterNote:
      "May mga LoRA adapter ang checkpoint na ito; ime-merge ang mga ito bago ang conversion.",
    notEligibleTitle: "Hindi available ang GGUF export",
    ggufOnly: "Sa GGUF lang maaaring i-export ang mga decision model",
    existing: "Na-export na: {quantizations}",
    outputNote: "Sine-save ang mga GGUF file sa folder ng run: {path}",
    methodLabel: "GGUF ng decision model",
    outputLabel: "Decision GGUF ({quantizations})",
  },
  decisions: {
    title: "Sumubok ng decision",
    description: "Tanungin ang decision model tungkol sa anumang text.",
    tryIt: "Subukan ito",
    tryItOff: 'I-on ang "Mag-serve ng mga request" para subukan ito.',
    typeNoul: "Oo / hindi",
    typeChoice: "Pagpipilian",
    typeScore: "Score",
    aboutNoul: "Sumasagot ng oo o hindi, kasama ang probability ng oo.",
    aboutChoice:
      "Pumipili ng isang opsyon mula sa listahan, may probability para sa bawat isa.",
    aboutScore:
      "Nire-rate ang state sa isang scale ng mga level, pinakamababa muna.",
    model: "Decision model",
    defaultModel: "Default ({model})",
    layaModel: "Laya {model}",
    thisMachine: "Ang machine na ito",
    modelReady: "Handa na",
    modelLoading: "Nilo-load",
    modelNotLoaded: "Ilo-load sa unang run",
    connectionNotDefault:
      "Itakda muna ito bilang model ng Decision API sa Mga Setting > API.",
    question: "Tanong",
    questionHelp: "Ang dapat pagpasyahan ng model.",
    result: "Resulta",
    form: "Form",
    json: "JSON",
    preview: "Preview",
    state: "State",
    stateHelp: "Ang text na pagpapasyahan. Plain text o JSON.",
    criteria: "Pamantayan",
    criteriaNoulHelp:
      "Opsyonal. Kung ano ang ituturing na oo at kung ano ang ituturing na hindi.",
    criteriaChoiceHelp:
      "Ang mga opsyong pagpipilian, bawat isa ay may maikling paglalarawan.",
    criteriaScoreHelp: "Ang mga level ng scale, pinakamababa muna.",
    yesWhen: "Oo kapag",
    noWhen: "Hindi kapag",
    optionName: "Opsyon",
    optionDescription: "Paglalarawan",
    optionNamePlaceholder: "pangalan",
    optionDescriptionPlaceholder: "Kailan ito pipiliin",
    optionNameLabel: "Pangalan ng opsyon {n}",
    optionDescriptionLabel: "Paglalarawan ng opsyon {n}",
    removeOption: "Alisin ang opsyon {n}",
    addOption: "Magdagdag ng opsyon",
    levelLabel: "Level {n}",
    removeLevel: "Alisin ang level {n}",
    addLevel: "Magdagdag ng level",
    jsonHelp:
      "Ang request body. Ipinapadala ito ng Patakbuhin nang eksakto sa pagkakasulat.",
    examples: "Mga halimbawa",
    presetToolApproval: "Aprubahan ang mga tool call ng agent",
    presetPromptInjection: "Hulihin ang prompt injection",
    presetTicketRouting: "I-route ang mga support ticket",
    presetModeration: "I-moderate ang mga mensahe",
    presetIntent: "Tukuyin ang intensyon",
    presetGrading: "Bigyan ng grado ang isang sagot",
    run: "Patakbuhin ang decision",
    running: "Tumatakbo",
    answer: "Sagot",
    yes: "Oo",
    no: "Hindi",
    scoreDetail: "Score na {score} sa scale na 0 hanggang {max}",
    latency: "{ms} ms",
    inputTokens: "{count} na input token",
    responseTitle: "Tugon",
    emptyTitle: "Wala pang resulta",
    emptyBody: "Pindutin ang Patakbuhin ang decision o {shortcut}.",
    cantRun: "Hindi mapatakbo",
    failed: "Error {status}",
    invalidJson: "Hindi valid na JSON ang request body.",
  },
  floatingPanels: {
    loadedModels: {
      title: "Mga na-load na model",
      countOne: "{count} model ang na-load",
      countMany: "{count} na model ang na-load",
      pillLabel: "{count}. Ipakita ang mga detalye, o i-drag para ilipat",
      dragToMove: "I-drag para ilipat",
      dragHandleLabel: "I-drag para ilipat",
      collapse: "I-collapse",
      collapseLabel: "I-collapse ang mga na-load na model",
      close: "Isara",
      closeLabel: "Isara ang mga na-load na model",
      backOnNextLoad: "Babalik sa susunod na pag-load ng model",
      ejectToFree: "I-eject para magbakante ng memory",
      ejectLabel: "I-eject ang {model}",
      openTarget: "Buksan ang {target}",
      rowLabel: "{model}. Buksan ang {target}",
      loading: "Nilo-load",
      ejecting: "Ine-eject",
      stillInMemory: "Nasa memory pa",
      alsoServing: "Nagse-serve din",
      noneLoaded: "Walang naka-load na model",
      noneLoadedDetail: "Lalabas dito ang mga model na ilo-load mo.",
      otherAccount: "Ginagamit ng ibang account",
      otherAccountDetail: "Hawak ang memory ng model na hindi mo nakikita",
      toastStillResident:
        'Na-load ang "{model}" habang nag-e-eject, kaya gumagamit pa rin ito ng memory. I-eject ulit para mapalaya ito.',
      toastUnverified:
        "Hiniling na i-unload ang {model}, pero hindi ito kinumpirma ng runtime nito. Tingnan ang card maya-maya.",
      toastAlreadyFree: "Hindi na naka-load ang {model}.",
      toastReplaced:
        'Hindi na naka-load ang {model}. Pumalit ang "{resident}" at hindi ito ginalaw.',
      toastEjected: "Na-eject ang {model}",
      toastFailed: "Hindi na-eject ang {model}",
    },
    monitor: {
      dragHandleLabel:
        "Ilipat ang live monitor. Iniuusog ito ng mga arrow key, ini-reset ito ng double-click.",
      reconnecting: "Muling kumokonekta...",
    },
  },
  projectsPage: {
    title: "Mga Proyekto",
    search: "Maghanap ng proyekto",
    importExport: "I-import / I-export ang mga proyekto",
    importChatsAction: "I-import ang mga chat…",
    exportAllProjects: "I-export ang Lahat ng Proyekto",
    exportProjectsAndRecents: "I-export ang Mga Proyekto + Mga Kamakailan",
    combined: "Pinagsama",
    perChat: "Bawat chat",
    newProject: "Bagong proyekto",
    name: "Pangalan",
    updated: "Na-update",
    newestFirst: "Pinakabago muna",
    oldestFirst: "Pinakaluma muna",
    today: "Ngayong araw",
    yesterday: "Kahapon",
    emptyDescription:
      "Pinagsasama-sama ng isang proyekto ang magkakaugnay na chat, na may mga shared na tagubilin at source na magagamit ng bawat chat dito.",
    createFirst: "Gumawa ng iyong unang proyekto",
    noMatches: "Walang proyektong tumutugma sa iyong paghahanap.",
    clearSearch: "I-clear ang paghahanap",
    showChats: "Ipakita ang mga chat",
    hideChats: "Itago ang mga chat",
    pinProject: "I-pin ang proyekto",
    unpinProject: "I-unpin ang proyekto",
    newChatInProject: "Bagong chat sa proyekto",
    projectOptions: "Mga opsyon ng proyekto",
    loadChatsFailed: "Hindi ma-load ang mga chat. Subukan ulit",
    refreshChatsFailed: "Hindi ma-refresh ang mga chat. Subukan ulit",
    noChats: "Walang chat",
    unread: "Hindi pa nababasa",
    pinChat: "I-pin ang chat",
    unpinChat: "I-unpin ang chat",
    chatOptions: "Mga opsyon ng chat",
    exportAllChats: "I-export ang lahat ng chat…",
    archive: "I-archive",
    renameChat: "Palitan ang pangalan ng chat",
    chatName: "Pangalan ng chat",
    deleteChat: "I-delete ang chat",
    deleteChatDescription:
      "Sigurado ka bang gusto mong i-delete ang {name}? Permanenteng ide-delete ang mga mensahe nito.",
    importChats: "I-import ang mga chat",
    importDestination: "Piliin kung saan ii-import ang {name}:",
    selectDestination: "Pumili ng destinasyon",
    recents: "Mga Kamakailan",
    import: "I-import",
    deleteProject: "I-delete ang proyekto",
    deleteProjectDescription:
      "Sigurado ka bang gusto mong i-delete ang {name}? Permanenteng ide-delete ang mga chat nito.",
    deleteProjectFiles: "Aalisin sa disk ang workspace folder ng proyekto.",
    deleteAll: "I-delete lahat",
    projectFallback: "proyekto",
    importFailed: "Nabigo ang pag-import.",
    nothingToExport: "Walang pag-uusap na ie-export.",
    exportFailed: "Nabigo ang pag-export.",
    renameChatFailed: "Hindi napalitan ang pangalan ng chat",
    archiveChatFailed: "Hindi na-archive ang chat",
    deleteChatFailed: "Hindi na-delete ang chat",
    deleteProjectFailed: "Hindi na-delete ang proyekto",
  },
  auth: {
    welcomeBack: "Maligayang pagbabalik",
    loginSubtitle: "Mag-sign in gamit ang iyong password.",
    setupTitle: "I-set up ang iyong account",
    setupSubtitle: "Gumawa ng bagong password",
    login: "Mag-sign in",
    changePassword: "Palitan ang password",
    alreadySetUp: "Naka-set up na ang password? ",
    backToLogin: "Bumalik sa pag-sign in",
    username: "Username",
    password: "Password",
    currentPassword: "Kasalukuyang password",
    newPassword: "Bagong password",
    confirmPassword: "Kumpirmahin ang password",
    setupCodeHint:
      "Unang pag-sign in? I-paste bilang password ang setup code na ibinigay sa iyo ng iyong administrator.",
    ownerHint:
      "May-ari ng installation? Mag-sign in bilang unsloth, o ang pangalang pang-sign in na nakikita sa Mga Setting > Mga Account.",
    currentSetupCodeHint:
      "I-paste bilang kasalukuyang password ang setup code na ginamit mo sa pag-sign in kanina.",
    minLength: "Dapat ay hindi bababa sa 8 character.",
    ensureMatch: "Pakitiyak na magkatugma ang mga password.",
    noSpaces: "Hindi puwedeng may space ang bagong password.",
    currentTooShort:
      "Dapat ay hindi bababa sa 8 character ang kasalukuyang password.",
    setupInitFailed:
      "Hindi masimulan ang setup. I-reload ang page at subukan ulit.",
    newTooShort: "Dapat ay hindi bababa sa 8 character ang bagong password.",
    mismatch: "Hindi magkatugma ang mga password.",
    mustDiffer:
      "Dapat ay iba ang bagong password sa iyong kasalukuyang password.",
    pleaseWait: "Pakihintay...",
    continueAsOwner: "Magpatuloy bilang may-ari",
    signInWithUsername: "Mag-sign in gamit ang iyong username at password.",
    bootstrapping: "Inihahanda pa ng auth ang default na admin account.",
    signInToChange:
      "Mag-sign in nang isang beses gamit ang mga paunang credential para palitan ang password.",
    alreadyUpdated:
      "Na-update na ang password. Gamitin ang screen ng pag-sign in.",
    busy: "Abala ang Unsloth. Subukan ulit maya-maya.",
    updateFailed: "Nabigo ang pag-update ng password.",
    loadFailed: "Hindi na-load.",
    failed: "Nabigo ang auth.",
    shuttingDown:
      "Nagsasara ang instance na ito: naaabot ito sa network at hindi kailanman napalitan ang default na password nito.",
    shutdownCountdown:
      "Naaabot sa network ang instance na ito at gumagamit pa rin ng default na password nito, kaya magsasara ito sa loob ng {time}. Kapag nagtakda ka ng password dito, patuloy itong tatakbo.",
  },
  routes: {
    chat: "Chat",
    api: "API",
    dataRecipes: "Data Recipes",
    login: "Mag-sign in",
    changePassword: "Palitan ang Password",
  },
  remoteCode: {
    unsafeTitle: "May natukoy na mga hindi ligtas na file",
    blockedTitle: "Naka-block ang custom code",
    enableTitle: "I-enable ang custom code para sa model na ito?",
    thisModel: "Ang model na ito",
    unsafeBody:
      "Naglalaman ang {model} ng mga file na minarkahang hindi ligtas ng security scan ng Hugging Face (halimbawa, isang mapanganib na pickle na magpapatakbo ng code kapag na-load ang model). Hindi ito maaaring i-load. Hindi kailanman na-download ang mga minarkahang file.",
    unsafeBodyFromProvider:
      "Naglalaman ang {model} mula sa {provider} ng mga file na minarkahang hindi ligtas ng security scan ng Hugging Face (halimbawa, isang mapanganib na pickle na magpapatakbo ng code kapag na-load ang model). Hindi ito maaaring i-load. Hindi kailanman na-download ang mga minarkahang file.",
    customCode:
      "Nagdedeklara ang {model} ng custom na Python code sa repository nito. {note}",
    customCodeFromProvider:
      "Nagdedeklara ang {model} mula sa {provider} ng custom na Python code sa repository nito. {note}",
    blockedNote:
      "Minarkahan ng security scan ang mga CRITICAL na isyu, kaya hindi ito maaaring i-enable.",
    reviewNote:
      "Suriin ang security scan sa ibaba. Magpatuloy lang kung pinagkakatiwalaan mo ang pinagmulan ng model.",
    trustNote:
      "Magpatuloy lang kung pinagkakatiwalaan mo ang pinagmulan ng model.",
    flaggedIntro:
      "Minarkahan ng aming awtomatikong scanner ang mga isyu, kabilang ang:",
    nothingFlagged:
      "Walang minarkahang nakababahalang file ang aming awtomatikong scanner, pero pakisuring muli.",
    enableAndContinue: "I-enable at magpatuloy",
  },
  tour: {
    badge: "gabay na tour",
    defaultTitle: "Mabilis na tour",
    defaultBody: "Ipapakilala namin sa iyo ang paligid.",
    skipTour: "Laktawan ang tour",
    skip: "Laktawan",
    back: "Bumalik",
    next: "Susunod",
    tip: "Tip: nilalaktawan ito ng `Esc`. Bina-block ng tour ang mga click para makapagbasa ka.",
    readMore: "Magbasa pa",
  },
  updateNotes: {
    openRelease: "Buksan ang release",
    openChangelog: "Buksan ang changelog",
    retry: "Subukan ulit",
    loadingChanges: "Nilo-load ang mga bagong pagbabago...",
    customLlamaRepo:
      "Sinusubaybayan ng install na ito ang custom na llama.cpp repository, kaya hindi maikukumpara ang mga pagbabago nito.",
    noItemisedNotes:
      "Mas luma ang build na ito kaysa sa mga naka-itemize na release note, kaya hindi maikukumpara ang mga pagbabago nito.",
    compareFailed: "Hindi maikumpara ang mga release na ito.",
    llamaChangesLabel:
      "Mga bagong pagbabago sa llama.cpp mula {from} hanggang {to}",
    showingChanges: "Ipinapakita ang {shown} sa {total} na pagbabago",
    noCarriedChanges:
      "Walang nakalistang bagong dalang pagbabago para sa build na ito.",
    notesFor: "Mga release note para sa {release}",
    notesForVersion: "Mga release note para sa bersyon {version}",
    notesTruncated: "Pinaikli ang mga note. Tingnan ang buong release notes.",
    summaryLabel: "Buod ng mga release note",
    moreCount: "+{count} pa",
    loadingNotes: "Nilo-load ang mga release note...",
    loadFailed: "Hindi ma-load ang mga release note.",
    noNotesFor: "Wala pang nailathalang release note para sa {release}.",
    noNotes: "Wala pang nailathalang release note.",
  },
  desktopStartup: {
    checking: "Sinusuri...",
    installPrompt: "Para i-install ang Unsloth, i-click ang Magsimula.",
    getStarted: "Magsimula",
    chatOnly: "Chat lang (GGUF): mas maliit na install, walang training",
    installLocation: "Lokasyon ng install:",
    changeLocation: "Baguhin",
    useDefaultLocation: "Gamitin ang default",
    longPaths: {
      title: "Hihingi ng pahintulot ang Windows",
      approve:
        "Aprubahan ang {tool} nang isang beses para i-on ang mahahabang file path para sa mga compiled kernel.",
      waiting: "Hinihintay ang pahintulot ng Windows...",
    },
    skip: "Laktawan",
    continue: "Magpatuloy",
    install: {
      preparing: "Inihahanda ang iyong workspace...",
      preparingDetail: "Sinusuri kung ano ang kailangan ng computer na ito.",
      downloading: "Dina-download ang mga kinakailangang component...",
      downloadingDetail: "Ginagawa ang iyong lokal na environment.",
      installing: "Ini-install ang Unsloth...",
      installingDetail: "Sine-set up ang mga lokal na AI tool...",
      finishing: "Tinatapos ang setup...",
      finishingDetail: "Inihahanda ang lahat.",
      continuesAutomatically: "Awtomatikong magpapatuloy ang setup.",
      fewMinutes: "Maaaring tumagal ito nang ilang minuto.",
      someTakeLonger: "Mas matagal i-configure ang ilang component.",
    },
    details: {
      showInstall: "Ipakita ang mga detalye ng installation",
      hideInstall: "Itago ang mga detalye ng installation",
      showSetup: "Ipakita ang mga detalye ng setup",
      hideSetup: "Itago ang mga detalye ng setup",
      showUpdate: "Ipakita ang mga detalye ng update",
      hideUpdate: "Itago ang mga detalye ng update",
    },
    repair: {
      upgrading: "Ina-upgrade sa buong install...",
      gettingReady: "Inihahanda ang mga bagay...",
      upgradingDetail:
        "Dina-download ang PyTorch at ang mga library nito para sa GPU ng machine na ito. Maaaring tumagal ito nang ilang minuto.",
      wontTakeLong: "Hindi ito magtatagal.",
      upgradeFailed: "Hindi ma-upgrade sa buong install",
      updateFailed: "Nabigo ang update",
    },
    closing: "Isinasara ang Unsloth Desktop...",
    closingDetail: "Pinapatay ang backend.",
    diskFull: "Kulang ang espasyo sa disk",
    setupFailed: "Nagkaproblema ang setup",
    tryAgain: "Subukan Ulit",
    retry: "Subukan ulit",
    elevation: {
      title: "Kailangan ng pahintulot",
      packages: "Kailangang i-install ang mga sumusunod na system package:",
      allow: "Payagan",
    },
    starting: "Sinisimulan ang Unsloth...",
    loadingProjects: "Nilo-load ang mga proyekto...",
    loadingModels: "Nilo-load ang mga model...",
    nearlyDone: "Halos tapos na...",
    finishingUpdate: "Tinatapos ang update...",
    serverStopped: "Nahinto ang server",
    startServer: "Simulan ang Server",
    somethingWrong: "Nagkaroon ng problema",
    diagnostics: {
      copy: "Kopyahin ang Diagnostics",
      copying: "Kinokopya...",
      clipboardFailed:
        "Nabigo ang pagkopya sa clipboard. Piliin at kopyahin ang diagnostics sa ibaba.",
      copyFailed: "Nabigo ang pagkopya ng diagnostics: {error}",
    },
  },
  desktopUpdate: {
    updatingBackend: "Ina-update ang backend...",
    downloadingApp: "Dina-download ang update ng app...",
    installing: "Ini-install ang update...",
    failed: "Nabigo ang update",
    updating: "Nag-a-update...",
    updatingBackendDetail:
      "Maaaring tumagal ito nang ilang minuto. Huwag isara ang app.",
    percentDownloaded: "{percent}% na-download",
    restartShortly: "Magre-restart ang app sa ilang sandali.",
    failedDetail: "Nagkaroon ng problema habang nag-a-update.",
    downloadingFile: "Dina-download ang {file}",
    downloadingAppLabel: "Dina-download ang update ng app",
    skipRestart: "Laktawan at I-restart",
    banner: {
      dismiss: "I-dismiss ang notification ng update ng app",
      failedTitle: "Nabigo ang update ng app",
      newVersion: "Bagong bersyon ng Unsloth",
      backendRecovered:
        "Naibalik ang backend. Available pa rin ang diagnostics.",
      linuxManual:
        "Buksan ang release page sa GitHub para i-install ang Linux package",
      available: "May bagong update ng app",
      showNotes: "Ipakita ang mga release note",
      hideNotes: "Itago ang mga release note",
      copyDiagnostics: "Kopyahin ang diagnostics",
      later: "Mamaya",
      remindLater: "Paalalahanan ako mamaya",
      openReleasePage: "Buksan ang release page",
      retry: "Subukan ulit ang update",
      update: "I-update",
    },
  },
  desktopShell: {
    navigation: "Sidebar at nabigasyon ng page",
    collapseSidebar: "I-collapse ang sidebar",
    expandSidebar: "I-expand ang sidebar",
    goBack: "Bumalik",
    goForward: "Sumulong",
    titlebar: "Titlebar ng window",
    windowControls: "Mga kontrol ng window",
    minimize: "I-minimize ang window",
    maximize: "I-maximize ang window",
    restore: "I-restore ang window",
    close: "Isara ang window",
  },
  runtimeUpdate: {
    installAudio: "I-install ang audio runtime",
    installingAudio: "Ini-install ang audio runtime...",
    unknownVersion: "hindi alam",
    failed: "Nabigo ang update: {error}",
    unknownError: "hindi kilalang error",
    skipVersions: "Laktawan ang mga bersyong ito",
    skipChange: "Laktawan ang pagbabagong ito",
    skipVersion: "Laktawan ang bersyong ito",
    skippedVersion: "Nilaktawan muna ang {component} {version}",
    skippedUpdate: "Nilaktawan muna ang update ng {component}",
    skippedUpdates: "Nilaktawan muna ang mga update ng {components}",
    vulkanFaster:
      "Mahigit 10% na mas mabilis ang Vulkan kaysa sa ROCM. Hindi kailangang mag-restart",
    noRestart: "Hindi kailangang mag-restart",
    downloadSize: "{size} na download",
    backendChange: "Backend: {change}",
    updating: "Ina-update ang {component}...",
    updatingLabel: "Ina-update ang {component}",
    installingPhase: "Ini-install ang {component} ({current} sa {total})...",
    installingCount: "Ini-install ang {count} na update...",
    bundleTitle: "{count} na runtime update",
    newBackend: "Bagong {component} backend",
    newUpdate: "Bagong update ng {component}",
    hideUpdate:
      "Itago ang update ng {component} hanggang muling buksan ang Unsloth",
    hideUpdates:
      "Itago ang mga update ng {components} hanggang muling buksan ang Unsloth",
    whatsNew: "Ano ang bago",
    hideWhatsNew: "Itago ang ano ang bago",
    later: "Mamaya",
    update: "I-update",
    undo: "I-undo",
  },
} satisfies DeepPartialMessageTree<typeof en>;
