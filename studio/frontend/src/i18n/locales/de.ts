// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DeepPartialMessageTree } from "../types";
import type { en } from "./en";

export const de = {
  picker: {
    onDevice: "Auf dem Gerät",
    huggingFace: "Hugging Face",
    retry: "Erneut versuchen",
    loadMore: "Mehr laden",
    offlineTitle: "Sie sind offline",
    offlineBody:
      "Wechseln Sie zu „Gerät“, um zwischengespeicherte oder lokale {noun} zu verwenden.",
    offlineSwitchDevice: "Gerät",
    searchAriaLabel: "Suche: {noun}",
    modelSourceAriaLabel: "Modellquelle",
    hubSectionAriaLabel: "Hub-Bereich",
    modelDropped: "Nicht mehr angeboten",
    modelDroppedByProvider: "{provider} · nicht mehr angeboten",
    modelDisabled: "Nicht aktiviert",
    modelDisabledByProvider: "{provider} · nicht aktiviert",
    multipleMatches:
      "Mehrere passende {noun} gefunden. Wählen Sie einen Eintrag aus der Liste aus.",
    rateLimitedTitle: "Hugging Face-Ratenlimit erreicht",
    rateLimitedBody:
      "Warten Sie einen Moment und wiederholen Sie dann die Suche ({noun}).",
    hfToken: {
      label: "HF-Token",
      saved: "Gespeichert",
      add: "Hinzufügen",
      savedAriaLabel: "Hugging Face-Token gespeichert",
      addAriaLabel: "Hugging Face-Token festlegen",
      savedHint:
        "Token gespeichert. Der Zugriff wird bei der Verwendung geprüft.",
      addHint:
        "Legen Sie ein Token fest, um auf private und zugriffsbeschränkte Repositories zuzugreifen.",
    },
  },
  common: {
    cancel: "Abbrechen",
    close: "Schließen",
    delete: "Löschen",
    done: "Fertig",
    error: "Fehler",
    export: "Exportieren",
    help: "Hilfe",
    loading: "Wird geladen…",
    new: "Neu",
    rename: "Umbenennen",
    save: "Speichern",
    saving: "Wird gespeichert…",
    search: "Suchen",
    shutdown: "Herunterfahren",
  },
  shell: {
    beta: "BETA",
    brand: "unsloth",
    product: "Unsloth",
    accountMenu: "Kontomenü von {name}",
    updateAvailable: "Update verfügbar",
    resize: {
      collapse: "Zum Einklappen klicken",
      expand: "Zum Ausklappen klicken",
      drag: "Zum Ändern der Größe ziehen",
    },
    aria: {
      home: "Unsloth-Startseite",
      closeSidebar: "Seitenleiste schließen",
      openSidebar: "Seitenleiste öffnen",
      resizeSidebar: "Seitenleiste anpassen oder einklappen",
      resizeRunSettings: "Ausführungseinstellungen anpassen oder schließen",
      openRunSettings: "Ausführungseinstellungen öffnen",
      chatOptions: "Chat-Optionen",
      runOptions: "Trainingslauf-Optionen",
    },
    navigation: {
      newChat: "Neuer Chat",
      returnToChat: "Zurück zum Chat",
      returnToChats: "Zurück zu {count} Chats",
      chatGenerating: "Wird generiert",
      compare: "Vergleichen",
      search: "Suchen",
      hub: "Modell-Hub",
      projects: "Projekte",
      train: "Trainieren",
      recipes: "Rezepte",
      images: "Bilder",
      video: "Video",
      audio: "Audio",
      trainChecking: "Dieser Rechner wird auf Trainingsunterstützung geprüft...",
      videoChecking: "Dieser Rechner wird auf Video-Unterstützung geprüft...",
      more: "Mehr",
      customizeSidebar: "Seitenleiste anpassen",
      newBadge: "Neu",
      export: "Exportieren",
      recents: "Zuletzt verwendet",
      noChatsYet: "Noch keine Chats",
      showMore: "Mehr anzeigen",
      showLess: "Weniger anzeigen",
      settings: "Einstellungen",
      api: "API",
      lightMode: "Heller Modus",
      darkMode: "Dunkler Modus",
      guidedTour: "Geführte Tour",
      help: "Hilfe",
      logOut: "Abmelden",
      shutdown: "Herunterfahren",
    },
    notFound: {
      title: "Seite nicht gefunden",
      description: "{path} existiert nicht.",
      backToChat: "Zurück zum Chat",
    },
    selection: {
      pinProjects: "Projekte anheften",
      unpinProjects: "Projekte lösen",
      deleteProjects: "Projekte löschen",
      deleteProjectsTitle: "Projekte löschen",
      deleteProjectsDescription:
        "{count} Projekte löschen? Ihre Chats werden dauerhaft gelöscht.",
      deleteProjectsFilesDescription:
        "Der Arbeitsbereich-Ordner jedes Projekts wird von der Festplatte entfernt.",
      countSelected: "{count} ausgewählt",
      pinChats: "Chats anheften",
      unpinChats: "Chats lösen",
      archiveChats: "Chats archivieren",
      markUnread: "Als ungelesen markieren",
      deleteChats: "Chats löschen",
      deleteTitle: "Chats löschen",
      deleteDescription: "{count} Chats löschen? Das lässt sich nicht rückgängig machen.",
      deleteFilesDescription:
        "Der eigene Sandbox-Ordner jedes Chats wird von der Festplatte entfernt. Dateien, die sie in einem Projekt erstellt haben, bleiben im Arbeitsbereich dieses Projekts.",
      deleteFilesLabel: "Dateien und Sandbox-Ordner löschen",
      deleteChatFilesDescription:
        "Der eigene Sandbox-Ordner dieses Chats wird von der Festplatte entfernt. Dateien, die er in einem Projekt geschrieben hat, bleiben im Arbeitsbereich des Projekts.",
    },
    organize: {
      sidebarHeading: "Seitenleiste organisieren",
      byProject: "Nach Projekt",
      inOneList: "In einer Liste",
      sortChatsBy: "Chats sortieren nach",
      sortPinnedBy: "Angeheftete sortieren nach",
      priority: "Priorität",
      lastUpdated: "Zuletzt aktualisiert",
      manualOrder: "Manuelle Reihenfolge",
      moveUp: "Nach oben",
      moveDown: "Nach unten",
      organizeChats: "Chats organisieren",
      organizeProjects: "Projekte organisieren",
      sortPinnedChats: "Angeheftete Chats sortieren",
    },
    dialog: {
      deleteChat: {
        title: "Chat löschen",
        description: "Möchten Sie den Chat „{name}“ wirklich löschen?",
      },
      deleteRun: {
        title: "Trainingslauf löschen",
        description: "Möchten Sie den Trainingslauf „{name}“ wirklich löschen?",
      },
      renameChat: {
        title: "Chat umbenennen",
        placeholder: "Chat-Titel",
      },
      renameRun: {
        title: "Lauf umbenennen",
        placeholder: "Laufname",
      },
    },
    toast: {
      cannotDeleteRunningRun:
        "Ein laufender Trainingslauf kann nicht gelöscht werden",
      failedToDeleteChat: "Chat konnte nicht gelöscht werden",
      failedToDeleteRun: "Lauf konnte nicht gelöscht werden",
      failedToRenameChat: "Chat konnte nicht umbenannt werden",
      failedToRenameRun: "Lauf konnte nicht umbenannt werden",
    },
  },
  settings: {
    title: "Einstellungen",
    dialog: {
      title: "Einstellungen",
      description: "Verwalten Sie Ihre Unsloth-Einstellungen.",
      closeAriaLabel: "Einstellungen schließen",
      searchPlaceholder: "Einstellungen durchsuchen…",
      searchNoResults: "Keine Einstellungen gefunden.",
      panelFailed: "Dieser Bereich konnte nicht geladen werden.",
      panelReload: "Neu laden",
    },
    tabs: {
      general: "Allgemein",
      profile: "Profil",
      appearance: "Darstellung",
      resources: "System",
      chat: "Chat",
      connections: "Verbindungen",
      apiKeys: "API",
      remoteLan: "Remote & LAN",
      about: "Info",
      data: "Daten",
      agents: "Agenten",
      debugging: "Protokolle",
      voice: "Sprachfunktionen",
      keyboardShortcuts: "Kürzel",
    },
    keyboardShortcuts: {
      title: "Tastenkürzel",
      description:
        "Ändere ein Kürzel oder lösche es, um die Tastenkombination für Browser oder Betriebssystem freizugeben.",
      searchPlaceholder: "Kürzel suchen…",
      noResults: "Keine Kürzel passen zu dieser Suche.",
      unassigned: "Nicht zugewiesen",
      recording: "Tasten drücken…",
      recordingHint: "Neue Tastenkombination drücken oder Esc zum Abbrechen.",
      needsModifier: "⌘, Strg oder Alt hinzufügen. Eine einzelne Taste würde die Eingabe verschlucken.",
      conflict: "Wird auch von einem anderen Kürzel verwendet",
      conflictShadowed: "Ein anderes Kürzel belegt diese Kombination und läuft stattdessen",
      edit: "Kürzel ändern",
      clear: "Kürzel entfernen",
      reset: "Standard wiederherstellen",
      resetAll: "Alle auf Standard zurücksetzen",
      primarySlot: "Kürzel",
      alternateSlot: "Alternatives Kürzel",
      browserReserved:
        "Dein Browser behält diese Tastenkombination unter Umständen für sich. In der Desktop-App funktioniert sie.",
      actions: {
        openSettings: {
          label: "Einstellungen öffnen",
          description: "Den Einstellungsdialog öffnen",
        },
        openKeyboardShortcuts: {
          label: "Tastenkürzel",
          description: "Diese Kürzelliste öffnen",
        },
        searchChats: {
          label: "Chats durchsuchen",
          description: "Die Chat-Suche öffnen",
        },
        openMcpServers: {
          label: "MCP-Server",
          description: "MCP-Server für diesen Chat einrichten",
        },
        logOut: {
          label: "Abmelden",
          description: "Von Unsloth abmelden",
        },
        approveToolRequest: {
          label: "Anfrage genehmigen",
          description: "Den wartenden Tool-Aufruf zulassen",
        },
        declineToolRequest: {
          label: "Anfrage ablehnen",
          description: "Den wartenden Tool-Aufruf ablehnen",
        },
        newChat: {
          label: "Neuer Chat",
          description: "Einen neuen Chat starten",
        },
        newTemporaryChat: {
          label: "Neuer temporärer Chat",
          description: "Einen Chat starten, der nicht im Verlauf gespeichert wird",
        },
        newStandaloneChat: {
          label: "Neuer eigenständiger Chat",
          description: "Einen neuen Chat außerhalb jedes Projekts starten",
        },
        archiveChat: {
          label: "Chat archivieren",
          description: "Die ausgewählten Chats archivieren, sonst den aktuellen",
        },
        markChatUnread: {
          label: "Als ungelesen markieren",
          description: "Die ausgewählten Chats als ungelesen markieren, sonst den aktuellen",
        },
        togglePinChat: {
          label: "Anheften umschalten",
          description: "Die ausgewählten Chats anheften oder lösen, sonst den aktuellen",
        },
        selectAllChats: {
          label: "Alle Chats auswählen",
          description: "Jeden Chat in der Seitenleiste auswählen",
        },
        clearChatSelection: {
          label: "Auswahl aufheben",
          description: "Die ausgewählten Chats abwählen. Escape hebt sie ebenfalls auf",
        },
        deleteSelectedChats: {
          label: "Ausgewählte Chats löschen",
          description: "Jeden ausgewählten Chat löschen",
        },
        nextRecentlyViewedChat: {
          label: "Nächster zuletzt geöffneter Chat",
          description: "Vorwärts durch zuletzt geöffnete Chats blättern",
        },
        previousRecentlyViewedChat: {
          label: "Vorheriger zuletzt geöffneter Chat",
          description: "Rückwärts durch zuletzt geöffnete Chats blättern",
        },
        nextChat: {
          label: "Nächster Chat",
          description: "Zum nächsten Chat in der Seitenleiste wechseln",
        },
        previousChat: {
          label: "Vorheriger Chat",
          description: "Zum vorherigen Chat in der Seitenleiste wechseln",
        },
        nextChatNeedingAttention: {
          label: "Nächster Chat mit Aktivität",
          description: "Zum nächsten generierenden, wartenden oder ungelesenen Chat wechseln",
        },
        clearAllUnreads: {
          label: "Alle als gelesen markieren",
          description: "Jeden Chat als gelesen markieren",
        },
        goToRecentChat1: {
          label: "Zu letztem Chat 1 wechseln",
          description: "Chat 1 unter „Zuletzt“ öffnen",
        },
        goToRecentChat2: {
          label: "Zu letztem Chat 2 wechseln",
          description: "Chat 2 unter „Zuletzt“ öffnen",
        },
        goToRecentChat3: {
          label: "Zu letztem Chat 3 wechseln",
          description: "Chat 3 unter „Zuletzt“ öffnen",
        },
        goToRecentChat4: {
          label: "Zu letztem Chat 4 wechseln",
          description: "Chat 4 unter „Zuletzt“ öffnen",
        },
        goToRecentChat5: {
          label: "Zu letztem Chat 5 wechseln",
          description: "Chat 5 unter „Zuletzt“ öffnen",
        },
        goToRecentChat6: {
          label: "Zu letztem Chat 6 wechseln",
          description: "Chat 6 unter „Zuletzt“ öffnen",
        },
        switchToChat: {
          label: "Zu Chat wechseln",
          description: "Zum Chat-Arbeitsbereich gehen",
        },
        switchToProjects: {
          label: "Zu Projekte wechseln",
          description: "Zum Projekte-Arbeitsbereich gehen",
        },
        switchToHub: {
          label: "Zu Modell-Hub wechseln",
          description: "Zum Modell-Hub gehen",
        },
        switchToTrain: {
          label: "Zu Training wechseln",
          description: "Zum Trainings-Arbeitsbereich gehen",
        },
        switchToRecipes: {
          label: "Zu Recipes wechseln",
          description: "Zu Data Recipes gehen",
        },
        switchToImages: {
          label: "Zu Bilder wechseln",
          description: "Zum Bilder-Arbeitsbereich gehen",
        },
        switchToVideo: {
          label: "Zu Video wechseln",
          description: "Zum Video-Arbeitsbereich gehen",
        },
        switchToAudio: {
          label: "Zu Audio wechseln",
          description: "Zum Audio-Arbeitsbereich gehen",
        },
        switchToExport: {
          label: "Zu Export wechseln",
          description: "Zum Export-Arbeitsbereich gehen",
        },
        toggleSidebar: {
          label: "Seitenleiste umschalten",
          description: "Seitenleiste ein- oder ausblenden",
        },
        toggleApiMonitor: {
          label: "API-Aktivität umschalten",
          description: "Den API-Aktivitätsmonitor ein- oder ausblenden",
        },
        openModelPicker: {
          label: "Modellauswahl öffnen",
          description: "Das Modell für diesen Chat wählen",
        },
        openProjectPicker: {
          label: "Projektauswahl öffnen",
          description: "Über die Chat-Kopfzeile zu einem anderen Projekt wechseln",
        },
        startDictation: {
          label: "Diktat",
          description: "Diktat im Eingabefeld starten oder stoppen",
        },
        attachFiles: {
          label: "Fotos und Dateien anhängen",
          description: "Einen Anhang zum Eingabefeld hinzufügen",
        },
        sendMessage: {
          label: "Nachricht senden",
          description: "Den Inhalt des Eingabefelds senden",
        },
        cycleReasoningEffort: {
          label: "Denkaufwand durchschalten",
          description: "Durch die Stufen des Denkaufwands blättern",
        },
        increaseReasoningEffort: {
          label: "Denkaufwand erhöhen",
          description: "Den Denkaufwand um eine Stufe anheben",
        },
        decreaseReasoningEffort: {
          label: "Denkaufwand verringern",
          description: "Den Denkaufwand um eine Stufe senken",
        },
        toggleFastMode: {
          label: "Fast-Modus umschalten",
          description: "Den Fast-Modus ein- oder ausschalten",
        },
        renameChat: {
          label: "Chat umbenennen",
          description: "Den aktuellen Chat umbenennen",
        },
        forkChat: {
          label: "Chat verzweigen",
          description: "Aus der letzten Nachricht einen neuen Chat abzweigen",
        },
        copyChatAsMarkdown: {
          label: "Als Markdown kopieren",
          description: "Den gesamten Chat als Markdown in die Zwischenablage kopieren",
        },
        copySessionId: {
          label: "Sitzungs-ID kopieren",
          description: "Die Sandbox-Sitzungs-ID dieses Chats kopieren",
        },
      },
    },
    debugging: {
      logSection: "Protokolldatei",
      source: "Protokolldatei",
      sourceHint: "Die Modell-Runner schreiben eigene Protokolle. Ein fehlgeschlagener Ladevorgang oder eine fehlgeschlagene Generierung wird deshalb oft dort erklärt und nicht im Server-Protokoll.",
      path: "Speicherort",
      pathCopy: "Pfad kopieren",
      refreshSection: "Aktualisierung",
      mode: "Modus",
      modeLive: "Live",
      modeInterval: "Alle 3 Sekunden",
      modeManual: "Manuell",
      refreshNow: "Jetzt aktualisieren",
      privacyNote: "Zugangsdaten werden in dieser Ansicht maskiert. In der Datei auf dem Datenträger sind sie nicht maskiert.",
      copyVisible: "Sichtbares Protokoll kopieren",
      empty: "Es wurde noch nichts protokolliert.",
      disabled: "Die Protokollierung in eine Datei ist deaktiviert (UNSLOTH_STUDIO_NO_FILE_LOG=1).",
      missing: "Es wurde keine Protokolldatei gefunden.",
      unreadable: "Die Protokolldatei konnte nicht gelesen werden.",
      timeout: "Die Protokollanfrage hat das Zeitlimit uberschritten. Der Server ist moglicherweise nicht erreichbar.",
      droppedNotice: "Einige Zeilen wurden übersprungen: Das Protokoll wurde schneller geschrieben, als es gelesen werden konnte.",
      morePending: "Weitere Zeilen werden noch gelesen; sie erscheinen bei der nachsten Aktualisierung.",
      staleSession: "Die Protokollierung in Dateien ist deaktiviert, daher ist dies eine fruhere Sitzung und wird nicht aktualisiert.",
      keywords: "Fehlersuche Debugging Protokoll Protokolle Log Logs Fehler Absturz Stacktrace Diagnose Problembehebung debug",
    },
    voice: {
      title: "Sprachfunktionen",
      description: "Mikrofon, Diktat, Spracherkennung und Vorlesen",
      dictation: {
        sectionTitle: "Diktat",
        engineLabel: "Diktat-Engine",
        engineBrowser: "Browser",
        engineBrowserDescription:
          "Transkribiert Audio über den Sprachdienst Ihres Browsers. Wählen Sie „Lokale Transkription“, um ein STT-Modell zu verwenden.",
        engineModel: "Lokale Transkription",
        engineModelDescription:
          "Führt ein Spracherkennungsmodell (STT) lokal aus und funktioniert offline. Zuerst herunterladen und laden; nach einer Zeit ohne Nutzung wird es automatisch wieder entladen.",
        engineCustom: "Benutzerdefinierter Endpunkt",
        engineCustomDescription:
          "Sendet Audioaufnahmen an einen OpenAI-kompatiblen STT-Server aus Ihren Verbindungen.",
        connectionLabel: "Verbindung",
        connectionDescription:
          "Fügen Sie unter Verbindungen einen OpenAI-kompatiblen Server und optional einen API-Schlüssel hinzu.",
        connectionPlaceholder: "Verbindung auswählen",
        connectionEmpty: "Keine Verbindungen verfügbar",
        customModelLabel: "Modell",
        customModelDescription:
          "Modellname, der an /v1/audio/transcriptions gesendet wird.",
        sttModelLabel: "Spracherkennungsmodell",
        sttModelDescription:
          "Wählen oder suchen Sie ein STT-Modell für die lokale Ausführung.",
        sttModelSearchPlaceholder: "Modell suchen",
        sttModelSearching: "Hugging Face wird durchsucht…",
        sttModelValidating: "Whisper-Kompatibilität wird geprüft…",
        sttModelNoResults: "Keine Whisper-Modelle gefunden",
        sttModelInvalid:
          "Dieses Repository kann nicht für das Diktat verwendet werden",
        sttModelFailed: "Das STT-Modell konnte nicht geladen werden",
        sttModelUnsupported:
          "Aufnahme wird in diesem Browser nicht unterstützt",
        sttChecking: "Wird geprüft…",
        sttOnDemand: "Heruntergeladen",
        sttLoadingModel: "Modell wird geladen…",
        sttReady: "Auf {device} geladen",
        sttLoaded: "Geladen",
        sttUnavailable:
          "Auf diesem Server nicht installiert. Führen Sie `unsloth studio update` aus, um das lokale Diktat zu aktivieren.",
        sttRetry: "Erneut versuchen",
        sttDownloadChecking: "Download-Status wird geprüft…",
        sttNotDownloaded: "Nicht heruntergeladen",
        sttDownloadStatusFailed:
          "Der Download-Status konnte nicht geprüft werden",
        sttDownload: "Herunterladen",
        sttDownloadConfirmTitle: "{model} herunterladen?",
        sttDownloadConfirmBody:
          "Das lokale Diktat läuft vollständig offline, benötigt aber zuerst das Spracherkennungsmodell {model}. Etwa {size}, wird einmalig in Ihren Hugging Face-Cache geladen.",
        sttDownloadConfirmBodyUnsized:
          "Das lokale Diktat läuft vollständig offline, benötigt aber zuerst das Spracherkennungsmodell {model}. Es wird einmalig in Ihren Hugging Face-Cache geladen.",
        sttOpenVoiceSettings: "Einstellungen für Sprachfunktionen öffnen",
        sttDownloadStarted: "{model} wird heruntergeladen",
        sttDownloading: "Wird heruntergeladen… {progress} %",
        sttCancelDownload: "Abbrechen",
        sttCancellingDownload: "Wird abgebrochen…",
        sttCancelDownloadFailed: "Der Download konnte nicht abgebrochen werden",
        sttDownloadComplete: "Spracherkennungsmodell heruntergeladen",
        sttModelReady: "{model} ist bereit für das Diktat",
        sttRecommended: "Empfohlen",
        sttDownloadFailed:
          "Das Spracherkennungsmodell konnte nicht heruntergeladen werden",
        sttLoad: "Laden",
        sttUnload: "Entladen",
        sttUnloading: "Wird entladen…",
        microphoneLabel: "Mikrofon",
        microphoneFallbackName: "Mikrofon {index}",
        microphoneDescription: "Wird für das Diktat verwendet",
        microphoneFallbackHint:
          "Wird für das Diktat verwendet. Greift auf den Systemstandard zurück, wenn die Sprach-Engine des Browsers dieses Gerät nicht nutzen kann",
        microphoneGrantDescription:
          "Erlauben Sie den Mikrofonzugriff, um Gerätenamen anzuzeigen",
        allowMicrophone: "Mikrofonzugriff erlauben",
        micAccessBlocked:
          "Der Mikrofonzugriff wurde blockiert. Erlauben Sie den Mikrofonzugriff für diese Unsloth-Seite und versuchen Sie es erneut.",
        micAccessBlockedDesktop:
          "Der Mikrofonzugriff wurde blockiert. Versuchen Sie es erneut und wählen Sie Zulassen, oder aktivieren Sie das Mikrofon in den Datenschutzeinstellungen des Systems.",
        micAccessUnsupported:
          "Der Mikrofonzugriff wird in diesem Browser oder Kontext nicht unterstützt.",
        systemDefault: "Systemstandard",
        savedMicDisconnected: "Gespeichertes Mikrofon (nicht verbunden)",
        languageLabel: "Diktatsprache",
        languageDescription: "Zu erkennende Sprache",
        languageAuto: "Automatisch (Browsersprache)",
        languageAutoDetect: "Automatisch (Sprache erkennen)",
      },
      dictionary: {
        sectionTitle: "Diktatwörterbuch",
        sectionDescription:
          "Legen Sie fest, wie das Diktat bestimmte Wörter oder Wendungen schreibt",
        manageLabel: "Eigene Schreibweisen",
        manage: "Verwalten",
        backToVoice: "Zurück zu den Sprachfunktionen",
        addEntry: "Eintrag hinzufügen",
        newEntryAria: "Neuer Wörterbucheintrag",
        entryPlaceholder: "Erika Mustermann",
        entryAria: "Wörterbucheintrag {index}",
        removeEntryAria: "Wörterbucheintrag {index} entfernen",
      },
      recents: {
        sectionTitle: "Diktatverlauf",
        sectionDescription:
          "Jedes Diktat wird hier gespeichert, damit Sie den Text wiederherstellen können",
        manageLabel: "Diktatverlauf",
        manage: "Verwalten",
        pageDescription:
          "Jedes Diktat wird gespeichert. Sie können Diktate ansehen, kopieren oder löschen oder den Chat öffnen, in dem ein Diktat verwendet wurde.",
        searchPlaceholder: "Diktate durchsuchen",
        sortLabel: "Diktate sortieren",
        sortNewest: "Neueste",
        sortOldest: "Älteste",
        sortAlpha: "A bis Z",
        noMatches: "Keine Diktate entsprechen Ihrer Suche",
        detailTitle: "Gespeichertes Diktat",
        backToVoice: "Zurück zu den Sprachfunktionen",
        backToRecents: "Zurück zu den letzten Diktaten",
        view: "Vollständiges Diktat ansehen",
        empty: "Noch keine Diktate",
        dictationColumn: "Diktat",
        dateColumn: "Erstellungsdatum",
        copy: "Diktat kopieren",
        copied: "In die Zwischenablage kopiert",
        copyFailed: "Kopieren in die Zwischenablage nicht möglich",
        delete: "Diktat löschen",
        deleteTitle: "Diktat löschen",
        deleteDescription:
          "Dieses gespeicherte Diktat löschen? Das lässt sich nicht rückgängig machen.",
        deleteLinkedDescription:
          "Dieses gespeicherte Diktat löschen? Sie können auch den Chat löschen, in dem es verwendet wurde. Das lässt sich nicht rückgängig machen.",
        deleteWithChat: "Chat und Diktat löschen",
        deleteWithChatFailed: "Der Chat konnte nicht gelöscht werden",
        clear: "Verlauf löschen",
        clearTitle: "Diktatverlauf löschen",
        clearDescription:
          "Alle gespeicherten Diktate löschen? Das lässt sich nicht rückgängig machen.",
        clearConfirm: "Alle löschen",
        showMore: "Mehr anzeigen ({count})",
        openChat: "Chat öffnen",
      },
      readAloud: {
        sectionTitle: "Vorlesen",
        buttonLabel: "Vorlesen-Schaltfläche",
        buttonDescription: "Bei Assistentenantworten anzeigen",
        engineLabel: "TTS-Engine",
        engineSystemDescription: "Auf dem Gerät integrierte Stimmen",
        engineStudioDescription:
          "Verwendet das geladene Audiomodell (z. B. Orpheus)",
        engineSystem: "Systemstimmen",
        engineStudio: "TTS-Modell laden",
        engineCustom: "Eigener Endpunkt",
        engineCustomDescription:
          "Ein OpenAI-kompatibler TTS-Server aus deinen Verbindungen (z. B. Kokoro)",
        connectionLabel: "Verbindung",
        connectionDescription:
          "Einen OpenAI-kompatiblen Server im Tab „Verbindungen“ hinzufügen",
        connectionPlaceholder: "Verbindung auswählen",
        customModelLabel: "Modell",
        customVoiceDescription:
          "Vom Endpunkt erwarteter Stimmenname; Standard ist alloy",
        modelLabel: "TTS-Modell",
        modelDescription:
          "Laden Sie ein Audiomodell über die Modellauswahl (z. B. Orpheus TTS)",
        openAudioAction: "Audio öffnen",
        voiceLabel: "Stimme",
        voiceDescription: "Beste Stimmen auf diesem Gerät",
        speedLabel: "Geschwindigkeit",
        pitchLabel: "Tonhöhe",
        volumeLabel: "Lautstärke",
        previewLabel: "Stimme anhören",
        previewDescription: "Eine kurze Hörprobe abspielen",
        previewFailed: "Die TTS-Vorschau ist fehlgeschlagen",
        previewAction: "Anhören",
        preparingAction: "Wird erzeugt…",
        stopAction: "Stopp",
        ttsLabel: "Sprachausgabe",
        notSupported: "In diesem Browser nicht unterstützt",
      },
    },
    general: {
      title: "Allgemein",
      description: "Globale Einstellungen für Unsloth.",
      account: "Konto",
      huggingFaceToken: "Hugging-Face-Token",
      huggingFaceTokenDescription:
        "Wird verwendet, um zugriffsbeschränkte Modelle zu laden und Artefakte hochzuladen.",
      hideToken: "Token verbergen",
      showToken: "Token anzeigen",
      clearToken: "Löschen",
      checkingToken: "Token wird geprüft...",
      tokenValidated: "Token validiert",
      password: "Passwort",
      passwordDescription:
        "Ändern Sie das Passwort für dieses Unsloth-Konto.",
      passwordDialog: {
        trigger: "Passwort ändern",
        title: "Passwort ändern",
        description:
          "Geben Sie Ihr aktuelles Passwort ein und wählen Sie ein neues (mindestens {minLength} Zeichen).",
        setTrigger: "Remote-Passwort festlegen",
        setTitle: "Remote-Passwort festlegen",
        setDescription:
          "Wählen Sie das Passwort, mit dem sich entfernte Browser als unsloth anmelden (mindestens {minLength} Zeichen). Die Unsloth Desktop-App meldet sich weiterhin automatisch an.",
        setSubmit: "Passwort festlegen",
        setting: "Wird festgelegt...",
        setDone: "Passwort festgelegt.",
        currentPassword: "Aktuelles Passwort",
        newPassword: "Neues Passwort",
        confirmPassword: "Neues Passwort bestätigen",
        currentTooShort:
          "Das aktuelle Passwort muss mindestens {minLength} Zeichen haben.",
        newTooShort:
          "Das neue Passwort muss mindestens {minLength} Zeichen haben.",
        mismatch: "Die Passwörter stimmen nicht überein.",
        samePassword:
          "Das neue Passwort muss sich vom aktuellen Passwort unterscheiden.",
        update: "Passwort aktualisieren",
        updating: "Wird aktualisiert...",
        updated: "Passwort aktualisiert.",
        updateFailed: "Passwortaktualisierung fehlgeschlagen.",
        newHasSpaces: "Das neue Passwort darf keine Leerzeichen enthalten.",
      },
      chatDefaults: "Chat-Standardeinstellungen",
      autoTitleNewChats: "Neue Chats automatisch benennen",
      autoTitleNewChatsDescription:
        "Erzeugt einen kurzen Titel aus der ersten Nachricht.",
      helperLlm: {
        sectionTitle: "Helfer-LLM",
        preloadOnStartup: "Helfer-LLM beim Start vorab zwischenspeichern",
        preloadOnStartupDescription:
          "Lädt das AI-Assist-Helfermodell beim Start im Hintergrund herunter. Standardmäßig aus; AI Assist kann es weiterhin bei Bedarf abrufen.",
        disabledByEnv:
          "Deaktiviert durch UNSLOTH_HELPER_MODEL_DISABLE in der Backend-Umgebung.",
        loadError: "Helfer-LLM-Einstellungen konnten nicht geladen werden.",
        saveError:
          "Helfer-LLM-Einstellungen konnten nicht gespeichert werden.",
      },
      modelAutoSwitch: {
        sectionTitle: "Automatischer Modellwechsel (OpenAI API)",
        enable: "Modell je Anfrage wechseln",
        enableDescription:
          "Lädt vor der Verarbeitung einer API-Anfrage ein darin angegebenes, bereits heruntergeladenes GGUF. Standardmäßig deaktiviert.",
        idleUnload: "Automatisches Entladen bei Inaktivität",
        idleUnloadDescription:
          "Gibt VRAM nach der angegebenen Anzahl von Sekunden ohne Aktivität frei. Bei 0 bleibt das Modell geladen; der Mindestwert ist 60.",
        idleSecondsAriaLabel:
          "Inaktivitätsdauer bis zum automatischen Entladen in Sekunden",
        mediaEnable: "Bild- und Videomodell je Anfrage wechseln",
        mediaEnableDescription:
          "Lädt vor der Generierung ein darin angegebenes, bereits heruntergeladenes Bild- oder Videomodell aus einer API-Anfrage. Eine eigene Einstellung: Die Einstellung darüber gilt nur für das Chat-Modell. Standardmäßig deaktiviert.",
        mediaIdleUnload:
          "Automatisches Entladen bei Inaktivität für Bild und Video",
        mediaIdleUnloadDescription:
          "Gibt VRAM frei, indem die Bild- und Videomodelle nach der angegebenen Anzahl von Sekunden ohne Aktivität entladen werden. Eine eigene Einstellung: Die Einstellung darüber gilt nur für das Chat-Modell. Bei 0 bleiben sie geladen; der Mindestwert ist 60.",
        mediaIdleSecondsAriaLabel:
          "Inaktivitätsdauer bis zum automatischen Entladen von Bild und Video in Sekunden",
        mediaIdlePaused:
          "Pausiert, solange „Modell im GPU-Speicher behalten“ aktiv ist.",
        idleNeedsEnable: "Aktivieren Sie zuerst „Modell je Anfrage wechseln“.",
        idleActiveViaEnv:
          "Automatisches Entladen bei Inaktivität ist über die Umgebungsvariable UNSLOTH_MODEL_IDLE_TTL aktiv.",
        loadError:
          "Einstellungen für automatischen Modellwechsel konnten nicht geladen werden.",
        saveError:
          "Einstellungen für automatischen Modellwechsel konnten nicht gespeichert werden.",
        idleError: "Geben Sie 0 ein, um das Modell geladen zu halten, oder mindestens 60 Sekunden.",
        autoDownload: "Fehlende Modelle herunterladen",
        autoDownloadDescription:
          "Lädt ein in einer API-Anfrage genanntes GGUF herunter, das noch nicht vorhanden ist. Wer einen API-Schlüssel hat, kann dann Speicherplatz und Bandbreite verbrauchen.",
        keepKv: "Chat-Kontext beim automatischen Entladen behalten",
        keepKvDescription:
          "Speichert den KV-Cache vor dem Entladen bei Inaktivität, damit der Verlauf bei fortgesetzten Chats nicht erneut eingelesen werden muss. Bis zu 10 GB auf der Festplatte.",
        apiOnly: "Nur über die API geladene Modelle entladen",
        apiOnlyDescription:
          "Das automatische Entladen bei Inaktivität lässt ein von dir in Unsloth geladenes Modell im Speicher und gibt nur solche frei, die eine API-Anfrage geladen hat.",
      },
      previewSharing: {
        sectionTitle: "Vorschau-Freigabe",
        enableLabel: "Öffentliche Vorschaulinks",
        enableDescription:
          "Erlaubt jedem mit einem signierten Link, mit einem fertigen Modell zu chatten, ohne Anmeldung. Deaktivieren Sie dies, um die öffentliche Vorschau offline zu nehmen; geteilte Links funktionieren dann nicht mehr.",
        loadError:
          "Einstellungen für die Vorschau-Freigabe konnten nicht geladen werden.",
        saveError:
          "Einstellungen für die Vorschau-Freigabe konnten nicht gespeichert werden.",
        revokeLabel: "Alle Vorschaulinks widerrufen",
        revokeDescription:
          "Rotiert das Signaturgeheimnis, sodass jeder von Ihnen geteilte Link nicht mehr funktioniert. Neu kopierte Links funktionieren weiterhin.",
        revokeAction: "Links widerrufen",
        revoking: "Wird widerrufen...",
        revokeConfirmTitle: "Alle Vorschaulinks widerrufen?",
        revokeConfirmDescription:
          "Jeder von Ihnen geteilte Vorschaulink funktioniert sofort nicht mehr. Dies kann nicht rückgängig gemacht werden.",
        revokeConfirmAction: "Alle Links widerrufen",
        revoked: "Alle Vorschaulinks widerrufen",
        revokeError: "Vorschaulinks konnten nicht widerrufen werden",
      },
      notifications: {
        sectionTitle: "Benachrichtigungen",
        showLlamaUpdates: "llama.cpp-Update-Benachrichtigungen",
        showLlamaUpdatesDescription:
          "Benachrichtigt, wenn ein neuerer llama.cpp-Build verfügbar ist, um neue Modelle auszuführen. Deaktivieren Sie dies, wenn Sie nur trainieren.",
        showLoadedModels: "Anzeige geladener Modelle",
        showLoadedModelsDescription:
          "Zeigt unten rechts eine kleine Karte mit allen derzeit im Speicher befindlichen Modellen (Chat, Sprache, Bild, Video) und einer Schaltfläche, um jedes einzeln zu entladen.",
      },
      startup: {
        sectionTitle: "Autostart",
        launchAtLogin: "Unsloth bei der Anmeldung starten",
        launchAtLoginDescription:
          "Startet Unsloth im Hintergrund, wenn Sie sich anmelden. Es bleibt in der Menüleiste bzw. im Infobereich, bis Sie es öffnen.",

        closeToTray: "In den Infobereich schließen",
        closeToTrayDescription:
          "Unsloth und seinen Server im Hintergrund weiterlaufen lassen, wenn Sie das Hauptfenster schließen.",
        closeToTraySaveError:
          "Die Einstellung zum Schließen in den Infobereich konnte nicht aktualisiert werden.",
        loadError: "Die Autostart-Einstellung konnte nicht geladen werden.",
        saveError:
          "Die Autostart-Einstellung konnte nicht aktualisiert werden.",
      },
      downloads: {
        sectionTitle: "Downloads",
        transport: "Download-Transport",
        transportDescription:
          "Wie Modell- und Datensatzdateien von Hugging Face geladen werden. HTTPS setzt an der Abbruchstelle fort; Xet ist beim ersten Download oft schneller, beginnt die Datei nach einem Abbruch aber neu.",
        transportHint:
          "HTTPS ist einfaches TLS: jedes Netzwerk, jeder Proxy und jedes VPN erlaubt es, ein abgebrochener Transfer setzt bei den bereits gespeicherten Bytes fort und der Speicherbedarf bleibt konstant. Xet lädt deduplizierte Blöcke, wodurch ein Repo mit bereits vorhandenen Daten viel schneller ankommt, benötigt aber hf_xet, mehr RAM, und ein Abbruch verwirft die laufende Datei. Auto entscheidet pro Rechner: es bewertet RAM und ob Xet hier ins Stocken gerät, und fällt auf HTTPS zurück.",
        https: "HTTPS",
        xet: "Xet",
        auto: "Auto",
        httpsHint:
          "Standard-TLS. Setzt nach einem Abbruch fort, funktioniert in jedem Netzwerk, gleichmäßiger Speicherbedarf.",
        transportDescriptionNoResume:
          "Wie Modell- und Datensatzdateien von Hugging Face geladen werden. Auf dieser Installation kann kein Transport fortsetzen, ein abgebrochener Download beginnt also neu; Xet ist beim ersten Download oft schneller.",
        httpsHintNoResume:
          "Standard-TLS. Funktioniert in jedem Netzwerk, gleichmäßiger Speicherbedarf. Diese Installation kann einen abgebrochenen Download nicht fortsetzen.",
        xetHint:
          "Deduplizierte Blockübertragung. Beim ersten Download oft schneller, beginnt die Datei nach einem Abbruch neu, braucht mehr Speicher.",
        autoHint:
          "Wählt pro Rechner und wechselt zu HTTPS, wenn Xet hier stockt oder fehlschlägt.",
        autoCurrently: "Auto verwendet auf diesem Rechner {transport}.",
        xetMissing: "Xet ist nicht verfügbar, da hf_xet nicht installiert ist.",
      },
      uploads: {
        sectionTitle: "Uploads",
        maxUploadSize: "Upload-Limit für Trainingsdatensätze",
        maxUploadSizeDescription: "Standard ist {defaultSize} MB.",
      },
      rag: {
        sectionTitle: "Dokumente & RAG",
        embeddingModel: "Embedding-Modell",
        embeddingModelDescription:
          "Hugging-Face-Modell oder lokaler Pfad zum Indexieren und Durchsuchen Ihrer Dokumente. Standard ist {defaultModel}.",
        searchPlaceholder: "Beliebiges Modell auf HF suchen",
        reindexWarning:
          "Betrifft nur neu indexierte Dokumente. Laden Sie bestehende nach einer Modelländerung erneut hoch.",
        emptyError:
          "Geben Sie eine Hugging-Face-Modell-ID oder einen lokalen Pfad ein.",
        loadError:
          "Die Embedding-Modell-Einstellung konnte nicht geladen werden.",
        saveError: "Das Embedding-Modell konnte nicht gespeichert werden.",
        saved: "Embedding-Modell gespeichert.",
        saveAnyway: "Trotzdem speichern",
        recommended: "Empfohlen",
        onDevice: "Auf dem Gerät",
        searching: "Hugging Face wird durchsucht…",
        checking: "Wird geprüft…",
        noResults: "Keine Embedding-Modelle gefunden",
        download: "Herunterladen",
        unload: "Entladen",
        unloadFailed: "Embedding-Modell konnte nicht entladen werden",
        downloadingStatus: "Wird heruntergeladen…",
        notDownloaded: "Nicht heruntergeladen",
        notDownloadedSized: "Nicht heruntergeladen · {size}",
        loaded: "Geladen",
        downloading: "{model} wird heruntergeladen",
        downloadingDescription:
          "Der Fortschritt steht im Downloads-Bereich. Nach Abschluss wird es für die Indizierung genutzt.",
        downloadFailed: "Download konnte nicht gestartet werden",
        downloadConflict: "Diesen Download im Hub fortsetzen",
        downloadBusy: "Download läuft bereits",
      },
      storage: {
        sectionTitle: "Speicher",
        modelsFolder: "Modell-Ordner",
        modelsFolderDescription:
          "Wo heruntergeladene Modelle gespeichert werden.",
        openAction: "Öffnen",
        copyAction: "Pfad kopieren",
        copied: "Pfad kopiert",
        openError: "Der Ordner konnte nicht geöffnet werden",
        copyError: "Der Pfad konnte nicht kopiert werden",
      },
      resetPreferences: {
        sectionTitle: "Gefahrenzone",
        label: "Alle lokalen Einstellungen zurücksetzen",
        description:
          "Löscht nur lokal gespeicherte Einstellungen. Chats, API-Zugriff und in der DB gespeicherte Einstellungen bleiben erhalten.",
        action: "Einstellungen zurücksetzen",
        confirmTitle: "Alle lokalen Einstellungen zurücksetzen?",
        confirmDescription:
          "Löscht nur lokal gespeicherte Einstellungen und lädt Unsloth neu. Chats, API-Zugriff und in der DB gespeicherte Einstellungen bleiben erhalten.",
        confirmAction: "Zurücksetzen und neu laden",
      },
      permissions: {
        sectionTitle: "Berechtigungen",
        bypassLabel: "Tool-Berechtigungen",
        bypassDescription:
          "Wie Unsloth Tool-Aufrufe im Chat (Terminal, Python, Web, MCP) vor der Ausführung freigibt. „Full access“ deaktiviert die Freigaben und die Code-Sandbox.",
      },
    },
    profile: {
      title: "Profil",
      description: "Wie Ihr Profil in Unsloth angezeigt wird.",
      changePicture: "Profilbild ändern",
      displayName: "Anzeigename",
      nickname: "Wie soll Unsloth Sie nennen?",
      nicknamePlaceholder: "Spitzname",
      nicknameSaved: "Bevorzugter Name gespeichert",
      avatarShape: "Form des Profilbilds",
      avatarShapeCircle: "Kreis",
      avatarShapeRounded: "Abgerundet",
      chooseSloth: "Oder wählen Sie ein Faultier",
      nameSaved: "Profilname gespeichert",
      namePersistErrorTitle: "Profilname konnte nicht gespeichert werden",
      namePersistErrorDescription:
        "Der Name wurde für diese Sitzung aktualisiert, bleibt nach dem Neuladen aber möglicherweise nicht erhalten.",
      photoUpdated: "Profilbild aktualisiert",
      photoPersistErrorTitle: "Profilbild konnte nicht gespeichert werden",
      photoPersistErrorDescription:
        "Das Foto wurde für diese Sitzung aktualisiert, bleibt nach dem Neuladen aber möglicherweise nicht erhalten.",
      photoUpdateErrorTitle: "Profilbild konnte nicht aktualisiert werden",
      imageUseError: "Dieses Bild konnte nicht verwendet werden.",
      uploadPhoto: "Foto hochladen",
      removePhoto: "Entfernen",
      pictureOptions: "Optionen für das Profilbild",
      greetingSloth: "Faultier in der Begrüßung",
      greetingSlothDescription: "Das Faultier in der Chat-Begrüßung anzeigen.",
      noPicture: "Kein Profilbild",
      noneLabel: "Keines",
      stats: {
        title: "Ihre Statistiken",
        subtitle:
          "Alles Folgende wird aus Ihrem eigenen Verlauf berechnet. Es wird nichts erfasst oder an Unsloth gesendet.",
        retry: "Erneut versuchen",
        privacyNote:
          "Die Statistiken werden aus dem lokalen Chat-, API-Nutzungs- und Trainingsverlauf Ihrer Unsloth-Installation berechnet. API-Eingaben, -Antworten und -Schlüssel werden dafür nie gespeichert. Nichts wird an Unsloth oder Dritte gesendet.",
        emptyChats:
          "Noch keine Chat- oder API-Nutzung. Starten Sie ein Gespräch oder senden Sie eine authentifizierte lokale API-Anfrage.",
        lifetimeTokens: "Tokens insgesamt",
        peakTokens: "Aktivster Tag",
        longestChat: "Längster Chat",
        currentStreak: "Aktuelle Serie",
        longestStreak: "Längste Serie",
        activityTitle: "Token-Aktivität",
        activityDescription: "Zeitraum: {weeks} · {total}",
        mode: {
          daily: "Täglich",
          weekly: "Wöchentlich",
          cumulative: "Kumuliert",
        },
        cellTooltip: "{date} · {tokens}, {messages}",
        weekTooltip: "Woche vom {date} · {tokens}",
        less: "Weniger",
        more: "Mehr",
        insightsTitle: "Aktivitätsauswertung",
        totalChats: "Chats insgesamt",
        totalMessages: "Nachrichten insgesamt",
        tokensIn: "Gesendete Tokens",
        tokensOut: "Erzeugte Tokens",
        totalTokens: "Tokens insgesamt",
        studioChatTokens: "Unsloth-Chat-Tokens",
        apiTokens: "API-Tokens",
        cachedTokens: "Zwischengespeicherte Tokens",
        cachedValue: "{tokens} ({percent} % der Eingabe)",
        avgTokensPerChat: "Durchschnittliche Tokens pro Chat",
        timeInChat: "Zeit im Chat",
        activeDays: "Aktive Tage",
        toolCalls: "Tool-Aufrufe",
        attachments: "Angehängte Dateien",
        avgSpeed: "Durchschnittliche Geschwindigkeit",
        bestSpeed: "Schnellste Antwort",
        firstToken: "Durchschnittliche Zeit bis zum ersten Token",
        tokensPerSecond: "{value} Tok/s",
        topModelsTitle: "Meistgenutzte Modelle",
        topModelsDescription: "Nach ausgetauschten Tokens sortiert",
        modelSummary: "{tokens} · {messages}",
        noModels: "Noch keine Modellnutzung vorhanden.",
        trainingTitle: "Training",
        trainingDescription: "Fine-Tuning-Läufe aus diesem Arbeitsbereich",
        trainingRuns: "Läufe",
        trainingCompleted: "Abgeschlossen",
        trainingSteps: "Schritte",
        trainingTokens: "Trainierte Tokens",
        trainingTime: "Trainingszeit",
        bestLoss: "Bester Loss",
        runSteps: "{steps}",
        runLoss: "Loss {loss}",
      },
    },
    appearance: {
      title: "Darstellung",
      description: "Wie Unsloth auf diesem Gerät aussieht.",
      theme: {
        title: "Design",
        label: "Farbschema",
        description: "Hell, dunkel oder dem System folgen.",
        system: "System",
        light: "Hell",
        dark: "Dunkel",
      },
      palette: {
        label: "Farbpalette",
        description: "Farben, die in Unsloth im hellen und dunklen Modus verwendet werden.",
        standard: "Standard",
        classic: "Klassisch",
        minimal: "Minimal",
      },
      custom: {
        reset: "Zurücksetzen",
        resetAll: "Anpassungen zurücksetzen",
        preferencesTitle: "Weitere Optionen",
        colors: {
          lightGroup: "Helles Design",
          darkGroup: "Dunkles Design",
          accent: "Akzentfarbe",
          background: "Hintergrund",
          foreground: "Vordergrund",
        },
        fontDefault: "Standard",
        fontBundledGroup: "Integriert",
        fontImportedGroup: "Importiert",
        fontDeviceGroup: "Auf diesem Gerät",
        fontFolderGroup: "Aus einem Ordner",
        fontDeviceLoading: "Schriftarten auf diesem Gerät werden gesucht…",
        fontSearch: "Schriften suchen…",
        fontNoResults: "Keine Schriften gefunden.",
        colorPicker: {
          hue: "Farbton",
          hex: "Hex-Farbe",
          eyedropper: "Eine Farbe vom Bildschirm auswählen",
        },
        uiFont: {
          label: "Schrift der Oberfläche",
        },
        headingFont: {
          label: "Schrift für Überschriften",
        },
        chatFont: {
          label: "Schrift im Chat",
        },
        codeFont: {
          label: "Schrift für Code",
        },
        importFont: {
          upload: "Hochladen",
          scanFolder: "Ordner auswählen",
          alreadyAvailable:
            "Diese Schrift ist bereits verfügbar, daher wird die vorhandene Kopie verwendet.",
          folderNoFonts: "In diesem Ordner wurden keine Schriftdateien gefunden.",
          remove: "Entfernen",
          errorInvalidType:
            "Nicht unterstützter Dateityp. Verwenden Sie .woff2, .woff, .ttf oder .otf.",
          errorTooLarge: "Die Schriftdatei ist zu groß (max. 1,5 MB).",
          errorLimit: "Sie können bis zu 3 Schriften importieren.",
          errorStorageFull:
            "Nicht genug lokaler Speicher für diese Schrift. Entfernen Sie zuerst eine importierte Schrift.",
          errorFailed: "Diese Schriftdatei konnte nicht geladen werden.",
        },
        uiFontSize: {
          label: "Schriftgröße der Oberfläche",
          description: "Passen Sie die Grundgröße der Unsloth-Oberfläche an.",
        },
        codeFontSize: {
          label: "Schriftgröße für Code",
          description: "Passen Sie die Grundgröße für Code an.",
        },
        fontSmoothing: {
          label: "Schriftglättung",
          description: "Kantenglättung für Schriften verwenden.",
        },
        contrast: {
          label: "Kontrast",
          description: "Intensität von Rahmen und sekundärem Text.",
        },
        reduceMotion: {
          label: "Bewegung reduzieren",
          description: "Animationen reduzieren oder Systemeinstellung übernehmen.",
          system: "System",
          on: "Ein",
          off: "Aus",
        },
        pointerCursors: {
          label: "Hand-Cursor verwenden",
          description:
            "Über interaktiven Elementen einen Hand-Cursor anzeigen.",
        },
      },
      language: {
        title: "Sprache",
        label: "Anzeigesprache",
        description: "Die von Unsloth verwendete Sprache.",
        autoDetect: "Automatisch erkennen",
      },
      layout: {
        title: "Layout",
        compactSidebar: "Seitenleiste standardmäßig anheften",
        compactSidebarDescription:
          "Hält die Seitenleiste ausgeklappt, statt sie zu Symbolen einzuklappen.",
      },
      sidebarNav: {
        title: "Seitenleisten-Navigation",
        description:
          "Tabs der Seitenleiste anheften und neu anordnen. Nicht angeheftete Tabs sammeln sich im Menü „Mehr“; ein einzelner nicht angehefteter Tab wird ausgeblendet, statt ein Menü mit nur einem Eintrag zu erhalten. „Neuer Chat“ bleibt fest.",
        dragToReorder: "Zum Neuanordnen ziehen",
        pinToSidebar: "{name} an die Seitenleiste anheften",
        moreHolds: "Mehr ({count})",
      },
      sidebarMenu: {
        title: "Seitenleistenmenü",
        description:
          "Elemente im Profilmenü der Seitenleiste anzeigen, ausblenden und neu anordnen. Einstellungen, Hilfe, Abmelden und Herunterfahren bleiben an ihrem Platz.",
        darkModeToggle: "Umschalter für den dunklen Modus",
        dragToReorder: "Zum Neuanordnen ziehen",
      },
    },
    resources: {
      title: "System",
      description:
        "Überwachen Sie Hardware und Speicher dieses Unsloth-Servers.",
      liveUpdates: "Live-Updates",
      floatingWindow: "Schwebendes Fenster",
      disableOverlay: "Overlay deaktivieren",
      liveMonitor: {
        title: "Live-Monitor",
        apiTitle: "API-Monitor",
        summary: "Aktive Anfragen, Fehler und Token-Nutzung",
        status: "Aktiv: {active} · kürzlich: {recent} · {model}",
        noModelLoaded: "kein Modell geladen",
        autoOpen: "Schwebenden Monitor automatisch anzeigen",
        autoOpenDescription:
          "Öffnet ein kleines Fenster, wenn API-Aktivität eingeht.",
        cpu: "CPU",
        ram: "RAM",
        disk: "Festplatte",
        vram: "VRAM",
        cpuCores: "{logical} logische / {physical} physische Kerne",
        currentLoad: "Aktuelle Auslastung",
        free: "{value} frei",
        noGpu: "Keine sichtbare GPU",
      },
      gpu: {
        title: "GPU-Geräte",
        ggufInference: "GGUF-Inferenz",
        unavailable: "nicht verfügbar",
        detecting: "Suche nach GPUs...",
        unreadable: "Die Hardware dieses Servers konnte nicht gelesen werden.",
        noGpu:
          "Keine sichtbare GPU erkannt. Oben werden nur die CPU-Ressourcen angezeigt.",
        unknownDevice: "Unbekannte GPU",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "{value} belegt",
        free: "{value} frei",
        total: "{value} gesamt",
      },
      llamaBackend: {
        title: "GGUF-Inferenz-Engine",
        label: "Compute-Backend",
        description: "Das Backend, mit dem llama.cpp GGUF-Modelle ausführt.",
        runningOn: "llama.cpp läuft derzeit mit {backend}.",
        hint: "Installiert den llama.cpp-Build für dieses Backend und behält ihn über Updates hinweg bei. Nützlich, wenn die automatische Wahl abstürzt oder dein GPU-Treiber sie nicht unterstützt. Es werden nur Backends aufgeführt, für die es einen Build für diesen Rechner gibt; das Training bleibt unberührt.",
        autoWith: "Automatisch ({backend})",
        apply: "Anwenden",
        applying: "Wird installiert ...",
        applyHint: "Lädt den neuen Build herunter und startet llama.cpp neu. Ein geladenes Modell wird entladen.",
        applyHintWithSize: "Lädt {size} herunter und startet llama.cpp neu. Ein geladenes Modell wird entladen.",
        switchedTo: "llama.cpp läuft jetzt mit {backend}.",
        switchFailed: "Das llama.cpp-Backend konnte nicht geändert werden.",
        switchInterrupted: "Der Wechsel wurde vor dem Abschluss unterbrochen.",
        envLocked: "Durch die Umgebungsvariable UNSLOTH_LLAMA_CPP_BACKEND auf {backend} festgelegt; sie überschreibt diese Einstellung.",
        customPath: {
          label: "Benutzerdefinierter llama.cpp-Ordner",
          description: "Verwende deinen eigenen llama-server-Build.",
          hint: "Wähle den llama.cpp-Ordner mit llama-server oder einen Build, in dem er unter build/bin liegt. Die benutzerdefinierte Laufzeit wird für GGUF-Chat, Einbettungen und unterstützte Sprachmodelle verwendet. Umgebungsvariablen haben weiterhin Vorrang.",
          automatic: "Automatisch (mitgeliefert)",
          bundled: "Verwendet die von Unsloth installierte llama.cpp-Laufzeit.",
          active: "Dein eigener llama-server wird beim nächsten Laden eines Modells verwendet.",
          environmentManaged: "Wird durch die Umgebungsvariable {variable} verwaltet.",
          missingBinary: "llama-server ist in diesem Ordner nicht mehr verfügbar. Wähle einen anderen Ordner oder verwende die mitgelieferte Laufzeit.",
          reloadRequired: "Lade das Modell neu, um den ausgewählten llama-server zu verwenden.",
          change: "Ändern",
          saving: "Speichern...",
          useBundled: "Mitgelieferte Version verwenden",
          chooseTitle: "llama.cpp-Ordner auswählen",
          chooseAction: "Diesen Ordner verwenden",
          saved: "llama.cpp-Ordner aktualisiert",
          saveError: "Der llama.cpp-Ordner konnte nicht aktualisiert werden",
        },
        backends: {
          auto: "Automatisch",
          cpu: "CPU",
          cuda: "CUDA",
          rocm: "ROCm",
          vulkan: "Vulkan",
          metal: "Metal",
        },
        unsupported: {
          notInstalled: "Es wurde keine von Unsloth verwaltete llama.cpp-Installation gefunden, daher gibt es kein Backend zum Wechseln.",
          localLink: "llama.cpp ist ein selbst verknüpftes lokales Verzeichnis, das Unsloth nicht ersetzt.",
          sourceBuild: "Dieses llama.cpp wurde aus dem Quellcode gebaut; sein Backend lässt sich hier nicht wechseln.",
          customPath: "Ein benutzerdefinierter llama.cpp-Ordner ist ausgewählt. Dessen Build bestimmt das Compute-Backend.",
          unresolved: "Die verfügbaren Backends konnten nicht geprüft werden. Prüfe deine Verbindung und versuche es erneut.",
        },
        // Wird nicht angezeigt: zusätzliche Begriffe für die Einstellungssuche.
        llamaBackendKeywords:
          "llama.cpp backend gguf inferenz cuda rocm hip vulkan metal cpu gpu beschleuniger prebuilt wechseln engine",
      },
      modelMemory: {
        title: "Modellspeicher",
        keepResident: "Modell im GPU-Speicher behalten",
        keepResidentDescription: "Zwischen Prompts im VRAM bleiben.",
        keepResidentHint: "Die Gewichte werden nicht an den System-RAM zurückgegeben, solange das Modell geladen bleibt. Deaktiviert das automatische Entladen im Leerlauf und übergibt zusätzlich --mlock, wenn die Gewichte tatsächlich im Host-RAM liegen (Unified Memory oder teilweises GPU-Offload), damit das Betriebssystem sie nicht auslagert und beim nächsten Prompt neu hochlädt.",
        noRamReserve: "Keinen System-RAM für das Modell reservieren",
        noRamReserveDescription: "Keine vollständige Kopie im RAM behalten.",
        noRamReserveHint: "Die Gewichte werden in den VRAM gestreamt, statt eine vollständige Kopie im RAM zu halten. Behält das speicherabgebildete Laden von llama.cpp bei und entfernt --no-mmap und --mlock.",
        mlockVetoed: "--mlock bleibt aus: das Fixieren des Modells würde RAM für das gesamte Modell reservieren. Das automatische Entladen im Leerlauf bleibt deaktiviert.",
        memlockCapped: "Dieses System begrenzt gesperrten Speicher auf {limit}. Ein größeres Modell wird nicht vollständig fixiert; erhöhen Sie das Limit mit ulimit -l.",
        reloadRequired: "Modell neu laden, um die neuen Speicheroptionen anzuwenden.",
        loadError: "Modellspeicher-Einstellungen konnten nicht geladen werden",
        saveError: "Modellspeicher-Einstellungen konnten nicht gespeichert werden",
        // Not rendered: extra terms the settings search matches these rows on.
        modelMemoryKeywords:
          "mlock memlock ulimit vram gpu speicher arbeitsspeicher ram resident anheften fixieren sperren geladen halten entladen leerlauf mmap no-mmap load-mode auslagern",
      },
      storage: {
        title: "Speicher",
        systemDisk: "Systemfestplatte",
        diskUsage: "{used} belegt / {total}",
        diskFree: "{free} frei",
        modelsFolder: "Modell-Ordner",
        modelsFolderDescription: "Wo heruntergeladene Modelle gespeichert werden.",
        modelsFolderHint: "Wo heruntergeladene Modelle gespeichert werden. Ändern Sie den Pfad, um Modelle von Ihrem Systemlaufwerk fernzuhalten. Gilt nur für neue Downloads. Bereits vorhandene Modelle bleiben, wo sie sind.",
        modelsFolderKeywords:
          "Modelle Ordner Verzeichnis Pfad Speicherort Download Downloads Cache Speicher Festplatte Laufwerk verschieben ändern hugging face",
        futureDownloads: "Nur neue Downloads",
        environmentManaged:
          "Wird über die Umgebungsvariable {variable} verwaltet.",
        locationFree: "{free} frei",
        openAction: "Öffnen",
        copyAction: "Pfad kopieren",
        changeAction: "Ändern",
        resetAction: "Standard verwenden",
        chooseTitle: "Speicherort für Modell-Downloads wählen",
        chooseAction: "Für künftige Downloads verwenden",
        cacheSaved: "Speicherort für Modell-Downloads aktualisiert",
        cacheSaveError:
          "Der Speicherort für Modell-Downloads konnte nicht geändert werden",
        cachePickerError: "Die Ordnerauswahl konnte nicht geöffnet werden",
        copied: "Pfad kopiert",
        openError: "Der Ordner konnte nicht geöffnet werden",
        copyError: "Der Pfad konnte nicht kopiert werden",
      },
      environment: {
        title: "Umgebung",
        backend: "Backend",
        python: "Python",
        torch: "Torch",
        transformers: "Transformers",
        uptime: "Betriebszeit",
        processMemory: "Prozessspeicher",
        notInstalled: "Nicht installiert",
        unknown: "Unbekannt",
        vramWithShared: "{vram} VRAM + {shared} gemeinsam genutzter Speicher",
      },
    },
    agents: {
      title: "Agenten",
      description:
        "Verbinden Sie Coding-Agenten wie Claude Code und Codex über unsloth start mit einem lokalen Modell.",
      intro:
        "verbindet Claude Code, Codex, Hermes, OpenClaw, OpenCode und weitere Agenten mit einem lokal von Unsloth bereitgestellten Modell, vollständig offline. Es startet einen OpenAI-kompatiblen Server und verändert nie die Konfigurationsdateien Ihres Agenten.",
      readDocs: "Dokumentation lesen",
      copy: "Kopieren",
      copied: "Kopiert",
      commandBuilder: "Befehlsgenerator",
      agent: "Coding-Agent",
      model: "Modell",
      searchModels: "GGUF-Modelle suchen...",
      noModels: "Keine passenden GGUF-Modelle.",
      showingModels:
        "{shown} von {total} Treffern werden angezeigt. Tippen Sie weiter, um die Liste einzugrenzen.",
      quantization: "Quantisierung",
      loadingQuantizations: "Quantisierungen werden geladen...",
      noQuantizations: "Keine separate Quantisierung",
      recommended: "Empfohlen",
      downloaded: "Heruntergeladen",
      quantizationLoadError:
        "Es konnten nicht alle Quantisierungen geladen werden. Der Befehl verwendet den verfügbaren Modellwert.",
      generatedCommand: "Generierter Befehl",
      docs: "Dokumentation",
      agentDocs: "Einrichtungsdokumentation zu {agent} öffnen",
      copyGeneratedCommand: "Generierten Befehl kopieren",
      // English is the baseline until these are translated. The three-part
      // sentence below is assembled in a fixed order around an inline link, so
      // it needs restructuring before it can be translated well.
      automaticSettingsNote:
        "Unsloth automatically applies the model’s recommended settings if you have not set any flags.",
      configurationNote:
        "You can also adjust any configuration. See further below or",
      configurationDocs: "docs",
      configurationFlagsSuffix: "for flags.",
      modelNote:
        "Codex benötigt ein GGUF-Modell, das von llama-server bereitgestellt wird. Andere Agenten können auch Transformer-basierte Modelle verwenden; lassen Sie --model weg, um das bereits in Unsloth geladene Modell zu nutzen.",
      subagent: {
        title: "Ein lokales Modell als Subagent verwenden",
        description:
          "Belassen Sie {agent} bei seinem aktuellen Modell und delegieren Sie ausgewählte Aufgaben an dieses lokale Unsloth-Modell.",
        setupCommand: "Einrichtungsbefehl",
        copySetupCommand: "Einrichtungsbefehl für den Subagenten kopieren",
        usagePrompt: "Geben Sie dann in {agent} Folgendes ein:",
        copyUsagePrompt: "Nutzungs-Prompt für den Subagenten kopieren",
        defaultPrompt:
          "Starte einen lokalen Agenten, um diese Funktion zu implementieren.",
        opencodePrompt: "@unsloth finde die Ursache dieses Testfehlers",
      },
      quickstart: {
        title: "Befehl zusammenstellen",
        description:
          "Starten Sie einen Agenten mit dem aktuell in Unsloth geladenen Modell. Laden Sie zuerst ein Modell und ersetzen Sie dann claude durch einen der unten aufgeführten Agenten.",
        noneDetected:
          "In Ihrem PATH wurden keine unterstützten Agent-CLIs gefunden.",
        installed: "Installiert",
      },
      supportedAgents: {
        title: "Unterstützte Agenten",
        description: "Jeder Agent startet mit seinem eigenen Befehl:",
        requiresGguf: "Benötigt ein GGUF-Modell",
      },
      models: {
        title: "Ein Modell auswählen",
        description:
          "Mit --model wählen Sie Modell und Quantisierung, mit --context-length das Kontextfenster. Verwenden Sie ein Quantisierungssuffix oder die explizite Option --gguf-variant.",
        suffixLabel: "Mit Quantisierungssuffix",
        variantLabel: "Mit expliziter Variantenoption",
      },
      options: {
        title: "Gängige Optionen",
        description:
          "Unsloth-Optionen werden zuerst ausgewertet; alles Unbekannte wird unverändert an den Agenten weitergereicht.",
        model:
          "Wählt ein Modell aus. Ohne --model verwendet unsloth start das aktuell in Unsloth geladene Modell und bricht mit einem Fehler ab, wenn keines geladen ist.",
        contextLength:
          "Legt die gewünschte Kontextlänge fest (Alias: --max-seq-length).",
        ggufVariant: "Wählt die GGUF-Quantisierungsvariante.",
        loadIn4bit:
          "Schaltet das 4-Bit-Laden für Hugging-Face-Modelle ein oder aus.",
        tensorParallel:
          "Schaltet Tensor-Parallelität über mehrere GPUs ein oder aus.",
        serve: "Aktiviert oder deaktiviert den automatischen lokalen Server.",
        launch:
          "Startet den Agenten oder gibt nur Befehl und Umgebung aus.",
        persist:
          "Behält die von Unsloth verwalteten Agent-Daten über Läufe hinweg bei.",
        asSubagent:
          "Belässt den übergeordneten Agenten bei seinem aktuellen Modell und registriert Unsloth als lokalen Subagenten (Claude Code, Codex und OpenCode).",
        apiKey:
          "Übergibt Ihren Unsloth-API-Schlüssel (alternativ UNSLOTH_API_KEY setzen).",
        reasoning:
          "Reasoning im Chat verwenden: on, off oder auto. Auto folgt der Chat-Vorlage des Modells, was meist on bedeutet.",
        reasoningEffort:
          "Reasoning-Aufwand, der an die Chat-Vorlage des Modells übergeben wird, z. B. medium. Die Stufen hängen vom Modell ab, verwenden Sie also eine, die es kennt. Ohne Angabe gilt die Stufe der Vorlage.",
        yolo:
          "Überspringt Bestätigungsabfragen. Nur in vertrauenswürdigen Umgebungen verwenden.",
      },
      remote: {
        title: "Mit einem entfernten Unsloth Studio verbinden",
        description:
          "Richten Sie unsloth start auf ein anderswo laufendes Unsloth Studio aus, indem Sie diese Variablen vor dem Aufruf setzen (oder --api-key direkt übergeben):",
      },
      passthrough: {
        title: "Argumente an den Agenten übergeben",
        description:
          "Argumente nach den Unsloth-Optionen werden an den Agenten selbst weitergereicht, sodass native Befehle wie resume weiterhin funktionieren:",
      },
      dryRun: {
        title: "Vorschau ohne Start",
        description:
          "Fügen Sie --no-launch hinzu, um Umgebung und Befehl auszugeben, statt den Agenten zu starten. Ist --model gesetzt, kann das Modell trotzdem aufgelöst und geladen werden.",
      },
    },
    chat: {
      projectsSection: "Projektbereich anzeigen",
      projectsSectionDescription:
        "Gruppiert Projekt-Chats unter einer Überschrift für Projekte. Deaktiviere dies, um sie stattdessen unter den zuletzt verwendeten Chats aufzulisten.",
      title: "Chat",
      description: "Passen Sie an, wie sich der Chat auf diesem Gerät verhält.",
      modelSelection: {
        title: "Einstellungen für die Modellauswahl",
        expandQuantizations: "Quantisierungen ausklappen",
        expandQuantizationsDescription:
          "Ein: Bei GGUF-Modellen unter „On Device“ werden die Quantisierungen sofort angezeigt. Aus: Klicken Sie auf ein Modell, um seine Quantisierungen anzuzeigen.",
        showAllQuantizations: "Alle Quantisierungen anzeigen",
        showAllQuantizationsDescription:
          "Ein: Alle Quantisierungen unter „On Device“ auflisten, auch nicht heruntergeladene. Aus: Nur heruntergeladene Quantisierungen anzeigen.",
        showMemoryBar: "VRAM-Auslastungsbalken anzeigen",
        showMemoryBarDescription:
          "Zeigt unter jeder Zeile eines heruntergeladenen Modells den geschätzten VRAM-Bedarf: Gewichte, KV-Cache bei der Kontextlänge, mit der das Modell geladen wird, und eine eventuelle Reserve für spekulatives Decoding.",
      },
      menu: {
        title: "Chatmenü",
        description:
          "Elemente im seitlichen Plus-Menü des Chats anheften. Die übrigen werden unter „Mehr“ angezeigt.",
        chatWithFiles: "Chat mit Dateien (RAG)",
        mcp: "MCP",
        savedPrompts: "Gespeicherte Prompts",
        compareChat: "Chats vergleichen",
        exportChat: "Chat exportieren",
      },
      pastedTextThreshold: "Lange Einfügungen verdichten",
      pastedTextThresholdDescription: "Eingefügter Text, der länger ist, wird zu einem .txt-Anhang, statt das Nachrichtenfeld zu füllen. Mit {shortcut} wird trotzdem in das Nachrichtenfeld eingefügt.",
      pastedTextThresholdOff: "Aus",
      showResponseModel: "Antwortmodell anzeigen",
      showResponseModelDescription:
        "Modellmetadaten in Antworten des Assistenten anzeigen.",
      modelDisclaimer: "Modell-Hinweis anzeigen",
      modelDisclaimerDescription:
        "Zeigt „LLMs können Fehler machen“ unter dem Chatfeld an.",
      projectAttachments: "Dateien projektweit teilen",
      projectAttachmentsDescription:
        "Standard für Dateien, die in einem Chat innerhalb eines Projekts angehängt werden: für das gesamte Projekt indizieren, damit jeder Chat darin sie nutzen kann. Jeder Chat kann dies im Anhangsmenü überschreiben.",
      rememberParamsPerModel: "Einstellungen pro Modell merken",
      rememberParamsPerModelDescription:
        "Beim Modellwechsel werden Temperatur, Prompt und die weiteren Einstellungen wiederhergestellt, die Sie zuletzt mit diesem Modell verwendet haben. Aus: ein Satz Einstellungen für alle Modelle.",
      autoCompact: "Lange Chats automatisch komprimieren",
      autoCompactDescription:
        "Wenn ein lokaler GGUF-Chat die festgelegte Kontextlänge erreicht, werden ältere Gesprächsrunden verworfen, statt einen Fehler zurückzugeben. Dies richtet sich nicht nach freiem VRAM.",
      compactionStyle: "Wenn der Kontext voll ist",
      compactionStyleDescription:
        "Die Servervorgabe behält UNSLOTH_CONTEXT_POLICY bei. Gespräch zurücksetzen behält die letzte Runde und dauerhafte Anweisungen. Ein gleitendes Fenster verwirft die ältesten Runden und kann mehr aktuellen Verlauf behalten.",
      compactionStyleInherit: "Servervorgabe verwenden",
      compactionStyleCheckpoint: "Gespräch zurücksetzen",
      compactionStyleRollingDefault:
        "Älteste Runden verwerfen (~25 % zusätzlicher Platz)",
      compactionStyleRolling10:
        "Älteste Runden verwerfen (~10 % zusätzlicher Platz)",
      compactionStyleRolling5:
        "Älteste Runden verwerfen (~5 % zusätzlicher Platz)",
      compactionStyleRollingNone:
        "Älteste Runden verwerfen (keine zusätzliche Kürzung)",
      autoCompactKeywords:
        "Komprimierung automatisch Kontext Fenster kürzen gleitend Prüfpunkt Reserve compaction rolling checkpoint headroom",
      thinking: {
        collapseByDefault: "Denken standardmäßig einklappen",
        collapseByDefaultDescription:
          "Das Denken bleibt eingeklappt, während das Modell denkt, statt automatisch aufzuklappen. Zum Lesen einen Block ausklappen.",
      },
      tools: {
        collapseByDefault: "Tool-Aktivität standardmäßig einklappen",
        collapseByDefaultDescription:
          "Tool-Eingaben und -Ausgaben bleiben während der Ausführung eingeklappt. Zum Prüfen eine Tool-Zeile ausklappen.",
      },
      webSearch: {
        title: "Websuche",
        images: "Bilder aus der Websuche anzeigen",
        imagesDescription:
          "Lässt die Websuche Bilder liefern und holt eines für jeden Punkt, den eine Antwort auflistet. Vorschaubilder lädt und verkleinert Unsloth, der Browser kontaktiert keine Bildhosts.",
      },
      artifacts: {
        title: "Canvas",
        collapseHtmlBlocks: "HTML-Blöcke einklappen",
        collapseHtmlBlocksDescription:
          "Der Canvas-Modus klappt vollständiges HTML automatisch ein. Aktivieren Sie diese Option, um auch HTML-Dokumente in Codeblöcken einzuklappen, wenn Canvas deaktiviert ist.",
        allowNetworkAccess: "Netzwerkzugriff für Canvas erlauben",
        allowNetworkAccessDescription:
          "Erlaubt Canvas-Vorschauen, Skripte, Stile, Schriftarten, Medien und Netzwerkressourcen von CDNs zu laden. Für vollständig offline nutzbare Vorschauen deaktiviert lassen.",
        blockedBanner: "{count} externe Ressource von {hosts} blockiert.",
        blockedBannerPlural: "{count} externe Ressourcen von {hosts} blockiert.",
        blockedBannerAction: "Für dieses Canvas erlauben",
      },
      data: "Daten",
      exportHistory: "Chatverlauf exportieren",
      exportHistoryDescription:
        "Alle Chats und Nachrichten als JSON herunterladen.",
      exportAction: "Exportieren",
      exportingAction: "Wird exportiert...",
      exportConversations: "„Zuletzt verwendet“ und Projekte exportieren",
      exportConversationsDescription:
        "Laden Sie Chats aus „Zuletzt verwendet“ oder zusätzlich auch Projekt-Chats als Training JSONL, CSV oder ShareGPT JSONL herunter, kombiniert oder einzeln pro Chat. Message JSONL ist nur einzeln pro Chat verfügbar.",
      exportConversationsAction: "Exportieren",
      exportScopeRecents: "Zuletzt verwendet",
      exportScopeAll: "Zuletzt verwendet + Projekte",
      exportCombinedSuffix: "(kombiniert)",
      exportPerChatSuffix: "(pro Chat)",
      importChats: "Chats importieren",
      importChatsDescription:
        "Einen Open-WebUI-, JSONL-, NDJSON- oder CSV-Export in Zuletzt importieren.",
      importChatsAction: "Importieren",
      importNoConversations: "Keine Konversationen in der Datei gefunden.",
      importedOneChat: "1 Konversation in „Zuletzt verwendet“ importiert.",
      importedChatCount:
        "{count} Konversationen in „Zuletzt verwendet“ importiert.",
      importingChats: "Chats werden importiert: bisher {count} ({percent}%)...",
      importedChatCountPartial: "{count} Unterhaltungen in Zuletzt importiert; {failed} konnten nicht gespeichert werden.",
      importFailed: "Import fehlgeschlagen.",
      clearHistory: "Chatverlauf löschen",
      clearHistoryDescription: "Chatverlauf von diesem Gerät löschen.",
      clearAction: "Löschen",
      clearAllChats: "Alle Chats löschen",
      clearAllChatsDescription:
        "Löscht dauerhaft jeden Chat auf diesem Gerät.",
      noChatsToClear: "Keine Chats zum Löschen.",
      clearOneChatDescription:
        "Löscht dauerhaft den einzigen Chat auf diesem Gerät.",
      clearChatCountDescription:
        "Löscht dauerhaft alle {count} Chats auf diesem Gerät.",
      clearChatsAction: "Chats löschen",
      clearOneChatTitle: "1 Chat löschen?",
      clearChatsTitle: "{count} Chats löschen?",
      clearChatsConfirmDescription:
        "Löscht dauerhaft jeden Chat auf diesem Gerät. Dies kann nicht rückgängig gemacht werden.",
      clearingAction: "Wird gelöscht...",
      clearOneChatAction: "1 Chat löschen",
      clearChatCountAction: "{count} Chats löschen",
      clearedAllChats: "Alle Chats gelöscht",
      clearedOneChat: "1 Chat gelöscht",
      clearedChatCount: "{count} Chats gelöscht",
      someChatsCouldNotBeCleared: "Einige Chats konnten nicht gelöscht werden",
      chatsClearedRemainOne:
        "{clearedCount} Chats gelöscht; 1 Chat verbleibt. Bitte erneut versuchen.",
      chatsClearedRemain:
        "{clearedCount} Chats gelöscht; {remainingCount} Chats verbleiben. Bitte erneut versuchen.",
      oneChatClearedRemain:
        "1 Chat gelöscht; {remainingCount} Chats verbleiben. Bitte erneut versuchen.",
      oneChatClearedRemainOne:
        "1 Chat gelöscht; 1 Chat verbleibt. Bitte erneut versuchen.",
      storageClearFailedOne:
        "Das Löschen aus dem Speicher ist fehlgeschlagen; möglicherweise verbleibt 1 Chat. Bitte erneut versuchen.",
      storageClearFailed:
        "Das Löschen aus dem Speicher ist fehlgeschlagen; möglicherweise verbleiben {count} Chats. Bitte erneut versuchen.",
      failedToClearChats: "Chats konnten nicht gelöscht werden",
    },
    data: {
      title: "Daten",
      backToData: "Zurück zu Daten",
      exportFailed: "Chats konnten nicht exportiert werden",
      description:
        "Verwalten Sie Chatverlauf und hochgeladene Dateien, die auf diesem Gerät gespeichert sind.",
      archivedChats: "Archivierte Chats",
      archivedChatsDescription:
        "Zeigen Sie die von Ihnen archivierten Chats an und verwalten Sie sie.",
      archivedImages: "Archivierte Bilder",
      archivedImagesDescription: "Bilder anzeigen und verwalten, die du archiviert hast.",
      archivedVideos: "Archivierte Videos",
      archivedVideosDescription: "Videos anzeigen und verwalten, die du archiviert hast.",
      manageAction: "Verwalten",
      manageChats: "Chats verwalten",
      manageChatsDescription:
        "Wählen Sie mehrere Chats aus, um sie zu verschieben, anzupinnen, zu archivieren, zu exportieren oder zu löschen.",
      exportArchivedChats: "Exportieren",
      exportingArchivedChats: "Wird exportiert...",
      exportedOneArchivedChat: "1 archivierter Chat exportiert",
      exportedArchivedChatCount: "{count} archivierte Chats exportiert",
      noArchivedChatsToExport: "Keine archivierten Chats zum Exportieren.",
      failedToExportArchivedChats:
        "Archivierte Chats konnten nicht exportiert werden",
      archiveAllChats: "Alle Chats archivieren",
      archiveAllChatsDescription:
        "Verschiebt alle Chats aus „Zuletzt verwendet“ und „Projekte“ ins Archiv.",
      noChatsToArchive: "Keine Chats zum Archivieren.",
      archiveAllAction: "Alle archivieren",
      archivingAction: "Wird archiviert...",
      archiveAllChatsTitle: "Alle Chats archivieren?",
      archiveAllChatsConfirmDescription:
        "Verschiebt alle Chats auf diesem Gerät ins Archiv. Archivierte Chats bleiben verfügbar und können jederzeit wieder aus dem Archiv geholt werden.",
      archivedAllChats: "Alle Chats archiviert",
      archivedOneChat: "1 Chat archiviert",
      archivedChatCount: "{count} Chats archiviert",
      failedToArchiveChats: "Chats konnten nicht archiviert werden",
      confirmBeforeDeleting: "Vor dem Löschen bestätigen",
      confirmBeforeDeletingDescription:
        "Fragt vor dem Löschen eines Chats nach einer Bestätigung. Deaktivieren, um sofort zu löschen.",
      alwaysDeleteFiles: "Dateien immer löschen",
      alwaysDeleteFilesDescription:
        "Beim Löschen eines Chats wird auch dessen eigener Sandbox-Ordner von der Festplatte entfernt. Dateien, die er in einem Projekt erstellt hat, bleiben im Arbeitsbereich dieses Projekts.",
      filesSection: "Dateien",
      uploadedFiles: "Hochgeladene Dateien",
      uploadedFilesDescription:
        "Zeigen Sie Dateien an, die in Chats, Projekte und Wissensdatenbanken hochgeladen wurden, und verwalten Sie sie.",
      fineTuneExport: "Chats als Trainingsdaten verwenden",
      fineTuneExportDescription:
        "Erstellen Sie aus Ihren Chats einen JSONL-Datensatz für das Fine-Tuning. Laden Sie ihn in „Trainieren“, verfeinern Sie ihn in „Rezepte“ oder exportieren Sie ihn.",
      fineTuneExportAction: "JSONL exportieren",
      fineTuneRunAction: "Ausführen",
      fineTuneExportingAction: "Wird exportiert...",
      fineTuneOpenRecipesAction: "In „Rezepte“ öffnen",
      fineTuneOpeningRecipesAction: "Wird geöffnet...",
      fineTuneTrainAction: "In den Tab „Trainieren“ laden",
      fineTuneTrainingAction: "Wird geladen...",
      fineTuneExportFailed: "Trainingsdaten konnten nicht exportiert werden",
      fineTuneRecipeFailed: "Chats konnten nicht in „Rezepte“ geöffnet werden",
      fineTuneTrainFailed:
        "Datensatz konnte nicht in den Tab „Trainieren“ geladen werden",
    },
    connections: {
      title: "Verbindungen",
      description: "Verwalten Sie Anbieter und externe Verbindungen.",
    },
    remoteLan: {
      title: "Remote & LAN",
      description:
        "Erreiche dieses Unsloth von deinen anderen Geräten über dein lokales Netzwerk oder eine temporäre öffentliche URL.",
    },
    apiKeys: {
      title: "API",
      description: "Zugriff auf Unsloth über die OpenAI-kompatible API.",
      readDocs: "API-Dokumentation lesen",
      noAccess: "Noch kein API-Zugriff.",
      accessTokens: "Zugriffstoken",
      loadError: "API-Zugriff konnte nicht geladen werden.",
      createError: "Zugriffstoken konnte nicht erstellt werden.",
      revokeError: "Zugriffstoken konnte nicht widerrufen werden.",
      never: "Nie",
      tokenNamePlaceholder: "Token-Name (z. B. production)",
      newAccessTokenName: "Name des neuen Zugriffstokens",
      createToken: "Token erstellen",
      creating: "Wird erstellt...",
      newTokenCreated: "Neues Zugriffstoken erstellt",
      accessTokenCopied: "Zugriffstoken kopiert",
      copyAccessToken: "Zugriffstoken kopieren",
      copyNow: "Jetzt kopieren - es wird nicht erneut angezeigt.",
      usageExamples: "Nutzungsbeispiele",
      usageTools: "Tools",
      exampleCurlTools: "curl + Tools",
      examplePythonTools: "Python + Tools",
      exampleJavaScriptTools: "JavaScript + Tools",
      exampleCurlAdvanced: "curl + erweitert",
      examplePythonAdvanced: "Python + erweitert",
      exampleJavaScriptAdvanced: "JavaScript + erweitert",
      osUnix: "Linux / macOS / WSL",
      osWindows: "Windows",
      secureHttps: "Sicheres HTTPS",
      secureHttpsHint:
        "Der 0.0.0.0-Port ist weiterhin global erreichbar. Für vollständige Sicherheit starten Sie Unsloth mit --secure, um nur diesen HTTPS-Link freizugeben.",
      copyTunnelUrl: "Tunnel-URL kopieren",
      copySnippet: "Snippet kopieren",
      copy: "Kopieren",
      copied: "Kopiert",
      setupDocs: "Einrichtungsdokumentation:",
      codingAgents: "Coding-Agents",
      codingAgentsHint:
        "Starten Sie einen Coding-Agenten für diesen Server. Er verwendet das geladene Modell; ein lokaler Server erstellt automatisch einen API-Schlüssel, ein Remote-Server fügt ihn dem Befehl hinzu.",
      codingAgentsSwap:
        "Ersetzen Sie claude durch codex, openclaw, opencode oder hermes.",
      codingAgentDetected: "Auf diesem Gerät installiert",
      codingAgentsDetectedHint: "Auf diesem Gerät erkannt: {agents}.",
      relativeNever: "nie",
      relativeJustNow: "gerade eben",
      expired: "abgelaufen",
      today: "heute",
      created: "Erstellt {value}",
      used: "Verwendet {value}",
      expires: "Läuft {value} ab",
      actionsFor: "Aktionen für {name}",
      copyPrefix: "Präfix kopieren",
      revokeToken: "Token widerrufen",
      revokeTitle: "Zugriffstoken „{name}“ widerrufen?",
      revokeDescription:
        "Apps, die dieses Token verwenden, verlieren sofort den Zugriff. Dies kann nicht rückgängig gemacht werden.",
      revokeAction: "„{name}“ widerrufen",
      revoking: "Wird widerrufen...",
      usageNoModel:
        "Laden Sie ein Modell oder laden Sie eines herunter, um ausführbare Beispiele zu sehen. Dieser Server kennt noch kein Modell, das in den Beispielen verwendet werden könnte.",
    },
    about: {
      title: "Info",
      description: "Dokumentation, Versionshinweise, Feedback und Build-Infos.",
      studioVersion: "Unsloth-Version",
      packageVersion: "Paketversion",
      desktopAppVersion: "Version der Desktop-App",
      desktopAppVersionUnavailable: "Nicht verfügbar",
      llamaCppVersion: "llama.cpp-Version",
      hardware: "Hardware",
      gpu: "GPU",
      cuda: "CUDA",
      rocm: "ROCm",
      xpu: "XPU",
      updates: "Update",
      help: "Hilfe",
      documentation: "Dokumentation",
      releaseNotes: "Versionshinweise",
      whatsNew: "Neuigkeiten",
      feedback: "Feedback",
      reportIssue: "Problem melden",
      license: {
        sectionTitle: "Lizenz",
        studioLabel: "Unsloth",
        studioLicense: "AGPL-3.0",
        studioDescription: "Open Source unter der GNU AGPL v3.0.",
        libraryLabel: "Unsloth Core",
        libraryLicense: "Apache-2.0",
        libraryDescription: "Lizenziert unter Apache 2.0.",
      },
      dangerZone: "Gefahrenzone",
      shutDownStudio: "Unsloth herunterfahren",
      shutDownStudioDescription:
        "Stoppt den Unsloth-Server und beendet Ihre Sitzung.",
      shutDown: "Herunterfahren",
      update: {
        title: "Unsloth aktualisieren",
        commandText: "{label} als Text",
        copied: "Kopiert",
        copyCommand: "Befehl kopieren",
        commandCopied: "{label} kopiert",
        copyNamedCommand: "{label} kopieren",
        checkingInstall: "Es wird geprüft, wie Unsloth installiert wurde...",
        installIntro: "So installieren oder aktualisieren Sie Unsloth:",
        localUpdateHeading: "Lokales Update",
        installCommandUnix: "macOS/Linux-Installationsbefehl",
        installCommandWindows: "Windows-Installationsbefehl",
        localInstallDetected:
          "Lokale Installation erkannt. Aktualisieren Sie aus Ihrem ursprünglichen Checkout, um zu vermeiden, dass er durch PyPI ersetzt wird.",
        pullThenUpdate:
          "Holen Sie die neuesten Änderungen und führen Sie dann den lokalen Installer aus:",
        gitPullCommand: "git pull-Befehl",
        localInstallerCommand: "Lokaler Installer-Befehl",
        sourceInstallDetected:
          "Quell- oder VCS-Paketinstallation erkannt. Installieren Sie erneut vom ursprünglichen lokalen Pfad oder der Git-URL.",
        repoCheckoutFallback:
          "Wenn Sie den Repository-Checkout noch haben, führen Sie den lokalen Installer daraus aus:",
        restartAfterUpdate: "Starten Sie Unsloth nach dem Update neu.",
        desktopManaged:
          "Die Desktop-App sucht automatisch nach neuen App-Versionen. Sie können hier auch jederzeit nach Updates suchen oder ein Update durchführen.",
        desktopReady: "Updates für die Desktop-App",
        desktopReadyDescription:
          "Prüfen Sie, ob eine neuere Version der Desktop-App verfügbar ist.",
        desktopChecking: "Es wird nach Updates gesucht",
        desktopCheckingDescription:
          "Dies dauert in der Regel einige Sekunden.",
        desktopAvailable: "Version {version} der Desktop-App ist verfügbar",
        desktopAvailableDescription:
          "Aktualisieren Sie jetzt. Die Desktop-App wird nach Abschluss des Updates neu gestartet.",
        desktopExternalServer:
          "Führen Sie `unsloth studio update` in dem Terminal aus, über das Sie Ihren Server gestartet haben.",
        desktopManualInstall:
          "Öffnen Sie die Release-Seite, um das neueste Linux-Paket zu installieren.",
        desktopCheckFailed: "Die Suche nach Updates ist fehlgeschlagen",
        desktopCheckFailedDescription:
          "Überprüfen Sie Ihre Verbindung und versuchen Sie es erneut.",
        desktopCurrent: "Die Desktop-App ist auf dem neuesten Stand",
        desktopCurrentDescription:
          "Unsloth sucht weiterhin automatisch nach Updates.",
        checkForUpdates: "Nach Updates suchen",
        checkAgain: "Erneut suchen",
        retryCheck: "Erneut versuchen",
        checking: "Wird geprüft...",
        updateNow: "Jetzt aktualisieren",
        openReleasePage: "Release-Seite öffnen",
        unknownInstall:
          "Es konnte nicht erkannt werden, wie Unsloth installiert wurde. Verwenden Sie für Installer- oder PyPI-Installationen die obigen Befehle.",
        localCheckout:
          "Führen Sie bei lokalen Checkout-Installationen den lokalen Installer aus diesem Checkout aus:",
        docs: "Installationsdokumentation:",
        docsInstall: "Installation",
        docsUpdating: "Aktualisierung",
        docsMac: "Mac",
        docsWindows: "Windows",
      },
    },
  },
  studio: {
    imageTraining: "Bildtraining",
    goToImageTraining: "Zum Bildtraining",
    routeTitle: "Trainieren",
    wizard: {
      modelTitle: "Modell",
      modelDescription: "Modell und Trainingsmethode auswählen",
      datasetTitle: "Datensatz",
      datasetDescription: "Trainingsdaten auswählen oder hochladen",
      paramsTitle: "Parameter",
      paramsDescription: "Trainingsparameter konfigurieren",
      configTitle: "Konfiguration",
      configDescription: "Konfigurationen speichern und laden",
      modelLabel: "Modell",
      methodLabel: "Methode",
      datasetLabel: "Datensatz",
      modelTooltip: "Das Basismodell, das du feinabstimmen möchtest.",
      methodTooltip: "Wie das Modell trainiert wird. LoRA und QLoRA aktualisieren kleine Adapter statt aller Gewichte.",
      datasetTooltip: "Die Trainingsdaten für die Feinabstimmung des Modells.",
      hfTokenDescription:
        "Erforderlich für zugriffsbeschränkte oder private Modelle und Datensätze.",
      uploadLocalLabel: "Oder eine lokale Datei hochladen",
      sourceBrowse: "Durchsuchen",
      releaseToUpload: "Zum Hochladen loslassen",
      loadYaml: "YAML laden",
      saveYaml: "YAML speichern",
      resetDefaults: "Auf Standardwerte zurücksetzen",
      cachedModelGoneTitle:
        "Zwischengespeichertes Modell nicht mehr verfügbar",
      cachedModelGoneDescription:
        "Die Modelldateien befinden sich nicht mehr auf diesem Gerät. Beim Training werden sie erneut heruntergeladen.",
      cachedDatasetGoneTitle:
        "Zwischengespeicherter Datensatz nicht mehr verfügbar",
      cachedDatasetGoneDescription:
        "Die Datensatzdateien befinden sich nicht mehr auf diesem Gerät. Beim Training werden sie erneut heruntergeladen.",
    },
    preview: {
      title: "Laufvorschau",
      ready: "Bereit",
      notReady: "Nicht bereit",
      modelPending: "Modell ausstehend",
      datasetPending: "Datensatz ausstehend",
      method: "Methode",
      length: "Länge",
      stepZero: "{count} Schritte",
      step: "{count} Schritt",
      stepTwo: "{count} Schritte",
      stepFew: "{count} Schritte",
      stepMany: "{count} Schritte",
      steps: "{count} Schritte",
      epochZero: "{count} Epochen",
      epoch: "{count} Epoche",
      epochTwo: "{count} Epochen",
      epochFew: "{count} Epochen",
      epochMany: "{count} Epochen",
      epochs: "{count} Epochen",
      batch: "Batch",
      context: "Kontext",
      lr: "LR",
      hardware: "Hardware",
      noGpu: "Keine GPU erkannt",
      hfToken: "HF-Token",
      saved: "Gespeichert",
      notSet: "Nicht festgelegt",
      files: "Dateien",
      model: "Modell",
      dataset: "Datensatz",
      downloadsOnStart: "Wird beim Start heruntergeladen",
      continuesOnStart: "Wird beim Start fortgesetzt",
      noticeModelDownload:
        "Dieses Modell befindet sich noch nicht auf dem Gerät. Beim Start des Trainings wird es automatisch heruntergeladen.",
      noticeModelPartial:
        "Vor dem Laden wird der unvollständige Modell-Download abgeschlossen.",
      noticeDatasetDownload:
        "Dieser Datensatz befindet sich noch nicht auf dem Gerät. Beim Start des Trainings wird er automatisch heruntergeladen.",
      noticeDatasetPartial:
        "Vor dem Einlesen wird der unvollständige Datensatz-Download abgeschlossen.",
      noticeTransformersUpgrade:
        "Keine installierte transformers-Version unterstützt diese Architektur bereits. Beim Start wird zuerst die Installation von transformers {version} angeboten.",
      noticeSixteenBitOnly:
        "Diese Architektur trainiert als 16-Bit-LoRA: 4 Bit ist dafür nicht verfügbar, der Lauf braucht also deutlich mehr VRAM als QLoRA.",
      noticeInstallSwitchesSixteenBit:
        "Wird diese Version installiert, statt den modelleigenen Code zu behalten, wechselt dieser Lauf zu 16-Bit-LoRA und braucht deutlich mehr VRAM als QLoRA.",
      advancedSettings: "Erweiterte Einstellungen",
      defaultAdvancedSettings: "Standardwerte",
      nonDefaultAdvancedSettings: "{count} abweichend",
    },
    datasetPicker: {
      noun: "Datensätze",
      selectDataset: "Datensatz auswählen",
      hubPlaceholder: "Hugging Face-Datensätze durchsuchen...",
      devicePlaceholder: "Lokale Datensätze durchsuchen...",
      useAsHubDataset: "Als Hugging Face-Datensatz verwenden",
      hfCacheLabel: "HF-Cache",
      scanningLocal: "Datensätze auf diesem Gerät werden durchsucht…",
      couldntScan: "Lokale Datensätze konnten nicht durchsucht werden",
      someLocationsUnscanned:
        "Einige Datensatzspeicherorte konnten nicht durchsucht werden.",
      noLocalDatasets:
        "Noch nichts auf diesem Gerät. Laden Sie einen Datensatz aus dem Hub herunter, erstellen Sie einen unter „Rezepte“ oder laden Sie eine Datei hoch.",
      openDataRecipes: "Rezepte öffnen",
      searchingHub: "Hugging Face wird durchsucht…",
      noDatasetsFound: "Keine Datensätze gefunden.",
      tokenRejectedTitle: "Hugging Face-Token abgelehnt",
      tokenRejectedBody:
        "Aktualisieren Sie Ihr Token unter Einstellungen → Allgemein und versuchen Sie es erneut.",
      hubUnreachable: "Hugging Face ist nicht erreichbar",
      cantUseDataset: "Datensatz kann nicht verwendet werden",
      reasonInvalidHubId:
        "Geben Sie eine gültige Hugging Face-Datensatz-ID ein: Repository oder Besitzer/Repository, nur mit Buchstaben, Zahlen, ., _ oder - (maximal 96 Zeichen pro Teil).",
      sourceRecipe: "Rezept",
      sourceUpload: "Upload",
      sourceLocal: "Lokal",
    },
    modelPicker: {
      noun: "Modelle",
      selectModel: "Modell auswählen",
      hubPlaceholder: "Hugging Face-ID suchen oder einfügen...",
      devicePlaceholder: "Lokale Modelle suchen oder einen Ordnerpfad einfügen...",
      useAsHubModel: "Als Hugging Face-Modell verwenden",
      useAsLocalPath: "Als lokalen Pfad verwenden",
      hfCacheLabel: "HF-Cache",
      scanningLocal: "Lokale Modelle werden durchsucht…",
      couldntScan: "Lokale Modelle konnten nicht durchsucht werden",
      someLocationsUnscanned:
        "Einige lokale Speicherorte konnten nicht durchsucht werden.",
      noLocalModels: "Keine lokalen Modelle gefunden.",
      noLocalModelsHint:
        "Fügen Sie oben einen Ordnerpfad ein oder wechseln Sie zu Hugging Face.",
      searchingHub: "Hugging Face wird durchsucht…",
      noModelsFound: "Keine Modelle gefunden.",
      tokenRejectedTitle: "Hugging Face-Token abgelehnt",
      tokenRejectedBody:
        "Aktualisieren Sie Ihr Token unter Einstellungen → Allgemein und versuchen Sie es erneut.",
      hubUnreachable: "Hugging Face ist nicht erreichbar",
      cantUseModel: "Modell kann nicht für das Training verwendet werden",
      reasonTypeMismatch:
        "Dieses Modell entspricht nicht dem im vorherigen Schritt ausgewählten Trainingstyp.",
      reasonEmptyId: "Geben Sie eine Modell-ID oder einen lokalen Modellpfad ein.",
      reasonInvalidHubId:
        "Geben Sie eine gültige Hugging Face-Modell-ID ein: Repository oder Besitzer/Repository, nur mit Buchstaben, Zahlen, ., _ oder - (maximal 96 Zeichen pro Teil).",
      reasonGguf: "GGUF-Modelle können nicht trainiert werden.",
      reasonAdapter:
        "Adapterausgaben können nicht als Basismodelle für das Training verwendet werden.",
      reasonNotTrainable:
        "Dieses Modell auf dem Gerät kann nicht trainiert werden.",
      reasonUnsupportedFormat:
        "Dieses Modellformat wird für das Training nicht unterstützt.",
      vramNeeds: "Benötigt ~{est} GB VRAM (GPU: {total} GiB)",
      vramTight: "~{est} GB VRAM (knapp bei {total} GiB)",
      vramApprox: "~{est} GB VRAM",
      sourceModelsFolder: "Modellordner",
      sourceHfCache: "HF-Cache",
      sourceLmStudio: "LM Studio",
      sourceOllama: "Ollama",
      sourceCustomFolder: "Benutzerdefinierter Ordner",
      sourceLocalModel: "Lokales Modell",
      vramOomBadge: "OOM",
      vramTightBadge: "Knapp",
    },
    methods: {
      qlora: {
        label: "QLoRA",
        hint: "4-Bit-Quantisierung. Niedrigster VRAM-Bedarf, schnellster Start.",
        note: "4-Bit",
      },
      lora: {
        label: "LoRA",
        hint: "16-Bit-Adapter. Ausgewogenes Verhältnis von Qualität und Speicherbedarf.",
        note: "16-Bit",
      },
      full: {
        label: "Vollständiges Fine-Tuning",
        hint: "Trainiert alle Gewichte. Höchste Qualität, benötigt am meisten VRAM.",
        note: "fp16",
      },
      cpt: {
        label: "Continued Pretraining",
        hint: "Fortgesetztes Vortraining für neue Domänen oder Sprachen.",
        note: "fortgesetzt",
      },
    },
    subtitles: {
      configure: "Training konfigurieren und starten",
      trainingInProgress: "Training läuft",
      viewPastRuns: "Frühere Trainingsläufe ansehen",
      viewingPastRun: "Früheren Lauf ansehen",
    },
    tabs: {
      configure: "Konfigurieren",
      currentRun: "Aktueller Lauf",
      history: "Verlauf",
    },
    loadingRuntime: "Trainingsumgebung wird geladen...",
    checkingSupport: "Dieser Rechner wird auf Trainingsunterstützung geprüft...",
    backToHistory: "Zurück zum Verlauf",
    dataset: {
      selectors: {
        subset: "Teilmenge",
        subsetTooltip:
          "Wähle die zu verwendende Teilmenge (Konfiguration) des Datensatzes aus.",
        trainSplit: "Trainingsaufteilung",
        trainSplitTooltip:
          "Wähle die Aufteilung aus, die für das Training verwendet werden soll.",
        evaluationSplit: "Evaluierungsaufteilung",
        evaluationSplitTooltip:
          "Wähle die Aufteilung für die Evaluierung aus. Keine bedeutet, dass während des Trainings keine Evaluierung stattfindet.",
        selectSubset: "Teilmenge auswählen...",
        selectSplit: "Aufteilung auswählen...",
        none: "Keine",
        loading: "Datensatzkonfigurationen und -aufteilungen werden geladen...",
        manualTitle: "Datensatzoptionen manuell eingeben",
        manualDescription:
          "Gib die genauen Namen der zu verwendenden Hugging-Face-Konfiguration und Aufteilungen ein.",
        manualSubsetPlaceholder: "Optionaler Konfigurationsname",
        manualRequired: "Eine Trainingsaufteilung ist erforderlich.",
        manualTooLong: "Verwende höchstens 128 Zeichen.",
        manualInvalid: "Dieser Wert enthält nicht unterstützte Zeichen.",
      },
      sourceAriaLabel: "Datensatzquelle",
      localDataset: "Lokaler Datensatz",
      localDatasetRows: " / {count} Zeilen",
      huggingFaceDataset: "Hugging-Face-Datensatz",
      localDatasetMetadata: "Metadaten des lokalen Datensatzes",
      dataRecipeOutput: "Data-Recipe-Ausgabe.",
      rows: "Zeilen",
      columns: "Spalten",
      batches: "Batches",
      updated: "Aktualisiert",
      evalDataset: "Eval-Datensatz",
      uploading: "Wird hochgeladen...",
      uploadEvalFile: "Eval-Datei hochladen",
      fileTooLarge: "Datei ist zu groß",
      fileTooLargeDescription:
        "{file} ist {size} groß. Trainings-Uploads unterstützen bis zu {limit}.",
      documentRedirect: {
        title: "Diese Datei muss zuerst konvertiert werden",
        genericFile: "Diese Datei",
        description:
          "{file} ist Quellmaterial und kein trainingsbereiter Datensatz. Verwenden Sie Data Recipes, um das Dokument in einen Datensatz umzuwandeln, und kehren Sie dann zum Fine-Tuning hierher zurück.",
        nextStepTitle: "Empfohlener nächster Schritt",
        nextStepDescription:
          "Öffnen Sie Learning Recipes und beginnen Sie mit einem dokumentbasierten Rezept wie PDF grounded QA.",
        openAction: "Learning Recipes öffnen",
      },
      evalDatasetDescription:
        "Optional. Wird keiner angegeben, wird ein kleiner Teil aus den Trainingsdaten abgetrennt.",
      advanced: "Erweitert",
      targetFormat: "Zielformat",
      targetFormatTooltip:
        "Format Ihrer Trainingsdaten. Die automatische Erkennung funktioniert bei den meisten Datensätzen.",
      streamingInfoAriaLabel: "Informationen zum Datensatz-Streaming",
      streaming: {
        label: "Streaming aktivieren",
        description:
          "Hugging-Face-Textdatensätze streamen, statt sie herunterzuladen.",
        unavailable: "Streaming nicht verfügbar. So aktivieren Sie es:",
        completionsUnavailable:
          "Nicht verfügbar, solange Datensatz-Streaming aktiviert ist.",
        blockers: {
          source:
            "Verwenden Sie einen Hugging-Face-Datensatz (keinen lokalen Upload und keine S3-Quelle).",
          maxSteps:
            "Setzen Sie Max. Schritte > 0 – Streaming-Datensätze haben keine bekannte Länge.",
          trainOnCompletions:
            "Deaktivieren Sie „Nur Assistenten-Antworten“.",
          evalSplit:
            "Wählen Sie einen separaten Eval-Split – die Evaluation ist aktiviert, aber es ist kein eigener Eval-Split festgelegt.",
          visionModel: "Vision-Modelle unterstützen kein Streaming.",
          audioModel: "Audio-Modelle unterstützen kein Streaming.",
          embeddingModel:
            "Embedding-Modelle unterstützen kein Streaming (das Training benötigt den vollständigen Datensatz).",
          imageDataset:
            "Dieser Datensatz scheint Bilder zu enthalten, die nicht gestreamt werden können.",
          audioDataset:
            "Dieser Datensatz scheint Audio zu enthalten, das nicht gestreamt werden kann.",
          appleSilicon:
            "Streaming wird auf Apple Silicon (MLX) noch nicht unterstützt.",
        },
        options: {
          trainOnCompletions: "nur Assistenten-Antworten",
          evaluation: "Evaluation (benötigt einen separaten Eval-Split)",
        },
        notifications: {
          turnedOffMaxSteps:
            "Streaming deaktiviert: Für Streaming muss „Max. Schritte“ auf einen festen Wert > 0 gesetzt sein.",
          adjusted:
            "Für Streaming angepasst. Inkompatible Optionen deaktiviert: {options}.",
          needsMaxSteps:
            "Streaming benötigt einen festen Wert für Max. Schritte (Streaming-Datensätze haben keine bekannte Länge). Setzen Sie zuerst Max. Schritte > 0.",
          enabledAdjusted:
            "Streaming aktiviert. Inkompatible Optionen deaktiviert: {options}.",
          disabledForDetectedModality:
            "Streaming wurde deaktiviert, da Bild- und Audio-Datensätze vollständig heruntergeladen werden müssen. Prüfen Sie die Einstellung und starten Sie das Training erneut.",
        },
      },
      auto: "Automatisch",
      rawText: "Rohtext",
      trainSplitStart: "Trainings-Split-Start",
      trainSplitStartTooltip:
        "Trainieren Sie nur auf einer Teilmenge Ihres Trainings-Splits, indem Sie einen Startzeilenindex angeben (inklusive, 0-basiert). Leer lassen, um bei der ersten Zeile zu beginnen.",
      trainSplitEnd: "Trainings-Split-Ende",
      trainSplitEndTooltip:
        "Letzter aus dem Trainings-Split einzuschließender Zeilenindex (inklusive, 0-basiert). Setzen Sie z. B. Start auf 0 und Ende auf 99, um auf den ersten 100 Zeilen zu trainieren. Leer lassen, um alle verbleibenden Zeilen zu verwenden.",
      endPlaceholder: "Ende",
      clear: "Leeren",
      dropFileOrClick: "1 Datei hier ablegen oder zum Hochladen klicken",
      uploadDetails: "Upload-Details",
      uploadDetailsTooltip:
        "Bis zu {limit} pro Datei. PDF-, DOCX- und TXT-Dateien sind keine trainingsfertigen Datensätze; konvertieren Sie sie daher zuerst in den Lernrezepten.",
      viewDataset: "Datensatz ansehen",
      uploadFailed: "Upload fehlgeschlagen",
      unknownError: "Unbekannter Fehler",
      unsupportedFileType: "Nicht unterstützter Dateityp",
      uploadOneFileType: "Laden Sie eine {types}-Datei hoch.",
      datasetUploaded: "Datensatz hochgeladen",
      evalDatasetUploaded: "Eval-Datensatz hochgeladen",
      uploadOneFileAtATime: "Laden Sie jeweils eine Datei hoch",
      uploadSingleFileDescription:
        "Der Trainingsdatensatz-Upload akzeptiert eine einzelne Datei.",
      previewLoadingHuggingFace:
        "Datensatzvorschau wird von Hugging Face abgerufen...",
      previewLoading: "Vorschau wird geladen...",
      mappingRequirements: {
        audioAndText: "Audio und Text",
        imageAndText: "Bild und Text",
        instructionAndOutput: "Anweisung und Ausgabe",
        humanAndGpt: "Mensch und GPT",
        userAndAssistant: "Benutzer und Assistent",
      },
      mappingStatus: {
        heuristicTitle: "Heuristisch erkannte Zuordnung",
        readyTitle: "Zuordnung bereit",
        requiredTitle: "Datensatzspalten zuordnen",
        heuristicDescription:
          "Die Spaltenzuordnung unten wurde automatisch anhand von Heuristiken erkannt. Prüfen und ändern Sie sie über die Dropdown-Menüs in den Spaltenüberschriften oder verwenden Sie KI-Unterstützung für eine intelligentere Zuordnung.",
        readyDescription:
          "Sieht gut aus. Dieser Datensatz wird automatisch konvertiert.",
        requiredDescription:
          "Weisen Sie den Spalten über die Dropdown-Menüs in den Überschriften Rollen zu. Weisen Sie mindestens {required} zu.",
      },
      s3: {
        title: "S3-Konfiguration",
        description:
          "Laden Sie .parquet-, .json-, .jsonl- oder .csv-Datensätze aus Amazon S3",
        bucket: "Bucket-Name",
        bucketPlaceholder: "my-training-data-bucket",
        region: "AWS-Region",
        regionPlaceholder: "us-east-1",
        prefix: "Pfad-Präfix",
        prefixPlaceholder: "datasets/whisper/",
        accessKeyId: "Access Key ID",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Secret Access Key",
        secretAccessKeyPlaceholder: "Ihr AWS Secret Access Key",
        useIamRole: "IAM-Rolle verwenden",
      },
    },
    params: {
      mode: {
        simple: "Einfach",
        advanced: "Erweitert",
        ariaLabel: "Parametermodus",
      },
      projectName: "Projektname",
      optional: "Optional",
      projectNameDescription:
        "Wird für Namen der Trainingsausgabe-Ordner, Export-Standardwerte und den Verlauf verwendet.",
      loraSettings: "LoRA-Einstellungen",
      trainingHyperparameters: "Trainings-Hyperparameter",
      maxSteps: "Max. Schritte",
      epochs: "Epochen",
      useMaxSteps: "Max. Schritte verwenden",
      useEpochs: "Epochen verwenden",
      maxStepsTooltip: "Überschreibt die gesamten Optimierer-Schritte.",
      epochsTooltip: "Anzahl vollständiger Durchläufe über den Datensatz.",
      contextLength: "Kontextlänge",
      contextLengthTooltip:
        "Maximale Anzahl an Token pro Trainingsbeispiel.",
      customContextLength: "Eigenen Wert eingeben",
      learningRate: "Lernrate",
      learningRateTooltip:
        "Schrittgröße für Gewichtsaktualisierungen. Niedrigere Werte trainieren langsamer, aber stabiler.",
      learningRateDescription:
        "Empfohlen: 2e-4 für LoRA, 5e-5 für CPT, 2e-5 für vollständiges Fine-Tuning",
      embeddingLearningRate: "Embedding-Lernrate",
      embeddingLearningRateTooltip:
        "Wird nur verwendet, wenn CPT embed_tokens trainiert. Embeddings destabilisieren leichter als LoRA-Gewichte und benötigen daher meist eine kleinere Lernrate. Leer lassen, um lr/10 zu verwenden; typischer Bereich ist 2- bis 10-mal kleiner als die Haupt-Lernrate. Erhöhen Sie sie nur, wenn die Anpassung von Vokabular oder Domänen-Token zu langsam ist.",
      rank: "Rank",
      rankTooltip:
        "Dimension der Low-Rank-Matrizen. Höher = mehr Kapazität.",
      alpha: "Alpha",
      alphaTooltip:
        "Skalierungsfaktor für LoRA-Updates. Üblicherweise 2x Rank.",
      dropout: "Dropout",
      dropoutTooltip:
        "Dropout-Wahrscheinlichkeit für LoRA-Schichten, um Overfitting zu reduzieren.",
      visionLayers: "Vision-Schichten",
      languageLayers: "Sprach-Schichten",
      attentionModules: "Attention-Module",
      mlpModules: "MLP-Module",
      targetModules: "Ziel-Module",
      enableLora: "LoRA aktivieren",
      trainWithLora: "Mit LoRA trainieren",
      stableRank: "Stable Rank",
      memoryEfficient: "Speichereffizient",
      weightDecomposed: "Gewichtszerlegt",
      notSupportedAppleSilicon: "Auf Apple Silicon nicht unterstützt",
      optimization: "Optimierung",
      schedule: "Zeitplan",
      memory: "Speicher",
      optimizer: "Optimierer",
      optimizerTooltip:
        "Optimierungsalgorithmus. 8-Bit-Varianten reduzieren den Speicherbedarf. Fused wird für Vision-Modelle empfohlen.",
      optimizerTooltipMlx:
        "Optimierungsalgorithmus. AdamW ist die Standardeinstellung. Lion benötigt weniger Speicher, braucht aber normalerweise eine niedrigere Lernrate.",
      lrScheduler: "LR-Scheduler",
      lrSchedulerTooltip:
        "Wie sich die Lernrate über das Training verändert. Linear fällt gleichmäßig; Cosine fällt in einer Kurve.",
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
      batchSize: "Batch-Größe",
      batchSizeTooltip:
        "Pro Schritt verarbeitete Beispiele. Höher verbraucht mehr VRAM.",
      gradAccum: "Grad-Akkumulation",
      gradAccumTooltip:
        "Simuliert größere Batch-Größen ohne zusätzlichen VRAM.",
      weightDecay: "Weight Decay",
      weightDecayTooltip: "L2-Regularisierung zur Vermeidung von Overfitting.",
      warmupSteps: "Warmup-Schritte",
      warmupStepsTooltip:
        "Erhöht die Lernrate zu Trainingsbeginn schrittweise für mehr Stabilität.",
      scheduleEpochsTooltip:
        "Anzahl vollständiger Durchläufe über den Datensatz. Auf 0 setzen, um nach max. Schritten zu laufen.",
      saveSteps: "Speicherintervall in Schritten",
      saveStepsTooltip:
        "Speichert alle N Schritte einen Checkpoint. 0 zum Deaktivieren.",
      evalSteps: "Eval-Schritte",
      evalStepsTooltip:
        "Anteil der gesamten Trainingsschritte zwischen Auswertungen (0-1). Auf 0 setzen, um die Auswertung zu deaktivieren. Z. B. 0.01 = alle 1 % der Schritte auswerten.",
      seed: "Seed",
      seedTooltip: "Zufalls-Seed für Reproduzierbarkeit.",
      gradCheckpoint: "Grad-Checkpoint",
      gradCheckpointTooltip:
        "Rechenaufwand gegen Speicher tauschen, indem Aktivierungen neu berechnet werden.",
      none: "Keine",
      standard: "Standard",
      enablePacking: "Packing aktivieren",
      assistantCompletionsOnly: "Nur Assistenten-Antworten",
      readMore: "Mehr erfahren",
    },
    training: {
      startTraining: "Training starten",
      starting: "Wird gestartet...",
      loadingModel: "Modell wird geladen...",
      checkingDataset: "Datensatz wird geprüft...",
      chooseModel: "Modell auswählen",
      chooseDataset: "Datensatz auswählen",
      chooseModelAndDataset: "Modell und Datensatz auswählen",
      modelUnverified:
        "Die Modelleinstellungen konnten nicht überprüft werden. Prüfen Sie Ihre Verbindung oder Ihr Hugging Face-Token und versuchen Sie es erneut.",
      legacyDatasetScriptUnsupported:
        "Dieser Hub-Datensatz basiert auf einem veralteten benutzerdefinierten Skript und wird in diesem Trainingsablauf nicht unterstützt.",
      hfModelAccessDenied:
        "Hugging Face hat den Zugriff auf dieses Modell verweigert. Fügen Sie ein gültiges Hugging Face-Token mit Repository-Zugriff hinzu, akzeptieren Sie erforderliche Zugriffsbedingungen und versuchen Sie es erneut.",
      hfModelVerificationRateLimited:
        "Die Hugging Face-Modellprüfung ist ratenbegrenzt. Versuchen Sie es in Kürze erneut.",
      hfModelVerificationFailed:
        "Das Hugging Face-Modell konnte nicht überprüft werden. Prüfen Sie die Repository-ID und Ihr Zugriffstoken.",
      hfModelMetadataUnavailable:
        "Die Hugging Face-Modellmetadaten sind vorübergehend nicht verfügbar. Versuchen Sie es erneut, bevor Sie das Training starten.",
      datasetUnverified:
        "Die Kompatibilität des Datensatzes mit diesem Modell konnte nicht überprüft werden. Prüfen Sie Ihre Verbindung oder Ihr Hugging Face-Token. Beim Start des Trainings wird die Prüfung erneut versucht.",
      setupChanged:
        "Die Trainingseinstellungen haben sich während der Überprüfung geändert. Prüfen Sie sie und starten Sie das Training erneut.",
      validation: {
        s3MultimodalUnsupported:
          "S3-Datensätze werden für Vision- oder Audio-Training noch nicht unterstützt.",
        s3BucketRequired: "Geben Sie zuerst den Namen eines S3-Buckets ein.",
        s3CredentialsRequired:
          "Geben Sie S3-Zugriffsschlüssel an oder aktivieren Sie die IAM-Rolle.",
        modelRequired: "Wählen Sie zuerst ein Basismodell aus.",
        learningRatePositive: "Geben Sie eine Lernrate größer als null ein.",
        embeddingLearningRateRange:
          "Geben Sie eine Embedding-Lernrate größer als 0 und kleiner als 1 ein.",
        hfDatasetRequired:
          "Wählen Sie zuerst einen Hugging Face-Datensatz aus.",
        hfDatasetSplitRequired:
          "Wählen Sie zuerst eine Trainingsaufteilung aus oder geben Sie sie ein.",
        localDatasetRequired: "Wählen Sie zuerst einen lokalen Datensatz aus.",
        unsupportedDatasetSource: "Nicht unterstützte Datensatzquelle.",
      },
      startFailed: "Training konnte nicht gestartet werden",
      startUnconfirmed:
        "Unsloth konnte nicht bestätigen, ob das Training gestartet wurde. Der Status wird im Hintergrund geprüft.",
      stopFailed: "Training konnte nicht gestoppt werden",
      trainingStillActiveTitle: "Training läuft noch",
      stopBeforeConfig:
        "Stoppen Sie zuerst das Training und kehren Sie dann zur Konfiguration zurück.",
      resumeFailed: "Fortsetzen des Trainings fehlgeschlagen",
      resumeFailedTitle: "Training konnte nicht fortgesetzt werden",
      resumeUnavailable:
        "Nur gestoppte oder fehlerhaft beendete Läufe mit einem gespeicherten Checkpoint können fortgesetzt werden.",
      uploadConfigTooltip: "Eine gespeicherte YAML-Konfiguration laden",
      saveConfigTooltip: "Aktuelle Konfiguration als YAML herunterladen",
      resetConfigTooltip: "Auf Modellstandardwerte zurücksetzen",
      configLoaded: "Konfiguration geladen",
      failedToLoadConfig: "Konfiguration konnte nicht geladen werden",
      invalidYamlFile: "Ungültige YAML-Datei",
      configTooLarge:
        "Die Trainingskonfiguration ist zu groß (maximal 1 MiB).",
      failedToReadFile: "Datei konnte nicht gelesen werden",
      failedToSaveConfig: "Konfiguration konnte nicht gespeichert werden",
      parametersReset: "Parameter auf Modellstandardwerte zurückgesetzt",
      audioIncompatible:
        "Dieses Modell unterstützt kein Audio. Wechseln Sie zu einem audiofähigen Modell oder wählen Sie einen Nicht-Audio-Datensatz.",
      visionIncompatible:
        "Ein Textmodell ist mit einem multimodalen Datensatz nicht kompatibel. Wechseln Sie zu einem Vision-Modell oder wählen Sie einen reinen Textdatensatz.",
      cancelTitle: "Training abbrechen",
      cancelDescription:
        "Möchten Sie den aktuellen Trainingslauf abbrechen?",
      continueAction: "Training fortsetzen",
      cancelAction: "Training abbrechen",
      stopTitle: "Training stoppen",
      stopDescription:
        "Wählen Sie, wie Sie den aktuellen Trainingslauf stoppen möchten. „Stoppen und speichern“ schreibt einen Checkpoint, von dem aus Sie später fortsetzen können; ein einfach gestopptes Training kann nicht fortgesetzt werden.",
      stopAction: "Stoppen",
      stopping: "Wird gestoppt...",
      stopAndSave: "Stoppen und speichern",
      compareInChat: "Im Chat vergleichen",
      exportModel: "Modell exportieren",
      milestone: "Meilenstein",
      halfwayDone: "Zur Hälfte fertig. Das Training ist über 50 %.",
      doneNextStep:
        "Training abgeschlossen. Nächster Schritt: Basis- und feinabgestimmte Ausgaben vergleichen.",
    },
    history: {
      title: "Verlauf",
      filesDeleted: "Dateien gelöscht",
      deleteArtifactsLabel: "Adapterdateien auf dem Datenträger ebenfalls löschen",
      deleteArtifactsDescription:
        "Entfernt den Ausgabeordner des Laufs einschließlich gespeicherter Adapter und Checkpoints.",
      deleteArtifactsSharedNote:
        "Ein anderer Lauf verwendet denselben Ausgabeordner. Die Dateien bleiben erhalten, bis der letzte Lauf gelöscht wird, der sie verwendet.",
      artifactsKeptShared:
        "Lauf gelöscht. Die Adapterdateien wurden beibehalten, da ein anderer Lauf denselben Ordner verwendet.",
      deleteArtifactsActiveError:
        "Diese Dateien werden vom laufenden Trainingslauf verwendet. Stoppen Sie das Training, bevor Sie sie löschen.",
      deleteArtifactsFailed:
        "Der Lauf wurde gelöscht, seine Dateien konnten jedoch nicht entfernt werden.",
      deleteArtifactsRetainedError:
        "Die Adapterdateien konnten nicht entfernt werden. Der Trainingslauf wurde daher im Verlauf beibehalten.",
      emptyDescription:
        "Noch keine Trainingsläufe. Starten Sie Ihren ersten Trainingslauf im Tab „Konfigurieren“.",
      loadError: "Trainingsläufe konnten nicht geladen werden",
      deleteError:
        "Trainingslauf konnte nicht gelöscht werden. Bitte erneut versuchen.",
      retry: "Erneut versuchen",
      loadMore: "Mehr laden",
      loading: "Wird geladen...",
      loadingRun: "Trainingslauf wird geladen...",
      runNotFound: "Lauf nicht gefunden",
      deleteTitle: "Trainingslauf löschen?",
      deleteDescription:
        "Dadurch werden dieser Trainingslauf und alle seine Metriken dauerhaft gelöscht. Diese Aktion kann nicht rückgängig gemacht werden.",
      resumeTraining: "Training fortsetzen",
      resuming: "Wird fortgesetzt...",
      deleteRun: "Lauf löschen",
      loss: "Loss",
      steps: "Schritte",
      lossTrendSparkline: "Loss-Trend-Sparkline",
      relativeJustNow: "gerade eben",
      status: {
        completed: "Abgeschlossen",
        stopped: "Gestoppt",
        error: "Fehler",
        running: "Läuft",
        continued: "Fortgesetzt",
      },
      message: {
        completed: "Training abgeschlossen",
        stopped: "Training gestoppt",
        running: "Training läuft",
        errored: "Training fehlgeschlagen",
      },
      copyPreviewLink: "Vorschaulink kopieren",
      previewLinkCopied: "Vorschaulink kopiert",
      previewLinkCopyFailed: "Der Link konnte nicht kopiert werden",
    },
    charts: {
      settings: "Diagramm-Einstellungen",
      settingsDescription:
        "Passen Sie die Diagrammdarstellung an, während das Training weiterläuft.",
      openSettings: "Diagramm-Einstellungen öffnen",
      viewWindow: "Ansichtsfenster",
      viewWindowDescription:
        "Nur die neuesten Schritte oder den gesamten Verlauf anzeigen.",
      window: "Fenster",
      all: "Alle",
      trainingLoss: "Trainings-Loss",
      trainingLossDescription: "Overlays und EMA-Glättung steuern.",
      smoothing: "Glättung",
      smoothingDescription:
        "Nach rechts bewegen für mehr Glättung. `0` = Rohdaten.",
      showRawLoss: "Roh-Loss anzeigen",
      showSmoothedLoss: "Geglätteten Loss anzeigen",
      showAverageLine: "Durchschnittslinie anzeigen",
      scaleAndCleanup: "Skala und Bereinigung",
      linear: "Linear",
      log: "Logarithmisch",
      noClip: "Kein Clipping",
      clipP99: "Clip p99",
      clipP95: "Clip p95",
      lossAxis: "Loss-Achse",
      gradientNormAxis: "Gradientennorm-Achse",
      learningRateAxis: "Lernraten-Achse",
      resetDefaults: "Standardwerte zurücksetzen",
      loss: "Loss",
      smoothed: "Geglättet",
      evalLoss: "Eval-Loss",
      learningRate: "Lernrate",
      lr: "LR",
      gradNorm: "Grad-Norm",
      gradientNorm: "Gradientennorm",
      step: "Schritt {step}",
      averageValue: "Ø {value}",
      waitingForFirstEvaluationStep:
        "Warte auf ersten Auswertungsschritt...",
      evaluationNotConfigured: "Auswertung nicht konfiguriert",
      evalChartWillAppear:
        "Das Diagramm erscheint, sobald eval_steps erreicht ist",
      setEvalDatasetAndSteps:
        "Legen Sie Eval-Datensatz & eval_steps fest, um den Eval-Loss zu verfolgen",
    },
    progress: {
      title: "Trainingsfortschritt",
      liveMetrics: "Live-Trainingsmetriken",
      exportGguf: "Nach GGUF exportieren",
      openConfig: "Trainingskonfiguration öffnen",
      configLabel: "Trainingskonfiguration",
      hyperparams: "Hyperparameter",
      epochs: "Epochen",
      batchSize: "Batch-Größe",
      learningRate: "Lernrate",
      optimizer: "Optimierer",
      maxSteps: "Max. Schritte",
      contextLength: "Kontextlänge",
      warmupSteps: "Warmup-Schritte",
      rank: "Rank",
      alpha: "Alpha",
      dropout: "Dropout",
      variant: "Variante",
      epoch: "Epoche {value}",
      percentComplete: "{percent} % abgeschlossen",
      stepProgress: "Schritt {current} / {total}",
      loss: "Loss",
      lr: "LR",
      gradNorm: "Grad-Norm",
      project: "Projekt",
      model: "Modell",
      method: "Methode",
      elapsed: "Vergangen: {value}",
      eta: "ETA: {value}",
      stepsPerSecond: "{value} Schritte/s",
      noStepsPerSecond: "-- Schritte/s",
      tokens: "Token: {value}",
      gpuMonitor: "GPU-Monitor",
      live: "Live",
      utilization: "Auslastung",
      temperature: "Temperatur",
      vram: "VRAM",
      power: "Leistung",
      phase: {
        idle: "Inaktiv",
        downloadingModel: "Modell wird heruntergeladen",
        downloadingDataset: "Datensatz wird heruntergeladen",
        loadingModel: "Modell wird geladen",
        loadingDataset: "Datensatz wird geladen",
        configuring: "Wird konfiguriert",
        training: "Training",
        finalizing: "Modell wird gespeichert",
        completed: "Abgeschlossen",
        error: "Fehler",
        stopped: "Gestoppt",
      },
    },
    trainingStart: {
      ready: "Bereit",
      downloading: "Wird heruntergeladen",
      preparing: "Wird vorbereitet",
      left: "{eta} verbleibend",
      downloaded: "{size} heruntergeladen",
      terminalStart: "> Unsloth-Training beginnt...",
      preparingResources: "> Modell und Datensatz werden vorbereitet...",
      gettingReady: "> Wir bereiten alles für Ihren Lauf vor...",
      waitingForFirstStep:
        "> {message} | warte auf ersten Schritt... ({step})",
      resumingTraining: "Training wird fortgesetzt...",
      startingTraining: "Training wird gestartet...",
      dataset: "Datensatz",
      datasetStreaming: "Datensatz: Streaming (kein vollständiger Download)",
      modelWeights: "Modellgewichte",
    },
  },
  modelMemory: {
    readout:
      "Gewichte {model} + Kontext {context} = {total} von {budget} nutzbarem VRAM",
    readoutWithSpec:
      "Gewichte {model} + KV {kv} + MTP-Entwurf {spec} = {total} von {budget} nutzbarem VRAM",
    kvRate: "KV reserviert, ca. {rate}/Token",
    oomLikely: "Mit den aktuellen Einstellungen ist ein Speicherüberlauf wahrscheinlich",
    tooLarge: "Größer als der VRAM, wird auf die CPU ausgelagert. Eine kleinere Quantisierung läuft schneller",
  },
} satisfies DeepPartialMessageTree<typeof en>;
