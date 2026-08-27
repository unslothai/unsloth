// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const it = {
  common: {
    cancel: "Annulla",
    close: "Chiudi",
    delete: "Elimina",
    done: "Fatto",
    error: "Errore",
    export: "Esporta",
    help: "Aiuto",
    loading: "Caricamento...",
    new: "Nuovo",
    rename: "Rinomina",
    save: "Salva",
    saving: "Salvataggio...",
    search: "Cerca",
    shutdown: "Arresta",
  },
  shell: {
    beta: "BETA",
    brand: "unsloth",
    product: "Unsloth",
    accountMenu: "Menu dell'account di {name}",
    updateAvailable: "Aggiornamento disponibile",
    resize: {
      collapse: "Clicca per ridurre",
      expand: "Clicca per espandere",
      drag: "Trascina per ridimensionare",
    },
    aria: {
      home: "Home di Unsloth",
      closeSidebar: "Chiudi la barra laterale",
      openSidebar: "Apri la barra laterale",
      resizeSidebar: "Ridimensiona o riduci la barra laterale",
      resizeRunSettings: "Ridimensiona o chiudi le impostazioni del run",
      openRunSettings: "Apri le impostazioni del run",
      chatOptions: "Opzioni della chat",
      runOptions: "Opzioni del run",
    },
    navigation: {
      newChat: "Nuova chat",
      returnToChat: "Torna alla chat",
      returnToChats: "Torna alle {count} chat",
      chatGenerating: "Generazione in corso",
      compare: "Confronta",
      search: "Cerca",
      hub: "Hub dei modelli",
      projects: "Progetti",
      train: "Addestra",
      recipes: "Ricette",
      images: "Immagini",
      video: "Video",
      audio: "Audio",
      trainChecking: "Verifica del supporto all'addestramento su questa macchina...",
      videoChecking: "Verifica del supporto video su questa macchina...",
      more: "Altro",
      customizeSidebar: "Personalizza la barra laterale",
      newBadge: "Novità",
      export: "Esporta",
      recents: "Recenti",
      noChatsYet: "Ancora nessuna chat",
      showMore: "Mostra altro",
      showLess: "Mostra meno",
      settings: "Impostazioni",
      api: "API",
      lightMode: "Tema chiaro",
      darkMode: "Tema scuro",
      guidedTour: "Tour guidato",
      help: "Aiuto",
      logOut: "Esci",
      shutdown: "Arresta",
    },
    notFound: {
      title: "Pagina non trovata",
      description: "{path} non esiste.",
      backToChat: "Torna alla chat",
    },
    selection: {
      pinProjects: "Fissa i progetti",
      unpinProjects: "Rimuovi i progetti fissati",
      deleteProjects: "Elimina i progetti",
      deleteProjectsTitle: "Elimina i progetti",
      deleteProjectsDescription:
        "Eliminare {count} progetti? Le loro chat vengono eliminate definitivamente.",
      deleteProjectsFilesDescription:
        "La cartella dello spazio di lavoro di ogni progetto viene rimossa dal disco.",
      countSelected: "{count} selezionate",
      pinChats: "Fissa le chat",
      unpinChats: "Rimuovi le chat fissate",
      archiveChats: "Archivia le chat",
      markUnread: "Segna come non letta",
      deleteChats: "Elimina le chat",
      deleteTitle: "Elimina le chat",
      deleteDescription: "Eliminare {count} chat? L'operazione non è reversibile.",
      deleteFilesDescription:
        "La cartella sandbox di ogni chat viene rimossa dal disco. I file scritti dentro un progetto restano nello spazio di lavoro di quel progetto.",
      deleteFilesLabel: "Elimina i file e la cartella sandbox",
      deleteChatFilesDescription:
        "La cartella sandbox di questa chat viene rimossa dal disco. I file scritti all'interno di un progetto restano nello spazio di lavoro del progetto.",
    },
    organize: {
      sidebarHeading: "Organizza la barra laterale",
      byProject: "Per progetto",
      inOneList: "In un solo elenco",
      sortChatsBy: "Ordina le chat per",
      sortPinnedBy: "Ordina i fissati per",
      priority: "Priorità",
      lastUpdated: "Ultimo aggiornamento",
      manualOrder: "Ordine manuale",
      moveUp: "Sposta su",
      moveDown: "Sposta giù",
      organizeChats: "Organizza le chat",
      organizeProjects: "Organizza i progetti",
      sortPinnedChats: "Ordina le chat fissate",
    },
    dialog: {
      deleteChat: {
        title: "Elimina la chat",
        description: "Vuoi davvero eliminare la chat «{name}»?",
      },
      deleteRun: {
        title: "Elimina il run di addestramento",
        description: "Vuoi davvero eliminare il run «{name}»?",
      },
      renameChat: {
        title: "Rinomina la chat",
        placeholder: "Titolo della chat",
      },
      renameRun: {
        title: "Rinomina il run",
        placeholder: "Nome del run",
      },
    },
    toast: {
      cannotDeleteRunningRun:
        "Non puoi eliminare un run di addestramento in corso",
      failedToDeleteChat: "Eliminazione della chat non riuscita",
      failedToDeleteRun: "Eliminazione del run non riuscita",
      failedToRenameChat: "Impossibile rinominare la chat",
      failedToRenameRun: "Impossibile rinominare il run",
    },
  },
  settings: {
    title: "Impostazioni",
    dialog: {
      title: "Impostazioni",
      description: "Gestisci le tue preferenze di Unsloth.",
      closeAriaLabel: "Chiudi le impostazioni",
      searchPlaceholder: "Cerca nelle impostazioni…",
      searchNoResults: "Nessuna impostazione trovata.",
      panelFailed: "Non è stato possibile caricare questa sezione.",
      panelReload: "Ricarica",
    },
    tabs: {
      general: "Generali",
      profile: "Profilo",
      appearance: "Aspetto",
      resources: "Sistema",
      chat: "Chat",
      voice: "Voce",
      connections: "Connessioni",
      data: "Dati",
      apiKeys: "API",
      remoteLan: "Remoto e LAN",
      agents: "Agenti",
      debugging: "Log",
      about: "Informazioni",
      keyboardShortcuts: "Scorciatoie",
    },
    keyboardShortcuts: {
      title: "Scorciatoie da tastiera",
      description:
        "Modifica una scorciatoia, oppure cancellala per liberare la combinazione per il browser o il sistema.",
      searchPlaceholder: "Cerca scorciatoie…",
      noResults: "Nessuna scorciatoia corrisponde alla ricerca.",
      unassigned: "Non assegnata",
      recording: "Premi i tasti…",
      recordingHint: "Premi la nuova combinazione, o Esc per annullare.",
      needsModifier: "Aggiungi ⌘, Ctrl o Alt. Un tasto singolo intercetterebbe la digitazione.",
      conflict: "Usata anche da un'altra scorciatoia",
      conflictShadowed: "Un'altra scorciatoia usa questa combinazione e viene eseguita al suo posto",
      edit: "Cambia scorciatoia",
      clear: "Rimuovi scorciatoia",
      reset: "Ripristina il valore predefinito",
      resetAll: "Ripristina tutto",
      primarySlot: "Scorciatoia",
      alternateSlot: "Scorciatoia alternativa",
      browserReserved:
        "Il browser potrebbe riservare questa combinazione per sé. Nell’app desktop funziona.",
      actions: {
        openSettings: {
          label: "Apri le impostazioni",
          description: "Apri la finestra delle impostazioni",
        },
        openKeyboardShortcuts: {
          label: "Scorciatoie da tastiera",
          description: "Apri questo elenco di scorciatoie",
        },
        searchChats: {
          label: "Cerca nelle chat",
          description: "Apri la ricerca delle chat",
        },
        openMcpServers: {
          label: "Server MCP",
          description: "Configura i server MCP per questa chat",
        },
        logOut: {
          label: "Esci",
          description: "Esci da Unsloth",
        },
        approveToolRequest: {
          label: "Approva la richiesta",
          description: "Consenti la chiamata allo strumento in attesa",
        },
        declineToolRequest: {
          label: "Rifiuta la richiesta",
          description: "Nega la chiamata allo strumento in attesa",
        },
        newChat: {
          label: "Nuova chat",
          description: "Avvia una nuova chat",
        },
        newTemporaryChat: {
          label: "Nuova chat temporanea",
          description: "Avvia una chat che non viene salvata nella cronologia",
        },
        newStandaloneChat: {
          label: "Nuova chat autonoma",
          description: "Avvia una chat fuori da qualsiasi progetto",
        },
        archiveChat: {
          label: "Archivia chat",
          description: "Archivia le chat selezionate, o quella corrente",
        },
        markChatUnread: {
          label: "Segna come non letta",
          description: "Segna come non lette le chat selezionate, o quella corrente",
        },
        togglePinChat: {
          label: "Fissa/sblocca",
          description: "Fissa o sblocca le chat selezionate, o quella corrente",
        },
        selectAllChats: {
          label: "Seleziona tutte le chat",
          description: "Seleziona tutte le chat nella barra laterale",
        },
        clearChatSelection: {
          label: "Annulla selezione",
          description: "Deseleziona le chat scelte. Anche Esc le annulla",
        },
        deleteSelectedChats: {
          label: "Elimina le chat selezionate",
          description: "Elimina tutte le chat selezionate",
        },
        nextRecentlyViewedChat: {
          label: "Chat vista successiva",
          description: "Avanza tra le chat viste di recente",
        },
        previousRecentlyViewedChat: {
          label: "Chat vista precedente",
          description: "Torna indietro tra le chat viste di recente",
        },
        nextChat: {
          label: "Chat successiva",
          description: "Passa alla chat successiva nella barra laterale",
        },
        previousChat: {
          label: "Chat precedente",
          description: "Passa alla chat precedente nella barra laterale",
        },
        nextChatNeedingAttention: {
          label: "Chat da seguire successiva",
          description: "Passa alla prossima chat in generazione, in coda o non letta",
        },
        clearAllUnreads: {
          label: "Segna tutto come letto",
          description: "Segna ogni chat come letta",
        },
        goToRecentChat1: {
          label: "Vai alla chat recente 1",
          description: "Apri la chat 1 in Recenti",
        },
        goToRecentChat2: {
          label: "Vai alla chat recente 2",
          description: "Apri la chat 2 in Recenti",
        },
        goToRecentChat3: {
          label: "Vai alla chat recente 3",
          description: "Apri la chat 3 in Recenti",
        },
        goToRecentChat4: {
          label: "Vai alla chat recente 4",
          description: "Apri la chat 4 in Recenti",
        },
        goToRecentChat5: {
          label: "Vai alla chat recente 5",
          description: "Apri la chat 5 in Recenti",
        },
        goToRecentChat6: {
          label: "Vai alla chat recente 6",
          description: "Apri la chat 6 in Recenti",
        },
        switchToChat: {
          label: "Vai a Chat",
          description: "Vai allo spazio di lavoro chat",
        },
        switchToProjects: {
          label: "Vai a Progetti",
          description: "Vai allo spazio di lavoro progetti",
        },
        switchToHub: {
          label: "Vai all’Hub modelli",
          description: "Vai all’hub dei modelli",
        },
        switchToTrain: {
          label: "Vai ad Addestramento",
          description: "Vai allo spazio di lavoro di addestramento",
        },
        switchToRecipes: {
          label: "Vai a Recipes",
          description: "Vai a Data Recipes",
        },
        switchToImages: {
          label: "Vai a Immagini",
          description: "Vai allo spazio di lavoro immagini",
        },
        switchToVideo: {
          label: "Vai a Video",
          description: "Vai allo spazio di lavoro video",
        },
        switchToAudio: {
          label: "Vai ad Audio",
          description: "Vai allo spazio di lavoro audio",
        },
        switchToExport: {
          label: "Vai a Esporta",
          description: "Vai allo spazio di lavoro di esportazione",
        },
        toggleSidebar: {
          label: "Mostra/nascondi barra laterale",
          description: "Mostra o nascondi la barra laterale",
        },
        toggleApiMonitor: {
          label: "Mostra/nascondi attività API",
          description: "Mostra o nascondi il monitor di attività API",
        },
        openModelPicker: {
          label: "Apri selettore modello",
          description: "Scegli il modello per questa chat",
        },
        openProjectPicker: {
          label: "Apri selettore progetto",
          description: "Passa a un altro progetto dall'intestazione della chat",
        },
        startDictation: {
          label: "Dettatura",
          description: "Avvia o interrompi la dettatura nell’editor",
        },
        attachFiles: {
          label: "Allega foto e file",
          description: "Aggiungi un allegato all’editor",
        },
        sendMessage: {
          label: "Invia messaggio",
          description: "Invia il contenuto dell’editor",
        },
        cycleReasoningEffort: {
          label: "Scorri lo sforzo di ragionamento",
          description: "Passa in rassegna i livelli di sforzo di ragionamento",
        },
        increaseReasoningEffort: {
          label: "Aumenta lo sforzo di ragionamento",
          description: "Alza di un livello lo sforzo di ragionamento",
        },
        decreaseReasoningEffort: {
          label: "Riduci lo sforzo di ragionamento",
          description: "Abbassa di un livello lo sforzo di ragionamento",
        },
        toggleFastMode: {
          label: "Attiva/disattiva la modalità Fast",
          description: "Attiva o disattiva la modalità Fast",
        },
        renameChat: {
          label: "Rinomina chat",
          description: "Rinomina la chat corrente",
        },
        forkChat: {
          label: "Dirama chat",
          description: "Crea una nuova chat a partire dall’ultimo messaggio",
        },
        copyChatAsMarkdown: {
          label: "Copia come Markdown",
          description: "Copia l’intera chat negli appunti come Markdown",
        },
        copySessionId: {
          label: "Copia ID sessione",
          description: "Copia l’ID di sessione sandbox di questa chat",
        },
      },
    },
    debugging: {
      logSection: "File di log",
      source: "File di log",
      sourceHint: "I runner dei modelli scrivono i propri log, quindi un caricamento o una generazione non riusciti sono spesso spiegati lì anziché nel log del server.",
      path: "Posizione",
      pathCopy: "Copia il percorso",
      refreshSection: "Aggiornamento",
      mode: "Modalità",
      modeLive: "In tempo reale",
      modeInterval: "Ogni 3 secondi",
      modeManual: "Manuale",
      refreshNow: "Aggiorna ora",
      privacyNote: "In questa vista le credenziali sono mascherate. Nel file su disco non lo sono.",
      copyVisible: "Copia il log visibile",
      empty: "Non è stato ancora registrato nulla.",
      disabled: "La registrazione su file è disattivata (UNSLOTH_STUDIO_NO_FILE_LOG=1).",
      missing: "Nessun file di log trovato.",
      unreadable: "Non è stato possibile leggere il file di log.",
      timeout: "La richiesta del log e scaduta. Il server potrebbe non essere raggiungibile.",
      droppedNotice: "Alcune righe sono state saltate: il log è stato scritto più velocemente di quanto potesse essere letto.",
      morePending: "Altre righe sono ancora in lettura; arriveranno al prossimo aggiornamento.",
      staleSession: "La registrazione su file e disattivata, quindi questa e una sessione precedente e non verra aggiornata.",
      keywords: "debug log logs registro registri errore errori crash traccia diagnostica risoluzione dei problemi",
    },
    voice: {
      title: "Voce",
      description:
        "Microfono, dettatura, riconoscimento vocale e lettura ad alta voce",
      dictation: {
        sectionTitle: "Dettatura",
        engineLabel: "Motore di dettatura",
        engineBrowser: "Browser",
        engineBrowserDescription:
          "Trascrive l'audio con il servizio vocale del browser. Seleziona «Trascrizione locale» per usare un modello STT.",
        engineModel: "Trascrizione locale",
        engineModelDescription:
          "Esegue un modello speech-to-text (STT) in locale e funziona offline. Puoi scaricare e caricare il modello, che verrà poi rimosso dalla memoria dopo un periodo di inattività.",
        engineCustom: "Endpoint personalizzato",
        engineCustomDescription:
          "Invia l'audio registrato a un server STT compatibile con OpenAI dalle tue connessioni.",
        connectionLabel: "Connessione",
        connectionDescription:
          "Aggiungi un server compatibile con OpenAI e una chiave API facoltativa in Connessioni.",
        connectionPlaceholder: "Seleziona una connessione",
        connectionEmpty: "Nessuna connessione disponibile",
        customModelLabel: "Modello",
        customModelDescription:
          "Nome del modello inviato a /v1/audio/transcriptions.",
        sttModelLabel: "Modello di riconoscimento vocale",
        sttModelDescription:
          "Scegli o cerca un modello STT da eseguire in locale.",
        sttModelSearchPlaceholder: "Cerca un modello",
        sttModelSearching: "Ricerca su Hugging Face…",
        sttModelValidating: "Verifica della compatibilità con Whisper…",
        sttModelNoResults: "Nessun modello Whisper trovato",
        sttModelInvalid:
          "Questo repository non può essere usato per la dettatura",
        sttModelFailed: "Impossibile caricare il modello STT",
        sttModelUnsupported:
          "La registrazione non è supportata in questo browser",
        sttChecking: "Verifica…",
        sttOnDemand: "Scaricato",
        sttLoadingModel: "Caricamento del modello…",
        sttReady: "Caricato su {device}",
        sttLoaded: "Caricato",
        sttUnavailable:
          "Non installato su questo server. Esegui `unsloth studio update` per abilitare la dettatura locale.",
        sttRetry: "Riprova",
        sttDownloadChecking: "Verifica dello stato del download…",
        sttNotDownloaded: "Non scaricato",
        sttDownloadStatusFailed: "Impossibile verificare lo stato del download",
        sttDownload: "Scarica",
        sttDownloadConfirmTitle: "Vuoi scaricare {model}?",
        sttDownloadConfirmBody:
          "La dettatura locale funziona completamente offline, ma prima ha bisogno del modello vocale {model}. Occupa circa {size} e viene scaricato una sola volta nella tua cache di Hugging Face.",
        sttDownloadConfirmBodyUnsized:
          "La dettatura locale funziona completamente offline, ma prima ha bisogno del modello vocale {model}. Viene scaricato una sola volta nella tua cache di Hugging Face.",
        sttOpenVoiceSettings: "Apri le impostazioni Voce",
        sttDownloadStarted: "Download di {model} in corso",
        sttDownloading: "Download in corso… {progress}%",
        sttCancelDownload: "Annulla",
        sttCancellingDownload: "Annullamento…",
        sttCancelDownloadFailed: "Impossibile annullare il download",
        sttDownloadComplete: "Modello di riconoscimento vocale scaricato",
        sttModelReady: "{model} è pronto per la dettatura",
        sttRecommended: "Consigliato",
        sttDownloadFailed:
          "Impossibile scaricare il modello di riconoscimento vocale",
        sttLoad: "Carica",
        sttUnload: "Rimuovi dalla memoria",
        sttUnloading: "Rimozione dalla memoria…",
        microphoneLabel: "Microfono",
        microphoneFallbackName: "Microfono {index}",
        microphoneDescription: "Usato per la dettatura",
        microphoneFallbackHint:
          "Usato per la dettatura. Se il motore vocale del browser non può usare questo dispositivo, si passa a quello predefinito di sistema",
        microphoneGrantDescription:
          "Consenti l'accesso al microfono per vedere i nomi dei dispositivi",
        allowMicrophone: "Consenti l'accesso al microfono",
        micAccessBlocked:
          "L'accesso al microfono è stato bloccato. Consenti l'accesso al microfono per questa pagina di Unsloth, poi riprova.",
        micAccessBlockedDesktop:
          "L'accesso al microfono è stato bloccato. Riprova e scegli Consenti, oppure abilita il microfono nelle impostazioni di privacy del sistema.",
        micAccessUnsupported:
          "L'accesso al microfono non è supportato in questo browser o contesto.",
        systemDefault: "Predefinito di sistema",
        savedMicDisconnected: "Microfono salvato (non collegato)",
        languageLabel: "Lingua di dettatura",
        languageDescription: "Lingua da riconoscere",
        languageAuto: "Automatica (lingua del browser)",
        languageAutoDetect: "Automatica (rileva lingua)",
      },
      dictionary: {
        sectionTitle: "Dizionario di dettatura",
        sectionDescription:
          "Definisci come la dettatura scrive parole o frasi specifiche",
        manageLabel: "Grafie personalizzate",
        manage: "Gestisci",
        backToVoice: "Torna a Voce",
        addEntry: "Aggiungi voce",
        newEntryAria: "Nuova voce del dizionario",
        entryPlaceholder: "Maria Rossi",
        entryAria: "Voce del dizionario {index}",
        removeEntryAria: "Rimuovi la voce del dizionario {index}",
      },
      recents: {
        sectionTitle: "Cronologia delle dettature",
        sectionDescription:
          "Ogni dettatura viene salvata qui, così puoi recuperare il testo",
        manageLabel: "Cronologia delle dettature",
        manage: "Gestisci",
        pageDescription:
          "Ogni dettatura viene salvata. Puoi visualizzarla, copiarla, eliminarla o aprire la chat in cui è stata usata.",
        searchPlaceholder: "Cerca nelle dettature",
        sortLabel: "Ordina le dettature",
        sortNewest: "Più recenti",
        sortOldest: "Più vecchie",
        sortAlpha: "Dalla A alla Z",
        noMatches: "Nessuna dettatura corrisponde alla ricerca",
        detailTitle: "Dettatura salvata",
        backToVoice: "Torna a Voce",
        backToRecents: "Torna alle dettature recenti",
        view: "Vedi la dettatura completa",
        empty: "Ancora nessuna dettatura",
        dictationColumn: "Dettatura",
        dateColumn: "Data di creazione",
        copy: "Copia la dettatura",
        copied: "Copiato negli appunti",
        copyFailed: "Impossibile copiare negli appunti",
        delete: "Elimina la dettatura",
        deleteTitle: "Elimina la dettatura",
        deleteDescription:
          "Vuoi eliminare questa dettatura salvata? L'operazione è irreversibile.",
        deleteLinkedDescription:
          "Vuoi eliminare questa dettatura salvata? Puoi eliminare anche la chat in cui è stata usata. L'operazione è irreversibile.",
        deleteWithChat: "Elimina chat e dettatura",
        deleteWithChatFailed: "Impossibile eliminare la chat",
        clear: "Svuota la cronologia",
        clearTitle: "Svuota la cronologia delle dettature",
        clearDescription:
          "Vuoi eliminare tutte le dettature salvate? L'operazione è irreversibile.",
        clearConfirm: "Elimina tutto",
        showMore: "Mostra altre ({count})",
        openChat: "Apri la chat",
      },
      readAloud: {
        sectionTitle: "Lettura ad alta voce",
        buttonLabel: "Pulsante di lettura ad alta voce",
        buttonDescription: "Mostralo nelle risposte dell'assistente",
        engineLabel: "Motore TTS",
        engineSystemDescription: "Voci integrate nel dispositivo",
        engineStudioDescription: "Usa il modello audio caricato (es. Orpheus)",
        engineSystem: "Voci di sistema",
        engineStudio: "Carica un modello TTS",
        engineCustom: "Endpoint personalizzato",
        engineCustomDescription:
          "Un server TTS compatibile con OpenAI dalle tue connessioni (ad es. Kokoro)",
        connectionLabel: "Connessione",
        connectionDescription:
          "Aggiungi un server compatibile con OpenAI nella scheda Connessioni",
        connectionPlaceholder: "Seleziona una connessione",
        customModelLabel: "Modello",
        customVoiceDescription:
          "Nome della voce atteso dall'endpoint; il valore predefinito è alloy",
        modelLabel: "Modello TTS",
        modelDescription:
          "Carica un modello audio dal selettore dei modelli (es. Orpheus TTS)",
        openAudioAction: "Apri Audio",
        voiceLabel: "Voce",
        voiceDescription: "Le migliori voci disponibili su questo dispositivo",
        speedLabel: "Velocità",
        pitchLabel: "Tono",
        volumeLabel: "Volume",
        previewLabel: "Ascolta la voce",
        previewDescription: "Riproduci un breve campione",
        previewAction: "Ascolta",
        preparingAction: "Generazione…",
        previewFailed: "Impossibile riprodurre l'anteprima TTS",
        stopAction: "Interrompi",
        ttsLabel: "Sintesi vocale",
        notSupported: "Non supportato in questo browser",
      },
    },
    general: {
      title: "Generali",
      description: "Preferenze globali di Unsloth.",
      account: "Account",
      huggingFaceToken: "Token Hugging Face",
      huggingFaceTokenDescription:
        "Usato per caricare modelli ad accesso limitato e pubblicare artefatti.",
      hideToken: "Nascondi il token",
      showToken: "Mostra il token",
      clearToken: "Cancella",
      checkingToken: "Verifica del token...",
      tokenValidated: "Token convalidato",
      password: "Password",
      passwordDescription: "Cambia la password di questo account Unsloth.",
      passwordDialog: {
        trigger: "Cambia password",
        title: "Cambia password",
        description:
          "Inserisci la password attuale e scegline una nuova (almeno {minLength} caratteri).",
        setTrigger: "Imposta la password remota",
        setTitle: "Imposta la password remota",
        setDescription:
          "Scegli la password con cui i browser remoti accedono come unsloth (almeno {minLength} caratteri). L'app desktop di Unsloth continua ad accedere automaticamente.",
        setSubmit: "Imposta la password",
        setting: "Impostazione...",
        setDone: "Password impostata.",
        currentPassword: "Password attuale",
        newPassword: "Nuova password",
        confirmPassword: "Conferma la nuova password",
        currentTooShort:
          "La password attuale deve avere almeno {minLength} caratteri.",
        newTooShort:
          "La nuova password deve avere almeno {minLength} caratteri.",
        newHasSpaces: "La nuova password non può contenere spazi.",
        mismatch: "Le password non corrispondono.",
        samePassword:
          "La nuova password deve essere diversa da quella attuale.",
        update: "Aggiorna la password",
        updating: "Aggiornamento...",
        updated: "Password aggiornata.",
        updateFailed: "Aggiornamento della password non riuscito.",
      },
      chatDefaults: "Impostazioni predefinite della chat",
      autoTitleNewChats: "Titola automaticamente le nuove chat",
      autoTitleNewChatsDescription:
        "Genera un titolo breve a partire dal primo messaggio.",
      helperLlm: {
        sectionTitle: "LLM di supporto",
        preloadOnStartup: "Scarica in anticipo l'LLM di supporto all'avvio",
        preloadOnStartupDescription:
          "Scarica in background il modello di supporto di AI Assist all'avvio. Disattivato per impostazione predefinita: AI Assist può comunque scaricarlo quando serve.",
        disabledByEnv:
          "Disattivato da UNSLOTH_HELPER_MODEL_DISABLE nell'ambiente del backend.",
        loadError:
          "Caricamento delle impostazioni dell'LLM di supporto non riuscito.",
        saveError:
          "Salvataggio delle impostazioni dell'LLM di supporto non riuscito.",
      },
      modelAutoSwitch: {
        sectionTitle: "Cambio automatico del modello (API OpenAI)",
        enable: "Cambia modello in base alla richiesta",
        enableDescription:
          "Carica un GGUF già scaricato indicato in una richiesta API prima di rispondere. Disattivato per impostazione predefinita.",
        autoDownload: "Scarica i modelli mancanti",
        autoDownloadDescription:
          "Scarica un GGUF indicato in una richiesta API se non è ancora presente. Chiunque abbia una chiave API potrà così consumare spazio su disco e banda.",
        idleUnload: "Scaricamento automatico dalla memoria per inattività",
        idleUnloadDescription:
          "Libera la VRAM dopo il numero indicato di secondi di inattività. 0 mantiene il modello in memoria; il minimo è 60.",
        idleSecondsAriaLabel:
          "Secondi di inattività prima dello scaricamento automatico",
        mediaEnable: "Cambia modello di immagini e video in base alla richiesta",
        mediaEnableDescription:
          "Carica un modello di immagini o video già scaricato indicato in una richiesta API prima di generare. È un'impostazione a sé: quella sopra riguarda solo il modello di chat. Disattivato per impostazione predefinita.",
        mediaIdleUnload:
          "Scaricamento automatico dalla memoria per inattività di immagini e video",
        mediaIdleUnloadDescription:
          "Libera la VRAM scaricando i modelli di immagini e video dopo il numero indicato di secondi di inattività. È un'impostazione a sé: quella qui sopra riguarda solo il modello di chat. 0 li mantiene in memoria; il minimo è 60.",
        mediaIdleSecondsAriaLabel:
          "Secondi di inattività prima dello scaricamento automatico di immagini e video",
        mediaIdlePaused:
          "In pausa finché «Mantieni il modello nella memoria della GPU» è attivo.",
        idleNeedsEnable:
          "Attiva prima «Cambia modello in base alla richiesta».",
        idleActiveViaEnv: "Attivo tramite UNSLOTH_MODEL_IDLE_TTL.",
        loadError:
          "Caricamento delle impostazioni di cambio automatico del modello non riuscito.",
        saveError:
          "Salvataggio delle impostazioni di cambio automatico del modello non riuscito.",
        idleError:
          "Inserisci 0 per mantenere il modello in memoria, oppure almeno 60 secondi.",
        keepKv:
          "Mantieni il contesto della chat dopo lo scaricamento dalla memoria per inattività",
        keepKvDescription:
          "Salva la cache KV prima dello scaricamento dalla memoria per inattività, così le chat riprese non rileggono la cronologia. Fino a 10 GB su disco.",
        apiOnly: "Scarica solo i modelli caricati dall'API",
        apiOnlyDescription:
          "Lo scaricamento per inattività lascia in memoria un modello caricato da te in Unsloth e libera solo quelli caricati da una richiesta API.",
      },
      previewSharing: {
        sectionTitle: "Condivisione delle anteprime",
        enableLabel: "Link di anteprima pubblici",
        enableDescription:
          "Permetti a chiunque abbia un link firmato di chattare con un modello addestrato, senza dover effettuare l'accesso. Disattiva l'opzione per rendere inaccessibile l'anteprima pubblica: i link condivisi smetteranno di funzionare.",
        loadError:
          "Caricamento delle impostazioni di condivisione delle anteprime non riuscito.",
        saveError:
          "Salvataggio delle impostazioni di condivisione delle anteprime non riuscito.",
        revokeLabel: "Revoca tutti i link di anteprima",
        revokeDescription:
          "Rigenera il segreto di firma, così tutti i link già condivisi smettono di funzionare. I nuovi link copiati da quel momento in poi continuano a funzionare.",
        revokeAction: "Revoca i link",
        revoking: "Revoca in corso...",
        revokeConfirmTitle: "Revocare tutti i link di anteprima?",
        revokeConfirmDescription:
          "Ogni link di anteprima che hai condiviso smetterà subito di funzionare. L'operazione è irreversibile.",
        revokeConfirmAction: "Revoca tutti i link",
        revoked: "Tutti i link di anteprima sono stati revocati",
        revokeError: "Impossibile revocare i link di anteprima",
      },
      permissions: {
        sectionTitle: "Autorizzazioni",
        bypassLabel: "Autorizzazioni degli strumenti",
        bypassDescription:
          "Come Unsloth approva le chiamate agli strumenti della chat (terminale, python, web, MCP) prima che vengano eseguite. La modalità «Full access» disattiva le approvazioni e la sandbox del codice.",
      },
      notifications: {
        sectionTitle: "Notifiche",
        showLlamaUpdates: "Notifiche di aggiornamento di llama.cpp",
        showLlamaUpdatesDescription:
          "Avvisa quando è disponibile una build più recente di llama.cpp per eseguire nuovi modelli. Disattiva le notifiche se usi Unsloth solo per l'addestramento.",
        showLoadedModels: "Indicatore dei modelli caricati",
        showLoadedModelsDescription:
          "Mostra una piccola scheda in basso a destra con tutti i modelli attualmente in memoria (chat, voce, immagini, video), con un pulsante per rimuovere ciascuno.",
      },
      startup: {
        sectionTitle: "Avvio",
        launchAtLogin: "Avvia Unsloth all'accesso",
        launchAtLoginDescription:
          "Avvia Unsloth in background quando accedi. Rimane nella barra dei menu o nell'area di notifica finché non lo apri.",

        closeToTray: "Chiudi nell'area di notifica",
        closeToTrayDescription:
          "Mantieni Unsloth e il suo server in esecuzione in background quando chiudi la finestra principale.",
        closeToTraySaveError:
          "Impossibile aggiornare l'impostazione di chiusura nell'area di notifica.",
        loadError: "Impossibile caricare l'impostazione di avvio all'accesso.",
        saveError:
          "Impossibile aggiornare l'impostazione di avvio all'accesso.",
      },
      downloads: {
        sectionTitle: "Download",
        transport: "Trasporto di download",
        transportDescription:
          "Come arrivano i file di modelli e dataset da Hugging Face. HTTPS riprende da dove si è fermato; Xet è spesso più veloce al primo download ma ricomincia il file se annulli.",
        transportHint:
          "HTTPS è normale TLS: ogni rete, proxy e VPN lo consente, un trasferimento annullato o interrotto riprende dai byte già su disco e l'uso di memoria resta costante. Xet scarica blocchi deduplicati, quindi un repository che condivide dati con uno che hai già può arrivare molto più rapidamente, ma richiede hf_xet, usa più RAM e un annullamento scarta il file in corso. Auto decide per macchina: valuta la RAM e se Xet si è bloccato qui, e ripiega su HTTPS.",
        https: "HTTPS",
        xet: "Xet",
        auto: "Auto",
        httpsHint:
          "TLS standard. Riprende dopo un annullamento, funziona su qualsiasi rete, memoria costante.",
        transportDescriptionNoResume:
          "Come i file di modelli e dataset vengono scaricati da Hugging Face. In questa installazione nessun trasporto può riprendere, quindi un download annullato ricomincia; Xet è spesso più veloce al primo download.",
        httpsHintNoResume:
          "TLS standard. Funziona con qualsiasi rete, uso di memoria costante. Questa installazione non può riprendere un download annullato.",
        xetHint:
          "Trasferimento a blocchi deduplicati. Spesso più veloce su un download nuovo, ricomincia il file se annulli, richiede più memoria.",
        autoHint:
          "Sceglie in base alla macchina e passa a HTTPS se Xet si blocca o fallisce qui.",
        autoCurrently: "Auto sta usando {transport} su questa macchina.",
        xetMissing: "Xet non è disponibile perché hf_xet non è installato.",
      },
      uploads: {
        sectionTitle: "Caricamenti",
        maxUploadSize: "Limite di caricamento del dataset di addestramento",
        maxUploadSizeDescription: "Il valore predefinito è {defaultSize} MB.",
      },
      rag: {
        sectionTitle: "Documenti e RAG",
        embeddingModel: "Modello di embedding",
        embeddingModelDescription:
          "Modello Hugging Face o percorso locale usato per indicizzare e cercare nei tuoi documenti. Il valore predefinito è {defaultModel}.",
        searchPlaceholder: "Cerca qualsiasi modello su HF",
        reindexWarning:
          "Vale solo per i documenti indicizzati da ora in poi. Ricarica quelli esistenti dopo aver cambiato modello.",
        emptyError:
          "Inserisci l'ID di un modello Hugging Face o un percorso locale.",
        loadError:
          "Caricamento dell'impostazione del modello di embedding non riuscito.",
        saveError: "Salvataggio del modello di embedding non riuscito.",
        saved: "Modello di embedding salvato.",
        saveAnyway: "Salva comunque",
        recommended: "Consigliato",
        onDevice: "Sul dispositivo",
        searching: "Ricerca su Hugging Face…",
        checking: "Verifica…",
        noResults: "Nessun modello di embedding trovato",
        download: "Scarica",
        unload: "Scarica dalla memoria",
        unloadFailed: "Impossibile scaricare il modello di embedding",
        downloadingStatus: "Download in corso…",
        notDownloaded: "Non scaricato",
        notDownloadedSized: "Non scaricato · {size}",
        loaded: "Caricato",
        downloading: "Download di {model}",
        downloadingDescription:
          "L'avanzamento è nel pannello dei download. L'indicizzazione lo userà al termine.",
        downloadFailed: "Impossibile avviare il download",
        downloadConflict: "Riprendi questo download dall'Hub",
        downloadBusy: "Download già in corso",
      },
      storage: {
        sectionTitle: "Archiviazione",
        modelsFolder: "Cartella dei modelli",
        modelsFolderDescription: "Dove vengono salvati i modelli scaricati.",
        openAction: "Apri",
        copyAction: "Copia il percorso",
        copied: "Percorso copiato",
        openError: "Impossibile aprire la cartella",
        copyError: "Impossibile copiare il percorso",
      },
      resetPreferences: {
        sectionTitle: "Zona pericolosa",
        label: "Reimposta tutte le preferenze locali",
        description:
          "Cancella solo le preferenze locali. Le chat, l'accesso API e le impostazioni salvate nel database restano invariati.",
        action: "Reimposta le preferenze",
        confirmTitle: "Reimpostare tutte le preferenze locali?",
        confirmDescription:
          "Cancella solo le preferenze locali e ricarica Unsloth. Le chat, l'accesso API e le impostazioni salvate nel database restano invariati.",
        confirmAction: "Reimposta e ricarica",
      },
    },
    profile: {
      title: "Profilo",
      description: "Come appare il tuo profilo in Unsloth.",
      changePicture: "Cambia immagine del profilo",
      uploadPhoto: "Carica una foto",
      removePhoto: "Rimuovi",
      pictureOptions: "Opzioni dell'immagine del profilo",
      displayName: "Nome visualizzato",
      nickname: "Come deve chiamarti Unsloth?",
      nicknamePlaceholder: "Soprannome",
      nicknameSaved: "Nome preferito salvato",
      avatarShape: "Forma dell'avatar",
      avatarShapeCircle: "Circolare",
      avatarShapeRounded: "Arrotondata",
      greetingSloth: "Bradipo nel saluto",
      greetingSlothDescription: "Mostra il bradipo nel saluto della chat.",
      chooseSloth: "Oppure scegli un bradipo",
      noPicture: "Nessuna immagine del profilo",
      noneLabel: "Nessuna",
      nameSaved: "Nome del profilo salvato",
      namePersistErrorTitle: "Impossibile salvare il nome del profilo",
      namePersistErrorDescription:
        "Il nome è stato aggiornato per questa sessione, ma potrebbe non essere mantenuto dopo il ricaricamento.",
      photoUpdated: "Foto del profilo aggiornata",
      photoPersistErrorTitle: "Impossibile salvare la foto del profilo",
      photoPersistErrorDescription:
        "La foto è stata aggiornata per questa sessione, ma potrebbe non essere mantenuta dopo il ricaricamento.",
      photoUpdateErrorTitle: "Impossibile aggiornare la foto del profilo",
      imageUseError: "Impossibile usare questa immagine.",
      stats: {
        title: "Le tue statistiche",
        subtitle:
          "Tutto quello che vedi qui è calcolato dalla tua cronologia. Nessun dato viene raccolto o inviato a Unsloth.",
        retry: "Riprova",
        privacyNote:
          "Le statistiche sono calcolate dalla cronologia locale di chat, utilizzo API e addestramenti della tua installazione di Unsloth. Prompt, risposte e chiavi API non vengono mai salvati per le statistiche. Nulla viene inviato a Unsloth o a terze parti.",
        emptyChats:
          "Nessun utilizzo della chat o dell'API. Inizia una conversazione o invia una richiesta autenticata all'API locale.",
        lifetimeTokens: "Token totali",
        peakTokens: "Giorno record",
        longestChat: "Chat più lunga",
        currentStreak: "Serie attuale",
        longestStreak: "Serie più lunga",
        activityTitle: "Attività in token",
        activityDescription: "Periodo: {weeks} · {total}",
        mode: {
          daily: "Giornaliera",
          weekly: "Settimanale",
          cumulative: "Cumulativa",
        },
        cellTooltip: "{date} · {tokens}, {messages}",
        weekTooltip: "Settimana del {date} · {tokens}",
        less: "Meno",
        more: "Più",
        insightsTitle: "Analisi dell'attività",
        totalChats: "Chat totali",
        totalMessages: "Messaggi totali",
        tokensIn: "Token inviati",
        tokensOut: "Token generati",
        totalTokens: "Token totali",
        studioChatTokens: "Token di Unsloth Chat",
        apiTokens: "Token API",
        cachedTokens: "Token in cache",
        cachedValue: "{tokens} ({percent}% dell'input)",
        avgTokensPerChat: "Token medi per chat",
        timeInChat: "Tempo in chat",
        activeDays: "Giorni attivi",
        toolCalls: "Chiamate agli strumenti",
        attachments: "File allegati",
        avgSpeed: "Velocità media",
        bestSpeed: "Risposta più rapida",
        firstToken: "Tempo medio al primo token",
        tokensPerSecond: "{value} tok/s",
        topModelsTitle: "Modelli più usati",
        topModelsDescription: "Ordinati per token scambiati",
        modelSummary: "{tokens} · {messages}",
        noModels: "Nessun utilizzo di modelli registrato.",
        trainingTitle: "Addestramento",
        trainingDescription: "Run di fine-tuning di questo spazio di lavoro",
        trainingRuns: "Run",
        trainingCompleted: "Completati",
        trainingSteps: "Step",
        trainingTokens: "Token usati nell'addestramento",
        trainingTime: "Tempo di addestramento",
        bestLoss: "Loss minima",
        runSteps: "{steps}",
        runLoss: "loss {loss}",
      },
    },
    appearance: {
      title: "Aspetto",
      description: "Come si presenta Unsloth su questo dispositivo.",
      theme: {
        title: "Tema",
        label: "Combinazione di colori",
        description: "Chiaro, scuro o in base alle impostazioni di sistema.",
        system: "Sistema",
        light: "Chiaro",
        dark: "Scuro",
      },
      palette: {
        label: "Tavolozza dei colori",
        description: "Colori usati in Unsloth, nei temi chiaro e scuro.",
        standard: "Standard",
        classic: "Classica",
        minimal: "Minimale",
      },
      custom: {
        reset: "Ripristina",
        resetAll: "Ripristina la personalizzazione",
        preferencesTitle: "Preferenze",
        colors: {
          lightGroup: "Tema chiaro",
          darkGroup: "Tema scuro",
          accent: "Accento",
          background: "Sfondo",
          foreground: "Primo piano",
        },
        fontDefault: "Predefinito",
        fontBundledGroup: "Integrati",
        fontImportedGroup: "Importati",
        fontDeviceGroup: "Su questo dispositivo",
        fontFolderGroup: "Da cartella",
        fontDeviceLoading: "Ricerca dei font del dispositivo…",
        fontSearch: "Cerca font…",
        fontNoResults: "Nessun font trovato.",
        colorPicker: {
          hue: "Tonalità",
          hex: "Colore esadecimale",
          eyedropper: "Preleva un colore dallo schermo",
        },
        uiFont: {
          label: "Font dell'interfaccia",
        },
        headingFont: {
          label: "Font dei titoli",
        },
        chatFont: {
          label: "Font della chat",
        },
        codeFont: {
          label: "Font del codice",
        },
        importFont: {
          upload: "Carica",
          scanFolder: "Seleziona una cartella",
          alreadyAvailable:
            "Questo font è già disponibile, quindi viene usata la copia esistente.",
          folderNoFonts: "Nessun file di font trovato in quella cartella.",
          remove: "Rimuovi",
          errorInvalidType:
            "Tipo di file non supportato. Usa .woff2, .woff, .ttf o .otf.",
          errorTooLarge: "Il file del font è troppo grande (max 1,5 MB).",
          errorLimit: "Puoi importare al massimo 3 font.",
          errorStorageFull:
            "Spazio di archiviazione locale insufficiente per questo font. Rimuovi prima un font importato.",
          errorFailed: "Impossibile caricare questo file di font.",
        },
        uiFontSize: {
          label: "Dimensione del font dell'interfaccia",
          description:
            "Regola la dimensione di base usata nell'interfaccia di Unsloth.",
        },
        codeFontSize: {
          label: "Dimensione del font del codice",
          description: "Regola la dimensione di base usata per il codice.",
        },
        fontSmoothing: {
          label: "Antialiasing dei caratteri",
          description:
            "Usa l'antialiasing per rendere più uniformi i caratteri.",
        },
        contrast: {
          label: "Contrasto",
          description: "Intensità dei bordi e del testo secondario.",
        },
        reduceMotion: {
          label: "Riduci le animazioni",
          description: "Riduci le animazioni o segui il sistema.",
          system: "Sistema",
          on: "Attivo",
          off: "Disattivo",
        },
        pointerCursors: {
          label: "Usa il cursore a mano",
          description:
            "Cambia il cursore in una mano quando passi sopra gli elementi interattivi.",
        },
      },
      language: {
        title: "Lingua",
        label: "Lingua dell'interfaccia",
        description: "La lingua usata da Unsloth.",
        autoDetect: "Rilevamento automatico",
      },
      layout: {
        title: "Layout",
        compactSidebar: "Fissa la barra laterale per impostazione predefinita",
        compactSidebarDescription:
          "Mantieni la barra laterale espansa invece di ridurla a icone.",
      },
      sidebarNav: {
        title: "Navigazione della barra laterale",
        description:
          "Fissa e riordina le schede della barra laterale. Le schede non fissate vengono raccolte nel menu «Altro»; se ne resta una sola non fissata viene nascosta invece di creare un menu con una voce sola. «Nuova chat» resta sempre al suo posto.",
        dragToReorder: "Trascina per riordinare",
        pinToSidebar: "Fissa {name} nella barra laterale",
        moreHolds: "Altro ({count})",
      },
      sidebarMenu: {
        title: "Menu della barra laterale",
        description:
          "Mostra, nascondi e riordina le voci del menu del profilo nella barra laterale. Impostazioni, Aiuto, Esci e Arresta restano fisse.",
        darkModeToggle: "Interruttore del tema scuro",
        dragToReorder: "Trascina per riordinare",
      },
    },
    resources: {
      title: "Sistema",
      description:
        "Monitora hardware e archiviazione di questo server Unsloth.",
      liveUpdates: "Aggiornamenti in tempo reale",
      floatingWindow: "Finestra mobile",
      disableOverlay: "Disattiva la sovrapposizione",
      liveMonitor: {
        title: "Monitor in tempo reale",
        apiTitle: "Monitor API",
        summary: "Richieste attive, errori e utilizzo dei token",
        status: "{active} attive · {recent} recenti · {model}",
        noModelLoaded: "nessun modello caricato",
        autoOpen: "Mostra automaticamente il monitor fluttuante",
        autoOpenDescription:
          "Apre un piccolo pannello quando arriva traffico API.",
        cpu: "CPU",
        ram: "RAM",
        disk: "Disco",
        vram: "VRAM",
        cpuCores: "Core logici: {logical} / fisici: {physical}",
        currentLoad: "Carico attuale",
        free: "Disponibili: {value}",
        noGpu: "Nessuna GPU visibile",
      },
      gpu: {
        title: "Dispositivi GPU",
        ggufInference: "Inferenza GGUF",
        unavailable: "non disponibile",
        detecting: "Ricerca di GPU...",
        unreadable: "Impossibile leggere l'hardware di questo server.",
        noGpu:
          "Nessuna GPU visibile rilevata. Sopra sono mostrate le risorse della sola CPU.",
        unknownDevice: "GPU sconosciuta",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "In uso: {value}",
        free: "Disponibili: {value}",
        total: "Totale: {value}",
      },
      llamaBackend: {
        title: "Motore di inferenza GGUF",
        label: "Backend di calcolo",
        description: "Il backend che llama.cpp usa per eseguire i modelli GGUF.",
        runningOn: "llama.cpp è in esecuzione su {backend}.",
        hint: "Installa la build di llama.cpp per questo backend e la mantiene tra un aggiornamento e l'altro. Utile se la scelta automatica va in crash o se il driver della GPU non la supporta. Sono elencati solo i backend con una build per questa macchina; l'addestramento non cambia.",
        autoWith: "Automatico ({backend})",
        apply: "Applica",
        applying: "Installazione...",
        applyHint: "Scarica la nuova build e riavvia llama.cpp. Il modello caricato verrà scaricato.",
        applyHintWithSize: "Scarica {size} e riavvia llama.cpp. Il modello caricato verrà scaricato.",
        switchedTo: "Ora llama.cpp è in esecuzione su {backend}.",
        switchFailed: "Impossibile cambiare il backend di llama.cpp.",
        switchInterrupted: "Il cambio è stato interrotto prima del completamento.",
        envLocked: "Impostato su {backend} dalla variabile d'ambiente UNSLOTH_LLAMA_CPP_BACKEND, che ha la precedenza su questa opzione.",
        customPath: {
          label: "Cartella llama.cpp personalizzata",
          description: "Usa la tua build di llama-server.",
          hint: "Scegli la cartella llama.cpp che contiene llama-server o una build in cui si trova sotto build/bin. Il runtime personalizzato viene usato per chat GGUF, embedding e modelli vocali supportati. Le variabili d'ambiente hanno comunque la priorità.",
          automatic: "Automatico (incluso)",
          bundled: "Usa il runtime llama.cpp installato da Unsloth.",
          active: "Il tuo llama-server personalizzato verrà usato al prossimo caricamento del modello.",
          environmentManaged: "Gestito dalla variabile d'ambiente {variable}.",
          missingBinary: "llama-server non è più disponibile in questa cartella. Scegli un'altra cartella o usa il runtime incluso.",
          reloadRequired: "Ricarica il modello per usare il llama-server selezionato.",
          change: "Cambia",
          saving: "Salvataggio...",
          useBundled: "Usa quello incluso",
          chooseTitle: "Scegli la cartella llama.cpp",
          chooseAction: "Usa questa cartella",
          saved: "Cartella llama.cpp aggiornata",
          saveError: "Impossibile aggiornare la cartella llama.cpp",
        },
        backends: {
          auto: "Automatico",
          cpu: "CPU",
          cuda: "CUDA",
          rocm: "ROCm",
          vulkan: "Vulkan",
          metal: "Metal",
        },
        unsupported: {
          notInstalled: "Nessuna installazione di llama.cpp gestita da Unsloth, quindi non c'è un backend da cambiare.",
          localLink: "llama.cpp è una cartella locale collegata da te, quindi Unsloth non la sostituirà.",
          sourceBuild: "Questo llama.cpp è stato compilato dai sorgenti, quindi il backend non è modificabile da qui.",
          customPath: "È selezionata una cartella llama.cpp personalizzata. La sua build determina il backend di calcolo.",
          unresolved: "Non è stato possibile controllare i backend disponibili. Verifica la connessione e riprova.",
        },
        // Non visualizzato: termini aggiuntivi per la ricerca nelle impostazioni.
        llamaBackendKeywords:
          "llama.cpp backend gguf inferenza cuda rocm hip vulkan metal cpu gpu acceleratore prebuilt cambiare motore",
      },
      modelMemory: {
        title: "Memoria del modello",
        keepResident: "Mantieni il modello nella memoria della GPU",
        keepResidentDescription: "Resta nella VRAM tra un prompt e l'altro.",
        keepResidentHint: "Non restituisce i pesi alla RAM di sistema finché il modello resta caricato. Disattiva lo scaricamento automatico in inattività e, quando i pesi risiedono davvero nella RAM host (memoria unificata o offload parziale sulla GPU), passa anche --mlock, così il sistema operativo non li pagina per ricaricarli al prompt successivo.",
        noRamReserve: "Non riservare RAM di sistema per il modello",
        noRamReserveDescription: "Non tiene una copia completa in RAM.",
        noRamReserveHint: "Trasferisce i pesi nella VRAM invece di tenerne una copia completa in RAM. Mantiene il caricamento mappato in memoria di llama.cpp e rimuove --no-mmap e --mlock.",
        mlockVetoed: "--mlock resta disattivato: bloccare il modello riserverebbe RAM per l'intero modello. Lo scaricamento automatico in inattività resta disattivato.",
        memlockCapped: "Questo sistema limita la memoria bloccata a {limit}. Un modello più grande non verrà bloccato del tutto; aumenta il limite con ulimit -l.",
        reloadRequired: "Ricarica il modello per applicare le nuove opzioni di memoria.",
        loadError: "Impossibile caricare le impostazioni di memoria del modello",
        saveError: "Impossibile salvare le impostazioni di memoria del modello",
        // Not rendered: extra terms the settings search matches these rows on.
        modelMemoryKeywords:
          "mlock memlock ulimit vram gpu memoria ram residente bloccare fissare mantenere caricato scaricare inattivo mmap no-mmap load-mode paginazione swap",
      },
      storage: {
        title: "Archiviazione",
        systemDisk: "Disco di sistema",
        diskUsage: "In uso: {used} / Totale: {total}",
        diskFree: "Disponibili: {free}",
        modelsFolder: "Cartella dei modelli",
        modelsFolderDescription: "Dove vengono salvati i modelli scaricati.",
        modelsFolderHint: "Dove vengono salvati i modelli scaricati. Cambialo per tenere i modelli fuori dall'unità di sistema. Vale solo per i nuovi download: i modelli che hai già restano dove sono.",
        // Non visualizzato: termini extra su cui la ricerca delle impostazioni trova questa riga.
        modelsFolderKeywords:
          "cartella modelli directory percorso posizione download scaricati cache archiviazione disco unità spostare sposta hugging face models folder path storage",
        futureDownloads: "Solo i nuovi download",
        environmentManaged: "Gestita dalla variabile d'ambiente {variable}.",
        locationFree: "Disponibili: {free}",
        openAction: "Apri",
        copyAction: "Copia il percorso",
        changeAction: "Cambia",
        resetAction: "Usa il valore predefinito",
        chooseTitle: "Scegli la posizione di download dei modelli",
        chooseAction: "Usa per i download futuri",
        cacheSaved: "Posizione di download dei modelli aggiornata",
        cacheSaveError:
          "Impossibile aggiornare la posizione di download dei modelli",
        cachePickerError: "Impossibile aprire il selettore di cartelle",
        copied: "Percorso copiato",
        openError: "Impossibile aprire la cartella",
        copyError: "Impossibile copiare il percorso",
      },
      environment: {
        title: "Ambiente",
        backend: "Backend",
        python: "Python",
        torch: "Torch",
        transformers: "Transformers",
        uptime: "Tempo di attività",
        processMemory: "Memoria del processo",
        notInstalled: "Non installato",
        unknown: "Sconosciuto",
        vramWithShared: "{vram} di VRAM + {shared} di memoria condivisa",
      },
    },
    agents: {
      title: "Agenti",
      description:
        "Collega agenti di programmazione come Claude Code e Codex a un modello locale con unsloth start.",
      intro:
        "collega Claude Code, Codex, Hermes, OpenClaw, OpenCode e altri agenti a un modello servito localmente da Unsloth, completamente offline. Avvia un server compatibile con le API OpenAI e non modifica i file di configurazione del tuo agente.",
      readDocs: "Leggi la documentazione",
      copy: "Copia",
      copied: "Copiato",
      commandBuilder: "Generatore di comandi",
      agent: "Agente di programmazione",
      model: "Modello",
      searchModels: "Cerca modelli GGUF...",
      noModels: "Nessun modello GGUF corrispondente.",
      showingModels:
        "Risultati mostrati: {shown} su {total}. Continua a digitare per restringere l'elenco.",
      quantization: "Quantizzazione",
      loadingQuantizations: "Caricamento delle quantizzazioni...",
      noQuantizations: "Nessuna quantizzazione separata",
      recommended: "Consigliato",
      downloaded: "Scaricato",
      quantizationLoadError:
        "Impossibile caricare tutte le quantizzazioni. Il comando userà il valore del modello disponibile.",
      generatedCommand: "Comando generato",
      docs: "Documentazione",
      agentDocs: "Apri la documentazione di configurazione di {agent}",
      copyGeneratedCommand: "Copia il comando generato",
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
        "Codex richiede un modello GGUF servito da llama-server. Gli altri agenti possono usare anche modelli basati su transformer; rimuovi --model per usare il modello già caricato in Unsloth.",
      subagent: {
        title: "Usa un modello locale come subagente",
        description:
          "Mantieni {agent} sul suo modello attuale e delega alcune attività a questo modello Unsloth locale.",
        setupCommand: "Comando di configurazione",
        copySetupCommand: "Copia il comando di configurazione del subagente",
        usagePrompt: "Poi, in {agent}, digita:",
        copyUsagePrompt: "Copia il prompt d'uso del subagente",
        defaultPrompt:
          "Avvia un agente locale per implementare questa funzione.",
        opencodePrompt: "@unsloth trova la causa di questo test che non passa",
      },
      quickstart: {
        title: "Costruisci un comando",
        description:
          "Avvia un agente sul modello attualmente caricato in Unsloth. Carica prima un modello, poi sostituisci claude con uno degli agenti supportati qui sotto.",
        noneDetected: "Nessuna CLI di agenti supportati trovata nel tuo PATH.",
        installed: "Installato",
      },
      supportedAgents: {
        title: "Agenti supportati",
        description: "Ogni agente si avvia con il proprio comando:",
        requiresGguf: "Richiede un modello GGUF",
      },
      models: {
        title: "Scegliere un modello",
        description:
          "Usa --model per scegliere un modello e una quantizzazione, e --context-length per impostare la finestra di contesto. Puoi usare un suffisso di quantizzazione oppure il flag esplicito --gguf-variant.",
        suffixLabel: "Con un suffisso di quantizzazione",
        variantLabel: "Con un flag di variante esplicito",
      },
      options: {
        title: "Opzioni comuni",
        description:
          "I flag di Unsloth vengono interpretati per primi; tutto ciò che Unsloth non riconosce viene passato direttamente all'agente.",
        model:
          "Seleziona un modello. Senza --model, unsloth start usa il modello attualmente caricato in Unsloth e restituisce un errore se non ce n'è nessuno.",
        contextLength:
          "Imposta la lunghezza di contesto richiesta (alias: --max-seq-length).",
        ggufVariant: "Scegli la variante di quantizzazione GGUF.",
        loadIn4bit:
          "Attiva o disattiva il caricamento a 4 bit per i modelli Hugging Face.",
        tensorParallel:
          "Attiva o disattiva il parallelismo tensoriale su più GPU.",
        serve: "Attiva o disattiva il server locale automatico.",
        launch:
          "Avvia l'agente oppure mostra soltanto il comando e le variabili d'ambiente.",
        persist:
          "Mantieni tra un'esecuzione e l'altra i dati dell'agente gestiti da Unsloth.",
        asSubagent:
          "Mantieni l'agente principale sul suo modello attuale e registra Unsloth come subagente locale (Claude Code, Codex e OpenCode).",
        apiKey:
          "Fornisci la tua chiave API Unsloth (oppure imposta UNSLOTH_API_KEY).",
        reasoning:
          "Usare il ragionamento nella chat: on, off o auto. Auto segue il chat template del modello, che di solito significa on.",
        reasoningEffort:
          "Impegno di ragionamento passato al chat template del modello, ad esempio medium. I livelli dipendono dal modello, quindi usane uno che accetta. Se non impostato resta quello del template.",
        yolo: "Salta le richieste di approvazione. Usa solo in ambienti fidati.",
      },
      remote: {
        title: "Connettersi a un Unsloth Studio remoto",
        description:
          "Punta unsloth start a un Unsloth Studio in esecuzione altrove impostando queste variabili prima dell'avvio (oppure passa direttamente --api-key):",
      },
      passthrough: {
        title: "Passare argomenti all'agente",
        description:
          "Gli argomenti che seguono i flag di Unsloth vengono inoltrati all'agente, così i comandi nativi come resume continuano a funzionare:",
      },
      dryRun: {
        title: "Anteprima senza avviare",
        description:
          "Aggiungi --no-launch per mostrare le variabili d'ambiente e il comando invece di avviare l'agente. Se --model è impostato, il modello potrebbe comunque essere individuato e caricato.",
      },
    },
    chat: {
      projectsSection: "Mostra la sezione Progetti",
      projectsSectionDescription:
        "Raggruppa le chat di progetto sotto un titolo Progetti. Disattivalo per elencarle in Recenti.",
      title: "Chat",
      description:
        "Personalizza il comportamento della chat su questo dispositivo.",
      modelSelection: {
        title: "Impostazioni di selezione del modello",
        expandQuantizations: "Espandi le quantizzazioni",
        expandQuantizationsDescription:
          "Attivata: i modelli GGUF in «On Device» mostrano subito le relative quantizzazioni. Disattivata: fai clic su un modello per visualizzarne le quantizzazioni.",
        showAllQuantizations: "Mostra tutte le quantizzazioni",
        showAllQuantizationsDescription:
          "Attivata: elenca tutte le quantizzazioni in «On Device», incluse quelle non scaricate. Disattivata: mostra solo le quantizzazioni scaricate.",
        showMemoryBar: "Mostra la barra di utilizzo della VRAM",
        showMemoryBarDescription:
          "Mostra sotto la riga di ogni modello scaricato il consumo stimato di VRAM: pesi, cache KV alla lunghezza di contesto con cui verrà caricato ed eventuale riserva per il draft speculativo.",
      },
      menu: {
        title: "Menu della chat",
        description:
          "Fissa le voci nel menu laterale «+» della chat. Le altre verranno spostate in «Altro».",
        chatWithFiles: "Chat con file (RAG)",
        mcp: "MCP",
        savedPrompts: "Prompt salvati",
        compareChat: "Confronta chat",
        exportChat: "Esporta chat",
      },
      pastedTextThreshold: "Condensa incolla lunghi",
      pastedTextThresholdDescription: "Il testo incollato più lungo di questo valore diventa un allegato .txt invece di riempire il campo del messaggio. Premi {shortcut} per incollare comunque nel campo del messaggio.",
      pastedTextThresholdOff: "Disattivato",
      showResponseModel: "Mostra il modello della risposta",
      showResponseModelDescription:
        "Mostra i metadati del modello nelle risposte dell'assistente.",
      modelDisclaimer: "Mostra l'avviso sul modello",
      modelDisclaimerDescription:
        "Mostra «Gli LLM possono commettere errori» sotto il campo della chat.",
      projectAttachments: "Condividi i file nel progetto",
      projectAttachmentsDescription:
        "Impostazione predefinita per i file allegati in una chat che appartiene a un progetto: indicizzarli per l'intero progetto in modo che ogni chat possa usarli. Ogni chat puo cambiarlo dal menu degli allegati.",
      rememberParamsPerModel: "Ricorda le impostazioni per modello",
      rememberParamsPerModelDescription:
        "Cambiando modello vengono ripristinati temperatura, prompt e le altre impostazioni usate l'ultima volta con quel modello. Disattivato, resta un unico set di impostazioni per tutti i modelli.",
      autoCompact: "Compatta automaticamente le chat lunghe",
      autoCompactDescription:
        "Quando una chat GGUF locale raggiunge la lunghezza di contesto impostata, elimina i turni precedenti invece di restituire un errore. Questa impostazione non dipende dalla VRAM libera.",
      compactionStyle: "Quando il contesto è pieno",
      compactionStyleDescription:
        "Il valore predefinito del server mantiene UNSLOTH_CONTEXT_POLICY. Reimpostare la conversazione conserva l'ultimo turno e le istruzioni permanenti. Una finestra scorrevole elimina i turni più vecchi e può conservare più cronologia recente.",
      compactionStyleInherit: "Usa il valore del server",
      compactionStyleCheckpoint: "Reimposta la conversazione",
      compactionStyleRollingDefault:
        "Elimina i turni precedenti (~25% di spazio aggiuntivo)",
      compactionStyleRolling10:
        "Elimina i turni precedenti (~10% di spazio aggiuntivo)",
      compactionStyleRolling5:
        "Elimina i turni precedenti (~5% di spazio aggiuntivo)",
      compactionStyleRollingNone:
        "Elimina i turni precedenti (nessun taglio aggiuntivo)",
      autoCompactKeywords:
        "compattazione automatica contesto finestra troncare scorrevole checkpoint margine compaction rolling headroom",
      thinking: {
        collapseByDefault: "Comprimi il ragionamento per impostazione predefinita",
        collapseByDefaultDescription:
          "Mantieni il ragionamento compresso mentre il modello pensa, invece di aprirlo automaticamente. Espandi un blocco per leggerlo.",
      },
      tools: {
        collapseByDefault: "Comprimi l’attività degli strumenti per impostazione predefinita",
        collapseByDefaultDescription:
          "Mantieni compressi input e output degli strumenti durante l’esecuzione. Espandi una riga per esaminarla.",
      },
      webSearch: {
        title: "Ricerca web",
        images: "Mostra immagini dalla ricerca web",
        imagesDescription:
          "Consente alla ricerca web di restituire immagini e ne recupera una per ogni elemento elencato in una risposta. Le miniature vengono scaricate e ridimensionate da Unsloth, quindi il browser non contatta mai gli host delle immagini.",
      },
      artifacts: {
        title: "Canvas",
        collapseHtmlBlocks: "Comprimi i blocchi HTML",
        collapseHtmlBlocksDescription:
          "La modalità Canvas comprime automaticamente l'HTML completo. Attiva questa opzione per comprimere anche i documenti HTML in blocchi di codice quando Canvas è disattivato.",
        allowNetworkAccess: "Consenti a Canvas di accedere alla rete",
        allowNetworkAccessDescription:
          "Consenti alle anteprime di Canvas di caricare script, stili, font, contenuti multimediali e risorse di rete dalle CDN. Lascia l'opzione disattivata per anteprime completamente offline.",
        blockedBanner: "Bloccata {count} risorsa esterna da {hosts}.",
        blockedBannerPlural: "Bloccate {count} risorse esterne da {hosts}.",
        blockedBannerAction: "Consenti per questo Canvas",
      },
      data: "Dati",
      exportHistory: "Esporta la cronologia delle chat",
      exportHistoryDescription: "Scarica tutte le chat e i messaggi in JSON.",
      exportAction: "Esporta",
      exportingAction: "Esportazione...",
      exportConversations: "Esporta Recenti e Progetti",
      exportConversationsDescription:
        "Scarica i Recenti, oppure i Recenti più le chat dei progetti, in Training JSONL, CSV o JSONL ShareGPT, in un unico file o uno per chat. Message JSONL è disponibile solo per singola chat.",
      exportConversationsAction: "Esporta",
      exportScopeRecents: "Recenti",
      exportScopeAll: "Recenti + Progetti",
      exportCombinedSuffix: "(file unico)",
      exportPerChatSuffix: "(uno per chat)",
      importChats: "Importa chat",
      importChatsDescription:
        "Importa un export Open WebUI, JSONL, NDJSON o CSV in Recenti.",
      importChatsAction: "Importa",
      importNoConversations: "Nessuna conversazione trovata nel file.",
      importedOneChat: "1 conversazione importata nei Recenti.",
      importedChatCount: "{count} conversazioni importate nei Recenti.",
      importingChats: "Importazione chat: {count} finora ({percent}%)...",
      importedChatCountPartial: "Importate {count} conversazioni in Recenti; {failed} non sono state salvate.",
      importFailed: "Importazione non riuscita.",
      clearHistory: "Cancella la cronologia delle chat",
      clearHistoryDescription:
        "Elimina la cronologia delle chat da questo dispositivo.",
      clearAction: "Cancella",
      clearAllChats: "Cancella tutte le chat",
      clearAllChatsDescription:
        "Elimina definitivamente ogni chat su questo dispositivo.",
      noChatsToClear: "Nessuna chat da cancellare.",
      clearOneChatDescription:
        "Elimina definitivamente l'unica chat presente su questo dispositivo.",
      clearChatCountDescription:
        "Elimina definitivamente tutte le {count} chat su questo dispositivo.",
      clearChatsAction: "Cancella le chat",
      clearOneChatTitle: "Cancellare 1 chat?",
      clearChatsTitle: "Cancellare {count} chat?",
      clearChatsConfirmDescription:
        "Elimina definitivamente ogni chat su questo dispositivo. L'operazione è irreversibile.",
      clearingAction: "Cancellazione...",
      clearOneChatAction: "Cancella 1 chat",
      clearChatCountAction: "Cancella {count} chat",
      clearedAllChats: "Tutte le chat sono state cancellate",
      clearedOneChat: "1 chat cancellata",
      clearedChatCount: "{count} chat cancellate",
      someChatsCouldNotBeCleared: "Alcune chat non sono state cancellate",
      chatsClearedRemainOne:
        "{clearedCount} chat cancellate; 1 chat rimane. Riprova.",
      chatsClearedRemain:
        "{clearedCount} chat cancellate; {remainingCount} chat rimangono. Riprova.",
      oneChatClearedRemain:
        "1 chat cancellata; {remainingCount} chat rimangono. Riprova.",
      oneChatClearedRemainOne: "1 chat cancellata; 1 chat rimane. Riprova.",
      storageClearFailedOne:
        "Un'operazione di cancellazione non è riuscita; 1 chat potrebbe rimanere. Riprova.",
      storageClearFailed:
        "Un'operazione di cancellazione non è riuscita; {count} chat potrebbero rimanere. Riprova.",
      failedToClearChats: "Cancellazione delle chat non riuscita",
    },
    data: {
      title: "Dati",
      backToData: "Torna a Dati",
      exportFailed: "Impossibile esportare le chat",
      description:
        "Gestisci la cronologia delle chat e i file caricati su questo dispositivo.",
      archivedChats: "Chat archiviate",
      archivedChatsDescription:
        "Visualizza e gestisci le chat che hai archiviato.",
      archivedImages: "Immagini archiviate",
      archivedImagesDescription: "Visualizza e gestisci le immagini che hai archiviato.",
      archivedVideos: "Video archiviati",
      archivedVideosDescription: "Visualizza e gestisci i video che hai archiviato.",
      manageAction: "Gestisci",
      manageChats: "Gestisci chat",
      manageChatsDescription:
        "Seleziona più chat da spostare, fissare, archiviare, esportare o eliminare.",
      exportArchivedChats: "Esporta",
      exportingArchivedChats: "Esportazione...",
      exportedOneArchivedChat: "1 chat archiviata esportata",
      exportedArchivedChatCount: "{count} chat archiviate esportate",
      noArchivedChatsToExport: "Nessuna chat archiviata da esportare.",
      failedToExportArchivedChats:
        "Esportazione delle chat archiviate non riuscita",
      archiveAllChats: "Archivia tutte le chat",
      archiveAllChatsDescription:
        "Sposta nell'archivio ogni chat presente in Recenti e Progetti.",
      noChatsToArchive: "Nessuna chat da archiviare.",
      archiveAllAction: "Archivia tutte",
      archivingAction: "Archiviazione...",
      archiveAllChatsTitle: "Archiviare tutte le chat?",
      archiveAllChatsConfirmDescription:
        "Sposta nell'archivio ogni chat su questo dispositivo. Le chat archiviate restano disponibili e possono essere ripristinate in qualsiasi momento.",
      archivedAllChats: "Tutte le chat sono state archiviate",
      archivedOneChat: "1 chat archiviata",
      archivedChatCount: "{count} chat archiviate",
      failedToArchiveChats: "Archiviazione delle chat non riuscita",
      confirmBeforeDeleting: "Chiedi conferma prima di eliminare",
      confirmBeforeDeletingDescription:
        "Richiedi una conferma prima di eliminare una chat. Disattiva l'opzione per eliminarla subito.",
      alwaysDeleteFiles: "Elimina sempre i file",
      alwaysDeleteFilesDescription:
        "L'eliminazione di una chat rimuove dal disco anche la sua cartella sandbox. I file scritti dentro un progetto restano nello spazio di lavoro di quel progetto.",
      filesSection: "File",
      uploadedFiles: "File caricati",
      uploadedFilesDescription:
        "Visualizza e gestisci i file caricati in chat, progetti e basi di conoscenza.",
      fineTuneExport: "Usa le chat come dati di addestramento",
      fineTuneExportDescription:
        "Crea un dataset JSONL di fine-tuning dalle tue chat. Caricalo in Addestra, perfezionalo in Ricette oppure esportalo.",
      fineTuneExportAction: "Esporta JSONL",
      fineTuneRunAction: "Esegui",
      fineTuneExportingAction: "Esportazione...",
      fineTuneOpenRecipesAction: "Apri in Ricette",
      fineTuneOpeningRecipesAction: "Apertura...",
      fineTuneTrainAction: "Carica nella scheda Addestra",
      fineTuneTrainingAction: "Caricamento...",
      fineTuneExportFailed:
        "Esportazione dei dati di addestramento non riuscita",
      fineTuneRecipeFailed: "Apertura delle chat in Ricette non riuscita",
      fineTuneTrainFailed:
        "Caricamento del dataset nella scheda Addestra non riuscito",
    },
    connections: {
      title: "Connessioni",
      description: "Gestisci provider e connessioni esterne.",
    },
    remoteLan: {
      title: "Remoto e LAN",
      description:
        "Raggiungi questo Unsloth dagli altri tuoi dispositivi, tramite la rete locale o un URL pubblico temporaneo.",
    },
    apiKeys: {
      title: "API",
      description: "Accedi a Unsloth tramite le API compatibili con OpenAI.",
      readDocs: "Leggi la documentazione delle API",
      noAccess: "Nessun accesso API configurato.",
      accessTokens: "Token di accesso",
      loadError: "Impossibile caricare gli accessi API.",
      createError: "Impossibile creare il token di accesso.",
      revokeError: "Impossibile revocare il token di accesso.",
      never: "Mai",
      tokenNamePlaceholder: "Nome del token (es. produzione)",
      newAccessTokenName: "Nome del nuovo token di accesso",
      createToken: "Crea token",
      creating: "Creazione...",
      newTokenCreated: "Nuovo token di accesso creato",
      accessTokenCopied: "Token di accesso copiato",
      copyAccessToken: "Copia il token di accesso",
      copyNow: "Copialo ora: non verrà mostrato di nuovo.",
      usageExamples: "Esempi d'uso",
      usageNoModel:
        "Carica o scarica un modello per vedere esempi eseguibili. Su questo server non è ancora disponibile alcun modello da usare negli esempi.",
      usageTools: "Strumenti",
      exampleCurlTools: "curl + strumenti",
      examplePythonTools: "Python + strumenti",
      exampleJavaScriptTools: "JavaScript + strumenti",
      exampleCurlAdvanced: "curl + avanzato",
      examplePythonAdvanced: "Python + avanzato",
      exampleJavaScriptAdvanced: "JavaScript + avanzato",
      osUnix: "Linux / macOS / WSL",
      osWindows: "Windows",
      secureHttps: "HTTPS sicuro",
      secureHttpsHint:
        "La porta su 0.0.0.0 resta raggiungibile da ovunque. Per la massima sicurezza avvia Unsloth con --secure, così viene esposto solo questo link HTTPS.",
      copyTunnelUrl: "Copia l'URL del tunnel",
      copySnippet: "Copia lo snippet",
      copy: "Copia",
      copied: "Copiato",
      setupDocs: "Documentazione di configurazione:",
      codingAgents: "Agenti di programmazione",
      codingAgentsHint:
        "Avvia un agente di programmazione configurato per usare questo server. Usa il modello caricato; un server locale genera automaticamente una chiave API, mentre uno remoto la include nel comando.",
      codingAgentsSwap:
        "Sostituisci claude con codex, openclaw, opencode o hermes.",
      codingAgentDetected: "Installato su questo computer",
      codingAgentsDetectedHint: "Rilevati su questo computer: {agents}.",
      relativeNever: "mai",
      relativeJustNow: "proprio ora",
      expired: "scaduta",
      today: "oggi",
      created: "Creato {value}",
      used: "Ultimo utilizzo: {value}",
      expires: "Scadenza: {value}",
      actionsFor: "Azioni per {name}",
      copyPrefix: "Copia il prefisso",
      revokeToken: "Revoca il token",
      revokeTitle: "Revocare il token di accesso «{name}»?",
      revokeDescription:
        "Le app che usano questo token perdono immediatamente l'accesso. L'operazione è irreversibile.",
      revokeAction: "Revoca «{name}»",
      revoking: "Revoca in corso...",
    },
    about: {
      title: "Informazioni",
      description:
        "Documentazione, note di rilascio, feedback e dettagli della build.",
      studioVersion: "Versione di Unsloth",
      desktopAppVersion: "Versione dell'app desktop",
      desktopAppVersionUnavailable: "Non disponibile",
      packageVersion: "Versione del pacchetto",
      llamaCppVersion: "Versione di llama.cpp",
      hardware: "Hardware",
      gpu: "GPU",
      cuda: "CUDA",
      rocm: "ROCm",
      xpu: "XPU",
      updates: "Aggiornamento",
      help: "Aiuto",
      documentation: "Documentazione",
      releaseNotes: "Note di rilascio",
      whatsNew: "Novità",
      feedback: "Feedback",
      reportIssue: "Segnala un problema",
      license: {
        sectionTitle: "Licenza",
        studioLabel: "Unsloth",
        studioLicense: "AGPL-3.0",
        studioDescription: "Open source con licenza GNU AGPL v3.0.",
        libraryLabel: "Unsloth Core",
        libraryLicense: "Apache-2.0",
        libraryDescription: "Distribuito con licenza Apache 2.0.",
      },
      dangerZone: "Zona pericolosa",
      shutDownStudio: "Arresta Unsloth",
      shutDownStudioDescription:
        "Ferma il server Unsloth e termina la tua sessione.",
      shutDown: "Arresta",
      update: {
        title: "Aggiorna Unsloth",
        commandText: "Testo del {label}",
        copied: "Copiato",
        copyCommand: "Copia il comando",
        commandCopied: "{label} copiato",
        copyNamedCommand: "Copia {label}",
        checkingInstall:
          "Verifica della modalità di installazione di Unsloth...",
        installIntro: "Per installare o aggiornare Unsloth:",
        localUpdateHeading: "Aggiornamento locale",
        installCommandUnix: "Comando di installazione per macOS/Linux",
        installCommandWindows: "Comando di installazione per Windows",
        localInstallDetected:
          "Rilevata un'installazione locale. Aggiornala dal checkout originale per evitare di sostituirla con la versione disponibile su PyPI.",
        pullThenUpdate:
          "Scarica le ultime modifiche, poi esegui l'installer locale:",
        gitPullCommand: "comando git pull",
        localInstallerCommand: "comando dell'installer locale",
        sourceInstallDetected:
          "Rilevata un'installazione da sorgente o da pacchetto VCS. Reinstalla dal percorso locale originale o dall'URL Git.",
        repoCheckoutFallback:
          "Se hai ancora il checkout del repository, esegui l'installer locale da lì:",
        restartAfterUpdate: "Riavvia Unsloth dopo l'aggiornamento.",
        desktopManaged:
          "L'app desktop verifica automaticamente la presenza di nuove versioni. Puoi anche controllare o aggiornare qui in qualsiasi momento.",
        desktopReady: "Aggiornamenti dell'app desktop",
        desktopReadyDescription:
          "Verifica se è disponibile una versione più recente dell'app desktop.",
        desktopChecking: "Verifica degli aggiornamenti",
        desktopCheckingDescription:
          "Questa operazione richiede in genere pochi secondi.",
        desktopAvailable:
          "È disponibile la versione {version} dell'app desktop",
        desktopAvailableDescription:
          "Aggiorna ora: al termine, l'app desktop verrà riavviata.",
        desktopExternalServer:
          "Esegui `unsloth studio update` nel terminale da cui hai avviato il server.",
        desktopManualInstall:
          "Apri la pagina della release per installare il pacchetto Linux più recente.",
        desktopCheckFailed:
          "Impossibile verificare la disponibilità di aggiornamenti",
        desktopCheckFailedDescription: "Controlla la connessione e riprova.",
        desktopCurrent: "L'app desktop è aggiornata",
        desktopCurrentDescription:
          "Unsloth continuerà a verificare automaticamente la disponibilità di aggiornamenti.",
        checkForUpdates: "Verifica aggiornamenti",
        checkAgain: "Verifica di nuovo",
        retryCheck: "Riprova",
        checking: "Verifica in corso...",
        updateNow: "Aggiorna ora",
        openReleasePage: "Apri la pagina della release",
        unknownInstall:
          "Impossibile rilevare come è stato installato Unsloth. Per installazioni tramite installer o PyPI, usa i comandi sopra.",
        localCheckout:
          "Per le installazioni da checkout locale, esegui l'installer locale da quel checkout:",
        docs: "Documentazione di installazione:",
        docsInstall: "Installazione",
        docsUpdating: "Aggiornamento",
        docsMac: "Mac",
        docsWindows: "Windows",
      },
    },
  },
  picker: {
    onDevice: "Sul dispositivo",
    huggingFace: "Hugging Face",
    retry: "Riprova",
    loadMore: "Carica altri",
    offlineTitle: "Sei offline",
    offlineBody:
      "Passa a Dispositivo per usare {noun} nella cache o in locale.",
    offlineSwitchDevice: "Dispositivo",
    searchAriaLabel: "Cerca {noun}",
    modelSourceAriaLabel: "Origine del modello",
    hubSectionAriaLabel: "Sezione Hub",
    modelDropped: "Non più offerto",
    modelDroppedByProvider: "{provider} · non più offerto",
    modelDisabled: "Non attivato",
    modelDisabledByProvider: "{provider} · non attivato",
    multipleMatches:
      "Sono stati trovati più {noun} corrispondenti. Scegline uno dall'elenco.",
    rateLimitedTitle: "Limite di richieste di Hugging Face raggiunto",
    rateLimitedBody: "Attendi un momento, poi riprova a cercare {noun}.",
    hfToken: {
      label: "Token HF",
      saved: "Salvato",
      add: "Non impostato",
      savedAriaLabel: "Token Hugging Face salvato",
      addAriaLabel: "Imposta il token Hugging Face",
      savedHint:
        "Token salvato. L'accesso viene verificato quando lo utilizzi.",
      addHint:
        "Imposta un token per accedere ai repository privati e con accesso limitato.",
    },
  },
  studio: {
    imageTraining: "Addestramento immagini",
    goToImageTraining: "Vai all'addestramento immagini",
    routeTitle: "Addestra",
    wizard: {
      modelTitle: "Modello",
      modelDescription: "Seleziona il modello e il metodo di addestramento",
      datasetTitle: "Dataset",
      datasetDescription: "Seleziona o carica i dati di addestramento",
      paramsTitle: "Parametri",
      paramsDescription: "Configura i parametri di addestramento",
      configTitle: "Configurazione",
      configDescription: "Salva e carica le configurazioni",
      modelLabel: "Modello",
      modelTooltip: "Il modello di base che vuoi sottoporre a fine-tuning.",
      methodLabel: "Metodo",
      methodTooltip:
        "Il modo in cui viene addestrato il modello. LoRA e QLoRA aggiornano piccoli adattatori anziché tutti i pesi.",
      datasetLabel: "Dataset",
      datasetTooltip:
        "I dati di addestramento usati per il fine-tuning del modello.",
      hfTokenDescription:
        "Necessario per modelli e dataset privati o con accesso limitato.",
      uploadLocalLabel: "Oppure carica un file locale",
      sourceBrowse: "Sfoglia",
      releaseToUpload: "Rilascia per caricare",
      loadYaml: "Carica YAML",
      saveYaml: "Salva YAML",
      resetDefaults: "Ripristina i valori predefiniti",
      cachedModelGoneTitle: "Il modello nella cache non è più disponibile",
      cachedModelGoneDescription:
        "I file del modello non sono più sul dispositivo, quindi l'addestramento li scaricherà di nuovo.",
      cachedDatasetGoneTitle: "Il dataset nella cache non è più disponibile",
      cachedDatasetGoneDescription:
        "I file del dataset non sono più sul dispositivo, quindi l'addestramento li scaricherà di nuovo.",
    },
    preview: {
      title: "Anteprima del run",
      ready: "Pronto",
      notReady: "Non pronto",
      modelPending: "Modello in attesa",
      datasetPending: "Dataset in attesa",
      method: "Metodo",
      length: "Durata",
      stepZero: "{count} step",
      step: "{count} step",
      stepTwo: "{count} step",
      stepFew: "{count} step",
      stepMany: "{count} step",
      steps: "{count} step",
      epochZero: "{count} epoche",
      epoch: "{count} epoca",
      epochTwo: "{count} epoche",
      epochFew: "{count} epoche",
      epochMany: "{count} epoche",
      epochs: "{count} epoche",
      batch: "Batch",
      context: "Contesto",
      lr: "LR",
      hardware: "Hardware",
      noGpu: "Nessuna GPU rilevata",
      hfToken: "Token HF",
      saved: "Salvato",
      notSet: "Non impostato",
      files: "File",
      model: "Modello",
      dataset: "Dataset",
      downloadsOnStart: "Download all'avvio",
      continuesOnStart: "Continua all'avvio",
      noticeModelDownload:
        "Questo modello non è ancora presente sul dispositivo. L'addestramento lo scaricherà automaticamente.",
      noticeModelPartial:
        "L'addestramento completerà il download parziale del modello prima di caricarlo.",
      noticeDatasetDownload:
        "Questo dataset non è ancora presente sul dispositivo. L'addestramento lo scaricherà automaticamente.",
      noticeDatasetPartial:
        "L'addestramento completerà il download parziale del dataset prima di leggerlo.",
      noticeTransformersUpgrade:
        "Nessuna versione installata di transformers supporta ancora questa architettura. All'avvio verrà proposto di installare prima transformers {version}.",
      noticeSixteenBitOnly:
        "Questa architettura si addestra in LoRA a 16 bit: i 4 bit non sono disponibili, quindi l'esecuzione richiede molta più VRAM di QLoRA.",
      noticeInstallSwitchesSixteenBit:
        "Installare quella versione invece di mantenere il codice del modello porta questa esecuzione a LoRA a 16 bit, che richiede molta più VRAM di QLoRA.",
      advancedSettings: "Impostazioni avanzate",
      defaultAdvancedSettings: "Predefinite",
      nonDefaultAdvancedSettings: "{count} non predefinite",
    },
    datasetPicker: {
      noun: "dataset",
      selectDataset: "Seleziona un dataset",
      hubPlaceholder: "Cerca dataset su Hugging Face...",
      devicePlaceholder: "Cerca dataset locali...",
      useAsHubDataset: "Usa come dataset Hugging Face",
      hfCacheLabel: "Cache HF",
      scanningLocal: "Analisi dei dataset sul dispositivo…",
      couldntScan: "Impossibile analizzare i dataset locali",
      someLocationsUnscanned:
        "Non è stato possibile analizzare alcune posizioni dei dataset.",
      noLocalDatasets:
        "Non c'è ancora nulla sul dispositivo. Scarica un dataset dall'Hub, creane uno in Ricette oppure carica un file.",
      openDataRecipes: "Apri le ricette per i dati",
      searchingHub: "Ricerca su Hugging Face…",
      noDatasetsFound: "Nessun dataset trovato.",
      tokenRejectedTitle: "Token Hugging Face rifiutato",
      tokenRejectedBody:
        "Aggiorna il token in Impostazioni → Generali, quindi riprova.",
      hubUnreachable: "Impossibile raggiungere Hugging Face",
      cantUseDataset: "Impossibile usare il dataset",
      reasonInvalidHubId:
        "Inserisci un ID dataset Hugging Face valido: repo oppure proprietario/repo, senza punti o trattini consecutivi e senza il suffisso .git (massimo 96 caratteri per parte).",
      sourceRecipe: "Ricetta",
      sourceUpload: "Caricamento",
      sourceLocal: "Locale",
    },
    modelPicker: {
      noun: "modelli",
      selectModel: "Seleziona un modello",
      hubPlaceholder: "Cerca o incolla un ID Hugging Face...",
      devicePlaceholder:
        "Cerca modelli locali o incolla il percorso di una cartella...",
      useAsHubModel: "Usa come modello Hugging Face",
      useAsLocalPath: "Usa come percorso locale",
      hfCacheLabel: "Cache HF",
      scanningLocal: "Analisi dei modelli locali…",
      couldntScan: "Impossibile analizzare i modelli locali",
      someLocationsUnscanned:
        "Non è stato possibile analizzare alcune posizioni locali.",
      noLocalModels: "Nessun modello locale trovato.",
      noLocalModelsHint:
        "Incolla qui sopra il percorso di una cartella oppure passa a Hugging Face.",
      searchingHub: "Ricerca su Hugging Face…",
      noModelsFound: "Nessun modello trovato.",
      tokenRejectedTitle: "Token Hugging Face rifiutato",
      tokenRejectedBody:
        "Aggiorna il token in Impostazioni → Generali, quindi riprova.",
      hubUnreachable: "Impossibile raggiungere Hugging Face",
      cantUseModel: "Impossibile usare il modello per l'addestramento",
      reasonTypeMismatch:
        "Questo modello non corrisponde al tipo di addestramento selezionato nel passaggio precedente.",
      reasonEmptyId: "Inserisci l'ID di un modello o un percorso locale.",
      reasonGguf:
        "I modelli GGUF non possono essere usati per l'addestramento.",
      reasonAdapter:
        "Gli output degli adattatori non possono essere usati come modelli di base per l'addestramento.",
      reasonNotTrainable:
        "Questo modello presente sul dispositivo non è addestrabile.",
      reasonUnsupportedFormat:
        "Questo formato di modello non è supportato per l'addestramento.",
      reasonInvalidHubId:
        "Inserisci un ID modello Hugging Face valido: repo oppure proprietario/repo, senza punti o trattini consecutivi e senza il suffisso .git (massimo 96 caratteri per parte).",
      sourceModelsFolder: "Cartella dei modelli",
      sourceHfCache: "Cache HF",
      sourceLmStudio: "LM Studio",
      sourceOllama: "Ollama",
      sourceCustomFolder: "Cartella personalizzata",
      sourceLocalModel: "Modello locale",
      vramOomBadge: "OOM",
      vramTightBadge: "Al limite",
      vramNeeds: "Richiede ~{est} GB di VRAM (GPU: {total} GiB)",
      vramTight: "~{est} GB di VRAM (al limite su {total} GiB)",
      vramApprox: "~{est} GB di VRAM",
    },
    methods: {
      qlora: {
        label: "QLoRA",
        hint: "Quantizzazione a 4 bit. VRAM minima e avvio più rapido.",
        note: "4 bit",
      },
      lora: {
        label: "LoRA",
        hint: "Adattatori a 16 bit. Equilibrio tra qualità e memoria.",
        note: "16 bit",
      },
      full: {
        label: "Fine-tuning completo",
        hint: "Addestra tutti i pesi. Qualità massima, ma richiede più VRAM.",
        note: "fp16",
      },
      cpt: {
        label: "Preaddestramento continuato",
        hint: "Preaddestramento continuato per nuovi domini o nuove lingue.",
        note: "continuato",
      },
    },
    subtitles: {
      configure: "Configura e avvia l'addestramento",
      trainingInProgress: "Addestramento in corso",
      viewPastRuns: "Consulta i run di addestramento passati",
      viewingPastRun: "Stai guardando un run passato",
    },
    tabs: {
      configure: "Configura",
      currentRun: "Run attuale",
      history: "Cronologia",
    },
    loadingRuntime: "Caricamento del runtime di addestramento...",
    checkingSupport: "Verifica del supporto all'addestramento su questa macchina...",
    backToHistory: "Torna alla cronologia",
    dataset: {
      selectors: {
        subset: "Sottoinsieme",
        subsetTooltip:
          "Seleziona il sottoinsieme (configurazione) del dataset da usare.",
        trainSplit: "Split di addestramento",
        trainSplitTooltip: "Seleziona lo split da usare per l'addestramento.",
        evaluationSplit: "Split di valutazione",
        evaluationSplitTooltip:
          "Seleziona lo split da usare per la valutazione. Nessuno significa che non verrà eseguita alcuna valutazione durante l'addestramento.",
        selectSubset: "Seleziona un sottoinsieme...",
        selectSplit: "Seleziona uno split...",
        none: "Nessuno",
        loading:
          "Caricamento delle configurazioni e degli split del dataset...",
        manualTitle: "Inserisci manualmente le opzioni del dataset",
        manualDescription:
          "Inserisci i nomi esatti della configurazione e degli split Hugging Face da usare.",
        manualSubsetPlaceholder: "Nome configurazione facoltativo",
        manualRequired: "È richiesto uno split di addestramento.",
        manualTooLong: "Usa al massimo 128 caratteri.",
        manualInvalid: "Questo valore contiene caratteri non supportati.",
      },
      sourceAriaLabel: "Origine del dataset",
      streamingInfoAriaLabel: "Informazioni sullo streaming del dataset",
      uploadDetails: "Dettagli del caricamento",
      uploadDetailsTooltip:
        "Fino a {limit} per file. PDF, DOCX e TXT non sono dataset pronti per l'addestramento: convertili prima nelle Ricette di apprendimento.",
      fileTooLarge: "File troppo grande",
      fileTooLargeDescription:
        "{file} occupa {size}. I caricamenti per l'addestramento supportano fino a {limit}.",
      documentRedirect: {
        title: "Questo file deve prima essere convertito",
        genericFile: "Questo file",
        description:
          "{file} è materiale di origine, non un dataset pronto per l'addestramento. Usa le Ricette per i dati per trasformare il documento in un dataset, quindi torna qui per eseguire il fine-tuning.",
        nextStepTitle: "Passaggio successivo consigliato",
        nextStepDescription:
          "Apri le Ricette di apprendimento e inizia con una ricetta basata su documenti, come le domande e risposte fondate su PDF.",
        openAction: "Apri le Ricette di apprendimento",
      },
      previewLoadingHuggingFace:
        "Recupero dell'anteprima del dataset da Hugging Face...",
      previewLoading: "Caricamento dell'anteprima...",
      mappingRequirements: {
        audioAndText: "audio e testo",
        imageAndText: "immagine e testo",
        instructionAndOutput: "istruzione e output",
        humanAndGpt: "umano e GPT",
        userAndAssistant: "utente e assistente",
      },
      mappingStatus: {
        heuristicTitle: "Mappatura rilevata con metodi euristici",
        readyTitle: "Mappatura pronta",
        requiredTitle: "Mappa le colonne del dataset",
        heuristicDescription:
          "Abbiamo rilevato automaticamente la mappatura delle colonne qui sotto usando metodi euristici. Controllala e modificala tramite i menu nelle intestazioni delle colonne oppure usa l'assistenza IA per una mappatura più precisa.",
        readyDescription:
          "È tutto corretto. Convertiremo automaticamente questo dataset.",
        requiredDescription:
          "Assegna i ruoli alle colonne tramite i menu nelle intestazioni. Assegna almeno {required}.",
      },
      localDataset: "Dataset locale",
      localDatasetRows: " / {count} righe",
      huggingFaceDataset: "Dataset Hugging Face",
      localDatasetMetadata: "Metadati del dataset locale",
      dataRecipeOutput: "Output di una ricetta.",
      rows: "Righe",
      columns: "Colonne",
      batches: "Batch",
      updated: "Aggiornato",
      evalDataset: "Dataset di valutazione",
      uploading: "Caricamento...",
      uploadEvalFile: "Carica il file di valutazione",
      evalDatasetDescription:
        "Facoltativo. Se non lo fornisci, una piccola parte verrà separata dai dati di addestramento.",
      advanced: "Avanzate",
      targetFormat: "Formato di destinazione",
      targetFormatTooltip:
        "Formato dei tuoi dati di addestramento. Il rilevamento automatico funziona con la maggior parte dei dataset.",
      auto: "Auto",
      rawText: "Testo grezzo",
      trainSplitStart: "Inizio dello split di addestramento",
      trainSplitStartTooltip:
        "Addestra solo su una parte del tuo split di addestramento indicando l'indice della riga iniziale (incluso, con numerazione a partire da 0). Lascia vuoto per partire dalla prima riga.",
      trainSplitEnd: "Fine dello split di addestramento",
      trainSplitEndTooltip:
        "Indice dell'ultima riga dello split di addestramento da includere (incluso, con numerazione a partire da 0). Per esempio, imposta Inizio su 0 e Fine su 99 per addestrare sulle prime 100 righe. Lascia vuoto per usare tutte le righe restanti.",
      endPlaceholder: "Fine",
      clear: "Deseleziona",
      dropFileOrClick: "Trascina qui 1 file oppure fai clic per caricarlo",
      viewDataset: "Vedi il dataset",
      uploadFailed: "Caricamento non riuscito",
      unknownError: "Errore sconosciuto",
      unsupportedFileType: "Tipo di file non supportato",
      uploadOneFileType: "Carica un file {types}.",
      datasetUploaded: "Dataset caricato",
      evalDatasetUploaded: "Dataset di valutazione caricato",
      uploadOneFileAtATime: "Carica un file alla volta",
      uploadSingleFileDescription:
        "Puoi caricare un solo file come dataset di addestramento.",
      streaming: {
        label: "Attiva lo streaming",
        description:
          "Esegui lo streaming dei dataset di testo di Hugging Face invece di scaricarli.",
        unavailable: "Streaming non disponibile. Per attivarlo:",
        completionsUnavailable:
          "Non disponibile mentre lo streaming del dataset è attivo.",
        blockers: {
          source:
            "Usa un dataset Hugging Face, non un caricamento locale o un'origine S3.",
          maxSteps:
            "Imposta Step massimi > 0: la lunghezza dei dataset in streaming non è nota.",
          trainOnCompletions: 'Disattiva "Solo completamenti dell’assistente".',
          evalSplit:
            "Scegli uno split di valutazione distinto: la valutazione è attiva, ma non è impostato uno split separato.",
          visionModel: "I modelli di visione non supportano lo streaming.",
          audioModel: "I modelli audio non supportano lo streaming.",
          embeddingModel:
            "I modelli di embedding non supportano lo streaming: l'addestramento richiede l'intero dataset.",
          imageDataset:
            "Questo dataset sembra contenere immagini, che non possono essere trasmesse in streaming.",
          audioDataset:
            "Questo dataset sembra contenere audio, che non può essere trasmesso in streaming.",
          appleSilicon:
            "Lo streaming non è ancora supportato su Apple Silicon (MLX).",
        },
        options: {
          trainOnCompletions: "solo completamenti dell'assistente",
          evaluation:
            "valutazione (richiede uno split di valutazione separato)",
        },
        notifications: {
          turnedOffMaxSteps:
            "Streaming disattivato: richiede un numero fisso di Step massimi > 0.",
          adjusted:
            "Impostazioni adattate per lo streaming. Opzioni incompatibili disattivate: {options}.",
          needsMaxSteps:
            "Lo streaming richiede un numero fisso di Step massimi perché la lunghezza dei dataset in streaming non è nota. Imposta prima Step massimi > 0.",
          enabledAdjusted:
            "Streaming attivato. Opzioni incompatibili disattivate: {options}.",
          disabledForDetectedModality:
            "Lo streaming è stato disattivato perché i dataset di immagini e audio richiedono un download completo. Controlla l'impostazione, quindi avvia di nuovo.",
        },
      },
      s3: {
        title: "Configurazione S3",
        description:
          "Carica dataset .parquet, .json, .jsonl o .csv da Amazon S3",
        bucket: "Nome del bucket",
        bucketPlaceholder: "mio-bucket-dati-addestramento",
        region: "Regione AWS",
        regionPlaceholder: "us-east-1",
        prefix: "Prefisso del percorso",
        prefixPlaceholder: "datasets/whisper/",
        accessKeyId: "ID chiave di accesso",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Chiave di accesso segreta",
        secretAccessKeyPlaceholder: "La tua chiave di accesso segreta AWS",
        useIamRole: "Usa un ruolo IAM",
      },
    },
    params: {
      mode: {
        simple: "Semplice",
        advanced: "Avanzata",
        ariaLabel: "Modalità dei parametri",
      },
      notSupportedAppleSilicon: "Non supportato su Apple Silicon",
      projectName: "Nome del progetto",
      optional: "Facoltativo",
      projectNameDescription:
        "Usato nei nomi delle cartelle di output, nei valori predefiniti di esportazione e nella cronologia.",
      loraSettings: "Impostazioni LoRA",
      trainingHyperparameters: "Iperparametri di addestramento",
      maxSteps: "Step massimi",
      epochs: "Epoche",
      useMaxSteps: "Usa gli step massimi",
      useEpochs: "Usa le epoche",
      maxStepsTooltip:
        "Sovrascrive il numero totale di step dell'ottimizzatore.",
      epochsTooltip: "Numero di passaggi completi sul dataset.",
      contextLength: "Lunghezza di contesto",
      contextLengthTooltip:
        "Numero massimo di token per campione di addestramento.",
      customContextLength: "Inserisci un valore personalizzato",
      learningRate: "Tasso di apprendimento",
      learningRateTooltip:
        "Ampiezza del passo negli aggiornamenti dei pesi. Con valori più bassi, l'addestramento è più lento ma più stabile.",
      learningRateDescription:
        "Consigliato: 2e-4 per LoRA, 5e-5 per CPT, 2e-5 per il fine-tuning completo",
      embeddingLearningRate: "Tasso di apprendimento degli embedding",
      embeddingLearningRateTooltip:
        "Usato solo quando il CPT addestra embed_tokens. Gli embedding si destabilizzano più facilmente dei pesi LoRA, quindi di solito richiedono un tasso di apprendimento più basso. Lascia vuoto per usare lr/10; in genere si usa un valore compreso tra lr/2 e lr/10. Aumentalo solo se l'adattamento del vocabolario o dei token specifici del dominio è troppo lento.",
      rank: "Rank",
      rankTooltip:
        "Dimensione delle matrici a rango ridotto. Più alto = più capacità.",
      alpha: "Alpha",
      alphaTooltip:
        "Fattore di scala degli aggiornamenti LoRA. Di solito il doppio del rank.",
      dropout: "Dropout",
      dropoutTooltip:
        "Probabilità di dropout dei livelli LoRA, per ridurre l'overfitting.",
      visionLayers: "Livelli visivi",
      languageLayers: "Livelli linguistici",
      attentionModules: "Moduli di attenzione",
      mlpModules: "Moduli MLP",
      targetModules: "Moduli target",
      enableLora: "Attiva LoRA",
      trainWithLora: "Addestra con LoRA",
      stableRank: "Rank stabile",
      memoryEfficient: "Efficiente in memoria",
      weightDecomposed: "Pesi decomposti",
      optimization: "Ottimizzazione",
      schedule: "Pianificazione",
      memory: "Memoria",
      optimizer: "Ottimizzatore",
      optimizerTooltip:
        "Algoritmo di ottimizzazione. Le varianti a 8 bit riducono il consumo di memoria. La variante fused è consigliata per i modelli di visione.",
      optimizerTooltipMlx:
        "Algoritmo di ottimizzazione. AdamW è il predefinito. Lion usa meno memoria ma di solito richiede un tasso di apprendimento più basso.",
      lrScheduler: "Pianificazione del tasso di apprendimento",
      lrSchedulerTooltip:
        "Il tasso di apprendimento diminuisce costantemente con la pianificazione lineare o secondo una curva del coseno.",
      optimizerOptions: {
        adamw8bit: "AdamW 8 bit",
        pagedAdamw8bit: "Paged AdamW 8 bit",
        adamwBnb8bit: "AdamW BNB 8 bit",
        pagedAdamw32bit: "Paged AdamW 32 bit",
        adamwTorch: "AdamW (PyTorch)",
        adamwTorchFused: "AdamW (PyTorch Fused)",
      },
      lrSchedulerOptions: {
        linear: "Lineare",
        cosine: "Coseno",
      },
      batchSize: "Dimensione del batch",
      batchSizeTooltip:
        "Numero di campioni elaborati per step. Un batch più grande richiede più VRAM.",
      gradAccum: "Accumulo del gradiente",
      gradAccumTooltip: "Simula batch più grandi senza VRAM aggiuntiva.",
      weightDecay: "Decadimento dei pesi",
      weightDecayTooltip: "Regolarizzazione L2 per evitare l'overfitting.",
      warmupSteps: "Step di warmup",
      warmupStepsTooltip:
        "Aumenta gradualmente il tasso di apprendimento all'inizio dell'addestramento, per stabilità.",
      scheduleEpochsTooltip:
        "Numero di passaggi completi sul dataset. Imposta 0 per procedere in base agli step massimi.",
      saveSteps: "Step fra i salvataggi",
      saveStepsTooltip: "Salva un checkpoint ogni N step. 0 per disattivare.",
      evalSteps: "Step fra le valutazioni",
      evalStepsTooltip:
        "Frazione degli step totali fra due valutazioni (0-1). Imposta 0 per disattivare la valutazione. Per esempio 0,01 = valuta ogni 1% degli step.",
      seed: "Seed",
      seedTooltip: "Seed casuale per la riproducibilità.",
      gradCheckpoint: "Checkpoint del gradiente",
      gradCheckpointTooltip:
        "Riduce l'uso della memoria ricalcolando le attivazioni, al costo di più calcoli.",
      none: "Nessuno",
      standard: "Standard",
      enablePacking: "Attiva il packing",
      assistantCompletionsOnly: "Addestra solo sulle risposte dell'assistente",
      readMore: "Scopri di più",
    },
    training: {
      chooseModel: "Scegli un modello",
      chooseDataset: "Scegli un dataset",
      chooseModelAndDataset: "Scegli modello e dataset",
      validation: {
        s3MultimodalUnsupported:
          "I dataset S3 non sono ancora supportati per l'addestramento visivo o audio.",
        s3BucketRequired: "Inserisci prima il nome di un bucket S3.",
        s3CredentialsRequired:
          "Fornisci le chiavi di accesso S3 oppure attiva il ruolo IAM.",
        modelRequired: "Seleziona prima un modello di base.",
        learningRatePositive:
          "Inserisci un tasso di apprendimento maggiore di zero.",
        embeddingLearningRateRange:
          "Inserisci un tasso di apprendimento degli embedding maggiore di 0 e minore di 1.",
        hfDatasetRequired: "Seleziona prima un dataset Hugging Face.",
        hfDatasetSplitRequired:
          "Seleziona o inserisci prima uno split di addestramento.",
        localDatasetRequired: "Seleziona prima un dataset locale.",
        unsupportedDatasetSource: "Origine del dataset non supportata.",
      },
      startFailed: "Avvio dell'addestramento non riuscito",
      startUnconfirmed:
        "Unsloth non ha potuto confermare l'avvio dell'addestramento. La verifica dello stato continua in background.",
      stopFailed: "Arresto dell'addestramento non riuscito",
      trainingStillActiveTitle: "Addestramento ancora attivo",
      stopBeforeConfig:
        "Ferma prima l'addestramento, quindi torna alla configurazione.",
      resumeFailed: "Ripresa dell'addestramento non riuscita",
      resumeFailedTitle: "Impossibile riprendere l'addestramento",
      resumeUnavailable:
        "È possibile riprendere solo i run fermati o con errori che dispongono di un checkpoint salvato.",
      modelUnverified:
        "Impossibile verificare le impostazioni del modello. Controlla la connessione o il token Hugging Face, quindi riprova.",
      legacyDatasetScriptUnsupported:
        "Questo dataset dell'Hub dipende da uno script personalizzato obsoleto e non è supportato in questo flusso di addestramento.",
      hfModelAccessDenied:
        "Hugging Face ha negato l'accesso a questo modello. Aggiungi un token Hugging Face valido con accesso al repository, accetta gli eventuali termini richiesti e riprova.",
      hfModelVerificationRateLimited:
        "La verifica del modello su Hugging Face è soggetta a un limite di richieste. Riprova tra poco.",
      hfModelVerificationFailed:
        "Impossibile verificare il modello Hugging Face. Controlla l'ID del repository e il token di accesso.",
      hfModelMetadataUnavailable:
        "I metadati del modello Hugging Face non sono temporaneamente disponibili. Riprova prima di avviare l'addestramento.",
      datasetUnverified:
        "Impossibile verificare che il dataset sia compatibile con questo modello. Controlla la connessione o il token Hugging Face: all'avvio, l'addestramento riproverà la verifica.",
      setupChanged:
        "La configurazione dell'addestramento è cambiata durante la verifica. Controllala e avvia di nuovo.",
      configTooLarge:
        "La configurazione dell'addestramento è troppo grande (massimo 1 MiB).",
      failedToSaveConfig: "Salvataggio della configurazione non riuscito",
      startTraining: "Avvia l'addestramento",
      starting: "Avvio...",
      loadingModel: "Caricamento del modello...",
      checkingDataset: "Verifica del dataset...",
      uploadConfigTooltip: "Carica una configurazione YAML salvata",
      saveConfigTooltip: "Scarica la configurazione attuale in YAML",
      resetConfigTooltip: "Torna ai valori predefiniti del modello",
      configLoaded: "Configurazione caricata",
      failedToLoadConfig: "Impossibile caricare la configurazione",
      invalidYamlFile: "File YAML non valido",
      failedToReadFile: "Impossibile leggere il file",
      parametersReset: "Parametri riportati ai valori predefiniti del modello",
      audioIncompatible:
        "Questo modello non supporta l'audio. Passa a un modello con supporto audio o scegli un dataset senza audio.",
      visionIncompatible:
        "Un modello di testo non è compatibile con un dataset multimodale. Passa a un modello di visione o scegli un dataset di solo testo.",
      cancelTitle: "Annulla l'addestramento",
      cancelDescription: "Vuoi annullare il run di addestramento in corso?",
      continueAction: "Continua l'addestramento",
      cancelAction: "Annulla l'addestramento",
      stopTitle: "Ferma l'addestramento",
      stopDescription:
        "Scegli come fermare il run di addestramento in corso. «Ferma e salva» crea un checkpoint da cui potrai riprendere più tardi; se lo fermi senza salvare non potrai riprendere l'addestramento.",
      stopAction: "Ferma",
      stopping: "Arresto in corso...",
      stopAndSave: "Ferma e salva",
      compareInChat: "Confronta in chat",
      exportModel: "Esporta il modello",
      milestone: "Traguardo",
      halfwayDone: "A metà strada. L'addestramento ha superato il 50%.",
      doneNextStep:
        "Addestramento completato. Passo successivo: confronta gli output del modello base con quelli del modello sottoposto a fine-tuning.",
    },
    history: {
      filesDeleted: "File eliminati",
      deleteArtifactsLabel: "Elimina anche i file degli adattatori dal disco",
      deleteArtifactsDescription:
        "Rimuove la cartella di output del run, inclusi gli adattatori e i checkpoint salvati.",
      deleteArtifactsSharedNote:
        "Un altro run condivide questa cartella di output. I file verranno conservati finché non sarà eliminato l'ultimo run che li usa.",
      artifactsKeptShared:
        "Run eliminato. I file degli adattatori sono stati conservati perché un altro run usa la stessa cartella.",
      deleteArtifactsActiveError:
        "Questi file sono usati dal run di addestramento in corso. Ferma l'addestramento prima di eliminarli.",
      deleteArtifactsFailed:
        "Il run è stato eliminato, ma non è stato possibile rimuoverne i file.",
      deleteArtifactsRetainedError:
        "Non è stato possibile rimuovere i file dell’adattatore, quindi il run è stato mantenuto nella cronologia.",
      title: "Cronologia",
      emptyDescription:
        "Nessun run di addestramento. Avvia il tuo primo run nella scheda Configura.",
      loadError: "Impossibile caricare i run di addestramento",
      deleteError: "Impossibile eliminare il run di addestramento. Riprova.",
      retry: "Riprova",
      loadMore: "Carica altri",
      loading: "Caricamento...",
      loadingRun: "Caricamento del run di addestramento...",
      runNotFound: "Run non trovato",
      deleteTitle: "Eliminare il run di addestramento?",
      deleteDescription:
        "Questo run di addestramento e tutte le sue metriche verranno eliminati definitivamente. L'operazione è irreversibile.",
      resumeTraining: "Riprendi l'addestramento",
      resuming: "Ripresa in corso...",
      deleteRun: "Elimina il run",
      loss: "Loss",
      steps: "Step",
      lossTrendSparkline: "Sparkline dell'andamento della loss",
      relativeJustNow: "proprio ora",
      status: {
        completed: "Completato",
        stopped: "Fermato",
        error: "Errore",
        running: "In esecuzione",
        continued: "Ripreso",
      },
      message: {
        completed: "Addestramento completato",
        stopped: "Addestramento fermato",
        running: "Addestramento in corso",
        errored: "Errore durante l'addestramento",
      },
      copyPreviewLink: "Copia il link di anteprima",
      previewLinkCopied: "Link di anteprima copiato",
      previewLinkCopyFailed: "Impossibile copiare il link",
    },
    charts: {
      settings: "Impostazioni dei grafici",
      settingsDescription:
        "Regola la visualizzazione dei grafici mentre l'addestramento continua.",
      openSettings: "Apri le impostazioni dei grafici",
      viewWindow: "Finestra di visualizzazione",
      viewWindowDescription:
        "Mostra solo gli step più recenti oppure tutta la cronologia.",
      window: "Finestra",
      all: "Tutti",
      trainingLoss: "Loss di addestramento",
      trainingLossDescription:
        "Controlla le sovrapposizioni e il livellamento EMA.",
      smoothing: "Livellamento",
      smoothingDescription:
        "Sposta il cursore verso destra per aumentare il livellamento. `0` = dati grezzi.",
      showRawLoss: "Mostra la loss grezza",
      showSmoothedLoss: "Mostra la loss livellata",
      showAverageLine: "Mostra la linea della media",
      scaleAndCleanup: "Scala e gestione degli outlier",
      linear: "Lineare",
      log: "Logaritmica",
      noClip: "Nessun clipping",
      clipP99: "Limita al p99",
      clipP95: "Limita al p95",
      lossAxis: "Asse della loss",
      gradientNormAxis: "Asse della norma del gradiente",
      learningRateAxis: "Asse del tasso di apprendimento",
      resetDefaults: "Ripristina i valori predefiniti",
      loss: "Loss",
      smoothed: "Livellata",
      evalLoss: "Loss di valutazione",
      learningRate: "Tasso di apprendimento",
      lr: "LR",
      gradNorm: "Norma grad.",
      gradientNorm: "Norma del gradiente",
      step: "Step {step}",
      averageValue: "media {value}",
      waitingForFirstEvaluationStep:
        "In attesa del primo step di valutazione...",
      evaluationNotConfigured: "Valutazione non configurata",
      evalChartWillAppear:
        "Il grafico comparirà al raggiungimento di eval_steps",
      setEvalDatasetAndSteps:
        "Imposta un dataset di valutazione ed eval_steps per monitorare la loss di valutazione",
    },
    progress: {
      title: "Avanzamento dell'addestramento",
      liveMetrics: "Metriche di addestramento in tempo reale",
      exportGguf: "Esporta in GGUF",
      openConfig: "Apri la configurazione di addestramento",
      configLabel: "Configurazione di addestramento",
      hyperparams: "Iperparametri",
      epochs: "Epoche",
      batchSize: "Dimensione del batch",
      learningRate: "Tasso di apprendimento",
      optimizer: "Ottimizzatore",
      maxSteps: "Step massimi",
      contextLength: "Lunghezza di contesto",
      warmupSteps: "Step di warmup",
      rank: "Rank",
      alpha: "Alpha",
      dropout: "Dropout",
      variant: "Variante",
      epoch: "Epoca {value}",
      percentComplete: "{percent}% completato",
      stepProgress: "Step {current} / {total}",
      loss: "Loss",
      lr: "LR",
      gradNorm: "Norma grad.",
      project: "Progetto",
      model: "Modello",
      method: "Metodo",
      elapsed: "Tempo trascorso: {value}",
      eta: "Tempo rimanente stimato: {value}",
      stepsPerSecond: "{value} step/s",
      noStepsPerSecond: "-- step/s",
      tokens: "Token: {value}",
      gpuMonitor: "Monitor della GPU",
      live: "In tempo reale",
      utilization: "Utilizzo",
      temperature: "Temperatura",
      vram: "VRAM",
      power: "Consumo",
      phase: {
        idle: "Inattivo",
        downloadingModel: "Download del modello",
        downloadingDataset: "Download del dataset",
        loadingModel: "Caricamento del modello",
        loadingDataset: "Caricamento del dataset",
        configuring: "Configurazione",
        training: "Addestramento",
        finalizing: "Salvataggio del modello",
        completed: "Completato",
        error: "Errore",
        stopped: "Fermato",
      },
    },
    trainingStart: {
      ready: "Pronto",
      downloading: "Download",
      preparing: "Preparazione",
      left: "tempo rimanente: {eta}",
      downloaded: "Dati scaricati: {size}",
      terminalStart: "> avvio dell'addestramento con Unsloth...",
      preparingResources: "> Preparazione del modello e del dataset...",
      gettingReady: "> Stiamo preparando tutto per il tuo run...",
      waitingForFirstStep: "> {message} | in attesa del primo step... ({step})",
      resumingTraining: "Ripresa dell'addestramento...",
      startingTraining: "avvio dell'addestramento...",
      dataset: "Dataset",
      datasetStreaming: "Dataset: in streaming (nessun download completo)",
      modelWeights: "Pesi del modello",
    },
  },
  modelMemory: {
    readout:
      "Pesi {model} + contesto {context} = {total} di {budget} di VRAM utilizzabile",
    readoutWithSpec:
      "Pesi {model} + KV {kv} + draft MTP {spec} = {total} di {budget} di VRAM utilizzabile",
    kvRate: "KV riservato, ~{rate}/token",
    oomLikely: "Con le impostazioni attuali è probabile un errore di memoria",
    tooLarge: "Più grande della VRAM, verrà scaricato sulla CPU. Una quantizzazione più piccola è più veloce",
  },
} as const;
