// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DeepPartialMessageTree } from "../types";
import type { en } from "./en";

export const fr = {
  picker: {
    onDevice: "Sur l'appareil",
    huggingFace: "Hugging Face",
    retry: "Réessayer",
    loadMore: "Charger plus",
    offlineTitle: "Vous êtes hors ligne",
    offlineBody:
      "Passez à « Appareil » pour utiliser des {noun} en cache ou locaux.",
    offlineSwitchDevice: "Appareil",
    searchAriaLabel: "Rechercher des {noun}",
    modelSourceAriaLabel: "Source du modèle",
    hubSectionAriaLabel: "Section du Hub",
    modelDropped: "N'est plus proposé",
    modelDroppedByProvider: "{provider} · n'est plus proposé",
    modelDisabled: "Non activé",
    modelDisabledByProvider: "{provider} · non activé",
    multipleMatches:
      "Plusieurs {noun} correspondent. Choisissez-en un dans la liste.",
    rateLimitedTitle: "Limite de requêtes Hugging Face atteinte",
    rateLimitedBody:
      "Patientez un instant, puis relancez la recherche de {noun}.",
    hfToken: {
      label: "Token HF",
      saved: "Enregistré",
      add: "Ajouter",
      savedAriaLabel: "Token Hugging Face enregistré",
      addAriaLabel: "Définir le token Hugging Face",
      savedHint:
        "Token enregistré. L'accès est vérifié au moment de son utilisation.",
      addHint:
        "Définissez un token pour accéder aux dépôts privés et à accès restreint.",
    },
  },
  common: {
    cancel: "Annuler",
    close: "Fermer",
    delete: "Supprimer",
    done: "Terminé",
    error: "Erreur",
    export: "Exporter",
    help: "Aide",
    loading: "Chargement...",
    new: "Nouveau",
    rename: "Renommer",
    save: "Enregistrer",
    saving: "Enregistrement...",
    search: "Rechercher",
    shutdown: "Arrêter",
  },
  shell: {
    beta: "BETA",
    brand: "unsloth",
    product: "Unsloth",
    accountMenu: "Menu du compte de {name}",
    updateAvailable: "Mise à jour disponible",
    resize: {
      collapse: "Cliquez pour réduire",
      expand: "Cliquez pour déployer",
      drag: "Faites glisser pour redimensionner",
    },
    aria: {
      home: "Accueil Unsloth",
      closeSidebar: "Fermer la barre latérale",
      openSidebar: "Ouvrir la barre latérale",
      resizeSidebar: "Redimensionner ou réduire la barre latérale",
      resizeRunSettings: "Redimensionner ou fermer les paramètres d'exécution",
      openRunSettings: "Ouvrir les paramètres d'exécution",
      chatOptions: "Options de discussion",
      runOptions: "Options d'exécution",
    },
    navigation: {
      newChat: "Nouvelle discussion",
      returnToChat: "Retour à la discussion",
      returnToChats: "Retour à {count} discussions",
      chatGenerating: "Génération en cours",
      compare: "Comparer",
      search: "Rechercher",
      hub: "Hub de modèles",
      projects: "Projets",
      train: "Entraîner",
      recipes: "Recettes",
      images: "Images",
      video: "Vidéo",
      audio: "Audio",
      trainChecking: "Vérification de la prise en charge de l'entraînement sur cette machine...",
      videoChecking: "Vérification de la prise en charge de la vidéo sur cette machine...",
      more: "Plus",
      customizeSidebar: "Personnaliser la barre latérale",
      newBadge: "Nouveau",
      export: "Exporter",
      recents: "Discussions récentes",
      noChatsYet: "Aucune discussion pour le moment",
      showMore: "Afficher plus",
      showLess: "Afficher moins",
      settings: "Paramètres",
      api: "API",
      lightMode: "Mode clair",
      darkMode: "Mode sombre",
      guidedTour: "Visite guidée",
      help: "Aide",
      logOut: "Se déconnecter",
      shutdown: "Arrêter",
    },
    notFound: {
      title: "Page introuvable",
      description: "{path} n'existe pas.",
      backToChat: "Retour à la discussion",
    },
    selection: {
      pinProjects: "Épingler les projets",
      unpinProjects: "Détacher les projets",
      deleteProjects: "Supprimer les projets",
      deleteProjectsTitle: "Supprimer les projets",
      deleteProjectsDescription:
        "Supprimer {count} projets ? Leurs discussions seront supprimées définitivement.",
      deleteProjectsFilesDescription:
        "Le dossier de travail de chaque projet est retiré du disque.",
      countSelected: "{count} sélectionnés",
      pinChats: "Épingler les discussions",
      unpinChats: "Détacher les discussions",
      archiveChats: "Archiver les discussions",
      markUnread: "Marquer comme non lu",
      deleteChats: "Supprimer les discussions",
      deleteTitle: "Supprimer les discussions",
      deleteDescription: "Supprimer {count} discussions ? Cette action est irréversible.",
      deleteFilesDescription:
        "Le dossier bac à sable propre à chaque discussion est supprimé du disque. Les fichiers écrits dans un projet restent dans l'espace de travail de ce projet.",
      deleteFilesLabel: "Supprimer les fichiers et le dossier bac à sable",
      deleteChatFilesDescription:
        "Le dossier bac à sable propre à cette discussion est retiré du disque. Les fichiers écrits dans un projet restent dans l'espace de travail du projet.",
    },
    organize: {
      sidebarHeading: "Organiser la barre latérale",
      byProject: "Par projet",
      inOneList: "Dans une seule liste",
      sortChatsBy: "Trier les discussions par",
      sortPinnedBy: "Trier les épinglés par",
      priority: "Priorité",
      lastUpdated: "Dernière mise à jour",
      manualOrder: "Ordre manuel",
      moveUp: "Monter",
      moveDown: "Descendre",
      organizeChats: "Organiser les discussions",
      organizeProjects: "Organiser les projets",
      sortPinnedChats: "Trier les discussions épinglées",
    },
    dialog: {
      deleteChat: {
        title: "Supprimer la discussion",
        description: 'Voulez-vous vraiment supprimer cette discussion "{name}" ?',
      },
      deleteRun: {
        title: "Supprimer l'entraînement",
        description: 'Voulez-vous vraiment supprimer cet entraînement "{name}" ?',
      },
      renameChat: {
        title: "Renommer la discussion",
        placeholder: "Titre de la discussion",
      },
      renameRun: {
        title: "Renommer l'entraînement",
        placeholder: "Nom de l'entraînement",
      },
    },
    toast: {
      cannotDeleteRunningRun:
        "Impossible de supprimer un entraînement en cours",
      failedToDeleteChat: "Échec de la suppression de la discussion",
      failedToDeleteRun: "Échec de la suppression de l'entraînement",
      failedToRenameChat: "Échec du renommage de la discussion",
      failedToRenameRun: "Échec du renommage de l'entraînement",
    },
  },
  settings: {
    title: "Paramètres",
    dialog: {
      title: "Paramètres",
      description: "Gérez vos préférences Unsloth.",
      closeAriaLabel: "Fermer les paramètres",
      searchPlaceholder: "Rechercher dans les paramètres…",
      searchNoResults: "Aucun paramètre trouvé.",
      panelFailed: "Cette section n'a pas pu être chargée.",
      panelReload: "Recharger",
    },
    tabs: {
      general: "Général",
      profile: "Profil",
      appearance: "Apparence",
      resources: "Système",
      chat: "Discussion",
      connections: "Connexions",
      apiKeys: "API",
      remoteLan: "Accès distant et LAN",
      about: "À propos",
      data: "Données",
      agents: "Agents",
      debugging: "Journaux",
      voice: "Voix",
      keyboardShortcuts: "Raccourcis",
    },
    keyboardShortcuts: {
      title: "Raccourcis clavier",
      description:
        "Modifiez un raccourci, ou effacez-le pour libérer la combinaison pour le navigateur ou le système.",
      searchPlaceholder: "Rechercher des raccourcis…",
      noResults: "Aucun raccourci ne correspond à cette recherche.",
      unassigned: "Non attribué",
      recording: "Appuyez sur les touches…",
      recordingHint: "Appuyez sur la nouvelle combinaison, ou Échap pour annuler.",
      needsModifier: "Ajoutez ⌘, Ctrl ou Alt. Une touche seule avalerait la saisie.",
      conflict: "Également utilisé par un autre raccourci",
      conflictShadowed: "Un autre raccourci utilise cette combinaison et s'exécute à sa place",
      edit: "Modifier le raccourci",
      clear: "Supprimer le raccourci",
      reset: "Rétablir la valeur par défaut",
      resetAll: "Tout réinitialiser",
      primarySlot: "Raccourci",
      alternateSlot: "Raccourci alternatif",
      browserReserved:
        "Votre navigateur peut réserver cette combinaison. Elle fonctionne dans l’application de bureau.",
      actions: {
        openSettings: {
          label: "Ouvrir les paramètres",
          description: "Ouvrir la fenêtre des paramètres",
        },
        openKeyboardShortcuts: {
          label: "Raccourcis clavier",
          description: "Ouvrir cette liste de raccourcis",
        },
        searchChats: {
          label: "Rechercher dans les discussions",
          description: "Ouvrir la recherche de discussions",
        },
        openMcpServers: {
          label: "Serveurs MCP",
          description: "Configurer les serveurs MCP de cette discussion",
        },
        logOut: {
          label: "Se déconnecter",
          description: "Se déconnecter d’Unsloth",
        },
        approveToolRequest: {
          label: "Approuver la demande",
          description: "Autoriser l’appel d’outil en attente",
        },
        declineToolRequest: {
          label: "Refuser la demande",
          description: "Refuser l’appel d’outil en attente",
        },
        newChat: {
          label: "Nouvelle discussion",
          description: "Démarrer une nouvelle discussion",
        },
        newTemporaryChat: {
          label: "Nouvelle discussion temporaire",
          description: "Démarrer une discussion qui n’est pas enregistrée dans l’historique",
        },
        newStandaloneChat: {
          label: "Nouvelle discussion autonome",
          description: "Démarrer une discussion en dehors de tout projet",
        },
        archiveChat: {
          label: "Archiver la discussion",
          description: "Archiver les discussions sélectionnées, sinon celle en cours",
        },
        markChatUnread: {
          label: "Marquer comme non lu",
          description: "Marquer comme non lues les discussions sélectionnées, sinon celle en cours",
        },
        togglePinChat: {
          label: "Épingler/désépingler",
          description: "Épingler ou désépingler les discussions sélectionnées, sinon celle en cours",
        },
        selectAllChats: {
          label: "Tout sélectionner",
          description: "Sélectionner tous les chats de la barre latérale",
        },
        clearChatSelection: {
          label: "Effacer la sélection",
          description: "Désélectionner les chats sélectionnés. Échap le fait aussi",
        },
        deleteSelectedChats: {
          label: "Supprimer les chats sélectionnés",
          description: "Supprimer tous les chats sélectionnés",
        },
        nextRecentlyViewedChat: {
          label: "Discussion consultée suivante",
          description: "Avancer dans les discussions récemment consultées",
        },
        previousRecentlyViewedChat: {
          label: "Discussion consultée précédente",
          description: "Reculer dans les discussions récemment consultées",
        },
        nextChat: {
          label: "Discussion suivante",
          description: "Passer à la discussion suivante de la barre latérale",
        },
        previousChat: {
          label: "Discussion précédente",
          description: "Passer à la discussion précédente de la barre latérale",
        },
        nextChatNeedingAttention: {
          label: "Discussion à traiter suivante",
          description: "Passer à la discussion suivante en cours, en file ou non lue",
        },
        clearAllUnreads: {
          label: "Tout marquer comme lu",
          description: "Marquer toutes les discussions comme lues",
        },
        goToRecentChat1: {
          label: "Aller à la discussion récente 1",
          description: "Ouvrir la discussion 1 sous Récents",
        },
        goToRecentChat2: {
          label: "Aller à la discussion récente 2",
          description: "Ouvrir la discussion 2 sous Récents",
        },
        goToRecentChat3: {
          label: "Aller à la discussion récente 3",
          description: "Ouvrir la discussion 3 sous Récents",
        },
        goToRecentChat4: {
          label: "Aller à la discussion récente 4",
          description: "Ouvrir la discussion 4 sous Récents",
        },
        goToRecentChat5: {
          label: "Aller à la discussion récente 5",
          description: "Ouvrir la discussion 5 sous Récents",
        },
        goToRecentChat6: {
          label: "Aller à la discussion récente 6",
          description: "Ouvrir la discussion 6 sous Récents",
        },
        switchToChat: {
          label: "Aller à Discussion",
          description: "Ouvrir l’espace de discussion",
        },
        switchToProjects: {
          label: "Aller à Projets",
          description: "Ouvrir l’espace des projets",
        },
        switchToHub: {
          label: "Aller au Hub de modèles",
          description: "Ouvrir le hub de modèles",
        },
        switchToTrain: {
          label: "Aller à Entraînement",
          description: "Ouvrir l’espace d’entraînement",
        },
        switchToRecipes: {
          label: "Aller à Recipes",
          description: "Ouvrir Data Recipes",
        },
        switchToImages: {
          label: "Aller à Images",
          description: "Ouvrir l’espace des images",
        },
        switchToVideo: {
          label: "Aller à Vidéo",
          description: "Ouvrir l’espace vidéo",
        },
        switchToAudio: {
          label: "Aller à Audio",
          description: "Ouvrir l’espace audio",
        },
        switchToExport: {
          label: "Aller à Export",
          description: "Ouvrir l’espace d’export",
        },
        toggleSidebar: {
          label: "Afficher/masquer la barre latérale",
          description: "Afficher ou masquer la barre latérale",
        },
        toggleApiMonitor: {
          label: "Afficher/masquer l’activité API",
          description: "Afficher ou masquer le moniteur d’activité API",
        },
        openModelPicker: {
          label: "Ouvrir le sélecteur de modèle",
          description: "Choisir le modèle de cette discussion",
        },
        openProjectPicker: {
          label: "Ouvrir le sélecteur de projet",
          description: "Passer à un autre projet depuis l'en-tête du chat",
        },
        startDictation: {
          label: "Dictée",
          description: "Démarrer ou arrêter la dictée dans la zone de saisie",
        },
        attachFiles: {
          label: "Joindre photos et fichiers",
          description: "Ajouter une pièce jointe à la zone de saisie",
        },
        sendMessage: {
          label: "Envoyer le message",
          description: "Envoyer le contenu de la zone de saisie",
        },
        cycleReasoningEffort: {
          label: "Faire défiler l’effort de raisonnement",
          description: "Parcourir les niveaux d’effort de raisonnement",
        },
        increaseReasoningEffort: {
          label: "Augmenter l’effort de raisonnement",
          description: "Monter d’un niveau l’effort de raisonnement",
        },
        decreaseReasoningEffort: {
          label: "Diminuer l’effort de raisonnement",
          description: "Descendre d’un niveau l’effort de raisonnement",
        },
        toggleFastMode: {
          label: "Activer/désactiver le mode Fast",
          description: "Activer ou désactiver le mode Fast",
        },
        renameChat: {
          label: "Renommer la discussion",
          description: "Renommer la discussion en cours",
        },
        forkChat: {
          label: "Dupliquer la discussion",
          description: "Créer une branche à partir du dernier message",
        },
        copyChatAsMarkdown: {
          label: "Copier en Markdown",
          description: "Copier toute la discussion dans le presse-papiers en Markdown",
        },
        copySessionId: {
          label: "Copier l’ID de session",
          description: "Copier l’ID de session sandbox de cette discussion",
        },
      },
    },
    debugging: {
      logSection: "Fichier journal",
      source: "Fichier journal",
      sourceHint: "Les exécuteurs de modèles écrivent leurs propres journaux : un chargement ou une génération en échec y est donc souvent expliqué plutôt que dans le journal du serveur.",
      path: "Emplacement",
      pathCopy: "Copier le chemin",
      refreshSection: "Actualisation",
      mode: "Mode",
      modeLive: "En direct",
      modeInterval: "Toutes les 3 secondes",
      modeManual: "Manuel",
      refreshNow: "Actualiser maintenant",
      privacyNote: "Les identifiants sont masqués dans cette vue. Dans le fichier sur le disque, ils ne le sont pas.",
      copyVisible: "Copier le journal visible",
      empty: "Rien n'a encore été consigné.",
      disabled: "La journalisation dans un fichier est désactivée (UNSLOTH_STUDIO_NO_FILE_LOG=1).",
      missing: "Aucun fichier journal n'a été trouvé.",
      unreadable: "Le fichier journal n'a pas pu être lu.",
      timeout: "La demande du journal a expire. Le serveur est peut-etre injoignable.",
      droppedNotice: "Certaines lignes ont été ignorées : le journal a été écrit plus vite qu'il ne pouvait être lu.",
      morePending: "D'autres lignes sont encore en cours de lecture ; elles arriveront au prochain rafraichissement.",
      staleSession: "La journalisation dans un fichier est desactivee : il s'agit d'une session anterieure, qui ne sera pas mise a jour.",
      keywords: "debogage deboguer journal journaux log logs erreur erreurs plantage trace diagnostic depannage debug",
    },
    voice: {
      title: "Voix",
      description: "Microphone, dictée, reconnaissance vocale et lecture à voix haute",
      dictation: {
        sectionTitle: "Dictée",
        engineLabel: "Moteur de dictée",
        engineBrowser: "Navigateur",
        engineBrowserDescription:
          "Transcrit l'audio via le service vocal de votre navigateur. Sélectionnez « Transcription locale » pour utiliser un modèle STT.",
        engineModel: "Transcription locale",
        engineModelDescription:
          "Exécute un modèle de reconnaissance vocale (STT) en local et fonctionne hors ligne. Téléchargez-le, chargez-le ; il se décharge après une période d'inactivité.",
        engineCustom: "Point de terminaison personnalisé",
        engineCustomDescription:
          "Envoie l'audio enregistré à un serveur STT compatible avec OpenAI depuis vos connexions.",
        connectionLabel: "Connexion",
        connectionDescription:
          "Ajoutez un serveur compatible avec OpenAI et éventuellement une clé API dans Connexions.",
        connectionPlaceholder: "Sélectionner une connexion",
        connectionEmpty: "Aucune connexion disponible",
        customModelLabel: "Modèle",
        customModelDescription:
          "Nom du modèle envoyé à /v1/audio/transcriptions.",
        sttModelLabel: "Modèle de reconnaissance vocale",
        sttModelDescription:
          "Choisissez ou recherchez un modèle STT à exécuter en local.",
        sttModelSearchPlaceholder: "Rechercher un modèle",
        sttModelSearching: "Recherche sur Hugging Face…",
        sttModelValidating: "Vérification de la compatibilité Whisper…",
        sttModelNoResults: "Aucun modèle Whisper trouvé",
        sttModelInvalid: "Ce dépôt ne peut pas être utilisé pour la dictée",
        sttModelFailed: "Impossible de charger le modèle STT",
        sttModelUnsupported:
          "L'enregistrement n'est pas pris en charge dans ce navigateur",
        sttChecking: "Vérification…",
        sttOnDemand: "Téléchargé",
        sttLoadingModel: "Chargement du modèle…",
        sttReady: "Chargé sur {device}",
        sttLoaded: "Chargé",
        sttUnavailable:
          "Non installé sur ce serveur. Exécutez `unsloth studio update` pour activer la dictée locale.",
        sttRetry: "Réessayer",
        sttDownloadChecking: "Vérification de l'état du téléchargement…",
        sttNotDownloaded: "Non téléchargé",
        sttDownloadStatusFailed:
          "Impossible de vérifier l'état du téléchargement",
        sttDownload: "Télécharger",
        sttDownloadConfirmTitle: "Télécharger {model} ?",
        sttDownloadConfirmBody:
          "La dictée locale fonctionne entièrement hors ligne, mais elle a d'abord besoin du modèle de reconnaissance vocale {model}. Environ {size}, téléchargé une seule fois dans votre cache Hugging Face.",
        sttDownloadConfirmBodyUnsized:
          "La dictée locale fonctionne entièrement hors ligne, mais elle a d'abord besoin du modèle de reconnaissance vocale {model}. Il est téléchargé une seule fois dans votre cache Hugging Face.",
        sttOpenVoiceSettings: "Ouvrir les paramètres Voix",
        sttDownloadStarted: "Téléchargement de {model}",
        sttDownloading: "Téléchargement… {progress} %",
        sttCancelDownload: "Annuler",
        sttCancellingDownload: "Annulation…",
        sttCancelDownloadFailed: "Impossible d'annuler le téléchargement",
        sttDownloadComplete: "Modèle de reconnaissance vocale téléchargé",
        sttModelReady: "{model} est prêt pour la dictée",
        sttRecommended: "Recommandé",
        sttDownloadFailed:
          "Impossible de télécharger le modèle de reconnaissance vocale",
        sttLoad: "Charger",
        sttUnload: "Décharger",
        sttUnloading: "Déchargement…",
        microphoneLabel: "Microphone",
        microphoneFallbackName: "Microphone {index}",
        microphoneDescription: "Utilisé pour la dictée",
        microphoneFallbackHint:
          "Utilisé pour la dictée. Revient au périphérique par défaut du système si le moteur vocal du navigateur ne peut pas utiliser ce périphérique",
        microphoneGrantDescription:
          "Autorisez l'accès au micro pour afficher le nom des périphériques",
        allowMicrophone: "Autoriser l’accès au microphone",
        micAccessBlocked:
          "L'accès au microphone a été bloqué. Autorisez l'accès au microphone pour cette page Unsloth, puis réessayez.",
        micAccessBlockedDesktop:
          "L'accès au microphone a été bloqué. Réessayez et choisissez Autoriser, ou activez le microphone dans les paramètres de confidentialité du système.",
        micAccessUnsupported:
          "L'accès au microphone n'est pas pris en charge dans ce navigateur ou ce contexte.",
        systemDefault: "Par défaut du système",
        savedMicDisconnected: "Microphone enregistré (non connecté)",
        languageLabel: "Langue de la dictée",
        languageDescription: "Langue à reconnaître",
        languageAuto: "Auto (langue du navigateur)",
        languageAutoDetect: "Auto (détecter la langue)",
      },
      dictionary: {
        sectionTitle: "Dictionnaire de dictée",
        sectionDescription:
          "Définissez l'orthographe employée par la dictée pour certains mots ou expressions",
        manageLabel: "Orthographes personnalisées",
        manage: "Gérer",
        backToVoice: "Retour à la section Voix",
        addEntry: "Ajouter une entrée",
        newEntryAria: "Nouvelle entrée du dictionnaire",
        entryPlaceholder: "Marie Dupont",
        entryAria: "Entrée du dictionnaire {index}",
        removeEntryAria: "Supprimer l’entrée {index} du dictionnaire",
      },
      recents: {
        sectionTitle: "Historique des dictées",
        sectionDescription:
          "Chaque dictée est enregistrée ici pour vous permettre de récupérer le texte",
        manageLabel: "Historique des dictées",
        manage: "Gérer",
        pageDescription:
          "Toutes les dictées sont enregistrées. Consultez-les, copiez-les ou supprimez-les, ou ouvrez la discussion dans laquelle une dictée a été utilisée.",
        searchPlaceholder: "Rechercher des dictées",
        sortLabel: "Trier les dictées",
        sortNewest: "Plus récentes",
        sortOldest: "Plus anciennes",
        sortAlpha: "De A à Z",
        noMatches: "Aucune dictée ne correspond à votre recherche",
        detailTitle: "Dictée enregistrée",
        backToVoice: "Retour à la section Voix",
        backToRecents: "Retour aux dictées récentes",
        view: "Voir la dictée complète",
        empty: "Aucune dictée pour le moment",
        dictationColumn: "Dictée",
        dateColumn: "Date de création",
        copy: "Copier la dictée",
        copied: "Dictée copiée dans le presse-papiers",
        copyFailed: "Impossible de copier dans le presse-papiers",
        delete: "Supprimer la dictée",
        deleteTitle: "Supprimer la dictée",
        deleteDescription:
          "Supprimer cette dictée enregistrée ? Cette action est irréversible.",
        deleteLinkedDescription:
          "Supprimer cette dictée enregistrée ? Vous pouvez aussi supprimer la discussion dans laquelle elle a été utilisée. Cette action est irréversible.",
        deleteWithChat: "Supprimer la discussion et la dictée",
        deleteWithChatFailed: "Impossible de supprimer la discussion",
        clear: "Effacer l'historique",
        clearTitle: "Effacer l'historique des dictées",
        clearDescription:
          "Supprimer toutes les dictées enregistrées ? Cette action est irréversible.",
        clearConfirm: "Tout effacer",
        showMore: "Afficher davantage ({count})",
        openChat: "Ouvrir la discussion",
      },
      readAloud: {
        sectionTitle: "Lecture à voix haute",
        buttonLabel: "Bouton de lecture à voix haute",
        buttonDescription: "Afficher dans les réponses de l’assistant",
        engineLabel: "Moteur TTS",
        engineSystemDescription: "Voix intégrées à l'appareil",
        engineStudioDescription:
          "Utilise le modèle audio chargé (par exemple Orpheus)",
        engineSystem: "Voix du système",
        engineStudio: "Charger un modèle TTS",
        engineCustom: "Endpoint personnalisé",
        engineCustomDescription:
          "Un serveur TTS compatible OpenAI parmi vos connexions (p. ex. Kokoro)",
        connectionLabel: "Connexion",
        connectionDescription:
          "Ajoutez un serveur compatible OpenAI dans l'onglet Connexions",
        connectionPlaceholder: "Sélectionner une connexion",
        customModelLabel: "Modèle",
        customVoiceDescription:
          "Nom de la voix attendu par l'endpoint ; alloy par défaut",
        modelLabel: "Modèle TTS",
        modelDescription:
          "Chargez un modèle audio depuis le sélecteur de modèles (par exemple Orpheus TTS)",
        openAudioAction: "Ouvrir Audio",
        voiceLabel: "Voix",
        voiceDescription: "Meilleures voix sur cet appareil",
        speedLabel: "Vitesse",
        pitchLabel: "Hauteur",
        volumeLabel: "Volume",
        previewLabel: "Écouter la voix",
        previewDescription: "Lire un court extrait",
        previewFailed: "Échec de l’aperçu de la synthèse vocale",
        previewAction: "Écouter",
        preparingAction: "Génération…",
        stopAction: "Arrêter",
        ttsLabel: "Synthèse vocale",
        notSupported: "Indisponible dans ce navigateur",
      },
    },
    general: {
      title: "Général",
      description: "Préférences globales pour Unsloth.",
      account: "Compte",
      huggingFaceToken: "Jeton Hugging Face",
      huggingFaceTokenDescription:
        "Utilisé pour charger des modèles restreints et publier des artefacts.",
      hideToken: "Masquer le jeton",
      showToken: "Afficher le jeton",
      clearToken: "Effacer",
      checkingToken: "Vérification du token...",
      tokenValidated: "Jeton validé",
      password: "Mot de passe",
      passwordDescription:
        "Changez le mot de passe de ce compte Unsloth.",
      passwordDialog: {
        trigger: "Changer le mot de passe",
        title: "Changer le mot de passe",
        description:
          "Saisissez votre mot de passe actuel et choisissez-en un nouveau (au moins {minLength} caractères).",
        setTrigger: "Définir le mot de passe distant",
        setTitle: "Définir le mot de passe distant",
        setDescription:
          "Choisissez le mot de passe utilisé par les navigateurs distants pour se connecter avec l'identifiant unsloth (au moins {minLength} caractères). L'application de bureau Unsloth continue de se connecter automatiquement.",
        setSubmit: "Définir le mot de passe",
        setting: "Définition...",
        setDone: "Mot de passe défini.",
        currentPassword: "Mot de passe actuel",
        newPassword: "Nouveau mot de passe",
        confirmPassword: "Confirmer le nouveau mot de passe",
        currentTooShort:
          "Le mot de passe actuel doit comporter au moins {minLength} caractères.",
        newTooShort:
          "Le nouveau mot de passe doit comporter au moins {minLength} caractères.",
        mismatch: "Les mots de passe ne correspondent pas.",
        samePassword:
          "Le nouveau mot de passe doit être différent de l'actuel.",
        update: "Mettre à jour le mot de passe",
        updating: "Mise à jour...",
        updated: "Mot de passe mis à jour.",
        updateFailed: "Échec de la mise à jour du mot de passe.",
        newHasSpaces: "Le nouveau mot de passe ne peut pas contenir d'espaces.",
      },
      chatDefaults: "Valeurs par défaut de discussion",
      autoTitleNewChats: "Titrer automatiquement les nouvelles discussions",
      autoTitleNewChatsDescription:
        "Générer un titre court à partir du premier message.",
      helperLlm: {
        sectionTitle: "LLM assistant",
        preloadOnStartup: "Précharger le LLM assistant au démarrage",
        preloadOnStartupDescription:
          "Télécharger le modèle assistant d'AI Assist en arrière-plan au démarrage. Désactivé par défaut ; AI Assist peut toujours le récupérer à la demande.",
        disabledByEnv:
          "Désactivé par UNSLOTH_HELPER_MODEL_DISABLE dans l'environnement du backend.",
        loadError: "Échec du chargement des paramètres du LLM assistant.",
        saveError: "Échec de l'enregistrement des paramètres du LLM assistant.",
      },
      modelAutoSwitch: {
        sectionTitle: "Changement automatique de modèle (API OpenAI)",
        enable: "Changer de modèle par requête",
        enableDescription:
          "Charger, avant de répondre, un GGUF téléchargé indiqué dans une requête API. Désactivé par défaut.",
        idleUnload: "Déchargement automatique en cas d'inactivité",
        idleUnloadDescription:
          "Libérer la VRAM après ce nombre de secondes d’inactivité. 0 maintient le modèle chargé ; le minimum est 60.",
        idleSecondsAriaLabel:
          "Délai d’inactivité avant le déchargement automatique, en secondes",
        mediaEnable: "Changer de modèle d’image et de vidéo par requête",
        mediaEnableDescription:
          "Charger, avant la génération, un modèle d’image ou de vidéo téléchargé indiqué dans une requête API. Réglage distinct : celui ci-dessus ne concerne que le modèle de discussion. Désactivé par défaut.",
        mediaIdleUnload:
          "Déchargement automatique en cas d’inactivité pour l’image et la vidéo",
        mediaIdleUnloadDescription:
          "Libérer la VRAM en déchargeant les modèles d’image et de vidéo après ce nombre de secondes d’inactivité. Réglage distinct : celui du dessus ne concerne que le modèle de discussion. 0 les maintient chargés ; le minimum est 60.",
        mediaIdleSecondsAriaLabel:
          "Délai d’inactivité avant le déchargement automatique de l’image et de la vidéo, en secondes",
        mediaIdlePaused:
          "En pause tant que « Conserver le modèle en mémoire GPU » est activé.",
        idleNeedsEnable:
          "Activez d’abord « Changer de modèle par requête ».",
        idleActiveViaEnv: "Actif via UNSLOTH_MODEL_IDLE_TTL.",
        loadError:
          "Échec du chargement des paramètres de changement automatique de modèle.",
        saveError:
          "Échec de l'enregistrement des paramètres de changement automatique de modèle.",
        idleError: "Saisissez 0 pour garder le modèle chargé, ou au moins 60 secondes.",
        autoDownload: "Télécharger les modèles manquants",
        autoDownloadDescription:
          "Récupérer un GGUF indiqué dans une requête API qui n'est pas encore téléchargé. Toute personne disposant d'une clé API peut alors consommer de l'espace disque et de la bande passante.",
        keepKv:
          "Conserver le contexte de la discussion après un déchargement en cas d'inactivité",
        keepKvDescription:
          "Enregistrer le cache KV avant un déchargement en cas d'inactivité, afin qu'une discussion reprise n'ait pas à relire l'historique. Jusqu'à 10 Go sur le disque.",
        apiOnly: "Décharger uniquement les modèles chargés par l'API",
        apiOnlyDescription:
          "Le déchargement en cas d'inactivité laisse en mémoire un modèle que vous avez chargé depuis Unsloth et ne libère que ceux chargés par une requête API.",
      },
      previewSharing: {
        sectionTitle: "Partage de l'aperçu",
        enableLabel: "Liens d'aperçu publics",
        enableDescription:
          "Permettre à quiconque disposant d'un lien signé de discuter avec un modèle finalisé, sans connexion. Désactivez cette option pour mettre l'aperçu public hors ligne ; les liens partagés cesseront de fonctionner.",
        loadError: "Échec du chargement des paramètres de partage d'aperçu.",
        saveError:
          "Échec de l'enregistrement des paramètres de partage d'aperçu.",
        revokeLabel: "Révoquer tous les liens d'aperçu",
        revokeDescription:
          "Renouveler le secret de signature pour que tous les liens partagés cessent de fonctionner. Les liens copiés après ce renouvellement continueront de fonctionner.",
        revokeAction: "Révoquer les liens",
        revoking: "Révocation...",
        revokeConfirmTitle: "Révoquer tous les liens d'aperçu ?",
        revokeConfirmDescription:
          "Tous les liens d'aperçu que vous avez partagés cesseront de fonctionner immédiatement. Cette action est irréversible.",
        revokeConfirmAction: "Révoquer tous les liens",
        revoked: "Tous les liens d'aperçu ont été révoqués",
        revokeError: "Impossible de révoquer les liens d'aperçu",
      },
      notifications: {
        sectionTitle: "Notifications",
        showLlamaUpdates: "Notifications de mise à jour de llama.cpp",
        showLlamaUpdatesDescription:
          "Notifier lorsqu'une nouvelle version de llama.cpp est disponible pour exécuter de nouveaux modèles. Désactivez si vous ne faites que de l'entraînement.",
        showLoadedModels: "Indicateur des modèles chargés",
        showLoadedModelsDescription:
          "Affiche une petite carte en bas à droite listant tous les modèles actuellement en mémoire (chat, voix, image, vidéo), avec un bouton pour éjecter chacun d'eux.",
      },
      startup: {
        sectionTitle: "Démarrage",
        launchAtLogin: "Lancer Unsloth à la connexion",
        launchAtLoginDescription:
          "Démarre Unsloth en arrière-plan lorsque vous vous connectez. Il reste dans la barre de menus ou la zone de notification jusqu'à ce que vous l'ouvriez.",

        closeToTray: "Fermer dans la zone de notification",
        closeToTrayDescription:
          "Laisser Unsloth et son serveur fonctionner en arrière-plan lorsque vous fermez la fenêtre principale.",
        closeToTraySaveError:
          "Impossible de mettre à jour le réglage de fermeture dans la zone de notification.",
        loadError:
          "Impossible de charger le réglage de lancement à la connexion.",
        saveError:
          "Impossible de mettre à jour le réglage de lancement à la connexion.",
      },
      downloads: {
        sectionTitle: "Téléchargements",
        transport: "Transport de téléchargement",
        transportDescription:
          "Comment les fichiers de modèles et de jeux de données arrivent depuis Hugging Face. HTTPS reprend là où il s'est arrêté ; Xet est souvent plus rapide au premier téléchargement mais recommence le fichier en cas d'annulation.",
        transportHint:
          "HTTPS, c'est du TLS classique : tous les réseaux, proxys et VPN l'autorisent, un transfert annulé ou coupé reprend à partir des octets déjà écrits et la mémoire reste stable. Xet récupère des blocs dédupliqués, donc un dépôt partageant des données avec un autre déjà présent peut arriver bien plus vite, mais il exige hf_xet, consomme plus de RAM, et une annulation jette le fichier en cours. Auto décide selon la machine : il pèse la RAM et les blocages récents de Xet ici, puis se rabat sur HTTPS.",
        https: "HTTPS",
        xet: "Xet",
        auto: "Auto",
        httpsHint:
          "TLS standard. Reprend après une annulation, fonctionne sur tous les réseaux, mémoire stable.",
        transportDescriptionNoResume:
          "Comment les fichiers de modèles et de jeux de données sont téléchargés depuis Hugging Face. Sur cette installation, aucun transport ne peut reprendre : un téléchargement annulé recommence ; Xet est souvent plus rapide au premier téléchargement.",
        httpsHintNoResume:
          "TLS standard. Fonctionne sur tous les réseaux, consommation mémoire stable. Cette installation ne peut pas reprendre un téléchargement annulé.",
        xetHint:
          "Transfert par blocs dédupliqués. Souvent plus rapide sur un premier téléchargement, recommence le fichier si vous annulez, demande plus de mémoire.",
        autoHint:
          "Choisit selon la machine et passe à HTTPS si Xet se bloque ou échoue ici.",
        autoCurrently: "Auto utilise {transport} sur cette machine.",
        xetMissing: "Xet est indisponible car hf_xet n'est pas installé.",
      },
      uploads: {
        sectionTitle: "Téléversements",
        maxUploadSize: "Limite de téléversement du jeu de données d'entraînement",
        maxUploadSizeDescription: "La valeur par défaut est {defaultSize} Mo.",
      },
      rag: {
        sectionTitle: "Documents et RAG",
        embeddingModel: "Modèle d'embedding",
        embeddingModelDescription:
          "Modèle Hugging Face ou chemin local utilisé pour indexer et rechercher vos documents. La valeur par défaut est {defaultModel}.",
        searchPlaceholder: "Rechercher n'importe quel modèle sur HF",
        reindexWarning:
          "N'affecte que les documents nouvellement indexés. Téléversez à nouveau les documents existants après avoir changé de modèle.",
        emptyError:
          "Saisissez un identifiant de modèle Hugging Face ou un chemin local.",
        loadError: "Échec du chargement du paramètre du modèle d'embedding.",
        saveError: "Échec de l'enregistrement du modèle d'embedding.",
        saved: "Modèle d'embedding enregistré.",
        saveAnyway: "Enregistrer quand même",
        recommended: "Recommandé",
        onDevice: "Sur l'appareil",
        searching: "Recherche sur Hugging Face…",
        checking: "Vérification…",
        noResults: "Aucun modèle d'embedding trouvé",
        download: "Télécharger",
        unload: "Décharger",
        unloadFailed: "Impossible de décharger le modèle d'embedding",
        downloadingStatus: "Téléchargement…",
        notDownloaded: "Non téléchargé",
        notDownloadedSized: "Non téléchargé · {size}",
        loaded: "Chargé",
        downloading: "Téléchargement de {model}",
        downloadingDescription:
          "La progression s'affiche dans le panneau des téléchargements. L'indexation l'utilisera une fois terminé.",
        downloadFailed: "Impossible de démarrer le téléchargement",
        downloadConflict: "Reprenez ce téléchargement depuis le Hub",
        downloadBusy: "Téléchargement déjà en cours",
      },
      storage: {
        sectionTitle: "Stockage",
        modelsFolder: "Dossier des modèles",
        modelsFolderDescription:
          "Emplacement de stockage des modèles téléchargés.",
        openAction: "Ouvrir",
        copyAction: "Copier le chemin",
        copied: "Chemin copié",
        openError: "Impossible d'ouvrir le dossier",
        copyError: "Impossible de copier le chemin",
      },
      resetPreferences: {
        sectionTitle: "Zone de danger",
        label: "Réinitialiser toutes les préférences locales",
        description:
          "Efface les préférences locales uniquement. Les discussions, l'accès API et les paramètres stockés en base de données sont conservés.",
        action: "Réinitialiser les préférences",
        confirmTitle: "Réinitialiser toutes les préférences locales ?",
        confirmDescription:
          "Efface les préférences locales uniquement et recharge Unsloth. Les discussions, l'accès API et les paramètres stockés en base de données sont conservés.",
        confirmAction: "Réinitialiser et recharger",
      },
      permissions: {
        sectionTitle: "Autorisations",
        bypassLabel: "Autorisations des outils",
        bypassDescription:
          "Comment Unsloth approuve les appels d'outils de la discussion (terminal, python, web, MCP) avant leur exécution. Le mode « Full access » désactive les demandes d'approbation et le bac à sable d'exécution du code.",
      },
    },
    profile: {
      title: "Profil",
      description: "Comment votre profil apparaît dans Unsloth.",
      changePicture: "Changer la photo de profil",
      displayName: "Nom affiché",
      nickname: "Comment Unsloth doit-il vous appeler ?",
      nicknamePlaceholder: "Surnom",
      nicknameSaved: "Nom préféré enregistré",
      avatarShape: "Forme de l'avatar",
      avatarShapeCircle: "Cercle",
      avatarShapeRounded: "Arrondi",
      chooseSloth: "Ou choisissez un paresseux",
      nameSaved: "Nom de profil enregistré",
      namePersistErrorTitle: "Impossible d'enregistrer le nom de profil",
      namePersistErrorDescription:
        "Le nom a été mis à jour pour cette session, mais risque de ne pas être conservé après le rechargement.",
      photoUpdated: "Photo de profil mise à jour",
      photoPersistErrorTitle: "Impossible d'enregistrer la photo de profil",
      photoPersistErrorDescription:
        "La photo a été mise à jour pour cette session, mais risque de ne pas être conservée après le rechargement.",
      photoUpdateErrorTitle: "Impossible de mettre à jour la photo de profil",
      imageUseError: "Impossible d'utiliser cette image.",
      uploadPhoto: "Importer une photo",
      removePhoto: "Retirer",
      pictureOptions: "Options de la photo de profil",
      greetingSloth: "Paresseux dans le message d'accueil",
      greetingSlothDescription:
        "Afficher le paresseux dans le message d'accueil de la discussion.",
      noPicture: "Aucune photo de profil",
      noneLabel: "Aucun",
      stats: {
        title: "Vos statistiques",
        subtitle:
          "Tout ce qui suit est calculé à partir de votre propre historique. Rien n'est collecté ni envoyé à Unsloth.",
        retry: "Réessayer",
        privacyNote:
          "Les statistiques sont calculées à partir de l'historique local des discussions, de l'utilisation de l'API et des entraînements de votre installation Unsloth. Les requêtes, réponses et clés API ne sont jamais stockées pour les statistiques. Rien n'est envoyé à Unsloth ni à un tiers.",
        emptyChats:
          "Aucune utilisation du chat ou de l'API pour le moment. Lancez une conversation ou effectuez une requête authentifiée vers l'API locale.",
        lifetimeTokens: "Tokens cumulés",
        peakTokens: "Jour record",
        longestChat: "Discussion la plus longue",
        currentStreak: "Série en cours",
        longestStreak: "Plus longue série",
        activityTitle: "Activité en tokens",
        activityDescription: "{total} au cours des {weeks} dernières",
        mode: {
          daily: "Quotidienne",
          weekly: "Hebdomadaire",
          cumulative: "Cumulée",
        },
        cellTooltip: "{date} · {tokens}, {messages}",
        weekTooltip: "Semaine du {date} · {tokens}",
        less: "Moins",
        more: "Plus",
        insightsTitle: "Analyse de l'activité",
        totalChats: "Discussions au total",
        totalMessages: "Messages au total",
        tokensIn: "Tokens envoyés",
        tokensOut: "Tokens générés",
        totalTokens: "Total des tokens",
        studioChatTokens: "Tokens de Unsloth Chat",
        apiTokens: "Tokens API",
        cachedTokens: "Tokens mis en cache",
        cachedValue: "{tokens} ({percent} % des tokens d'entrée)",
        avgTokensPerChat: "Moyenne de tokens par discussion",
        timeInChat: "Temps passé en discussion",
        activeDays: "Jours actifs",
        toolCalls: "Appels d'outils",
        attachments: "Fichiers joints",
        avgSpeed: "Vitesse moyenne",
        bestSpeed: "Réponse la plus rapide",
        firstToken: "Temps moyen jusqu'au premier token",
        tokensPerSecond: "{value} tok/s",
        topModelsTitle: "Modèles les plus utilisés",
        topModelsDescription: "Classés par tokens échangés",
        modelSummary: "{tokens} · {messages}",
        noModels: "Aucune utilisation de modèle enregistrée pour l'instant.",
        trainingTitle: "Entraînement",
        trainingDescription: "Sessions de fine-tuning de cet espace de travail",
        trainingRuns: "Sessions",
        trainingCompleted: "Terminées",
        trainingSteps: "Étapes",
        trainingTokens: "Tokens d'entraînement",
        trainingTime: "Temps d'entraînement",
        bestLoss: "Perte minimale",
        runSteps: "{steps}",
        runLoss: "perte {loss}",
      },
    },
    appearance: {
      title: "Apparence",
      description: "L'apparence d'Unsloth sur cet appareil.",
      theme: {
        title: "Thème",
        label: "Mode de couleur",
        description: "Clair, sombre ou selon votre système.",
        system: "Système",
        light: "Clair",
        dark: "Sombre",
      },
      palette: {
        label: "Palette",
        description: "Couleurs utilisées dans Unsloth, en mode clair et sombre.",
        standard: "Standard",
        classic: "Classique",
        minimal: "Minimale",
      },
      custom: {
        reset: "Réinitialiser",
        resetAll: "Réinitialiser la personnalisation",
        preferencesTitle: "Préférences",
        colors: {
          lightGroup: "Thème clair",
          darkGroup: "Thème sombre",
          accent: "Accent",
          background: "Arrière-plan",
          foreground: "Premier plan",
        },
        fontDefault: "Par défaut",
        fontBundledGroup: "Intégrées",
        fontImportedGroup: "Importées",
        fontDeviceGroup: "Sur cet appareil",
        fontFolderGroup: "Depuis un dossier",
        fontDeviceLoading: "Recherche des polices de l'appareil…",
        fontSearch: "Rechercher des polices…",
        fontNoResults: "Aucune police trouvée.",
        colorPicker: {
          hue: "Teinte",
          hex: "Couleur hexadécimale",
          eyedropper: "Sélectionner une couleur à l'écran",
        },
        uiFont: {
          label: "Police de l'interface",
        },
        headingFont: {
          label: "Police des titres",
        },
        chatFont: {
          label: "Police de la discussion",
        },
        codeFont: {
          label: "Police du code",
        },
        importFont: {
          upload: "Importer",
          scanFolder: "Sélectionner un dossier",
          alreadyAvailable:
            "Cette police est déjà disponible ; la copie existante est utilisée.",
          folderNoFonts: "Aucun fichier de police trouvé dans ce dossier.",
          remove: "Retirer",
          errorInvalidType:
            "Type de fichier non pris en charge. Utilisez .woff2, .woff, .ttf ou .otf.",
          errorTooLarge: "Le fichier de police est trop volumineux (1,5 Mo max.).",
          errorLimit: "Vous pouvez importer jusqu'à 3 polices.",
          errorStorageFull:
            "Stockage local insuffisant pour cette police. Retirez d'abord une police importée.",
          errorFailed: "Impossible de charger ce fichier de police.",
        },
        uiFontSize: {
          label: "Taille de police de l'interface",
          description: "Ajustez la taille de base utilisée pour l'interface Unsloth.",
        },
        codeFontSize: {
          label: "Taille de police du code",
          description: "Ajustez la taille de base utilisée pour le code.",
        },
        fontSmoothing: {
          label: "Lissage des polices",
          description: "Utiliser le lissage des polices.",
        },
        contrast: {
          label: "Contraste",
          description: "Intensité des bordures et du texte secondaire.",
        },
        reduceMotion: {
          label: "Réduire les animations",
          description: "Réduire les animations ou suivre votre système.",
          system: "Système",
          on: "Activé",
          off: "Désactivé",
        },
        pointerCursors: {
          label: "Utiliser un curseur en forme de main",
          description:
            "Afficher un curseur en forme de main au survol des éléments interactifs.",
        },
      },
      language: {
        title: "Langue",
        label: "Langue d'affichage",
        description: "La langue utilisée par Unsloth.",
        autoDetect: "Détection automatique",
      },
      layout: {
        title: "Disposition",
        compactSidebar: "Épingler la barre latérale par défaut",
        compactSidebarDescription:
          "Garder la barre latérale déployée au lieu de la réduire en icônes.",
      },
      sidebarNav: {
        title: "Navigation de la barre latérale",
        description:
          "Épinglez et réorganisez les onglets de la barre latérale. Les onglets non épinglés sont regroupés dans le menu « Plus » ; s'il ne reste qu'un seul onglet non épinglé, il est masqué au lieu de créer un menu à une seule entrée. « Nouvelle discussion » reste fixe.",
        dragToReorder: "Faites glisser pour réorganiser",
        pinToSidebar: "Épingler {name} dans la barre latérale",
        moreHolds: "Plus ({count})",
      },
      sidebarMenu: {
        title: "Menu de la barre latérale",
        description:
          "Affichez, masquez et réorganisez les éléments du menu de profil de la barre latérale. Paramètres, Aide, Se déconnecter et Arrêter restent fixes.",
        darkModeToggle: "Bascule du mode sombre",
        dragToReorder: "Faites glisser pour réorganiser",
      },
    },
    resources: {
      title: "Système",
      description:
        "Surveillez le matériel et le stockage de ce serveur Unsloth.",
      liveUpdates: "Mises à jour en direct",
      floatingWindow: "Fenêtre flottante",
      disableOverlay: "Désactiver la superposition",
      liveMonitor: {
        title: "Moniteur en direct",
        apiTitle: "Moniteur d’API",
        summary: "Requêtes en cours, erreurs et utilisation des jetons",
        status: "{active} actives · {recent} récentes · {model}",
        noModelLoaded: "aucun modèle chargé",
        autoOpen: "Afficher automatiquement le moniteur flottant",
        autoOpenDescription:
          "Ouvre un petit panneau lorsque l’API reçoit du trafic.",
        cpu: "CPU",
        ram: "RAM",
        disk: "Disque",
        vram: "VRAM",
        cpuCores: "{logical} cœurs logiques / {physical} physiques",
        currentLoad: "Charge actuelle",
        free: "Disponible : {value}",
        noGpu: "Aucun GPU visible",
      },
      gpu: {
        title: "Périphériques GPU",
        ggufInference: "Inférence GGUF",
        unavailable: "indisponible",
        detecting: "Recherche de GPU...",
        unreadable: "Impossible de lire le matériel de ce serveur.",
        noGpu:
          "Aucun GPU visible n'a été détecté. Seules les ressources du CPU sont affichées ci-dessus.",
        unknownDevice: "GPU inconnu",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "Utilisé : {value}",
        free: "Disponible : {value}",
        total: "{value} au total",
      },
      llamaBackend: {
        title: "Moteur d'inférence GGUF",
        label: "Backend de calcul",
        description: "Le backend utilisé par llama.cpp pour exécuter les modèles GGUF.",
        runningOn: "llama.cpp fonctionne actuellement sur {backend}.",
        hint: "Installe la version de llama.cpp pour ce backend et la conserve lors des mises à jour. Utile si le choix automatique plante ou si votre pilote GPU ne le prend pas en charge. Seuls les backends disposant d'une version pour cette machine sont proposés ; l'entraînement n'est pas affecté.",
        autoWith: "Automatique ({backend})",
        apply: "Appliquer",
        applying: "Installation...",
        applyHint: "Télécharge la nouvelle version et redémarre llama.cpp. Un modèle chargé sera déchargé.",
        applyHintWithSize: "Télécharge {size} et redémarre llama.cpp. Un modèle chargé sera déchargé.",
        switchedTo: "llama.cpp fonctionne maintenant sur {backend}.",
        switchFailed: "Impossible de changer le backend llama.cpp.",
        switchInterrupted: "Le changement a été interrompu avant d’être terminé.",
        envLocked: "Fixé à {backend} par la variable d'environnement UNSLOTH_LLAMA_CPP_BACKEND, qui prévaut sur ce réglage.",
        customPath: {
          label: "Dossier llama.cpp personnalisé",
          description: "Utilisez votre propre build de llama-server.",
          hint: "Choisissez le dossier llama.cpp contenant llama-server, ou un build où il se trouve sous build/bin. Le runtime personnalisé est utilisé pour le chat GGUF, les embeddings et les modèles vocaux compatibles. Les variables d'environnement restent prioritaires.",
          automatic: "Automatique (fourni)",
          bundled: "Utilise le runtime llama.cpp installé par Unsloth.",
          active: "Votre llama-server personnalisé sera utilisé au prochain chargement de modèle.",
          environmentManaged: "Géré par la variable d'environnement {variable}.",
          missingBinary: "llama-server n'est plus disponible dans ce dossier. Choisissez un autre dossier ou utilisez le runtime fourni.",
          reloadRequired: "Rechargez le modèle pour utiliser le llama-server sélectionné.",
          change: "Modifier",
          saving: "Enregistrement...",
          useBundled: "Utiliser la version fournie",
          chooseTitle: "Choisir le dossier llama.cpp",
          chooseAction: "Utiliser ce dossier",
          saved: "Dossier llama.cpp mis à jour",
          saveError: "Impossible de mettre à jour le dossier llama.cpp",
        },
        backends: {
          auto: "Automatique",
          cpu: "CPU",
          cuda: "CUDA",
          rocm: "ROCm",
          vulkan: "Vulkan",
          metal: "Metal",
        },
        unsupported: {
          notInstalled: "Aucune installation llama.cpp gérée n'a été trouvée, il n'y a donc pas de backend à changer.",
          localLink: "llama.cpp est un dossier local que vous avez lié vous-même ; Unsloth ne le remplacera pas.",
          sourceBuild: "Ce llama.cpp a été compilé depuis les sources, son backend ne peut pas être changé ici.",
          customPath: "Un dossier llama.cpp personnalisé est sélectionné. Son build détermine le backend de calcul.",
          unresolved: "Impossible de vérifier les backends disponibles. Vérifiez votre connexion et réessayez.",
        },
        // Non affiché : termes supplémentaires pour la recherche dans les réglages.
        llamaBackendKeywords:
          "llama.cpp backend gguf inférence cuda rocm hip vulkan metal cpu gpu accélérateur prebuilt changer moteur",
      },
      modelMemory: {
        title: "Mémoire du modèle",
        keepResident: "Conserver le modèle dans la mémoire du GPU",
        keepResidentDescription: "Reste en VRAM entre les messages.",
        keepResidentHint: "Ne rend pas les poids à la RAM système tant que le modèle reste chargé. Désactive le déchargement automatique en veille et, lorsque les poids résident réellement en RAM hôte (mémoire unifiée ou déchargement GPU partiel), passe aussi --mlock afin que le système ne les décharge pas pour les retransférer au prochain message.",
        noRamReserve: "Ne pas réserver de RAM système pour le modèle",
        noRamReserveDescription: "Ne garde aucune copie complète en RAM.",
        noRamReserveHint: "Transfère les poids vers la VRAM au lieu d'en garder une copie complète en RAM. Conserve le chargement mappé en mémoire de llama.cpp et supprime --no-mmap et --mlock.",
        mlockVetoed: "--mlock reste désactivé : épingler le modèle réserverait de la RAM pour l'intégralité de celui-ci. Le déchargement automatique en veille reste désactivé.",
        memlockCapped: "Ce système limite la mémoire verrouillée à {limit}. Un modèle plus grand ne sera pas entièrement épinglé ; augmentez la limite avec ulimit -l.",
        reloadRequired: "Rechargez le modèle pour appliquer les nouvelles options de mémoire.",
        loadError: "Impossible de charger les paramètres de mémoire du modèle",
        saveError: "Impossible d'enregistrer les paramètres de mémoire du modèle",
        // Not rendered: extra terms the settings search matches these rows on.
        modelMemoryKeywords:
          "mlock memlock ulimit vram gpu memoire ram resident epingler verrouiller garder charge decharger inactif mmap no-mmap load-mode pagination echange",
      },
      storage: {
        title: "Stockage",
        systemDisk: "Disque système",
        diskUsage: "Espace utilisé : {used} / Total : {total}",
        diskFree: "Espace libre : {free}",
        modelsFolder: "Dossier des modèles",
        modelsFolderKeywords:
          "modeles modèles dossier repertoire répertoire chemin emplacement telechargements téléchargements cache stockage disque lecteur deplacer déplacer changer models folder path hugging face",
        modelsFolderDescription: "Où sont stockés les modèles téléchargés.",
        modelsFolderHint: "Où sont stockés les modèles téléchargés. Modifiez-le pour garder les modèles hors de votre disque système. S'applique uniquement aux nouveaux téléchargements. Les modèles que vous avez déjà restent où ils sont.",
        openAction: "Ouvrir",
        copyAction: "Copier le chemin",
        copied: "Chemin copié",
        openError: "Impossible d'ouvrir le dossier",
        copyError: "Impossible de copier le chemin",
        futureDownloads: "Nouveaux téléchargements uniquement",
        environmentManaged: "Géré par la variable d'environnement {variable}.",
        locationFree: "Espace libre : {free}",
        changeAction: "Modifier",
        resetAction: "Utiliser la valeur par défaut",
        chooseTitle: "Choisir l'emplacement de téléchargement des modèles",
        chooseAction: "Utiliser pour les prochains téléchargements",
        cacheSaved: "Emplacement de téléchargement des modèles mis à jour",
        cacheSaveError:
          "Impossible de mettre à jour l'emplacement de téléchargement des modèles",
        cachePickerError: "Impossible d'ouvrir le sélecteur de dossier",
      },
      environment: {
        title: "Environnement",
        backend: "Backend",
        python: "Python",
        torch: "Torch",
        transformers: "Transformers",
        uptime: "Temps de fonctionnement",
        processMemory: "Mémoire du processus",
        notInstalled: "Non installé",
        unknown: "Inconnu",
        vramWithShared: "{vram} de VRAM + {shared} de mémoire partagée",
      },
    },
    agents: {
      title: "Agents",
      description:
        "Connectez des agents de codage comme Claude Code et Codex à un modèle local avec unsloth start.",
      intro:
        "connecte Claude Code, Codex, Hermes, OpenClaw, OpenCode et d'autres agents à un modèle servi localement par Unsloth, entièrement hors ligne. Il lance un serveur compatible OpenAI et ne touche jamais aux fichiers de configuration de votre agent.",
      readDocs: "Lire la documentation",
      copy: "Copier",
      copied: "Copié",
      commandBuilder: "Générateur de commande",
      agent: "Agent de codage",
      model: "Modèle",
      searchModels: "Rechercher des modèles GGUF...",
      noModels: "Aucun modèle GGUF correspondant.",
      showingModels:
        "Affichage de {shown} résultats sur {total}. Continuez à taper pour affiner la liste.",
      quantization: "Quantification",
      loadingQuantizations: "Chargement des quantifications...",
      noQuantizations: "Aucune quantification distincte",
      recommended: "Recommandé",
      downloaded: "Téléchargé",
      quantizationLoadError:
        "Impossible de charger toutes les quantifications. La commande utilisera la valeur de modèle disponible.",
      generatedCommand: "Commande générée",
      docs: "Documentation",
      agentDocs: "Ouvrir la documentation de configuration de {agent}",
      copyGeneratedCommand: "Copier la commande générée",
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
        "Codex nécessite un modèle GGUF servi par llama-server. Les autres agents peuvent aussi utiliser des modèles basés sur transformers ; retirez --model pour utiliser le modèle déjà chargé dans Unsloth.",
      subagent: {
        title: "Utiliser un modèle local comme sous-agent",
        description:
          "Gardez {agent} sur son modèle actuel et déléguez certaines tâches à ce modèle Unsloth local.",
        setupCommand: "Commande de configuration",
        copySetupCommand: "Copier la commande de configuration du sous-agent",
        usagePrompt: "Ensuite, dans {agent}, tapez :",
        copyUsagePrompt: "Copier le prompt d'utilisation du sous-agent",
        defaultPrompt: "Lance un agent local pour implémenter cette fonction.",
        opencodePrompt: "@unsloth trouve la cause de cet échec de test",
      },
      quickstart: {
        title: "Construire une commande",
        description:
          "Lancez un agent sur le modèle actuellement chargé dans Unsloth. Chargez d'abord un modèle, puis remplacez claude par n'importe quel agent pris en charge ci-dessous.",
        noneDetected:
          "Aucune CLI d'agent prise en charge n'a été trouvée dans votre PATH.",
        installed: "Installé",
      },
      supportedAgents: {
        title: "Agents pris en charge",
        description: "Chaque agent se lance avec sa propre commande :",
        requiresGguf: "Nécessite un modèle GGUF",
      },
      models: {
        title: "Choisir un modèle",
        description:
          "Utilisez --model pour choisir un modèle et une quantification, et --context-length pour définir la fenêtre de contexte. Utilisez un suffixe de quantification ou l'option explicite --gguf-variant.",
        suffixLabel: "Avec un suffixe de quantification",
        variantLabel: "Avec une option de variante explicite",
      },
      options: {
        title: "Options courantes",
        description:
          "Les options Unsloth sont analysées en premier ; tout ce qu'Unsloth ne reconnaît pas est transmis tel quel à l'agent.",
        model:
          "Sélectionne un modèle. Sans --model, unsloth start utilise le modèle actuellement chargé dans Unsloth et échoue si aucun modèle n'est chargé.",
        contextLength:
          "Définit la longueur de contexte demandée (alias : --max-seq-length).",
        ggufVariant: "Choisit la variante de quantification GGUF.",
        loadIn4bit:
          "Active ou désactive le chargement en 4 bits pour les modèles Hugging Face.",
        tensorParallel:
          "Active ou désactive le parallélisme de tenseurs sur plusieurs GPU.",
        serve: "Active ou désactive le serveur local automatique.",
        launch:
          "Lance l'agent, ou affiche simplement la commande et l'environnement.",
        persist:
          "Conserve d'une exécution à l'autre les données d'agent gérées par Unsloth.",
        asSubagent:
          "Garde l'agent parent sur son modèle actuel et enregistre Unsloth comme sous-agent local (Claude Code, Codex et OpenCode).",
        apiKey:
          "Fournit votre clé API Unsloth, ou lit la variable UNSLOTH_API_KEY.",
        reasoning:
          "Utiliser le raisonnement dans le chat : on, off ou auto. Auto suit le modèle de chat du modèle, ce qui veut généralement dire on.",
        reasoningEffort:
          "Effort de raisonnement transmis au modèle de chat du modèle, par exemple medium. Les niveaux dépendent du modèle, utilisez-en un qu'il accepte. Sans valeur, le niveau du modèle de chat s'applique.",
        yolo:
          "Ignore les demandes d'autorisation. À n'utiliser que dans des environnements de confiance.",
      },
      remote: {
        title: "Se connecter à un Unsloth Studio distant",
        description:
          "Faites pointer unsloth start vers un Unsloth Studio exécuté ailleurs en définissant ces variables avant le lancement (ou passez --api-key directement) :",
      },
      passthrough: {
        title: "Transmettre des arguments à l'agent",
        description:
          "Les arguments placés après les options Unsloth sont transmis à l'agent lui-même, donc les commandes natives comme resume fonctionnent toujours :",
      },
      dryRun: {
        title: "Prévisualiser sans lancer",
        description:
          "Ajoutez --no-launch pour afficher l'environnement et la commande au lieu de lancer l'agent. Si --model est défini, le modèle peut tout de même être résolu et chargé.",
      },
    },
    chat: {
      projectsSection: "Afficher la section Projets",
      projectsSectionDescription:
        "Regroupe les discussions de projet sous un titre Projets. Désactivez cette option pour les lister dans Récents.",
      title: "Discussion",
      description: "Personnalisez le fonctionnement du chat sur cet appareil.",
      modelSelection: {
        title: "Paramètres de sélection du modèle",
        expandQuantizations: "Développer les quantifications",
        expandQuantizationsDescription:
          "Activé : les modèles GGUF de « On Device » affichent immédiatement leurs quantifications. Désactivé : cliquez sur un modèle pour afficher ses quantifications.",
        showAllQuantizations: "Afficher toutes les quantifications",
        showAllQuantizationsDescription:
          "Activé : affiche toutes les quantifications de « On Device », y compris celles qui ne sont pas téléchargées. Désactivé : affiche uniquement les quantifications téléchargées.",
        showMemoryBar: "Afficher la barre d’utilisation de la VRAM",
        showMemoryBarDescription:
          "Affiche sous la ligne de chaque modèle téléchargé son utilisation estimée de la VRAM : poids, cache KV à la longueur de contexte avec laquelle il sera chargé, et toute réserve de brouillon spéculatif.",
      },
      menu: {
        title: "Menu du chat",
        description:
          "Épinglez des éléments dans le menu latéral + du chat. Les autres seront placés dans « Plus ».",
        chatWithFiles: "Discuter avec des fichiers (RAG)",
        mcp: "MCP",
        savedPrompts: "Invites enregistrées",
        compareChat: "Comparer le chat",
        exportChat: "Exporter le chat",
      },
      pastedTextThreshold: "Condenser les collages longs",
      pastedTextThresholdDescription: "Le texte collé plus long que cette valeur devient une pièce jointe .txt au lieu de remplir le champ de message. Appuyez sur {shortcut} pour coller quand même dans le champ de message.",
      pastedTextThresholdOff: "Désactivé",
      showResponseModel: "Afficher le modèle de réponse",
      showResponseModelDescription:
        "Afficher les métadonnées du modèle dans les réponses de l’assistant.",
      modelDisclaimer: "Afficher l'avertissement du modèle",
      modelDisclaimerDescription:
        'Afficher "Les LLM peuvent faire des erreurs" sous la zone de discussion.',
      projectAttachments: "Partager les fichiers dans tout le projet",
      projectAttachmentsDescription:
        "Valeur par defaut pour les fichiers joints dans une discussion appartenant a un projet : les indexer pour tout le projet afin que chaque discussion puisse les utiliser. Chaque discussion peut le modifier depuis le menu des pieces jointes.",
      rememberParamsPerModel: "Mémoriser les réglages par modèle",
      rememberParamsPerModelDescription:
        "Changer de modèle restaure la température, le prompt et les autres réglages utilisés en dernier avec ce modèle. Désactivé, un seul jeu de réglages s'applique à tous les modèles.",
      autoCompact: "Compacter automatiquement les longues discussions",
      autoCompactDescription:
        "Lorsqu’une discussion GGUF locale atteint la longueur de contexte définie, supprimez les anciens tours au lieu de renvoyer une erreur. Ce réglage ne dépend pas de la VRAM libre.",
      compactionStyle: "Lorsque le contexte est plein",
      compactionStyleDescription:
        "La valeur par défaut du serveur conserve UNSLOTH_CONTEXT_POLICY. Réinitialiser la discussion garde le dernier tour et les instructions permanentes. Une fenêtre glissante supprime les tours les plus anciens et peut conserver davantage d’historique récent.",
      compactionStyleInherit: "Utiliser la valeur du serveur",
      compactionStyleCheckpoint: "Réinitialiser la discussion",
      compactionStyleRollingDefault:
        "Supprimer les anciens tours (~25 % d’espace supplémentaire)",
      compactionStyleRolling10:
        "Supprimer les anciens tours (~10 % d’espace supplémentaire)",
      compactionStyleRolling5:
        "Supprimer les anciens tours (~5 % d’espace supplémentaire)",
      compactionStyleRollingNone:
        "Supprimer les anciens tours (sans réduction supplémentaire)",
      autoCompactKeywords:
        "compaction automatique contexte fenêtre tronquer glissante point de contrôle marge compaction rolling checkpoint headroom",
      thinking: {
        collapseByDefault: "Replier la réflexion par défaut",
        collapseByDefaultDescription:
          "Garde la réflexion repliée pendant que le modèle réfléchit, au lieu de l’ouvrir automatiquement. Dépliez un bloc pour le lire.",
      },
      tools: {
        collapseByDefault: "Replier l’activité des outils par défaut",
        collapseByDefaultDescription:
          "Garde les entrées et sorties des outils repliées pendant leur exécution. Dépliez une ligne d’outil pour l’examiner.",
      },
      webSearch: {
        title: "Recherche web",
        images: "Afficher les images de la recherche web",
        imagesDescription:
          "Permet à la recherche web de renvoyer des images et en récupère une pour chaque élément listé dans une réponse. Unsloth télécharge et redimensionne les vignettes : le navigateur ne contacte jamais les hébergeurs d'images.",
      },
      artifacts: {
        title: "Canvas",
        collapseHtmlBlocks: "Réduire les blocs HTML",
        collapseHtmlBlocksDescription:
          "Le mode Canvas réduit automatiquement les pages HTML complètes. Activez cette option pour réduire également les documents HTML placés dans des blocs de code lorsque Canvas est désactivé.",
        allowNetworkAccess: "Autoriser l'accès réseau du canvas",
        allowNetworkAccessDescription:
          "Permettre aux aperçus Canvas de charger des scripts, des styles, des polices, des médias et d'autres ressources depuis des CDN. Laissez cette option désactivée pour des aperçus entièrement hors ligne.",
        blockedBanner: "{count} ressource externe bloquée depuis {hosts}.",
        blockedBannerPlural: "{count} ressources externes bloquées depuis {hosts}.",
        blockedBannerAction: "Autoriser pour ce Canvas",
      },
      data: "Données",
      exportHistory: "Exporter l'historique des discussions",
      exportHistoryDescription:
        "Télécharger toutes les discussions et messages au format JSON.",
      exportAction: "Exporter",
      exportingAction: "Exportation...",
      exportConversations: "Exporter Récents et Projets",
      exportConversationsDescription:
        "Télécharger Récents ou Récents plus les discussions de projet au format Training JSONL, CSV ou JSONL ShareGPT, combinés ou par discussion. Message JSONL est disponible uniquement par discussion.",
      exportConversationsAction: "Exporter",
      exportScopeRecents: "Récents",
      exportScopeAll: "Récents + Projets",
      exportCombinedSuffix: "(combiné)",
      exportPerChatSuffix: "(par discussion)",
      importChats: "Importer des discussions",
      importChatsDescription:
        "Importez un export Open WebUI, JSONL, NDJSON ou CSV dans Récents.",
      importChatsAction: "Importer",
      importNoConversations: "Aucune conversation trouvée dans le fichier.",
      importedOneChat: "1 conversation importée dans Récents.",
      importedChatCount: "{count} conversations importées dans Récents.",
      importingChats: "Import des discussions : {count} jusqu'ici ({percent}%)...",
      importedChatCountPartial: "{count} conversations importées dans Récents ; {failed} n'ont pas pu être enregistrées.",
      importFailed: "Échec de l'importation.",
      clearHistory: "Effacer l'historique des discussions",
      clearHistoryDescription:
        "Supprimer l'historique des discussions de cet appareil.",
      clearAction: "Effacer",
      clearAllChats: "Effacer toutes les discussions",
      clearAllChatsDescription:
        "Supprimer définitivement toutes les discussions de cet appareil.",
      noChatsToClear: "Aucune discussion à effacer.",
      clearOneChatDescription:
        "Supprimer définitivement la seule discussion de cet appareil.",
      clearChatCountDescription:
        "Supprimer définitivement les {count} discussions de cet appareil.",
      clearChatsAction: "Effacer les discussions",
      clearOneChatTitle: "Effacer 1 discussion ?",
      clearChatsTitle: "Effacer {count} discussions ?",
      clearChatsConfirmDescription:
        "Supprime définitivement toutes les discussions de cet appareil. Cette action est irréversible.",
      clearingAction: "Effacement...",
      clearOneChatAction: "Effacer 1 discussion",
      clearChatCountAction: "Effacer {count} discussions",
      clearedAllChats: "Toutes les discussions ont été effacées",
      clearedOneChat: "1 discussion effacée",
      clearedChatCount: "{count} discussions effacées",
      someChatsCouldNotBeCleared:
        "Certaines discussions n'ont pas pu être effacées",
      chatsClearedRemainOne:
        "{clearedCount} discussions effacées ; 1 discussion reste. Veuillez réessayer.",
      chatsClearedRemain:
        "{clearedCount} discussions effacées ; {remainingCount} discussions restent. Veuillez réessayer.",
      oneChatClearedRemain:
        "1 discussion effacée ; {remainingCount} discussions restent. Veuillez réessayer.",
      oneChatClearedRemainOne:
        "1 discussion effacée ; 1 discussion reste. Veuillez réessayer.",
      storageClearFailedOne:
        "Un effacement du stockage a échoué ; 1 discussion peut rester. Veuillez réessayer.",
      storageClearFailed:
        "Un effacement du stockage a échoué ; {count} discussions peuvent rester. Veuillez réessayer.",
      failedToClearChats: "Échec de l'effacement des discussions",
    },
    data: {
      title: "Données",
      backToData: "Retour aux données",
      exportFailed: "Impossible d’exporter les chats",
      description:
        "Gérez l'historique des discussions et les fichiers importés conservés sur cet appareil.",
      archivedChats: "Discussions archivées",
      archivedChatsDescription:
        "Consultez et gérez les discussions que vous avez archivées.",
      archivedImages: "Images archivées",
      archivedImagesDescription: "Consultez et gérez les images que vous avez archivées.",
      archivedVideos: "Vidéos archivées",
      archivedVideosDescription: "Consultez et gérez les vidéos que vous avez archivées.",
      manageAction: "Gérer",
      manageChats: "Gérer les discussions",
      manageChatsDescription:
        "Sélectionnez plusieurs discussions pour les déplacer, les épingler, les archiver, les exporter ou les supprimer.",
      exportArchivedChats: "Exporter",
      exportingArchivedChats: "Exportation...",
      exportedOneArchivedChat: "1 discussion archivée a été exportée",
      exportedArchivedChatCount: "{count} discussions archivées ont été exportées",
      noArchivedChatsToExport: "Aucune discussion archivée à exporter.",
      failedToExportArchivedChats:
        "Échec de l'exportation des discussions archivées",
      archiveAllChats: "Archiver toutes les discussions",
      archiveAllChatsDescription:
        "Déplace vers l'archive toutes les discussions de Récents et Projets.",
      noChatsToArchive: "Aucune discussion à archiver.",
      archiveAllAction: "Tout archiver",
      archivingAction: "Archivage...",
      archiveAllChatsTitle: "Archiver toutes les discussions ?",
      archiveAllChatsConfirmDescription:
        "Déplace vers l'archive toutes les discussions de cet appareil. Les discussions archivées restent disponibles et peuvent être désarchivées à tout moment.",
      archivedAllChats: "Toutes les discussions ont été archivées",
      archivedOneChat: "1 discussion archivée",
      archivedChatCount: "{count} discussions archivées",
      failedToArchiveChats: "Échec de l'archivage des discussions",
      confirmBeforeDeleting: "Confirmer avant de supprimer",
      confirmBeforeDeletingDescription:
        "Demande une confirmation avant de supprimer une discussion. Désactivez cette option pour supprimer immédiatement.",
      alwaysDeleteFiles: "Toujours supprimer les fichiers",
      alwaysDeleteFilesDescription:
        "La suppression d'une discussion retire aussi son dossier bac à sable du disque. Les fichiers écrits dans un projet restent dans l'espace de travail de ce projet.",
      filesSection: "Fichiers",
      uploadedFiles: "Fichiers importés",
      uploadedFilesDescription:
        "Consultez et gérez les fichiers importés dans les discussions, les projets et les bases de connaissances.",
      fineTuneExport: "Utiliser les discussions comme données d'entraînement",
      fineTuneExportDescription:
        "Créez un jeu de données JSONL de fine-tuning à partir de vos discussions. Chargez-le dans Entraîner, affinez-le dans Recettes ou exportez-le.",
      fineTuneExportAction: "Exporter en JSONL",
      fineTuneRunAction: "Exécuter",
      fineTuneExportingAction: "Exportation...",
      fineTuneOpenRecipesAction: "Ouvrir dans Recettes",
      fineTuneOpeningRecipesAction: "Ouverture...",
      fineTuneTrainAction: "Charger dans l'onglet Entraîner",
      fineTuneTrainingAction: "Chargement...",
      fineTuneExportFailed: "Échec de l'exportation des données d'entraînement",
      fineTuneRecipeFailed:
        "Échec de l'ouverture des discussions dans Recettes",
      fineTuneTrainFailed:
        "Échec du chargement du jeu de données dans l'onglet Entraîner",
    },
    connections: {
      title: "Connexions",
      description: "Gérez les fournisseurs et les connexions externes.",
    },
    remoteLan: {
      title: "Accès distant et LAN",
      description:
        "Accédez à cet Unsloth depuis vos autres appareils, via votre réseau local ou une URL publique temporaire.",
    },
    apiKeys: {
      title: "API",
      description: "Accédez à Unsloth via l'API compatible OpenAI.",
      readDocs: "Lire la documentation de l'API",
      noAccess: "Aucun accès API pour le moment.",
      accessTokens: "Jetons d’accès",
      loadError: "Impossible de charger l'accès API.",
      createError: "Impossible de créer le jeton d’accès.",
      revokeError: "Impossible de révoquer le jeton d’accès.",
      never: "Jamais",
      tokenNamePlaceholder: "Nom du jeton (par ex. production)",
      newAccessTokenName: "Nom du nouveau jeton d’accès",
      createToken: "Créer un jeton",
      creating: "Création...",
      newTokenCreated: "Nouveau jeton d’accès créé",
      accessTokenCopied: "Jeton d’accès copié",
      copyAccessToken: "Copier le jeton d’accès",
      copyNow: "Copiez le jeton maintenant : il ne sera plus affiché.",
      usageExamples: "Exemples d'utilisation",
      usageTools: "Outils",
      exampleCurlTools: "curl + outils",
      examplePythonTools: "Python + outils",
      exampleJavaScriptTools: "JavaScript + outils",
      exampleCurlAdvanced: "curl + avancé",
      examplePythonAdvanced: "Python + avancé",
      exampleJavaScriptAdvanced: "JavaScript + avancé",
      osUnix: "Linux / macOS / WSL",
      osWindows: "Windows",
      secureHttps: "HTTPS sécurisé",
      secureHttpsHint:
        "Le service lié à l’adresse 0.0.0.0 reste accessible sur toutes les interfaces réseau. Pour une sécurité complète, lancez Unsloth avec --secure afin de n’exposer que ce lien HTTPS.",
      copyTunnelUrl: "Copier l'URL du tunnel",
      copySnippet: "Copier l'extrait",
      copy: "Copier",
      copied: "Copié",
      setupDocs: "Documentation de configuration :",
      codingAgents: "Agents de codage",
      codingAgentsHint:
        "Lancez un agent de codage sur ce serveur. Il utilise le modèle chargé ; un serveur local génère automatiquement une clé API, un serveur distant l'inclut dans la commande.",
      codingAgentsSwap:
        "Remplacez claude par codex, openclaw, opencode ou hermes.",
      codingAgentDetected: "Installé sur cette machine",
      codingAgentsDetectedHint: "Détecté sur cette machine : {agents}.",
      relativeNever: "jamais",
      relativeJustNow: "à l'instant",
      expired: "expiré",
      today: "aujourd'hui",
      created: "Créé {value}",
      used: "Utilisé {value}",
      expires: "Expire {value}",
      actionsFor: "Actions pour {name}",
      copyPrefix: "Copier le préfixe",
      revokeToken: "Révoquer le jeton",
      revokeTitle: "Révoquer le jeton d’accès « {name} » ?",
      revokeDescription:
        "Les applications utilisant ce jeton perdent immédiatement l’accès. Cette action est irréversible.",
      revokeAction: 'Révoquer "{name}"',
      revoking: "Révocation...",
      usageNoModel:
        "Chargez ou téléchargez un modèle pour voir des exemples exécutables. Aucun modèle n'est encore disponible sur ce serveur pour figurer dans les exemples.",
    },
    about: {
      title: "À propos",
      description:
        "Documentation, notes de version, retours et informations de compilation.",
      studioVersion: "Version d'Unsloth",
      packageVersion: "Version du paquet",
      desktopAppVersion: "Version de l’application de bureau",
      desktopAppVersionUnavailable: "Indisponible",
      llamaCppVersion: "Version de llama.cpp",
      hardware: "Matériel",
      gpu: "GPU",
      cuda: "CUDA",
      rocm: "ROCm",
      xpu: "XPU",
      updates: "Mise à jour",
      help: "Aide",
      documentation: "Documentation",
      releaseNotes: "Notes de version",
      whatsNew: "Nouveautés",
      feedback: "Retours",
      reportIssue: "Signaler un problème",
      license: {
        sectionTitle: "Licence",
        studioLabel: "Unsloth",
        studioLicense: "AGPL-3.0",
        studioDescription: "Open source sous licence GNU AGPL v3.0.",
        libraryLabel: "Unsloth Core",
        libraryLicense: "Apache-2.0",
        libraryDescription: "Sous licence Apache 2.0.",
      },
      dangerZone: "Zone de danger",
      shutDownStudio: "Arrêter Unsloth",
      shutDownStudioDescription:
        "Arrête le serveur Unsloth et met fin à votre session.",
      shutDown: "Arrêter",
      update: {
        title: "Mettre à jour Unsloth",
        commandText: "Texte de la {label}",
        copied: "Copié",
        copyCommand: "Copier la commande",
        commandCopied: "{label} copiée",
        copyNamedCommand: "Copier la {label}",
        checkingInstall: "Vérification du mode d'installation d'Unsloth...",
        installIntro: "Pour installer ou mettre à jour Unsloth :",
        localUpdateHeading: "Mise à jour locale",
        installCommandUnix: "Commande d'installation macOS/Linux",
        installCommandWindows: "Commande d'installation Windows",
        localInstallDetected:
          "Installation locale détectée. Mettez à jour depuis votre checkout d'origine pour éviter de le remplacer par PyPI.",
        pullThenUpdate:
          "Récupérez les dernières modifications, puis lancez l'installateur local :",
        gitPullCommand: "commande git pull",
        localInstallerCommand: "commande de l'installateur local",
        sourceInstallDetected:
          "Installation depuis la source ou un paquet VCS détectée. Réinstallez depuis le chemin local d'origine ou l'URL Git.",
        repoCheckoutFallback:
          "Si vous avez encore le checkout du dépôt, lancez l'installateur local depuis celui-ci :",
        restartAfterUpdate: "Redémarrez Unsloth après la mise à jour.",
        desktopManaged:
          "L’application de bureau recherche automatiquement les nouvelles versions. Vous pouvez également rechercher ou installer une mise à jour ici à tout moment.",
        desktopReady: "Mises à jour de l’application de bureau",
        desktopReadyDescription:
          "Vérifiez si une version plus récente de l’application de bureau est disponible.",
        desktopChecking: "Recherche de mises à jour",
        desktopCheckingDescription:
          "Cette opération ne prend généralement que quelques secondes.",
        desktopAvailable:
          "La version {version} de l’application de bureau est disponible",
        desktopAvailableDescription:
          "Effectuez la mise à jour maintenant. L’application de bureau redémarrera une fois l’opération terminée.",
        desktopExternalServer:
          "Exécutez `unsloth studio update` dans le terminal depuis lequel vous avez lancé le serveur.",
        desktopManualInstall:
          "Ouvrez la page des versions pour installer le dernier paquet Linux.",
        desktopCheckFailed: "Impossible de rechercher les mises à jour",
        desktopCheckFailedDescription:
          "Vérifiez votre connexion, puis réessayez.",
        desktopCurrent: "L’application de bureau est à jour",
        desktopCurrentDescription:
          "Unsloth continuera à rechercher automatiquement les mises à jour.",
        checkForUpdates: "Rechercher les mises à jour",
        checkAgain: "Rechercher à nouveau",
        retryCheck: "Réessayer",
        checking: "Vérification...",
        updateNow: "Mettre à jour maintenant",
        openReleasePage: "Ouvrir la page des versions",
        unknownInstall:
          "Impossible de détecter le mode d'installation d'Unsloth. Pour les installations via installateur ou PyPI, utilisez les commandes ci-dessus.",
        localCheckout:
          "Pour les installations depuis un checkout local, lancez l'installateur local depuis ce checkout :",
        docs: "Documentation d'installation :",
        docsInstall: "Installation",
        docsUpdating: "Mise à jour",
        docsMac: "Mac",
        docsWindows: "Windows",
      },
    },
  },
  studio: {
    imageTraining: "Entraînement d'images",
    goToImageTraining: "Aller à l'entraînement d'images",
    routeTitle: "Entraîner",
    wizard: {
      modelTitle: "Modèle",
      modelDescription: "Sélectionner le modèle et la méthode d'entraînement",
      datasetTitle: "Jeu de données",
      datasetDescription:
        "Sélectionner ou téléverser des données d'entraînement",
      paramsTitle: "Paramètres",
      paramsDescription: "Configurer les paramètres d'entraînement",
      configTitle: "Configuration",
      configDescription: "Enregistrer et charger des configurations",
      modelLabel: "Modèle",
      methodLabel: "Méthode",
      datasetLabel: "Jeu de données",
      modelTooltip: "Le modèle de base que vous souhaitez affiner.",
      methodTooltip: "Comment le modèle est entraîné. LoRA et QLoRA mettent à jour de petits adaptateurs au lieu de tous les poids.",
      datasetTooltip: "Les données d'entraînement utilisées pour affiner le modèle.",
      hfTokenDescription:
        "Nécessaire pour les modèles et jeux de données restreints ou privés.",
      uploadLocalLabel: "Ou téléverser un fichier local",
      sourceBrowse: "Parcourir",
      releaseToUpload: "Relâchez pour téléverser",
      loadYaml: "Charger le YAML",
      saveYaml: "Enregistrer le YAML",
      resetDefaults: "Rétablir les valeurs par défaut",
      cachedModelGoneTitle: "Modèle en cache indisponible",
      cachedModelGoneDescription:
        "Les fichiers du modèle ne sont plus sur cet appareil. L'entraînement les téléchargera à nouveau.",
      cachedDatasetGoneTitle: "Jeu de données en cache indisponible",
      cachedDatasetGoneDescription:
        "Les fichiers du jeu de données ne sont plus sur cet appareil. L'entraînement les téléchargera à nouveau.",
    },
    preview: {
      title: "Aperçu de l'exécution",
      ready: "Prêt",
      notReady: "Pas prêt",
      modelPending: "Modèle en attente",
      datasetPending: "Jeu de données en attente",
      method: "Méthode",
      length: "Durée",
      stepZero: "{count} étape",
      step: "{count} étape",
      stepTwo: "{count} étapes",
      stepFew: "{count} étapes",
      stepMany: "{count} étapes",
      steps: "{count} étapes",
      epochZero: "{count} époque",
      epoch: "{count} époque",
      epochTwo: "{count} époques",
      epochFew: "{count} époques",
      epochMany: "{count} époques",
      epochs: "{count} époques",
      batch: "Lot",
      context: "Contexte",
      lr: "LR",
      hardware: "Matériel",
      noGpu: "Aucun GPU détecté",
      hfToken: "Token HF",
      saved: "Enregistré",
      notSet: "Non défini",
      files: "Fichiers",
      model: "Modèle",
      dataset: "Jeu de données",
      downloadsOnStart: "Téléchargement au démarrage",
      continuesOnStart: "Reprise au démarrage",
      noticeModelDownload:
        "Ce modèle n'est pas encore sur cet appareil. Il sera téléchargé automatiquement au démarrage de l'entraînement.",
      noticeModelPartial:
        "L'entraînement terminera le téléchargement partiel du modèle avant de le charger.",
      noticeDatasetDownload:
        "Ce jeu de données n'est pas encore sur cet appareil. Il sera téléchargé automatiquement au démarrage de l'entraînement.",
      noticeDatasetPartial:
        "L'entraînement terminera le téléchargement partiel du jeu de données avant de le lire.",
      noticeTransformersUpgrade:
        "Aucune version installée de transformers ne prend encore en charge cette architecture. Au démarrage, l'installation de transformers {version} sera proposée d'abord.",
      noticeSixteenBitOnly:
        "Cette architecture s'entraîne en LoRA 16 bits : le 4 bits n'est pas disponible, donc l'exécution demande beaucoup plus de VRAM que QLoRA.",
      noticeInstallSwitchesSixteenBit:
        "Installer cette version au lieu de conserver le code propre au modèle fait passer cette exécution en LoRA 16 bits, qui demande beaucoup plus de VRAM que QLoRA.",
      advancedSettings: "Paramètres avancés",
      defaultAdvancedSettings: "Valeurs par défaut",
      nonDefaultAdvancedSettings: "{count} non standard",
    },
    datasetPicker: {
      noun: "jeux de données",
      selectDataset: "Sélectionner un jeu de données",
      hubPlaceholder: "Rechercher des jeux de données Hugging Face...",
      devicePlaceholder: "Rechercher des jeux de données locaux...",
      useAsHubDataset: "Utiliser comme jeu de données Hugging Face",
      hfCacheLabel: "Cache HF",
      scanningLocal: "Recherche des jeux de données sur cet appareil…",
      couldntScan: "Impossible d'analyser les jeux de données locaux",
      someLocationsUnscanned:
        "Certains emplacements de jeux de données n'ont pas pu être analysés.",
      noLocalDatasets:
        "Rien sur cet appareil pour le moment. Téléchargez un jeu de données depuis le Hub, créez-en un dans Recettes ou téléversez un fichier.",
      openDataRecipes: "Ouvrir les recettes de données",
      searchingHub: "Recherche sur Hugging Face…",
      noDatasetsFound: "Aucun jeu de données trouvé.",
      tokenRejectedTitle: "Token Hugging Face refusé",
      tokenRejectedBody:
        "Mettez à jour votre token dans Paramètres → Général, puis réessayez.",
      hubUnreachable: "Impossible de joindre Hugging Face",
      cantUseDataset: "Impossible d'utiliser le jeu de données",
      reasonInvalidHubId:
        "Saisissez un ID de jeu de données Hugging Face valide : dépôt ou propriétaire/dépôt, composé uniquement de lettres, chiffres, ., _ ou - (96 caractères maximum par partie).",
      sourceRecipe: "Recette",
      sourceUpload: "Téléversement",
      sourceLocal: "Local",
    },
    modelPicker: {
      noun: "modèles",
      selectModel: "Sélectionner un modèle",
      hubPlaceholder: "Rechercher ou coller un ID Hugging Face...",
      devicePlaceholder:
        "Rechercher des modèles locaux ou coller un chemin de dossier...",
      useAsHubModel: "Utiliser comme modèle Hugging Face",
      useAsLocalPath: "Utiliser comme chemin local",
      hfCacheLabel: "Cache HF",
      scanningLocal: "Recherche des modèles locaux…",
      couldntScan: "Impossible d'analyser les modèles locaux",
      someLocationsUnscanned:
        "Certains emplacements locaux n'ont pas pu être analysés.",
      noLocalModels: "Aucun modèle local trouvé.",
      noLocalModelsHint:
        "Collez un chemin de dossier ci-dessus ou passez à Hugging Face.",
      searchingHub: "Recherche sur Hugging Face…",
      noModelsFound: "Aucun modèle trouvé.",
      tokenRejectedTitle: "Token Hugging Face refusé",
      tokenRejectedBody:
        "Mettez à jour votre token dans Paramètres → Général, puis réessayez.",
      hubUnreachable: "Impossible de joindre Hugging Face",
      cantUseModel: "Impossible d'utiliser le modèle pour l'entraînement",
      reasonTypeMismatch:
        "Ce modèle ne correspond pas au type d’entraînement sélectionné à l’étape précédente.",
      reasonEmptyId:
        "Saisissez un ID de modèle ou le chemin d'un modèle local.",
      reasonInvalidHubId:
        "Saisissez un ID de modèle Hugging Face valide : dépôt ou propriétaire/dépôt, composé uniquement de lettres, chiffres, ., _ ou - (96 caractères maximum par partie).",
      reasonGguf: "Les modèles GGUF ne peuvent pas être entraînés.",
      reasonAdapter:
        "Les sorties d'adaptateur ne peuvent pas servir de modèles de base pour l'entraînement.",
      reasonNotTrainable:
        "Ce modèle présent sur l'appareil ne peut pas être entraîné.",
      reasonUnsupportedFormat:
        "Ce format de modèle n'est pas pris en charge pour l'entraînement.",
      vramNeeds: "Nécessite environ {est} Go de VRAM (GPU : {total} Gio)",
      vramTight: "Environ {est} Go de VRAM (limite sur {total} Gio)",
      vramApprox: "Environ {est} Go de VRAM",
      sourceModelsFolder: "Dossier des modèles",
      sourceHfCache: "Cache HF",
      sourceLmStudio: "LM Studio",
      sourceOllama: "Ollama",
      sourceCustomFolder: "Dossier personnalisé",
      sourceLocalModel: "Modèle local",
      vramOomBadge: "OOM",
      vramTightBadge: "Limite",
    },
    methods: {
      qlora: {
        label: "QLoRA",
        hint: "Quantification 4 bits. VRAM minimale et démarrage le plus rapide.",
        note: "4 bits",
      },
      lora: {
        label: "LoRA",
        hint: "Adaptateurs 16 bits. Équilibre entre qualité et mémoire.",
        note: "16 bits",
      },
      full: {
        label: "Fine-tuning complet",
        hint: "Entraîne tous les poids. Qualité maximale, mais nécessite le plus de VRAM.",
        note: "fp16",
      },
      cpt: {
        label: "Pré-entraînement continu",
        hint: "Pré-entraînement continu pour de nouveaux domaines ou de nouvelles langues.",
        note: "continu",
      },
    },
    subtitles: {
      configure: "Configurer et démarrer l'entraînement",
      trainingInProgress: "Entraînement en cours",
      viewPastRuns: "Voir les entraînements passés",
      viewingPastRun: "Consultation d'un entraînement passé",
    },
    tabs: {
      configure: "Configurer",
      currentRun: "Entraînement actuel",
      history: "Historique",
    },
    loadingRuntime: "Chargement de l'environnement d'entraînement...",
    checkingSupport: "Vérification de la prise en charge de l'entraînement sur cette machine...",
    backToHistory: "Retour à l'historique",
    dataset: {
      selectors: {
        subset: "Sous-ensemble",
        subsetTooltip:
          "Sélectionnez le sous-ensemble (configuration) du jeu de données à utiliser.",
        trainSplit: "Partition d’entraînement",
        trainSplitTooltip:
          "Sélectionnez la partition à utiliser pour l’entraînement.",
        evaluationSplit: "Partition d’évaluation",
        evaluationSplitTooltip:
          "Sélectionnez la partition à utiliser pour l’évaluation. Aucune signifie qu’aucune évaluation ne sera effectuée pendant l’entraînement.",
        selectSubset: "Sélectionnez un sous-ensemble...",
        selectSplit: "Sélectionnez une partition...",
        none: "Aucune",
        loading:
          "Chargement des configurations et partitions du jeu de données...",
        manualTitle: "Saisir manuellement les options du jeu de données",
        manualDescription:
          "Saisissez les noms exacts de la configuration et des partitions Hugging Face à utiliser.",
        manualSubsetPlaceholder: "Nom de configuration facultatif",
        manualRequired: "Une partition d’entraînement est requise.",
        manualTooLong: "Utilisez au maximum 128 caractères.",
        manualInvalid: "Cette valeur contient des caractères non pris en charge.",
      },
      sourceAriaLabel: "Source du jeu de données",
      localDataset: "Jeu de données local",
      localDatasetRows: " / {count} lignes",
      huggingFaceDataset: "Jeu de données Hugging Face",
      localDatasetMetadata: "Métadonnées du jeu de données local",
      dataRecipeOutput: "Sortie de Data Recipe.",
      rows: "Lignes",
      columns: "Colonnes",
      batches: "Batchs",
      updated: "Mis à jour",
      evalDataset: "Jeu de données d'évaluation",
      uploading: "Téléversement...",
      uploadEvalFile: "Téléverser un fichier d'évaluation",
      fileTooLarge: "Fichier trop volumineux",
      fileTooLargeDescription:
        "{file} fait {size}. Les téléversements d’entraînement sont limités à {limit}.",
      documentRedirect: {
        title: "Ce fichier doit d’abord être converti",
        genericFile: "Ce fichier",
        description:
          "{file} est un document source, pas un jeu de données prêt pour l’entraînement. Utilisez Data Recipes pour convertir le document en jeu de données, puis revenez ici pour l’affinage.",
        nextStepTitle: "Étape suivante recommandée",
        nextStepDescription:
          "Ouvrez Learning Recipes et commencez par une recette basée sur un document, comme PDF grounded QA.",
        openAction: "Ouvrir Learning Recipes",
      },
      evalDatasetDescription:
        "Facultatif. Si non fourni, une petite portion sera prélevée sur les données d'entraînement.",
      advanced: "Avancé",
      targetFormat: "Format cible",
      targetFormatTooltip:
        "Format de vos données d'entraînement. La détection automatique fonctionne pour la plupart des jeux de données.",
      streamingInfoAriaLabel:
        "Informations sur le streaming du jeu de données",
      streaming: {
        label: "Activer le streaming",
        description:
          "Utilisez les jeux de données textuels de Hugging Face en streaming au lieu de les télécharger.",
        unavailable: "Streaming indisponible. Pour l'activer :",
        completionsUnavailable:
          "Indisponible lorsque le streaming du jeu de données est activé.",
        blockers: {
          source:
            "Utilisez un jeu de données Hugging Face (pas un téléversement local ni une source S3).",
          maxSteps:
            "Définissez le nombre max d'étapes > 0 : les jeux de données en streaming n'ont pas de longueur connue.",
          trainOnCompletions:
            'Désactivez "Réponses de l’assistant uniquement".',
          evalSplit:
            "Choisissez un split d'évaluation distinct : l'évaluation est activée, mais aucun split distinct n'est défini.",
          visionModel:
            "Les modèles de vision ne prennent pas en charge le streaming.",
          audioModel:
            "Les modèles audio ne prennent pas en charge le streaming.",
          embeddingModel:
            "Les modèles d'embeddings ne prennent pas en charge le streaming (l'entraînement nécessite le jeu de données complet).",
          imageDataset:
            "Ce jeu de données semble contenir des images, ce qui empêche son utilisation en streaming.",
          audioDataset:
            "Ce jeu de données semble contenir de l'audio, ce qui empêche son utilisation en streaming.",
          appleSilicon:
            "Le streaming n'est pas encore pris en charge sur Apple Silicon (MLX).",
        },
        options: {
          trainOnCompletions: "réponses de l'assistant uniquement",
          evaluation:
            "évaluation (nécessite un split d'évaluation distinct)",
        },
        notifications: {
          turnedOffMaxSteps:
            "Streaming désactivé : il nécessite un nombre max d'étapes fixe > 0.",
          adjusted:
            "Paramètres ajustés pour le streaming. Options incompatibles désactivées : {options}.",
          needsMaxSteps:
            "Le streaming nécessite un nombre max d'étapes fixe (les jeux de données en streaming n'ont pas de longueur connue). Définissez d'abord le nombre max d'étapes > 0.",
          enabledAdjusted:
            "Streaming activé. Options incompatibles désactivées : {options}.",
          disabledForDetectedModality:
            "Le streaming a été désactivé, car les jeux de données d'images et audio doivent être téléchargés intégralement. Vérifiez le réglage, puis relancez l'entraînement.",
        },
      },
      auto: "Auto",
      rawText: "Texte brut",
      trainSplitStart: "Début du split d'entraînement",
      trainSplitStartTooltip:
        "N'entraîner que sur un sous-ensemble de votre split d'entraînement en spécifiant un indice de ligne de départ (inclus, base 0). Laissez vide pour commencer à la première ligne.",
      trainSplitEnd: "Fin du split d'entraînement",
      trainSplitEndTooltip:
        "Dernier indice de ligne à inclure du split d'entraînement (inclus, base 0). Par exemple, définissez Début à 0 et Fin à 99 pour entraîner sur les 100 premières lignes. Laissez vide pour utiliser toutes les lignes restantes.",
      endPlaceholder: "Fin",
      clear: "Effacer",
      dropFileOrClick: "Déposez 1 fichier ici ou cliquez pour téléverser",
      uploadDetails: "Détails du téléversement",
      uploadDetailsTooltip:
        "Jusqu’à {limit} par fichier. Les fichiers PDF, DOCX et TXT ne sont pas des jeux de données prêts pour l’entraînement ; convertissez-les d’abord dans les Recettes.",
      viewDataset: "Voir le jeu de données",
      uploadFailed: "Échec du téléversement",
      unknownError: "Erreur inconnue",
      unsupportedFileType: "Type de fichier non pris en charge",
      uploadOneFileType: "Téléversez un fichier {types}.",
      datasetUploaded: "Jeu de données téléversé",
      evalDatasetUploaded: "Jeu de données d'évaluation téléversé",
      uploadOneFileAtATime: "Téléversez un fichier à la fois",
      uploadSingleFileDescription:
        "Le téléversement du jeu de données d'entraînement accepte un seul fichier.",
      previewLoadingHuggingFace:
        "Récupération de l’aperçu du jeu de données depuis Hugging Face...",
      previewLoading: "Chargement de l’aperçu...",
      mappingRequirements: {
        audioAndText: "audio et texte",
        imageAndText: "image et texte",
        instructionAndOutput: "instruction et sortie",
        humanAndGpt: "humain et GPT",
        userAndAssistant: "utilisateur et assistant",
      },
      mappingStatus: {
        heuristicTitle: "Mappage détecté par heuristique",
        readyTitle: "Mappage prêt",
        requiredTitle: "Mapper les colonnes du jeu de données",
        heuristicDescription:
          "Nous avons détecté automatiquement le mappage des colonnes ci-dessous à l’aide d’heuristiques. Vérifiez-le et ajustez-le avec les menus des en-têtes de colonnes, ou utilisez l’assistance IA pour un mappage plus précis.",
        readyDescription:
          "Tout est prêt. Nous convertirons automatiquement ce jeu de données.",
        requiredDescription:
          "Attribuez des rôles aux colonnes à l’aide des menus des en-têtes. Attribuez au minimum {required}.",
      },
      s3: {
        title: "Configuration S3",
        description:
          "Charger des jeux de données .parquet, .json, .jsonl ou .csv depuis Amazon S3",
        bucket: "Nom du bucket",
        bucketPlaceholder: "mon-bucket-donnees-entrainement",
        region: "Région AWS",
        regionPlaceholder: "us-east-1",
        prefix: "Préfixe de chemin",
        prefixPlaceholder: "datasets/whisper/",
        accessKeyId: "ID de clé d'accès",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Clé d'accès secrète",
        secretAccessKeyPlaceholder: "Votre clé d'accès secrète AWS",
        useIamRole: "Utiliser un rôle IAM",
      },
    },
    params: {
      mode: {
        simple: "Simple",
        advanced: "Avancé",
        ariaLabel: "Mode des paramètres",
      },
      projectName: "Nom du projet",
      optional: "Facultatif",
      projectNameDescription:
        "Utilisé dans les noms des dossiers de sortie d'entraînement, les valeurs d'export par défaut et l'historique.",
      loraSettings: "Paramètres LoRA",
      trainingHyperparameters: "Hyperparamètres d'entraînement",
      maxSteps: "Nombre max d'étapes",
      epochs: "Époques",
      useMaxSteps: "Utiliser le nombre max d'étapes",
      useEpochs: "Utiliser les époques",
      maxStepsTooltip: "Remplace le nombre total d'étapes d'optimisation.",
      epochsTooltip: "Nombre de passages complets sur le jeu de données.",
      contextLength: "Longueur de contexte",
      contextLengthTooltip:
        "Nombre maximal de tokens par échantillon d'entraînement.",
      customContextLength: "Saisir une valeur personnalisée",
      learningRate: "Taux d'apprentissage",
      learningRateTooltip:
        "Taille du pas pour les mises à jour des poids. Des valeurs plus faibles entraînent plus lentement mais de manière plus stable.",
      learningRateDescription:
        "Recommandé : 2e-4 pour LoRA, 5e-5 pour CPT, 2e-5 pour le fine-tune complet",
      embeddingLearningRate: "Taux d'apprentissage des embeddings",
      embeddingLearningRateTooltip:
        "Utilisé uniquement lorsque CPT entraîne embed_tokens. Les embeddings sont plus faciles à déstabiliser que les poids LoRA, ils nécessitent donc généralement un taux plus faible. Laissez vide pour utiliser lr/10 ; la plage typique est 2x à 10x plus petite que le taux principal. Augmentez-le seulement si l'adaptation du vocabulaire ou des tokens de domaine est trop lente.",
      rank: "Rang",
      rankTooltip:
        "Dimension des matrices de bas rang. Plus élevé = plus de capacité.",
      alpha: "Alpha",
      alphaTooltip:
        "Facteur d'échelle pour les mises à jour LoRA. Généralement 2x le rang.",
      dropout: "Dropout",
      dropoutTooltip:
        "Probabilité de dropout pour les couches LoRA afin de réduire le surapprentissage.",
      visionLayers: "Couches de vision",
      languageLayers: "Couches de langage",
      attentionModules: "Modules d'attention",
      mlpModules: "Modules MLP",
      targetModules: "Modules cibles",
      enableLora: "Activer LoRA",
      trainWithLora: "Entraîner avec LoRA",
      stableRank: "Rang stable",
      memoryEfficient: "Économe en mémoire",
      weightDecomposed: "Poids décomposés",
      notSupportedAppleSilicon: "Non pris en charge sur Apple Silicon",
      optimization: "Optimisation",
      schedule: "Planification",
      memory: "Mémoire",
      optimizer: "Optimiseur",
      optimizerTooltip:
        "Algorithme d'optimisation. Les variantes 8 bits réduisent l'usage mémoire. Fused est recommandé pour les modèles de vision.",
      optimizerTooltipMlx:
        "Algorithme d'optimisation. AdamW est utilisé par défaut. Lion consomme moins de mémoire, mais nécessite généralement un taux d'apprentissage plus faible.",
      lrScheduler: "Planificateur de taux d'apprentissage",
      lrSchedulerTooltip:
        "Comment le taux d'apprentissage évolue au cours de l'entraînement. Linear décroît régulièrement ; cosine décroît selon une courbe.",
      optimizerOptions: {
        adamw8bit: "AdamW 8 bits",
        pagedAdamw8bit: "Paged AdamW 8 bits",
        adamwBnb8bit: "AdamW BNB 8 bits",
        pagedAdamw32bit: "Paged AdamW 32 bits",
        adamwTorch: "AdamW (PyTorch)",
        adamwTorchFused: "AdamW (PyTorch Fused)",
      },
      lrSchedulerOptions: {
        linear: "Linéaire",
        cosine: "Cosinus",
      },
      batchSize: "Taille de batch",
      batchSizeTooltip:
        "Échantillons traités par étape. Plus élevé utilise plus de VRAM.",
      gradAccum: "Accumulation de gradient",
      gradAccumTooltip:
        "Simule des tailles de batch plus grandes sans VRAM supplémentaire.",
      weightDecay: "Décroissance des poids",
      weightDecayTooltip:
        "Régularisation L2 pour prévenir le surapprentissage.",
      warmupSteps: "Étapes de préchauffage",
      warmupStepsTooltip:
        "Augmenter progressivement le taux d'apprentissage au début de l'entraînement pour plus de stabilité.",
      scheduleEpochsTooltip:
        "Nombre de passages complets sur le jeu de données. Définissez 0 pour fonctionner par nombre max d'étapes.",
      saveSteps: "Étapes de sauvegarde",
      saveStepsTooltip:
        "Enregistrer un checkpoint toutes les N étapes. 0 pour désactiver.",
      evalSteps: "Étapes d'évaluation",
      evalStepsTooltip:
        "Fraction du nombre total d'étapes d'entraînement entre les évaluations (0-1). Définissez 0 pour désactiver l'évaluation. Ex. 0,01 = évaluer tous les 1 % d'étapes.",
      seed: "Graine",
      seedTooltip: "Graine aléatoire pour la reproductibilité.",
      gradCheckpoint: "Checkpoint de gradient",
      gradCheckpointTooltip:
        "Échanger du calcul contre de la mémoire en recalculant les activations.",
      none: "Aucun",
      standard: "Standard",
      enablePacking: "Activer le packing",
      assistantCompletionsOnly: "Complétions de l'assistant uniquement",
      readMore: "En savoir plus",
    },
    training: {
      startTraining: "Démarrer l'entraînement",
      starting: "Démarrage...",
      loadingModel: "Chargement du modèle...",
      checkingDataset: "Vérification du jeu de données...",
      chooseModel: "Choisir un modèle",
      chooseDataset: "Choisir un jeu de données",
      chooseModelAndDataset: "Choisir un modèle et un jeu de données",
      modelUnverified:
        "Impossible de vérifier les paramètres de ce modèle. Vérifiez votre connexion ou votre token Hugging Face, puis réessayez.",
      legacyDatasetScriptUnsupported:
        "Ce jeu de données du Hub repose sur un ancien script personnalisé et n’est pas pris en charge dans ce flux d’entraînement.",
      hfModelAccessDenied:
        "Hugging Face a refusé l’accès à ce modèle. Ajoutez un jeton Hugging Face valide ayant accès au dépôt, acceptez les éventuelles conditions d’accès, puis réessayez.",
      hfModelVerificationRateLimited:
        "La vérification du modèle Hugging Face est limitée. Réessayez dans quelques instants.",
      hfModelVerificationFailed:
        "Le modèle Hugging Face n’a pas pu être vérifié. Vérifiez l’identifiant du dépôt et votre jeton d’accès.",
      hfModelMetadataUnavailable:
        "Les métadonnées du modèle Hugging Face sont temporairement indisponibles. Réessayez avant de démarrer l’entraînement.",
      datasetUnverified:
        "Impossible de vérifier si le jeu de données est compatible avec ce modèle. Vérifiez votre connexion ou votre token Hugging Face ; le démarrage de l'entraînement relancera la vérification.",
      setupChanged:
        "La configuration de l'entraînement a changé pendant sa vérification. Vérifiez-la, puis relancez l'entraînement.",
      validation: {
        s3MultimodalUnsupported:
          "Les jeux de données S3 ne sont pas encore pris en charge pour l'entraînement de modèles de vision ou audio.",
        s3BucketRequired: "Saisissez d'abord le nom d'un bucket S3.",
        s3CredentialsRequired:
          "Indiquez des clés d'accès S3 ou activez le rôle IAM.",
        modelRequired: "Sélectionnez d'abord un modèle de base.",
        learningRatePositive: "Saisissez un taux d'apprentissage supérieur à zéro.",
        embeddingLearningRateRange:
          "Saisissez un taux d'apprentissage des embeddings supérieur à 0 et inférieur à 1.",
        hfDatasetRequired:
          "Sélectionnez d'abord un jeu de données Hugging Face.",
        hfDatasetSplitRequired:
          "Sélectionnez ou saisissez d’abord une partition d’entraînement.",
        localDatasetRequired: "Sélectionnez d'abord un jeu de données local.",
        unsupportedDatasetSource:
          "Source de jeu de données non prise en charge.",
      },
      startFailed: "Échec du démarrage de l'entraînement",
      startUnconfirmed:
        "Unsloth n'a pas pu confirmer le démarrage de l'entraînement. Vérification de l'état en arrière-plan.",
      stopFailed: "Échec de l'arrêt de l'entraînement",
      trainingStillActiveTitle: "L'entraînement est toujours actif",
      stopBeforeConfig:
        "Arrêtez d'abord l'entraînement, puis revenez à la configuration.",
      resumeFailed: "Échec de la reprise de l'entraînement",
      resumeFailedTitle: "Impossible de reprendre l'entraînement",
      resumeUnavailable:
        "Seuls les entraînements arrêtés ou en erreur disposant d'un checkpoint enregistré peuvent être repris.",
      uploadConfigTooltip: "Charger une configuration YAML enregistrée",
      saveConfigTooltip: "Télécharger la configuration actuelle au format YAML",
      resetConfigTooltip: "Réinitialiser aux valeurs par défaut du modèle",
      configLoaded: "Configuration chargée",
      failedToLoadConfig: "Échec du chargement de la configuration",
      invalidYamlFile: "Fichier YAML invalide",
      configTooLarge:
        "Le fichier de configuration d’entraînement est trop volumineux (maximum 1 Mio).",
      failedToReadFile: "Échec de la lecture du fichier",
      failedToSaveConfig: "Échec de l'enregistrement de la configuration",
      parametersReset: "Paramètres réinitialisés aux valeurs par défaut du modèle",
      audioIncompatible:
        "Ce modèle ne prend pas en charge l'audio. Passez à un modèle compatible audio ou choisissez un jeu de données sans audio.",
      visionIncompatible:
        "Un modèle textuel n'est pas compatible avec un jeu de données multimodal. Passez à un modèle de vision ou choisissez un jeu de données uniquement textuel.",
      cancelTitle: "Annuler l'entraînement",
      cancelDescription: "Voulez-vous annuler l'entraînement en cours ?",
      continueAction: "Continuer l'entraînement",
      cancelAction: "Annuler l'entraînement",
      stopTitle: "Arrêter l'entraînement",
      stopDescription:
        "Choisissez comment arrêter l’entraînement en cours. « Arrêter et enregistrer » crée un point de contrôle qui permettra de le reprendre plus tard ; un entraînement simplement arrêté ne peut pas être repris.",
      stopAction: "Arrêter",
      stopping: "Arrêt...",
      stopAndSave: "Arrêter et enregistrer",
      compareInChat: "Comparer dans la discussion",
      exportModel: "Exporter le modèle",
      milestone: "Étape clé",
      halfwayDone: "À mi-chemin. L'entraînement a dépassé 50 %.",
      doneNextStep:
        "Entraînement terminé. Étape suivante : comparer les sorties de base et affinées.",
    },
    history: {
      title: "Historique",
      emptyDescription:
        "Aucun entraînement pour le moment. Démarrez votre premier entraînement dans l'onglet Configurer.",
      loadError: "Échec du chargement des entraînements",
      deleteError:
        "Échec de la suppression de l'entraînement. Veuillez réessayer.",
      retry: "Réessayer",
      loadMore: "Charger plus",
      loading: "Chargement...",
      loadingRun: "Chargement de l'entraînement...",
      runNotFound: "Entraînement introuvable",
      deleteTitle: "Supprimer l'entraînement ?",
      deleteDescription:
        "Cette action supprimera définitivement cet entraînement et toutes ses métriques. Elle est irréversible.",
      filesDeleted: "Fichiers supprimés",
      deleteArtifactsLabel:
        "Supprimer également les fichiers de l'adaptateur du disque",
      deleteArtifactsDescription:
        "Supprime le dossier de sortie de l'exécution, y compris les adaptateurs et les checkpoints enregistrés.",
      deleteArtifactsSharedNote:
        "Une autre exécution partage ce dossier de sortie. Les fichiers sont conservés jusqu'à la suppression de la dernière exécution qui les utilise.",
      artifactsKeptShared:
        "Exécution supprimée. Les fichiers de l'adaptateur ont été conservés, car une autre exécution utilise le même dossier.",
      deleteArtifactsActiveError:
        "Ces fichiers sont utilisés par l'entraînement en cours. Arrêtez l'entraînement avant de les supprimer.",
      deleteArtifactsFailed:
        "L'exécution a été supprimée, mais ses fichiers n'ont pas pu être effacés.",
      deleteArtifactsRetainedError:
        "Les fichiers d’adaptateur n’ont pas pu être supprimés. L’entraînement a donc été conservé dans l’historique.",
      resumeTraining: "Reprendre l'entraînement",
      resuming: "Reprise...",
      deleteRun: "Supprimer l'entraînement",
      loss: "Perte",
      steps: "Étapes",
      lossTrendSparkline: "Sparkline de tendance de la perte",
      relativeJustNow: "à l'instant",
      status: {
        completed: "Terminé",
        stopped: "Arrêté",
        error: "Erreur",
        running: "En cours",
        continued: "Poursuivi",
      },
      message: {
        completed: "Entraînement terminé",
        stopped: "Entraînement arrêté",
        running: "Entraînement en cours",
        errored: "Erreur d'entraînement",
      },
      copyPreviewLink: "Copier le lien d'aperçu",
      previewLinkCopied: "Lien d'aperçu copié",
      previewLinkCopyFailed: "Impossible de copier le lien",
    },
    charts: {
      settings: "Paramètres des graphiques",
      settingsDescription:
        "Ajustez la présentation des graphiques pendant que l'entraînement continue.",
      openSettings: "Ouvrir les paramètres des graphiques",
      viewWindow: "Fenêtre d'affichage",
      viewWindowDescription:
        "Afficher uniquement les dernières étapes ou tout l'historique.",
      window: "Fenêtre",
      all: "Tout",
      trainingLoss: "Perte d'entraînement",
      trainingLossDescription: "Contrôlez les superpositions et le lissage EMA.",
      smoothing: "Lissage",
      smoothingDescription:
        "Déplacez vers la droite pour plus de lissage. `0` = brut.",
      showRawLoss: "Afficher la perte brute",
      showSmoothedLoss: "Afficher la perte lissée",
      showAverageLine: "Afficher la ligne moyenne",
      scaleAndCleanup: "Échelle et nettoyage",
      linear: "Linéaire",
      log: "Log",
      noClip: "Sans écrêtage",
      clipP99: "Écrêter p99",
      clipP95: "Écrêter p95",
      lossAxis: "Axe de la perte",
      gradientNormAxis: "Axe de la norme du gradient",
      learningRateAxis: "Axe du taux d'apprentissage",
      resetDefaults: "Réinitialiser par défaut",
      loss: "Perte",
      smoothed: "Lissé",
      evalLoss: "Perte d'évaluation",
      learningRate: "Taux d'apprentissage",
      lr: "LR",
      gradNorm: "Norme du gradient",
      gradientNorm: "Norme du gradient",
      step: "Étape {step}",
      averageValue: "moy {value}",
      waitingForFirstEvaluationStep:
        "En attente de la première étape d'évaluation...",
      evaluationNotConfigured: "Évaluation non configurée",
      evalChartWillAppear:
        "Le graphique apparaîtra une fois eval_steps atteint",
      setEvalDatasetAndSteps:
        "Définissez le jeu de données d'évaluation et eval_steps pour suivre la perte d'évaluation",
    },
    progress: {
      title: "Progression de l'entraînement",
      liveMetrics: "Métriques d'entraînement en direct",
      exportGguf: "Exporter en GGUF",
      openConfig: "Ouvrir la configuration d'entraînement",
      configLabel: "Configuration d'entraînement",
      hyperparams: "Hyperparamètres",
      epochs: "Époques",
      batchSize: "Taille de batch",
      learningRate: "Taux d'apprentissage",
      optimizer: "Optimiseur",
      maxSteps: "Nombre max d'étapes",
      contextLength: "Longueur de contexte",
      warmupSteps: "Étapes de préchauffage",
      rank: "Rang",
      alpha: "Alpha",
      dropout: "Dropout",
      variant: "Variante",
      epoch: "Époque {value}",
      percentComplete: "{percent} % terminé",
      stepProgress: "Étape {current} / {total}",
      loss: "Perte",
      lr: "LR",
      gradNorm: "Norme du gradient",
      project: "Projet",
      model: "Modèle",
      method: "Méthode",
      elapsed: "Écoulé : {value}",
      eta: "ETA : {value}",
      stepsPerSecond: "{value} étapes/s",
      noStepsPerSecond: "-- étapes/s",
      tokens: "Tokens : {value}",
      gpuMonitor: "Moniteur GPU",
      live: "En direct",
      utilization: "Utilisation",
      temperature: "Température",
      vram: "VRAM",
      power: "Puissance",
      phase: {
        idle: "Inactif",
        downloadingModel: "Téléchargement du modèle",
        downloadingDataset: "Téléchargement du jeu de données",
        loadingModel: "Chargement du modèle",
        loadingDataset: "Chargement du jeu de données",
        configuring: "Configuration",
        training: "Entraînement",
        finalizing: "Enregistrement du modèle",
        completed: "Terminé",
        error: "Erreur",
        stopped: "Arrêté",
      },
    },
    trainingStart: {
      ready: "Prêt",
      downloading: "Téléchargement",
      preparing: "Préparation",
      left: "{eta} restant",
      downloaded: "{size} téléchargé",
      terminalStart: "> l'entraînement unsloth démarre...",
      preparingResources: "> Préparation du modèle et du jeu de données...",
      gettingReady: "> Nous préparons tout pour votre entraînement...",
      waitingForFirstStep:
        "> {message} | en attente de la première étape... ({step})",
      resumingTraining: "Reprise de l'entraînement...",
      startingTraining: "démarrage de l'entraînement...",
      dataset: "Jeu de données",
      datasetStreaming: "Jeu de données : streaming (pas de téléchargement complet)",
      modelWeights: "Poids du modèle",
    },
  },
  modelMemory: {
    readout:
      "Poids {model} + contexte {context} = {total} sur {budget} de VRAM utilisable",
    readoutWithSpec:
      "Poids {model} + KV {kv} + brouillon MTP {spec} = {total} sur {budget} de VRAM utilisable",
    kvRate: "KV réservé, ~{rate}/token",
    oomLikely: "Avec les réglages actuels, un dépassement de mémoire est probable",
    tooLarge: "Plus volumineux que la VRAM, sera déchargé sur le CPU. Une quantification plus petite est plus rapide",
  },
} satisfies DeepPartialMessageTree<typeof en>;
