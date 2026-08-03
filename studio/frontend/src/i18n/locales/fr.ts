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
    pickModelFile: "Choisir un fichier de modèle sur le disque",
    ejectLoadedModel: "Éjecter le modèle chargé",
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
      expand: "Cliquez pour développer",
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
      export: "Exporter",
      recents: "Récents",
      noChatsYet: "Aucune discussion pour le moment",
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
    },
    tabs: {
      general: "Général",
      profile: "Profil",
      appearance: "Apparence",
      resources: "Système",
      chat: "Discussion",
      connections: "Connexions",
      apiKeys: "API",
      about: "À propos",
      data: "Données",
      agents: "Agents",
      voice: "Voix",
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
        sttDownloading: "Téléchargement… {progress} %",
        sttCancelDownload: "Annuler",
        sttCancellingDownload: "Annulation…",
        sttDownloadComplete: "Modèle de reconnaissance vocale téléchargé",
        sttDownloadFailed:
          "Impossible de télécharger le modèle de reconnaissance vocale",
        sttLoad: "Charger",
        sttUnload: "Décharger",
        sttUnloading: "Déchargement…",
        microphoneLabel: "Microphone",
        microphoneDescription: "Utilisé pour la dictée",
        microphoneFallbackHint:
          "Utilisé pour la dictée. Revient au périphérique par défaut du système si le moteur vocal du navigateur ne peut pas utiliser ce périphérique",
        microphoneGrantDescription:
          "Autorisez l'accès au micro pour afficher le nom des périphériques",
        allowMicrophone: "Autoriser le microphone",
        micAccessBlocked:
          "L'accès au microphone a été bloqué. Autorisez l'accès au microphone pour cette page Unsloth, puis réessayez.",
        micAccessUnsupported:
          "L'accès au microphone n'est pas pris en charge dans ce navigateur ou ce contexte.",
        systemDefault: "Par défaut",
        savedMicDisconnected: "Microphone enregistré (non connecté)",
        languageLabel: "Langue de la dictée",
        languageDescription: "Langue à reconnaître",
        languageAuto: "Auto (langue du navigateur)",
      },
      dictionary: {
        sectionTitle: "Dictionnaire de dictée",
        sectionDescription:
          "Définissez l'orthographe employée par la dictée pour certains mots ou expressions",
        manageLabel: "Orthographes personnalisées",
        manage: "Gérer",
        backToVoice: "Retour à la section Voix",
        addEntry: "Ajouter une entrée",
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
        copied: "Copié dans le presse-papiers",
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
        showMore: "Afficher plus ({count})",
        openChat: "Ouvrir la discussion",
      },
      readAloud: {
        sectionTitle: "Lecture à voix haute",
        buttonLabel: "Bouton de lecture à voix haute",
        buttonDescription: "Afficher sur les réponses de l'assistant",
        engineLabel: "Moteur TTS",
        engineSystemDescription: "Voix intégrées à l'appareil",
        engineStudioDescription:
          "Utilise le modèle audio chargé (par exemple Orpheus)",
        engineSystem: "Voix du système",
        engineStudio: "Charger un modèle TTS",
        modelLabel: "Modèle TTS",
        modelDescription:
          "Chargez un modèle audio depuis le sélecteur de modèles (par exemple Orpheus TTS)",
        voiceLabel: "Voix",
        voiceDescription: "Meilleures voix sur cet appareil",
        speedLabel: "Vitesse",
        pitchLabel: "Hauteur",
        volumeLabel: "Volume",
        previewLabel: "Écouter la voix",
        previewDescription: "Lire un court extrait",
        previewAction: "Écouter",
        stopAction: "Arrêter",
        ttsLabel: "Synthèse vocale",
        notSupported: "Non pris en charge dans ce navigateur",
      },
    },
    general: {
      title: "Général",
      description: "Préférences globales pour Unsloth.",
      account: "Compte",
      huggingFaceToken: "Token Hugging Face",
      huggingFaceTokenDescription:
        "Utilisé pour charger des modèles restreints et publier des artefacts.",
      hideToken: "Masquer le token",
      showToken: "Afficher le token",
      tokenValidated: "Jeton validé",
      password: "Mot de passe",
      passwordDescription:
        "Changez le mot de passe de ce compte Unsloth.",
      passwordDialog: {
        trigger: "Changer le mot de passe",
        title: "Changer le mot de passe",
        description:
          "Saisissez votre mot de passe actuel et choisissez-en un nouveau (au moins {minLength} caractères).",
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
          "Lorsqu'une requête compatible OpenAI nomme un autre GGUF téléchargé, le charger avant de répondre. Désactivé par défaut ; les noms inconnus continuent de servir le modèle chargé.",
        idleUnload: "Déchargement automatique en cas d'inactivité",
        idleUnloadDescription:
          "Décharger le modèle après ce nombre de secondes d'inactivité pour libérer la VRAM ; la requête suivante le recharge. 0 le maintient chargé. Minimum 60 secondes.",
        idleNeedsEnable:
          "Activez Changer de modèle par requête pour qu'un modèle déchargé se recharge à la prochaine utilisation.",
        idleActiveViaEnv:
          "Le déchargement automatique en cas d'inactivité est actif via la variable d'environnement UNSLOTH_MODEL_IDLE_TTL.",
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
      },
      previewSharing: {
        sectionTitle: "Partage de l'aperçu",
        enableLabel: "Liens d'aperçu publics",
        enableDescription:
          "Permettre à quiconque disposant d'un lien signé de discuter avec un modèle terminé, sans connexion. Désactivez pour mettre l'aperçu public hors ligne ; les liens partagés cessent de fonctionner.",
        loadError: "Échec du chargement des paramètres de partage d'aperçu.",
        saveError:
          "Échec de l'enregistrement des paramètres de partage d'aperçu.",
        revokeLabel: "Révoquer tous les liens d'aperçu",
        revokeDescription:
          "Renouveler le secret de signature pour que tous les liens partagés cessent de fonctionner. Les liens nouvellement copiés continuent de fonctionner.",
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
      },
      gettingStarted: "Prise en main",
      startOnboarding: "Démarrer l'intégration",
      startOnboardingDescription:
        "Rouvrir l'assistant de configuration sans modifier votre compte.",
      startOnboardingAction: "Démarrer l'intégration",
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
        reindexWarning:
          "N'affecte que les documents nouvellement indexés. Téléversez à nouveau les documents existants après avoir changé de modèle.",
        emptyError:
          "Saisissez un identifiant de modèle Hugging Face ou un chemin local.",
        loadError: "Échec du chargement du paramètre du modèle d'embedding.",
        saveError: "Échec de l'enregistrement du modèle d'embedding.",
        saved: "Modèle d'embedding enregistré.",
        saveAnyway: "Enregistrer quand même",
        resetAction: "Réinitialiser par défaut",
      },
      storage: {
        sectionTitle: "Stockage",
        modelsFolder: "Dossier des modèles",
        modelsFolderDescription: "Emplacement de stockage des modèles téléchargés.",
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
      avatarShape: "Forme de la photo de profil",
      avatarShapeCircle: "Cercle",
      avatarShapeRounded: "Arrondi",
      chooseSloth: "Ou choisissez un paresseux",
      nameSaved: "Nom de profil enregistré",
      namePersistErrorTitle: "Impossible d'enregistrer le nom de profil",
      namePersistErrorDescription:
        "Nom mis à jour pour cette session, mais il pourrait ne pas persister après rechargement.",
      photoUpdated: "Photo de profil mise à jour",
      photoPersistErrorTitle: "Impossible d'enregistrer la photo de profil",
      photoPersistErrorDescription:
        "Photo mise à jour pour cette session, mais elle pourrait ne pas persister après rechargement.",
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
          "Les statistiques sont calculées à partir de l'historique des discussions et des entraînements conservé par votre installation Unsloth. Rien n'est collecté, et rien n'est envoyé à Unsloth ni à un tiers.",
        emptyChats:
          "Aucune discussion pour le moment. Lancez une conversation et vos statistiques apparaîtront ici.",
        lifetimeTokens: "Tokens cumulés",
        peakTokens: "Jour record",
        longestChat: "Discussion la plus longue",
        currentStreak: "Série en cours",
        longestStreak: "Plus longue série",
        activityTitle: "Activité en tokens",
        activityDescription: "Période : {weeks} · {total}",
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
        trainingDescription: "Exécutions de fine-tuning de cet espace de travail",
        trainingRuns: "Exécutions",
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
        label: "Palette de couleurs",
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
        cpu: "CPU",
        ram: "RAM",
        disk: "Disque",
        vram: "VRAM",
        cpuCores: "{logical} cœurs logiques / {physical} physiques",
        currentLoad: "Charge actuelle",
        free: "{value} libre",
        noGpu: "Aucun GPU visible",
      },
      gpu: {
        title: "Périphériques GPU",
        noGpu:
          "Aucun GPU visible détecté. Les ressources CPU uniquement sont affichées ci-dessus.",
        unknownDevice: "GPU inconnu",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "{value} utilisé",
        free: "{value} libre",
        total: "{value} au total",
      },
      storage: {
        title: "Stockage",
        systemDisk: "Disque système",
        diskUsage: "{used} utilisé / {total}",
        diskFree: "{free} libre",
        modelsFolder: "Dossier des modèles",
        modelsFolderKeywords:
          "modeles modèles dossier repertoire répertoire chemin emplacement telechargements téléchargements cache stockage disque lecteur deplacer déplacer changer models folder path hugging face",
        modelsFolderDescription: "Emplacement de stockage des modèles téléchargés.",
        openAction: "Ouvrir",
        copyAction: "Copier le chemin",
        copied: "Chemin copié",
        openError: "Impossible d'ouvrir le dossier",
        copyError: "Impossible de copier le chemin",
        futureDownloads: "Nouveaux téléchargements uniquement",
        environmentManaged: "Géré par la variable d'environnement {variable}.",
        locationFree: "{free} libres",
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
          "Lancez un agent sur le modèle actuellement chargé dans Studio. Chargez d'abord un modèle, puis remplacez claude par n'importe quel agent pris en charge ci-dessous.",
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
          "Sélectionne un modèle. Sans --model, unsloth start utilise le modèle actuellement chargé dans Studio et échoue si aucun modèle n'est chargé.",
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
        yolo:
          "Ignore les demandes de confirmation. À n'utiliser que dans des environnements de confiance.",
      },
      remote: {
        title: "Se connecter à un Studio distant",
        description:
          "Faites pointer unsloth start vers un Studio exécuté ailleurs en définissant ces variables avant le lancement (ou passez --api-key directement) :",
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
      title: "Discussion",
      description: "Gérez l'historique des discussions stocké sur cet appareil.",
      modelDisclaimer: "Afficher l'avertissement du modèle",
      modelDisclaimerDescription:
        'Afficher "Les LLM peuvent faire des erreurs" sous la zone de discussion.',
      artifacts: {
        title: "Canvas",
        collapseHtmlBlocks: "Réduire les blocs HTML",
        collapseHtmlBlocksDescription:
          "Le mode Canvas réduit automatiquement le HTML complet. Activez ceci pour aussi réduire les documents HTML encadrés lorsque Canvas est désactivé.",
        allowNetworkAccess: "Autoriser l'accès réseau du canvas",
        allowNetworkAccessDescription:
          "Permettre aux aperçus de canvas de charger scripts, styles, polices, médias et ressources réseau depuis des CDN. Gardez désactivé pour des aperçus entièrement hors ligne.",
      },
      data: "Données",
      exportHistory: "Exporter l'historique des discussions",
      exportHistoryDescription:
        "Télécharger toutes les discussions et messages au format JSON.",
      exportAction: "Exporter",
      exportingAction: "Exportation...",
      exportConversations: "Exporter Récents et Projets",
      exportConversationsDescription:
        "Télécharger Récents ou Récents plus les discussions de projet au format JSONL brut, CSV ou JSONL ShareGPT, combinés ou par discussion.",
      exportConversationsAction: "Exporter",
      exportScopeRecents: "Récents",
      exportScopeAll: "Récents + Projets",
      exportCombinedSuffix: "(combiné)",
      exportPerChatSuffix: "(par discussion)",
      importChats: "Importer des discussions",
      importChatsDescription:
        "Importer un export JSONL, NDJSON ou CSV dans Récents.",
      importChatsAction: "Importer",
      importNoConversations: "Aucune conversation trouvée dans le fichier.",
      importedOneChat: "1 conversation importée dans Récents.",
      importedChatCount: "{count} conversations importées dans Récents.",
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
      description:
        "Gérez l'historique des discussions et les fichiers importés conservés sur cet appareil.",
      archivedChats: "Discussions archivées",
      archivedChatsDescription:
        "Consultez et gérez les discussions que vous avez archivées.",
      manageAction: "Gérer",
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
    apiKeys: {
      title: "API",
      description: "Accédez à Unsloth via l'API compatible OpenAI.",
      readDocs: "Lire la documentation de l'API",
      noAccess: "Aucun accès API pour le moment.",
      accessTokens: "Tokens d'accès",
      loadError: "Impossible de charger l'accès API.",
      createError: "Impossible de créer le token d'accès.",
      revokeError: "Impossible de révoquer le token d'accès.",
      never: "Jamais",
      tokenNamePlaceholder: "Nom du token (ex. production)",
      newAccessTokenName: "Nom du nouveau token d'accès",
      createToken: "Créer un token",
      creating: "Création...",
      newTokenCreated: "Nouveau token d'accès créé",
      accessTokenCopied: "Token d'accès copié",
      copyAccessToken: "Copier le token d'accès",
      copyNow: "Copiez maintenant - il ne sera plus affiché.",
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
        "Le port 0.0.0.0 reste accessible globalement. Pour une sécurité complète, lancez Unsloth avec --secure afin de n'exposer que ce lien HTTPS.",
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
      relativeHoursAgo: "il y a {count} h",
      relativeDaysAgo: "il y a {count} j",
      relativeMonthsAgo: "il y a {count} mois",
      relativeYearsAgo: "il y a {count} an(s)",
      expired: "expiré",
      today: "aujourd'hui",
      inDays: "dans {count} j",
      created: "Créé {value}",
      used: "Utilisé {value}",
      expires: "Expire {value}",
      actionsFor: "Actions pour {name}",
      copyPrefix: "Copier le préfixe",
      revokeToken: "Révoquer le token",
      revokeTitle: 'Révoquer le token d\'accès "{name}" ?',
      revokeDescription:
        "Les applications utilisant ce token perdent immédiatement l'accès. Cette action est irréversible.",
      revokeAction: 'Révoquer "{name}"',
      revoking: "Révocation...",
      usageNoModel:
        "Chargez ou téléchargez un modèle pour voir des exemples exécutables. Aucun modèle n'est encore disponible sur ce serveur pour figurer dans les exemples.",
    },
    about: {
      title: "À propos",
      description:
        "Documentation, notes de version, retours et informations de build.",
      studioVersion: "Version d'Unsloth",
      packageVersion: "Version du paquet",
      llamaCppVersion: "Version de llama.cpp",
      hardware: "Matériel",
      gpu: "GPU",
      cuda: "CUDA",
      rocm: "ROCm",
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
        commandText: "Texte de {label}",
        copied: "Copié",
        copyCommand: "Copier la commande",
        commandCopied: "{label} copié",
        copyNamedCommand: "Copier {label}",
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
          "L'application de bureau maintient son backend intégré à jour et vous avertira lorsqu'une nouvelle version sera disponible.",
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
      hfTokenLabel: "Jeton Hugging Face",
      hfTokenDescription:
        "Nécessaire pour les modèles et jeux de données restreints ou privés.",
      hfTokenGet: "Obtenir un jeton",
      hfTokenChecking: "Vérification du jeton…",
      modelPickerDescription:
        "Recherchez sur Hugging Face ou choisissez un modèle entraînable déjà présent sur cet appareil.",
      trainingMethod: "Méthode d'entraînement",
      trainingMethodDescription: "Choisissez comment affiner {model}",
      trainingMethodTooltip:
        "QLoRA utilise une quantification 4 bits pour réduire au minimum l'utilisation de la VRAM. LoRA utilise des poids 16 bits, tandis que l'affinage complet met à jour tous les poids.",
      datasetPickerDescription:
        "Recherchez sur Hugging Face ou choisissez un jeu de données déjà présent sur cet appareil.",
      uploadDataset: "Téléverser un jeu de données",
      uploadDatasetDescription:
        "Prend en charge CSV, JSONL, JSON et Parquet.",
      chooseFile: "Choisir un fichier",
      format: "Format",
      autoDetect: "Détection automatique",
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
      vramNeeds: "Nécessite environ {est} Go de VRAM (GPU : {total} Go)",
      vramTight: "Environ {est} Go de VRAM (limite sur {total} Go)",
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
    backToHistory: "Retour à l'historique",
    sections: {
      model: "Modèle",
      dataset: "Jeu de données",
      params: "Paramètres",
      training: "Entraînement",
      charts: "Graphiques",
      progress: "Progression de l'entraînement",
    },
    configure: {
      title: "Configurer",
      description:
        "Choisissez un modèle, un jeu de données et les paramètres d'entraînement.",
      startTraining: "Démarrer l'entraînement",
      starting: "Démarrage...",
      loadingModel: "Chargement du modèle...",
      checkingDataset: "Vérification du jeu de données...",
      trainingConfig: "Configuration d'entraînement",
    },
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
      },
      source: "Source du jeu de données",
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
      uploadLimitsHint:
        "CSV, JSONL, JSON, Parquet · jusqu’à {limit} ; PDF/DOCX/TXT → Learning Recipes",
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
      preview: "Aperçu du jeu de données",
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
      split: "Split",
      subset: "Sous-ensemble",
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
        prefixTooltip:
          "Chemin facultatif dans le bucket vers vos fichiers de jeu de données",
        accessKeyId: "ID de clé d'accès",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Clé d'accès secrète",
        secretAccessKeyPlaceholder: "Votre clé d'accès secrète AWS",
        useIamRole: "Utiliser un rôle IAM",
        useIamRoleTooltip:
          "Utiliser les identifiants d'un rôle IAM au lieu de clés d'accès (recommandé pour EC2/SageMaker)",
        testConnection: "Tester la connexion",
        connectionSuccess: "Connexion au bucket S3 réussie",
        connectionFailed: "Échec de la connexion au bucket S3",
        comingSoon: "Intégration S3 bientôt disponible",
        comingSoonDescription:
          "Le chargement de jeux de données S3 nécessite boto3. Cette fonctionnalité est en cours de développement.",
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
        "Choisissez comment vous souhaitez arrêter l'entraînement en cours.",
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
      emptyTitle: "Aucun entraînement pour le moment",
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
      runCount: "{count} entraînements",
      oneRun: "1 entraînement",
      resume: "Reprendre",
      resumeTraining: "Reprendre l'entraînement",
      resuming: "Reprise...",
      deleteRun: "Supprimer l'entraînement",
      loss: "Perte",
      steps: "Étapes",
      lossTrendSparkline: "Sparkline de tendance de la perte",
      relativeJustNow: "à l'instant",
      relativeMinutesAgo: "il y a {count} min",
      relativeHoursAgo: "il y a {count} h",
      relativeDaysAgo: "il y a {count} j",
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
    tour: {
      guidedTour: "Visite guidée",
    },
  },
} satisfies DeepPartialMessageTree<typeof en>;
