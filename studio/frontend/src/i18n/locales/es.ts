// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DeepPartialMessageTree } from "../types";
import type { en } from "./en";

export const es = {
  picker: {
    onDevice: "En el dispositivo",
    huggingFace: "Hugging Face",
    retry: "Reintentar",
    loadMore: "Cargar más",
    offlineTitle: "Estás sin conexión",
    offlineBody:
      "Cambia a Dispositivo para usar {noun} locales o en caché.",
    offlineSwitchDevice: "Dispositivo",
    searchAriaLabel: "Buscar {noun}",
    modelSourceAriaLabel: "Origen del modelo",
    hubSectionAriaLabel: "Sección del Hub",
    modelDropped: "Ya no se ofrece",
    modelDroppedByProvider: "{provider} · ya no se ofrece",
    modelDisabled: "No activado",
    modelDisabledByProvider: "{provider} · no activado",
    multipleMatches:
      "Hay varios {noun} coincidentes. Elige uno de la lista.",
    rateLimitedTitle: "Se alcanzó el límite de solicitudes de Hugging Face",
    rateLimitedBody:
      "Espera un momento y vuelve a intentar la búsqueda de {noun}.",
    hfToken: {
      label: "Token de HF",
      saved: "Guardado",
      add: "Añadir",
      savedAriaLabel: "Token de Hugging Face guardado",
      addAriaLabel: "Configurar token de Hugging Face",
      savedHint: "Token guardado. El acceso se comprueba cuando lo usas.",
      addHint:
        "Configura un token para acceder a repositorios privados y restringidos.",
    },
  },
  common: {
    cancel: "Cancelar",
    close: "Cerrar",
    delete: "Eliminar",
    done: "Listo",
    error: "Error",
    export: "Exportar",
    help: "Ayuda",
    loading: "Cargando...",
    new: "Nuevo",
    rename: "Renombrar",
    save: "Guardar",
    saving: "Guardando...",
    search: "Buscar",
    shutdown: "Apagar",
  },
  shell: {
    beta: "BETA",
    brand: "unsloth",
    product: "Unsloth",
    accountMenu: "Menú de cuenta de {name}",
    updateAvailable: "Actualización disponible",
    resize: {
      collapse: "Haz clic para contraer",
      expand: "Haz clic para expandir",
      drag: "Arrastra para redimensionar",
    },
    aria: {
      home: "Inicio de Unsloth",
      closeSidebar: "Cerrar barra lateral",
      openSidebar: "Abrir barra lateral",
      resizeSidebar: "Redimensionar o contraer la barra lateral",
      resizeRunSettings: "Redimensionar o cerrar los ajustes de ejecución",
      openRunSettings: "Abrir los ajustes de ejecución",
      chatOptions: "Opciones de chat",
      runOptions: "Opciones de ejecución",
    },
    navigation: {
      newChat: "Nuevo chat",
      returnToChat: "Volver al chat",
      returnToChats: "Volver a {count} chats",
      chatGenerating: "Generando",
      compare: "Comparar",
      search: "Buscar",
      hub: "Centro de modelos",
      projects: "Proyectos",
      train: "Entrenar",
      recipes: "Recetas",
      images: "Imágenes",
      video: "Vídeo",
      audio: "Audio",
      trainChecking: "Comprobando si este equipo admite entrenamiento...",
      videoChecking: "Comprobando si este equipo admite vídeo...",
      more: "Más",
      customizeSidebar: "Personalizar la barra lateral",
      newBadge: "Nuevo",
      export: "Exportar",
      recents: "Recientes",
      noChatsYet: "Aún no hay chats",
      showMore: "Mostrar más",
      showLess: "Mostrar menos",
      settings: "Configuración",
      api: "API",
      lightMode: "Modo claro",
      darkMode: "Modo oscuro",
      guidedTour: "Recorrido guiado",
      help: "Ayuda",
      logOut: "Cerrar sesión",
      shutdown: "Apagar",
    },
    notFound: {
      title: "Página no encontrada",
      description: "{path} no existe.",
      backToChat: "Volver al chat",
    },
    selection: {
      pinProjects: "Fijar proyectos",
      unpinProjects: "Dejar de fijar proyectos",
      deleteProjects: "Eliminar proyectos",
      deleteProjectsTitle: "Eliminar proyectos",
      deleteProjectsDescription:
        "¿Eliminar {count} proyectos? Sus chats se eliminan de forma permanente.",
      deleteProjectsFilesDescription:
        "La carpeta del espacio de trabajo de cada proyecto se elimina del disco.",
      countSelected: "{count} seleccionados",
      pinChats: "Fijar chats",
      unpinChats: "Dejar de fijar chats",
      archiveChats: "Archivar chats",
      markUnread: "Marcar como no leído",
      deleteChats: "Eliminar chats",
      deleteTitle: "Eliminar chats",
      deleteDescription: "¿Eliminar {count} chats? Esta acción no se puede deshacer.",
      deleteFilesDescription:
        "Se elimina del disco la carpeta de entorno aislado de cada chat. Los archivos que hayan escrito dentro de un proyecto permanecen en el espacio de trabajo de ese proyecto.",
      deleteFilesLabel: "Eliminar archivos y carpeta de espacio aislado",
      deleteChatFilesDescription:
        "La carpeta de espacio aislado propia de este chat se elimina del disco. Los archivos que escribió dentro de un proyecto permanecen en el espacio de trabajo de ese proyecto.",
    },
    organize: {
      sidebarHeading: "Organizar la barra lateral",
      byProject: "Por proyecto",
      inOneList: "En una sola lista",
      sortChatsBy: "Ordenar chats por",
      sortPinnedBy: "Ordenar fijados por",
      priority: "Prioridad",
      lastUpdated: "Última actualización",
      manualOrder: "Orden manual",
      moveUp: "Subir",
      moveDown: "Bajar",
      organizeChats: "Organizar chats",
      organizeProjects: "Organizar proyectos",
      sortPinnedChats: "Ordenar chats fijados",
    },
    dialog: {
      deleteChat: {
        title: "Eliminar chat",
        description: '¿Seguro que quieres eliminar este chat "{name}"?',
      },
      deleteRun: {
        title: "Eliminar ejecución de entrenamiento",
        description: '¿Seguro que quieres eliminar esta ejecución "{name}"?',
      },
      renameChat: {
        title: "Renombrar chat",
        placeholder: "Título del chat",
      },
      renameRun: {
        title: "Renombrar ejecución",
        placeholder: "Nombre de la ejecución",
      },
    },
    toast: {
      cannotDeleteRunningRun:
        "No se puede eliminar una ejecución de entrenamiento en curso",
      failedToDeleteChat: "No se pudo eliminar el chat",
      failedToDeleteRun: "No se pudo eliminar la ejecución",
      failedToRenameChat: "No se pudo renombrar el chat",
      failedToRenameRun: "No se pudo renombrar la ejecución",
    },
  },
  settings: {
    title: "Configuración",
    dialog: {
      title: "Configuración",
      description: "Gestiona tus preferencias de Unsloth.",
      closeAriaLabel: "Cerrar configuración",
      searchPlaceholder: "Buscar en la configuración…",
      searchNoResults: "No se encontró ninguna opción.",
      panelFailed: "No se pudo cargar esta sección.",
      panelReload: "Recargar",
    },
    tabs: {
      general: "General",
      profile: "Perfil",
      appearance: "Apariencia",
      resources: "Sistema",
      chat: "Chat",
      connections: "Conexiones",
      apiKeys: "API",
      remoteLan: "Remoto y LAN",
      about: "Acerca de",
      data: "Datos",
      agents: "Agentes",
      debugging: "Registros",
      voice: "Voz",
      keyboardShortcuts: "Atajos",
    },
    keyboardShortcuts: {
      title: "Atajos de teclado",
      description:
        "Cambia cualquier atajo, o bórralo para liberar la combinación para el navegador o el sistema.",
      searchPlaceholder: "Buscar atajos…",
      noResults: "Ningún atajo coincide con esa búsqueda.",
      unassigned: "Sin asignar",
      recording: "Pulsa las teclas…",
      recordingHint: "Pulsa la nueva combinación, o Esc para cancelar.",
      needsModifier: "Añade ⌘, Ctrl o Alt. Una tecla suelta se tragaría lo que escribes.",
      conflict: "También lo usa otro atajo",
      conflictShadowed: "Otro atajo tiene esta combinación y se ejecuta en lugar de este",
      edit: "Cambiar atajo",
      clear: "Quitar atajo",
      reset: "Restaurar el valor predeterminado",
      resetAll: "Restablecer todo",
      primarySlot: "Atajo",
      alternateSlot: "Atajo alternativo",
      browserReserved:
        "Puede que tu navegador se reserve esta combinación. En la app de escritorio funciona.",
      actions: {
        openSettings: {
          label: "Abrir ajustes",
          description: "Abrir el diálogo de ajustes",
        },
        openKeyboardShortcuts: {
          label: "Atajos de teclado",
          description: "Abrir esta lista de atajos",
        },
        searchChats: {
          label: "Buscar chats",
          description: "Abrir el diálogo de búsqueda de chats",
        },
        openMcpServers: {
          label: "Servidores MCP",
          description: "Configurar los servidores MCP de este chat",
        },
        logOut: {
          label: "Cerrar sesión",
          description: "Salir de Unsloth",
        },
        approveToolRequest: {
          label: "Aprobar solicitud",
          description: "Permitir la llamada de herramienta en espera",
        },
        declineToolRequest: {
          label: "Rechazar solicitud",
          description: "Denegar la llamada de herramienta en espera",
        },
        newChat: {
          label: "Nuevo chat",
          description: "Iniciar un chat nuevo",
        },
        newTemporaryChat: {
          label: "Nuevo chat temporal",
          description: "Iniciar un chat que no se guarda en el historial",
        },
        newStandaloneChat: {
          label: "Nuevo chat independiente",
          description: "Iniciar un chat fuera de cualquier proyecto",
        },
        archiveChat: {
          label: "Archivar chat",
          description: "Archivar los chats seleccionados o el chat actual",
        },
        markChatUnread: {
          label: "Marcar como no leído",
          description: "Marcar como no leídos los chats seleccionados o el chat actual",
        },
        togglePinChat: {
          label: "Alternar fijado",
          description: "Fijar o desfijar los chats seleccionados o el chat actual",
        },
        selectAllChats: {
          label: "Seleccionar todos los chats",
          description: "Seleccionar todos los chats de la barra lateral",
        },
        clearChatSelection: {
          label: "Borrar selección",
          description: "Anular la selección de los chats. Escape también la anula",
        },
        deleteSelectedChats: {
          label: "Eliminar los chats seleccionados",
          description: "Eliminar todos los chats seleccionados",
        },
        nextRecentlyViewedChat: {
          label: "Siguiente chat visto",
          description: "Avanzar por los chats vistos recientemente",
        },
        previousRecentlyViewedChat: {
          label: "Chat visto anterior",
          description: "Retroceder por los chats vistos recientemente",
        },
        nextChat: {
          label: "Chat siguiente",
          description: "Ir al siguiente chat de la barra lateral",
        },
        previousChat: {
          label: "Chat anterior",
          description: "Ir al chat anterior de la barra lateral",
        },
        nextChatNeedingAttention: {
          label: "Siguiente chat con actividad",
          description: "Ir al siguiente chat generando, en cola o no leído",
        },
        clearAllUnreads: {
          label: "Marcar todo como leído",
          description: "Marcar todos los chats como leídos",
        },
        goToRecentChat1: {
          label: "Ir al chat reciente 1",
          description: "Abrir el chat 1 de Recientes",
        },
        goToRecentChat2: {
          label: "Ir al chat reciente 2",
          description: "Abrir el chat 2 de Recientes",
        },
        goToRecentChat3: {
          label: "Ir al chat reciente 3",
          description: "Abrir el chat 3 de Recientes",
        },
        goToRecentChat4: {
          label: "Ir al chat reciente 4",
          description: "Abrir el chat 4 de Recientes",
        },
        goToRecentChat5: {
          label: "Ir al chat reciente 5",
          description: "Abrir el chat 5 de Recientes",
        },
        goToRecentChat6: {
          label: "Ir al chat reciente 6",
          description: "Abrir el chat 6 de Recientes",
        },
        switchToChat: {
          label: "Ir a Chat",
          description: "Ir al espacio de trabajo de chat",
        },
        switchToProjects: {
          label: "Ir a Proyectos",
          description: "Ir al espacio de trabajo de proyectos",
        },
        switchToHub: {
          label: "Ir al Hub de modelos",
          description: "Ir al hub de modelos",
        },
        switchToTrain: {
          label: "Ir a Entrenar",
          description: "Ir al espacio de trabajo de entrenamiento",
        },
        switchToRecipes: {
          label: "Ir a Recipes",
          description: "Ir a Data Recipes",
        },
        switchToImages: {
          label: "Ir a Imágenes",
          description: "Ir al espacio de trabajo de imágenes",
        },
        switchToVideo: {
          label: "Ir a Vídeo",
          description: "Ir al espacio de trabajo de vídeo",
        },
        switchToAudio: {
          label: "Ir a Audio",
          description: "Ir al espacio de trabajo de audio",
        },
        switchToExport: {
          label: "Ir a Exportar",
          description: "Ir al espacio de trabajo de exportación",
        },
        toggleSidebar: {
          label: "Alternar barra lateral",
          description: "Mostrar u ocultar la barra lateral",
        },
        toggleApiMonitor: {
          label: "Alternar actividad de API",
          description: "Mostrar u ocultar el monitor de actividad de API",
        },
        openModelPicker: {
          label: "Abrir selector de modelo",
          description: "Elegir el modelo de este chat",
        },
        openProjectPicker: {
          label: "Abrir selector de proyecto",
          description: "Cambiar a otro proyecto desde la cabecera del chat",
        },
        startDictation: {
          label: "Dictado",
          description: "Iniciar o detener el dictado en el redactor",
        },
        attachFiles: {
          label: "Adjuntar fotos y archivos",
          description: "Añadir un adjunto al redactor",
        },
        sendMessage: {
          label: "Enviar mensaje",
          description: "Enviar lo que hay en el redactor",
        },
        cycleReasoningEffort: {
          label: "Alternar esfuerzo de razonamiento",
          description: "Recorrer los niveles de esfuerzo de razonamiento",
        },
        increaseReasoningEffort: {
          label: "Aumentar esfuerzo de razonamiento",
          description: "Subir un nivel el esfuerzo de razonamiento",
        },
        decreaseReasoningEffort: {
          label: "Reducir esfuerzo de razonamiento",
          description: "Bajar un nivel el esfuerzo de razonamiento",
        },
        toggleFastMode: {
          label: "Alternar modo Fast",
          description: "Activar o desactivar el modo Fast",
        },
        renameChat: {
          label: "Renombrar chat",
          description: "Renombrar el chat actual",
        },
        forkChat: {
          label: "Bifurcar chat",
          description: "Ramificar un chat nuevo desde el último mensaje",
        },
        copyChatAsMarkdown: {
          label: "Copiar como Markdown",
          description: "Copiar todo el chat al portapapeles como Markdown",
        },
        copySessionId: {
          label: "Copiar ID de sesión",
          description: "Copiar el ID de sesión de sandbox de este chat",
        },
      },
    },
    debugging: {
      logSection: "Archivo de registro",
      source: "Archivo de registro",
      sourceHint: "Los ejecutores de modelos escriben sus propios registros, así que un fallo al cargar o al generar suele explicarse ahí y no en el registro del servidor.",
      path: "Ubicación",
      pathCopy: "Copiar ruta",
      refreshSection: "Actualización",
      mode: "Modo",
      modeLive: "En vivo",
      modeInterval: "Cada 3 segundos",
      modeManual: "Manual",
      refreshNow: "Actualizar ahora",
      privacyNote: "Las credenciales se enmascaran en esta vista. El archivo en disco no está enmascarado.",
      copyVisible: "Copiar el registro visible",
      empty: "Todavía no se ha registrado nada.",
      disabled: "El registro en archivo está desactivado (UNSLOTH_STUDIO_NO_FILE_LOG=1).",
      missing: "No se encontró ningún archivo de registro.",
      unreadable: "No se pudo leer el archivo de registro.",
      timeout: "La solicitud del registro agoto el tiempo de espera. Puede que el servidor no este accesible.",
      droppedNotice: "Se omitieron algunas líneas: el registro se escribió más rápido de lo que se podía leer.",
      morePending: "Aun se estan leyendo mas lineas; llegaran en la proxima actualizacion.",
      staleSession: "El registro en archivo esta desactivado, por lo que esta es una sesion anterior y no se actualizara.",
      keywords: "depuracion depurar registro registros log logs error errores fallo traza diagnostico solucion de problemas debug",
    },
    voice: {
      title: "Voz",
      description: "Micrófono, dictado, voz a texto y lectura en voz alta",
      dictation: {
        sectionTitle: "Dictado",
        engineLabel: "Motor de dictado",
        engineBrowser: "Navegador",
        engineBrowserDescription:
          "Transcribe el audio con el servicio de voz de tu navegador. Selecciona 'Transcripción local' para usar un modelo STT.",
        engineModel: "Transcripción local",
        engineModelDescription:
          "Ejecuta un modelo de voz a texto (STT) en local y funciona sin conexión. Descárgalo, cárgalo y se liberará de la memoria tras un tiempo de inactividad.",
        engineCustom: "Endpoint personalizado",
        engineCustomDescription:
          "Envía el audio grabado a un servidor STT compatible con OpenAI desde tus conexiones.",
        connectionLabel: "Conexión",
        connectionDescription:
          "Añade un servidor compatible con OpenAI y una clave API opcional en Conexiones.",
        connectionPlaceholder: "Selecciona una conexión",
        connectionEmpty: "No hay conexiones disponibles",
        customModelLabel: "Modelo",
        customModelDescription:
          "Nombre del modelo enviado a /v1/audio/transcriptions.",
        sttModelLabel: "Modelo de reconocimiento de voz",
        sttModelDescription:
          "Elige o busca un modelo STT para ejecutarlo en local.",
        sttModelSearchPlaceholder: "Buscar modelo",
        sttModelSearching: "Buscando en Hugging Face…",
        sttModelValidating: "Comprobando la compatibilidad con Whisper…",
        sttModelNoResults: "No se encontraron modelos Whisper",
        sttModelInvalid: "Este repositorio no se puede usar para el dictado",
        sttModelFailed: "No se pudo cargar el modelo STT",
        sttModelUnsupported: "La grabación no es compatible con este navegador",
        sttChecking: "Comprobando…",
        sttOnDemand: "Descargado",
        sttLoadingModel: "Cargando el modelo…",
        sttReady: "Cargado en {device}",
        sttLoaded: "Cargado",
        sttUnavailable:
          "No está instalado en este servidor. Ejecuta `unsloth studio update` para habilitar el dictado local.",
        sttRetry: "Reintentar",
        sttDownloadChecking: "Comprobando el estado de la descarga…",
        sttNotDownloaded: "Sin descargar",
        sttDownloadStatusFailed:
          "No se pudo comprobar el estado de la descarga",
        sttDownload: "Descargar",
        sttDownloadConfirmTitle: "¿Descargar {model}?",
        sttDownloadConfirmBody:
          "El dictado local funciona totalmente sin conexión, pero antes necesita el modelo de reconocimiento de voz {model}. Ocupa unos {size} y se descarga una sola vez en tu caché de Hugging Face.",
        sttDownloadConfirmBodyUnsized:
          "El dictado local funciona totalmente sin conexión, pero antes necesita el modelo de reconocimiento de voz {model}. Se descarga una sola vez en tu caché de Hugging Face.",
        sttOpenVoiceSettings: "Abrir la configuración de Voz",
        sttDownloadStarted: "Descargando {model}",
        sttDownloading: "Descargando… {progress}%",
        sttCancelDownload: "Cancelar",
        sttCancellingDownload: "Cancelando…",
        sttCancelDownloadFailed: "No se pudo cancelar la descarga",
        sttDownloadComplete: "Modelo de reconocimiento de voz descargado",
        sttModelReady: "{model} está listo para el dictado",
        sttRecommended: "Recomendado",
        sttDownloadFailed:
          "No se pudo descargar el modelo de reconocimiento de voz",
        sttLoad: "Cargar",
        sttUnload: "Liberar memoria",
        sttUnloading: "Liberando memoria…",
        microphoneLabel: "Micrófono",
        microphoneFallbackName: "Micrófono {index}",
        microphoneDescription: "Se usa para el dictado",
        microphoneFallbackHint:
          "Se usa para el dictado. Recurre al predeterminado del sistema si el motor de voz del navegador no puede usar este dispositivo",
        microphoneGrantDescription:
          "Permite el acceso al micrófono para ver los nombres de los dispositivos",
        allowMicrophone: "Permitir el acceso al micrófono",
        micAccessBlocked:
          "Se bloqueó el acceso al micrófono. Permite el acceso al micrófono para esta página de Unsloth e inténtalo de nuevo.",
        micAccessBlockedDesktop:
          "Se bloqueó el acceso al micrófono. Inténtalo de nuevo y elige Permitir, o activa el micrófono en la configuración de privacidad del sistema.",
        micAccessUnsupported:
          "El acceso al micrófono no es compatible con este navegador o contexto.",
        systemDefault: "Predeterminado del sistema",
        savedMicDisconnected: "Micrófono guardado (no conectado)",
        languageLabel: "Idioma del dictado",
        languageDescription: "Idioma que se reconocerá",
        languageAuto: "Automático (idioma del navegador)",
        languageAutoDetect: "Automático (detectar idioma)",
      },
      dictionary: {
        sectionTitle: "Diccionario de dictado",
        sectionDescription:
          "Define cómo se escriben determinadas palabras o frases al dictar",
        manageLabel: "Grafías personalizadas",
        manage: "Gestionar",
        backToVoice: "Volver a Voz",
        addEntry: "Añadir entrada",
        newEntryAria: "Nueva entrada del diccionario",
        entryPlaceholder: "María García",
        entryAria: "Entrada {index} del diccionario",
        removeEntryAria: "Eliminar la entrada {index} del diccionario",
      },
      recents: {
        sectionTitle: "Historial de dictados",
        sectionDescription:
          "Cada dictado se guarda aquí para que puedas recuperar el texto",
        manageLabel: "Historial de dictados",
        manage: "Gestionar",
        pageDescription:
          "Todos los dictados se guardan. Consúltalos, cópialos o elimínalos, o abre el chat en el que se usó cada uno.",
        searchPlaceholder: "Buscar dictados",
        sortLabel: "Ordenar dictados",
        sortNewest: "Más recientes",
        sortOldest: "Más antiguos",
        sortAlpha: "De la A a la Z",
        noMatches: "Ningún dictado coincide con tu búsqueda",
        detailTitle: "Dictado guardado",
        backToVoice: "Volver a Voz",
        backToRecents: "Volver a los dictados recientes",
        view: "Ver el dictado completo",
        empty: "Todavía no hay dictados",
        dictationColumn: "Dictado",
        dateColumn: "Fecha de creación",
        copy: "Copiar el dictado",
        copied: "Copiado al portapapeles",
        copyFailed: "No se pudo copiar al portapapeles",
        delete: "Eliminar el dictado",
        deleteTitle: "Eliminar el dictado",
        deleteDescription:
          "¿Eliminar este dictado guardado? Esta acción no se puede deshacer.",
        deleteLinkedDescription:
          "¿Eliminar este dictado guardado? También puedes eliminar el chat en el que se usó. Esta acción no se puede deshacer.",
        deleteWithChat: "Eliminar el chat y el dictado",
        deleteWithChatFailed: "No se pudo eliminar el chat",
        clear: "Borrar el historial",
        clearTitle: "Borrar el historial de dictados",
        clearDescription:
          "¿Eliminar todos los dictados guardados? Esta acción no se puede deshacer.",
        clearConfirm: "Borrar todo",
        showMore: "Mostrar más ({count})",
        openChat: "Abrir el chat",
      },
      readAloud: {
        sectionTitle: "Lectura en voz alta",
        buttonLabel: "Botón de lectura en voz alta",
        buttonDescription: "Mostrar en las respuestas del asistente",
        engineLabel: "Motor de TTS",
        engineSystemDescription: "Voces integradas del dispositivo",
        engineStudioDescription:
          "Usa el modelo de audio cargado (por ejemplo, Orpheus)",
        engineSystem: "Voces del sistema",
        engineStudio: "Cargar un modelo de TTS",
        engineCustom: "Endpoint personalizado",
        engineCustomDescription:
          "Un servidor TTS compatible con OpenAI de tus conexiones (p. ej., Kokoro)",
        connectionLabel: "Conexión",
        connectionDescription:
          "Añade un servidor compatible con OpenAI en la pestaña Conexiones",
        connectionPlaceholder: "Selecciona una conexión",
        customModelLabel: "Modelo",
        customVoiceDescription:
          "Nombre de la voz que espera el endpoint; el valor predeterminado es alloy",
        modelLabel: "Modelo de TTS",
        modelDescription:
          "Carga un modelo de audio desde el selector de modelos (por ejemplo, Orpheus TTS)",
        openAudioAction: "Abrir Audio",
        voiceLabel: "Voz",
        voiceDescription: "Las mejores voces de este dispositivo",
        speedLabel: "Velocidad",
        pitchLabel: "Tono",
        volumeLabel: "Volumen",
        previewLabel: "Escuchar la voz",
        previewDescription: "Reproducir una muestra corta",
        previewAction: "Escuchar",
        preparingAction: "Generando…",
        previewFailed: "No se pudo reproducir la muestra de TTS",
        stopAction: "Detener",
        ttsLabel: "Texto a voz",
        notSupported: "No es compatible con este navegador",
      },
    },
    general: {
      title: "General",
      description: "Preferencias globales de Unsloth.",
      account: "Cuenta",
      huggingFaceToken: "Token de Hugging Face",
      huggingFaceTokenDescription:
        "Se usa para cargar modelos restringidos y subir artefactos.",
      hideToken: "Ocultar token",
      showToken: "Mostrar token",
      clearToken: "Borrar",
      checkingToken: "Comprobando token...",
      tokenValidated: "Token validado",
      password: "Contraseña",
      passwordDescription:
        "Cambia la contraseña de esta cuenta de Unsloth.",
      passwordDialog: {
        trigger: "Cambiar contraseña",
        title: "Cambiar contraseña",
        description:
          "Introduce tu contraseña actual y elige una nueva (al menos {minLength} caracteres).",
        setTrigger: "Establecer contraseña remota",
        setTitle: "Establecer contraseña remota",
        setDescription:
          "Elige la contraseña con la que los navegadores remotos inician sesión como unsloth (al menos {minLength} caracteres). La app de escritorio de Unsloth sigue iniciando sesión automáticamente.",
        setSubmit: "Establecer contraseña",
        setting: "Estableciendo...",
        setDone: "Contraseña establecida.",
        currentPassword: "Contraseña actual",
        newPassword: "Contraseña nueva",
        confirmPassword: "Confirmar contraseña nueva",
        currentTooShort:
          "La contraseña actual debe tener al menos {minLength} caracteres.",
        newTooShort:
          "La contraseña nueva debe tener al menos {minLength} caracteres.",
        mismatch: "Las contraseñas no coinciden.",
        samePassword:
          "La contraseña nueva debe ser distinta de la actual.",
        update: "Actualizar contraseña",
        updating: "Actualizando...",
        updated: "Contraseña actualizada.",
        updateFailed: "No se pudo actualizar la contraseña.",
        newHasSpaces: "La nueva contraseña no puede contener espacios.",
      },
      chatDefaults: "Valores predeterminados del chat",
      autoTitleNewChats: "Titular automáticamente los chats nuevos",
      autoTitleNewChatsDescription:
        "Genera un título breve a partir del primer mensaje.",
      helperLlm: {
        sectionTitle: "LLM auxiliar",
        preloadOnStartup: "Precargar el LLM auxiliar al iniciar",
        preloadOnStartupDescription:
          "Descarga en segundo plano el modelo auxiliar de AI Assist al iniciar. Desactivado por defecto; AI Assist aún puede obtenerlo bajo demanda.",
        disabledByEnv:
          "Desactivado por UNSLOTH_HELPER_MODEL_DISABLE en el entorno del backend.",
        loadError: "No se pudo cargar la configuración del LLM auxiliar.",
        saveError: "No se pudo guardar la configuración del LLM auxiliar.",
      },
      modelAutoSwitch: {
        sectionTitle: "Cambio automático de modelo (API de OpenAI)",
        enable: "Cambiar de modelo según la solicitud",
        enableDescription:
          "Si una solicitud de la API especifica un GGUF que ya está descargado, carga ese modelo antes de responder. Desactivado por defecto.",
        idleUnload: "Liberar automáticamente por inactividad",
        idleUnloadDescription:
          "Libera la VRAM después de este número de segundos de inactividad. El valor 0 mantiene el modelo cargado; el mínimo es 60.",
        idleSecondsAriaLabel:
          "Segundos de inactividad antes de liberar el modelo",
        mediaEnable: "Cambiar de modelo de imagen y vídeo según la solicitud",
        mediaEnableDescription:
          "Si una solicitud de la API especifica un modelo de imagen o vídeo ya descargado, lo carga antes de generar. Es una opción independiente: la de arriba solo se aplica al modelo de chat. Desactivado por defecto.",
        mediaIdleUnload:
          "Liberar imagen y vídeo automáticamente por inactividad",
        mediaIdleUnloadDescription:
          "Libera la VRAM descargando los modelos de imagen y vídeo después de este número de segundos de inactividad. Es una configuración independiente: la de arriba solo afecta al modelo de chat. El valor 0 los mantiene cargados; el mínimo es 60.",
        mediaIdleSecondsAriaLabel:
          "Segundos de inactividad antes de liberar los modelos de imagen y vídeo",
        mediaIdlePaused:
          "En pausa mientras «Mantener el modelo en la memoria de la GPU» está activado.",
        idleNeedsEnable: "Activa primero «Cambiar de modelo según la solicitud».",
        idleActiveViaEnv:
          "La descarga automática por inactividad está activa mediante la variable de entorno UNSLOTH_MODEL_IDLE_TTL.",
        loadError:
          "No se pudo cargar la configuración de cambio automático de modelo.",
        saveError:
          "No se pudo guardar la configuración de cambio automático de modelo.",
        idleError: "Introduce 0 para mantener el modelo cargado, o al menos 60 segundos.",
        autoDownload: "Descargar los modelos que falten",
        autoDownloadDescription:
          "Descarga un GGUF que se indique en una solicitud de la API y que aún no esté descargado. Cualquiera con una clave de API podría así consumir disco y ancho de banda.",
        keepKv: "Conservar el contexto del chat al liberar el modelo por inactividad",
        keepKvDescription:
          "Guarda la caché KV antes de liberar el modelo por inactividad para que los chats reanudados no vuelvan a leer el historial. Hasta 10 GB en disco.",
        apiOnly: "Liberar solo los modelos cargados por la API",
        apiOnlyDescription:
          "La liberación por inactividad mantiene en memoria el modelo que cargaste desde Unsloth y solo libera los que cargó una solicitud a la API.",
      },
      previewSharing: {
        sectionTitle: "Compartir vista previa",
        enableLabel: "Enlaces públicos de vista previa",
        enableDescription:
          "Permite que cualquiera con un enlace firmado chatee con un modelo terminado, sin necesidad de iniciar sesión. Desactívalo para retirar la vista previa pública; los enlaces compartidos dejan de funcionar.",
        loadError:
          "No se pudo cargar la configuración para compartir vista previa.",
        saveError:
          "No se pudo guardar la configuración para compartir vista previa.",
        revokeLabel: "Revocar todos los enlaces de vista previa",
        revokeDescription:
          "Rota el secreto de firma para que todos los enlaces que hayas compartido dejen de funcionar. Los enlaces que copies después seguirán funcionando.",
        revokeAction: "Revocar enlaces",
        revoking: "Revocando...",
        revokeConfirmTitle: "¿Revocar todos los enlaces de vista previa?",
        revokeConfirmDescription:
          "Todos los enlaces de vista previa que hayas compartido dejarán de funcionar de inmediato. Esto no se puede deshacer.",
        revokeConfirmAction: "Revocar todos los enlaces",
        revoked: "Todos los enlaces de vista previa revocados",
        revokeError: "No se pudieron revocar los enlaces de vista previa",
      },
      notifications: {
        sectionTitle: "Notificaciones",
        showLlamaUpdates: "Notificaciones de actualización de llama.cpp",
        showLlamaUpdatesDescription:
          "Avisa cuando haya una compilación más reciente de llama.cpp para ejecutar nuevos modelos. Desactívalo si solo entrenas.",
        showLoadedModels: "Indicador de modelos cargados",
        showLoadedModelsDescription:
          "Muestra una pequeña tarjeta en la esquina inferior derecha con todos los modelos actualmente en memoria (chat, voz, imagen, vídeo), con un botón para expulsar cada uno.",
      },
      startup: {
        sectionTitle: "Inicio",
        launchAtLogin: "Ejecutar Unsloth al iniciar sesión",
        launchAtLoginDescription:
          "Inicia Unsloth en segundo plano cuando inicias sesión. Permanece en la barra de menús o en la bandeja del sistema hasta que lo abras.",

        closeToTray: "Cerrar en la bandeja del sistema",
        closeToTrayDescription:
          "Mantén Unsloth y su servidor ejecutándose en segundo plano al cerrar la ventana principal.",
        closeToTraySaveError:
          "No se pudo actualizar el ajuste de cierre en la bandeja del sistema.",
        loadError: "No se pudo cargar el ajuste de inicio automático.",
        saveError: "No se pudo actualizar el ajuste de inicio automático.",
      },
      downloads: {
        sectionTitle: "Descargas",
        transport: "Transporte de descarga",
        transportDescription:
          "Cómo llegan los archivos de modelos y conjuntos de datos desde Hugging Face. HTTPS retoma donde se detuvo; Xet suele ser más rápido en la primera descarga, pero reinicia el archivo si cancelas.",
        transportHint:
          "HTTPS es TLS normal: cualquier red, proxy o VPN lo permite, una transferencia cancelada o cortada continúa desde los bytes ya guardados y el uso de memoria se mantiene estable. Xet descarga bloques deduplicados, así que un repositorio que comparte datos con otro que ya tienes puede llegar mucho más rápido, pero necesita hf_xet, usa más RAM y una cancelación descarta el archivo en curso. Auto decide según la máquina: valora la RAM y si Xet se ha estado atascando aquí, y recurre a HTTPS.",
        https: "HTTPS",
        xet: "Xet",
        auto: "Auto",
        httpsHint:
          "TLS estándar. Retoma tras una cancelación, funciona en cualquier red, uso de memoria estable.",
        transportDescriptionNoResume:
          "Cómo se descargan los archivos de modelos y conjuntos de datos desde Hugging Face. En esta instalación ningún transporte puede reanudarse, así que una descarga cancelada vuelve a empezar; Xet suele ser más rápido en la primera descarga.",
        httpsHintNoResume:
          "TLS estándar. Funciona en cualquier red, uso de memoria estable. Esta instalación no puede reanudar una descarga cancelada.",
        xetHint:
          "Transferencia por bloques deduplicados. Suele ser más rápida en una descarga nueva, reinicia el archivo si cancelas y necesita más memoria.",
        autoHint:
          "Elige según la máquina y cambia a HTTPS si Xet se atasca o falla aquí.",
        autoCurrently: "Auto está usando {transport} en esta máquina.",
        xetMissing: "Xet no está disponible porque hf_xet no está instalado.",
      },
      uploads: {
        sectionTitle: "Subidas",
        maxUploadSize: "Límite de subida del conjunto de datos de entrenamiento",
        maxUploadSizeDescription: "El valor predeterminado es {defaultSize} MB.",
      },
      rag: {
        sectionTitle: "Documentos y RAG",
        embeddingModel: "Modelo de embeddings",
        embeddingModelDescription:
          "Modelo de Hugging Face o ruta local usada para indexar y buscar tus documentos. El valor predeterminado es {defaultModel}.",
        searchPlaceholder: "Buscar cualquier modelo en HF",
        reindexWarning:
          "Solo afecta a los documentos recién indexados. Vuelve a subir los existentes tras cambiar el modelo.",
        emptyError:
          "Introduce un id de modelo de Hugging Face o una ruta local.",
        loadError:
          "No se pudo cargar la configuración del modelo de embeddings.",
        saveError: "No se pudo guardar el modelo de embeddings.",
        saved: "Modelo de embeddings guardado.",
        saveAnyway: "Guardar de todos modos",
        recommended: "Recomendado",
        onDevice: "En el dispositivo",
        searching: "Buscando en Hugging Face…",
        checking: "Comprobando…",
        noResults: "No se encontraron modelos de embedding",
        download: "Descargar",
        unload: "Descargar de memoria",
        unloadFailed: "No se pudo descargar el modelo de embedding",
        downloadingStatus: "Descargando…",
        notDownloaded: "No descargado",
        notDownloadedSized: "No descargado · {size}",
        loaded: "Cargado",
        downloading: "Descargando {model}",
        downloadingDescription:
          "El progreso está en el panel de descargas. La indexación lo usará cuando termine.",
        downloadFailed: "No se pudo iniciar la descarga",
        downloadConflict: "Reanuda esta descarga desde el Hub",
        downloadBusy: "La descarga ya está en curso",
      },
      storage: {
        sectionTitle: "Almacenamiento",
        modelsFolder: "Carpeta de modelos",
        modelsFolderDescription:
          "Dónde se almacenan los modelos descargados.",
        openAction: "Abrir",
        copyAction: "Copiar ruta",
        copied: "Ruta copiada",
        openError: "No se pudo abrir la carpeta",
        copyError: "No se pudo copiar la ruta",
      },
      resetPreferences: {
        sectionTitle: "Zona de peligro",
        label: "Restablecer todas las preferencias locales",
        description:
          "Borra las preferencias solo locales. Se conservan los chats, el acceso a la API y la configuración almacenada en la base de datos.",
        action: "Restablecer preferencias",
        confirmTitle: "¿Restablecer todas las preferencias locales?",
        confirmDescription:
          "Borra las preferencias solo locales y recarga Unsloth. Se conservan los chats, el acceso a la API y la configuración almacenada en la base de datos.",
        confirmAction: "Restablecer y recargar",
      },
      permissions: {
        sectionTitle: "Permisos",
        bypassLabel: "Permisos de herramientas",
        bypassDescription:
          "Cómo aprueba Unsloth las llamadas a herramientas del chat (terminal, python, web, MCP) antes de ejecutarlas. El modo «Full access» desactiva las aprobaciones y el sandbox de código.",
      },
    },
    profile: {
      title: "Perfil",
      description: "Cómo aparece tu perfil en Unsloth.",
      changePicture: "Cambiar foto de perfil",
      displayName: "Nombre visible",
      nickname: "¿Cómo debería llamarte Unsloth?",
      nicknamePlaceholder: "Apodo",
      nicknameSaved: "Nombre preferido guardado",
      avatarShape: "Forma de la foto de perfil",
      avatarShapeCircle: "Círculo",
      avatarShapeRounded: "Redondeada",
      chooseSloth: "O elige un perezoso",
      nameSaved: "Nombre de perfil guardado",
      namePersistErrorTitle: "No se pudo guardar el nombre de perfil",
      namePersistErrorDescription:
        "El nombre se actualizó para esta sesión, pero podría no conservarse tras recargar.",
      photoUpdated: "Foto de perfil actualizada",
      photoPersistErrorTitle: "No se pudo guardar la foto de perfil",
      photoPersistErrorDescription:
        "La foto se actualizó para esta sesión, pero podría no conservarse tras recargar.",
      photoUpdateErrorTitle: "No se pudo actualizar la foto de perfil",
      imageUseError: "No se pudo usar esta imagen.",
      uploadPhoto: "Subir foto",
      removePhoto: "Quitar",
      pictureOptions: "Opciones de la foto de perfil",
      greetingSloth: "Perezoso en el saludo",
      greetingSlothDescription: "Muestra el perezoso en el saludo del chat.",
      noPicture: "Sin foto de perfil",
      noneLabel: "Ninguno",
      stats: {
        title: "Tus estadísticas",
        subtitle:
          "Todo lo que aparece a continuación se calcula a partir de tu propio historial. No se recopila ni se envía nada a Unsloth.",
        retry: "Volver a intentar",
        privacyNote:
          "Las estadísticas se calculan a partir del historial local de chats, uso de la API y entrenamientos de tu instalación de Unsloth. Nunca se guardan solicitudes, respuestas ni claves de API para las estadísticas. No se envía nada a Unsloth ni a terceros.",
        emptyChats:
          "Todavía no hay uso de chats ni de la API. Empieza una conversación o haz una solicitud autenticada a la API local.",
        lifetimeTokens: "Tokens acumulados",
        peakTokens: "Día más activo",
        longestChat: "Chat más largo",
        currentStreak: "Racha actual",
        longestStreak: "Racha más larga",
        activityTitle: "Actividad de tokens",
        activityDescription: "Período: {weeks} · {total}",
        mode: {
          daily: "Diaria",
          weekly: "Semanal",
          cumulative: "Acumulada",
        },
        cellTooltip: "{date} · {tokens}, {messages}",
        weekTooltip: "Semana del {date} · {tokens}",
        less: "Menos",
        more: "Más",
        insightsTitle: "Análisis de actividad",
        totalChats: "Chats totales",
        totalMessages: "Mensajes totales",
        tokensIn: "Tokens enviados",
        tokensOut: "Tokens generados",
        totalTokens: "Tokens totales",
        studioChatTokens: "Tokens de Unsloth Chat",
        apiTokens: "Tokens de la API",
        cachedTokens: "Tokens en caché",
        cachedValue: "{tokens} ({percent}% de la entrada)",
        avgTokensPerChat: "Media de tokens por chat",
        timeInChat: "Tiempo en chats",
        activeDays: "Días activos",
        toolCalls: "Llamadas a herramientas",
        attachments: "Archivos adjuntos",
        avgSpeed: "Velocidad media",
        bestSpeed: "Respuesta más rápida",
        firstToken: "Tiempo medio hasta el primer token",
        tokensPerSecond: "{value} tok/s",
        topModelsTitle: "Modelos más usados",
        topModelsDescription: "Ordenados por tokens intercambiados",
        modelSummary: "{tokens} · {messages}",
        noModels: "Todavía no se ha registrado uso de modelos.",
        trainingTitle: "Entrenamiento",
        trainingDescription: "Ejecuciones de fine-tuning de este espacio de trabajo",
        trainingRuns: "Ejecuciones",
        trainingCompleted: "Completadas",
        trainingSteps: "Pasos",
        trainingTokens: "Tokens entrenados",
        trainingTime: "Tiempo de entrenamiento",
        bestLoss: "Pérdida mínima",
        runSteps: "{steps}",
        runLoss: "pérdida {loss}",
      },
    },
    appearance: {
      title: "Apariencia",
      description: "Cómo se ve Unsloth en este dispositivo.",
      theme: {
        title: "Tema",
        label: "Esquema de color",
        description: "Claro, oscuro o según tu sistema.",
        system: "Sistema",
        light: "Claro",
        dark: "Oscuro",
      },
      palette: {
        label: "Paleta de colores",
        description: "Colores usados en Unsloth, en modo claro y oscuro.",
        standard: "Estándar",
        classic: "Clásica",
        minimal: "Minimalista",
      },
      custom: {
        reset: "Restablecer",
        resetAll: "Restablecer la personalización",
        preferencesTitle: "Preferencias",
        colors: {
          lightGroup: "Tema claro",
          darkGroup: "Tema oscuro",
          accent: "Acento",
          background: "Fondo",
          foreground: "Primer plano",
        },
        fontDefault: "Predeterminada",
        fontBundledGroup: "Integradas",
        fontImportedGroup: "Importadas",
        fontDeviceGroup: "En este dispositivo",
        fontFolderGroup: "Desde una carpeta",
        fontDeviceLoading: "Buscando las fuentes del dispositivo…",
        fontSearch: "Buscar fuentes…",
        fontNoResults: "No se encontraron fuentes.",
        colorPicker: {
          hue: "Tono",
          hex: "Color hexadecimal",
          eyedropper: "Seleccionar un color de la pantalla",
        },
        uiFont: {
          label: "Fuente de la interfaz",
        },
        headingFont: {
          label: "Fuente de los títulos",
        },
        chatFont: {
          label: "Fuente del chat",
        },
        codeFont: {
          label: "Fuente del código",
        },
        importFont: {
          upload: "Subir",
          scanFolder: "Seleccionar carpeta",
          alreadyAvailable:
            "Esta fuente ya está disponible, así que se usa la copia existente.",
          folderNoFonts: "No se encontraron archivos de fuente en esa carpeta.",
          remove: "Quitar",
          errorInvalidType:
            "Tipo de archivo no admitido. Usa .woff2, .woff, .ttf o .otf.",
          errorTooLarge: "El archivo de fuente es demasiado grande (máx. 1,5 MB).",
          errorLimit: "Puedes importar hasta 3 fuentes.",
          errorStorageFull:
            "No hay suficiente almacenamiento local para esta fuente. Quita antes una fuente importada.",
          errorFailed: "No se pudo cargar este archivo de fuente.",
        },
        uiFontSize: {
          label: "Tamaño de fuente de la interfaz",
          description: "Ajusta el tamaño base usado en la interfaz de Unsloth.",
        },
        codeFontSize: {
          label: "Tamaño de fuente del código",
          description: "Ajusta el tamaño base usado para el código.",
        },
        fontSmoothing: {
          label: "Suavizado de fuentes",
          description: "Usa antialiasing para suavizar el texto.",
        },
        contrast: {
          label: "Contraste",
          description: "Intensidad de los bordes y del texto secundario.",
        },
        reduceMotion: {
          label: "Reducir el movimiento",
          description: "Reduce las animaciones o sigue la configuración del sistema.",
          system: "Sistema",
          on: "Activado",
          off: "Desactivado",
        },
        pointerCursors: {
          label: "Usar cursores de puntero",
          description:
            "Cambia el cursor a un puntero al pasar por encima de elementos interactivos.",
        },
      },
      language: {
        title: "Idioma",
        label: "Idioma de la interfaz",
        description: "El idioma que usa Unsloth.",
        autoDetect: "Detección automática",
      },
      layout: {
        title: "Diseño",
        compactSidebar: "Fijar la barra lateral por defecto",
        compactSidebarDescription:
          "Mantén la barra lateral expandida en lugar de contraerla a iconos.",
      },
      sidebarNav: {
        title: "Navegación de la barra lateral",
        description:
          "Fija y reordena las pestañas de la barra lateral. Las pestañas sin fijar se agrupan en el menú «Más»; si solo queda una pestaña sin fijar, se oculta en lugar de crear un menú de un único elemento. «Nuevo chat» queda fijo.",
        dragToReorder: "Arrastra para reordenar",
        pinToSidebar: "Fijar {name} en la barra lateral",
        moreHolds: "Más ({count})",
      },
      sidebarMenu: {
        title: "Menú de la barra lateral",
        description:
          "Muestra, oculta y reordena los elementos del menú de perfil de la barra lateral. Configuración, Ayuda, Cerrar sesión y Apagar quedan fijos.",
        darkModeToggle: "Modo oscuro",
        dragToReorder: "Arrastra para reordenar",
      },
    },
    resources: {
      title: "Sistema",
      description:
        "Monitorea el hardware y el almacenamiento de este servidor de Unsloth.",
      liveUpdates: "Actualizaciones en vivo",
      floatingWindow: "Ventana flotante",
      disableOverlay: "Desactivar superposición",
      liveMonitor: {
        title: "Monitor en vivo",
        apiTitle: "Monitor de API",
        summary: "Solicitudes en curso, errores y uso de tokens",
        status: "{active} activas · {recent} recientes · {model}",
        noModelLoaded: "ningún modelo cargado",
        autoOpen: "Mostrar automáticamente el monitor flotante",
        autoOpenDescription:
          "Abre un panel pequeño cuando llega tráfico de la API.",
        cpu: "CPU",
        ram: "RAM",
        disk: "Disco",
        vram: "VRAM",
        cpuCores: "{logical} núcleos lógicos / {physical} físicos",
        currentLoad: "Carga actual",
        free: "Libre: {value}",
        noGpu: "No hay GPU visible",
      },
      gpu: {
        title: "Dispositivos GPU",
        ggufInference: "Inferencia GGUF",
        unavailable: "No disponible",
        detecting: "Buscando GPU...",
        unreadable: "No se pudo leer el hardware de este servidor.",
        noGpu:
          "No se detectó ninguna GPU visible. Arriba se muestran los recursos solo de CPU.",
        unknownDevice: "GPU desconocida",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "{value} en uso",
        free: "Libre: {value}",
        total: "{value} en total",
      },
      llamaBackend: {
        title: "Motor de inferencia GGUF",
        label: "Backend de cómputo",
        description: "El backend que llama.cpp usa para ejecutar modelos GGUF.",
        runningOn: "llama.cpp se ejecuta actualmente en {backend}.",
        hint: "Instala la compilación de llama.cpp para este backend y la mantiene entre actualizaciones. Útil cuando la elección automática falla o el controlador de tu GPU no la admite. Solo se listan los backends con una compilación para este equipo; el entrenamiento no se ve afectado.",
        autoWith: "Automático ({backend})",
        apply: "Aplicar",
        applying: "Instalando...",
        applyHint: "Descarga la nueva compilación y reinicia llama.cpp. Se descargará el modelo que esté cargado.",
        applyHintWithSize: "Descarga {size} y reinicia llama.cpp. Se descargará el modelo que esté cargado.",
        switchedTo: "llama.cpp ahora se ejecuta en {backend}.",
        switchFailed: "No se pudo cambiar el backend de llama.cpp.",
        switchInterrupted: "El cambio se interrumpió antes de completarse.",
        envLocked: "Fijado en {backend} por la variable de entorno UNSLOTH_LLAMA_CPP_BACKEND, que tiene prioridad sobre este ajuste.",
        customPath: {
          label: "Carpeta personalizada de llama.cpp",
          description: "Usa tu propia compilación de llama-server.",
          hint: "Elige la carpeta de llama.cpp que contiene llama-server o una compilación donde esté en build/bin. El runtime personalizado se usa para chat GGUF, embeddings y modelos de voz compatibles. Las variables de entorno siguen teniendo prioridad.",
          automatic: "Automático (incluido)",
          bundled: "Usa el runtime de llama.cpp instalado por Unsloth.",
          active: "Tu llama-server personalizado se usará la próxima vez que cargues un modelo.",
          environmentManaged: "Gestionado por la variable de entorno {variable}.",
          missingBinary: "llama-server ya no está disponible en esta carpeta. Elige otra carpeta o usa el runtime incluido.",
          reloadRequired: "Vuelve a cargar el modelo para usar el llama-server seleccionado.",
          change: "Cambiar",
          saving: "Guardando...",
          useBundled: "Usar incluido",
          chooseTitle: "Elegir carpeta de llama.cpp",
          chooseAction: "Usar esta carpeta",
          saved: "Carpeta de llama.cpp actualizada",
          saveError: "No se pudo actualizar la carpeta de llama.cpp",
        },
        backends: {
          auto: "Automático",
          cpu: "CPU",
          cuda: "CUDA",
          rocm: "ROCm",
          vulkan: "Vulkan",
          metal: "Metal",
        },
        unsupported: {
          notInstalled: "No se encontró una instalación de llama.cpp gestionada, así que no hay backend que cambiar.",
          localLink: "llama.cpp es un directorio local que enlazaste tú, así que Unsloth no lo reemplazará.",
          sourceBuild: "Este llama.cpp se compiló desde el código fuente, así que su backend no se puede cambiar desde aquí.",
          customPath: "Hay una carpeta personalizada de llama.cpp seleccionada. Su compilación determina el backend de cómputo.",
          unresolved: "No se pudieron consultar los backends disponibles. Revisa tu conexión e inténtalo de nuevo.",
        },
        // No se muestra: términos adicionales para la búsqueda de ajustes.
        llamaBackendKeywords:
          "llama.cpp backend gguf inferencia cuda rocm hip vulkan metal cpu gpu acelerador prebuilt cambiar motor",
      },
      modelMemory: {
        title: "Memoria del modelo",
        keepResident: "Mantener el modelo en la memoria de la GPU",
        keepResidentDescription: "Permanece en la VRAM entre mensajes.",
        keepResidentHint: "No devuelve los pesos a la RAM del sistema mientras el modelo siga cargado. Desactiva la descarga automática por inactividad y, cuando los pesos sí residen en la RAM del host (memoria unificada o descarga parcial a la GPU), también pasa --mlock para que el sistema operativo no los pagine ni los vuelva a subir en tu siguiente mensaje.",
        noRamReserve: "No reservar RAM del sistema para el modelo",
        noRamReserveDescription: "No mantiene una copia completa en la RAM.",
        noRamReserveHint: "Transfiere los pesos a la VRAM en lugar de mantener una copia completa en la RAM. Conserva la carga mapeada en memoria de llama.cpp y elimina --no-mmap y --mlock.",
        mlockVetoed: "--mlock permanece desactivado: fijar el modelo reservaría RAM para todo él. La descarga automática por inactividad sigue desactivada.",
        memlockCapped: "Este sistema limita la memoria bloqueada a {limit}. Un modelo mayor no quedará fijado por completo; aumenta el límite con ulimit -l.",
        reloadRequired: "Vuelve a cargar el modelo para aplicar las nuevas opciones de memoria.",
        loadError: "No se pudieron cargar los ajustes de memoria del modelo",
        saveError: "No se pudieron guardar los ajustes de memoria del modelo",
        // Not rendered: extra terms the settings search matches these rows on.
        modelMemoryKeywords:
          "mlock memlock ulimit vram gpu memoria ram residente fijar anclar bloquear mantener cargado descargar inactivo mmap no-mmap load-mode paginacion intercambio",
      },
      storage: {
        title: "Almacenamiento",
        systemDisk: "Disco del sistema",
        diskUsage: "{used} en uso / {total}",
        diskFree: "Libre: {free}",
        modelsFolder: "Carpeta de modelos",
        modelsFolderKeywords:
          "modelos carpeta directorio ruta ubicacion ubicación descargas descarga cache caché almacenamiento disco unidad mover cambiar models folder path hugging face",
        modelsFolderDescription: "Dónde se guardan los modelos descargados.",
        modelsFolderHint: "Dónde se guardan los modelos descargados. Cámbialo para mantener los modelos fuera de tu unidad del sistema. Solo se aplica a las descargas nuevas: los modelos que ya tienes se quedan donde están.",
        openAction: "Abrir",
        copyAction: "Copiar ruta",
        copied: "Ruta copiada",
        openError: "No se pudo abrir la carpeta",
        copyError: "No se pudo copiar la ruta",
        futureDownloads: "Solo las descargas nuevas",
        environmentManaged: "Gestionado por la variable de entorno {variable}.",
        locationFree: "{free} libres",
        changeAction: "Cambiar",
        resetAction: "Usar la ubicación predeterminada",
        chooseTitle: "Elegir la ubicación de descarga de los modelos",
        chooseAction: "Usar para las próximas descargas",
        cacheSaved: "Se actualizó la ubicación de descarga de los modelos",
        cacheSaveError:
          "No se pudo actualizar la ubicación de descarga de los modelos",
        cachePickerError: "No se pudo abrir el selector de carpetas",
      },
      environment: {
        title: "Entorno",
        backend: "Backend",
        python: "Python",
        torch: "Torch",
        transformers: "Transformers",
        uptime: "Tiempo activo",
        processMemory: "Memoria del proceso",
        notInstalled: "No instalado",
        unknown: "Desconocido",
        vramWithShared: "{vram} de VRAM + {shared} de memoria compartida",
      },
    },
    agents: {
      title: "Agentes",
      description:
        "Conecta agentes de programación como Claude Code y Codex a un modelo local con unsloth start.",
      intro:
        "conecta Claude Code, Codex, Hermes, OpenClaw, OpenCode y otros agentes a un modelo servido localmente por Unsloth, totalmente sin conexión. Ejecuta un servidor compatible con OpenAI y nunca modifica los archivos de configuración de tu agente.",
      readDocs: "Leer la documentación",
      copy: "Copiar",
      copied: "Copiado",
      commandBuilder: "Generador de comandos",
      agent: "Agente de programación",
      model: "Modelo",
      searchModels: "Buscar modelos GGUF...",
      noModels: "No hay modelos GGUF que coincidan.",
      showingModels:
        "Mostrando {shown} de {total} coincidencias. Sigue escribiendo para acotar la lista.",
      quantization: "Cuantización",
      loadingQuantizations: "Cargando cuantizaciones...",
      noQuantizations: "Sin cuantización independiente",
      recommended: "Recomendado",
      downloaded: "Descargado",
      quantizationLoadError:
        "No se pudieron cargar todas las cuantizaciones. El comando usará el valor de modelo que esté disponible.",
      generatedCommand: "Comando generado",
      docs: "Documentación",
      agentDocs: "Abrir la documentación de configuración de {agent}",
      copyGeneratedCommand: "Copiar el comando generado",
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
        "Codex requiere un modelo GGUF servido por llama-server. Otros agentes también pueden usar modelos basados en transformers; quita --model para usar el modelo ya cargado en Unsloth.",
      subagent: {
        title: "Usar un modelo local como subagente",
        description:
          "Mantén {agent} en su modelo actual y delega tareas concretas en este modelo local de Unsloth.",
        setupCommand: "Comando de configuración",
        copySetupCommand: "Copiar el comando de configuración del subagente",
        usagePrompt: "Luego, en {agent}, escribe:",
        copyUsagePrompt: "Copiar el prompt de uso del subagente",
        defaultPrompt: "Crea un agente local para implementar esta función.",
        opencodePrompt: "@unsloth encuentra la causa de este fallo en las pruebas",
      },
      quickstart: {
        title: "Crear un comando",
        description:
          "Inicia un agente con el modelo cargado actualmente en Unsloth. Carga primero un modelo y luego cambia claude por cualquiera de los agentes compatibles que aparecen abajo.",
        noneDetected:
          "No se encontró ninguna CLI de agente compatible en tu PATH.",
        installed: "Instalado",
      },
      supportedAgents: {
        title: "Agentes compatibles",
        description: "Cada agente se inicia con su propio comando:",
        requiresGguf: "Necesita un modelo GGUF",
      },
      models: {
        title: "Elegir un modelo",
        description:
          "Usa --model para elegir un modelo y una cuantización, y --context-length para definir la ventana. Usa un sufijo de cuantización o la opción explícita --gguf-variant.",
        suffixLabel: "Con un sufijo de cuantización",
        variantLabel: "Con una opción explícita de variante",
      },
      options: {
        title: "Opciones comunes",
        description:
          "Primero se procesan las opciones de Unsloth; todo lo que Unsloth no reconoce se pasa directamente al agente.",
        model:
          "Selecciona un modelo. Sin --model, unsloth start usa el modelo cargado actualmente en Unsloth y da error si no hay ninguno.",
        contextLength:
          "Define la longitud de contexto solicitada (alias: --max-seq-length).",
        ggufVariant: "Elige la variante de cuantización GGUF.",
        loadIn4bit:
          "Activa o desactiva la carga en 4 bits para modelos de Hugging Face.",
        tensorParallel:
          "Activa o desactiva el paralelismo de tensores entre varias GPU.",
        serve: "Activa o desactiva el servidor local automático.",
        launch: "Inicia el agente, o solo imprime el comando y el entorno.",
        persist:
          "Conserva entre ejecuciones el almacenamiento de agentes que gestiona Unsloth.",
        asSubagent:
          "Mantén el agente principal en su modelo actual y registra Unsloth como subagente local (Claude Code, Codex y OpenCode).",
        apiKey: "Indica tu clave de API de Unsloth (o define UNSLOTH_API_KEY).",
        reasoning:
          "Usar razonamiento en el chat: on, off o auto. Auto sigue la plantilla de chat del modelo, lo que suele significar on.",
        reasoningEffort:
          "Esfuerzo de razonamiento que se pasa a la plantilla de chat del modelo, por ejemplo medium. Los niveles dependen del modelo, así que usa uno que acepte. Sin valor se mantiene el de la plantilla.",
        yolo:
          "Omite las solicitudes de aprobación. Úsalo solo en entornos de confianza.",
      },
      remote: {
        title: "Conectar con un Unsloth Studio remoto",
        description:
          "Apunta unsloth start a un Unsloth Studio que se ejecuta en otro lugar definiendo estas variables antes de iniciar el agente (o pasa --api-key directamente):",
      },
      passthrough: {
        title: "Pasar argumentos al agente",
        description:
          "Los argumentos que van después de las opciones de Unsloth se reenvían al propio agente, así que comandos nativos como resume siguen funcionando:",
      },
      dryRun: {
        title: "Previsualizar sin iniciar",
        description:
          "Añade --no-launch para imprimir el entorno y el comando en vez de iniciar el agente. Si --model está definido, el modelo aún puede resolverse y cargarse.",
      },
    },
    chat: {
      projectsSection: "Mostrar la sección Proyectos",
      projectsSectionDescription:
        "Agrupa los chats de proyecto bajo un encabezado Proyectos. Desactívalo para listarlos en Recientes.",
      title: "Chat",
      description: "Personaliza cómo funciona el chat en este dispositivo.",
      modelSelection: {
        title: "Configuración de selección del modelo",
        expandQuantizations: "Expandir las cuantizaciones",
        expandQuantizationsDescription:
          "Activado: los modelos GGUF de «On Device» muestran sus cuantizaciones de inmediato. Desactivado: haz clic en un modelo para ver sus cuantizaciones.",
        showAllQuantizations: "Mostrar todas las cuantizaciones",
        showAllQuantizationsDescription:
          "Activado: muestra todas las cuantizaciones de «On Device», incluidas las que no están descargadas. Desactivado: muestra solo las cuantizaciones descargadas.",
        showMemoryBar: "Mostrar barra de uso de VRAM",
        showMemoryBarDescription:
          "Muestra debajo de la fila de cada modelo descargado su uso estimado de VRAM: pesos, caché KV con la longitud de contexto con la que se cargará y cualquier reserva de borrador especulativo.",
      },
      menu: {
        title: "Menú del chat",
        description:
          "Fija elementos en el menú lateral «+» del chat. Los demás pasarán a «Más».",
        chatWithFiles: "Chat con archivos (RAG)",
        mcp: "MCP",
        savedPrompts: "Prompts guardados",
        compareChat: "Comparar chats",
        exportChat: "Exportar chat",
      },
      pastedTextThreshold: "Condensar pegados largos",
      pastedTextThresholdDescription: "El texto pegado más largo que esto se convierte en un adjunto .txt en lugar de llenar el cuadro de mensaje. Pulsa {shortcut} para pegar en el cuadro de mensaje de todos modos.",
      pastedTextThresholdOff: "Desactivado",
      showResponseModel: "Mostrar el modelo de respuesta",
      showResponseModelDescription:
        "Muestra los metadatos del modelo en las respuestas del asistente.",
      modelDisclaimer: "Mostrar aviso del modelo",
      modelDisclaimerDescription:
        'Muestra "Los LLM pueden cometer errores" bajo el cuadro de chat.',
      projectAttachments: "Compartir archivos en todo el proyecto",
      projectAttachmentsDescription:
        "Valor predeterminado para los archivos adjuntos en un chat que pertenece a un proyecto: indexarlos para todo el proyecto para que cualquier chat pueda usarlos. Cada chat puede cambiarlo desde el menú de adjuntos.",
      rememberParamsPerModel: "Recordar los ajustes por modelo",
      rememberParamsPerModelDescription:
        "Al cambiar de modelo se restauran la temperatura, el prompt y los demás ajustes que usaste por última vez con ese modelo. Desactivado, se mantiene un único conjunto de ajustes para todos los modelos.",
      autoCompact: "Compactar automáticamente chats largos",
      autoCompactDescription:
        "Cuando un chat GGUF local alcance la longitud de contexto configurada, descarta los turnos antiguos en vez de devolver un error. Esto no depende de la VRAM libre.",
      compactionStyle: "Cuando se llena el contexto",
      compactionStyleDescription:
        "Usar el valor del servidor conserva UNSLOTH_CONTEXT_POLICY. Restablecer la conversación mantiene el último turno y las instrucciones permanentes. Una ventana deslizante descarta los turnos más antiguos y puede conservar más historial reciente.",
      compactionStyleInherit: "Usar valor del servidor",
      compactionStyleCheckpoint: "Restablecer conversación",
      compactionStyleRollingDefault:
        "Descartar turnos antiguos (~25% de espacio extra)",
      compactionStyleRolling10:
        "Descartar turnos antiguos (~10% de espacio extra)",
      compactionStyleRolling5:
        "Descartar turnos antiguos (~5% de espacio extra)",
      compactionStyleRollingNone:
        "Descartar turnos antiguos (sin recorte adicional)",
      autoCompactKeywords:
        "compactación compactar automáticamente contexto ventana truncar deslizante checkpoint margen compaction rolling headroom",
      thinking: {
        collapseByDefault: "Contraer el razonamiento de forma predeterminada",
        collapseByDefaultDescription:
          "Mantén el razonamiento contraído mientras el modelo piensa, en lugar de abrirlo automáticamente. Expande cualquier bloque para leerlo.",
      },
      tools: {
        collapseByDefault: "Contraer la actividad de herramientas por defecto",
        collapseByDefaultDescription:
          "Mantén contraídas las entradas y salidas de las herramientas mientras se ejecutan. Expande cualquier fila para inspeccionarla.",
      },
      webSearch: {
        title: "Búsqueda web",
        images: "Mostrar imágenes de la búsqueda web",
        imagesDescription:
          "Permite que la búsqueda web devuelva imágenes y obtiene una por cada elemento que enumera una respuesta. Unsloth descarga y redimensiona las miniaturas, así que el navegador nunca contacta con los servidores de imágenes.",
      },
      artifacts: {
        title: "Canvas",
        collapseHtmlBlocks: "Contraer bloques HTML",
        collapseHtmlBlocksDescription:
          "El modo Canvas contrae el HTML completo automáticamente. Activa esta opción para contraer también los documentos HTML delimitados por bloques de código cuando Canvas esté desactivado.",
        allowNetworkAccess: "Permitir acceso de red en Canvas",
        allowNetworkAccessDescription:
          "Permite que las vistas previas de Canvas carguen scripts, estilos, fuentes, medios y recursos de red desde CDNs. Mantenlo desactivado para vistas previas totalmente sin conexión.",
        blockedBanner: "Se bloqueó {count} recurso externo de {hosts}.",
        blockedBannerPlural: "Se bloquearon {count} recursos externos de {hosts}.",
        blockedBannerAction: "Permitir en este Canvas",
      },
      data: "Datos",
      exportHistory: "Exportar historial de chat",
      exportHistoryDescription:
        "Descarga todos los chats y mensajes como JSON.",
      exportAction: "Exportar",
      exportingAction: "Exportando...",
      exportConversations: "Exportar Recientes y Proyectos",
      exportConversationsDescription:
        "Descarga Recientes, o Recientes más los chats de proyectos, como Training JSONL, CSV o JSONL de ShareGPT, combinados o por chat. Message JSONL solo está disponible por chat.",
      exportConversationsAction: "Exportar",
      exportScopeRecents: "Recientes",
      exportScopeAll: "Recientes + Proyectos",
      exportCombinedSuffix: "(combinado)",
      exportPerChatSuffix: "(por chat)",
      importChats: "Importar chats",
      importChatsDescription:
        "Importa una exportación de Open WebUI, JSONL, NDJSON o CSV a Recientes.",
      importChatsAction: "Importar",
      importNoConversations: "No se encontraron conversaciones en el archivo.",
      importedOneChat: "Se importó 1 conversación a Recientes.",
      importedChatCount: "Se importaron {count} conversaciones a Recientes.",
      importingChats: "Importando chats: {count} hasta ahora ({percent}%)...",
      importedChatCountPartial: "Se importaron {count} conversaciones a Recientes; {failed} no se pudieron guardar.",
      importFailed: "La importación falló.",
      clearHistory: "Borrar historial de chat",
      clearHistoryDescription: "Elimina el historial de chat de este dispositivo.",
      clearAction: "Borrar",
      clearAllChats: "Borrar todos los chats",
      clearAllChatsDescription:
        "Elimina permanentemente todos los chats de este dispositivo.",
      noChatsToClear: "No hay chats para borrar.",
      clearOneChatDescription:
        "Elimina permanentemente el único chat de este dispositivo.",
      clearChatCountDescription:
        "Elimina permanentemente los {count} chats de este dispositivo.",
      clearChatsAction: "Borrar chats",
      clearOneChatTitle: "¿Borrar 1 chat?",
      clearChatsTitle: "¿Borrar {count} chats?",
      clearChatsConfirmDescription:
        "Elimina permanentemente todos los chats de este dispositivo. Esto no se puede deshacer.",
      clearingAction: "Borrando...",
      clearOneChatAction: "Borrar 1 chat",
      clearChatCountAction: "Borrar {count} chats",
      clearedAllChats: "Se borraron todos los chats",
      clearedOneChat: "Se borró 1 chat",
      clearedChatCount: "Se borraron {count} chats",
      someChatsCouldNotBeCleared: "Algunos chats no se pudieron borrar",
      chatsClearedRemainOne:
        "{clearedCount} chats borrados; queda 1 chat. Inténtalo de nuevo.",
      chatsClearedRemain:
        "{clearedCount} chats borrados; quedan {remainingCount} chats. Inténtalo de nuevo.",
      oneChatClearedRemain:
        "1 chat borrado; quedan {remainingCount} chats. Inténtalo de nuevo.",
      oneChatClearedRemainOne:
        "1 chat borrado; queda 1 chat. Inténtalo de nuevo.",
      storageClearFailedOne:
        "Falló un borrado del almacenamiento; puede quedar 1 chat. Inténtalo de nuevo.",
      storageClearFailed:
        "Falló un borrado del almacenamiento; pueden quedar {count} chats. Inténtalo de nuevo.",
      failedToClearChats: "No se pudieron borrar los chats",
    },
    data: {
      title: "Datos",
      backToData: "Volver a Datos",
      exportFailed: "No se pudieron exportar los chats",
      description:
        "Gestiona el historial de chats y los archivos subidos que se guardan en este dispositivo.",
      archivedChats: "Chats archivados",
      archivedChatsDescription:
        "Consulta y gestiona los chats que has archivado.",
      archivedImages: "Imágenes archivadas",
      archivedImagesDescription: "Consulta y gestiona las imágenes que has archivado.",
      archivedVideos: "Vídeos archivados",
      archivedVideosDescription: "Consulta y gestiona los vídeos que has archivado.",
      manageAction: "Gestionar",
      manageChats: "Gestionar chats",
      manageChatsDescription:
        "Selecciona varios chats para moverlos, fijarlos, archivarlos, exportarlos o eliminarlos.",
      exportArchivedChats: "Exportar",
      exportingArchivedChats: "Exportando...",
      exportedOneArchivedChat: "Se exportó 1 chat archivado",
      exportedArchivedChatCount: "Se exportaron {count} chats archivados",
      noArchivedChatsToExport: "No hay chats archivados para exportar.",
      failedToExportArchivedChats:
        "No se pudieron exportar los chats archivados",
      archiveAllChats: "Archivar todos los chats",
      archiveAllChatsDescription:
        "Mueve al archivo todos los chats de Recientes y Proyectos.",
      noChatsToArchive: "No hay chats para archivar.",
      archiveAllAction: "Archivar todos",
      archivingAction: "Archivando...",
      archiveAllChatsTitle: "¿Archivar todos los chats?",
      archiveAllChatsConfirmDescription:
        "Mueve al archivo todos los chats de este dispositivo. Los chats archivados siguen disponibles y se pueden desarchivar en cualquier momento.",
      archivedAllChats: "Se archivaron todos los chats",
      archivedOneChat: "Se archivó 1 chat",
      archivedChatCount: "Se archivaron {count} chats",
      failedToArchiveChats: "No se pudieron archivar los chats",
      confirmBeforeDeleting: "Confirmar antes de eliminar",
      confirmBeforeDeletingDescription:
        "Pide confirmación antes de eliminar un chat. Desactívalo para eliminar los chats al instante.",
      alwaysDeleteFiles: "Eliminar siempre los archivos",
      alwaysDeleteFilesDescription:
        "Al eliminar un chat también se quita del disco su carpeta de entorno aislado. Los archivos que haya escrito dentro de un proyecto permanecen en el espacio de trabajo de ese proyecto.",
      filesSection: "Archivos",
      uploadedFiles: "Archivos subidos",
      uploadedFilesDescription:
        "Consulta y gestiona los archivos subidos a chats, proyectos y bases de conocimiento.",
      fineTuneExport: "Usar los chats como datos de entrenamiento",
      fineTuneExportDescription:
        "Crea un conjunto de datos JSONL de fine-tuning a partir de tus chats. Cárgalo en Entrenar, refínalo en Recetas o expórtalo.",
      fineTuneExportAction: "Exportar JSONL",
      fineTuneRunAction: "Ejecutar",
      fineTuneExportingAction: "Exportando...",
      fineTuneOpenRecipesAction: "Abrir en Recetas",
      fineTuneOpeningRecipesAction: "Abriendo...",
      fineTuneTrainAction: "Cargar en la pestaña Entrenar",
      fineTuneTrainingAction: "Cargando...",
      fineTuneExportFailed:
        "No se pudieron exportar los datos de entrenamiento",
      fineTuneRecipeFailed: "No se pudieron abrir los chats en Recetas",
      fineTuneTrainFailed:
        "No se pudo cargar el conjunto de datos en la pestaña Entrenar",
    },
    connections: {
      title: "Conexiones",
      description: "Gestiona proveedores y conexiones externas.",
    },
    remoteLan: {
      title: "Remoto y LAN",
      description:
        "Accede a este Unsloth desde tus otros dispositivos, por tu red local o mediante una URL pública temporal.",
    },
    apiKeys: {
      title: "API",
      description: "Accede a Unsloth mediante la API compatible con OpenAI.",
      readDocs: "Leer la documentación de la API",
      noAccess: "Aún no hay acceso a la API.",
      accessTokens: "Tokens de acceso",
      loadError: "No se pudo cargar el acceso a la API.",
      createError: "No se pudo crear el token de acceso.",
      revokeError: "No se pudo revocar el token de acceso.",
      never: "Nunca",
      tokenNamePlaceholder: "Nombre del token (p. ej. producción)",
      newAccessTokenName: "Nombre del nuevo token de acceso",
      createToken: "Crear token",
      creating: "Creando...",
      newTokenCreated: "Nuevo token de acceso creado",
      accessTokenCopied: "Token de acceso copiado",
      copyAccessToken: "Copiar token de acceso",
      copyNow: "Cópialo ahora: no se volverá a mostrar.",
      usageExamples: "Ejemplos de uso",
      usageTools: "Herramientas",
      exampleCurlTools: "curl + herramientas",
      examplePythonTools: "Python + herramientas",
      exampleJavaScriptTools: "JavaScript + herramientas",
      exampleCurlAdvanced: "curl + avanzado",
      examplePythonAdvanced: "Python + avanzado",
      exampleJavaScriptAdvanced: "JavaScript + avanzado",
      osUnix: "Linux / macOS / WSL",
      osWindows: "Windows",
      secureHttps: "HTTPS seguro",
      secureHttpsHint:
        "El puerto 0.0.0.0 sigue siendo accesible globalmente. Para máxima seguridad, inicia Unsloth con --secure para exponer solo este enlace HTTPS.",
      copyTunnelUrl: "Copiar URL del túnel",
      copySnippet: "Copiar fragmento",
      copy: "Copiar",
      copied: "Copiado",
      setupDocs: "Documentación de configuración:",
      codingAgents: "Agentes de programación",
      codingAgentsHint:
        "Inicia un agente de programación conectado a este servidor. Usa el modelo cargado; un servidor local genera una clave de API automáticamente y uno remoto la incluye en el comando.",
      codingAgentsSwap:
        "Reemplaza claude por codex, openclaw, opencode o hermes.",
      codingAgentDetected: "Instalado en esta máquina",
      codingAgentsDetectedHint: "Detectados en esta máquina: {agents}.",
      relativeNever: "nunca",
      relativeJustNow: "justo ahora",
      expired: "caducado",
      today: "hoy",
      created: "Creado {value}",
      used: "Usado {value}",
      expires: "Caduca {value}",
      actionsFor: "Acciones para {name}",
      copyPrefix: "Copiar prefijo",
      revokeToken: "Revocar token",
      revokeTitle: '¿Revocar el token de acceso "{name}"?',
      revokeDescription:
        "Las apps que usan este token pierden el acceso de inmediato. Esto no se puede deshacer.",
      revokeAction: 'Revocar "{name}"',
      revoking: "Revocando...",
      usageNoModel:
        "Carga o descarga un modelo para ver ejemplos ejecutables. Este servidor todavía no tiene ningún modelo que indicar.",
    },
    about: {
      title: "Acerca de",
      description:
        "Documentación, notas de la versión, comentarios e información de compilación.",
      studioVersion: "Versión de Unsloth",
      packageVersion: "Versión del paquete",
      desktopAppVersion: "Versión de la app de escritorio",
      desktopAppVersionUnavailable: "No disponible",
      llamaCppVersion: "Versión de llama.cpp",
      hardware: "Hardware",
      gpu: "GPU",
      cuda: "CUDA",
      rocm: "ROCm",
      xpu: "XPU",
      updates: "Actualización",
      help: "Ayuda",
      documentation: "Documentación",
      releaseNotes: "Notas de la versión",
      whatsNew: "Novedades",
      feedback: "Comentarios",
      reportIssue: "Reportar un problema",
      license: {
        sectionTitle: "Licencia",
        studioLabel: "Unsloth",
        studioLicense: "AGPL-3.0",
        studioDescription: "Código abierto bajo la GNU AGPL v3.0.",
        libraryLabel: "Unsloth Core",
        libraryLicense: "Apache-2.0",
        libraryDescription: "Con licencia Apache 2.0.",
      },
      dangerZone: "Zona de peligro",
      shutDownStudio: "Apagar Unsloth",
      shutDownStudioDescription:
        "Detiene el servidor de Unsloth y finaliza tu sesión.",
      shutDown: "Apagar",
      update: {
        title: "Actualizar Unsloth",
        commandText: "Texto de {label}",
        copied: "Copiado",
        copyCommand: "Copiar comando",
        commandCopied: "{label} copiado",
        copyNamedCommand: "Copiar {label}",
        checkingInstall: "Comprobando cómo se instaló Unsloth...",
        installIntro: "Para instalar o actualizar Unsloth:",
        localUpdateHeading: "Actualización local",
        installCommandUnix: "Comando de instalación para macOS/Linux",
        installCommandWindows: "Comando de instalación para Windows",
        localInstallDetected:
          "Instalación local detectada. Actualiza desde tu checkout original para evitar reemplazarlo con PyPI.",
        pullThenUpdate:
          "Descarga los últimos cambios y luego ejecuta el instalador local:",
        gitPullCommand: "comando git pull",
        localInstallerCommand: "comando del instalador local",
        sourceInstallDetected:
          "Instalación de paquete desde código fuente o VCS detectada. Reinstala desde la ruta local original o la URL de Git.",
        repoCheckoutFallback:
          "Si aún tienes el checkout del repositorio, ejecuta el instalador local desde él:",
        restartAfterUpdate: "Reinicia Unsloth después de actualizar.",
        desktopManaged:
          "La app de escritorio busca nuevas versiones automáticamente. También puedes buscar o instalar una actualización aquí en cualquier momento.",
        desktopReady: "Actualizaciones de la app de escritorio",
        desktopReadyDescription:
          "Comprueba si hay disponible una versión más reciente de la app de escritorio.",
        desktopChecking: "Buscando actualizaciones",
        desktopCheckingDescription: "Esto suele tardar unos segundos.",
        desktopAvailable:
          "La versión {version} de la app de escritorio está disponible",
        desktopAvailableDescription:
          "Actualiza ahora y la app de escritorio se reiniciará cuando termine.",
        desktopExternalServer:
          "Ejecuta `unsloth studio update` desde el terminal que inició el servidor.",
        desktopManualInstall:
          "Abre la página de versiones para instalar el paquete más reciente para Linux.",
        desktopCheckFailed: "No se pudo buscar actualizaciones",
        desktopCheckFailedDescription:
          "Comprueba la conexión e inténtalo de nuevo.",
        desktopCurrent: "La app de escritorio está actualizada",
        desktopCurrentDescription:
          "Unsloth seguirá buscando actualizaciones automáticamente.",
        checkForUpdates: "Buscar actualizaciones",
        checkAgain: "Buscar de nuevo",
        retryCheck: "Intentarlo de nuevo",
        checking: "Buscando...",
        updateNow: "Actualizar ahora",
        openReleasePage: "Abrir la página de versiones",
        unknownInstall:
          "No se pudo detectar cómo se instaló Unsloth. Para instalaciones con el instalador o desde PyPI, usa los comandos anteriores.",
        localCheckout:
          "Para instalaciones desde un checkout local, ejecuta el instalador local desde ese checkout:",
        docs: "Documentación de instalación:",
        docsInstall: "Instalación",
        docsUpdating: "Actualización",
        docsMac: "Mac",
        docsWindows: "Windows",
      },
    },
  },
  studio: {
    imageTraining: "Entrenamiento de imágenes",
    goToImageTraining: "Ir al entrenamiento de imágenes",
    routeTitle: "Entrenar",
    wizard: {
      modelTitle: "Modelo",
      modelDescription: "Selecciona el modelo y el método de entrenamiento",
      datasetTitle: "Conjunto de datos",
      datasetDescription: "Selecciona o sube datos de entrenamiento",
      paramsTitle: "Parámetros",
      paramsDescription: "Configura los parámetros de entrenamiento",
      configTitle: "Configuración",
      configDescription: "Guarda y carga configuraciones",
      modelLabel: "Modelo",
      methodLabel: "Método",
      datasetLabel: "Conjunto de datos",
      modelTooltip: "El modelo base que quieres ajustar.",
      methodTooltip: "Cómo se entrena el modelo. LoRA y QLoRA actualizan adaptadores pequeños en lugar de todos los pesos.",
      datasetTooltip: "Los datos de entrenamiento usados para ajustar el modelo.",
      hfTokenDescription:
        "Necesario para modelos y conjuntos de datos restringidos o privados.",
      uploadLocalLabel: "O sube un archivo local",
      sourceBrowse: "Explorar",
      releaseToUpload: "Suelta para subir",
      loadYaml: "Cargar YAML",
      saveYaml: "Guardar YAML",
      resetDefaults: "Restablecer valores predeterminados",
      cachedModelGoneTitle: "El modelo en caché ya no está disponible",
      cachedModelGoneDescription:
        "Los archivos del modelo ya no están en este dispositivo, por lo que el entrenamiento volverá a descargarlos.",
      cachedDatasetGoneTitle:
        "El conjunto de datos en caché ya no está disponible",
      cachedDatasetGoneDescription:
        "Los archivos del conjunto de datos ya no están en este dispositivo, por lo que el entrenamiento volverá a descargarlos.",
    },
    preview: {
      title: "Vista previa de la ejecución",
      ready: "Listo",
      notReady: "No está listo",
      modelPending: "Modelo pendiente",
      datasetPending: "Conjunto de datos pendiente",
      method: "Método",
      length: "Duración",
      stepZero: "{count} pasos",
      step: "{count} paso",
      stepTwo: "{count} pasos",
      stepFew: "{count} pasos",
      stepMany: "{count} pasos",
      steps: "{count} pasos",
      epochZero: "{count} épocas",
      epoch: "{count} época",
      epochTwo: "{count} épocas",
      epochFew: "{count} épocas",
      epochMany: "{count} épocas",
      epochs: "{count} épocas",
      batch: "Lote",
      context: "Contexto",
      lr: "LR",
      hardware: "Hardware",
      noGpu: "No se detectó ninguna GPU",
      hfToken: "Token de HF",
      saved: "Guardado",
      notSet: "Sin configurar",
      files: "Archivos",
      model: "Modelo",
      dataset: "Conjunto de datos",
      downloadsOnStart: "Se descarga al iniciar",
      continuesOnStart: "Continúa al iniciar",
      noticeModelDownload:
        "Este modelo aún no está en el dispositivo. Se descargará automáticamente al iniciar el entrenamiento.",
      noticeModelPartial:
        "El entrenamiento completará la descarga parcial del modelo antes de cargarlo.",
      noticeDatasetDownload:
        "Este conjunto de datos aún no está en el dispositivo. Se descargará automáticamente al iniciar el entrenamiento.",
      noticeDatasetPartial:
        "El entrenamiento completará la descarga parcial del conjunto de datos antes de leerlo.",
      noticeTransformersUpgrade:
        "Ninguna versión instalada de transformers admite todavía esta arquitectura. Al iniciar la ejecución se ofrecerá instalar transformers {version} primero.",
      noticeSixteenBitOnly:
        "Esta arquitectura se entrena en LoRA de 16 bits: 4 bits no está disponible, así que la ejecución necesita mucha más VRAM que QLoRA.",
      noticeInstallSwitchesSixteenBit:
        "Instalar esa versión en lugar de conservar el código propio del modelo cambia esta ejecución a LoRA de 16 bits, que necesita mucha más VRAM que QLoRA.",
      advancedSettings: "Ajustes avanzados",
      defaultAdvancedSettings: "Valores predeterminados",
      nonDefaultAdvancedSettings: "{count} no predeterminados",
    },
    datasetPicker: {
      noun: "conjuntos de datos",
      selectDataset: "Seleccionar conjunto de datos",
      hubPlaceholder: "Buscar conjuntos de datos en Hugging Face...",
      devicePlaceholder: "Buscar conjuntos de datos locales...",
      useAsHubDataset: "Usar como conjunto de datos de Hugging Face",
      hfCacheLabel: "Caché de HF",
      scanningLocal: "Buscando conjuntos de datos en este dispositivo…",
      couldntScan: "No se pudieron buscar conjuntos de datos locales",
      someLocationsUnscanned:
        "No se pudieron explorar algunas ubicaciones de conjuntos de datos.",
      noLocalDatasets:
        "Todavía no hay nada en este dispositivo. Descarga un conjunto de datos del Hub, crea uno en Recetas o sube un archivo.",
      openDataRecipes: "Abrir recetas de datos",
      searchingHub: "Buscando en Hugging Face…",
      noDatasetsFound: "No se encontraron conjuntos de datos.",
      tokenRejectedTitle: "Hugging Face rechazó el token",
      tokenRejectedBody:
        "Actualiza el token en Configuración → General y vuelve a intentarlo.",
      hubUnreachable: "No se pudo acceder a Hugging Face",
      cantUseDataset: "No se puede usar el conjunto de datos",
      reasonInvalidHubId:
        "Introduce un ID válido de conjunto de datos de Hugging Face: repositorio o propietario/repositorio, usando solo letras, números, ., _ o - (máximo 96 caracteres por parte).",
      sourceRecipe: "Receta",
      sourceUpload: "Subida",
      sourceLocal: "Local",
    },
    modelPicker: {
      noun: "modelos",
      selectModel: "Seleccionar modelo",
      hubPlaceholder: "Busca o pega un ID de Hugging Face...",
      devicePlaceholder: "Busca modelos locales o pega una ruta de carpeta...",
      useAsHubModel: "Usar como modelo de Hugging Face",
      useAsLocalPath: "Usar como ruta local",
      hfCacheLabel: "Caché de HF",
      scanningLocal: "Buscando modelos locales…",
      couldntScan: "No se pudieron buscar modelos locales",
      someLocationsUnscanned:
        "No se pudieron explorar algunas ubicaciones locales.",
      noLocalModels: "No se encontraron modelos locales.",
      noLocalModelsHint:
        "Pega una ruta de carpeta arriba o cambia a Hugging Face.",
      searchingHub: "Buscando en Hugging Face…",
      noModelsFound: "No se encontraron modelos.",
      tokenRejectedTitle: "Hugging Face rechazó el token",
      tokenRejectedBody:
        "Actualiza el token en Configuración → General y vuelve a intentarlo.",
      hubUnreachable: "No se pudo acceder a Hugging Face",
      cantUseModel: "No se puede usar el modelo para entrenar",
      reasonTypeMismatch:
        "Este modelo no coincide con el tipo de entrenamiento seleccionado en el paso anterior.",
      reasonEmptyId: "Introduce un ID de modelo o una ruta de modelo local.",
      reasonInvalidHubId:
        "Introduce un ID válido de modelo de Hugging Face: repositorio o propietario/repositorio, usando solo letras, números, ., _ o - (máximo 96 caracteres por parte).",
      reasonGguf: "Los modelos GGUF no se pueden usar para entrenar.",
      reasonAdapter:
        "Las salidas de adaptadores no se pueden usar como modelos base de entrenamiento.",
      reasonNotTrainable:
        "Este modelo del dispositivo no se puede usar para entrenar.",
      reasonUnsupportedFormat:
        "Este formato de modelo no es compatible con el entrenamiento.",
      vramNeeds: "Necesita ~{est} GB de VRAM (GPU: {total} GiB)",
      vramTight: "~{est} GB de VRAM (ajustado para {total} GiB)",
      vramApprox: "~{est} GB de VRAM",
      sourceModelsFolder: "Carpeta de modelos",
      sourceHfCache: "Caché de HF",
      sourceLmStudio: "LM Studio",
      sourceOllama: "Ollama",
      sourceCustomFolder: "Carpeta personalizada",
      sourceLocalModel: "Modelo local",
      vramOomBadge: "OOM",
      vramTightBadge: "Ajustado",
    },
    methods: {
      qlora: {
        label: "QLoRA",
        hint: "Cuantización de 4 bits. Menor uso de VRAM y arranque más rápido.",
        note: "4 bits",
      },
      lora: {
        label: "LoRA",
        hint: "Adaptadores de 16 bits. Equilibrio entre calidad y memoria.",
        note: "16 bits",
      },
      full: {
        label: "Fine-tuning completo",
        hint: "Entrena todos los pesos. Máxima calidad y mayor uso de VRAM.",
        note: "fp16",
      },
      cpt: {
        label: "Preentrenamiento continuo",
        hint: "Preentrenamiento continuo para nuevos dominios o idiomas.",
        note: "continuo",
      },
    },
    subtitles: {
      configure: "Configura e inicia el entrenamiento",
      trainingInProgress: "Entrenamiento en curso",
      viewPastRuns: "Ver ejecuciones de entrenamiento anteriores",
      viewingPastRun: "Viendo una ejecución anterior",
    },
    tabs: {
      configure: "Configurar",
      currentRun: "Ejecución actual",
      history: "Historial",
    },
    loadingRuntime: "Cargando entorno de entrenamiento...",
    checkingSupport: "Comprobando si este equipo admite entrenamiento...",
    backToHistory: "Volver al historial",
    dataset: {
      selectors: {
        subset: "Subconjunto",
        subsetTooltip:
          "Selecciona el subconjunto (configuración) del conjunto de datos que se utilizará.",
        trainSplit: "División de entrenamiento",
        trainSplitTooltip:
          "Selecciona la división que se utilizará para el entrenamiento.",
        evaluationSplit: "División de evaluación",
        evaluationSplitTooltip:
          "Selecciona la división que se utilizará para la evaluación. Ninguna significa que no se evaluará durante el entrenamiento.",
        selectSubset: "Selecciona un subconjunto...",
        selectSplit: "Selecciona una división...",
        none: "Ninguna",
        loading:
          "Cargando configuraciones y divisiones del conjunto de datos...",
        manualTitle: "Introduce manualmente las opciones del conjunto de datos",
        manualDescription:
          "Introduce los nombres exactos de la configuración y las divisiones de Hugging Face que se utilizarán.",
        manualSubsetPlaceholder: "Nombre de configuración opcional",
        manualRequired: "Se requiere una división de entrenamiento.",
        manualTooLong: "Usa 128 caracteres o menos.",
        manualInvalid: "Este valor contiene caracteres no compatibles.",
      },
      sourceAriaLabel: "Origen del conjunto de datos",
      localDataset: "Conjunto de datos local",
      localDatasetRows: " / {count} filas",
      huggingFaceDataset: "Conjunto de datos de Hugging Face",
      localDatasetMetadata: "Metadatos del conjunto de datos local",
      dataRecipeOutput: "Salida de Data Recipe.",
      rows: "Filas",
      columns: "Columnas",
      batches: "Lotes",
      updated: "Actualizado",
      evalDataset: "Conjunto de datos de evaluación",
      uploading: "Subiendo...",
      uploadEvalFile: "Subir archivo de evaluación",
      fileTooLarge: "El archivo es demasiado grande",
      fileTooLargeDescription:
        "{file} ocupa {size}. Las cargas de entrenamiento admiten hasta {limit}.",
      documentRedirect: {
        title: "Este archivo debe convertirse primero",
        genericFile: "Este archivo",
        description:
          "{file} es material de origen, no un conjunto de datos listo para entrenar. Usa Data Recipes para convertir el documento en un conjunto de datos y vuelve aquí para ajustar el modelo.",
        nextStepTitle: "Mejor siguiente paso",
        nextStepDescription:
          "Abre Learning Recipes y empieza con una receta basada en documentos, como PDF grounded QA.",
        openAction: "Abrir Learning Recipes",
      },
      evalDatasetDescription:
        "Opcional. Si no se proporciona, se separará una pequeña porción de los datos de entrenamiento.",
      advanced: "Avanzado",
      targetFormat: "Formato de destino",
      targetFormatTooltip:
        "Formato de tus datos de entrenamiento. La detección automática funciona para la mayoría de los conjuntos de datos.",
      streamingInfoAriaLabel:
        "Información sobre el streaming del conjunto de datos",
      streaming: {
        label: "Activar streaming",
        description:
          "Transmite conjuntos de datos de texto de Hugging Face en lugar de descargarlos.",
        unavailable: "Streaming no disponible. Para activarlo:",
        completionsUnavailable:
          "No disponible mientras el streaming del conjunto de datos esté activado.",
        blockers: {
          source:
            "Usa un conjunto de datos de Hugging Face (no una subida local ni un origen de S3).",
          maxSteps:
            "Establece Pasos máximos > 0; los conjuntos de datos en streaming no tienen una longitud conocida.",
          trainOnCompletions:
            'Desactiva "Solo respuestas del asistente".',
          evalSplit:
            "Elige una partición de evaluación aparte: la evaluación está activada, pero no hay una partición distinta configurada.",
          visionModel: "Los modelos de visión no admiten streaming.",
          audioModel: "Los modelos de audio no admiten streaming.",
          embeddingModel:
            "Los modelos de embeddings no admiten streaming (el entrenamiento necesita el conjunto de datos completo).",
          imageDataset:
            "Este conjunto de datos parece contener imágenes, que no se pueden transmitir.",
          audioDataset:
            "Este conjunto de datos parece contener audio, que no se puede transmitir.",
          appleSilicon:
            "El streaming todavía no es compatible con Apple Silicon (MLX).",
        },
        options: {
          trainOnCompletions: "solo respuestas del asistente",
          evaluation:
            "evaluación (necesita una partición de evaluación aparte)",
        },
        notifications: {
          turnedOffMaxSteps:
            "Streaming desactivado: se necesita un valor fijo de Pasos máximos > 0.",
          adjusted:
            "Ajustado para streaming. Opciones incompatibles desactivadas: {options}.",
          needsMaxSteps:
            "El streaming necesita un valor fijo de Pasos máximos (los conjuntos de datos en streaming no tienen una longitud conocida). Establece primero Pasos máximos > 0.",
          enabledAdjusted:
            "Streaming activado. Opciones incompatibles desactivadas: {options}.",
          disabledForDetectedModality:
            "El streaming se desactivó porque los conjuntos de datos de imagen y audio deben descargarse por completo. Revisa la opción y vuelve a iniciar el entrenamiento.",
        },
      },
      auto: "Automático",
      rawText: "Texto sin procesar",
      trainSplitStart: "Inicio de la partición de entrenamiento",
      trainSplitStartTooltip:
        "Entrena solo con un subconjunto de tu partición de entrenamiento especificando un índice de fila inicial (inclusivo, base 0). Déjalo vacío para empezar desde la primera fila.",
      trainSplitEnd: "Fin de la partición de entrenamiento",
      trainSplitEndTooltip:
        "Último índice de fila a incluir de la partición de entrenamiento (inclusivo, base 0). Por ejemplo, pon Inicio en 0 y Fin en 99 para entrenar con las primeras 100 filas. Déjalo vacío para usar todas las filas restantes.",
      endPlaceholder: "Fin",
      clear: "Borrar",
      dropFileOrClick: "Suelta 1 archivo aquí o haz clic para subir",
      uploadDetails: "Detalles de la subida",
      uploadDetailsTooltip:
        "Hasta {limit} por archivo. Los archivos PDF, DOCX y TXT no son conjuntos de datos listos para entrenar, así que conviértelos primero en Recetas.",
      viewDataset: "Ver conjunto de datos",
      uploadFailed: "La subida falló",
      unknownError: "Error desconocido",
      unsupportedFileType: "Tipo de archivo no admitido",
      uploadOneFileType: "Sube un archivo {types}.",
      datasetUploaded: "Conjunto de datos subido",
      evalDatasetUploaded: "Conjunto de datos de evaluación subido",
      uploadOneFileAtATime: "Sube un archivo a la vez",
      uploadSingleFileDescription:
        "La subida del conjunto de datos de entrenamiento acepta un solo archivo.",
      previewLoadingHuggingFace:
        "Obteniendo la vista previa del conjunto de datos desde Hugging Face...",
      previewLoading: "Cargando vista previa...",
      mappingRequirements: {
        audioAndText: "audio y texto",
        imageAndText: "imagen y texto",
        instructionAndOutput: "instrucción y salida",
        humanAndGpt: "humano y GPT",
        userAndAssistant: "usuario y asistente",
      },
      mappingStatus: {
        heuristicTitle: "Asignación detectada mediante heurísticas",
        readyTitle: "Asignación lista",
        requiredTitle: "Asignar columnas del conjunto de datos",
        heuristicDescription:
          "Detectamos automáticamente la asignación de columnas mediante heurísticas. Revísala y ajústala con los menús desplegables de los encabezados, o usa la asistencia de IA para obtener una asignación más precisa.",
        readyDescription:
          "Todo está listo. Convertiremos este conjunto de datos automáticamente.",
        requiredDescription:
          "Asigna roles a las columnas con los menús desplegables de los encabezados. Como mínimo, asigna {required}.",
      },
      s3: {
        title: "Configuración de S3",
        description:
          "Carga conjuntos de datos .parquet, .json, .jsonl o .csv desde Amazon S3",
        bucket: "Nombre del bucket",
        bucketPlaceholder: "mi-bucket-de-datos-de-entrenamiento",
        region: "Región de AWS",
        regionPlaceholder: "us-east-1",
        prefix: "Prefijo de ruta",
        prefixPlaceholder: "datasets/whisper/",
        accessKeyId: "ID de clave de acceso",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Clave de acceso secreta",
        secretAccessKeyPlaceholder: "Tu clave de acceso secreta de AWS",
        useIamRole: "Usar rol de IAM",
      },
    },
    params: {
      mode: {
        simple: "Simple",
        advanced: "Avanzado",
        ariaLabel: "Modo de parámetros",
      },
      projectName: "Nombre del proyecto",
      optional: "Opcional",
      projectNameDescription:
        "Se usa en los nombres de las carpetas de salida del entrenamiento, los valores predeterminados de exportación y el historial.",
      loraSettings: "Configuración de LoRA",
      trainingHyperparameters: "Hiperparámetros de entrenamiento",
      maxSteps: "Pasos máximos",
      epochs: "Épocas",
      useMaxSteps: "Usar pasos máximos",
      useEpochs: "Usar épocas",
      maxStepsTooltip: "Anula el total de pasos del optimizador.",
      epochsTooltip: "Número de pasadas completas sobre el conjunto de datos.",
      contextLength: "Longitud de contexto",
      contextLengthTooltip:
        "Número máximo de tokens por muestra de entrenamiento.",
      customContextLength: "Introduce un valor personalizado",
      learningRate: "Tasa de aprendizaje",
      learningRateTooltip:
        "Tamaño de paso para las actualizaciones de pesos. Valores más bajos entrenan más lento pero con mayor estabilidad.",
      learningRateDescription:
        "Recomendado: 2e-4 para LoRA, 5e-5 para CPT, 2e-5 para fine-tune completo",
      embeddingLearningRate: "Tasa de aprendizaje de embeddings",
      embeddingLearningRateTooltip:
        "Solo se usa cuando CPT entrena embed_tokens. Los embeddings son más fáciles de desestabilizar que los pesos de LoRA, por lo que suelen necesitar una LR más pequeña. Déjalo en blanco para usar lr/10; el rango de trabajo típico es de 2x a 10x más pequeño que la LR principal. Auméntalo solo si la adaptación de vocabulario o de tokens de dominio es demasiado lenta.",
      rank: "Rango",
      rankTooltip:
        "Dimensión de las matrices de bajo rango. Mayor = más capacidad.",
      alpha: "Alpha",
      alphaTooltip:
        "Factor de escalado para las actualizaciones de LoRA. Normalmente 2x el rango.",
      dropout: "Dropout",
      dropoutTooltip:
        "Probabilidad de dropout para las capas de LoRA para reducir el sobreajuste.",
      visionLayers: "Capas de visión",
      languageLayers: "Capas de lenguaje",
      attentionModules: "Módulos de atención",
      mlpModules: "Módulos MLP",
      targetModules: "Módulos objetivo",
      enableLora: "Habilitar LoRA",
      trainWithLora: "Entrenar con LoRA",
      stableRank: "Rango estable",
      memoryEfficient: "Eficiente en memoria",
      weightDecomposed: "Pesos descompuestos",
      notSupportedAppleSilicon: "No compatible con Apple Silicon",
      optimization: "Optimización",
      schedule: "Programación",
      memory: "Memoria",
      optimizer: "Optimizador",
      optimizerTooltip:
        "Algoritmo de optimización. Las variantes de 8 bits reducen el uso de memoria. Se recomienda Fused para modelos de visión.",
      optimizerTooltipMlx:
        "Algoritmo de optimización. AdamW es la opción predeterminada. Lion usa menos memoria, pero normalmente requiere una tasa de aprendizaje menor.",
      lrScheduler: "Planificador de LR",
      lrSchedulerTooltip:
        "Cómo cambia la tasa de aprendizaje durante el entrenamiento. El planificador lineal disminuye de forma constante; el de coseno disminuye siguiendo una curva.",
      optimizerOptions: {
        adamw8bit: "AdamW 8 bits",
        pagedAdamw8bit: "Paged AdamW 8 bits",
        adamwBnb8bit: "AdamW BNB 8 bits",
        pagedAdamw32bit: "Paged AdamW 32 bits",
        adamwTorch: "AdamW (PyTorch)",
        adamwTorchFused: "AdamW (PyTorch Fused)",
      },
      lrSchedulerOptions: {
        linear: "Lineal",
        cosine: "Coseno",
      },
      batchSize: "Tamaño de batch",
      batchSizeTooltip:
        "Muestras procesadas por paso. Un valor más alto usa más VRAM.",
      gradAccum: "Acum. de gradiente",
      gradAccumTooltip:
        "Simula tamaños de batch más grandes sin VRAM adicional.",
      weightDecay: "Decaimiento de pesos",
      weightDecayTooltip: "Regularización L2 para prevenir el sobreajuste.",
      warmupSteps: "Pasos de calentamiento",
      warmupStepsTooltip:
        "Aumenta gradualmente la LR al inicio del entrenamiento para dar estabilidad.",
      scheduleEpochsTooltip:
        "Número de pasadas completas sobre el conjunto de datos. Pon 0 para ejecutar por pasos máximos.",
      saveSteps: "Pasos entre guardados",
      saveStepsTooltip: "Guarda un checkpoint cada N pasos. 0 para desactivar.",
      evalSteps: "Pasos entre evaluaciones",
      evalStepsTooltip:
        "Fracción del total de pasos de entrenamiento entre evaluaciones (0-1). Pon 0 para desactivar la evaluación. P. ej. 0.01 = evaluar cada 1 % de los pasos.",
      seed: "Semilla",
      seedTooltip: "Semilla aleatoria para reproducibilidad.",
      gradCheckpoint: "Checkpoint de gradiente",
      gradCheckpointTooltip:
        "Intercambia cómputo por memoria recalculando las activaciones.",
      none: "Ninguno",
      standard: "Estándar",
      enablePacking: "Habilitar packing",
      assistantCompletionsOnly: "Solo completaciones del asistente",
      readMore: "Leer más",
    },
    training: {
      startTraining: "Iniciar entrenamiento",
      starting: "Iniciando...",
      loadingModel: "Cargando modelo...",
      checkingDataset: "Comprobando conjunto de datos...",
      chooseModel: "Elige un modelo",
      chooseDataset: "Elige un conjunto de datos",
      chooseModelAndDataset: "Elige un modelo y un conjunto de datos",
      modelUnverified:
        "No se pudo verificar la configuración de este modelo. Comprueba la conexión o el token de Hugging Face y vuelve a intentarlo.",
      legacyDatasetScriptUnsupported:
        "Este conjunto de datos de Hub depende de un script personalizado antiguo y no es compatible con este flujo de entrenamiento.",
      hfModelAccessDenied:
        "Hugging Face denegó el acceso a este modelo. Añade un token de Hugging Face válido con acceso al repositorio, acepta las condiciones de acceso necesarias y vuelve a intentarlo.",
      hfModelVerificationRateLimited:
        "La verificación del modelo de Hugging Face está limitada temporalmente. Vuelve a intentarlo en breve.",
      hfModelVerificationFailed:
        "No se pudo verificar el modelo de Hugging Face. Comprueba el ID del repositorio y tu token de acceso.",
      hfModelMetadataUnavailable:
        "Los metadatos del modelo de Hugging Face no están disponibles temporalmente. Vuelve a intentarlo antes de iniciar el entrenamiento.",
      datasetUnverified:
        "No se pudo verificar si el conjunto de datos es compatible con este modelo. Comprueba la conexión o el token de Hugging Face; al iniciar el entrenamiento se volverá a intentar la comprobación.",
      setupChanged:
        "La configuración del entrenamiento cambió mientras se comprobaba. Revísala y vuelve a iniciar el entrenamiento.",
      validation: {
        s3MultimodalUnsupported:
          "Los conjuntos de datos de S3 todavía no son compatibles con el entrenamiento de visión o audio.",
        s3BucketRequired: "Introduce primero el nombre de un bucket de S3.",
        s3CredentialsRequired:
          "Proporciona las claves de acceso de S3 o activa el rol de IAM.",
        modelRequired: "Selecciona primero un modelo base.",
        learningRatePositive: "Introduce una tasa de aprendizaje mayor que cero.",
        embeddingLearningRateRange:
          "Introduce una tasa de aprendizaje de embeddings mayor que 0 y menor que 1.",
        hfDatasetRequired:
          "Selecciona primero un conjunto de datos de Hugging Face.",
        hfDatasetSplitRequired:
          "Selecciona o introduce primero una división de entrenamiento.",
        localDatasetRequired: "Selecciona primero un conjunto de datos local.",
        unsupportedDatasetSource: "Origen de conjunto de datos no compatible.",
      },
      startFailed: "No se pudo iniciar el entrenamiento",
      startUnconfirmed:
        "Unsloth no pudo confirmar si el entrenamiento se inició. Se está comprobando el estado en segundo plano.",
      stopFailed: "No se pudo detener el entrenamiento",
      trainingStillActiveTitle: "El entrenamiento sigue activo",
      stopBeforeConfig:
        "Detén primero el entrenamiento y vuelve después a la configuración.",
      resumeFailed: "No se pudo reanudar el entrenamiento",
      resumeFailedTitle: "Error al reanudar el entrenamiento",
      resumeUnavailable:
        "Solo se pueden reanudar las ejecuciones detenidas o con errores que tengan un checkpoint guardado.",
      uploadConfigTooltip: "Carga una configuración YAML guardada",
      saveConfigTooltip: "Descarga la configuración actual como YAML",
      resetConfigTooltip: "Restablece a los valores predeterminados del modelo",
      configLoaded: "Configuración cargada",
      failedToLoadConfig: "No se pudo cargar la configuración",
      invalidYamlFile: "Archivo YAML no válido",
      configTooLarge:
        "El archivo de configuración de entrenamiento es demasiado grande (máximo 1 MiB).",
      failedToReadFile: "No se pudo leer el archivo",
      failedToSaveConfig: "No se pudo guardar la configuración",
      parametersReset:
        "Parámetros restablecidos a los valores predeterminados del modelo",
      audioIncompatible:
        "Este modelo no admite audio. Cambia a un modelo compatible con audio o elige un conjunto de datos sin audio.",
      visionIncompatible:
        "El modelo de texto no es compatible con un conjunto de datos multimodal. Cambia a un modelo de visión o elige un conjunto de datos solo de texto.",
      cancelTitle: "Cancelar entrenamiento",
      cancelDescription:
        "¿Quieres cancelar la ejecución de entrenamiento actual?",
      continueAction: "Continuar entrenamiento",
      cancelAction: "Cancelar entrenamiento",
      stopTitle: "Detener entrenamiento",
      stopDescription:
        "Elige cómo quieres detener la ejecución de entrenamiento actual. «Detener y guardar» crea un punto de control desde el que podrás reanudarla más adelante; si la detienes sin guardar, no podrás reanudarla.",
      stopAction: "Detener",
      stopping: "Deteniendo...",
      stopAndSave: "Detener y guardar",
      compareInChat: "Comparar en el chat",
      exportModel: "Exportar modelo",
      milestone: "Hito",
      halfwayDone: "A mitad de camino. El entrenamiento superó el 50 %.",
      doneNextStep:
        "Entrenamiento terminado. Siguiente paso: comparar las salidas base y ajustadas.",
    },
    history: {
      title: "Historial",
      filesDeleted: "Archivos eliminados",
      deleteArtifactsLabel:
        "Eliminar también los archivos del adaptador del disco",
      deleteArtifactsDescription:
        "Elimina la carpeta de salida de la ejecución, incluidos los adaptadores y checkpoints guardados.",
      deleteArtifactsSharedNote:
        "Otra ejecución comparte esta carpeta de salida. Los archivos se conservarán hasta que se elimine la última ejecución que los usa.",
      artifactsKeptShared:
        "Se eliminó la ejecución. Los archivos del adaptador se conservaron porque otra ejecución usa la misma carpeta.",
      deleteArtifactsActiveError:
        "La ejecución de entrenamiento en curso está usando estos archivos. Detén el entrenamiento antes de eliminarlos.",
      deleteArtifactsFailed:
        "Se eliminó la ejecución, pero no se pudieron borrar sus archivos.",
      deleteArtifactsRetainedError:
        "No se pudieron eliminar los archivos del adaptador, por lo que la ejecución se conservó en el historial.",
      emptyDescription:
        "Aún no hay ejecuciones de entrenamiento. Inicia tu primera ejecución en la pestaña Configurar.",
      loadError: "No se pudieron cargar las ejecuciones de entrenamiento",
      deleteError:
        "No se pudo eliminar la ejecución de entrenamiento. Inténtalo de nuevo.",
      retry: "Reintentar",
      loadMore: "Cargar más",
      loading: "Cargando...",
      loadingRun: "Cargando ejecución de entrenamiento...",
      runNotFound: "Ejecución no encontrada",
      deleteTitle: "¿Eliminar ejecución de entrenamiento?",
      deleteDescription:
        "Esto eliminará permanentemente esta ejecución de entrenamiento y todas sus métricas. Esta acción no se puede deshacer.",
      resumeTraining: "Reanudar entrenamiento",
      resuming: "Reanudando...",
      deleteRun: "Eliminar ejecución",
      loss: "Pérdida",
      steps: "Pasos",
      lossTrendSparkline: "Minigráfica de tendencia de pérdida",
      relativeJustNow: "justo ahora",
      status: {
        completed: "Completado",
        stopped: "Detenido",
        error: "Error",
        running: "En ejecución",
        continued: "Reanudado",
      },
      message: {
        completed: "Entrenamiento completado",
        stopped: "Entrenamiento detenido",
        running: "Entrenamiento en curso",
        errored: "El entrenamiento tuvo un error",
      },
      copyPreviewLink: "Copiar enlace de vista previa",
      previewLinkCopied: "Enlace de vista previa copiado",
      previewLinkCopyFailed: "No se pudo copiar el enlace",
    },
    charts: {
      settings: "Configuración de gráficas",
      settingsDescription:
        "Ajusta la presentación de las gráficas mientras el entrenamiento sigue en curso.",
      openSettings: "Abrir configuración de gráficas",
      viewWindow: "Ventana de visualización",
      viewWindowDescription:
        "Muestra solo los últimos pasos o todo el historial.",
      window: "Ventana",
      all: "Todo",
      trainingLoss: "Pérdida de entrenamiento",
      trainingLossDescription: "Controla las superposiciones y el suavizado EMA.",
      smoothing: "Suavizado",
      smoothingDescription:
        "Mueve a la derecha para más suavizado. `0` = sin procesar.",
      showRawLoss: "Mostrar pérdida sin procesar",
      showSmoothedLoss: "Mostrar pérdida suavizada",
      showAverageLine: "Mostrar línea de promedio",
      scaleAndCleanup: "Escala y limpieza",
      linear: "Lineal",
      log: "Logarítmica",
      noClip: "Sin recorte",
      clipP99: "Recortar p99",
      clipP95: "Recortar p95",
      lossAxis: "Eje de pérdida",
      gradientNormAxis: "Eje de norma del gradiente",
      learningRateAxis: "Eje de tasa de aprendizaje",
      resetDefaults: "Restablecer valores predeterminados",
      loss: "Pérdida",
      smoothed: "Suavizada",
      evalLoss: "Pérdida de evaluación",
      learningRate: "Tasa de aprendizaje",
      lr: "LR",
      gradNorm: "Norma de grad.",
      gradientNorm: "Norma del gradiente",
      step: "Paso {step}",
      averageValue: "prom. {value}",
      waitingForFirstEvaluationStep:
        "Esperando el primer paso de evaluación...",
      evaluationNotConfigured: "Evaluación no configurada",
      evalChartWillAppear:
        "La gráfica aparecerá cuando se alcance eval_steps",
      setEvalDatasetAndSteps:
        "Configura el conjunto de datos de evaluación y eval_steps para seguir la pérdida de evaluación",
    },
    progress: {
      title: "Progreso del entrenamiento",
      liveMetrics: "Métricas de entrenamiento en vivo",
      exportGguf: "Exportar a GGUF",
      openConfig: "Abrir configuración de entrenamiento",
      configLabel: "Configuración de entrenamiento",
      hyperparams: "Hiperparámetros",
      epochs: "Épocas",
      batchSize: "Tamaño de batch",
      learningRate: "Tasa de aprendizaje",
      optimizer: "Optimizador",
      maxSteps: "Pasos máximos",
      contextLength: "Longitud de contexto",
      warmupSteps: "Pasos de calentamiento",
      rank: "Rango",
      alpha: "Alpha",
      dropout: "Dropout",
      variant: "Variante",
      epoch: "Época {value}",
      percentComplete: "{percent} % completado",
      stepProgress: "Paso {current} / {total}",
      loss: "Pérdida",
      lr: "LR",
      gradNorm: "Norma de grad.",
      project: "Proyecto",
      model: "Modelo",
      method: "Método",
      elapsed: "Transcurrido: {value}",
      eta: "ETA: {value}",
      stepsPerSecond: "{value} pasos/s",
      noStepsPerSecond: "-- pasos/s",
      tokens: "Tokens: {value}",
      gpuMonitor: "Monitor de GPU",
      live: "En vivo",
      utilization: "Utilización",
      temperature: "Temperatura",
      vram: "VRAM",
      power: "Potencia",
      phase: {
        idle: "Inactivo",
        downloadingModel: "Descargando modelo",
        downloadingDataset: "Descargando conjunto de datos",
        loadingModel: "Cargando modelo",
        loadingDataset: "Cargando conjunto de datos",
        configuring: "Configurando",
        training: "Entrenando",
        finalizing: "Guardando modelo",
        completed: "Completado",
        error: "Error",
        stopped: "Detenido",
      },
    },
    trainingStart: {
      ready: "Listo",
      downloading: "Descargando",
      preparing: "Preparando",
      left: "{eta} restante",
      downloaded: "{size} descargado",
      terminalStart: "> el entrenamiento de unsloth comienza...",
      preparingResources: "> Preparando modelo y conjunto de datos...",
      gettingReady:
        "> Estamos dejando todo listo para tu ejecución...",
      waitingForFirstStep:
        "> {message} | esperando el primer paso... ({step})",
      resumingTraining: "Reanudando entrenamiento...",
      startingTraining: "iniciando entrenamiento...",
      dataset: "Conjunto de datos",
      datasetStreaming:
        "Conjunto de datos: streaming (sin descarga completa)",
      modelWeights: "Pesos del modelo",
    },
  },
  modelMemory: {
    readout:
      "Pesos {model} + contexto {context} = {total} de {budget} de VRAM utilizable",
    readoutWithSpec:
      "Pesos {model} + KV {kv} + borrador MTP {spec} = {total} de {budget} de VRAM utilizable",
    kvRate: "KV reservado, ~{rate}/token",
    oomLikely: "Con la configuración actual es probable un error de memoria",
    tooLarge: "Más grande que la VRAM, se descargará a la CPU. Una cuantización más pequeña es más rápida",
  },
} satisfies DeepPartialMessageTree<typeof en>;
