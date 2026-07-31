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
    product: "Unsloth Studio",
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
      train: "Entrenar",
      recipes: "Recetas",
      export: "Exportar",
      recents: "Recientes",
      noChatsYet: "Aún no hay chats",
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
    },
    tabs: {
      general: "General",
      profile: "Perfil",
      appearance: "Apariencia",
      resources: "Sistema",
      chat: "Chat",
      connections: "Conexiones",
      apiKeys: "API",
      about: "Acerca de",
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
      tokenValidated: "Token validado",
      password: "Contraseña",
      passwordDescription:
        "Cambia la contraseña de esta cuenta de Unsloth.",
      passwordDialog: {
        trigger: "Cambiar contraseña",
        title: "Cambiar contraseña",
        description:
          "Introduce tu contraseña actual y elige una nueva (al menos {minLength} caracteres).",
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
          "Cuando una solicitud compatible con OpenAI nombra un GGUF descargado distinto, se carga antes de responder. Desactivado por defecto; los nombres desconocidos siguen usando el modelo cargado.",
        idleUnload: "Descarga automática por inactividad",
        idleUnloadDescription:
          "Descarga el modelo tras este número de segundos inactivo para liberar VRAM; la siguiente solicitud lo recarga. 0 lo mantiene cargado. Mínimo 60 segundos.",
        idleNeedsEnable:
          "Activa Cambiar de modelo según la solicitud para que un modelo descargado se recargue en el próximo uso.",
        idleActiveViaEnv:
          "La descarga automática por inactividad está activa mediante la variable de entorno UNSLOTH_MODEL_IDLE_TTL.",
        loadError:
          "No se pudo cargar la configuración de cambio automático de modelo.",
        saveError:
          "No se pudo guardar la configuración de cambio automático de modelo.",
        idleError: "Introduce 0 para mantener el modelo cargado, o al menos 60 segundos.",
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
          "Rota el secreto de firma para que todos los enlaces que hayas compartido dejen de funcionar. Los enlaces recién copiados siguen funcionando.",
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
      },
      gettingStarted: "Primeros pasos",
      startOnboarding: "Iniciar la introducción",
      startOnboardingDescription:
        "Reabre el asistente de configuración sin cambiar tu cuenta.",
      startOnboardingAction: "Iniciar la introducción",
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
        reindexWarning:
          "Solo afecta a los documentos recién indexados. Vuelve a subir los existentes tras cambiar el modelo.",
        emptyError:
          "Introduce un id de modelo de Hugging Face o una ruta local.",
        loadError:
          "No se pudo cargar la configuración del modelo de embeddings.",
        saveError: "No se pudo guardar el modelo de embeddings.",
        saved: "Modelo de embeddings guardado.",
        saveAnyway: "Guardar de todos modos",
        resetAction: "Restablecer al valor predeterminado",
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
    },
    appearance: {
      title: "Apariencia",
      description: "Cómo se ve Unsloth Studio en este dispositivo.",
      theme: {
        title: "Tema",
        label: "Esquema de color",
        description: "Claro, oscuro o según tu sistema.",
        system: "Sistema",
        light: "Claro",
        dark: "Oscuro",
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
        cpu: "CPU",
        ram: "RAM",
        disk: "Disco",
        vram: "VRAM",
        cpuCores: "{logical} núcleos lógicos / {physical} físicos",
        currentLoad: "Carga actual",
        free: "{value} libre",
        noGpu: "No hay GPU visible",
      },
      gpu: {
        title: "Dispositivos GPU",
        noGpu:
          "No se detectó ninguna GPU visible. Arriba se muestran los recursos solo de CPU.",
        unknownDevice: "GPU desconocida",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "{value} en uso",
        free: "{value} libre",
        total: "{value} en total",
      },
      storage: {
        title: "Almacenamiento",
        systemDisk: "Disco del sistema",
        diskUsage: "{used} en uso / {total}",
        diskFree: "{free} libre",
        modelsFolder: "Carpeta de modelos",
        modelsFolderKeywords:
          "modelos carpeta directorio ruta ubicacion ubicación descargas descarga cache caché almacenamiento disco unidad mover cambiar models folder path hugging face",
        modelsFolderDescription:
          "Dónde se almacenan los modelos descargados.",
        openAction: "Abrir",
        copyAction: "Copiar ruta",
        copied: "Ruta copiada",
        openError: "No se pudo abrir la carpeta",
        copyError: "No se pudo copiar la ruta",
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
      },
    },
    chat: {
      title: "Chat",
      description: "Gestiona el historial de chat almacenado en este dispositivo.",
      modelDisclaimer: "Mostrar aviso del modelo",
      modelDisclaimerDescription:
        'Muestra "Los LLM pueden cometer errores" bajo el cuadro de chat.',
      artifacts: {
        title: "Canvas",
        collapseHtmlBlocks: "Contraer bloques HTML",
        collapseHtmlBlocksDescription:
          "El modo Canvas contrae el HTML completo automáticamente. Activa esto para contraer también los documentos HTML entre delimitadores cuando Canvas está desactivado.",
        allowNetworkAccess: "Permitir acceso de red en Canvas",
        allowNetworkAccessDescription:
          "Permite que las vistas previas de Canvas carguen scripts, estilos, fuentes, medios y recursos de red desde CDNs. Mantenlo desactivado para vistas previas totalmente sin conexión.",
      },
      data: "Datos",
      exportHistory: "Exportar historial de chat",
      exportHistoryDescription:
        "Descarga todos los chats y mensajes como JSON.",
      exportAction: "Exportar",
      exportingAction: "Exportando...",
      exportConversations: "Exportar Recientes y Proyectos",
      exportConversationsDescription:
        "Descarga Recientes, o Recientes más los chats de proyectos, como JSONL sin procesar, CSV o JSONL de ShareGPT, combinados o por chat.",
      exportConversationsAction: "Exportar",
      exportScopeRecents: "Recientes",
      exportScopeAll: "Recientes + Proyectos",
      exportCombinedSuffix: "(combinado)",
      exportPerChatSuffix: "(por chat)",
      importChats: "Importar chats",
      importChatsDescription:
        "Importa una exportación JSONL, NDJSON o CSV a Recientes.",
      importChatsAction: "Importar",
      importNoConversations: "No se encontraron conversaciones en el archivo.",
      importedOneChat: "Se importó 1 conversación a Recientes.",
      importedChatCount: "Se importaron {count} conversaciones a Recientes.",
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
    connections: {
      title: "Conexiones",
      description: "Gestiona proveedores y conexiones externas.",
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
        "El puerto 0.0.0.0 sigue siendo accesible globalmente. Para máxima seguridad, inicia Unsloth Studio con --secure para exponer solo este enlace HTTPS.",
      copyTunnelUrl: "Copiar URL del túnel",
      copySnippet: "Copiar fragmento",
      copy: "Copiar",
      copied: "Copiado",
      setupDocs: "Documentación de configuración:",
      codingAgents: "Agentes de programación",
      codingAgentsHint:
        "Inicia un agente de programación contra este servidor. Usa el modelo cargado; un servidor local genera una clave de API automáticamente y uno remoto la incluye en el comando.",
      codingAgentsSwap:
        "Reemplaza claude por codex, openclaw, opencode o hermes.",
      codingAgentDetected: "Instalado en esta máquina",
      codingAgentsDetectedHint: "Detectados en esta máquina: {agents}.",
      relativeNever: "nunca",
      relativeJustNow: "justo ahora",
      relativeHoursAgo: "hace {count} h",
      relativeDaysAgo: "hace {count} d",
      relativeMonthsAgo: "hace {count} mes(es)",
      relativeYearsAgo: "hace {count} año(s)",
      expired: "caducado",
      today: "hoy",
      inDays: "en {count} d",
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
    },
    about: {
      title: "Acerca de",
      description:
        "Documentación, notas de la versión, comentarios e información de compilación.",
      studioVersion: "Versión de Unsloth",
      packageVersion: "Versión del paquete",
      llamaCppVersion: "Versión de llama.cpp",
      hardware: "Hardware",
      gpu: "GPU",
      cuda: "CUDA",
      rocm: "ROCm",
      updates: "Actualizar",
      help: "Ayuda",
      documentation: "Documentación",
      releaseNotes: "Notas de la versión",
      whatsNew: "Novedades",
      feedback: "Comentarios",
      reportIssue: "Reportar un problema",
      license: {
        sectionTitle: "Licencia",
        studioLabel: "Unsloth Studio",
        studioLicense: "AGPL-3.0",
        studioDescription: "Código abierto bajo la GNU AGPL v3.0.",
        libraryLabel: "Unsloth Core",
        libraryLicense: "Apache-2.0",
        libraryDescription: "Con licencia Apache 2.0.",
      },
      dangerZone: "Zona de peligro",
      shutDownStudio: "Apagar Unsloth Studio",
      shutDownStudioDescription:
        "Detiene el servidor de Unsloth y finaliza tu sesión.",
      shutDown: "Apagar",
      update: {
        title: "Actualizar Unsloth Studio",
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
          "La app de escritorio mantiene actualizado su backend integrado y avisará cuando haya una nueva versión disponible.",
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
      vramNeeds: "Necesita ~{est} GB de VRAM (GPU: {total} GB)",
      vramTight: "~{est} GB de VRAM (ajustado para {total} GB)",
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
    backToHistory: "Volver al historial",
    sections: {
      model: "Modelo",
      dataset: "Conjunto de datos",
      params: "Parámetros",
      training: "Entrenamiento",
      charts: "Gráficas",
      progress: "Progreso del entrenamiento",
    },
    configure: {
      title: "Configurar",
      description:
        "Elige un modelo, un conjunto de datos y la configuración de entrenamiento.",
      startTraining: "Iniciar entrenamiento",
      starting: "Iniciando...",
      loadingModel: "Cargando modelo...",
      checkingDataset: "Comprobando conjunto de datos...",
      trainingConfig: "Configuración de entrenamiento",
    },
    dataset: {
      source: "Origen del conjunto de datos",
      sourceAriaLabel: "Origen del conjunto de datos",
      failedToLoadLocalDatasets:
        "No se pudieron cargar los conjuntos de datos locales.",
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
      uploadLimitsHint:
        "CSV, JSONL, JSON, Parquet · hasta {limit}; PDF/DOCX/TXT → Learning Recipes",
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
      checkingToken: "Comprobando token...",
      preview: "Vista previa del conjunto de datos",
      split: "Partición",
      subset: "Subconjunto",
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
        prefixTooltip:
          "Ruta opcional dentro del bucket hacia los archivos de tu conjunto de datos",
        accessKeyId: "ID de clave de acceso",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Clave de acceso secreta",
        secretAccessKeyPlaceholder: "Tu clave de acceso secreta de AWS",
        useIamRole: "Usar rol de IAM",
        useIamRoleTooltip:
          "Usa credenciales de rol de IAM en lugar de claves de acceso (recomendado para EC2/SageMaker)",
        testConnection: "Probar conexión",
        connectionSuccess: "Conexión al bucket de S3 exitosa",
        connectionFailed: "No se pudo conectar al bucket de S3",
        comingSoon: "Integración con S3 próximamente",
        comingSoonDescription:
          "La carga de conjuntos de datos desde S3 requiere boto3. Esta función está en desarrollo.",
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
      epochsDescription:
        "Cada época es una pasada completa sobre tu conjunto de datos.",
      maxStepsDescription:
        "Limita el entrenamiento a un número fijo de pasos del optimizador.",
      contextLength: "Longitud de contexto",
      contextLengthTooltip:
        "Número máximo de tokens por muestra de entrenamiento.",
      customContextLength: "Introduce un valor personalizado",
      contextLengthDescription:
        "Longitud máxima de secuencia para las muestras de entrenamiento",
      learningRate: "Tasa de aprendizaje",
      learningRateTooltip:
        "Tamaño de paso para las actualizaciones de pesos. Valores más bajos entrenan más lento pero con mayor estabilidad.",
      learningRateDescription:
        "Recomendado: 2e-4 para LoRA, 5e-5 para CPT, 2e-5 para fine-tune completo",
      embeddingLearningRate: "Tasa de aprendizaje de embeddings",
      embeddingLearningRateTooltip:
        "Solo se usa cuando CPT entrena embed_tokens. Los embeddings son más fáciles de desestabilizar que los pesos de LoRA, por lo que suelen necesitar una LR más pequeña. Déjalo en blanco para usar lr/10; el rango de trabajo típico es de 2x a 10x más pequeño que la LR principal. Auméntalo solo si la adaptación de vocabulario o de tokens de dominio es demasiado lenta.",
      embeddingLearningRateDescription:
        "Déjalo en blanco para usar lr/10 (recomendado). El rango típico es de 2x a 10x más pequeño que la tasa de aprendizaje principal.",
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
      optimization: "Optimización",
      schedule: "Programación",
      memory: "Memoria",
      optimizer: "Optimizador",
      optimizerTooltip:
        "Algoritmo de optimización. Las variantes de 8 bits reducen el uso de memoria. Se recomienda Fused para modelos de visión.",
      lrScheduler: "Planificador de LR",
      lrSchedulerTooltip:
        "Cómo cambia la tasa de aprendizaje durante el entrenamiento. Linear decae de forma constante; cosine decae en una curva.",
      optimizerOptions: {
        adamw8bit: "AdamW 8 bits",
        pagedAdamw8bit: "Paged AdamW 8 bits",
        adamwBnb8bit: "AdamW BNB 8 bits",
        pagedAdamw32bit: "Paged AdamW 32 bits",
        adamwTorch: "AdamW (PyTorch)",
        adamwTorchFused: "AdamW (PyTorch Fused)",
      },
      lrSchedulerOptions: {
        linear: "Linear",
        cosine: "Cosine",
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
        hfDatasetRequired:
          "Selecciona primero un conjunto de datos de Hugging Face.",
        localDatasetRequired: "Selecciona primero un conjunto de datos local.",
        unsupportedDatasetSource: "Origen de conjunto de datos no compatible.",
      },
      startFailed: "No se pudo iniciar el entrenamiento",
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
        "Elige cómo quieres detener la ejecución de entrenamiento actual.",
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
      emptyTitle: "Aún no hay ejecuciones de entrenamiento",
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
      runCount: "{count} ejecuciones",
      oneRun: "1 ejecución",
      resume: "Reanudar",
      resumeTraining: "Reanudar entrenamiento",
      resuming: "Reanudando...",
      deleteRun: "Eliminar ejecución",
      loss: "Pérdida",
      steps: "Pasos",
      lossTrendSparkline: "Minigráfica de tendencia de pérdida",
      relativeJustNow: "justo ahora",
      relativeMinutesAgo: "hace {count} min",
      relativeHoursAgo: "hace {count} h",
      relativeDaysAgo: "hace {count} d",
      status: {
        completed: "Completado",
        stopped: "Detenido",
        error: "Error",
        running: "En ejecución",
        continued: "Continuado",
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
    tour: {
      guidedTour: "Recorrido guiado",
    },
  },
} satisfies DeepPartialMessageTree<typeof en>;
