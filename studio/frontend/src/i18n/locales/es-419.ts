// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DeepPartialMessageTree } from "../types";
import type { en } from "./en.ts";

export const es419 = {
  chatBackground: {
    noBackground: "Sin fondo",
    sizing: "Tamaño de imagen",
    fill: "Rellenar",
    fit: "Ajustar",
    stretch: "Estirar",
    original: "Tamaño original",
    zoom: "Zoom",
    outline: "Contorno del campo de mensaje",
    fade: "Desvanecer hacia el fondo",
    title: "Imagen de fondo",
    description: "Una imagen local para tu espacio de trabajo. JPEG, PNG o WebP, hasta 20 MiB. El texto permanece legible automáticamente.",
    upload: "Subir imagen",
    replace: "Reemplazar imagen",
    remove: "Quitar fondo",
    reset: "Restablecer ajustes",
    effect: "Efecto",
    none: "Ninguno",
    dither: "Tramado",
    ascii: "ASCII",
    halftone: "Semitono",
    scanlines: "Líneas de barrido",
    visibility: "Visibilidad",
    blur: "Desenfoque de imagen",
    position: "Posición",
    positionValue: "Horizontal {x} %, vertical {y} %",
    newChat: "Nuevo chat",
    conversation: "Conversación",
    prompt: "¿Qué te gustaría explorar?",
    response: "Tu conversación sigue siendo legible sobre el fondo.",
    storageError: "No se pudo guardar el fondo. Comprueba el almacenamiento del navegador.",
    fileError: "Elige una imagen JPEG, PNG o WebP de menos de 20 MiB.",
    dimensionsError: "Elige una imagen de menos de 40 megapíxeles.",
    decodeError: "No se pudo leer esta imagen. Prueba otro archivo.",
    effectError: "No se pudo generar el efecto. Elige Ninguno u otra imagen.",
  },

  linkMenu: {
    openInBrowser: "Abrir en el navegador de Unsloth",
    openExternal: "Abrir en el navegador externo",
    copyLink: "Copiar enlace",
    saveLinkAs: "Guardar enlace como…",
    alwaysInBrowser: "Abrir siempre los enlaces en el navegador de Unsloth",
    linkCopied: "Enlace copiado",
    copyFailed: "No se pudo copiar.",
    saveFailed: "No se pudo guardar el archivo.",
    openFile: "Abrir archivo",
    openDefaultApp: "Abrir en la app predeterminada",
    openWith: "Abrir con",
    unslothBrowser: "Navegador de Unsloth",
    newChat: "Nuevo chat",
    browserTab: "Nueva pestaña del navegador",
    saveAs: "Guardar como…",
    copyPath: "Copiar ruta",
    copyContents: "Copiar contenido del archivo",
    revealFinder: "Mostrar en Finder",
    revealExplorer: "Mostrar en el Explorador",
    revealFolder: "Mostrar en la carpeta",
    pathCopied: "Ruta copiada",
    contentsCopied: "Contenido copiado",
    openFailed: "No se pudo abrir {name}.",
    revealFailed: "No se pudo mostrar {name} en su carpeta.",
  },
  imageViewer: {
    title: "Imagen",
    description:
      "Usa las teclas de flecha para moverte entre imágenes y las teclas más o menos para hacer zoom.",
    failed: "No se pudo cargar esta imagen.",
    downloadFailed: "No se pudo guardar la imagen.",
    openSource: "Abrir página de origen",
    download: "Descargar",
    close: "Cerrar",
    previous: "Imagen anterior",
    next: "Imagen siguiente",
    zoomIn: "Acercar",
    zoomOut: "Alejar",
    fit: "Ajustar a la ventana",
    copy: "Copiar imagen",
    copied: "Imagen copiada",
    copyFailed: "No se pudo copiar la imagen.",
    removeFromMessage: "Quitar del mensaje",
  },
  browser: {
    drafts: {
      title: "¿Descartar una página con texto escrito?",
      description:
        "Para que el chat siga respondiendo con fluidez, solo se mantienen activas tres páginas ocultas con texto escrito. Si continúas, se descartará {pages} y podría perderse el texto sin guardar. Cancelar te regresa a la pestaña anterior.",
      confirm: "Descartar y continuar",
    },
    agentSettings: {
      title: "Uso del navegador",
      docs: "Documentación",
      docsLabel: "Abrir la documentación del uso del navegador",
      enable: "Permitir que los modelos usen el navegador",
      enableDescription:
        "Los modelos abren páginas en su propia pestaña del navegador, las leen y hacen clic y escriben por ti. También está en el menú + del redactor.",
      history: "Historial de navegación",
      historyDescription:
        "Acceso total omite la aprobación; Nunca sigue bloqueando el acceso",
      alwaysAsk: "Preguntar siempre",
      alwaysAllow: "Permitir siempre",
      never: "Nunca",
      scripts: "Ejecutar scripts",
      scriptsDescription:
        "Elige si los modelos pueden ejecutar scripts en las páginas. Verás cada script antes de que se ejecute",
      cursor: "Mostrar cursor",
      cursorDescription:
        "Muestra un puntero donde los modelos hacen clic, escriben y arrastran en la página",
      askEveryTime: "Preguntar cada vez",
      permissionsTitle: "Permisos del agente",
      permissionsDescription:
        "Define valores predeterminados y excepciones por sitio. Los campos de contraseña y de pago nunca se completan.",
      site: "Sitio o patrón",
      sitePlaceholder: "example.com o *.example.com",
      default: "Predeterminado",
      browsing: "Navegación",
      actions: "Acciones",
      downloads: "Descargas",
      unlabelled: "Clics sin etiqueta",
      askNewSites: "Preguntar en sitios nuevos",
      requiresApproval: "Requiere aprobación",
      blocked: "Bloqueado",
      addSite: "Agregar",
      add: "Agregar",
      cancel: "Cancelar",
      removeSite: "Quitar {site}",
      permissionsNote:
        "Aquí solo aparecen los sitios con permisos propios. Las acciones son inicios de sesión, envíos de formularios y botones que compran, envían o eliminan.",
      siteInvalid: "Escribe un sitio como example.com o *.example.com.",
    },
    agent: {
      working: "Unsloth está navegando en esta pestaña",
      waiting: "Unsloth necesita tu respuesta en el navegador para {chat}",
      otherChat: "otro chat",
      show: "Mostrar",
      stop: "Detener",
      allow: "Permitir",
      deny: "Denegar",
      ask: {
        open: "¿Permitir que Unsloth abra {host}?",
        signIn: "¿Permitir que Unsloth inicie sesión en {host}?",
        submit: "¿Permitir que Unsloth envíe este formulario en {host}?",
        click: "¿Permitir que Unsloth haga clic en “{name}” en {host}?",
        history: "¿Permitir que Unsloth lea tu historial de navegación?",
        script: "¿Permitir que Unsloth ejecute este script en {host}?",
        pointer:
          "¿Permitir que Unsloth haga clic en partes sin etiqueta de {host}, como un lienzo?",
      },
      step: {
        open: "Abriendo {detail}",
        back: "Volviendo atrás",
        forward: "Avanzando",
        reload: "Recargando la página",
        read: "Leyendo la página",
        find: "Buscando “{detail}”",
        click: "Haciendo clic en “{detail}”",
        type: "Escribiendo en “{detail}”",
        select: "Eligiendo “{detail}”",
        press: "Presionando {detail}",
        scroll: "Desplazándose",
        screenshot: "Haciendo una captura de pantalla",
        evaluate: "Ejecutando un script en la página",
        history: "Leyendo tu historial de navegación",
        move: "Moviendo el mouse",
        drag: "Arrastrando",
      },
    },
    title: "Navegador",
    tabs: "Pestañas del navegador",
    newTab: "Nueva pestaña",
    newTabFullView: "Nueva pestaña en vista completa",
    closeTab: "Cerrar pestaña",
    close: "Cerrar navegador",
    show: "Mostrar navegador",
    back: "Atrás",
    forward: "Adelante",
    reload: "Recargar",
    more: "Más",
    addressPlaceholder: "Busca o escribe una URL",
    local: "Local",
    browserUse: {
      label: "Uso del navegador",
      description: "Permite que los modelos controlen el navegador integrado",
      unavailable:
        "Carga un modelo que pueda llamar a herramientas para que use el navegador.",
    },
    suggested: "Sugeridos",
    suggestedMenu: {
      open: "Abrir",
      openInNewTab: "Abrir en una pestaña nueva",
      remove: "Quitar de sugeridos",
    },
    tabMenu: {
      duplicate: "Duplicar",
      closeOthers: "Cerrar otras pestañas",
      pin: "Fijar en la barra lateral",
      unpin: "Dejar de fijar en la barra lateral",
      newTabRight: "Nueva pestaña a la derecha",
      fork: "Bifurcar",
      forkNewChat: "Nuevo chat",
      forkTemporaryChat: "Chat temporal",
      copyUrl: "Copiar URL",
      openExternal: "Abrir en el navegador externo",
      rename: "Cambiar nombre",
      renameLabel: "Nombre de la pestaña",
      mute: "Silenciar pestaña",
      unmute: "Activar sonido de la pestaña",
      close: "Cerrar",
      closeRight: "Cerrar pestañas a la derecha",
      pageOptions: "Opciones de la página",
    },
    siteInfo: {
      label: "Información del sitio",
      secure: "Conexión segura",
      insecure: "Conexión no segura",
      back: "Atrás",
      close: "Cerrar",
      security: "Seguridad",
      secureTitle: "La conexión es segura",
      secureDescription:
        "Tu información, como contraseñas o números de tarjeta, es privada cuando se envía a este sitio.",
      insecureTitle: "La conexión no es segura",
      insecureDescription:
        "Otras personas podrían ver la información que envíes a este sitio, como contraseñas o números de tarjeta.",
      learnMore: "Más información",
      certificateValid: "El certificado es válido",
    },
    tools: "Herramientas",
    recents: "Recientes",
    website: "Sitio web",
    recentNewer: "Más recientes",
    recentOlder: "Más antiguos",
    openExternal: "Abrir en el navegador del sistema",
    copyLink: "Copiar enlace",
    linkCopied: "Enlace copiado",
    download: "Descargar",
    settings: "Configuración del navegador",
    cannotShowFile:
      "Este archivo no se puede mostrar aquí. Descárgalo para abrirlo.",
    fileGone:
      "Este archivo ya no está disponible. Vuelve a abrirlo desde el chat.",
    settingsTitle: "Navegador",
    linksTitle: "Enlaces y archivos",
    addressBarTitle: "Barra de direcciones",
    openLinksSetting: "Abrir enlaces en",
    openLinksSettingDescription:
      "Dónde se abren los enlaces del chat: en tu navegador predeterminado o en una pestaña del navegador junto a la conversación.",
    linkDestinationDefault: "Navegador predeterminado",
    linkDestinationPanel: "Navegador de Unsloth",
    openFilesSetting: "Abrir archivos en el navegador",
    openFilesSettingDescription:
      "Los documentos que adjuntas, como PDF, archivos de Office, HTML y texto, se abren en una pestaña del navegador junto a la conversación en lugar de una ventana de vista previa.",
    searchEngineSetting: "Motor de búsqueda",
    searchEngineSettingDescription:
      "Se usa cuando la barra de direcciones recibe palabras en lugar de una URL.",
    showFullUrlSetting: "Mostrar la URL completa",
    showFullUrlSettingDescription:
      "Incluye la consulta y el fragmento en la barra de direcciones, no solo el sitio y la ruta.",
    bookmarksTitle: "Marcadores",
    bookmarksToolbarSetting: "Barra de marcadores",
    bookmarksToolbarSettingDescription:
      "Cuándo mostrar tus páginas guardadas debajo de la barra de direcciones.",
    bookmarkEditorSettingDescription:
      "Ponle nombre a un marcador y elige su ubicación al guardarlo.",
    bookmarksSettingDescription:
      "Consulta, edita y quita las páginas guardadas en el navegador.",
    browsingDataTitle: "Datos de navegación",
    historySetting: "Historial de navegación",
    historySettingDescription:
      "Consulta y administra las páginas visitadas en el navegador.",
    downloadsSetting: "Historial de descargas",
    downloadsSettingDescription:
      "Consulta y administra los archivos descargados desde el navegador.",
    pagesFromChat: "Abre un chat para ver esto en el navegador.",
    tabsTitle: "Pestañas",
    switchToNewTabsSetting: "Cambiar a las pestañas nuevas al instante",
    switchToNewTabsSettingDescription:
      "Cuando abres un enlace en una pestaña nueva con ⌘ o Ctrl, se muestra en lugar de abrirse en segundo plano.",
    switchToNewTabsNative:
      "Las páginas en la aplicación de escritorio siempre abren las pestañas nuevas en primer plano.",
    defaultZoomSetting: "Zoom predeterminado",
    defaultZoomSettingDescription:
      "El tamaño de las páginas al abrirse. Restablecer el zoom de una página vuelve a este valor.",
    newTabPageTitle: "Página de nueva pestaña",
    showSuggestedSetting: "Mostrar sitios sugeridos",
    showSuggestedSettingDescription:
      "Sitios que visitas con frecuencia, como accesos directos en una pestaña nueva.",
    showRecentsSetting: "Mostrar páginas recientes",
    showRecentsSettingDescription:
      "Páginas que visitaste hace poco, en una pestaña nueva.",
    hiddenSuggestionsSetting: "Sugerencias quitadas",
    hiddenSuggestionsSettingDescription:
      "Sitios que quitaste de Sugeridos: {count}. Restáuralos para que vuelvan a aparecer.",
    hiddenSuggestionsNone:
      "Los sitios que quites de Sugeridos se pueden restaurar aquí.",
    restore: "Restaurar",
    importBookmarksSetting: "Importar y exportar",
    importBookmarksSettingDescription:
      "Importa marcadores de Chrome, Safari, Firefox o Edge con el archivo HTML que exportan, o guarda los tuyos en uno.",
    importBookmarks: "Importar…",
    exportBookmarks: "Exportar",
    importBookmarksDoneOne: "Se importó {count} marcador.",
    importBookmarksDoneMany: "Se importaron {count} marcadores.",
    importBookmarksNone: "No hay marcadores nuevos en ese archivo.",
    importBookmarksPartial:
      "Se importaron {count}, pero {leftOut} más no cupieron: Unsloth guarda hasta {max} marcadores.",
    importBookmarksFailed:
      "No se pudo leer ese archivo. Elige un archivo HTML de marcadores exportado desde un navegador.",
    downloadsTitle: "Descargas",
    askWhereToSaveSetting: "Preguntar dónde guardar cada archivo",
    askWhereToSaveSettingDescription:
      "Muestra un cuadro de diálogo para guardar los archivos que descargas en el navegador.",
    askWhereToSaveUnsupported:
      "Tu navegador web guarda las descargas en su propia carpeta de descargas.",
    downloadLocationSetting: "Ubicación",
    downloadLocationDefault: "Carpeta de descargas del sistema",
    downloadLocationChange: "Cambiar",
    downloadLocationReset: "Restablecer",
    askBeforeDownloadingSetting: "Preguntar antes de descargar",
    askBeforeDownloadingSettingDescription:
      "Confirma los archivos de sitios web antes de guardarlos.",
    downloadSitesSetting: "Permisos de descarga por sitio",
    downloadSitesSettingDescription:
      "Sitios cuyas descargas elegiste permitir o bloquear siempre.",
    downloadSitesNone:
      "Marca Recordar para la próxima vez cuando se te pregunte y el sitio aparecerá aquí.",
    downloadSitesFor: "Descargas de {host}",
    downloadSitesRemove: "Olvidar {host}",
    downloadSiteAllow: "Permitir siempre",
    downloadSiteBlock: "Bloquear siempre",
    downloadPrompt: {
      title: "¿Descargar este archivo?",
      description: "{host} quiere guardar {name} en tu computadora.",
      dangerous:
        "Los archivos como este pueden ejecutar programas en tu computadora. Descárgalo solo si confías en {host}.",
      remember: "Recordar para la próxima vez",
      cancel: "Cancelar",
      download: "Descargar",
      blocked: "Se bloqueó una descarga de {host}.",
      ready: "{name} está listo para guardar",
      save: "Guardar",
    },
    saveDownloadHistorySetting: "Conservar el historial de descargas",
    saveDownloadHistorySettingDescription:
      "Lista los archivos que descargas del navegador en Historial de descargas.",
    saveHistorySetting: "Guardar historial de navegación",
    saveHistorySettingDescription:
      "Recuerda las páginas que visitas, para Historial, Recientes y Sugeridos.",
    historyRetentionSetting: "Conservar el historial durante",
    historyRetentionSettingDescription:
      "Las visitas más antiguas que este plazo se eliminan automáticamente.",
    retention: {
      forever: "Hasta que se borre",
      days90: "3 meses",
      days30: "1 mes",
      days7: "1 semana",
      days1: "1 día",
    },
    annotationsTitle: "Anotaciones",
    annotationScreenshotsSetting: "Capturas de pantalla de anotaciones",
    annotationScreenshotsSettingDescription:
      "Una captura de la página con tus marcas ayuda a los modelos de visión a entender tus comentarios, pero usa más contexto.",
    annotationScreenshotsUnsupported:
      "Requiere la aplicación de escritorio o un navegador basado en Chromium, como Chrome o Edge.",
    annotationScreenshotsHint:
      "En un navegador web, se te pedirá compartir la pestaña cada vez que envíes anotaciones.",
    annotationScreenshotsAlways: "Incluir siempre",
    annotationScreenshotsNever: "No incluir nunca",
    clearDataSetting: "Borrar datos de navegación",
    clearDataSettingDescription:
      "Borra el historial de navegación, el historial de descargas y las páginas en caché. Las páginas conservan las cookies y los datos de sitios solo hasta que se recarga Unsloth.",
    manage: "Administrar",
    fullView: {
      enter: "Entrar a la vista completa",
      exit: "Salir de la vista completa",
      minimize: "Minimizar chat",
      showChat: "Mostrar chat",
      showConversation: "Mostrar conversación",
      hideConversation: "Ocultar conversación",
      newChat: "Nuevo chat",
      options: "Opciones del chat",
      openChat: "Abrir en vista completa",
      moveLeft: "Mover al panel izquierdo",
      moveRight: "Mover al panel derecho",
    },
    annotate: {
      countOne: "1 anotación",
      countMany: "{count} anotaciones",
      cancel: "Cancelar",
      send: "Enviar",
      move: "Mover",
      placeholder: "Describe un cambio o haz una pregunta",
      areaQuote: "Área de la página",
      imageQuote: "Imagen",
      page: "Preguntar sobre esta página",
      pagePlaceholder: "Haz una pregunta sobre esto",
      save: "Guardar comentario",
      dictate: "Dictar",
      stopDictating: "Dejar de dictar",
      dictateFailed: "No se pudo iniciar el dictado",
    },
    menu: {
      find: "Buscar en la página",
      print: "Imprimir",
      printFailed: "No se pudo imprimir esta página",
      screenshot: "Tomar una captura de pantalla",
      zoom: "Zoom",
      zoomIn: "Acercar",
      zoomOut: "Alejar",
      zoomReset: "Restablecer zoom",
      deviceToolbar: "Mostrar barra de dispositivos",
      clearData: "Borrar datos de navegación",
    },
    screenshot: {
      added: "Captura agregada al chat",
      taken: "Captura tomada",
      save: "Guardar",
      failed: "No se pudo tomar la captura de pantalla",
      otherSurface: "Comparte esta pestaña para capturar la página",
    },
    device: {
      mobile: "Celular",
      tablet: "Tableta",
      close: "Ocultar barra de dispositivos",
    },
    downloads: {
      title: "Descargas",
      inProgressLabel: "Descargando",
      inProgress: "Descargando…",
      complete: "Descarga completada",
      failed: "La descarga falló",
      downloaded: "Descargado",
      missing: "Archivo movido o eliminado",
      open: "Abrir",
      openFailed: "No se pudo abrir {name}.",
      copyLink: "Copiar enlace de descarga",
      showAll: "Mostrar todo",
      savePage: "Guardar página como…",
    },
    file: {
      openIn: "Abrir en",
      newChat: "Nuevo chat",
      newBrowserTab: "Nueva pestaña del navegador",
      requestEdits: "Solicitar cambios",
      requestEditsPrompt: "Edita {name}: ",
      preview: "Vista previa",
      source: "Código fuente",
      viewMode: "Vista del archivo",
      runAgain: "Ejecutar de nuevo",
      console: "Consola",
      copy: "Copiar contenido",
      copied: "Copiado",
      copyFailed: "No se pudo copiar el archivo.",
      copyPath: "Copiar ruta",
      pathCopied: "Ruta copiada",
      openFailed: "No se pudo abrir {name}.",
      wrap: "Ajustar líneas",
      reactNeedsNode:
        "Las vistas previas de React necesitan Node.js 20.19 o posterior. Instálalo si falta y vuelve a ejecutar la configuración de Unsloth.",
      reactFailed:
        "No se pudo preparar esta vista previa. Vuelve a ejecutarla para intentarlo de nuevo.",
    },
    video: {
      play: "Reproducir",
      pause: "Pausar",
      seek: "Buscar",
      volume: "Volumen",
      mute: "Silenciar",
      unmute: "Activar sonido",
      more: "Más opciones de reproducción",
      speed: "Velocidad de reproducción",
      normalSpeed: "Normal",
      loop: "Repetir",
      pictureInPicture: "Imagen en imagen",
      fullscreen: "Pantalla completa",
      exitFullscreen: "Salir de pantalla completa",
      copyFrame: "Copiar fotograma",
      frameCopied: "Fotograma copiado",
      copyFrameFailed: "No se pudo copiar el fotograma.",
      copyOptions: "Opciones de copia",
      open: "Abrir",
      openOptions: "Opciones de apertura",
    },
    pages: {
      searchDownloads: "Buscar en el historial de descargas",
      clearAll: "Borrar todo",
      deleted: "Eliminado",
      showInFinder: "Mostrar en Finder",
      showInExplorer: "Mostrar en el Explorador",
      showInFolder: "Mostrar en la carpeta",
      revealFailed: "No se pudo mostrar {name} en su carpeta.",
      history: "Historial",
      downloads: "Descargas",
      bookmarks: "Marcadores",
      search: "Buscar",
      today: "Hoy",
      yesterday: "Ayer",
      remove: "Quitar",
      clearHistory: "Borrar historial",
      clearDownloads: "Borrar descargas",
      noHistory: "Las páginas que visites en el navegador aparecerán aquí.",
      showMore: "Mostrar más",
      noDownloads:
        "Los archivos que descargues desde el navegador aparecerán aquí.",
      noBookmarks:
        "Las páginas que guardes con la estrella de la barra de direcciones aparecerán aquí.",
      noMatches: "No hay resultados para tu búsqueda.",
      fromChat: "Desde el chat",
      searchHistory: "Buscar en el historial de navegación",
      filter: "Filtrar",
      allTime: "Desde siempre",
      allTimeHistory: "Todo el historial",
      lastWeek: "Últimos 7 días",
      lastMonth: "Últimos 30 días",
      customDates: "Elegir fechas…",
      clearFilter: "Quitar filtro",
      openPage: "Abrir página",
      removeFromHistory: "Quitar del historial",
      selected: "{count} seleccionados",
      cancel: "Cancelar",
      select: "Seleccionar {title}",
      collapse: "Contraer {date}",
      expand: "Expandir {date}",
      pageActions: "Acciones para {title}",
    },
    bookmarks: {
      bookmarkPage: "Agregar esta página a marcadores",
      add: "Agregar marcador",
      edit: "Editar marcador",
      name: "Nombre",
      location: "Ubicación",
      toolbar: "Barra de marcadores",
      other: "Otros marcadores",
      showEditor: "Mostrar el editor al guardar",
      cancel: "Cancelar",
      remove: "Quitar marcador",
      delete: "Eliminar marcador",
      save: "Guardar",
      saved: "Guardado en marcadores",
      removed: "Marcador eliminado",
      undo: "Deshacer",
      manage: "Administrar marcadores",
      more: "Más marcadores",
      toolbarEmpty:
        "Guarda una página con la estrella de la barra de direcciones para agregarla aquí.",
      toolbarAlways: "Mostrar siempre",
      toolbarNewTab: "Mostrar solo en pestañas nuevas",
      toolbarNever: "No mostrar nunca",
    },
    clearData: {
      title: "¿Borrar datos de navegación?",
      description:
        "Esto borra el historial de navegación, el historial de descargas, las páginas en caché y las cookies y los datos de sitios, lo que cierra tu sesión en los sitios. Los archivos descargados permanecen en tu computadora.",
      cancel: "Cancelar",
      confirm: "Borrar datos",
      done: "Datos de navegación borrados",
    },
    error: {
      title: "No se pudo cargar esta página",
      description:
        "{host} no respondió, rechazó la conexión o bloqueó la solicitud.",
      retry: "Intentar de nuevo",
      botCheckTitle: "Este sitio verifica que usas un navegador real",
      botCheckDescription:
        "{host} pide a los visitantes que demuestren que son personas, y la vista de la aplicación no puede hacerlo aquí. Ábrelo en tu navegador.",
      botCheckDescriptionWeb:
        "{host} pide a los visitantes que demuestren que son personas, y la vista de la aplicación no puede hacerlo aquí. Ábrelo en tu navegador o en {app}, que sí puede abrirlo.",
      desktopApp: "Unsloth Desktop",
      resubmit:
        "Esta página proviene de un formulario. Volver a cargarla envía el formulario de nuevo, lo que podría repetir lo que hizo.",
      upload:
        "La subida de archivos no funciona en esta vista. Abre la página en el navegador del sistema para enviar el archivo.",
      scriptNavigation:
        "Esta página intentó ir a otra dirección mediante un script, algo que esta vista no puede seguir. Ábrela en el navegador del sistema.",
      needsBrowser:
        "Esta página intentó usar una función del navegador que esta vista bloquea, así que puede que partes de ella no funcionen. Ábrela en el navegador del sistema.",
    },
    native: {
      externalPrompt: "{host} quiere abrir {url}",
      open: "Abrir",
      downloading: "Descargando {name}",
      downloaded: "{name} se guardó en Descargas",
      downloadFailed: "No se pudo descargar {name}",
      notMarked:
        "{name} se guardó, pero no se pudo marcar como descargado de internet, así que tu sistema no avisará antes de abrirlo.",
      blocked:
        "Esta dirección no se puede abrir en el panel del navegador. Solo abre sitios web públicos.",
      blockedLocal:
        "Esta dirección local no se puede abrir aquí. Escríbela en la barra de direcciones para abrirla.",
      studioAddress:
        "Esta es la dirección del propio Unsloth, así que no se puede abrir en el panel del navegador.",
      clearDataSettingDescription:
        "Borra el historial de navegación, el historial de descargas, las páginas en caché y las cookies y los datos de sitios que guardan las páginas, lo que cierra tu sesión en los sitios.",
      clearDataDescription:
        "Esto borra el historial de navegación, el historial de descargas, las páginas en caché y las cookies y los datos de sitios, lo que cierra tu sesión en los sitios. Los archivos descargados permanecen en tu computadora.",
      clearDataFailed:
        "No se pudieron borrar los datos de sitios. Inténtalo de nuevo.",
    },
  },
  managedEngines: {
    gpu: "GPU",
    gpuHelp: "Elige qué GPU usar.",
    parallelism: "Modo multi-GPU",
    tensor: "Paralelismo de tensores",
    pipeline: "Paralelismo de pipeline",
    data: "Réplicas (paralelismo de datos)",
    pipelineHelp:
      "Divide las capas del modelo entre las GPU. Cada GPU debe poder alojar sus capas y su caché.",
    dataHelp:
      "Reparte las solicitudes entre las GPU. Los modelos densos deben caber en cada GPU; los modelos MoE pueden compartir expertos.",
    tensorParallel: "Divide los cálculos de las capas entre {count} GPU.",
    inUse:
      "Descarga el modelo de la memoria antes de reparar, restaurar o quitar su motor.",
    title: "Más motores",
    description:
      "Motores adicionales para GPU (beta). Pueden ejecutar modelos más rápido, sobre todo en inferencia por lotes. Instálalos aquí y luego elígelos en la configuración de ejecución del modelo.",
    picker: "Motor de inferencia",
    precision: "Precisión",
    precisionAuto: "Predeterminado del modelo",
    precisionHelp:
      "El predeterminado del modelo conserva la precisión almacenada. Las demás opciones convierten los pesos sin cuantizar al cargar.",
    kvCache: "Caché KV",
    kvCacheFp8: "FP8 (la mitad de memoria)",
    kvCacheHelp:
      "FP8 guarda la caché de contexto en 8 bits, así que cabe aproximadamente el doble de contexto en la misma memoria. Sin escalas calibradas en el checkpoint, las respuestas pueden ser algo menos precisas.",
    gpuMemory: "Límite de memoria de GPU",
    gpuMemoryAuto: "Automático",
    gpuMemoryHelp:
      "Lo máximo de cada GPU seleccionada que puede usar el motor. Automático usa lo que esté libre; bájalo para dejar espacio a otros programas.",
    seed: "Semilla",
    seedPlaceholder: "Predeterminado del motor",
    seedHelp:
      "Establece la semilla aleatoria del motor. Los chats simultáneos aún pueden cambiar las respuestas muestreadas.",
    default: "Predeterminado",
    installRequired: "requiere instalación",
    ownerRequired: "Pide al propietario de Unsloth que instale este motor.",
    cancelInstall: "Cancelar instalación",
    installed: "Instalado: {version}",
    repair: "Reparar",
    update: "Actualizar motor",
    install: "Instalar motor",
    installing: "Instalando el motor. Consulta el progreso en Descargas.",
    remove: "Quitar motor",
    rollback: "Restaurar la instalación anterior",
    installTitle: "Instalar {engine}",
    installAndLoad: "Instalar y cargar",
    confirm: "¿Instalar {engine} {version}? Puede descargar varios GB.",
    confirmSized: "¿Instalar {engine} {version}? Unos {size} de descarga.",
    wslSetup:
      "En Windows, {engine} se ejecuta en un entorno WSL2 privado que configura Unsloth. Verás una solicitud de permisos de administrador y quizá tengas que reiniciar.",
    wslReady:
      "En Windows, {engine} se ejecuta dentro del entorno WSL2 privado de Unsloth.",
    wslRestart:
      "Reinicia Windows para terminar de activar WSL2 y luego vuelve a hacer clic en Instalar.",
    background: "Se instala en segundo plano.",
    failed:
      "No se pudo instalar el motor. Vuelve a intentarlo o usa el motor predeterminado.",
    details: "Detalles técnicos",
    cancelled: "Instalación cancelada. Puedes volver a intentarlo.",
    scope:
      "Chat de texto e imagen en GPU NVIDIA. La primera carga puede tardar más.",
  },
  sandboxSetup: {
    levelTitle: "El sandbox del sistema no está disponible",
    levelDescription:
      "Instálalo para usar Alto o mantén Bajo (sandbox de software).",
    useLow: "Usar sandbox Bajo",
    checking: "Comprobando el sandbox en esta computadora…",
    install: "Instalar sandbox",
    windowsSetup: "Configurar el sandbox de Windows",
    copyCommand: "Copiar comando",
    copied: "Comando copiado",
    copyFailed: "No se pudo copiar el comando",
    copiedRunIt:
      "Comando copiado. Ejecútalo en una terminal de la computadora donde se ejecuta Unsloth y luego cambia a Alto.",
    ownerOnly:
      "Solo el propietario puede instalarlo, desde la computadora donde se ejecuta Unsloth.",
    windowsNote:
      "Requiere un paso de administrador después de cada reinicio. MXC es una vista previa, no un límite de seguridad.",
    running:
      "Configurando. Aprueba la solicitud de contraseña o de permisos de administrador en la computadora donde se ejecuta Unsloth…",
    succeeded: "El sandbox del sistema está listo. Sandbox Alto está activado.",
    stillUnavailable:
      "La configuración terminó, pero el sandbox del sistema todavía no supera su comprobación.",
    declined:
      "Se rechazó la solicitud de contraseña o de permisos de administrador.",
    failed: "La configuración falló.",
    startError: "No se pudo iniciar la configuración del sandbox",
    checkFailed: "No se pudo comprobar el sandbox en esta computadora.",
    retry: "Reintentar",
    keepsRunning:
      "Cerrar esto no detiene la configuración. El resultado aparece en Configuración > Seguridad.",
  },
  composerSettings: {
    title: "Redactor",
    plainText: "Redactor de texto sin formato",
    plainTextDescription:
      "Mantiene el código, Markdown y los enlaces como texto literal. Desactívalo para mostrar una vista previa con formato.",
    showContext: "Uso de la ventana de contexto",
    sendShortcut: "Atajo para enviar",
    sendMultiline: "{mod} + Enter en mensajes de varias líneas",
    sendAlways: "{mod} + Enter siempre",
    sendEnterDescription: "Enter envía. Shift + Enter agrega una línea nueva.",
    sendMultilineDescription:
      "Enter envía un mensaje de una línea. Cuando tiene más de una línea, Enter agrega una línea nueva y {mod} + Enter envía.",
    sendAlwaysDescription: "{mod} + Enter envía. Enter agrega una línea nueva.",
    followUp: "Mensajes de seguimiento",
    followUpDescription:
      "Qué ocurre cuando envías durante una respuesta. Presiona {shortcut} para hacer lo contrario una vez.",
    followUpMultilineShortcut:
      "{shortcut} ({multiline} en un mensaje de varias líneas)",
    queue: "Poner en cola",
    steer: "Redirigir",
    steerDescription:
      "Redirigir detiene la respuesta actual y envía tu mensaje de seguimiento a continuación.",
    settings: "Configuración del redactor",
    preview: "Vista previa con formato",
  },
  promptQueue: {
    loading: "Cargando mensajes en cola",
    listLabel: "Mensajes en cola",
    regionLabel: "Cola de mensajes, {current} de {total}",
    itemLabel: "Mensaje en cola {position} de {total}: {prompt}",
    reorderInstructions:
      "Arrastra el control de arrastre para reordenar. Con el control enfocado, usa Arriba o Abajo para moverlo una posición, o Inicio o Fin para moverlo al principio o al final.",
    dragTooltip: "Arrastra para reordenar",
    reorderLabel: "Reordenar el mensaje en cola {position} de {total}",
    paused: "En pausa",
    steer: "Redirigir",
    steerTooltip:
      "Interrumpir la respuesta y enviar este mensaje a continuación",
    steerLabel: "Redirigir con el mensaje en cola {position}",
    removeTooltip: "Quitar de la cola",
    removeLabel: "Quitar el mensaje en cola {position}",
    moreTooltip: "Más opciones",
    moreLabel: "Más opciones para el mensaje en cola {position}",
    editItem: "Editar mensaje",
    copyItem: "Copiar mensaje",
    editLabel: "Editar el mensaje en cola {position}",
    cancel: "Cancelar",
    save: "Guardar",
    turnOffQueueing: "Desactivar la cola",
    turnOnQueueing: "Activar la cola",
    resume: "Reanudar la cola",
    queueButton: "Poner el mensaje en cola",
    steerButton: "Redirigir la respuesta",
    sendTooltip: "Enviar mensaje ({shortcut})",
    sendLabel: "Enviar mensaje",
    followUpTooltip: "{action} ({send}) · {opposite} para lo contrario",
    announceUpdated: "Mensaje en cola actualizado.",
    announceEditFailed:
      "Este mensaje ya no se puede editar porque la cola cambió.",
    announceRemoved: "Mensaje quitado de la cola.",
    announceSteered: "Este mensaje redirigirá la respuesta a continuación.",
    announceSteerFailed:
      "Este mensaje no pudo redirigir la respuesta. Revisa la cola e inténtalo de nuevo.",
    announceCopied: "Mensaje copiado.",
    announceCopyFailed: "No se pudo copiar este mensaje. Inténtalo de nuevo.",
    announceQueueingOn:
      "Los nuevos mensajes de seguimiento se pondrán en cola después de la respuesta actual.",
    announceQueueingOff:
      "Los nuevos mensajes de seguimiento redirigirán la respuesta actual.",
    announceMoved: "Mensaje movido a la posición {position} de {total}.",
    announceMoveFailed:
      "La cola cambió antes de poder mover este mensaje. Inténtalo de nuevo.",
    announceDragReset:
      "La cola cambió. Arrastra de nuevo para reordenar los mensajes restantes.",
    editingHint: "Editando el mensaje",
    queueingOffHint:
      "Los mensajes nuevos interrumpen y se ejecutan a continuación.",
    queueingOnHint: "Los mensajes nuevos esperan su turno.",
    queueingHintShared: "La cola se conserva.",
  },
  rewind: {
    title: "Rebobinar",
    confirmTitle: "¿Rebobinar hasta este mensaje?",
    restoreLabel: "Restaurar",
    pickDescription: "Elige un mensaje al que volver.",
    pickEmpty: "Todavía no hay mensajes a los que volver.",
    attachmentOnly: "Solo archivos adjuntos",
    modeBoth: "Conversación y archivos",
    modeConversation: "Solo conversación",
    modeFiles: "Solo archivos",
    checkFailed: "No se pudieron comprobar los cambios en archivos: {error}",
    summary:
      "Archivos por restaurar: {restore}. Por eliminar: {remove}. Por recrear: {recreate}.",
    conflicts: "Cambiados desde entonces, se omitirán",
    untracked: "No se pueden restaurar",
    scopeNote:
      "Los archivos incluyen código. Los programas, paquetes y archivos fuera de la carpeta de este chat no se modifican.",
    back: "Atrás",
    cancel: "Cancelar",
    confirm: "Rebobinar",
    waitForResponse: "Espera a que termine la respuesta actual",
    restoredOne: "1 archivo restaurado",
    restoredMany: "{count} archivos restaurados",
    nothingRestored: "No había archivos que restaurar",
    skippedCount: "{count} omitidos",
    filesNotRestored:
      "{count} archivos modificados no se pueden restaurar y quedan como están.",
    restoreFailed: "No se pudieron restaurar los archivos",
    undo: "Deshacer",
    undone: "Restauración de archivos deshecha",
    undoFailed: "No se pudo deshacer la restauración de archivos",
  },
  chatMenu: {
    more: "Opciones del chat",
    copy: "Copiar",
    compact: "Compactar conversación",
    compactQueued: "Se solicitó la compactación para el siguiente mensaje",
    compactEmpty: "Aún no hay nada que compactar",
    compactUnsupported:
      "La compactación requiere un modelo GGUF o MLX compatible con herramientas",
  },
  halloween: {
    navLabel: "Truco o trato…",
    title: "¡¿Truco o trato?!",
    description:
      "Preparamos algunos sustos para Unsloth, disponibles el resto de octubre.",
    chooseTheme: "Elige un tema terrorífico",
    spookyGreetings: "Saludos terroríficos",
    spookyGreetingsDescription:
      "El perezoso recibe los chats nuevos con un susto.",
    decorations: "Arañas y murciélagos",
    decorationsDescription: "Telarañas, y alguna que otra araña o murciélago.",
    surprise: "¡Truco o trato!",
    surpriseDescription: "Deja que el perezoso elija un tema por ti.",
    themesTitle: "Temas terroríficos",
    back: "Atrás",
    footer: "Unsloth vuelve a la normalidad a medianoche del 1 de noviembre.",
    removeNow: "Quitar ahora",
    hideFromSidebar: "Ocultar de la barra lateral",
    greeting: {
      pumpkins: "Las calabazas ya están talladas. ¿Qué sigue?",
      brewing: "¡A preparar la poción!",
      trickOrTreat: "¿Truco o trato?",
      manor: "Te damos la bienvenida a la mansión",
      manorName: "Te damos la bienvenida a la mansión, {name}",
      happyHalloween: "¡Feliz Halloween!",
      happyHalloweenName: "¡Feliz Halloween, {name}!",
      spirits: "Los espíritus están escuchando…",
      mysteries: "¿Qué misterios descubriremos esta noche?",
      conjure: "¿Qué puedo conjurar para ti?",
    },
  },
  welcome: {
    goodToSeeYou: "Me alegra verte",
    goodToSeeYouName: "Me alegra verte, {name}",
    readyWhenYouAre: "Listo cuando tú lo estés",
    onYourMind: "¿Qué tienes en mente hoy?",
    howCanIHelp: "¿En qué puedo ayudarte?",
    goodMorning: "Buenos días",
    goodMorningName: "Buenos días, {name}",
    goodEvening: "Buenas noches",
    goodEveningName: "Buenas noches, {name}",
    tonight: "¿Qué planes tienes para esta noche?",
    nightOwl: "¿Modo noctámbulo?",
    lateNightIdeas: "¿Ideas de madrugada?",
    upLate: "¿Una idea a estas horas?",
    nightShift: "Empieza el turno de noche",
    nightShiftName: "Empieza el turno de noche, {name}",
    temporaryChat: "Chat temporal",
    temporaryChatDescription:
      "Este chat no aparecerá en tu historial ni se guardará. Desaparece cuando salgas.",
  },
  picker: {
    unifiedTightFit: "Ajuste justo",
    unifiedTightFitHint:
      "Puede cargarse con un contexto más corto o funcionar más lento en esta Mac.",
    onDevice: "En el dispositivo",
    huggingFace: "Hugging Face",
    retry: "Reintentar",
    loadMore: "Cargar más",
    offlineTitle: "Estás sin conexión",
    offlineBody: "Cambia a Dispositivo para usar {noun} locales o en caché.",
    offlineSwitchDevice: "Dispositivo",
    searchAriaLabel: "Buscar {noun}",
    modelSourceAriaLabel: "Origen del modelo",
    hubSectionAriaLabel: "Sección del Hub",
    modelDropped: "Ya no se ofrece",
    modelDroppedByProvider: "{provider} · ya no se ofrece",
    modelDisabled: "No activado",
    modelDisabledByProvider: "{provider} · no activado",
    multipleMatches: "Hay varios {noun} coincidentes. Elige uno de la lista.",
    rateLimitedTitle: "Se alcanzó el límite de solicitudes de Hugging Face",
    rateLimitedBody:
      "Espera un momento y vuelve a intentar la búsqueda de {noun}.",
    modelScope: "ModelScope",
    useModelScope: "Usar ModelScope",
    useModelScopeHint:
      "Busca y descarga desde ModelScope en su lugar. Puedes volver a cambiarlo en Configuración.",
    useModelScopeFailed: "No se pudo cambiar a ModelScope.",
    updateToken: "Actualizar token",
    hfToken: {
      label: "Token de HF",
      saved: "Guardado",
      add: "Sin configurar",
      savedAriaLabel: "Token de Hugging Face guardado",
      addAriaLabel: "Configurar token de Hugging Face",
      savedHint: "Token guardado. El acceso se comprueba cuando lo usas.",
      addHint:
        "Configura un token para acceder a repositorios privados y restringidos.",
    },
  },
  common: {
    duplicate: "Duplicar",
    cancel: "Cancelar",
    close: "Cerrar",
    delete: "Eliminar",
    done: "Listo",
    error: "Error",
    export: "Exportar",
    help: "Ayuda",
    todayAt: "Hoy, {time}",
    yesterdayAt: "Ayer, {time}",
    loading: "Cargando...",
    new: "Nuevo",
    rename: "Renombrar",
    save: "Guardar",
    saving: "Guardando...",
    search: "Buscar",
    shutdown: "Apagar",
  },
  shell: {
    helpMenu: {
      documentation: "Documentación",
      keyboardShortcuts: "Atajos de teclado",
      whatsNew: "Novedades",
      troubleshooting: "Solución de problemas",
      systemStatus: "Estado del sistema",
      sendFeedback: "Enviar comentarios",
      about: "Acerca de Unsloth",
    },
    find: {
      label: "Buscar en la página",
      previous: "Coincidencia anterior",
      next: "Coincidencia siguiente",
      close: "Cerrar búsqueda",
      truncated: "Esta página es demasiado larga para buscarla por completo.",
      searchChat: "Buscar en el chat",
      searchBrowser: "Buscar en la página del navegador",
    },
    zoom: {
      label: "Zoom",
      zoomOut: "Alejar",
      zoomIn: "Acercar",
      reset: "Restablecer",
      announce: "Zoom {percent}%",
    },
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
      resizeRunSettings: "Redimensionar o cerrar la configuración de ejecución",
      openRunSettings: "Abrir la configuración de ejecución",
      chatOptions: "Opciones del chat",
      runOptions: "Opciones de ejecución",
    },
    navigation: {
      newChat: "Nuevo chat",
      returnToChat: "Volver al chat",
      returnToChats: "Volver a {count} chats",
      chatGenerating: "Generando",
      chatAwaitingApproval: "Esperando aprobación",
      chatUnread: "No leído",
      compare: "Comparar",
      search: "Buscar",
      projects: "Proyectos",
      library: "Biblioteca",
      hub: "Centro de modelos",
      train: "Entrenar",
      recipes: "Recetas",
      images: "Imágenes",
      video: "Video",
      audio: "Audio",
      tasks: "Programadas",
      trainChecking: "Comprobando si esta computadora admite entrenamiento...",
      videoChecking: "Comprobando si esta computadora admite video...",
      more: "Más",
      customizeSidebar: "Personalizar la barra lateral",
      newBadge: "Nuevo",
      export: "Exportar",
      recents: "Recientes",
      noChatsYet: "Aún no hay chats",
      noChats: "Sin chats",
      allProjectsPinned: "Todos los proyectos están fijados",
      allProjectsFiled: "Todos los proyectos están fijados o en secciones",
      noProjects: "No hay proyectos",
      showMore: "Mostrar más",
      showLess: "Mostrar menos",
      settings: "Configuración",
      api: "API",
      lightMode: "Modo claro",
      darkMode: "Modo oscuro",
      guidedTour: "Recorrido guiado",
      help: "Ayuda",
      logOut: "Cerrar sesión",
      logOutNeedsPassword:
        "Define una contraseña en Configuración > Remoto y LAN antes de cerrar sesión para poder volver a iniciar sesión.",
      logOutFailed:
        "No se pudo cerrar sesión de forma segura. Inténtalo de nuevo en un momento.",
      shutdown: "Apagar",
    },
    commandPalette: {
      placeholder: "Escribe un comando o busca...",
      noResults: "No se encontraron resultados.",
      navigation: "Navegación",
      actions: "Acciones",
      chat: "Chat",
      searchChats: "Buscar chats...",
    },
    search: {
      placeholder: "Buscar",
      tabs: {
        all: "Todo",
        chats: "Chats",
        projects: "Proyectos",
        files: "Archivos",
        models: "Modelos",
      },
      recents: "Recientes",
      actions: "Acciones",
      newChat: "Nuevo chat",
      newTemporaryChat: "Nuevo chat temporal",
      fineTune: "Ajustar un modelo",
      generateImage: "Generar una imagen",
      generateVideo: "Generar un video",
      untitledChat: "Chat sin título",
      compare: "Comparar",
      loading: "Cargando…",
      empty: {
        all: "Aún no hay nada que buscar.",
        chats: "Aún no hay chats.",
        projects: "Aún no hay proyectos.",
        files: "Aún no hay archivos en tu Biblioteca.",
        models: "Aún no hay modelos descargados.",
      },
      noMatches: "Sin resultados.",
      when: {
        today: "Hoy",
        pastWeek: "Última semana",
        pastMonth: "Último mes",
        older: "Más antiguo",
      },
      footer: {
        close: "Cerrar",
        changeType: "Cambiar tipo",
        open: "Abrir",
      },
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
      markRead: "Marcar como leído",
      deleteChats: "Eliminar chats",
      deleteTitle: "Eliminar chats",
      deleteDescription:
        "¿Eliminar {count} chats? Esta acción no se puede deshacer.",
      deleteFilesDescription:
        "Se elimina del disco la carpeta de sandbox propia de cada chat. Los archivos que hayan escrito dentro de un proyecto permanecen en el espacio de trabajo de ese proyecto.",
      deleteFilesLabel: "Eliminar archivos y carpeta del sandbox",
      deleteChatFilesDescription:
        "Se elimina del disco la carpeta de sandbox propia de este chat. Los archivos que escribió dentro de un proyecto permanecen en el espacio de trabajo de ese proyecto.",
    },
    organize: {
      sidebarHeading: "Organizar la barra lateral",
      byProject: "Por proyecto",
      inOneList: "En una sola lista",
      sortChatsBy: "Ordenar chats por",
      projectsHome: "Inicio de proyectos",
      lastUpdated: "Última actualización",
      manualOrder: "Orden manual",
      switchedToManual: "Orden manual: arrastra las filas para reordenarlas",
      organizeChats: "Organizar chats",
      organizeProjects: "Organizar proyectos",
      sortPinnedChats: "Ordenar chats fijados",
      show: "Mostrar",
      newSection: "Nueva sección",
    },
    sections: {
      createTitle: "Nueva sección",
      createDescription: "Agrupa chats y proyectos como quieras",
      namePlaceholder: "Nombre de la sección",
      create: "Crear sección",
      renameTitle: "Renombrar sección",
      renameDescription: "Dale un nombre nuevo a esta sección",
      edit: "Editar",
      remove: "Quitar sección",
      markAllRead: "Marcar todo como leído",
      moveTo: "Mover a",
      section: "Sección",
      sectionsHeading: "Secciones",
      removeFromProject: "Quitar del proyecto",
      newSection: "Nueva sección",
      removeFromSection: "Quitar de la sección",
      removeFrom: "Quitar de {name}",
      empty: "Arrastra chats o proyectos aquí",
      sectionOptions: "Opciones de sección",
      newChatInSection: "Nuevo chat en {name}",
      deleted: "Sección “{name}” eliminada",
      undo: "Deshacer",
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
    accounts: {
      signsInAs: "Inicia sesión como {username}",
      signsInAsOwner: "Inicia sesión como {alias} o {username}",
      signInName: "Tu nombre de inicio de sesión",
      signInNameDescription:
        "El nombre que escribes para iniciar sesión cuando esta instalación tiene más de una cuenta. unsloth también funciona siempre. Déjalo vacío para usar solo unsloth.",
      signInNameSave: "Guardar nombre",
      title: "Cuentas",
      description:
        "Crea cuentas privadas de Unsloth. Los nuevos usuarios inician sesión con un código de configuración de un solo uso y eligen una contraseña.",
      username: "Nombre de usuario",
      create: "Crear cuenta",
      createDescription:
        "Comparte un código de configuración para que la persona elija su propia contraseña.",
      actionsFor: "Administrar {username}",
      actions: "Acciones",
      search: "Buscar cuentas",
      noResults: "No hay cuentas coincidentes",
      created: "Creada",
      status: "Estado",
      loginHint:
        "Inicia sesión como {username} con este código como contraseña y luego elige una nueva.",
      privateAccount: "Cuenta privada",
      modelAccess: "Acceso a modelos",
      modelAccessTitle: "Acceso a modelos de {username}",
      modelAccessDescription:
        "Elige qué modelos de Hugging Face puede ver, descargar, cargar y usar esta cuenta mediante sus claves de API.",
      modelAccessLabel: "Modelos permitidos",
      modelAccessHint:
        "Un modelo por línea, por ejemplo unsloth/Qwen3-4B-GGUF, o unsloth/* para toda una organización. Déjalo vacío para permitir todos los modelos.",
      modelAccessSave: "Guardar",
      modelAccessLimited: "Acceso a modelos limitado",
      empty: "Aún no hay otras cuentas",
      setupCode: "Código de configuración",
      setupFor: "Código de configuración para {username}",
      shownOnce:
        "Copia este código ahora y compártelo con el titular de la cuenta. Solo se muestra aquí y puede usarse una vez en un plazo de 60 minutos.",
      expires: "Vence {expiry}",
      copy: "Copiar código de configuración",
      copied: "Copiado",
      copyFailed:
        "No se pudo copiar. Selecciona y copia el código de configuración de arriba.",
      dismiss: "Listo",
      owner: "Propietario de la instalación",
      active: "Activada",
      inactive: "Desactivada",
      regenerate: "Regenerar código de configuración",
      resetTitle: "¿Restablecer la contraseña de {username}?",
      resetDescription:
        "Regenerar el código de configuración reemplaza la contraseña de {username}, cierra sus sesiones y revoca sus claves de API. Dale el código nuevo para que vuelva a elegir una contraseña.",
      deactivate: "Desactivar",
      reactivate: "Activar",
      delete: "Eliminar cuenta",
      deleteTitle: "¿Eliminar {username}?",
      deleteDescription:
        "Esto revoca las sesiones de {username} y cancela su trabajo. Sus chats, configuración, credenciales, archivos subidos, conjuntos de datos, ejecuciones de entrenamiento, resultados, exportaciones, galerías, sandboxes, proyectos y archivos temporales se retiran. Los directorios se renombran y se apartan, nunca se eliminan. Volver a crear este nombre de usuario inicia una cuenta nueva sin ninguno de estos datos.",
      cancel: "Cancelar",
      retry: "Actualizar cuentas",
      failed: "La solicitud de cuenta falló.",
    },
    title: "Configuración",
    dialog: {
      title: "Configuración",
      description: "Administra tus preferencias de Unsloth.",
      closeAriaLabel: "Cerrar configuración",
      searchPlaceholder: "Buscar en la configuración…",
      searchNoResults: "No se encontró ninguna opción.",
      panelFailed: "No se pudo cargar esta sección.",
      panelReload: "Recargar",
    },
    library: {
      layoutSection: "Diseño",
      cardSize: "Tamaño de tarjeta",
      cardSizeDescription: "Cuántas tarjetas caben en una fila.",
      small: "Pequeño",
      medium: "Mediano",
      large: "Grande",
      imageLayout: "Tarjetas de imagen",
      imageLayoutDescription:
        "Conserva la forma de cada imagen o recórtalas en cuadrados iguales.",
      masonry: "Forma original",
      square: "Cuadrada",
      showCardDates: "Mostrar fechas en las tarjetas",
      showCardDatesDescription:
        "Muestra cuándo cambió cada archivo por última vez.",
      browsingSection: "Navegación",
      startTab: "Abrir en",
      startTabDescription: "La pestaña con la que se abre la Biblioteca.",
      suggested: "Sugeridos",
      all: "Todo",
      sort: "Ordenar por",
      sortDescription:
        "Orden de archivos y carpetas. Sugeridos siempre muestra los más recientes.",
      recent: "Más recientes",
      oldest: "Más antiguos",
      name: "Nombre",
      suggestedLimit: "Elementos sugeridos",
      suggestedLimitDescription:
        "Cuántos archivos recientes muestra Sugeridos.",
      tabAuto: "Cuando tengan archivos",
      tabAlways: "Siempre",
      contentSection: "Contenido",
      contentDescription:
        "Elige qué muestra la Biblioteca. Los archivos ocultos no se eliminan.",
      showChatAttachments: "Adjuntos de chat",
      showChatAttachmentsDescription: "Archivos que adjuntaste a los chats.",
      showChatToolFiles: "Archivos de herramientas del chat",
      showChatToolFilesDescription:
        "Archivos que el código y las herramientas crearon en los chats.",
      showGeneratedMedia: "Contenido generado",
      showGeneratedMediaDescription:
        "Imágenes, videos y audio de las páginas Imágenes, Video y Audio.",
      showFineTunesDescription:
        "Modelos que entrenaste o exportaste en Unsloth.",
      deletingSection: "Eliminación",
      confirmDelete: "Confirmar antes de eliminar",
      confirmDeleteDescription:
        "Pregunta antes de eliminar archivos y carpetas de la Biblioteca.",
      reset: "Restablecer la configuración de la Biblioteca",
      resetDescription:
        "Devuelve cada opción anterior a su valor predeterminado.",
      resetButton: "Restablecer",
      resetDone: "Configuración de la Biblioteca restablecida",
      storageSection: "Almacenamiento",
      downloadsSection: "Descargas de imágenes",
      imageInfoStamp: "Sello de información",
      imageInfoStampDescription:
        "Muestra el modelo, la fecha y la semilla en la parte inferior de las imágenes que descargas desde la página Imágenes. La configuración de generación se guarda dentro del archivo de todos modos.",
      storageDescription:
        "Abre una categoría para ver sus archivos de mayor a menor y elimina lo que ya no necesites.",
      storageUsed: "{size} usados por la Biblioteca",
      storageDisk: "{free} libres de {total} en este disco",
      storageEmpty: "Todavía no hay nada guardado en la Biblioteca.",
      storageHidden: "{size} ocultos en la configuración de Contenido",
      storageHiddenLabel: "Oculto",
      storageRefresh: "Actualizar",
      storageOpen: "Abrir",
      storageError: "No se pudo leer el almacenamiento de la Biblioteca.",
      locationsSection: "Archivos en el disco",
      locationsDescription:
        "Dónde guarda Unsloth cada tipo de archivo. Las carpetas de la Biblioteca son etiquetas dentro de Unsloth, no carpetas en el disco. Los modelos ajustados y las exportaciones se quedan donde están, porque el entrenamiento y los chats los encuentran por su ruta.",
      locationUploads: "Archivos subidos a la Biblioteca",
      locationExports: "Exportaciones",
      revealFailed: "No se pudo abrir el administrador de archivos",
      locationReset: "Restablecer",
      locationMoveTitle: "Mover {name} a una carpeta nueva",
      locationMoveDescription:
        "Elige cualquier carpeta. Si ya tiene archivos, Unsloth crea su propia carpeta dentro. Todo lo que ya está aquí se mueve también.",
      locationMoveAction: "Mover aquí",
      locationMoving: "Moviendo {name}…",
      locationMoved: "{name} movido",
      locationMoveFailed: "No se pudo mover {name}",
      locationFree: "{free} libres en esta unidad",
      locationUnavailable:
        "No disponible. Vuelve a conectar su unidad o restablece la carpeta.",
      locationResetUnavailableTitle: "¿Restablecer {name}?",
      locationResetUnavailableDescription:
        "Su unidad no está conectada. Los archivos de {path} se quedan en esa unidad y no aparecerán en la Biblioteca hasta que los vuelvas a mover.",
      locationLeftBehind: "Los archivos de {path} se quedan en esa unidad.",
      dataStorage: "Almacenamiento de la Biblioteca",
      manageStorage: "Administrar almacenamiento",
      categoryImages: "Imágenes",
      categoryVideos: "Videos",
      categoryFineTunes: "Modelos ajustados",
      itemCountOne: "1 elemento",
      itemCount: "{count} elementos",
      tabsSection: "Pestañas",
      tabsDescription:
        "Elige qué pestañas muestra la Biblioteca. Una pestaña abierta desde un enlace sigue apareciendo.",
      tabHidden: "Oculta",
      lastVisited: "Última visitada",
    },
    tabs: {
      sandbox: "Seguridad",
      accounts: "Cuentas",
      general: "General",
      profile: "Perfil",
      appearance: "Apariencia",
      resources: "Sistema",
      chat: "Chat",
      voice: "Voz",
      connections: "Conexiones",
      data: "Datos",
      apiKeys: "API",
      remoteLan: "Remoto y LAN",
      agents: "Agentes",
      keyboardShortcuts: "Atajos",
      debugging: "Registros",
      about: "Acerca de",
    },
    keyboardShortcuts: {
      title: "Atajos de teclado",
      intro:
        "Para cambiar un atajo, selecciona la combinación de teclas y luego presiona las nuevas teclas",
      searchPlaceholder: "Buscar atajos…",
      keystrokePlaceholder: "Presiona un atajo para buscar",
      searchByKeystrokes: "Buscar por teclas",
      searchByName: "Buscar por nombre",
      noResults: "Ningún atajo coincide con esa búsqueda.",
      unassigned: "Sin asignar",
      recording: "Presiona la secuencia de teclas",
      needsModifier:
        "Agrega ⌘, Ctrl o Alt. Una tecla sola capturaría lo que escribes.",
      conflict: "También lo usa otro atajo",
      conflictShadowed:
        "Otro atajo tiene esta combinación y se ejecuta en lugar de este",
      edit: "Cambiar atajo",
      reset: "Restaurar valor predeterminado",
      resetAll: "Restaurar valores predeterminados",
      browserReserved:
        "Es posible que tu navegador se reserve esta combinación. En la aplicación de escritorio sí funciona.",
      metaUnsupported: "La tecla Windows no se puede usar en atajos.",
      altGraphUnsupported:
        "Las combinaciones con AltGr no se pueden usar como atajos.",
      desktopOnly: "Solo en la aplicación de escritorio",
      groups: {
        composer: "Redactor",
        chat: "Chat",
        navigation: "Navegación",
        app: "Aplicación",
      },
      actions: {
        findInPage: {
          label: "Buscar en la página",
          description: "Buscar el texto de esta página",
        },
        newBrowserTab: {
          label: "Nueva pestaña del navegador",
          description: "Abrir una pestaña nueva en el navegador junto al chat",
        },
        toggleBrowserFullView: {
          label: "Alternar vista completa del navegador",
          description:
            "Mostrar el navegador en toda la ventana, con el chat flotando encima",
        },
        openSettings: {
          label: "Abrir configuración",
          description: "Abrir el cuadro de diálogo de configuración",
        },
        openKeyboardShortcuts: {
          label: "Atajos de teclado",
          description: "Abrir esta lista de atajos",
        },
        openCommandPalette: {
          label: "Paleta de comandos",
          description: "Abrir la paleta de comandos",
        },
        searchChats: {
          label: "Buscar chats",
          description: "Abrir el cuadro de diálogo de búsqueda de chats",
        },
        openMcpServers: {
          label: "Servidores MCP",
          description: "Configurar los servidores MCP de este chat",
        },
        logOut: {
          label: "Cerrar sesión",
          description: "Cerrar sesión en Unsloth",
        },
        approveToolRequest: {
          label: "Aprobar solicitud",
          description:
            "Permitir la llamada a herramienta que espera aprobación",
        },
        declineToolRequest: {
          label: "Rechazar solicitud",
          description: "Denegar la llamada a herramienta que espera aprobación",
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
          description:
            "Marcar como no leídos los chats seleccionados o el chat actual",
        },
        togglePinChat: {
          label: "Alternar fijado",
          description:
            "Fijar o dejar de fijar los chats seleccionados o el chat actual",
        },
        selectAllChats: {
          label: "Seleccionar todos los chats",
          description: "Seleccionar todos los chats de la barra lateral",
        },
        clearChatSelection: {
          label: "Borrar selección",
          description:
            "Anular la selección de los chats. Escape también la anula",
        },
        deleteSelectedChats: {
          label: "Eliminar los chats seleccionados",
          description: "Eliminar todos los chats seleccionados",
        },
        nextRecentlyViewedChat: {
          label: "Siguiente chat visto recientemente",
          description: "Avanzar por los chats vistos recientemente",
        },
        previousRecentlyViewedChat: {
          label: "Chat visto recientemente anterior",
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
          label: "Siguiente chat que requiere atención",
          description:
            "Ir al siguiente chat que esté generando, en cola o no leído",
        },
        clearAllUnreads: {
          label: "Marcar todo como leído",
          description: "Marcar todos los chats como leídos",
        },
        goToRecentChat1: {
          label: "Ir al chat reciente 1",
          description: "Abrir el primer chat de Recientes",
        },
        goToRecentChat2: {
          label: "Ir al chat reciente 2",
          description: "Abrir el segundo chat de Recientes",
        },
        goToRecentChat3: {
          label: "Ir al chat reciente 3",
          description: "Abrir el tercer chat de Recientes",
        },
        goToRecentChat4: {
          label: "Ir al chat reciente 4",
          description: "Abrir el cuarto chat de Recientes",
        },
        goToRecentChat5: {
          label: "Ir al chat reciente 5",
          description: "Abrir el quinto chat de Recientes",
        },
        goToRecentChat6: {
          label: "Ir al chat reciente 6",
          description: "Abrir el sexto chat de Recientes",
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
          label: "Ir al Centro de modelos",
          description: "Ir al centro de modelos",
        },
        switchToTrain: {
          label: "Ir a Entrenar",
          description: "Ir al espacio de trabajo de entrenamiento",
        },
        switchToRecipes: {
          label: "Ir a Recetas",
          description: "Ir a Data Recipes",
        },
        switchToImages: {
          label: "Ir a Imágenes",
          description: "Ir al espacio de trabajo de imágenes",
        },
        switchToVideo: {
          label: "Ir a Video",
          description: "Ir al espacio de trabajo de video",
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
          description: "Cambiar a otro proyecto desde el encabezado del chat",
        },
        startDictation: {
          label: "Dictado",
          description: "Iniciar o detener el dictado en el redactor",
        },
        attachFiles: {
          label: "Adjuntar fotos y archivos",
          description: "Agregar un adjunto al redactor",
        },
        openComposerMenu: {
          label: "Abrir menú +",
          description:
            "Abrir el menú + del redactor con herramientas y adjuntos",
        },
        sendMessage: {
          label: "Enviar mensaje",
          description: "Enviar lo que hay en el redactor",
        },
        queueMessage: {
          label: "Poner el mensaje en cola",
          description: "Enviar el borrador al final de la cola de mensajes",
        },
        steerMessage: {
          label: "Redirigir la respuesta",
          description:
            "Detener la respuesta actual y enviar el borrador a continuación",
        },
        stopGeneration: {
          label: "Detener generación",
          description: "Detener la respuesta que se está escribiendo",
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
        renameChat: {
          label: "Renombrar chat",
          description: "Renombrar el chat actual",
        },
        forkChat: {
          label: "Bifurcar chat",
          description:
            "Crear un chat nuevo como rama a partir del último mensaje",
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
    sandbox: {
      title: "Seguridad",
      description:
        "Si las llamadas a herramientas de Python y Terminal se ejecutan dentro de un sandbox del sistema en esta computadora.",
      docs: "Documentación",
      docsLabel: "Abrir la documentación del sandbox",
      toolsSection: "Esta computadora",
      refresh: "Actualizar",
      python: "Python",
      terminal: "Terminal",
      osIsolation: "Sandbox del sistema ({backend})",
      softwareSafeguards: "Protecciones por software",
      noReason:
        "Aquí todavía no funciona ningún sandbox del sistema, así que Unsloth revisa cada llamada.",
      runsInCmd:
        "Se ejecuta en cmd.exe, porque Git Bash no puede iniciarse en el sandbox de Windows.",
      loadError: "No se pudo cargar el estado del sandbox",
      saveError: "No se pudo guardar la opción del sandbox",
      prepareError: "No se pudo preparar esta computadora",
      unsupported:
        "Este servidor de Unsloth aún no informa el estado del sandbox. Actualiza Unsloth para administrarlo aquí.",
      windowsSection: "Sandbox de Windows (versión preliminar)",
      windowsDescription:
        "El sandbox MXC de Microsoft. Usa el contenedor integrado en Windows o, en versiones de Windows sin él, una alternativa.",
      builtinLabel: "Contenedor integrado",
      builtinDescription:
        "El sandbox de Windows se ejecuta en el contenedor integrado de Windows, sin ningún paso de administrador.",
      builtinInUse: "En uso",
      optInLabel: "Usar el sandbox alternativo",
      optInDescription:
        "Para versiones de Windows sin el contenedor integrado. Así, Python y Terminal solo pueden modificar archivos en la carpeta del chat.",
      optInNotNeeded:
        "No hace falta en esta computadora, porque el contenedor integrado funciona.",
      disclosure:
        "La alternativa requiere un paso único de administrador, Preparar esta computadora, que hay que repetir después de cada reinicio. Mientras se ejecuta una herramienta, Unsloth le da acceso temporal a las carpetas que necesita. MXC todavía es una versión preliminar, no un límite de seguridad.",
      lockedDacl: "Definido por UNSLOTH_MXC_ALLOW_DACL_FALLBACK",
      grantsLabel: "Inicio más rápido de herramientas",
      grantsDescription:
        "Mantiene el acceso de solo lectura a las carpetas del entorno de ejecución propias de Unsloth entre llamadas. Si está desactivado, cada llamada tarda unos segundos más en iniciarse.",
      lockedGrants: "Definido por UNSLOTH_MXC_PERSISTENT_READ_GRANTS",
      memoryLabel: "Límite de memoria",
      memoryDescription:
        "Memoria máxima que puede reservar una llamada de Python o Terminal en el sandbox. Se aplica desde la próxima llamada. El valor predeterminado es {defaultSize} GB.",
      memoryLocked: "Definido por UNSLOTH_STUDIO_SANDBOX_AS_GB",
      memoryInvalid: "Ingresa un número entero de {min} a {max}.",
      restored: "Se quitó el acceso a {count} carpetas.",
      hostPrepLabel: "Configuración de administrador",
      prepPrepared: "Listo",
      prepNeeds: "Necesario",
      prepNeedsAgain: "Necesario de nuevo después del último reinicio",
      prepUnknown: "Desconocido",
      prepareButton: "Preparar esta computadora",
      preparing: "Esperando la solicitud de permisos de administrador…",
      prepSucceeded: "Esta computadora está preparada.",
      prepDeclined: "Se rechazó la solicitud de permisos de administrador.",
      prepFailed: "La preparación falló.",
      setupLabel: "Sandbox del sistema",
      setupCommandHint:
        "Ejecuta esto en la computadora donde se ejecuta Unsloth:",
      macosBuiltIn:
        "macOS tiene un sandbox integrado, así que no hay nada que instalar.",
      setupKeywords:
        "instalar configurar configuración install setup bubblewrap bwrap seatbelt mxc sudo pkexec apparmor uac",
      setupSucceeded: "El sandbox del sistema está listo.",
      runtimeMissing:
        "Aún no está instalado. Instalarlo no requiere permisos de administrador.",
      runtimeLabel: "Runtime del sandbox de Windows",
      installRuntime: "Instalar runtime",
      installingRuntime: "Instalando…",
      installRuntimeFailed: "No se pudo instalar el runtime.",
      installRuntimeError: "No se pudo iniciar la instalación del runtime.",
      unsupportedArch:
        "El sandbox de Windows (MXC) solo funciona en Windows x64 y ARM64, así que no está disponible en esta computadora.",
      builtinNeedsUpdate:
        "El contenedor integrado de Windows necesita la compilación {required} o posterior, y esta computadora usa {current}. Instala la última actualización de Windows para usarlo sin un paso de administrador, o activa la alternativa de abajo.",
      unsupportedBuild:
        "El sandbox de Windows (MXC) necesita Windows 11 24H2 (compilación 26100) o posterior, así que no está disponible en esta computadora.",
      permissionLabel: "Permisos de herramientas",
      levelRowLabel: "Nivel de sandbox",
      levelPickerTitle: "¿Cómo debe aislarse el código?",
      levelOff: "Desactivado",
      levelOffShort: "El acceso total desactiva el sandbox",
      levelHighDetail:
        "Python y Terminal se ejecutan dentro del sandbox del sistema (bubblewrap, Seatbelt o MXC) además del sandbox de software. Sin un sandbox del sistema que funcione, las llamadas riesgosas preguntan primero.",
      levelLowDetail:
        "Python y Terminal se ejecutan solo con el sandbox de software, fuera del sandbox del sistema. Las llamadas riesgosas preguntan primero, a menos que hayas elegido Acceso total.",
      levelLabel: "Sandbox",
      levelHigh: "Alto",
      levelLow: "Bajo",
      levelHighShort: "Agrega aislamiento del sistema a Python y Terminal",
      levelLowShort: "Solo software. El código riesgoso pregunta primero",
      levelFullAccessNote:
        "El acceso total desactiva el sandbox. Esta opción vuelve a aplicarse cuando elijas otro nivel.",
      learnMore: "Más información",
      managedNote:
        "El propietario de este servidor de Unsloth configura el sandbox del sistema.",
    },
    debugging: {
      description: "Lee los registros de Unsloth sin salir de la aplicación.",
      logSection: "Archivo de registro",
      source: "Archivo de registro",
      sourceHint:
        "Los ejecutores de modelos escriben sus propios registros, así que un error al cargar o al generar suele explicarse ahí y no en el registro del servidor.",
      path: "Ubicación",
      pathCopy: "Copiar ruta",
      currentSession: "Actual",
      statusLive: "En vivo",
      statusPaused: "En pausa",
      statusStale: "Desactualizado",
      filterPlaceholder: "Filtrar líneas",
      lineCount: "{count} líneas",
      filteredLineCount: "{shown} de {total} líneas",
      wrapLines: "Ajustar líneas",
      jumpToLatest: "Más reciente",
      resizeLog: "Arrastra para cambiar el tamaño del registro",
      noMatches: "Ninguna línea coincide con el filtro.",
      mode: "Modo",
      modeLive: "En vivo",
      modeInterval: "Cada 3 segundos",
      modeManual: "Manual",
      refreshNow: "Actualizar ahora",
      privacyNote:
        "Las credenciales se enmascaran en esta vista. El archivo en disco no está enmascarado.",
      copyVisible: "Copiar el registro visible",
      empty: "Todavía no se registró nada.",
      disabled:
        "El registro en archivo está desactivado (UNSLOTH_STUDIO_NO_FILE_LOG=1).",
      missing: "No se encontró ningún archivo de registro.",
      unreadable: "No se pudo leer el archivo de registro.",
      viewLogs: "Ver registros",
      timeout:
        "Se agotó el tiempo de espera de la solicitud del registro. Es posible que no se pueda acceder al servidor.",
      droppedNotice:
        "Se omitieron algunas líneas: el registro se escribió más rápido de lo que se podía leer.",
      morePending:
        "Aún se están leyendo más líneas; llegarán en la próxima actualización.",
      staleSession:
        "El registro en archivo está desactivado, así que esta es una sesión anterior y no se actualizará.",
      downloadAllLogs: "Descargar todos los registros (.zip)",
      downloadingAllLogs: "Empaquetando los registros...",
      exportMaskedNote:
        "Las credenciales se enmascaran en los archivos exportados. Los registros muy grandes conservan solo sus líneas más recientes y algunos registros pueden quedar fuera por completo; consulta EXPORT_WARNINGS.txt en el archivo comprimido.",
      downloadedTo: "Guardado en {path}",
      downloadedToBrowser: "Se inició la descarga.",
      showInFolder: "Mostrar en la carpeta",
      openLogsFolder: "Abrir la carpeta de registros",
      openLogsFolderFailed: "No se pudo abrir la carpeta de registros.",
      exportFailed: "No se pudieron descargar los registros.",
      exportTooOld:
        "El backend de Unsloth en ejecución es demasiado antiguo para exportar registros. Actualiza ese backend y reinícialo.",
      exportForbidden:
        "Para descargar todos los registros se necesita una sesión de Unsloth iniciada. Una clave de API no es suficiente.",
      keywords:
        "depuración depurar registro registros log logs error errores falla fallo traza traceback stack trace diagnóstico solución de problemas debug",
    },
    voice: {
      title: "Voz",
      description: "Micrófono, dictado, voz a texto y lectura en voz alta",
      dictation: {
        audioUploadTitle: "Dictar con una grabación",
        audioUploadDescription:
          "Tu navegador requiere una conexión segura (HTTPS) para grabar directamente en Unsloth. Con esta conexión HTTP, usa la grabadora de tu teléfono o elige una grabación guardada. Tu servidor de Unsloth transcribirá el audio y agregará el texto a tu mensaje.",
        audioUploadChooseFile: "Elegir grabación",
        audioUploadRecord: "Grabar audio",
        audioUploadIphoneHint:
          "En iPhone, guarda una grabación de Notas de voz o de otra app de grabación y luego elígela aquí.",
        audioUploadServerModelNote:
          "Esta grabación usa el modelo local que se muestra arriba en el servidor de Unsloth, aunque el dictado normal use un motor diferente.",
        audioUploadAutomatic: "Automático",
        audioUploadModelReady: "Listo en este servidor de Unsloth",
        audioUploadRetryTitle: "No se pudo transcribir {file}",
        audioUploadRetry: "Reintentar la transcripción",
        audioUploadChooseModel:
          "Elige un modelo local de reconocimiento de voz en la configuración de Voz.",
        audioUploadEmpty: "La grabación seleccionada está vacía.",
        audioUploadTooLarge: "Elige una grabación de menos de {size}.",
        audioUploadVideoUnsupported:
          "Elige una grabación de audio, no un video.",
        audioUploadNoSpeech: "El modelo no detectó voz en esa grabación.",
        audioUploadFailed: "No se pudo transcribir la grabación.",
        audioUploadCancel: "Cancelar transcripción",
        audioUploadTranscribing: "Transcribiendo audio…",
        sectionTitle: "Dictado",
        engineLabel: "Motor de dictado",
        engineBrowser: "Navegador",
        engineBrowserDescription:
          "Transcribe el audio con el servicio de voz de tu navegador. Selecciona “Transcripción local” para usar un modelo STT.",
        engineModel: "Transcripción local",
        engineModelDescription:
          "Ejecuta un modelo de voz a texto (STT) de forma local y funciona sin conexión. Descárgalo y cárgalo; se descarga de la memoria tras un tiempo de inactividad.",
        engineCustom: "Endpoint personalizado",
        engineCustomDescription:
          "Envía el audio grabado a un servidor STT compatible con OpenAI de tus Conexiones.",
        connectionLabel: "Conexión",
        connectionDescription:
          "Agrega un servidor compatible con OpenAI y una clave de API opcional en Conexiones.",
        connectionPlaceholder: "Selecciona una conexión",
        connectionEmpty: "No hay conexiones disponibles",
        customModelLabel: "Modelo",
        customModelDescription:
          "Nombre del modelo que se envía a /v1/audio/transcriptions.",
        sttModelLabel: "Modelo de reconocimiento de voz",
        sttModelDescription:
          "Elige o busca un modelo STT para ejecutarlo de forma local.",
        sttQuantLabel: "Cuantización",
        sttQuantDescription:
          "Elige o busca un modelo STT para ejecutarlo de forma local y luego su cuantización.",
        sttDeviceLabel: "Cargar en",
        sttDeviceAuto: "GPU cuando esté disponible",
        sttDeviceCpu: "RAM de la CPU",
        sttDeviceAutoDescription:
          "Usar la GPU cuando haya una y, si no, la CPU.",
        sttDeviceCpuDescription:
          "Mantener el modelo en la RAM del sistema. La transcripción es más lenta, pero no se usa memoria de GPU.",
        sttModelSearchPlaceholder: "Buscar cualquier modelo en HF",
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
          "No está instalado en este servidor. Ejecuta `unsloth studio update` para activar el dictado local.",
        sttRetry: "Reintentar",
        sttDownloadChecking: "Comprobando el estado de la descarga…",
        sttNotDownloaded: "Sin descargar",
        sttDownloadStatusFailed:
          "No se pudo comprobar el estado de la descarga",
        sttDownload: "Descargar",
        sttDownloadConfirmTitle: "¿Descargar {model}?",
        sttDownloadConfirmBody:
          "El dictado local funciona totalmente sin conexión, pero antes necesita el modelo de voz {model}. Ocupa unos {size} y se descarga una sola vez en tu caché de Hugging Face.",
        sttDownloadConfirmBodyUnsized:
          "El dictado local funciona totalmente sin conexión, pero antes necesita el modelo de voz {model}. Se descarga una sola vez en tu caché de Hugging Face.",
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
          "Se usa para el dictado. Se usa el predeterminado del sistema si el motor de voz del navegador no puede usar este dispositivo",
        microphoneGrantDescription:
          "Permite el acceso al micrófono para ver los nombres de los dispositivos",
        allowMicrophone: "Permitir el micrófono",
        micAccessBlocked:
          "Se bloqueó el acceso al micrófono. Permite el acceso al micrófono para esta página de Unsloth y vuelve a intentarlo.",
        micAccessBlockedDesktop:
          "Se bloqueó el acceso al micrófono. Vuelve a intentarlo y elige Permitir, o activa el micrófono en la configuración de privacidad del sistema.",
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
          "Define cómo escribe el dictado determinadas palabras o frases",
        manageLabel: "Grafías personalizadas",
        manage: "Administrar",
        backToVoice: "Volver a Voz",
        addEntry: "Agregar entrada",
        newEntryAria: "Nueva entrada del diccionario",
        entryPlaceholder: "María García",
        entryAria: "Entrada {index} del diccionario",
        removeEntryAria: "Quitar la entrada {index} del diccionario",
      },
      recents: {
        sectionTitle: "Historial de dictados",
        sectionDescription:
          "Cada dictado se guarda aquí para que puedas recuperar el texto",
        manageLabel: "Historial de dictados",
        manage: "Administrar",
        pageDescription:
          "Todos los dictados se guardan. Consúltalos, cópialos o elimínalos, o abre el chat en el que se usó un dictado.",
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
        autoLabel: "Leer las respuestas automáticamente",
        autoDescription: "Lee en voz alta cada respuesta en cuanto termina",
        engineLabel: "Motor de TTS",
        engineSystemDescription: "Voces integradas del dispositivo",
        engineStudioDescription:
          "Habla con un modelo cargado en Audio › Texto a voz",
        engineSystem: "Voces del sistema",
        engineStudio: "Modelo de Unsloth",
        engineCustom: "Endpoint personalizado",
        engineCustomDescription:
          "Un servidor TTS compatible con OpenAI de tus conexiones (p. ej., Kokoro)",
        connectionLabel: "Conexión",
        connectionDescription:
          "Agrega un servidor compatible con OpenAI en la pestaña Conexiones",
        connectionPlaceholder: "Selecciona una conexión",
        customModelLabel: "Modelo",
        customVoiceDescription:
          "Nombre de la voz que espera el endpoint; el valor predeterminado es alloy",
        modelLabel: "Modelo de TTS",
        modelDescription:
          "No hay ninguno cargado. Carga uno en Audio › Texto a voz.",
        modelLoaded: "{model} está cargado",
        keepBothLabel: "Mantener cargado el modelo del chat",
        keepBothDescription:
          "“Mantener varios modelos cargados” está desactivado, así que cargar un modelo de voz descarga de la memoria el modelo del chat.",
        keepBothAction: "Mantener ambos cargados",
        openAudioAction: "Abrir Audio",
        voiceLabel: "Voz",
        voiceDescription: "Las mejores voces de este dispositivo",
        studioVoiceDescription:
          "Las voces guardadas en Audio requieren un modelo de clonación",
        studioVoiceDefault: "Voz propia del modelo",
        studioVoiceSaved: "Voz guardada",
        speedLabel: "Velocidad",
        pitchLabel: "Tono",
        volumeLabel: "Volumen",
        previewLabel: "Escuchar la voz",
        previewDescription: "Reproducir una muestra corta",
        previewFailed: "No se pudo reproducir la muestra de TTS",
        previewAction: "Escuchar",
        preparingAction: "Generando…",
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
      checkingToken: "Comprobando el token...",
      tokenValidated: "Token validado",
      password: "Contraseña",
      passwordDescription: "Cambia la contraseña de esta cuenta de Unsloth.",
      passwordDialog: {
        trigger: "Cambiar contraseña",
        title: "Cambiar contraseña",
        description:
          "Ingresa tu contraseña actual y elige una nueva (al menos {minLength} caracteres).",
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
        newHasSpaces: "La contraseña nueva no puede contener espacios.",
        mismatch: "Las contraseñas no coinciden.",
        samePassword: "La contraseña nueva debe ser distinta de la actual.",
        update: "Actualizar contraseña",
        updating: "Actualizando...",
        updated: "Contraseña actualizada.",
        updateFailed: "No se pudo actualizar la contraseña.",
      },
      chatDefaults: "Valores predeterminados del chat",
      autoTitleNewChats: "Poner título automático a los chats nuevos",
      autoTitleNewChatsDescription:
        "Genera un título a partir del primer mensaje.",
      helperLlm: {
        sectionTitle: "LLM auxiliar",
        preloadOnStartup: "Precargar el LLM auxiliar en caché al iniciar",
        preloadOnStartupDescription:
          "Descarga en segundo plano el modelo auxiliar de AI Assist al iniciar. Desactivado de forma predeterminada; AI Assist puede obtenerlo igualmente cuando lo necesite.",
        disabledByEnv:
          "Desactivado por UNSLOTH_HELPER_MODEL_DISABLE en el entorno del backend.",
        loadError: "No se pudo cargar la configuración del LLM auxiliar.",
        saveError: "No se pudo guardar la configuración del LLM auxiliar.",
      },
      developerMode: {
        sectionTitle: "Modo desarrollador",
        videoCpu: "Permitir la generación de video en CPU (experimental)",
        videoCpuDescription:
          "Permite usar la página de Video en un equipo sin GPU compatible. Desactivado de forma predeterminada; no tiene soporte y es muy lento.",
        notApplicable:
          "No tiene efecto en esta computadora: el video ya se ejecuta en su GPU o está bloqueado por algo que la CPU no puede sortear.",
        forcedOnByEnv:
          "Activado por UNSLOTH_VIDEO_ALLOW_CPU en el entorno del backend.",
        forcedOffByEnv:
          "Desactivado por UNSLOTH_VIDEO_ALLOW_CPU en el entorno del backend.",
        confirmTitle: "¿Activar la generación de video en CPU?",
        confirmWarning:
          "La generación de video en CPU puede tardar horas, consumir grandes cantidades de RAM y congelar tu sistema. Continúa bajo tu propio riesgo.",
        ramBelowMinimum:
          "Esta computadora tiene {host} GB de RAM, menos de los {min} GB que necesita el modelo de video más pequeño en CPU. Es probable que la carga falle o congele el sistema.",
        ramOk:
          "Esta computadora tiene {host} GB de RAM. El modelo de video más pequeño necesita unos {min} GB en CPU; los modelos más grandes necesitan mucho más.",
        ramUnknown:
          "El modelo de video más pequeño necesita unos {min} GB de RAM en CPU; los modelos más grandes necesitan mucho más.",
        performanceEstimate:
          "Espera minutos por cada paso de eliminación de ruido: un clip de 5 fotogramas a 832x480 de Wan2.1 1.3B con 3 pasos tardó unos 4 minutos en 16 hilos de CPU, y un clip de duración completa con la configuración predeterminada puede tardar horas.",
        confirmAction: "Activar de todos modos",
        loadError: "No se pudo cargar la configuración del modo desarrollador.",
        saveError:
          "No se pudo guardar la configuración del modo desarrollador.",
      },
      modelAutoSwitch: {
        sectionTitle: "Cambio automático de modelo (API de OpenAI)",
        enable: "Cambiar de modelo según la solicitud",
        enableDescription:
          "Carga un modelo descargado que se indique en una solicitud de la API antes de responder. Desactivado de forma predeterminada.",
        autoDownload: "Descargar los modelos que falten",
        autoDownloadDescription:
          "Descarga un GGUF que se indique en una solicitud de la API y que aún no esté descargado. Así, cualquiera con una clave de API puede consumir disco y ancho de banda.",
        idleUnload: "Liberar automáticamente por inactividad",
        idleUnloadDescription:
          "Libera la VRAM después de esta cantidad de segundos de inactividad. El valor 0 mantiene el modelo cargado; el mínimo es 60.",
        idleSecondsAriaLabel:
          "Segundos de inactividad antes de liberar el modelo",
        mediaEnable: "Cambiar de modelo de imagen y video según la solicitud",
        mediaEnableDescription:
          "Carga un modelo de imagen o video descargado que se indique en una solicitud de la API antes de generar. Es independiente del cambio de modelo de chat. Desactivado de forma predeterminada.",
        mediaIdleUnload:
          "Liberar imagen y video automáticamente por inactividad",
        mediaIdleUnloadDescription:
          "Libera la VRAM descargando de la memoria los modelos de imagen y video después de esta cantidad de segundos de inactividad. Es independiente del cambio de modelo de chat. El valor 0 los mantiene cargados; el mínimo es 60.",
        mediaIdleSecondsAriaLabel:
          "Segundos de inactividad antes de liberar los modelos de imagen y video",
        mediaIdlePaused:
          "En pausa mientras “Mantener el modelo en la memoria de GPU” está activado.",
        idleNeedsEnable:
          "Primero activa “Cambiar de modelo según la solicitud”.",
        idleActiveViaEnv: "Activo mediante UNSLOTH_MODEL_IDLE_TTL.",
        loadError:
          "No se pudo cargar la configuración de cambio automático de modelo.",
        saveError:
          "No se pudo guardar la configuración de cambio automático de modelo.",
        idleError:
          "Ingresa 0 para mantener el modelo cargado, o al menos 60 segundos.",
        keepKv: "Conservar el contexto del chat entre descargas de memoria",
        keepKvDescription:
          "Guarda la caché KV antes de liberar el modelo por inactividad, o antes de que un modelo de imagen o video ocupe la GPU, para que los chats reanudados no tengan que volver a leer el historial. Hasta 10 GB en disco.",
        keepKvKeywords:
          "kv cache caché contexto cambio swap reanudar imagen video",
        apiOnly: "Liberar solo los modelos cargados por la API",
        apiOnlyDescription:
          "La liberación por inactividad mantiene en memoria el modelo que cargaste desde Unsloth y solo libera los que cargó una solicitud de la API.",
      },
      modelAliases: {
        sectionTitle: "Alias de modelos (API de OpenAI)",
        description:
          "Nombres estables para clientes de la API, como fast-chat o code. Una solicitud que nombra un alias la atiende el modelo al que apunta, así las apps conectadas siguen funcionando cuando cambias de modelo. Activa “Cambiar de modelo según la solicitud” para que un alias pueda cargar su modelo.",
        namePlaceholder: "fast-chat",
        nameAriaLabel: "Nombre del alias",
        modelAriaLabel: "Modelo al que apunta el alias",
        modelPlaceholder: "unsloth/Qwen3-4B-GGUF:Q4_K_M",
        removeAriaLabel: "Quitar alias",
        add: "Agregar alias",
        loadError: "No se pudieron cargar los alias de modelos.",
        saveError: "No se pudieron guardar los alias de modelos.",
      },
      previewSharing: {
        sectionTitle: "Compartir vista previa",
        enableLabel: "Enlaces públicos de vista previa",
        enableDescription:
          "Permite que cualquiera con un enlace firmado chatee con un modelo terminado, sin necesidad de iniciar sesión. Desactívalo para desconectar la vista previa pública; los enlaces compartidos dejan de funcionar.",
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
          "Todos los enlaces de vista previa que hayas compartido dejarán de funcionar de inmediato. Esta acción no se puede deshacer.",
        revokeConfirmAction: "Revocar todos los enlaces",
        revoked: "Se revocaron todos los enlaces de vista previa",
        revokeError: "No se pudieron revocar los enlaces de vista previa",
      },
      managedProviderUrls: {
        sectionTitle: "Cuentas administradas",
        enableLabel: "Conexiones locales y de red",
        enableDescription:
          "Permite que las cuentas administradas apunten sus conexiones a direcciones locales o de red, como un servidor Ollama o llama.cpp en esta computadora o en tu LAN. Desactivado de forma predeterminada, porque permite que esas cuentas accedan a servicios que se ejecutan en tu computadora y en tu red.",
        lockedByEnvironment:
          "Definido por UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1 en este servidor, que rechaza las direcciones privadas para todas las cuentas.",
        loadError:
          "No se pudo cargar la configuración de conexiones de las cuentas administradas.",
        saveError:
          "No se pudo guardar la configuración de conexiones de las cuentas administradas.",
      },
      permissions: {
        sectionTitle: "Permisos",
        names: {
          ask: "Pedir aprobación",
          auto: "Aprobar por mí",
          off: "Ejecutar automáticamente",
          full: "Acceso total",
        },
        details: {
          ask: "Pregunta antes de cada llamada a herramienta, incluido el código de terminal y Python, las búsquedas web, la edición de archivos y las herramientas MCP. Las herramientas que ejecuta un proveedor externo no se pausan.",
          auto: "Ejecuta por sí solo las llamadas a herramientas habituales y solo pregunta cuando una acción parece riesgosa, como leer credenciales, elevar privilegios o ejecutar comandos destructivos.",
          off: "Ejecuta todas las llamadas a herramientas sin preguntar. El código de Python y terminal sigue ejecutándose en el sandbox, que limita a qué archivos de tu computadora puede acceder. Sin un sandbox del sistema que funcione, las llamadas riesgosas preguntan primero.",
          full: "Ejecuta todas las llamadas a herramientas sin preguntar y desactiva el sandbox de código, así que el código de Python y terminal puede leer y cambiar cualquier archivo al que tenga acceso tu cuenta.",
        },
      },
      notifications: {
        sectionTitle: "Notificaciones",
        showLoadedModels: "Tarjeta de modelos cargados",
        showLoadedModelsDescription:
          "Una tarjeta en la esquina con cada modelo cargado, para descargarlo de la memoria con un clic.",
        updateReminders: "Recordatorios de actualización",
        updateRemindersDescription:
          "Con qué frecuencia ofrecer una versión nueva cuando esté lista.",
        llamaHint: "Modelos de chat y GGUF",
        audioHint: "Voz, música y transcripción",
        frequency: {
          always: "Siempre que haya una",
          daily: "Una vez al día",
          weekly: "Una vez a la semana",
          biweekly: "Cada dos semanas",
          monthly: "Una vez al mes",
          off: "Desactivado",
        },
      },
      startup: {
        sectionTitle: "Inicio",
        launchAtLogin: "Ejecutar Unsloth al iniciar sesión",
        launchAtLoginDescription:
          "Inicia Unsloth en segundo plano cuando inicias sesión. Permanece en la barra de menús o en la bandeja del sistema hasta que lo abras.",
        closeToTray: "Cerrar en la bandeja del sistema",
        closeToTrayDescription:
          "Mantiene Unsloth y su servidor ejecutándose en segundo plano cuando cierras la ventana principal.",
        closeToTraySaveError:
          "No se pudo actualizar la opción de cerrar en la bandeja del sistema.",
        menuBarIcon: "Mostrar en la barra de menús",
        menuBarIconDescription:
          "Muestra el ícono de Unsloth en la barra de menús de macOS. Cuando está oculto, Unsloth permanece en el Dock, incluso cuando se abre al iniciar sesión.",
        menuBarIconSaveError:
          "No se pudo actualizar la opción del ícono de la barra de menús.",
        loadError: "No se pudo cargar la opción de ejecutar al iniciar sesión.",
        saveError:
          "No se pudo actualizar la opción de ejecutar al iniciar sesión.",
      },
      hub: {
        source: "Origen de los modelos",
        sourceDescription:
          "Dónde busca y descarga el hub de modelos. Elige ModelScope si Hugging Face está bloqueado o es lento en tu red.",
        sourceFallback:
          "No se pudo iniciar ModelScope, así que se usa Hugging Face. Revisa los registros de Unsloth.",
        autoSourceTitle: "Se cambió el origen de los modelos a ModelScope",
        autoSourceDescription:
          "Hugging Face suele ser lento o estar bloqueado en tu región, así que Unsloth ahora descarga los modelos de ModelScope.",
        autoSourceAction: "Abrir la configuración del origen de los modelos",
        sectionTitle: "Hub de modelos",
        endpoint: "Endpoint de Hugging Face",
        endpointDescription:
          "De dónde se descargan los modelos y conjuntos de datos. Déjalo vacío para usar huggingface.co o ingresa un espejo como https://hf-mirror.com.",
        datasetsServer: "Usarlo para las vistas previas de conjuntos de datos",
        datasetsServerDescription:
          "Envía también las solicitudes de vista previa de conjuntos de datos (HF_DATASETS_SERVER) a este endpoint. Actívalo solo si el espejo las atiende.",
        invalidEndpoint:
          "Ingresa una URL http(s) sin credenciales ni parámetros de consulta. http sin cifrar solo funciona con una dirección local.",
        saveFailed: "No se pudo guardar la configuración del hub.",
        loadFailed: "No se pudo cargar la configuración del hub.",
      },
      downloads: {
        sectionTitle: "Descargas",
        transport: "Transporte de descarga",
        transportDescription:
          "Cómo llegan los archivos de modelos y conjuntos de datos desde Hugging Face. HTTPS retoma donde se detuvo; Xet suele ser más rápido en la primera descarga, pero reinicia el archivo si cancelas.",
        transportHint:
          "HTTPS es TLS normal: cualquier red, proxy o VPN lo permite, una transferencia cancelada o cortada continúa desde los bytes ya guardados en disco y el uso de memoria se mantiene estable. Xet descarga bloques deduplicados, así que un repositorio que comparte datos con otro que ya tienes puede llegar mucho más rápido, pero necesita hf_xet, usa más RAM y una cancelación descarta el archivo en curso. Auto decide según esta computadora: tiene en cuenta la RAM y si Xet se ha estado atascando aquí, y recurre a HTTPS.",
        https: "HTTPS",
        xet: "Xet",
        auto: "Auto",
        httpsHint:
          "TLS estándar. Se reanuda después de cancelar, funciona en cualquier red y el uso de memoria es estable.",
        transportDescriptionNoResume:
          "Cómo llegan los archivos de modelos y conjuntos de datos desde Hugging Face. En esta instalación ningún transporte puede reanudarse, así que una descarga cancelada vuelve a empezar; Xet suele ser más rápido en la primera descarga.",
        httpsHintNoResume:
          "TLS estándar. Funciona en cualquier red y el uso de memoria es estable. Esta instalación no puede reanudar una descarga cancelada.",
        xetHint:
          "Transferencia por bloques deduplicados. Suele ser más rápida en una descarga nueva, reinicia el archivo si cancelas y necesita más memoria.",
        autoHint:
          "Elige según la computadora y cambia a HTTPS si Xet se atasca o falla aquí.",
        autoCurrently: "Auto está usando {transport} en esta computadora.",
        xetMissing: "Xet no está disponible porque hf_xet no está instalado.",
      },
      uploads: {
        sectionTitle: "Subidas",
        maxUploadSize:
          "Límite de subida de conjuntos de datos de entrenamiento",
        maxUploadSizeDescription:
          "El valor predeterminado es {defaultSize} MB.",
      },
      rag: {
        sectionTitle: "Documentos y RAG",
        embeddingModel: "Modelo de embeddings",
        embeddingModelShort:
          "Convierte tus documentos en vectores para que el chat pueda buscar en ellos.",
        embeddingModelDescription:
          "Modelo de Hugging Face o ruta local que se usa para indexar y buscar tus documentos. El valor predeterminado es {defaultModel}.",
        searchPlaceholder: "Buscar cualquier modelo en HF",
        reindexWarning:
          "Solo afecta a los documentos que se indexen a partir de ahora. Vuelve a generar los embeddings de los existentes después de cambiar el modelo.",
        reembed: {
          label: "Volver a generar embeddings de documentos",
          staleOne:
            "1 documento se indexó con otro modelo de embeddings, así que la búsqueda semántica lo omite.",
          staleMany:
            "{count} documentos se indexaron con otro modelo de embeddings, así que la búsqueda semántica los omite.",
          staleArchived:
            "El historial de chat anterior se indexó con otro modelo de embeddings, así que la búsqueda semántica lo omite.",
          running: "Volviendo a generar embeddings: {done} de {total}",
          action: "Volver a generar",
          stop: "Detener",
          startFailed: "No se pudo iniciar la regeneración de embeddings",
          stopped: "Se detuvo la regeneración de embeddings",
          finished: "Se volvieron a generar los embeddings de los documentos",
          finishedWithFailures:
            "No se pudieron volver a generar los embeddings de {count}",
        },
        emptyError: "Ingresa un ID de modelo de Hugging Face o una ruta local.",
        loadError:
          "No se pudo cargar la configuración del modelo de embeddings.",
        saveError: "No se pudo guardar el modelo de embeddings.",
        saved: "Modelo de embeddings guardado.",
        saveAnyway: "Guardar de todos modos",
        recommended: "Recomendado",
        onDevice: "En el dispositivo",
        searching: "Buscando en Hugging Face…",
        checking: "Comprobando…",
        noResults: "No se encontraron modelos de embeddings",
        download: "Descargar",
        unload: "Expulsar",
        unloadFailed: "No se pudo expulsar el modelo de embeddings",
        ejected: "Se expulsó el modelo de embeddings",
        ejectModel: "Expulsar modelo",
        onDeviceHint:
          "Descargado en esta computadora. Solo se carga en memoria mientras se indexan o buscan documentos, así que no usa RAM ni VRAM hasta entonces.",
        loadedHint:
          "Está en memoria ahora y usa RAM o VRAM. Pasa el cursor sobre el selector de modelo y haz clic en el ícono rojo para expulsarlo.",
        notLoaded: "No cargado",
        notLoadedHint:
          "No está en memoria ahora. Se carga solo la próxima vez que se indexen o busquen documentos, y luego puedes expulsarlo desde el selector.",
        downloadingStatus: "Descargando…",
        notDownloaded: "No descargado",
        notDownloadedSized: "No descargado · {size}",
        loaded: "Cargado",
        downloading: "Descargando {model}",
        downloadingDescription:
          "El progreso aparece en el panel de descargas. La indexación lo usará cuando termine de descargarse.",
        downloadFailed: "No se pudo iniciar la descarga",
        downloadConflict: "Reanuda esta descarga desde el Hub",
        downloadBusy: "La descarga ya está en curso",
        pin: "Fijar en el menú Archivos",
        unpin: "Quitar del menú Archivos",
        menuChip: "Modelo",
        moreModels: "Más modelos",
        pinMoreModels: "Fijar más modelos",
        defaultTag: "Predeterminado",
        localModel: "Modelo local",
        switched: "Modelo de embeddings cambiado a {model}",
        switchedNeedsDownload: "{model} aún no está descargado",
        switchedNeedsDownloadDescription:
          "Descárgalo en Configuración antes de indexar archivos nuevos.",
        switchedUncheckedDescription:
          "No se pudo comprobar si está descargado. Revísalo en Configuración antes de indexar archivos nuevos.",
        openSettings: "Abrir Configuración",
        switchFailed: "No se pudo cambiar el modelo de embeddings",
        back: "Volver a las fuentes",
        menuSubtitle: "Permite buscar en archivos",
        device: "Dispositivo de embeddings",
        deviceDescription:
          "Dónde se ejecuta el modelo de embeddings, para todos los modelos. La GPU indexa grandes conjuntos de documentos más rápido, pero mantiene algo de VRAM en uso hasta que Unsloth se reinicie.",
        deviceAuto: "Automático",
        deviceCpu: "CPU",
        deviceGpu: "GPU",
        deviceLocked: "Definido por {name}.",
        deviceSaveError: "No se pudo cambiar el dispositivo de embeddings",
      },
      storage: {
        sectionTitle: "Almacenamiento",
        modelsFolder: "Carpeta de modelos",
        modelsFolderDescription: "Dónde se almacenan los modelos descargados.",
        openAction: "Abrir",
        copyAction: "Copiar ruta",
        copied: "Ruta copiada",
        openError: "No se pudo abrir la carpeta",
        copyError: "No se pudo copiar la ruta",
      },
      fullInstall: {
        label: "Tipo de instalación",
        description:
          "Solo chat (GGUF). La instalación completa agrega entrenamiento, exportación, video, modelos safetensors y más funciones de generación de imágenes.",
        action: "Actualizar a la instalación completa",
        confirmTitle: "¿Actualizar a la instalación completa?",
        confirmDescription:
          "Esta instalación de solo chat ejecuta modelos GGUF. La instalación completa agrega PyTorch para la GPU de esta computadora, lo que habilita el entrenamiento, la exportación, el video, los modelos safetensors y más funciones de generación de imágenes. Descarga varios gigabytes (al menos 5 GB de espacio en disco) y reinicia el servidor. Se conservan los chats, los modelos y la configuración. Puede tardar varios minutos.",
        confirmAction: "Actualizar",
        keywords:
          "full install upgrade tipo de instalación instalación completa actualizar solo chat chat only gguf entrenamiento entrenar train training fine-tune ajuste fino pytorch torch exportar export video generación de imágenes safetensors",
      },
      repairInstall: {
        label: "Reparar la instalación",
        description:
          "Vuelve a ejecutar el instalador sobre el entorno administrado. Úsalo si no se detecta la GPU o si la aplicación no se inicia.",
        action: "Reparar instalación",
        confirmTitle: "¿Reparar esta instalación?",
        confirmDescription:
          "Detiene el servidor y vuelve a ejecutar el instalador, que reinstala PyTorch para la GPU de esta computadora en la versión compatible más reciente. Se conservan los chats y la configuración. Puede tardar varios minutos.",
        confirmDescriptionChatOnly:
          "Detiene el servidor y vuelve a ejecutar el instalador. Esta es una instalación de solo chat, así que el instalador también la actualiza a la instalación completa con PyTorch, varios gigabytes en total. Se conservan los chats y la configuración. Puede tardar varios minutos.",
        confirmAction: "Reparar ahora",
      },
      resetPreferences: {
        sectionTitle: "Zona de peligro",
        label: "Restablecer todas las preferencias locales",
        description:
          "Borra las preferencias que solo se guardan localmente. Se conservan los chats, el acceso a la API y la configuración almacenada en la base de datos.",
        action: "Restablecer preferencias",
        confirmTitle: "¿Restablecer todas las preferencias locales?",
        confirmDescription:
          "Borra las preferencias que solo se guardan localmente y vuelve a cargar Unsloth. Se conservan los chats, el acceso a la API y la configuración almacenada en la base de datos.",
        confirmAction: "Restablecer y volver a cargar",
      },
    },
    profile: {
      title: "Perfil",
      description: "Cómo aparece tu perfil en Unsloth.",
      changePicture: "Cambiar foto de perfil",
      uploadPhoto: "Subir foto",
      removePhoto: "Quitar",
      pictureOptions: "Opciones de la foto de perfil",
      displayName: "Nombre visible",
      nickname: "¿Cómo debería llamarte Unsloth?",
      nicknamePlaceholder: "Apodo",
      nicknameSaved: "Nombre preferido guardado",
      avatarShape: "Forma del avatar",
      avatarShapeCircle: "Círculo",
      avatarShapeRounded: "Redondeada",
      greetingSloth: "Saludo",
      greetingSlothDescription: "Mostrar el saludo en los chats nuevos.",
      chooseSloth: "O elige un perezoso",
      noPicture: "Sin foto de perfil",
      noneLabel: "Ninguno",
      nameSaved: "Nombre de perfil guardado",
      namePersistErrorTitle: "No se pudo guardar el nombre de perfil",
      namePersistErrorDescription:
        "El nombre se actualizó para esta sesión, pero podría no conservarse al volver a cargar.",
      photoUpdated: "Foto de perfil actualizada",
      photoPersistErrorTitle: "No se pudo guardar la foto de perfil",
      photoPersistErrorDescription:
        "La foto se actualizó para esta sesión, pero podría no conservarse al volver a cargar.",
      photoUpdateErrorTitle: "No se pudo actualizar la foto de perfil",
      imageUseError: "No se pudo usar esta imagen.",
      stats: {
        title: "Tus estadísticas",
        subtitle:
          "Todo lo que aparece a continuación se calcula a partir de tu propio historial. No se recopila ni se envía nada a Unsloth.",
        retry: "Volver a intentar",
        exportCsv: "Exportar CSV",
        exportError: "No se pudo exportar tu uso de tokens",
        privacyNote:
          "Las estadísticas se calculan a partir del historial local de chats, uso de la API y entrenamientos que guarda tu instalación de Unsloth. Nunca se guardan prompts, respuestas ni claves de la API para las estadísticas. No se envía nada a Unsloth ni a terceros.",
        emptyChats:
          "Todavía no hay uso de chats ni de la API. Empieza una conversación o haz una solicitud autenticada a la API local, y tus estadísticas aparecerán aquí.",
        lifetimeTokens: "Tokens acumulados",
        peakTokens: "Día más activo",
        longestChat: "Chat más largo",
        currentStreak: "Racha actual",
        longestStreak: "Racha más larga",
        activityTitle: "Actividad de tokens",
        activityDescription: {
          daily: "{total} desde el {since}",
          weekly: "Semana más activa {total} · desde el {since}",
          cumulative: "{total} en total desde el {since}",
        },
        mode: {
          daily: "Diaria",
          weekly: "Semanal",
          cumulative: "Acumulada",
        },
        cellTooltip: "{date} · {tokens}, {messages}",
        emptyDayTooltip: "{date} · Sin actividad",
        apiRequests: "{count} vía API",
        weekTooltip: "Semana del {date} · {tokens}",
        cumulativeTooltip: "Hasta la semana del {date} · {tokens} en total",
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
        avgTokensPerChat: "Promedio de tokens por chat",
        timeInChat: "Tiempo en chats",
        activeDays: "Días activos",
        toolCalls: "Llamadas a herramientas",
        attachments: "Archivos adjuntos",
        avgSpeed: "Velocidad promedio",
        bestSpeed: "Respuesta más rápida",
        firstToken: "Tiempo promedio hasta el primer token",
        tokensPerSecond: "{value} tok/s",
        topModelsTitle: "Modelos más usados",
        topModelsDescription: "Ordenados por tokens intercambiados",
        modelSummary: "{tokens} · {messages}",
        noModels: "Todavía no se registró uso de modelos.",
        trainingTitle: "Entrenamiento",
        trainingDescription:
          "Ejecuciones de ajuste fino de este espacio de trabajo",
        trainingRuns: "Ejecuciones",
        trainingCompleted: "Completadas",
        trainingSteps: "Pasos",
        trainingTokens: "Tokens entrenados",
        trainingTime: "Tiempo de entrenamiento",
        bestLoss: "Mejor pérdida",
        runSteps: "{steps}",
        runLoss: "pérdida {loss}",
      },
    },
    appearance: {
      title: "Apariencia",
      description: "Cómo se ve Unsloth en este dispositivo.",
      gpuRendering: {
        sectionTitle: "Renderizado",
        label: "Renderizado por GPU",
        description:
          "Dibuja Unsloth con tu tarjeta gráfica. Auto la usa donde se sabe que funciona. Los cambios se aplican cuando Unsloth se reinicia.",
        auto: "Auto",
        on: "Activado",
        off: "Desactivado",
        oldWebkitWarning:
          "WebKitGTK {version} en Wayland: el renderizado por GPU puede fallar al iniciar en NVIDIA con escalado fraccionario (WebKit bug 280210). Si ocurre, Unsloth vuelve por sí solo al renderizado por CPU. En WebKitGTK 2.50.4 no mostró ninguna mejora medible (8 fps en ambos casos), así que no se recomienda Activado aquí.",
        oldDriverWarning:
          "Controlador NVIDIA {version} en X11: Unsloth mantiene desactivado el renderizado por GPU hasta que el controlador sea 610 o posterior. Los controladores anteriores pierden un descriptor de archivo por fotograma y la ventana deja de dibujarse en pocos minutos.",
        fallbackNote:
          "El renderizado por GPU falló al iniciar en este sistema, así que Unsloth usa el renderizado por CPU.",
        tryAgain: "Volver a probar la GPU",
        envOverride:
          "Definido por la variable de entorno {name}. Esta opción no tiene efecto hasta que se elimine.",
        macNote:
          "macOS siempre renderiza con la GPU. WebKit no ofrece ninguna opción compatible para desactivarlo.",
        restartPending: "Reinicia Unsloth para aplicarlo.",
        restartNow: "Reiniciar ahora",
        loadError: "No se pudo cargar la opción de renderizado por GPU.",
        saveError: "No se pudo guardar la opción de renderizado por GPU.",
        restartError: "No se pudo reiniciar Unsloth.",
      },
      theme: {
        title: "Estilo visual",
        label: "Modo",
        description: "Claro, oscuro o según tu sistema.",
        system: "Sistema",
        light: "Claro",
        dark: "Oscuro",
      },
      palette: {
        label: "Tema",
        description: "Colores que se usan en Unsloth, en modo claro y oscuro.",
        standard: "Estándar",
        classic: "Clásico",
        minimal: "Minimalista",
        moreThemes: "Más temas",
        halloween: "Halloween",
      },
      trickOrTreat: {
        label: "Dulce o truco",
        description:
          "Temas de Halloween y saludos terroríficos, hasta el 1 de noviembre.",
      },
      custom: {
        chatWidth: {
          label: "Ancho del chat",
          description: "Ancho de los mensajes y del cuadro de redacción.",
          standard: "Estándar",
          wide: "Amplio",
          full: "Ancho completo",
        },
        sentAttachments: {
          label: "Adjuntos en los mensajes enviados",
          description:
            "Estándar muestra cada archivo con su tipo; Compacto muestra más en cada línea. Automático cambia a Compacto con más de seis archivos.",
          auto: "Automático",
          list: "Estándar",
          chips: "Compacto",
        },
        reset: "Restablecer",
        resetAll: "Restablecer la personalización",
        preferencesTitle: "Preferencias",
        colors: {
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
            "Tipo de archivo no compatible. Usa .woff2, .woff, .ttf o .otf.",
          errorTooLarge:
            "El archivo de fuente es demasiado grande (máx. 1.5 MB).",
          errorLimit: "Puedes importar hasta 3 fuentes.",
          errorStorageFull:
            "No hay suficiente almacenamiento local para esta fuente. Primero quita una fuente importada.",
          errorFailed: "No se pudo cargar este archivo de fuente.",
        },
        interfaceScale: {
          label: "Escala de la interfaz",
          description:
            "Cambia el tamaño de todos los elementos de la interfaz.",
        },
        uiFontSize: {
          label: "Tamaño de fuente de la interfaz",
          description:
            "Ajusta el tamaño base que se usa en la interfaz de Unsloth.",
        },
        codeFontSize: {
          label: "Tamaño de fuente del código",
          description: "Ajusta el tamaño base que se usa para el código.",
        },
        fontSmoothing: {
          label: "Suavizado de fuentes",
          description: "Usa antialiasing para suavizar las fuentes.",
        },
        contrast: {
          label: "Contraste",
        },
        reduceMotion: {
          label: "Reducir el movimiento",
          description:
            "Reduce las animaciones o sigue la configuración del sistema.",
          system: "Sistema",
          on: "Activado",
          off: "Desactivado",
        },
        pointerCursors: {
          label: "Usar cursores de puntero",
          description:
            "Cambia el cursor a un puntero al pasar sobre elementos interactivos.",
        },
      },
      language: {
        title: "Idioma",
        label: "Idioma de la interfaz",
        description: "El idioma que usa Unsloth.",
        autoDetect: "Detección automática",
        search: "Buscar idiomas",
        noResults: "No hay idiomas que coincidan",
        spellCheck: "Corrector ortográfico",
        spellCheckDescription:
          "Subraya las palabras mal escritas mientras escribes.",
      },
      layout: {
        title: "Diseño",
        compactSidebar: "Fijar la barra lateral de forma predeterminada",
        compactSidebarDescription:
          "Mantiene la barra lateral expandida en lugar de contraerla a íconos.",
      },
      sidebarNav: {
        title: "Navegación de la barra lateral",
        description:
          "Fija y reordena las pestañas de la barra lateral. Las pestañas que no fijes pasan al menú “Más”.",
        dragToReorder: "Arrastra para reordenar",
        pinToSidebar: "Fijar {name} en la barra lateral",
        moreHolds: "Más ({count})",
      },
      sidebarMenu: {
        title: "Menú de perfil",
        description:
          "Elige y reordena los accesos directos que aparecen bajo tu nombre.",
        darkModeToggle: "Interruptor de modo oscuro",
        dragToReorder: "Arrastra para reordenar",
      },
    },
    resources: {
      title: "Sistema y motor",
      description:
        "Supervisa el hardware y el almacenamiento de este servidor de Unsloth.",
      liveUpdates: "Actualizaciones en vivo",
      floatingWindow: "Ventana flotante",
      showFloatingWindow: "Mostrar ventana flotante",
      showFloatingWindowDescription:
        "Mantiene a la vista el uso de CPU, RAM y GPU sobre Unsloth.",
      liveMonitor: {
        title: "Monitor en vivo",
        apiTitle: "Monitor de API",
        apiDescription:
          "Mira lo que otras apps envían a la API de este servidor en cuanto llega.",
        summary: "Solicitudes en vivo, errores y uso de tokens",
        status: "{active} activas · {recent} recientes · {model}",
        noModelLoaded: "ningún modelo cargado",
        autoOpen: "El monitor flotante se abre automáticamente",
        autoOpenDescription:
          "Abre un panel pequeño cuando llega tráfico de la API.",
        cpu: "CPU",
        ram: "RAM",
        disk: "Disco",
        modelsDisk: "Disco de modelos",
        vram: "VRAM",
        cpuCores: "{logical} núcleos lógicos / {physical} físicos",
        currentLoad: "Carga actual",
        free: "{value} libre",
        noGpu: "No hay GPU visible",
        gpuUnusable: "GPU no utilizable",
        gpuUnusableDetail: "Detectada, pero PyTorch no puede usarla",
        detailedView: "Vista detallada",
        detailedViewDescription:
          "Muestra la carga de la CPU y la carga, temperatura y consumo de cada GPU en la ventana flotante.",
        cpuLoad: "Carga de CPU",
        gpuLoad: "Carga de GPU",
        temperature: "Temperatura",
        power: "Consumo",
      },
      gpu: {
        title: "Dispositivos GPU",
        memory: "Memoria de GPU",
        sharedWithSystemRam: "Compartida con la RAM del sistema",
        estimatedAvailable: "Disponible estimado: {value}",
        sharedEstimatedAvailable:
          "RAM compartida del sistema: {value} disponibles (estimado)",
        ggufInference: "Memoria para modelos GGUF",
        unavailable: "no disponible",
        detecting: "Buscando GPU...",
        unreadable: "No se pudo leer el hardware de este servidor.",
        noGpu:
          "No se detectó ninguna GPU visible. Arriba se muestran los recursos solo de CPU.",
        noUsableGpu: "PyTorch no puede usar ninguna GPU de esta computadora.",
        mismatchCpuBuild:
          "PyTorch es una compilación solo para CPU ({version}), así que las GPU de abajo no se pueden usar. Repara la instalación para restaurar la compatibilidad con GPU.",
        mismatchUnavailable:
          "PyTorch ({version}) no puede inicializar las GPU de abajo, así que no se pueden usar. Revisa el controlador de la GPU o repara la instalación.",
        driverIdleEvict:
          "El controlador de AMD {version} tiene un error conocido que puede congelar Windows cuando una GPU de AMD está inactiva, sobre todo con más de una GPU. Actualiza a Adrenalin 26.9.2 o posterior.",
        driverIdleEvictDetails: "Detalles",
        dismissNotice: "Descartar",
        unusableDevice: "no utilizable",
        unknownDevice: "GPU desconocida",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "{value} en uso",
        free: "{value} libre",
        total: "{value} en total",
        usedOf: "{used} en uso de {total}",
        segmentUsed: "En uso por la GPU",
        segmentSystem: "En uso por el sistema",
        segmentAvailable: "Disponible",
      },
      llamaBackend: {
        title: "Motor de inferencia GGUF",
        label: "Backend de cómputo",
        description: "El backend que usa llama.cpp para ejecutar modelos GGUF.",
        runningOn: "llama.cpp se ejecuta actualmente en {backend}.",
        hint: "Instala la compilación de llama.cpp para este backend y la mantiene entre actualizaciones. Útil cuando la elección automática falla o el controlador de tu GPU no la admite. Solo se muestran los backends con una compilación para esta computadora; el entrenamiento no se ve afectado.",
        autoWith: "Automático ({backend})",
        apply: "Aplicar",
        applying: "Instalando...",
        applyHint:
          "Descarga la nueva compilación y reinicia llama.cpp. Se descargará de la memoria el modelo cargado.",
        applyHintWithSize:
          "Descarga {size} y reinicia llama.cpp. Se descargará de la memoria el modelo cargado.",
        switchedTo: "llama.cpp ahora se ejecuta en {backend}.",
        switchFailed: "No se pudo cambiar el backend de llama.cpp.",
        switchInterrupted: "El cambio se interrumpió antes de completarse.",
        envLocked:
          "La variable de entorno UNSLOTH_LLAMA_CPP_BACKEND lo fija en {backend} y tiene prioridad sobre esta opción.",
        customPath: {
          label: "Carpeta personalizada de llama.cpp",
          description: "Usa tu propia compilación de llama-server.",
          hint: "Elige la carpeta de llama.cpp que contiene llama-server o una compilación donde esté en build/bin. El runtime personalizado se usa para chat GGUF, embeddings y modelos de voz compatibles. Las variables de entorno siguen teniendo prioridad.",
          automatic: "Automático (incluido)",
          bundled: "Usa el runtime de llama.cpp instalado por Unsloth.",
          active:
            "Tu llama-server personalizado se usará la próxima vez que se cargue un modelo.",
          environmentManaged:
            "Administrado por la variable de entorno {variable}.",
          missingBinary:
            "llama-server ya no está disponible en esta carpeta. Elige otra carpeta o usa el runtime incluido.",
          reloadRequired:
            "Vuelve a cargar el modelo para usar el llama-server seleccionado.",
          change: "Cambiar",
          saving: "Guardando...",
          useBundled: "Usar el incluido",
          chooseTitle: "Elegir la carpeta de llama.cpp",
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
          notInstalled:
            "No se encontró una instalación administrada de llama.cpp, así que no hay ningún backend que cambiar.",
          localLink:
            "llama.cpp es un directorio local que enlazaste tú, así que Unsloth no lo reemplazará.",
          sourceBuild:
            "Este llama.cpp se compiló desde el código fuente, así que su backend no se puede cambiar desde aquí.",
          customPath:
            "Hay una carpeta personalizada de llama.cpp seleccionada. Su compilación determina el backend de cómputo.",
          unresolved:
            "No se pudieron consultar los backends disponibles. Revisa tu conexión y vuelve a intentarlo.",
          updateChecksDisabled:
            "Las comprobaciones de actualizaciones están desactivadas (UNSLOTH_DISABLE_UPDATE_CHECK=1), así que no se consultan los backends disponibles.",
        },
        llamaBackendKeywords:
          "llama.cpp backend gguf inferencia inference cuda rocm hip vulkan metal cpu gpu acelerador accelerator prebuilt precompilado cambiar switch motor engine personalizado custom ruta path carpeta folder directorio directory llama-server ejecutable executable binario binary",
      },
      modelMemory: {
        title: "Memoria del modelo",
        keepResident: "Mantener el modelo en la memoria de GPU",
        keepResidentDescription: "Permanece en la VRAM entre prompts.",
        keepResidentHint:
          "No devuelve los pesos a la RAM del sistema mientras el modelo siga cargado. Desactiva la liberación automática por inactividad y, cuando los pesos sí están en la RAM del host (memoria unificada o carga parcial en la GPU), también pasa --mlock para que el sistema operativo no pueda paginarlos y volver a subirlos en tu siguiente prompt.",
        noRamReserve: "No reservar RAM del sistema para el modelo",
        noRamReserveDescription:
          "Reduce la RAM del host que ocupan los pesos del modelo.",
        noRamReserveHint:
          "Omite la carga mapeada en memoria en las compilaciones de Windows compatibles cuando el modelo está cargado por completo en la GPU, para que sus páginas no queden residentes. En caso contrario, conserva la carga mapeada en memoria. Los búferes de CPU necesarios pueden seguir usando RAM. Quita --no-mmap y --mlock.",
        multiModel: "Mantener varios modelos cargados",
        openSystemSettings: "Abrir en Sistema",
        multiModelOff: "Desactivado",
        multiModelFit: "Cuando quepan",
        multiModelAlways: "Siempre",
        multiModelDescription:
          "Al cargar un modelo de chat, voz, imagen o video, se descargan los demás de la memoria.",
        multiModelDescriptionFit:
          "Al cargar un modelo, los demás se mantienen si caben. Si no caben, primero se descargan de la memoria los que no se están usando y después el activo.",
        multiModelDescriptionAlways:
          "Al cargar un modelo nunca se descarga otro de la memoria. Un modelo que no cabe no se carga. La liberación automática por inactividad y el entrenamiento siguen liberando memoria.",
        multiModelHint:
          "Se aplica a los modelos de chat, voz, imagen y video. Cada modelo de chat cargado responde a las solicitudes que lo nombran. Con varias GPU, Unsloth intenta colocar un modelo nuevo en una GPU que ningún otro modelo usa. “Cuando quepan” deja aproximadamente 1 GB libre. Un modelo cargado junto a otros recibe la memoria que estos dejan libre, así que puede empezar con un contexto más corto. “Cuando quepan” y “Siempre” nunca liberan un modelo de otra GPU.",
        allowOversizedLoad: "Permitir cargar modelos demasiado grandes",
        allowOversizedLoadDescription:
          "Carga modelos de imagen y video aunque la comprobación de memoria indique que no caben.",
        allowOversizedLoadHint:
          "Se aplica cuando la CPU y la GPU comparten una sola memoria (Apple Silicon, GPU integradas). Unsloth rechaza la carga cuando su estimación de los pesos del modelo supera la memoria que queda después de reservar espacio para el sistema operativo. Con esta opción activada, se intenta cargar de todos modos. La estimación puede equivocarse en ambos sentidos: una carga demasiado grande puede fallar, agotar toda la memoria, hacer que la computadora use mucho la memoria de intercambio o deje de responder, o detener Unsloth. Desactivado de forma predeterminada.",
        mediaIdleUnloadDescription:
          "Libera la VRAM descargando de la memoria los modelos de imagen y video después de esta cantidad de segundos de inactividad. El valor 0 los mantiene cargados; el mínimo es 60.",
        mlockVetoed:
          "--mlock permanece desactivado: fijar el modelo en memoria reservaría RAM para todo el modelo. La liberación automática por inactividad sigue desactivada.",
        mlockNotApplicable:
          "Totalmente en la GPU: no hay nada que bloquear en la RAM del sistema. La liberación automática por inactividad sigue desactivada.",
        memlockCapped:
          "Este sistema limita la memoria bloqueada a {limit}. Un modelo más grande no quedará fijado por completo; aumenta el límite con ulimit -l.",
        reloadRequired:
          "Vuelve a cargar el modelo para aplicar las nuevas opciones de memoria.",
        loadError: "No se pudo cargar la configuración de memoria del modelo",
        saveError: "No se pudo guardar la configuración de memoria del modelo",
        modelMemoryKeywords:
          "mlock memlock ulimit vram gpu memoria memory ram residente resident residency fijar pin pinned anclar página page bloquear lock locked mantener cargado keep loaded descargar unload inactivo idle mmap no-mmap load-mode paginación paging intercambio swap",
        multiModelKeywords:
          "varios múltiples multiple several modelos models juntos together a la vez alongside cambiar swap caber fit descargar unload expulsar evict vram gpu imagen image video chat voz speech tts voice leer en voz alta read aloud siempre always inactivo idle",
        mediaIdleUnloadKeywords:
          "imagen image video difusión diffusion inactivo idle descargar unload ttl tiempo de espera timeout segundos seconds liberar free vram gpu memoria memory",
      },
      storage: {
        title: "Almacenamiento",
        systemDisk: "Disco del sistema",
        systemUsed: "{used} en uso de {total} en el disco del sistema",
        modelsDiskSummary:
          "Unidad de la carpeta de modelos: {used} en uso de {total}, {free} libres",
        segmentModels: "Modelos",
        segmentCaches: "Cachés",
        segmentOther: "Otros",
        segmentFree: "Libre",
        refreshAction: "Actualizar",
        modelsDisk: "Disco de modelos",
        diskUsage: "{used} en uso / {total}",
        diskFree: "{free} libres",
        modelsFolder: "Carpeta de modelos",
        modelsFolderDescription: "Dónde se guardan los modelos descargados.",
        modelsFolderHint:
          "Dónde se guardan los modelos descargados. Cámbiala para mantener los modelos fuera de la unidad del sistema. Solo se aplica a las descargas nuevas. Los modelos que ya tienes se quedan donde están.",
        modelsFolderKeywords:
          "modelos models carpeta folder directorio directory ruta path ubicación location descarga descargas download downloads caché cache almacenamiento storage disco disk unidad drive mover move reubicar relocate hugging face",
        futureDownloads: "Solo las descargas nuevas",
        environmentManaged:
          "Administrado por la variable de entorno {variable}.",
        locationFree: "{free} libres",
        openAction: "Abrir",
        copyAction: "Copiar ruta",
        changeAction: "Cambiar",
        resetAction: "Usar la predeterminada",
        chooseTitle: "Elegir la ubicación de descarga de los modelos",
        chooseAction: "Usar para las próximas descargas",
        cacheSaved: "Se actualizó la ubicación de descarga de los modelos",
        cacheSaveError:
          "No se pudo actualizar la ubicación de descarga de los modelos",
        cachePickerError: "No se pudo abrir el selector de carpetas",
        copied: "Ruta copiada",
        openError: "No se pudo abrir la carpeta",
        copyError: "No se pudo copiar la ruta",
        caches: {
          label: "Archivos de caché",
          description:
            "{size} en cachés, de los cuales {reclaimable} se pueden borrar ahora.",
          hint: "Descargas de paquetes, kernels compilados y cachés de transferencia que Unsloth vuelve a crear cuando los necesita. Los modelos descargados, los proyectos, los chats, la configuración y tu token de Hugging Face nunca se borran aquí.",
          keywords:
            "caché cachés borrar limpiar vaciar purgar liberar espacio disco temporal compilado cache caches purge prune clear clean free space disk uv pip npm bun triton inductor cuda numba matplotlib vllm compiled xet temporary",
          measuring: "Midiendo el tamaño de las cachés...",
          measureFailed: "No se pudieron medir las cachés",
          empty: "No se encontraron archivos de caché.",
          detailsAction: "Detalles",
          recheckAction: "Volver a comprobar",
          hideDetailsAction: "Ocultar detalles",
          showFolders: "Mostrar carpetas",
          hideFolders: "Ocultar carpetas",
          previousLocations:
            "Incluye ubicaciones anteriores. El botón Borrar principal afecta a la carpeta de caché actual. Los botones de cada modelo borran solo la copia seleccionada.",
          clearModelDescription:
            "Esto elimina solo esta copia en caché. Se volverá a descargar cuando sea necesario.",
          clearAction: "Borrar cachés",
          clearOneAction: "Borrar",
          clearingAction: "Borrando...",
          confirmTitle: "¿Borrar los archivos en caché?",
          confirmDescription: "Esto libera unos {size}.",
          confirmOneTitle: "¿Borrar {name}?",
          safety:
            "Unsloth vuelve a crear una caché la próxima vez que la necesita. Los modelos descargados, los proyectos, los chats, la configuración y tu token de Hugging Face no se tocan.",
          hubCost:
            "Esta es la caché de modelos. Si la borras, esos modelos se volverán a descargar la próxima vez que los uses.",
          datasetsCost:
            "Si borras esto, esos conjuntos de datos se volverán a descargar la próxima vez que los uses.",
          blocked: "No se borró: {reason}",
          cleared: "Se borraron {size}",
          partial: "Algunos archivos de caché no se pudieron eliminar",
          clearFailed: "No se pudieron borrar las cachés",
          names: {
            uv: "Caché de paquetes de uv",
            pip: "Caché de descargas de pip",
            npm: "Caché de paquetes de npm",
            bun: "Caché de paquetes de Bun",
            torchInductor: "Caché de compilación de Torch Inductor",
            torchExtensions: "Compilaciones de extensiones de Torch",
            triton: "Caché de kernels de Triton",
            cuda: "Caché de kernels de CUDA",
            numba: "Caché de compilación de Numba",
            matplotlib: "Caché de fuentes de Matplotlib",
            vllm: "Caché de vLLM",
            unslothCompiled: "Módulos compilados de Unsloth",
            hfXet: "Caché de transferencia de Hugging Face",
            hfAssets: "Caché de recursos de Hugging Face",
            hfDatasets: "Caché de conjuntos de datos de Hugging Face",
            hfHub: "Caché de modelos de Hugging Face",
          },
        },
        lowDisk: {
          title: "Queda poco espacio en disco",
          criticalTitle: "El espacio en disco es críticamente bajo",
          description:
            "{free} libres de {total}. Borrar las cachés puede liberar espacio.",
          action: "Revisar cachés",
        },
      },
      environment: {
        title: "Entorno",
        backend: "Backend",
        python: "Python",
        torch: "Torch",
        transformers: "Transformers",
        mlxLm: "mlx-lm",
        mlxVlm: "mlx-vlm",
        uptime: "Tiempo activo",
        processMemory: "Memoria del proceso",
        notInstalled: "No instalado",
        unknown: "Desconocido",
        vramWithShared: "{vram} de VRAM + {shared} compartida",
      },
    },
    agents: {
      title: "Agentes",
      description:
        "Conecta agentes de programación como Claude Code y Codex a un modelo local con unsloth start.",
      agentKeywords:
        "vs code vscode copilot pi mistral vibe deepseek harness hermes openclaw opencode claude codex agente agentes programación",
      vscodeNote:
        "VS Code es la excepción: este comando agrega el modelo servido a la lista de modelos de Copilot Chat de VS Code (chatLanguageModels.json).",
      intro:
        "conecta Claude Code, Codex, DeepSeek Harness, Hermes, OpenClaw, OpenCode y otros agentes a un modelo servido localmente por Unsloth, totalmente sin conexión. Ejecuta un servidor compatible con OpenAI y nunca modifica los archivos de configuración de tu agente.",
      readDocs: "Leer la documentación",
      copy: "Copiar",
      copied: "Copiado",
      commandBuilder: "Generador de comandos",
      commandShell: "Shell para todos los comandos",
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
        "No se pudieron cargar todas las cuantizaciones. El comando usará el valor de modelo disponible.",
      generatedCommand: "Comando generado",
      docs: "Documentación",
      docsLabel: "Abrir la documentación de unsloth start",
      agentDocs: "Abrir la documentación de configuración de {agent}",
      copyGeneratedCommand: "Copiar el comando generado",
      automaticSettingsNote:
        "Unsloth aplica automáticamente la configuración recomendada del modelo si no definiste ninguna opción.",
      configurationNote:
        "También puedes ajustar cualquier configuración. Consulta más abajo o la",
      configurationDocs: "documentación",
      configurationFlagsSuffix: "para ver las opciones.",
      modelNote:
        "Codex requiere un modelo GGUF servido por llama-server. Otros agentes también pueden usar modelos basados en transformers; quita --model para usar el modelo que ya está cargado en Unsloth.",
      subagent: {
        title: "Usar un modelo local como subagente",
        description:
          "Mantén {agent} en su modelo actual y delega tareas seleccionadas a este modelo local de Unsloth.",
        setupCommand: "Comando de configuración",
        copySetupCommand: "Copiar el comando de configuración del subagente",
        usagePrompt: "Luego, en {agent}, escribe:",
        copyUsagePrompt: "Copiar el prompt de uso del subagente",
        defaultPrompt: "Inicia un agente local para implementar esta función.",
        opencodePrompt:
          "@unsloth encuentra la causa de esta falla en la prueba",
      },
      quickstart: {
        title: "Crear un comando",
        description:
          "Inicia un agente con el modelo cargado actualmente en Unsloth. Primero carga un modelo y luego cambia claude por cualquiera de los agentes compatibles que aparecen abajo.",
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
          "Usa --model para elegir un modelo y una cuantización, y --context-length para definir la ventana de contexto. Usa un sufijo de cuantización o la opción explícita --gguf-variant.",
        suffixLabel: "Con un sufijo de cuantización",
        variantLabel: "Con una opción explícita de variante",
      },
      options: {
        title: "Opciones comunes",
        description:
          "Primero se procesan las opciones de Unsloth; todo lo que no reconoce se pasa directamente al agente.",
        model:
          "Selecciona un modelo. Sin --model, unsloth start usa el modelo cargado actualmente en Unsloth y da error si no hay ninguno cargado.",
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
          "Conserva entre ejecuciones el almacenamiento de agentes que administra Unsloth.",
        asSubagent:
          "Mantén el agente principal en su modelo actual y registra Unsloth como subagente local (Claude Code, Codex, OpenCode y Pi).",
        apiKey:
          "Proporciona tu clave de API de Unsloth (o define UNSLOTH_API_KEY).",
        reasoning:
          "Usa razonamiento en el chat: on, off o auto. Auto sigue la plantilla de chat del modelo, lo que suele significar on.",
        reasoningEffort:
          "Esfuerzo de razonamiento que se pasa a la plantilla de chat del modelo, por ejemplo medium. Los niveles dependen del modelo, así que usa uno que ese modelo acepte. Sin valor, se mantiene el de la plantilla.",
        yolo: "Omite las solicitudes de aprobación. Úsalo solo en entornos de confianza.",
      },
      remote: {
        title: "Conectarse a un Unsloth remoto",
        description:
          "Apunta unsloth start a una instancia de Unsloth que se ejecuta en otro lugar definiendo estas variables antes de iniciar (o pasa --api-key directamente):",
      },
      passthrough: {
        title: "Pasar argumentos al agente",
        description:
          "Los argumentos que van después de las opciones de Unsloth se reenvían al propio agente, así que los comandos nativos como resume siguen funcionando:",
      },
      dryRun: {
        title: "Vista previa sin iniciar",
        description:
          "Agrega --no-launch para imprimir el entorno y el comando en lugar de iniciar el agente. Si --model está definido, el modelo aún puede resolverse y cargarse.",
      },
    },
    chat: {
      groups: {
        conversations: {
          title: "Conversaciones",
        },
        files: {
          title: "Archivos y pegado",
        },
        display: {
          title: "Visualización",
        },
        composer: {
          title: "Cuadro de mensaje",
        },
        menu: {
          title: "Menú del chat",
        },
        advanced: {
          title: "Avanzado",
        },
        contextTitle: "Contexto",
      },
      library: {
        label: "Administrar chats",
        description: "Busca, ordena y organiza chats, proyectos y secciones.",
        action: "Abrir Biblioteca",
      },
      projectAttachmentsHint:
        "Cambia esta opción en cada chat desde su menú de archivos adjuntos.",
      rememberParamsPerModelHint:
        "Si está desactivado, todos los modelos usan la misma configuración.",
      autoCompactHint:
        "Chats GGUF y MLX locales: los turnos expulsados se indexan para que el modelo pueda recuperarlos con una búsqueda, y un reinicio cita palabra por palabra las instrucciones permanentes que quepan, dando prioridad a las más antiguas y a las más recientes sobre las intermedias. Archivar requiere un chat guardado y el índice vectorial; sin ellos, los turnos antiguos se descartan. Usa la longitud de contexto que configuraste, no la VRAM disponible. Modelos de API: el proveedor resume los turnos antiguos cuando puede; si no, se descartan los más antiguos cuando el chat se acerca a la ventana de contexto publicada del modelo.",
      pastedTextShortDescription:
        "El texto pegado de {count} caracteres o más se convierte en un adjunto .txt.",
      pastedTextOffDescription:
        "El texto pegado siempre se queda en el cuadro de mensaje.",
      title: "Chat",
      description: "Personaliza cómo funciona el chat en este dispositivo.",
      modelSelection: {
        title: "Selección del modelo",
        expandQuantizations: "Expandir las cuantizaciones",
        expandQuantizationsDescription:
          "Muestra las cuantizaciones GGUF en “En el dispositivo” sin abrir cada modelo.",
        showAllQuantizations: "Todas las cuantizaciones",
        showAllQuantizationsDescription:
          "Incluye en “En el dispositivo” las cuantizaciones que no están descargadas.",
        showMemoryBar: "Barra de uso de VRAM",
        showMemoryBarDescription:
          "Muestra la VRAM estimada para los pesos del modelo, el contexto y la decodificación especulativa.",
        fitOnDeviceOnly: "Mostrar solo modelos que caben",
        fitOnDeviceOnlyDescription:
          "Oculta en el selector de modelos y en el Hub los modelos demasiado grandes para la memoria de GPU y la RAM de este dispositivo. Los modelos descargados siguen visibles.",
      },
      menu: {
        title: "Menú del chat",
        description:
          "Fija elementos en el menú + del chat y arrástralos para reordenarlos. Los demás pasan a “Más”.",
        deepResearch: "Investigación profunda",
        browserUse: "Uso del navegador",
        chatWithFiles: "Chatear con archivos (RAG)",
        mcp: "MCP",
        skills: "Skills",
        savedPrompts: "Prompts guardados",
        compareChat: "Comparar chats",
        exportChat: "Exportar chat",
        reset: "Restablecer valores predeterminados",
        addFiles: "Agregar fotos y archivos",
        webSearch: "Búsqueda web",
        tools: "Herramientas",
        more: "Más",
      },
      pastedTextThreshold: "Condensar textos pegados largos",
      pastedTextThresholdDescription:
        "Presiona {shortcut} para pegar directamente en el cuadro de mensaje.",
      pastedTextThresholdOff: "Desactivado",
      autoScroll: "Desplazamiento al generar",
      autoScrollDescription:
        "El desplazamiento automático mantiene visible el texto nuevo. Manual se queda fijo para que puedas leer desde el principio.",
      autoScrollAuto: "Desplazamiento automático",
      autoScrollManual: "Manual",
      autoScrollKeywords:
        "desplazamiento desplazar automático seguir abajo saltar transmisión generar vista bloquear fijar manual scroll autoscroll auto-scroll follow stick bottom jump streaming generating viewport lock hold auto",
      replyNotification: "Avisar cuando termine una respuesta",
      replyNotificationDescription:
        "Muestra una notificación del sistema cuando termina una respuesta mientras Unsloth está minimizado o en segundo plano.",
      replyNotificationBlocked:
        "Las notificaciones están bloqueadas. Permítelas para Unsloth en la configuración del navegador o del sistema y vuelve a intentarlo.",
      replyNotificationKeywords:
        "notificación avisar alerta terminado completado listo respuesta segundo plano minimizado escritorio sonido notify notification alert done finished complete ready response reply background minimized desktop sound",
      showResponseModel: "Modelo de la respuesta",
      showResponseModelDescription:
        "Muestra los detalles del modelo en las respuestas del asistente.",
      showReplyTimes: "Hora de las respuestas",
      showReplyTimesDescription:
        "Muestra debajo de cada respuesta cuándo terminó.",
      showReplyTimesKeywords:
        "marca de tiempo hora reloj fecha cuándo terminó llegó respuesta mensaje timestamp time clock date hour when finished arrived reply response message",
      showGenerationDetails: "Detalles de generación",
      showGenerationDetailsDescription:
        "Muestra en vivo las velocidades del prompt y de la respuesta con el recuento de tokens, y dónde se cargaron las capas de un modelo.",
      showGenerationDetailsKeywords:
        "detallado desarrollador estadísticas velocidad capas trasladar expertos verbose developer stats speed tok/s prefill decode layers offload vram experts",
      inlineEditResponse: "Editar respuesta en las respuestas",
      inlineEditResponseDescription:
        "Mantén Editar respuesta en cada respuesta, en lugar de en el menú Más.",
      composerAtBottom: "Cuadro de mensaje abajo",
      composerAtBottomDescription:
        "Empieza los chats nuevos con el cuadro de mensaje en la parte inferior de la página en lugar del centro.",
      projectAttachments: "Compartir archivos en todo el proyecto",
      projectAttachmentsDescription:
        "Haz que los nuevos archivos adjuntos del chat estén disponibles para todos los chats del proyecto.",
      rememberParamsPerModel: "Recordar la configuración por modelo",
      rememberParamsPerModelDescription:
        "Restaura el último prompt, la temperatura y las demás opciones usadas con cada modelo.",
      autoCompact: "Compactar automáticamente los chats largos",
      autoCompactDescription:
        "Los turnos antiguos pasan a un archivo consultable cuando un chat llena su contexto.",
      autoCompactKeywords:
        "compactación compactar automáticamente contexto ventana truncar deslizante checkpoint margen archivo recuperación búsqueda compaction compact auto-compact context window truncate rolling headroom archive retrieval recall rag search",
      keepRunningWhenTabClosed: "Seguir ejecutándose al cerrar la pestaña",
      keepRunningWhenTabClosedDescription:
        "Las respuestas, la búsqueda web y Deep Research siguen en el servidor después de cerrar la pestaña. Desactívalo para detenerlas.",
      keepRunningWhenTabClosedHint:
        "Vuelve a abrir el chat en cualquier dispositivo para retomar una ejecución que siguió en marcha. Si está desactivado, una ejecución se detiene aproximadamente un minuto después de cerrarse su última pestaña, así que recargar no la detiene. Se aplica a las ejecuciones iniciadas después de cambiar esta opción. Los chats temporales, las respuestas de modelos de API o a archivos adjuntos y los turnos que usan el panel del navegador siempre se detienen con su pestaña.",
      keepRunningWhenTabClosedKeywords:
        "segundo plano pestaña cerrar cerrada salir seguir ejecutando continuar detener cancelar investigación búsqueda web reanudar reconectar dispositivo celular teléfono background tab close closed leave keep running continue stop cancel deep research web search resume reconnect device phone",
      firstTokenLabel: "Límite de espera del primer token",
      firstTokenDescription:
        "Cuánto pueden esperar los modelos GGUF locales sin avance en el procesamiento del prompt antes del primer token. Se aplica a las nuevas esperas de todas las cuentas. Con “Sin límite” se sigue esperando hasta que lo detengas.",
      firstTokenDefault: "20 min (predeterminado)",
      firstTokenCustom: "Personalizado",
      firstTokenUnlimited: "Sin límite",
      firstTokenMinutes: "Espera del primer token en minutos",
      firstTokenRange: "Minutos (1–10,080)",
      firstTokenSave: "Guardar",
      firstTokenSaving: "Guardando…",
      firstTokenLoadError:
        "No se pudo cargar el límite de espera del primer token.",
      firstTokenSaveError:
        "No se pudo guardar el límite de espera del primer token.",
      firstTokenKeywords:
        "tiempo de espera primer token procesamiento del prompt lento CPU toda la noche sin límite límite de espera timeout first token prefill slow overnight unlimited waiting limit",
      toolResultLimit: "Longitud de la salida de herramientas",
      toolResultLimitDescription:
        "Caracteres de un resultado de herramienta de Python, Terminal o MCP que lee el modelo. El valor predeterminado es {count}.",
      toolResultLimitHint:
        "La salida más larga de Python y Terminal se guarda en un archivo que el modelo puede leer por partes. Un límite mayor se aplica a los modelos que se ejecutan en Unsloth, y las ventanas de contexto pequeñas siguen recibiendo menos. Los modelos de API mantienen el valor predeterminado a menos que lo bajes.",
      toolResultLimitDefault: "{count} (predeterminado)",
      toolResultLimitLocked: "Definido por UNSLOTH_TOOL_RESULT_MAX_CHARS",
      toolResultLimitLoadError:
        "No se pudo cargar la longitud de la salida de herramientas.",
      toolResultLimitSaveError:
        "No se pudo guardar la longitud de la salida de herramientas.",
      toolResultLimitKeywords:
        "salida resultado truncar recortar límite caracteres longitud herramienta tool output result truncate truncated truncation limit cap characters chars length terminal python mcp",
      visibility: {
        collapsed: "Contraído",
        auto: "Expandir mientras se ejecuta",
        expanded: "Siempre expandido",
      },
      visibilityKeywords:
        "contraer contraído expandir expandido abierto cerrado razonamiento pensamiento llamadas a herramientas actividad agrupar transmisión collapse collapsed expand expanded open closed reasoning thinking tool calls tool activity fold group streaming",
      thinking: {
        visibility: "Razonamiento",
        visibilityDescription:
          "Cómo se abre el razonamiento. Puedes expandir o contraer cualquier bloque por tu cuenta.",
      },
      memory: {
        title: "Memoria",
        description:
          "Notas que lee cada chat, con cualquier modelo, local o en la nube. Los chats de un proyecto también leen el AGENTS.md de ese proyecto.",
        enable: "Usar memoria en los chats",
        enableDescription: "Envía estas notas con cada chat.",
        locked: "Desactivado en esta instalación por UNSLOTH_STUDIO_AGENTS_MD.",
        editor: "Notas de memoria",
        placeholder:
          "Por ejemplo:\n- Escribo en Python y uso uv.\n- Da respuestas breves.",
        hint: "Se guarda en {path}. También puedes editar ese archivo con cualquier editor de texto.",
        size: "{used} / {max} KB",
        save: "Guardar",
        saved: "Memoria guardada",
        loadError: "No se pudo cargar la memoria.",
        saveError: "No se pudo guardar la memoria.",
        keywords:
          "memoria recordar personalización perfil preferencias notas datos instrucciones personalizadas sobre mí memory remember personalization personalisation profile preferences notes facts custom instructions about me agents.md claude.md",
      },
      currentDate: {
        label: "Indicarle al modelo la fecha de hoy",
        description:
          "Incluye la fecha de hoy para que la búsqueda y la investigación encuentren fuentes recientes.",
        loadError: "No se pudo cargar la configuración de la fecha actual",
        saveError: "No se pudo actualizar la configuración de la fecha actual",
      },
      tools: {
        visibility: "Llamadas a herramientas",
        visibilityDescription:
          "Cómo se abre la actividad de las herramientas. Puedes expandir o contraer cualquier llamada por tu cuenta.",
        foldIntoThinking: "Agrupar las llamadas a herramientas en Razonamiento",
        foldIntoThinkingDescription:
          "Muestra las llamadas a herramientas de un turno dentro de su bloque de Razonamiento en lugar de en filas propias.",
        foldIntoThinkingBlocked:
          "No disponible mientras las llamadas a herramientas estén en “Siempre expandido”, que las mantiene en filas propias.",
      },
      webSearch: {
        title: "Búsqueda web",
        images: "Imágenes de la búsqueda web",
        imagesDescription: "Incluye imágenes en los resultados de búsqueda.",
      },
      artifacts: {
        title: "Páginas HTML",
        collapseHtmlBlocks: "Contraer bloques HTML",
        collapseHtmlBlocksDescription:
          "Muestra los documentos HTML completos del chat como una tarjeta que se abre en el navegador, en lugar de como código.",
        allowNetworkAccess: "Permitir acceso a la red a las páginas HTML",
        allowNetworkAccessDescription:
          "Permite que las páginas HTML del chat carguen scripts, estilos, fuentes y contenido multimedia externos. Desactívalo para vistas previas sin conexión.",
        blockedBanner: "Se bloqueó {count} recurso externo de {hosts}.",
        blockedBannerPlural:
          "Se bloquearon {count} recursos externos de {hosts}.",
        blockedBannerAction: "Permitir para esta página",
        blockedTitle:
          "El acceso a la red está desactivado para las páginas HTML",
        blockedHint:
          "Activa “{setting}” en Configuración → Navegador para que las páginas HTML carguen recursos externos, o permítelo solo para esta página.",
        blockedSettingsAction: "Abrir Configuración",
        blockedDismiss: "Descartar",
        errorTitle: "Esta página tuvo un error",
        errorTitlePlural: "Esta página tuvo {count} errores",
        errorHint:
          "“Corregir con el modelo” pone el error en el cuadro de mensaje. No se envía nada hasta que tú lo envíes.",
        errorBannerAction: "Corregir con el modelo",
        errorConsoleAction: "Abrir consola",
        errorConsoleHideAction: "Ocultar consola",
        errorLocation: "línea {line}, columna {column}",
        errorLine: "línea {line}",
        consoleTitle: "Consola",
        reloadCanvas: "Ejecutar de nuevo",
        consoleMessageCount: "{count} mensaje",
        consoleMessageCountPlural: "{count} mensajes",
        consoleClear: "Borrar consola",
        consoleClose: "Cerrar consola",
        consoleEmpty: "Aún no hay salida en la consola.",
        consoleCapped:
          "Solo se conservan las últimas {count} entradas; la salida anterior se descartó.",
      },
      data: "Datos",
      exportHistory: "Exportar historial de chat",
      exportHistoryDescription:
        "Descarga todos los chats y mensajes como JSON.",
      exportAction: "Exportar",
      exportingAction: "Exportando...",
      exportConversations: "Exportar Recientes y Proyectos",
      exportConversationsDescription:
        "Descarga Recientes, o Recientes más los chats de proyectos, como Training JSONL, CSV, ShareGPT JSONL o Markdown, combinados o por chat. Message JSONL está disponible por chat.",
      exportConversationsAction: "Exportar",
      exportScopeRecents: "Recientes",
      exportScopeAll: "Recientes + Proyectos",
      exportCombinedSuffix: "(combinado)",
      exportPerChatSuffix: "(por chat)",
      importChats: "Importar chats",
      importChatsDescription:
        "Importa archivos JSON, JSONL, NDJSON, CSV o Markdown a Recientes.",
      importChatsAction: "Importar",
      importNoConversations: "No se encontraron conversaciones en el archivo.",
      importedOneChat: "Se importó 1 conversación a Recientes.",
      importedChatCount: "Se importaron {count} conversaciones a Recientes.",
      importingChats: "Importando chats: {count} hasta ahora ({percent}%)...",
      importedChatCountPartial:
        "Se importaron {count} conversaciones a Recientes; {failed} no se pudieron guardar.",
      importFailed: "La importación falló.",
      importFromSource: "Importar desde {source}",
      importFromSourceDescription:
        "Copia tus conversaciones de {source} a Unsloth, agrupadas por el proyecto al que pertenecen.",
      importingAction: "Importando...",
      importedSourceOneChat: "Se importó 1 conversación desde {source}.",
      importedSourceChatCount:
        "Se importaron {count} conversaciones desde {source}.",
      sourceUpdated: "Se agregaron {count} mensajes nuevos desde {source}.",
      sourceUpToDate: "Las conversaciones de {source} ya están actualizadas.",
      importedSourcePartial:
        "No se pudieron importar algunas conversaciones de {source}.",
      clearHistory: "Borrar historial de chat",
      clearHistoryDescription:
        "Elimina el historial de chat de este dispositivo.",
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
        "Elimina permanentemente todos los chats de este dispositivo. Esta acción no se puede deshacer.",
      clearingAction: "Borrando...",
      clearOneChatAction: "Borrar 1 chat",
      clearChatCountAction: "Borrar {count} chats",
      clearedAllChats: "Se borraron todos los chats",
      clearedOneChat: "Se borró 1 chat",
      clearedChatCount: "Se borraron {count} chats",
      someChatsCouldNotBeCleared: "Algunos chats no se pudieron borrar",
      chatsClearedRemainOne:
        "{clearedCount} chats borrados; queda 1 chat. Vuelve a intentarlo.",
      chatsClearedRemain:
        "{clearedCount} chats borrados; quedan {remainingCount} chats. Vuelve a intentarlo.",
      oneChatClearedRemain:
        "1 chat borrado; quedan {remainingCount} chats. Vuelve a intentarlo.",
      oneChatClearedRemainOne:
        "1 chat borrado; queda 1 chat. Vuelve a intentarlo.",
      storageClearFailedOne:
        "Falló un borrado del almacenamiento; puede quedar 1 chat. Vuelve a intentarlo.",
      storageClearFailed:
        "Falló un borrado del almacenamiento; pueden quedar {count} chats. Vuelve a intentarlo.",
      failedToClearChats: "No se pudieron borrar los chats",
    },
    data: {
      library: {
        defaultSort: "Predeterminado",
        updated: "Actualización",
        created: "Creación",
        oldest: "Más antiguos primero",
        alphabetical: "Orden alfabético",
        allChats: "Todos los chats",
        singleChats: "Chats individuales",
        compareChats: "Chats de comparación",
        allProjects: "Todos los proyectos",
        noProject: "Sin proyecto",
        unavailableProject: "Proyecto no disponible",
        filterSort: "Filtrar y ordenar",
        sort: "Ordenar",
        type: "Tipo",
        sortBy: "Ordenar por",
        filterProject: "Filtrar por proyecto",
        searchProjects: "Buscar proyectos",
        noProjects: "No se encontraron proyectos.",
        untitled: "Sin título",
        oneChat: "1 chat",
        chatCount: "{count} chats",
        searchChats: "Buscar chats o proyectos",
        searchArchivedChats: "Buscar chats o proyectos archivados",
        searchImages: "Buscar imágenes archivadas",
        searchVideos: "Buscar videos archivados",
        searchAudio: "Buscar audio archivado",
        unarchive: "Desarchivar",
        unarchiveAll: "Desarchivar todo",
        unarchiveAllInProject: "Desarchivar todo en el proyecto",
        deleteAllInProject: "Eliminar todo en el proyecto",
        groupActions: "Acciones de {group}",
        unarchiveResults: "Desarchivar resultados",
        deleteResults: "Eliminar resultados",
        noChats: "Ningún chat coincide con tu búsqueda.",
        noArchivedChats: "No hay chats archivados.",
        noArchivedMatches: "Ningún chat archivado coincide con tu búsqueda.",
        noMedia: "No hay elementos archivados.",
        noMediaMatches: "Ningún elemento archivado coincide con tu búsqueda.",
        itemCount: "Elementos: {count}",
        incompleteSearch:
          "La búsqueda está incompleta. Vuelve a intentar cargar los elementos restantes.",
        searchingRemaining:
          "Buscando los elementos restantes ({count} cargados)...",
        noLoadedMatches:
          "No hay coincidencias entre los elementos cargados hasta ahora.",
        deleteItem: "Eliminar: {title}",
        unarchiveItem: "Desarchivar: {title}",
        deleteItemsTitle: "Eliminar elementos archivados ({count})",
        unarchiveItemsTitle: "Desarchivar elementos ({count})",
        deleteFilesWarning:
          "Estos archivos se eliminarán permanentemente. Esta acción no se puede deshacer.",
        restoreWarning: "Estos elementos volverán a la galería.",
        working: "Procesando...",
        loadFailed: "No se pudieron cargar los elementos archivados",
        loadMoreFailed: "No se pudieron cargar más elementos archivados",
        restoreFailed: "No se pudieron desarchivar los elementos seleccionados",
        deleteFailed: "No se pudieron eliminar los elementos seleccionados",
        projectsFailed: "No se pudieron cargar los proyectos archivados",
        pageStalled: "La página de archivados no avanzó. Vuelve a intentarlo.",
        pageChanged:
          "Los archivados cambiaron durante la carga. Vuelve a intentarlo.",
        selectAll: "Seleccionar todos los chats visibles",
        selectItem: 'Seleccionar "{title}"',
        selectedChats: "Chats seleccionados: {count}",
        move: "Mover",
        pin: "Fijar",
        unpin: "Desfijar",
        archive: "Archivar",
        deletedChats: "Chats eliminados: {count}",
        restoredChats: "Chats desarchivados: {count}",
        pinnedChats: "Chats fijados: {count}",
        unpinnedChats: "Chats desfijados: {count}",
        movedChatsToProject: "Chats movidos a {project}: {count}",
        movedChatsToRecents: "Chats movidos a Recientes: {count}",
        moveFailed: "No se pudieron mover los chats",
        deleteChatsTitle: "Eliminar chats ({count})",
        deleteChatsWarning:
          "¿Eliminar los chats seleccionados ({count})? Esta acción no se puede deshacer.",
        deleteArchivedTitle: "Eliminar {count} chats archivados",
        deleteArchivedWarning:
          "¿Eliminar los chats archivados seleccionados ({count})? Esta acción no se puede deshacer.",
        deleting: "Eliminando...",
      },
      title: "Datos",
      backToData: "Volver a Datos",
      exportFailed: "No se pudieron exportar los chats",
      description:
        "Administra el historial de chats y los archivos subidos que se guardan en este dispositivo.",
      manageFiles: {
        label: "Administrar archivos",
        description:
          "Busca, ordena y organiza imágenes, videos, audio y archivos.",
        action: "Abrir Biblioteca",
      },
      archivedChats: "Chats archivados",
      archivedChatsDescription:
        "Consulta y administra los chats que archivaste.",
      archivedImages: "Imágenes archivadas",
      archivedImagesDescription:
        "Consulta y administra las imágenes que archivaste.",
      archivedVideos: "Videos archivados",
      archivedVideosDescription:
        "Consulta y administra los videos que archivaste.",
      archivedAudio: "Audio archivado",
      archivedAudioDescription:
        "Consulta y administra los clips de audio que archivaste.",
      manageAction: "Administrar",
      exportArchivedChats: "Exportar",
      exportingArchivedChats: "Exportando...",
      exportedOneArchivedChat: "Se exportó 1 chat archivado",
      exportedArchivedChatCount: "Se exportaron {count} chats archivados",
      noArchivedChatsToExport: "No hay chats archivados para exportar.",
      failedToExportArchivedChats:
        "No se pudieron exportar los chats archivados",
      archiveAllChats: "Archivar todos los chats",
      archiveAllChatsDescription:
        "Mueve a archivados todos los chats de Recientes y Proyectos.",
      noChatsToArchive: "No hay chats para archivar.",
      archiveAllAction: "Archivar todos",
      archivingAction: "Archivando...",
      archiveAllChatsTitle: "¿Archivar todos los chats?",
      archiveAllChatsConfirmDescription:
        "Mueve a archivados todos los chats de este dispositivo. Los chats archivados siguen disponibles y se pueden desarchivar en cualquier momento.",
      archivedAllChats: "Se archivaron todos los chats",
      archivedOneChat: "Se archivó 1 chat",
      archivedChatCount: "Se archivaron {count} chats",
      failedToArchiveChats: "No se pudieron archivar los chats",
      confirmBeforeDeleting: "Confirmar antes de eliminar",
      confirmBeforeDeletingDescription:
        "Pide confirmación antes de eliminar un chat. Desactívalo para eliminar al instante.",
      alwaysDeleteFiles: "Eliminar siempre los archivos",
      alwaysDeleteFilesDescription:
        "Al eliminar un chat también se elimina del disco su propia carpeta de sandbox. Los archivos que haya escrito dentro de un proyecto permanecen en el espacio de trabajo de ese proyecto.",
      chatsSection: "Chats y archivados",
      transferSection: "Importar y exportar",
      deletionSection: "Eliminación",
      archives: "Archivados",
      archiveChatsLabel: "Chats",
      archiveVideosLabel: "Videos",
      exportFormatsSummary:
        "Exporta los chats de Recientes y de proyectos en el formato que prefieras.",
      trainingSummary:
        "Convierte tus chats en un conjunto de datos para ajuste fino.",
      confirmDeletionSummary:
        "Pide confirmación antes de eliminar chats individuales.",
      sandboxFiles: "Archivos del sandbox del chat",
      sandboxFilesDescription: "Acción predeterminada al eliminar un chat.",
      keepSandboxFiles: "Conservar archivos del sandbox",
      deleteSandboxFiles: "Eliminar archivos del sandbox",
      projectFilesKept:
        "Los sandboxes son carpetas creadas para cada chat. Se conservan los archivos guardados en los espacios de trabajo de los proyectos.",
      deleteAllAction: "Eliminar todo",
      deleteChatsOnly: "Eliminar solo los chats…",
      deleteChatsAndSandboxes: "Eliminar chats y sandboxes…",
      deleteSandboxFilesDescription:
        "Elimina también la carpeta del sandbox de cada chat y los archivos que contiene. Se conservan los archivos de los espacios de trabajo de los proyectos.",
      filesSection: "Archivos",
      uploadedFiles: "Archivos subidos",
      uploadedFilesDescription:
        "Consulta y administra los archivos subidos a chats, proyectos y bases de conocimiento.",
      fineTuneExport: "Usar los chats como datos de entrenamiento",
      fineTuneExportDescription:
        "Crea un conjunto de datos JSONL para ajuste fino a partir de tus chats. Cárgalo en Entrenar, refínalo en Recetas o expórtalo.",
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
      description: "Administra proveedores y conexiones externas.",
    },
    remoteLan: {
      title: "Remoto y LAN",
      description:
        "Accede a este Unsloth desde tus otros dispositivos, a través de tu red local o de una URL pública temporal.",
    },
    apiKeys: {
      title: "API",
      description:
        "Accede a Unsloth mediante la API compatible con OpenAI y Anthropic.",
      readDocs: "Leer la documentación de la API",
      noAccess: "Aún no hay acceso a la API.",
      accessTokens: "Tokens de acceso",
      quickstartTitle: "Inicio rápido",
      columnName: "Nombre",
      columnToken: "Token",
      columnCreated: "Creado",
      columnLastUsed: "Último uso",
      columnExpires: "Vence",
      loadError: "No se pudo cargar el acceso a la API.",
      createError: "No se pudo crear el token de acceso.",
      revokeError: "No se pudo revocar el token de acceso.",
      never: "Nunca",
      tokenNamePlaceholder: "Nombre del token",
      newAccessTokenName: "Nombre del nuevo token de acceso",
      createToken: "Crear token",
      neverExpiresHint: "Nunca vence",
      expiresInDaysHint: "Vence en {n} días",
      createTokenHint: 'Crear un token llamado "{name}"',
      defaultTokenName: "Token nuevo {n}",
      creating: "Creando...",
      newTokenCreated: "Se creó un nuevo token de acceso",
      accessTokenCopied: "Token de acceso copiado",
      copyAccessToken: "Copiar token de acceso",
      copyNow: "Cópialo ahora: no se volverá a mostrar.",
      usageExamples: "API de chat",
      usageNoModel:
        "Carga o descarga un modelo para ver ejemplos ejecutables. Este servidor todavía no tiene ningún modelo que indicar.",
      usageTools: "Herramientas",
      exampleLanguage: "Lenguaje",
      exampleRequest: "Solicitud",
      examplePlatform: "Plataforma",
      requestPlain: "Básica",
      requestTools: "Con herramientas",
      requestAdvanced: "Muestreo avanzado",
      exampleKind: "Ejemplo",
      osUnix: "Linux / macOS / WSL",
      osWindows: "Windows",
      secureHttps: "HTTPS seguro",
      secureHttpsHint:
        "El puerto 0.0.0.0 sigue siendo accesible globalmente. Para máxima seguridad, inicia Unsloth con --secure para exponer solo este enlace HTTPS.",
      copyTunnelUrl: "Copiar URL del túnel",
      copySnippet: "Copiar fragmento",
      copy: "Copiar",
      copied: "Copiado",
      codingAgents: "Agentes de programación",
      codingAgentsHint:
        "Inicia un agente de programación conectado a este servidor. Usa el modelo cargado; un servidor local genera una clave de API automáticamente y uno remoto la incluye en el comando.",
      openAgentsTab: "Abrir en Agentes",
      codingAgentsSwap:
        "Reemplaza claude por codex, openclaw, opencode o hermes.",
      relativeNever: "nunca",
      relativeJustNow: "justo ahora",
      expired: "vencido",
      expiredStatus: "Vencido",
      today: "hoy",
      actionsFor: "Acciones para {name}",
      copyPrefix: "Copiar prefijo",
      copyFailed: "No se pudo copiar",
      revokeToken: "Revocar token",
      regenerateToken: "Regenerar token",
      regenerateTitle: '¿Regenerar el token de acceso "{name}"?',
      regenerateDescription:
        "Un token nuevo con el mismo nombre y periodo de vencimiento reemplaza a este. Las apps que usan el token actual pierden el acceso de inmediato.",
      regenerateAction: "Regenerar",
      regenerating: "Regenerando...",
      regenerateError: "No se pudo regenerar el token de acceso.",
      revokeTitle: '¿Revocar el token de acceso "{name}"?',
      revokeDescription:
        "Las apps que usan este token pierden el acceso de inmediato. Esta acción no se puede deshacer.",
      revokeAction: 'Revocar "{name}"',
      revoking: "Revocando...",
      audioApi: {
        title: "API de audio",
        description:
          "Síntesis de voz, clonación de voz, transcripción y los flujos de trabajo de la página Audio.",
        speak: "Hablar",
        clone: "Clonar",
        transcribe: "Transcribir",
        workflows: "Flujos de trabajo",
        separate: "Separar",
        convert: "Convertir",
        music: "Música",
        edit: "Editar",
        placeholderModel:
          "Aún no hay ningún modelo descargado que pueda ejecutar este ejemplo, así que el modelo que aparece es provisional. Primero descarga uno en la página Audio.",
        autoSwitchOff:
          "El modelo solo se carga por su nombre mientras el cambio automático de modelo está activado. Actívalo o primero carga el modelo en su página de Audio.",
        useViaApi: "Usar mediante API",
      },
      mediaApi: {
        title: "API de imágenes y video",
        description:
          "Genera imágenes y videos desde tu propio código con los endpoints compatibles con OpenAI.",
        images: "Imágenes",
        video: "Video",
        placeholderModel:
          "El modelo que aparece es un ejemplo. Carga un modelo en la página Imágenes o Video y abre esto desde su botón de API para completarlo.",
        autoSwitchOff:
          "El cambio automático de medios está desactivado, así que las solicitudes se ejecutan en el modelo que tiene cargado la página, sin importar cuál indiquen.",
      },
      trainingApi: {
        title: "API de entrenamiento",
        description:
          "Haz fine-tuning de modelos desde tu propio código con los endpoints de ajuste fino compatibles con OpenAI: sube un archivo JSONL, crea un trabajo, sigue sus eventos y luego chatea con el modelo entrenado.",
        placeholder:
          "El modelo y la configuración que aparecen son un ejemplo. Abre esto desde el botón de API de la página Entrenar para completar tu configuración actual.",
        hubDataset:
          "Tu página Entrenar usa un conjunto de datos de Hugging Face o S3. La API entrena con archivos subidos, así que guarda el conjunto de datos como JSONL y súbelo.",
        batching:
          "Los trabajos compatibles que esperan en la cola se entrenan juntos automáticamente.",
        chatNote:
          "Para chatear con el modelo entrenado por su nombre, activa el cambio automático de modelo o primero cárgalo en la página Chat.",
        cancelTitle: "Cancelar un trabajo",
        cancelDescription:
          "Detiene un trabajo en cola o en ejecución. No forma parte del ejemplo anterior.",
      },
      decisionApi: {
        title: "API de decisiones",
        docs: "Documentación",
        docsLabel: "Abrir la documentación de la API de decisiones",
        description:
          "Responde preguntas de sí/no, de opción múltiple y de puntuación sobre texto con un modelo en esta computadora o un modelo de decisiones de Conexiones. Funciona con el SDK de TypeSafe.",
        enable: "Atender solicitudes",
        enableDescription:
          "Atiende /v1/systemone. Al activarlo se descarga el modelo.",
        enableRemoteDescription: "Atiende /v1/systemone.",
        lockedByEnv: "Definido por {name}.",
        model: "Modelo",
        thisMachine: "Esta computadora",
        modelMultilingual: "Multilingüe",
        modelEnglish: "Inglés",
        modelTypedDecisions: "Decisiones tipadas",
        modelClefFlash: "Clef-flash",
        modelClef: "Clef",
        recommended: "Recomendado",
        device: "Ejecutar en",
        deviceDescription:
          "La GPU responde más rápido y libera su memoria cuando el modelo se descarga de la memoria.",
        deviceCpu: "CPU",
        deviceGpu: "GPU",
        backend: "Entorno de ejecución",
        backendAuto: "Automático",
        backendDescription:
          "Automático sirve Clef mediante llama.cpp cuando el modelo tiene una versión GGUF y, si no, recurre a PyTorch. llama.cpp también lee imágenes.",
        backendDescriptionMlx:
          "Automático sirve Clef mediante MLX en Apple Silicon y mediante llama.cpp cuando solo la versión GGUF del modelo está cargada o descargada. MLX lee imágenes solo con modelos Clef. llama.cpp también lee imágenes.",
        backendStatus: "Entorno de ejecución: {backend}",
        backendNone: "no disponible",
        mediaImages: "Lee texto e imágenes.",
        mediaText: "Solo lee texto.",
        checking: "Verificando…",
        notDownloaded: "No descargado · {size}",
        downloading: "Descargando…",
        downloaded: "Descargado · se carga con la primera solicitud",
        ready: "Listo · se carga con la primera solicitud",
        installing: "Instalando…",
        loading: "Cargando…",
        loadedOn: "Cargado en {device}",
        download: "Descargar",
        downloadConfirmTitle: "¿Descargar Laya {model}?",
        downloadConfirmTitleModel: "¿Descargar {model}?",
        downloadConfirmBody:
          "La API de decisiones necesita este modelo para responder solicitudes. Ocupa unos {size} y se descarga una sola vez en tu caché de Hugging Face.",
        unload: "Descargar de memoria",
        downloadBusy:
          "Ya se está descargando un modelo de la API de decisiones.",
        downloadFailed: "No se pudo iniciar la descarga.",
        saveFailed: "No se pudo guardar la opción de la API de decisiones.",
        loadError:
          "No se pudo cargar la configuración de la API de decisiones.",
        sendsTo: "Las solicitudes se envían a {provider}.",
        connectionMissing:
          "Esta conexión ya no existe o no tiene modelos de decisiones. Elige otro modelo.",
        addConnection:
          "Para usar un modelo de decisiones alojado en la nube, agrega uno en ",
        connectionsLink: "Conexiones",
        addConnectionSuffix: ".",
      },
      mcp: {
        title: "Acceso de agentes (MCP)",
        description:
          "Permite que agentes de programación como Claude Code y Codex usen Unsloth mediante MCP. Los agentes inician sesión con una clave de API de Unsloth de esta página.",
        enable: "Permitir conexiones de agentes",
        enableDescription:
          "Sirve /mcp/ a las solicitudes que incluyen una clave de API de Unsloth.",
        lockedByEnv: "Definido por {name}.",
        loadError: "No se pudo cargar la configuración de acceso de agentes.",
        saveError: "No se pudo guardar la opción de acceso de agentes.",
        agent: "Agente",
        exportKeyHint:
          "Define {name} con una clave de API de Unsloth de esta página antes de iniciar el agente.",
        configFileHint: "Agrega esto a {path}.",
        keywords:
          "mcp protocolo de contexto de modelo agentes agente herramientas model context protocol agents claude codex tools",
      },
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
      audioCppVersion: "Versión de audio.cpp",
      mlxVersion: "Versión de MLX",
      whisperCppVersion: "Versión de whisper.cpp",
      engines: {
        title: "Motores",
        description:
          "Los entornos de ejecución que usa Unsloth para ejecutar modelos. Las actualizaciones se instalan en segundo plano, sin reiniciar.",
        updatesLabel: "Actualizaciones de motores",
        notInstalled: "No instalado",
        available: "{version} disponible",
        upToDate: "Actualizado",
        updatesAvailable: "Hay actualizaciones disponibles",
        checking: "Buscando actualizaciones...",
        checkFailed:
          "No se pudieron buscar actualizaciones. Vuelve a intentarlo más tarde.",
        unsupported:
          "Los motores de esta instalación se administran fuera de Unsloth.",
        checkForUpdates: "Buscar actualizaciones",
        update: "Actualizar",
        updating: "Actualizando...",
        updated: "Motores actualizados.",
        updateFailed: "Error al actualizar: {error}",
      },
      hardware: "Hardware",
      gpu: "GPU",
      gpuIntegrated: "integrada",
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
        checkingInstall: "Verificando cómo se instaló Unsloth...",
        installIntro: "Para instalar o actualizar Unsloth:",
        localUpdateHeading: "Actualización local",
        installCommandUnix: "Comando de instalación para macOS/Linux",
        installCommandWindows: "Comando de instalación para Windows",
        localInstallDetected:
          "Se detectó una instalación local. Actualiza desde tu checkout original para no reemplazarla con PyPI.",
        pullThenUpdate:
          "Descarga los últimos cambios y luego ejecuta el instalador local:",
        gitPullCommand: "comando git pull",
        localInstallerCommand: "comando del instalador local",
        sourceInstallDetected:
          "Se detectó una instalación del paquete desde código fuente o VCS. Reinstala desde la ruta local original o la URL de Git.",
        repoCheckoutFallback:
          "Si aún tienes el checkout del repositorio, ejecuta el instalador local desde ahí:",
        restartAfterUpdate: "Reinicia Unsloth después de actualizar.",
        desktopManaged:
          "La app de escritorio busca nuevas versiones automáticamente. También puedes buscar o actualizar aquí en cualquier momento.",
        desktopReady: "Actualizaciones de la app de escritorio",
        desktopReadyDescription:
          "Verifica si hay disponible una versión más reciente de la app de escritorio.",
        desktopChecking: "Buscando actualizaciones",
        desktopCheckingDescription: "Esto suele tardar unos segundos.",
        desktopAvailable:
          "La versión {version} de la app de escritorio está disponible",
        desktopAvailableDescription:
          "Actualiza ahora y la app de escritorio se reiniciará cuando termine.",
        desktopExternalServer:
          "La aplicación se conectó a un servidor de Unsloth que ya estaba en ejecución y no puede actualizarlo. Detén ese servidor, cierra la aplicación de escritorio y vuelve a abrirla para actualizar.",
        desktopManualInstall:
          "Abre la página de versiones para instalar el paquete más reciente para Linux.",
        desktopCheckFailed: "No se pudieron buscar actualizaciones",
        desktopCheckFailedDescription:
          "Verifica tu conexión y vuelve a intentarlo.",
        desktopCurrent: "La app de escritorio está actualizada",
        desktopCurrentDescription:
          "Unsloth seguirá buscando actualizaciones automáticamente.",
        checkForUpdates: "Buscar actualizaciones",
        checkAgain: "Buscar de nuevo",
        retryCheck: "Volver a intentarlo",
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
        docsLinux: "Linux",
        docsDocker: "Docker",
        docsAmd: "AMD",
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
      modelTooltip: "El modelo base que quieres ajustar.",
      methodLabel: "Método",
      methodTooltip:
        "Cómo se entrena el modelo. LoRA y QLoRA actualizan adaptadores pequeños en lugar de todos los pesos.",
      trainAsLabel: "Entrenar como",
      trainAsTooltip:
        "Un modelo de lenguaje escribe texto. Un modelo de decisión elige una de las opciones que le das, con una probabilidad, y se ejecuta en la API de decisiones.",
      trainAsLanguage: "Modelo de lenguaje",
      trainAsDecision: "Modelo de decisión",
      checkpointLabel: "Checkpoint",
      checkpointTooltip:
        "El checkpoint de Laya que se ajustará. Multilingüe sirve para la mayoría de los conjuntos de datos.",
      datasetLabel: "Conjunto de datos",
      datasetTooltip:
        "Los datos de entrenamiento que se usan para ajustar el modelo.",
      hfTokenDescription:
        "Necesario para modelos y conjuntos de datos restringidos o privados.",
      uploadLocalLabel: "Subir un archivo local",
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
      vram: "VRAM",
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
        "Este modelo aún no está en este dispositivo. El entrenamiento lo descargará automáticamente.",
      noticeModelPartial:
        "El entrenamiento completará la descarga parcial del modelo antes de cargarlo.",
      noticeDatasetDownload:
        "Este conjunto de datos aún no está en este dispositivo. El entrenamiento lo descargará automáticamente.",
      noticeDatasetPartial:
        "El entrenamiento completará la descarga parcial del conjunto de datos antes de leerlo.",
      noticeTransformersUpgrade:
        "Ninguna versión instalada de transformers admite todavía esta arquitectura. Al iniciar la ejecución se ofrecerá instalar primero transformers {version}.",
      noticeSixteenBitOnly:
        "Esta arquitectura se entrena en LoRA de 16 bits: 4 bits no está disponible para ella, así que la ejecución necesita mucha más VRAM que QLoRA.",
      noticeInstallSwitchesSixteenBit:
        "Instalar esa versión en lugar de conservar el código propio del modelo cambia esta ejecución a LoRA de 16 bits, que necesita mucha más VRAM que QLoRA.",
      advancedSettings: "Configuración avanzada",
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
      openDataRecipes: "Abrir Data Recipes",
      searchingHub: "Buscando en Hugging Face…",
      noDatasetsFound: "No se encontraron conjuntos de datos.",
      tokenRejectedTitle: "Hugging Face rechazó el token",
      tokenRejectedBody:
        "Actualiza tu token en Configuración → General y vuelve a intentarlo.",
      hubUnreachable: "No se pudo conectar con Hugging Face",
      cantUseDataset: "No se puede usar el conjunto de datos",
      reasonInvalidHubId:
        "Ingresa un ID válido de conjunto de datos de Hugging Face: repo o propietario/repo, sin puntos ni guiones consecutivos y sin el sufijo .git (máximo 96 caracteres por parte).",
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
        "Actualiza tu token en Configuración → General y vuelve a intentarlo.",
      tokenRejectedAnonymousBody:
        "Los modelos públicos se muestran sin él. Actualiza tu token en Configuración → General para acceder a modelos privados y restringidos.",
      hubUnreachable: "No se pudo conectar con Hugging Face",
      cantUseModel: "No se puede usar el modelo para entrenar",
      reasonTypeMismatch:
        "Este modelo no coincide con el tipo de entrenamiento seleccionado en el paso anterior.",
      reasonEmptyId: "Ingresa un ID de modelo o una ruta de modelo local.",
      reasonGguf: "Los modelos GGUF no se pueden usar para entrenar.",
      reasonAdapter:
        "Las salidas de adaptadores no se pueden usar como modelos base de entrenamiento.",
      reasonNotTrainable: "Este modelo del dispositivo no se puede entrenar.",
      reasonUnsupportedFormat:
        "Este formato de modelo no es compatible con el entrenamiento.",
      reasonInvalidHubId:
        "Ingresa un ID válido de modelo de Hugging Face: repo o propietario/repo, sin puntos ni guiones consecutivos y sin el sufijo .git (máximo 96 caracteres por parte).",
      sourceModelsFolder: "Carpeta de modelos",
      sourceHfCache: "Caché de HF",
      sourceLmStudio: "LM Studio",
      sourceOllama: "Ollama",
      sourceHermes: "Hermes",
      sourceCustomFolder: "Carpeta personalizada",
      sourceLocalModel: "Modelo local",
      vramOomBadge: "OOM",
      vramTightBadge: "Ajustado",
      vramNeeds: "Necesita ~{est} GB de VRAM (GPU: {total} GiB)",
      vramTight: "~{est} GB de VRAM (ajustado para {total} GiB)",
      vramApprox: "~{est} GB de VRAM",
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
        hint: "Entrena todos los pesos. Máxima calidad, pero requiere la mayor cantidad de VRAM.",
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
    goToLibrary: "Ir a la Biblioteca",
    imageTraining: "Entrenamiento de imágenes",
    goToImageTraining: "Entrenar LoRA de imagen",
    loadingRuntime: "Cargando el entorno de entrenamiento...",
    checkingSupport: "Verificando si esta computadora admite entrenamiento...",
    backToHistory: "Volver al historial",
    dataset: {
      selectors: {
        subset: "Subconjunto",
        subsetTooltip:
          "Selecciona qué subconjunto (configuración) del conjunto de datos usar.",
        trainSplit: "División de entrenamiento",
        trainSplitTooltip:
          "Selecciona qué división usar para el entrenamiento.",
        evaluationSplit: "División de evaluación",
        evaluationSplitTooltip:
          "Selecciona qué división usar para la evaluación. Ninguna significa que no habrá evaluación durante el entrenamiento.",
        selectSubset: "Selecciona un subconjunto...",
        selectSplit: "Selecciona una división...",
        none: "Ninguna",
        loading:
          "Cargando configuraciones y divisiones del conjunto de datos...",
        manualTitle: "Ingresa manualmente las opciones del conjunto de datos",
        manualDescription:
          "Ingresa los nombres exactos de la configuración y la división de Hugging Face que quieres usar.",
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
        "{file} ocupa {size}. Las subidas para entrenamiento admiten hasta {limit}.",
      documentRedirect: {
        title: "Primero hay que convertir este archivo",
        genericFile: "Este archivo",
        description:
          "{file} es material de origen, no un conjunto de datos listo para entrenar. Usa Data Recipes para convertir el documento en un conjunto de datos y luego vuelve aquí para hacer el fine-tuning.",
        nextStepTitle: "Siguiente paso recomendado",
        nextStepDescription:
          "Abre Learning Recipes y empieza con una receta basada en documentos, como PDF grounded QA.",
        openAction: "Abrir Learning Recipes",
      },
      evalDatasetDescription:
        "Opcional. Si no se proporciona, se separará una pequeña parte de los datos de entrenamiento.",
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
            "Establece Pasos máximos > 0: los conjuntos de datos en streaming no tienen una longitud conocida.",
          trainOnCompletions: 'Desactiva "Solo respuestas del asistente".',
          evalSplit:
            "Elige una división de evaluación aparte: la evaluación está activada, pero no hay una división de evaluación distinta configurada.",
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
          evaluation: "evaluación (necesita una división de evaluación aparte)",
        },
        notifications: {
          turnedOffMaxSteps:
            "Streaming desactivado: el streaming necesita un valor fijo de Pasos máximos > 0.",
          adjusted:
            "Ajustado para streaming. Opciones incompatibles desactivadas: {options}.",
          needsMaxSteps:
            "El streaming necesita un valor fijo de Pasos máximos (los conjuntos de datos en streaming no tienen una longitud conocida). Primero establece Pasos máximos > 0.",
          enabledAdjusted:
            "Streaming activado. Opciones incompatibles desactivadas: {options}.",
          disabledForDetectedModality:
            "El streaming se desactivó porque los conjuntos de datos de imagen y audio requieren una descarga completa. Revisa la opción y vuelve a iniciar.",
        },
      },
      auto: "Automático",
      rawText: "Texto sin procesar",
      trainSplitStart: "Inicio de la división de entrenamiento",
      trainSplitStartTooltip:
        "Entrena solo con un subconjunto de tu división de entrenamiento especificando un índice de fila inicial (inclusivo, base 0). Déjalo vacío para empezar desde la primera fila.",
      trainSplitEnd: "Fin de la división de entrenamiento",
      trainSplitEndTooltip:
        "Último índice de fila que se incluye de la división de entrenamiento (inclusivo, base 0). Por ejemplo, pon Inicio en 0 y Fin en 99 para entrenar con las primeras 100 filas. Déjalo vacío para usar todas las filas restantes.",
      endPlaceholder: "Fin",
      clear: "Borrar",
      dropFileOrClick: "Suelta un archivo o haz clic para subirlo",
      uploadDetails: "Detalles de la subida",
      uploadDetailsTooltip:
        "Hasta {limit} por archivo. Los archivos PDF, DOCX y TXT no son conjuntos de datos listos para entrenar, así que primero conviértelos en Learning Recipes.",
      viewDataset: "Ver conjunto de datos",
      previewTokens: "Vista previa de tokens",
      tokenPreview: {
        title: "Vista previa de tokens",
        description:
          "Cómo {model} tokeniza estos datos para el entrenamiento. Solo se descarga el tokenizador.",
        datasetTab: "Muestras del conjunto de datos",
        textTab: "Tu texto",
        trained: "Entrenado",
        masked: "Enmascarado",
        completionsOnly: "Solo respuestas del asistente",
        allTokens: "Se entrenan todos los tokens",
        responseMarker: "Marcador de respuesta",
        maskingNotApplied:
          "No se encontraron los marcadores del chat, así que se entrenan todos los tokens.",
        loading:
          "Cargando el tokenizador. La primera vista previa tarda un poco más...",
        shuffle: "Mezclar muestras",
        showIds: "Mostrar IDs de tokens",
        copyIds: "Copiar IDs",
        copied: "IDs de tokens copiados",
        sample: "Muestra {index}",
        stats: "{tokens} tokens, {trained} entrenados ({percent}%)",
        truncated:
          "Recortado a {max} de {full} tokens por la longitud de contexto.",
        noTrained:
          "Ningún token de esta fila se entrena, así que el entrenamiento la omite.",
        hiddenTokens: "{count} tokens más que no se muestran",
        multimodal:
          "Las filas de imagen y audio no se muestran aquí. Aun así, puedes probar el tokenizador con tu propio texto.",
        textPlaceholder: "Escribe o pega texto para ver sus tokens...",
        textHint:
          "Con Solo respuestas del asistente, solo se entrena el texto que va después de un marcador del asistente.",
        insertExample: "Insertar ejemplo de chat",
        vocab: "Vocabulario: {count} tokens",
        retry: "Volver a intentarlo",
        packing:
          "El empaquetado está activado: la ejecución une filas en secuencias de la longitud de contexto, así que el punto donde se corta una fila puede diferir de esta vista por fila.",
      },
      uploadFailed: "La subida falló",
      unknownError: "Error desconocido",
      unsupportedFileType: "Tipo de archivo no compatible",
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
          "Detectamos automáticamente la siguiente asignación de columnas mediante heurísticas. Revísala y ajústala con los menús desplegables de los encabezados de columna, o usa AI Assist para una asignación más inteligente.",
        readyDescription:
          "Todo se ve bien. Convertiremos este conjunto de datos automáticamente.",
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
      drafterSettings: "Modelo de borrador para decodificación especulativa",
      drafterTrain:
        "Entrenar un modelo de borrador para decodificación especulativa",
      drafterTrainTooltip:
        "Entrena una pequeña cabeza de borrador después del ajuste fino para que la inferencia sea más rápida en llama.cpp y vLLM; las salidas no cambian.",
      drafterChecking: "Verificando este modelo...",
      drafterUnavailable: "No disponible para este modelo: {reason}",
      drafterRetrainMtp: "Vuelve a entrenar la cabeza MTP de este modelo",
      drafterEagle3:
        "Entrena un modelo de borrador EAGLE-3 (este modelo no tiene cabeza MTP)",
      drafterDataSource: "Datos de entrenamiento",
      drafterDataSourceDataset: "Respuestas del conjunto de datos",
      drafterDataSourceRegenerate:
        "Respuestas del propio modelo (más lento, mejor)",
      drafterMaxExamples: "Ejemplos",
      drafterEpochs: "Épocas",
      drafterLearningRate: "Tasa de aprendizaje",
      drafterLearningRateDefault: "Predeterminado",
      drafterDraftSteps: "Tokens de borrador",
      maxSteps: "Pasos máximos",
      epochs: "Épocas",
      useMaxSteps: "Usar pasos máximos",
      useEpochs: "Usar épocas",
      maxStepsTooltip: "Reemplaza el total de pasos del optimizador.",
      epochsTooltip: "Número de pasadas completas sobre el conjunto de datos.",
      contextLength: "Longitud de contexto",
      contextLengthTooltip:
        "Número máximo de tokens por muestra de entrenamiento.",
      customContextLength: "Ingresa un valor personalizado",
      learningRate: "Tasa de aprendizaje",
      learningRateTooltip:
        "Tamaño de paso para las actualizaciones de pesos. Los valores más bajos entrenan más lento, pero con mayor estabilidad.",
      learningRateDescription:
        "Recomendado: 2e-4 para LoRA, 5e-5 para CPT, 2e-5 para fine-tuning completo",
      embeddingLearningRate: "Tasa de aprendizaje de embeddings",
      embeddingLearningRateTooltip:
        "Solo se usa cuando CPT entrena embed_tokens. Los embeddings se desestabilizan más fácilmente que los pesos de LoRA, por lo que suelen necesitar una LR más pequeña. Déjalo en blanco para usar lr/10; el rango de trabajo típico es de 2x a 10x más pequeño que la LR principal. Auméntalo solo si la adaptación del vocabulario o de los tokens de dominio es demasiado lenta.",
      rank: "Rango",
      rankTooltip:
        "Dimensión de las matrices de bajo rango. Mayor = más capacidad.",
      alpha: "Alpha",
      alphaTooltip:
        "Factor de escala para las actualizaciones de LoRA. Normalmente 2x el rango.",
      dropout: "Dropout",
      dropoutTooltip:
        "Probabilidad de dropout de las capas de LoRA para reducir el sobreajuste.",
      visionLayers: "Capas de visión",
      languageLayers: "Capas de lenguaje",
      attentionModules: "Módulos de atención",
      mlpModules: "Módulos MLP",
      targetModules: "Módulos objetivo",
      enableLora: "LoRA",
      trainWithLora: "Entrenar con LoRA",
      stableRank: "LoRA con rango estabilizado",
      memoryEfficient: "LoRA inicializado con SVD",
      weightDecomposed: "LoRA con multiplicadores",
      notSupportedAppleSilicon: "No compatible con Apple Silicon",
      doraNeedsVisionLayersOff:
        "Desactiva el entrenamiento de las capas de visión para usar DoRA",
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
        "Cómo cambia la tasa de aprendizaje durante el entrenamiento. Lineal disminuye de forma constante; coseno disminuye siguiendo una curva.",
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
      batchSize: "Tamaño de lote",
      batchSizeTooltip:
        "Muestras procesadas por paso. Un valor más alto usa más VRAM.",
      gradAccum: "Acum. de gradiente",
      gradAccumTooltip:
        "Simula tamaños de lote más grandes sin VRAM adicional.",
      weightDecay: "Decaimiento de pesos",
      weightDecayTooltip: "Regularización L2 para prevenir el sobreajuste.",
      warmupSteps: "Pasos de calentamiento",
      warmupStepsTooltip:
        "Aumenta gradualmente la LR al inicio del entrenamiento para dar estabilidad.",
      scheduleEpochsTooltip:
        "Número de pasadas completas sobre el conjunto de datos. Usa 0 para ejecutar por pasos máximos.",
      saveSteps: "Pasos entre guardados",
      saveStepsTooltip: "Guarda un checkpoint cada N pasos. 0 para desactivar.",
      evalSteps: "Pasos entre evaluaciones",
      evalStepsTooltip:
        "Fracción del total de pasos de entrenamiento entre evaluaciones (0-1). Usa 0 para desactivar la evaluación. P. ej. 0.01 = evaluar cada 1 % de los pasos.",
      earlyStoppingPatience: "Paciencia de parada temprana",
      earlyStoppingPatienceTooltip:
        "Detiene el entrenamiento tras este número de evaluaciones seguidas sin una pérdida de evaluación menor y conserva los pesos de la mejor evaluación. Requiere evaluación (pasos entre evaluaciones mayor que 0). 0 para desactivar.",
      seed: "Semilla",
      seedTooltip: "Semilla aleatoria para reproducibilidad.",
      offloadLayers: "Trasladar capas",
      offloadLayersTooltip:
        "Mantiene capas del decodificador en la RAM del sistema y transfiere cada una a la GPU justo antes de ejecutarla, para entrenar con LoRA un modelo más grande que tu VRAM. Auto traslada las mínimas que hagan falta; un número traslada exactamente esa cantidad. Requiere gradient checkpointing.",
      offloadOff: "Desactivado",
      offloadAuto: "Auto",
      offloadCount: "Número",
      offloadVramBudget: "Presupuesto de VRAM (GiB)",
      offloadVramBudgetTooltip:
        "La VRAM máxima que puede usar este entrenamiento. Auto traslada las capas necesarias para caber, y dos entrenamientos pueden compartir una tarjeta. Vacío usa toda la tarjeta.",
      offloadWholeCard: "toda la tarjeta",
      offloadVramBudgetGpu: "Presupuesto de VRAM GPU {index} (GiB)",
      prefetchDepth: "Profundidad de precarga",
      prefetchDepthTooltip:
        "Cuántas capas trasladadas se copian antes de la que se ejecuta. Auto empieza en 1 y solo mantiene más si reduce de forma medible la espera de copias.",
      offloadPanelTitle: "Capas trasladadas",
      offloadPanelGpu: "En la GPU",
      offloadPanelHost: "En la RAM del sistema",
      offloadPanelCopying: "Copiando",
      offloadPanelSwapped: "trasladadas",
      offloadPanelStall: "esperando copias",
      offloadPanelCopy: "copia por capa",
      offloadPanelCompute: "cálculo por capa",
      offloadPanelVram: "VRAM",
      offloadPanelDepth: "profundidad de precarga",
      offloadPanelPinned: "fijada",
      offloadPanelSweepNote:
        "La ventana móvil muestra el orden en que se traen las capas, ralentizado para que se vea; los números se miden en el último paso.",
      offloadPanelCard: "GPU {index}",
      offloadPanelCardUnknown: "Tarjeta desconocida",
      gradCheckpoint: "Checkpoint de gradiente",
      gradCheckpointTooltip:
        "Intercambia cómputo por memoria recalculando las activaciones.",
      none: "Ninguno",
      standard: "Estándar",
      enablePacking: "Activar packing",
      assistantCompletionsOnly: "Solo completaciones del asistente",
      readMore: "Leer más",
    },
    training: {
      duplicateFailed: "No se pudo duplicar la ejecución",
      duplicateDraftChanged:
        "La configuración del entrenamiento cambió durante la carga. Vuelve a intentar duplicarla.",
      duplicateNoModel:
        "Esta ejecución no tiene una configuración de modelo guardada.",
      startTraining: "Iniciar entrenamiento",
      starting: "Iniciando...",
      loadingModel: "Cargando modelo...",
      checkingDataset: "Verificando conjunto de datos...",
      chooseModel: "Elige un modelo",
      chooseDataset: "Elige un conjunto de datos",
      chooseModelAndDataset: "Elige un modelo y un conjunto de datos",
      validation: {
        s3MultimodalUnsupported:
          "Los conjuntos de datos de S3 todavía no son compatibles con el entrenamiento de visión.",
        s3BucketRequired: "Ingresa primero el nombre de un bucket de S3.",
        s3CredentialsRequired:
          "Proporciona las claves de acceso de S3 o activa el rol de IAM.",
        modelRequired: "Selecciona primero un modelo base.",
        learningRatePositive: "Ingresa una tasa de aprendizaje mayor que cero.",
        embeddingLearningRateRange:
          "Ingresa una tasa de aprendizaje de embeddings mayor que 0 y menor que 1.",
        decisionColumnsMissing:
          "Los modelos de decisión necesitan las columnas state, questions y gold (o answers). Faltan: {columns}.",
        decisionOwnerOnly:
          "Solo el propietario de Unsloth puede ajustar modelos de decisión.",
        hfDatasetRequired:
          "Selecciona primero un conjunto de datos de Hugging Face.",
        hfDatasetSplitRequired:
          "Selecciona o ingresa primero una división de entrenamiento.",
        localDatasetRequired: "Selecciona primero un conjunto de datos local.",
        unsupportedDatasetSource: "Origen de conjunto de datos no compatible.",
      },
      startFailed: "No se pudo iniciar el entrenamiento",
      startUnconfirmed:
        "Unsloth no pudo confirmar si el entrenamiento se inició. Se está verificando el estado en segundo plano.",
      stopFailed: "No se pudo detener el entrenamiento",
      trainingStillActiveTitle: "El entrenamiento sigue activo",
      stopBeforeConfig:
        "Detén primero el entrenamiento y vuelve después a la configuración.",
      resumeFailed: "No se pudo reanudar el entrenamiento",
      resumeFailedTitle: "Error al reanudar el entrenamiento",
      resumeUnavailable:
        "Solo se pueden reanudar las ejecuciones detenidas o con errores que tengan un checkpoint guardado.",
      modelUnverified:
        "No se pudo verificar la configuración de este modelo. Verifica la conexión o el token de Hugging Face y vuelve a intentarlo.",
      legacyDatasetScriptUnsupported:
        "Este conjunto de datos de Hub depende de un script personalizado antiguo y no es compatible con este flujo de entrenamiento.",
      hfModelAccessDenied:
        "Hugging Face denegó el acceso a este modelo. Agrega un token de Hugging Face válido con acceso al repositorio, acepta las condiciones de acceso necesarias y vuelve a intentarlo.",
      hfModelVerificationRateLimited:
        "La verificación del modelo de Hugging Face está limitada temporalmente. Vuelve a intentarlo en breve.",
      hfModelVerificationFailed:
        "No se pudo verificar el modelo de Hugging Face. Verifica el ID del repositorio y tu token de acceso.",
      hfModelMetadataUnavailable:
        "Los metadatos del modelo de Hugging Face no están disponibles temporalmente. Vuelve a intentarlo antes de iniciar el entrenamiento.",
      datasetUnverified:
        "No se pudo verificar si el conjunto de datos es compatible con este modelo. Verifica la conexión o el token de Hugging Face; al iniciar el entrenamiento se volverá a intentar la verificación.",
      setupChanged:
        "La configuración del entrenamiento cambió mientras se verificaba. Revísala y vuelve a iniciar el entrenamiento.",
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
        "Elige cómo quieres detener la ejecución de entrenamiento actual. “Detener y guardar” crea un checkpoint desde el que podrás reanudarla más adelante; si la detienes sin guardar, no podrás reanudarla.",
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
      relativeJustNow: "hace un momento",
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
      trainingLossDescription:
        "Controla las superposiciones y el suavizado EMA.",
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
      evalChartWillAppear: "La gráfica aparecerá cuando se alcance eval_steps",
      setEvalDatasetAndSteps:
        "Configura el conjunto de datos de evaluación y eval_steps para seguir la pérdida de evaluación",
    },
    progress: {
      title: "Progreso del entrenamiento",
      liveMetrics: "Métricas de entrenamiento en vivo",
      exportGguf: "Exportar a GGUF",
      useInDecisionApi: "Usar en la API de decisiones",
      decisionApiEnabled: "La API de decisiones ahora usa {name}.",
      decisionApiFailed:
        "No se pudo cambiar la API de decisiones a este modelo.",
      openConfig: "Abrir configuración de entrenamiento",
      configLabel: "Configuración de entrenamiento",
      hyperparams: "Hiperparámetros",
      epochs: "Épocas",
      batchSize: "Tamaño de lote",
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
        trainingDrafter: "Entrenando drafter",
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
      gettingReady: "> Estamos dejando todo listo para tu ejecución...",
      resumingTraining: "Reanudando entrenamiento...",
      startingTraining: "iniciando entrenamiento...",
      dataset: "Conjunto de datos",
      datasetStreaming: "Conjunto de datos: streaming (sin descarga completa)",
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
    tooLarge:
      "Más grande que la VRAM, una parte se trasladará a la CPU. Una cuantización más pequeña es más rápida",
  },
  skills: {
    title: "Skills",
    description:
      "Las skills se detectan en tus carpetas de agentes estándar. Actívalas aquí y luego escribe @ en el chat para mencionar una.",
    precedence: "~/.agents/skills tiene prioridad sobre ~/.claude/skills.",
    refresh: "Actualizar",
    empty:
      "No se encontraron skills. Agrega una carpeta con SKILL.md en ~/.agents/skills o ~/.claude/skills y actualiza.",
    sourceAgents: "Agents",
    sourceClaude: "Claude",
    sourceBundled: "Incluida",
    shadowed: "Oculta",
    invalid: "No válida",
    compatibility: "Compatibilidad: {value}",
    shadowedBy: "Otra skill ({source}) con este nombre tiene prioridad.",
    enable: "Activar {name}",
    disable: "Desactivar {name}",
    updateError: "No se pudo actualizar la skill",
    newSkill: "Nueva skill",
    import: "Importar",
    delete: "Eliminar {name}",
    createDescription:
      "Se guarda como SKILL.md en una carpeta nueva dentro de ~/.agents/skills y se activa en cuanto existe.",
    nameLabel: "Nombre",
    namePlaceholder: "p. ej. code-arithmetic",
    nameHint:
      "Minúsculas, números y guiones simples. También es el nombre de la carpeta.",
    nameInvalid:
      "Usa de 1 a 64 minúsculas, números o guiones simples, sin empezar ni terminar con guion.",
    descriptionLabel: "Descripción",
    descriptionPlaceholder:
      "Usar cuando el usuario pida cálculos aritméticos. Calcularlos siempre con la herramienta de código.",
    descriptionHint:
      "La línea que el modelo ve en su catálogo de skills. Con ella decide cuándo leer la skill.",
    instructionsLabel: "Instrucciones",
    instructionsPlaceholder: "Qué hacer cuando aplica esta skill, en Markdown.",
    instructionsHint:
      "El cuerpo de SKILL.md, que el modelo lee completo cuando elige la skill.",
    create: "Crear skill",
    save: "Guardar cambios",
    created: "{name} creada",
    saved: "{name} guardada",
    deleted: "{name} eliminada",
    importedOne: "{count} skill agregada",
    importedMany: "{count} skills agregadas",
    importPartial: "{added} de {total} skills agregadas",
    saveError: "No se pudo guardar la skill",
    openError: "No se pudo abrir la skill",
    deleteError: "No se pudo eliminar la skill",
    importError: "No se pudo importar la skill",
    exportError: "No se pudo exportar la skill",
    deleteTitle: "Eliminar skill",
    deleteDescription:
      '¿Eliminar "{name}" y todo lo que hay en su carpeta dentro de ~/.agents/skills? Esto no se puede deshacer.',
    search: "Buscar skills por nombre o descripción…",
    noMatch: "Ninguna skill coincide con “{query}”",
    clearSearch: "Borrar búsqueda",
    readOnlyClaude:
      "Solo lectura: está en ~/.claude/skills. Edita el archivo allí.",
    readOnlyBundled: "Solo lectura: incluida con Unsloth.",
    readOnlyLinked:
      "Solo lectura: esta entrada es un enlace. Edita los archivos en su ubicación original.",
    characters: "{count} caracteres",
    unsaved: "cambios sin guardar",
    revert: "Revertir",
    enabledLabel: "Activada",
    licenseLabel: "Licencia",
    compatibilityLabel: "Compatibilidad",
    allowedToolsLabel: "Herramientas permitidas",
    sectionAgents: "Tus skills",
    sectionAgentsHint:
      "Guardadas en ~/.agents/skills. Créalas, edítalas y elimínalas aquí.",
    sectionClaude: "Skills de Claude",
    sectionClaudeHint:
      "Encontradas en ~/.claude/skills. Solo lectura aquí; edita los archivos allí.",
    sectionBundled: "Incluidas",
    sectionBundledHint: "Vienen con Unsloth. Solo lectura.",
    linked: "Enlace",
    discardTitle: "¿Descartar los cambios?",
    discardDescription:
      "Los cambios en {name} que no se hayan guardado se perderán.",
    discard: "Descartar",
    mentions: "Skills",
    manage: "Administrar skills",
    turnOff: "Desactivar skills",
    bulkActions: "Acciones de skills",
    enableAll: "Activar todas",
    disableAll: "Desactivar todas",
    resetAll: "Restablecer valores predeterminados",
    resetTitle: "¿Restablecer todas las skills?",
    resetDescription:
      "Cada skill vuelve a como está en una instalación nueva: tus skills y las de Claude activadas, las incluidas desactivadas. Se borran tus elecciones de activar y desactivar.",
    reset: "Restablecer",
  },
  scheduledTasks: {
    title: "Programadas",
    description:
      "Ejecuta prompts de forma programada con cualquiera de tus modelos.",
    newTask: "Nueva tarea",
    upcoming: "Próximas",
    columns: {
      repeats: "Repetición",
      time: "Hora",
      nextRun: "Próxima ejecución",
    },
    monthlyDay: "Mensual, día {day}",
    filterBy: "Filtrar por",
    filter: "Filtrar tareas",
    active: "Activas",
    noMatches: "No hay tareas programadas",
    search: "Buscar tareas programadas",
    newResult: "Nuevo resultado",
    sentBy: "Enviado por una tarea programada",
    chatMark: "Chat de una tarea programada",
    taskMissing: "Esta tarea ya no existe.",
    emptyTitle: "Aún no hay tareas programadas",
    emptyDescription:
      "Un resumen diario, un recordatorio semanal o cualquier otra cosa que quieras hacer de forma programada.",
    createFirst: "Crear una tarea",
    loadFailed: "No se pudieron cargar las tareas programadas.",
    retry: "Reintentar",
    paused: "En pausa",
    completed: "Completada",
    running: "En ejecución",
    lastRun: "Última ejecución {when}",
    lastRunFailed: "La última ejecución falló {when}: {reason}",
    openChat: "Abrir chat",
    runNow: "Ejecutar ahora",
    pause: "Pausar",
    resume: "Reanudar",
    edit: "Editar",
    options: "Opciones de la tarea",
    schedule: {
      hourly: "Cada hora",
      everyHours: "Cada {hours} horas",
      once: "Una vez, el {date} a las {time}",
      daily: "Todos los días a las {time}",
      weekdays: "Entre semana a las {time}",
      weekly: "Cada {days} a las {time}",
      monthly: "Cada mes el día {day} a las {time}",
    },
    form: {
      hourly: "Cada hora",
      intervalHours: "Intervalo (horas)",
      hourlyNote: "La primera ejecución ocurre un intervalo después de guardar. Cambiar el intervalo o reanudar reinicia la espera.",
      createTitle: "Programar una tarea",
      editTitle: "Editar tarea programada",
      name: "Nombre",
      namePlaceholder: "Nombre (opcional)",
      instructions: "Instrucciones",
      instructionsHint: "Se envían al modelo cada vez que se ejecuta la tarea.",
      instructionsPlaceholder: "Instrucciones…",
      repeat: "Repetir",
      once: "Una vez",
      daily: "Diaria",
      weekdays: "Entre semana",
      weekly: "Semanal",
      monthly: "Mensual",
      date: "Fecha",
      time: "Hora",
      days: "Días",
      dayOfMonth: "Día del mes",
      lastDayNote: "En los meses sin este día, se ejecuta el último día.",
      advanced: "Avanzado",
      notifications: "Notificaciones",
      notifyAll: "Cada ejecución",
      notifyFailures: "Solo si falla",
      notifyNone: "Desactivadas",
      archiveSuccessful: "Archivar ejecuciones correctas",
      project: "Proyecto",
      projectNone: "Ninguno",
      model: "Modelo",
      modelLoaded: "El modelo que esté cargado",
      modelOnDevice: "En este dispositivo",
      modelNote:
        "Si este modelo no está cargado cuando se ejecuta la tarea, Unsloth pregunta primero.",
      modelNoteAllowed:
        "Cambia a este modelo solo mientras no usas Unsloth y luego vuelve al que tenías.",
      modelEmpty:
        "Descarga un modelo en Model Hub o agrega una conexión para elegir otro.",
      modelMissing: "{model} (no está en este dispositivo)",
      allowLoad: "Cambiar automáticamente cuando no esté",
      allowLoadHint:
        "Cuando toca esta tarea y no has usado Unsloth en 5 minutos, carga este modelo sin preguntar, ejecuta la tarea y luego vuelve al modelo que tenías. Si estás usando Unsloth o entrenando en ese momento, la ejecución se omite y tu modelo no se toca. Si está desactivado, Unsloth pregunta antes de cargarlo.",
      reasoning: "Esfuerzo de pensamiento",
      reasoningDefault: "Predeterminado",
      reasoningOff: "Desactivado",
      reasoningLowest: "Mínimo",
      reasoningHighest: "Máximo",
      create: "Crear",
    },
    deleteDialog: {
      title: "¿Eliminar la tarea programada?",
      description:
        "“{title}” dejará de ejecutarse. Los chats que ya creó se conservan.",
    },
    errors: {
      noModel:
        "No había ningún modelo cargado. Carga un modelo en el chat y vuelve a ejecutar la tarea.",
      modelNotLoaded:
        "El modelo de la tarea no estaba cargado. Cárgalo en el chat y vuelve a ejecutar la tarea.",
      modelLoadFailed:
        "No se pudo cargar el modelo de la tarea. Verifica que siga descargado en Model Hub.",
      modelBusy:
        "Omitida mientras usabas Unsloth, así que tu modelo no se tocó.",
      connectionUnavailable:
        "La conexión de la tarea no existe o está desactivada. Revísala en Conexiones.",
      timeout: "El modelo tardó demasiado en responder.",
      cancelled: "La ejecución se detuvo antes de terminar.",
      interrupted: "Unsloth se cerró antes de que terminara la ejecución.",
      empty: "El modelo devolvió una respuesta vacía.",
      unknown: "La ejecución falló.",
      saveFailed: "No se pudo guardar la tarea",
      updateFailed: "No se pudo actualizar la tarea",
      deleteFailed: "No se pudo eliminar la tarea",
      runFailed: "No se pudo iniciar la tarea",
    },
    loadPrompt: {
      title: "¿Cargar {model}?",
      description:
        "“{title}” usa {model}, que no está cargado. Ejecutarla cambia el chat a {model} hasta que termine y luego vuelve.",
      always: "No volver a preguntar. Cambiar solo mientras no uso Unsloth.",
      notNow: "Ahora no",
      load: "Cargar y ejecutar",
    },
    toast: {
      finished: "“{title}” está lista",
      failed: "“{title}” falló",
      many: "{count} tareas programadas terminaron",
      open: "Abrir",
      view: "Ver",
    },
    notificationBody:
      "Tu tarea programada terminó. Abre Unsloth para leer la respuesta.",
  },
  library: {
    tabs: {
      ariaLabel: "Secciones de la Biblioteca",
      suggested: "Sugerencias",
      favorites: "Favoritos",
      folders: "Carpetas",
      images: "Imágenes",
      videos: "Videos",
      audio: "Audio",
      models: "Modelos ajustados",
      all: "Todo",
      chats: "Chats",
    },
    chats: {
      sections: {
        chats: "Chats",
        projects: "Proyectos",
        archived: "Archivados",
        ariaLabel: "Secciones de chats",
      },
      search: {
        chats: "Buscar chats",
        projects: "Buscar proyectos",
        archived: "Buscar chats archivados",
        project: "Buscar en “{project}”",
        sections: "Buscar secciones",
      },
      toolbar: {
        show: "Mostrar",
        pinned: "Fijados",
        forks: "Bifurcaciones",
        comparisons: "Comparaciones",
        project: "Proyecto",
        model: "Modelo",
        groupBy: "Agrupar por",
        groupNone: "Sin agrupar",
        groupDate: "Fecha",
        groupProject: "Proyecto",
        pinnedFirst: "Fijados primero",
        newChat: "Nuevo chat",
        newProject: "Nuevo proyecto",
        sortChats: "Número de chats",
        withoutSection: "Sin sección",
      },
      groups: {
        today: "Hoy",
        yesterday: "Ayer",
        week: "Últimos 7 días",
        month: "Últimos 30 días",
      },
      list: {
        created: "Creado",
        lastActive: "Última actividad",
        location: "Ubicación",
        lastModified: "Última modificación",
        contents: "Contenido",
        oneMessage: "1 mensaje",
        messageCount: "{count} mensajes",
        showMore: "Mostrar más ({count})",
      },
      badges: {
        pinned: "Fijado",
        fork: "Bifurcación",
        compare: "Comparación",
      },
      menu: {
        fork: "Bifurcar",
        moveTo: "Mover a proyecto",
        newChatInProject: "Nuevo chat en el proyecto",
        edit: "Editar",
        deleteProject: "Eliminar proyecto",
        newChatInSection: "Nuevo chat en la sección",
      },
      project: {
        oneProject: "1 proyecto",
        projectCount: "{count} proyectos",
      },
      empty: {
        chatsTitle: "Aún no hay chats",
        chatsDescription:
          "Tus conversaciones aparecen aquí, separadas de tus archivos.",
        archivedTitle: "Nada archivado",
        archivedDescription: "Los chats que archives aparecen aquí.",
        sectionsTitle: "Aún no hay secciones",
        sectionsDescription:
          "Las secciones agrupan chats y proyectos en la barra lateral como quieras.",
        sectionTitle: "Aún no hay nada en esta sección",
        sectionDescription:
          "Mueve chats aquí desde su menú o inicia un chat nuevo en esta sección.",
      },
      dialog: {
        renameChat: "Renombrar chat",
        deleteChatDescription:
          "Este chat se eliminará de forma permanente. Esta acción no se puede deshacer.",
        deleteProjectDescription:
          "El proyecto y sus chats ({count}) se eliminarán de forma permanente. Esta acción no se puede deshacer.",
        deleteFilesMany:
          "Eliminar también los archivos y carpetas de sandbox que crearon estos chats.",
        deleteProjectFilesLabel: "Eliminar la carpeta de trabajo del proyecto",
        deleteProjectFiles:
          "La carpeta de trabajo del proyecto se eliminará del disco.",
      },
      toast: {
        projectMoved: "{project} se movió a {section}",
        projectUnfiled: "{project} se quitó de {section}",
        forkFailed: "No se pudo bifurcar",
        projectDeleted: "Se eliminó “{name}”",
        projectDeleteFailed: "No se pudo eliminar el proyecto",
        removedFromSection: "Quitados de la sección: {count}",
      },
      folder: {
        openChat: "Abrir carpeta del chat",
        openChatTitle:
          "Abrir la carpeta que leen y escriben las herramientas de este chat",
        openProject: "Abrir carpeta del proyecto",
        openProjectTitle:
          "Abrir la carpeta que leen y escriben los chats de este proyecto",
        chatHint:
          "Solo la app de escritorio puede abrir la carpeta de archivos de un chat. En el navegador, descarga el archivo desde el resultado de la herramienta que lo escribió.",
        projectHint:
          "Solo la app de escritorio puede abrir la carpeta de un proyecto. En el navegador, descarga los archivos desde los resultados de las herramientas que los escribieron.",
        manyFolders: "Este chat escribió en más de una carpeta.",
        manyFoldersDescription:
          "Ejecutó herramientas antes y después de moverse, así que abre la carpeta desde una tarjeta de herramienta.",
        chatFailed: "No se pudo abrir la carpeta del chat.",
        projectFailed: "No se pudo abrir la carpeta del proyecto.",
        chatMissing: "Este chat aún no tiene carpeta.",
        projectMissing: "Este proyecto aún no tiene carpeta.",
        missingDescription:
          "Se crea la primera vez que una herramienta escribe un archivo.",
        upload: "Subir a la carpeta del chat",
        uploadTitle:
          "Agregar archivos a la carpeta que leen y escriben las herramientas de este chat",
        uploaded: "{name} se agregó a la carpeta del chat",
        uploadedMany: "Se agregaron {count} archivos a la carpeta del chat",
        uploadedDescription:
          "Las herramientas de este chat ya pueden abrirlos. No se agregan a la conversación.",
        uploadExists: "{name} ya está en la carpeta del chat",
        uploadExistsMany: "{count} archivos ya están en la carpeta del chat",
        replace: "Reemplazar",
        uploadFailed: "No se pudo subir {name}",
      },
    },
    empty: {
      suggestedTitle: "Tu biblioteca está vacía",
      suggestedDescription:
        "Los archivos que subas o crees en los chats aparecen aquí.",
      favoritesTitle: "Aún no hay favoritos",
      favoritesDescription:
        "Agrega archivos a tus favoritos para encontrarlos aquí rápidamente.",
      foldersTitle: "Crea tu primera carpeta",
      foldersDescription:
        "Crea carpetas para organizar los elementos de tu biblioteca.",
      imagesTitle: "Aún no hay imágenes",
      imagesDescription: "Las imágenes que subas o generes aparecen aquí.",
      videosTitle: "Aún no hay videos",
      videosDescription: "Los videos que subas o generes aparecen aquí.",
      audioTitle: "Aún no hay audio",
      audioDescription:
        "La voz, la música y las pistas que crees en la página Audio aparecen aquí.",
      modelsTitle: "Aún no hay modelos ajustados",
      modelsDescription:
        "Los modelos que entrenes o exportes en Unsloth aparecen aquí.",
      createFolder: "Crear carpeta",
      generateImage: "Generar una imagen",
      generateVideo: "Generar un video",
      generateAudio: "Generar audio",
      trainModel: "Entrenar un modelo",
      uploadFiles: "Subir archivos",
      dropHere: "o suelta archivos aquí",
      noMatchesTitle: "Sin resultados",
      noMatchesDescription: "Prueba con otra búsqueda o borra los filtros.",
      loadErrorTitle: "No se pudo cargar tu Biblioteca",
      loadErrorFallback: "Algo salió mal.",
      tryAgain: "Volver a intentar",
    },
    sections: {
      items: "Elementos",
    },
    breadcrumb: "Ruta de navegación",
    searchLibrary: "Buscar en la biblioteca",
    searchFolder: "Buscar en la carpeta",
    dropToUpload: "Suelta para subir",
    dropToUploadInto: "Suelta para subir a “{folder}”",
    itemCountOne: "{count} elemento",
    itemCount: "{count} elementos",
    selectItem: "Seleccionar {name}",
    selection: {
      startChat: "Iniciar chat",
      move: "Mover",
      selectAll: "Seleccionar todo",
      countOfTotal: "{count} de {total} seleccionados",
      clear: "Borrar selección",
    },
    menu: {
      moreActions: "Más acciones",
      chatAboutThis: "Chatear sobre esto",
      chatWithModel: "Chatear con este modelo",
      editImage: "Editar imagen",
      addToFavorites: "Agregar a Favoritos",
      regenerateTitle: "Regenerar título",
      removeFromFavorites: "Quitar de Favoritos",
      download: "Descargar",
      addToFolder: "Agregar a carpeta",
      newFolder: "Nueva carpeta",
      noFolder: "Biblioteca (sin carpeta)",
      deleteFolder: "Eliminar carpeta",
    },
    reveal: {
      finder: "Mostrar en el Finder",
      explorer: "Mostrar en el Explorador",
      files: "Mostrar en la carpeta",
    },
    toolbar: {
      filter: "Filtrar",
      source: "Origen",
      fileType: "Tipo de archivo",
      clearFilters: "Borrar filtros",
      workflow: "Creado con",
      uploaded: "Subidos",
      generated: "Generados",
      documents: "Documentos",
      spreadsheets: "Hojas de cálculo",
      presentations: "Presentaciones",
      pdfs: "PDF",
      gridView: "Vista de cuadrícula",
      listView: "Vista de lista",
      grid: "Cuadrícula",
      list: "Lista",
      settings: "Configuración de la Biblioteca",
      sort: "Ordenar",
      sortDefault: "Orden predeterminado",
      sortName: "Nombre",
      sortModified: "Modificado",
      sortSize: "Tamaño",
      sortAscending: "Ascendente",
      sortDescending: "Descendente",
    },
    audio: {
      play: "Reproducir {name}",
      pause: "Pausar {name}",
      playFailed: "No se pudo reproducir {name}",
      fromRun: "De la misma ejecución",
      downloadRun: "Descargar todo",
    },
    create: {
      note: "Nota",
      image: "Imagen",
      video: "Video",
      audio: "Audio",
      model: "Modelo",
      folder: "Carpeta",
      untitledNote: "Nota sin título",
    },
    list: {
      selectAll: "Seleccionar todo",
      name: "Nombre",
      lastActivity: "Última actividad",
      favorite: "Favorito",
      modified: "Modificado {time}",
      opened: "Abierto {time}",
      modifiedColumn: "Modificado",
      justNow: "hace un momento",
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
      fullFineTune: "Fine-tuning completo",
      loraExport: "Exportación LoRA",
      mergedExport: "Exportación fusionada",
      ggufExport: "Exportación GGUF",
      model: "Modelo",
    },
    dialog: {
      create: "Crear",
      renameFolder: "Renombrar carpeta",
      renameFile: "Renombrar archivo",
      deleteTitle: "¿Eliminar “{name}”?",
      deleteManyTitle: "¿Eliminar {count} elementos?",
      deleteManyDescription:
        "Los archivos se eliminan de donde estén guardados. Lo que haya dentro de una carpeta eliminada sube un nivel.",
      deleteFolderIntoParent:
        "Todo su contenido se mueve a “{folder}”. No se elimina ningún archivo.",
      deleteFolderIntoLibrary:
        "Todo su contenido se mueve a tu Biblioteca. No se elimina ningún archivo.",
      deleteUpload: "Esto elimina el archivo de forma permanente.",
      deleteAttachment: "También se quitará del chat al que estaba adjunto.",
      deleteImage: "También se quitará de tu galería de Imágenes.",
      deleteVideo: "También se quitará de tu galería de Video.",
      deleteAudio: "También se quitará de tu galería de Audio.",
      deleteSandbox: "También se quitará del chat que lo creó.",
      deleteModel: "Esto elimina el modelo del disco de forma permanente.",
      unsavedTitle: "Tus cambios no se guardaron",
      unsavedDescription:
        "{reason} Vuelve a intentarlo o descarta los cambios para cerrar el archivo.",
      keepEditing: "Seguir editando",
      discardChanges: "Descartar cambios",
    },
    preview: {
      code: "Código",
      preview: "Vista previa",
      fromChat: "Del chat",
      viewOriginalChat: "Ver el chat original",
      viewInImages: "Ver en Imágenes",
      viewInVideo: "Ver en Video",
      viewInAudio: "Ver en Audio",
      viewOriginalProject: "Ver proyecto original",
      viewTrainingRun: "Ver ejecución de entrenamiento",
      sheetTruncated: "Mostrando parte de esta hoja",
      documentTruncated: "Mostrando el principio de este documento",
      emptyDocument: "Este archivo está vacío.",
      cannotPreview:
        "No se puede previsualizar este archivo aquí. Descárgalo para abrirlo.",
      noPreview: "No hay vista previa para este tipo de archivo.",
      tooLargeToPreview:
        "Este archivo es demasiado grande para previsualizarlo aquí. Descárgalo para abrirlo.",
      startWriting: "Empieza a escribir…",
      type: "Tipo",
      baseModel: "Modelo base",
      size: "Tamaño",
      location: "Ubicación",
      unknown: "Desconocido",
      readOnlyUtf16:
        "Este archivo no es texto UTF-16 válido, así que aquí se abre en modo de solo lectura.",
      readOnlyNotUtf8:
        "Este archivo no es texto UTF-8, así que aquí se abre en modo de solo lectura. Es posible que algunos caracteres no se muestren correctamente.",
    },
    viewer: {
      scale: "Escala",
      fit: "Ajustar",
      moreActionsImage: "Más acciones para esta imagen",
      moreActionsVideo: "Más acciones para este video",
      moreActionsClip: "Más acciones para este clip",
      moreActionsFile: "Más acciones para este archivo",
      openImage: "Abrir imagen",
      openImageNamed: "Abrir imagen: {prompt}",
      openVideo: "Abrir video",
      untitledImage: "Imagen sin título",
      untitledVideo: "Video sin título",
    },
    project: {
      label: "Proyecto",
      newProject: "Nuevo proyecto",
      noProjects: "Aún no hay proyectos",
      alreadyIn: "Ya está en {project}",
      addedTo: "Agregado a {project}",
      createAndAdd: "Crear y agregar",
      addFailedImage: "No se pudo agregar la imagen al proyecto",
      addFailedVideo: "No se pudo agregar el video al proyecto",
      addFailedClip: "No se pudo agregar el clip al proyecto",
      addFailedFile: "No se pudo agregar el archivo al proyecto",
      newProjectImage: "Agregar imagen a un nuevo proyecto",
      newProjectVideo: "Agregar video a un nuevo proyecto",
      newProjectClip: "Agregar clip a un nuevo proyecto",
      newProjectFile: "Agregar archivo a un nuevo proyecto",
    },
    toast: {
      missingItem: "Ese archivo ya no está en la Biblioteca",
      moveFailed: "No se pudo mover",
      movedToFolder: "Movido a {folder}",
      movedToLibrary: "Movido a la Biblioteca",
      folderFallback: "carpeta",
      favoritesFailed: "No se pudieron actualizar los favoritos",
      addedToFavorites: "Agregado a Favoritos",
      removedFromFavorites: "Quitado de Favoritos",
      uploadingOne: "Subiendo {name}…",
      uploadingMany: "Subiendo {count} archivos…",
      uploadedOne: "Archivo subido",
      uploadedMany: "{count} archivos subidos",
      uploadFailed: "Error al subir",
      uploadTooLarge: "{name} supera los 512 MB.",
      readDropsFailed: "No se pudieron leer los archivos soltados",
      createNoteFailed: "No se pudo crear la nota",
      renameFailed: "No se pudo renombrar",
      createFolderFailed: "No se pudo crear la carpeta",
      signedOutBeforeFolder: "Se cerró la sesión antes de crear la carpeta.",
      signedOutBeforeSave: "Se cerró la sesión antes de guardar el cambio.",
      signedOutBeforeUpload: "Se cerró la sesión antes de terminar la subida.",
      deleteFailed: "No se pudieron eliminar algunos elementos",
      saveFailed: "No se pudo guardar",
      saveNoteFirst: "Guarda primero la nota.",
      revealFailed: "No se pudo abrir el administrador de archivos",
      downloadFailed: "No se pudo descargar {name}",
      downloadManyFailed: "No se pudieron descargar los archivos",
      downloadingMany: "Descargando {count} archivos",
      downloadingManyDescription:
        "Si tu navegador lo pide, permite que este sitio descargue varios archivos.",
      preparingMany: "Preparando {count} archivos…",
      zipFileName: "Archivos de la Biblioteca",
      nothingToChat: "Todavía no hay nada sobre lo que chatear",
      emptyFolder: "Esta carpeta no tiene archivos.",
      tooLargeOne: "{name} es demasiado grande para adjuntarlo",
      tooLargeMany: "Estos archivos son demasiado grandes para adjuntarlos",
      attachedOne: "{count} archivo adjuntado",
      attachedMany: "{count} archivos adjuntados",
      skippedTooLarge: "{count} demasiado grandes para adjuntar.",
      skippedOverLimit:
        "{count} más por encima del límite de {limit} archivos.",
      chatFilesWaiting:
        "{count} se adjuntarán cuando se cargue un modelo que pueda leerlos.",
      openFilesFailed: "No se pudieron abrir los archivos",
      editImageFailed: "No se pudo abrir la imagen para editarla",
      speechModel: "{name} es un modelo de voz",
      speechModelDescription:
        "Selecciónalo en el menú de modelos de la página Audio.",
      noMediaLink:
        "El servidor no devolvió ningún enlace para reproducir este archivo.",
      imageNotFound: "No se encontró esta imagen",
      clipNotFound: "No se encontró este clip",
      notFoundDescription: "Puede que se haya archivado o eliminado.",
      imageTooLarge: "Esta imagen es demasiado grande para adjuntarla",
      videoTooLarge: "Este video es demasiado grande para adjuntarlo",
      attachmentLimit: "Los adjuntos del chat tienen un límite de {size} MB.",
      attachingImage: "Adjuntando la imagen…",
      attachingVideo: "Adjuntando el video…",
      readImageFailed: "No se pudo leer la imagen ({status}).",
      readVideoFailed: "No se pudo leer el video ({status}).",
      openFileFailed: "No se pudo abrir el archivo",
    },
  },
  exportDecision: {
    title: "Modelo de decisión",
    description:
      "Los modelos de decisión {layout} se exportan a GGUF para el servidor de decisiones de llama.cpp. Elige una o más cuantizaciones.",
    adapterNote:
      "Este checkpoint contiene adaptadores LoRA; se fusionan antes de la conversión.",
    notEligibleTitle: "Exportación GGUF no disponible",
    ggufOnly: "Los modelos de decisión solo se exportan a GGUF",
    existing: "Ya exportado: {quantizations}",
    outputNote:
      "Los archivos GGUF se guardan en la carpeta de la ejecución: {path}",
    methodLabel: "GGUF del modelo de decisión",
    outputLabel: "GGUF de decisión ({quantizations})",
  },
  decisions: {
    title: "Probar una decisión",
    description:
      "Hazle una pregunta al modelo de decisiones sobre cualquier texto.",
    tryIt: "Probar",
    tryItOff: "Activa Atender solicitudes para probarlo.",
    typeNoul: "Sí / no",
    typeChoice: "Elección",
    typeScore: "Puntaje",
    aboutNoul: "Responde sí o no, con la probabilidad de sí.",
    aboutChoice:
      "Elige una opción de una lista, con una probabilidad para cada una.",
    aboutScore:
      "Sitúa el estado en una escala de niveles, del más bajo al más alto.",
    model: "Modelo de decisión",
    defaultModel: "Predeterminado ({model})",
    layaModel: "Laya {model}",
    thisMachine: "Esta computadora",
    modelReady: "Listo",
    modelLoading: "Cargando",
    modelNotLoaded: "Se carga en la primera ejecución",
    connectionNotDefault:
      "Primero configúrala como modelo de la API de decisiones en Configuración > API.",
    question: "Pregunta",
    questionHelp: "Lo que el modelo debe decidir.",
    result: "Resultado",
    form: "Formulario",
    json: "JSON",
    preview: "Vista previa",
    state: "Estado",
    stateHelp: "El texto sobre el que decidir. Texto plano o JSON.",
    criteria: "Criterios",
    criteriaNoulHelp: "Opcional. Qué cuenta como sí y qué como no.",
    criteriaChoiceHelp:
      "Las opciones para elegir, cada una con una breve descripción.",
    criteriaScoreHelp: "Los niveles de la escala, del más bajo al más alto.",
    yesWhen: "Sí cuando",
    noWhen: "No cuando",
    optionName: "Opción",
    optionDescription: "Descripción",
    optionNamePlaceholder: "nombre",
    optionDescriptionPlaceholder: "Cuándo elegirla",
    optionNameLabel: "Nombre de la opción {n}",
    optionDescriptionLabel: "Descripción de la opción {n}",
    removeOption: "Quitar la opción {n}",
    addOption: "Agregar opción",
    levelLabel: "Nivel {n}",
    removeLevel: "Quitar el nivel {n}",
    addLevel: "Agregar nivel",
    jsonHelp: "El cuerpo de la solicitud. Ejecutar lo envía tal cual.",
    examples: "Ejemplos",
    presetToolApproval: "Aprobar llamadas a herramientas de agentes",
    presetPromptInjection: "Detectar inyección de prompts",
    presetTicketRouting: "Enrutar tickets de soporte",
    presetModeration: "Moderar mensajes",
    presetIntent: "Detectar la intención",
    presetGrading: "Calificar una respuesta",
    run: "Ejecutar decisión",
    running: "Ejecutando",
    answer: "Respuesta",
    yes: "Sí",
    no: "No",
    scoreDetail: "Puntaje {score} en una escala de 0 a {max}",
    latency: "{ms} ms",
    inputTokens: "{count} tokens de entrada",
    responseTitle: "Respuesta",
    emptyTitle: "Aún no hay resultado",
    emptyBody: "Presiona Ejecutar decisión o {shortcut}.",
    cantRun: "No se pudo ejecutar",
    failed: "Error {status}",
    invalidJson: "El cuerpo de la solicitud no es JSON válido.",
  },
  floatingPanels: {
    loadedModels: {
      title: "Modelos cargados",
      countOne: "{count} modelo cargado",
      countMany: "{count} modelos cargados",
      pillLabel: "{count}. Mostrar detalles o arrastrar para mover",
      dragToMove: "Arrastra para mover",
      dragHandleLabel: "Arrastra para mover",
      collapse: "Contraer",
      collapseLabel: "Contraer los modelos cargados",
      close: "Cerrar",
      closeLabel: "Cerrar los modelos cargados",
      backOnNextLoad: "Vuelve al cargar el siguiente modelo",
      ejectToFree: "Expulsar para liberar memoria",
      ejectLabel: "Expulsar {model}",
      openTarget: "Abrir {target}",
      rowLabel: "{model}. Abrir {target}",
      loading: "Cargando",
      ejecting: "Expulsando",
      stillInMemory: "Sigue en memoria",
      alsoServing: "También en servicio",
      noneLoaded: "No hay modelos cargados",
      noneLoadedDetail: "Los modelos que cargues aparecerán aquí.",
      otherAccount: "En uso por otra cuenta",
      otherAccountDetail: "La memoria la ocupa un modelo que no puedes ver",
      toastStillResident:
        "“{model}” se cargó mientras se expulsaba, así que sigue usando memoria. Expúlsalo de nuevo para liberarla.",
      toastUnverified:
        "Se pidió descargar {model} de la memoria, pero el entorno de ejecución no lo confirmó. Revisa la tarjeta en un momento.",
      toastAlreadyFree: "{model} ya no estaba cargado.",
      toastReplaced:
        "{model} ya no está cargado. “{resident}” ocupó su lugar y no se tocó.",
      toastEjected: "{model} expulsado",
      toastFailed: "No se pudo expulsar {model}",
    },
    monitor: {
      dragHandleLabel:
        "Mover el monitor en vivo. Las teclas de flecha lo desplazan un poco; con doble clic se restablece.",
      reconnecting: "Reconectando...",
    },
  },
  projectsPage: {
    title: "Proyectos",
    search: "Buscar proyectos",
    importExport: "Importar / exportar proyectos",
    importChatsAction: "Importar chats…",
    exportAllProjects: "Exportar todos los proyectos",
    exportProjectsAndRecents: "Exportar proyectos + recientes",
    combined: "Combinado",
    perChat: "Por chat",
    newProject: "Nuevo proyecto",
    name: "Nombre",
    updated: "Actualizado",
    newestFirst: "Más recientes primero",
    oldestFirst: "Más antiguos primero",
    today: "Hoy",
    yesterday: "Ayer",
    emptyDescription:
      "Un proyecto reúne chats relacionados, con instrucciones y fuentes compartidas que puede usar cada chat que contiene.",
    createFirst: "Crea tu primer proyecto",
    noMatches: "Ningún proyecto coincide con tu búsqueda.",
    clearSearch: "Borrar búsqueda",
    showChats: "Mostrar chats",
    hideChats: "Ocultar chats",
    pinProject: "Fijar proyecto",
    unpinProject: "Desfijar proyecto",
    newChatInProject: "Nuevo chat en el proyecto",
    projectOptions: "Opciones del proyecto",
    loadChatsFailed: "No se pudieron cargar los chats. Reintentar",
    refreshChatsFailed: "No se pudieron actualizar los chats. Reintentar",
    noChats: "No hay chats",
    unread: "No leído",
    pinChat: "Fijar chat",
    unpinChat: "Desfijar chat",
    chatOptions: "Opciones del chat",
    exportAllChats: "Exportar todos los chats…",
    archive: "Archivar",
    renameChat: "Renombrar chat",
    chatName: "Nombre del chat",
    deleteChat: "Eliminar chat",
    deleteChatDescription:
      "¿Seguro que quieres eliminar {name}? Sus mensajes se eliminarán de forma permanente.",
    importChats: "Importar chats",
    importDestination: "Elige dónde importar {name}:",
    selectDestination: "Seleccionar destino",
    recents: "Recientes",
    import: "Importar",
    deleteProject: "Eliminar proyecto",
    deleteProjectDescription:
      "¿Seguro que quieres eliminar {name}? Sus chats se eliminarán de forma permanente.",
    deleteProjectFiles:
      "La carpeta de trabajo del proyecto se eliminará del disco.",
    deleteAll: "Eliminar todo",
    projectFallback: "proyecto",
    importFailed: "Error al importar.",
    nothingToExport: "No hay conversaciones para exportar.",
    exportFailed: "Error al exportar.",
    renameChatFailed: "No se pudo renombrar el chat",
    archiveChatFailed: "No se pudo archivar el chat",
    deleteChatFailed: "No se pudo eliminar el chat",
    deleteProjectFailed: "No se pudo eliminar el proyecto",
  },
  auth: {
    welcomeBack: "Bienvenido de nuevo",
    loginSubtitle: "Inicia sesión con tu contraseña.",
    setupTitle: "Configura tu cuenta",
    setupSubtitle: "Crea una contraseña nueva",
    login: "Iniciar sesión",
    changePassword: "Cambiar contraseña",
    alreadySetUp: "¿Ya configuraste la contraseña? ",
    backToLogin: "Volver al inicio de sesión",
    username: "Nombre de usuario",
    password: "Contraseña",
    currentPassword: "Contraseña actual",
    newPassword: "Contraseña nueva",
    confirmPassword: "Confirmar contraseña",
    setupCodeHint:
      "¿Primer inicio de sesión? Pega como contraseña el código de configuración que te dio tu administrador.",
    ownerHint:
      "¿Eres el propietario de la instalación? Inicia sesión como unsloth o con el nombre de inicio de sesión que aparece en Configuración > Cuentas.",
    currentSetupCodeHint:
      "Pega como contraseña actual el código de configuración con el que acabas de iniciar sesión.",
    minLength: "Debe tener al menos 8 caracteres.",
    ensureMatch: "Asegúrate de que las contraseñas coincidan.",
    noSpaces: "La contraseña nueva no puede contener espacios.",
    currentTooShort: "La contraseña actual debe tener al menos 8 caracteres.",
    setupInitFailed:
      "No se pudo iniciar la configuración. Recarga la página e inténtalo de nuevo.",
    newTooShort: "La contraseña nueva debe tener al menos 8 caracteres.",
    mismatch: "Las contraseñas no coinciden.",
    mustDiffer: "La contraseña nueva debe ser distinta de la actual.",
    pleaseWait: "Espera un momento...",
    continueAsOwner: "Continuar como propietario",
    signInWithUsername: "Inicia sesión con tu nombre de usuario y contraseña.",
    bootstrapping:
      "La autenticación aún está creando la cuenta de administrador predeterminada.",
    signInToChange:
      "Inicia sesión una vez con las credenciales iniciales para cambiar la contraseña.",
    alreadyUpdated:
      "La contraseña ya se cambió. Usa la pantalla de inicio de sesión.",
    busy: "Unsloth está ocupado. Inténtalo de nuevo en un momento.",
    updateFailed: "No se pudo cambiar la contraseña.",
    loadFailed: "Error al cargar.",
    failed: "Error de autenticación.",
    shuttingDown:
      "Esta instancia se está cerrando: era accesible en la red y su contraseña predeterminada nunca se cambió.",
    shutdownCountdown:
      "Esta instancia es accesible en la red y aún usa su contraseña predeterminada, así que se cerrará en {time}. Si defines una contraseña aquí, seguirá en funcionamiento.",
  },
  routes: {
    chat: "Chat",
    api: "API",
    dataRecipes: "Data Recipes",
    login: "Iniciar sesión",
    changePassword: "Cambiar contraseña",
  },
  remoteCode: {
    unsafeTitle: "Archivos no seguros detectados",
    blockedTitle: "Código personalizado bloqueado",
    enableTitle: "¿Activar código personalizado para este modelo?",
    thisModel: "Este modelo",
    unsafeBody:
      "{model} contiene archivos que el análisis de seguridad de Hugging Face marcó como no seguros (por ejemplo, un pickle malicioso que ejecutaría código al cargar el modelo). No se puede cargar. Los archivos marcados nunca se descargaron.",
    unsafeBodyFromProvider:
      "{model} de {provider} contiene archivos que el análisis de seguridad de Hugging Face marcó como no seguros (por ejemplo, un pickle malicioso que ejecutaría código al cargar el modelo). No se puede cargar. Los archivos marcados nunca se descargaron.",
    customCode:
      "{model} declara código Python personalizado en su repositorio. {note}",
    customCodeFromProvider:
      "{model} de {provider} declara código Python personalizado en su repositorio. {note}",
    blockedNote:
      "Un análisis de seguridad detectó problemas CRITICAL, así que no se puede activar.",
    reviewNote:
      "Revisa el análisis de seguridad de abajo. Continúa solo si confías en el origen del modelo.",
    trustNote: "Continúa solo si confías en el origen del modelo.",
    flaggedIntro:
      "Nuestro analizador automático detectó problemas, entre ellos:",
    nothingFlagged:
      "Nuestro analizador automático no detectó archivos preocupantes, pero verifícalo tú también.",
    enableAndContinue: "Activar y continuar",
  },
  tour: {
    badge: "recorrido guiado",
    defaultTitle: "Recorrido rápido",
    defaultBody: "Vamos a orientarte.",
    skipTour: "Omitir recorrido",
    skip: "Omitir",
    back: "Atrás",
    next: "Siguiente",
    tip: "Consejo: `Esc` omite. El recorrido bloquea los clics para que puedas leer.",
    readMore: "Más información",
  },
  updateNotes: {
    openRelease: "Abrir versión",
    openChangelog: "Abrir registro de cambios",
    retry: "Reintentar",
    loadingChanges: "Cargando cambios nuevos...",
    customLlamaRepo:
      "Esta instalación sigue un repositorio personalizado de llama.cpp, así que sus cambios no se pueden comparar.",
    noItemisedNotes:
      "Esta compilación es anterior a las notas de la versión detalladas, así que sus cambios no se pueden comparar.",
    compareFailed: "No se pudieron comparar estas versiones.",
    llamaChangesLabel: "Cambios nuevos de llama.cpp de {from} a {to}",
    showingChanges: "Mostrando {shown} de {total} cambios",
    noCarriedChanges:
      "No se indican cambios nuevos incorporados para esta compilación.",
    notesFor: "Notas de la versión {release}",
    notesForVersion: "Notas de la versión {version}",
    notesTruncated:
      "Notas recortadas. Consulta las notas de la versión completas.",
    summaryLabel: "Resumen de las notas de la versión",
    moreCount: "+{count} más",
    loadingNotes: "Cargando notas de la versión...",
    loadFailed: "No se pudieron cargar las notas de la versión.",
    noNotesFor: "Aún no hay notas de la versión publicadas para {release}.",
    noNotes: "Aún no hay notas de la versión publicadas.",
  },
  desktopStartup: {
    checking: "Verificando...",
    installPrompt: "Para instalar Unsloth, haz clic en Empezar.",
    getStarted: "Empezar",
    chatOnly: "Solo chat (GGUF): instalación más ligera, sin entrenamiento",
    installLocation: "Ubicación de instalación:",
    changeLocation: "Cambiar",
    useDefaultLocation: "Usar predeterminada",
    longPaths: {
      title: "Windows pedirá permiso",
      approve:
        "Aprueba {tool} una vez para activar las rutas de archivo largas para los kernels compilados.",
      waiting: "Esperando el permiso de Windows...",
    },
    skip: "Omitir",
    continue: "Continuar",
    install: {
      preparing: "Preparando tu espacio de trabajo...",
      preparingDetail: "Verificando qué necesita esta computadora.",
      downloading: "Descargando los componentes necesarios...",
      downloadingDetail: "Creando tu entorno local.",
      installing: "Instalando Unsloth...",
      installingDetail: "Configurando herramientas de IA locales...",
      finishing: "Terminando la configuración...",
      finishingDetail: "Dejando todo listo.",
      continuesAutomatically: "La configuración continuará automáticamente.",
      fewMinutes: "Esto puede tardar unos minutos.",
      someTakeLonger: "Algunos componentes tardan más en configurarse.",
    },
    details: {
      showInstall: "Mostrar detalles de la instalación",
      hideInstall: "Ocultar detalles de la instalación",
      showSetup: "Mostrar detalles de la configuración",
      hideSetup: "Ocultar detalles de la configuración",
      showUpdate: "Mostrar detalles de la actualización",
      hideUpdate: "Ocultar detalles de la actualización",
    },
    repair: {
      upgrading: "Pasando a la instalación completa...",
      gettingReady: "Preparando todo...",
      upgradingDetail:
        "Descargando PyTorch y sus bibliotecas para la GPU de esta computadora. Esto puede tardar varios minutos.",
      wontTakeLong: "No tardará mucho.",
      upgradeFailed: "No se pudo pasar a la instalación completa",
      updateFailed: "Error en la actualización",
    },
    closing: "Cerrando Unsloth Desktop...",
    closingDetail: "Deteniendo el backend.",
    diskFull: "No hay suficiente espacio en disco",
    setupFailed: "La configuración tuvo un problema",
    tryAgain: "Intentar de nuevo",
    retry: "Reintentar",
    elevation: {
      title: "Se necesita permiso",
      packages: "Hay que instalar los siguientes paquetes del sistema:",
      allow: "Permitir",
    },
    starting: "Iniciando Unsloth...",
    loadingProjects: "Cargando proyectos...",
    loadingModels: "Cargando modelos...",
    nearlyDone: "Casi listo...",
    finishingUpdate: "Terminando la actualización...",
    serverStopped: "Servidor detenido",
    startServer: "Iniciar servidor",
    somethingWrong: "Algo salió mal",
    diagnostics: {
      copy: "Copiar diagnóstico",
      copying: "Copiando...",
      clipboardFailed:
        "No se pudo copiar al portapapeles. Selecciona y copia el diagnóstico de abajo.",
      copyFailed: "No se pudo copiar el diagnóstico: {error}",
    },
  },
  desktopUpdate: {
    updatingBackend: "Actualizando el backend...",
    downloadingApp: "Descargando la actualización de la app...",
    installing: "Instalando la actualización...",
    failed: "Error en la actualización",
    updating: "Actualizando...",
    updatingBackendDetail: "Esto puede tardar unos minutos. No cierres la app.",
    percentDownloaded: "{percent} % descargado",
    restartShortly: "La app se reiniciará en breve.",
    failedDetail: "Algo salió mal durante la actualización.",
    downloadingFile: "Descargando {file}",
    downloadingAppLabel: "Descargando la actualización de la app",
    skipRestart: "Omitir y reiniciar",
    banner: {
      dismiss: "Descartar la notificación de actualización de la app",
      failedTitle: "Error al actualizar la app",
      newVersion: "Nueva versión de Unsloth",
      backendRecovered: "Backend recuperado. El diagnóstico sigue disponible.",
      linuxManual:
        "Abre la página de la versión en GitHub para instalar el paquete de Linux",
      available: "Hay una nueva actualización de la app disponible",
      showNotes: "Mostrar notas de la versión",
      hideNotes: "Ocultar notas de la versión",
      copyDiagnostics: "Copiar diagnóstico",
      later: "Más tarde",
      remindLater: "Recordármelo más tarde",
      openReleasePage: "Abrir página de la versión",
      retry: "Reintentar actualización",
      update: "Actualizar",
    },
  },
  desktopShell: {
    navigation: "Barra lateral y navegación de páginas",
    collapseSidebar: "Contraer barra lateral",
    expandSidebar: "Expandir barra lateral",
    goBack: "Atrás",
    goForward: "Adelante",
    titlebar: "Barra de título de la ventana",
    windowControls: "Controles de la ventana",
    minimize: "Minimizar ventana",
    maximize: "Maximizar ventana",
    restore: "Restaurar ventana",
    close: "Cerrar ventana",
  },
  runtimeUpdate: {
    installAudio: "Instalar el runtime de audio",
    installingAudio: "Instalando el runtime de audio...",
    unknownVersion: "desconocida",
    failed: "Error en la actualización: {error}",
    unknownError: "error desconocido",
    skipVersions: "Omitir estas versiones",
    skipChange: "Omitir este cambio",
    skipVersion: "Omitir esta versión",
    skippedVersion: "{component} {version} omitido por ahora",
    skippedUpdate: "Actualización de {component} omitida por ahora",
    skippedUpdates: "Actualizaciones de {components} omitidas por ahora",
    vulkanFaster:
      "Vulkan es más de un 10 % más rápido que ROCM. No hace falta reiniciar",
    noRestart: "No hace falta reiniciar",
    downloadSize: "Descarga de {size}",
    backendChange: "Backend {change}",
    updating: "Actualizando {component}...",
    updatingLabel: "Actualizando {component}",
    installingPhase: "Instalando {component} ({current} de {total})...",
    installingCount: "Instalando {count} actualizaciones...",
    bundleTitle: "{count} actualizaciones del runtime",
    newBackend: "Nuevo backend de {component}",
    newUpdate: "Nueva actualización de {component}",
    hideUpdate:
      "Ocultar la actualización de {component} hasta que se vuelva a abrir Unsloth",
    hideUpdates:
      "Ocultar las actualizaciones de {components} hasta que se vuelva a abrir Unsloth",
    whatsNew: "Novedades",
    hideWhatsNew: "Ocultar novedades",
    later: "Más tarde",
    update: "Actualizar",
    undo: "Deshacer",
  },
} satisfies DeepPartialMessageTree<typeof en>;
