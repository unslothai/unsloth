// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DeepPartialMessageTree } from "../types";
import type { en } from "./en.ts";

export const ptPT = {
  linkMenu: {
    openInBrowser: "Abrir no navegador do Unsloth",
    openExternal: "Abrir no navegador externo",
    copyLink: "Copiar ligação",
    saveLinkAs: "Guardar ligação como…",
    alwaysInBrowser: "Abrir sempre as ligações no navegador do Unsloth",
    linkCopied: "Ligação copiada",
    copyFailed: "Não foi possível copiar.",
    saveFailed: "Não foi possível guardar o ficheiro.",
    openFile: "Abrir ficheiro",
    openDefaultApp: "Abrir na aplicação predefinida",
    openWith: "Abrir com",
    unslothBrowser: "Navegador do Unsloth",
    newChat: "Novo chat",
    browserTab: "Novo separador do navegador",
    saveAs: "Guardar como…",
    copyPath: "Copiar caminho",
    copyContents: "Copiar conteúdo do ficheiro",
    revealFinder: "Mostrar no Finder",
    revealExplorer: "Mostrar no Explorador",
    revealFolder: "Mostrar na pasta",
    pathCopied: "Caminho copiado",
    contentsCopied: "Conteúdo copiado",
    openFailed: "Não foi possível abrir {name}.",
    revealFailed: "Não foi possível mostrar {name} na respetiva pasta.",
  },
  imageViewer: {
    title: "Imagem",
    description:
      "Utilize as teclas de seta para navegar entre imagens e as teclas mais e menos para aplicar zoom.",
    failed: "Não foi possível carregar esta imagem.",
    downloadFailed: "Não foi possível guardar a imagem.",
    openSource: "Abrir página de origem",
    download: "Transferir",
    close: "Fechar",
    previous: "Imagem anterior",
    next: "Imagem seguinte",
    zoomIn: "Ampliar",
    zoomOut: "Reduzir",
    fit: "Ajustar à janela",
    copy: "Copiar imagem",
    copied: "Imagem copiada",
    copyFailed: "Não foi possível copiar a imagem.",
    removeFromMessage: "Remover da mensagem",
  },
  browser: {
    drafts: {
      title: "Descartar uma página com texto introduzido?",
      description:
        "Para manter o chat responsivo, apenas três páginas ocultas com texto introduzido continuam em execução. Se continuar, {pages} será descartado, o que pode fazer perder texto não guardado. Cancelar volta ao separador anterior.",
      confirm: "Descartar e continuar",
    },
    agentSettings: {
      title: "Utilização do navegador",
      docs: "Documentação",
      docsLabel: "Abrir a documentação de utilização do navegador",
      enable: "Permitir que os modelos utilizem o navegador",
      enableDescription:
        "Os modelos abrem páginas no seu próprio separador do navegador, leem-nas e clicam e escrevem por si. Também disponível no menu + da caixa de mensagem.",
      history: "Histórico de navegação",
      historyDescription:
        "O acesso total dispensa aprovação; Nunca continua a bloquear o acesso",
      alwaysAsk: "Perguntar sempre",
      alwaysAllow: "Permitir sempre",
      never: "Nunca",
      scripts: "Executar scripts",
      scriptsDescription:
        "Escolha se os modelos podem executar scripts nas páginas. Cada script é-lhe mostrado antes de ser executado",
      cursor: "Mostrar cursor",
      cursorDescription:
        "Mostra um ponteiro onde os modelos clicam, escrevem e arrastam na página",
      askEveryTime: "Perguntar sempre",
      permissionsTitle: "Permissões do agente",
      permissionsDescription:
        "Defina predefinições e exceções por site. Os campos de palavra-passe e de pagamento nunca são preenchidos.",
      site: "Site ou padrão",
      sitePlaceholder: "example.com ou *.example.com",
      default: "Predefinição",
      browsing: "Navegação",
      actions: "Ações",
      downloads: "Transferências",
      unlabelled: "Cliques sem etiqueta",
      askNewSites: "Perguntar em sites novos",
      requiresApproval: "Requer aprovação",
      blocked: "Bloqueado",
      addSite: "Adicionar",
      add: "Adicionar",
      cancel: "Cancelar",
      removeSite: "Remover {site}",
      permissionsNote:
        "Só aparecem aqui os sites com permissões próprias. As ações são inícios de sessão, envios de formulários e botões que compram, enviam ou eliminam.",
      siteInvalid: "Introduza um site como example.com ou *.example.com.",
    },
    agent: {
      working: "O Unsloth está a navegar neste separador",
      waiting: "O Unsloth precisa da sua resposta no navegador para {chat}",
      otherChat: "outro chat",
      show: "Mostrar",
      stop: "Parar",
      allow: "Permitir",
      deny: "Recusar",
      ask: {
        open: "Permitir que o Unsloth abra {host}?",
        signIn: "Permitir que o Unsloth inicie sessão em {host}?",
        submit: "Permitir que o Unsloth envie este formulário em {host}?",
        click: "Permitir que o Unsloth clique em “{name}” em {host}?",
        history: "Permitir que o Unsloth leia o seu histórico de navegação?",
        script: "Permitir que o Unsloth execute este script em {host}?",
        pointer:
          "Permitir que o Unsloth clique em partes sem etiqueta de {host}, como um canvas?",
      },
      step: {
        open: "A abrir {detail}",
        back: "A retroceder",
        forward: "A avançar",
        reload: "A recarregar a página",
        read: "A ler a página",
        find: "A procurar “{detail}”",
        click: "A clicar em “{detail}”",
        type: "A escrever em “{detail}”",
        select: "A escolher “{detail}”",
        press: "A premir {detail}",
        scroll: "A deslocar",
        screenshot: "A fazer uma captura de ecrã",
        evaluate: "A executar um script na página",
        history: "A ler o seu histórico de navegação",
        move: "A mover o rato",
        drag: "A arrastar",
      },
    },
    title: "Navegador",
    tabs: "Separadores do navegador",
    newTab: "Novo separador",
    newTabFullView: "Novo separador em vista completa",
    closeTab: "Fechar separador",
    close: "Fechar navegador",
    show: "Mostrar navegador",
    back: "Retroceder",
    forward: "Avançar",
    reload: "Recarregar",
    more: "Mais",
    addressPlaceholder: "Pesquise ou introduza um URL",
    local: "Local",
    browserUse: {
      label: "Utilização do navegador",
      description: "Permitir que os modelos controlem o navegador integrado",
      unavailable:
        "Carregue um modelo capaz de chamar ferramentas para que possa utilizar o navegador.",
    },
    suggested: "Sugestões",
    suggestedMenu: {
      open: "Abrir",
      openInNewTab: "Abrir num novo separador",
      remove: "Remover das sugestões",
    },
    tabMenu: {
      duplicate: "Duplicar",
      closeOthers: "Fechar outros separadores",
      pin: "Afixar na barra lateral",
      unpin: "Desafixar da barra lateral",
      newTabRight: "Novo separador à direita",
      fork: "Ramificar",
      forkNewChat: "Novo chat",
      forkTemporaryChat: "Chat temporário",
      copyUrl: "Copiar URL",
      openExternal: "Abrir no navegador externo",
      rename: "Mudar o nome",
      renameLabel: "Nome do separador",
      mute: "Desativar som do separador",
      unmute: "Ativar som do separador",
      close: "Fechar",
      closeRight: "Fechar separadores à direita",
      pageOptions: "Opções da página",
    },
    siteInfo: {
      label: "Informações do site",
      secure: "Ligação segura",
      insecure: "Ligação não segura",
      back: "Retroceder",
      close: "Fechar",
      security: "Segurança",
      secureTitle: "A ligação é segura",
      secureDescription:
        "As suas informações, como palavras-passe ou números de cartão, são privadas quando são enviadas para este site.",
      insecureTitle: "A ligação não é segura",
      insecureDescription:
        "As informações que enviar para este site, como palavras-passe ou números de cartão, podem ser vistas por outras pessoas.",
      learnMore: "Saiba mais",
      certificateValid: "O certificado é válido",
    },
    tools: "Ferramentas",
    recents: "Recentes",
    website: "Site",
    recentNewer: "Mais recentes",
    recentOlder: "Mais antigos",
    openExternal: "Abrir no navegador do sistema",
    copyLink: "Copiar ligação",
    linkCopied: "Ligação copiada",
    download: "Transferir",
    settings: "Definições do navegador",
    cannotShowFile:
      "Não é possível mostrar este ficheiro aqui. Transfira-o para o abrir.",
    fileGone:
      "Este ficheiro já não está disponível. Abra-o novamente a partir do chat.",
    settingsTitle: "Navegador",
    linksTitle: "Ligações e ficheiros",
    addressBarTitle: "Barra de endereço",
    openLinksSetting: "Abrir ligações em",
    openLinksSettingDescription:
      "Onde abrem as ligações do chat: no navegador predefinido ou num separador do navegador ao lado da conversa.",
    linkDestinationDefault: "Navegador predefinido",
    linkDestinationPanel: "Navegador do Unsloth",
    openFilesSetting: "Abrir ficheiros no navegador",
    openFilesSettingDescription:
      "Os documentos que anexar, como PDFs, ficheiros do Office, HTML e texto, abrem num separador do navegador ao lado da conversa em vez de numa janela de pré-visualização.",
    searchEngineSetting: "Motor de pesquisa",
    searchEngineSettingDescription:
      "Utilizado quando a barra de endereço recebe palavras em vez de um URL.",
    showFullUrlSetting: "Mostrar URL completo",
    showFullUrlSettingDescription:
      "Incluir a consulta e o fragmento na barra de endereço, e não apenas o site e o caminho.",
    bookmarksTitle: "Marcadores",
    bookmarksToolbarSetting: "Barra de marcadores",
    bookmarksToolbarSettingDescription:
      "Quando mostrar as páginas guardadas por baixo da barra de endereço.",
    bookmarkEditorSettingDescription:
      "Dê um nome ao marcador e escolha a respetiva localização ao guardá-lo.",
    bookmarksSettingDescription:
      "Ver, editar e remover páginas guardadas no navegador.",
    browsingDataTitle: "Dados de navegação",
    historySetting: "Histórico de navegação",
    historySettingDescription: "Ver e gerir as páginas visitadas no navegador.",
    downloadsSetting: "Histórico de transferências",
    downloadsSettingDescription:
      "Ver e gerir os ficheiros transferidos a partir do navegador.",
    pagesFromChat: "Abra um chat para ver isto no navegador.",
    tabsTitle: "Separadores",
    switchToNewTabsSetting: "Mudar imediatamente para os novos separadores",
    switchToNewTabsSettingDescription:
      "Quando abrir uma ligação num novo separador com ⌘ ou Ctrl, mostrá-lo em vez de o abrir em segundo plano.",
    switchToNewTabsNative:
      "Na aplicação para computador, as páginas abrem sempre os novos separadores em primeiro plano.",
    defaultZoomSetting: "Zoom predefinido",
    defaultZoomSettingDescription:
      "O tamanho com que as páginas aparecem ao abrir. Repor o zoom de uma página volta a este valor.",
    newTabPageTitle: "Página de novo separador",
    showSuggestedSetting: "Mostrar sites sugeridos",
    showSuggestedSettingDescription:
      "Sites que visita com frequência, como atalhos num novo separador.",
    showRecentsSetting: "Mostrar páginas recentes",
    showRecentsSettingDescription:
      "Páginas visitadas recentemente, listadas num novo separador.",
    hiddenSuggestionsSetting: "Sugestões removidas",
    hiddenSuggestionsSettingDescription:
      "Sites que removeu das Sugestões: {count}. Restaure-os para voltarem a aparecer.",
    hiddenSuggestionsNone:
      "Os sites que remover das Sugestões podem ser restaurados aqui.",
    restore: "Restaurar",
    importBookmarksSetting: "Importar e exportar",
    importBookmarksSettingDescription:
      "Importe marcadores do Chrome, Safari, Firefox ou Edge através do ficheiro HTML que estes exportam, ou guarde os seus nesse formato.",
    importBookmarks: "Importar…",
    exportBookmarks: "Exportar",
    importBookmarksDoneOne: "{count} marcador importado.",
    importBookmarksDoneMany: "{count} marcadores importados.",
    importBookmarksNone: "Não há marcadores novos nesse ficheiro.",
    importBookmarksPartial:
      "{count} importados, mas outros {leftOut} não couberam: o Unsloth guarda até {max} marcadores.",
    importBookmarksFailed:
      "Não foi possível ler esse ficheiro. Escolha um ficheiro HTML de marcadores exportado de um navegador.",
    downloadsTitle: "Transferências",
    askWhereToSaveSetting: "Perguntar onde guardar cada ficheiro",
    askWhereToSaveSettingDescription:
      "Mostrar uma caixa de diálogo de gravação para os ficheiros que transferir no navegador.",
    askWhereToSaveUnsupported:
      "O seu navegador web guarda as transferências na respetiva pasta de transferências.",
    downloadLocationSetting: "Localização",
    downloadLocationDefault: "Pasta Transferências do sistema",
    downloadLocationChange: "Alterar",
    downloadLocationReset: "Repor",
    askBeforeDownloadingSetting: "Perguntar antes de transferir",
    askBeforeDownloadingSettingDescription:
      "Confirmar os ficheiros de sites antes de serem guardados.",
    downloadSitesSetting: "Permissões de transferência por site",
    downloadSitesSettingDescription:
      "Sites dos quais escolheu permitir ou bloquear sempre as transferências.",
    downloadSitesNone:
      "Assinale Lembrar para a próxima vez quando lhe for perguntado e o site aparecerá aqui.",
    downloadSitesFor: "Transferências de {host}",
    downloadSitesRemove: "Esquecer {host}",
    downloadSiteAllow: "Permitir sempre",
    downloadSiteBlock: "Bloquear sempre",
    downloadPrompt: {
      title: "Transferir este ficheiro?",
      description: "{host} pretende guardar {name} no seu computador.",
      dangerous:
        "Ficheiros como este podem executar programas no seu computador. Transfira-o apenas se confiar em {host}.",
      remember: "Lembrar para a próxima vez",
      cancel: "Cancelar",
      download: "Transferir",
      blocked: "Foi bloqueada uma transferência de {host}.",
      ready: "{name} está pronto para ser guardado",
      save: "Guardar",
    },
    saveDownloadHistorySetting: "Manter histórico de transferências",
    saveDownloadHistorySettingDescription:
      "Listar os ficheiros que transferir no navegador em Histórico de transferências.",
    saveHistorySetting: "Guardar histórico de navegação",
    saveHistorySettingDescription:
      "Lembrar as páginas que visita, para o Histórico, Recentes e Sugestões.",
    historyRetentionSetting: "Manter histórico durante",
    historyRetentionSettingDescription:
      "As visitas mais antigas do que isto são eliminadas automaticamente.",
    retention: {
      forever: "Até ser limpo",
      days90: "3 meses",
      days30: "1 mês",
      days7: "1 semana",
      days1: "1 dia",
    },
    annotationsTitle: "Anotações",
    annotationScreenshotsSetting: "Capturas de ecrã das anotações",
    annotationScreenshotsSettingDescription:
      "Uma captura de ecrã da página com as suas marcações ajuda os modelos de visão a compreender os seus comentários, mas utiliza mais contexto.",
    annotationScreenshotsUnsupported:
      "Requer a aplicação para computador ou um navegador baseado em Chromium, como o Chrome ou o Edge.",
    annotationScreenshotsHint:
      "Num navegador web, é-lhe pedido que partilhe o separador sempre que enviar anotações.",
    annotationScreenshotsAlways: "Incluir sempre",
    annotationScreenshotsNever: "Nunca incluir",
    clearDataSetting: "Limpar dados de navegação",
    clearDataSettingDescription:
      "Limpar o histórico de navegação, o histórico de transferências e as páginas em cache. As páginas mantêm os cookies e os dados dos sites apenas até o Unsloth ser recarregado.",
    manage: "Gerir",
    fullView: {
      enter: "Entrar na vista completa",
      exit: "Sair da vista completa",
      minimize: "Minimizar chat",
      showChat: "Mostrar chat",
      showConversation: "Mostrar conversa",
      hideConversation: "Ocultar conversa",
      newChat: "Novo chat",
      options: "Opções do chat",
      openChat: "Abrir em vista completa",
      moveLeft: "Mover para o painel esquerdo",
      moveRight: "Mover para o painel direito",
    },
    annotate: {
      countOne: "1 anotação",
      countMany: "{count} anotações",
      cancel: "Cancelar",
      send: "Enviar",
      move: "Mover",
      placeholder: "Descreva uma alteração ou faça uma pergunta",
      areaQuote: "Área da página",
      imageQuote: "Imagem",
      page: "Perguntar sobre esta página",
      pagePlaceholder: "Faça uma pergunta sobre isto",
      save: "Guardar comentário",
      dictate: "Ditar",
      stopDictating: "Parar de ditar",
      dictateFailed: "Não foi possível iniciar o ditado",
    },
    menu: {
      find: "Localizar na página",
      print: "Imprimir",
      printFailed: "Não foi possível imprimir esta página",
      screenshot: "Fazer uma captura de ecrã",
      zoom: "Zoom",
      zoomIn: "Ampliar",
      zoomOut: "Reduzir",
      zoomReset: "Repor zoom",
      deviceToolbar: "Mostrar barra de dispositivos",
      clearData: "Limpar dados de navegação",
    },
    screenshot: {
      added: "Captura de ecrã adicionada ao chat",
      taken: "Captura de ecrã efetuada",
      save: "Guardar",
      failed: "Não foi possível fazer a captura de ecrã",
      otherSurface:
        "Partilhe este separador para fazer uma captura de ecrã da página",
    },
    device: {
      mobile: "Telemóvel",
      tablet: "Tablet",
      close: "Ocultar barra de dispositivos",
    },
    downloads: {
      title: "Transferências",
      inProgressLabel: "A transferir",
      inProgress: "A transferir…",
      complete: "Transferência concluída",
      failed: "Falha na transferência",
      downloaded: "Transferido",
      missing: "Ficheiro movido ou eliminado",
      open: "Abrir",
      openFailed: "Não foi possível abrir {name}.",
      copyLink: "Copiar ligação de transferência",
      showAll: "Mostrar tudo",
      savePage: "Guardar página como…",
    },
    file: {
      openIn: "Abrir em",
      newChat: "Novo chat",
      newBrowserTab: "Novo separador do navegador",
      requestEdits: "Pedir alterações",
      requestEditsPrompt: "Edite {name}: ",
      preview: "Pré-visualização",
      source: "Código-fonte",
      viewMode: "Vista do ficheiro",
      runAgain: "Executar novamente",
      console: "Consola",
      copy: "Copiar conteúdo",
      copied: "Copiado",
      copyFailed: "Não foi possível copiar o ficheiro.",
      copyPath: "Copiar caminho",
      pathCopied: "Caminho copiado",
      openFailed: "Não foi possível abrir {name}.",
      wrap: "Moldar linhas",
      reactNeedsNode:
        "As pré-visualizações de React requerem o Node.js 20.19 ou posterior. Instale-o se estiver em falta e, em seguida, execute novamente a configuração do Unsloth.",
      reactFailed:
        "Não foi possível preparar esta pré-visualização. Execute-a novamente para tentar outra vez.",
    },
    video: {
      play: "Reproduzir",
      pause: "Pausa",
      seek: "Procurar",
      volume: "Volume",
      mute: "Desativar som",
      unmute: "Ativar som",
      more: "Mais opções de reprodução",
      speed: "Velocidade de reprodução",
      normalSpeed: "Normal",
      loop: "Repetir",
      pictureInPicture: "Imagem na imagem",
      fullscreen: "Ecrã inteiro",
      exitFullscreen: "Sair do ecrã inteiro",
      copyFrame: "Copiar fotograma",
      frameCopied: "Fotograma copiado",
      copyFrameFailed: "Não foi possível copiar o fotograma.",
      copyOptions: "Opções de cópia",
      open: "Abrir",
      openOptions: "Opções de abertura",
    },
    pages: {
      searchDownloads: "Pesquisar histórico de transferências",
      clearAll: "Limpar tudo",
      deleted: "Eliminado",
      showInFinder: "Mostrar no Finder",
      showInExplorer: "Mostrar no Explorador",
      showInFolder: "Mostrar na pasta",
      revealFailed: "Não foi possível mostrar {name} na respetiva pasta.",
      history: "Histórico",
      downloads: "Transferências",
      bookmarks: "Marcadores",
      search: "Pesquisar",
      today: "Hoje",
      yesterday: "Ontem",
      remove: "Remover",
      clearHistory: "Limpar histórico",
      clearDownloads: "Limpar transferências",
      noHistory: "As páginas que visitar no navegador aparecem aqui.",
      showMore: "Mostrar mais",
      noDownloads:
        "Os ficheiros que transferir a partir do navegador aparecem aqui.",
      noBookmarks:
        "As páginas que adicionar aos marcadores com a estrela na barra de endereço aparecem aqui.",
      noMatches: "Nada corresponde à sua pesquisa.",
      fromChat: "Do chat",
      searchHistory: "Pesquisar histórico de navegação",
      filter: "Filtrar",
      allTime: "Sempre",
      allTimeHistory: "Todo o histórico",
      lastWeek: "Últimos 7 dias",
      lastMonth: "Últimos 30 dias",
      customDates: "Escolher datas…",
      clearFilter: "Limpar filtro",
      openPage: "Abrir página",
      removeFromHistory: "Remover do histórico",
      selected: "{count} selecionados",
      cancel: "Cancelar",
      select: "Selecionar {title}",
      collapse: "Fechar {date}",
      expand: "Expandir {date}",
      pageActions: "Ações para {title}",
    },
    bookmarks: {
      bookmarkPage: "Adicionar esta página aos marcadores",
      add: "Adicionar marcador",
      edit: "Editar marcador",
      name: "Nome",
      location: "Localização",
      toolbar: "Barra de marcadores",
      other: "Outros marcadores",
      showEditor: "Mostrar editor ao guardar",
      cancel: "Cancelar",
      remove: "Remover marcador",
      delete: "Eliminar marcador",
      save: "Guardar",
      saved: "Guardado nos marcadores",
      removed: "Marcador removido",
      undo: "Anular",
      manage: "Gerir marcadores",
      more: "Mais marcadores",
      toolbarEmpty:
        "Adicione uma página aos marcadores com a estrela na barra de endereço para a adicionar aqui.",
      toolbarAlways: "Mostrar sempre",
      toolbarNewTab: "Mostrar apenas num novo separador",
      toolbarNever: "Nunca mostrar",
    },
    clearData: {
      title: "Limpar dados de navegação?",
      description:
        "Esta ação limpa o histórico de navegação, o histórico de transferências, as páginas em cache e os cookies e dados dos sites, o que termina a sua sessão nos sites. Os ficheiros transferidos permanecem no seu computador.",
      cancel: "Cancelar",
      confirm: "Limpar dados",
      done: "Dados de navegação limpos",
    },
    error: {
      title: "Não foi possível carregar esta página",
      description:
        "{host} não respondeu, recusou a ligação ou bloqueou o pedido.",
      retry: "Tentar novamente",
      botCheckTitle: "Este site verifica se está a utilizar um navegador real",
      botCheckDescription:
        "{host} pede aos visitantes que provem que são humanos, o que a vista na aplicação não consegue fazer aqui. Em alternativa, abra-o no seu navegador.",
      botCheckDescriptionWeb:
        "{host} pede aos visitantes que provem que são humanos, o que a vista na aplicação não consegue fazer aqui. Abra-o no seu navegador ou no {app}, que o consegue abrir.",
      desktopApp: "Unsloth Desktop",
      resubmit:
        "Esta página resultou de um formulário. Carregá-la novamente volta a enviar o formulário, o que pode repetir o que este fez.",
      upload:
        "O carregamento de ficheiros não funciona nesta vista. Abra a página no navegador do sistema para o enviar.",
      scriptNavigation:
        "Esta página tentou ir para outro endereço através de um script, o que esta vista não consegue seguir. Em alternativa, abra-a no navegador do sistema.",
      needsBrowser:
        "Esta página tentou utilizar uma funcionalidade do navegador que esta vista bloqueia, pelo que algumas partes podem não funcionar. Em alternativa, abra-a no navegador do sistema.",
    },
    native: {
      externalPrompt: "{host} pretende abrir {url}",
      open: "Abrir",
      downloading: "A transferir {name}",
      downloaded: "{name} guardado em Transferências",
      downloadFailed: "Não foi possível transferir {name}",
      notMarked:
        "{name} foi guardado, mas não foi possível marcá-lo como transferido da Internet, pelo que o sistema não o avisará antes de o abrir.",
      blocked:
        "Este endereço não pode ser aberto no painel do navegador. Este só abre sites públicos.",
      blockedLocal:
        "Este endereço local não pode ser aberto aqui. Escreva-o na barra de endereço para o abrir.",
      studioAddress:
        "Este é o endereço do próprio Unsloth, pelo que não pode ser aberto no painel do navegador.",
      clearDataSettingDescription:
        "Limpar o histórico de navegação, o histórico de transferências, as páginas em cache e os cookies e dados que as páginas guardam, o que termina a sua sessão nos sites.",
      clearDataDescription:
        "Esta ação limpa o histórico de navegação, o histórico de transferências, as páginas em cache e os cookies e dados dos sites, o que termina a sua sessão nos sites. Os ficheiros transferidos permanecem no seu computador.",
      clearDataFailed:
        "Não foi possível limpar os dados dos sites. Tente novamente.",
    },
  },
  managedEngines: {
    gpu: "GPUs",
    gpuHelp: "Escolha as GPUs a utilizar.",
    parallelism: "Modo multi-GPU",
    tensor: "Paralelismo de tensores",
    pipeline: "Paralelismo de pipeline",
    data: "Réplicas (paralelismo de dados)",
    pipelineHelp:
      "Divide as camadas do modelo pelas GPUs. Cada GPU tem de comportar as suas camadas e a cache.",
    dataHelp:
      "Distribui os pedidos pelas GPUs. Os modelos densos têm de caber em cada GPU; os modelos MoE podem partilhar especialistas.",
    tensorParallel: "Divide os cálculos das camadas por {count} GPUs.",
    inUse:
      "Descarregue o modelo antes de reparar, restaurar ou remover o respetivo motor.",
    title: "Mais motores",
    description:
      "Motores adicionais para GPUs (Beta). Podem executar modelos mais rapidamente, sobretudo em inferência em lote. Instale-os aqui e, em seguida, escolha-os nas definições de execução do modelo.",
    picker: "Motor de inferência",
    precision: "Precisão",
    precisionAuto: "Predefinição do modelo",
    precisionHelp:
      "A predefinição do modelo preserva a precisão armazenada. As outras opções convertem os pesos não quantizados durante o carregamento.",
    kvCache: "Cache KV",
    kvCacheFp8: "FP8 (metade da memória)",
    kvCacheHelp:
      "O FP8 armazena a cache de contexto em 8 bits, o que permite colocar cerca do dobro do contexto na mesma memória. Sem escalas calibradas no checkpoint, as respostas podem ser ligeiramente menos precisas.",
    gpuMemory: "Limite de memória da GPU",
    gpuMemoryAuto: "Automático",
    gpuMemoryHelp:
      "O máximo de cada GPU selecionada que o motor pode utilizar. Automático utiliza o que estiver livre; reduza-o para deixar espaço para outros programas.",
    seed: "Semente",
    seedPlaceholder: "Predefinição do motor",
    seedHelp:
      "Define a semente aleatória do motor. Chats simultâneos podem ainda assim alterar as respostas amostradas.",
    default: "Predefinição",
    installRequired: "instalação necessária",
    ownerRequired: "Peça ao proprietário do Unsloth para instalar este motor.",
    cancelInstall: "Cancelar instalação",
    installed: "{version} instalada",
    repair: "Reparar",
    update: "Atualizar motor",
    install: "Instalar motor",
    installing: "A instalar o motor. Consulte o progresso em Transferências.",
    remove: "Remover motor",
    rollback: "Restaurar instalação anterior",
    installTitle: "Instalar {engine}",
    installAndLoad: "Instalar e carregar",
    confirm:
      "Instalar {engine} {version}? Esta ação pode transferir vários GB.",
    confirmSized: "Instalar {engine} {version}? Cerca de {size} a transferir.",
    wslSetup:
      "No Windows, {engine} é executado num ambiente WSL2 privado configurado pelo Unsloth. Será apresentado um pedido de permissão de administrador e poderá ser necessário reiniciar.",
    wslReady:
      "No Windows, {engine} é executado no ambiente WSL2 privado do Unsloth.",
    wslRestart:
      "Reinicie o Windows para concluir a ativação do WSL2 e, em seguida, clique novamente em Instalar.",
    background: "Instala em segundo plano.",
    failed:
      "A instalação do motor falhou. Tente novamente ou utilize o motor predefinido.",
    details: "Detalhes técnicos",
    cancelled: "Instalação cancelada. Pode tentar novamente.",
    scope:
      "Chat de texto e imagem em GPUs NVIDIA. O primeiro carregamento pode demorar mais tempo.",
  },
  sandboxSetup: {
    levelTitle: "A sandbox do sistema operativo não está disponível",
    levelDescription:
      "Instale-a para utilizar o nível Alto ou mantenha o nível Baixo (sandbox por software).",
    useLow: "Utilizar sandbox Baixa",
    checking: "A verificar a sandbox neste computador…",
    install: "Instalar sandbox",
    windowsSetup: "Configurar a sandbox do Windows",
    copyCommand: "Copiar comando",
    copied: "Comando copiado",
    copyFailed: "Não foi possível copiar o comando",
    copiedRunIt:
      "Comando copiado. Execute-o num terminal no computador onde o Unsloth está a ser executado e, em seguida, mude para Alto.",
    ownerOnly:
      "Apenas o proprietário a pode instalar, a partir do computador onde o Unsloth está a ser executado.",
    windowsNote:
      "Requer um passo de administrador após cada reinício. O MXC é uma pré-visualização, não uma barreira de segurança.",
    running:
      "A configurar. Aprove o pedido de palavra-passe ou de permissão de administrador no computador onde o Unsloth está a ser executado…",
    succeeded:
      "A sandbox do sistema operativo está pronta. A sandbox Alta está ativada.",
    stillUnavailable:
      "A configuração terminou, mas a sandbox do sistema operativo continua a não passar na verificação.",
    declined:
      "O pedido de palavra-passe ou de permissão de administrador foi recusado.",
    failed: "A configuração falhou.",
    startError: "Não foi possível iniciar a configuração da sandbox",
    checkFailed: "Não foi possível verificar a sandbox neste computador.",
    retry: "Tentar novamente",
    keepsRunning:
      "Fechar esta janela não interrompe a configuração. O resultado aparece em Definições > Segurança.",
  },
  composerSettings: {
    title: "Caixa de mensagem",
    plainText: "Caixa de mensagem em texto simples",
    plainTextDescription:
      "Manter o código, o Markdown e as ligações como texto literal. Desative para mostrar uma pré-visualização formatada.",
    showContext: "Utilização da janela de contexto",
    sendShortcut: "Atalho de envio",
    sendMultiline: "{mod} + Enter para prompts com várias linhas",
    sendAlways: "Sempre {mod} + Enter",
    sendEnterDescription: "Enter envia. Shift + Enter adiciona uma nova linha.",
    sendMultilineDescription:
      "Enter envia um prompt de uma linha. Quando tiver mais de uma linha, Enter adiciona uma nova linha e {mod} + Enter envia.",
    sendAlwaysDescription:
      "{mod} + Enter envia. Enter adiciona uma nova linha.",
    followUp: "Comportamento das mensagens seguintes",
    followUpDescription:
      "O que acontece quando envia durante uma resposta. Prima {shortcut} para fazer o contrário uma vez.",
    followUpMultilineShortcut:
      "{shortcut} ({multiline} num prompt com várias linhas)",
    queue: "Colocar em fila",
    steer: "Redirecionar",
    steerDescription:
      "Redirecionar interrompe a resposta atual e envia a sua mensagem seguinte de imediato.",
    settings: "Definições da caixa de mensagem",
    preview: "Pré-visualização formatada",
  },
  promptQueue: {
    loading: "A carregar prompts em fila",
    listLabel: "Prompts em fila",
    regionLabel: "Fila de prompts, {current} de {total}",
    itemLabel: "Prompt em fila {position} de {total}: {prompt}",
    reorderInstructions:
      "Arraste a pega para reordenar. Com a pega em foco, utilize Seta para cima ou Seta para baixo para mover uma posição, ou Home ou End para mover para o início ou o fim.",
    dragTooltip: "Arraste para reordenar",
    reorderLabel: "Reordenar o prompt em fila {position} de {total}",
    paused: "Em pausa",
    steer: "Redirecionar",
    steerTooltip: "Interromper a resposta e enviar este prompt a seguir",
    steerLabel: "Redirecionar com o prompt em fila {position}",
    removeTooltip: "Remover da fila",
    removeLabel: "Remover o prompt em fila {position}",
    moreTooltip: "Mais opções",
    moreLabel: "Mais opções para o prompt em fila {position}",
    editItem: "Editar mensagem",
    copyItem: "Copiar mensagem",
    editLabel: "Editar o prompt em fila {position}",
    cancel: "Cancelar",
    save: "Guardar",
    turnOffQueueing: "Desativar a fila",
    turnOnQueueing: "Ativar a fila",
    resume: "Retomar a fila",
    queueButton: "Colocar mensagem em fila",
    steerButton: "Redirecionar a resposta",
    sendTooltip: "Enviar mensagem ({shortcut})",
    sendLabel: "Enviar mensagem",
    followUpTooltip: "{action} ({send}) · {opposite} para o contrário",
    announceUpdated: "Prompt em fila atualizado.",
    announceEditFailed:
      "Já não é possível editar este prompt porque a fila foi alterada.",
    announceRemoved: "Prompt removido da fila.",
    announceSteered: "Este prompt vai redirecionar a resposta a seguir.",
    announceSteerFailed:
      "Não foi possível redirecionar a resposta com este prompt. Verifique a fila e tente novamente.",
    announceCopied: "Prompt copiado.",
    announceCopyFailed: "Não foi possível copiar este prompt. Tente novamente.",
    announceQueueingOn:
      "As novas mensagens seguintes ficarão em fila após a resposta atual.",
    announceQueueingOff:
      "As novas mensagens seguintes vão redirecionar a resposta atual.",
    announceMoved: "Prompt movido para a posição {position} de {total}.",
    announceMoveFailed:
      "A fila foi alterada antes de ser possível mover este prompt. Tente novamente.",
    announceDragReset:
      "A fila foi alterada. Arraste novamente para reordenar os prompts restantes.",
    editingHint: "A editar a mensagem",
    queueingOffHint:
      "As novas mensagens interrompem e são executadas a seguir.",
    queueingOnHint: "As novas mensagens aguardam a sua vez.",
    queueingHintShared: "A fila é mantida.",
  },
  rewind: {
    title: "Recuar",
    confirmTitle: "Recuar até esta mensagem?",
    restoreLabel: "Restaurar",
    pickDescription: "Escolha uma mensagem até à qual recuar.",
    pickEmpty: "Ainda não há mensagens até às quais recuar.",
    attachmentOnly: "Apenas anexos",
    modeBoth: "Conversa e ficheiros",
    modeConversation: "Apenas a conversa",
    modeFiles: "Apenas os ficheiros",
    checkFailed:
      "Não foi possível verificar as alterações aos ficheiros: {error}",
    summary:
      "Ficheiros a restaurar: {restore}. A remover: {remove}. A recriar: {recreate}.",
    conflicts: "Alterados entretanto, serão ignorados",
    untracked: "Não podem ser restaurados",
    scopeNote:
      "Os ficheiros incluem código. Os programas, pacotes e ficheiros fora da pasta deste chat não são alterados.",
    back: "Voltar",
    cancel: "Cancelar",
    confirm: "Recuar",
    waitForResponse: "Aguarde que a resposta atual termine",
    restoredOne: "1 ficheiro restaurado",
    restoredMany: "{count} ficheiros restaurados",
    nothingRestored: "Não foi necessário restaurar nenhum ficheiro",
    skippedCount: "{count} ignorados",
    filesNotRestored:
      "Não é possível restaurar {count} ficheiros alterados, que ficam como estão.",
    restoreFailed: "Não foi possível restaurar os ficheiros",
    undo: "Anular",
    undone: "Restauro de ficheiros anulado",
    undoFailed: "Não foi possível anular o restauro de ficheiros",
  },
  chatMenu: {
    more: "Opções do chat",
    copy: "Copiar",
    compact: "Compactar conversa",
    compactQueued: "Compactação pedida para a próxima mensagem",
    compactEmpty: "Ainda não há nada para compactar",
    compactUnsupported:
      "A compactação requer um modelo GGUF ou MLX com suporte de ferramentas",
  },
  halloween: {
    navLabel: "Doçura ou travessura…",
    title: "Doçura ou travessura?!",
    description:
      "Preparámos alguns sustos para o Unsloth, disponíveis até ao fim de outubro.",
    chooseTheme: "Escolha um tema assustador",
    spookyGreetings: "Saudações arrepiantes",
    spookyGreetingsDescription:
      "A preguiça recebe os novos chats com um susto.",
    decorations: "Aranhas e morcegos",
    decorationsDescription:
      "Teias e, de vez em quando, uma aranha ou um morcego.",
    surprise: "Doçura ou travessura!",
    surpriseDescription: "Deixe a preguiça escolher um tema por si.",
    themesTitle: "Temas assustadores",
    back: "Voltar",
    footer: "O Unsloth volta ao normal à meia-noite de 1 de novembro.",
    removeNow: "Remover agora",
    hideFromSidebar: "Ocultar da barra lateral",
    greeting: {
      pumpkins: "As abóboras já estão esculpidas. E agora?",
      brewing: "Vamos preparar a poção",
      trickOrTreat: "Doçura ou travessura?",
      manor: "Bem-vindo à mansão",
      manorName: "Bem-vindo à mansão, {name}",
      happyHalloween: "Feliz Halloween",
      happyHalloweenName: "Feliz Halloween, {name}",
      spirits: "Os espíritos estão a ouvir…",
      mysteries: "Que mistérios vamos desvendar esta noite?",
      conjure: "O que posso conjurar por si?",
    },
  },
  welcome: {
    goodToSeeYou: "É bom vê-lo por cá",
    goodToSeeYouName: "É bom vê-lo por cá, {name}",
    readyWhenYouAre: "Pronto quando quiser",
    onYourMind: "Em que está a pensar hoje?",
    howCanIHelp: "Como posso ajudar?",
    goodMorning: "Bom dia",
    goodMorningName: "Bom dia, {name}",
    goodEvening: "Boa noite",
    goodEveningName: "Boa noite, {name}",
    tonight: "Quais são os planos para esta noite?",
    nightOwl: "Modo noctívago?",
    lateNightIdeas: "Ideias de madrugada?",
    upLate: "Acordado até tarde com uma ideia?",
    nightShift: "Começa o turno da noite",
    nightShiftName: "Começa o turno da noite, {name}",
    temporaryChat: "Chat temporário",
    temporaryChatDescription:
      "Este chat não aparecerá no seu histórico e não é guardado. Desaparece quando sair.",
  },
  picker: {
    unifiedTightFit: "Ajuste apertado",
    unifiedTightFitHint:
      "Pode carregar com um contexto mais curto ou ser executado mais lentamente neste Mac.",
    onDevice: "No dispositivo",
    huggingFace: "Hugging Face",
    retry: "Tentar novamente",
    loadMore: "Carregar mais",
    offlineTitle: "Está offline",
    offlineBody:
      "Mude para Dispositivo para utilizar {noun} locais ou em cache.",
    offlineSwitchDevice: "Dispositivo",
    searchAriaLabel: "Pesquisar {noun}",
    modelSourceAriaLabel: "Origem do modelo",
    hubSectionAriaLabel: "Secção do Hub",
    modelDropped: "Já não disponibilizado",
    modelDroppedByProvider: "{provider} · já não disponibilizado",
    modelDisabled: "Não ativado",
    modelDisabledByProvider: "{provider} · não ativado",
    multipleMatches: "Vários {noun} correspondentes. Escolha um da lista.",
    rateLimitedTitle: "Limite de pedidos do Hugging Face atingido",
    rateLimitedBody:
      "Aguarde um momento e, em seguida, tente pesquisar {noun} novamente.",
    modelScope: "ModelScope",
    useModelScope: "Utilizar o ModelScope",
    useModelScopeHint:
      "Pesquisar e transferir a partir do ModelScope. Pode voltar atrás nas Definições.",
    useModelScopeFailed: "Não foi possível mudar para o ModelScope.",
    updateToken: "Atualizar token",
    hfToken: {
      label: "Token do HF",
      saved: "Guardado",
      add: "Não definido",
      savedAriaLabel: "Token do Hugging Face guardado",
      addAriaLabel: "Definir token do Hugging Face",
      savedHint: "Token guardado. O acesso é verificado quando o utilizar.",
      addHint:
        "Defina um token para aceder a repositórios privados e restritos.",
    },
  },
  common: {
    duplicate: "Duplicar",
    cancel: "Cancelar",
    close: "Fechar",
    delete: "Eliminar",
    done: "Concluir",
    error: "Erro",
    export: "Exportar",
    help: "Ajuda",
    todayAt: "Hoje, {time}",
    yesterdayAt: "Ontem, {time}",
    loading: "A carregar...",
    new: "Novo",
    rename: "Mudar o nome",
    save: "Guardar",
    saving: "A guardar...",
    search: "Pesquisar",
    shutdown: "Encerrar",
  },
  shell: {
    helpMenu: {
      documentation: "Documentação",
      keyboardShortcuts: "Atalhos de teclado",
      whatsNew: "Novidades",
      troubleshooting: "Resolução de problemas",
      systemStatus: "Estado do sistema",
      sendFeedback: "Enviar comentários",
      about: "Acerca do Unsloth",
    },
    find: {
      label: "Localizar na página",
      previous: "Correspondência anterior",
      next: "Correspondência seguinte",
      close: "Fechar pesquisa",
      truncated:
        "Esta página é demasiado longa para ser pesquisada na totalidade.",
      searchChat: "Pesquisar no chat",
      searchBrowser: "Pesquisar na página do navegador",
    },
    zoom: {
      label: "Zoom",
      zoomOut: "Reduzir",
      zoomIn: "Ampliar",
      reset: "Repor",
      announce: "Zoom {percent}%",
    },
    beta: "BETA",
    brand: "unsloth",
    product: "Unsloth",
    accountMenu: "Menu da conta de {name}",
    updateAvailable: "Atualização disponível",
    resize: {
      collapse: "Clique para fechar",
      expand: "Clique para expandir",
      drag: "Arraste para redimensionar",
    },
    aria: {
      home: "Página inicial do Unsloth",
      closeSidebar: "Fechar barra lateral",
      openSidebar: "Abrir barra lateral",
      resizeSidebar: "Redimensionar ou fechar a barra lateral",
      resizeRunSettings: "Redimensionar ou fechar as definições de execução",
      openRunSettings: "Abrir as definições de execução",
      chatOptions: "Opções do chat",
      runOptions: "Opções de execução",
    },
    navigation: {
      newChat: "Novo chat",
      returnToChat: "Voltar ao chat",
      returnToChats: "Voltar aos {count} chats",
      chatGenerating: "A gerar",
      chatAwaitingApproval: "A aguardar aprovação",
      chatUnread: "Não lido",
      compare: "Comparar",
      search: "Pesquisar",
      projects: "Projetos",
      library: "Biblioteca",
      hub: "Hub de modelos",
      train: "Treinar",
      recipes: "Receitas",
      images: "Imagens",
      video: "Vídeo",
      audio: "Áudio",
      tasks: "Agendadas",
      trainChecking: "A verificar se este computador suporta treino...",
      videoChecking: "A verificar se este computador suporta vídeo...",
      more: "Mais",
      customizeSidebar: "Personalizar barra lateral",
      newBadge: "Novo",
      export: "Exportar",
      recents: "Recentes",
      noChatsYet: "Ainda não há chats",
      noChats: "Nenhum chat",
      allProjectsPinned: "Todos os projetos estão afixados",
      allProjectsFiled: "Todos os projetos estão afixados ou em secções",
      noProjects: "Nenhum projeto",
      showMore: "Mostrar mais",
      showLess: "Mostrar menos",
      settings: "Definições",
      api: "API",
      lightMode: "Modo claro",
      darkMode: "Modo escuro",
      guidedTour: "Visita guiada",
      help: "Ajuda",
      logOut: "Terminar sessão",
      logOutNeedsPassword:
        "Defina uma palavra-passe em Definições > Remoto e LAN antes de terminar sessão, para poder voltar a iniciar sessão.",
      logOutFailed:
        "Não foi possível terminar sessão em segurança. Tente novamente daqui a pouco.",
      shutdown: "Encerrar",
    },
    commandPalette: {
      placeholder: "Escreva um comando ou pesquise...",
      noResults: "Nenhum resultado encontrado.",
      navigation: "Navegação",
      actions: "Ações",
      chat: "Chat",
      searchChats: "Pesquisar chats...",
    },
    search: {
      placeholder: "Pesquisar",
      tabs: {
        all: "Tudo",
        chats: "Chats",
        projects: "Projetos",
        files: "Ficheiros",
        models: "Modelos",
      },
      recents: "Recentes",
      actions: "Ações",
      newChat: "Novo chat",
      newTemporaryChat: "Novo chat temporário",
      fineTune: "Afinar um modelo",
      generateImage: "Gerar uma imagem",
      generateVideo: "Gerar um vídeo",
      untitledChat: "Chat sem título",
      compare: "Comparar",
      loading: "A carregar…",
      empty: {
        all: "Ainda não há nada para pesquisar.",
        chats: "Ainda não há chats.",
        projects: "Ainda não há projetos.",
        files: "Ainda não há ficheiros na sua Biblioteca.",
        models: "Ainda não há modelos transferidos.",
      },
      noMatches: "Sem resultados.",
      when: {
        today: "Hoje",
        pastWeek: "Última semana",
        pastMonth: "Último mês",
        older: "Mais antigos",
      },
      footer: {
        close: "Fechar",
        changeType: "Mudar tipo",
        open: "Abrir",
      },
    },
    notFound: {
      title: "Página não encontrada",
      description: "{path} não existe.",
      backToChat: "Voltar ao chat",
    },
    selection: {
      pinProjects: "Afixar projetos",
      unpinProjects: "Desafixar projetos",
      deleteProjects: "Eliminar projetos",
      deleteProjectsTitle: "Eliminar projetos",
      deleteProjectsDescription:
        "Eliminar {count} projetos? Os respetivos chats são eliminados permanentemente.",
      deleteProjectsFilesDescription:
        "A pasta do espaço de trabalho de cada projeto é removida do disco.",
      countSelected: "{count} selecionados",
      pinChats: "Afixar chats",
      unpinChats: "Desafixar chats",
      archiveChats: "Arquivar chats",
      markUnread: "Marcar como não lido",
      markRead: "Marcar como lido",
      deleteChats: "Eliminar chats",
      deleteTitle: "Eliminar chats",
      deleteDescription:
        "Eliminar {count} chats? Esta ação não pode ser anulada.",
      deleteFilesDescription:
        "A pasta da sandbox própria de cada chat é removida do disco. Os ficheiros que estes escreveram dentro de um projeto permanecem no espaço de trabalho desse projeto.",
      deleteFilesLabel: "Eliminar ficheiros e pasta da sandbox",
      deleteChatFilesDescription:
        "A pasta da sandbox própria deste chat é removida do disco. Os ficheiros que este escreveu dentro de um projeto permanecem no espaço de trabalho desse projeto.",
    },
    organize: {
      sidebarHeading: "Organizar a barra lateral",
      byProject: "Por projeto",
      inOneList: "Numa só lista",
      sortChatsBy: "Ordenar chats por",
      projectsHome: "Página inicial dos projetos",
      lastUpdated: "Última atualização",
      manualOrder: "Ordem manual",
      switchedToManual: "Ordenação manual: arraste as linhas para reordenar",
      organizeChats: "Organizar chats",
      organizeProjects: "Organizar projetos",
      sortPinnedChats: "Ordenar chats afixados",
      show: "Mostrar",
      newSection: "Nova secção",
    },
    sections: {
      createTitle: "Nova secção",
      createDescription: "Agrupe chats e projetos como preferir",
      namePlaceholder: "Nome da secção",
      create: "Criar secção",
      renameTitle: "Mudar o nome da secção",
      renameDescription: "Dê um novo nome a esta secção",
      edit: "Editar",
      remove: "Remover secção",
      markAllRead: "Marcar tudo como lido",
      moveTo: "Mover para",
      section: "Secção",
      sectionsHeading: "Secções",
      removeFromProject: "Remover do projeto",
      newSection: "Nova secção",
      removeFromSection: "Remover da secção",
      removeFrom: "Remover de {name}",
      empty: "Arraste chats ou projetos para aqui",
      sectionOptions: "Opções da secção",
      newChatInSection: "Novo chat em {name}",
      deleted: "Secção “{name}” eliminada",
      undo: "Anular",
    },
    dialog: {
      deleteChat: {
        title: "Eliminar chat",
        description:
          'Tem a certeza de que pretende eliminar este chat "{name}"?',
      },
      deleteRun: {
        title: "Eliminar execução de treino",
        description:
          'Tem a certeza de que pretende eliminar esta execução "{name}"?',
      },
      renameChat: {
        title: "Mudar o nome do chat",
        placeholder: "Título do chat",
      },
      renameRun: {
        title: "Mudar o nome da execução",
        placeholder: "Nome da execução",
      },
    },
    toast: {
      cannotDeleteRunningRun:
        "Não é possível eliminar uma execução de treino em curso",
      failedToDeleteChat: "Não foi possível eliminar o chat",
      failedToDeleteRun: "Não foi possível eliminar a execução",
      failedToRenameChat: "Não foi possível mudar o nome do chat",
      failedToRenameRun: "Não foi possível mudar o nome da execução",
    },
  },
  settings: {
    accounts: {
      signsInAs: "Inicia sessão como {username}",
      signsInAsOwner: "Inicia sessão como {alias} ou {username}",
      signInName: "O seu nome de início de sessão",
      signInNameDescription:
        "O nome que escreve para iniciar sessão quando esta instalação tem mais de uma conta. unsloth também funciona sempre. Deixe vazio para utilizar apenas unsloth.",
      signInNameSave: "Guardar nome",
      title: "Contas",
      description:
        "Crie contas privadas do Unsloth. Os novos utilizadores iniciam sessão com um código de configuração de utilização única e escolhem uma palavra-passe.",
      username: "Nome de utilizador",
      create: "Criar conta",
      createDescription:
        "Partilhe um código de configuração para que a pessoa possa escolher a sua própria palavra-passe.",
      actionsFor: "Gerir {username}",
      actions: "Ações",
      search: "Pesquisar contas",
      noResults: "Nenhuma conta correspondente",
      created: "Criada",
      status: "Estado",
      loginHint:
        "Inicie sessão como {username} com este código como palavra-passe e, em seguida, escolha uma nova palavra-passe.",
      privateAccount: "Conta privada",
      modelAccess: "Acesso a modelos",
      modelAccessTitle: "Acesso a modelos de {username}",
      modelAccessDescription:
        "Escolha os modelos do Hugging Face que esta conta pode ver, transferir, carregar e utilizar através das respetivas chaves de API.",
      modelAccessLabel: "Modelos permitidos",
      modelAccessHint:
        "Um modelo por linha, por exemplo unsloth/Qwen3-4B-GGUF, ou unsloth/* para uma organização inteira. Deixe vazio para permitir todos os modelos.",
      modelAccessSave: "Guardar",
      modelAccessLimited: "Acesso a modelos limitado",
      empty: "Ainda não há outras contas",
      setupCode: "Código de configuração",
      setupFor: "Código de configuração para {username}",
      shownOnce:
        "Copie este código agora e partilhe-o com o titular da conta. Só é mostrado aqui e pode ser utilizado uma vez no prazo de 60 minutos.",
      expires: "Expira {expiry}",
      copy: "Copiar código de configuração",
      copied: "Copiado",
      copyFailed:
        "Não foi possível copiar. Selecione e copie o código de configuração acima.",
      dismiss: "Concluir",
      owner: "Proprietário da instalação",
      active: "Ativada",
      inactive: "Desativada",
      regenerate: "Gerar novo código de configuração",
      resetTitle: "Repor a palavra-passe de {username}?",
      resetDescription:
        "Gerar um novo código de configuração substitui a palavra-passe de {username}, termina as respetivas sessões e revoga as respetivas chaves de API. Entregue-lhe o novo código para que possa escolher novamente uma palavra-passe.",
      deactivate: "Desativar",
      reactivate: "Ativar",
      delete: "Eliminar conta",
      deleteTitle: "Eliminar {username}?",
      deleteDescription:
        "Esta ação revoga as sessões de {username} e cancela o respetivo trabalho. Os chats, definições, credenciais, carregamentos, conjuntos de dados, execuções de treino, resultados, exportações, galerias, sandboxes, projetos e ficheiros temporários desta conta são retirados. Os diretórios são renomeados e colocados à parte, nunca eliminados. Criar novamente este nome de utilizador inicia uma conta nova sem nenhum destes dados.",
      cancel: "Cancelar",
      retry: "Atualizar contas",
      failed: "O pedido da conta falhou.",
    },
    title: "Definições",
    dialog: {
      title: "Definições",
      description: "Gira as suas preferências do Unsloth.",
      closeAriaLabel: "Fechar definições",
      searchPlaceholder: "Pesquisar definições…",
      searchNoResults: "Nenhuma definição encontrada.",
      panelFailed: "Não foi possível carregar esta secção.",
      panelReload: "Recarregar",
    },
    library: {
      layoutSection: "Esquema",
      cardSize: "Tamanho dos cartões",
      cardSizeDescription: "Quantos cartões cabem numa linha.",
      small: "Pequeno",
      medium: "Médio",
      large: "Grande",
      imageLayout: "Cartões de imagem",
      imageLayoutDescription:
        "Manter o formato de cada imagem ou recortá-las em quadrados iguais.",
      masonry: "Formato original",
      square: "Quadrado",
      showCardDates: "Mostrar datas nos cartões",
      showCardDatesDescription:
        "Mostrar quando cada ficheiro foi alterado pela última vez.",
      browsingSection: "Navegação",
      startTab: "Abrir em",
      startTabDescription: "O separador em que a Biblioteca abre.",
      suggested: "Sugestões",
      all: "Tudo",
      sort: "Ordenar por",
      sortDescription:
        "Ordem dos ficheiros e pastas. Sugestões mostra sempre os mais recentes.",
      recent: "Mais recentes",
      oldest: "Mais antigos",
      name: "Nome",
      suggestedLimit: "Itens sugeridos",
      suggestedLimitDescription:
        "Quantos ficheiros recentes são mostrados em Sugestões.",
      tabAuto: "Quando tiverem ficheiros",
      tabAlways: "Sempre",
      contentSection: "Conteúdo",
      contentDescription:
        "Escolha o que a Biblioteca mostra. Os ficheiros ocultos não são eliminados.",
      showChatAttachments: "Anexos de chats",
      showChatAttachmentsDescription: "Ficheiros que anexou a chats.",
      showChatToolFiles: "Ficheiros de ferramentas do chat",
      showChatToolFilesDescription:
        "Ficheiros criados por código e ferramentas durante os chats.",
      showGeneratedMedia: "Multimédia gerada",
      showGeneratedMediaDescription:
        "Imagens, vídeos e áudio das páginas Imagens, Vídeo e Áudio.",
      showFineTunesDescription: "Modelos que treinou ou exportou no Unsloth.",
      deletingSection: "Eliminação",
      confirmDelete: "Confirmar antes de eliminar",
      confirmDeleteDescription:
        "Perguntar antes de eliminar ficheiros e pastas da Biblioteca.",
      reset: "Repor definições da Biblioteca",
      resetDescription:
        "Repor cada uma das opções acima com o valor predefinido.",
      resetButton: "Repor",
      resetDone: "Definições da Biblioteca repostas",
      storageSection: "Armazenamento",
      downloadsSection: "Transferências de imagens",
      imageInfoStamp: "Carimbo de informações",
      imageInfoStampDescription:
        "Mostrar o modelo, a data e a semente na parte inferior das imagens que transferir da página Imagens. As definições de geração são sempre guardadas dentro do ficheiro.",
      storageDescription:
        "Abra uma categoria para ver os respetivos ficheiros, do maior para o menor, e elimine o que já não precisa.",
      storageUsed: "{size} utilizados pela Biblioteca",
      storageDisk: "{free} livres de {total} neste disco",
      storageEmpty: "Ainda não há nada armazenado na Biblioteca.",
      storageHidden: "{size} ocultos nas definições de Conteúdo",
      storageHiddenLabel: "Oculto",
      storageRefresh: "Atualizar",
      storageOpen: "Abrir",
      storageError: "Não foi possível ler o armazenamento da Biblioteca.",
      locationsSection: "Ficheiros no disco",
      locationsDescription:
        "Onde o Unsloth guarda cada tipo de ficheiro. As pastas da Biblioteca são etiquetas dentro do Unsloth, não pastas no disco. Os modelos afinados e as exportações ficam onde estão, pois o treino e os chats referem-se a eles pelo caminho.",
      locationUploads: "Carregamentos da Biblioteca",
      locationExports: "Exportações",
      revealFailed: "Não foi possível abrir o gestor de ficheiros",
      locationReset: "Repor",
      locationMoveTitle: "Mover {name} para uma nova pasta",
      locationMoveDescription:
        "Escolha qualquer pasta. Se já tiver ficheiros, o Unsloth cria uma pasta própria dentro dela. Tudo o que já está aqui também é movido.",
      locationMoveAction: "Mover para aqui",
      locationMoving: "A mover {name}…",
      locationMoved: "{name} movido",
      locationMoveFailed: "Não foi possível mover {name}",
      locationFree: "{free} livres nesta unidade",
      locationUnavailable:
        "Indisponível. Volte a ligar a respetiva unidade ou reponha a pasta.",
      locationResetUnavailableTitle: "Repor {name}?",
      locationResetUnavailableDescription:
        "A respetiva unidade não está ligada. Os ficheiros em {path} permanecem nessa unidade e não aparecerão na Biblioteca até os mover de volta.",
      locationLeftBehind: "Os ficheiros em {path} permanecem nessa unidade.",
      dataStorage: "Armazenamento da Biblioteca",
      manageStorage: "Gerir armazenamento",
      categoryImages: "Imagens",
      categoryVideos: "Vídeos",
      categoryFineTunes: "Modelos afinados",
      itemCountOne: "1 item",
      itemCount: "{count} itens",
      tabsSection: "Separadores",
      tabsDescription:
        "Escolha os separadores que a Biblioteca mostra. Um separador aberto a partir de uma ligação continua a aparecer.",
      tabHidden: "Oculto",
      lastVisited: "Último visitado",
    },
    tabs: {
      sandbox: "Segurança",
      accounts: "Contas",
      general: "Geral",
      profile: "Perfil",
      appearance: "Aspeto",
      resources: "Sistema",
      chat: "Chat",
      voice: "Voz",
      connections: "Ligações",
      data: "Dados",
      apiKeys: "API",
      remoteLan: "Remoto e LAN",
      agents: "Agentes",
      keyboardShortcuts: "Atalhos",
      debugging: "Registos",
      about: "Acerca de",
    },
    keyboardShortcuts: {
      title: "Atalhos de teclado",
      intro:
        "Para alterar um atalho, selecione a combinação de teclas e, em seguida, prima as novas teclas",
      searchPlaceholder: "Pesquisar atalhos…",
      keystrokePlaceholder: "Prima um atalho para pesquisar",
      searchByKeystrokes: "Pesquisar por teclas",
      searchByName: "Pesquisar por nome",
      noResults: "Nenhum atalho corresponde a essa pesquisa.",
      unassigned: "nenhum",
      recording: "Prima a sequência de teclas",
      needsModifier:
        "Adicione ⌘, Ctrl ou Alt. Uma tecla isolada interferiria com a escrita.",
      conflict: "Também utilizado por outro atalho",
      conflictShadowed:
        "Outro atalho tem esta combinação e é executado em vez deste",
      edit: "Alterar atalho",
      reset: "Restaurar predefinição",
      resetAll: "Restaurar predefinições",
      browserReserved:
        "O seu navegador pode reservar esta combinação para si próprio. Funciona na aplicação para computador.",
      metaUnsupported: "A tecla Windows não pode ser utilizada em atalhos.",
      altGraphUnsupported:
        "Não é possível utilizar combinações com AltGr como atalhos.",
      desktopOnly: "Apenas na aplicação para computador",
      groups: {
        composer: "Caixa de mensagem",
        chat: "Chat",
        navigation: "Navegação",
        app: "Aplicação",
      },
      actions: {
        findInPage: {
          label: "Localizar na página",
          description: "Pesquisar o texto nesta página",
        },
        newBrowserTab: {
          label: "Novo separador do navegador",
          description: "Abrir um novo separador no navegador ao lado do chat",
        },
        toggleBrowserFullView: {
          label: "Alternar vista completa do navegador",
          description:
            "Mostrar o navegador em toda a janela, com o chat a flutuar por cima",
        },
        openSettings: {
          label: "Abrir definições",
          description: "Abrir a caixa de diálogo de definições",
        },
        openKeyboardShortcuts: {
          label: "Atalhos de teclado",
          description: "Abrir esta lista de atalhos",
        },
        openCommandPalette: {
          label: "Paleta de comandos",
          description: "Abrir a paleta de comandos",
        },
        searchChats: {
          label: "Pesquisar chats",
          description: "Abrir a caixa de diálogo de pesquisa de chats",
        },
        openMcpServers: {
          label: "Servidores MCP",
          description: "Configurar os servidores MCP para este chat",
        },
        logOut: {
          label: "Terminar sessão",
          description: "Terminar sessão no Unsloth",
        },
        approveToolRequest: {
          label: "Aprovar pedido",
          description: "Permitir a chamada de ferramenta que aguarda aprovação",
        },
        declineToolRequest: {
          label: "Recusar pedido",
          description: "Recusar a chamada de ferramenta que aguarda aprovação",
        },
        newChat: {
          label: "Novo chat",
          description: "Iniciar um novo chat",
        },
        newTemporaryChat: {
          label: "Novo chat temporário",
          description: "Iniciar um chat que não é guardado no histórico",
        },
        newStandaloneChat: {
          label: "Novo chat independente",
          description: "Iniciar um novo chat fora de qualquer projeto",
        },
        archiveChat: {
          label: "Arquivar chat",
          description: "Arquivar os chats selecionados ou o chat atual",
        },
        markChatUnread: {
          label: "Marcar como não lido",
          description:
            "Marcar como não lidos os chats selecionados ou o chat atual",
        },
        togglePinChat: {
          label: "Alternar afixação",
          description:
            "Afixar ou desafixar os chats selecionados ou o chat atual",
        },
        selectAllChats: {
          label: "Selecionar todos os chats",
          description: "Selecionar todos os chats na barra lateral",
        },
        clearChatSelection: {
          label: "Limpar seleção",
          description:
            "Desmarcar os chats selecionados. A tecla Esc também o faz",
        },
        deleteSelectedChats: {
          label: "Eliminar chats selecionados",
          description: "Eliminar todos os chats selecionados",
        },
        nextRecentlyViewedChat: {
          label: "Chat visto recentemente seguinte",
          description: "Avançar pelos chats vistos recentemente",
        },
        previousRecentlyViewedChat: {
          label: "Chat visto recentemente anterior",
          description: "Recuar pelos chats vistos recentemente",
        },
        nextChat: {
          label: "Chat seguinte",
          description: "Mudar para o chat seguinte na barra lateral",
        },
        previousChat: {
          label: "Chat anterior",
          description: "Mudar para o chat anterior na barra lateral",
        },
        nextChatNeedingAttention: {
          label: "Chat seguinte que requer atenção",
          description:
            "Mudar para o chat seguinte que esteja a gerar, em fila ou não lido",
        },
        clearAllUnreads: {
          label: "Limpar todos os não lidos",
          description: "Marcar todos os chats como lidos",
        },
        goToRecentChat1: {
          label: "Ir para o chat recente 1",
          description: "Abrir o primeiro chat em Recentes",
        },
        goToRecentChat2: {
          label: "Ir para o chat recente 2",
          description: "Abrir o segundo chat em Recentes",
        },
        goToRecentChat3: {
          label: "Ir para o chat recente 3",
          description: "Abrir o terceiro chat em Recentes",
        },
        goToRecentChat4: {
          label: "Ir para o chat recente 4",
          description: "Abrir o quarto chat em Recentes",
        },
        goToRecentChat5: {
          label: "Ir para o chat recente 5",
          description: "Abrir o quinto chat em Recentes",
        },
        goToRecentChat6: {
          label: "Ir para o chat recente 6",
          description: "Abrir o sexto chat em Recentes",
        },
        switchToChat: {
          label: "Mudar para Chat",
          description: "Ir para o espaço de trabalho de chat",
        },
        switchToProjects: {
          label: "Mudar para Projetos",
          description: "Ir para o espaço de trabalho de projetos",
        },
        switchToHub: {
          label: "Mudar para Hub de modelos",
          description: "Ir para o hub de modelos",
        },
        switchToTrain: {
          label: "Mudar para Treinar",
          description: "Ir para o espaço de trabalho de treino",
        },
        switchToRecipes: {
          label: "Mudar para Receitas",
          description: "Ir para Data Recipes",
        },
        switchToImages: {
          label: "Mudar para Imagens",
          description: "Ir para o espaço de trabalho de imagens",
        },
        switchToVideo: {
          label: "Mudar para Vídeo",
          description: "Ir para o espaço de trabalho de vídeo",
        },
        switchToAudio: {
          label: "Mudar para Áudio",
          description: "Ir para o espaço de trabalho de áudio",
        },
        switchToExport: {
          label: "Mudar para Exportar",
          description: "Ir para o espaço de trabalho de exportação",
        },
        toggleSidebar: {
          label: "Alternar barra lateral",
          description: "Mostrar ou ocultar a barra lateral",
        },
        toggleApiMonitor: {
          label: "Alternar atividade da API",
          description: "Mostrar ou ocultar o monitor de atividade da API",
        },
        openModelPicker: {
          label: "Abrir seletor de modelos",
          description: "Escolher o modelo para este chat",
        },
        openProjectPicker: {
          label: "Abrir seletor de projetos",
          description: "Mudar para outro projeto a partir do cabeçalho do chat",
        },
        startDictation: {
          label: "Ditado",
          description: "Iniciar ou parar o ditado na caixa de mensagem",
        },
        attachFiles: {
          label: "Anexar fotografias e ficheiros",
          description: "Adicionar um anexo à caixa de mensagem",
        },
        openComposerMenu: {
          label: "Abrir menu +",
          description:
            "Abrir o menu + de ferramentas e anexos da caixa de mensagem",
        },
        sendMessage: {
          label: "Enviar mensagem",
          description: "Enviar o que está na caixa de mensagem",
        },
        queueMessage: {
          label: "Colocar mensagem em fila",
          description: "Enviar o rascunho para o fim da fila de prompts",
        },
        steerMessage: {
          label: "Redirecionar a resposta",
          description: "Parar a resposta atual e enviar o rascunho a seguir",
        },
        stopGeneration: {
          label: "Parar a geração",
          description: "Parar a resposta que está a ser escrita",
        },
        cycleReasoningEffort: {
          label: "Alternar esforço de raciocínio",
          description: "Percorrer os níveis de esforço de raciocínio",
        },
        increaseReasoningEffort: {
          label: "Aumentar esforço de raciocínio",
          description: "Subir um nível no esforço de raciocínio",
        },
        decreaseReasoningEffort: {
          label: "Diminuir esforço de raciocínio",
          description: "Descer um nível no esforço de raciocínio",
        },
        renameChat: {
          label: "Mudar o nome do chat",
          description: "Mudar o nome do chat atual",
        },
        forkChat: {
          label: "Ramificar chat",
          description:
            "Criar um novo chat ramificado a partir da última mensagem",
        },
        copyChatAsMarkdown: {
          label: "Copiar como Markdown",
          description:
            "Copiar todo o chat para a área de transferência como Markdown",
        },
        copySessionId: {
          label: "Copiar ID da sessão",
          description: "Copiar o ID da sessão da sandbox deste chat",
        },
      },
    },
    sandbox: {
      title: "Segurança",
      description:
        "Se as chamadas das ferramentas Python e Terminal são executadas dentro de uma sandbox do sistema operativo neste computador.",
      docs: "Documentação",
      docsLabel: "Abrir a documentação da sandbox",
      toolsSection: "Este computador",
      refresh: "Atualizar",
      python: "Python",
      terminal: "Terminal",
      osIsolation: "Sandbox do sistema operativo ({backend})",
      softwareSafeguards: "Salvaguardas por software",
      noReason:
        "Ainda não funciona aqui nenhuma sandbox do sistema operativo, pelo que o Unsloth verifica cada chamada.",
      runsInCmd:
        "É executado no cmd.exe, uma vez que o Git Bash não consegue iniciar na sandbox do Windows.",
      loadError: "Não foi possível carregar o estado da sandbox",
      saveError: "Não foi possível guardar a definição da sandbox",
      prepareError: "Não foi possível preparar este PC",
      unsupported:
        "Este servidor Unsloth ainda não comunica o estado da sandbox. Atualize o Unsloth para a gerir aqui.",
      windowsSection: "Sandbox do Windows (pré-visualização)",
      windowsDescription:
        "A sandbox MXC da Microsoft. Utiliza o contentor integrado no Windows ou uma alternativa nas versões do Windows que não o têm.",
      builtinLabel: "Contentor integrado",
      builtinDescription:
        "A sandbox do Windows é executada no contentor integrado do Windows, sem qualquer passo de administrador.",
      builtinInUse: "Em utilização",
      optInLabel: "Utilizar a sandbox alternativa",
      optInDescription:
        "Para versões do Windows sem o contentor integrado. O Python e o Terminal só poderão então alterar ficheiros na pasta do chat.",
      optInNotNeeded:
        "Não é necessária neste PC, uma vez que o contentor integrado funciona.",
      disclosure:
        "A alternativa requer um passo de administrador único, Preparar este PC, que tem de ser repetido após cada reinício. Enquanto uma ferramenta é executada, o Unsloth concede-lhe acesso temporário às pastas de que necessita. O MXC ainda é uma pré-visualização, não uma barreira de segurança.",
      lockedDacl: "Definido por UNSLOTH_MXC_ALLOW_DACL_FALLBACK",
      grantsLabel: "Início mais rápido das ferramentas",
      grantsDescription:
        "Mantém o acesso só de leitura às pastas de runtime do próprio Unsloth entre chamadas. Quando desativado, cada chamada demora mais alguns segundos a iniciar.",
      lockedGrants: "Definido por UNSLOTH_MXC_PERSISTENT_READ_GRANTS",
      memoryLabel: "Limite de memória",
      memoryDescription:
        "Memória máxima que uma chamada isolada de Python ou Terminal pode reservar. Aplica-se a partir da próxima chamada. A predefinição é {defaultSize} GB.",
      memoryLocked: "Definido por UNSLOTH_STUDIO_SANDBOX_AS_GB",
      memoryInvalid: "Introduza um número inteiro de {min} a {max}.",
      restored: "Acesso removido de {count} pastas.",
      hostPrepLabel: "Configuração de administrador",
      prepPrepared: "Concluída",
      prepNeeds: "Necessária",
      prepNeedsAgain: "Necessária novamente após o último reinício",
      prepUnknown: "Desconhecido",
      prepareButton: "Preparar este PC",
      preparing: "A aguardar o pedido de permissão de administrador…",
      prepSucceeded: "Este PC está preparado.",
      prepDeclined: "O pedido de permissão de administrador foi recusado.",
      prepFailed: "A preparação falhou.",
      setupLabel: "Sandbox do SO",
      setupCommandHint:
        "Execute isto no computador onde o Unsloth está em execução:",
      macosBuiltIn:
        "O macOS tem uma sandbox integrada, pelo que não há nada a instalar.",
      setupKeywords:
        "instalar configurar configuração install setup bubblewrap bwrap seatbelt mxc sudo pkexec apparmor uac",
      setupSucceeded: "A sandbox do SO está pronta.",
      runtimeMissing:
        "Ainda não instalado. A instalação não requer permissão de administrador.",
      runtimeLabel: "Runtime da sandbox do Windows",
      installRuntime: "Instalar runtime",
      installingRuntime: "A instalar…",
      installRuntimeFailed: "A instalação do runtime falhou.",
      installRuntimeError: "Não foi possível iniciar a instalação do runtime.",
      unsupportedArch:
        "A sandbox do Windows (MXC) só funciona no Windows x64 e ARM64, pelo que não está disponível neste PC.",
      builtinNeedsUpdate:
        "O contentor integrado do Windows requer a compilação {required} ou mais recente, e este PC tem a {current}. Instale a atualização mais recente do Windows para o utilizar sem um passo de administrador, ou ative a alternativa abaixo.",
      unsupportedBuild:
        "A sandbox do Windows (MXC) requer o Windows 11 24H2 (compilação 26100) ou mais recente, pelo que não está disponível neste PC.",
      permissionLabel: "Permissões das ferramentas",
      levelRowLabel: "Nível da sandbox",
      levelPickerTitle: "Como deve o código ser isolado?",
      levelOff: "Desativada",
      levelOffShort: "O acesso total desativa a sandbox",
      levelHighDetail:
        "O Python e o Terminal são executados dentro da sandbox do SO (bubblewrap, Seatbelt ou MXC), além do isolamento por software. Sem uma sandbox do SO funcional, as chamadas de risco pedem confirmação primeiro.",
      levelLowDetail:
        "O Python e o Terminal são executados apenas com isolamento por software, fora da sandbox do SO. As chamadas de risco pedem confirmação primeiro, exceto se tiver escolhido Acesso total.",
      levelLabel: "Sandbox",
      levelHigh: "Alto",
      levelLow: "Baixo",
      levelHighShort: "Adiciona isolamento do SO ao Python e ao Terminal",
      levelLowShort:
        "Apenas software. O código de risco pede confirmação primeiro",
      levelFullAccessNote:
        "O acesso total desativa a sandbox. Esta definição volta a aplicar-se quando escolher outro nível.",
      learnMore: "Saber mais",
      managedNote:
        "O proprietário deste servidor Unsloth configura a sandbox do SO.",
    },
    debugging: {
      description: "Leia os registos do Unsloth sem sair da aplicação.",
      logSection: "Ficheiro de registo",
      source: "Ficheiro de registo",
      sourceHint:
        "Os executores de modelos escrevem os seus próprios registos, pelo que uma falha ao carregar ou gerar é muitas vezes explicada aí e não no registo do servidor.",
      path: "Localização",
      pathCopy: "Copiar caminho",
      currentSession: "Atual",
      statusLive: "Em direto",
      statusPaused: "Em pausa",
      statusStale: "Desatualizado",
      filterPlaceholder: "Filtrar linhas",
      lineCount: "{count} linhas",
      filteredLineCount: "{shown} de {total} linhas",
      wrapLines: "Moldar linhas",
      jumpToLatest: "Mais recente",
      resizeLog: "Arraste para redimensionar o registo",
      noMatches: "Nenhuma linha corresponde ao filtro.",
      mode: "Modo",
      modeLive: "Em direto",
      modeInterval: "A cada 3 segundos",
      modeManual: "Manual",
      refreshNow: "Atualizar agora",
      privacyNote:
        "As credenciais estão ocultadas nesta vista. O ficheiro no disco não está ocultado.",
      copyVisible: "Copiar registo visível",
      empty: "Ainda não foi registado nada.",
      disabled:
        "O registo em ficheiro está desativado (UNSLOTH_STUDIO_NO_FILE_LOG=1).",
      missing: "Não foi encontrado nenhum ficheiro de registo.",
      unreadable: "Não foi possível ler o ficheiro de registo.",
      viewLogs: "Ver registos",
      timeout:
        "O pedido de registos excedeu o tempo limite. O servidor pode estar inacessível.",
      droppedNotice:
        "Algumas linhas foram ignoradas: o registo foi escrito mais depressa do que podia ser lido.",
      morePending:
        "Ainda estão a ser lidas mais linhas; chegam na próxima atualização.",
      staleSession:
        "O registo em ficheiro está desativado, pelo que esta é uma sessão anterior e não será atualizada.",
      downloadAllLogs: "Transferir todos os registos (.zip)",
      downloadingAllLogs: "A compactar registos...",
      exportMaskedNote:
        "As credenciais estão ocultadas nos ficheiros exportados. Os registos muito grandes mantêm apenas as linhas mais recentes, e alguns registos podem ser totalmente omitidos; consulte EXPORT_WARNINGS.txt no arquivo.",
      downloadedTo: "Guardado em {path}",
      downloadedToBrowser: "Transferência iniciada.",
      showInFolder: "Mostrar na pasta",
      openLogsFolder: "Abrir pasta de registos",
      openLogsFolderFailed: "Não foi possível abrir a pasta de registos.",
      exportFailed: "Não foi possível transferir os registos.",
      exportTooOld:
        "O backend do Unsloth em execução é demasiado antigo para exportar registos. Atualize esse backend e reinicie.",
      exportForbidden:
        "Transferir todos os registos requer uma sessão iniciada no Unsloth. Uma chave de API não é suficiente.",
      keywords:
        "depuração depurar registo registos erro erros falha bloqueio rastreio diagnóstico resolução de problemas debug log logs error crash traceback stack trace troubleshoot diagnostics failed",
    },
    voice: {
      title: "Voz",
      description: "Microfone, ditado, voz para texto e leitura em voz alta",
      dictation: {
        audioUploadTitle: "Ditar com uma gravação",
        audioUploadDescription:
          "O navegador requer uma ligação segura (HTTPS) para gravar diretamente no Unsloth. Nesta ligação HTTP, utilize o gravador do telemóvel ou escolha uma gravação guardada. O servidor Unsloth irá transcrever o áudio e adicionar o texto à mensagem.",
        audioUploadChooseFile: "Escolher gravação",
        audioUploadRecord: "Gravar áudio",
        audioUploadIphoneHint:
          "No iPhone, guarde uma gravação a partir do Memorandos de voz ou de outra aplicação de gravação e, em seguida, escolha-a aqui.",
        audioUploadServerModelNote:
          "Esta gravação utiliza o modelo local indicado acima no servidor Unsloth, mesmo que o ditado normal utilize outro motor.",
        audioUploadAutomatic: "Automático",
        audioUploadModelReady: "Pronto neste servidor Unsloth",
        audioUploadRetryTitle: "Não foi possível transcrever {file}",
        audioUploadRetry: "Tentar transcrever novamente",
        audioUploadChooseModel:
          "Escolha um modelo local de reconhecimento de voz nas definições de Voz.",
        audioUploadEmpty: "A gravação selecionada está vazia.",
        audioUploadTooLarge: "Escolha uma gravação com menos de {size}.",
        audioUploadVideoUnsupported:
          "Escolha uma gravação de áudio, não um vídeo.",
        audioUploadNoSpeech: "O modelo não detetou fala nessa gravação.",
        audioUploadFailed: "Não foi possível transcrever a gravação.",
        audioUploadCancel: "Cancelar transcrição",
        audioUploadTranscribing: "A transcrever áudio…",
        sectionTitle: "Ditado",
        engineLabel: "Motor de ditado",
        engineBrowser: "Navegador",
        engineBrowserDescription:
          'Transcreve o áudio com o serviço de voz do navegador. Selecione "Transcrição local" para utilizar um modelo STT.',
        engineModel: "Transcrição local",
        engineModelDescription:
          "Executa localmente um modelo de voz para texto (STT) e funciona offline. Transfira, carregue e o modelo é descarregado após inatividade.",
        engineCustom: "Endpoint personalizado",
        engineCustomDescription:
          "Envia o áudio gravado para um servidor STT compatível com OpenAI das suas Ligações.",
        connectionLabel: "Ligação",
        connectionDescription:
          "Adicione um servidor compatível com OpenAI e uma chave de API opcional em Ligações.",
        connectionPlaceholder: "Selecione uma ligação",
        connectionEmpty: "Nenhuma ligação disponível",
        customModelLabel: "Modelo",
        customModelDescription:
          "Nome do modelo enviado para /v1/audio/transcriptions.",
        sttModelLabel: "Modelo de reconhecimento de voz",
        sttModelDescription:
          "Escolha ou pesquise um modelo STT para executar localmente.",
        sttQuantLabel: "Quantização",
        sttQuantDescription:
          "Escolha ou pesquise um modelo STT para executar localmente e, em seguida, a respetiva quantização.",
        sttDeviceLabel: "Carregar em",
        sttDeviceAuto: "GPU quando disponível",
        sttDeviceCpu: "RAM da CPU",
        sttDeviceAutoDescription:
          "Utilizar a GPU quando existir e, caso contrário, a CPU.",
        sttDeviceCpuDescription:
          "Manter o modelo na RAM do sistema. A transcrição é mais lenta, mas não utiliza memória da GPU.",
        sttModelSearchPlaceholder: "Pesquisar qualquer modelo no HF",
        sttModelSearching: "A pesquisar no Hugging Face…",
        sttModelValidating: "A verificar a compatibilidade com Whisper…",
        sttModelNoResults: "Nenhum modelo Whisper encontrado",
        sttModelInvalid: "Este repositório não pode ser utilizado para ditado",
        sttModelFailed: "Não foi possível carregar o modelo STT",
        sttModelUnsupported: "A gravação não é suportada neste navegador",
        sttChecking: "A verificar…",
        sttOnDemand: "Transferido",
        sttLoadingModel: "A carregar modelo…",
        sttReady: "Carregado em {device}",
        sttLoaded: "Carregado",
        sttUnavailable:
          "Não instalado neste servidor. Execute `unsloth studio update` para ativar o ditado local.",
        sttRetry: "Tentar novamente",
        sttDownloadChecking: "A verificar o estado da transferência…",
        sttNotDownloaded: "Não transferido",
        sttDownloadStatusFailed:
          "Não foi possível verificar o estado da transferência",
        sttDownload: "Transferir",
        sttDownloadConfirmTitle: "Transferir {model}?",
        sttDownloadConfirmBody:
          "O ditado local funciona totalmente offline, mas primeiro necessita do modelo de voz {model}. Cerca de {size}, obtido uma única vez para a cache do Hugging Face.",
        sttDownloadConfirmBodyUnsized:
          "O ditado local funciona totalmente offline, mas primeiro necessita do modelo de voz {model}. É obtido uma única vez para a cache do Hugging Face.",
        sttOpenVoiceSettings: "Abrir definições de Voz",
        sttDownloadStarted: "A transferir {model}",
        sttDownloading: "A transferir… {progress}%",
        sttCancelDownload: "Cancelar",
        sttCancellingDownload: "A cancelar…",
        sttCancelDownloadFailed: "Não foi possível cancelar a transferência",
        sttDownloadComplete: "Modelo de reconhecimento de voz transferido",
        sttModelReady: "{model} está pronto para ditado",
        sttRecommended: "Recomendado",
        sttDownloadFailed:
          "Não foi possível transferir o modelo de reconhecimento de voz",
        sttLoad: "Carregar",
        sttUnload: "Descarregar",
        sttUnloading: "A descarregar…",
        microphoneLabel: "Microfone",
        microphoneFallbackName: "Microfone {index}",
        microphoneDescription: "Utilizado para ditado",
        microphoneFallbackHint:
          "Utilizado para ditado. Recorre à predefinição do sistema se o motor de voz do navegador não puder utilizar este dispositivo",
        microphoneGrantDescription:
          "Permita o acesso ao microfone para mostrar os nomes dos dispositivos",
        allowMicrophone: "Permitir microfone",
        micAccessBlocked:
          "O acesso ao microfone foi bloqueado. Permita o acesso ao microfone para esta página do Unsloth e tente novamente.",
        micAccessBlockedDesktop:
          "O acesso ao microfone foi bloqueado. Tente novamente e escolha Permitir, ou ative o microfone nas definições de privacidade do sistema.",
        micAccessUnsupported:
          "O acesso ao microfone não é suportado neste navegador ou contexto.",
        systemDefault: "Predefinição do sistema",
        savedMicDisconnected: "Microfone guardado (não ligado)",
        languageLabel: "Idioma do ditado",
        languageDescription: "Idioma a reconhecer",
        languageAuto: "Automático (idioma do navegador)",
        languageAutoDetect: "Automático (detetar idioma)",
      },
      dictionary: {
        sectionTitle: "Dicionário de ditado",
        sectionDescription:
          "Defina como o ditado escreve palavras ou expressões específicas",
        manageLabel: "Grafias personalizadas",
        manage: "Gerir",
        backToVoice: "Voltar a Voz",
        addEntry: "Adicionar entrada",
        newEntryAria: "Nova entrada do dicionário",
        entryPlaceholder: "João Silva",
        entryAria: "Entrada do dicionário {index}",
        removeEntryAria: "Remover entrada do dicionário {index}",
      },
      recents: {
        sectionTitle: "Histórico de ditados",
        sectionDescription:
          "Todos os ditados são guardados aqui para que possa recuperar o texto",
        manageLabel: "Histórico de ditados",
        manage: "Gerir",
        pageDescription:
          "Todos os ditados são guardados. Veja, copie ou elimine-os, ou abra o chat onde um ditado foi utilizado.",
        searchPlaceholder: "Pesquisar ditados",
        sortLabel: "Ordenar ditados",
        sortNewest: "Mais recentes",
        sortOldest: "Mais antigos",
        sortAlpha: "A a Z",
        noMatches: "Nenhum ditado corresponde à pesquisa",
        detailTitle: "Ditado guardado",
        backToVoice: "Voltar a Voz",
        backToRecents: "Voltar aos ditados recentes",
        view: "Ver ditado completo",
        empty: "Ainda não há ditados",
        dictationColumn: "Ditado",
        dateColumn: "Data de criação",
        copy: "Copiar ditado",
        copied: "Copiado para a área de transferência",
        copyFailed: "Não foi possível copiar para a área de transferência",
        delete: "Eliminar ditado",
        deleteTitle: "Eliminar ditado",
        deleteDescription:
          "Eliminar este ditado guardado? Esta ação não pode ser anulada.",
        deleteLinkedDescription:
          "Eliminar este ditado guardado? Também pode eliminar o chat onde foi utilizado. Esta ação não pode ser anulada.",
        deleteWithChat: "Eliminar chat e ditado",
        deleteWithChatFailed: "Não foi possível eliminar o chat",
        clear: "Limpar histórico",
        clearTitle: "Limpar histórico de ditados",
        clearDescription:
          "Eliminar todos os ditados guardados? Esta ação não pode ser anulada.",
        clearConfirm: "Limpar tudo",
        showMore: "Mostrar mais ({count})",
        openChat: "Abrir chat",
      },
      readAloud: {
        sectionTitle: "Ler em voz alta",
        buttonLabel: "Botão Ler em voz alta",
        buttonDescription: "Mostrar nas respostas do assistente",
        autoLabel: "Ler respostas automaticamente",
        autoDescription: "Ler cada resposta em voz alta assim que terminar",
        engineLabel: "Motor TTS",
        engineSystemDescription: "Vozes integradas do dispositivo",
        engineStudioDescription:
          "Fala com um modelo carregado em Áudio › Texto para voz",
        engineSystem: "Vozes do sistema",
        engineStudio: "Modelo Unsloth",
        engineCustom: "Endpoint personalizado",
        engineCustomDescription:
          "Um servidor TTS compatível com OpenAI das suas ligações (p. ex., Kokoro)",
        connectionLabel: "Ligação",
        connectionDescription:
          "Adicione um servidor compatível com OpenAI no separador Ligações",
        connectionPlaceholder: "Selecione uma ligação",
        customModelLabel: "Modelo",
        customVoiceDescription:
          "Nome da voz esperado pelo endpoint; a predefinição é alloy",
        modelLabel: "Modelo TTS",
        modelDescription:
          "Nenhum carregado. Carregue um em Áudio › Texto para voz.",
        modelLoaded: "{model} está carregado",
        keepBothLabel: "Manter o modelo de chat carregado",
        keepBothDescription:
          "A opção Manter vários modelos carregados está desativada, pelo que carregar um modelo de voz descarrega o modelo de chat.",
        keepBothAction: "Manter ambos carregados",
        openAudioAction: "Abrir Áudio",
        voiceLabel: "Voz",
        voiceDescription: "Melhores vozes neste dispositivo",
        studioVoiceDescription:
          "As vozes guardadas em Áudio requerem um modelo de clonagem",
        studioVoiceDefault: "Voz própria do modelo",
        studioVoiceSaved: "Voz guardada",
        speedLabel: "Velocidade",
        pitchLabel: "Tom",
        volumeLabel: "Volume",
        previewLabel: "Pré-visualizar voz",
        previewDescription: "Reproduzir uma amostra curta",
        previewFailed: "A pré-visualização TTS falhou",
        previewAction: "Pré-visualizar",
        preparingAction: "A gerar…",
        stopAction: "Parar",
        ttsLabel: "Texto para voz",
        notSupported: "Não suportado neste navegador",
      },
    },
    general: {
      title: "Geral",
      description: "Preferências globais do Unsloth.",
      account: "Conta",
      huggingFaceToken: "Token do Hugging Face",
      huggingFaceTokenDescription:
        "Utilizado para carregar modelos restritos e enviar artefactos.",
      hideToken: "Ocultar token",
      showToken: "Mostrar token",
      clearToken: "Limpar",
      checkingToken: "A verificar o token...",
      tokenValidated: "Token validado",
      password: "Palavra-passe",
      passwordDescription: "Alterar a palavra-passe desta conta Unsloth.",
      passwordDialog: {
        trigger: "Alterar palavra-passe",
        title: "Alterar palavra-passe",
        description:
          "Introduza a palavra-passe atual e escolha uma nova (pelo menos {minLength} carateres).",
        setTrigger: "Definir palavra-passe remota",
        setTitle: "Definir palavra-passe remota",
        setDescription:
          "Escolha a palavra-passe com que os navegadores remotos iniciam sessão como unsloth (pelo menos {minLength} carateres). A aplicação Unsloth Desktop continua a iniciar sessão automaticamente.",
        setSubmit: "Definir palavra-passe",
        setting: "A definir...",
        setDone: "Palavra-passe definida.",
        currentPassword: "Palavra-passe atual",
        newPassword: "Nova palavra-passe",
        confirmPassword: "Confirmar nova palavra-passe",
        currentTooShort:
          "A palavra-passe atual deve ter pelo menos {minLength} carateres.",
        newTooShort:
          "A nova palavra-passe deve ter pelo menos {minLength} carateres.",
        newHasSpaces: "A nova palavra-passe não pode conter espaços.",
        mismatch: "As palavras-passe não coincidem.",
        samePassword:
          "A nova palavra-passe tem de ser diferente da palavra-passe atual.",
        update: "Atualizar palavra-passe",
        updating: "A atualizar...",
        updated: "Palavra-passe atualizada.",
        updateFailed: "Não foi possível atualizar a palavra-passe.",
      },
      chatDefaults: "Predefinições do chat",
      autoTitleNewChats: "Atribuir título automático a novos chats",
      autoTitleNewChatsDescription:
        "Gerar um título a partir da primeira mensagem.",
      helperLlm: {
        sectionTitle: "LLM auxiliar",
        preloadOnStartup: "Pré-colocar o LLM auxiliar em cache no arranque",
        preloadOnStartupDescription:
          "Transferir o modelo auxiliar do Assistente de IA em segundo plano no arranque. Desativado por predefinição; o Assistente de IA pode obtê-lo quando for necessário.",
        disabledByEnv:
          "Desativado por UNSLOTH_HELPER_MODEL_DISABLE no ambiente do backend.",
        loadError: "Não foi possível carregar as definições do LLM auxiliar.",
        saveError: "Não foi possível guardar as definições do LLM auxiliar.",
      },
      developerMode: {
        sectionTitle: "Modo de programador",
        videoCpu: "Permitir geração de vídeo na CPU (experimental)",
        videoCpuDescription:
          "Permite que a página Vídeo seja executada num anfitrião sem GPU suportada. Desativado por predefinição; não suportado e muito lento.",
        notApplicable:
          "Não tem efeito neste computador: o vídeo já é executado na GPU ou está bloqueado por algo que a CPU não consegue contornar.",
        forcedOnByEnv:
          "Ativado por UNSLOTH_VIDEO_ALLOW_CPU no ambiente do backend.",
        forcedOffByEnv:
          "Desativado por UNSLOTH_VIDEO_ALLOW_CPU no ambiente do backend.",
        confirmTitle: "Ativar a geração de vídeo na CPU?",
        confirmWarning:
          "A geração de vídeo na CPU pode demorar horas, consumir grandes quantidades de RAM e bloquear o sistema. Continue por sua conta e risco.",
        ramBelowMinimum:
          "Este computador tem {host} GB de RAM, abaixo dos {min} GB de que o modelo de vídeo mais pequeno necessita na CPU. É provável que o carregamento falhe ou bloqueie o sistema.",
        ramOk:
          "Este computador tem {host} GB de RAM. O modelo de vídeo mais pequeno necessita de cerca de {min} GB na CPU; os modelos maiores necessitam de muito mais.",
        ramUnknown:
          "O modelo de vídeo mais pequeno necessita de cerca de {min} GB de RAM na CPU; os modelos maiores necessitam de muito mais.",
        performanceEstimate:
          "Conte com minutos por passo de remoção de ruído: um clipe de 5 fotogramas a 832x480 do Wan2.1 1.3B com 3 passos demorou cerca de 4 minutos em 16 threads de CPU, e um clipe completo com as definições predefinidas pode demorar horas.",
        confirmAction: "Ativar mesmo assim",
        loadError:
          "Não foi possível carregar as definições do Modo de programador.",
        saveError:
          "Não foi possível guardar as definições do Modo de programador.",
      },
      modelAutoSwitch: {
        sectionTitle: "Mudança automática de modelo (API OpenAI)",
        enable: "Mudar de modelo por pedido",
        enableDescription:
          "Carregar um modelo transferido indicado num pedido de API antes de responder. Desativado por predefinição.",
        autoDownload: "Transferir modelos em falta",
        autoDownloadDescription:
          "Obter um GGUF indicado num pedido de API que ainda não foi transferido. Qualquer pessoa com uma chave de API pode então utilizar disco e largura de banda.",
        idleUnload: "Descarregamento automático por inatividade",
        idleUnloadDescription:
          "Libertar VRAM após este número de segundos de inatividade. 0 mantém-no carregado; mínimo de 60.",
        idleSecondsAriaLabel:
          "Segundos para descarregamento automático por inatividade",
        mediaEnable: "Mudar de modelo de imagem e vídeo por pedido",
        mediaEnableDescription:
          "Carregar um modelo de imagem ou vídeo transferido indicado num pedido de API antes de gerar. Independente da mudança de modelo de chat. Desativado por predefinição.",
        mediaIdleUnload:
          "Descarregamento automático por inatividade para imagem e vídeo",
        mediaIdleUnloadDescription:
          "Libertar VRAM ao descarregar os modelos de imagem e vídeo após este número de segundos de inatividade. Independente da mudança de modelo de chat. 0 mantém-nos carregados; mínimo de 60.",
        mediaIdleSecondsAriaLabel:
          "Segundos para descarregamento automático por inatividade de imagem e vídeo",
        mediaIdlePaused:
          "Em pausa enquanto Manter modelo na memória da GPU estiver ativado.",
        idleNeedsEnable: "Ative primeiro Mudar de modelo por pedido.",
        idleActiveViaEnv: "Ativo através de UNSLOTH_MODEL_IDLE_TTL.",
        loadError:
          "Não foi possível carregar as definições de mudança automática de modelo.",
        saveError:
          "Não foi possível guardar as definições de mudança automática de modelo.",
        idleError:
          "Introduza 0 para manter o modelo carregado, ou pelo menos 60 segundos.",
        keepKv: "Manter o contexto do chat entre descarregamentos",
        keepKvDescription:
          "Guardar a cache KV antes de um descarregamento por inatividade, ou antes de um modelo de imagem ou vídeo ocupar a GPU, para que os chats retomados não tenham de reler o histórico. Até 10 GB em disco.",
        keepKvKeywords:
          "cache kv contexto troca retomar imagem vídeo kv cache context swap resume image video",
        apiOnly: "Descarregar apenas modelos carregados pela API",
        apiOnlyDescription:
          "O descarregamento por inatividade mantém em memória um modelo que tenha carregado a partir do Unsloth, e liberta apenas os que foram carregados por um pedido de API.",
      },
      modelAliases: {
        sectionTitle: "Aliases de modelos (API OpenAI)",
        description:
          "Nomes estáveis para clientes de API, como fast-chat ou code. Um pedido que indique um alias é servido pelo modelo para o qual aponta, pelo que as aplicações ligadas continuam a funcionar quando troca de modelo. Ative Mudar de modelo por pedido para que um alias possa carregar o respetivo modelo.",
        namePlaceholder: "fast-chat",
        nameAriaLabel: "Nome do alias",
        modelAriaLabel: "Modelo para o qual o alias aponta",
        modelPlaceholder: "unsloth/Qwen3-4B-GGUF:Q4_K_M",
        removeAriaLabel: "Remover alias",
        add: "Adicionar alias",
        loadError: "Não foi possível carregar os aliases de modelos.",
        saveError: "Não foi possível guardar os aliases de modelos.",
      },
      previewSharing: {
        sectionTitle: "Partilha de pré-visualizações",
        enableLabel: "Ligações públicas de pré-visualização",
        enableDescription:
          "Permitir que qualquer pessoa com uma ligação assinada converse com um modelo concluído, sem necessidade de iniciar sessão. Desative para colocar offline a superfície pública de pré-visualização; as ligações partilhadas deixam de funcionar.",
        loadError:
          "Não foi possível carregar as definições de partilha de pré-visualizações.",
        saveError:
          "Não foi possível guardar as definições de partilha de pré-visualizações.",
        revokeLabel: "Revogar todas as ligações de pré-visualização",
        revokeDescription:
          "Rodar o segredo de assinatura para que todas as ligações que partilhou deixem de funcionar. As ligações copiadas posteriormente continuam a funcionar.",
        revokeAction: "Revogar ligações",
        revoking: "A revogar...",
        revokeConfirmTitle: "Revogar todas as ligações de pré-visualização?",
        revokeConfirmDescription:
          "Todas as ligações de pré-visualização que partilhou deixarão de funcionar imediatamente. Esta ação não pode ser anulada.",
        revokeConfirmAction: "Revogar todas as ligações",
        revoked: "Todas as ligações de pré-visualização foram revogadas",
        revokeError: "Não foi possível revogar as ligações de pré-visualização",
      },
      managedProviderUrls: {
        sectionTitle: "Contas geridas",
        enableLabel: "Ligações locais e de rede",
        enableDescription:
          "Permitir que as contas geridas apontem as suas ligações para endereços locais ou de rede, como um servidor Ollama ou llama.cpp neste computador ou na rede local (LAN). Desativado por predefinição, porque permite que essas contas acedam a serviços em execução no seu computador e na sua rede.",
        lockedByEnvironment:
          "Definido por UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1 neste servidor, que recusa endereços privados para todas as contas.",
        loadError:
          "Não foi possível carregar as definições de ligações das contas geridas.",
        saveError:
          "Não foi possível guardar as definições de ligações das contas geridas.",
      },
      permissions: {
        sectionTitle: "Permissões",
        names: {
          ask: "Pedir aprovação",
          auto: "Aprovar por mim",
          off: "Executar automaticamente",
          full: "Acesso total",
        },
        details: {
          ask: "Pede confirmação antes de cada chamada de ferramenta, incluindo código de terminal e Python, pesquisas na Web, edições de ficheiros e ferramentas MCP. As ferramentas executadas por um fornecedor externo não são colocadas em pausa.",
          auto: "Executa autonomamente as chamadas de ferramentas de rotina e só pede confirmação quando uma ação parece arriscada, como ler credenciais, elevar privilégios ou executar comandos destrutivos.",
          off: "Executa todas as chamadas de ferramentas sem perguntar. O código de Python e de terminal continua a ser executado na sandbox, o que limita os ficheiros a que pode aceder no seu computador. Sem uma sandbox do SO funcional, as chamadas de risco pedem confirmação primeiro.",
          full: "Executa todas as chamadas de ferramentas sem perguntar e desativa a sandbox de código, pelo que o código de Python e de terminal pode ler e alterar qualquer ficheiro a que a sua conta tenha acesso.",
        },
      },
      notifications: {
        sectionTitle: "Notificações",
        showLoadedModels: "Cartão de modelos carregados",
        showLoadedModelsDescription:
          "Um cartão no canto com todos os modelos carregados, para descarregar um com um clique.",
        updateReminders: "Lembretes de atualização",
        updateRemindersDescription:
          "Com que frequência sugerir uma nova versão quando estiver disponível.",
        llamaHint: "Modelos de chat e GGUF",
        audioHint: "Voz, música e transcrição",
        frequency: {
          always: "Sempre que disponível",
          daily: "Uma vez por dia",
          weekly: "Uma vez por semana",
          biweekly: "De duas em duas semanas",
          monthly: "Uma vez por mês",
          off: "Desativado",
        },
      },
      startup: {
        sectionTitle: "Arranque",
        launchAtLogin: "Executar o Unsloth ao iniciar sessão",
        launchAtLoginDescription:
          "Iniciar o Unsloth em segundo plano quando inicia sessão. Fica na barra de menus ou na área de notificação até o abrir.",
        closeToTray: "Fechar para a área de notificação",
        closeToTrayDescription:
          "Manter o Unsloth e o respetivo servidor em execução em segundo plano quando fecha a janela principal.",
        closeToTraySaveError:
          "Não foi possível atualizar a definição de fechar para a área de notificação.",
        menuBarIcon: "Mostrar na barra de menus",
        menuBarIconDescription:
          "Mostrar o ícone do Unsloth na barra de menus do macOS. Quando oculto, o Unsloth fica na Dock, incluindo quando é iniciado ao iniciar sessão.",
        menuBarIconSaveError:
          "Não foi possível atualizar a definição do ícone da barra de menus.",
        loadError:
          "Não foi possível carregar a definição de iniciar ao iniciar sessão.",
        saveError:
          "Não foi possível atualizar a definição de iniciar ao iniciar sessão.",
      },
      hub: {
        source: "Origem dos modelos",
        sourceDescription:
          "Onde o hub de modelos pesquisa e transfere. Escolha ModelScope se o Hugging Face estiver bloqueado ou lento na sua rede.",
        sourceFallback:
          "Não foi possível iniciar o ModelScope, pelo que está a ser utilizado o Hugging Face. Consulte os registos do Unsloth.",
        autoSourceTitle: "Origem dos modelos alterada para ModelScope",
        autoSourceDescription:
          "O Hugging Face é frequentemente lento ou bloqueado na sua região, pelo que o Unsloth passa a transferir modelos do ModelScope.",
        autoSourceAction: "Abrir definições de Origem dos modelos",
        sectionTitle: "Hub de modelos",
        endpoint: "Endpoint do Hugging Face",
        endpointDescription:
          "De onde são transferidos os modelos e conjuntos de dados. Deixe vazio para huggingface.co, ou introduza um espelho como https://hf-mirror.com.",
        datasetsServer: "Utilizar para pré-visualizações de conjuntos de dados",
        datasetsServerDescription:
          "Enviar também os pedidos de pré-visualização de conjuntos de dados (HF_DATASETS_SERVER) para este endpoint. Ative apenas se o espelho os servir.",
        invalidEndpoint:
          "Introduza um URL http(s) sem credenciais nem consulta. O http simples só funciona para um endereço local.",
        saveFailed: "Não foi possível guardar as definições do hub.",
        loadFailed: "Não foi possível carregar as definições do hub.",
      },
      downloads: {
        sectionTitle: "Transferências",
        transport: "Transporte de transferência",
        transportDescription:
          "Como os ficheiros de modelos e conjuntos de dados são transferidos do Hugging Face. O HTTPS retoma onde parou; o Xet é muitas vezes mais rápido numa primeira transferência, mas recomeça o ficheiro se cancelar.",
        transportHint:
          "O HTTPS é TLS simples: todas as redes, proxies e VPN o permitem, uma transferência cancelada ou interrompida é retomada a partir dos bytes já em disco, e a utilização de memória mantém-se estável. O Xet obtém blocos sem duplicados, pelo que um repositório que partilhe dados com outro que já tenha pode chegar muito mais depressa, mas requer o hf_xet, utiliza mais RAM, e um cancelamento descarta o ficheiro em curso. O modo Automático avalia este computador: pondera a RAM e se o Xet tem bloqueado aqui, e recorre ao HTTPS.",
        https: "HTTPS",
        xet: "Xet",
        auto: "Automático",
        httpsHint:
          "TLS padrão. Retoma após um cancelamento, funciona em qualquer rede, utilização de memória estável.",
        transportDescriptionNoResume:
          "Como os ficheiros de modelos e conjuntos de dados são transferidos do Hugging Face. Nenhum dos transportes consegue retomar nesta instalação, pelo que uma transferência cancelada recomeça; o Xet é muitas vezes mais rápido numa primeira transferência.",
        httpsHintNoResume:
          "TLS padrão. Funciona em qualquer rede, utilização de memória estável. Esta instalação não consegue retomar uma transferência cancelada.",
        xetHint:
          "Transferência de blocos sem duplicados. Muitas vezes mais rápido numa transferência nova, recomeça o ficheiro se cancelar, requer mais memória.",
        autoHint:
          "Escolhe por computador e muda para HTTPS se o Xet bloquear ou falhar aqui.",
        autoCurrently:
          "O modo Automático está a utilizar {transport} neste computador.",
        xetMissing:
          "O Xet não está disponível porque o hf_xet não está instalado.",
      },
      uploads: {
        sectionTitle: "Carregamentos",
        maxUploadSize: "Limite de carregamento de conjuntos de dados de treino",
        maxUploadSizeDescription: "A predefinição é {defaultSize} MB.",
      },
      rag: {
        sectionTitle: "Documentos e RAG",
        embeddingModel: "Modelo de embeddings",
        embeddingModelShort:
          "Converte os seus documentos em vetores para que o chat os possa pesquisar.",
        embeddingModelDescription:
          "Modelo do Hugging Face ou caminho local utilizado para indexar e pesquisar os seus documentos. A predefinição é {defaultModel}.",
        searchPlaceholder: "Pesquisar qualquer modelo no HF",
        reindexWarning:
          "Afeta apenas os documentos indexados de novo. Recalcule os embeddings dos documentos existentes depois de alterar o modelo.",
        reembed: {
          label: "Recalcular embeddings dos documentos",
          staleOne:
            "1 documento foi indexado com um modelo de embeddings diferente, pelo que a pesquisa semântica o ignora.",
          staleMany:
            "{count} documentos foram indexados com um modelo de embeddings diferente, pelo que a pesquisa semântica os ignora.",
          staleArchived:
            "O histórico de chat anterior foi indexado com um modelo de embeddings diferente, pelo que a pesquisa semântica o ignora.",
          running: "A recalcular embeddings: {done} de {total}",
          action: "Recalcular embeddings",
          stop: "Parar",
          startFailed: "Não foi possível iniciar o recálculo de embeddings",
          stopped: "Recálculo de embeddings parado",
          finished: "Embeddings dos documentos recalculados",
          finishedWithFailures:
            "Não foi possível recalcular os embeddings de {count}",
        },
        emptyError:
          "Introduza um ID de modelo do Hugging Face ou um caminho local.",
        loadError:
          "Não foi possível carregar a definição do modelo de embeddings.",
        saveError: "Não foi possível guardar o modelo de embeddings.",
        saved: "Modelo de embeddings guardado.",
        saveAnyway: "Guardar mesmo assim",
        recommended: "Recomendado",
        onDevice: "No dispositivo",
        searching: "A pesquisar no Hugging Face…",
        checking: "A verificar…",
        noResults: "Nenhum modelo de embeddings encontrado",
        download: "Transferir",
        unload: "Ejetar",
        unloadFailed: "Não foi possível ejetar o modelo de embeddings",
        ejected: "Modelo de embeddings ejetado",
        ejectModel: "Ejetar modelo",
        onDeviceHint:
          "Transferido para este computador. Só é carregado na memória enquanto os documentos são indexados ou pesquisados, pelo que não utiliza RAM nem VRAM até lá.",
        loadedHint:
          "Neste momento na memória, a utilizar RAM ou VRAM. Passe o cursor sobre o seletor de modelos e clique no ícone vermelho para o ejetar.",
        notLoaded: "Não carregado",
        notLoadedHint:
          "Neste momento não está na memória. É carregado automaticamente da próxima vez que os documentos forem indexados ou pesquisados e, depois, pode ejetá-lo a partir do seletor.",
        downloadingStatus: "A transferir…",
        notDownloaded: "Não transferido",
        notDownloadedSized: "Não transferido · {size}",
        loaded: "Carregado",
        downloading: "A transferir {model}",
        downloadingDescription:
          "O progresso está no painel de transferências. A indexação passa a utilizá-lo quando estiver disponível.",
        downloadFailed: "Não foi possível iniciar a transferência",
        downloadConflict: "Retomar esta transferência a partir do Hub",
        downloadBusy: "Transferência já em curso",
        pin: "Afixar no menu Ficheiros",
        unpin: "Desafixar do menu Ficheiros",
        menuChip: "Modelo",
        moreModels: "Mais modelos",
        pinMoreModels: "Afixar mais modelos",
        defaultTag: "Predefinição",
        localModel: "Modelo local",
        switched: "Modelo de embeddings definido como {model}",
        switchedNeedsDownload: "{model} ainda não foi transferido",
        switchedNeedsDownloadDescription:
          "Transfira-o nas Definições antes de indexar novos ficheiros.",
        switchedUncheckedDescription:
          "Não foi possível verificar se está transferido. Verifique nas Definições antes de indexar novos ficheiros.",
        openSettings: "Abrir definições",
        switchFailed: "Não foi possível mudar o modelo de embeddings",
        back: "Voltar às origens",
        menuSubtitle: "Suporta a pesquisa de ficheiros",
        device: "Dispositivo de embeddings",
        deviceDescription:
          "Onde é executado o modelo de embeddings, para todos os modelos. A GPU indexa grandes conjuntos de documentos mais depressa, mas mantém alguma VRAM ocupada até o Unsloth ser reiniciado.",
        deviceAuto: "Automático",
        deviceCpu: "CPU",
        deviceGpu: "GPU",
        deviceLocked: "Definido por {name}.",
        deviceSaveError: "Não foi possível alterar o dispositivo de embeddings",
      },
      storage: {
        sectionTitle: "Armazenamento",
        modelsFolder: "Pasta de modelos",
        modelsFolderDescription:
          "Onde são armazenados os modelos transferidos.",
        openAction: "Abrir",
        copyAction: "Copiar caminho",
        copied: "Caminho copiado",
        openError: "Não foi possível abrir a pasta",
        copyError: "Não foi possível copiar o caminho",
      },
      fullInstall: {
        label: "Tipo de instalação",
        description:
          "Apenas chat (GGUF). A instalação completa adiciona treino, exportação, vídeo, modelos safetensors e funcionalidades adicionais de geração de imagens.",
        action: "Atualizar para a instalação completa",
        confirmTitle: "Atualizar para a instalação completa?",
        confirmDescription:
          "Esta instalação Apenas chat executa modelos GGUF. A instalação completa adiciona o PyTorch para a GPU deste computador, o que permite treino, exportação, vídeo, modelos safetensors e funcionalidades adicionais de geração de imagens. Transfere vários gigabytes (pelo menos 5 GB de espaço em disco) e reinicia o servidor. Os chats, modelos e definições são mantidos. Isto pode demorar vários minutos.",
        confirmAction: "Atualizar",
        keywords:
          "instalação completa atualizar tipo de instalação apenas chat treino treinar afinar exportar vídeo geração de imagens full install upgrade install type chat only gguf add training train fine-tune pytorch torch export video image generation safetensors",
      },
      repairInstall: {
        label: "Reparar instalação",
        description:
          "Volta a executar o instalador sobre o ambiente gerido. Utilize esta opção se a GPU não for detetada ou se a aplicação não iniciar.",
        action: "Reparar instalação",
        confirmTitle: "Reparar esta instalação?",
        confirmDescription:
          "Para o servidor e volta a executar o instalador, que reinstala o PyTorch para a GPU deste computador na versão suportada mais recente. Os chats e as definições são mantidos. Isto pode demorar vários minutos.",
        confirmDescriptionChatOnly:
          "Para o servidor e volta a executar o instalador. Esta é uma instalação Apenas chat, pelo que o instalador também a atualiza para a instalação completa com PyTorch, vários gigabytes no total. Os chats e as definições são mantidos. Isto pode demorar vários minutos.",
        confirmAction: "Reparar agora",
      },
      resetPreferences: {
        sectionTitle: "Zona de perigo",
        label: "Repor todas as preferências locais",
        description:
          "Limpa as preferências apenas locais. Os chats, o acesso à API e as definições guardadas na base de dados são mantidos.",
        action: "Repor preferências",
        confirmTitle: "Repor todas as preferências locais?",
        confirmDescription:
          "Limpa as preferências apenas locais e recarrega o Unsloth. Os chats, o acesso à API e as definições guardadas na base de dados são mantidos.",
        confirmAction: "Repor e recarregar",
      },
    },
    profile: {
      title: "Perfil",
      description: "Como o seu perfil aparece no Unsloth.",
      changePicture: "Alterar imagem de perfil",
      uploadPhoto: "Carregar fotografia",
      removePhoto: "Remover",
      pictureOptions: "Opções da imagem de perfil",
      displayName: "Nome a apresentar",
      nickname: "Como deve o Unsloth tratá-lo?",
      nicknamePlaceholder: "Alcunha",
      nicknameSaved: "Nome preferido guardado",
      avatarShape: "Forma do avatar",
      avatarShapeCircle: "Círculo",
      avatarShapeRounded: "Arredondado",
      greetingSloth: "Saudação",
      greetingSlothDescription: "Mostrar a saudação em novos chats.",
      chooseSloth: "Ou escolha uma preguiça",
      noPicture: "Sem imagem de perfil",
      noneLabel: "Nenhuma",
      nameSaved: "Nome do perfil guardado",
      namePersistErrorTitle: "Não foi possível guardar o nome do perfil",
      namePersistErrorDescription:
        "O nome foi atualizado para esta sessão, mas pode não ser mantido após recarregar.",
      photoUpdated: "Fotografia de perfil atualizada",
      photoPersistErrorTitle: "Não foi possível guardar a fotografia de perfil",
      photoPersistErrorDescription:
        "A fotografia foi atualizada para esta sessão, mas pode não ser mantida após recarregar.",
      photoUpdateErrorTitle:
        "Não foi possível atualizar a fotografia de perfil",
      imageUseError: "Não foi possível utilizar esta imagem.",
      stats: {
        title: "As suas estatísticas",
        subtitle:
          "Tudo o que se segue é contabilizado a partir do seu próprio histórico. Nada é recolhido nem enviado para a Unsloth.",
        retry: "Tentar novamente",
        exportCsv: "Exportar CSV",
        exportError: "Não foi possível exportar a utilização de tokens",
        privacyNote:
          "As estatísticas são calculadas a partir do histórico local de chat, utilização da API e treino guardado na sua instalação do Unsloth. Os prompts, respostas e chaves da API nunca são armazenados para estatísticas. Nada é enviado para a Unsloth nem para terceiros.",
        emptyChats:
          "Ainda não há utilização de chat nem da API. Inicie uma conversa ou faça um pedido autenticado à API local e as suas estatísticas aparecerão aqui.",
        lifetimeTokens: "Tokens desde sempre",
        peakTokens: "Dia de pico",
        longestChat: "Chat mais longo",
        currentStreak: "Sequência atual",
        longestStreak: "Sequência mais longa",
        activityTitle: "Atividade de tokens",
        activityDescription: {
          daily: "{total} desde {since}",
          weekly: "Semana mais ativa {total} · desde {since}",
          cumulative: "{total} no total desde {since}",
        },
        mode: {
          daily: "Diário",
          weekly: "Semanal",
          cumulative: "Cumulativo",
        },
        cellTooltip: "{date} · {tokens}, {messages}",
        emptyDayTooltip: "{date} · Sem atividade",
        apiRequests: "{count} através da API",
        weekTooltip: "Semana de {date} · {tokens}",
        cumulativeTooltip: "Até à semana de {date} · {tokens} no total",
        less: "Menos",
        more: "Mais",
        insightsTitle: "Informações de atividade",
        totalChats: "Total de chats",
        totalMessages: "Total de mensagens",
        tokensIn: "Tokens enviados",
        tokensOut: "Tokens gerados",
        totalTokens: "Total de tokens",
        studioChatTokens: "Tokens do Unsloth Chat",
        apiTokens: "Tokens da API",
        cachedTokens: "Tokens em cache",
        cachedValue: "{tokens} ({percent}% da entrada)",
        avgTokensPerChat: "Média de tokens por chat",
        timeInChat: "Tempo em chat",
        activeDays: "Dias ativos",
        toolCalls: "Chamadas de ferramentas",
        attachments: "Ficheiros anexados",
        avgSpeed: "Velocidade média",
        bestSpeed: "Resposta mais rápida",
        firstToken: "Tempo médio até ao primeiro token",
        tokensPerSecond: "{value} tok/s",
        topModelsTitle: "Modelos mais utilizados",
        topModelsDescription: "Ordenados por tokens trocados",
        modelSummary: "{tokens} · {messages}",
        noModels: "Ainda não foi registada utilização de modelos.",
        trainingTitle: "Treino",
        trainingDescription: "Execuções de afinação deste espaço de trabalho",
        trainingRuns: "Execuções",
        trainingCompleted: "Concluídas",
        trainingSteps: "Passos",
        trainingTokens: "Tokens treinados",
        trainingTime: "Tempo de treino",
        bestLoss: "Melhor perda",
        runSteps: "{steps}",
        runLoss: "perda {loss}",
      },
    },
    appearance: {
      title: "Aspeto",
      description: "O aspeto do Unsloth neste dispositivo.",
      gpuRendering: {
        sectionTitle: "Composição",
        label: "Composição por GPU",
        description:
          "Desenhar o Unsloth com a placa gráfica. O modo Automático utiliza-a onde se sabe que funciona. As alterações aplicam-se depois de reiniciar o Unsloth.",
        auto: "Automático",
        on: "Ativado",
        off: "Desativado",
        oldWebkitWarning:
          "WebKitGTK {version} em Wayland: a composição por GPU pode falhar no arranque em NVIDIA com escala fracionária (erro 280210 do WebKit). Se isso acontecer, o Unsloth volta automaticamente à composição por CPU. No WebKitGTK 2.50.4 não trouxe ganho de velocidade mensurável (8 fps em ambos os casos), pelo que Ativado não é recomendado aqui.",
        oldDriverWarning:
          "Controlador NVIDIA {version} em X11: o Unsloth mantém a composição por GPU desativada até o controlador ser 610 ou mais recente. Os controladores mais antigos perdem um descritor de ficheiro por fotograma, e a janela deixa de ser desenhada em poucos minutos.",
        fallbackNote:
          "A composição por GPU falhou no arranque neste sistema, pelo que o Unsloth está a utilizar a composição por CPU.",
        tryAgain: "Tentar GPU novamente",
        envOverride:
          "Definido pela variável de ambiente {name}. Esta definição não tem efeito até essa variável ser removida.",
        macNote:
          "O macOS compõe sempre com a GPU. O WebKit não tem uma opção suportada para a desativar.",
        restartPending: "Reinicie o Unsloth para aplicar.",
        restartNow: "Reiniciar agora",
        loadError:
          "Não foi possível carregar a definição de composição por GPU.",
        saveError:
          "Não foi possível guardar a definição de composição por GPU.",
        restartError: "Não foi possível reiniciar o Unsloth.",
      },
      theme: {
        title: "Estilo visual",
        label: "Modo",
        description: "Claro, escuro ou de acordo com o sistema.",
        system: "Sistema",
        light: "Claro",
        dark: "Escuro",
      },
      palette: {
        label: "Tema",
        description:
          "Cores utilizadas em todo o Unsloth, nos modos claro e escuro.",
        standard: "Padrão",
        classic: "Clássico",
        minimal: "Minimalista",
        moreThemes: "Mais temas",
        halloween: "Halloween",
      },
      trickOrTreat: {
        label: "Doçura ou travessura",
        description:
          "Temas de Halloween e saudações assustadoras, até 1 de novembro.",
      },
      custom: {
        chatWidth: {
          label: "Largura do chat",
          description: "Largura das mensagens e da caixa de composição.",
          standard: "Padrão",
          wide: "Larga",
          full: "Largura total",
        },
        sentAttachments: {
          label: "Anexos nas mensagens enviadas",
          description:
            "Padrão mostra cada ficheiro com o respetivo tipo; Compacto mostra mais em cada linha. Automático muda para Compacto a partir de seis ficheiros.",
          auto: "Automático",
          list: "Padrão",
          chips: "Compacto",
        },
        reset: "Repor",
        resetAll: "Repor personalização",
        preferencesTitle: "Preferências",
        colors: {
          accent: "Destaque",
          background: "Fundo",
          foreground: "Primeiro plano",
        },
        fontDefault: "Predefinição",
        fontBundledGroup: "Integrados",
        fontImportedGroup: "Importados",
        fontDeviceGroup: "Neste dispositivo",
        fontFolderGroup: "Da pasta",
        fontDeviceLoading: "A procurar tipos de letra do dispositivo…",
        fontSearch: "Pesquisar tipos de letra…",
        fontNoResults: "Nenhum tipo de letra encontrado.",
        colorPicker: {
          hue: "Matiz",
          hex: "Cor hexadecimal",
          eyedropper: "Escolher uma cor do ecrã",
        },
        uiFont: {
          label: "Tipo de letra da interface",
        },
        headingFont: {
          label: "Tipo de letra dos títulos",
        },
        chatFont: {
          label: "Tipo de letra do chat",
        },
        codeFont: {
          label: "Tipo de letra do código",
        },
        importFont: {
          upload: "Carregar",
          scanFolder: "Selecionar pasta",
          alreadyAvailable:
            "Este tipo de letra já está disponível, pelo que é utilizada a cópia existente.",
          folderNoFonts:
            "Não foram encontrados ficheiros de tipos de letra nessa pasta.",
          remove: "Remover",
          errorInvalidType:
            "Tipo de ficheiro não suportado. Utilize .woff2, .woff, .ttf ou .otf.",
          errorTooLarge:
            "O ficheiro do tipo de letra é demasiado grande (máx. 1,5 MB).",
          errorLimit: "Pode importar até 3 tipos de letra.",
          errorStorageFull:
            "Não há armazenamento local suficiente para este tipo de letra. Remova primeiro um tipo de letra importado.",
          errorFailed:
            "Não foi possível carregar este ficheiro de tipo de letra.",
        },
        interfaceScale: {
          label: "Escala da interface",
          description: "Redimensionar todos os elementos da interface.",
        },
        uiFontSize: {
          label: "Tamanho do tipo de letra da interface",
          description:
            "Ajustar o tamanho base utilizado na interface do Unsloth.",
        },
        codeFontSize: {
          label: "Tamanho do tipo de letra do código",
          description: "Ajustar o tamanho base utilizado no código.",
        },
        fontSmoothing: {
          label: "Suavização do tipo de letra",
          description: "Utilizar anti-aliasing suavizado nos tipos de letra.",
        },
        contrast: {
          label: "Contraste",
        },
        reduceMotion: {
          label: "Reduzir movimento",
          description: "Reduzir as animações ou seguir o sistema.",
          system: "Sistema",
          on: "Ativado",
          off: "Desativado",
        },
        pointerCursors: {
          label: "Utilizar cursores de ponteiro",
          description:
            "Mudar o cursor para um ponteiro ao passar sobre elementos interativos.",
        },
      },
      language: {
        title: "Idioma",
        label: "Idioma de apresentação",
        description: "O idioma utilizado pelo Unsloth.",
        autoDetect: "Detetar automaticamente",
        search: "Pesquisar idiomas",
        noResults: "Nenhum idioma correspondente",
        spellCheck: "Verificação ortográfica",
        spellCheckDescription:
          "Sublinhar as palavras com erros ortográficos enquanto escreve.",
      },
      layout: {
        title: "Esquema",
        compactSidebar: "Afixar a barra lateral por predefinição",
        compactSidebarDescription:
          "Manter a barra lateral expandida em vez de a reduzir a ícones.",
      },
      sidebarNav: {
        title: "Navegação da barra lateral",
        description:
          "Afixe e reordene os separadores da barra lateral. Os separadores não afixados vão para o menu Mais.",
        dragToReorder: "Arraste para reordenar",
        pinToSidebar: "Afixar {name} na barra lateral",
        moreHolds: "Mais ({count})",
      },
      sidebarMenu: {
        title: "Menu do perfil",
        description: "Escolha e reordene os atalhos sob o seu nome.",
        darkModeToggle: "Alternar modo escuro",
        dragToReorder: "Arraste para reordenar",
      },
    },
    resources: {
      title: "Sistema e motor",
      description:
        "Monitorize o hardware e o armazenamento deste servidor Unsloth.",
      liveUpdates: "Atualizações em direto",
      floatingWindow: "Janela flutuante",
      showFloatingWindow: "Mostrar janela flutuante",
      showFloatingWindowDescription:
        "Manter a utilização de CPU, RAM e GPU no ecrã sobre o Unsloth.",
      liveMonitor: {
        title: "Monitor em direto",
        apiTitle: "Monitor da API",
        apiDescription:
          "Veja o que outras aplicações enviam para a API deste servidor à medida que chega.",
        summary: "Pedidos, erros e utilização de tokens em direto",
        status: "{active} ativos · {recent} recentes · {model}",
        noModelLoaded: "nenhum modelo carregado",
        autoOpen: "O monitor flutuante abre automaticamente",
        autoOpenDescription:
          "Abre um pequeno painel quando chega tráfego da API.",
        cpu: "CPU",
        ram: "RAM",
        disk: "Disco",
        modelsDisk: "Disco dos modelos",
        vram: "VRAM",
        cpuCores: "{logical} núcleos lógicos / {physical} físicos",
        currentLoad: "Carga atual",
        free: "{value} livres",
        noGpu: "Nenhuma GPU visível",
        gpuUnusable: "GPU inutilizável",
        gpuUnusableDetail: "Detetada, mas o PyTorch não a consegue utilizar",
        detailedView: "Vista detalhada",
        detailedViewDescription:
          "Mostrar na janela flutuante a carga da CPU e a carga, temperatura e consumo de cada GPU.",
        cpuLoad: "Carga da CPU",
        gpuLoad: "Carga da GPU",
        temperature: "Temperatura",
        power: "Consumo",
      },
      gpu: {
        title: "Dispositivos GPU",
        memory: "Memória da GPU",
        sharedWithSystemRam: "Partilhada com a RAM do sistema",
        estimatedAvailable: "Disponível estimada: {value}",
        sharedEstimatedAvailable:
          "RAM do sistema partilhada: {value} disponíveis (estimativa)",
        ggufInference: "Memória dos modelos GGUF",
        unavailable: "indisponível",
        detecting: "A procurar GPUs...",
        unreadable: "Não foi possível ler o hardware deste servidor.",
        noGpu:
          "Nenhuma GPU visível detetada. Os recursos apenas de CPU são apresentados acima.",
        noUsableGpu:
          "Nenhuma GPU deste computador pode ser utilizada pelo PyTorch.",
        mismatchCpuBuild:
          "O PyTorch é uma compilação apenas para CPU ({version}), pelo que as GPUs abaixo não podem ser utilizadas. Repare a instalação para restaurar o suporte de GPU.",
        mismatchUnavailable:
          "O PyTorch ({version}) não consegue inicializar as GPUs abaixo, pelo que não podem ser utilizadas. Verifique o controlador da GPU ou repare a instalação.",
        driverIdleEvict:
          "O controlador AMD {version} tem um erro conhecido que pode bloquear o Windows quando uma GPU AMD está inativa, sobretudo com mais de uma GPU. Atualize para o Adrenalin 26.9.2 ou posterior.",
        driverIdleEvictDetails: "Detalhes",
        dismissNotice: "Dispensar",
        unusableDevice: "inutilizável",
        unknownDevice: "GPU desconhecida",
        deviceWithIndex: "GPU {index}",
        vramUtilization: "VRAM",
        used: "{value} utilizados",
        free: "{value} livres",
        total: "{value} no total",
        usedOf: "{used} utilizados de {total}",
        segmentUsed: "Utilizada pela GPU",
        segmentSystem: "Utilizada pelo sistema",
        segmentAvailable: "Disponível",
      },
      llamaBackend: {
        title: "Motor de inferência GGUF",
        label: "Backend de computação",
        description:
          "O backend que o llama.cpp utiliza para executar modelos GGUF.",
        runningOn: "O llama.cpp é atualmente executado em {backend}.",
        hint: "Instala a compilação do llama.cpp para este backend e mantém-na entre atualizações. Útil quando a escolha automática falha ou o controlador da GPU não a suporta. Só são listados os backends com uma compilação para este computador; o treino não é afetado.",
        autoWith: "Automático ({backend})",
        apply: "Aplicar",
        applying: "A instalar...",
        applyHint:
          "Transfere a nova compilação e reinicia o llama.cpp. Um modelo carregado é descarregado.",
        applyHintWithSize:
          "Transfere {size} e reinicia o llama.cpp. Um modelo carregado é descarregado.",
        switchedTo: "O llama.cpp é agora executado em {backend}.",
        switchFailed: "Não foi possível alterar o backend do llama.cpp.",
        switchInterrupted: "A mudança foi interrompida antes de terminar.",
        envLocked:
          "Definido como {backend} pela variável de ambiente UNSLOTH_LLAMA_CPP_BACKEND, que se sobrepõe a esta definição.",
        customPath: {
          label: "Pasta personalizada do llama.cpp",
          description: "Utilize a sua própria compilação do llama-server.",
          hint: "Escolha a pasta do llama.cpp que contém o llama-server, ou uma compilação onde este esteja em build/bin. O runtime personalizado é utilizado para chat GGUF, embeddings e modelos de voz suportados. As variáveis de ambiente continuam a ter prioridade.",
          automatic: "Automático (incluído)",
          bundled: "Utiliza o runtime do llama.cpp instalado pelo Unsloth.",
          active:
            "O seu llama-server personalizado será utilizado no próximo carregamento de modelo.",
          environmentManaged: "Gerido pela variável de ambiente {variable}.",
          missingBinary:
            "O llama-server já não está disponível nesta pasta. Escolha outra pasta ou utilize o runtime incluído.",
          reloadRequired:
            "Recarregue o modelo para utilizar o llama-server selecionado.",
          change: "Alterar",
          saving: "A guardar...",
          useBundled: "Utilizar o incluído",
          chooseTitle: "Escolher pasta do llama.cpp",
          chooseAction: "Utilizar esta pasta",
          saved: "Pasta do llama.cpp atualizada",
          saveError: "Não foi possível atualizar a pasta do llama.cpp",
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
            "Não foi encontrada nenhuma instalação gerida do llama.cpp, pelo que não há nenhum backend para mudar.",
          localLink:
            "O llama.cpp é um diretório local que ligou manualmente, pelo que o Unsloth não o irá substituir.",
          sourceBuild:
            "Este llama.cpp foi compilado a partir do código-fonte, pelo que o respetivo backend não pode ser mudado aqui.",
          customPath:
            "Está selecionada uma pasta personalizada do llama.cpp. A respetiva compilação determina o backend de computação.",
          unresolved:
            "Não foi possível verificar os backends disponíveis. Verifique a ligação e tente novamente.",
          updateChecksDisabled:
            "As verificações de atualizações estão desativadas (UNSLOTH_DISABLE_UPDATE_CHECK=1), pelo que os backends disponíveis não são consultados.",
        },
        llamaBackendKeywords:
          "llama.cpp backend gguf inferência cuda rocm hip vulkan metal cpu gpu acelerador pré-compilado mudar motor caminho personalizado pasta diretório executável binário inference accelerator prebuilt switch engine custom path folder directory llama-server executable binary",
      },
      modelMemory: {
        title: "Memória dos modelos",
        keepResident: "Manter modelo na memória da GPU",
        keepResidentDescription: "Permanecer na VRAM entre prompts.",
        keepResidentHint:
          "Não devolver os pesos à RAM do sistema enquanto o modelo estiver carregado. Desativa o descarregamento automático por inatividade e, quando os pesos ficam na RAM do anfitrião (memória unificada ou carregamento parcial na GPU), também passa --mlock para que o SO não os possa paginar para fora e voltar a carregá-los no próximo prompt.",
        noRamReserve: "Não reservar RAM do sistema para o modelo",
        noRamReserveDescription:
          "Reduzir a RAM do anfitrião retida para os pesos do modelo.",
        noRamReserveHint:
          "Ignora o carregamento mapeado em memória nas compilações do Windows suportadas quando o modelo é totalmente carregado na GPU, para que as respetivas páginas não fiquem residentes. Caso contrário, mantém o carregamento mapeado em memória. Os buffers de CPU necessários podem continuar a utilizar RAM. Remove --no-mmap e --mlock.",
        multiModel: "Manter vários modelos carregados",
        openSystemSettings: "Abrir em Sistema",
        multiModelOff: "Desativado",
        multiModelFit: "Quando couberem",
        multiModelAlways: "Sempre",
        multiModelDescription:
          "Carregar um modelo de chat, voz, imagem ou vídeo descarrega os restantes.",
        multiModelDescriptionFit:
          "Carregar um modelo mantém os restantes quando couberem. Quando não couberem, descarrega primeiro os que não estão em utilização e, depois, o ativo.",
        multiModelDescriptionAlways:
          "Carregar um modelo nunca descarrega outro. Um modelo que não caiba não é carregado. O descarregamento automático por inatividade e o treino continuam a libertar memória.",
        multiModelHint:
          "Abrange modelos de chat, voz, imagem e vídeo. Cada modelo de chat carregado responde aos pedidos que o indicam. Com várias GPUs, o Unsloth tenta colocar um novo modelo numa GPU que nenhum outro modelo utilize. “Quando couberem” deixa cerca de 1 GB livre. Um modelo carregado ao lado de outros recebe a memória que estes deixam livre, pelo que pode começar com um contexto mais curto. “Quando couberem” e “Sempre” nunca descarregam um modelo noutra GPU.",
        allowOversizedLoad:
          "Permitir carregamento de modelos demasiado grandes",
        allowOversizedLoadDescription:
          "Carregar modelos de imagem e vídeo mesmo quando a verificação de memória indica que não vão caber.",
        allowOversizedLoadHint:
          "Aplica-se quando a CPU e a GPU partilham um único conjunto de memória (Apple Silicon, GPUs integradas). O Unsloth recusa um carregamento quando a estimativa dos pesos do modelo é superior à memória restante depois de reservar espaço para o sistema operativo. Com esta opção ativada, esses carregamentos são tentados mesmo assim. A estimativa pode estar errada em qualquer sentido: um carregamento demasiado grande pode falhar, esgotar toda a memória, fazer com que o computador recorra intensivamente à memória virtual ou deixe de responder, ou parar o Unsloth. Desativado por predefinição.",
        mediaIdleUnloadDescription:
          "Libertar VRAM ao descarregar os modelos de imagem e vídeo após este número de segundos de inatividade. 0 mantém-nos carregados; mínimo de 60.",
        mlockVetoed:
          "O --mlock permanece desativado: fixar o modelo reservaria RAM para o modelo inteiro. O descarregamento automático por inatividade continua desativado.",
        mlockNotApplicable:
          "Totalmente na GPU: nada na RAM do sistema para bloquear. O descarregamento automático por inatividade permanece desativado.",
        memlockCapped:
          "Este sistema limita a memória bloqueada a {limit}. Um modelo maior não ficará totalmente fixado; aumente o limite com ulimit -l.",
        reloadRequired:
          "Recarregue o modelo para aplicar os novos parâmetros de memória.",
        loadError:
          "Não foi possível carregar as definições de memória dos modelos",
        saveError:
          "Não foi possível guardar as definições de memória dos modelos",
        modelMemoryKeywords:
          "memória residente fixar fixado bloquear bloqueado manter carregado descarregar inatividade paginação troca mlock memlock ulimit vram gpu memory ram resident residency pin pinned page lock locked keep loaded unload idle mmap no-mmap load-mode swap paging",
        multiModelKeywords:
          "vários modelos juntos em simultâneo trocar caber descarregar vram gpu imagem vídeo chat voz ler em voz alta sempre inatividade multiple several models together alongside swap fit unload evict image video speech tts voice read aloud always idle",
        mediaIdleUnloadKeywords:
          "imagem vídeo difusão inatividade descarregar tempo limite segundos libertar vram memória da gpu image video diffusion idle unload ttl timeout seconds free gpu memory",
      },
      storage: {
        title: "Armazenamento",
        systemDisk: "Disco do sistema",
        systemUsed: "{used} utilizados de {total} no disco do sistema",
        modelsDiskSummary:
          "Unidade da pasta de modelos: {used} utilizados de {total}, {free} livres",
        segmentModels: "Modelos",
        segmentCaches: "Caches",
        segmentOther: "Outros",
        segmentFree: "Livre",
        refreshAction: "Atualizar",
        modelsDisk: "Disco dos modelos",
        diskUsage: "{used} utilizados / {total}",
        diskFree: "{free} livres",
        modelsFolder: "Pasta de modelos",
        modelsFolderDescription:
          "Onde são armazenados os modelos transferidos.",
        modelsFolderHint:
          "Onde são armazenados os modelos transferidos. Altere-a para manter os modelos fora da unidade do sistema. Aplica-se apenas a novas transferências. Os modelos que já tem permanecem onde estão.",
        modelsFolderKeywords:
          "pasta de modelos diretório caminho localização transferir transferências cache armazenamento disco unidade mover realocar models folder directory path location download downloads storage disk drive move relocate hugging face",
        futureDownloads: "Apenas novas transferências",
        environmentManaged: "Gerido pela variável de ambiente {variable}.",
        locationFree: "{free} livres",
        openAction: "Abrir",
        copyAction: "Copiar caminho",
        changeAction: "Alterar",
        resetAction: "Utilizar predefinição",
        chooseTitle: "Escolher localização de transferência dos modelos",
        chooseAction: "Utilizar para futuras transferências",
        cacheSaved: "Localização de transferência dos modelos atualizada",
        cacheSaveError:
          "Não foi possível atualizar a localização de transferência dos modelos",
        cachePickerError: "Não foi possível abrir o seletor de pastas",
        copied: "Caminho copiado",
        openError: "Não foi possível abrir a pasta",
        copyError: "Não foi possível copiar o caminho",
        caches: {
          label: "Ficheiros de cache",
          description:
            "{size} em caches, dos quais {reclaimable} podem ser limpos agora.",
          hint: "Transferências de pacotes, kernels compilados e caches de transferência que o Unsloth reconstrói quando necessário. Os modelos transferidos, projetos, chats, definições e o seu token do Hugging Face nunca são limpos aqui.",
          keywords:
            "cache caches limpar libertar espaço disco temporário compilado purge prune clear clean free space disk uv pip npm bun triton inductor cuda numba matplotlib vllm compiled xet temporary",
          measuring: "A medir o tamanho das caches...",
          measureFailed: "Não foi possível medir as caches",
          empty: "Não foram encontrados ficheiros de cache.",
          detailsAction: "Detalhes",
          recheckAction: "Verificar novamente",
          hideDetailsAction: "Ocultar detalhes",
          showFolders: "Mostrar pastas",
          hideFolders: "Ocultar pastas",
          previousLocations:
            "Inclui localizações anteriores. O botão Limpar principal afeta a pasta de cache atual. Os botões de modelo limpam apenas a cópia selecionada.",
          clearModelDescription:
            "Esta ação remove apenas esta cópia em cache. Será transferida novamente quando for necessária.",
          clearAction: "Limpar caches",
          clearOneAction: "Limpar",
          clearingAction: "A limpar...",
          confirmTitle: "Limpar ficheiros em cache?",
          confirmDescription: "Esta ação liberta cerca de {size}.",
          confirmOneTitle: "Limpar {name}?",
          safety:
            "O Unsloth reconstrói uma cache da próxima vez que precisar dela. Os modelos transferidos, projetos, chats, definições e o seu token do Hugging Face não são afetados.",
          hubCost:
            "Esta é a cache de modelos. Se a limpar, esses modelos serão transferidos novamente da próxima vez que os utilizar.",
          datasetsCost:
            "Se a limpar, esses conjuntos de dados serão transferidos novamente da próxima vez que os utilizar.",
          blocked: "Não limpa: {reason}",
          cleared: "{size} limpos",
          partial: "Não foi possível remover alguns ficheiros de cache",
          clearFailed: "Não foi possível limpar as caches",
          names: {
            uv: "Cache de pacotes do uv",
            pip: "Cache de transferências do pip",
            npm: "Cache de pacotes do npm",
            bun: "Cache de pacotes do Bun",
            torchInductor: "Cache de compilação do Torch Inductor",
            torchExtensions: "Compilações de extensões do Torch",
            triton: "Cache de kernels do Triton",
            cuda: "Cache de kernels CUDA",
            numba: "Cache de compilação do Numba",
            matplotlib: "Cache de tipos de letra do Matplotlib",
            vllm: "Cache do vLLM",
            unslothCompiled: "Módulos compilados do Unsloth",
            hfXet: "Cache de transferências do Hugging Face",
            hfAssets: "Cache de recursos do Hugging Face",
            hfDatasets: "Cache de conjuntos de dados do Hugging Face",
            hfHub: "Cache de modelos do Hugging Face",
          },
        },
        lowDisk: {
          title: "O espaço em disco está a esgotar-se",
          criticalTitle: "O espaço em disco está criticamente baixo",
          description:
            "{free} livres de {total}. Limpar as caches pode libertar espaço.",
          action: "Rever caches",
        },
      },
      environment: {
        title: "Ambiente",
        backend: "Backend",
        python: "Python",
        torch: "Torch",
        transformers: "Transformers",
        mlxLm: "mlx-lm",
        mlxVlm: "mlx-vlm",
        uptime: "Tempo de atividade",
        processMemory: "Memória do processo",
        notInstalled: "Não instalado",
        unknown: "Desconhecido",
        vramWithShared: "{vram} de VRAM + {shared} partilhada",
      },
    },
    agents: {
      title: "Agentes",
      description:
        "Ligue agentes de programação como o Claude Code e o Codex a um modelo local com unsloth start.",
      agentKeywords:
        "agentes programação vs code vscode copilot pi mistral vibe deepseek harness hermes openclaw opencode claude codex",
      vscodeNote:
        "O VS Code é a exceção: este comando adiciona o modelo servido à lista de modelos do Copilot Chat do VS Code (chatLanguageModels.json).",
      intro:
        "liga o Claude Code, Codex, DeepSeek Harness, Hermes, OpenClaw, OpenCode e outros agentes a um modelo servido localmente pelo Unsloth, totalmente offline. Executa um servidor compatível com OpenAI e nunca mexe nos ficheiros de configuração do seu agente.",
      readDocs: "Ler a documentação",
      copy: "Copiar",
      copied: "Copiado",
      commandBuilder: "Construtor de comandos",
      commandShell: "Shell para todos os comandos",
      agent: "Agente de programação",
      model: "Modelo",
      searchModels: "Pesquisar modelos GGUF...",
      noModels: "Nenhum modelo GGUF correspondente.",
      showingModels:
        "A mostrar {shown} de {total} correspondências. Continue a escrever para restringir a lista.",
      quantization: "Quantização",
      loadingQuantizations: "A carregar quantizações...",
      noQuantizations: "Sem quantização separada",
      recommended: "Recomendada",
      downloaded: "Transferida",
      quantizationLoadError:
        "Não foi possível carregar todas as quantizações. O comando utilizará o valor de modelo disponível.",
      generatedCommand: "Comando gerado",
      docs: "Documentação",
      docsLabel: "Abrir a documentação do unsloth start",
      agentDocs: "Abrir a documentação de configuração do {agent}",
      copyGeneratedCommand: "Copiar comando gerado",
      automaticSettingsNote:
        "O Unsloth aplica automaticamente as definições recomendadas do modelo se não tiver definido nenhuma opção.",
      configurationNote:
        "Também pode ajustar qualquer configuração. Veja mais abaixo ou consulte a",
      configurationDocs: "documentação",
      configurationFlagsSuffix: "para conhecer as opções.",
      modelNote:
        "O Codex requer um modelo GGUF servido pelo llama-server. Os outros agentes também podem utilizar modelos baseados em transformers; remova --model para utilizar o modelo já carregado no Unsloth.",
      subagent: {
        title: "Utilizar um modelo local como subagente",
        description:
          "Manter o {agent} no modelo atual e delegar tarefas selecionadas a este modelo local do Unsloth.",
        setupCommand: "Comando de configuração",
        copySetupCommand: "Copiar comando de configuração do subagente",
        usagePrompt: "Depois, no {agent}, escreva:",
        copyUsagePrompt: "Copiar prompt de utilização do subagente",
        defaultPrompt: "Crie um agente local para implementar esta função.",
        opencodePrompt: "@unsloth encontre a causa desta falha de teste",
      },
      quickstart: {
        title: "Criar um comando",
        description:
          "Inicie um agente com o modelo atualmente carregado no Unsloth. Carregue primeiro um modelo e, depois, substitua claude por qualquer agente suportado abaixo.",
        noneDetected:
          "Não foram encontradas CLIs de agentes suportados no seu PATH.",
        installed: "Instalado",
      },
      supportedAgents: {
        title: "Agentes suportados",
        description: "Cada agente é iniciado com o seu próprio comando:",
        requiresGguf: "Requer um modelo GGUF",
      },
      models: {
        title: "Escolher um modelo",
        description:
          "Passe --model para escolher um modelo e uma quantização, e --context-length para definir a janela. Utilize um sufixo de quantização ou uma opção --gguf-variant explícita.",
        suffixLabel: "Com um sufixo de quantização",
        variantLabel: "Com uma flag de variante explícita",
      },
      options: {
        title: "Opções comuns",
        description:
          "As flags do Unsloth são analisadas primeiro; tudo o que não reconhecer é passado diretamente ao agente.",
        model:
          "Selecione um modelo. Sem --model, o unsloth start utiliza o modelo atualmente carregado no Unsloth e devolve um erro se nenhum estiver carregado.",
        contextLength:
          "Define o comprimento do contexto pedido (alias: --max-seq-length).",
        ggufVariant: "Escolha a variante de quantização GGUF.",
        loadIn4bit:
          "Ativa ou desativa o carregamento em 4 bits para modelos do Hugging Face.",
        tensorParallel:
          "Ativa ou desativa o paralelismo de tensores em várias GPUs.",
        serve: "Ativa ou desativa o servidor local automático.",
        launch: "Inicia o agente ou apenas mostra o comando e o ambiente.",
        persist:
          "Mantém o armazenamento do agente gerido pelo Unsloth entre execuções.",
        asSubagent:
          "Mantém o agente principal no modelo atual e regista o Unsloth como subagente local (Claude Code, Codex, OpenCode e Pi).",
        apiKey:
          "Indique a sua chave de API do Unsloth (ou defina UNSLOTH_API_KEY).",
        reasoning:
          "Utilizar raciocínio no chat: on, off ou auto. Auto segue o modelo de chat do modelo, o que normalmente significa ativado.",
        reasoningEffort:
          "Esforço de raciocínio passado ao modelo de chat do modelo, por exemplo medium. Os níveis dependem de cada modelo, por isso utilize um que esse modelo aceite. Se não for definido, mantém o do próprio modelo de chat.",
        yolo: "Ignora os pedidos de aprovação. Utilize apenas em ambientes fidedignos.",
      },
      remote: {
        title: "Ligar a um Unsloth remoto",
        description:
          "Aponte o unsloth start para uma instância do Unsloth em execução noutro local: defina estas variáveis antes de iniciar (ou passe --api-key diretamente):",
      },
      passthrough: {
        title: "Passar argumentos ao agente",
        description:
          "Os argumentos a seguir às flags do Unsloth são reencaminhados para o próprio agente, pelo que comandos nativos como resume continuam a funcionar:",
      },
      dryRun: {
        title: "Pré-visualizar sem iniciar",
        description:
          "Adicione --no-launch para mostrar o ambiente e o comando em vez de iniciar o agente. Se --model estiver definido, o modelo pode ainda assim ser resolvido e carregado.",
      },
    },
    chat: {
      groups: {
        conversations: {
          title: "Conversas",
        },
        files: {
          title: "Ficheiros e colagem",
        },
        display: {
          title: "Apresentação",
        },
        composer: {
          title: "Caixa de mensagem",
        },
        menu: {
          title: "Menu do chat",
        },
        advanced: {
          title: "Avançado",
        },
        contextTitle: "Contexto",
      },
      library: {
        label: "Gerir chats",
        description: "Pesquise, ordene e organize chats, projetos e secções.",
        action: "Abrir Biblioteca",
      },
      projectAttachmentsHint:
        "Substitua esta definição no menu de anexos de cada chat.",
      rememberParamsPerModelHint:
        "Quando desativado, utiliza as mesmas definições para todos os modelos.",
      autoCompactHint:
        "Chats locais GGUF e MLX: os turnos removidos são indexados para que o modelo os possa voltar a pesquisar, e uma reposição cita literalmente as instruções permanentes que couberem e mantém as mais antigas e as mais recentes em detrimento das intermédias. O arquivamento requer um chat guardado e o índice vetorial; sem eles, os turnos mais antigos são descartados. Utiliza o comprimento do contexto definido, não a VRAM disponível. Modelos de API: o fornecedor resume os turnos mais antigos quando possível; caso contrário, os turnos mais antigos são descartados à medida que o chat se aproxima da janela de contexto publicada do modelo.",
      pastedTextShortDescription:
        "Colagens com {count} carateres ou mais tornam-se anexos .txt.",
      pastedTextOffDescription:
        "O texto colado fica sempre na caixa de mensagem.",
      title: "Chat",
      description: "Personalize o comportamento do chat neste dispositivo.",
      modelSelection: {
        title: "Seleção de modelos",
        expandQuantizations: "Expandir quantizações",
        expandQuantizationsDescription:
          "Mostra as quantizações GGUF em No dispositivo sem abrir cada modelo.",
        showAllQuantizations: "Todas as quantizações",
        showAllQuantizationsDescription:
          "Inclui em No dispositivo as quantizações que não foram transferidas.",
        showMemoryBar: "Barra de utilização de VRAM",
        showMemoryBarDescription:
          "Mostra a VRAM estimada para os pesos do modelo, o contexto e a descodificação especulativa.",
        fitOnDeviceOnly: "Mostrar apenas modelos que cabem",
        fitOnDeviceOnlyDescription:
          "Oculta no seletor de modelos e no Hub os modelos demasiado grandes para a memória da GPU e a RAM deste dispositivo. Os modelos transferidos continuam visíveis.",
      },
      menu: {
        title: "Menu do chat",
        description:
          "Afixe itens no menu + do chat e arraste-os para os reordenar. Os restantes passam para “Mais”.",
        deepResearch: "Pesquisa aprofundada",
        browserUse: "Utilização do navegador",
        chatWithFiles: "Conversar com ficheiros (RAG)",
        mcp: "MCP",
        skills: "Competências",
        savedPrompts: "Prompts guardados",
        compareChat: "Comparar chat",
        exportChat: "Exportar chat",
        reset: "Repor predefinições",
        addFiles: "Adicionar fotografias e ficheiros",
        webSearch: "Pesquisa na Web",
        tools: "Ferramentas",
        more: "Mais",
      },
      pastedTextThreshold: "Condensar colagens longas",
      pastedTextThresholdDescription:
        "Prima {shortcut} para colar diretamente na caixa de mensagem.",
      pastedTextThresholdOff: "Desativado",
      autoScroll: "Deslocar durante a geração",
      autoScrollDescription:
        "A deslocação automática mantém o texto novo visível. A manual mantém a posição para que possa ler desde o início.",
      autoScrollAuto: "Deslocação automática",
      autoScrollManual: "Manual",
      autoScrollKeywords:
        "deslocar deslocação automática scroll autoscroll seguir fixar fundo saltar streaming gerar geração área visível bloquear manter auto manual",
      replyNotification: "Notificar quando uma resposta terminar",
      replyNotificationDescription:
        "Mostra uma notificação do sistema quando uma resposta termina enquanto o Unsloth está minimizado ou em segundo plano.",
      replyNotificationBlocked:
        "As notificações estão bloqueadas. Permita-as para o Unsloth nas definições do navegador ou do sistema e tente novamente.",
      replyNotificationKeywords:
        "notificar notificação alerta concluído terminado completo pronto resposta segundo plano minimizado ambiente de trabalho som",
      showResponseModel: "Modelo da resposta",
      showResponseModelDescription:
        "Mostra os detalhes do modelo nas respostas do assistente.",
      showReplyTimes: "Horas das respostas",
      showReplyTimesDescription:
        "Mostra, por baixo de cada resposta, quando esta terminou.",
      showReplyTimesKeywords:
        "carimbo de data/hora hora relógio data quando terminado chegou resposta mensagem timestamp",
      showGenerationDetails: "Detalhes da geração",
      showGenerationDetailsDescription:
        "Mostra em tempo real as velocidades do prompt e da resposta com as contagens de tokens, e onde foram carregadas as camadas do modelo.",
      showGenerationDetailsKeywords:
        "detalhado programador estatísticas velocidade tok/s prefill decode camadas offload vram experts",
      inlineEditResponse: "Editar resposta nas respostas",
      inlineEditResponseDescription:
        "Mantém Editar resposta em cada resposta, em vez de no menu Mais.",
      composerAtBottom: "Caixa de composição em baixo",
      composerAtBottomDescription:
        "Inicia novos chats com a caixa de composição no fundo da página em vez de no meio.",
      projectAttachments: "Partilhar ficheiros num projeto",
      projectAttachmentsDescription:
        "Torna os novos anexos de chat disponíveis para todos os chats do projeto.",
      rememberParamsPerModel: "Memorizar definições por modelo",
      rememberParamsPerModelDescription:
        "Restaura o último prompt, a temperatura e as outras definições utilizadas em cada modelo.",
      autoCompact: "Compactar automaticamente chats longos",
      autoCompactDescription:
        "Os turnos mais antigos passam para um arquivo pesquisável quando um chat preenche o contexto.",
      autoCompactKeywords:
        "compactação compactar compactação automática janela de contexto truncar contínuo checkpoint margem arquivo recuperação memória rag pesquisa",
      keepRunningWhenTabClosed: "Continuar em execução após fechar o separador",
      keepRunningWhenTabClosedDescription:
        "As respostas, a pesquisa na Web e a Pesquisa aprofundada continuam no servidor depois de o separador ser fechado. Desative para as parar.",
      keepRunningWhenTabClosedHint:
        "Reabra o chat em qualquer dispositivo para retomar uma execução que continuou. Quando desativado, uma execução para cerca de um minuto depois de fechar o último separador, pelo que recarregar a página não a interrompe. Aplica-se às execuções iniciadas depois de alterar esta opção. Chats temporários, respostas de modelos de API ou a anexos e turnos que utilizam o painel do navegador param sempre com o respetivo separador.",
      keepRunningWhenTabClosedKeywords:
        "segundo plano separador fechar fechado sair continuar em execução continuar parar cancelar pesquisa aprofundada pesquisa na web retomar religar dispositivo telemóvel",
      firstTokenLabel: "Limite de espera pelo primeiro token",
      firstTokenDescription:
        "Tempo que os modelos GGUF locais podem esperar sem progresso no processamento do prompt antes do primeiro token. Aplica-se a novas esperas em todas as contas. Ilimitado continua à espera até parar.",
      firstTokenDefault: "20 min (predefinição)",
      firstTokenCustom: "Personalizado",
      firstTokenUnlimited: "Ilimitado",
      firstTokenMinutes: "Espera pelo primeiro token em minutos",
      firstTokenRange: "Minutos (1–10 080)",
      firstTokenSave: "Guardar",
      firstTokenSaving: "A guardar…",
      firstTokenLoadError:
        "Não foi possível carregar o limite de espera pelo primeiro token.",
      firstTokenSaveError:
        "Não foi possível guardar o limite de espera pelo primeiro token.",
      firstTokenKeywords:
        "tempo limite timeout primeiro token prefill lento CPU durante a noite ilimitado espera limite",
      toolResultLimit: "Comprimento da saída das ferramentas",
      toolResultLimitDescription:
        "Carateres de um resultado de ferramenta Python, Terminal ou MCP que o modelo lê. A predefinição é {count}.",
      toolResultLimitHint:
        "A saída mais longa de Python e Terminal é guardada num ficheiro que o modelo pode ler por partes. Um limite mais alto aplica-se aos modelos executados no Unsloth, e as janelas de contexto pequenas continuam a receber menos. Os modelos de API mantêm a predefinição, a menos que a reduza.",
      toolResultLimitDefault: "{count} (predefinição)",
      toolResultLimitLocked: "Definido por UNSLOTH_TOOL_RESULT_MAX_CHARS",
      toolResultLimitLoadError:
        "Não foi possível carregar o comprimento da saída das ferramentas.",
      toolResultLimitSaveError:
        "Não foi possível guardar o comprimento da saída das ferramentas.",
      toolResultLimitKeywords:
        "ferramenta saída resultado truncar truncado truncagem limite máximo carateres comprimento terminal python mcp",
      visibility: {
        collapsed: "Fechado",
        auto: "Expandir durante a execução",
        expanded: "Sempre expandido",
      },
      visibilityKeywords:
        "fechar fechado expandir expandido aberto raciocínio pensamento chamadas de ferramentas atividade das ferramentas agrupar grupo streaming",
      thinking: {
        visibility: "Raciocínio",
        visibilityDescription:
          "Como o raciocínio é apresentado. Pode continuar a expandir ou fechar qualquer bloco manualmente.",
      },
      memory: {
        title: "Memória",
        description:
          "Notas que todos os chats leem, para todos os modelos, locais ou na nuvem. Os chats de um projeto também leem o AGENTS.md desse projeto.",
        enable: "Utilizar memória nos chats",
        enableDescription: "Envia estas notas com cada chat.",
        locked: "Desativado nesta instalação por UNSLOTH_STUDIO_AGENTS_MD.",
        editor: "Notas de memória",
        placeholder:
          "Por exemplo:\n- Escrevo em Python e utilizo o uv.\n- Respostas curtas.",
        hint: "Guardado em {path}. Também pode editar esse ficheiro em qualquer editor de texto.",
        size: "{used} / {max} KB",
        save: "Guardar",
        saved: "Memória guardada",
        loadError: "Não foi possível carregar a memória.",
        saveError: "Não foi possível guardar a memória.",
        keywords:
          "memória memorizar personalização perfil preferências notas factos instruções personalizadas sobre mim memory agents.md claude.md",
      },
      currentDate: {
        label: "Indicar a data de hoje ao modelo",
        description:
          "Inclui a data de hoje para que a pesquisa e a investigação encontrem fontes recentes.",
        loadError: "Não foi possível carregar as definições da data atual",
        saveError: "Não foi possível atualizar as definições da data atual",
      },
      tools: {
        visibility: "Chamadas de ferramentas",
        visibilityDescription:
          "Como a atividade das ferramentas é apresentada. Pode continuar a expandir ou fechar qualquer chamada manualmente.",
        foldIntoThinking: "Agrupar chamadas de ferramentas em Raciocínio",
        foldIntoThinkingDescription:
          "Mostra as chamadas de ferramentas de um turno dentro do respetivo bloco de Raciocínio em vez de em linhas próprias.",
        foldIntoThinkingBlocked:
          "Indisponível enquanto as chamadas de ferramentas estiverem definidas como Sempre expandido, o que as mantém em linhas próprias.",
      },
      webSearch: {
        title: "Pesquisa na Web",
        images: "Imagens da pesquisa na Web",
        imagesDescription: "Inclui imagens nos resultados da pesquisa.",
      },
      artifacts: {
        title: "Páginas HTML",
        collapseHtmlBlocks: "Fechar blocos HTML",
        collapseHtmlBlocksDescription:
          "Mostra documentos HTML completos no chat como um cartão que abre no navegador, em vez de como código.",
        allowNetworkAccess: "Permitir acesso à rede às páginas HTML",
        allowNetworkAccessDescription:
          "Permite que as páginas HTML do chat carreguem scripts, estilos, tipos de letra e multimédia externos. Desative para pré-visualizações offline.",
        blockedBanner: "Foi bloqueado {count} recurso externo de {hosts}.",
        blockedBannerPlural:
          "Foram bloqueados {count} recursos externos de {hosts}.",
        blockedBannerAction: "Permitir para esta página",
        blockedTitle: "O acesso à rede está desativado para as páginas HTML",
        blockedHint:
          "Ative “{setting}” em Definições → Navegador para permitir que as páginas HTML carreguem recursos externos, ou permita-o apenas para esta página.",
        blockedSettingsAction: "Abrir Definições",
        blockedDismiss: "Ignorar",
        errorTitle: "Esta página encontrou um erro",
        errorTitlePlural: "Esta página encontrou {count} erros",
        errorHint:
          "Corrigir com o modelo coloca o erro na caixa de mensagem. Nada é enviado até o enviar.",
        errorBannerAction: "Corrigir com o modelo",
        errorConsoleAction: "Abrir consola",
        errorConsoleHideAction: "Ocultar consola",
        errorLocation: "linha {line}, coluna {column}",
        errorLine: "linha {line}",
        consoleTitle: "Consola",
        reloadCanvas: "Executar novamente",
        consoleMessageCount: "{count} mensagem",
        consoleMessageCountPlural: "{count} mensagens",
        consoleClear: "Limpar consola",
        consoleClose: "Fechar consola",
        consoleEmpty: "Ainda não há saída na consola.",
        consoleCapped:
          "Apenas são mantidas as últimas {count} entradas; a saída anterior foi descartada.",
      },
      data: "Dados",
      exportHistory: "Exportar histórico de chats",
      exportHistoryDescription: "Transfere todos os chats e mensagens em JSON.",
      exportAction: "Exportar",
      exportingAction: "A exportar...",
      exportConversations: "Exportar Recentes e Projetos",
      exportConversationsDescription:
        "Transfira os Recentes, ou os Recentes e os chats dos projetos, como Training JSONL, CSV, ShareGPT JSONL ou Markdown, combinados ou por chat. O Message JSONL está disponível por chat.",
      exportConversationsAction: "Exportar",
      exportScopeRecents: "Recentes",
      exportScopeAll: "Recentes + Projetos",
      exportCombinedSuffix: "(combinados)",
      exportPerChatSuffix: "(por chat)",
      importChats: "Importar chats",
      importChatsDescription:
        "Importe ficheiros JSON, JSONL, NDJSON, CSV ou Markdown para os Recentes.",
      importChatsAction: "Importar",
      importNoConversations: "Não foram encontradas conversas no ficheiro.",
      importedOneChat: "Foi importada 1 conversa para os Recentes.",
      importedChatCount: "Foram importadas {count} conversas para os Recentes.",
      importingChats: "A importar chats: {count} até agora ({percent}%)...",
      importedChatCountPartial:
        "Foram importadas {count} conversas para os Recentes; não foi possível guardar {failed}.",
      importFailed: "A importação falhou.",
      importFromSource: "Importar de {source}",
      importFromSourceDescription:
        "Copie as suas conversas do {source} para o Unsloth, agrupadas pelo projeto a que pertencem.",
      importingAction: "A importar...",
      importedSourceOneChat: "Foi importada 1 conversa de {source}.",
      importedSourceChatCount:
        "Foram importadas {count} conversas de {source}.",
      sourceUpdated: "Foram adicionadas {count} mensagens novas de {source}.",
      sourceUpToDate: "As conversas de {source} já estão atualizadas.",
      importedSourcePartial:
        "Não foi possível importar algumas conversas de {source}.",
      clearHistory: "Limpar histórico de chats",
      clearHistoryDescription:
        "Elimina o histórico de chats deste dispositivo.",
      clearAction: "Limpar",
      clearAllChats: "Limpar todos os chats",
      clearAllChatsDescription:
        "Elimina permanentemente todos os chats deste dispositivo.",
      noChatsToClear: "Não há chats para limpar.",
      clearOneChatDescription:
        "Elimina permanentemente o único chat deste dispositivo.",
      clearChatCountDescription:
        "Elimina permanentemente todos os {count} chats deste dispositivo.",
      clearChatsAction: "Limpar chats",
      clearOneChatTitle: "Limpar 1 chat?",
      clearChatsTitle: "Limpar {count} chats?",
      clearChatsConfirmDescription:
        "Elimina permanentemente todos os chats deste dispositivo. Esta ação não pode ser anulada.",
      clearingAction: "A limpar...",
      clearOneChatAction: "Limpar 1 chat",
      clearChatCountAction: "Limpar {count} chats",
      clearedAllChats: "Todos os chats foram limpos",
      clearedOneChat: "1 chat limpo",
      clearedChatCount: "{count} chats limpos",
      someChatsCouldNotBeCleared: "Não foi possível limpar alguns chats",
      chatsClearedRemainOne:
        "{clearedCount} chats limpos; resta 1 chat. Tente novamente.",
      chatsClearedRemain:
        "{clearedCount} chats limpos; restam {remainingCount} chats. Tente novamente.",
      oneChatClearedRemain:
        "1 chat limpo; restam {remainingCount} chats. Tente novamente.",
      oneChatClearedRemainOne: "1 chat limpo; resta 1 chat. Tente novamente.",
      storageClearFailedOne:
        "Falha ao limpar o armazenamento; pode restar 1 chat. Tente novamente.",
      storageClearFailed:
        "Falha ao limpar o armazenamento; podem restar {count} chats. Tente novamente.",
      failedToClearChats: "Não foi possível limpar os chats",
    },
    data: {
      library: {
        defaultSort: "Predefinição",
        updated: "Atualizado",
        created: "Criado",
        oldest: "Mais antigos primeiro",
        alphabetical: "Alfabética",
        allChats: "Todos os chats",
        singleChats: "Chats individuais",
        compareChats: "Chats de comparação",
        allProjects: "Todos os projetos",
        noProject: "Sem projeto",
        unavailableProject: "Projeto indisponível",
        filterSort: "Filtrar e ordenar",
        sort: "Ordenar",
        type: "Tipo",
        sortBy: "Ordenar por",
        filterProject: "Filtrar por projeto",
        searchProjects: "Pesquisar projetos",
        noProjects: "Não foram encontrados projetos.",
        untitled: "Sem título",
        oneChat: "1 chat",
        chatCount: "{count} chats",
        searchChats: "Pesquisar chats ou projetos",
        searchArchivedChats: "Pesquisar chats ou projetos arquivados",
        searchImages: "Pesquisar imagens arquivadas",
        searchVideos: "Pesquisar vídeos arquivados",
        searchAudio: "Pesquisar áudio arquivado",
        unarchive: "Desarquivar",
        unarchiveAll: "Desarquivar tudo",
        unarchiveAllInProject: "Desarquivar tudo no projeto",
        deleteAllInProject: "Eliminar tudo no projeto",
        groupActions: "Ações para {group}",
        unarchiveResults: "Desarquivar resultados",
        deleteResults: "Eliminar resultados",
        noChats: "Nenhum chat corresponde à pesquisa.",
        noArchivedChats: "Não há chats arquivados.",
        noArchivedMatches: "Nenhum chat arquivado corresponde à pesquisa.",
        noMedia: "Não há itens arquivados.",
        noMediaMatches: "Nenhum item arquivado corresponde à pesquisa.",
        itemCount: "Itens: {count}",
        incompleteSearch:
          "A pesquisa está incompleta. Tente carregar novamente os itens restantes.",
        searchingRemaining:
          "A pesquisar os itens restantes ({count} carregados)...",
        noLoadedMatches:
          "Não há correspondências nos itens carregados até agora.",
        deleteItem: "Eliminar: {title}",
        unarchiveItem: "Desarquivar: {title}",
        deleteItemsTitle: "Eliminar itens arquivados ({count})",
        unarchiveItemsTitle: "Desarquivar itens ({count})",
        deleteFilesWarning:
          "Estes ficheiros serão eliminados permanentemente. Esta ação não pode ser anulada.",
        restoreWarning: "Estes itens voltarão para a galeria.",
        working: "A processar...",
        loadFailed: "Não foi possível carregar os itens arquivados",
        loadMoreFailed: "Não foi possível carregar mais itens arquivados",
        restoreFailed: "Não foi possível desarquivar os itens selecionados",
        deleteFailed: "Não foi possível eliminar os itens selecionados",
        projectsFailed: "Não foi possível carregar os projetos arquivados",
        pageStalled: "A página do arquivo não avançou. Tente novamente.",
        pageChanged:
          "O arquivo foi alterado durante o carregamento. Tente novamente.",
        selectAll: "Selecionar todos os chats visíveis",
        selectItem: "Selecionar “{title}”",
        selectedChats: "Chats selecionados: {count}",
        move: "Mover",
        pin: "Afixar",
        unpin: "Desafixar",
        archive: "Arquivar",
        deletedChats: "Chats eliminados: {count}",
        restoredChats: "Chats desarquivados: {count}",
        pinnedChats: "Chats afixados: {count}",
        unpinnedChats: "Chats desafixados: {count}",
        movedChatsToProject: "Chats movidos para {project}: {count}",
        movedChatsToRecents: "Chats movidos para Recentes: {count}",
        moveFailed: "Não foi possível mover os chats",
        deleteChatsTitle: "Eliminar chats ({count})",
        deleteChatsWarning:
          "Eliminar os chats selecionados ({count})? Esta ação não pode ser anulada.",
        deleteArchivedTitle: "Eliminar {count} chats arquivados",
        deleteArchivedWarning:
          "Eliminar os chats arquivados selecionados ({count})? Esta ação não pode ser anulada.",
        deleting: "A eliminar...",
      },
      title: "Dados",
      backToData: "Voltar a Dados",
      exportFailed: "Não foi possível exportar os chats",
      description:
        "Gira o histórico de chats e os ficheiros carregados armazenados neste dispositivo.",
      manageFiles: {
        label: "Gerir ficheiros",
        description:
          "Pesquise, ordene e organize imagens, vídeos, áudio e ficheiros.",
        action: "Abrir Biblioteca",
      },
      archivedChats: "Chats arquivados",
      archivedChatsDescription: "Veja e gira os chats que arquivou.",
      archivedImages: "Imagens arquivadas",
      archivedImagesDescription: "Veja e gira as imagens que arquivou.",
      archivedVideos: "Vídeos arquivados",
      archivedVideosDescription: "Veja e gira os vídeos que arquivou.",
      archivedAudio: "Áudio arquivado",
      archivedAudioDescription: "Veja e gira os clipes de áudio que arquivou.",
      manageAction: "Gerir",
      exportArchivedChats: "Exportar",
      exportingArchivedChats: "A exportar...",
      exportedOneArchivedChat: "Foi exportado 1 chat arquivado",
      exportedArchivedChatCount: "Foram exportados {count} chats arquivados",
      noArchivedChatsToExport: "Não há chats arquivados para exportar.",
      failedToExportArchivedChats:
        "Não foi possível exportar os chats arquivados",
      archiveAllChats: "Arquivar todos os chats",
      archiveAllChatsDescription:
        "Move todos os chats de Recentes e Projetos para o arquivo.",
      noChatsToArchive: "Não há chats para arquivar.",
      archiveAllAction: "Arquivar tudo",
      archivingAction: "A arquivar...",
      archiveAllChatsTitle: "Arquivar todos os chats?",
      archiveAllChatsConfirmDescription:
        "Move todos os chats deste dispositivo para o arquivo. Os chats arquivados continuam disponíveis e podem ser desarquivados a qualquer momento.",
      archivedAllChats: "Todos os chats foram arquivados",
      archivedOneChat: "1 chat arquivado",
      archivedChatCount: "{count} chats arquivados",
      failedToArchiveChats: "Não foi possível arquivar os chats",
      confirmBeforeDeleting: "Confirmar antes de eliminar",
      confirmBeforeDeletingDescription:
        "Pede confirmação antes de eliminar um chat. Desative para eliminar imediatamente.",
      alwaysDeleteFiles: "Eliminar sempre os ficheiros",
      alwaysDeleteFilesDescription:
        "Eliminar um chat também remove do disco a respetiva pasta da sandbox. Os ficheiros que escreveu dentro de um projeto permanecem no espaço de trabalho desse projeto.",
      chatsSection: "Chats e arquivos",
      transferSection: "Importação e exportação",
      deletionSection: "Eliminação",
      archives: "Arquivos",
      archiveChatsLabel: "Chats",
      archiveVideosLabel: "Vídeos",
      exportFormatsSummary:
        "Exporte os Recentes e os chats dos projetos no formato que preferir.",
      trainingSummary:
        "Transforme os seus chats num conjunto de dados para afinação.",
      confirmDeletionSummary: "Perguntar antes de eliminar chats individuais.",
      sandboxFiles: "Ficheiros da sandbox do chat",
      sandboxFilesDescription: "Ação predefinida ao eliminar um chat.",
      keepSandboxFiles: "Manter os ficheiros da sandbox",
      deleteSandboxFiles: "Eliminar os ficheiros da sandbox",
      projectFilesKept:
        "As sandboxes são pastas criadas para chats individuais. Os ficheiros guardados nos espaços de trabalho dos projetos são mantidos.",
      deleteAllAction: "Eliminar tudo",
      deleteChatsOnly: "Eliminar apenas os chats…",
      deleteChatsAndSandboxes: "Eliminar chats e sandboxes…",
      deleteSandboxFilesDescription:
        "Elimina também a pasta da sandbox de cada chat e os ficheiros que contém. Os ficheiros nos espaços de trabalho dos projetos são mantidos.",
      filesSection: "Ficheiros",
      uploadedFiles: "Ficheiros carregados",
      uploadedFilesDescription:
        "Veja e gira os ficheiros carregados para chats, projetos e bases de conhecimento.",
      fineTuneExport: "Utilizar chats como dados de treino",
      fineTuneExportDescription:
        "Crie um conjunto de dados JSONL para afinação a partir dos seus chats. Carregue-o em Treinar, refine-o em Receitas ou exporte-o.",
      fineTuneExportAction: "Exportar JSONL",
      fineTuneRunAction: "Executar",
      fineTuneExportingAction: "A exportar...",
      fineTuneOpenRecipesAction: "Abrir em Receitas",
      fineTuneOpeningRecipesAction: "A abrir...",
      fineTuneTrainAction: "Carregar no separador Treinar",
      fineTuneTrainingAction: "A carregar...",
      fineTuneExportFailed: "Não foi possível exportar os dados de treino",
      fineTuneRecipeFailed: "Não foi possível abrir os chats em Receitas",
      fineTuneTrainFailed:
        "Não foi possível carregar o conjunto de dados no separador Treinar",
    },
    connections: {
      title: "Ligações",
      description: "Gira fornecedores e ligações externas.",
    },
    remoteLan: {
      title: "Remoto e LAN",
      description:
        "Aceda a este Unsloth a partir dos seus outros dispositivos, através da rede local ou de um URL público temporário.",
    },
    apiKeys: {
      title: "API",
      description:
        "Aceda ao Unsloth através da API compatível com OpenAI e Anthropic.",
      readDocs: "Ler a documentação da API",
      noAccess: "Ainda não há acesso à API.",
      accessTokens: "Tokens de acesso",
      quickstartTitle: "Início rápido",
      columnName: "Nome",
      columnToken: "Token",
      columnCreated: "Criado",
      columnLastUsed: "Última utilização",
      columnExpires: "Expira",
      loadError: "Não foi possível carregar o acesso à API.",
      createError: "Não foi possível criar o token de acesso.",
      revokeError: "Não foi possível revogar o token de acesso.",
      never: "Nunca",
      tokenNamePlaceholder: "Nome do token",
      newAccessTokenName: "Nome do novo token de acesso",
      createToken: "Criar token",
      neverExpiresHint: "Nunca expira",
      expiresInDaysHint: "Expira dentro de {n} dias",
      createTokenHint: "Criar um token com o nome “{name}”",
      defaultTokenName: "Novo token {n}",
      creating: "A criar...",
      newTokenCreated: "Novo token de acesso criado",
      accessTokenCopied: "Token de acesso copiado",
      copyAccessToken: "Copiar token de acesso",
      copyNow: "Copie-o agora: não voltará a ser mostrado.",
      usageExamples: "API de chat",
      usageNoModel:
        "Carregue ou transfira um modelo para ver exemplos executáveis. Este servidor ainda não tem nenhum modelo para indicar.",
      usageTools: "Ferramentas",
      exampleLanguage: "Linguagem",
      exampleRequest: "Pedido",
      examplePlatform: "Plataforma",
      requestPlain: "Básico",
      requestTools: "Com ferramentas",
      requestAdvanced: "Amostragem avançada",
      exampleKind: "Exemplo",
      osUnix: "Linux / macOS / WSL",
      osWindows: "Windows",
      secureHttps: "HTTPS seguro",
      secureHttpsHint:
        "A porta 0.0.0.0 continua acessível globalmente. Para segurança total, inicie o Unsloth com --secure para expor apenas esta ligação HTTPS.",
      copyTunnelUrl: "Copiar URL do túnel",
      copySnippet: "Copiar excerto",
      copy: "Copiar",
      copied: "Copiado",
      codingAgents: "Agentes de programação",
      codingAgentsHint:
        "Inicie um agente de programação ligado a este servidor. Utiliza o modelo carregado; um servidor local gera automaticamente uma chave de API e um servidor remoto inclui-a no comando.",
      openAgentsTab: "Abrir em Agentes",
      codingAgentsSwap:
        "Substitua claude por codex, openclaw, opencode ou hermes.",
      relativeNever: "nunca",
      relativeJustNow: "agora mesmo",
      expired: "expirado",
      expiredStatus: "Expirado",
      today: "hoje",
      actionsFor: "Ações para {name}",
      copyPrefix: "Copiar prefixo",
      copyFailed: "Não foi possível copiar",
      revokeToken: "Revogar token",
      regenerateToken: "Gerar novamente o token",
      regenerateTitle: "Gerar novamente o token de acesso “{name}”?",
      regenerateDescription:
        "Um novo token com o mesmo nome e o mesmo período de validade substitui este. As aplicações que utilizam o token atual perdem o acesso imediatamente.",
      regenerateAction: "Gerar novamente",
      regenerating: "A gerar novamente...",
      regenerateError: "Não foi possível gerar novamente o token de acesso.",
      revokeTitle: "Revogar o token de acesso “{name}”?",
      revokeDescription:
        "As aplicações que utilizam este token perdem o acesso imediatamente. Esta ação não pode ser anulada.",
      revokeAction: "Revogar “{name}”",
      revoking: "A revogar...",
      audioApi: {
        title: "API de áudio",
        description:
          "Fala, clonagem de voz, transcrição e os fluxos de trabalho da página Áudio.",
        speak: "Falar",
        clone: "Clonar",
        transcribe: "Transcrever",
        workflows: "Fluxos de trabalho",
        separate: "Separar",
        convert: "Converter",
        music: "Música",
        edit: "Editar",
        placeholderModel:
          "Ainda nenhum modelo transferido consegue executar este exemplo, pelo que o modelo mostrado é um substituto. Transfira primeiro um modelo na página Áudio.",
        autoSwitchOff:
          "O modelo só é carregado pelo nome enquanto a Troca automática de modelo estiver ativada. Ative-a ou carregue primeiro o modelo na respetiva página Áudio.",
        useViaApi: "Utilizar através da API",
      },
      mediaApi: {
        title: "API de imagens / vídeo",
        description:
          "Gere imagens e vídeos a partir do seu próprio código com os endpoints compatíveis com OpenAI.",
        images: "Imagens",
        video: "Vídeo",
        placeholderModel:
          "O modelo mostrado é um exemplo. Carregue um modelo na página Imagens ou Vídeo e abra esta janela a partir do respetivo botão API para o preencher.",
        autoSwitchOff:
          "A troca automática de multimédia está desativada, pelo que os pedidos são executados no modelo que a página tem carregado, independentemente do nome indicado.",
      },
      trainingApi: {
        title: "API de treino",
        description:
          "Afine modelos a partir do seu próprio código com os endpoints de afinação compatíveis com OpenAI: carregue um ficheiro JSONL, crie uma tarefa, acompanhe os eventos e depois converse com o modelo treinado.",
        placeholder:
          "O modelo e as definições mostrados são um exemplo. Abra esta janela a partir do botão API na página Treinar para preencher a sua configuração atual.",
        hubDataset:
          "A página Treinar utiliza um conjunto de dados do Hugging Face ou do S3. A API treina com ficheiros carregados, por isso guarde o conjunto de dados como JSONL e carregue esse ficheiro.",
        batching:
          "As tarefas compatíveis em espera na fila são treinadas em conjunto automaticamente.",
        chatNote:
          "Para conversar com o modelo treinado pelo nome, é necessário ter a Troca automática de modelo ativada; caso contrário, carregue-o primeiro na página Chat.",
        cancelTitle: "Cancelar uma tarefa",
        cancelDescription:
          "Para uma tarefa em fila ou em execução. Não faz parte do exemplo acima.",
      },
      decisionApi: {
        title: "API de decisão",
        docs: "Documentação",
        docsLabel: "Abrir a documentação da API de decisão",
        description:
          "Responda a perguntas de sim/não, de escolha múltipla e de pontuação sobre texto com um modelo neste computador ou um modelo de decisão de Ligações. Funciona com o SDK TypeSafe.",
        enable: "Servir pedidos",
        enableDescription:
          "Serve /v1/systemone. Ao ativar, o modelo é transferido.",
        enableRemoteDescription: "Serve /v1/systemone.",
        lockedByEnv: "Definido por {name}.",
        model: "Modelo",
        thisMachine: "Este computador",
        modelMultilingual: "Multilingue",
        modelEnglish: "Inglês",
        modelTypedDecisions: "Decisões tipificadas",
        modelClefFlash: "Clef-flash",
        modelClef: "Clef",
        recommended: "Recomendado",
        device: "Executar em",
        deviceDescription:
          "A GPU responde mais depressa e liberta a memória quando o modelo é descarregado.",
        deviceCpu: "CPU",
        deviceGpu: "GPU",
        backend: "Runtime",
        backendAuto: "Automático",
        backendDescription:
          "Automático serve o Clef através do llama.cpp quando o modelo tem uma versão GGUF e recorre ao PyTorch caso contrário. O llama.cpp também lê imagens.",
        backendDescriptionMlx:
          "Automático serve o Clef através do MLX no Apple Silicon e através do llama.cpp quando apenas a versão GGUF do modelo está carregada ou transferida. O MLX lê imagens apenas para modelos Clef. O llama.cpp também lê imagens.",
        backendStatus: "Runtime: {backend}",
        backendNone: "indisponível",
        mediaImages: "Lê texto e imagens.",
        mediaText: "Lê apenas texto.",
        checking: "A verificar…",
        notDownloaded: "Não transferido · {size}",
        downloading: "A transferir…",
        downloaded: "Transferido · carrega no primeiro pedido",
        ready: "Pronto · carrega no primeiro pedido",
        installing: "A instalar…",
        loading: "A carregar…",
        loadedOn: "Carregado em {device}",
        download: "Transferir",
        downloadConfirmTitle: "Transferir Laya {model}?",
        downloadConfirmTitleModel: "Transferir {model}?",
        downloadConfirmBody:
          "A API de decisão precisa deste modelo para responder aos pedidos. Cerca de {size}, obtido uma única vez para a sua cache do Hugging Face.",
        unload: "Descarregar",
        downloadBusy: "Já está a ser transferido um modelo da API de decisão.",
        downloadFailed: "Não foi possível iniciar a transferência.",
        saveFailed: "Não foi possível guardar a definição da API de decisão.",
        loadError: "Não foi possível carregar as definições da API de decisão.",
        sendsTo: "Os pedidos são enviados para {provider}.",
        connectionMissing:
          "Esta ligação já não existe ou não tem modelos de decisão. Escolha outro modelo.",
        addConnection:
          "Para utilizar um modelo de decisão alojado na nuvem, adicione um em ",
        connectionsLink: "Ligações",
        addConnectionSuffix: ".",
      },
      mcp: {
        title: "Acesso de agentes (MCP)",
        description:
          "Permita que agentes de programação como o Claude Code e o Codex utilizem o Unsloth através de MCP. Os agentes iniciam sessão com uma chave de API do Unsloth desta página.",
        enable: "Permitir ligações de agentes",
        enableDescription:
          "Serve /mcp/ aos pedidos que incluam uma chave de API do Unsloth.",
        lockedByEnv: "Definido por {name}.",
        loadError:
          "Não foi possível carregar as definições de acesso de agentes.",
        saveError: "Não foi possível guardar a definição de acesso de agentes.",
        agent: "Agente",
        exportKeyHint:
          "Defina {name} com uma chave de API do Unsloth desta página antes de iniciar o agente.",
        configFileHint: "Adicione isto a {path}.",
        keywords:
          "mcp model context protocol protocolo de contexto de modelo agentes claude codex ferramentas",
      },
    },
    about: {
      title: "Acerca de",
      description:
        "Documentação, notas de versão, comentários e informações da compilação.",
      studioVersion: "Versão do Unsloth",
      packageVersion: "Versão do pacote",
      desktopAppVersion: "Versão da aplicação de ambiente de trabalho",
      desktopAppVersionUnavailable: "Indisponível",
      llamaCppVersion: "Versão do llama.cpp",
      audioCppVersion: "Versão do audio.cpp",
      mlxVersion: "Versão do MLX",
      whisperCppVersion: "Versão do whisper.cpp",
      engines: {
        title: "Motores",
        description:
          "Os runtimes que o Unsloth utiliza para executar modelos. As atualizações são instaladas em segundo plano, sem reiniciar.",
        updatesLabel: "Atualizações dos motores",
        notInstalled: "Não instalado",
        available: "{version} disponível",
        upToDate: "Atualizado",
        updatesAvailable: "Atualizações disponíveis",
        checking: "A procurar atualizações...",
        checkFailed:
          "Não foi possível procurar atualizações. Tente novamente mais tarde.",
        unsupported: "Os motores desta instalação são geridos fora do Unsloth.",
        checkForUpdates: "Procurar atualizações",
        update: "Atualizar",
        updating: "A atualizar...",
        updated: "Motores atualizados.",
        updateFailed: "A atualização falhou: {error}",
      },
      hardware: "Hardware",
      gpu: "GPU",
      gpuIntegrated: "integrada",
      cuda: "CUDA",
      rocm: "ROCm",
      xpu: "XPU",
      updates: "Atualização",
      help: "Ajuda",
      documentation: "Documentação",
      releaseNotes: "Notas de versão",
      whatsNew: "Novidades",
      feedback: "Comentários",
      reportIssue: "Comunicar um problema",
      license: {
        sectionTitle: "Licença",
        studioLabel: "Unsloth",
        studioLicense: "AGPL-3.0",
        studioDescription: "Código aberto sob a GNU AGPL v3.0.",
        libraryLabel: "Unsloth Core",
        libraryLicense: "Apache-2.0",
        libraryDescription: "Licenciado ao abrigo da Apache 2.0.",
      },
      dangerZone: "Zona de perigo",
      shutDownStudio: "Encerrar o Unsloth",
      shutDownStudioDescription:
        "Para o servidor do Unsloth e termina a sua sessão.",
      shutDown: "Encerrar",
      update: {
        title: "Atualizar o Unsloth",
        commandText: "Texto de {label}",
        copied: "Copiado",
        copyCommand: "Copiar comando",
        commandCopied: "{label} copiado",
        copyNamedCommand: "Copiar {label}",
        checkingInstall: "A verificar como o Unsloth foi instalado...",
        installIntro: "Para instalar ou atualizar o Unsloth:",
        localUpdateHeading: "Atualização local",
        installCommandUnix: "Comando de instalação para macOS/Linux",
        installCommandWindows: "Comando de instalação para Windows",
        localInstallDetected:
          "Foi detetada uma instalação local. Atualize a partir do seu checkout original para evitar substituí-lo pela versão do PyPI.",
        pullThenUpdate:
          "Obtenha as alterações mais recentes e, em seguida, execute o instalador local:",
        gitPullCommand: "comando git pull",
        localInstallerCommand: "comando do instalador local",
        sourceInstallDetected:
          "Foi detetada uma instalação a partir do código-fonte ou de um pacote VCS. Reinstale a partir do caminho local ou do URL Git original.",
        repoCheckoutFallback:
          "Se ainda tiver o checkout do repositório, execute o instalador local a partir dele:",
        restartAfterUpdate: "Reinicie o Unsloth após a atualização.",
        desktopManaged:
          "A aplicação de ambiente de trabalho procura automaticamente novas versões. Também pode procurar ou atualizar aqui a qualquer momento.",
        desktopReady: "Atualizações da aplicação de ambiente de trabalho",
        desktopReadyDescription:
          "Verifique se está disponível uma versão mais recente da aplicação de ambiente de trabalho.",
        desktopChecking: "A procurar atualizações",
        desktopCheckingDescription: "Normalmente demora alguns segundos.",
        desktopAvailable:
          "Está disponível a aplicação de ambiente de trabalho {version}",
        desktopAvailableDescription:
          "Atualize agora e a aplicação de ambiente de trabalho será reiniciada quando terminar.",
        desktopExternalServer:
          "A aplicação ligou-se a um servidor do Unsloth que já estava em execução e não o consegue atualizar. Pare esse servidor e, em seguida, saia e volte a abrir a aplicação de ambiente de trabalho para atualizar.",
        desktopManualInstall:
          "Abra a página da versão para instalar o pacote Linux mais recente.",
        desktopCheckFailed: "Não foi possível procurar atualizações",
        desktopCheckFailedDescription:
          "Verifique a sua ligação e tente novamente.",
        desktopCurrent: "A aplicação de ambiente de trabalho está atualizada",
        desktopCurrentDescription:
          "O Unsloth continuará a procurar atualizações automaticamente.",
        checkForUpdates: "Procurar atualizações",
        checkAgain: "Procurar novamente",
        retryCheck: "Tentar novamente",
        checking: "A procurar...",
        updateNow: "Atualizar agora",
        openReleasePage: "Abrir página da versão",
        unknownInstall:
          "Não foi possível detetar como o Unsloth foi instalado. Para instalações através do instalador ou do PyPI, utilize os comandos acima.",
        localCheckout:
          "Para instalações a partir de um checkout local, execute o instalador local a partir desse checkout:",
        docs: "Documentação de instalação:",
        docsInstall: "Instalação",
        docsUpdating: "Atualização",
        docsMac: "Mac",
        docsWindows: "Windows",
        docsLinux: "Linux",
        docsDocker: "Docker",
        docsAmd: "AMD",
      },
    },
  },
  studio: {
    routeTitle: "Treinar",
    wizard: {
      modelTitle: "Modelo",
      modelDescription: "Selecione o modelo e o método de treino",
      datasetTitle: "Conjunto de dados",
      datasetDescription: "Selecione ou carregue dados de treino",
      paramsTitle: "Parâmetros",
      paramsDescription: "Configure os parâmetros de treino",
      configTitle: "Configuração",
      configDescription: "Guarde e carregue configurações",
      modelLabel: "Modelo",
      modelTooltip: "O modelo base que pretende afinar.",
      methodLabel: "Método",
      methodTooltip:
        "Como o modelo é treinado. O LoRA e o QLoRA atualizam pequenos adaptadores em vez de todos os pesos.",
      trainAsLabel: "Treinar como",
      trainAsTooltip:
        "Um modelo de linguagem escreve texto. Um modelo de decisão escolhe uma das opções que lhe forem dadas, com uma probabilidade, e é executado na API de decisão.",
      trainAsLanguage: "Modelo de linguagem",
      trainAsDecision: "Modelo de decisão",
      checkpointLabel: "Checkpoint",
      checkpointTooltip:
        "O checkpoint Laya a afinar. O Multilingue adequa-se à maioria dos conjuntos de dados.",
      datasetLabel: "Conjunto de dados",
      datasetTooltip: "Os dados de treino utilizados para afinar o modelo.",
      hfTokenDescription:
        "Obrigatório para modelos e conjuntos de dados restritos ou privados.",
      uploadLocalLabel: "Carregar um ficheiro local",
      sourceBrowse: "Procurar",
      releaseToUpload: "Largue para carregar",
      loadYaml: "Carregar YAML",
      saveYaml: "Guardar YAML",
      resetDefaults: "Repor predefinições",
      cachedModelGoneTitle: "O modelo em cache já não está disponível",
      cachedModelGoneDescription:
        "Os ficheiros do modelo já não estão neste dispositivo, pelo que o treino irá transferi-los novamente.",
      cachedDatasetGoneTitle:
        "O conjunto de dados em cache já não está disponível",
      cachedDatasetGoneDescription:
        "Os ficheiros do conjunto de dados já não estão neste dispositivo, pelo que o treino irá transferi-los novamente.",
    },
    preview: {
      title: "Pré-visualização da execução",
      ready: "Pronto",
      notReady: "Não está pronto",
      modelPending: "Modelo pendente",
      datasetPending: "Conjunto de dados pendente",
      method: "Método",
      length: "Duração",
      stepZero: "{count} passos",
      step: "{count} passo",
      stepTwo: "{count} passos",
      stepFew: "{count} passos",
      stepMany: "{count} passos",
      steps: "{count} passos",
      epochZero: "{count} épocas",
      epoch: "{count} época",
      epochTwo: "{count} épocas",
      epochFew: "{count} épocas",
      epochMany: "{count} épocas",
      epochs: "{count} épocas",
      batch: "Lote",
      context: "Contexto",
      lr: "TA",
      hardware: "Hardware",
      vram: "VRAM",
      noGpu: "Nenhuma GPU detetada",
      hfToken: "Token do HF",
      saved: "Guardado",
      notSet: "Não definido",
      files: "Ficheiros",
      model: "Modelo",
      dataset: "Conjunto de dados",
      downloadsOnStart: "Transferido ao iniciar",
      continuesOnStart: "Continua ao iniciar",
      noticeModelDownload:
        "Este modelo ainda não está neste dispositivo. O treino irá transferi-lo automaticamente.",
      noticeModelPartial:
        "O treino irá concluir a transferência parcial do modelo antes de o carregar.",
      noticeDatasetDownload:
        "Este conjunto de dados ainda não está neste dispositivo. O treino irá transferi-lo automaticamente.",
      noticeDatasetPartial:
        "O treino irá concluir a transferência parcial do conjunto de dados antes de o ler.",
      noticeTransformersUpgrade:
        "Nenhuma versão instalada do transformers suporta ainda esta arquitetura. Ao iniciar a execução, será proposta primeiro a instalação do transformers {version}.",
      noticeSixteenBitOnly:
        "Esta arquitetura treina em LoRA de 16 bits: os 4 bits não estão disponíveis para ela, pelo que a execução precisa de muito mais VRAM do que o QLoRA.",
      noticeInstallSwitchesSixteenBit:
        "Instalar essa versão em vez de manter o código próprio do modelo muda esta execução para LoRA de 16 bits, que precisa de muito mais VRAM do que o QLoRA.",
      advancedSettings: "Definições avançadas",
      defaultAdvancedSettings: "Predefinições",
      nonDefaultAdvancedSettings: "{count} não predefinidas",
    },
    datasetPicker: {
      noun: "conjuntos de dados",
      selectDataset: "Selecionar conjunto de dados",
      hubPlaceholder: "Pesquisar conjuntos de dados do Hugging Face...",
      devicePlaceholder: "Pesquisar conjuntos de dados locais...",
      useAsHubDataset: "Utilizar como conjunto de dados do Hugging Face",
      hfCacheLabel: "Cache do HF",
      scanningLocal: "A analisar os conjuntos de dados neste dispositivo…",
      couldntScan: "Não foi possível analisar os conjuntos de dados locais",
      someLocationsUnscanned:
        "Não foi possível analisar algumas localizações de conjuntos de dados.",
      noLocalDatasets:
        "Ainda não há nada neste dispositivo. Transfira um conjunto de dados do Hub, crie um em Receitas ou carregue um ficheiro.",
      openDataRecipes: "Abrir Data Recipes",
      searchingHub: "A pesquisar no Hugging Face…",
      noDatasetsFound: "Não foram encontrados conjuntos de dados.",
      tokenRejectedTitle: "Token do Hugging Face rejeitado",
      tokenRejectedBody:
        "Atualize o seu token em Definições → Geral e tente novamente.",
      hubUnreachable: "Não foi possível aceder ao Hugging Face",
      cantUseDataset: "Não é possível utilizar o conjunto de dados",
      reasonInvalidHubId:
        "Introduza um ID de conjunto de dados do Hugging Face válido: repo ou owner/repo, sem pontos ou hífenes consecutivos e sem o sufixo .git (máximo de 96 carateres por parte).",
      sourceRecipe: "Receita",
      sourceUpload: "Carregamento",
      sourceLocal: "Local",
    },
    modelPicker: {
      noun: "modelos",
      selectModel: "Selecionar modelo",
      hubPlaceholder: "Pesquise ou cole um ID do Hugging Face...",
      devicePlaceholder:
        "Pesquise modelos locais ou cole o caminho de uma pasta...",
      useAsHubModel: "Utilizar como modelo do Hugging Face",
      useAsLocalPath: "Utilizar como caminho local",
      hfCacheLabel: "Cache do HF",
      scanningLocal: "A analisar os modelos locais…",
      couldntScan: "Não foi possível analisar os modelos locais",
      someLocationsUnscanned:
        "Não foi possível analisar algumas localizações locais.",
      noLocalModels: "Não foram encontrados modelos locais.",
      noLocalModelsHint:
        "Cole o caminho de uma pasta acima ou mude para o Hugging Face.",
      searchingHub: "A pesquisar no Hugging Face…",
      noModelsFound: "Não foram encontrados modelos.",
      tokenRejectedTitle: "Token do Hugging Face rejeitado",
      tokenRejectedBody:
        "Atualize o seu token em Definições → Geral e tente novamente.",
      tokenRejectedAnonymousBody:
        "Os modelos públicos são mostrados sem token. Atualize o seu token em Definições → Geral para aceder a modelos privados e restritos.",
      hubUnreachable: "Não foi possível aceder ao Hugging Face",
      cantUseModel: "Não é possível utilizar o modelo para treino",
      reasonTypeMismatch:
        "Este modelo não corresponde ao tipo de treino selecionado no passo anterior.",
      reasonEmptyId:
        "Introduza um ID de modelo ou o caminho de um modelo local.",
      reasonGguf: "Os modelos GGUF não podem ser utilizados para treino.",
      reasonAdapter:
        "As saídas de adaptadores não podem ser utilizadas como modelos base de treino.",
      reasonNotTrainable: "Este modelo no dispositivo não pode ser treinado.",
      reasonUnsupportedFormat:
        "Este formato de modelo não é suportado para treino.",
      reasonInvalidHubId:
        "Introduza um ID de modelo do Hugging Face válido: repo ou owner/repo, sem pontos ou hífenes consecutivos e sem o sufixo .git (máximo de 96 carateres por parte).",
      sourceModelsFolder: "Pasta de modelos",
      sourceHfCache: "Cache do HF",
      sourceLmStudio: "LM Studio",
      sourceOllama: "Ollama",
      sourceHermes: "Hermes",
      sourceCustomFolder: "Pasta personalizada",
      sourceLocalModel: "Modelo local",
      vramOomBadge: "OOM",
      vramTightBadge: "Justo",
      vramNeeds: "Precisa de ~{est}GB de VRAM (GPU: {total}GiB)",
      vramTight: "~{est}GB de VRAM (justo em {total}GiB)",
      vramApprox: "~{est}GB de VRAM",
    },
    methods: {
      qlora: {
        label: "QLoRA",
        hint: "Quantização de 4 bits. Menor VRAM, início mais rápido.",
        note: "4 bits",
      },
      lora: {
        label: "LoRA",
        hint: "Adaptadores de 16 bits. Equilíbrio entre qualidade e memória.",
        note: "16 bits",
      },
      full: {
        label: "Afinação completa",
        hint: "Treina todos os pesos. Qualidade máxima, precisa de mais VRAM.",
        note: "fp16",
      },
      cpt: {
        label: "Pré-treino contínuo",
        hint: "Pré-treino contínuo para novos domínios ou idiomas.",
        note: "contínuo",
      },
    },
    subtitles: {
      configure: "Configure e inicie o treino",
      trainingInProgress: "Treino em curso",
      viewPastRuns: "Ver execuções de treino anteriores",
      viewingPastRun: "A ver uma execução anterior",
    },
    tabs: {
      configure: "Configurar",
      currentRun: "Execução atual",
      history: "Histórico",
    },
    goToLibrary: "Ir para a Biblioteca",
    imageTraining: "Treino de imagens",
    goToImageTraining: "Treinar LoRAs de imagem",
    loadingRuntime: "A carregar o runtime de treino...",
    checkingSupport: "A verificar o suporte de treino neste computador...",
    backToHistory: "Voltar ao histórico",
    dataset: {
      selectors: {
        subset: "Subconjunto",
        subsetTooltip:
          "Selecione o subconjunto (config) do conjunto de dados a utilizar.",
        trainSplit: "Divisão de treino",
        trainSplitTooltip: "Selecione a divisão a utilizar para o treino.",
        evaluationSplit: "Divisão de avaliação",
        evaluationSplitTooltip:
          "Selecione a divisão a utilizar para a avaliação. Nenhuma significa que não há avaliação durante o treino.",
        selectSubset: "Selecione um subconjunto...",
        selectSplit: "Selecione uma divisão...",
        none: "Nenhuma",
        loading:
          "A carregar as configurações e divisões do conjunto de dados...",
        manualTitle: "Introduzir manualmente as opções do conjunto de dados",
        manualDescription:
          "Introduza os nomes exatos da configuração e da divisão do Hugging Face a utilizar.",
        manualSubsetPlaceholder: "Nome da configuração (opcional)",
        manualRequired: "É obrigatória uma divisão de treino.",
        manualTooLong: "Utilize 128 carateres ou menos.",
        manualInvalid: "Este valor contém carateres não suportados.",
      },
      sourceAriaLabel: "Origem do conjunto de dados",
      localDataset: "Conjunto de dados local",
      localDatasetRows: " / {count} linhas",
      huggingFaceDataset: "Conjunto de dados do Hugging Face",
      localDatasetMetadata: "Metadados do conjunto de dados local",
      dataRecipeOutput: "Saída de Data Recipe.",
      rows: "Linhas",
      columns: "Colunas",
      batches: "Lotes",
      updated: "Atualizado",
      evalDataset: "Conjunto de dados de avaliação",
      uploading: "A carregar...",
      uploadEvalFile: "Carregar ficheiro de avaliação",
      fileTooLarge: "Ficheiro demasiado grande",
      fileTooLargeDescription:
        "{file} tem {size}. Os carregamentos para treino suportam até {limit}.",
      documentRedirect: {
        title: "Este ficheiro tem de ser convertido primeiro",
        genericFile: "Este ficheiro",
        description:
          "{file} é material de origem, não um conjunto de dados pronto para treino. Utilize as Data Recipes para transformar o documento num conjunto de dados e, em seguida, volte aqui para afinar.",
        nextStepTitle: "Melhor passo seguinte",
        nextStepDescription:
          "Abra as Learning Recipes e comece com uma receita baseada em documentos, como PDF grounded QA.",
        openAction: "Abrir Learning Recipes",
      },
      evalDatasetDescription:
        "Opcional. Se não for indicado, será separada uma pequena parte dos dados de treino.",
      advanced: "Avançado",
      targetFormat: "Formato de destino",
      targetFormatTooltip:
        "Formato dos seus dados de treino. A deteção automática funciona para a maioria dos conjuntos de dados.",
      streamingInfoAriaLabel:
        "Informações sobre o streaming do conjunto de dados",
      streaming: {
        label: "Ativar streaming",
        description:
          "Faz streaming de conjuntos de dados de texto do Hugging Face em vez de os transferir.",
        unavailable: "Streaming indisponível. Para o ativar:",
        completionsUnavailable:
          "Indisponível enquanto o streaming do conjunto de dados estiver ativado.",
        blockers: {
          source:
            "Utilize um conjunto de dados do Hugging Face (não um carregamento local nem uma origem S3).",
          maxSteps:
            "Defina Máximo de passos > 0 — os conjuntos de dados em streaming não têm comprimento conhecido.",
          trainOnCompletions: "Desative “Apenas conclusões do assistente”.",
          evalSplit:
            "Escolha uma divisão de avaliação separada — a avaliação está ativada, mas não está definida nenhuma divisão de avaliação distinta.",
          visionModel: "Os modelos de visão não suportam streaming.",
          audioModel: "Os modelos de áudio não suportam streaming.",
          embeddingModel:
            "Os modelos de embeddings não suportam streaming (o treino precisa do conjunto de dados completo).",
          imageDataset:
            "Este conjunto de dados parece conter imagens, que não suportam streaming.",
          audioDataset:
            "Este conjunto de dados parece conter áudio, que não suporta streaming.",
          appleSilicon:
            "O streaming ainda não é suportado no Apple Silicon (MLX).",
        },
        options: {
          trainOnCompletions: "apenas conclusões do assistente",
          evaluation:
            "avaliação (precisa de uma divisão de avaliação separada)",
        },
        notifications: {
          turnedOffMaxSteps:
            "Streaming desativado: o streaming precisa de um Máximo de passos fixo > 0.",
          adjusted:
            "Ajustado para streaming. Opções incompatíveis desativadas: {options}.",
          needsMaxSteps:
            "O streaming precisa de um Máximo de passos fixo (os conjuntos de dados em streaming não têm comprimento conhecido). Defina primeiro Máximo de passos > 0.",
          enabledAdjusted:
            "Streaming ativado. Opções incompatíveis desativadas: {options}.",
          disabledForDetectedModality:
            "O streaming foi desativado porque os conjuntos de dados de imagem e áudio requerem uma transferência completa. Reveja a definição e inicie novamente.",
        },
      },
      auto: "Automático",
      rawText: "Texto simples",
      trainSplitStart: "Início da divisão de treino",
      trainSplitStartTooltip:
        "Treine apenas com uma parte da divisão de treino: indique o índice da linha inicial (inclusive, a partir de 0). Deixe vazio para começar na primeira linha.",
      trainSplitEnd: "Fim da divisão de treino",
      trainSplitEndTooltip:
        "Índice da última linha a incluir da divisão de treino (inclusive, a partir de 0). Por exemplo, defina Início como 0 e Fim como 99 para treinar com as primeiras 100 linhas. Deixe vazio para utilizar todas as linhas restantes.",
      endPlaceholder: "Fim",
      clear: "Limpar",
      dropFileOrClick: "Largue um ficheiro ou clique para carregar",
      uploadDetails: "Detalhes do carregamento",
      uploadDetailsTooltip:
        "Até {limit} por ficheiro. PDF, DOCX e TXT não são conjuntos de dados prontos para treino, por isso converta-os primeiro nas Learning Recipes.",
      viewDataset: "Ver conjunto de dados",
      previewTokens: "Pré-visualizar tokens",
      tokenPreview: {
        title: "Pré-visualização de tokens",
        description:
          "Como o {model} tokeniza estes dados para o treino. Apenas o tokenizador é transferido.",
        datasetTab: "Amostras do conjunto de dados",
        textTab: "O seu texto",
        trained: "Treinado",
        masked: "Mascarado",
        completionsOnly: "Apenas conclusões do assistente",
        allTokens: "Todos os tokens treinados",
        responseMarker: "Marcador de resposta",
        maskingNotApplied:
          "Os marcadores de chat não foram encontrados, pelo que todos os tokens são treinados.",
        loading:
          "A carregar o tokenizador. A primeira pré-visualização demora um pouco mais...",
        shuffle: "Baralhar amostras",
        showIds: "Mostrar IDs dos tokens",
        copyIds: "Copiar IDs",
        copied: "IDs dos tokens copiados",
        sample: "Amostra {index}",
        stats: "{tokens} tokens, {trained} treinados ({percent}%)",
        truncated:
          "Cortado para {max} de {full} tokens pelo comprimento do contexto.",
        noTrained:
          "Nenhum token desta linha é treinado, pelo que o treino a ignora.",
        hiddenTokens: "Mais {count} tokens não mostrados",
        multimodal:
          "As linhas de imagem e áudio não são mostradas aqui. Pode continuar a experimentar o tokenizador com o seu próprio texto.",
        textPlaceholder:
          "Escreva ou cole texto para ver os respetivos tokens...",
        textHint:
          "Com Apenas conclusões do assistente, só o texto após um marcador do assistente é treinado.",
        insertExample: "Inserir exemplo de chat",
        vocab: "Vocabulário: {count} tokens",
        retry: "Tentar novamente",
        packing:
          "O empacotamento está ativado: a execução junta linhas em sequências com o comprimento do contexto, pelo que o ponto em que uma linha é cortada pode diferir desta vista por linha.",
      },
      uploadFailed: "O carregamento falhou",
      unknownError: "Erro desconhecido",
      unsupportedFileType: "Tipo de ficheiro não suportado",
      uploadOneFileType: "Carregue um ficheiro {types}.",
      datasetUploaded: "Conjunto de dados carregado",
      evalDatasetUploaded: "Conjunto de dados de avaliação carregado",
      uploadOneFileAtATime: "Carregue um ficheiro de cada vez",
      uploadSingleFileDescription:
        "O carregamento do conjunto de dados de treino aceita um único ficheiro.",
      previewLoadingHuggingFace:
        "A obter a pré-visualização do conjunto de dados do Hugging Face...",
      previewLoading: "A carregar a pré-visualização...",
      mappingRequirements: {
        audioAndText: "áudio e texto",
        imageAndText: "imagem e texto",
        instructionAndOutput: "instrução e saída",
        humanAndGpt: "humano e GPT",
        userAndAssistant: "utilizador e assistente",
      },
      mappingStatus: {
        heuristicTitle: "Mapeamento detetado por heurística",
        readyTitle: "Mapeamento pronto",
        requiredTitle: "Mapear colunas do conjunto de dados",
        heuristicDescription:
          "O mapeamento de colunas abaixo foi detetado automaticamente com heurísticas. Reveja-o e ajuste-o através das listas pendentes nos cabeçalhos das colunas, ou utilize o AI Assist para um mapeamento mais inteligente.",
        readyDescription:
          "Está tudo certo. Este conjunto de dados será convertido automaticamente.",
        requiredDescription:
          "Atribua funções às colunas através das listas pendentes nos cabeçalhos. No mínimo, atribua {required}.",
      },
      s3: {
        title: "Configuração do S3",
        description:
          "Carregue conjuntos de dados .parquet, .json, .jsonl ou .csv do Amazon S3",
        bucket: "Nome do bucket",
        bucketPlaceholder: "my-training-data-bucket",
        region: "Região da AWS",
        regionPlaceholder: "us-east-1",
        prefix: "Prefixo do caminho",
        prefixPlaceholder: "datasets/whisper/",
        accessKeyId: "ID da chave de acesso",
        accessKeyIdPlaceholder: "AKIAIOSFODNN7EXAMPLE",
        secretAccessKey: "Chave de acesso secreta",
        secretAccessKeyPlaceholder: "A sua chave de acesso secreta da AWS",
        useIamRole: "Utilizar função IAM",
      },
    },
    params: {
      mode: {
        simple: "Simples",
        advanced: "Avançado",
        ariaLabel: "Modo de parâmetros",
      },
      projectName: "Nome do projeto",
      optional: "Opcional",
      projectNameDescription:
        "Utilizado nos nomes das pastas de saída do treino, nas predefinições de exportação e no histórico.",
      loraSettings: "Definições de LoRA",
      trainingHyperparameters: "Hiperparâmetros de treino",
      drafterSettings: "Drafter de descodificação especulativa",
      drafterTrain: "Treinar um drafter de descodificação especulativa",
      drafterTrainTooltip:
        "Treina uma pequena cabeça de rascunho após a afinação para que a inferência seja mais rápida no llama.cpp e no vLLM; as saídas não se alteram.",
      drafterChecking: "A verificar este modelo...",
      drafterUnavailable: "Indisponível para este modelo: {reason}",
      drafterRetrainMtp: "Volta a treinar a cabeça MTP deste modelo",
      drafterEagle3:
        "Treina um drafter EAGLE-3 (este modelo não tem cabeça MTP)",
      drafterDataSource: "Dados de treino",
      drafterDataSourceDataset: "Respostas do conjunto de dados",
      drafterDataSourceRegenerate:
        "Respostas do próprio modelo (mais lento, melhor)",
      drafterMaxExamples: "Exemplos",
      drafterEpochs: "Épocas",
      drafterLearningRate: "Taxa de aprendizagem",
      drafterLearningRateDefault: "Predefinição",
      drafterDraftSteps: "Tokens de rascunho",
      maxSteps: "Máximo de passos",
      epochs: "Épocas",
      useMaxSteps: "Utilizar máximo de passos",
      useEpochs: "Utilizar épocas",
      maxStepsTooltip: "Substitui o número total de passos do otimizador.",
      epochsTooltip: "Número de passagens completas pelo conjunto de dados.",
      contextLength: "Comprimento do contexto",
      contextLengthTooltip: "Número máximo de tokens por amostra de treino.",
      customContextLength: "Introduza um valor personalizado",
      learningRate: "Taxa de aprendizagem",
      learningRateTooltip:
        "Tamanho do passo nas atualizações dos pesos. Valores mais baixos treinam mais devagar, mas de forma mais estável.",
      learningRateDescription:
        "Recomendado: 2e-4 para LoRA, 5e-5 para CPT, 2e-5 para afinação completa",
      embeddingLearningRate: "Taxa de aprendizagem dos embeddings",
      embeddingLearningRateTooltip:
        "Utilizada apenas quando o CPT treina embed_tokens. Os embeddings são mais fáceis de desestabilizar do que os pesos LoRA, pelo que normalmente precisam de uma taxa de aprendizagem menor. Deixe em branco para utilizar lr/10; o intervalo habitual é 2x a 10x menor do que a taxa de aprendizagem principal. Aumente-a apenas se a adaptação do vocabulário ou dos tokens do domínio for demasiado lenta.",
      rank: "Rank",
      rankTooltip:
        "Dimensão das matrizes de baixo rank. Mais alto = mais capacidade.",
      alpha: "Alpha",
      alphaTooltip:
        "Fator de escala das atualizações LoRA. Normalmente 2x o rank.",
      dropout: "Dropout",
      dropoutTooltip:
        "Probabilidade de dropout das camadas LoRA para reduzir o sobreajuste.",
      visionLayers: "Camadas de visão",
      languageLayers: "Camadas de linguagem",
      attentionModules: "Módulos de atenção",
      mlpModules: "Módulos MLP",
      targetModules: "Módulos de destino",
      enableLora: "LoRA",
      trainWithLora: "Treinar com LoRA",
      stableRank: "LoRA com rank estabilizado",
      memoryEfficient: "LoRA inicializado por SVD",
      weightDecomposed: "LoRA com multiplicadores",
      notSupportedAppleSilicon: "Não suportado no Apple Silicon",
      doraNeedsVisionLayersOff:
        "Desative o treino das camadas de visão para utilizar o DoRA",
      optimization: "Otimização",
      schedule: "Programação",
      memory: "Memória",
      optimizer: "Otimizador",
      optimizerTooltip:
        "Algoritmo de otimização. As variantes de 8 bits reduzem a utilização de memória. Fused é recomendado para modelos de visão.",
      optimizerTooltipMlx:
        "Algoritmo de otimização. AdamW é a predefinição. Lion utiliza menos memória, mas normalmente precisa de uma taxa de aprendizagem mais baixa.",
      lrScheduler: "Programador da taxa de aprendizagem",
      lrSchedulerTooltip:
        "Como a taxa de aprendizagem varia ao longo do treino. Linear diminui de forma constante; cosseno diminui em curva.",
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
        cosine: "Cosseno",
      },
      batchSize: "Tamanho do lote",
      batchSizeTooltip:
        "Amostras processadas por passo. Valores mais altos usam mais VRAM.",
      gradAccum: "Passos de acumulação de gradiente",
      gradAccumTooltip: "Simula tamanhos de lote maiores sem VRAM adicional.",
      weightDecay: "Decaimento de pesos",
      weightDecayTooltip: "Regularização L2 para evitar sobreajuste.",
      warmupSteps: "Passos de aquecimento",
      warmupStepsTooltip:
        "Aumenta gradualmente a LR no início do treino para maior estabilidade.",
      scheduleEpochsTooltip:
        "Número de passagens completas pelo conjunto de dados. Defina 0 para executar até ao número máximo de passos.",
      saveSteps: "Passos entre gravações",
      saveStepsTooltip:
        "Guarda um checkpoint a cada N passos. 0 para desativar.",
      evalSteps: "Passos de avaliação",
      evalStepsTooltip:
        "Fração do total de passos de treino entre avaliações (0-1). Defina 0 para desativar a avaliação. Por exemplo, 0.01 = avaliar a cada 1% dos passos.",
      earlyStoppingPatience: "Paciência da paragem antecipada",
      earlyStoppingPatienceTooltip:
        "Para o treino após este número de avaliações consecutivas sem uma perda de avaliação inferior e mantém os pesos da melhor avaliação. Requer avaliação (Passos de avaliação acima de 0). 0 para desativar.",
      seed: "Semente",
      seedTooltip: "Semente aleatória para reprodutibilidade.",
      offloadLayers: "Deslocar camadas",
      offloadLayersTooltip:
        "Mantém as camadas do descodificador na RAM do sistema e transmite cada uma para a GPU imediatamente antes de ser executada, para que um modelo maior do que a VRAM possa ser treinado com LoRA. Automático desloca o mínimo de camadas necessário; um número desloca exatamente essa quantidade. Requer gradient checkpointing.",
      offloadOff: "Desativado",
      offloadAuto: "Automático",
      offloadCount: "Número",
      offloadVramBudget: "Orçamento de VRAM (GiB)",
      offloadVramBudgetTooltip:
        "A VRAM máxima que esta execução pode usar. Automático desloca camadas suficientes para ficar abaixo deste valor, e duas execuções podem partilhar uma placa, cada uma a utilizar parte dela. Vazio usa a placa inteira.",
      offloadWholeCard: "placa inteira",
      offloadVramBudgetGpu: "Orçamento de VRAM da GPU {index} (GiB)",
      prefetchDepth: "Profundidade de pré-carregamento",
      prefetchDepthTooltip:
        "Quantas camadas deslocadas são copiadas antecipadamente em relação à que está em execução. Automático começa em 1 e só mantém um conjunto mais profundo enquanto este reduzir de forma mensurável a espera pelas cópias.",
      offloadPanelTitle: "Camadas deslocadas",
      offloadPanelGpu: "Na GPU",
      offloadPanelHost: "Na RAM do sistema",
      offloadPanelCopying: "A copiar",
      offloadPanelSwapped: "deslocadas",
      offloadPanelStall: "à espera de cópias",
      offloadPanelCopy: "cópia por camada",
      offloadPanelCompute: "computação por camada",
      offloadPanelVram: "VRAM",
      offloadPanelDepth: "profundidade de pré-carregamento",
      offloadPanelPinned: "fixada",
      offloadPanelSweepNote:
        "A janela móvel mostra a ordem pela qual as camadas são obtidas, abrandada para ser visível; os valores são medidos no último passo.",
      offloadPanelCard: "GPU {index}",
      offloadPanelCardUnknown: "Placa desconhecida",
      gradCheckpoint: "Gradient checkpointing",
      gradCheckpointTooltip:
        "Troca computação por memória ao recalcular as ativações.",
      none: "Nenhum",
      standard: "Padrão",
      enablePacking: "Ativar packing",
      assistantCompletionsOnly: "Apenas respostas do assistente",
      readMore: "Saber mais",
    },
    training: {
      duplicateFailed: "Não foi possível duplicar a execução",
      duplicateDraftChanged:
        "As definições de treino foram alteradas durante o carregamento. Tente duplicar novamente.",
      duplicateNoModel:
        "Esta execução não tem nenhuma configuração de modelo guardada.",
      startTraining: "Iniciar treino",
      starting: "A iniciar...",
      loadingModel: "A carregar o modelo...",
      checkingDataset: "A verificar o conjunto de dados...",
      chooseModel: "Escolha um modelo",
      chooseDataset: "Escolha um conjunto de dados",
      chooseModelAndDataset: "Escolha o modelo e o conjunto de dados",
      validation: {
        s3MultimodalUnsupported:
          "Os conjuntos de dados S3 ainda não são suportados no treino de visão.",
        s3BucketRequired: "Introduza primeiro o nome de um bucket S3.",
        s3CredentialsRequired:
          "Forneça chaves de acesso S3 ou ative a função IAM.",
        modelRequired: "Selecione primeiro um modelo base.",
        learningRatePositive:
          "Introduza uma taxa de aprendizagem superior a zero.",
        embeddingLearningRateRange:
          "Introduza uma taxa de aprendizagem de embeddings superior a 0 e inferior a 1.",
        decisionColumnsMissing:
          "Os modelos de decisão necessitam das colunas state, questions e gold (ou answers). Em falta: {columns}.",
        decisionOwnerOnly:
          "Apenas o proprietário do Unsloth pode afinar modelos de decisão.",
        hfDatasetRequired:
          "Selecione primeiro um conjunto de dados do Hugging Face.",
        hfDatasetSplitRequired:
          "Selecione ou introduza primeiro uma divisão de treino.",
        localDatasetRequired: "Selecione primeiro um conjunto de dados local.",
        unsupportedDatasetSource: "Origem do conjunto de dados não suportada.",
      },
      startFailed: "Não foi possível iniciar o treino",
      startUnconfirmed:
        "O Unsloth não conseguiu confirmar se o treino foi iniciado. A verificar o estado em segundo plano.",
      stopFailed: "Não foi possível parar o treino",
      trainingStillActiveTitle: "O treino continua ativo",
      stopBeforeConfig: "Pare primeiro o treino e depois volte à configuração.",
      resumeFailed: "Não foi possível retomar o treino",
      resumeFailedTitle: "Não foi possível retomar o treino",
      resumeUnavailable:
        "Só é possível retomar execuções paradas ou com erro que tenham um checkpoint guardado.",
      modelUnverified:
        "Não foi possível verificar as definições deste modelo. Verifique a ligação ou o token do Hugging Face e tente novamente.",
      legacyDatasetScriptUnsupported:
        "Este conjunto de dados do Hub depende de um script personalizado antigo e não é suportado neste fluxo de treino.",
      hfModelAccessDenied:
        "O Hugging Face negou o acesso a este modelo. Adicione um token do Hugging Face válido com acesso ao repositório, aceite os termos de acesso necessários e tente novamente.",
      hfModelVerificationRateLimited:
        "A verificação de modelos do Hugging Face está limitada por taxa. Tente novamente daqui a pouco.",
      hfModelVerificationFailed:
        "Não foi possível verificar o modelo do Hugging Face. Verifique o ID do repositório e o token de acesso.",
      hfModelMetadataUnavailable:
        "Os metadados do modelo do Hugging Face estão temporariamente indisponíveis. Tente novamente antes de iniciar o treino.",
      datasetUnverified:
        "Não foi possível verificar se o conjunto de dados é compatível com este modelo. Verifique a ligação ou o token do Hugging Face; ao iniciar o treino, a verificação será repetida.",
      setupChanged:
        "A configuração do treino foi alterada enquanto estava a ser verificada. Reveja-a e inicie novamente.",
      uploadConfigTooltip: "Carregar uma configuração YAML guardada",
      saveConfigTooltip: "Transferir a configuração atual como YAML",
      resetConfigTooltip: "Repor as predefinições do modelo",
      configLoaded: "Configuração carregada",
      failedToLoadConfig: "Não foi possível carregar a configuração",
      invalidYamlFile: "Ficheiro YAML inválido",
      configTooLarge:
        "A configuração de treino é demasiado grande (máximo de 1 MiB).",
      failedToReadFile: "Não foi possível ler o ficheiro",
      failedToSaveConfig: "Não foi possível guardar a configuração",
      parametersReset: "Parâmetros repostos para as predefinições do modelo",
      audioIncompatible:
        "Este modelo não suporta áudio. Mude para um modelo com suporte de áudio ou escolha um conjunto de dados sem áudio.",
      visionIncompatible:
        "Um modelo de texto não é compatível com um conjunto de dados multimodal. Mude para um modelo de visão ou escolha um conjunto de dados apenas de texto.",
      cancelTitle: "Cancelar treino",
      cancelDescription: "Pretende cancelar a execução de treino atual?",
      continueAction: "Continuar treino",
      cancelAction: "Cancelar treino",
      stopTitle: "Parar treino",
      stopDescription:
        "Escolha como pretende parar a execução de treino atual. Parar e guardar grava um checkpoint a partir do qual pode retomar mais tarde; parar sem guardar não permite retomar.",
      stopAction: "Parar",
      stopping: "A parar...",
      stopAndSave: "Parar e guardar",
      compareInChat: "Comparar no chat",
      exportModel: "Exportar modelo",
      milestone: "Marco",
      halfwayDone: "A meio caminho. O treino ultrapassou os 50%.",
      doneNextStep:
        "Treino concluído. Passo seguinte: comparar as respostas do modelo base com as do modelo afinado.",
    },
    history: {
      title: "Histórico",
      filesDeleted: "Ficheiros eliminados",
      deleteArtifactsLabel:
        "Eliminar também os ficheiros do adaptador no disco",
      deleteArtifactsDescription:
        "Remove a pasta de saída da execução, incluindo os adaptadores e checkpoints guardados.",
      deleteArtifactsSharedNote:
        "Outra execução partilha esta pasta de saída. Os ficheiros são mantidos até que a última execução que os utiliza seja eliminada.",
      artifactsKeptShared:
        "Execução eliminada. Os ficheiros do adaptador foram mantidos porque outra execução utiliza a mesma pasta.",
      deleteArtifactsActiveError:
        "Estes ficheiros estão a ser utilizados pela execução de treino em curso. Pare o treino antes de os eliminar.",
      deleteArtifactsFailed:
        "A execução foi eliminada, mas não foi possível remover os respetivos ficheiros.",
      deleteArtifactsRetainedError:
        "Não foi possível remover os ficheiros do adaptador, pelo que a execução de treino foi mantida no histórico.",
      emptyDescription:
        "Ainda não há execuções de treino. Inicie a primeira execução de treino no separador Configurar.",
      loadError: "Não foi possível carregar as execuções de treino",
      deleteError:
        "Não foi possível eliminar a execução de treino. Tente novamente.",
      retry: "Tentar novamente",
      loadMore: "Carregar mais",
      loading: "A carregar...",
      loadingRun: "A carregar a execução de treino...",
      runNotFound: "Execução não encontrada",
      deleteTitle: "Eliminar a execução de treino?",
      deleteDescription:
        "Esta ação elimina permanentemente esta execução de treino e todas as respetivas métricas. Esta ação não pode ser anulada.",
      resumeTraining: "Retomar treino",
      resuming: "A retomar...",
      deleteRun: "Eliminar execução",
      loss: "Perda",
      steps: "Passos",
      lossTrendSparkline: "Minigráfico da tendência da perda",
      relativeJustNow: "agora mesmo",
      status: {
        completed: "Concluída",
        stopped: "Interrompida",
        error: "Erro",
        running: "Em execução",
        continued: "Continuada",
      },
      message: {
        completed: "Treino concluído",
        stopped: "Treino interrompido",
        running: "Treino em curso",
        errored: "O treino terminou com erro",
      },
      copyPreviewLink: "Copiar ligação de pré-visualização",
      previewLinkCopied: "Ligação de pré-visualização copiada",
      previewLinkCopyFailed: "Não foi possível copiar a ligação",
    },
    charts: {
      settings: "Definições do gráfico",
      settingsDescription:
        "Ajuste a apresentação do gráfico enquanto o treino continua.",
      openSettings: "Abrir as definições do gráfico",
      viewWindow: "Janela de visualização",
      viewWindowDescription:
        "Mostrar apenas os passos mais recentes ou todo o histórico.",
      window: "Janela",
      all: "Tudo",
      trainingLoss: "Perda de treino",
      trainingLossDescription: "Controle as sobreposições e a suavização EMA.",
      smoothing: "Suavização",
      smoothingDescription:
        "Mova para a direita para obter mais suavização. `0` = valores brutos.",
      showRawLoss: "Mostrar perda bruta",
      showSmoothedLoss: "Mostrar perda suavizada",
      showAverageLine: "Mostrar linha da média",
      scaleAndCleanup: "Escala e limpeza",
      linear: "Linear",
      log: "Logarítmica",
      noClip: "Sem corte",
      clipP99: "Cortar em p99",
      clipP95: "Cortar em p95",
      lossAxis: "Eixo da perda",
      gradientNormAxis: "Eixo da norma do gradiente",
      learningRateAxis: "Eixo da taxa de aprendizagem",
      resetDefaults: "Repor predefinições",
      loss: "Perda",
      smoothed: "Suavizada",
      evalLoss: "Perda de avaliação",
      learningRate: "Taxa de aprendizagem",
      lr: "LR",
      gradNorm: "Norma do gradiente",
      gradientNorm: "Norma do gradiente",
      step: "Passo {step}",
      averageValue: "média {value}",
      waitingForFirstEvaluationStep:
        "À espera do primeiro passo de avaliação...",
      evaluationNotConfigured: "Avaliação não configurada",
      evalChartWillAppear:
        "O gráfico será apresentado quando eval_steps for atingido",
      setEvalDatasetAndSteps:
        "Defina o conjunto de dados de avaliação e eval_steps para acompanhar a perda de avaliação",
    },
    progress: {
      title: "Progresso do treino",
      liveMetrics: "Métricas de treino em tempo real",
      exportGguf: "Exportar para GGUF",
      useInDecisionApi: "Utilizar na Decision API",
      decisionApiEnabled: "A Decision API passou a servir {name}.",
      decisionApiFailed:
        "Não foi possível mudar a Decision API para este modelo.",
      openConfig: "Abrir a configuração de treino",
      configLabel: "Configuração de treino",
      hyperparams: "Hiperparâmetros",
      epochs: "Épocas",
      batchSize: "Tamanho do lote",
      learningRate: "Taxa de aprendizagem",
      optimizer: "Otimizador",
      maxSteps: "Máximo de passos",
      contextLength: "Comprimento do contexto",
      warmupSteps: "Passos de aquecimento",
      rank: "Rank",
      alpha: "Alpha",
      dropout: "Dropout",
      variant: "Variante",
      epoch: "Época {value}",
      percentComplete: "{percent}% concluído",
      stepProgress: "Passo {current} / {total}",
      loss: "Perda",
      lr: "LR",
      gradNorm: "Norma do gradiente",
      project: "Projeto",
      model: "Modelo",
      method: "Método",
      elapsed: "Decorrido: {value}",
      eta: "Tempo restante: {value}",
      stepsPerSecond: "{value} passos/s",
      noStepsPerSecond: "-- passos/s",
      tokens: "Tokens: {value}",
      gpuMonitor: "Monitor da GPU",
      live: "Em direto",
      utilization: "Utilização",
      temperature: "Temperatura",
      vram: "VRAM",
      power: "Consumo",
      phase: {
        idle: "Inativo",
        downloadingModel: "A transferir o modelo",
        downloadingDataset: "A transferir o conjunto de dados",
        loadingModel: "A carregar o modelo",
        loadingDataset: "A carregar o conjunto de dados",
        configuring: "A configurar",
        training: "A treinar",
        finalizing: "A guardar o modelo",
        trainingDrafter: "A treinar o modelo de rascunho",
        completed: "Concluído",
        error: "Erro",
        stopped: "Interrompido",
      },
    },
    trainingStart: {
      ready: "Pronto",
      downloading: "A transferir",
      preparing: "A preparar",
      left: "faltam {eta}",
      downloaded: "{size} transferidos",
      terminalStart: "> o treino do unsloth está a começar...",
      preparingResources: "> A preparar o modelo e o conjunto de dados...",
      gettingReady: "> Estamos a preparar tudo para a sua execução...",
      resumingTraining: "A retomar o treino...",
      startingTraining: "a iniciar o treino...",
      dataset: "Conjunto de dados",
      datasetStreaming:
        "Conjunto de dados: em streaming (sem transferência completa)",
      modelWeights: "Pesos do modelo",
    },
  },
  modelMemory: {
    readout:
      "Pesos {model} + contexto {context} = {total} de {budget} de VRAM utilizável",
    readoutWithSpec:
      "Pesos {model} + KV {kv} + rascunho MTP {spec} = {total} de {budget} de VRAM utilizável",
    kvRate: "KV reservada, ~{rate}/token",
    oomLikely: "Com as definições atuais, é provável ocorrer OOM",
    tooLarge:
      "Maior do que a VRAM; parte será deslocada para a CPU. Uma quantização mais pequena é mais rápida",
  },
  skills: {
    title: "Competências",
    description:
      "As competências são detetadas nas pastas habituais de agentes. Ative-as aqui e escreva @ no chat para mencionar uma.",
    precedence: "~/.agents/skills tem precedência sobre ~/.claude/skills.",
    refresh: "Atualizar",
    empty:
      "Não foram encontradas competências. Adicione uma pasta com SKILL.md em ~/.agents/skills ou ~/.claude/skills e atualize.",
    sourceAgents: "Agentes",
    sourceClaude: "Claude",
    sourceBundled: "Incluídas",
    shadowed: "Sobreposta",
    invalid: "Inválida",
    compatibility: "Compatibilidade: {value}",
    shadowedBy: "Outra competência de {source} com este nome tem precedência.",
    enable: "Ativar {name}",
    disable: "Desativar {name}",
    updateError: "Não foi possível atualizar a competência",
    newSkill: "Nova competência",
    import: "Importar",
    delete: "Eliminar {name}",
    createDescription:
      "Guardada como SKILL.md numa nova pasta em ~/.agents/skills e ativada assim que existir.",
    nameLabel: "Nome",
    namePlaceholder: "p. ex., code-arithmetic",
    nameHint:
      "Letras minúsculas, números e hífenes simples. Este é também o nome da pasta.",
    nameInvalid:
      "Utilize 1-64 letras minúsculas, números ou hífenes simples, sem começar nem terminar com hífen.",
    descriptionLabel: "Descrição",
    descriptionPlaceholder:
      "Utilizar quando o utilizador pedir cálculos aritméticos. Calcular sempre com a ferramenta de código.",
    descriptionHint:
      "A linha que o modelo vê no seu catálogo de competências. É com base nela que decide quando ler a competência.",
    instructionsLabel: "Instruções",
    instructionsPlaceholder:
      "O que fazer quando esta competência se aplica, em Markdown.",
    instructionsHint:
      "O corpo de SKILL.md, lido na íntegra quando o modelo escolhe a competência.",
    create: "Criar competência",
    save: "Guardar alterações",
    created: "{name} criada",
    saved: "{name} guardada",
    deleted: "{name} eliminada",
    importedOne: "{count} competência adicionada",
    importedMany: "{count} competências adicionadas",
    importPartial: "{added} de {total} competências adicionadas",
    saveError: "Não foi possível guardar a competência",
    openError: "Não foi possível abrir a competência",
    deleteError: "Não foi possível eliminar a competência",
    importError: "Não foi possível importar a competência",
    exportError: "Não foi possível exportar a competência",
    deleteTitle: "Eliminar competência",
    deleteDescription:
      'Eliminar "{name}" e todo o conteúdo da respetiva pasta em ~/.agents/skills? Esta ação não pode ser anulada.',
    search: "Pesquisar competências por nome ou descrição…",
    noMatch: "Nenhuma competência corresponde a “{query}”",
    clearSearch: "Limpar pesquisa",
    readOnlyClaude:
      "Só de leitura: encontrada em ~/.claude/skills. Edite o ficheiro nessa localização.",
    readOnlyBundled: "Só de leitura: incluída com o Unsloth.",
    readOnlyLinked:
      "Só de leitura: esta entrada é uma ligação. Edite os ficheiros na sua localização original.",
    characters: "{count} carateres",
    unsaved: "alterações não guardadas",
    revert: "Reverter",
    enabledLabel: "Ativada",
    licenseLabel: "Licença",
    compatibilityLabel: "Compatibilidade",
    allowedToolsLabel: "Ferramentas permitidas",
    sectionAgents: "As suas competências",
    sectionAgentsHint:
      "Guardadas em ~/.agents/skills. Crie, edite e elimine-as aqui.",
    sectionClaude: "Competências do Claude",
    sectionClaudeHint:
      "Encontradas em ~/.claude/skills. Só de leitura aqui; edite os ficheiros nessa localização.",
    sectionBundled: "Incluídas",
    sectionBundledHint: "Fornecidas com o Unsloth. Só de leitura.",
    linked: "Ligação",
    discardTitle: "Descartar alterações?",
    discardDescription: "As edições não guardadas a {name} serão perdidas.",
    discard: "Descartar",
    mentions: "Competências",
    manage: "Gerir competências",
    turnOff: "Desativar competências",
    bulkActions: "Ações de competências",
    enableAll: "Ativar todas",
    disableAll: "Desativar todas",
    resetAll: "Repor predefinições",
    resetTitle: "Repor todas as competências?",
    resetDescription:
      "Todas as competências voltam ao estado de uma instalação nova: as suas competências e as competências do Claude ativadas, as competências incluídas desativadas. As suas escolhas de ativação e desativação são limpas.",
    reset: "Repor",
  },
  scheduledTasks: {
    title: "Agendadas",
    description:
      "Execute prompts de forma agendada com qualquer um dos seus modelos.",
    newTask: "Nova tarefa",
    upcoming: "Próximas",
    columns: {
      repeats: "Repetição",
      time: "Hora",
      nextRun: "Próxima execução",
    },
    monthlyDay: "Mensal, dia {day}",
    filterBy: "Filtrar por",
    filter: "Filtrar tarefas",
    active: "Ativas",
    noMatches: "Nenhuma tarefa agendada",
    search: "Pesquisar tarefas agendadas",
    newResult: "Novo resultado",
    sentBy: "Enviado por uma tarefa agendada",
    chatMark: "Chat de uma tarefa agendada",
    taskMissing: "Esta tarefa já não existe.",
    emptyTitle: "Ainda não há tarefas agendadas",
    emptyDescription:
      "Receba um resumo diário, um lembrete semanal ou qualquer outra coisa que pretenda fazer de forma agendada.",
    createFirst: "Criar uma tarefa",
    loadFailed: "Não foi possível carregar as tarefas agendadas.",
    retry: "Tentar novamente",
    paused: "Em pausa",
    completed: "Concluída",
    running: "Em execução",
    lastRun: "Última execução {when}",
    lastRunFailed: "A última execução falhou {when}: {reason}",
    openChat: "Abrir chat",
    runNow: "Executar agora",
    pause: "Pausar",
    resume: "Retomar",
    edit: "Editar",
    options: "Opções da tarefa",
    schedule: {
      once: "Uma vez, {date} às {time}",
      daily: "Todos os dias às {time}",
      weekdays: "Todos os dias úteis às {time}",
      weekly: "Todas as semanas ({days}) às {time}",
      monthly: "Todos os meses, no dia {day}, às {time}",
    },
    form: {
      createTitle: "Agendar uma tarefa",
      editTitle: "Editar tarefa agendada",
      name: "Nome",
      namePlaceholder: "Nome (opcional)",
      instructions: "Instruções",
      instructionsHint: "Enviadas ao modelo sempre que a tarefa é executada.",
      instructionsPlaceholder: "Instruções…",
      repeat: "Repetir",
      once: "Uma vez",
      daily: "Diariamente",
      weekdays: "Dias úteis",
      weekly: "Semanalmente",
      monthly: "Mensalmente",
      date: "Data",
      time: "Hora",
      days: "Dias",
      dayOfMonth: "Dia do mês",
      lastDayNote:
        "Nos meses sem este dia, a tarefa é executada no último dia do mês.",
      advanced: "Avançado",
      notifications: "Notificações",
      notifyAll: "Todas as execuções",
      notifyFailures: "Apenas falhas",
      notifyNone: "Desativadas",
      archiveSuccessful: "Arquivar execuções bem-sucedidas",
      project: "Projeto",
      projectNone: "Nenhum",
      model: "Modelo",
      modelLoaded: "O modelo que estiver carregado",
      modelOnDevice: "Neste dispositivo",
      modelNote:
        "Se este modelo não estiver carregado quando a tarefa for executada, o Unsloth pergunta primeiro.",
      modelNoteAllowed:
        "Carrega este modelo apenas enquanto estiver ausente do Unsloth e, em seguida, repõe o que tinha.",
      modelEmpty:
        "Transfira um modelo no Model hub ou adicione uma ligação para escolher outro.",
      modelMissing: "{model} (não está neste dispositivo)",
      allowLoad: "Mudar automaticamente quando estiver ausente",
      allowLoadHint:
        "Quando esta tarefa estiver agendada e o Unsloth não tiver sido utilizado durante 5 minutos, este modelo é carregado sem perguntar, a tarefa é executada e, em seguida, é reposto o modelo que tinha. Se estiver a utilizar o Unsloth ou a treinar nesse momento, a execução é ignorada e o seu modelo não é alterado. Se estiver desativado, o Unsloth pergunta antes de carregar.",
      reasoning: "Esforço de raciocínio",
      reasoningDefault: "Predefinição",
      reasoningOff: "Desativado",
      reasoningLowest: "Mínimo",
      reasoningHighest: "Máximo",
      create: "Criar",
    },
    deleteDialog: {
      title: "Eliminar a tarefa agendada?",
      description:
        "“{title}” deixará de ser executada. Os chats que já criou são mantidos.",
    },
    errors: {
      noModel:
        "Nenhum modelo estava carregado. Carregue um modelo no Chat e execute novamente a tarefa.",
      modelNotLoaded:
        "O modelo da tarefa não estava carregado. Carregue-o no Chat e execute novamente a tarefa.",
      modelLoadFailed:
        "Não foi possível carregar o modelo da tarefa. Confirme que continua transferido no Model hub.",
      modelBusy:
        "Ignorada enquanto estava a utilizar o Unsloth, pelo que o seu modelo não foi alterado.",
      connectionUnavailable:
        "A ligação da tarefa não existe ou está desativada. Verifique-a em Ligações.",
      timeout: "O modelo demorou demasiado tempo a responder.",
      cancelled: "A execução foi parada antes de terminar.",
      interrupted: "O Unsloth foi fechado antes de a execução terminar.",
      empty: "O modelo devolveu uma resposta vazia.",
      unknown: "A execução falhou.",
      saveFailed: "Não foi possível guardar a tarefa",
      updateFailed: "Não foi possível atualizar a tarefa",
      deleteFailed: "Não foi possível eliminar a tarefa",
      runFailed: "Não foi possível iniciar a tarefa",
    },
    loadPrompt: {
      title: "Carregar {model}?",
      description:
        "“{title}” é executada com {model}, que não está carregado. Ao executá-la, {model} fica carregado até terminar e, em seguida, é reposto o que tinha.",
      always:
        "Não perguntar novamente. Mudar apenas quando estiver ausente do Unsloth.",
      notNow: "Agora não",
      load: "Carregar e executar",
    },
    toast: {
      finished: "“{title}” está pronta",
      failed: "“{title}” falhou",
      many: "{count} tarefas agendadas concluídas",
      open: "Abrir",
      view: "Ver",
    },
    notificationBody:
      "A sua tarefa agendada foi concluída. Abra o Unsloth para ler a resposta.",
  },
  library: {
    tabs: {
      ariaLabel: "Secções da biblioteca",
      suggested: "Sugeridos",
      favorites: "Favoritos",
      folders: "Pastas",
      images: "Imagens",
      videos: "Vídeos",
      audio: "Áudio",
      models: "Modelos afinados",
      all: "Tudo",
      chats: "Chats",
    },
    chats: {
      sections: {
        chats: "Chats",
        projects: "Projetos",
        archived: "Arquivados",
        ariaLabel: "Secções de chats",
      },
      search: {
        chats: "Pesquisar chats",
        projects: "Pesquisar projetos",
        archived: "Pesquisar chats arquivados",
        project: 'Pesquisar em "{project}"',
        sections: "Pesquisar secções",
      },
      toolbar: {
        show: "Mostrar",
        pinned: "Afixados",
        forks: "Ramificações",
        comparisons: "Comparações",
        project: "Projeto",
        model: "Modelo",
        groupBy: "Agrupar por",
        groupNone: "Sem agrupamento",
        groupDate: "Data",
        groupProject: "Projeto",
        pinnedFirst: "Afixados primeiro",
        newChat: "Novo chat",
        newProject: "Novo projeto",
        sortChats: "Número de chats",
        withoutSection: "Sem secção",
      },
      groups: {
        today: "Hoje",
        yesterday: "Ontem",
        week: "Últimos 7 dias",
        month: "Últimos 30 dias",
      },
      list: {
        created: "Criado",
        lastActive: "Última atividade",
        location: "Localização",
        lastModified: "Última modificação",
        contents: "Conteúdo",
        oneMessage: "1 mensagem",
        messageCount: "{count} mensagens",
        showMore: "Mostrar mais ({count})",
      },
      badges: {
        pinned: "Afixado",
        fork: "Ramificação",
        compare: "Comparação",
      },
      menu: {
        fork: "Ramificar chat",
        moveTo: "Mover para projeto",
        newChatInProject: "Novo chat no projeto",
        edit: "Editar",
        deleteProject: "Eliminar projeto",
        newChatInSection: "Novo chat na secção",
      },
      project: {
        oneProject: "1 projeto",
        projectCount: "{count} projetos",
      },
      empty: {
        chatsTitle: "Ainda não há chats",
        chatsDescription:
          "As suas conversas aparecem aqui, separadas dos seus ficheiros.",
        archivedTitle: "Nada arquivado",
        archivedDescription: "Os chats que arquivar aparecem aqui.",
        sectionsTitle: "Ainda não há secções",
        sectionsDescription:
          "As secções agrupam chats e projetos na barra lateral da forma que preferir.",
        sectionTitle: "Ainda não há nada nesta secção",
        sectionDescription:
          "Mova chats para aqui a partir do respetivo menu ou inicie um novo chat nesta secção.",
      },
      dialog: {
        renameChat: "Mudar o nome do chat",
        deleteChatDescription:
          "Este chat será eliminado permanentemente. Esta ação não pode ser anulada.",
        deleteProjectDescription:
          "O projeto e os respetivos chats ({count}) serão eliminados permanentemente. Esta ação não pode ser anulada.",
        deleteFilesMany:
          "Remover também os ficheiros e as pastas da sandbox criados por estes chats.",
        deleteProjectFilesLabel:
          "Eliminar a pasta do espaço de trabalho do projeto",
        deleteProjectFiles:
          "A pasta do espaço de trabalho do projeto será removida do disco.",
      },
      toast: {
        projectMoved: "{project} movido para {section}",
        projectUnfiled: "{project} removido de {section}",
        forkFailed: "Não foi possível ramificar",
        projectDeleted: '"{name}" eliminado',
        projectDeleteFailed: "Não foi possível eliminar o projeto",
        removedFromSection: "Removidos da secção: {count}",
      },
      folder: {
        openChat: "Abrir pasta do chat",
        openChatTitle:
          "Abrir a pasta que as chamadas de ferramentas deste chat leem e escrevem",
        openProject: "Abrir pasta do projeto",
        openProjectTitle:
          "Abrir a pasta que os chats deste projeto leem e escrevem",
        chatHint:
          "Apenas a aplicação de ambiente de trabalho pode abrir a pasta de ficheiros de um chat. Num navegador, transfira o ficheiro a partir do resultado da ferramenta que o escreveu.",
        projectHint:
          "Apenas a aplicação de ambiente de trabalho pode abrir a pasta de um projeto. Num navegador, transfira os ficheiros a partir dos resultados das ferramentas que os escreveram.",
        manyFolders: "Este chat escreveu em mais do que uma pasta.",
        manyFoldersDescription:
          "Executou ferramentas antes e depois de uma mudança de localização, por isso abra a pasta a partir de um cartão de ferramenta.",
        chatFailed: "Não foi possível abrir a pasta do chat.",
        projectFailed: "Não foi possível abrir a pasta do projeto.",
        chatMissing: "Este chat ainda não tem pasta.",
        projectMissing: "Este projeto ainda não tem pasta.",
        missingDescription:
          "É criada na primeira vez que uma ferramenta escreve um ficheiro.",
        upload: "Carregar para a pasta do chat",
        uploadTitle:
          "Adicionar ficheiros à pasta que as chamadas de ferramentas deste chat leem e escrevem",
        uploaded: "{name} adicionado à pasta do chat",
        uploadedMany: "{count} ficheiros adicionados à pasta do chat",
        uploadedDescription:
          "As ferramentas deste chat já os podem abrir. Não são adicionados à conversa.",
        uploadExists: "{name} já está na pasta do chat",
        uploadExistsMany: "{count} ficheiros já estão na pasta do chat",
        replace: "Substituir",
        uploadFailed: "Não foi possível carregar {name}",
      },
    },
    empty: {
      suggestedTitle: "A sua biblioteca está vazia",
      suggestedDescription:
        "Os ficheiros que carregar ou criar nos chats aparecem aqui.",
      favoritesTitle: "Ainda não há favoritos",
      favoritesDescription:
        "Adicione ficheiros aos favoritos para os encontrar aqui rapidamente.",
      foldersTitle: "Crie a sua primeira pasta",
      foldersDescription:
        "Crie pastas para organizar os itens da sua biblioteca.",
      imagesTitle: "Ainda não há imagens",
      imagesDescription: "As imagens que carregar ou gerar aparecem aqui.",
      videosTitle: "Ainda não há vídeos",
      videosDescription: "Os vídeos que carregar ou gerar aparecem aqui.",
      audioTitle: "Ainda não há áudio",
      audioDescription:
        "A fala, a música e as faixas separadas que criar na página Áudio aparecem aqui.",
      modelsTitle: "Ainda não há modelos afinados",
      modelsDescription:
        "Os modelos que treinar ou exportar no Unsloth aparecem aqui.",
      createFolder: "Criar pasta",
      generateImage: "Gerar uma imagem",
      generateVideo: "Gerar um vídeo",
      generateAudio: "Gerar áudio",
      trainModel: "Treinar um modelo",
      uploadFiles: "Carregar ficheiros",
      dropHere: "ou largue ficheiros aqui",
      noMatchesTitle: "Sem correspondências",
      noMatchesDescription: "Experimente outra pesquisa ou limpe os filtros.",
      loadErrorTitle: "Não foi possível carregar a sua Biblioteca",
      loadErrorFallback: "Ocorreu um erro.",
      tryAgain: "Tentar novamente",
    },
    sections: {
      items: "Itens",
    },
    breadcrumb: "Trilho de navegação",
    searchLibrary: "Pesquisar na biblioteca",
    searchFolder: "Pesquisar na pasta",
    dropToUpload: "Largue para carregar",
    dropToUploadInto: 'Largue para carregar para "{folder}"',
    itemCountOne: "{count} item",
    itemCount: "{count} itens",
    selectItem: "Selecionar {name}",
    selection: {
      startChat: "Iniciar chat",
      move: "Mover",
      selectAll: "Selecionar tudo",
      countOfTotal: "{count} de {total} selecionados",
      clear: "Limpar seleção",
    },
    menu: {
      moreActions: "Mais ações",
      chatAboutThis: "Conversar sobre isto",
      chatWithModel: "Conversar com este modelo",
      editImage: "Editar imagem",
      addToFavorites: "Adicionar aos Favoritos",
      regenerateTitle: "Gerar novamente o título",
      removeFromFavorites: "Remover dos Favoritos",
      download: "Transferir",
      addToFolder: "Adicionar a uma pasta",
      newFolder: "Nova pasta",
      noFolder: "Biblioteca (sem pasta)",
      deleteFolder: "Eliminar pasta",
    },
    reveal: {
      finder: "Mostrar no Finder",
      explorer: "Mostrar no Explorador",
      files: "Mostrar na pasta",
    },
    toolbar: {
      filter: "Filtrar",
      source: "Origem",
      fileType: "Tipo de ficheiro",
      clearFilters: "Limpar filtros",
      workflow: "Criado com",
      uploaded: "Carregados",
      generated: "Gerados",
      documents: "Documentos",
      spreadsheets: "Folhas de cálculo",
      presentations: "Apresentações",
      pdfs: "PDFs",
      gridView: "Vista de grelha",
      listView: "Vista de lista",
      grid: "Grelha",
      list: "Lista",
      settings: "Definições da biblioteca",
      sort: "Ordenar",
      sortDefault: "Ordem predefinida",
      sortName: "Nome",
      sortModified: "Modificado",
      sortSize: "Tamanho",
      sortAscending: "Ascendente",
      sortDescending: "Descendente",
    },
    audio: {
      play: "Reproduzir {name}",
      pause: "Pausar {name}",
      playFailed: "Não foi possível reproduzir {name}",
      fromRun: "Da mesma execução",
      downloadRun: "Transferir tudo",
    },
    create: {
      note: "Nota",
      image: "Imagem",
      video: "Vídeo",
      audio: "Áudio",
      model: "Modelo",
      folder: "Pasta",
      untitledNote: "Nota sem título",
    },
    list: {
      selectAll: "Selecionar tudo",
      name: "Nome",
      lastActivity: "Última atividade",
      favorite: "Favorito",
      modified: "Modificado {time}",
      opened: "Aberto {time}",
      modifiedColumn: "Modificado",
      justNow: "agora mesmo",
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
      fullFineTune: "Afinação completa",
      loraExport: "Exportação LoRA",
      mergedExport: "Exportação combinada",
      ggufExport: "Exportação GGUF",
      model: "Modelo",
    },
    dialog: {
      create: "Criar",
      renameFolder: "Mudar o nome da pasta",
      renameFile: "Mudar o nome do ficheiro",
      deleteTitle: 'Eliminar "{name}"?',
      deleteManyTitle: "Eliminar {count} itens?",
      deleteManyDescription:
        "Os ficheiros são eliminados da localização onde se encontram. O conteúdo de uma pasta eliminada sobe um nível.",
      deleteFolderIntoParent:
        'Todo o conteúdo é movido para "{folder}". Nenhum ficheiro é eliminado.',
      deleteFolderIntoLibrary:
        "Todo o conteúdo é movido para a sua Biblioteca. Nenhum ficheiro é eliminado.",
      deleteUpload: "Esta ação elimina permanentemente o ficheiro.",
      deleteAttachment: "Também é removido do chat ao qual foi anexado.",
      deleteImage: "Também é removida da sua galeria de Imagens.",
      deleteVideo: "Também é removido da sua galeria de Vídeo.",
      deleteAudio: "Também é removido da sua galeria de Áudio.",
      deleteSandbox: "Também é removido do chat que o criou.",
      deleteModel: "Esta ação elimina permanentemente o modelo do disco.",
      unsavedTitle: "As suas alterações não foram guardadas",
      unsavedDescription:
        "{reason} Tente novamente ou descarte as alterações para fechar o ficheiro.",
      keepEditing: "Continuar a editar",
      discardChanges: "Descartar alterações",
    },
    preview: {
      code: "Código",
      preview: "Pré-visualização",
      fromChat: "Do chat",
      viewOriginalChat: "Ver o chat original",
      viewInImages: "Ver em Imagens",
      viewInVideo: "Ver em Vídeo",
      viewInAudio: "Ver em Áudio",
      viewOriginalProject: "Ver o projeto original",
      viewTrainingRun: "Ver a execução de treino",
      sheetTruncated: "A mostrar parte desta folha",
      documentTruncated: "A mostrar o início deste documento",
      emptyDocument: "Este ficheiro está vazio.",
      cannotPreview:
        "Não é possível pré-visualizar este ficheiro aqui. Transfira-o para o abrir.",
      noPreview: "Sem pré-visualização para este tipo de ficheiro.",
      tooLargeToPreview:
        "Este ficheiro é demasiado grande para ser pré-visualizado aqui. Transfira-o para o abrir.",
      startWriting: "Comece a escrever…",
      type: "Tipo",
      baseModel: "Modelo base",
      size: "Tamanho",
      location: "Localização",
      unknown: "Desconhecido",
      readOnlyUtf16:
        "Este ficheiro não é texto UTF-16 válido, pelo que é aberto aqui só de leitura.",
      readOnlyNotUtf8:
        "Este ficheiro não é texto UTF-8, pelo que é aberto aqui só de leitura. Alguns carateres podem não ser apresentados corretamente.",
    },
    viewer: {
      scale: "Escala",
      fit: "Ajustar à janela",
      moreActionsImage: "Mais ações para esta imagem",
      moreActionsVideo: "Mais ações para este vídeo",
      moreActionsClip: "Mais ações para este clip",
      moreActionsFile: "Mais ações para este ficheiro",
      openImage: "Abrir imagem",
      openImageNamed: "Abrir imagem: {prompt}",
      openVideo: "Abrir vídeo",
      untitledImage: "Imagem sem título",
      untitledVideo: "Vídeo sem título",
    },
    project: {
      label: "Projeto",
      newProject: "Novo projeto",
      noProjects: "Ainda não há projetos",
      alreadyIn: "Já está em {project}",
      addedTo: "Adicionado a {project}",
      createAndAdd: "Criar e adicionar",
      addFailedImage: "Não foi possível adicionar a imagem ao projeto",
      addFailedVideo: "Não foi possível adicionar o vídeo ao projeto",
      addFailedClip: "Não foi possível adicionar o clip ao projeto",
      addFailedFile: "Não foi possível adicionar o ficheiro ao projeto",
      newProjectImage: "Adicionar imagem a um novo projeto",
      newProjectVideo: "Adicionar vídeo a um novo projeto",
      newProjectClip: "Adicionar clip a um novo projeto",
      newProjectFile: "Adicionar ficheiro a um novo projeto",
    },
    toast: {
      missingItem: "Esse ficheiro já não está na Biblioteca",
      moveFailed: "Não foi possível mover",
      movedToFolder: "Movido para {folder}",
      movedToLibrary: "Movido para a Biblioteca",
      folderFallback: "pasta",
      favoritesFailed: "Não foi possível atualizar os favoritos",
      addedToFavorites: "Adicionado aos Favoritos",
      removedFromFavorites: "Removido dos Favoritos",
      uploadingOne: "A carregar {name}…",
      uploadingMany: "A carregar {count} ficheiros…",
      uploadedOne: "Ficheiro carregado",
      uploadedMany: "{count} ficheiros carregados",
      uploadFailed: "O carregamento falhou",
      uploadTooLarge: "{name} tem mais de 512 MB.",
      readDropsFailed: "Não foi possível ler os ficheiros largados",
      createNoteFailed: "Não foi possível criar a nota",
      renameFailed: "Não foi possível mudar o nome",
      createFolderFailed: "Não foi possível criar a pasta",
      signedOutBeforeFolder:
        "A sessão foi terminada antes de a pasta ser criada.",
      signedOutBeforeSave:
        "A sessão foi terminada antes de a alteração ser guardada.",
      signedOutBeforeUpload:
        "A sessão foi terminada antes de o carregamento terminar.",
      deleteFailed: "Não foi possível eliminar alguns itens",
      saveFailed: "Não foi possível guardar",
      saveNoteFirst: "Guarde primeiro a nota.",
      revealFailed: "Não foi possível abrir o gestor de ficheiros",
      downloadFailed: "Não foi possível transferir {name}",
      downloadManyFailed: "Não foi possível transferir os ficheiros",
      downloadingMany: "A transferir {count} ficheiros",
      downloadingManyDescription:
        "Se o navegador pedir, permita que este site transfira vários ficheiros.",
      preparingMany: "A preparar {count} ficheiros…",
      zipFileName: "Ficheiros da biblioteca",
      nothingToChat: "Ainda não há nada sobre o qual conversar",
      emptyFolder: "Esta pasta não tem ficheiros.",
      tooLargeOne: "{name} é demasiado grande para ser anexado",
      tooLargeMany: "Estes ficheiros são demasiado grandes para serem anexados",
      attachedOne: "{count} ficheiro anexado",
      attachedMany: "{count} ficheiros anexados",
      skippedTooLarge: "{count} demasiado grandes para anexar.",
      skippedOverLimit: "Mais {count} acima do limite de {limit} ficheiros.",
      chatFilesWaiting:
        "{count} serão anexados quando for carregado um modelo que os consiga ler.",
      openFilesFailed: "Não foi possível abrir os ficheiros",
      editImageFailed: "Não foi possível abrir a imagem para edição",
      speechModel: "{name} é um modelo de voz",
      speechModelDescription: "Escolha-o no menu de modelos da página Áudio.",
      noMediaLink:
        "O servidor não devolveu nenhuma ligação para reproduzir este ficheiro.",
      imageNotFound: "Não foi possível encontrar esta imagem",
      clipNotFound: "Não foi possível encontrar este clip",
      notFoundDescription: "Pode ter sido arquivado ou eliminado.",
      imageTooLarge: "Esta imagem é demasiado grande para ser anexada",
      videoTooLarge: "Este vídeo é demasiado grande para ser anexado",
      attachmentLimit: "Os anexos do chat estão limitados a {size} MB.",
      attachingImage: "A anexar a imagem…",
      attachingVideo: "A anexar o vídeo…",
      readImageFailed: "Não foi possível ler a imagem ({status}).",
      readVideoFailed: "Não foi possível ler o vídeo ({status}).",
      openFileFailed: "Não foi possível abrir o ficheiro",
    },
  },
  exportDecision: {
    title: "Modelo de decisão",
    description:
      "Os modelos de decisão {layout} são exportados para GGUF para o servidor de decisão do llama.cpp. Escolha uma ou mais quantizações.",
    adapterNote:
      "Este checkpoint contém adaptadores LoRA; são combinados antes da conversão.",
    notEligibleTitle: "Exportação GGUF indisponível",
    ggufOnly: "Os modelos de decisão só podem ser exportados para GGUF",
    existing: "Já exportado: {quantizations}",
    outputNote: "Os ficheiros GGUF são guardados na pasta da execução: {path}",
    methodLabel: "GGUF de modelo de decisão",
    outputLabel: "GGUF de decisão ({quantizations})",
  },
  decisions: {
    title: "Experimentar uma decisão",
    description: "Faça uma pergunta ao modelo de decisão sobre qualquer texto.",
    tryIt: "Experimentar",
    tryItOff: "Ative Servir pedidos para experimentar.",
    typeNoul: "Sim / não",
    typeChoice: "Escolha",
    typeScore: "Pontuação",
    aboutNoul: "Responde sim ou não, com a probabilidade de sim.",
    aboutChoice:
      "Escolhe uma opção de uma lista, com uma probabilidade para cada uma.",
    aboutScore:
      "Classifica o estado numa escala de níveis, do mais baixo para o mais alto.",
    model: "Modelo de decisão",
    defaultModel: "Predefinido ({model})",
    layaModel: "Laya {model}",
    thisMachine: "Este computador",
    modelReady: "Pronto",
    modelLoading: "A carregar",
    modelNotLoaded: "É carregado na primeira execução",
    connectionNotDefault:
      "Defina-o primeiro como o modelo da Decision API em Definições > API.",
    question: "Pergunta",
    questionHelp: "O que o modelo deve decidir.",
    result: "Resultado",
    form: "Formulário",
    json: "JSON",
    preview: "Pré-visualização",
    state: "Estado",
    stateHelp: "O texto sobre o qual decidir. Texto simples ou JSON.",
    criteria: "Critérios",
    criteriaNoulHelp: "Opcional. O que conta como sim e o que conta como não.",
    criteriaChoiceHelp:
      "As opções entre as quais escolher, cada uma com uma descrição breve.",
    criteriaScoreHelp: "Os níveis da escala, do mais baixo para o mais alto.",
    yesWhen: "Sim quando",
    noWhen: "Não quando",
    optionName: "Opção",
    optionDescription: "Descrição",
    optionNamePlaceholder: "nome",
    optionDescriptionPlaceholder: "Quando a escolher",
    optionNameLabel: "Nome da opção {n}",
    optionDescriptionLabel: "Descrição da opção {n}",
    removeOption: "Remover a opção {n}",
    addOption: "Adicionar opção",
    levelLabel: "Nível {n}",
    removeLevel: "Remover o nível {n}",
    addLevel: "Adicionar nível",
    jsonHelp: "O corpo do pedido. Executar envia-o tal como está escrito.",
    examples: "Exemplos",
    presetToolApproval: "Aprovar chamadas de ferramentas de agentes",
    presetPromptInjection: "Detetar injeção de prompts",
    presetTicketRouting: "Encaminhar pedidos de suporte",
    presetModeration: "Moderar mensagens",
    presetIntent: "Detetar intenção",
    presetGrading: "Avaliar uma resposta",
    run: "Executar decisão",
    running: "Em execução",
    answer: "Resposta",
    yes: "Sim",
    no: "Não",
    scoreDetail: "Pontuação {score} numa escala de 0 a {max}",
    latency: "{ms} ms",
    inputTokens: "{count} tokens de entrada",
    responseTitle: "Resposta",
    emptyTitle: "Ainda não há resultados",
    emptyBody: "Prima Executar decisão ou {shortcut}.",
    cantRun: "Não foi possível executar",
    failed: "Erro {status}",
    invalidJson: "O corpo do pedido não é JSON válido.",
  },
  floatingPanels: {
    loadedModels: {
      title: "Modelos carregados",
      countOne: "{count} modelo carregado",
      countMany: "{count} modelos carregados",
      pillLabel: "{count}. Mostrar detalhes ou arrastar para mover",
      dragToMove: "Arraste para mover",
      dragHandleLabel: "Arraste para mover",
      collapse: "Fechar",
      collapseLabel: "Reduzir os modelos carregados",
      close: "Fechar",
      closeLabel: "Fechar os modelos carregados",
      backOnNextLoad: "Volta a aparecer no próximo carregamento de modelo",
      ejectToFree: "Ejetar para libertar memória",
      ejectLabel: "Ejetar {model}",
      openTarget: "Abrir {target}",
      rowLabel: "{model}. Abrir {target}",
      loading: "A carregar",
      ejecting: "A ejetar",
      stillInMemory: "Ainda em memória",
      alsoServing: "Também a servir",
      noneLoaded: "Nenhum modelo carregado",
      noneLoadedDetail: "Os modelos que carregar aparecem aqui.",
      otherAccount: "Em utilização por outra conta",
      otherAccountDetail:
        "A memória está ocupada por um modelo que não consegue ver",
      toastStillResident:
        '"{model}" foi carregado durante a ejeção, pelo que continua a utilizar memória. Ejete-o novamente para a libertar.',
      toastUnverified:
        "Foi pedido a {model} que descarregasse, mas o respetivo runtime não o confirmou. Verifique o cartão daqui a pouco.",
      toastAlreadyFree: "{model} já não estava carregado.",
      toastReplaced:
        '{model} já não está carregado. "{resident}" ocupou o seu lugar e não foi alterado.',
      toastEjected: "{model} ejetado",
      toastFailed: "Não foi possível ejetar {model}",
    },
    monitor: {
      dragHandleLabel:
        "Mover o monitor em tempo real. As teclas de seta deslocam-no ligeiramente; duplo clique repõe-no.",
      reconnecting: "A restabelecer a ligação...",
    },
  },
  projectsPage: {
    title: "Projetos",
    search: "Pesquisar projetos",
    importExport: "Importar / exportar projetos",
    importChatsAction: "Importar chats…",
    exportAllProjects: "Exportar todos os projetos",
    exportProjectsAndRecents: "Exportar projetos + recentes",
    combined: "Combinado",
    perChat: "Por chat",
    newProject: "Novo projeto",
    name: "Nome",
    updated: "Atualizado",
    newestFirst: "Mais recentes primeiro",
    oldestFirst: "Mais antigos primeiro",
    today: "Hoje",
    yesterday: "Ontem",
    emptyDescription:
      "Um projeto mantém chats relacionados juntos, com instruções e fontes partilhadas que todos os chats do projeto podem utilizar.",
    createFirst: "Crie o seu primeiro projeto",
    noMatches: "Nenhum projeto corresponde à pesquisa.",
    clearSearch: "Limpar pesquisa",
    showChats: "Mostrar chats",
    hideChats: "Ocultar chats",
    pinProject: "Afixar projeto",
    unpinProject: "Desafixar projeto",
    newChatInProject: "Novo chat no projeto",
    projectOptions: "Opções do projeto",
    loadChatsFailed: "Não foi possível carregar os chats. Tentar novamente",
    refreshChatsFailed: "Não foi possível atualizar os chats. Tentar novamente",
    noChats: "Sem chats",
    unread: "Não lido",
    pinChat: "Afixar chat",
    unpinChat: "Desafixar chat",
    chatOptions: "Opções do chat",
    exportAllChats: "Exportar todos os chats…",
    archive: "Arquivar",
    renameChat: "Mudar o nome do chat",
    chatName: "Nome do chat",
    deleteChat: "Eliminar chat",
    deleteChatDescription:
      "Tem a certeza de que pretende eliminar {name}? As respetivas mensagens serão eliminadas permanentemente.",
    importChats: "Importar chats",
    importDestination: "Escolha para onde importar {name}:",
    selectDestination: "Selecionar destino",
    recents: "Recentes",
    import: "Importar",
    deleteProject: "Eliminar projeto",
    deleteProjectDescription:
      "Tem a certeza de que pretende eliminar {name}? Os respetivos chats serão eliminados permanentemente.",
    deleteProjectFiles:
      "A pasta do espaço de trabalho do projeto será removida do disco.",
    deleteAll: "Eliminar tudo",
    projectFallback: "projeto",
    importFailed: "A importação falhou.",
    nothingToExport: "Não há conversas para exportar.",
    exportFailed: "A exportação falhou.",
    renameChatFailed: "Não foi possível mudar o nome do chat",
    archiveChatFailed: "Não foi possível arquivar o chat",
    deleteChatFailed: "Não foi possível eliminar o chat",
    deleteProjectFailed: "Não foi possível eliminar o projeto",
  },
  auth: {
    welcomeBack: "Bem-vindo de volta",
    loginSubtitle: "Inicie sessão com a sua palavra-passe.",
    setupTitle: "Configure a sua conta",
    setupSubtitle: "Crie uma nova palavra-passe",
    login: "Iniciar sessão",
    changePassword: "Alterar palavra-passe",
    alreadySetUp: "Já configurou a palavra-passe? ",
    backToLogin: "Voltar ao início de sessão",
    username: "Nome de utilizador",
    password: "Palavra-passe",
    currentPassword: "Palavra-passe atual",
    newPassword: "Nova palavra-passe",
    confirmPassword: "Confirmar palavra-passe",
    setupCodeHint:
      "Primeiro início de sessão? Cole como palavra-passe o código de configuração que o administrador lhe deu.",
    ownerHint:
      "É o proprietário da instalação? Inicie sessão como unsloth ou com o nome de início de sessão apresentado em Definições > Contas.",
    currentSetupCodeHint:
      "Cole como palavra-passe atual o código de configuração com que acabou de iniciar sessão.",
    minLength: "Tem de ter pelo menos 8 carateres.",
    ensureMatch: "Certifique-se de que as palavras-passe coincidem.",
    noSpaces: "A nova palavra-passe não pode conter espaços.",
    currentTooShort: "A palavra-passe atual tem de ter pelo menos 8 carateres.",
    setupInitFailed:
      "Não foi possível inicializar a configuração. Recarregue a página e tente novamente.",
    newTooShort: "A nova palavra-passe tem de ter pelo menos 8 carateres.",
    mismatch: "As palavras-passe não coincidem.",
    mustDiffer:
      "A nova palavra-passe tem de ser diferente da palavra-passe atual.",
    pleaseWait: "Aguarde...",
    continueAsOwner: "Continuar como proprietário",
    signInWithUsername:
      "Inicie sessão com o seu nome de utilizador e palavra-passe.",
    bootstrapping:
      "A autenticação ainda está a inicializar a conta de administrador predefinida.",
    signInToChange:
      "Inicie sessão uma vez com as credenciais iniciais para alterar a palavra-passe.",
    alreadyUpdated:
      "A palavra-passe já foi atualizada. Utilize o ecrã de início de sessão.",
    busy: "O Unsloth está ocupado. Tente novamente daqui a pouco.",
    updateFailed: "Não foi possível atualizar a palavra-passe.",
    loadFailed: "Não foi possível carregar.",
    failed: "A autenticação falhou.",
    shuttingDown:
      "Esta instância está a encerrar: estava acessível na rede e a palavra-passe predefinida nunca foi alterada.",
    shutdownCountdown:
      "Esta instância está acessível na rede e continua a utilizar a palavra-passe predefinida, pelo que será encerrada dentro de {time}. Definir aqui uma palavra-passe mantém-na em execução.",
  },
  routes: {
    chat: "Chat",
    api: "API",
    dataRecipes: "Data Recipes",
    login: "Iniciar sessão",
    changePassword: "Alterar palavra-passe",
  },
  remoteCode: {
    unsafeTitle: "Ficheiros não seguros detetados",
    blockedTitle: "Código personalizado bloqueado",
    enableTitle: "Ativar código personalizado para este modelo?",
    thisModel: "Este modelo",
    unsafeBody:
      "{model} contém ficheiros que a análise de segurança do Hugging Face assinalou como não seguros (por exemplo, um pickle malicioso que executaria código quando o modelo fosse carregado). Não pode ser carregado. Os ficheiros assinalados nunca foram transferidos.",
    unsafeBodyFromProvider:
      "{model} de {provider} contém ficheiros que a análise de segurança do Hugging Face assinalou como não seguros (por exemplo, um pickle malicioso que executaria código quando o modelo fosse carregado). Não pode ser carregado. Os ficheiros assinalados nunca foram transferidos.",
    customCode:
      "{model} declara código Python personalizado no respetivo repositório. {note}",
    customCodeFromProvider:
      "{model} de {provider} declara código Python personalizado no respetivo repositório. {note}",
    blockedNote:
      "Uma análise de segurança assinalou problemas CRÍTICOS, pelo que não pode ser ativado.",
    reviewNote:
      "Reveja a análise de segurança abaixo. Continue apenas se confiar na origem do modelo.",
    trustNote: "Continue apenas se confiar na origem do modelo.",
    flaggedIntro:
      "O nosso analisador automático assinalou problemas, incluindo:",
    nothingFlagged:
      "O nosso analisador automático não assinalou nenhum ficheiro preocupante, mas verifique novamente.",
    enableAndContinue: "Ativar e continuar",
  },
  tour: {
    badge: "visita guiada",
    defaultTitle: "Visita rápida",
    defaultBody: "Vamos ajudá-lo a orientar-se.",
    skipTour: "Ignorar visita",
    skip: "Ignorar",
    back: "Voltar",
    next: "Seguinte",
    tip: "Sugestão: `Esc` ignora a visita. A visita bloqueia os cliques para que possa ler.",
    readMore: "Saber mais",
  },
  updateNotes: {
    openRelease: "Abrir versão",
    openChangelog: "Abrir registo de alterações",
    retry: "Tentar novamente",
    loadingChanges: "A carregar as novas alterações...",
    customLlamaRepo:
      "Esta instalação segue um repositório personalizado do llama.cpp, pelo que não é possível comparar as respetivas alterações.",
    noItemisedNotes:
      "Esta compilação é anterior às notas de versão detalhadas, pelo que não é possível comparar as respetivas alterações.",
    compareFailed: "Não foi possível comparar estas versões.",
    llamaChangesLabel: "Novas alterações do llama.cpp de {from} para {to}",
    showingChanges: "A mostrar {shown} de {total} alterações",
    noCarriedChanges:
      "Não estão listadas novas alterações incluídas para esta compilação.",
    notesFor: "Notas de versão de {release}",
    notesForVersion: "Notas de versão da versão {version}",
    notesTruncated: "Notas truncadas. Consulte as notas de versão completas.",
    summaryLabel: "Resumo das notas de versão",
    moreCount: "+{count} mais",
    loadingNotes: "A carregar as notas de versão...",
    loadFailed: "Não foi possível carregar as notas de versão.",
    noNotesFor: "Ainda não foram publicadas notas de versão para {release}.",
    noNotes: "Ainda não foram publicadas notas de versão.",
  },
  desktopStartup: {
    checking: "A verificar...",
    installPrompt: "Para instalar o Unsloth, clique em Começar.",
    getStarted: "Começar",
    chatOnly: "Apenas chat (GGUF): instalação mais pequena, sem treino",
    installLocation: "Localização da instalação:",
    changeLocation: "Alterar",
    useDefaultLocation: "Utilizar predefinição",
    longPaths: {
      title: "O Windows irá pedir permissão",
      approve:
        "Aprove {tool} uma vez para ativar caminhos de ficheiro longos para os kernels compilados.",
      waiting: "À espera da permissão do Windows...",
    },
    skip: "Ignorar",
    continue: "Continuar",
    install: {
      preparing: "A preparar o seu espaço de trabalho...",
      preparingDetail: "A verificar o que este computador necessita.",
      downloading: "A transferir os componentes necessários...",
      downloadingDetail: "A criar o seu ambiente local.",
      installing: "A instalar o Unsloth...",
      installingDetail: "A configurar as ferramentas de IA locais...",
      finishing: "A concluir a configuração...",
      finishingDetail: "A preparar tudo.",
      continuesAutomatically: "A configuração continuará automaticamente.",
      fewMinutes: "Isto pode demorar alguns minutos.",
      someTakeLonger: "Alguns componentes demoram mais tempo a configurar.",
    },
    details: {
      showInstall: "Mostrar detalhes da instalação",
      hideInstall: "Ocultar detalhes da instalação",
      showSetup: "Mostrar detalhes da configuração",
      hideSetup: "Ocultar detalhes da configuração",
      showUpdate: "Mostrar detalhes da atualização",
      hideUpdate: "Ocultar detalhes da atualização",
    },
    repair: {
      upgrading: "A atualizar para a instalação completa...",
      gettingReady: "A preparar tudo...",
      upgradingDetail:
        "A transferir o PyTorch e as respetivas bibliotecas para a GPU deste computador. Isto pode demorar vários minutos.",
      wontTakeLong: "Não vai demorar muito.",
      upgradeFailed: "Não foi possível atualizar para a instalação completa",
      updateFailed: "A atualização falhou",
    },
    closing: "A fechar o Unsloth Desktop...",
    closingDetail: "A encerrar o backend.",
    diskFull: "Espaço em disco insuficiente",
    setupFailed: "A configuração encontrou um problema",
    tryAgain: "Tentar novamente",
    retry: "Tentar novamente",
    elevation: {
      title: "Permissão necessária",
      packages: "É necessário instalar os seguintes pacotes do sistema:",
      allow: "Permitir",
    },
    starting: "A iniciar o Unsloth...",
    loadingProjects: "A carregar os projetos...",
    loadingModels: "A carregar os modelos...",
    nearlyDone: "Quase concluído...",
    finishingUpdate: "A concluir a atualização...",
    serverStopped: "Servidor parado",
    startServer: "Iniciar servidor",
    somethingWrong: "Ocorreu um erro",
    diagnostics: {
      copy: "Copiar diagnóstico",
      copying: "A copiar...",
      clipboardFailed:
        "Não foi possível copiar para a área de transferência. Selecione e copie o diagnóstico abaixo.",
      copyFailed: "Não foi possível copiar o diagnóstico: {error}",
    },
  },
  desktopUpdate: {
    updatingBackend: "A atualizar o backend...",
    downloadingApp: "A transferir a atualização da aplicação...",
    installing: "A instalar a atualização...",
    failed: "A atualização falhou",
    updating: "A atualizar...",
    updatingBackendDetail:
      "Isto pode demorar alguns minutos. Não feche a aplicação.",
    percentDownloaded: "{percent}% transferido",
    restartShortly: "A aplicação será reiniciada em breve.",
    failedDetail: "Ocorreu um erro durante a atualização.",
    downloadingFile: "A transferir {file}",
    downloadingAppLabel: "A transferir a atualização da aplicação",
    skipRestart: "Ignorar e reiniciar",
    banner: {
      dismiss: "Dispensar a notificação de atualização da aplicação",
      failedTitle: "A atualização da aplicação falhou",
      newVersion: "Nova versão do Unsloth",
      backendRecovered:
        "O backend foi recuperado. O diagnóstico continua disponível.",
      linuxManual:
        "Abra a página da versão no GitHub para instalar o pacote Linux",
      available: "Está disponível uma nova atualização da aplicação",
      showNotes: "Mostrar notas de versão",
      hideNotes: "Ocultar notas de versão",
      copyDiagnostics: "Copiar diagnóstico",
      later: "Mais tarde",
      remindLater: "Lembrar mais tarde",
      openReleasePage: "Abrir página da versão",
      retry: "Tentar atualizar novamente",
      update: "Atualizar",
    },
  },
  desktopShell: {
    navigation: "Barra lateral e navegação de páginas",
    collapseSidebar: "Fechar a barra lateral",
    expandSidebar: "Expandir a barra lateral",
    goBack: "Retroceder",
    goForward: "Avançar",
    titlebar: "Barra de título da janela",
    windowControls: "Controlos da janela",
    minimize: "Minimizar a janela",
    maximize: "Maximizar a janela",
    restore: "Restaurar a janela",
    close: "Fechar a janela",
  },
  runtimeUpdate: {
    installAudio: "Instalar o runtime de áudio",
    installingAudio: "A instalar o runtime de áudio...",
    unknownVersion: "desconhecida",
    failed: "A atualização falhou: {error}",
    unknownError: "erro desconhecido",
    skipVersions: "Ignorar estas versões",
    skipChange: "Ignorar esta alteração",
    skipVersion: "Ignorar esta versão",
    skippedVersion: "{component} {version} ignorado por agora",
    skippedUpdate: "Atualização de {component} ignorada por agora",
    skippedUpdates: "Atualizações de {components} ignoradas por agora",
    vulkanFaster:
      "O Vulkan é >10% mais rápido do que o ROCM. Não é necessário reiniciar",
    noRestart: "Não é necessário reiniciar",
    downloadSize: "Transferência de {size}",
    backendChange: "Backend {change}",
    updating: "A atualizar {component}...",
    updatingLabel: "A atualizar {component}",
    installingPhase: "A instalar {component} ({current} de {total})...",
    installingCount: "A instalar {count} atualizações...",
    bundleTitle: "{count} atualizações de runtime",
    newBackend: "Novo backend {component}",
    newUpdate: "Nova atualização de {component}",
    hideUpdate:
      "Ocultar a atualização de {component} até o Unsloth ser reaberto",
    hideUpdates:
      "Ocultar as atualizações de {components} até o Unsloth ser reaberto",
    whatsNew: "Novidades",
    hideWhatsNew: "Ocultar novidades",
    later: "Mais tarde",
    update: "Atualizar",
    undo: "Anular",
  },
} satisfies DeepPartialMessageTree<typeof en>;
