// SPDX-License-Identifier: MIT

/**
 * Type definitions for i18n translation keys
 * This provides IntelliSense support and catches typos at compile time
 */

export type TranslationKey<Namespace extends string = string> = `${Namespace}:${string}` | string;

// Define your namespaces
export type Namespace = 
  | 'common'
  | 'nav'
  | 'auth'
  | 'chat'
  | 'studio'
  | 'training'
  | 'dataset'
  | 'export'
  | 'tour'
  | 'errors'
  | 'vram'
  | 'update';

// Common usage helper
export const i18nKeys = {
  // Navigation
  nav: {
    studio: "nav.studio" as const,
    recipes: "nav.recipes" as const,
    export: "nav.export" as const,
    chat: "nav.chat" as const,
    tour: "nav.tour" as const,
    update: "nav.update" as const,
    learnMore: "nav.learnMore" as const,
    docs: "nav.docs" as const,
    menu: "nav.menu" as const,
    navigate: "nav.navigate" as const,
    theme: "nav.theme" as const,
    quit: "nav.quit" as const,
  },

  // Auth
  auth: {
    welcomeBack: "auth.welcomeBack" as const,
    setupAccount: "auth.setupAccount" as const,
    login: "auth.login" as const,
    changePassword: "auth.changePassword" as const,
    password: "auth.password" as const,
    newPassword: "auth.newPassword" as const,
    confirmPassword: "auth.confirmPassword" as const,
    backToLogin: "auth.backToLogin" as const,
  },

  // Chat
  chat: {
    title: "chat.title" as const,
    settings: "chat.settings" as const,
    model: "chat.model" as const,
    sampling: "chat.sampling" as const,
    temperature: "chat.temperature" as const,
    tools: "chat.tools" as const,
    preferences: "chat.preferences" as const,
    systemPrompt: "chat.systemPrompt" as const,
    contextLength: "chat.contextLength" as const,
    savePreset: "chat.savePreset" as const,
    presetName: "chat.presetName" as const,
  },

  // Studio
  studio: {
    title: "studio.title" as const,
    configure: "studio.configure" as const,
    currentRun: "studio.currentRun" as const,
    history: "studio.history" as const,
    configureTraining: "studio.configureTraining" as const,
    trainingInProgress: "studio.trainingInProgress" as const,
    loadingRuntime: "studio.loadingRuntime" as const,
  },

  // Training
  training: {
    modelSelection: "training.modelSelection" as const,
    dataset: "training.dataset" as const,
    parameters: "training.parameters" as const,
    startTraining: "training.startTraining" as const,
    stopTraining: "training.stopTraining" as const,
  },

  // Common
  common: {
    save: "common.save" as const,
    cancel: "common.cancel" as const,
    delete: "common.delete" as const,
    edit: "common.edit" as const,
    loading: "common.loading" as const,
    confirm: "common.confirm" as const,
    close: "common.close" as const,
    toggleTheme: "common.toggleTheme" as const,
    off: "common.off" as const,
    max: "common.max" as const,
  },

  // Errors
  errors: {
    loginFailed: "errors.loginFailed" as const,
    authFailed: "errors.authFailed" as const,
    modelLoadFailed: "errors.modelLoadFailed" as const,
    trainingFailed: "errors.trainingFailed" as const,
  },
};
