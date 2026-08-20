import { getProductCapability } from "./platform-capabilities";

/**
 * Route-level compatibility exports. The product capability registry is the
 * single source of truth; legacy pages stay mounted in source for rollback but
 * cannot issue requests while unavailable.
 */
export const FEATURE_IMAGES =
  getProductCapability("image-generation").available;
export const FEATURE_AUDIO =
  getProductCapability("audio-generation").available;
export const FEATURE_TRAIN = getProductCapability("training").available;
export const FEATURE_PROJECTS = getProductCapability("projects").available;
export const FEATURE_VIDEO = getProductCapability("video-generation").available;
export const FEATURE_RECIPES = getProductCapability("recipes").available;
export const FEATURE_EXPORT = getProductCapability("export").available;
export const FEATURE_API_MONITOR =
  getProductCapability("api-monitor").available;
export const FEATURE_AGENTS_NAV =
  getProductCapability("agents").visibleInNavigation;
export const FEATURE_FILES_NAV =
  getProductCapability("files").visibleInNavigation;
export const FEATURE_MEMORY_NAV =
  getProductCapability("memory").visibleInNavigation;
export const FEATURE_SEARCH_NAV =
  getProductCapability("search").visibleInNavigation;
export const FEATURE_MANAGEMENT_NAV =
  getProductCapability("management").visibleInNavigation;
