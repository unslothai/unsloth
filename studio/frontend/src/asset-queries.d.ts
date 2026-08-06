


// Vite `?inline` imports (data URIs); vite/client only types bare extensions.
declare module "*?inline" {
  const src: string;
  // biome-ignore lint/style/noDefaultExport: Vite asset modules export default.
  export default src;
}
