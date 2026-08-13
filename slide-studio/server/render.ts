import type { Deck } from "./schema.js";

export interface RenderOptions {
  title?: string;
  assetBase?: string;
  assetMap?: Record<string, string>;
  themeHref?: string;
  animationsHref?: string;
  runtimeCssHref?: string;
  runtimeJsHref?: string;
  noIndex?: boolean;
  editorPreview?: boolean;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[character] || character);
}

function safeJson(value: unknown): string {
  return JSON.stringify(value).replace(/</g, "\\u003c").replace(/-->/g, "--\\u003e");
}

export function renderDeckHtml(deck: Deck, options: RenderOptions = {}): string {
  const config = {
    assetBase: options.assetBase || "",
    assetMap: options.assetMap || {},
    editorPreview: Boolean(options.editorPreview)
  };
  return `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  ${options.noIndex === false ? "" : '<meta name="robots" content="noindex,nofollow">'}
  <meta name="theme-color" content="#080b14">
  <title>${escapeHtml(options.title || deck.title)}</title>
  <link rel="stylesheet" href="${escapeHtml(options.runtimeCssHref || "/studio-runtime.css")}">
  <link rel="stylesheet" href="${escapeHtml(options.themeHref || "./theme.css")}">
  <link rel="stylesheet" href="${escapeHtml(options.animationsHref || "./animations.css")}">
</head>
<body>
  <main class="studio-deck" data-studio-deck aria-label="${escapeHtml(deck.title)}"></main>
  <script id="deck-data" type="application/json">${safeJson(deck)}</script>
  <script id="deck-config" type="application/json">${safeJson(config)}</script>
  <script src="${escapeHtml(options.runtimeJsHref || "/studio-runtime.js")}" defer></script>
</body>
</html>`;
}
