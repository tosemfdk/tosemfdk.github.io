import { readFile } from "node:fs/promises";
import { deckSchema, type Deck } from "./schema.js";

const FORBIDDEN_CSS = [
  /@import\b/i,
  /url\s*\(\s*["']?\s*(?:https?:|data:|javascript:)/i,
  /expression\s*\(/i,
  /javascript\s*:/i,
  /behavior\s*:/i,
  /-moz-binding\s*:/i,
  /<\/?style/i,
  /<script/i
];

export function validateCss(css: string, label = "CSS"): void {
  if (Buffer.byteLength(css) > 256 * 1024) throw new Error(`${label} exceeds 256KB`);
  for (const pattern of FORBIDDEN_CSS) {
    if (pattern.test(css)) throw new Error(`${label} contains forbidden content: ${pattern.source}`);
  }
  const opens = [...css].filter((character) => character === "{").length;
  const closes = [...css].filter((character) => character === "}").length;
  if (opens !== closes) throw new Error(`${label} has unbalanced braces`);
}

export function validateDeck(value: unknown, assetIds?: Set<string>): Deck {
  const deck = deckSchema.parse(value);
  if (assetIds) {
    for (const slide of deck.slides) {
      for (const object of slide.objects) {
        if (object.assetId && !assetIds.has(object.assetId)) {
          throw new Error(`Object ${object.id} references unknown asset ${object.assetId}`);
        }
      }
    }
  }
  return deck;
}

export async function validateWorkspace(
  deckPath: string,
  themePath: string,
  animationsPath: string,
  assetIds?: Set<string>
): Promise<{ deck: Deck; themeCss: string; animationsCss: string }> {
  const [deckSource, themeCss, animationsCss] = await Promise.all([
    readFile(deckPath, "utf8"),
    readFile(themePath, "utf8"),
    readFile(animationsPath, "utf8")
  ]);
  const deck = validateDeck(JSON.parse(deckSource), assetIds);
  validateCss(themeCss, "theme.css");
  validateCss(animationsCss, "animations.css");
  return { deck, themeCss, animationsCss };
}
