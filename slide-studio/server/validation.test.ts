import { describe, expect, it } from "vitest";
import { createDeck } from "./schema.js";
import { validateCss, validateDeck } from "./validation.js";

describe("Slide Studio validation", () => {
  it("accepts the canonical 1920x1080 deck", () => {
    const deck = createDeck("Demo", "demo");
    expect(validateDeck(deck)).toEqual(deck);
  });

  it.each([
    '@import "https://example.com/a.css";',
    ".x { background: url(https://example.com/a.png); }",
    ".x { width: expression(alert(1)); }",
    "</style><script>alert(1)</script>"
  ])("rejects network-capable or executable CSS: %s", (css) => {
    expect(() => validateCss(css)).toThrow(/forbidden/i);
  });

  it("rejects media objects that reference unavailable assets", () => {
    const deck = createDeck("Demo", "demo");
    deck.slides[0].objects.push({
      id: "image-1", type: "image", assetId: "a892074f-18c9-4ecf-a0dd-5c022f6c38dc",
      x: 0, y: 0, width: 100, height: 100, rotation: 0, zIndex: 1, styles: {}
    });
    expect(() => validateDeck(deck, new Set())).toThrow(/unknown asset/i);
  });

  it("normalizes abbreviated and CSS-style animation timing", () => {
    const deck = createDeck("Demo", "demo");
    deck.slides[0].objects.push({
      id: "shape-1",
      type: "shape",
      x: 100,
      y: 100,
      width: 200,
      height: 200,
      rotation: 0,
      zIndex: 1,
      styles: {},
      animation: {
        name: "zoom-in",
        trigger: "slide-enter",
        duration: "0.6s",
        timingFunction: "ease-out"
      }
    } as unknown as (typeof deck.slides)[number]["objects"][number]);

    const normalized = validateDeck(deck);
    expect(normalized.slides[0].objects[0].animation).toEqual({
      name: "zoom-in",
      trigger: "slide-enter",
      durationMs: 600,
      delayMs: 0,
      easing: "ease-out",
      iterationCount: 1
    });
  });
});
