import { describe, expect, it } from "vitest";
import { enforceAnimationTriggerIntent } from "./animation-policy.js";
import { createDeck } from "./schema.js";

function decks() {
  const original = createDeck("Demo", "demo");
  original.slides[0].objects.push({
    id: "shape-1", type: "shape", x: 0, y: 0, width: 100, height: 100,
    rotation: 0, zIndex: 1, styles: {}
  });
  const edited = structuredClone(original);
  edited.slides[0].objects[0].animation = {
    name: "pop-in", trigger: "slide-enter", durationMs: 700,
    delayMs: 0, easing: "ease", iterationCount: 1
  };
  return { original, edited };
}

describe("AI animation trigger policy", () => {
  it("defaults newly generated animations to the presentation arrow", () => {
    const { original, edited } = decks();
    expect(enforceAnimationTriggerIntent(original, edited, "중앙에서 나타나는 애니메이션 만들어줘")
      .slides[0].objects[0].animation?.trigger).toBe("click");
  });

  it("keeps automatic entry when the user explicitly requests it", () => {
    const { original, edited } = decks();
    expect(enforceAnimationTriggerIntent(original, edited, "슬라이드 진입 시 자동으로 재생해줘")
      .slides[0].objects[0].animation?.trigger).toBe("slide-enter");
  });
});
