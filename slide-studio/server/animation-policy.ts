import type { Deck } from "./schema.js";

const AUTOMATIC_TRIGGER_REQUEST = /(?:자동(?:으로)?|열자마자|시작하자마자|즉시\s*(?:재생|실행)|슬라이드\s*(?:진입|시작|열)|autoplay|auto-play|on\s*(?:load|enter)|slide[- ]?enter)/i;

export function enforceAnimationTriggerIntent(original: Deck, edited: Deck, prompt: string): Deck {
  if (AUTOMATIC_TRIGGER_REQUEST.test(prompt)) return edited;

  const originals = new Map(
    original.slides.flatMap((slide) => slide.objects.map((object) => [object.id, object] as const))
  );
  for (const slide of edited.slides) {
    for (const object of slide.objects) {
      if (!object.animation || object.animation.trigger === "click") continue;
      const previous = originals.get(object.id)?.animation;
      const triggerWasIntroduced = !previous || previous.trigger !== object.animation.trigger;
      if (triggerWasIntroduced) object.animation.trigger = "click";
    }
  }
  return edited;
}
