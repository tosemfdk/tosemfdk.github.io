import { z } from "zod";
import { randomUUID } from "node:crypto";

export const CANVAS_WIDTH = 1920;
export const CANVAS_HEIGHT = 1080;

const styleValueSchema = z.union([z.string().max(500), z.number().finite()]);

export const animationSchema = z.object({
  name: z.string().regex(/^[a-zA-Z_][a-zA-Z0-9_-]{0,79}$/),
  trigger: z.enum(["click", "slide-enter", "with-previous", "after-previous"]),
  durationMs: z.number().int().min(0).max(120_000),
  delayMs: z.number().int().min(0).max(120_000),
  easing: z.string().max(120),
  iterationCount: z.number().int().min(1).max(100)
});

export const deckObjectSchema = z.object({
  id: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}$/),
  type: z.enum(["text", "image", "video", "audio", "pdf", "attachment", "shape"]),
  x: z.number().finite().min(-3840).max(5760),
  y: z.number().finite().min(-2160).max(3240),
  width: z.number().finite().positive().max(7680),
  height: z.number().finite().positive().max(4320),
  rotation: z.number().finite().min(-3600).max(3600),
  zIndex: z.number().int().min(-10_000).max(10_000),
  content: z.string().max(100_000).optional(),
  assetId: z.string().uuid().optional(),
  className: z.string().regex(/^[a-zA-Z0-9 _-]{0,200}$/).optional(),
  styles: z.record(z.string().max(80), styleValueSchema),
  animation: animationSchema.optional()
}).superRefine((object, context) => {
  if (["image", "video", "audio", "pdf", "attachment"].includes(object.type) && !object.assetId) {
    context.addIssue({ code: z.ZodIssueCode.custom, message: `${object.type} object requires assetId` });
  }
});

export const slideSchema = z.object({
  id: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}$/),
  title: z.string().max(500),
  background: z.string().max(500),
  objects: z.array(deckObjectSchema).max(2000)
});

export const deckSchema = z.object({
  schemaVersion: z.literal(1),
  title: z.string().min(1).max(500),
  slug: z.string().regex(/^[a-z0-9][a-z0-9-]{0,79}$/),
  width: z.literal(CANVAS_WIDTH),
  height: z.literal(CANVAS_HEIGHT),
  slides: z.array(slideSchema).min(1).max(500)
}).superRefine((deck, context) => {
  const ids = new Set<string>();
  for (const slide of deck.slides) {
    if (ids.has(slide.id)) context.addIssue({ code: z.ZodIssueCode.custom, message: `Duplicate id: ${slide.id}` });
    ids.add(slide.id);
    for (const object of slide.objects) {
      if (ids.has(object.id)) context.addIssue({ code: z.ZodIssueCode.custom, message: `Duplicate id: ${object.id}` });
      ids.add(object.id);
    }
  }
});

export type Deck = z.infer<typeof deckSchema>;

export function createDeck(title: string, slug: string): Deck {
  return {
    schemaVersion: 1,
    title,
    slug,
    width: CANVAS_WIDTH,
    height: CANVAS_HEIGHT,
    slides: [{ id: randomUUID(), title: "첫 번째 슬라이드", background: "#ffffff", objects: [] }]
  };
}
