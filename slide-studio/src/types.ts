export const CANVAS_WIDTH = 1920;
export const CANVAS_HEIGHT = 1080;

export type ObjectType =
  | "text"
  | "image"
  | "video"
  | "audio"
  | "pdf"
  | "attachment"
  | "shape";

export type AnimationTrigger = "click" | "slide-enter" | "with-previous" | "after-previous";

export interface DeckAnimation {
  name: string;
  trigger: AnimationTrigger;
  durationMs: number;
  delayMs: number;
  easing: string;
  iterationCount: number;
}

export interface DeckObject {
  id: string;
  type: ObjectType;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  zIndex: number;
  content?: string;
  assetId?: string;
  className?: string;
  styles: Record<string, string | number>;
  animation?: DeckAnimation;
}

export interface Slide {
  id: string;
  title: string;
  background: string;
  objects: DeckObject[];
}

export interface Deck {
  schemaVersion: 1;
  title: string;
  slug: string;
  width: 1920;
  height: 1080;
  slides: Slide[];
}

export interface Project {
  id: string;
  name: string;
  slug: string;
  createdAt: string;
  updatedAt: string;
  latestReleaseId?: string | null;
  publicUrl?: string | null;
}

export interface Asset {
  id: string;
  projectId: string;
  originalName: string;
  mime: string;
  size: number;
  sha256: string;
  createdAt: string;
}

export interface SelectionContext {
  slideId: string;
  selectedObjectIds: string[];
  point?: { x: number; y: number };
  region?: { x: number; y: number; width: number; height: number };
}

export interface AiJob {
  id: string;
  projectId: string;
  status: "queued" | "running" | "ready" | "accepted" | "rejected" | "failed" | "cancelled";
  prompt: string;
  context: SelectionContext;
  summary?: string | null;
  error?: string | null;
  createdAt: string;
  updatedAt: string;
}
