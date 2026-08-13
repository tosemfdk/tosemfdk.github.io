import type { Asset, Deck, DeckObject, ObjectType, Slide } from "./types";

export function cloneDeck(deck: Deck): Deck {
  return structuredClone(deck);
}

export function createSlide(title: string): Slide {
  return { id: crypto.randomUUID(), title, background: "#ffffff", objects: [] };
}

export function objectTypeForAsset(asset: Asset): ObjectType {
  if (asset.mime.startsWith("image/")) return "image";
  if (asset.mime.startsWith("video/")) return "video";
  if (asset.mime.startsWith("audio/")) return "audio";
  if (asset.mime === "application/pdf") return "pdf";
  return "attachment";
}

export function createAssetObject(asset: Asset, point?: { x: number; y: number }): DeckObject {
  const type = objectTypeForAsset(asset);
  const dimensions = {
    image: [720, 480], video: [960, 540], audio: [700, 110], pdf: [900, 820], attachment: [520, 220]
  }[type as "image" | "video" | "audio" | "pdf" | "attachment"] || [520, 220];
  const [width, height] = dimensions;
  return {
    id: `object-${crypto.randomUUID()}`,
    type,
    x: Math.round((point?.x ?? 960) - width / 2),
    y: Math.round((point?.y ?? 540) - height / 2),
    width,
    height,
    rotation: 0,
    zIndex: 10,
    content: asset.originalName,
    assetId: asset.id,
    styles: {
      objectFit: "contain",
      borderRadius: type === "image" || type === "video" ? "18px" : "0px"
    }
  };
}

export function clampObject(object: DeckObject): DeckObject {
  return {
    ...object,
    x: Math.max(-object.width, Math.min(1920, object.x)),
    y: Math.max(-object.height, Math.min(1080, object.y)),
    width: Math.max(1, Math.min(7680, object.width)),
    height: Math.max(1, Math.min(4320, object.height))
  };
}

export function findObject(deck: Deck, slideId: string, objectId: string): DeckObject | undefined {
  return deck.slides.find((slide) => slide.id === slideId)?.objects.find((object) => object.id === objectId);
}
