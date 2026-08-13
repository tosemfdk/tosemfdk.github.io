import { describe, expect, it } from "vitest";
import { createAssetObject, objectTypeForAsset } from "./deck-utils";
import type { Asset } from "./types";

const asset = (mime: string): Asset => ({
  id: "a892074f-18c9-4ecf-a0dd-5c022f6c38dc",
  projectId: "project",
  originalName: "sample.bin",
  mime,
  size: 10,
  sha256: "x",
  createdAt: new Date(0).toISOString()
});

describe("deck asset placement", () => {
  it.each([
    ["image/png", "image"], ["video/mp4", "video"], ["audio/mpeg", "audio"],
    ["application/pdf", "pdf"], ["application/zip", "attachment"]
  ])("maps %s to %s", (mime, expected) => {
    expect(objectTypeForAsset(asset(mime))).toBe(expected);
  });

  it("centers an uploaded asset around the selected canvas point", () => {
    const object = createAssetObject(asset("image/png"), { x: 1000, y: 600 });
    expect(object.x + object.width / 2).toBe(1000);
    expect(object.y + object.height / 2).toBe(600);
  });
});
