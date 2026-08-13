import { expect, test } from "@playwright/test";

test("creates a deck and adds positioned content", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "LOE Slide Studio" })).toBeVisible();
  await page.getByPlaceholder("발표자료 제목").fill(`E2E ${Date.now()}`);
  await page.getByRole("button", { name: "새 덱 만들기" }).click();
  await expect(page.getByText("Codex 디자인 편집")).toBeVisible();
  await page.getByRole("button", { name: "T 텍스트" }).click();
  await expect(page.locator(".slide-canvas .slide-object__text", { hasText: "텍스트를 입력하세요" })).toBeVisible();
  await page.getByRole("button", { name: "점 좌표" }).click();
  const canvas = page.locator(".slide-canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Canvas is not visible");
  await page.mouse.click(box.x + box.width * 0.7, box.y + box.height * 0.35);
  await expect(page.locator(".context-chips")).toContainText("@point");
});
