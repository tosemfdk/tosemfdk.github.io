import { expect, test } from "@playwright/test";

test("creates a deck and adds positioned content", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "LOE Slide Studio" })).toBeVisible();
  await page.getByPlaceholder("발표자료 제목").fill(`E2E ${Date.now()}`);
  await page.getByRole("button", { name: "새 덱 만들기" }).click();
  await expect(page.getByText("Codex 디자인 편집")).toBeVisible();
  await page.getByRole("button", { name: "T 텍스트" }).click();
  await expect(page.locator(".slide-canvas .slide-object__text", { hasText: "텍스트를 입력하세요" })).toBeVisible();
  await page.getByRole("button", { name: "○ 도형" }).click();
  const textLayer = page.locator(".layer-list button", { hasText: "텍스트를 입력하세요" });
  await textLayer.click({ modifiers: ["Shift"] });
  await expect(page.locator(".slide-canvas .editor-object.is-selected")).toHaveCount(2);
  await expect(page.locator(".context-chips .object-context-chip")).toHaveCount(2);

  let submittedObjectIds: string[] = [];
  await page.route("**/api/projects/*/ai-jobs", async (route) => {
    const body = route.request().postDataJSON();
    submittedObjectIds = body.context.selectedObjectIds;
    const now = new Date().toISOString();
    await route.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify({
      id: "multi-object-job", projectId: "test-project", status: "ready", prompt: body.prompt,
      context: body.context, summary: "test", error: null, createdAt: now, updatedAt: now
    }) });
  });
  await page.getByPlaceholder(/선택한 객체들을/).fill("선택한 객체를 함께 정렬해줘");
  await page.getByRole("button", { name: /Codex 변경안 만들기/ }).click();
  await expect.poll(() => submittedObjectIds.length).toBe(2);
  expect(new Set(submittedObjectIds).size).toBe(2);

  await page.getByRole("button", { name: "점 좌표" }).click();
  const canvas = page.locator(".slide-canvas");
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Canvas is not visible");
  await page.mouse.click(box.x + box.width * 0.7, box.y + box.height * 0.35);
  await expect(page.locator(".context-chips")).toContainText("@point");
});

test("fits a 16:9 preview and reveals click animations with the right arrow", async ({ page, request }) => {
  const projectResponse = await request.post("/api/projects", { data: { name: `Animation ${Date.now()}` } });
  const project = await projectResponse.json();
  const detailResponse = await request.get(`/api/projects/${project.id}`);
  const detail = await detailResponse.json();
  detail.deck.slides[0].objects.push({
    id: "animated-shape",
    type: "shape",
    x: 760,
    y: 390,
    width: 400,
    height: 300,
    rotation: 0,
    zIndex: 1,
    styles: { backgroundColor: "#5b7cfa", borderRadius: "40px" },
    animation: {
      name: "zoom-in",
      trigger: "click",
      durationMs: 100,
      delayMs: 0,
      easing: "linear",
      iterationCount: 1
    }
  });
  await request.put(`/api/projects/${project.id}/deck`, { data: detail.deck });

  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto(`/api/projects/${project.id}/preview`);
  const viewport = page.locator(".studio-viewport");
  const box = await viewport.boundingBox();
  expect(box?.x).toBeCloseTo(0, 0);
  expect(box?.y).toBeCloseTo(0, 0);
  expect(box?.width).toBeCloseTo(1280, 0);
  expect(box?.height).toBeCloseTo(720, 0);

  const object = page.locator('[data-object-id="animated-shape"]');
  await expect(object).toHaveCSS("visibility", "hidden");
  const next = page.locator('[data-action="next"]');
  await expect(next).toHaveClass(/has-pending-animation/);
  await next.click();
  await expect(object).toHaveClass(/is-visible/);
  await expect(next).not.toHaveClass(/has-pending-animation/);
});
