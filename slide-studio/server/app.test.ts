import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import request from "supertest";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createApp } from "./app.js";
import { CodexJobManager } from "./codex.js";
import { StudioDatabase } from "./db.js";
import { StudioStorage } from "./storage.js";

describe("Slide Studio API", () => {
  let root: string;
  let database: StudioDatabase;
  let storage: StudioStorage;
  let app: ReturnType<typeof createApp>;

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "slide-studio-test-"));
    storage = new StudioStorage(join(root, "data"), { maxFileBytes: 1024 * 1024, maxProjectBytes: 4 * 1024 * 1024 });
    await storage.initialize();
    database = new StudioDatabase(join(root, "data", "test.sqlite3"));
    const fakeCodex = join(root, "fake-codex");
    await writeFile(fakeCodex, `#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";
const args = process.argv.slice(2);
const output = args[args.indexOf("--output-last-message") + 1];
process.stdin.resume();
process.stdin.on("end", () => {
  const deck = JSON.parse(readFileSync("deck.json", "utf8"));
  deck.slides[0].title = "Codex updated";
  deck.slides[0].objects.push({
    id: "codex-shape", type: "shape", x: 100, y: 100, width: 200, height: 200,
    rotation: 0, zIndex: 1, styles: { backgroundColor: "#123456" },
    animation: { name: "zoom-in", duration: "600ms", timingFunction: "ease-out", trigger: "slide-enter" }
  });
  writeFileSync("deck.json", JSON.stringify(deck, null, 2));
  writeFileSync("theme.css", readFileSync("theme.css", "utf8") + "\\n.codex-change { color: #123456; }\\n");
  writeFileSync(output, JSON.stringify({ summary: "Updated the slide", changes: ["Changed title"], warnings: [] }));
  console.log(JSON.stringify({ type: "result", ok: true }));
});
`);
    await chmod(fakeCodex, 0o755);
    const jobs = new CodexJobManager(database, storage, resolve("../.codex/skills/slide-studio/SKILL.md"), fakeCodex, 5000);
    app = createApp({
      database, storage, jobs,
      auth: { allowDevelopment: true },
      runtimeDir: resolve("runtime"),
      publicBaseUrl: "https://slides.example.com"
    });
  });

  afterEach(async () => {
    database.close();
    await rm(root, { recursive: true, force: true });
  });

  it("creates, edits, uploads, publishes, streams, and exports a deck", async () => {
    const created = await request(app).post("/api/projects").send({ name: "Demo Deck" }).expect(201);
    const projectId = created.body.id;
    const detail = await request(app).get(`/api/projects/${projectId}`).expect(200);
    expect(detail.body.deck.width).toBe(1920);
    const theme = await request(app).get(`/api/projects/${projectId}/files/theme.css`).expect(200);
    expect(theme.text).toContain("--deck-font");
    const animations = await request(app).get(`/api/projects/${projectId}/files/animations.css`).expect(200);
    expect(animations.text).toContain("@keyframes zoom-in");

    const upload = await request(app)
      .post(`/api/projects/${projectId}/assets`)
      .set("Content-Type", "image/png")
      .set("X-File-Name", encodeURIComponent("한글 이미지.png"))
      .send(Buffer.from("fake-png-content"))
      .expect(201);
    const assetId = upload.body.id;
    const jsonUpload = await request(app)
      .post(`/api/projects/${projectId}/assets`)
      .set("Content-Type", "application/json")
      .set("X-File-Name", "reference.json")
      .send('{"reference":true}')
      .expect(201);
    expect(jsonUpload.body.mime).toBe("application/json");

    const deck = detail.body.deck;
    deck.slides[0].objects.push({
      id: "image-1", type: "image", assetId, x: 100, y: 100, width: 600, height: 400,
      rotation: 0, zIndex: 1, content: "sample", styles: { objectFit: "contain" }
    });
    await request(app).put(`/api/projects/${projectId}/deck`).send(deck).expect(200);

    const range = await request(app).get(`/api/projects/${projectId}/asset-content/${assetId}`).set("Range", "bytes=0-3").expect(206);
    expect(range.headers["content-range"]).toContain("bytes 0-3/");

    const published = await request(app).post(`/api/projects/${projectId}/publish`).send({}).expect(201);
    expect(published.body.publicUrl).toBe("https://slides.example.com/p/demo-deck");
    const publicDeck = await request(app).get("/p/demo-deck").expect(200);
    expect(publicDeck.text).toContain("noindex,nofollow");
    expect(publicDeck.text).toContain(`/published/${published.body.releaseId}/assets/`);

    const archive = await request(app).get(`/api/projects/${projectId}/export`).buffer(true).parse(binaryParser).expect(200);
    expect(archive.headers["content-type"]).toContain("application/zip");
    expect(Buffer.from(archive.body).subarray(0, 2).toString()).toBe("PK");
  });

  it("runs Codex in a job copy and applies it only after approval", async () => {
    const created = await request(app).post("/api/projects").send({ name: "AI Deck" }).expect(201);
    const projectId = created.body.id;
    const started = await request(app).post(`/api/projects/${projectId}/ai-jobs`).send({
      prompt: "첫 슬라이드 제목을 바꿔줘",
      context: { slideId: "slide", selectedObjectIds: ["object-a", "object-b", "object-a"] }
    }).expect(202);
    expect(started.body.context.selectedObjectIds).toEqual(["object-a", "object-b"]);

    let job = started.body;
    for (let attempt = 0; attempt < 50 && ["queued", "running"].includes(job.status); attempt += 1) {
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 30));
      job = (await request(app).get(`/api/ai-jobs/${job.id}`).expect(200)).body;
    }
    expect(job.status, job.error).toBe("ready");
    await request(app).get(`/api/ai-jobs/${job.id}/files/theme.css`).expect(200);
    await request(app).get(`/api/ai-jobs/${job.id}/files/animations.css`).expect(200);

    const before = await request(app).get(`/api/projects/${projectId}`).expect(200);
    expect(before.body.deck.slides[0].title).not.toBe("Codex updated");
    await request(app).post(`/api/ai-jobs/${job.id}/accept`).send({}).expect(200);
    const after = await request(app).get(`/api/projects/${projectId}`).expect(200);
    expect(after.body.deck.slides[0].title).toBe("Codex updated");
    expect(after.body.deck.slides[0].objects[0].animation).toEqual({
      name: "zoom-in", trigger: "click", durationMs: 600,
      delayMs: 0, easing: "ease-out", iterationCount: 1
    });
    expect(after.body.versions).toHaveLength(1);
  });

  it("creates a schema-safe slug for a Korean-only project name", async () => {
    const created = await request(app).post("/api/projects").send({ name: "로봇 발표자료" }).expect(201);
    expect(created.body.slug).toMatch(/^deck-[a-z0-9]+$/);
    await request(app).post(`/api/projects/${created.body.id}/publish`).send({}).expect(201);
  });
});

function binaryParser(response: NodeJS.ReadableStream, callback: (error: Error | null, value?: Buffer) => void): void {
  const chunks: Buffer[] = [];
  response.on("data", (chunk: Buffer) => chunks.push(chunk));
  response.on("end", () => callback(null, Buffer.concat(chunks)));
  response.on("error", callback);
}
