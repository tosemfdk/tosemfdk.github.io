import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createSlideEditorServer,
  sanitizeFileName,
  sanitizeSlug,
} from "./slide-editor-server.mjs";

const TOKEN = "test-token-that-is-long-enough-for-authentication";
const SLUG = "active-scene-change-detection";

async function startServer(t) {
  const root = await mkdtemp(join(tmpdir(), "slide-editor-test-"));
  const server = createSlideEditorServer({
    root,
    token: TOKEN,
    skipDeploy: true,
    skipGit: true,
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();

  t.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  });

  return {
    root,
    request: async (path, options = {}) => {
      const response = await fetch(`http://127.0.0.1:${port}/slide-editor-api${path}`, {
        ...options,
        headers: {
          ...(options.authorized === false ? {} : { Authorization: `Bearer ${TOKEN}` }),
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...(options.headers || {}),
        },
      });
      const contentType = response.headers.get("content-type") || "";
      return {
        response,
        body: contentType.startsWith("application/json")
          ? await response.json()
          : Buffer.from(await response.arrayBuffer()),
      };
    },
  };
}

test("slug and file-name sanitizers reject traversal", () => {
  assert.equal(sanitizeSlug(SLUG), SLUG);
  assert.throws(() => sanitizeSlug("../secret"), /Invalid deck slug/);
  assert.equal(sanitizeFileName("../../deck image.png"), "deck-image.png");
});

test("API requires authentication and persists sanitized draft state", async (t) => {
  const { request } = await startServer(t);

  const health = await request("/health", { authorized: false });
  assert.equal(health.response.status, 200);
  assert.equal(health.body.ok, true);

  const denied = await request(`/decks/${SLUG}`, { authorized: false });
  assert.equal(denied.response.status, 401);

  const initial = await request(`/decks/${SLUG}`);
  assert.equal(initial.response.status, 200);
  assert.equal(initial.body.revision, 0);
  assert.match(initial.response.headers.get("set-cookie"), /HttpOnly; Secure; SameSite=Strict/);

  const saved = await request(`/decks/${SLUG}`, {
    method: "PUT",
    body: JSON.stringify({
      ...initial.body,
      objectPatches: {
        "slide-1-h1-1": {
          text: "Edited title",
          styles: {
            color: "#112233",
            fontSize: "48px",
            position: "fixed",
          },
          animation: "slide-in-right",
        },
      },
      comments: [
        {
          id: "comment-1",
          objectId: "slide-1-h1-1",
          body: "Move this to the left",
          status: "open",
        },
      ],
    }),
  });

  assert.equal(saved.response.status, 200);
  assert.equal(saved.body.revision, 1);
  assert.equal(saved.body.objectPatches["slide-1-h1-1"].text, "Edited title");
  assert.equal(saved.body.objectPatches["slide-1-h1-1"].styles.position, undefined);
  assert.equal(saved.body.comments.length, 1);

  const stale = await request(`/decks/${SLUG}`, {
    method: "PUT",
    body: JSON.stringify(initial.body),
  });
  assert.equal(stale.response.status, 409);
});

test("image upload and publish create a public overlay without comments", async (t) => {
  const { root, request } = await startServer(t);
  const pixelPng =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

  const uploaded = await request(`/uploads/${SLUG}`, {
    method: "POST",
    body: JSON.stringify({
      name: "diagram.png",
      type: "image/png",
      data: pixelPng,
    }),
  });
  assert.equal(uploaded.response.status, 201);
  assert.match(uploaded.body.previewUrl, /uploads-public/);

  const privateUpload = await request(
    uploaded.body.previewUrl.replace("/slide-editor-api", ""),
    { authorized: false }
  );
  assert.equal(privateUpload.response.status, 401);

  const authenticatedUpload = await request(
    uploaded.body.previewUrl.replace("/slide-editor-api", "")
  );
  assert.equal(authenticatedUpload.response.status, 200);
  assert.equal(authenticatedUpload.response.headers.get("content-type"), "image/png");

  const initial = (await request(`/decks/${SLUG}`)).body;
  const saved = await request(`/decks/${SLUG}`, {
    method: "PUT",
    body: JSON.stringify({
      ...initial,
      additions: [
        {
          id: "added-image-1",
          type: "image",
          slideIndex: 0,
          src: uploaded.body.previewUrl,
          alt: "Diagram",
          styles: { left: "50%", top: "50%", width: "20rem" },
          animation: "fade-in",
        },
        {
          id: "added-external-image",
          type: "image",
          slideIndex: 0,
          src: "https://tracker.example/image.png",
          alt: "External image",
          styles: {},
        },
      ],
      comments: [
        {
          id: "comment-private",
          objectId: "added-image-1",
          body: "Private instruction",
          status: "open",
        },
      ],
    }),
  });
  assert.equal(saved.response.status, 200);
  assert.equal(saved.body.additions[1].src, "");

  const published = await request(`/publish/${SLUG}`, {
    method: "POST",
    body: "{}",
  });
  assert.equal(published.response.status, 200);

  const publicState = JSON.parse(
    await readFile(join(root, "assets", "data", "slides", `${SLUG}.json`), "utf8")
  );
  assert.equal(publicState.frozen, true);
  assert.equal(publicState.comments, undefined);
  assert.match(publicState.additions[0].src, /^\/assets\/uploads\/slides\//);

  const frozenDraft = (
    await request(`/decks/${SLUG}`)
  ).body;
  assert.equal(frozenDraft.frozen, true);
  assert.equal(frozenDraft.comments.length, 1);
});
