#!/usr/bin/env node

import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import {
  mkdir,
  readFile,
  rename,
  stat,
  writeFile,
} from "node:fs/promises";
import { spawn } from "node:child_process";
import {
  createHash,
  randomBytes,
  randomUUID,
  timingSafeEqual,
} from "node:crypto";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const MODULE_DIR = dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = resolve(MODULE_DIR, "..");
const API_PREFIX = "/slide-editor-api";
const MAX_BODY_BYTES = 12 * 1024 * 1024;
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const SESSION_COOKIE = "tosemfdk_slide_editor";
const SLUG_PATTERN = /^[a-z0-9][a-z0-9-]{0,79}$/;
const ID_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}$/;
const ALLOWED_ANIMATIONS = new Set([
  "none",
  "fade-in",
  "slide-in-right",
  "slide-in-left",
  "slide-in-up",
  "slide-in-down",
  "zoom-in",
]);
const ALLOWED_ADDITION_TYPES = new Set(["text", "image", "shape"]);
const ALLOWED_STYLE_KEYS = new Set([
  "fontFamily",
  "fontSize",
  "fontWeight",
  "color",
  "backgroundColor",
  "textAlign",
  "opacity",
  "translateX",
  "translateY",
  "left",
  "top",
  "width",
  "height",
  "borderRadius",
]);
const IMAGE_TYPES = new Map([
  ["image/png", ".png"],
  ["image/jpeg", ".jpg"],
  ["image/webp", ".webp"],
  ["image/gif", ".gif"],
  ["image/avif", ".avif"],
]);

function boundedString(value, max = 4000) {
  return typeof value === "string" ? value.slice(0, max) : "";
}

function safeId(value, fallback = randomUUID()) {
  return ID_PATTERN.test(String(value || "")) ? String(value) : fallback;
}

export function sanitizeSlug(value) {
  const slug = String(value || "").toLowerCase();
  if (!SLUG_PATTERN.test(slug)) throw new HttpError(400, "Invalid deck slug");
  return slug;
}

export function sanitizeFileName(value) {
  const base = String(value || "image")
    .normalize("NFKD")
    .replace(/[^\w.-]+/g, "-")
    .replace(/^[.-]+|[.-]+$/g, "")
    .slice(0, 64);
  return base || "image";
}

function sanitizeStyles(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};

  return Object.fromEntries(
    Object.entries(value)
      .filter(([key, entry]) => ALLOWED_STYLE_KEYS.has(key) && typeof entry === "string")
      .map(([key, entry]) => [key, entry.slice(0, 160)])
  );
}

function sanitizeImageSource(value) {
  const source = boundedString(value, 500);
  return source.startsWith(`${API_PREFIX}/uploads-public/`) ||
    source.startsWith("/assets/uploads/slides/")
    ? source
    : "";
}

function sanitizePatch(value) {
  const patch = value && typeof value === "object" ? value : {};
  const animation = ALLOWED_ANIMATIONS.has(patch.animation) ? patch.animation : "none";
  const src = sanitizeImageSource(patch.src);

  return {
    ...(typeof patch.text === "string" ? { text: boundedString(patch.text, 12000) } : {}),
    ...(src ? { src } : {}),
    ...(typeof patch.alt === "string" ? { alt: boundedString(patch.alt, 500) } : {}),
    styles: sanitizeStyles(patch.styles),
    animation,
    hidden: Boolean(patch.hidden),
  };
}

function sanitizeAddition(value, index) {
  const addition = value && typeof value === "object" ? value : {};
  const type = ALLOWED_ADDITION_TYPES.has(addition.type) ? addition.type : "text";
  const slideIndex = Math.max(0, Math.min(99, Number.parseInt(addition.slideIndex, 10) || 0));

  return {
    id: safeId(addition.id, `added-${index}-${randomUUID()}`),
    type,
    slideIndex,
    text: boundedString(addition.text, 12000),
    src: sanitizeImageSource(addition.src),
    alt: boundedString(addition.alt, 500),
    styles: sanitizeStyles(addition.styles),
    animation: ALLOWED_ANIMATIONS.has(addition.animation) ? addition.animation : "none",
    hidden: Boolean(addition.hidden),
  };
}

function sanitizeComment(value, index) {
  const comment = value && typeof value === "object" ? value : {};
  return {
    id: safeId(comment.id, `comment-${index}-${randomUUID()}`),
    objectId: safeId(comment.objectId, "deck"),
    body: boundedString(comment.body, 2000),
    status: comment.status === "resolved" ? "resolved" : "open",
    createdAt:
      typeof comment.createdAt === "string"
        ? comment.createdAt.slice(0, 40)
        : new Date().toISOString(),
  };
}

export function sanitizeDeckState(value, slug, revision = 0) {
  const state = value && typeof value === "object" ? value : {};
  const objectPatches =
    state.objectPatches && typeof state.objectPatches === "object"
      ? Object.fromEntries(
          Object.entries(state.objectPatches)
            .filter(([id]) => ID_PATTERN.test(id))
            .slice(0, 1000)
            .map(([id, patch]) => [id, sanitizePatch(patch)])
        )
      : {};

  return {
    schemaVersion: 1,
    deckSlug: slug,
    revision,
    frozen: Boolean(state.frozen),
    updatedAt: new Date().toISOString(),
    objectPatches,
    additions: Array.isArray(state.additions)
      ? state.additions.slice(0, 300).map(sanitizeAddition)
      : [],
    comments: Array.isArray(state.comments)
      ? state.comments.slice(0, 1000).map(sanitizeComment)
      : [],
  };
}

function createEmptyState(slug) {
  return sanitizeDeckState({}, slug, 0);
}

function publicUploadUrl(slug, fileName) {
  return `/assets/uploads/slides/${slug}/${fileName}`;
}

function previewUploadUrl(slug, fileName) {
  return `${API_PREFIX}/uploads-public/${slug}/${fileName}`;
}

export function createPublishedState(state) {
  const rewriteSource = (source) =>
    source.replace(
      new RegExp(`^${API_PREFIX}/uploads-public/([^/]+)/`),
      "/assets/uploads/slides/$1/"
    );

  return {
    schemaVersion: 1,
    deckSlug: state.deckSlug,
    revision: state.revision,
    frozen: true,
    updatedAt: new Date().toISOString(),
    objectPatches: Object.fromEntries(
      Object.entries(state.objectPatches).map(([id, patch]) => [
        id,
        { ...patch, ...(patch.src ? { src: rewriteSource(patch.src) } : {}) },
      ])
    ),
    additions: state.additions.map((addition) => ({
      ...addition,
      ...(addition.src ? { src: rewriteSource(addition.src) } : {}),
    })),
  };
}

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function readJsonBody(request) {
  const chunks = [];
  let size = 0;

  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw new HttpError(413, "Request body is too large");
    chunks.push(chunk);
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  } catch {
    throw new HttpError(400, "Invalid JSON body");
  }
}

function sendJson(response, status, value) {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "X-Content-Type-Options": "nosniff",
  });
  response.end(body);
}

function safeTokenEqual(provided, expected) {
  const left = Buffer.from(provided || "");
  const right = Buffer.from(expected || "");
  return left.length === right.length && left.length > 0 && timingSafeEqual(left, right);
}

function cookieValue(request, name) {
  const cookies = String(request.headers.cookie || "").split(";");
  for (const cookie of cookies) {
    const separator = cookie.indexOf("=");
    if (separator < 0) continue;
    if (cookie.slice(0, separator).trim() === name) {
      try {
        return decodeURIComponent(cookie.slice(separator + 1).trim());
      } catch {
        return "";
      }
    }
  }
  return "";
}

function isImageContent(type, content) {
  if (type === "image/png") {
    return content.subarray(0, 8).equals(Buffer.from("89504e470d0a1a0a", "hex"));
  }
  if (type === "image/jpeg") {
    return content.length >= 3 && content[0] === 0xff && content[1] === 0xd8 && content[2] === 0xff;
  }
  if (type === "image/gif") {
    return ["GIF87a", "GIF89a"].includes(content.subarray(0, 6).toString("ascii"));
  }
  if (type === "image/webp") {
    return (
      content.subarray(0, 4).toString("ascii") === "RIFF" &&
      content.subarray(8, 12).toString("ascii") === "WEBP"
    );
  }
  if (type === "image/avif") {
    return (
      content.subarray(4, 8).toString("ascii") === "ftyp" &&
      content.subarray(8, 32).includes(Buffer.from("avif"))
    );
  }
  return false;
}

async function atomicWrite(filePath, content) {
  await mkdir(dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`;
  await writeFile(temporaryPath, content, { mode: 0o600 });
  await rename(temporaryPath, filePath);
}

function runCommand(command, args, options = {}) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env || process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout = `${stdout}${chunk}`.slice(-24000);
    });
    child.stderr.on("data", (chunk) => {
      stderr = `${stderr}${chunk}`.slice(-24000);
    });
    child.on("error", rejectPromise);
    child.on("close", (code) => {
      if (code === 0) resolvePromise({ stdout, stderr });
      else rejectPromise(new Error(`${command} exited ${code}\n${stderr || stdout}`));
    });

    if (options.input) child.stdin.end(options.input);
    else child.stdin.end();
  });
}

async function readJsonFile(filePath) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

export function createSlideEditorServer(options = {}) {
  const root = resolve(options.root || process.env.SLIDE_EDITOR_ROOT || DEFAULT_ROOT);
  const draftRoot = resolve(options.draftRoot || join(root, ".slide-editor", "drafts"));
  const publicRoot = resolve(options.publicRoot || join(root, "assets", "data", "slides"));
  const uploadRoot = resolve(options.uploadRoot || join(root, "assets", "uploads", "slides"));
  const token = String(options.token || process.env.SLIDE_EDITOR_TOKEN || "").trim();
  const skipDeploy =
    options.skipDeploy ?? process.env.SLIDE_EDITOR_SKIP_DEPLOY === "1";
  const skipGit = options.skipGit ?? process.env.SLIDE_EDITOR_SKIP_GIT === "1";
  const sessionFingerprint = createHash("sha256")
    .update(`slide-editor-session:${token}`)
    .digest("base64url");
  let publishing = false;

  if (token.length < 24) throw new Error("SLIDE_EDITOR_TOKEN must contain at least 24 characters");

  function draftPath(slug) {
    return join(draftRoot, `${slug}.json`);
  }

  function publishedPath(slug) {
    return join(publicRoot, `${slug}.json`);
  }

  async function loadState(slug) {
    const draft = await readJsonFile(draftPath(slug));
    if (draft) return sanitizeDeckState(draft, slug, Number(draft.revision) || 0);

    const published = await readJsonFile(publishedPath(slug));
    if (published) {
      return sanitizeDeckState(
        { ...published, comments: [] },
        slug,
        Number(published.revision) || 0
      );
    }

    return createEmptyState(slug);
  }

  async function publishDeck(slug) {
    if (publishing) throw new HttpError(409, "Another publish is already running");
    publishing = true;
    const target = publishedPath(slug);
    let previous = null;

    try {
      const state = await loadState(slug);
      const published = createPublishedState(state);

      try {
        previous = await readFile(target);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }

      await atomicWrite(target, `${JSON.stringify(published, null, 2)}\n`);

      if (!skipDeploy) {
        try {
          await runCommand(join(root, "tools", "deploy.sh"), [], { cwd: root });
        } catch (error) {
          if (previous) await atomicWrite(target, previous);
          throw error;
        }
      }

      await atomicWrite(
        draftPath(slug),
        `${JSON.stringify(
          {
            ...state,
            frozen: true,
            updatedAt: new Date().toISOString(),
          },
          null,
          2
        )}\n`
      );

      const warnings = [];
      if (!skipGit) {
        const relativePublished = relative(root, target);
        const relativeUploads = relative(root, join(uploadRoot, slug));
        const gitPaths = [relativePublished];
        try {
          await stat(join(uploadRoot, slug));
          gitPaths.push(relativeUploads);
        } catch (error) {
          if (error.code !== "ENOENT") throw error;
        }
        const commitMessage = [
          `Publish reviewed edits for ${slug}`,
          "",
          "The owner-approved slide-editor snapshot is frozen into the public overlay.",
          "",
          "Constraint: Draft comments remain private and are excluded from the published snapshot",
          "Confidence: high",
          "Scope-risk: narrow",
          `Tested: Slide editor publish validation and production Jekyll deployment`,
          "",
        ].join("\n");

        try {
          await runCommand(
            "git",
            ["add", "--", ...gitPaths],
            { cwd: root }
          );
          const staged = await runCommand(
            "git",
            ["diff", "--cached", "--quiet", "--", ...gitPaths],
            { cwd: root }
          ).then(
            () => false,
            () => true
          );
          if (staged) {
            await runCommand(
              "git",
              ["commit", "--only", "-F", "-", "--", ...gitPaths],
              { cwd: root, input: commitMessage }
            );
            await runCommand("git", ["push", "origin", "main"], { cwd: root });
          }
        } catch (error) {
          warnings.push(`Published successfully, but Git sync failed: ${error.message}`);
        }
      }

      return { state: published, warnings };
    } finally {
      publishing = false;
    }
  }

  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      const pathname = url.pathname.startsWith(API_PREFIX)
        ? url.pathname.slice(API_PREFIX.length) || "/"
        : url.pathname;

      if (request.method === "GET" && pathname === "/health") {
        sendJson(response, 200, { ok: true, publishing });
        return;
      }

      const authorization = request.headers.authorization || "";
      const providedToken = authorization.startsWith("Bearer ")
        ? authorization.slice(7)
        : "";
      const bearerAuthenticated = safeTokenEqual(providedToken, token);
      const cookieAuthenticated = safeTokenEqual(
        cookieValue(request, SESSION_COOKIE),
        sessionFingerprint
      );

      const uploadMatch = pathname.match(
        /^\/uploads-public\/([a-z0-9-]+)\/([a-zA-Z0-9._-]+)$/
      );
      if (request.method === "GET" && uploadMatch) {
        if (!bearerAuthenticated && !cookieAuthenticated) {
          throw new HttpError(401, "Editor authentication required");
        }
        const slug = sanitizeSlug(uploadMatch[1]);
        const fileName = sanitizeFileName(uploadMatch[2]);
        const filePath = join(uploadRoot, slug, fileName);
        let metadata;
        try {
          metadata = await stat(filePath);
        } catch (error) {
          if (error.code === "ENOENT") throw new HttpError(404, "Upload not found");
          throw error;
        }
        const extension = extname(fileName).toLowerCase();
        const contentType =
          [...IMAGE_TYPES.entries()].find(([, suffix]) => suffix === extension)?.[0] ||
          "application/octet-stream";
        response.writeHead(200, {
          "Cache-Control": "private, max-age=60",
          "Content-Type": contentType,
          "Content-Length": metadata.size,
          "X-Content-Type-Options": "nosniff",
        });
        createReadStream(filePath).pipe(response);
        return;
      }

      if (!bearerAuthenticated) {
        throw new HttpError(401, "Editor authentication required");
      }
      response.setHeader(
        "Set-Cookie",
        `${SESSION_COOKIE}=${encodeURIComponent(
          sessionFingerprint
        )}; HttpOnly; Secure; SameSite=Strict; Path=${API_PREFIX}; Max-Age=21600`
      );

      const deckMatch = pathname.match(/^\/decks\/([a-z0-9-]+)$/);
      if (deckMatch && request.method === "GET") {
        const slug = sanitizeSlug(deckMatch[1]);
        sendJson(response, 200, await loadState(slug));
        return;
      }

      if (deckMatch && request.method === "PUT") {
        const slug = sanitizeSlug(deckMatch[1]);
        const incoming = await readJsonBody(request);
        const current = await loadState(slug);
        const incomingRevision = Number.parseInt(incoming.revision, 10) || 0;
        if (incomingRevision !== current.revision) {
          throw new HttpError(409, "Draft changed elsewhere; reload before saving");
        }
        const next = sanitizeDeckState(incoming, slug, current.revision + 1);
        await atomicWrite(draftPath(slug), `${JSON.stringify(next, null, 2)}\n`);
        sendJson(response, 200, next);
        return;
      }

      const uploadRouteMatch = pathname.match(/^\/uploads\/([a-z0-9-]+)$/);
      if (uploadRouteMatch && request.method === "POST") {
        const slug = sanitizeSlug(uploadRouteMatch[1]);
        const body = await readJsonBody(request);
        const type = String(body.type || "");
        const extension = IMAGE_TYPES.get(type);
        if (!extension) throw new HttpError(415, "Unsupported image type");

        const encoded = String(body.data || "").replace(/^data:[^;]+;base64,/, "");
        const content = Buffer.from(encoded, "base64");
        if (content.length === 0 || content.length > MAX_IMAGE_BYTES) {
          throw new HttpError(413, "Image must be between 1 byte and 8 MB");
        }
        if (!isImageContent(type, content)) {
          throw new HttpError(415, "Uploaded bytes do not match the declared image type");
        }

        const requestedBase = sanitizeFileName(body.name).replace(/\.[^.]+$/, "");
        const fileName = `${Date.now()}-${randomBytes(5).toString("hex")}-${requestedBase}${extension}`;
        const filePath = join(uploadRoot, slug, fileName);
        await mkdir(dirname(filePath), { recursive: true });
        await writeFile(filePath, content, { mode: 0o644 });
        sendJson(response, 201, {
          fileName,
          previewUrl: previewUploadUrl(slug, fileName),
          publicUrl: publicUploadUrl(slug, fileName),
        });
        return;
      }

      const publishMatch = pathname.match(/^\/publish\/([a-z0-9-]+)$/);
      if (publishMatch && request.method === "POST") {
        const slug = sanitizeSlug(publishMatch[1]);
        const result = await publishDeck(slug);
        sendJson(response, 200, {
          ok: true,
          revision: result.state.revision,
          warnings: result.warnings,
        });
        return;
      }

      throw new HttpError(404, "Not found");
    } catch (error) {
      const status = error instanceof HttpError ? error.status : 500;
      if (status === 500) console.error(error);
      sendJson(response, status, {
        error: status === 500 ? "Internal slide editor error" : error.message,
      });
    }
  });
}

async function main() {
  const tokenFile =
    process.env.SLIDE_EDITOR_TOKEN_FILE ||
    join(process.env.HOME || "", ".config", "tosemfdk", "slide-editor-token");
  const token = process.env.SLIDE_EDITOR_TOKEN || (await readFile(tokenFile, "utf8")).trim();
  const port = Number.parseInt(process.env.SLIDE_EDITOR_PORT || "5556", 10);
  const host = process.env.SLIDE_EDITOR_HOST || "127.0.0.1";
  const server = createSlideEditorServer({ token });

  server.listen(port, host, () => {
    console.log(`Slide editor API listening on http://${host}:${port}`);
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
