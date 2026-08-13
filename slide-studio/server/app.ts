import archiver from "archiver";
import express, { type Express, type NextFunction, type Request, type Response } from "express";
import { randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { createSessionHandler, requireAuth, type AuthOptions } from "./auth.js";
import { CodexJobManager, type JobContext } from "./codex.js";
import { assetJson, jobJson, projectJson, StudioDatabase } from "./db.js";
import { renderDeckHtml } from "./render.js";
import { StudioStorage } from "./storage.js";
import { validateDeck, validateWorkspace } from "./validation.js";

export interface AppOptions {
  database: StudioDatabase;
  storage: StudioStorage;
  jobs: CodexJobManager;
  auth: AuthOptions;
  runtimeDir: string;
  clientDir?: string;
  publicBaseUrl?: string;
}

export function createApp(options: AppOptions): Express {
  const { database, storage, jobs } = options;
  const app = express();
  app.disable("x-powered-by");
  app.set("trust proxy", "loopback");
  app.use((_request, response, next) => {
    response.setHeader("X-Content-Type-Options", "nosniff");
    response.setHeader("Referrer-Policy", "no-referrer");
    response.setHeader("X-Frame-Options", "SAMEORIGIN");
    response.setHeader("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'self'");
    next();
  });
  app.use(express.json({
    limit: "16mb",
    type: (request) => {
      const isAssetUpload = request.method === "POST" && /^\/api\/projects\/[^/]+\/assets\/?$/.test(String(request.url || "").split("?")[0]);
      const contentType = String(request.headers["content-type"] || "").split(";", 1)[0].trim().toLowerCase();
      return !isAssetUpload && (contentType === "application/json" || contentType.endsWith("+json"));
    }
  }));

  app.get("/api/health", (_request, response) => response.json({ ok: true }));
  app.post("/api/session", createSessionHandler(options.auth));

  app.get("/studio-runtime.css", (_request, response, next) => response.sendFile(join(options.runtimeDir, "studio-runtime.css"), next));
  app.get("/studio-runtime.js", (_request, response, next) => response.sendFile(join(options.runtimeDir, "studio-runtime.js"), next));

  const authenticated = express.Router();
  authenticated.use(requireAuth(options.auth));

  authenticated.get("/projects", (request, response) => {
    response.json(database.listProjects().map((project) => projectJson(project, publicBase(request, options.publicBaseUrl))));
  });

  authenticated.post("/projects", asyncHandler(async (request, response) => {
    const name = String(request.body?.name || "새 발표자료").trim().slice(0, 160) || "새 발표자료";
    const id = randomUUID();
    const baseSlug = slugify(String(request.body?.slug || name));
    let slug = baseSlug;
    let suffix = 1;
    while (database.getProjectBySlug(slug)) slug = `${baseSlug.slice(0, 72)}-${++suffix}`;
    const now = new Date().toISOString();
    await storage.createProject(id, name, slug);
    database.insertProject({ id, name, slug, created_at: now, updated_at: now, latest_release_id: null });
    response.status(201).json(projectJson(database.getProject(id)!, publicBase(request, options.publicBaseUrl)));
  }));

  authenticated.get("/projects/:id", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    const workspace = await storage.readWorkspace(storage.projectDir(project.id));
    response.json({
      project: projectJson(project, publicBase(request, options.publicBaseUrl)),
      deck: workspace.deck,
      themeCss: workspace.themeCss,
      animationsCss: workspace.animationsCss,
      codexSettings: jobs.settings(),
      assets: database.listAssets(project.id).map(assetJson),
      jobs: database.listJobs(project.id).map(jobJson),
      versions: database.listVersions(project.id).map((version) => ({ id: version.id, reason: version.reason, createdAt: version.created_at }))
    });
  }));

  authenticated.put("/projects/:id", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    const name = String(request.body?.name || "").trim().slice(0, 160);
    if (!name) throw httpError(400, "Project name is required");
    database.touchProject(project.id, name);
    response.json(projectJson(database.getProject(project.id)!, publicBase(request, options.publicBaseUrl)));
  }));

  authenticated.get("/projects/:id/deck", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    response.json(await storage.readWorkspace(storage.projectDir(project.id)));
  }));

  authenticated.put("/projects/:id/deck", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    const assetIds = new Set(database.listAssets(project.id).map((asset) => asset.id));
    const deck = validateDeck(request.body, assetIds);
    if (deck.slug !== project.slug) throw httpError(400, "Deck slug cannot differ from its project slug");
    await storage.writeDeck(project.id, deck);
    database.touchProject(project.id);
    response.json({ saved: true, updatedAt: database.getProject(project.id)!.updated_at });
  }));

  authenticated.get("/projects/:id/assets", (request, response) => {
    const projectId = param(request, "id");
    requiredProject(database, projectId);
    response.json(database.listAssets(projectId).map(assetJson));
  });

  authenticated.post("/projects/:id/assets", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    const currentBytes = database.listAssets(project.id).reduce((sum, asset) => sum + asset.size, 0);
    const uploaded = await storage.storeUpload(request, project.id, currentBytes);
    const row = { ...uploaded, project_id: project.id, created_at: new Date().toISOString() };
    database.insertAsset(row);
    database.touchProject(project.id);
    response.status(201).json(assetJson(row));
  }));

  authenticated.get("/projects/:id/assets/:assetId/content", asyncHandler(async (request, response) => {
    const asset = database.getAsset(param(request, "id"), param(request, "assetId"));
    if (!asset) throw httpError(404, "Asset not found");
    sendRangedFile(request, response, storage.blobPath(asset.stored_name), asset.mime, asset.size, asset.original_name);
  }));

  authenticated.get("/projects/:id/preview", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    const workspace = await validatedProject(database, storage, project.id);
    response.type("html").send(renderDeckHtml(workspace.deck, {
      title: `${project.name} · Draft`,
      assetBase: `/api/projects/${project.id}/asset-content/`,
      themeHref: `/api/projects/${project.id}/files/theme.css`,
      animationsHref: `/api/projects/${project.id}/files/animations.css`,
      editorPreview: true
    }));
  }));

  authenticated.get("/projects/:id/asset-content/:assetId", asyncHandler(async (request, response) => {
    const asset = database.getAsset(param(request, "id"), param(request, "assetId"));
    if (!asset) throw httpError(404, "Asset not found");
    sendRangedFile(request, response, storage.blobPath(asset.stored_name), asset.mime, asset.size, asset.original_name);
  }));

  authenticated.get("/projects/:id/files/:name", asyncHandler(async (request, response) => {
    const projectId = param(request, "id");
    const name = param(request, "name");
    requiredProject(database, projectId);
    if (!new Set(["theme.css", "animations.css"]).has(name)) throw httpError(404, "File not found");
    response.type("text/css").send(await readFile(join(storage.projectDir(projectId), name), "utf8"));
  }));

  authenticated.post("/projects/:id/ai-jobs", asyncHandler(async (request, response) => {
    const projectId = param(request, "id");
    requiredProject(database, projectId);
    const prompt = String(request.body?.prompt || "").trim();
    if (!prompt) throw httpError(400, "Prompt is required");
    const context = normalizeContext(request.body?.context);
    const job = await jobs.create(projectId, prompt, context, request.body?.screenshotDataUrl);
    response.status(202).json(job);
  }));

  authenticated.get("/ai-jobs/:id", (request, response) => {
    const job = database.getJob(param(request, "id"));
    if (!job) throw httpError(404, "Job not found");
    response.json(jobJson(job));
  });

  authenticated.get("/ai-jobs/:id/events", (request, response) => {
    const job = database.getJob(param(request, "id"));
    if (!job) throw httpError(404, "Job not found");
    response.setHeader("Content-Type", "text/event-stream");
    response.setHeader("Cache-Control", "no-cache, no-transform");
    response.setHeader("Connection", "keep-alive");
    response.flushHeaders();
    jobs.subscribe(job.id, response);
  });

  authenticated.delete("/ai-jobs/:id", (request, response) => {
    if (!jobs.cancel(param(request, "id"))) throw httpError(409, "Job is not active");
    response.status(204).end();
  });

  authenticated.get("/ai-jobs/:id/preview", asyncHandler(async (request, response) => {
    const job = requiredReadyJob(database, param(request, "id"));
    const project = requiredProject(database, job.project_id);
    const assets = database.listAssets(project.id);
    const workspace = await validateWorkspace(
      join(storage.jobDir(job.id), "deck.json"), join(storage.jobDir(job.id), "theme.css"), join(storage.jobDir(job.id), "animations.css"),
      new Set(assets.map((asset) => asset.id))
    );
    response.type("html").send(renderDeckHtml(workspace.deck, {
      title: `${project.name} · Codex Preview`,
      assetBase: `/api/projects/${project.id}/asset-content/`,
      themeHref: `/api/ai-jobs/${job.id}/files/theme.css`,
      animationsHref: `/api/ai-jobs/${job.id}/files/animations.css`,
      editorPreview: true
    }));
  }));

  authenticated.get("/ai-jobs/:id/files/:name", asyncHandler(async (request, response) => {
    const job = requiredReadyJob(database, param(request, "id"));
    const name = param(request, "name");
    if (!new Set(["theme.css", "animations.css"]).has(name)) throw httpError(404, "File not found");
    response.type("text/css").send(await readFile(join(storage.jobDir(job.id), name), "utf8"));
  }));

  authenticated.post("/ai-jobs/:id/accept", asyncHandler(async (request, response) => {
    const job = requiredReadyJob(database, param(request, "id"));
    const snapshot = await storage.snapshot(job.project_id, `Before accepting Codex job ${job.id}`);
    database.insertVersion({ id: snapshot.id, project_id: job.project_id, path: snapshot.path, reason: `Before AI: ${job.prompt.slice(0, 160)}`, created_at: snapshot.createdAt });
    await storage.acceptJob(job.project_id, job.id);
    database.updateJob(job.id, { status: "accepted", error: null });
    database.touchProject(job.project_id);
    response.json({ accepted: true, versionId: snapshot.id, job: jobJson(database.getJob(job.id)!) });
  }));

  authenticated.post("/ai-jobs/:id/reject", (request, response) => {
    const job = requiredReadyJob(database, param(request, "id"));
    database.updateJob(job.id, { status: "rejected", error: null });
    response.json({ rejected: true, job: jobJson(database.getJob(job.id)!) });
  });

  authenticated.post("/projects/:id/versions/:versionId/restore", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    const version = database.getVersion(param(request, "versionId"));
    if (!version || version.project_id !== project.id) throw httpError(404, "Version not found");
    const current = await storage.snapshot(project.id, `Before restoring ${version.id}`);
    database.insertVersion({ id: current.id, project_id: project.id, path: current.path, reason: `Before restore ${version.id}`, created_at: current.createdAt });
    await storage.restoreSnapshot(project.id, version.path);
    database.touchProject(project.id);
    response.json({ restored: true, safetyVersionId: current.id });
  }));

  authenticated.post("/projects/:id/publish", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    await validatedProject(database, storage, project.id);
    const releaseId = randomUUID();
    const path = await storage.publish(project.id, releaseId, database.listAssets(project.id));
    database.insertRelease({ id: releaseId, project_id: project.id, slug: project.slug, path, created_at: new Date().toISOString() });
    database.setLatestRelease(project.id, releaseId);
    response.status(201).json({ releaseId, publicUrl: `${publicBase(request, options.publicBaseUrl)}/p/${project.slug}` });
  }));

  authenticated.get("/projects/:id/export", asyncHandler(async (request, response) => {
    const project = requiredProject(database, param(request, "id"));
    const workspace = await validatedProject(database, storage, project.id);
    const assets = database.listAssets(project.id);
    const assetMap = Object.fromEntries(assets.map((asset) => [asset.id, `assets/${asset.id}-${exportFileName(asset.original_name)}`]));
    const html = renderDeckHtml(workspace.deck, {
      assetMap,
      runtimeCssHref: "./studio-runtime.css",
      runtimeJsHref: "./studio-runtime.js",
      themeHref: "./theme.css",
      animationsHref: "./animations.css"
    });
    response.setHeader("Content-Type", "application/zip");
    response.setHeader("Content-Disposition", `attachment; filename="${project.slug}.zip"`);
    const archive = archiver("zip", { zlib: { level: 9 } });
    archive.on("error", (error) => response.destroy(error));
    archive.pipe(response);
    archive.append(html, { name: "index.html" });
    archive.file(join(options.runtimeDir, "studio-runtime.css"), { name: "studio-runtime.css" });
    archive.file(join(options.runtimeDir, "studio-runtime.js"), { name: "studio-runtime.js" });
    archive.file(join(storage.projectDir(project.id), "theme.css"), { name: "theme.css" });
    archive.file(join(storage.projectDir(project.id), "animations.css"), { name: "animations.css" });
    for (const asset of assets) archive.file(storage.blobPath(asset.stored_name), { name: assetMap[asset.id] });
    await archive.finalize();
  }));

  app.use("/api", authenticated);

  app.get("/p/:slug", asyncHandler(async (request, response) => {
    const project = database.getProjectBySlug(param(request, "slug"));
    if (!project?.latest_release_id) throw httpError(404, "Published deck not found");
    const release = database.getRelease(project.latest_release_id);
    if (!release) throw httpError(404, "Published deck not found");
    const deck = validateDeck(JSON.parse(await readFile(join(release.path, "deck.json"), "utf8")));
    response.type("html").send(renderDeckHtml(deck, {
      assetBase: `/published/${release.id}/assets/`,
      themeHref: `/published/${release.id}/theme.css`,
      animationsHref: `/published/${release.id}/animations.css`
    }));
  }));

  app.get("/published/:releaseId/:file", asyncHandler(async (request, response) => {
    const release = database.getRelease(param(request, "releaseId"));
    const file = param(request, "file");
    if (!release || !new Set(["theme.css", "animations.css"]).has(file)) throw httpError(404, "Published file not found");
    response.setHeader("Cache-Control", "public, max-age=31536000, immutable");
    response.type("text/css").send(await readFile(join(release.path, file), "utf8"));
  }));

  app.get("/published/:releaseId/assets/:assetId", asyncHandler(async (request, response) => {
    const release = database.getRelease(param(request, "releaseId"));
    if (!release) throw httpError(404, "Release not found");
    const manifest = JSON.parse(await readFile(join(release.path, "assets.json"), "utf8")) as Array<{ id: string; originalName: string; mime: string; size: number }>;
    const asset = manifest.find((item) => item.id === param(request, "assetId"));
    if (!asset) throw httpError(404, "Published asset not found");
    response.setHeader("Cache-Control", "public, max-age=31536000, immutable");
    sendRangedFile(request, response, join(release.path, "assets", asset.id), asset.mime, asset.size, asset.originalName);
  }));

  if (options.clientDir) {
    app.use(express.static(options.clientDir, { index: false, maxAge: process.env.NODE_ENV === "production" ? "1h" : 0 }));
    app.get(/^\/(?!api|p\/|published\/|studio-runtime\.).*/, (_request, response, next) => response.sendFile(join(options.clientDir!, "index.html"), next));
  }

  app.use((error: Error & { status?: number }, _request: Request, response: Response, _next: NextFunction) => {
    console.error(error);
    response.status(error.status || 500).json({ error: error.message || "Internal server error" });
  });
  return app;
}

function asyncHandler(handler: (request: Request, response: Response, next: NextFunction) => Promise<void>) {
  return (request: Request, response: Response, next: NextFunction): void => { handler(request, response, next).catch(next); };
}

function requiredProject(database: StudioDatabase, id: string) {
  const project = database.getProject(id);
  if (!project) throw httpError(404, "Project not found");
  return project;
}

function requiredReadyJob(database: StudioDatabase, id: string) {
  const job = database.getJob(id);
  if (!job) throw httpError(404, "Job not found");
  if (job.status !== "ready") throw httpError(409, "Job does not have a reviewable result");
  return job;
}

async function validatedProject(database: StudioDatabase, storage: StudioStorage, projectId: string) {
  return validateWorkspace(
    join(storage.projectDir(projectId), "deck.json"), join(storage.projectDir(projectId), "theme.css"), join(storage.projectDir(projectId), "animations.css"),
    new Set(database.listAssets(projectId).map((asset) => asset.id))
  );
}

function normalizeContext(value: unknown): JobContext {
  const context = value && typeof value === "object" ? value as Record<string, unknown> : {};
  const slideId = String(context.slideId || "").slice(0, 120);
  const selectedObjectIds = Array.isArray(context.selectedObjectIds)
    ? [...new Set(context.selectedObjectIds.map(String).filter(Boolean))].slice(0, 100)
    : [];
  const point = numericRect(context.point, false) as JobContext["point"];
  const region = numericRect(context.region, true) as JobContext["region"];
  return { slideId, selectedObjectIds, ...(point ? { point } : {}), ...(region ? { region } : {}) };
}

function numericRect(value: unknown, includeSize: boolean): Record<string, number> | undefined {
  if (!value || typeof value !== "object") return undefined;
  const source = value as Record<string, unknown>;
  const keys = includeSize ? ["x", "y", "width", "height"] : ["x", "y"];
  const result: Record<string, number> = {};
  for (const key of keys) {
    const number = Number(source[key]);
    if (!Number.isFinite(number)) return undefined;
    result[key] = Math.round(number * 100) / 100;
  }
  return result;
}

function slugify(value: string): string {
  const slug = value.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);
  return slug || `deck-${Date.now().toString(36)}`;
}

function publicBase(request: Request, configured?: string): string {
  return (configured || `${request.protocol}://${request.get("host")}`).replace(/\/$/, "");
}

function param(request: Request, name: string): string {
  const value = request.params[name];
  return Array.isArray(value) ? value[0] : value;
}

function isInlineMime(mime: string): boolean {
  return /^(image|video|audio)\//.test(mime) || mime === "application/pdf";
}

function sendRangedFile(request: Request, response: Response, path: string, mime: string, size: number, name: string): void {
  response.setHeader("Accept-Ranges", "bytes");
  response.setHeader("Content-Type", mime);
  response.setHeader("Content-Disposition", `${isInlineMime(mime) ? "inline" : "attachment"}; filename*=UTF-8''${encodeURIComponent(name)}`);
  const match = String(request.header("range") || "").match(/^bytes=(\d*)-(\d*)$/);
  if (!match) {
    response.setHeader("Content-Length", String(size));
    createReadStream(path).pipe(response);
    return;
  }
  const start = match[1] ? Number(match[1]) : 0;
  const end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || start >= size) {
    response.status(416).setHeader("Content-Range", `bytes */${size}`);
    response.end();
    return;
  }
  response.status(206);
  response.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
  response.setHeader("Content-Length", String(end - start + 1));
  createReadStream(path, { start, end }).pipe(response);
}

function exportFileName(value: string): string {
  return basename(value).normalize("NFKC").replace(/[^\p{L}\p{N}._-]+/gu, "-").slice(0, 140) || "attachment";
}

function httpError(status: number, message: string): Error & { status: number } {
  return Object.assign(new Error(message), { status });
}
