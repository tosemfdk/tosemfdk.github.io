import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { access, copyFile, cp, mkdir, readFile, rename, rm, stat, unlink, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { Request } from "express";
import type { AssetRow } from "./db.js";
import { createDeck, type Deck } from "./schema.js";

const DEFAULT_THEME = `:root {
  --deck-font: Inter, Pretendard, system-ui, sans-serif;
  --deck-accent: #5b7cfa;
  --deck-text: #10172a;
}
.slide-object { font-family: var(--deck-font); color: var(--deck-text); }
`;

const DEFAULT_ANIMATIONS = `@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes slide-in-right {
  from { opacity: 0; transform: translateX(120px); }
  to { opacity: 1; transform: translateX(0); }
}
@keyframes zoom-in {
  from { opacity: 0; transform: scale(.75); }
  to { opacity: 1; transform: scale(1); }
}
`;

export class StudioStorage {
  readonly root: string;
  readonly maxFileBytes: number;
  readonly maxProjectBytes: number;

  constructor(root: string, options: { maxFileBytes?: number; maxProjectBytes?: number } = {}) {
    this.root = root;
    this.maxFileBytes = options.maxFileBytes ?? 2 * 1024 * 1024 * 1024;
    this.maxProjectBytes = options.maxProjectBytes ?? 20 * 1024 * 1024 * 1024;
  }

  async initialize(): Promise<void> {
    await Promise.all(["projects", "blobs", "jobs", "versions", "published", "tmp"].map((entry) => mkdir(join(this.root, entry), { recursive: true })));
  }

  projectDir(projectId: string): string {
    return join(this.root, "projects", projectId);
  }

  jobDir(jobId: string): string {
    return join(this.root, "jobs", jobId);
  }

  blobPath(storedName: string): string {
    return join(this.root, "blobs", storedName);
  }

  async createProject(projectId: string, name: string, slug: string): Promise<void> {
    const directory = this.projectDir(projectId);
    await mkdir(directory, { recursive: true });
    await Promise.all([
      writeFile(join(directory, "deck.json"), JSON.stringify(createDeck(name, slug), null, 2) + "\n"),
      writeFile(join(directory, "theme.css"), DEFAULT_THEME),
      writeFile(join(directory, "animations.css"), DEFAULT_ANIMATIONS)
    ]);
  }

  async readWorkspace(directory: string): Promise<{ deck: unknown; themeCss: string; animationsCss: string }> {
    const [deck, themeCss, animationsCss] = await Promise.all([
      readFile(join(directory, "deck.json"), "utf8"),
      readFile(join(directory, "theme.css"), "utf8"),
      readFile(join(directory, "animations.css"), "utf8")
    ]);
    return { deck: JSON.parse(deck), themeCss, animationsCss };
  }

  async writeDeck(projectId: string, deck: Deck): Promise<void> {
    const target = join(this.projectDir(projectId), "deck.json");
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, JSON.stringify(deck, null, 2) + "\n");
    await rename(temporary, target);
  }

  async prepareJob(jobId: string, projectId: string, assets: AssetRow[], screenshotDataUrl?: string): Promise<string> {
    const directory = this.jobDir(jobId);
    await rm(directory, { recursive: true, force: true });
    await mkdir(directory, { recursive: true });
    for (const file of ["deck.json", "theme.css", "animations.css"]) {
      await copyFile(join(this.projectDir(projectId), file), join(directory, file));
    }
    await writeFile(join(directory, "assets.json"), JSON.stringify(assets.map((asset) => ({
      id: asset.id,
      name: asset.original_name,
      mime: asset.mime,
      size: asset.size
    })), null, 2));
    if (screenshotDataUrl?.startsWith("data:image/png;base64,")) {
      await writeFile(join(directory, "context.png"), Buffer.from(screenshotDataUrl.slice("data:image/png;base64,".length), "base64"));
    }
    return directory;
  }

  async snapshot(projectId: string, reason: string): Promise<{ id: string; path: string; createdAt: string }> {
    const id = randomUUID();
    const createdAt = new Date().toISOString();
    const path = join(this.root, "versions", projectId, id);
    await mkdir(path, { recursive: true });
    for (const file of ["deck.json", "theme.css", "animations.css"]) {
      await copyFile(join(this.projectDir(projectId), file), join(path, file));
    }
    await writeFile(join(path, "meta.json"), JSON.stringify({ reason, createdAt }, null, 2));
    return { id, path, createdAt };
  }

  async restoreSnapshot(projectId: string, snapshotPath: string): Promise<void> {
    for (const file of ["deck.json", "theme.css", "animations.css"]) {
      await copyFile(join(snapshotPath, file), join(this.projectDir(projectId), file));
    }
  }

  async acceptJob(projectId: string, jobId: string): Promise<void> {
    for (const file of ["deck.json", "theme.css", "animations.css"]) {
      await copyFile(join(this.jobDir(jobId), file), join(this.projectDir(projectId), file));
    }
  }

  async publish(projectId: string, releaseId: string, assets: AssetRow[]): Promise<string> {
    const path = join(this.root, "published", releaseId);
    await mkdir(join(path, "assets"), { recursive: true });
    for (const file of ["deck.json", "theme.css", "animations.css"]) {
      await copyFile(join(this.projectDir(projectId), file), join(path, file));
    }
    for (const asset of assets) {
      await copyFile(this.blobPath(asset.stored_name), join(path, "assets", asset.id));
    }
    await writeFile(join(path, "assets.json"), JSON.stringify(assets.map((asset) => ({
      id: asset.id,
      originalName: asset.original_name,
      mime: asset.mime,
      size: asset.size
    })), null, 2));
    return path;
  }

  async storeUpload(request: Request, projectId: string, currentProjectBytes: number): Promise<Omit<AssetRow, "project_id" | "created_at">> {
    const id = randomUUID();
    let suppliedName = String(request.header("x-file-name") || "attachment");
    try { suppliedName = decodeURIComponent(suppliedName); } catch { /* keep the literal header */ }
    const originalName = sanitizeFileName(suppliedName);
    const mime = String(request.header("content-type") || "application/octet-stream").slice(0, 200);
    const temporary = join(this.root, "tmp", `${id}.upload`);
    const hash = createHash("sha256");
    let size = 0;
    const limit = new Transform({
      transform: (chunk: Buffer, _encoding, callback) => {
        size += chunk.length;
        if (size > this.maxFileBytes) return callback(new Error("File exceeds configured upload limit"));
        if (currentProjectBytes + size > this.maxProjectBytes) return callback(new Error("Project exceeds configured storage limit"));
        hash.update(chunk);
        callback(null, chunk);
      }
    });
    try {
      await pipeline(request, limit, createWriteStream(temporary, { flags: "wx" }));
      if (!size) throw new Error("Uploaded file is empty");
      const sha256 = hash.digest("hex");
      const destination = this.blobPath(sha256);
      try {
        await access(destination);
        await unlink(temporary);
      } catch {
        await rename(temporary, destination);
      }
      return { id, original_name: originalName, stored_name: sha256, mime, size, sha256 };
    } catch (error) {
      await rm(temporary, { force: true });
      throw error;
    }
  }

  assetStream(asset: AssetRow) {
    return createReadStream(this.blobPath(asset.stored_name));
  }

  async publishedAsset(releasePath: string, assetId: string): Promise<{ path: string; size: number }> {
    if (!/^[0-9a-f-]{36}$/i.test(assetId)) throw new Error("Invalid asset id");
    const path = join(releasePath, "assets", assetId);
    const details = await stat(path);
    return { path, size: details.size };
  }
}

function sanitizeFileName(value: string): string {
  return basename(value)
    .normalize("NFKC")
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .replace(/[^\p{L}\p{N}._() -]+/gu, "-")
    .slice(0, 180) || "attachment";
}
