import Database from "better-sqlite3";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

export interface ProjectRow {
  id: string;
  name: string;
  slug: string;
  created_at: string;
  updated_at: string;
  latest_release_id: string | null;
}

export interface AssetRow {
  id: string;
  project_id: string;
  original_name: string;
  stored_name: string;
  mime: string;
  size: number;
  sha256: string;
  created_at: string;
}

export interface JobRow {
  id: string;
  project_id: string;
  status: string;
  prompt: string;
  context_json: string;
  summary: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ReleaseRow {
  id: string;
  project_id: string;
  slug: string;
  path: string;
  created_at: string;
}

export class StudioDatabase {
  readonly raw: Database.Database;

  constructor(path: string) {
    mkdirSync(dirname(path), { recursive: true });
    this.raw = new Database(path);
    this.raw.pragma("journal_mode = WAL");
    this.raw.pragma("foreign_keys = ON");
    this.migrate();
  }

  private migrate(): void {
    this.raw.exec(`
      CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        slug TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        latest_release_id TEXT
      );
      CREATE TABLE IF NOT EXISTS assets (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        original_name TEXT NOT NULL,
        stored_name TEXT NOT NULL,
        mime TEXT NOT NULL,
        size INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS assets_project_id ON assets(project_id);
      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        prompt TEXT NOT NULL,
        context_json TEXT NOT NULL,
        summary TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS jobs_project_id ON jobs(project_id, created_at DESC);
      CREATE TABLE IF NOT EXISTS versions (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        path TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS releases (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        slug TEXT NOT NULL,
        path TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
    `);
    this.raw.prepare("UPDATE jobs SET status = 'failed', error = 'Server restarted during job', updated_at = ? WHERE status IN ('queued', 'running')")
      .run(new Date().toISOString());
  }

  close(): void {
    this.raw.close();
  }

  listProjects(): ProjectRow[] {
    return this.raw.prepare("SELECT * FROM projects ORDER BY updated_at DESC").all() as ProjectRow[];
  }

  getProject(id: string): ProjectRow | undefined {
    return this.raw.prepare("SELECT * FROM projects WHERE id = ?").get(id) as ProjectRow | undefined;
  }

  getProjectBySlug(slug: string): ProjectRow | undefined {
    return this.raw.prepare("SELECT * FROM projects WHERE slug = ?").get(slug) as ProjectRow | undefined;
  }

  insertProject(row: ProjectRow): void {
    this.raw.prepare(`INSERT INTO projects (id, name, slug, created_at, updated_at, latest_release_id)
      VALUES (@id, @name, @slug, @created_at, @updated_at, @latest_release_id)`).run(row);
  }

  touchProject(id: string, name?: string): void {
    const now = new Date().toISOString();
    if (name) this.raw.prepare("UPDATE projects SET name = ?, updated_at = ? WHERE id = ?").run(name, now, id);
    else this.raw.prepare("UPDATE projects SET updated_at = ? WHERE id = ?").run(now, id);
  }

  setLatestRelease(projectId: string, releaseId: string): void {
    this.raw.prepare("UPDATE projects SET latest_release_id = ?, updated_at = ? WHERE id = ?")
      .run(releaseId, new Date().toISOString(), projectId);
  }

  listAssets(projectId: string): AssetRow[] {
    return this.raw.prepare("SELECT * FROM assets WHERE project_id = ? ORDER BY created_at DESC").all(projectId) as AssetRow[];
  }

  getAsset(projectId: string, assetId: string): AssetRow | undefined {
    return this.raw.prepare("SELECT * FROM assets WHERE project_id = ? AND id = ?").get(projectId, assetId) as AssetRow | undefined;
  }

  insertAsset(row: AssetRow): void {
    this.raw.prepare(`INSERT INTO assets (id, project_id, original_name, stored_name, mime, size, sha256, created_at)
      VALUES (@id, @project_id, @original_name, @stored_name, @mime, @size, @sha256, @created_at)`).run(row);
  }

  insertJob(row: JobRow): void {
    this.raw.prepare(`INSERT INTO jobs (id, project_id, status, prompt, context_json, summary, error, created_at, updated_at)
      VALUES (@id, @project_id, @status, @prompt, @context_json, @summary, @error, @created_at, @updated_at)`).run(row);
  }

  getJob(id: string): JobRow | undefined {
    return this.raw.prepare("SELECT * FROM jobs WHERE id = ?").get(id) as JobRow | undefined;
  }

  listJobs(projectId: string): JobRow[] {
    return this.raw.prepare("SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at DESC LIMIT 30").all(projectId) as JobRow[];
  }

  activeJob(projectId: string): JobRow | undefined {
    return this.raw.prepare("SELECT * FROM jobs WHERE project_id = ? AND status IN ('queued', 'running') LIMIT 1")
      .get(projectId) as JobRow | undefined;
  }

  updateJob(id: string, values: { status: string; summary?: string | null; error?: string | null }): void {
    this.raw.prepare(`UPDATE jobs SET status = ?, summary = COALESCE(?, summary), error = ?, updated_at = ? WHERE id = ?`)
      .run(values.status, values.summary ?? null, values.error ?? null, new Date().toISOString(), id);
  }

  insertVersion(row: { id: string; project_id: string; path: string; reason: string; created_at: string }): void {
    this.raw.prepare("INSERT INTO versions (id, project_id, path, reason, created_at) VALUES (@id, @project_id, @path, @reason, @created_at)")
      .run(row);
  }

  listVersions(projectId: string): Array<{ id: string; project_id: string; path: string; reason: string; created_at: string }> {
    return this.raw.prepare("SELECT * FROM versions WHERE project_id = ? ORDER BY created_at DESC LIMIT 50").all(projectId) as Array<{ id: string; project_id: string; path: string; reason: string; created_at: string }>;
  }

  getVersion(id: string): { id: string; project_id: string; path: string; reason: string; created_at: string } | undefined {
    return this.raw.prepare("SELECT * FROM versions WHERE id = ?").get(id) as { id: string; project_id: string; path: string; reason: string; created_at: string } | undefined;
  }

  insertRelease(row: ReleaseRow): void {
    this.raw.prepare("INSERT INTO releases (id, project_id, slug, path, created_at) VALUES (@id, @project_id, @slug, @path, @created_at)").run(row);
  }

  getRelease(id: string): ReleaseRow | undefined {
    return this.raw.prepare("SELECT * FROM releases WHERE id = ?").get(id) as ReleaseRow | undefined;
  }
}

export function projectJson(row: ProjectRow, publicBase = ""): Record<string, unknown> {
  return {
    id: row.id,
    name: row.name,
    slug: row.slug,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    latestReleaseId: row.latest_release_id,
    publicUrl: row.latest_release_id ? `${publicBase}/p/${row.slug}` : null
  };
}

export function assetJson(row: AssetRow): Record<string, unknown> {
  return {
    id: row.id,
    projectId: row.project_id,
    originalName: row.original_name,
    mime: row.mime,
    size: row.size,
    sha256: row.sha256,
    createdAt: row.created_at
  };
}

export function jobJson(row: JobRow): Record<string, unknown> {
  return {
    id: row.id,
    projectId: row.project_id,
    status: row.status,
    prompt: row.prompt,
    context: JSON.parse(row.context_json),
    summary: row.summary,
    error: row.error,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}
