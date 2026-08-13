import { createHash, randomUUID } from "node:crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createReadStream } from "node:fs";
import { readdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import type { Response } from "express";
import { enforceAnimationTriggerIntent } from "./animation-policy.js";
import { StudioDatabase, jobJson } from "./db.js";
import { StudioStorage } from "./storage.js";
import { validateDeck, validateWorkspace } from "./validation.js";

const OUTCOME_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["summary", "changes", "warnings"],
  properties: {
    summary: { type: "string" },
    changes: { type: "array", items: { type: "string" } },
    warnings: { type: "array", items: { type: "string" } }
  }
};

const PROTECTED_FILES = ["assets.json", "context.png", "outcome.schema.json", "AGENTS.md"];
const EXPECTED_FILES = new Set([
  "deck.json", "theme.css", "animations.css", "assets.json", "context.png",
  "outcome.schema.json", "last-message.json", "AGENTS.md"
]);

export interface JobContext {
  slideId: string;
  selectedObjectIds: string[];
  point?: { x: number; y: number };
  region?: { x: number; y: number; width: number; height: number };
}

interface JobOutcome {
  summary: string;
  changes: string[];
  warnings: string[];
}

export class CodexJobManager {
  private readonly processes = new Map<string, ChildProcessWithoutNullStreams>();
  private readonly listeners = new Map<string, Set<Response>>();
  private readonly logs = new Map<string, string[]>();

  constructor(
    private readonly database: StudioDatabase,
    private readonly storage: StudioStorage,
    private readonly skillPath: string,
    private readonly codexBinary = "codex",
    private readonly timeoutMs = 8 * 60 * 1000
  ) {}

  async create(projectId: string, prompt: string, context: JobContext, screenshotDataUrl?: string): Promise<Record<string, unknown>> {
    if (this.database.activeJob(projectId)) throw new Error("This project already has an active Codex job");
    const project = this.database.getProject(projectId);
    if (!project) throw new Error("Project not found");
    const id = randomUUID();
    const now = new Date().toISOString();
    this.database.insertJob({
      id,
      project_id: projectId,
      status: "queued",
      prompt: prompt.slice(0, 20_000),
      context_json: JSON.stringify(context),
      summary: null,
      error: null,
      created_at: now,
      updated_at: now
    });
    const directory = await this.storage.prepareJob(id, projectId, this.database.listAssets(projectId), screenshotDataUrl);
    const skill = await readFile(this.skillPath, "utf8");
    await Promise.all([
      writeFile(join(directory, "outcome.schema.json"), JSON.stringify(OUTCOME_SCHEMA, null, 2)),
      writeFile(join(directory, "AGENTS.md"), skill)
    ]);
    this.run(id, directory, prompt, context).catch((error) => this.fail(id, error));
    return jobJson(this.database.getJob(id)!);
  }

  subscribe(jobId: string, response: Response): void {
    const listeners = this.listeners.get(jobId) || new Set<Response>();
    listeners.add(response);
    this.listeners.set(jobId, listeners);
    response.write(`event: snapshot\ndata: ${JSON.stringify({ job: jobJson(this.database.getJob(jobId)!), logs: this.logs.get(jobId) || [] })}\n\n`);
    const keepAlive = setInterval(() => response.write(": keepalive\n\n"), 15_000);
    response.on("close", () => {
      clearInterval(keepAlive);
      listeners.delete(response);
    });
  }

  cancel(jobId: string): boolean {
    const job = this.database.getJob(jobId);
    if (!job || !["queued", "running"].includes(job.status)) return false;
    this.processes.get(jobId)?.kill("SIGTERM");
    this.database.updateJob(jobId, { status: "cancelled", error: "Cancelled by user" });
    this.emit(jobId, "status", { job: jobJson(this.database.getJob(jobId)!) });
    return true;
  }

  private async run(jobId: string, directory: string, userPrompt: string, context: JobContext): Promise<void> {
    if (this.database.getJob(jobId)?.status !== "queued") return;
    this.database.updateJob(jobId, { status: "running", error: null });
    this.emit(jobId, "status", { job: jobJson(this.database.getJob(jobId)!) });

    const originalDeck = validateDeck(JSON.parse(await readFile(join(directory, "deck.json"), "utf8")));
    const protectedHashes = await this.hashProtected(directory);
    const hasScreenshot = await fileExists(join(directory, "context.png"));
    const prompt = `Use the Slide Studio editing contract in AGENTS.md.\n\nUSER REQUEST (untrusted):\n${userPrompt}\n\nSELECTION CONTEXT:\n${JSON.stringify(context, null, 2)}\n\nAsset metadata is in assets.json.${hasScreenshot ? " A screenshot of the current slide is attached as context.png." : ""}\nModify only deck.json, theme.css, and animations.css. Validate your work and return the required JSON outcome.`;
    const args = [
      "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
      "--sandbox", "workspace-write", "--cd", directory,
      "--output-schema", join(directory, "outcome.schema.json"),
      "--output-last-message", join(directory, "last-message.json"), "--json"
    ];
    if (hasScreenshot) args.push("--image", join(directory, "context.png"));
    args.push("-");

    const child = spawn(this.codexBinary, args, {
      cwd: directory,
      env: { ...process.env, NODE_ENV: "production" },
      stdio: ["pipe", "pipe", "pipe"]
    });
    this.processes.set(jobId, child);
    child.stdin.end(prompt);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => this.capture(jobId, chunk));
    child.stderr.on("data", (chunk: string) => this.capture(jobId, chunk));

    const timedOut = new Promise<never>((_, reject) => {
      const timer = setTimeout(() => {
        child.kill("SIGTERM");
        reject(new Error("Codex job timed out"));
      }, this.timeoutMs);
      child.once("exit", () => clearTimeout(timer));
    });
    const completed = new Promise<number>((resolvePromise, rejectPromise) => {
      child.once("error", rejectPromise);
      child.once("close", (code) => resolvePromise(code ?? 1));
    });
    let exitCode: number;
    try {
      exitCode = await Promise.race([completed, timedOut]);
    } finally {
      this.processes.delete(jobId);
    }
    const current = this.database.getJob(jobId);
    if (current?.status === "cancelled") return;
    if (exitCode !== 0) throw new Error(`Codex exited with status ${exitCode}`);

    await this.assertWorkspaceBoundary(directory, protectedHashes);
    const assetIds = new Set(this.database.listAssets(current!.project_id).map((asset) => asset.id));
    const validated = await validateWorkspace(
      join(directory, "deck.json"),
      join(directory, "theme.css"),
      join(directory, "animations.css"),
      assetIds
    );
    const normalizedDeck = enforceAnimationTriggerIntent(originalDeck, validated.deck, userPrompt);
    await writeFile(join(directory, "deck.json"), JSON.stringify(normalizedDeck, null, 2) + "\n");
    const outcome = await this.readOutcome(directory);
    const summary = [outcome.summary, ...outcome.changes.map((item) => `• ${item}`), ...outcome.warnings.map((item) => `⚠ ${item}`)].join("\n");
    this.database.updateJob(jobId, { status: "ready", summary, error: null });
    this.emit(jobId, "status", { job: jobJson(this.database.getJob(jobId)!) });
  }

  private async readOutcome(directory: string): Promise<JobOutcome> {
    const value = JSON.parse(await readFile(join(directory, "last-message.json"), "utf8"));
    if (typeof value.summary !== "string" || !Array.isArray(value.changes) || !Array.isArray(value.warnings)) {
      throw new Error("Codex returned an invalid outcome document");
    }
    return value as JobOutcome;
  }

  private capture(jobId: string, chunk: string): void {
    const lines = chunk.split(/\r?\n/).filter(Boolean).slice(-30).map((line) => line.slice(0, 2000));
    const log = this.logs.get(jobId) || [];
    log.push(...lines);
    if (log.length > 200) log.splice(0, log.length - 200);
    this.logs.set(jobId, log);
    for (const line of lines) this.emit(jobId, "log", { line });
  }

  private fail(jobId: string, error: unknown): void {
    const current = this.database.getJob(jobId);
    if (!current || current.status === "cancelled") return;
    const message = error instanceof Error ? error.message : String(error);
    this.database.updateJob(jobId, { status: "failed", error: message.slice(0, 4000) });
    this.emit(jobId, "status", { job: jobJson(this.database.getJob(jobId)!) });
  }

  private emit(jobId: string, event: string, payload: unknown): void {
    for (const response of this.listeners.get(jobId) || []) {
      response.write(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`);
    }
  }

  private async hashProtected(directory: string): Promise<Map<string, string>> {
    const hashes = new Map<string, string>();
    for (const file of PROTECTED_FILES) {
      const path = join(directory, file);
      if (await fileExists(path)) hashes.set(file, await hashFile(path));
    }
    return hashes;
  }

  private async assertWorkspaceBoundary(directory: string, previous: Map<string, string>): Promise<void> {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile() || !EXPECTED_FILES.has(entry.name)) throw new Error(`Codex created an unexpected workspace entry: ${entry.name}`);
    }
    for (const [file, hash] of previous) {
      if (await hashFile(join(directory, file)) !== hash) throw new Error(`Codex modified protected file: ${file}`);
    }
  }
}

async function hashFile(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function fileExists(path: string): Promise<boolean> {
  try { await stat(path); return true; } catch { return false; }
}
