import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createApp } from "./app.js";
import { CodexJobManager } from "./codex.js";
import { StudioDatabase } from "./db.js";
import { StudioStorage } from "./storage.js";

const moduleDir = dirname(fileURLToPath(import.meta.url));
const studioRoot = resolve(moduleDir, "..");
const repositoryRoot = resolve(studioRoot, "..");
const dataRoot = resolve(process.env.SLIDE_STUDIO_DATA_DIR || resolve(repositoryRoot, ".slide-studio-data"));
const port = Number(process.env.SLIDE_STUDIO_PORT || 5560);
const host = process.env.SLIDE_STUDIO_HOST || "127.0.0.1";
const production = process.env.NODE_ENV === "production";

const auth = {
  adminEmail: process.env.SLIDE_STUDIO_ADMIN_EMAIL,
  adminToken: process.env.SLIDE_STUDIO_ADMIN_TOKEN,
  allowDevelopment: !production,
  secureCookie: production
};

if (production && !auth.adminEmail && !auth.adminToken) {
  throw new Error("Production requires SLIDE_STUDIO_ADMIN_EMAIL or SLIDE_STUDIO_ADMIN_TOKEN");
}

const storage = new StudioStorage(dataRoot, {
  maxFileBytes: parseBytes(process.env.SLIDE_STUDIO_MAX_FILE_BYTES, 2 * 1024 ** 3),
  maxProjectBytes: parseBytes(process.env.SLIDE_STUDIO_MAX_PROJECT_BYTES, 20 * 1024 ** 3)
});
await storage.initialize();
const database = new StudioDatabase(resolve(dataRoot, "studio.sqlite3"));
const jobs = new CodexJobManager(
  database,
  storage,
  resolve(repositoryRoot, ".codex/skills/slide-studio/SKILL.md"),
  process.env.SLIDE_STUDIO_CODEX_BINARY || "codex",
  Number(process.env.SLIDE_STUDIO_CODEX_TIMEOUT_MS || 8 * 60 * 1000)
);
const runtimeDir = production ? resolve(studioRoot, "runtime") : resolve(studioRoot, "runtime");
const clientDir = production ? resolve(studioRoot, "dist/client") : undefined;
const app = createApp({
  database,
  storage,
  jobs,
  auth,
  runtimeDir,
  clientDir,
  publicBaseUrl: process.env.SLIDE_STUDIO_PUBLIC_URL
});

const server = app.listen(port, host, () => console.log(`Slide Studio listening on http://${host}:${port}`));
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => server.close(() => { database.close(); process.exit(0); }));
}

function parseBytes(value: string | undefined, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}
