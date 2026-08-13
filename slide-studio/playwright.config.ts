import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:5173", trace: "retain-on-failure" },
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:5173",
    timeout: 120_000,
    reuseExistingServer: true,
    env: { SLIDE_STUDIO_DATA_DIR: "/tmp/tosemfdk-slide-studio-e2e" }
  }
});
