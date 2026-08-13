import type { AiJob, Asset, Deck, Project } from "./types";

export class AuthenticationError extends Error {}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body && typeof options.body === "string" ? { "Content-Type": "application/json" } : {}),
      ...options.headers
    }
  });
  if (response.status === 401) throw new AuthenticationError("Authentication required");
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export interface ProjectDetail {
  project: Project;
  deck: Deck;
  themeCss: string;
  animationsCss: string;
  codexSettings: { model: string; reasoningEffort: string; serviceTier: string };
  assets: Asset[];
  jobs: AiJob[];
  versions: Array<{ id: string; reason: string; createdAt: string }>;
}

export const api = {
  login: (token: string) => request<{ authenticated: true }>("/api/session", { method: "POST", body: JSON.stringify({ token }) }),
  listProjects: () => request<Project[]>("/api/projects"),
  createProject: (name: string) => request<Project>("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),
  getProject: (id: string) => request<ProjectDetail>(`/api/projects/${id}`),
  saveDeck: (id: string, deck: Deck) => request<{ saved: true; updatedAt: string }>(`/api/projects/${id}/deck`, { method: "PUT", body: JSON.stringify(deck) }),
  uploadAsset: async (projectId: string, file: File): Promise<Asset> => request<Asset>(`/api/projects/${projectId}/assets`, {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream", "X-File-Name": encodeURIComponent(file.name) },
    body: file
  }),
  createJob: (projectId: string, body: { prompt: string; context: unknown; screenshotDataUrl?: string }) =>
    request<AiJob>(`/api/projects/${projectId}/ai-jobs`, { method: "POST", body: JSON.stringify(body) }),
  getJob: (id: string) => request<AiJob>(`/api/ai-jobs/${id}`),
  cancelJob: (id: string) => request<void>(`/api/ai-jobs/${id}`, { method: "DELETE" }),
  acceptJob: (id: string) => request(`/api/ai-jobs/${id}/accept`, { method: "POST", body: "{}" }),
  rejectJob: (id: string) => request(`/api/ai-jobs/${id}/reject`, { method: "POST", body: "{}" }),
  publish: (projectId: string) => request<{ releaseId: string; publicUrl: string }>(`/api/projects/${projectId}/publish`, { method: "POST", body: "{}" }),
  restoreVersion: (projectId: string, versionId: string) => request(`/api/projects/${projectId}/versions/${versionId}/restore`, { method: "POST", body: "{}" })
};

export function assetContentUrl(projectId: string, assetId: string): string {
  return `/api/projects/${projectId}/asset-content/${assetId}`;
}
