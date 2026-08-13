import { timingSafeEqual } from "node:crypto";
import type { Request, Response, NextFunction } from "express";

export interface AuthOptions {
  adminEmail?: string;
  adminToken?: string;
  allowDevelopment?: boolean;
  secureCookie?: boolean;
}

function safeEqual(left: string, right: string): boolean {
  const a = Buffer.from(left);
  const b = Buffer.from(right);
  return a.length === b.length && a.length > 0 && timingSafeEqual(a, b);
}

function cookie(request: Request, name: string): string {
  for (const item of String(request.headers.cookie || "").split(";")) {
    const separator = item.indexOf("=");
    if (separator > 0 && item.slice(0, separator).trim() === name) {
      return decodeURIComponent(item.slice(separator + 1).trim());
    }
  }
  return "";
}

export function isAuthenticated(request: Request, options: AuthOptions): boolean {
  if (options.allowDevelopment && !options.adminEmail && !options.adminToken) return true;
  const email = String(request.header("cf-access-authenticated-user-email") || "").toLowerCase();
  if (options.adminEmail && email === options.adminEmail.toLowerCase()) return true;
  const authorization = String(request.header("authorization") || "");
  const bearer = authorization.startsWith("Bearer ") ? authorization.slice(7) : "";
  const provided = bearer || cookie(request, "slide_studio_session");
  return Boolean(options.adminToken && safeEqual(provided, options.adminToken));
}

export function requireAuth(options: AuthOptions) {
  return (request: Request, response: Response, next: NextFunction): void => {
    if (isAuthenticated(request, options)) return next();
    response.status(401).json({ error: "Authentication required" });
  };
}

export function createSessionHandler(options: AuthOptions) {
  return (request: Request, response: Response): void => {
    const token = typeof request.body?.token === "string" ? request.body.token : "";
    if (!options.adminToken || !safeEqual(token, options.adminToken)) {
      response.status(401).json({ error: "Invalid administrator token" });
      return;
    }
    response.setHeader(
      "Set-Cookie",
      `slide_studio_session=${encodeURIComponent(token)}; Path=/; HttpOnly; SameSite=Strict; Max-Age=43200${options.secureCookie ? "; Secure" : ""}`
    );
    response.json({ authenticated: true });
  };
}
