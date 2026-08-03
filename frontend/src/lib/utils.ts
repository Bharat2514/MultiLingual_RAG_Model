import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

const LANGUAGE_LABELS: Record<string, string> = {
  hi: "हिन्दी",
  en: "English",
  hinglish: "Hinglish",
  mixed: "Mixed",
  unknown: "—",
};

export function languageLabel(code: string): string {
  return LANGUAGE_LABELS[code] ?? code;
}

/** Devanagari needs a taller line-height and a font that actually has the
 *  glyphs, so text direction-agnostic styling is applied per string (NFR-10). */
export function hasDevanagari(text: string): boolean {
  return /[ऀ-ॿ]/.test(text);
}
