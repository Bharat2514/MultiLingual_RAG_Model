"use client";

import { useSyncExternalStore } from "react";

/**
 * Colour theme, stored in localStorage and stamped onto `<html data-theme>`.
 *
 * An external store rather than `useEffect` + `setState`, for the same reason
 * as the session id: reading a browser API in an effect forces a second render
 * on every mount.
 */

export type Theme = "light" | "dark";

const STORAGE_KEY = "polydoc.theme";

let cached: Theme | null = null;
const listeners = new Set<() => void>();

function snapshot(): Theme {
  if (cached !== null) return cached;
  const stored = window.localStorage.getItem(STORAGE_KEY);
  cached =
    stored === "light" || stored === "dark"
      ? stored
      : window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
  return cached;
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  // Follow the OS only while the user has not made an explicit choice.
  const onMedia = () => {
    if (window.localStorage.getItem(STORAGE_KEY)) return;
    cached = media.matches ? "dark" : "light";
    onChange();
  };
  media.addEventListener("change", onMedia);
  return () => {
    listeners.delete(onChange);
    media.removeEventListener("change", onMedia);
  };
}

/** The inline script in `layout.tsx` has already applied the stored theme before
 *  hydration, so "light" here is only a placeholder for markup generation. */
const serverSnapshot = (): Theme => "light";

export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, snapshot, serverSnapshot);
}

export function setTheme(theme: Theme): void {
  cached = theme;
  window.localStorage.setItem(STORAGE_KEY, theme);
  document.documentElement.setAttribute("data-theme", theme);
  for (const listener of listeners) listener();
}
