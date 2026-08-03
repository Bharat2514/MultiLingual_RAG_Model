"use client";

import { useSyncExternalStore } from "react";

/**
 * Session identity.
 *
 * v1 scopes documents by a client-held key (§9). It is persisted so a reload
 * keeps the user's corpus, and it is the *only* thing standing between two
 * users' documents — the backend applies it as a hard filter inside every
 * query, so a fresh key means a genuinely empty workspace.
 *
 * Exposed through `useSyncExternalStore` rather than `useEffect` + `setState`:
 * localStorage is external state, and reading it in an effect causes a
 * cascading render on every mount.
 */

const STORAGE_KEY = "polydoc.session_id";

let cached: string | null = null;
const listeners = new Set<() => void>();

function generate(): string {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `sess_${hex}`;
}

/** Reads (or mints) the id once and caches it, so `getSnapshot` is referentially
 *  stable — returning a fresh value each call would loop forever. */
function snapshot(): string {
  if (cached !== null) return cached;
  let id = window.localStorage.getItem(STORAGE_KEY);
  if (!id) {
    id = generate();
    window.localStorage.setItem(STORAGE_KEY, id);
  }
  cached = id;
  return cached;
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  // Keep tabs in step if the user resets the workspace in another one.
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) {
      cached = event.newValue;
      onChange();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(onChange);
    window.removeEventListener("storage", onStorage);
  };
}

/** Empty during SSR — there is no localStorage on the server, and the hooks
 *  that consume this stay idle until it is non-empty. */
const serverSnapshot = () => "";

export function useSessionId(): string {
  return useSyncExternalStore(subscribe, snapshot, serverSnapshot);
}

export function resetSessionId(): void {
  cached = generate();
  window.localStorage.setItem(STORAGE_KEY, cached);
  for (const listener of listeners) listener();
}
