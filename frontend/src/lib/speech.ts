"use client";

import { useCallback, useRef, useState, useSyncExternalStore } from "react";

/**
 * Voice in and out through the browser's own Web Speech API — no model to
 * download and no audio sent to our backend.
 *
 * Reading Devanagari on a phone keyboard is the barrier for a lot of the people
 * these documents are written for (scheme applicants, pensioners). Speaking the
 * question in Hindi and hearing the answer read back removes it. Recognition
 * runs in `hi-IN` or `en-IN`; both handle code-mixed Hinglish reasonably, and
 * the answer is read in the voice that matches its script.
 *
 * Browser support varies (Chrome/Edge/Safari recognise speech; Firefox does
 * not), so every capability is feature-detected and the UI hides what is missing.
 */

type RecognitionCtor = new () => SpeechRecognitionLike;

interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
}

function recognitionCtor(): RecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: RecognitionCtor;
    webkitSpeechRecognition?: RecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export type SpeechLang = "hi-IN" | "en-IN";

// Browser capabilities never change during a page's life, so the "store" has
// no updates to subscribe to; the server snapshot is `false` (no window).
const noSubscribe = () => () => {};
const useCapability = (probe: () => boolean) => useSyncExternalStore(noSubscribe, probe, () => false);
const canRecognise = () => recognitionCtor() !== null;
const canSpeak = () => "speechSynthesis" in window;

export function useDictation(onText: (text: string) => void) {
  const [listening, setListening] = useState(false);
  const supported = useCapability(canRecognise);
  const [error, setError] = useState<string | null>(null);
  const recognition = useRef<SpeechRecognitionLike | null>(null);

  const start = useCallback(
    (lang: SpeechLang) => {
      const Ctor = recognitionCtor();
      if (!Ctor) return;
      const r = new Ctor();
      r.lang = lang;
      r.interimResults = false;
      r.continuous = false;
      r.onresult = (event) => {
        const text = Array.from(event.results)
          .map((result) => result[0]?.transcript ?? "")
          .join(" ")
          .trim();
        if (text) onText(text);
      };
      r.onerror = (event) => {
        setError(event.error === "not-allowed" ? "Microphone permission was denied." : null);
        setListening(false);
      };
      r.onend = () => setListening(false);
      recognition.current = r;
      setError(null);
      setListening(true);
      r.start();
    },
    [onText],
  );

  const stop = useCallback(() => {
    recognition.current?.stop();
    setListening(false);
  }, []);

  return { supported, listening, error, start, stop };
}

/** Strip Markdown and citation markers so the voice reads prose, not syntax. */
function speakable(text: string): string {
  return text
    .replace(/\[[^\]]{1,120}\]/g, "")
    .replace(/[*_`#>]+/g, "")
    .replace(/^\s*[-•]\s+/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function useSpeaker() {
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const supported = useCapability(canSpeak);

  const speak = useCallback((id: string, text: string) => {
    const synth = window.speechSynthesis;
    synth.cancel();
    const utterance = new SpeechSynthesisUtterance(speakable(text));
    const hindi = /[ऀ-ॿ]/.test(text);
    utterance.lang = hindi ? "hi-IN" : "en-IN";
    const voice = synth.getVoices().find((v) => v.lang === utterance.lang);
    if (voice) utterance.voice = voice;
    utterance.onend = () => setSpeakingId(null);
    utterance.onerror = () => setSpeakingId(null);
    setSpeakingId(id);
    synth.speak(utterance);
  }, []);

  const stop = useCallback(() => {
    window.speechSynthesis.cancel();
    setSpeakingId(null);
  }, []);

  return { supported, speakingId, speak, stop };
}
