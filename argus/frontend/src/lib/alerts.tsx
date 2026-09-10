import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { getJson } from "./api";

/** Live Feed event shape (subset of /api/events/ fields used here). */
export interface FeedEvent {
  id: number;
  behaviour_type: string;
  tier: string;
  risk_score: number;
  status: string;
  confidence: number;
  timestamp: string | null;
  vlm_explanation?: string | null;
}

export interface AlertText {
  event_id: number;
  tier: string;
  language: string;
  translated: boolean;
  text: string;
  base_text: string;
}

export interface Toast {
  id: number;
  event: FeedEvent;
  text: string | null;
  language: string;
  translated: boolean;
}

interface AlertCenterValue {
  /** Events that arrived since mount — Live Feed shows the "+N new" badge. */
  newCount: number;
  clearNewCount: () => void;
  /** Play alert (sound + optional voice + toast) for one event. */
  fireAlert: (event: FeedEvent, options?: { silent?: boolean }) => Promise<void>;
  /** Manually replay a toast for an event (e.g. the bell button on a row). */
  replayAlert: (event: FeedEvent) => void;
  toasts: Toast[];
  dismissToast: (id: number) => void;
  muted: boolean;
  setMuted: (m: boolean) => void;
  lang: string;
  setLang: (l: string) => void;
  languages: Array<{ code: string; name: string }>;
  voiceAvailable: boolean;
}

const AlertCenterContext = createContext<AlertCenterValue | null>(null);

const LANG_KEY = "argus.alertLang";

/** Tier→color tokens kept in sync with tailwind.config argus palette. */
export const TIER_HEX: Record<string, string> = {
  Critical: "#E4483A",
  High: "#F0A63A",
  Medium: "#E8C547",
  Low: "#3FA687",
};

function beep(ctx: AudioContext, at: number, freq: number, duration: number) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.12, at);
  gain.gain.exponentialRampToValueAtTime(0.0001, at + duration);
  osc.connect(gain).connect(ctx.destination);
  osc.start(at);
  osc.stop(at + duration);
}

/** Two-tone alert chirp via WebAudio — no asset file needed for the demo. */
function playChime(tier: string) {
  try {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const t = ctx.currentTime;
    if (tier === "Critical") {
      beep(ctx, t, 880, 0.12);
      beep(ctx, t + 0.15, 880, 0.12);
      beep(ctx, t + 0.3, 660, 0.2);
    } else {
      beep(ctx, t, 660, 0.15);
      beep(ctx, t + 0.2, 880, 0.2);
    }
  } catch {
    /* audio not available — silent */
  }
}

/** Speak text with the browser's SpeechSynthesis in the chosen language. */
function speak(text: string, lang: string) {
  try {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === "en" ? "en-US" : lang;
    u.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  } catch {
    /* no voice — silent */
  }
}

export function AlertCenterProvider({ children }: { children: ReactNode }) {
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const [newCount, setNewCount] = useState(0);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [muted, setMutedState] = useState(false);
  const [lang, setLangState] = useState<string>(() => localStorage.getItem(LANG_KEY) || "en");
  const [languages, setLanguages] = useState<Array<{ code: string; name: string }>>([
    { code: "en", name: "English" },
    { code: "hi", name: "Hindi" },
  ]);
  const knownIds = useRef<Set<number>>(new Set());
  const firstLoad = useRef(true);
  const toastSeq = useRef(0);
  const mutedRef = useRef(muted);
  const langRef = useRef(lang);

  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);
  useEffect(() => {
    langRef.current = lang;
    localStorage.setItem(LANG_KEY, lang);
  }, [lang]);

  // Load available alert languages from the backend (graceful fallback above).
  useEffect(() => {
    getJson<Array<{ code: string; name: string }>>("/api/alerts/languages").then((langs) => {
      if (langs && langs.length > 0) setLanguages(langs);
    });
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const fireAlert = useCallback(async (event: FeedEvent, options?: { silent?: boolean }) => {
    // Toast state
    const toastId = ++toastSeq.current;
    const toast: Toast = {
      id: toastId,
      event,
      text: null,
      language: langRef.current,
      translated: false,
    };
    setToasts((prev) => [toast, ...prev].slice(0, 3));
    window.setTimeout(() => dismissToast(toastId), 10000);

    // Sound (unless muted or silent replay)
    if (!mutedRef.current && !options?.silent) playChime(event.tier);

    // Voice: fetch alert text (translated server-side when configured)
    const at = await getJson<AlertText>(`/api/alerts/text?event_id=${event.id}&lang=${langRef.current}`);
    if (at?.text) {
      setToasts((prev) => prev.map((t) => (t.id === toastId ? { ...t, text: at.text, language: at.language, translated: at.translated } : t)));
      if (!mutedRef.current && !options?.silent) speak(at.text, at.language);
    }
  }, [dismissToast]);

  const replayAlert = useCallback((event: FeedEvent) => {
    void fireAlert(event);
  }, [fireAlert]);

  // Poll the live feed and detect new High/Critical events.
  useEffect(() => {
    let cancelled = false;

    async function poll() {
      const fresh = await getJson<FeedEvent[]>("/api/events/?limit=50");
      if (cancelled || !fresh) return; // keep last known state; LiveFeed handles its own error UI

      const incoming = [...fresh].sort((a, b) => a.id - b.id); // oldest first for detection order
      for (const ev of incoming) {
        if (knownIds.current.has(ev.id)) continue;
        knownIds.current.add(ev.id);
        if (firstLoad.current) continue; // don't alert on the initial backlog
        if (ev.tier === "High" || ev.tier === "Critical") {
          void fireAlert(ev);
          setNewCount((c) => c + 1);
        }
      }
      firstLoad.current = false;
      setEvents(fresh);
    }

    void poll();
    const interval = window.setInterval(poll, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [fireAlert]);

  const value: AlertCenterValue = {
    newCount,
    clearNewCount: () => setNewCount(0),
    fireAlert,
    replayAlert,
    toasts,
    dismissToast,
    muted,
    setMuted: setMutedState,
    lang,
    setLang: setLangState,
    languages,
    voiceAvailable: typeof window !== "undefined" && "speechSynthesis" in window,
  };

  return <AlertCenterContext.Provider value={value}>{children}</AlertCenterContext.Provider>;
}

export function useAlertCenter(): AlertCenterValue {
  const ctx = useContext(AlertCenterContext);
  if (!ctx) throw new Error("useAlertCenter must be used inside AlertCenterProvider");
  return ctx;
}
