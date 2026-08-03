"use client";

import { Languages, Moon, RefreshCw, Sun } from "lucide-react";
import { Button } from "@/components/ui/primitives";
import { setTheme, useTheme } from "@/lib/theme";

function ThemeToggle() {
  const theme = useTheme();
  const isDark = theme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
      aria-label="Toggle colour theme"
    >
      {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}

export function AppHeader({
  sessionId,
  onNewSession,
}: {
  sessionId: string;
  onNewSession: () => void;
}) {
  return (
    <header className="flex items-center gap-3 border-b border-border bg-surface/80 px-4 py-3 backdrop-blur">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-white">
        <Languages className="h-4 w-4" aria-hidden />
      </div>

      <div className="min-w-0">
        <h1 className="text-[15px] font-semibold leading-tight text-ink">PolyDoc</h1>
        <p className="text-[11px] leading-tight text-ink-faint">
          Hindi · English · Hinglish document assistant
        </p>
      </div>

      <div className="ml-auto flex items-center gap-1">
        <span
          className="hidden font-mono text-[11px] text-ink-faint sm:inline"
          title="Your workspace. Documents are scoped to this key."
        >
          {sessionId.slice(0, 13)}…
        </span>
        <Button
          variant="ghost"
          size="icon"
          onClick={onNewSession}
          title="Start a new workspace — your documents stay, but this session no longer sees them"
          aria-label="Start a new workspace"
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
        <ThemeToggle />
      </div>
    </header>
  );
}
