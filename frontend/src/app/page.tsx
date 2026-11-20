"use client";

import * as React from "react";
import { AppHeader } from "@/components/app-header";
import { ChatPanel } from "@/components/chat-panel";
import { DocumentPanel } from "@/components/document-panel";
import { ToolsPanel } from "@/components/tools-panel";
import { resetSessionId, useSessionId } from "@/lib/session";
import { useChat } from "@/lib/use-chat";
import { useDocuments } from "@/lib/use-documents";

export default function Home() {
  // Empty during SSR — the hooks below stay idle until the client supplies it.
  const sessionId = useSessionId();
  const docs = useDocuments(sessionId);
  const chat = useChat(sessionId);
  const [view, setView] = React.useState<"ask" | "tools">("ask");

  const startNewSession = React.useCallback(() => {
    chat.clear();
    resetSessionId();
  }, [chat]);

  return (
    <div className="flex h-dvh flex-col bg-canvas">
      <AppHeader sessionId={sessionId} onNewSession={startNewSession} />

      <main className="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[minmax(280px,340px)_1fr]">
        {/* On narrow screens the document panel sits above the chat and scrolls
            with the page; from lg up the two columns scroll independently. */}
        <div className="min-h-[220px] lg:min-h-0">
          <DocumentPanel
            documents={docs.documents}
            loading={docs.loading}
            uploading={docs.uploading}
            error={docs.error}
            onUpload={docs.upload}
            onDelete={docs.remove}
            onDismissError={docs.clearError}
          />
        </div>

        <div className="flex min-h-[420px] flex-col gap-2 lg:min-h-0">
          <div
            role="tablist"
            aria-label="Workspace view"
            className="flex w-fit shrink-0 rounded-lg border border-border bg-surface p-0.5 text-[12.5px]"
          >
            {(
              [
                ["ask", "Ask"],
                ["tools", "Document tools"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={view === id}
                onClick={() => setView(id)}
                className={
                  view === id
                    ? "rounded-md bg-accent px-3 py-1 font-medium text-white"
                    : "rounded-md px-3 py-1 text-ink-muted hover:text-ink"
                }
              >
                {label}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1">
            {/* Both stay mounted so switching views keeps the conversation. */}
            <div className={view === "ask" ? "h-full" : "hidden"}>
              <ChatPanel
                sessionId={sessionId}
                messages={chat.messages}
                busy={chat.busy}
                hasReadyDocuments={docs.ready.length > 0}
                style={chat.style}
                onStyleChange={chat.setStyle}
                onSend={(query) => chat.send(query)}
                onStop={chat.stop}
                onClear={chat.clear}
                onRate={chat.rate}
                onShare={chat.share}
              />
            </div>
            {view === "tools" ? <ToolsPanel sessionId={sessionId} documents={docs.ready} /> : null}
          </div>
        </div>
      </main>
    </div>
  );
}
