"use client";

import * as React from "react";
import { AppHeader } from "@/components/app-header";
import { ChatPanel } from "@/components/chat-panel";
import { DocumentPanel } from "@/components/document-panel";
import { resetSessionId, useSessionId } from "@/lib/session";
import { useChat } from "@/lib/use-chat";
import { useDocuments } from "@/lib/use-documents";

export default function Home() {
  // Empty during SSR — the hooks below stay idle until the client supplies it.
  const sessionId = useSessionId();
  const docs = useDocuments(sessionId);
  const chat = useChat(sessionId);

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

        <div className="min-h-[420px] lg:min-h-0">
          <ChatPanel
            messages={chat.messages}
            busy={chat.busy}
            hasReadyDocuments={docs.ready.length > 0}
            onSend={(query) => chat.send(query)}
            onStop={chat.stop}
            onClear={chat.clear}
          />
        </div>
      </main>
    </div>
  );
}
