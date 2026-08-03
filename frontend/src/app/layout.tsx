import type { Metadata, Viewport } from "next";
import { Inter, Noto_Sans_Devanagari } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

/*
 * Without a Devanagari-capable webfont the browser falls back to whatever the
 * OS provides, which on many Windows installs renders matras and conjuncts
 * incorrectly — unacceptable for a product whose whole point is Hindi (NFR-10).
 */
const devanagari = Noto_Sans_Devanagari({
  subsets: ["devanagari", "latin"],
  weight: ["400", "500", "600"],
  variable: "--font-devanagari",
  display: "swap",
});

export const metadata: Metadata = {
  title: "PolyDoc — Multilingual Document Assistant",
  description:
    "Ask questions in Hindi, English, or Hinglish about your PDFs and Word documents, and get cited answers.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#1a1c20" },
  ],
};

/**
 * Applies the stored theme before first paint. Without this the page renders
 * in the system theme and then snaps to the user's choice — a visible flash.
 */
const THEME_BOOTSTRAP = `
(function () {
  try {
    var stored = localStorage.getItem("polydoc.theme");
    if (stored === "light" || stored === "dark") {
      document.documentElement.setAttribute("data-theme", stored);
    }
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
      </head>
      <body className={`${inter.variable} ${devanagari.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
