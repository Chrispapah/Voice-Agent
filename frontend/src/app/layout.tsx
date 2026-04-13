import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI SDR - Voicebot Builder",
  description: "Self-service platform for creating AI voice agents",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
