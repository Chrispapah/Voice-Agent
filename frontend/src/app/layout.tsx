import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Akoi",
  description: "Akoi — self-service platform for creating AI voice agents",
  icons: { icon: "/akoi-mark.png" },
  openGraph: {
    title: "Akoi",
    description: "Akoi — self-service platform for creating AI voice agents",
    images: ["/akoi-mark.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
