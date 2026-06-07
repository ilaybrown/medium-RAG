import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Medium RAG Assistant",
  description: "RAG assistant for Medium articles assignment",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}