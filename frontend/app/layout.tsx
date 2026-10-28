// Next.js 根布局：统一 HTML 语言、元信息、全局样式与导航外壳；各路由页面作为 children 渲染。
import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/app-shell";

export const metadata: Metadata = {
  title: "DevFlow AI",
  description: "PR / Issue intelligent collaboration agent for engineering teams"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
