// Next.js 构建配置入口；这些选项影响开发/构建流程，不是浏览器中可任意修改的业务状态。
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Keep the dev server output separate from production build artifacts.
  distDir: process.env.NODE_ENV === "development" ? ".next-dev" : ".next",
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
  }
};

module.exports = nextConfig;
