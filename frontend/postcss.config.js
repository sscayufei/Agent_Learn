// CSS 编译流水线：由 PostCSS 调用 Tailwind 与 Autoprefixer，源码样式不直接等同于最终输出 CSS。
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}
  }
};
