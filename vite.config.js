import { defineConfig } from 'vite';

export default defineConfig({
  // 用相对路径产出，dist 放到任意子目录 / 静态托管都能直接跑
  base: './',
  resolve: {
    alias: [
      // 与原文件 importmap 保持一致：'three' 也指向 WebGPU 构建，
      // 这样 three、three/webgpu、three/addons/* 共用同一份实例
      { find: /^three$/, replacement: 'three/webgpu' },
    ],
    dedupe: ['three'],
  },
  server: {
    port: 5173,
    host: true,
  },
  build: {
    // three/webgpu + TSL 用到了较新的语法特性
    target: 'esnext',
    chunkSizeWarningLimit: 4096,
  },
});
