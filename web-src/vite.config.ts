import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

export default defineConfig(({ mode }) => {
  const page = mode === 'extrinsic' ? 'extrinsic' : 'intrinsic'
  return {
    define: {
      __CALIBRATION_PAGE__: JSON.stringify(page),
      'process.env.NODE_ENV': JSON.stringify('production'),
    },
    plugins: [react()],
    build: {
      emptyOutDir: false,
      lib: {
        entry: resolve(import.meta.dirname, 'src/main.tsx'),
        formats: ['es'],
        fileName: () => 'app.js',
        cssFileName: 'styles',
      },
      minify: false,
      outDir: resolve(import.meta.dirname, `../xgc_camera_calibration/web/${page}`),
      rollupOptions: {
        output: {
          assetFileNames: 'styles.[ext]',
          inlineDynamicImports: true,
        },
      },
    },
  }
})
