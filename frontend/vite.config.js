import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxying /api and /ws to the backend means the dashboard uses relative URLs
// everywhere -- no CORS in development, and nothing to change when the whole
// thing is served from one origin later.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
