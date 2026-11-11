import path from "path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: { port: 5180, strictPort: true, proxy: { "/api": process.env.CORNSIGHT_API ?? "http://127.0.0.1:8010" } },
});
