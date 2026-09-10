/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "-apple-system", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        argus: {
          dark: "#0f172a",
          card: "#1e293b",
          border: "#334155",
          // UI accent (nav, buttons, focus) — deliberately NOT a risk color.
          // Amber is reserved EXCLUSIVELY for the High risk tier so badge,
          // chart and nav colors never collide (design review fix).
          accent: "#3b82f6",
          danger: "#ef4444",
          success: "#22c55e",
          info: "#3b82f6",
          // Risk tier palette — the ONLY sources of tier color in the UI.
          critical: "#ef4444",
          high: "#f59e0b",
          medium: "#eab308",
          low: "#22c55e",
        },
      },
    },
  },
  plugins: [],
};
