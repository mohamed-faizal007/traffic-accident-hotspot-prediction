/**
 * Slow-drifting mesh-gradient blobs in the tier palette, purely atmospheric (no data is plotted
 * here — the Risk Map page is where real geography lives). CSS-only so it costs nothing on the
 * main thread, and prefers-reduced-motion disables the animation via index.css.
 */
export default function HeroCanvas() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="hero-blob hero-blob-critical" />
      <div className="hero-blob hero-blob-high" />
      <div className="hero-blob hero-blob-medium" />
      <style>{`
        .hero-blob {
          position: absolute;
          border-radius: 50%;
          filter: blur(60px);
          opacity: 0.18;
          mix-blend-mode: screen;
        }
        .hero-blob-critical {
          top: -10%;
          left: 5%;
          width: 42vw;
          height: 42vw;
          background: var(--color-tier-critical);
          animation: drift-a 48s ease-in-out infinite alternate;
        }
        .hero-blob-high {
          top: 15%;
          right: 0%;
          width: 36vw;
          height: 36vw;
          background: var(--color-tier-high);
          animation: drift-b 56s ease-in-out infinite alternate;
        }
        .hero-blob-medium {
          bottom: -15%;
          left: 30%;
          width: 38vw;
          height: 38vw;
          background: var(--color-tier-medium);
          opacity: 0.14;
          animation: drift-c 64s ease-in-out infinite alternate;
        }
        @keyframes drift-a {
          from { transform: translate(0, 0) scale(1); }
          to { transform: translate(6%, 8%) scale(1.08); }
        }
        @keyframes drift-b {
          from { transform: translate(0, 0) scale(1); }
          to { transform: translate(-8%, 6%) scale(0.95); }
        }
        @keyframes drift-c {
          from { transform: translate(0, 0) scale(1); }
          to { transform: translate(5%, -6%) scale(1.05); }
        }
      `}</style>
    </div>
  );
}
