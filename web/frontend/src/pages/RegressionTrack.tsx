import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import AblationsSection from "./regression/AblationsSection";
import ComparisonSection from "./regression/ComparisonSection";
import ForecastSection from "./regression/ForecastSection";
import TiersSection from "./regression/TiersSection";

const SECTIONS = [
  { id: "comparison", title: "Model comparison" },
  { id: "forecast", title: "Forecast for the next month" },
  { id: "tiers", title: "Quantile tiers" },
  { id: "ablations", title: "Ablations" },
];

export default function RegressionTrack() {
  const { hash } = useLocation();

  // Jump to a section when arriving with #section. Content above it grows as data loads, so re-align a few
  // times, and stop as soon as the reader scrolls themselves.
  useEffect(() => {
    if (!hash) return;
    let userScrolled = false;
    const stop = () => {
      userScrolled = true;
    };
    window.addEventListener("wheel", stop, { passive: true });
    window.addEventListener("touchmove", stop, { passive: true });
    window.addEventListener("keydown", stop);
    const align = () => {
      if (!userScrolled) document.getElementById(hash.slice(1))?.scrollIntoView({ block: "start" });
    };
    const timers = [0, 400, 1000, 2000, 3500].map((ms) => window.setTimeout(align, ms));
    return () => {
      timers.forEach(window.clearTimeout);
      window.removeEventListener("wheel", stop);
      window.removeEventListener("touchmove", stop);
      window.removeEventListener("keydown", stop);
    };
  }, [hash]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 md:py-12 lg:px-16">
      <h1 className="font-display text-2xl font-bold text-text-primary sm:text-3xl">📈 Regression track</h1>
      <p className="mt-2 max-w-3xl text-sm text-text-secondary">
        A second, separately frozen model that predicts how many collisions a cell will have next month, built to match the original proposal (regression against a historical-average baseline, LOW/MEDIUM/HIGH tiers). It sits beside the classifier and does not feed into it.
      </p>

      <nav aria-label="Sections on this page" className="mt-4 flex flex-wrap gap-2">
        {SECTIONS.map((s) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className="rounded-pill border border-border bg-card px-3.5 py-1.5 text-sm text-accent hover:bg-card-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {s.title}
          </a>
        ))}
      </nav>

      <div className="mt-8 space-y-12">
        <section id="comparison" aria-labelledby="h-comparison" className="scroll-mt-28">
          <h2 id="h-comparison" className="font-display text-2xl font-semibold text-text-primary">
            Model comparison
          </h2>
          <div className="mt-4">
            <ComparisonSection />
          </div>
        </section>

        <section id="forecast" aria-labelledby="h-forecast" className="scroll-mt-28">
          <h2 id="h-forecast" className="font-display text-2xl font-semibold text-text-primary">
            Forecast for the next month
          </h2>
          <div className="mt-4">
            <ForecastSection />
          </div>
        </section>

        <section id="tiers" aria-labelledby="h-tiers" className="scroll-mt-28">
          <h2 id="h-tiers" className="font-display text-2xl font-semibold text-text-primary">
            Quantile tiers
          </h2>
          <div className="mt-4">
            <TiersSection />
          </div>
        </section>

        <section id="ablations" aria-labelledby="h-ablations" className="scroll-mt-28">
          <h2 id="h-ablations" className="font-display text-2xl font-semibold text-text-primary">
            Ablations
          </h2>
          <p className="mt-2 max-w-3xl text-sm text-text-secondary">
            Extra inputs that were tested on the 2024 validation set against a rule written before looking at results. The 2025 test set was not used for any of them.
          </p>
          <div className="mt-4">
            <AblationsSection />
          </div>
        </section>
      </div>
    </div>
  );
}
