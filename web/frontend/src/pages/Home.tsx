import { api } from "../api";
import HeroCanvas from "../components/HeroCanvas";
import InfoCard from "../components/InfoCard";
import MetricCard from "../components/MetricCard";
import { ErrorState, LoadingState } from "../components/PageState";
import { useAsync } from "../hooks/useAsync";

function pct(x: number, digits = 1) {
  return `${(100 * x).toFixed(digits)}%`;
}

export default function Home() {
  const { data, error, loading } = useAsync(() => api.summary(), []);

  if (loading) return <LoadingState />;
  if (error || !data) return <ErrorState message={error ?? "Failed to load summary."} />;

  const fractions = data.tier_top_fractions;

  return (
    <div>
      <section className="relative overflow-hidden">
        <HeroCanvas />
        <div className="relative mx-auto flex min-h-[70vh] max-w-7xl flex-col justify-center px-6 py-24 lg:px-16">
          <h1 className="font-display text-[clamp(2.75rem,6vw,4.75rem)] font-bold leading-[1.05] text-text-primary">
            Traffic Accident <span className="text-accent-amber">Hotspot</span> Prediction
          </h1>
          <p className="mt-6 max-w-xl text-lg text-text-secondary">
            UK road-safety collisions 2021&ndash;2025, aggregated to 500&nbsp;m British National Grid cells and
            months &mdash; ranking the risk that each modelled cell has at least two collisions next month.
          </p>
          {data.frozen?.status === "frozen" && (
            <span className="mt-8 inline-flex items-center gap-2 self-start rounded-pill border border-border bg-card px-3.5 py-1.5 font-mono text-xs text-text-secondary">
              🔒 frozen {data.frozen.config_sha256?.slice(0, 10)} · {data.frozen.frozen_at_utc?.slice(0, 10)}
            </span>
          )}
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-6 pb-20 lg:px-16">
        <p className="mb-8 max-w-3xl text-text-secondary">
          DBSCAN is used to define and analyse hotspot structure; the cluster features tested in an ablation did{" "}
          <strong className="text-text-primary">not</strong> improve prediction and are not used in the final model.
        </p>

        <div className="mb-2 text-sm font-medium text-text-secondary">{data.split_label}</div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="Base rate (hotspot cell-months)" value={pct(data.base_rate, 2)} />
          <MetricCard
            label="Average precision"
            value={data.average_precision.toFixed(3)}
            sub={`${data.ap_lift_over_base_rate.toFixed(1)}x base rate`}
          />
          <MetricCard
            label="Precision in top 1% each month"
            value={pct(data.top_1pct_precision)}
            sub={`${data.top_1pct_lift.toFixed(1)}x base rate`}
          />
          <MetricCard label="Same, trailing-12-month baseline" value={pct(data.baseline_trailing12_top_1pct_precision)} />
        </div>

        <div className="mt-6 grid grid-cols-1 gap-4">
          <InfoCard tone="info">
            Average precision is {data.average_precision.toFixed(3)} versus {data.baseline_trailing12_average_precision.toFixed(3)}{" "}
            for simply ranking cells by their collisions in the last 12 months ({data.ap_gain_over_trailing12 >= 0 ? "+" : ""}
            {data.ap_gain_over_trailing12.toFixed(3)}). See Model Performance for confidence intervals and the full comparison.
          </InfoCard>
          <InfoCard tone="warning">
            <strong className="text-text-primary">Coverage ceiling.</strong> Only cells with &ge; 3 collisions in
            2021&ndash;2023 are modelled. They contain {pct(data.coverage)} of all hotspot cell-months, so system-wide
            recall cannot exceed that.
          </InfoCard>
        </div>

        <div className="mt-12">
          <h2 className="font-display text-lg font-semibold text-text-primary">📖 How to read the outputs</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-text-secondary">
            <li>
              <strong className="text-text-primary">Risk tier</strong> is rank-based within each month: Critical = top{" "}
              {pct(fractions.Critical, 0)} of cells, High = next {pct(fractions.High - fractions.Critical, 0)}, Medium = next{" "}
              {pct(fractions.Medium - fractions.High, 0)}, otherwise Low.
            </li>
            <li>
              <strong className="text-text-primary">Score</strong> is the raw model output; it is <em>not</em> a probability.
            </li>
            <li>
              <strong className="text-text-primary">Estimated probability</strong> is the score passed through a{" "}
              {data.calibration_method} calibrator fitted on 2024 data.
            </li>
          </ul>
        </div>

        <div className="mt-10">
          <h2 className="font-display text-lg font-semibold text-text-primary">⚠ Limitations</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-text-secondary">
            <li>The model is trained on targets up to 2023-12 only; 2024 is used for calibration and the threshold, so recent data are not in the fit.</li>
            <li>Grids first active after 2023 are not modelled.</li>
            <li>Neighbouring cells are spatially correlated; metrics assume independent cells and understate uncertainty.</li>
            <li>Confidence intervals resample grids only (not years); one test year is a single draw.</li>
            <li>The label is a count rule (&ge; 2 collisions), not a causal risk measure; reporting is police-recorded injury collisions only.</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
