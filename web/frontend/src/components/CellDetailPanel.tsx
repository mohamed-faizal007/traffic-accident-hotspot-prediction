import { Link } from "react-router-dom";
import { api, type CellDetail } from "../api";
import { fixed, int, pct } from "../format";
import { useAsync } from "../hooks/useAsync";
import { ErrorState, LoadingState } from "./PageState";
import Sparkline from "./Sparkline";
import TierPill from "./TierPill";

const ROAD_TEXT = { single: "Single carriageway", other: "Other road type" } as const;
const WEATHER_TEXT = { adverse: "Mostly weather other than fine", fine: "Mostly fine weather" } as const;

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-text-muted">{label}</dt>
      <dd className="mt-0.5 text-sm text-text-primary">{children}</dd>
    </div>
  );
}

export default function CellDetailPanel({ gridId, onClose }: { gridId: string; onClose: () => void }) {
  const { data, error, loading } = useAsync(() => api.cell(gridId), [gridId]);

  return (
    <aside
      aria-label={`Details for cell ${gridId}`}
      className="rounded-card border border-border bg-card p-5 lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto"
    >
      <div className="flex items-start justify-between gap-3">
        <h2 className="font-display text-lg font-semibold text-text-primary">
          Cell <span className="font-mono text-base">{gridId}</span>
        </h2>
        <button
          type="button"
          onClick={onClose}
          className="rounded-sm border border-border px-2.5 py-1 text-sm text-text-secondary hover:bg-card-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          Close
        </button>
      </div>

      {loading && <LoadingState />}
      {error && (
        <div className="mt-4">
          <ErrorState message={error} />
        </div>
      )}
      {data && <Body d={data} />}
    </aside>
  );
}

function Body({ d }: { d: CellDetail }) {
  return (
    <div className="mt-4 space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <TierPill tier={d.risk_tier} />
        <span className="rounded-pill bg-bg-raised px-2.5 py-0.5 text-xs font-semibold text-text-secondary">
          {d.quantile_tier} (tercile)
        </span>
        <span className="text-xs text-text-muted">rank {int(d.rank)} for {d.target_month}</span>
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
        <Stat label="Estimated probability of ≥ 2 collisions">{pct(d.probability, 1)}</Stat>
        <Stat label="Score (not a probability)">{fixed(d.score, 3)}</Stat>
        <Stat label="Predicted collisions, XGBoost">
          {d.predicted_count_xgboost === null ? "n/a" : fixed(d.predicted_count_xgboost, 2)}
        </Stat>
        <Stat label="Predicted collisions, historical average">
          {d.predicted_count_baseline === null ? "n/a" : fixed(d.predicted_count_baseline, 2)}
        </Stat>
      </dl>
      <p className="-mt-2 text-xs text-text-muted">Predicted counts: {d.status}.</p>

      <section>
        <h3 className="text-sm font-semibold text-text-primary">Collisions per month, last 12 months</h3>
        <div className="mt-2">
          <Sparkline values={d.history_12m} months={d.history_months} width={300} height={56} showAxis />
        </div>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-text-primary">What stands out about this cell</h3>
        <p className="mt-1 text-xs text-text-muted">{d.drivers_note}</p>
        <p className="mt-2 text-sm text-text-secondary">{d.summary}</p>
        <ul className="mt-3 divide-y divide-gridline rounded-sm border border-border text-sm">
          {d.drivers.map((x) => (
            <li key={x.key} className="flex items-baseline justify-between gap-3 px-3 py-2">
              <span className="text-text-secondary">{x.label}</span>
              <span className="whitespace-nowrap tabular-nums text-text-primary">
                {Number.isInteger(x.value) ? int(x.value) : fixed(x.value, 2)}
                {x.percentile !== null && (
                  <span className="ml-2 text-xs text-text-muted">above {int(Math.floor(x.percentile))}% of cells</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-text-primary">Road and weather profile</h3>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-3">
          <Stat label="Road type">
            {ROAD_TEXT[d.road]} ({pct(d.shares.single_carriageway, 0)} of collisions)
          </Stat>
          <Stat label="Weather">
            {WEATHER_TEXT[d.weather]} ({pct(d.shares.adverse_weather, 0)} not fine)
          </Stat>
          <Stat label="Road surface not dry">{pct(d.shares.wet_road, 0)}</Stat>
          <Stat label="Speed limit 40 mph or more">{pct(d.shares.high_speed, 0)}</Stat>
          <Stat label="Rush hour">{pct(d.shares.rush_hour, 0)}</Stat>
          <Stat label="Weekend / night">
            {pct(d.shares.weekend, 0)} / {pct(d.shares.night, 0)}
          </Stat>
        </dl>
      </section>

      <section className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <span className="text-text-secondary">
          {fixed(d.grid_latitude, 5)}, {fixed(d.grid_longitude, 5)}
        </span>
        <a
          className="text-accent underline-offset-2 hover:underline"
          href={`https://www.openstreetmap.org/?mlat=${d.grid_latitude}&mlon=${d.grid_longitude}#map=16/${d.grid_latitude}/${d.grid_longitude}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          Open in OpenStreetMap ↗
        </a>
        <Link className="text-accent underline-offset-2 hover:underline" to="/risk-map">
          Risk map
        </Link>
      </section>
    </div>
  );
}
