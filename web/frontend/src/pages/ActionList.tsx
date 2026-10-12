import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, FOLIUM_MAP_URL, type CellRecord, type Tier } from "../api";
import CellDetailPanel from "../components/CellDetailPanel";
import ExternalLink from "../components/ExternalLink";
import { ErrorState, LoadingState } from "../components/PageState";
import ShapeFilters from "../components/ShapeFilters";
import { NO_SHAPE_FILTER, toApiFilter, type ShapeFilterState } from "../components/shapeFilter";
import Sparkline from "../components/Sparkline";
import TierFilterChips from "../components/TierFilterChips";
import TierPill from "../components/TierPill";
import { fixed, int, pct } from "../format";
import { useAsync } from "../hooks/useAsync";

const PAGE_SIZES = [10, 20, 50];
const DEFAULT_TIERS: Tier[] = ["Critical", "High"];

const selectClass =
  "rounded-sm border border-border bg-card px-3 py-1.5 text-sm text-text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent";

export default function ActionList() {
  const [params, setParams] = useSearchParams();
  const selected = params.get("cell");

  const [tiers, setTiers] = useState<Tier[]>(DEFAULT_TIERS);
  const [shape, setShape] = useState<ShapeFilterState>(NO_SHAPE_FILTER);
  const [sort, setSort] = useState<"classifier" | "regression">("classifier");
  const [n, setN] = useState(PAGE_SIZES[1]);
  const [offset, setOffset] = useState(0);
  const [find, setFind] = useState("");
  const panelRef = useRef<HTMLDivElement>(null);

  const { data, error, loading } = useAsync(
    () => api.actionList({ n, offset, tiers, sort, ...toApiFilter(shape) }),
    [tiers.join(","), shape.road, shape.weather, shape.cut, sort, n, offset],
  );

  useEffect(() => {
    if (selected && panelRef.current && window.innerWidth < 1024) panelRef.current.scrollIntoView({ block: "start" });
  }, [selected]);

  function select(gridId: string | null) {
    const next = new URLSearchParams(params);
    if (gridId) next.set("cell", gridId);
    else next.delete("cell");
    setParams(next, { replace: true });
  }

  function resetFilters() {
    setTiers(DEFAULT_TIERS);
    setShape(NO_SHAPE_FILTER);
    setOffset(0);
  }

  const total = data?.total ?? 0;
  const firstShown = total === 0 ? 0 : offset + 1;
  const lastShown = Math.min(offset + n, total);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 md:py-12 lg:px-16">
      <h1 className="font-display text-2xl font-bold text-text-primary sm:text-3xl">
        🎯 Action list{data?.target_month ? `: cells to prioritise in ${data.target_month}` : ""}
      </h1>
      <p className="mt-2 max-w-3xl text-sm text-text-secondary">
        Cells ranked by the frozen classifier's forecast from data up to the previous month, with the facts about each cell
        that stand out. Select a row for its detail.
      </p>
      {data && (
        <p className="mt-3 inline-block rounded-sm border border-border bg-card px-3 py-1.5 text-sm text-tier-medium-text" role="note">
          Forecast status: {data.status}.
        </p>
      )}

      <div className="mt-6 space-y-4">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
          <TierFilterChips
            selected={tiers}
            onChange={(t) => {
              setTiers(t);
              setOffset(0);
            }}
          />
          <label className="flex items-center gap-2 text-sm text-text-secondary">
            Sort by
            <select
              className={selectClass}
              value={sort}
              onChange={(e) => {
                setSort(e.target.value as "classifier" | "regression");
                setOffset(0);
              }}
            >
              <option value="classifier">Classifier rank</option>
              <option value="regression" disabled={data ? !data.regression_available : false}>
                Predicted collisions (XGBoost)
              </option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-text-secondary">
            Show
            <select
              className={selectClass}
              value={n}
              onChange={(e) => {
                setN(Number(e.target.value));
                setOffset(0);
              }}
            >
              {PAGE_SIZES.map((s) => (
                <option key={s} value={s}>
                  {s} cells
                </option>
              ))}
            </select>
          </label>
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (find.trim()) select(find.trim());
            }}
          >
            <label className="sr-only" htmlFor="find-cell">
              Find a cell by grid id
            </label>
            <input
              id="find-cell"
              value={find}
              onChange={(e) => setFind(e.target.value)}
              placeholder="Find a cell, e.g. 1059_362"
              className={`${selectClass} w-52 placeholder:text-text-muted`}
            />
            <button type="submit" className="rounded-sm border border-border bg-card px-3 py-1.5 text-sm text-text-primary hover:bg-card-hover">
              Find
            </button>
          </form>
          <ExternalLink href={FOLIUM_MAP_URL}>Open the Folium map</ExternalLink>
        </div>

        <ShapeFilters
          state={shape}
          onChange={(s) => {
            setShape(s);
            setOffset(0);
          }}
          percentiles={data?.share_percentiles}
          matching={data?.total}
          rule={data?.road_weather_rule}
        />
      </div>

      <div className={`mt-6 grid gap-6 ${selected ? "lg:grid-cols-[minmax(0,1fr)_26rem]" : ""}`}>
        {selected && (
          <div ref={panelRef} className="order-first scroll-mt-28 lg:order-last">
            <CellDetailPanel gridId={selected} onClose={() => select(null)} />
          </div>
        )}

        <div className="min-w-0">
          {loading && !data && <LoadingState />}
          {error && <ErrorState message={`Could not load the action list: ${error}`} />}

          {data && (
            <div className={loading ? "opacity-60 transition-opacity" : ""} aria-busy={loading}>
              <div className="mb-3 text-sm text-text-muted" role="status">
                {total === 0
                  ? "No cells match."
                  : `Showing ${int(firstShown)}–${int(lastShown)} of ${int(total)} cells (${int(data.total_unfiltered)} modelled).`}
              </div>

              {data.rows.length === 0 ? (
                <div className="rounded-card border border-border bg-card p-8 text-center">
                  <p className="text-text-primary">No cells match the current filters.</p>
                  <p className="mt-1 text-sm text-text-secondary">Try selecting more risk tiers or relaxing the road and weather filters.</p>
                  <button
                    type="button"
                    onClick={resetFilters}
                    className="mt-4 rounded-pill border border-border bg-card-hover px-4 py-1.5 text-sm text-text-primary hover:bg-bg-raised"
                  >
                    Reset filters
                  </button>
                </div>
              ) : (
                <>
                  <DesktopTable rows={data.rows} selected={selected} onSelect={select} status={data.status} note={data.drivers_note} compact={!!selected} />
                  <MobileCards rows={data.rows} selected={selected} onSelect={select} />
                </>
              )}

              <div className="mt-4 flex items-center justify-between text-sm text-text-secondary">
                <span>
                  Page {int(Math.floor(offset / n) + 1)} of {int(Math.max(1, Math.ceil(total / n)))}
                </span>
                <div className="flex gap-2">
                  <button
                    type="button"
                    disabled={offset <= 0}
                    onClick={() => setOffset(Math.max(0, offset - n))}
                    className="rounded-pill border border-border bg-card px-4 py-1.5 disabled:opacity-40"
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    disabled={offset + n >= total}
                    onClick={() => setOffset(offset + n)}
                    className="rounded-pill border border-border bg-card px-4 py-1.5 disabled:opacity-40"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface ListProps {
  rows: CellRecord[];
  selected: string | null;
  onSelect: (gridId: string) => void;
}

function topFacts(r: CellRecord): string {
  const last12 = r.drivers.find((d) => d.key === "collisions_last_12_months");
  const since = r.drivers.find((d) => d.key === "months_since_last_collision");
  const parts: string[] = [];
  if (last12) parts.push(`${int(last12.value)} collisions in 12 months`);
  if (since) parts.push(`last collision ${int(since.value)} month${since.value === 1 ? "" : "s"} ago`);
  return parts.join(", ");
}

function DesktopTable({ rows, selected, onSelect, status, note, compact }: ListProps & { status: string; note: string; compact: boolean }) {
  return (
    <div className="hidden overflow-hidden rounded-card border border-border md:block">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-bg-raised text-left text-text-secondary">
            <tr>
              <th className="px-3 py-3 text-right font-medium">Rank</th>
              <th className="px-3 py-3 font-medium">Cell</th>
              <th className="px-3 py-3 font-medium">Risk tier</th>
              <th className="px-3 py-3 text-right font-medium">Est. probability</th>
              <th className="px-3 py-3 text-right font-medium">
                Predicted collisions
                <div className="text-xs font-normal text-text-muted">{status}</div>
              </th>
              <th className="px-3 py-3 font-medium">Last 12 months</th>
              {!compact && (
                <th className="px-3 py-3 font-medium">
                  What stands out
                  <div className="text-xs font-normal text-text-muted">{note}</div>
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.grid_id}
                onClick={() => onSelect(r.grid_id)}
                className={`cursor-pointer border-t border-gridline hover:bg-card-hover ${selected === r.grid_id ? "bg-card-hover" : "bg-card"}`}
              >
                <td className="px-3 py-2.5 text-right tabular-nums">{int(r.rank)}</td>
                <td className="px-3 py-2.5">
                  <button
                    type="button"
                    aria-label={`Show details for cell ${r.grid_id}`}
                    aria-current={selected === r.grid_id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect(r.grid_id);
                    }}
                    className="font-mono text-xs text-accent underline-offset-2 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    {r.grid_id}
                  </button>
                  <div className="text-xs text-text-muted">
                    {fixed(r.grid_latitude, 4)}, {fixed(r.grid_longitude, 4)}
                  </div>
                </td>
                <td className="px-3 py-2.5">
                  <TierPill tier={r.risk_tier} />
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums">{pct(r.probability, 1)}</td>
                <td className="px-3 py-2.5 text-right tabular-nums">
                  {r.predicted_count_xgboost === null ? "n/a" : fixed(r.predicted_count_xgboost, 2)}
                </td>
                <td className="px-3 py-2.5">
                  <Sparkline values={r.history_12m} months={r.history_months} />
                </td>
                {!compact && <td className="max-w-xs px-3 py-2.5 text-xs text-text-secondary">{topFacts(r)}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MobileCards({ rows, selected, onSelect }: ListProps) {
  return (
    <ul className="space-y-3 md:hidden">
      {rows.map((r) => (
        <li key={r.grid_id}>
          <button
            type="button"
            onClick={() => onSelect(r.grid_id)}
            aria-current={selected === r.grid_id}
            className={`w-full rounded-card border border-border p-4 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              selected === r.grid_id ? "bg-card-hover" : "bg-card"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-xs text-accent">
                #{int(r.rank)} · {r.grid_id}
              </span>
              <TierPill tier={r.risk_tier} />
            </div>
            <div className="mt-2 flex items-end justify-between gap-3">
              <div className="text-sm text-text-primary">
                <div>{pct(r.probability, 1)} est. probability</div>
                <div className="text-text-secondary">
                  {r.predicted_count_xgboost === null ? "n/a" : fixed(r.predicted_count_xgboost, 2)} predicted collisions
                </div>
              </div>
              <Sparkline values={r.history_12m} months={r.history_months} />
            </div>
            <div className="mt-2 text-xs text-text-secondary">{topFacts(r)}</div>
          </button>
        </li>
      ))}
    </ul>
  );
}
