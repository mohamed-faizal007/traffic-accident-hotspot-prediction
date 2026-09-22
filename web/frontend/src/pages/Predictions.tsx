import { useState } from "react";
import { api, type Tier } from "../api";
import DataTable, { type Column } from "../components/DataTable";
import { ErrorState, LoadingState } from "../components/PageState";
import TierFilterChips from "../components/TierFilterChips";
import TierPill from "../components/TierPill";
import { useAsync } from "../hooks/useAsync";

interface Row {
  rank: number;
  grid_id: string;
  risk_tier: Tier;
  score: number;
  probability: number;
  grid_latitude: number;
  grid_longitude: number;
}

const PAGE_SIZE = 50;

const columns: Column<Row>[] = [
  { key: "rank", header: "Rank", align: "right", render: (r) => r.rank },
  { key: "grid_id", header: "Grid ID", render: (r) => <span className="font-mono text-xs">{r.grid_id}</span> },
  { key: "risk_tier", header: "Risk tier", render: (r) => <TierPill tier={r.risk_tier} /> },
  { key: "score", header: "Score (not a probability)", align: "right", render: (r) => r.score.toFixed(4) },
  { key: "probability", header: "Estimated probability", align: "right", render: (r) => r.probability.toFixed(4) },
  { key: "lat", header: "Latitude", align: "right", render: (r) => r.grid_latitude.toFixed(4) },
  { key: "lon", header: "Longitude", align: "right", render: (r) => r.grid_longitude.toFixed(4) },
];

export default function Predictions() {
  const [tiers, setTiers] = useState<Tier[]>(["Critical", "High"]);
  const [gridId, setGridId] = useState("");
  const [page, setPage] = useState(1);

  const { data, error, loading } = useAsync(
    () => api.predictions({ tiers, gridId: gridId || undefined, page, pageSize: PAGE_SIZE }),
    [tiers.join(","), gridId, page],
  );

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div className="mx-auto max-w-7xl px-6 py-12 lg:px-16">
      <h1 className="font-display text-3xl font-bold text-text-primary">🔮 Hotspot forecast</h1>
      {data?.target_month && (
        <p className="mt-2 text-sm text-text-secondary">
          Forecast for {data.target_month} from data up to the previous month. Tiers are rank-based; probabilities are
          calibrated on 2024 data.
        </p>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-4">
        <TierFilterChips
          selected={tiers}
          onChange={(t) => {
            setTiers(t);
            setPage(1);
          }}
        />
        <input
          type="text"
          placeholder="Filter by grid id"
          value={gridId}
          onChange={(e) => {
            setGridId(e.target.value);
            setPage(1);
          }}
          className="rounded-sm border border-border bg-card px-3 py-1.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-accent"
        />
      </div>

      {loading && <div className="mt-6"><LoadingState /></div>}
      {error && <div className="mt-6"><ErrorState message={error} /></div>}

      {data && (
        <>
          <div className="mt-4 text-sm text-text-muted">
            {data.total.toLocaleString()} of {data.total_unfiltered.toLocaleString()} modelled cells
          </div>
          <div className="mt-3">
            <DataTable columns={columns} rows={data.rows} rowKey={(r) => `${r.grid_id}-${r.rank}`} />
          </div>
          <div className="mt-4 flex items-center justify-between text-sm text-text-secondary">
            <span>
              Page {data.page} of {totalPages}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="rounded-pill border border-border bg-card px-4 py-1.5 disabled:opacity-40"
              >
                Previous
              </button>
              <button
                type="button"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                className="rounded-pill border border-border bg-card px-4 py-1.5 disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
