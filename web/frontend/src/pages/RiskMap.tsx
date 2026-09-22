import { useState } from "react";
import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { api, TIER_ORDER, type Tier } from "../api";
import { ErrorState, LoadingState } from "../components/PageState";
import TierFilterChips from "../components/TierFilterChips";
import { useAsync } from "../hooks/useAsync";

const TIER_HEX: Record<Tier, string> = {
  Critical: "#d03b3b",
  High: "#ec835a",
  Medium: "#fab219",
  Low: "#0ca30c",
};

export default function RiskMap() {
  const [source, setSource] = useState<"forecast" | "test">("forecast");
  const [month, setMonth] = useState<string | undefined>(undefined);
  const [tiers, setTiers] = useState<Tier[]>(["Critical", "High", "Medium"]);

  const { data, error, loading } = useAsync(() => api.riskMap({ source, month, tiers }), [source, month, tiers.join(",")]);

  return (
    <div className="mx-auto max-w-7xl px-6 py-12 lg:px-16">
      <h1 className="font-display text-3xl font-bold text-text-primary">🗺 Risk map</h1>

      <div className="mt-6 flex flex-wrap items-center gap-4">
        <div className="flex rounded-pill border border-border bg-card p-1">
          {(["forecast", "test"] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => {
                setSource(s);
                setMonth(undefined);
              }}
              className={`rounded-pill px-3.5 py-1.5 text-sm font-medium transition-colors ${
                source === s ? "bg-accent text-bg" : "text-text-secondary hover:text-text-primary"
              }`}
            >
              {s === "forecast" ? "Forecast (next month)" : "2025 retrospective (test predictions)"}
            </button>
          ))}
        </div>

        {data && data.months.length > 0 && (
          <select
            value={data.month ?? ""}
            onChange={(e) => setMonth(e.target.value)}
            className="rounded-sm border border-border bg-card px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
          >
            {data.months.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        )}

        <TierFilterChips selected={tiers} onChange={setTiers} />
      </div>

      {loading && <div className="mt-6"><LoadingState /></div>}
      {error && <div className="mt-6"><ErrorState message={error} /></div>}

      {data && (
        <>
          {data.truncated && (
            <p className="mt-4 text-sm text-tier-medium-text">Showing the 30,000 highest-scoring cells.</p>
          )}
          <div className="mt-4 overflow-hidden rounded-card border border-border" style={{ height: "70vh" }}>
            {data.cells.length > 0 ? (
              <MapContainer
                center={[data.cells[0].grid_latitude, data.cells[0].grid_longitude]}
                zoom={6}
                style={{ height: "100%", width: "100%" }}
              >
                <TileLayer
                  url="https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png"
                  attribution='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                {data.cells.map((cell) => (
                  <CircleMarker
                    key={cell.grid_id}
                    center={[cell.grid_latitude, cell.grid_longitude]}
                    radius={cell.hotspot === 1 ? 6 : 4}
                    pathOptions={{
                      color: cell.hotspot === 1 ? "#ffffff" : TIER_HEX[cell.risk_tier],
                      weight: cell.hotspot === 1 ? 2 : 0,
                      fillColor: TIER_HEX[cell.risk_tier],
                      fillOpacity: 0.85,
                    }}
                  >
                    <Popup>
                      <div className="font-mono text-xs">
                        {cell.grid_id}
                        <br />
                        {cell.risk_tier}
                        <br />
                        score {cell.score.toFixed(3)}
                        <br />
                        prob {cell.probability.toFixed(3)}
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
              </MapContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-text-muted">No cells match the current filters.</div>
            )}
          </div>
          <div className="mt-4 flex flex-wrap gap-3">
            {TIER_ORDER.map((tier) => (
              <span key={tier} className="flex items-center gap-1.5 text-sm text-text-secondary">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: TIER_HEX[tier] }} />
                {tier}
              </span>
            ))}
          </div>
          <p className="mt-2 text-xs text-text-muted">
            Tiers are rank-based within the month. {data.source === "test" && "White rings mark cells that actually had ≥ 2 collisions."}
          </p>
        </>
      )}
    </div>
  );
}
