import { useState } from "react";
import { Link } from "react-router-dom";
import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { api, FOLIUM_MAP_URL, TIER_ORDER, type RiskMapCell, type Tier } from "../api";
import ExternalLink from "../components/ExternalLink";
import { ErrorState, LoadingState } from "../components/PageState";
import ShapeFilters from "../components/ShapeFilters";
import { NO_SHAPE_FILTER, toApiFilter, type ShapeFilterState } from "../components/shapeFilter";
import TierFilterChips from "../components/TierFilterChips";
import { fixed, int } from "../format";
import { useTileFallback } from "../components/useTileFallback";
import { useAsync } from "../hooks/useAsync";

const TIER_HEX: Record<Tier, string> = {
  Critical: "#d03b3b",
  High: "#ec835a",
  Medium: "#fab219",
  Low: "#0ca30c",
};

const DEFAULT_TIERS: Tier[] = ["Critical", "High", "Medium"];

function boundsOf(cells: RiskMapCell[]): [[number, number], [number, number]] {
  const lats = cells.map((c) => c.grid_latitude);
  const lons = cells.map((c) => c.grid_longitude);
  return [
    [Math.min(...lats), Math.min(...lons)],
    [Math.max(...lats), Math.max(...lons)],
  ];
}

export default function RiskMap() {
  const [source, setSource] = useState<"forecast" | "test">("forecast");
  const [month, setMonth] = useState<string | undefined>(undefined);
  const [tiers, setTiers] = useState<Tier[]>(DEFAULT_TIERS);
  const [shape, setShape] = useState<ShapeFilterState>(NO_SHAPE_FILTER);
  const tiles = useTileFallback();

  // Percentiles and the Folium rule text come from the action-list endpoint (one small request).
  const meta = useAsync(() => api.actionList({ n: 1, offset: 0, tiers: [], sort: "classifier" }), []);
  const percentiles = meta.data?.share_percentiles;

  const { data, error, loading } = useAsync(
    () => api.riskMap({ source, month, tiers, ...(source === "forecast" ? toApiFilter(shape) : {}) }),
    [source, month, tiers.join(","), shape.road, shape.weather, shape.cut],
  );


  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 md:py-12 lg:px-16">
      <h1 className="font-display text-2xl font-bold text-text-primary sm:text-3xl">🗺 Risk map</h1>

      <div className="mt-6 flex flex-wrap items-center gap-x-4 gap-y-3">
        <div className="flex flex-wrap rounded-pill border border-border bg-card p-1">
          {(["forecast", "test"] as const).map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={source === s}
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
          <label className="flex items-center gap-2 text-sm text-text-secondary">
            Month
            <select
              value={data.month ?? ""}
              onChange={(e) => setMonth(e.target.value)}
              className="rounded-sm border border-border bg-card px-3 py-1.5 text-sm text-text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              {data.months.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        )}

        <TierFilterChips selected={tiers} onChange={setTiers} />
        <ExternalLink href={FOLIUM_MAP_URL}>Open the Folium map</ExternalLink>
      </div>

      <div className="mt-4">
        <ShapeFilters
          state={shape}
          onChange={setShape}
          percentiles={percentiles}
          matching={data?.total_matching}
          rule={meta.data?.road_weather_rule}
          disabledReason={
            source === "test"
              ? "Road-type and weather filters apply to the forecast only: they use each cell's collision profile up to the latest month, which is not leakage-safe for 2025 months."
              : undefined
          }
        />
      </div>

      {loading && !data && <div className="mt-6"><LoadingState /></div>}
      {error && (
        <div className="mt-6">
          <ErrorState message={`Could not load the map data: ${error}`} />
        </div>
      )}

      {data && (
        <div className={loading ? "opacity-60 transition-opacity" : ""} aria-busy={loading}>
          <p className={`mt-4 text-sm ${data.truncated ? "text-tier-medium-text" : "text-text-secondary"}`} role="status">
            {data.truncated
              ? `Showing the ${int(data.count)} highest-scoring of ${int(data.total_matching)} matching cells (display limit ${int(data.limit)}).`
              : `${int(data.total_matching)} cells shown.`}
          </p>

          <div className="mt-3 h-[60vh] overflow-hidden rounded-card border border-border md:h-[70vh]">
            {data.cells.length > 0 ? (
              <MapContainer
                key={source}
                bounds={boundsOf(data.cells)}
                boundsOptions={{ padding: [24, 24] }}
                preferCanvas
                style={{ height: "100%", width: "100%" }}
              >
                <TileLayer key={tiles.provider.name} {...tiles.layerProps} />
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
                      <div className="font-mono text-xs text-bg">
                        {cell.grid_id}
                        <br />
                        {cell.risk_tier}
                        <br />
                        score {fixed(cell.score, 3)}
                        <br />
                        prob {fixed(cell.probability, 3)}
                        {data.source === "forecast" && (
                          <>
                            <br />
                            <Link className="underline" to={`/action-list?cell=${encodeURIComponent(cell.grid_id)}`}>
                              Details
                            </Link>
                          </>
                        )}
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
              </MapContainer>
            ) : (
              <div className="flex h-full items-center justify-center p-6 text-center text-text-secondary">
                {tiers.length === 0
                  ? "Select at least one risk tier to see cells."
                  : "No cells match the current filters. Select more tiers or relax the road and weather filters."}
              </div>
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
            Tiers are rank-based within the month. {data.source === "test" && "White rings mark cells that actually had ≥ 2 collisions. "}
            {data.source === "forecast" && "The forecast has no outcomes yet, so it is unvalidated. "}
            Select a point for its score; forecast points link to the cell's details.
          </p>
          <p className="mt-1 text-xs text-text-muted" role="note">
            Base-map tiles are loaded from the internet ({tiles.provider.name}
            {tiles.fallbackInUse ? ", fallback in use" : ""}; falls back to CARTO, then OpenStreetMap). Offline, the coloured cells still
            appear on a blank background. The Folium map also needs an internet connection for its scripts and tiles.
          </p>
        </div>
      )}
    </div>
  );
}
