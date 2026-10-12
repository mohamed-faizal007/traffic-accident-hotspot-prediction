import { useState } from "react";
import { Link } from "react-router-dom";
import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { api, type ClusterRow, type DbscanSummary, type MoransI } from "../../api";
import Async from "../../components/Async";
import { ChartCard } from "../../components/Takeaway";
import { useTileFallback } from "../../components/useTileFallback";
import { fixed, int, pct } from "../../format";
import { useAsync } from "../../hooks/useAsync";

const DISPLAY_CAP = 1000; // largest clusters first; keeps the map fast
const SWEEP_HEADERS = ["eps (km)", "min samples", "clusters", "clustered points", "clustered share", "largest cluster", "largest share", "median size", "95th pct size"];

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-card border border-border bg-card p-4">
      <div className="text-sm text-text-secondary">{label}</div>
      <div className="mt-1 font-display text-2xl font-semibold text-text-primary">{value}</div>
      {sub && <div className="mt-1 text-xs text-text-muted">{sub}</div>}
    </div>
  );
}

export default function ClustersTab() {
  const summary = useAsync(() => api.dbscanSummary(), []);
  const ablations = useAsync(() => api.ablations(), []);
  const dbscanAblation = ablations.data?.items.find((i) => i.id === "dbscan_clusters");

  return (
    <div className="space-y-6">
      <Async state={summary} what="the DBSCAN summary">
        {(s) => (
          <>
            <Stats s={s} />

            <div className="rounded-card border border-border border-l-4 border-l-accent-amber bg-card p-4 text-text-primary" role="note">
              <strong>Cluster features did not improve prediction.</strong>{" "}
              {dbscanAblation ? (
                <>
                  Adding them to the classifier {dbscanAblation.helps ? "improved" : "did not improve"} validation {dbscanAblation.metric}: {fixed(dbscanAblation.without, 4)} without, {fixed(dbscanAblation.with, 4)} with (difference{" "}
                  {dbscanAblation.difference > 0 ? "+" : ""}
                  {fixed(dbscanAblation.difference, 4)}; better in {dbscanAblation.seeds_with_better} of {dbscanAblation.seeds} seeds).{" "}
                  {dbscanAblation.used_in_final_model ? "They are used in the final model." : "They are not used in the final model."}{" "}
                </>
              ) : (
                <>{s.ablation_note} </>
              )}
              <Link className="text-accent underline underline-offset-2" to="/regression-track#ablations">
                See the ablations
              </Link>
              .
            </div>

            <ClusterMap s={s} />
            <SweepTable s={s} />
            <Moran m={s.morans_i} />
          </>
        )}
      </Async>
    </div>
  );
}

function Stats({ s }: { s: DbscanSummary }) {
  const st = s.stats;
  return (
    <section aria-label="DBSCAN fit statistics">
      <p className="mb-3 text-sm text-text-secondary">
        DBSCAN fitted on {s.fit_period} collisions only, with eps = {fixed(s.chosen.eps_km, 2)} km and min samples = {int(s.chosen.min_samples)} (haversine distance).
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Clusters" value={int(st.clusters)} sub={`median size ${fixed(st.median_cluster_size, 0)}, 95th percentile ${fixed(st.p95_cluster_size, 0)}`} />
        <Stat label="Coverage of training collisions" value={pct(st.clustered_fraction, 1)} sub={`${int(st.clustered_points)} collisions are in a cluster`} />
        <Stat label="Largest cluster" value={int(st.largest_cluster_points)} sub={`${pct(st.largest_cluster_share, 1)} of clustered points`} />
        <Stat label="Outside any cluster" value={pct(1 - st.clustered_fraction, 1)} sub="treated as noise by DBSCAN" />
      </div>
    </section>
  );
}

function radiusFor(size: number, max: number): number {
  return 3 + 13 * Math.sqrt(size / max);
}

function ClusterMap({ s }: { s: DbscanSummary }) {
  const [minSize, setMinSize] = useState(0);
  const tiles = useTileFallback();
  const max = s.stats.largest_cluster_points;
  const state = useAsync(() => api.dbscanClusters({ minSize, limit: DISPLAY_CAP }), [minSize]);

  const presets = [
    { text: "All clusters", value: 0 },
    { text: `Median size or larger (≥ ${fixed(s.stats.median_cluster_size, 0)})`, value: Math.ceil(s.stats.median_cluster_size) },
    { text: `95th percentile or larger (≥ ${fixed(s.stats.p95_cluster_size, 0)})`, value: Math.ceil(s.stats.p95_cluster_size) },
  ];

  return (
    <ChartCard title="Cluster map" subtitle="Each circle is one DBSCAN cluster; circle area grows with the number of collisions in it.">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
        <label className="flex items-center gap-3 text-sm text-text-secondary">
          Minimum cluster size
          <input
            type="range"
            min={0}
            max={max}
            step={1}
            value={minSize}
            onChange={(e) => setMinSize(Number(e.target.value))}
            className="w-48 accent-[var(--color-accent)] sm:w-64"
            aria-valuetext={`${int(minSize)} collisions`}
          />
          <output className="min-w-12 font-semibold tabular-nums text-text-primary">{int(minSize)}</output>
        </label>
        <div className="flex flex-wrap gap-2">
          {presets.map((p) => (
            <button
              key={p.text}
              type="button"
              aria-pressed={minSize === p.value}
              onClick={() => setMinSize(p.value)}
              className={`rounded-pill border px-3 py-1 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                minSize === p.value ? "border-accent bg-accent text-bg" : "border-border bg-card text-text-secondary hover:bg-card-hover"
              }`}
            >
              {p.text}
            </button>
          ))}
        </div>
      </div>

      <Async state={state} what="the clusters">
        {(d) => (
          <div className={state.loading ? "opacity-60 transition-opacity" : ""}>
            <p className="mt-3 text-sm text-text-secondary" role="status">
              {d.matching === 0
                ? "No clusters of that size."
                : d.matching > d.returned
                  ? `Showing the ${int(d.returned)} largest of ${int(d.matching)} matching clusters (${int(d.total)} in total; display limit ${int(DISPLAY_CAP)}).`
                  : `Showing ${int(d.returned)} of ${int(d.total)} clusters.`}
            </p>
            <div className="mt-3 h-[55vh] overflow-hidden rounded-card border border-border md:h-[65vh]">
              {d.clusters.length > 0 ? (
                <MapContainer
                  key="clusters"
                  bounds={[
                    [Math.min(...d.clusters.map((c) => c.lat)), Math.min(...d.clusters.map((c) => c.lon))],
                    [Math.max(...d.clusters.map((c) => c.lat)), Math.max(...d.clusters.map((c) => c.lon))],
                  ]}
                  boundsOptions={{ padding: [24, 24] }}
                  preferCanvas
                  style={{ height: "100%", width: "100%" }}
                >
                  <TileLayer key={tiles.provider.name} {...tiles.layerProps} />
                  {d.clusters.map((c) => (
                    <CircleMarker
                      key={c.hotspot_id}
                      center={[c.lat, c.lon]}
                      radius={radiusFor(c.size, max)}
                      pathOptions={{ color: "#ffffff", weight: 1, fillColor: "#4c8dff", fillOpacity: 0.6 }}
                    >
                      <Popup>
                        <ClusterPopup c={c} />
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>
              ) : (
                <div className="flex h-full items-center justify-center p-6 text-center text-text-secondary">No clusters of that size. Lower the minimum size.</div>
              )}
            </div>
          </div>
        )}
      </Async>

      <div className="mt-3 flex flex-wrap items-end gap-x-6 gap-y-2 text-sm text-text-secondary" aria-label="Circle size key">
        <span>Circle size key:</span>
        {[
          { label: "median", size: s.stats.median_cluster_size },
          { label: "95th percentile", size: s.stats.p95_cluster_size },
          { label: "largest", size: max },
        ].map((k) => (
          <span key={k.label} className="flex items-center gap-2">
            <svg width={34} height={34} aria-hidden="true">
              <circle cx={17} cy={17} r={radiusFor(k.size, max)} fill="#4c8dff" fillOpacity={0.6} stroke="#ffffff" />
            </svg>
            {k.label}: {int(k.size)}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-text-muted" role="note">
        Base-map tiles are loaded from the internet ({tiles.provider.name}
        {tiles.fallbackInUse ? ", fallback in use" : ""}). Offline, the circles still appear on a blank background.
      </p>
    </ChartCard>
  );
}

function ClusterPopup({ c }: { c: ClusterRow }) {
  return (
    <div className="font-mono text-xs text-bg">
      cluster {c.hotspot_id}
      <br />
      {int(c.size)} collisions in {int(c.grid_cells)} grid cells
      <br />
      {int(c.total_casualties)} casualties
      <br />
      radius {fixed(c.radius_km, 2)} km
      <br />
      {c.first_collision} to {c.last_collision}
    </div>
  );
}

function SweepTable({ s }: { s: DbscanSummary }) {
  return (
    <ChartCard title="Parameter sweep" subtitle={`Selection criterion: ${s.criterion}. The chosen row is marked.`}>
      <div className="overflow-x-auto rounded-card border border-border">
        <table className="w-full min-w-[720px] text-sm">
          <caption className="sr-only">DBSCAN parameter sweep with the chosen parameters marked</caption>
          <thead className="bg-bg-raised text-text-secondary">
            <tr>
              <th className="px-3 py-3 text-left font-medium">Chosen</th>
              {SWEEP_HEADERS.map((h) => (
                <th key={h} className="px-3 py-3 text-right font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {s.sweep.map((r) => {
              const chosen = r.eps_km === s.chosen.eps_km && r.min_samples === s.chosen.min_samples;
              return (
                <tr key={`${r.eps_km}-${r.min_samples}`} className={`border-t border-gridline tabular-nums ${chosen ? "bg-card-hover font-semibold text-text-primary" : "bg-card text-text-secondary"}`} aria-current={chosen}>
                  <td className="px-3 py-2 text-left">{chosen ? <span className="rounded-pill bg-accent px-2 py-0.5 text-xs font-semibold text-bg">chosen</span> : ""}</td>
                  <td className="px-3 py-2 text-right">{fixed(r.eps_km, 2)}</td>
                  <td className="px-3 py-2 text-right">{int(r.min_samples)}</td>
                  <td className="px-3 py-2 text-right">{int(r.clusters)}</td>
                  <td className="px-3 py-2 text-right">{int(r.clustered_points)}</td>
                  <td className="px-3 py-2 text-right">{pct(r.clustered_fraction, 1)}</td>
                  <td className="px-3 py-2 text-right">{int(r.largest_cluster_points)}</td>
                  <td className="px-3 py-2 text-right">{pct(r.largest_cluster_share, 1)}</td>
                  <td className="px-3 py-2 text-right">{fixed(r.median_cluster_size, 0)}</td>
                  <td className="px-3 py-2 text-right">{fixed(r.p95_cluster_size, 0)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </ChartCard>
  );
}

function Moran({ m }: { m: MoransI | null }) {
  return (
    <ChartCard title="Spatial autocorrelation (Moran's I)" subtitle="Do neighbouring grid cells have similar collision counts?">
      {m ? (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <Stat label="Moran's I" value={fixed(m.morans_i, 3)} sub={`expected ${fixed(m.expected_under_randomness, 4)} if independent`} />
            <Stat label="p-value" value={fixed(m.p_value, 3)} sub={`${int(m.permutations)} random permutations`} />
            <Stat label="Cells used" value={int(m.n_cells_used)} sub={`${int(m.n_cells_without_modelled_neighbour)} had no modelled neighbour`} />
            <Stat label="Variable" value={m.variable} sub={m.weights} />
          </div>
          <p className="mt-3 border-l-2 border-accent pl-3 text-sm text-text-primary">{m.interpretation}</p>
        </>
      ) : (
        <p className="text-text-secondary">Moran's I has not been computed (outputs/dashboard/morans_i.json is missing). Run src/dashboard_exports/morans_i.py.</p>
      )}
    </ChartCard>
  );
}
