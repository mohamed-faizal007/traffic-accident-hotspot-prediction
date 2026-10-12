import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type QuantileTiers, type TierBlockRow } from "../../api";
import Async from "../../components/Async";
import { tick, tooltipProps } from "../../components/chartStyle";
import Takeaway, { ChartCard } from "../../components/Takeaway";
import TierPill from "../../components/TierPill";
import { fixed, int, pct } from "../../format";
import { useAsync } from "../../hooks/useAsync";
import type { Tier } from "../../api";

function RateChart({ rows, max, label }: { rows: TierBlockRow[]; max: number; label: string }) {
  const step = 0.1;
  const ticks = Array.from({ length: Math.ceil(max / step) + 1 }, (_, i) => Math.round(i * step * 100) / 100);
  return (
    <div className="h-64" role="img" aria-label={label}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 22 }}>
          <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
          <XAxis dataKey="tier" tick={tick} tickLine={false} />
          <YAxis tick={tick} tickLine={false} width={52} domain={[0, ticks[ticks.length - 1]]} ticks={ticks} tickFormatter={(v: number) => pct(v, 0)} />
          <Tooltip {...tooltipProps} formatter={(v) => (typeof v === "number" ? pct(v, 2) : String(v ?? ""))} />
          <Bar dataKey="hotspot_rate" name="observed hotspot rate" fill="#4c8dff" radius={[4, 4, 0, 0]} isAnimationActive={false}>
            <LabelList dataKey="hotspot_rate" position="top" formatter={(v) => (typeof v === "number" ? pct(v, 1) : "")} fill="var(--color-text-primary)" fontSize={14} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function TierTable({ title, rows }: { title: string; rows: TierBlockRow[] }) {
  return (
    <div className="overflow-x-auto rounded-card border border-border">
      <table className="w-full min-w-[420px] text-sm">
        <caption className="sr-only">{title}</caption>
        <thead className="bg-bg-raised text-text-secondary">
          <tr>
            <th className="px-3 py-2.5 text-left font-medium">Tier</th>
            <th className="px-3 py-2.5 text-right font-medium">Cell-months</th>
            <th className="px-3 py-2.5 text-right font-medium">Hotspots</th>
            <th className="px-3 py-2.5 text-right font-medium">Observed rate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.tier} className="border-t border-gridline bg-card tabular-nums">
              <td className="px-3 py-2 text-left text-text-primary">{r.tier}</td>
              <td className="px-3 py-2 text-right text-text-secondary">{int(r.cell_months)}</td>
              <td className="px-3 py-2 text-right text-text-secondary">{int(r.hotspots)}</td>
              <td className="px-3 py-2 text-right text-text-primary">{pct(r.hotspot_rate, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Body({ q }: { q: QuantileTiers }) {
  const max = Math.max(...q.quantile_tiers.map((t) => t.hotspot_rate), ...q.rank_tiers.map((t) => t.hotspot_rate));
  const qr = q.quantile_tiers.map((t) => t.hotspot_rate);
  const monotone = qr.every((v, i) => i === 0 || v >= qr[i - 1]);
  const rr = q.rank_tiers.map((t) => t.hotspot_rate);
  const rMonotone = rr.every((v, i) => i === 0 || v <= rr[i - 1]);
  const top = q.quantile_tiers[q.quantile_tiers.length - 1];
  const crit = q.rank_tiers[0];
  const low = q.rank_tiers[q.rank_tiers.length - 1];
  const fr = q.rank_rule_top_fractions;

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-sm text-text-secondary">
        Two ways to turn the same classifier score into tiers. Rank tiers (the validated system): {fr ? `Critical is the top ${pct(fr.Critical, 0)} of cells each month, High the next ${pct(fr.High - fr.Critical, 0)}, Medium the next ${pct(fr.Medium - fr.High, 0)}, otherwise Low.` : "defined by rank within each month."}{" "}
        Quantile tiers (the proposal's LOW/MEDIUM/HIGH): {q.quantile_rule}. Observed rates below are on {q.split_label}; the overall hotspot rate is {pct(q.overall_hotspot_rate, 2)}.
      </p>

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="Quantile tiers (LOW / MEDIUM / HIGH)" subtitle="Share of cell-months in each tier that were hotspots.">
          <RateChart rows={q.quantile_tiers} max={max} label="Observed hotspot rate by quantile tier" />
          <div className="mt-3">
            <TierTable title="Quantile tiers" rows={q.quantile_tiers} />
          </div>
        </ChartCard>
        <ChartCard title="Rank tiers (Critical / High / Medium / Low)" subtitle="Same measure, same axis scale.">
          <RateChart rows={q.rank_tiers} max={max} label="Observed hotspot rate by rank tier" />
          <div className="mt-3">
            <TierTable title="Rank tiers" rows={q.rank_tiers} />
          </div>
        </ChartCard>
      </div>

      <Takeaway>
        The observed hotspot rate {monotone ? "rises" : "does not rise steadily"} from {pct(qr[0], 2)} in {q.quantile_tiers[0].tier} to {pct(top.hotspot_rate, 2)} in {top.tier} ({fixed(top.hotspot_rate / q.overall_hotspot_rate, 1)}× the overall rate). The rank tiers {rMonotone ? "fall" : "do not fall steadily"} from{" "}
        {pct(crit.hotspot_rate, 1)} in {crit.tier} to {pct(low.hotspot_rate, 2)} in {low.tier}, which is {fixed(crit.hotspot_rate / top.hotspot_rate, 1)}× the rate of the {top.tier} tercile.
      </Takeaway>

      <ChartCard title="How the two systems overlap" subtitle="Cell-months by rank tier (rows) and quantile tier (columns); the percentage is the share of the row.">
        <div className="overflow-x-auto rounded-card border border-border">
          <table className="w-full min-w-[480px] text-sm">
            <caption className="sr-only">Cross-tabulation of rank tiers by quantile tiers</caption>
            <thead className="bg-bg-raised text-text-secondary">
              <tr>
                <th className="px-4 py-3 text-left font-medium">Rank tier</th>
                {q.crosstab.columns.map((c) => (
                  <th key={c} className="px-4 py-3 text-right font-medium">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {q.crosstab.rows.map((row, i) => {
                const total = q.crosstab.counts[i].reduce((a, b) => a + b, 0);
                return (
                  <tr key={row} className="border-t border-gridline bg-card tabular-nums">
                    <td className="px-4 py-2.5">
                      <TierPill tier={row as Tier} />
                    </td>
                    {q.crosstab.counts[i].map((v, j) => (
                      <td key={j} className="px-4 py-2.5 text-right text-text-primary">
                        {int(v)} <span className="text-xs text-text-secondary">({total ? pct(v / total, 1) : "–"})</span>
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ChartCard>

      {q.forecast && (
        <ChartCard title={`Forecast month ${q.forecast.target_month}: cells per tier`} subtitle={`Forecast status: ${q.forecast.status}; counts only, there are no outcomes to measure rates against.`}>
          <div className="grid gap-4 sm:grid-cols-2">
            <ul className="divide-y divide-gridline rounded-card border border-border text-sm">
              {q.forecast.rank_tiers.map((t) => (
                <li key={t.tier} className="flex items-center justify-between bg-card px-4 py-2.5">
                  <TierPill tier={t.tier as Tier} />
                  <span className="tabular-nums text-text-primary">{int(t.cells)} cells</span>
                </li>
              ))}
            </ul>
            <ul className="divide-y divide-gridline rounded-card border border-border text-sm">
              {q.forecast.quantile_tiers.map((t) => (
                <li key={t.tier} className="flex items-center justify-between bg-card px-4 py-2.5">
                  <span className="font-semibold text-text-primary">{t.tier}</span>
                  <span className="tabular-nums text-text-primary">{int(t.cells)} cells</span>
                </li>
              ))}
            </ul>
          </div>
        </ChartCard>
      )}
    </div>
  );
}

export default function TiersSection() {
  const state = useAsync(() => api.quantileTiers(), []);
  return (
    <Async state={state} what="the quantile tiers">
      {(q) => <Body q={q} />}
    </Async>
  );
}
