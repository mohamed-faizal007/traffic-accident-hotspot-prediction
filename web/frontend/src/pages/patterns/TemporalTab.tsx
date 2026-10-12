import { Bar, BarChart, CartesianGrid, LabelList, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type TemporalPatterns } from "../../api";
import Async from "../../components/Async";
import { legendStyle, SERIES, tick, tooltipProps } from "../../components/chartStyle";
import Heatmap from "../../components/Heatmap";
import Takeaway, { ChartCard } from "../../components/Takeaway";
import { fixed, int, pct } from "../../format";
import { useAsync } from "../../hooks/useAsync";

const hh = (h: number) => `${String(h).padStart(2, "0")}:00`;
const monthName = (i: number) => new Date(2000, i, 1).toLocaleString("en-GB", { month: "short" });
const argMax = (xs: number[]) => xs.indexOf(Math.max(...xs));
const argMin = (xs: number[]) => xs.indexOf(Math.min(...xs));
const cap = (x: string) => x.charAt(0).toUpperCase() + x.slice(1);
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

export default function TemporalTab() {
  const state = useAsync(() => api.temporalPatterns(), []);
  return (
    <Async state={state} what="temporal patterns">
      {(t) => <Charts t={t} />}
    </Async>
  );
}

function Charts({ t }: { t: TemporalPatterns }) {
  const total = t.n_collisions;
  const hourLabels = t.hours.map((h) => String(h).padStart(2, "0"));

  // 1. hour x weekday
  const peakHour = argMax(t.by_hour);
  const peakDay = argMax(t.by_weekday);
  const rowSums = t.hour_by_weekday;
  const weekendRows = t.weekday_order.map((d, i) => (d === "Saturday" || d === "Sunday" ? i : -1)).filter((i) => i >= 0);
  const weekdayRows = t.weekday_order.map((_, i) => i).filter((i) => !weekendRows.includes(i));
  const hourOf = (rows: number[]) => argMax(t.hours.map((h) => sum(rows.map((r) => rowSums[r][h]))));
  const wkPeak = weekdayRows.length ? hourOf(weekdayRows) : null;
  const wePeak = weekendRows.length ? hourOf(weekendRows) : null;

  // 2. monthly
  const years = Object.keys(t.monthly_by_year);
  const monthly = t.monthly_total.map((_, m) => ({ month: monthName(m), ...Object.fromEntries(years.map((y) => [y, t.monthly_by_year[y][m]])) }));
  const hiM = argMax(t.monthly_total);
  const loM = argMin(t.monthly_total);
  const cells = years.flatMap((y) => t.monthly_by_year[y].map((v, m) => ({ y, m, v })));
  const lowest = cells.reduce((a, b) => (b.v < a.v ? b : a));
  const yearTotals = years.map((y) => sum(t.monthly_by_year[y]));
  const yHi = Math.max(...yearTotals);
  const yLo = Math.min(...yearTotals);

  // 3. weekend
  const wk = t.weekend;
  const wkShare = wk.weekend / (wk.weekend + wk.weekday);
  const weekendData = [
    { name: "Of collisions", value: wkShare },
    { name: "Of calendar days", value: wk.weekend_share_of_days },
  ];

  // 4. time of day
  const bins = t.time_of_day.hour_bins_right_inclusive;
  const buckets = t.time_of_day.buckets.map((b, i) => {
    const from = bins[i] + 1;
    const to = bins[i + 1];
    return { name: `${b.replace("_", " ")} (${String(from).padStart(2, "0")}–${String(to).padStart(2, "0")} h)`, short: b.replace("_", " "), collisions: t.time_of_day.collisions[i], hours: to - from + 1 };
  });
  const bTotal = sum(buckets.map((b) => b.collisions));
  const bTop = buckets[argMax(buckets.map((b) => b.collisions))];
  const bDense = buckets[argMax(buckets.map((b) => b.collisions / b.hours))];

  // 5. clustered vs other
  const cl = t.clustered_vs_other_hours;
  const clSum = sum(cl.clustered);
  const otSum = sum(cl.other);
  const hourShare = t.hours.map((h) => ({ hour: String(h).padStart(2, "0"), clustered: cl.clustered[h] / clSum, other: cl.other[h] / otSum }));
  const diff = hourShare.map((r) => r.clustered - r.other);
  const dMax = argMax(diff);
  const dMin = argMin(diff);

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-sm text-text-secondary">{cap(t.scope)}. {int(total)} collisions, {t.years[0]}–{t.years[t.years.length - 1]}.</p>

      <ChartCard title="Hour of day by weekday" subtitle="Collisions in each hour-and-weekday cell. Darker blue and a larger printed value mean more collisions.">
        <Heatmap rowLabels={t.weekday_order} columnLabels={hourLabels} values={t.hour_by_weekday} caption="Collisions by weekday and hour" valueNoun="collisions" />
        <Takeaway>
          Collisions peak in the {hh(peakHour)} hour ({pct(t.by_hour[peakHour] / total, 1)} of all collisions) and {t.weekday_order[peakDay]} has the most of any weekday
          ({pct(t.by_weekday[peakDay] / total, 1)}).
          {wkPeak !== null && wePeak !== null && <> The weekday peak hour is {hh(wkPeak)}; the weekend peak hour is {hh(wePeak)}.</>}
        </Takeaway>
      </ChartCard>

      <ChartCard title="Monthly seasonality by year" subtitle="Collisions per calendar month; one line per year; the legend shows each year's line style and marker.">
        <div className="h-80" role="img" aria-label="Collisions per month, one line per year">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={monthly} margin={{ right: 16, top: 8 }}>
              <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
              <XAxis dataKey="month" tick={tick} tickLine={false} />
              <YAxis tick={tick} tickLine={false} width={56} tickFormatter={(v: number) => int(v)} />
              <Tooltip {...tooltipProps} formatter={(v) => (typeof v === "number" ? int(v) : String(v ?? ""))} />
              <Legend wrapperStyle={legendStyle} />
              {years.map((y, i) => (
                <Line
                  key={y}
                  type="monotone"
                  dataKey={y}
                  name={y}
                  stroke={SERIES[i % SERIES.length].color}
                  strokeDasharray={SERIES[i % SERIES.length].dash}
                  strokeWidth={3}
                  dot={{ r: 3 + (i % 3) }}
                  isAnimationActive={false}
                >
                </Line>
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Takeaway>
          Across all years combined, {monthName(hiM)} is the busiest month ({int(t.monthly_total[hiM])}) and {monthName(loM)} the quietest ({int(t.monthly_total[loM])}). Yearly totals range from{" "}
          {int(yLo)} to {int(yHi)} ({pct((yHi - yLo) / yLo, 1)} apart). The lowest single month in the data is {monthName(lowest.m)} {lowest.y} ({int(lowest.v)}).
        </Takeaway>
      </ChartCard>

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="Weekend vs weekday" subtitle="The weekend share of all collisions, next to the weekend share of all calendar days.">
          <div className="h-64" role="img" aria-label="Weekend share of collisions and of calendar days">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weekendData} margin={{ top: 22 }}>
                <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
                <XAxis dataKey="name" tick={tick} tickLine={false} interval={0} />
                <YAxis tick={tick} tickLine={false} width={56} tickFormatter={(v: number) => pct(v, 1)} />
                <Tooltip {...tooltipProps} formatter={(v) => (typeof v === "number" ? pct(v, 1) : String(v ?? ""))} />
                <Bar dataKey="value" fill="var(--color-accent)" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  <LabelList dataKey="value" position="top" formatter={(v) => (typeof v === "number" ? pct(v, 1) : "")} fill="var(--color-text-primary)" fontSize={14} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Takeaway>
            {wk.per_day_ratio_weekend_vs_weekday !== null && (
              <>
                A weekend day averages {fixed(wk.per_day_ratio_weekend_vs_weekday, 2)}× the collisions of a weekday ({wk.per_day_ratio_weekend_vs_weekday < 1 ? "fewer" : "more"} per day).
              </>
            )}{" "}
            Weekends hold {pct(wkShare, 1)} of collisions on {pct(wk.weekend_share_of_days, 1)} of days.
          </Takeaway>
        </ChartCard>

        <ChartCard title="Time-of-day buckets" subtitle="Collisions per bucket; the hours each bucket covers are in its label.">
          <div className="h-64" role="img" aria-label="Collisions per time-of-day bucket">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={buckets} margin={{ top: 22 }}>
                <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
                <XAxis dataKey="short" tick={tick} tickLine={false} interval={0} angle={-30} textAnchor="end" height={64} />
                <YAxis tick={tick} tickLine={false} width={64} tickFormatter={(v: number) => int(v)} />
                <Tooltip {...tooltipProps} formatter={(v) => (typeof v === "number" ? int(v) : String(v ?? ""))} labelFormatter={(_, p) => p?.[0]?.payload?.name ?? ""} />
                <Bar dataKey="collisions" fill="var(--color-accent)" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  <LabelList dataKey="collisions" position="top" formatter={(v) => (typeof v === "number" ? int(v) : "")} fill="var(--color-text-primary)" fontSize={13} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Takeaway>
            {cap(bTop.short)} has the most collisions ({pct(bTop.collisions / bTotal, 1)}), while {bDense.short} is the busiest per hour ({int(Math.round(bDense.collisions / bDense.hours))} collisions per hour within the bucket, {bDense.name.split("(")[1].replace(")", "")}).
          </Takeaway>
        </ChartCard>
      </div>

      <ChartCard title="Hour profile: inside DBSCAN clusters vs elsewhere" subtitle={`Share of each group's collisions by hour (${cl.period}). Solid line with round markers: clustered; dashed line: other.`}>
        <div className="h-80" role="img" aria-label="Hourly share of collisions, clustered versus other">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={hourShare} margin={{ right: 12, top: 8 }}>
              <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
              <XAxis dataKey="hour" tick={tick} tickLine={false} interval={1} />
              <YAxis tick={tick} tickLine={false} width={56} tickFormatter={(v: number) => pct(v, 0)} />
              <Tooltip {...tooltipProps} formatter={(v) => (typeof v === "number" ? pct(v, 1) : String(v ?? ""))} />
              <Legend wrapperStyle={legendStyle} />
              <Line type="monotone" dataKey="clustered" name="Clustered collisions" stroke={SERIES[0].color} strokeWidth={3} dot={{ r: 4 }} isAnimationActive={false} />
              <Line type="monotone" dataKey="other" name="Other collisions" stroke={SERIES[1].color} strokeDasharray={SERIES[1].dash} strokeWidth={3} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Takeaway>
          Clustered collisions peak at {hh(argMax(cl.clustered))} and other collisions at {hh(argMax(cl.other))}. The clustered group has its largest relative excess at {hh(dMax)} ({fixed(100 * diff[dMax], 1)} points more of its collisions) and its
          largest relative deficit at {hh(dMin)} ({fixed(100 * -diff[dMin], 1)} points fewer).
        </Takeaway>
      </ChartCard>
    </div>
  );
}
