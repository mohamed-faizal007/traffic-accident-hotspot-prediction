import { Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type RegressionResults, type RegressionSplit } from "../../api";
import Async from "../../components/Async";
import { legendStyle, tick, tooltipProps } from "../../components/chartStyle";
import IntervalPlot from "../../components/IntervalPlot";
import { fixed, int } from "../../format";
import { useAsync } from "../../hooks/useAsync";

const MODEL_NAME: Record<string, string> = {
  xgboost: "XGBoost",
  random_forest: "Random forest",
  historical_average_baseline: "Historical-average baseline",
};
const name = (k: string) => MODEL_NAME[k] ?? k;
const signed = (v: number, d: number) => `${v > 0 ? "+" : ""}${fixed(v, d)}`;

function best(rows: RegressionSplit["models"], key: "mae" | "rmse" | "r2"): string {
  const pick = key === "r2" ? Math.max : Math.min;
  const target = pick(...rows.map((r) => r[key]));
  return rows.find((r) => r[key] === target)!.model;
}

function SplitTable({ split, id }: { split: RegressionSplit; id: string }) {
  const bestMae = best(split.models, "mae");
  const bestRmse = best(split.models, "rmse");
  const bestR2 = best(split.models, "r2");
  const cell = (isBest: boolean, text: string) => (
    <td className={`px-4 py-3 text-right tabular-nums ${isBest ? "font-semibold text-text-primary" : "text-text-secondary"}`}>
      {text}
      {isBest && <span className="ml-1.5 text-xs font-medium text-accent">best</span>}
    </td>
  );
  return (
    <div className="min-w-0">
      <h4 id={id} className="font-display text-base font-semibold text-text-primary">
        {split.split_label}
      </h4>
      <p className="mt-1 text-sm text-text-secondary">
        {int(split.n)} cell-months; mean actual count {fixed(split.mean_actual, 3)} per cell-month.
        {split.evaluated_at_utc && <> Evaluated once at {split.evaluated_at_utc}.</>}
      </p>
      <div className="mt-3 overflow-x-auto rounded-card border border-border">
        <table className="w-full min-w-[480px] text-sm" aria-labelledby={id}>
          <thead className="bg-bg-raised text-text-secondary">
            <tr>
              <th className="px-4 py-3 text-left font-medium">Model</th>
              <th className="px-4 py-3 text-right font-medium">MAE (lower is better)</th>
              <th className="px-4 py-3 text-right font-medium">RMSE (lower is better)</th>
              <th className="px-4 py-3 text-right font-medium">R² (higher is better)</th>
            </tr>
          </thead>
          <tbody>
            {split.models.map((m) => (
              <tr key={m.model} className="border-t border-gridline bg-card">
                <td className="px-4 py-3 text-text-primary">{name(m.model)}</td>
                {cell(m.model === bestMae, fixed(m.mae, 4))}
                {cell(m.model === bestRmse, fixed(m.rmse, 4))}
                {cell(m.model === bestR2, fixed(m.r2, 4))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Callout({ r }: { r: RegressionResults }) {
  const v = r.validation;
  const t = r.test;
  if (!t) {
    return <p className="rounded-card border border-border bg-card p-4 text-sm text-text-secondary">2025 test results are not available, so only the validation comparison is shown.</p>;
  }
  const tx = t.models.find((m) => m.model === "xgboost")!;
  const tb = t.models.find((m) => m.model === "historical_average_baseline")!;
  // "XGBoost's value is lower / higher than the baseline's", from the sign of (XGBoost minus baseline)
  const word = (xgboostMinusBaseline: number) => (xgboostMinusBaseline < 0 ? "lower" : xgboostMinusBaseline > 0 ? "higher" : "equal");
  const gv = v.mae_gap.xgboost_minus_baseline;
  const gt = t.mae_gap.xgboost_minus_baseline;
  const reversed = Math.sign(gv) !== Math.sign(gt);
  return (
    <div className="rounded-card border border-border border-l-4 border-l-accent-amber bg-card p-4 text-sm text-text-primary" role="note">
      <h4 className="font-display text-base font-semibold">MAE on the 2025 test set, XGBoost against the baseline</h4>
      <p className="mt-2">
        XGBoost's MAE is {fixed(t.mae_gap.xgboost_mae, 4)} and the baseline's is {fixed(t.mae_gap.baseline_mae, 4)}: a gap of {signed(gt, 4)} ({signed(100 * t.mae_gap.relative_difference, 1)}%), so XGBoost's MAE is {word(gt)} than the baseline's.
        Its RMSE is {word(tx.rmse - tb.rmse)} ({fixed(tx.rmse, 4)} against {fixed(tb.rmse, 4)}) and its R² is {word(tx.r2 - tb.r2)} ({fixed(tx.r2, 4)} against {fixed(tb.r2, 4)}).
      </p>
      <p className="mt-2 text-text-secondary">
        On the 2024 validation set the MAE gap was {signed(gv, 4)} ({signed(100 * v.mae_gap.relative_difference, 1)}%).{" "}
        {reversed ? "The direction of the MAE gap differs between validation and test." : "The MAE gap has the same direction on validation and test."} The test figures were computed once and nothing was tuned after seeing them.
      </p>
    </div>
  );
}

function MaeChart({ r }: { r: RegressionResults }) {
  const keys = r.validation.models.map((m) => m.model);
  const data = keys.map((k) => ({
    model: name(k),
    validation: r.validation.models.find((m) => m.model === k)?.mae,
    test: r.test?.models.find((m) => m.model === k)?.mae,
  }));
  return (
    <div className="h-80" role="img" aria-label="MAE by model on validation and test">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 24 }}>
          <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
          <XAxis dataKey="model" tick={tick} tickLine={false} interval={0} angle={-20} textAnchor="end" height={96} />
          <YAxis tick={tick} tickLine={false} width={52} domain={[0, "auto"]} tickFormatter={(v: number) => fixed(v, 2)} />
          <Tooltip {...tooltipProps} formatter={(v) => (typeof v === "number" ? fixed(v, 4) : String(v ?? ""))} />
          <Legend wrapperStyle={legendStyle} />
          <Bar dataKey="validation" name="2024 validation" fill="#4c8dff" radius={[4, 4, 0, 0]} isAnimationActive={false}>
            <LabelList dataKey="validation" position="top" formatter={(v) => (typeof v === "number" ? fixed(v, 3) : "")} fill="var(--color-text-primary)" fontSize={13} />
          </Bar>
          {r.test && (
            <Bar dataKey="test" name="2025 test" fill="#f2b84b" radius={[4, 4, 0, 0]} isAnimationActive={false}>
              <LabelList dataKey="test" position="top" formatter={(v) => (typeof v === "number" ? fixed(v, 3) : "")} fill="var(--color-text-primary)" fontSize={13} />
            </Bar>
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function ComparisonSection() {
  const state = useAsync(() => api.regressionResults(), []);
  return (
    <Async state={state} what="the regression results">
      {(r) => {
        const notOnTest = r.test ? r.validation.models.filter((m) => !r.test!.models.some((x) => x.model === m.model)).map((m) => name(m.model)) : [];
        const d = r.selection_decision;
        return (
          <div className="space-y-6">
            <p className="max-w-3xl text-sm text-text-secondary">
              Models predict next-month collision <em>count</em> per 500 m cell ({r.target_transform}). Validation (2024) was used to tune and choose; the 2025 test set was evaluated once.
            </p>
            <div className="grid gap-6 lg:grid-cols-2">
              <SplitTable split={r.validation} id="tbl-validation" />
              {r.test ? <SplitTable split={r.test} id="tbl-test" /> : <p className="text-sm text-text-secondary">2025 test results are not available.</p>}
            </div>
            {notOnTest.length > 0 && <p className="text-sm text-text-muted">Not evaluated on the 2025 test set: {notOnTest.join(", ")} (only the selected model and the baseline were).</p>}
            <Callout r={r} />
            <MaeChart r={r} />

            <div className="rounded-card border border-border bg-card p-4">
              <h4 className="font-display text-base font-semibold text-text-primary">How {name(d.chosen)} was chosen</h4>
              <p className="mt-2 text-sm text-text-secondary">
                Rule saved before tuning: take the lowest validation MAE; if the top two are within bootstrap noise, take the simpler. Outcome: {d.reason}. Top two ({d.top_two.map(name).join(" vs ")}): MAE difference {signed(d.top_two_mae_diff.point_diff, 4)}, 95% interval {fixed(d.top_two_mae_diff.ci95[0], 4)} to{" "}
                {fixed(d.top_two_mae_diff.ci95[1], 4)} ({int(d.top_two_mae_diff.n_bootstrap)} grid bootstrap resamples).
              </p>
              <IntervalPlot low={d.top_two_mae_diff.ci95[0]} high={d.top_two_mae_diff.ci95[1]} point={d.top_two_mae_diff.point_diff} label="Validation MAE difference, top two models" />
              <details className="mt-3 text-sm text-text-secondary">
                <summary className="cursor-pointer text-text-primary">Model settings and the {int(r.features.length)} features</summary>
                <p className="mt-2">Parameters: {Object.entries(r.params).map(([k, v]) => `${k} = ${v}`).join(", ")}.</p>
                <p className="mt-1">{r.features.join(", ")}.</p>
                <p className="mt-1">Counts are scored on the raw scale after inverting the transform.</p>
              </details>
            </div>
          </div>
        );
      }}
    </Async>
  );
}
