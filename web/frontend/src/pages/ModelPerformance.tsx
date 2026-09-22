import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import DataTable, { type Column } from "../components/DataTable";
import { ErrorState, LoadingState } from "../components/PageState";
import { useAsync } from "../hooks/useAsync";
import type { ComparisonRow, ModelPerformance as ModelPerformanceData } from "../api";

function pct(x: number, digits = 1) {
  return `${(100 * x).toFixed(digits)}%`;
}

const comparisonColumns: Column<ComparisonRow>[] = [
  { key: "method", header: "Method", render: (r) => r.method },
  { key: "ap", header: "Avg precision", align: "right", render: (r) => r.average_precision.toFixed(3) },
  { key: "lift", header: "Lift over base", align: "right", render: (r) => `${r.ap_lift_over_base_rate.toFixed(1)}x` },
  { key: "auc", header: "ROC-AUC (ref.)", align: "right", render: (r) => r.roc_auc.toFixed(3) },
  { key: "p1", header: "Precision @ top 1%", align: "right", render: (r) => pct(r.precision_top_1pct) },
  { key: "r1", header: "Recall @ top 1%", align: "right", render: (r) => pct(r.recall_top_1pct) },
  { key: "p5", header: "Precision @ top 5%", align: "right", render: (r) => pct(r.precision_top_5pct) },
  { key: "r5", header: "Recall @ top 5%", align: "right", render: (r) => pct(r.recall_top_5pct) },
];

function ConfusionMatrix({ conf }: { conf: ModelPerformanceData["confusion"] }) {
  const cell = (label: string, value: number, correct: boolean) => (
    <div
      className={`rounded-sm border border-border p-4 text-center ${correct ? "bg-accent/10" : "bg-card"}`}
    >
      <div className="text-xs text-text-secondary">{label}</div>
      <div className="mt-1 font-display text-xl font-semibold text-text-primary">{value.toLocaleString()}</div>
    </div>
  );
  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        {cell("True negative", conf.tn, true)}
        {cell("False positive", conf.fp, false)}
        {cell("False negative", conf.fn, false)}
        {cell("True positive", conf.tp, true)}
      </div>
      <p className="mt-3 text-xs text-text-muted">
        Precision {conf.precision.toFixed(3)}, recall {conf.recall.toFixed(3)}, F1 {conf.f1.toFixed(3)}; flags{" "}
        {pct(conf.flagged_fraction)} of cell-months.
      </p>
    </div>
  );
}

export default function ModelPerformance() {
  const { data, error, loading } = useAsync(() => api.modelPerformance(), []);

  if (loading) return <div className="mx-auto max-w-7xl px-6 py-12 lg:px-16"><LoadingState /></div>;
  if (error || !data) return <div className="mx-auto max-w-7xl px-6 py-12 lg:px-16"><ErrorState message={error ?? "Failed to load."} /></div>;

  const perMonth = Object.entries(data.per_month_average_precision).map(([month, ap]) => ({ month, ap }));

  return (
    <div className="mx-auto max-w-7xl px-6 py-12 lg:px-16">
      <h1 className="font-display text-3xl font-bold text-text-primary">📊 Model performance</h1>
      <p className="mt-2 text-sm text-text-secondary">{data.split_label}</p>

      <div className="mt-6">
        <DataTable columns={comparisonColumns} rows={data.comparison_rows} rowKey={(r) => r.method} />
      </div>
      <p className="mt-3 text-xs text-text-muted">
        Base rate {pct(data.base_rate, 2)} on {data.n.toLocaleString()} cell-months ({data.positives.toLocaleString()} hotspots).
        Top-k is taken within each month. Recall is on modelled cells.
      </p>

      {data.system_wide && (
        <div className="mt-10">
          <h2 className="font-display text-lg font-semibold text-text-primary">Recall on modelled cells vs system-wide</h2>
          <div className="mt-3">
            <DataTable
              columns={[
                { key: "op", header: "Operating point", render: (r: { op: string; modelled: number; system: number }) => r.op },
                { key: "modelled", header: "Recall (modelled cells)", align: "right", render: (r) => r.modelled.toFixed(3) },
                { key: "system", header: "Recall (system-wide)", align: "right", render: (r) => r.system.toFixed(3) },
              ]}
              rows={[
                {
                  op: "Frozen F1 threshold",
                  modelled: data.system_wide.frozen_threshold.recall_on_modelled_grids,
                  system: data.system_wide.frozen_threshold.recall_system_wide,
                },
                ...Object.entries(data.system_wide.top_k_per_month).map(([k, v]) => ({
                  op: `Top ${k.split("_")[1].replace("pct", "%")} per month`,
                  modelled: v.recall_on_modelled_grids,
                  system: v.recall_system_wide,
                })),
              ]}
              rowKey={(r) => r.op}
            />
          </div>
          <p className="mt-2 text-xs text-text-muted">
            {data.system_wide.coverage.in_modelled_grids.toLocaleString()} of{" "}
            {data.system_wide.coverage.hotspot_grid_months_all_grids.toLocaleString()} hotspot cell-months (
            {pct(data.system_wide.coverage.coverage)}) lie in modelled cells.
          </p>
        </div>
      )}

      <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h2 className="font-display text-lg font-semibold text-text-primary">
            Confusion matrix at the frozen threshold (score &ge; {data.confusion.threshold.toFixed(3)}, tuned on 2024)
          </h2>
          <div className="mt-3">
            <ConfusionMatrix conf={data.confusion} />
          </div>
        </div>
        <div>
          <h2 className="font-display text-lg font-semibold text-text-primary">Calibration (Brier score, lower is better)</h2>
          <p className="mt-3 text-sm text-text-secondary">
            Calibrated probabilities: <strong className="text-text-primary">{data.calibration.brier.toFixed(4)}</strong> vs
            constant base-rate forecast {data.calibration.brier_base_rate_reference.toFixed(4)}. Raw scores are not
            probabilities (mean score is far above the base rate).
          </p>
        </div>
      </div>

      <div className="mt-10">
        <h2 className="font-display text-lg font-semibold text-text-primary">Per-month average precision</h2>
        <div className="mt-3 h-72 rounded-card border border-border bg-card p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={perMonth}>
              <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
              <XAxis dataKey="month" stroke="var(--color-text-muted)" fontSize={12} tickLine={false} />
              <YAxis stroke="var(--color-text-muted)" fontSize={12} tickLine={false} width={40} />
              <Tooltip
                contentStyle={{ background: "var(--color-bg-raised)", border: "1px solid var(--color-border)", borderRadius: 8 }}
                labelStyle={{ color: "var(--color-text-primary)" }}
                itemStyle={{ color: "var(--color-text-secondary)" }}
              />
              <Bar dataKey="ap" name="average precision" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mt-10">
        <h2 className="font-display text-lg font-semibold text-text-primary">Validation (2024) and model selection</h2>
        <div className="mt-3">
          <DataTable
            columns={[
              { key: "model", header: "Model", render: (r: (typeof data.validation_candidates)[number]) => r.model },
              { key: "ap", header: "Val AP", align: "right", render: (r) => r.val_average_precision.toFixed(3) },
              { key: "auc", header: "Val ROC-AUC", align: "right", render: (r) => r.val_roc_auc.toFixed(3) },
              { key: "p1", header: "Val top-1% precision", align: "right", render: (r) => r.val_top_1pct_precision.toFixed(3) },
            ]}
            rows={data.validation_candidates}
            rowKey={(r) => r.model}
          />
        </div>
        <p className="mt-3 text-sm text-text-secondary">
          Selection rule (saved before tuning): highest validation AP; if the top two are within bootstrap noise, take the
          simpler. Outcome: <strong className="text-text-primary">{data.selection_decision.chosen}</strong> (
          {data.selection_decision.reason}). Top-two AP difference {data.selection_decision.top_two_ap_point_diff >= 0 ? "+" : ""}
          {data.selection_decision.top_two_ap_point_diff.toFixed(4)}, 95% CI [
          {data.selection_decision.top_two_ap_ci95[0].toFixed(4)}, {data.selection_decision.top_two_ap_ci95[1].toFixed(4)}].
        </p>

        {data.bootstrap_comparisons && (
          <div className="mt-6">
            <h3 className="text-sm font-semibold text-text-primary">Validation AP differences (grid bootstrap)</h3>
            <div className="mt-2">
              <DataTable
                columns={[
                  { key: "c", header: "Comparison", render: (r: (typeof data.bootstrap_comparisons)[number]) => r.comparison },
                  { key: "d", header: "AP difference", align: "right", render: (r) => r.ap_point_diff.toFixed(4) },
                  { key: "lo", header: "95% CI low", align: "right", render: (r) => r.ap_ci95_low.toFixed(4) },
                  { key: "hi", header: "95% CI high", align: "right", render: (r) => r.ap_ci95_high.toFixed(4) },
                ]}
                rows={data.bootstrap_comparisons}
                rowKey={(r) => r.comparison}
              />
            </div>
          </div>
        )}

        {data.cluster_ablation && (
          <p className="mt-6 text-sm text-text-secondary">
            <strong className="text-text-primary">DBSCAN cluster-feature ablation (validation).</strong> Average precision
            without cluster features {data.cluster_ablation.without_cluster_mean.toFixed(4)}, with{" "}
            {data.cluster_ablation.with_cluster_mean.toFixed(4)} (difference{" "}
            {data.cluster_ablation.diff_mean >= 0 ? "+" : ""}
            {data.cluster_ablation.diff_mean.toFixed(4)}, mean over seeds). Cluster features are not used in the final model.
          </p>
        )}

        {data.alignment_ablation && (
          <p className="mt-3 text-sm text-text-secondary">
            Feature timing ablation (validation AP): original alignment {data.alignment_ablation.original_alignment_ap.toFixed(4)}{" "}
            vs months &le; t {data.alignment_ablation.months_leq_t_ap.toFixed(4)}.
          </p>
        )}

        {data.test_evaluated && data.test_evaluated_at_utc && (
          <p className="mt-6 text-xs text-text-muted">Test evaluated once at {data.test_evaluated_at_utc}.</p>
        )}
      </div>
    </div>
  );
}
