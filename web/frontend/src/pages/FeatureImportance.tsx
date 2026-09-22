import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type FeatureImportanceRow } from "../api";
import DataTable, { type Column } from "../components/DataTable";
import { ErrorState, LoadingState } from "../components/PageState";
import { useAsync } from "../hooks/useAsync";

const columns: Column<FeatureImportanceRow>[] = [
  { key: "feature", header: "Feature", render: (r) => r.feature },
  { key: "importance", header: "Importance", align: "right", render: (r) => r.importance.toFixed(4) },
  { key: "std", header: "Std", align: "right", render: (r) => r.std.toFixed(4) },
];

export default function FeatureImportance() {
  const { data, error, loading } = useAsync(() => api.featureImportance(), []);

  if (loading) return <div className="mx-auto max-w-7xl px-6 py-12 lg:px-16"><LoadingState /></div>;
  if (error || !data) return <div className="mx-auto max-w-7xl px-6 py-12 lg:px-16"><ErrorState message={error ?? "Failed to load."} /></div>;

  const rows = [...data.rows].sort((a, b) => b.importance - a.importance);
  const chartData = rows.map((r) => ({ feature: r.feature, importance: r.importance }));

  return (
    <div className="mx-auto max-w-7xl px-6 py-12 lg:px-16">
      <h1 className="font-display text-3xl font-bold text-text-primary">🧩 Feature importance</h1>
      <p className="mt-2 max-w-2xl text-sm text-text-secondary">
        Permutation importance on a 2024 validation sample: the drop in average precision when a feature is shuffled.
        Correlated features share credit, so small values do not mean a feature is useless.
      </p>

      <div className="mt-6 rounded-card border border-border bg-card p-4" style={{ height: Math.max(320, rows.length * 32) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
            <CartesianGrid stroke="var(--color-gridline)" horizontal={false} />
            <XAxis type="number" stroke="var(--color-text-muted)" fontSize={12} tickLine={false} />
            <YAxis dataKey="feature" type="category" stroke="var(--color-text-muted)" fontSize={12} tickLine={false} width={200} />
            <Tooltip
              contentStyle={{ background: "var(--color-bg-raised)", border: "1px solid var(--color-border)", borderRadius: 8 }}
              labelStyle={{ color: "var(--color-text-primary)" }}
              itemStyle={{ color: "var(--color-text-secondary)" }}
            />
            <Bar dataKey="importance" name="importance" fill="var(--color-accent)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6">
        <DataTable columns={columns} rows={rows} rowKey={(r) => r.feature} />
      </div>
    </div>
  );
}
