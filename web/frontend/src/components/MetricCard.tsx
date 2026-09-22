export default function MetricCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-card border border-border bg-card p-5">
      <div className="text-sm text-text-secondary">{label}</div>
      <div className="mt-1 font-display text-2xl font-semibold text-text-primary">{value}</div>
      {sub && <div className="mt-1 text-xs text-text-muted">{sub}</div>}
    </div>
  );
}
