/** One plain-language sentence under a chart. Always built from API values by the caller. */
export default function Takeaway({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-3 border-l-2 border-accent pl-3 text-sm text-text-primary" data-takeaway>
      {children}
    </p>
  );
}

export function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="min-w-0 rounded-card border border-border bg-card p-4 sm:p-5">
      <h3 className="font-display text-lg font-semibold text-text-primary">{title}</h3>
      {subtitle && <p className="mt-1 text-sm text-text-secondary">{subtitle}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}
