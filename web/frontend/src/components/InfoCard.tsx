export default function InfoCard({
  tone,
  children,
}: {
  tone: "info" | "warning";
  children: React.ReactNode;
}) {
  const border = tone === "info" ? "border-l-accent-amber" : "border-l-tier-critical";
  return (
    <div className={`rounded-card border border-border border-l-4 ${border} bg-card p-4 text-sm text-text-secondary`}>
      {children}
    </div>
  );
}
