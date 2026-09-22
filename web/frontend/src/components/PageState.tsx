export function LoadingState() {
  return <div className="py-16 text-center text-sm text-text-muted">Loading…</div>;
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-card border border-border border-l-4 border-l-tier-critical bg-card p-4 text-sm text-tier-critical-text">
      {message}
    </div>
  );
}
