export default function ExternalLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 rounded-pill border border-border bg-card px-3.5 py-1.5 text-sm font-medium text-accent hover:bg-card-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
    >
      {children} <span aria-hidden="true">↗</span>
    </a>
  );
}
