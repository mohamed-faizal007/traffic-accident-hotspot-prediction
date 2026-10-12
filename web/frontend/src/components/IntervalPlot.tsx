import { fixed } from "../format";

/** A 95% interval with a zero line, labelled in numbers at both ends so it never relies on colour alone. */
export default function IntervalPlot({
  low,
  high,
  point,
  digits = 4,
  label,
}: {
  low: number;
  high: number;
  point: number;
  digits?: number;
  label: string;
}) {
  const W = 420;
  const LABEL_ROOM = 64; // px kept free at each end so the numeric labels never clip
  const lo0 = Math.min(low, 0);
  const hi0 = Math.max(high, 0);
  const span = hi0 - lo0 || 0.001;
  const margin = (span * LABEL_ROOM) / (W - 2 * LABEL_ROOM);
  const lo = lo0 - margin;
  const hi = hi0 + margin;
  const x = (v: number) => ((v - lo) / (hi - lo)) * W;
  const sign = (v: number) => (v > 0 ? "+" : "");
  return (
    <svg viewBox={`0 0 ${W} 64`} className="h-16 w-full max-w-md" role="img" aria-label={`${label}: ${sign(point)}${fixed(point, digits)}, 95% interval ${fixed(low, digits)} to ${fixed(high, digits)}`}>
      <line x1={x(0)} x2={x(0)} y1={6} y2={34} stroke="var(--color-text-secondary)" strokeWidth={2} strokeDasharray="4 3" />
      <text x={x(0)} y={48} textAnchor="middle" fontSize={13} fill="var(--color-text-secondary)">
        0
      </text>
      <line x1={x(low)} x2={x(high)} y1={20} y2={20} stroke="var(--color-accent)" strokeWidth={4} strokeLinecap="round" />
      <line x1={x(low)} x2={x(low)} y1={12} y2={28} stroke="var(--color-accent)" strokeWidth={3} />
      <line x1={x(high)} x2={x(high)} y1={12} y2={28} stroke="var(--color-accent)" strokeWidth={3} />
      <rect x={x(point) - 5} y={15} width={10} height={10} fill="var(--color-text-primary)" transform={`rotate(45 ${x(point)} 20)`} />
      <text x={x(low) - 2} y={62} textAnchor="end" fontSize={13} fill="var(--color-text-primary)">
        {fixed(low, digits)}
      </text>
      <text x={x(high) + 2} y={62} textAnchor="start" fontSize={13} fill="var(--color-text-primary)">
        {fixed(high, digits)}
      </text>
    </svg>
  );
}
