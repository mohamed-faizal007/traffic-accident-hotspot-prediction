import { int } from "../format";

/** Tiny bar chart of monthly counts. Colour is never the only cue: the maximum is labelled in text. */
export default function Sparkline({
  values,
  months,
  width = 96,
  height = 28,
  showAxis = false,
}: {
  values: number[];
  months?: string[] | null;
  width?: number;
  height?: number;
  showAxis?: boolean;
}) {
  const max = Math.max(1, ...values);
  const gap = 2;
  const barW = (width - gap * (values.length - 1)) / values.length;
  const label = `Monthly collisions${months ? ` ${months[0]} to ${months[months.length - 1]}` : ""}: ${values.join(", ")}`;

  return (
    <div className="inline-flex flex-col">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
        <title>{label}</title>
        {values.map((v, i) => {
          const h = v === 0 ? 1 : Math.max(2, (v / max) * (height - 2));
          return (
            <rect
              key={i}
              x={i * (barW + gap)}
              y={height - h}
              width={barW}
              height={h}
              rx={1}
              fill={v === 0 ? "var(--color-text-muted)" : "var(--color-accent)"}
              opacity={v === 0 ? 0.5 : 1}
            />
          );
        })}
      </svg>
      {showAxis && months && (
        <div className="mt-1 flex justify-between text-xs text-text-muted" style={{ width }}>
          <span>{months[0]}</span>
          <span>peak {int(max)}</span>
          <span>{months[months.length - 1]}</span>
        </div>
      )}
    </div>
  );
}
