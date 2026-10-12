// Shared recharts styling: large enough to read on a projector, theme tokens only.
export const tick = { fill: "var(--color-text-secondary)", fontSize: 14 };

export const tooltipProps = {
  contentStyle: { background: "var(--color-bg-raised)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 14 },
  labelStyle: { color: "var(--color-text-primary)" },
  itemStyle: { color: "var(--color-text-secondary)" },
};

export const legendStyle = { fontSize: 14, color: "var(--color-text-primary)" };

// Series are told apart by colour AND dash pattern AND (for lines) marker shape, never colour alone.
export const SERIES = [
  { color: "#4c8dff", dash: undefined },
  { color: "#f2b84b", dash: "8 4" },
  { color: "#ec835a", dash: "2 3" },
  { color: "#6fe36f", dash: "10 3 2 3" },
  { color: "#c8c8c4", dash: "1 5" },
];
