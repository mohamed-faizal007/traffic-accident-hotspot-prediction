import type { SharePercentiles } from "../api";
import { fixed, int } from "../format";
import { PRESET_KEY, presetCutoff, type ShapeFilterState } from "./shapeFilter";

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; text: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div role="group" aria-label={label} className="flex flex-wrap items-center gap-2">
      <span className="text-sm text-text-secondary">{label}</span>
      <div className="flex flex-wrap rounded-pill border border-border bg-card p-1">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            aria-pressed={value === o.value}
            onClick={() => onChange(o.value)}
            className={`rounded-pill px-3 py-1 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              value === o.value ? "bg-accent text-bg" : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {o.text}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ShapeFilters({
  state,
  onChange,
  percentiles,
  matching,
  rule,
  disabledReason,
}: {
  state: ShapeFilterState;
  onChange: (s: ShapeFilterState) => void;
  percentiles: SharePercentiles | undefined;
  matching: number | undefined;
  rule: string | undefined;
  disabledReason?: string;
}) {
  const cut = presetCutoff(percentiles);
  if (disabledReason) {
    return <p className="text-sm text-text-muted">{disabledReason}</p>;
  }
  const topPct = 100 - Number(PRESET_KEY.slice(1));
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <Segmented
          label="Road type"
          value={state.road}
          onChange={(road) => onChange({ ...state, road })}
          options={[
            { value: "any", text: "Any" },
            { value: "single", text: "Single carriageway" },
            { value: "other", text: "Other road type" },
          ]}
        />
        <Segmented
          label="Weather"
          value={state.weather}
          onChange={(weather) => onChange({ ...state, weather, cut: weather === "top" ? cut : undefined })}
          options={[
            { value: "any", text: "Any" },
            ...(cut !== undefined ? [{ value: "top" as const, text: `Adverse-weather share: top ${topPct}% of cells` }] : []),
            { value: "adverse", text: "Mostly adverse (Folium rule)" },
          ]}
        />
      </div>
      {(state.road !== "any" || state.weather !== "any") && matching !== undefined && (
        <p className="text-sm text-text-secondary" role="status">
          <strong className="text-text-primary">{int(matching)}</strong> cells match
          {state.weather === "top" && state.cut !== undefined && (
            <>
              {" "}
              (adverse-weather share ≥ {fixed(state.cut, 2)}, the {PRESET_KEY.slice(1)}th percentile of modelled cells)
            </>
          )}
          .
        </p>
      )}
      <p className="text-xs text-text-muted">
        Road type and weather describe each cell's past collisions (share on single carriageways / in adverse weather, up to the
        latest month). The Folium map uses a different, stricter cut-off{rule ? ` (${rule})` : ""}.
      </p>
    </div>
  );
}
