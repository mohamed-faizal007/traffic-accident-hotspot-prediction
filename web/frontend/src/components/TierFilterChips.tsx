import { TIER_ORDER, type Tier } from "../api";

const ACTIVE_STYLES: Record<Tier, string> = {
  Critical: "bg-tier-critical-bg text-tier-critical-text border-tier-critical-bg",
  High: "bg-tier-high-bg text-tier-high-text border-tier-high-bg",
  Medium: "bg-tier-medium-bg text-tier-medium-text border-tier-medium-bg",
  Low: "bg-tier-low-bg text-tier-low-text border-tier-low-bg",
};

export default function TierFilterChips({
  selected,
  onChange,
}: {
  selected: Tier[];
  onChange: (tiers: Tier[]) => void;
}) {
  function toggle(tier: Tier) {
    onChange(selected.includes(tier) ? selected.filter((t) => t !== tier) : [...selected, tier]);
  }

  return (
    <div className="flex flex-wrap gap-2">
      {TIER_ORDER.map((tier) => {
        const active = selected.includes(tier);
        return (
          <button
            key={tier}
            type="button"
            onClick={() => toggle(tier)}
            className={`rounded-pill border px-3.5 py-1.5 text-sm font-medium transition-colors ${
              active ? ACTIVE_STYLES[tier] : "border-border bg-card text-text-secondary hover:bg-card-hover"
            }`}
          >
            {tier}
          </button>
        );
      })}
    </div>
  );
}
