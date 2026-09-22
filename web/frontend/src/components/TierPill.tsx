import type { Tier } from "../api";

const STYLES: Record<Tier, string> = {
  Critical: "bg-tier-critical-bg text-tier-critical-text",
  High: "bg-tier-high-bg text-tier-high-text",
  Medium: "bg-tier-medium-bg text-tier-medium-text",
  Low: "bg-tier-low-bg text-tier-low-text",
};

export default function TierPill({ tier }: { tier: Tier }) {
  return (
    <span
      className={`inline-flex items-center rounded-pill px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap ${STYLES[tier]}`}
    >
      {tier}
    </span>
  );
}
