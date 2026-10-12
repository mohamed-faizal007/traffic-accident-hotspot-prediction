import { useSearchParams } from "react-router-dom";
import Tabs from "../components/Tabs";
import ClustersTab from "./patterns/ClustersTab";
import TemporalTab from "./patterns/TemporalTab";

type TabId = "temporal" | "clusters";
const TABS: { id: TabId; text: string }[] = [
  { id: "temporal", text: "Temporal patterns" },
  { id: "clusters", text: "DBSCAN clusters" },
];

export default function Patterns() {
  const [params, setParams] = useSearchParams();
  const tab: TabId = params.get("tab") === "clusters" ? "clusters" : "temporal";

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 md:py-12 lg:px-16">
      <h1 className="font-display text-2xl font-bold text-text-primary sm:text-3xl">🔎 Patterns</h1>
      <p className="mt-2 max-w-3xl text-sm text-text-secondary">
        What the collision data shows over time and in space. These are descriptive; the frozen classifier does not use the temporal measures, and DBSCAN cluster features were tested and not adopted.
      </p>

      <div className="mt-6">
        <Tabs label="Pattern views" value={tab} tabs={TABS} onChange={(id) => setParams({ tab: id }, { replace: true })} />
      </div>

      <div className="mt-6" role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
        {tab === "temporal" ? <TemporalTab /> : <ClustersTab />}
      </div>
    </div>
  );
}
