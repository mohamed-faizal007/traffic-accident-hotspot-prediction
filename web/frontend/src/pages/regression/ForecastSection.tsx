import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api";
import Async from "../../components/Async";
import { fixed, int } from "../../format";
import { useAsync } from "../../hooks/useAsync";

const SIZES = [10, 20, 50];
const selectClass =
  "rounded-sm border border-border bg-card px-3 py-1.5 text-sm text-text-primary focus:outline-none focus-visible:ring-2 focus-visible:ring-accent";

export default function ForecastSection() {
  const [n, setN] = useState(SIZES[1]);
  const state = useAsync(() => api.regressionForecast({ n }), [n]);

  return (
    <Async state={state} what="the regression forecast">
      {(f) => (
        <div className={`space-y-4 ${state.loading ? "opacity-60 transition-opacity" : ""}`}>
          <p className="inline-block rounded-sm border border-border bg-card px-3 py-1.5 text-sm text-tier-medium-text" role="note">
            Forecast status: {f.status}.
          </p>
          <p className="max-w-3xl text-sm text-text-secondary">
            Expected collision count per cell for {f.target_month}, from the frozen {f.model ?? "regression"} model run once on data through the previous month (inference only;
            {f.model_hash_verified ? " the model file hash matched its frozen configuration before and after loading" : " model hash verification is not recorded"}
            {f.generated_at_utc ? `; generated ${f.generated_at_utc}` : ""}). {int(f.total)} cells were scored. There are no outcomes for this month yet, so these numbers have not been checked against reality.
          </p>
          <label className="flex items-center gap-2 text-sm text-text-secondary">
            Show the top
            <select className={selectClass} value={n} onChange={(e) => setN(Number(e.target.value))}>
              {SIZES.map((s) => (
                <option key={s} value={s}>
                  {s} cells
                </option>
              ))}
            </select>
            by predicted count
          </label>

          {f.rows.length === 0 ? (
            <p className="rounded-card border border-border bg-card p-6 text-center text-text-secondary">No forecast rows.</p>
          ) : (
            <div className="overflow-x-auto rounded-card border border-border">
              <table className="w-full min-w-[560px] text-sm">
                <caption className="sr-only">Cells with the highest predicted collision count (unvalidated forecast)</caption>
                <thead className="bg-bg-raised text-text-secondary">
                  <tr>
                    <th className="px-4 py-3 text-right font-medium">Rank</th>
                    <th className="px-4 py-3 text-left font-medium">Cell</th>
                    <th className="px-4 py-3 text-right font-medium">
                      XGBoost predicted count
                      <div className="text-xs font-normal text-text-muted">{f.status}</div>
                    </th>
                    <th className="px-4 py-3 text-right font-medium">Historical-average predicted count</th>
                    <th className="px-4 py-3 text-right font-medium">Latitude, longitude</th>
                  </tr>
                </thead>
                <tbody>
                  {f.rows.map((r) => (
                    <tr key={r.grid_id} className="border-t border-gridline bg-card tabular-nums">
                      <td className="px-4 py-2.5 text-right">{int(r.rank)}</td>
                      <td className="px-4 py-2.5">
                        <Link className="font-mono text-xs text-accent underline-offset-2 hover:underline" to={`/action-list?cell=${encodeURIComponent(r.grid_id)}`}>
                          {r.grid_id}
                        </Link>
                      </td>
                      <td className="px-4 py-2.5 text-right text-text-primary">{fixed(r.predicted_count_xgboost, 2)}</td>
                      <td className="px-4 py-2.5 text-right text-text-secondary">{fixed(r.predicted_count_baseline, 2)}</td>
                      <td className="px-4 py-2.5 text-right text-text-secondary">
                        {fixed(r.grid_latitude, 4)}, {fixed(r.grid_longitude, 4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </Async>
  );
}
