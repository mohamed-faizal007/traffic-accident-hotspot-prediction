import { api, type AblationItem } from "../../api";
import Async from "../../components/Async";
import IntervalPlot from "../../components/IntervalPlot";
import { fixed, int } from "../../format";
import { useAsync } from "../../hooks/useAsync";

const signed = (v: number, d: number) => `${v > 0 ? "+" : ""}${fixed(v, d)}`;

type Effect = "better" | "worse" | "indistinguishable";

/** Direction of the average change and whether the interval rules out zero, from the saved numbers. */
function effect(i: AblationItem): Effect {
  const improved = i.higher_is_better ? i.difference > 0 : i.difference < 0;
  const excludesZero = i.ci95 ? i.ci95[0] > 0 || i.ci95[1] < 0 : false;
  if (!excludesZero) return "indistinguishable";
  return improved ? "better" : "worse";
}

const EFFECT_TEXT: Record<Effect, string> = {
  better: "improved the result (the interval excludes zero)",
  worse: "worsened the result (the interval excludes zero)",
  indistinguishable: "made no distinguishable difference (the interval includes zero)",
};

function Card({ i }: { i: AblationItem }) {
  const e = effect(i);
  const digits = 4;
  return (
    <article className="rounded-card border border-border bg-card p-4 sm:p-5" aria-labelledby={`abl-${i.id}`}>
      <div className="flex flex-wrap items-center gap-2">
        <h4 id={`abl-${i.id}`} className="font-display text-lg font-semibold text-text-primary">
          {i.title}
        </h4>
        <span className="rounded-pill bg-bg-raised px-2.5 py-0.5 text-xs font-semibold text-text-secondary">{i.track} track</span>
      </div>
      <p className="mt-1 text-sm text-text-secondary">
        {i.split}; {i.metric} ({i.higher_is_better ? "higher is better" : "lower is better"}).
      </p>

      <dl className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <div>
          <dt className="text-xs text-text-muted">Without</dt>
          <dd className="mt-0.5 tabular-nums text-text-primary">{fixed(i.without, digits)}</dd>
        </div>
        <div>
          <dt className="text-xs text-text-muted">With</dt>
          <dd className="mt-0.5 tabular-nums text-text-primary">{fixed(i.with, digits)}</dd>
        </div>
        <div>
          <dt className="text-xs text-text-muted">Difference</dt>
          <dd className="mt-0.5 tabular-nums font-semibold text-text-primary">{signed(i.difference, digits)}</dd>
        </div>
      </dl>

      {i.ci95 && (
        <div className="mt-3">
          <p className="text-xs text-text-muted">95% interval of the difference{i.ci_note ? ` (${i.ci_note})` : ""}</p>
          <IntervalPlot low={i.ci95[0]} high={i.ci95[1]} point={i.difference} digits={digits} label={`${i.title} difference`} />
        </div>
      )}

      <p className="mt-3 text-sm text-text-primary">
        Adding it {EFFECT_TEXT[e]}. It was better in {int(i.seeds_with_better)} of {int(i.seeds)} seeds, and {i.used_in_final_model ? "it is used" : "it is not used"} in the final model.
      </p>
      <p className="mt-2 text-xs text-text-muted">Decision rule: {i.decision_rule}</p>
    </article>
  );
}

function Together({ items }: { items: AblationItem[] }) {
  const helped = items.filter((i) => i.helps).length;
  const parts = items.map((i) => `${i.title} on the ${i.track} (${effect(i) === "better" ? "improved" : effect(i) === "worse" ? "worsened" : "no distinguishable change"})`);
  return (
    <p className="rounded-card border border-border border-l-4 border-l-accent-amber bg-card p-4 text-sm text-text-primary" role="note">
      <strong>Taken together:</strong>{" "}
      {helped === 0
        ? `none of the ${int(items.length)} additions met its decision rule, so none of them is used in a final model. `
        : `${int(helped)} of ${int(items.length)} additions met their decision rule. `}
      Per addition: {parts.join("; ")}.
      {helped === 0 && " None of them added anything the decision rules could confirm, so a cell's own collision history remains the main input."}
    </p>
  );
}

export default function AblationsSection() {
  const state = useAsync(() => api.ablations(), []);
  return (
    <Async state={state} what="the ablations">
      {(d) =>
        d.items.length === 0 ? (
          <p className="rounded-card border border-border bg-card p-6 text-center text-text-secondary">No ablation results are available.</p>
        ) : (
          <div className="space-y-4">
            <Together items={d.items} />
            <div className="grid gap-4 lg:grid-cols-3">
              {d.items.map((i) => (
                <Card key={i.id} i={i} />
              ))}
            </div>
          </div>
        )
      }
    </Async>
  );
}
