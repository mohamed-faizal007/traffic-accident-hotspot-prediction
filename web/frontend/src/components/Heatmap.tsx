import { fixed, int } from "../format";

// Cell colour: the accent blue blended over the card background; intensity also drives the printed value,
// so the encoding is never colour alone. Text colour is chosen for the best contrast against the blended cell.
const BG = [21, 21, 26]; // --color-card
const ACCENT = [76, 141, 255]; // --color-accent
// Pure black / white text: with these, the better of the two always reaches 4.5:1 on every blended cell colour.
const DARK = [0, 0, 0];
const LIGHT = [255, 255, 255];

function blend(t: number): number[] {
  const a = 0.08 + 0.92 * t;
  return BG.map((b, i) => Math.round(b * (1 - a) + ACCENT[i] * a));
}

function luminance(c: number[]): number {
  const f = (v: number) => {
    const x = v / 255;
    return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
}

function contrast(a: number[], b: number[]): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const compact = new Intl.NumberFormat("en-GB", { notation: "compact", maximumFractionDigits: 1 });

export default function Heatmap({
  rowLabels,
  columnLabels,
  values,
  caption,
  valueNoun,
}: {
  rowLabels: string[];
  columnLabels: string[];
  values: number[][];
  caption: string;
  valueNoun: string;
}) {
  const flat = values.flat();
  const max = Math.max(...flat);
  const min = Math.min(...flat);
  let peak = { r: 0, c: 0 };
  for (let r = 0; r < values.length; r++) {
    for (let c = 0; c < values[r].length; c++) {
      if (values[r][c] > values[peak.r][peak.c]) peak = { r, c };
    }
  }
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length;

  return (
    <figure>
      <p className="mb-2 text-xs text-text-muted md:hidden">Scroll sideways to see all {columnLabels.length} columns.</p>
      <div className="overflow-x-auto">
        <div className="min-w-[760px]" role="table" aria-label={caption}>
          <div className="grid items-center gap-[2px]" style={{ gridTemplateColumns: `6.5rem repeat(${columnLabels.length}, minmax(0, 1fr))` }} role="row">
            <div />
            {columnLabels.map((c) => (
              <div key={c} role="columnheader" className="text-center text-xs text-text-secondary">
                {c}
              </div>
            ))}
          </div>
          {values.map((row, r) => (
            <div
              key={rowLabels[r]}
              role="row"
              className="mt-[2px] grid items-stretch gap-[2px]"
              style={{ gridTemplateColumns: `6.5rem repeat(${columnLabels.length}, minmax(0, 1fr))` }}
            >
              <div role="rowheader" className="flex items-center text-sm text-text-secondary">
                {rowLabels[r]}
              </div>
              {row.map((v, c) => {
                const t = max === min ? 0 : (v - min) / (max - min);
                const bg = blend(t);
                const fg = contrast(bg, LIGHT) >= contrast(bg, DARK) ? LIGHT : DARK;
                const isPeak = peak.r === r && peak.c === c;
                return (
                  <div
                    key={c}
                    role="cell"
                    title={`${rowLabels[r]} ${columnLabels[c]}: ${int(v)} ${valueNoun}`}
                    aria-label={`${rowLabels[r]} ${columnLabels[c]}: ${int(v)} ${valueNoun}${isPeak ? " (busiest)" : ""}`}
                    className={`flex h-9 items-center justify-center rounded-[3px] text-xs tabular-nums ${isPeak ? "outline outline-2 outline-offset-1 outline-text-primary" : ""}`}
                    style={{ background: `rgb(${bg.join(",")})`, color: `rgb(${fg.join(",")})` }}
                  >
                    {compact.format(v)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <figcaption className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-text-secondary">
        <span className="flex items-center gap-2">
          {int(min)}
          <span
            aria-hidden="true"
            className="inline-block h-3 w-32 rounded-sm"
            style={{ background: `linear-gradient(to right, rgb(${blend(0).join(",")}), rgb(${blend(1).join(",")}))` }}
          />
          {int(max)} {valueNoun}
        </span>
        <span className="flex items-center gap-2">
          <span aria-hidden="true" className="inline-block h-4 w-4 rounded-[3px] outline outline-2 outline-text-primary" />
          busiest cell: {rowLabels[peak.r]} {columnLabels[peak.c]} ({int(values[peak.r][peak.c])}; {fixed(values[peak.r][peak.c] / mean, 1)}× the average cell)
        </span>
      </figcaption>
    </figure>
  );
}
