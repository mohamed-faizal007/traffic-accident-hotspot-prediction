// All numbers are shown with one locale so that grouping is consistent (348,525, not 3,48,525).
const LOCALE = "en-GB";

const integer = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 });

export function int(n: number): string {
  return integer.format(n);
}

export function fixed(n: number, digits: number): string {
  return new Intl.NumberFormat(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(n);
}

export function pct(fraction: number, digits = 1): string {
  return `${fixed(100 * fraction, digits)}%`;
}
