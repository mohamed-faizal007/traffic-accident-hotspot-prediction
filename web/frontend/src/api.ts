export type Tier = "Critical" | "High" | "Medium" | "Low";

export const TIER_ORDER: Tier[] = ["Critical", "High", "Medium", "Low"];

export interface Summary {
  model_family: string;
  task: string;
  frozen: { status: string | null; config_sha256: string | null; frozen_at_utc: string | null } | null;
  test_evaluated: boolean;
  split_label: string;
  base_rate: number;
  average_precision: number;
  ap_lift_over_base_rate: number;
  top_1pct_precision: number;
  top_1pct_lift: number;
  baseline_trailing12_top_1pct_precision: number;
  baseline_trailing12_average_precision: number;
  ap_gain_over_trailing12: number;
  coverage: number;
  tier_top_fractions: Record<string, number>;
  calibration_method: string;
}

export interface PredictionRow {
  rank: number;
  grid_id: string;
  risk_tier: Tier;
  score: number;
  probability: number;
  grid_latitude: number;
  grid_longitude: number;
}

export interface PredictionsResponse {
  target_month: string | null;
  total: number;
  total_unfiltered: number;
  page: number;
  page_size: number;
  rows: PredictionRow[];
}

export interface RiskMapCell {
  grid_id: string;
  grid_latitude: number;
  grid_longitude: number;
  risk_tier: Tier;
  score: number;
  probability: number;
  hotspot?: number;
}

export interface RiskMapResponse {
  source: "forecast" | "test";
  months: string[];
  month: string | null;
  truncated: boolean;
  count: number;
  cells: RiskMapCell[];
}

export interface ComparisonRow {
  method: string;
  average_precision: number;
  ap_lift_over_base_rate: number;
  roc_auc: number;
  precision_top_1pct: number;
  recall_top_1pct: number;
  precision_top_5pct: number;
  recall_top_5pct: number;
}

export interface ModelPerformance {
  split_label: string;
  base_rate: number;
  n: number;
  positives: number;
  comparison_rows: ComparisonRow[];
  system_wide: {
    frozen_threshold: { recall_on_modelled_grids: number; recall_system_wide: number };
    top_k_per_month: Record<string, { recall_on_modelled_grids: number; recall_system_wide: number }>;
    coverage: { in_modelled_grids: number; hotspot_grid_months_all_grids: number; coverage: number };
  } | null;
  confusion: { tn: number; fp: number; fn: number; tp: number; threshold: number; precision: number; recall: number; f1: number; flagged_fraction: number };
  calibration: { brier: number; brier_base_rate_reference: number };
  per_month_average_precision: Record<string, number>;
  validation_candidates: { model: string; val_average_precision: number; val_roc_auc: number; val_top_1pct_precision: number }[];
  selection_decision: { chosen: string; reason: string; top_two_ap_point_diff: number; top_two_ap_ci95: [number, number] };
  bootstrap_comparisons: { comparison: string; ap_point_diff: number; ap_ci95_low: number; ap_ci95_high: number }[] | null;
  cluster_ablation: { without_cluster_mean: number; with_cluster_mean: number; diff_mean: number } | null;
  alignment_ablation: { original_alignment_ap: number; months_leq_t_ap: number } | null;
  test_evaluated: boolean;
  test_evaluated_at_utc: string | null;
}

export interface FeatureImportanceRow {
  feature: string;
  importance: number;
  std: number;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `Request to ${path} failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  summary: () => getJSON<Summary>("/api/summary"),
  predictions: (params: { tiers?: Tier[]; gridId?: string; page?: number; pageSize?: number }) => {
    const q = new URLSearchParams();
    if (params.tiers?.length) q.set("tiers", params.tiers.join(","));
    if (params.gridId) q.set("grid_id", params.gridId);
    q.set("page", String(params.page ?? 1));
    q.set("page_size", String(params.pageSize ?? 50));
    return getJSON<PredictionsResponse>(`/api/predictions?${q.toString()}`);
  },
  riskMap: (params: { source: "forecast" | "test"; month?: string; tiers?: Tier[] }) => {
    const q = new URLSearchParams();
    q.set("source", params.source);
    if (params.month) q.set("month", params.month);
    if (params.tiers?.length) q.set("tiers", params.tiers.join(","));
    return getJSON<RiskMapResponse>(`/api/risk-map?${q.toString()}`);
  },
  modelPerformance: () => getJSON<ModelPerformance>("/api/model-performance"),
  featureImportance: () => getJSON<{ rows: FeatureImportanceRow[] }>("/api/feature-importance"),
};
