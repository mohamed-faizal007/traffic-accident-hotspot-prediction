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
  tiers: Tier[];
  limit: number;
  total_matching: number;
  road_weather_filters_applied: boolean;
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

export interface RoadWeatherFilter {
  road?: "single" | "other";
  weather?: "adverse";
  minAdverseShare?: number;
}

export interface Driver {
  key: string;
  label: string;
  value: number;
  percentile: number | null;
}

export interface CellRecord {
  rank: number;
  grid_id: string;
  grid_latitude: number;
  grid_longitude: number;
  risk_tier: Tier;
  quantile_tier: "LOW" | "MEDIUM" | "HIGH";
  score: number;
  probability: number;
  predicted_count_xgboost: number | null;
  predicted_count_baseline: number | null;
  road: "single" | "other";
  weather: "adverse" | "fine";
  shares: Record<string, number>;
  drivers: Driver[];
  summary: string;
  history_12m: number[];
  history_months: string[] | null;
}

export interface SharePercentiles {
  adverse_weather: Record<string, number>;
  single_carriageway: Record<string, number>;
}

export interface ActionListResponse {
  target_month: string | null;
  status: string;
  drivers_note: string;
  regression_available: boolean;
  total: number;
  total_unfiltered: number;
  offset: number;
  n: number;
  road_weather_rule: string;
  share_percentiles: SharePercentiles;
  rows: CellRecord[];
}

export type CellDetail = CellRecord & { target_month: string; status: string; drivers_note: string };

export interface TemporalPatterns {
  scope: string;
  n_collisions: number;
  years: number[];
  hours: number[];
  weekday_order: string[];
  hour_by_weekday: number[][];
  by_hour: number[];
  by_weekday: number[];
  monthly_by_year: Record<string, number[]>;
  monthly_total: number[];
  weekend: { weekend: number; weekday: number; weekend_share_of_days: number; per_day_ratio_weekend_vs_weekday: number | null };
  time_of_day: { buckets: string[]; hour_bins_right_inclusive: number[]; collisions: number[] };
  clustered_vs_other_hours: { period: string; clustered: number[]; other: number[] };
}

export interface DbscanStats {
  clusters: number;
  clustered_points: number;
  clustered_fraction: number;
  largest_cluster_points: number;
  largest_cluster_share: number;
  median_cluster_size: number;
  p95_cluster_size: number;
}

export interface DbscanSweepRow extends DbscanStats {
  eps_km: number;
  min_samples: number;
}

export interface MoransI {
  morans_i: number;
  expected_under_randomness: number;
  p_value: number;
  permutations: number;
  n_cells_used: number;
  n_cells_without_modelled_neighbour: number;
  variable: string;
  weights: string;
  interpretation: string;
}

export interface DbscanSummary {
  fit_period: string;
  criterion: string;
  chosen: { eps_km: number; min_samples: number };
  stats: DbscanStats;
  sweep: DbscanSweepRow[];
  morans_i: MoransI | null;
  ablation_note: string;
}

export interface ClusterRow {
  hotspot_id: number;
  size: number;
  lat: number;
  lon: number;
  grid_cells: number;
  total_casualties: number;
  first_collision: string;
  last_collision: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  radius_km: number;
}

export interface DbscanClusters {
  fit_period: string | null;
  total: number;
  matching: number;
  returned: number;
  clusters: ClusterRow[];
}

export interface RegressionModelRow {
  model: string;
  mae: number;
  rmse: number;
  r2: number;
}

export interface MaeGap {
  xgboost_mae: number;
  baseline_mae: number;
  xgboost_minus_baseline: number;
  relative_difference: number;
}

export interface RegressionSplit {
  split_label: string;
  n: number;
  models: RegressionModelRow[];
  mean_actual: number;
  mae_gap: MaeGap;
  evaluated_at_utc?: string;
  note?: string;
}

export interface RegressionResults {
  protocol: string;
  model: string;
  params: Record<string, number>;
  target_transform: string;
  features: string[];
  selection_decision: {
    chosen: string;
    reason: string;
    within_noise: boolean;
    top_two: string[];
    top_two_mae_diff: { point_diff: number; ci95: [number, number]; n_bootstrap: number };
  };
  validation: RegressionSplit;
  test: RegressionSplit | null;
}

export interface RegressionForecastRow {
  grid_id: string;
  grid_latitude: number;
  grid_longitude: number;
  target_month: string;
  predicted_count_xgboost: number;
  predicted_count_baseline: number;
  rank: number;
}

export interface RegressionForecast {
  status: string;
  target_month: string | null;
  total: number;
  model: string | null;
  model_hash_verified: boolean | null;
  generated_at_utc: string | null;
  rows: RegressionForecastRow[];
}

export interface TierBlockRow {
  tier: string;
  cell_months: number;
  hotspots: number;
  hotspot_rate: number;
}

export interface QuantileTiers {
  split_label: string;
  quantile_rule: string;
  rank_rule_top_fractions: Record<string, number> | null;
  overall_hotspot_rate: number;
  quantile_tiers: TierBlockRow[];
  rank_tiers: TierBlockRow[];
  crosstab: { rows: string[]; columns: string[]; counts: number[][] };
  forecast: {
    target_month: string;
    status: string;
    quantile_tiers: { tier: string; cells: number }[];
    rank_tiers: { tier: string; cells: number }[];
  } | null;
}

export interface AblationItem {
  id: string;
  track: "classifier" | "regression";
  title: string;
  split: string;
  metric: string;
  higher_is_better: boolean;
  without: number;
  with: number;
  difference: number;
  ci95: [number, number] | null;
  ci_note: string | null;
  seeds: number;
  seeds_with_better: number;
  helps: boolean;
  used_in_final_model: boolean;
  decision_rule: string;
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

function appendShapeFilters(q: URLSearchParams, f: RoadWeatherFilter) {
  if (f.road) q.set("road", f.road);
  if (f.weather) q.set("weather", f.weather);
  if (f.minAdverseShare !== undefined) q.set("min_adverse_share", String(f.minAdverseShare));
}

export const FOLIUM_MAP_URL = "/api/map/folium";

export const api = {
  summary: () => getJSON<Summary>("/api/summary"),
  riskMap: (params: { source: "forecast" | "test"; month?: string; tiers: Tier[] } & RoadWeatherFilter) => {
    const q = new URLSearchParams();
    q.set("source", params.source);
    if (params.month) q.set("month", params.month);
    q.set("tiers", params.tiers.join(",")); // an empty list means "no tiers", not "the default"
    appendShapeFilters(q, params);
    return getJSON<RiskMapResponse>(`/api/risk-map?${q.toString()}`);
  },
  actionList: (params: { n: number; offset: number; tiers: Tier[]; sort: "classifier" | "regression" } & RoadWeatherFilter) => {
    const q = new URLSearchParams({ n: String(params.n), offset: String(params.offset), sort: params.sort });
    q.set("tiers", params.tiers.join(","));
    appendShapeFilters(q, params);
    return getJSON<ActionListResponse>(`/api/action-list?${q.toString()}`);
  },
  temporalPatterns: () => getJSON<TemporalPatterns>("/api/temporal-patterns"),
  dbscanSummary: () => getJSON<DbscanSummary>("/api/dbscan/summary"),
  dbscanClusters: (params: { minSize: number; limit: number }) =>
    getJSON<DbscanClusters>(`/api/dbscan/clusters?min_size=${params.minSize}&limit=${params.limit}`),
  regressionResults: () => getJSON<RegressionResults>("/api/regression/results"),
  regressionForecast: (params: { n: number }) => getJSON<RegressionForecast>(`/api/regression/forecast?n=${params.n}`),
  quantileTiers: () => getJSON<QuantileTiers>("/api/quantile-tiers"),
  ablations: () => getJSON<{ items: AblationItem[] }>("/api/ablations"),
  cell: (gridId: string) => getJSON<CellDetail>(`/api/cell/${encodeURIComponent(gridId)}`),
  modelPerformance: () => getJSON<ModelPerformance>("/api/model-performance"),
  featureImportance: () => getJSON<{ rows: FeatureImportanceRow[] }>("/api/feature-importance"),
};
