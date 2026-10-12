import type { RoadWeatherFilter, SharePercentiles } from "../api";

export type WeatherChoice = "any" | "adverse" | "top";

export interface ShapeFilterState {
  road: "any" | "single" | "other";
  weather: WeatherChoice;
  /** Adverse-weather share cut-off, fixed from the API's percentiles when the "top" preset is chosen. */
  cut?: number;
}

export const NO_SHAPE_FILTER: ShapeFilterState = { road: "any", weather: "any" };

/** The percentile key used for the "top N% of cells" preset; the label's N is derived from it. */
export const PRESET_KEY = "p90";

export function presetCutoff(p: SharePercentiles | undefined): number | undefined {
  return p?.adverse_weather?.[PRESET_KEY];
}

/** Translate the UI choice into API parameters. */
export function toApiFilter(state: ShapeFilterState): RoadWeatherFilter {
  const out: RoadWeatherFilter = {};
  if (state.road !== "any") out.road = state.road;
  if (state.weather === "adverse") out.weather = "adverse";
  if (state.weather === "top" && state.cut !== undefined) out.minAdverseShare = state.cut;
  return out;
}
