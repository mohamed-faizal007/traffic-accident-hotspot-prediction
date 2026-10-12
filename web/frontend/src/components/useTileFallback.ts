import { useRef, useState } from "react";

// Base-map providers, tried in order. A provider is abandoned after several consecutive tile errors.
export const TILE_PROVIDERS = [
  {
    name: "Stadia Maps",
    url: "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png",
    attribution:
      '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
  {
    name: "CARTO",
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
  },
  {
    name: "OpenStreetMap",
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
];
const ERRORS_BEFORE_FALLBACK = 4;

/** Props for a react-leaflet TileLayer that falls back to the next provider after repeated tile errors. */
export function useTileFallback() {
  const [index, setIndex] = useState(0);
  const errors = useRef(0);
  const provider = TILE_PROVIDERS[index];
  return {
    provider,
    fallbackInUse: index > 0,
    layerProps: {
      url: provider.url,
      attribution: provider.attribution,
      eventHandlers: {
        tileload: () => {
          errors.current = 0;
        },
        tileerror: () => {
          errors.current += 1;
          if (errors.current >= ERRORS_BEFORE_FALLBACK && index < TILE_PROVIDERS.length - 1) {
            errors.current = 0;
            setIndex(index + 1);
          }
        },
      },
    },
  };
}
