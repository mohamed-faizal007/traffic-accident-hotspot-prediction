"""Post-hoc REPORTING: grid-bootstrap CIs on the saved 2025 predictions.

Reads outputs/test_predictions_2025.parquet (already produced by evaluate_test.py) and adds
`test.bootstrap` to metrics.json. Does not touch the model, threshold, calibrator or any point metric.
"""

import json

import pandas as pd

from config import METRICS_PATH, OUTPUTS_DIR
from validation_comparison import bootstrap


def main():
    test = pd.read_parquet(OUTPUTS_DIR / "test_predictions_2025.parquet")
    scores = {"frozen_model": test["score"].to_numpy(), "trailing_12_month": test["base_trailing_12"].to_numpy()}
    out = bootstrap(test, scores, comparisons=[("frozen_model", "trailing_12_month")])
    metrics = json.loads(METRICS_PATH.read_text())
    metrics["test"]["bootstrap"] = out
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    for name, entry in out["comparisons"].items():
        for metric, e in entry.items():
            print(f"{name} {metric}: {e['point_diff']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
