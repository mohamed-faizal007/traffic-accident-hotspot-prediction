# Old run (pre-audit) — kept for before/after comparison

Produced by the original pipeline: DBSCAN/grid/active-grid filter computed on all 5 years,
features shifted so month t's own count was unused, hotspot threshold (0.90) chosen by best F1
on the 2025 test set, uncalibrated RF scores presented as probabilities, metrics hard-coded in the dashboard.
The reported F1 0.2621 / ROC-AUC 0.8508 came from that test-tuned protocol. Do not compare directly to
the new validation (2024) numbers; the new test (2025) numbers are in ../metrics.json once produced.
Model pickles are not copied (large, gitignored); predictions CSV is the original outputs/final_hotspot_predictions.csv.
