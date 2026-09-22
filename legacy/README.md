# legacy/

Scripts from the pre-audit pipeline, kept only for reference. They read files that no longer exist
(`ml_features*.csv`), tune thresholds on the 2025 test set, and are **not part of the pipeline**.
See `results/old_run/` for the outputs they produced and `README.md` for the current pipeline.

`legacy/streamlit_dashboard/` is different from the rest of this directory: it is the original,
fully working Streamlit dashboard (superseded by `web/` as the primary interface), kept runnable
as a documented fallback rather than as dead reference code. See the main `README.md`'s Dashboard
section for how to run it.
