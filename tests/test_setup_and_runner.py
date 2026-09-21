"""Fresh-clone regressions: output folders are created, and run_pipeline reports stage failures cleanly."""

import sys

import pandas as pd
import pytest

import config
import preprocessing
import run_pipeline

DIRS = ["data/raw", "data/interim", "data/processed", "models", "outputs", "results"]


def test_ensure_output_dirs_creates_everything_in_an_empty_directory(tmp_path):
    assert not any((tmp_path / d).exists() for d in DIRS)
    config.ensure_output_dirs(tmp_path)
    assert all((tmp_path / d).is_dir() for d in DIRS)
    config.ensure_output_dirs(tmp_path)          # idempotent


def test_preprocessing_runs_when_only_the_raw_file_exists(tmp_path, monkeypatch):
    """Regression: in a fresh clone only data/raw/collisions_raw.csv exists (placed by the user);
    preprocessing used to crash writing the quality report because data/interim was missing."""
    raw = tmp_path / "data" / "raw" / "collisions_raw.csv"
    raw.parent.mkdir(parents=True)
    rows = [dict(collision_index=f"c{i}", collision_year=2021, longitude=-1.0 - i * 0.01, latitude=51.0 + i * 0.01,
                 date="15/03/2021", time="12:30", day_of_week=2, collision_severity=3, number_of_vehicles=2,
                 number_of_casualties=1, road_type=6, speed_limit=30, junction_detail=0, junction_control=4,
                 light_conditions=1, weather_conditions=1, road_surface_conditions=1, urban_or_rural_area=1)
            for i in range(5)]
    pd.DataFrame(rows).to_csv(raw, index=False)
    interim = tmp_path / "data" / "interim"
    assert not interim.exists()
    monkeypatch.setattr(preprocessing, "RAW_COLLISIONS_PATH", raw)
    monkeypatch.setattr(preprocessing, "CLEAN_COLLISIONS_PATH", interim / "collisions_clean.csv")
    monkeypatch.setattr(preprocessing, "QUALITY_REPORT_PATH", interim / "data_quality_report.csv")
    monkeypatch.setattr(preprocessing, "ensure_output_dirs", lambda: config.ensure_output_dirs(tmp_path), raising=False)
    preprocessing.run_preprocessing()
    assert len(pd.read_csv(interim / "collisions_clean.csv")) == 5
    assert (interim / "data_quality_report.csv").exists()


def _run_failing_stage(tmp_path, monkeypatch, capsys, with_lock, stage="train"):
    (tmp_path / "results").mkdir()
    if with_lock:
        (tmp_path / "results" / "test_evaluated.lock").write_text("done")
    monkeypatch.setattr(run_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(run_pipeline, "STAGES", [(stage, ["-c", "import sys; sys.exit(3)"])])
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"])
    with pytest.raises(SystemExit) as exit_info:
        run_pipeline.main()
    return exit_info.value.code, capsys.readouterr().err


def test_failed_stage_reports_name_and_exit_code_without_traceback(tmp_path, monkeypatch, capsys):
    code, err = _run_failing_stage(tmp_path, monkeypatch, capsys, with_lock=False)
    assert code == 3
    assert "stage train failed (exit 3)" in err
    assert "Traceback" not in err and "CalledProcessError" not in err
    assert "frozen by design" not in err


def test_lock_refusal_adds_frozen_by_design_hint(tmp_path, monkeypatch, capsys):
    code, err = _run_failing_stage(tmp_path, monkeypatch, capsys, with_lock=True)
    assert code == 3
    assert "stage train failed (exit 3)" in err and "frozen by design" in err


def test_lock_hint_is_only_for_training_stages(tmp_path, monkeypatch, capsys):
    code, err = _run_failing_stage(tmp_path, monkeypatch, capsys, with_lock=True, stage="preprocess")
    assert code == 3 and "stage preprocess failed (exit 3)" in err and "frozen by design" not in err
