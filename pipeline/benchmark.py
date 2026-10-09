"""Initial, leak-free-at-fit-time SberIndex baseline benchmark; no tuned model claims."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_MODELS = ("last_value", "seasonal_naive", "damped_trend", "seasonal_scaled")


def baseline_forecasts(history, horizon):
    """Use only the history supplied by the caller; never interpolate future values."""
    y = np.asarray(history, dtype=float)
    observed = np.flatnonzero(np.isfinite(y))
    if not len(observed):
        raise ValueError("No observed history")
    last_idx = observed[-1]
    last = y[last_idx]
    target = len(y) + horizon - 1
    seasonal_index = target - 12
    seasonal = y[seasonal_index] if 0 <= seasonal_index < len(y) else np.nan
    if not np.isfinite(seasonal):
        seasonal = last
    recent = observed[-6:]
    slope = np.polyfit(recent, y[recent], 1)[0] if len(recent) > 1 else 0.0
    steps = target - last_idx
    damped = last + slope * sum(0.8 ** k for k in range(1, steps + 1))
    ratios = [y[i] / y[i - 12] for i in range(12, len(y))
              if np.isfinite(y[i]) and np.isfinite(y[i-12]) and y[i-12] > 0]
    scale = float(np.median(ratios)) if ratios else 1.0
    return dict(zip(BASE_MODELS, map(float, np.maximum(
        [last, seasonal, damped, seasonal * scale], 0.0))))


def load_data(path):
    df = pd.read_parquet(path)
    required = {"date", "territory_id", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing fields: {required - set(df.columns)}")
    value_field = "value" if "value" in df else "consumption"
    df = df.rename(columns={value_field: "y"})
    if df.duplicated(["territory_id", "category", "date"]).any():
        raise ValueError("Duplicate series-month keys")
    if df[["date", "territory_id", "category", "y"]].isna().any().any():
        raise ValueError("Null values require an explicit policy")
    df["y"] = pd.to_numeric(df["y"], errors="raise")
    if not np.isfinite(df.y).all() or (df.y < 0).any():
        raise ValueError("Invalid numeric values")
    df["ds"] = pd.to_datetime(df.date, format="%Y-%m", errors="raise")
    dates = pd.date_range(df.ds.min(), df.ds.max(), freq="MS")
    panel = df.pivot(index=["territory_id", "category"], columns="ds", values="y")
    panel = panel.reindex(columns=dates).sort_index()
    return df, panel, dates, value_field


def pilot_ids(panel, count):
    """Coverage-based diagnostic cohort; size ranking uses the first year only."""
    complete = panel.notna().all(axis=1)
    groups = complete.groupby(level=0).agg(["all", "count"])
    eligible = groups.index[groups["all"] & (groups["count"] == 6)]
    all_cat = panel.xs("Все категории", level=1).loc[eligible]
    ranking = all_cat.iloc[:, :12].mean(axis=1).sort_values(kind="stable")
    positions = np.minimum((np.arange(count) + 0.5) * len(ranking) / count,
                           len(ranking) - 1).astype(int)
    return sorted(set(map(int, ranking.index[positions])))


def metric_records(rows):
    result = []
    for (cohort, model, horizon), g in rows.groupby(["cohort", "model", "horizon"]):
        actual, predicted = g.actual.to_numpy(), g.predicted.to_numpy()
        err = predicted - actual
        denominator = np.sum((actual - actual.mean()) ** 2)
        result.append({"cohort": cohort, "model": model, "horizon": int(horizon),
                       "n": len(g), "mae": float(np.mean(np.abs(err))),
                       "r2_pooled": float(1-np.sum(err**2)/denominator) if denominator else None,
                       "wape": float(np.sum(np.abs(err))/np.sum(np.abs(actual)))})
    return result


def run(data_path, output, config, skip_prophet=False):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    df, panel, dates, value_field = load_data(data_path)
    selected = pilot_ids(panel, config["pilot_territories"])
    rows, failures = [], []
    for (territory, category), series in panel.iterrows():
        values = series.to_numpy(dtype=float)
        for cutoff in config["cutoffs"]:
            if cutoff >= len(dates) or not np.isfinite(values[:cutoff]).any():
                continue
            for h in config["horizons"]:
                target = cutoff + h - 1
                if target >= len(dates) or not np.isfinite(values[target]):
                    continue
                meta = {"territory_id": int(territory), "category": category,
                        "cutoff": str(dates[cutoff-1].date()), "horizon": h,
                        "target": str(dates[target].date()), "actual": float(values[target])}
                for model, pred in baseline_forecasts(values[:cutoff], h).items():
                    rows.append({**meta, "model": model, "predicted": pred, "cohort": "full_available"})
                    if territory in selected:
                        rows.append({**meta, "model": model, "predicted": pred, "cohort": "paired_pilot"})
    print(f"Simple baselines finished: {len(rows)} forecast rows", flush=True)
    if not skip_prophet:
        from prophet import Prophet
        logging.getLogger("cmdstanpy").disabled = True
        logging.getLogger("prophet").setLevel(logging.ERROR)
        pilot = panel.loc[panel.index.get_level_values(0).isin(selected)]
        for number, ((territory, category), series) in enumerate(pilot.iterrows(), 1):
            values = series.to_numpy(dtype=float)
            for cutoff in config["cutoffs"]:
                horizons = [h for h in config["horizons"] if cutoff+h <= len(dates)]
                if not horizons:
                    continue
                training = pd.DataFrame({"ds": dates[:cutoff], "y": values[:cutoff]}).dropna()
                future = pd.DataFrame({"ds": [dates[cutoff+h-1] for h in horizons]})
                for model_name, seasonal in [("prophet_default", "auto"), ("prophet_annual3", False)]:
                    try:
                        model = Prophet(yearly_seasonality=seasonal, weekly_seasonality=False,
                                        daily_seasonality=False, uncertainty_samples=0,
                                        changepoint_prior_scale=config["prophet"]["changepoint_prior_scale"],
                                        seasonality_prior_scale=config["prophet"]["seasonality_prior_scale"])
                        if model_name == "prophet_annual3":
                            model.add_seasonality("annual", period=365.25,
                                                  fourier_order=config["prophet"]["annual_fourier_order"])
                        model.fit(training, seed=config["seed"])
                        predictions = model.predict(future).yhat.to_numpy()
                        if not np.isfinite(predictions).all():
                            raise ValueError("Non-finite Prophet prediction")
                        for h, pred in zip(horizons, predictions):
                            target = cutoff+h-1
                            rows.append({"territory_id": int(territory), "category": category,
                                         "cutoff": str(dates[cutoff-1].date()), "horizon": h,
                                         "target": str(dates[target].date()), "actual": float(values[target]),
                                         "model": model_name, "predicted": float(max(pred, 0)),
                                         "cohort": "paired_pilot"})
                    except Exception as exc:
                        failures.append({"territory_id": int(territory), "category": category,
                                         "cutoff": cutoff, "model": model_name, "error": str(exc)})
            if number % 6 == 0:
                print(f"Prophet progress: {number}/{len(pilot)} series", flush=True)
    predictions = pd.DataFrame(rows)
    pilot_rows = predictions[predictions.cohort == "paired_pilot"]
    if not skip_prophet and not failures:
        paired_counts = pilot_rows.groupby(["model", "horizon"]).size().unstack(0)
        if paired_counts.nunique(axis=1).max() != 1:
            raise AssertionError("Models were not evaluated on the same pilot observations")
    sizes = panel.notna().sum(axis=1)
    summary = {"stage": "initial_baselines_not_final_submission", "config": config,
               "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
               "data": {"rows": len(df), "territories": int(df.territory_id.nunique()),
                        "categories": sorted(df.category.unique()), "months": len(dates),
                        "source_value_field": value_field, "series": len(panel),
                        "complete_series": int((sizes == len(dates)).sum()),
                        "incomplete_series": int((sizes < len(dates)).sum())},
               "pilot_territory_ids": selected,
               "metrics": metric_records(predictions), "failures": failures,
               "runtime_seconds": round(time.perf_counter()-start, 3),
               "versions": {name: importlib.metadata.version(name)
                            for name in ["numpy", "pandas", "pyarrow"] + ([] if skip_prophet else ["prophet"])},
               "python": platform.python_version(),
               "limitations": ["Pilot excludes incomplete series based on observed coverage; not a population estimate.",
                               "Only 24 months: horizon 12 has one forecast origin and no robust tuning split.",
                               "Overlapping evaluation targets are not independent observations.",
                               "Pooled R2 is dominated by between-series scale; not within-series skill.",
                               "Organizer Prophet configuration is unavailable; these are our reference implementations.",
                               "No structural-break detection, news or foundation model yet."]}
    (output/"results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions.to_json(output/"predictions.jsonl", orient="records", lines=True, force_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} fit failures; do not compare unpaired metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--skip-prophet", action="store_true")
    args = parser.parse_args()
    run(args.data, args.output, json.loads(args.config.read_text()), args.skip_prophet)
