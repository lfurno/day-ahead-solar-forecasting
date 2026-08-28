#!/usr/bin/env python3
"""Train and evaluate TabM on the final photovoltaic (PV) dataset.

The default setup uses:

- 20 regional shortwave-radiation forecasts;
- 20 regional direct-normal-irradiance forecasts;
- 20 regional global-tilted-irradiance forecasts;
- 20 regional cloud-cover forecasts;
- national installed PV capacity;
- the same month of the previous year as validation;
- training history beginning with the first row of the final dataset;
- January through June 2026 as the default forecast period.

To use a different training start date, rebuild the final dataset with the
desired start date.

Use --feature-groups to add or remove complete groups of model inputs. Run
--check-only to inspect the selected features and time splits without training.
Use --forecast-start and --forecast-end to change the months being evaluated.

Monthly metrics use the fixed local-time window from 06:00 through 20:59.
Aggregate metrics are reported both for this window and for all hours. The
reported metrics are the Coefficient of Determination (R²), Root Mean Squared
Error (RMSE), Mean Absolute Error (MAE), Mean Bias Error (MBE), and Weighted
Absolute Percentage Error (WAPE).

The script saves forecast records, monthly metrics, aggregate metrics, and the
selected feature names in the output directory. It also plots actual versus
predicted generation, monthly RMSE, and residuals by local hour.

Dependencies:

    python -m pip install -r requirements-model.txt

Examples:

    python src/train_tabm.py

    python src/train_tabm.py \
        --input-csv path/to/final_dataset.csv \
        --output-dir path/to/output-dir \
        --forecast-start 2026-01 \
        --forecast-end 2026-06

    python src/train_tabm.py \
        --feature-groups shortwave-radiation global-tilted-irradiance \
        temperature cloud-cover national-capacity
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rtdl_num_embeddings import PeriodicEmbeddings
from sklearn.preprocessing import StandardScaler
from tabm import TabM
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "share" / "final_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "share" / "tabm"
DEFAULT_FORECAST_START = "2026-01"
DEFAULT_FORECAST_END = "2026-06"

TIME_UTC_COLUMN = "time_utc"
TIME_LOCAL_COLUMN = "time_local"
ACTUAL_GENERATION_COLUMN = "actual_generation_gw"
NATIONAL_CAPACITY_COLUMN = "national_installed_pv_mw_start"
REGIONAL_CAPACITY_SUFFIX = "_installed_pv_mw_start"

WEATHER_GROUPS = {
    "shortwave-radiation": "_shortwave_radiation_previous_day1",
    "direct-radiation": "_direct_radiation_previous_day1",
    "diffuse-radiation": "_diffuse_radiation_previous_day1",
    "direct-normal-irradiance": "_direct_normal_irradiance_previous_day1",
    "global-tilted-irradiance": "_global_tilted_irradiance_previous_day1",
    "precipitation": "_precipitation_previous_day1",
    "temperature": "_temperature_2m_previous_day1",
    "cloud-cover": "_cloud_cover_previous_day1",
    "wind-speed": "_wind_speed_10m_previous_day1",
    "wind-direction": "_wind_direction_10m_previous_day1",
    "relative-humidity": "_relative_humidity_2m_previous_day1"
}
CAPACITY_GROUPS = (
    "national-capacity",
    "regional-capacity"
)
FEATURE_GROUPS = (*WEATHER_GROUPS, *CAPACITY_GROUPS)
DEFAULT_FEATURE_GROUPS = (
    "shortwave-radiation",
    "direct-normal-irradiance",
    "global-tilted-irradiance",
    "cloud-cover",
    "national-capacity"
)

ENSEMBLE_MEMBERS = 32
BLOCK_WIDTH = 256
NUMBER_OF_BLOCKS = 2
DROPOUT = 0.10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0003
BATCH_SIZE = 512
MAX_EPOCHS = 80
PATIENCE = 8
MINIMUM_IMPROVEMENT = 0.0001
EMBEDDING_SIZE = 16
NUMBER_OF_FREQUENCIES = 16
FREQUENCY_SCALE = 0.1


def parse_args() -> argparse.Namespace:
    """Read the dataset, feature groups, and run settings."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-csv",
        dest="input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Final dataset CSV. Default: share/final_dataset.csv."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model results. Default: share/tabm."
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=FEATURE_GROUPS,
        default=list(DEFAULT_FEATURE_GROUPS),
        help=(
            "Feature groups passed to TabM. Default: shortwave-radiation, "
            "direct-normal-irradiance, global-tilted-irradiance, "
            "cloud-cover, and national-capacity."
        )
    )
    parser.add_argument(
        "--forecast-start",
        default=DEFAULT_FORECAST_START,
        help="First forecast month in YYYY-MM format. Default: 2026-01."
    )
    parser.add_argument(
        "--forecast-end",
        default=DEFAULT_FORECAST_END,
        help="Last forecast month in YYYY-MM format. Default: 2026-06."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads used by PyTorch. Default: 4."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check the dataset, features, and splits without training."
    )

    return parser.parse_args()


def parse_forecast_month(value: str, option: str) -> pd.Timestamp:
    """Read a forecast month written as YYYY-MM."""

    try:
        month = pd.to_datetime(value, format="%Y-%m", errors="raise")
    except ValueError as exc:
        raise ValueError(f"{option} must use YYYY-MM format.") from exc

    if month.strftime("%Y-%m") != value:
        raise ValueError(f"{option} must use YYYY-MM format.")

    return month


def create_forecast_months(start: str, end: str) -> list[pd.Timestamp]:
    """List every month from the first forecast month to the last."""

    first_month = parse_forecast_month(start, "--forecast-start")
    last_month = parse_forecast_month(end, "--forecast-end")

    if last_month < first_month:
        raise ValueError("--forecast-end must not be before --forecast-start.")

    return list(pd.date_range(first_month, last_month, freq="MS"))


def select_features(
    columns: list[str],
    selected_groups: list[str]
) -> tuple[list[str], dict[str, list[str]]]:
    """Find the dataset columns belonging to the selected feature groups."""

    groups: dict[str, list[str]] = {}

    for name in selected_groups:
        if name in WEATHER_GROUPS:
            suffix = WEATHER_GROUPS[name]
            selected = [column for column in columns if column.endswith(suffix)]

            if len(selected) != 20:
                raise RuntimeError(
                    f"Feature group {name!r} has {len(selected)} columns. "
                    "Expected 20."
                )

        elif name == "national-capacity":
            selected = [NATIONAL_CAPACITY_COLUMN]

            if NATIONAL_CAPACITY_COLUMN not in columns:
                raise RuntimeError(
                    f"Missing feature column: {NATIONAL_CAPACITY_COLUMN}."
                )

        else:
            selected = [
                column
                for column in columns
                if column.endswith(REGIONAL_CAPACITY_SUFFIX)
                and column != NATIONAL_CAPACITY_COLUMN
            ]

            if len(selected) != 20:
                raise RuntimeError(
                    f"Feature group {name!r} has {len(selected)} columns. "
                    "Expected 20."
                )

        groups[name] = selected

    # Keep weather features in the same order as the final CSV.
    weather_suffixes = [
        WEATHER_GROUPS[name]
        for name in selected_groups
        if name in WEATHER_GROUPS
    ]
    features = [
        column
        for column in columns
        if any(column.endswith(suffix) for suffix in weather_suffixes)
    ]

    if "national-capacity" in groups:
        features.extend(groups["national-capacity"])

    if "regional-capacity" in groups:
        features.extend(groups["regional-capacity"])

    return features, groups


def read_dataset_and_select_features(
    path: Path,
    selected_groups: list[str]
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Read the final dataset and select the requested model features."""

    if not path.exists():
        raise FileNotFoundError(f"Final dataset not found: {path}.")

    data = pd.read_csv(path)

    required = {
        TIME_UTC_COLUMN,
        TIME_LOCAL_COLUMN,
        ACTUAL_GENERATION_COLUMN
    }

    missing = required - set(data.columns)

    if missing:
        raise RuntimeError(f"Final dataset is missing columns: {sorted(missing)}.")

    features, groups = select_features(list(data.columns), selected_groups)

    data[TIME_UTC_COLUMN] = pd.to_datetime(
        data[TIME_UTC_COLUMN],
        utc=True,
        errors="raise"
    )
    data[TIME_LOCAL_COLUMN] = pd.to_datetime(
        data[TIME_LOCAL_COLUMN],
        errors="raise"
    )
    numeric_columns = [ACTUAL_GENERATION_COLUMN, *features]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="raise"
    )
    data = data.sort_values(TIME_UTC_COLUMN).reset_index(drop=True)

    if data.empty:
        raise RuntimeError("Final dataset is empty.")

    if data[TIME_UTC_COLUMN].duplicated().any():
        raise RuntimeError("Final dataset contains duplicate UTC timestamps.")

    if data[TIME_LOCAL_COLUMN].duplicated().any():
        raise RuntimeError("Final dataset contains duplicate local timestamps.")

    if data[numeric_columns].isna().any(axis=None):
        raise RuntimeError("Selected model columns contain missing values.")

    infinite_values = [
        float("inf"),
        float("-inf")
    ]

    if data[numeric_columns].isin(infinite_values).any(axis=None):
        raise RuntimeError("Selected model columns contain infinite values.")

    return data, features, groups


def make_split(
    data: pd.DataFrame,
    forecast_month: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into training, seasonal validation, full, and test sets."""

    test_end = forecast_month + pd.DateOffset(months=1)
    validation_start = forecast_month - pd.DateOffset(years=1)
    validation_end = validation_start + pd.DateOffset(months=1)
    local_time = data[TIME_LOCAL_COLUMN]

    history = data.loc[local_time < forecast_month]
    validation_mask = history[TIME_LOCAL_COLUMN].between(
        validation_start,
        validation_end,
        inclusive="left"
    )
    training = history.loc[~validation_mask].copy()
    validation = history.loc[validation_mask].copy()
    full = history.copy()
    test = data.loc[
        local_time.between(forecast_month, test_end, inclusive="left")
    ].copy()

    if min(len(training), len(validation), len(full), len(test)) == 0:
        raise RuntimeError(
            f"{forecast_month:%Y-%m} has an empty split: "
            f"training={len(training)}, "
            f"validation={len(validation)}, "
            f"full={len(full)}, "
            f"test={len(test)}."
        )

    return training, validation, full, test


def create_model(number_of_features: int) -> TabM:
    """Create the TabM model."""

    embeddings = PeriodicEmbeddings(
        number_of_features,
        d_embedding=EMBEDDING_SIZE,
        n_frequencies=NUMBER_OF_FREQUENCIES,
        frequency_init_scale=FREQUENCY_SCALE,
        lite=False
    )

    return TabM.make(
        n_num_features=number_of_features,
        d_out=1,
        num_embeddings=embeddings,
        n_blocks=NUMBER_OF_BLOCKS,
        d_block=BLOCK_WIDTH,
        dropout=DROPOUT,
        k=ENSEMBLE_MEMBERS,
        start_scaling_init="random-signs"
    )


def create_loader(
    data: pd.DataFrame,
    features: list[str],
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    seed: int
) -> DataLoader:
    """Scale the training data and create shuffled batches."""

    inputs = torch.tensor(
        input_scaler.transform(data[features]).astype(np.float32)
    )
    target = torch.tensor(
        target_scaler.transform(
            data[[ACTUAL_GENERATION_COLUMN]]
        ).ravel().astype(np.float32)
    )

    return DataLoader(
        TensorDataset(inputs, target),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )


def fit_scalers(
    data: pd.DataFrame,
    features: list[str]
) -> tuple[StandardScaler, StandardScaler]:
    """Fit input and target scaling on training rows only."""

    input_scaler = StandardScaler().fit(
        data[features].astype(np.float32)
    )
    target_scaler = StandardScaler().fit(
        data[[ACTUAL_GENERATION_COLUMN]].astype(np.float32)
    )

    return input_scaler, target_scaler


def train_epoch(
    model: TabM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer
) -> None:
    """Train TabM for one epoch."""

    model.train()

    for inputs, target in loader:
        optimizer.zero_grad(set_to_none=True)
        member_predictions = model(inputs).squeeze(-1)
        loss = ((member_predictions - target[:, None]) ** 2).mean()
        loss.backward()
        optimizer.step()


def select_best_epoch(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    seed: int
) -> tuple[int, float]:
    """Use seasonal validation to select the number of training epochs."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    input_scaler, target_scaler = fit_scalers(training, features)
    loader = create_loader(
        training,
        features,
        input_scaler,
        target_scaler,
        seed
    )
    validation_inputs = torch.tensor(
        input_scaler.transform(validation[features]).astype(np.float32)
    )
    validation_target = torch.tensor(
        target_scaler.transform(
            validation[[ACTUAL_GENERATION_COLUMN]]
        ).ravel().astype(np.float32)
    )
    model = create_model(len(features))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    best_loss = float("inf")
    best_epoch = 1
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, loader, optimizer)
        model.eval()

        with torch.no_grad():
            prediction = model(validation_inputs).squeeze(-1).mean(dim=1)
            validation_loss = (
                (prediction - validation_target) ** 2
            ).mean().item()

        if validation_loss < best_loss - MINIMUM_IMPROVEMENT:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    return best_epoch, best_loss


def train_and_predict(
    full: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seed: int,
    epochs: int
) -> np.ndarray:
    """Retrain TabM on all available history and predict the forecast month."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    input_scaler, target_scaler = fit_scalers(full, features)
    loader = create_loader(
        full,
        features,
        input_scaler,
        target_scaler,
        seed
    )
    model = create_model(len(features))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    for _ in range(epochs):
        train_epoch(model, loader, optimizer)

    test_inputs = torch.tensor(
        input_scaler.transform(test[features]).astype(np.float32)
    )
    model.eval()

    with torch.no_grad():
        scaled_prediction = (
            model(test_inputs).squeeze(-1).mean(dim=1).numpy()
        )

    prediction = target_scaler.inverse_transform(
        scaled_prediction[:, None]
    ).ravel()

    return np.clip(prediction, 0.0, None)


def evaluation_rows_06_20(data: pd.DataFrame) -> np.ndarray:
    """Select the fixed evaluation window from 06:00 through 20:59 local time."""

    return data[TIME_LOCAL_COLUMN].dt.hour.between(6, 20).to_numpy()


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray
) -> dict[str, float]:
    """Calculate the prediction errors."""

    errors = prediction - actual
    absolute_errors = np.abs(errors)
    residual_sum = float(np.sum(errors ** 2))
    total_sum = float(np.sum((actual - actual.mean()) ** 2))

    # WAPE compares the total absolute error with total actual generation.
    wape = 100.0 * np.sum(absolute_errors) / np.sum(actual)

    return {
        "r2": 1.0 - residual_sum / total_sum,
        "rmse_gw": float(np.sqrt(np.mean(errors ** 2))),
        "mae_gw": float(np.mean(absolute_errors)),
        "mbe_gw": float(np.mean(errors)),
        "wape_percent": float(wape)
    }


def aggregate_metrics(forecast_records: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics over all forecast months."""

    evaluation_rows = evaluation_rows_06_20(forecast_records)
    evaluation_periods = {
        "daylight_06_20": evaluation_rows,
        "all_hours": np.ones(len(forecast_records), dtype=bool)
    }
    rows = []

    for period, selected in evaluation_periods.items():
        metrics = calculate_metrics(
            forecast_records.loc[
                selected,
                ACTUAL_GENERATION_COLUMN
            ].to_numpy(),
            forecast_records.loc[selected, "prediction_gw"].to_numpy()
        )
        rows.append({"scope": period, **metrics})

    return pd.DataFrame(rows)


def run_backtest(
    data: pd.DataFrame,
    features: list[str],
    forecast_months: list[pd.Timestamp],
    seed: int,
    output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate one TabM model for each forecast month."""

    forecast_records = pd.DataFrame()
    monthly_rows = []

    for forecast_month in forecast_months:
        started = time.time()
        training, validation, full, test = make_split(
            data,
            forecast_month
        )
        validation_month = forecast_month - pd.DateOffset(years=1)

        print(
            f"Test {forecast_month:%Y-%m}: validation month "
            f"{validation_month:%Y-%m}, training {len(training):,}, "
            f"validation {len(validation):,}, full {len(full):,}, "
            f"test {len(test):,}."
        )

        best_epoch, validation_loss = select_best_epoch(
            training,
            validation,
            features,
            seed
        )
        prediction = train_and_predict(
            full,
            test,
            features,
            seed,
            best_epoch
        )
        test["prediction_gw"] = prediction
        test["forecast_month"] = (
            forecast_month.strftime("%Y-%m")
        )
        forecast_records = pd.concat(
            [
                forecast_records,
                test[
                    [
                        TIME_UTC_COLUMN,
                        TIME_LOCAL_COLUMN,
                        ACTUAL_GENERATION_COLUMN,
                        "prediction_gw",
                        "forecast_month"
                    ]
                ]
            ],
            ignore_index=True
        )

        evaluation_rows = evaluation_rows_06_20(test)
        metrics = calculate_metrics(
            test.loc[
                evaluation_rows,
                ACTUAL_GENERATION_COLUMN
            ].to_numpy(),
            test.loc[
                evaluation_rows,
                "prediction_gw"
            ].to_numpy()
        )
        monthly_rows.append(
            {
                "forecast_month": forecast_month.strftime("%Y-%m"),
                "validation_month": validation_month.strftime("%Y-%m"),
                "best_epoch": best_epoch,
                "validation_scaled_mse": validation_loss,
                **metrics,
                "seconds": round(time.time() - started, 1),
                "training_rows": len(training),
                "validation_rows": len(validation),
                "full_rows": len(full),
                "test_rows": len(test)
            }
        )

        monthly_metrics = pd.DataFrame(monthly_rows)
        forecast_records.to_csv(
            output_dir / "forecast_records.csv",
            index=False
        )
        monthly_metrics.to_csv(output_dir / "monthly_metrics.csv", index=False)

        message = (
            f"Best epoch: {best_epoch}. RMSE for local hours 06-20: "
            f"{metrics['rmse_gw']:.6f} GW. WAPE: "
            f"{metrics['wape_percent']:.3f}%."
        )

        print(message)

    return forecast_records, monthly_metrics


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save a plot and close it."""

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_actual_vs_predicted(
    forecast_records: pd.DataFrame,
    output_dir: Path
) -> None:
    """Compare actual and predicted generation during evaluation hours."""

    selected = forecast_records.loc[
        evaluation_rows_06_20(forecast_records)
    ]
    lower = min(
        selected[ACTUAL_GENERATION_COLUMN].min(),
        selected["prediction_gw"].min()
    )
    upper = max(
        selected[ACTUAL_GENERATION_COLUMN].max(),
        selected["prediction_gw"].max()
    )

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        selected[ACTUAL_GENERATION_COLUMN],
        selected["prediction_gw"],
        color="tab:blue",
        alpha=0.25,
        s=12
    )
    axis.plot(
        [lower, upper],
        [lower, upper],
        color="tab:red",
        linewidth=1.5,
        label="Perfect prediction"
    )
    axis.set_title("Actual versus predicted PV generation, local hours 06-20")
    axis.set_xlabel("Actual generation (GW)")
    axis.set_ylabel("Predicted generation (GW)")
    axis.legend()

    save_figure(figure, output_dir / "actual_vs_predicted.png")


def plot_monthly_rmse(
    monthly_metrics: pd.DataFrame,
    output_dir: Path
) -> None:
    """Plot RMSE for each forecast month."""

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        monthly_metrics["forecast_month"],
        monthly_metrics["rmse_gw"],
        color="tab:blue"
    )
    axis.set_title("Monthly RMSE, local hours 06-20")
    axis.set_xlabel("Forecast month")
    axis.set_ylabel("RMSE (GW)")
    axis.tick_params(axis="x", rotation=45)

    save_figure(figure, output_dir / "monthly_rmse.png")


def plot_residuals_by_local_hour(
    forecast_records: pd.DataFrame,
    output_dir: Path
) -> None:
    """Plot prediction errors for every local hour."""

    residuals = (
        forecast_records["prediction_gw"]
        - forecast_records[ACTUAL_GENERATION_COLUMN]
    )
    local_hours = forecast_records[TIME_LOCAL_COLUMN].dt.hour
    hours = sorted(local_hours.unique())
    values = [residuals.loc[local_hours.eq(hour)] for hour in hours]

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(
        values,
        tick_labels=hours,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "tab:blue"},
        medianprops={"color": "black"}
    )
    axis.axhline(0, color="tab:red", linewidth=1)
    axis.set_title("Prediction residuals by local hour")
    axis.set_xlabel("Local hour")
    axis.set_ylabel("Prediction minus actual generation (GW)")

    save_figure(figure, output_dir / "residuals_by_local_hour.png")


def create_plots(
    forecast_records: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create the diagnostic plots for the completed backtest."""

    plot_actual_vs_predicted(forecast_records, output_dir)
    plot_monthly_rmse(monthly_metrics, output_dir)
    plot_residuals_by_local_hour(forecast_records, output_dir)


def main() -> int:
    """Check the inputs, run the monthly backtest, and save its results."""

    args = parse_args()

    if args.threads < 1:
        raise ValueError("--threads must be positive.")

    forecast_months = create_forecast_months(
        args.forecast_start,
        args.forecast_end
    )

    data, features, groups = read_dataset_and_select_features(
        args.input.resolve(),
        args.feature_groups
    )

    print(
        f"Dataset: {len(data):,} rows from "
        f"{data[TIME_LOCAL_COLUMN].min()} to "
        f"{data[TIME_LOCAL_COLUMN].max()}."
    )
    print(f"Selected features: {len(features)}.")

    for name, columns in groups.items():
        print(f"- {name}: {len(columns)}.")

    for forecast_month in forecast_months:
        training, validation, full, test = make_split(
            data,
            forecast_month
        )
        print(
            f"- {forecast_month:%Y-%m}: training {len(training):,}, "
            f"validation {len(validation):,}, full {len(full):,}, "
            f"test {len(test):,}."
        )

    if args.check_only:
        print("Dataset and split checks completed. No model was trained.")

        return 0

    torch.set_num_threads(args.threads)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": features}).to_csv(
        output_dir / "selected_features.csv",
        index=False
    )
    forecast_records, monthly_metrics = run_backtest(
        data,
        features,
        forecast_months,
        args.seed,
        output_dir
    )
    metrics = aggregate_metrics(forecast_records)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    print("Final metrics.")
    print(metrics.to_string(index=False))
    print(f"Results: {output_dir}.")

    create_plots(forecast_records, monthly_metrics, output_dir)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
