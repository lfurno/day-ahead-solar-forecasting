#!/usr/bin/env python3
"""Explore the final hourly photovoltaic (PV) dataset.

The script reads the final CSV without modifying it and produces:

- column_summary.csv with data quality and descriptive statistics;
- daily_coverage.csv with expected and observed local hours;
- generation summaries by hour, month, and year when PV generation is present;
- feature_target_correlations.csv with correlations with PV generation;
- weather_variable_correlations.csv with regional correlation summaries;
- generation_solar_shift_correlations.csv to check the temporal alignment
  between PV generation and the national solar proxy;
- capacity_by_month.csv with national and regional installed PV capacity;
- diagnostic plots for data coverage, weather correlations, PV capacity factor,
  and its relationship with the national solar proxy.

The available analyses depend on the columns included in the final dataset.
The three time columns must always be present.

Dependencies:

    python -m pip install -r requirements-eda.txt

Examples:

    python src/explore_final_dataset.py

    python src/explore_final_dataset.py \
        --input-csv path/to/final_dataset.csv \
        --output-dir path/to/output-dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROME_TIMEZONE = ZoneInfo("Europe/Rome")

DEFAULT_INPUT = PROJECT_ROOT / "share" / "final_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "share" / "eda"

TIME_COLUMNS = (
    "time_utc",
    "time_local",
    "utc_offset"
)
ACTUAL_GENERATION_COLUMN = "actual_generation_gw"
NATIONAL_CAPACITY_COLUMN = "national_installed_pv_mw_start"
REGIONAL_CAPACITY_SUFFIX = "_installed_pv_mw_start"
REGION_NAMES = (
    "abruzzo",
    "basilicata",
    "calabria",
    "campania",
    "emilia_romagna",
    "friuli_venezia_giulia",
    "lazio",
    "liguria",
    "lombardia",
    "marche",
    "molise",
    "piemonte",
    "puglia",
    "sardegna",
    "sicilia",
    "toscana",
    "trentino_alto_adige",
    "umbria",
    "valle_d_aosta",
    "veneto"
)

ColumnGroups = dict[str, list[str]]


def parse_args() -> argparse.Namespace:
    """Read the input file, output directory, and plot options."""

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
        help="Directory for EDA results. Default: share/eda."
    )
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    """Read the final dataset and prepare its time and numeric columns."""

    data = pd.read_csv(path)

    if data.empty:
        raise RuntimeError(f"Input dataset is empty: {path}.")

    missing_time_columns = set(TIME_COLUMNS) - set(data.columns)

    if missing_time_columns:
        raise RuntimeError(
            "The final dataset is missing time columns: "
            f"{sorted(missing_time_columns)}."
        )

    data["time_utc"] = pd.to_datetime(
        data["time_utc"],
        utc=True,
        errors="raise"
    )
    data["time_local"] = pd.to_datetime(
        data["time_local"],
        errors="raise"
    )

    # The data are expected to have been validated when the final dataset was
    # created and to contain only time identifiers and numeric model data.
    numeric_columns = [
        column
        for column in data.columns
        if column not in TIME_COLUMNS
    ]

    numeric_data = data[numeric_columns].apply(
        pd.to_numeric,
        errors="raise"
    )
    data = pd.concat(
        [
            data[list(TIME_COLUMNS)],
            numeric_data
        ],
        axis=1
    )

    return data.sort_values("time_utc").reset_index(drop=True).copy()


def classify_columns(data: pd.DataFrame) -> ColumnGroups:
    """Separate time, target, capacity, and weather columns."""

    regional_capacity = [
        column
        for column in data.columns
        if column.endswith(REGIONAL_CAPACITY_SUFFIX)
        and column != NATIONAL_CAPACITY_COLUMN
    ]
    national_capacity = []

    if NATIONAL_CAPACITY_COLUMN in data.columns:
        national_capacity.append(NATIONAL_CAPACITY_COLUMN)

    actual_generation = []

    if ACTUAL_GENERATION_COLUMN in data.columns:
        actual_generation.append(ACTUAL_GENERATION_COLUMN)

    known_columns = {
        *TIME_COLUMNS,
        *actual_generation,
        *national_capacity,
        *regional_capacity
    }
    weather = [
        column
        for column in data.columns
        if column not in known_columns
    ]

    return {
        "time": list(TIME_COLUMNS),
        "actual_generation": actual_generation,
        "national_capacity": national_capacity,
        "regional_capacity": regional_capacity,
        "weather": weather
    }


def category_by_column(groups: ColumnGroups) -> dict[str, str]:
    """Map each column name to its category."""

    return {
        column: category
        for category, columns in groups.items()
        for column in columns
    }


def build_column_summary(
    data: pd.DataFrame,
    groups: ColumnGroups
) -> pd.DataFrame:
    """Calculate quality checks and descriptive statistics for each column."""

    categories = category_by_column(groups)
    rows = []

    for column in data.columns:
        values = data[column]
        is_numeric = pd.api.types.is_numeric_dtype(values)
        row = {
            "column": column,
            "category": categories[column],
            "rows": len(values),
            "missing_values": int(values.isna().sum()),
            "missing_fraction": float(values.isna().mean()),
            "unique_values": int(values.nunique(dropna=True)),
            "is_constant": values.nunique(dropna=False) <= 1,
            "infinite_values": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan
        }

        if is_numeric:
            finite = values.replace([np.inf, -np.inf], np.nan)
            row.update(
                {
                    "infinite_values": int(np.isinf(values).sum()),
                    "mean": finite.mean(),
                    "std": finite.std(),
                    "min": finite.min(),
                    "q25": finite.quantile(0.25),
                    "median": finite.median(),
                    "q75": finite.quantile(0.75),
                    "max": finite.max()
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def expected_local_rows(day: pd.Timestamp) -> int:
    """Return the expected unique local hours for an Italian calendar day,
    accounting for clock changes."""

    start = day.tz_localize(ROME_TIMEZONE)
    end = (day + pd.Timedelta(days=1)).tz_localize(ROME_TIMEZONE)
    physical_hours = int(
        (end.tz_convert("UTC") - start.tz_convert("UTC"))
        / pd.Timedelta(hours=1)
    )

    # Autumn has 25 physical hours, but the repeated hour is kept only
    # once in the final dataset.
    return min(physical_hours, 24)


def build_daily_coverage(data: pd.DataFrame) -> pd.DataFrame:
    """Compare observed rows with the expected hours of each local day."""

    local_dates = data["time_local"].dt.normalize()
    calendar = pd.date_range(
        local_dates.min(),
        local_dates.max(),
        freq="1D"
    )

    observed_rows = local_dates.value_counts().reindex(
        calendar,
        fill_value=0
    )
    expected_rows = pd.Series(
        [expected_local_rows(day) for day in calendar],
        index=calendar
    )

    missing_rows = (expected_rows - observed_rows).clip(lower=0)
    extra_rows = (observed_rows - expected_rows).clip(lower=0)

    return pd.DataFrame(
        {
            "local_date": calendar,
            "expected_rows": expected_rows.to_numpy(),
            "observed_rows": observed_rows.to_numpy(),
            "missing_rows": missing_rows.to_numpy(),
            "extra_rows": extra_rows.to_numpy()
        }
    )


def summarize_generation(
    data: pd.DataFrame,
    group_column: str
) -> pd.DataFrame:
    """Calculate generation statistics by hour, month or year."""

    grouped = data.groupby(group_column)[ACTUAL_GENERATION_COLUMN]
    summary = grouped.agg(
        rows="size",
        mean_gw="mean",
        median_gw="median",
        standard_deviation_gw="std",
        minimum_gw="min",
        maximum_gw="max"
    )
    summary["zero_fraction"] = grouped.apply(lambda values: values.eq(0).mean())

    return summary.reset_index()


def build_feature_correlations(
    data: pd.DataFrame,
    groups: ColumnGroups
) -> pd.DataFrame:
    """Calculate correlations between selected features and actual generation."""

    features = [
        *groups["weather"],
        *groups["national_capacity"],
        *groups["regional_capacity"]
    ]
    categories = category_by_column(groups)
    correlations = data[features].corrwith(
        data[ACTUAL_GENERATION_COLUMN]
    ).dropna()
    result = correlations.rename("correlation").to_frame()
    result["absolute_correlation"] = result["correlation"].abs()
    result["category"] = [
        categories[column]
        for column in result.index
    ]

    return (
        result.sort_values("absolute_correlation", ascending=False)
        .rename_axis("feature")
        .reset_index()
    )


def weather_variable_name(column: str) -> str:
    """Remove the regional prefix and forecast suffix from a weather column."""

    for region in REGION_NAMES:
        prefix = f"{region}_"

        if column.startswith(prefix):
            variable = column.removeprefix(prefix)

            return variable.removesuffix("_previous_day1")

    return column.removesuffix("_previous_day1")


def build_weather_variable_correlations(
    correlations: pd.DataFrame
) -> pd.DataFrame:
    """Summarize regional correlations for each type of weather variable."""

    weather = correlations.loc[
        correlations["category"] == "weather"
    ].copy()
    weather["weather_variable"] = weather["feature"].map(
        weather_variable_name
    )
    summary = weather.groupby("weather_variable")["correlation"].agg(
        regional_columns="size",
        mean_correlation="mean",
        median_correlation="median",
        minimum_correlation="min",
        maximum_correlation="max"
    )
    summary["absolute_median_correlation"] = summary[
        "median_correlation"
    ].abs()

    return summary.sort_values(
        "absolute_median_correlation",
        ascending=False
    ).reset_index()


def build_capacity_by_month(
    data: pd.DataFrame,
    groups: ColumnGroups
) -> pd.DataFrame:
    """Summarize national and regional installed PV capacity by month."""

    capacity_columns = [
        *groups["national_capacity"],
        *groups["regional_capacity"]
    ]
    working = data.assign(
        month=data["time_local"].dt.to_period("M").astype(str)
    )
    # Each monthly capacity value is repeated across hourly rows, so keep only
    # the first value for each month.
    monthly = working.groupby("month")[capacity_columns].first().reset_index()

    if groups["regional_capacity"]:
        monthly["regional_capacity_total_mw"] = monthly[
            groups["regional_capacity"]
        ].sum(axis=1)

    if groups["national_capacity"] and groups["regional_capacity"]:
        monthly["regional_minus_national_mw"] = (
            monthly["regional_capacity_total_mw"]
            - monthly[NATIONAL_CAPACITY_COLUMN]
        )

    return monthly


def find_solar_columns(weather_columns: list[str]) -> list[str]:
    """Select regional solar columns for the national solar proxy."""

    # Prefer tilted irradiance because it better represents
    # PV-oriented surfaces.
    tilted = [
        column
        for column in weather_columns
        if "global_tilted_irradiance" in column
    ]

    if tilted:
        return tilted

    return [
        column
        for column in weather_columns
        if "shortwave_radiation" in column
    ]


def build_national_solar_proxy(
    data: pd.DataFrame,
    solar_columns: list[str],
    groups: ColumnGroups
) -> pd.Series:
    """Combine regional solar data into one national value, weighted by PV
    capacity when available."""

    regional_pairs = []

    for region in REGION_NAMES:
        solar_column = next(
            (
                column
                for column in solar_columns
                if column.startswith(f"{region}_")
            ),
            None
        )
        capacity_column = f"{region}{REGIONAL_CAPACITY_SUFFIX}"

        if (
            solar_column is not None
            and capacity_column in groups["regional_capacity"]
        ):
            regional_pairs.append((solar_column, capacity_column))

    # Use capacity weights only when every selected solar region has a match.
    if len(regional_pairs) == len(solar_columns):
        weighted_solar = pd.Series(0.0, index=data.index)
        total_capacity = pd.Series(0.0, index=data.index)

        for solar_column, capacity_column in regional_pairs:
            weighted_solar += data[solar_column] * data[capacity_column]
            total_capacity += data[capacity_column]

        national_solar_proxy = weighted_solar / total_capacity

        return national_solar_proxy

    return data[solar_columns].mean(axis=1)


def build_solar_shift_correlations(
    data: pd.DataFrame,
    national_solar_proxy: pd.Series
) -> pd.DataFrame:
    """Check how solar-generation correlation changes with hourly time shifts."""

    generation_by_time = pd.Series(
        data[ACTUAL_GENERATION_COLUMN].to_numpy(),
        index=data["time_utc"]
    )

    solar_by_time = pd.Series(
        national_solar_proxy.to_numpy(),
        index=data["time_utc"]
    )

    rows = []

    for shift_hours in range(-3, 4):
        shifted_solar = solar_by_time.shift(
            freq=pd.Timedelta(hours=shift_hours)
        )

        paired = pd.concat(
            [generation_by_time, shifted_solar],
            axis=1,
            join="inner"
        ).dropna()

        rows.append(
            {
                "solar_shift_hours": shift_hours,
                "correlation": paired.iloc[:, 0].corr(paired.iloc[:, 1]),
                "paired_hours": len(paired)
            }
        )

    return pd.DataFrame(rows)


def build_capacity_factor(
    data: pd.DataFrame,
    groups: ColumnGroups
) -> pd.Series | None:
    """Calculate actual generation divided by available installed PV capacity."""

    if not groups["actual_generation"]:
        return None

    if groups["national_capacity"]:
        capacity_mw = data[NATIONAL_CAPACITY_COLUMN]
    elif groups["regional_capacity"]:
        capacity_mw = data[groups["regional_capacity"]].sum(axis=1)
    else:
        return None

    capacity_gw = capacity_mw / 1000
    capacity_factor = data[ACTUAL_GENERATION_COLUMN] / capacity_gw

    return capacity_factor.replace([np.inf, -np.inf], np.nan)


def save_figure(figure: Figure, path: Path) -> None:
    """Save a plot and close it."""

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_coverage_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    """Show observed and missing local hours across the complete date range."""

    first_date = data["time_local"].min().normalize()
    last_date = data["time_local"].max().normalize()
    dates = pd.date_range(first_date, last_date, freq="1D")
    date_positions = {
        day: position
        for position, day in enumerate(dates)
    }
    coverage = np.zeros((24, len(dates)), dtype=float)

    for local_time in data["time_local"]:
        day = local_time.normalize()
        coverage[local_time.hour, date_positions[day]] = 1

    # The local 02:00 hour does not exist when clocks move forward in March.
    for position, day in enumerate(dates):
        if expected_local_rows(day) == 23:
            coverage[2, position] = np.nan

    colours = ListedColormap(["tab:red", "tab:blue"])
    colours.set_bad("lightgray")

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.imshow(
        coverage,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=colours,
        vmin=0,
        vmax=1
    )
    month_positions = [
        position
        for position, day in enumerate(dates)
        if day.day == 1 or position == 0
    ]
    axis.set_xticks(month_positions)
    axis.set_xticklabels(
        [dates[position].strftime("%Y-%m") for position in month_positions],
        rotation=45,
        ha="right"
    )
    axis.set_yticks(range(24))
    axis.set_title("Hourly data coverage")
    axis.set_xlabel("Local date")
    axis.set_ylabel("Local hour")
    axis.legend(
        handles=[
            Patch(color="tab:blue", label="Observed"),
            Patch(color="tab:red", label="Missing"),
            Patch(color="lightgray", label="Non-existent DST hour")
        ],
        loc="upper right"
    )
    save_figure(figure, output_dir / "hourly_data_coverage.png")


def save_capacity_factor_heatmap(
    data: pd.DataFrame,
    capacity_factor: pd.Series,
    output_dir: Path
) -> None:
    """Show the average capacity factor by local month and hour."""

    working = pd.DataFrame(
        {
            "month": data["time_local"].dt.strftime("%Y-%m"),
            "hour": data["time_local"].dt.hour,
            "capacity_factor": capacity_factor
        }
    )
    monthly_hourly = working.pivot_table(
        index="month",
        columns="hour",
        values="capacity_factor",
        aggfunc="mean"
    ).reindex(columns=range(24))

    figure_height = max(4.5, len(monthly_hourly) * 0.28)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    image = axis.imshow(
        monthly_hourly,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    axis.set_xticks(range(24))
    axis.set_xticklabels(range(24))
    axis.set_yticks(range(len(monthly_hourly)))
    axis.set_yticklabels(monthly_hourly.index)
    axis.set_title("Mean photovoltaic capacity factor by month and local hour")
    axis.set_xlabel("Local hour")
    axis.set_ylabel("Local month")
    figure.colorbar(image, ax=axis, label="Capacity factor")
    save_figure(figure, output_dir / "capacity_factor_month_hour.png")


def save_capacity_factor_vs_national_solar_proxy(
    capacity_factor: pd.Series,
    national_solar_proxy: pd.Series,
    output_dir: Path
) -> None:
    """Show the daylight relationship between capacity factor and national solar proxy."""

    daylight = national_solar_proxy.gt(0) & capacity_factor.notna()

    if not daylight.any():
        return

    figure, axis = plt.subplots(figsize=(8, 5))
    plot = axis.hexbin(
        national_solar_proxy.loc[daylight],
        capacity_factor.loc[daylight],
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="viridis"
    )
    axis.set_title("Capacity factor against national solar proxy")
    axis.set_xlabel("National solar proxy")
    axis.set_ylabel("Capacity factor")
    figure.colorbar(plot, ax=axis, label="Logarithmic row count")
    save_figure(figure, output_dir / "capacity_factor_vs_national_solar_proxy.png")


def save_weather_correlation_plot(
    weather_correlations: pd.DataFrame,
    output_dir: Path
) -> None:
    """Show the median and regional range for each weather variable."""

    if weather_correlations.empty:
        return

    ordered = weather_correlations.sort_values("median_correlation")
    median = ordered["median_correlation"]
    lower = median - ordered["minimum_correlation"]
    upper = ordered["maximum_correlation"] - median

    figure_height = max(4.5, len(ordered) * 0.45)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.errorbar(
        median,
        ordered["weather_variable"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title("Weather correlation with actual generation")
    axis.set_xlabel("Pearson correlation")
    axis.set_ylabel("Weather variable")
    save_figure(figure, output_dir / "weather_variable_correlations.png")


def save_monthly_capacity_factor_boxplot(
    data: pd.DataFrame,
    capacity_factor: pd.Series,
    national_solar_proxy: pd.Series,
    output_dir: Path
) -> None:
    """Compare daylight capacity factor distributions across months."""

    daylight = national_solar_proxy.gt(0) & capacity_factor.notna()
    months = []
    values = []

    for month in range(1, 13):
        selected = daylight & data["time_local"].dt.month.eq(month)
        month_capacity_factors = capacity_factor.loc[selected].dropna()

        if not month_capacity_factors.empty:
            months.append(month)
            values.append(month_capacity_factors)

    if not values:
        return

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(values, showfliers=False)
    axis.set_xticks(range(1, len(months) + 1), months)
    axis.set_title("Daylight capacity factor by month")
    axis.set_xlabel("Month")
    axis.set_ylabel("Capacity factor")
    save_figure(figure, output_dir / "capacity_factor_by_month.png")


def main() -> int:
    """Run the EDA and save the available tables and plots."""

    args = parse_args()
    data = read_dataset(args.input.resolve())
    groups = classify_columns(data)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calendar fields are used only for EDA grouping and are not model features.
    data = data.assign(
        local_hour=data["time_local"].dt.hour,
        calendar_month=data["time_local"].dt.month,
        calendar_year=data["time_local"].dt.year
    )

    column_summary = build_column_summary(
        data.drop(columns=["local_hour", "calendar_month", "calendar_year"]),
        groups
    )
    daily_coverage = build_daily_coverage(data)
    tables = {
        "column_summary": column_summary,
        "daily_coverage": daily_coverage
    }
    save_coverage_heatmap(data, output_dir)

    correlations = None
    weather_correlations = None
    capacity_by_month = None
    capacity_factor = build_capacity_factor(data, groups)
    solar_shift_correlations = None
    solar_columns = find_solar_columns(groups["weather"])
    national_solar_proxy = None

    if groups["actual_generation"]:
        generation_groups = {
            "hour": "local_hour",
            "month": "calendar_month",
            "year": "calendar_year"
        }

        for name, group_column in generation_groups.items():
            tables[f"generation_by_{name}"] = summarize_generation(
                data,
                group_column
            )

        feature_columns = [
            *groups["weather"],
            *groups["national_capacity"],
            *groups["regional_capacity"]
        ]

        if feature_columns:
            correlations = build_feature_correlations(data, groups)
            tables["feature_target_correlations"] = correlations

            if groups["weather"]:
                weather_correlations = build_weather_variable_correlations(
                    correlations
                )
                tables["weather_variable_correlations"] = weather_correlations
                save_weather_correlation_plot(
                    weather_correlations,
                    output_dir
                )

        if solar_columns:
            national_solar_proxy = (
                build_national_solar_proxy(data, solar_columns, groups)
            )
            solar_shift_correlations = build_solar_shift_correlations(
                data,
                national_solar_proxy
            )
            tables["generation_solar_shift_correlations"] = (
                solar_shift_correlations
            )

    if capacity_factor is not None:
        save_capacity_factor_heatmap(data, capacity_factor, output_dir)

        if national_solar_proxy is not None:
            save_capacity_factor_vs_national_solar_proxy(
                capacity_factor,
                national_solar_proxy,
                output_dir
            )
            save_monthly_capacity_factor_boxplot(
                data,
                capacity_factor,
                national_solar_proxy,
                output_dir
            )

    if groups["national_capacity"] or groups["regional_capacity"]:
        capacity_by_month = build_capacity_by_month(data, groups)
        tables["capacity_by_month"] = capacity_by_month

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    print("Final dataset EDA.")
    print(f"Input: {args.input.resolve()}.")
    print(f"Rows analysed: {len(data):,}.")
    print(f"Columns analysed: {len(data.columns) - 3:,}.")
    print(f"Results: {output_dir}.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
