#!/usr/bin/env python3
"""Build the final hourly solar dataset from selected CSV files.

Four datasets are available:

- actual-generation: Terna photovoltaic (PV) actual generation;
- weather: regional Open-Meteo forecasts;
- national-capacity: national PV capacity at the start of each month;
- regional-capacity: regional PV capacity at the start of each month.

At least one hourly dataset, actual-generation or weather, must be included.
The monthly capacity files cannot define the hourly timestamps by themselves.

Terna values that are already hourly are kept as they are. Four complete
15-minute values are averaged to obtain one hourly value. Missing or incomplete
hours are excluded and are not interpolated.

Some Open-Meteo variables describe the previous hour, while others describe
the stated instant. Previous-hour values are aligned using the next physical
row. Instant values remain on their original row.

When Italian clocks move backward, the local hour from 02:00 to 03:00 occurs
twice. If both occurrences are available, the later duplicate is removed, so
the final dataset contains at most one row for each local date and time.

The input CSV files are expected to come from the companion download and build
scripts, which already validate their values. This script checks only what is
needed to align and merge them correctly.

Examples:

    python src/build_final_dataset.py

    python src/build_final_dataset.py --include actual-generation weather

    python src/build_final_dataset.py --start 2025-01-01 --end 2026-06-30

    python src/build_final_dataset.py \
        --actual-generation-csv path/to/actual_generation.csv \
        --weather-csv path/to/regional_weather.csv \
        --national-capacity-csv path/to/national_pv_capacity_monthly.csv \
        --regional-capacity-csv path/to/regional_pv_capacity_monthly.csv \
        --output-csv path/to/final_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROME_TIMEZONE = ZoneInfo("Europe/Rome")

DEFAULT_ACTUAL_GENERATION = PROJECT_ROOT / "share" / "terna" / "actual_generation.csv"
DEFAULT_WEATHER = (
    PROJECT_ROOT / "share" / "open-meteo" / "regional_weather.csv"
)
DEFAULT_NATIONAL_CAPACITY = (
    PROJECT_ROOT / "share" / "terna" / "national_pv_capacity_monthly.csv"
)
DEFAULT_REGIONAL_CAPACITY = (
    PROJECT_ROOT / "share" / "terna" / "regional_pv_capacity_monthly.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "share" / "final_dataset.csv"

DATASETS = (
    "actual-generation",
    "weather",
    "national-capacity",
    "regional-capacity"
)
TIME_COLUMNS = (
    "time_utc",
    "time_local",
    "utc_offset"
)
PREVIOUS_HOUR_VARIABLES = (
    "shortwave_radiation_previous_day1",
    "direct_radiation_previous_day1",
    "diffuse_radiation_previous_day1",
    "direct_normal_irradiance_previous_day1",
    "global_tilted_irradiance_previous_day1",
    "precipitation_previous_day1"
)
ACTUAL_GENERATION_COLUMNS = (
    "time_utc",
    "time_local",
    "utc_offset",
    "timezone",
    "primary_source",
    "actual_generation_gwh"
)
NATIONAL_CAPACITY_COLUMNS = (
    "month",
    "capacity_start_mw",
    "monthly_change_mw",
    "capacity_end_mw"
)
REGIONAL_CAPACITY_COLUMNS = (
    "month",
    "region",
    "capacity_start_mw",
    "monthly_change_mw",
    "capacity_end_mw",
    "capacity_method"
)

HourlyRows = dict[datetime, dict[str, str]]
RegionalCapacity = dict[str, dict[str, str]]
GenerationStats = tuple[datetime, datetime, int]


def parse_args() -> argparse.Namespace:
    """Read the selected datasets, dates, input files, and output file."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help=(
            "Select which data sources are included in the final "
            "dataset. Choices: actual-generation, weather, national-capacity, "
            "regional-capacity. Default: include all four."
        )
    )
    parser.add_argument(
        "--start",
        help="First local date to keep in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end",
        help="Last local date to keep in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--actual-generation-csv",
        dest="actual_generation",
        type=Path,
        default=DEFAULT_ACTUAL_GENERATION,
        help=(
            "Terna actual-generation CSV. Default: "
            "share/terna/actual_generation.csv."
        )
    )
    parser.add_argument(
        "--weather-csv",
        dest="weather",
        type=Path,
        default=DEFAULT_WEATHER,
        help=(
            "Open-Meteo weather CSV. Default: "
            "share/open-meteo/regional_weather.csv."
        )
    )
    parser.add_argument(
        "--national-capacity-csv",
        dest="national_capacity",
        type=Path,
        default=DEFAULT_NATIONAL_CAPACITY,
        help=(
            "National monthly capacity CSV. Default: "
            "share/terna/national_pv_capacity_monthly.csv."
        )
    )
    parser.add_argument(
        "--regional-capacity-csv",
        dest="regional_capacity",
        type=Path,
        default=DEFAULT_REGIONAL_CAPACITY,
        help=(
            "Regional monthly capacity CSV. Default: "
            "share/terna/regional_pv_capacity_monthly.csv."
        )
    )
    parser.add_argument(
        "--output-csv",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Final CSV to create. Default: share/final_dataset.csv."
    )

    return parser.parse_args()


def parse_date(value: str | None, option: str) -> date | None:
    """Convert an optional YYYY-MM-DD argument to a date."""

    if value is None:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{option} must use YYYY-MM-DD: {value!r}."
        ) from exc


def parse_utc(value: str, label: str) -> datetime:
    """Convert a timestamp with an offset to a UTC date and time."""

    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid UTC timestamp in {label}: {value!r}."
        ) from exc

    if parsed.tzinfo is None:
        raise ValueError(
            f"UTC timestamp has no offset in {label}: {value!r}."
        )

    return parsed.astimezone(timezone.utc)


def utc_text(value: datetime) -> str:
    """Convert a date and time to the UTC text used in the output."""

    return value.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace(
        "+00:00",
        "Z"
    )


def time_fields(value: datetime) -> dict[str, str]:
    """Create the UTC time, Italian local time, and UTC offset fields."""

    local = value.astimezone(ROME_TIMEZONE)
    offset = local.strftime("%z")

    return {
        "time_utc": utc_text(value),
        "time_local": local.replace(tzinfo=None).isoformat(timespec="seconds"),
        "utc_offset": f"{offset[:3]}:{offset[3:]}"
    }


def local_month(instant: datetime) -> str:
    """Return the local month used to join the capacity files."""

    return instant.astimezone(ROME_TIMEZONE).strftime("%Y-%m")


def read_csv(
    path: Path,
    expected_columns: tuple[str, ...]
) -> list[dict[str, str]]:
    """Read a non-empty CSV file with the expected columns."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if tuple(reader.fieldnames or ()) != expected_columns:
            raise RuntimeError(f"Unexpected columns in {path.name}.")

        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Input file is empty: {path}.")

    return rows


def read_actual_generation(path: Path) -> tuple[HourlyRows, GenerationStats]:
    """Convert complete Terna observations to hourly actual_generation values."""

    actual_generation_by_time: dict[datetime, float] = {}

    for row_number, row in enumerate(
        read_csv(path, ACTUAL_GENERATION_COLUMNS),
        start=2
    ):
        instant = parse_utc(
            row["time_utc"],
            f"{path.name} row {row_number}"
        )

        if instant in actual_generation_by_time:
            raise RuntimeError(
                f"Duplicate Terna timestamp: {utc_text(instant)}."
            )

        try:
            actual_generation_by_time[instant] = float(
                row["actual_generation_gwh"]
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid Terna actual generation at {utc_text(instant)}."
            ) from exc

    # Group all Terna observations by the hour they belong to.
    actual_generation_by_hour: dict[
        datetime,
        dict[int, float]
    ] = defaultdict(dict)

    for instant, actual_generation in actual_generation_by_time.items():
        hour = instant.replace(minute=0, second=0, microsecond=0)
        actual_generation_by_hour[hour][instant.minute] = actual_generation

    hours = sorted(actual_generation_by_hour)
    first_hour = hours[0]
    last_hour = hours[-1]

    # The first hour containing :15, :30, or :45 marks the transition from
    # hourly to 15-minute Terna data.
    quarter_start = next(
        (
            hour
            for hour in hours
            if any(
                minute != 0
                for minute in actual_generation_by_hour[hour]
            )
        ),
        None
    )

    hourly: HourlyRows = {}
    incomplete_hours = 0

    for hour in hours:
        observations = actual_generation_by_hour[hour]

        if quarter_start is not None and hour >= quarter_start:
            expected_minutes = {0, 15, 30, 45}
        else:
            expected_minutes = {0}

        # Never calculate an hourly value from an incomplete group.
        if set(observations) != expected_minutes:
            incomplete_hours += 1
            continue

        hourly[hour] = {
            "actual_generation_gw": format(
                fmean(observations.values()),
                ".12g"
            )
        }

    stats = (
        first_hour,
        last_hour,
        incomplete_hours
    )

    return hourly, stats


def read_weather(path: Path) -> tuple[HourlyRows, list[str]]:
    """Read the weather data and align previous-hour variables."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())

        if columns[:3] != TIME_COLUMNS or len(columns) <= len(TIME_COLUMNS):
            raise RuntimeError(f"Unexpected columns in {path.name}.")

        weather_columns = list(columns[3:])
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Input file is empty: {path}.")

    rows_by_time: HourlyRows = {}

    for row_number, row in enumerate(rows, start=2):
        instant = parse_utc(
            row["time_utc"],
            f"{path.name} row {row_number}"
        )

        if instant in rows_by_time:
            raise RuntimeError(
                f"Duplicate weather timestamp: {utc_text(instant)}."
            )

        rows_by_time[instant] = row

    # Only variables defined as averages of the previous hour must be shifted.
    columns_to_shift = {
        column
        for column in weather_columns
        if any(
            column.endswith(f"_{variable}")
            for variable in PREVIOUS_HOUR_VARIABLES
        )
    }

    ordered_times = sorted(rows_by_time)
    aligned: HourlyRows = {}

    for current, following in zip(ordered_times, ordered_times[1:]):
        # Consecutive physical rows are required for a correct one-hour shift.
        if following - current != timedelta(hours=1):
            raise RuntimeError(
                "Weather timestamps are not hourly: "
                f"{utc_text(current)} -> {utc_text(following)}."
            )

        aligned[current] = {
            column: rows_by_time[
                following if column in columns_to_shift else current
            ][column]
            for column in weather_columns
        }

    return aligned, weather_columns


def read_national_capacity(path: Path) -> dict[str, str]:
    """Read national PV capacity at the start of each month."""

    capacity_by_month: dict[str, str] = {}

    for row in read_csv(path, NATIONAL_CAPACITY_COLUMNS):
        month = row["month"].strip()

        if month in capacity_by_month:
            raise RuntimeError(f"Duplicate national capacity month: {month}.")

        capacity_by_month[month] = row["capacity_start_mw"]

    return capacity_by_month


def read_regional_capacity(
    path: Path
) -> tuple[RegionalCapacity, list[str]]:
    """Read regional PV capacity at the start of each month."""

    capacity_by_month: RegionalCapacity = defaultdict(dict)
    regions: set[str] = set()

    for row in read_csv(path, REGIONAL_CAPACITY_COLUMNS):
        month = row["month"].strip()
        region = row["region"].strip()

        if region in capacity_by_month[month]:
            raise RuntimeError(
                f"Duplicate regional capacity: {month}, {region}."
            )

        capacity_by_month[month][region] = row["capacity_start_mw"]
        regions.add(region)

    # Every available month must contain the same set of regions.
    for month, capacities in capacity_by_month.items():
        if set(capacities) != regions:
            raise RuntimeError(
                f"Regional capacity is incomplete for {month}."
            )

    return dict(capacity_by_month), sorted(regions)


def select_times(
    selected: set[str],
    actual_generation: HourlyRows,
    weather: HourlyRows,
    national_capacity: dict[str, str],
    regional_capacity: RegionalCapacity,
    start: date | None,
    end: date | None
) -> tuple[list[datetime], int]:
    """Select the hours shared by the requested datasets and date limits."""

    hourly_sets = []

    if "actual-generation" in selected:
        hourly_sets.append(set(actual_generation))

    if "weather" in selected:
        hourly_sets.append(set(weather))

    if not hourly_sets:
        raise ValueError(
            "Select actual-generation or weather to define hourly timestamps."
        )

    selected_times = set.intersection(*hourly_sets)

    # Keep only months covered by every selected capacity file.
    covered_times = set()

    for instant in selected_times:
        month = local_month(instant)

        if (
            "national-capacity" in selected
            and month not in national_capacity
        ):
            continue

        if (
            "regional-capacity" in selected
            and month not in regional_capacity
        ):
            continue

        covered_times.add(instant)

    selected_times = covered_times

    # Select only the hours within the requested local date range.
    filtered_times = []

    for instant in sorted(selected_times):
        local_date = instant.astimezone(ROME_TIMEZONE).date()

        if start is not None and local_date < start:
            continue

        if end is not None and local_date > end:
            continue

        filtered_times.append(instant)

    unique_times = []
    seen_local_times: set[datetime] = set()
    repeated_hours = 0

    # Remove the second occurrence of the repeated local hour in October.
    for instant in filtered_times:
        local_time = instant.astimezone(ROME_TIMEZONE).replace(tzinfo=None)

        if local_time in seen_local_times:
            repeated_hours += 1
            continue

        seen_local_times.add(local_time)
        unique_times.append(instant)

    if not unique_times:
        raise RuntimeError("The selected datasets and dates do not overlap.")

    return unique_times, repeated_hours


def build_rows(
    times: list[datetime],
    selected: set[str],
    actual_generation: HourlyRows,
    weather: HourlyRows,
    national_capacity: dict[str, str],
    regional_capacity: RegionalCapacity,
    regions: list[str]
) -> list[dict[str, str]]:
    """Combine the requested values for every selected hour."""

    result = []

    for instant in times:
        row = time_fields(instant)
        month = local_month(instant)

        if "actual-generation" in selected:
            row.update(actual_generation[instant])

        if "weather" in selected:
            row.update(weather[instant])

        if "national-capacity" in selected:
            row["national_installed_pv_mw_start"] = national_capacity[month]

        if "regional-capacity" in selected:
            for region in regions:
                column = f"{region}_installed_pv_mw_start"
                row[column] = regional_capacity[month][region]

        result.append(row)

    return result


def output_columns(
    selected: set[str],
    weather_columns: list[str],
    regions: list[str]
) -> list[str]:
    """Create the output column order for the selected datasets."""

    columns = list(TIME_COLUMNS)

    if "actual-generation" in selected:
        columns.append("actual_generation_gw")

    if "national-capacity" in selected:
        columns.append("national_installed_pv_mw_start")

    if "regional-capacity" in selected:
        columns.extend(
            f"{region}_installed_pv_mw_start"
            for region in regions
        )

    if "weather" in selected:
        columns.extend(weather_columns)

    return columns


def write_csv_atomic(
    rows: list[dict[str, str]],
    columns: list[str],
    output: Path
) -> None:
    """Save the final dataset without leaving a partial CSV."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp" + output.suffix)

    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)

            writer.writeheader()
            writer.writerows(rows)

        temporary.replace(output)
    finally:

        if temporary.exists():
            temporary.unlink()


def main() -> int:
    """Read, align, merge, and save the selected datasets."""

    args = parse_args()
    selected = set(args.include)
    start = parse_date(args.start, "--start")
    end = parse_date(args.end, "--end")

    if start is not None and end is not None and end < start:
        raise ValueError("--end must not be earlier than --start.")

    actual_generation: HourlyRows = {}
    weather: HourlyRows = {}
    national_capacity: dict[str, str] = {}
    regional_capacity: RegionalCapacity = {}
    weather_columns: list[str] = []
    regions: list[str] = []
    actual_generation_stats: GenerationStats | None = None

    if "actual-generation" in selected:
        actual_generation, actual_generation_stats = read_actual_generation(
            args.actual_generation.resolve()
        )

    if "weather" in selected:
        weather, weather_columns = read_weather(
            args.weather.resolve()
        )

    if "national-capacity" in selected:
        national_capacity = read_national_capacity(
            args.national_capacity.resolve()
        )

    if "regional-capacity" in selected:
        regional_capacity, regions = read_regional_capacity(
            args.regional_capacity.resolve()
        )

    times, repeated_hours = select_times(
        selected,
        actual_generation,
        weather,
        national_capacity,
        regional_capacity,
        start,
        end
    )
    rows = build_rows(
        times,
        selected,
        actual_generation,
        weather,
        national_capacity,
        regional_capacity,
        regions
    )
    columns = output_columns(selected, weather_columns, regions)
    output = args.output.resolve()

    write_csv_atomic(rows, columns, output)

    print("Final hourly dataset.")
    print(
        "Included: "
        + ", ".join(item for item in DATASETS if item in selected)
        + "."
    )

    if actual_generation_stats is not None:
        first_hour, last_hour, incomplete_hours = actual_generation_stats
        expected_hours = int(
            (last_hour - first_hour).total_seconds() // 3600
        ) + 1
        missing_hours = expected_hours - len(actual_generation) - incomplete_hours

        print(f"Complete Terna hours: {len(actual_generation):,}.")
        print(f"Missing Terna hours: {missing_hours:,}; not interpolated.")
        print(f"Incomplete Terna hours: {incomplete_hours:,}; excluded.")

    if "weather" in selected:
        print(f"Aligned weather hours: {len(weather):,}.")

    print(f"Repeated local hours removed: {repeated_hours:,}.")
    print(f"Saved: {output}.")
    print(f"Rows: {len(rows):,} | columns: {len(columns)}.")
    print(f"Local range: {rows[0]['time_local']} -> "
          f"{rows[-1]['time_local']}.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
