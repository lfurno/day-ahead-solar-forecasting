#!/usr/bin/env python3
"""Create a monthly photovoltaic (PV) capacity CSV for Italy's regions.

Terna provides regional PV capacity for December 2024 and monthly regional
capacity changes from January 2025 onward.

For January through November 2024, the script estimates the regional values
using the national capacity reported for each month. It preserves the
differences between regions observed in December and ensures that the regional
values add up to the national total.

The capacity_method column indicates whether each value is estimated
or based on observed regional changes.

Required input files:

- installed_renewables_2024.csv
- installed_renewables_2025.csv
- installed_renewables_2026.csv
  created with download_terna.py;

- national_pv_capacity_monthly.csv
  created with build_national_pv_capacity.py.

Example with all required steps:

    python src/download_terna.py installed-renewables --start 2024 --end 2026
    python src/build_national_pv_capacity.py
    python src/build_regional_pv_capacity.py

    python src/build_regional_pv_capacity.py \
        --input-dir path/to/input-dir \
        --national-capacity-csv path/to/national_pv_capacity_monthly.csv \
        --output-csv path/to/regional_pv_capacity_monthly.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERNA_DATA_DIR = PROJECT_ROOT / "share" / "terna"
DEFAULT_NATIONAL_CAPACITY = (
    DEFAULT_TERNA_DATA_DIR / "national_pv_capacity_monthly.csv"
)
DEFAULT_OUTPUT = DEFAULT_TERNA_DATA_DIR / "regional_pv_capacity_monthly.csv"

INSTALLED_RENEWABLE_COLUMNS = (
    "year",
    "month",
    "market_zone",
    "region",
    "province",
    "plant_power_class",
    "change_type",
    "source",
    "voltage_level",
    "nominal_active_power_mw",
    "plant_count"
)
NATIONAL_CAPACITY_COLUMNS = (
    "month",
    "capacity_start_mw",
    "monthly_change_mw",
    "capacity_end_mw"
)
OUTPUT_COLUMNS = (
    "month",
    "region",
    "capacity_start_mw",
    "monthly_change_mw",
    "capacity_end_mw",
    "capacity_method"
)

SOLAR_SOURCE = "Solare"
ESTIMATED_METHOD = "estimated_from_end_2024_regional_share"
OBSERVED_METHOD = "observed_regional_change"
FIRST_MONTH = "2024-01"
LAST_MONTH = "2026-06"

REGION_TO_DATASET_NAME = {
    "Abruzzo": "abruzzo",
    "Basilicata": "basilicata",
    "Calabria": "calabria",
    "Campania": "campania",
    "Emilia Romagna": "emilia_romagna",
    "Friuli Venezia Giulia": "friuli_venezia_giulia",
    "Lazio": "lazio",
    "Liguria": "liguria",
    "Lombardia": "lombardia",
    "Marche": "marche",
    "Molise": "molise",
    "Piemonte": "piemonte",
    "Puglia": "puglia",
    "Sardegna": "sardegna",
    "Sicilia": "sicilia",
    "Toscana": "toscana",
    "Trentino Alto Adige": "trentino_alto_adige",
    "Umbria": "umbria",
    "Valle d'aosta": "valle_d_aosta",
    "Veneto": "veneto"
}
NUM_REGIONS =  len(REGION_TO_DATASET_NAME)
NUM_MONTHS = 30 # They cover January 2024 through June 2026.

def parse_args() -> argparse.Namespace:
    """Read the folders and files used to build the regional CSV."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_TERNA_DATA_DIR,
        help=(
            "Folder containing installed_renewables_2024.csv, "
            "installed_renewables_2025.csv, and "
            "installed_renewables_2026.csv. Default: share/terna."
        )
    )

    parser.add_argument(
        "--national-capacity-csv",
        dest="national_capacity",
        type=Path,
        default=DEFAULT_NATIONAL_CAPACITY,
        help=(
            "National monthly capacity CSV created by "
            "build_national_pv_capacity.py. Default: "
            "share/terna/national_pv_capacity_monthly.csv."
        )
    )

    parser.add_argument(
        "--output-csv",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Regional monthly capacity CSV to create. Default: "
            "share/terna/regional_pv_capacity_monthly.csv."
        )
    )

    return parser.parse_args()


def create_month_list(
    start_year: int,
    start_month: int,
    number_of_months: int
) -> list[str]:
    """Create a list of months in YYYY-MM format."""

    months: list[str] = []
    year = start_year
    month = start_month

    for _ in range(number_of_months):
        months.append(f"{year:04d}-{month:02d}")

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return months


def decimal_text(value: Decimal) -> str:
    """Convert a number to text with six decimal places."""

    return format(value, ".6f")


def read_national_capacity(path: Path) -> dict[str, dict[str, Decimal]]:
    """Read and check Italy's monthly capacity CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"National monthly capacity file not found: {path}."
        )

    result: dict[str, dict[str, Decimal]] = {}

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if tuple(reader.fieldnames or ()) != NATIONAL_CAPACITY_COLUMNS:
            raise RuntimeError(
                f"Unexpected national capacity columns in {path.name}."
            )

        for row in reader:
            month = row["month"].strip()

            if month in result:
                raise RuntimeError(f"Duplicate national capacity month: {month}.")

            try:
                start = Decimal(row["capacity_start_mw"])
                change = Decimal(row["monthly_change_mw"])
                end = Decimal(row["capacity_end_mw"])
            except (InvalidOperation, TypeError) as exc:
                raise RuntimeError(
                    f"Invalid national capacity values for {month}."
                ) from exc

            if start + change != end:
                raise RuntimeError(
                    f"National capacity arithmetic is inconsistent for {month}."
                )

            result[month] = {"start": start, "change": change, "end": end}

    expected = create_month_list(2024, 1, NUM_MONTHS)

    if list(result) != expected:
        raise RuntimeError(
            "National capacity must cover every month from 2024-01 through "
            "2026-06 in chronological order."
        )

    return result


def read_solar_rows(
    path: Path,
    expected_year: int
) -> list[tuple[int, str, Decimal]]:
    """Read and convert the solar rows from one annual Terna CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Installed-renewables file not found: {path}.")

    solar_rows: list[tuple[int, str, Decimal]] = []

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if tuple(reader.fieldnames or ()) != INSTALLED_RENEWABLE_COLUMNS:
            raise RuntimeError(f"Unexpected columns in {path.name}.")

        for row_number, row in enumerate(reader, start=2):
            try:
                year = int(row["year"])
                month = int(row["month"])
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Invalid date in {path.name} at row {row_number}."
                ) from exc

            if year != expected_year:
                raise RuntimeError(
                    f"{path.name} contains year {year}, expected {expected_year}."
                )

            if not 1 <= month <= 12:
                raise RuntimeError(
                    f"{path.name} contains invalid month {month}."
                )

            if row["source"].strip() != SOLAR_SOURCE:
                continue

            region = row["region"].strip()

            if region not in REGION_TO_DATASET_NAME:
                raise RuntimeError(
                    f"Unknown Terna region in {path.name}: {region!r}."
                )

            try:
                power = Decimal(row["nominal_active_power_mw"])
            except (InvalidOperation, TypeError) as exc:
                raise RuntimeError(
                    f"Invalid power in {path.name} at row {row_number}."
                ) from exc

            solar_rows.append((month, region, power))

    if not solar_rows:
        raise RuntimeError(f"No solar rows found in {path.name}.")

    return solar_rows


def build_anchor(
    rows_2024: list[tuple[int, str, Decimal]]
) -> dict[str, Decimal]:
    """Calculate each region's photovoltaic (PV) capacity at the end of 2024."""

    months = {month for month, _, _ in rows_2024}

    if months != {12}:
        raise RuntimeError(
            "The 2024 installed-renewables file must be a December snapshot."
        )

    anchor = {region: Decimal("0") for region in REGION_TO_DATASET_NAME}

    for _, region, power in rows_2024:
        anchor[region] += power

    missing = [
        region
        for region, capacity in anchor.items()
        if capacity <= 0
    ]

    if missing:
        raise RuntimeError(
            f"Non-positive or missing 2024 regional anchors: {missing}."
        )

    return anchor


def build_monthly_changes(
    annual_rows: dict[int, list[tuple[int, str, Decimal]]]
) -> dict[str, dict[str, Decimal]]:
    """Add together the photovoltaic (PV) capacity changes for each region and month."""

    # These 18 months cover January 2025 through June 2026.
    required_months = create_month_list(2025, 1, 18)

    changes = {
        month: {region: Decimal("0") for region in REGION_TO_DATASET_NAME}
        for month in required_months
    }

    observed_months: set[str] = set()

    for year in (2025, 2026):
        for month_number, region, power in annual_rows[year]:
            month = f"{year:04d}-{month_number:02d}"

            if month not in changes:
                continue

            observed_months.add(month)
            changes[month][region] += power

    if observed_months != set(required_months):
        raise RuntimeError(
            "Regional changes must cover every month from 2025-01 through "
            "2026-06."
        )

    return changes


def output_row(
    month: str,
    region_name: str,
    start: Decimal,
    change: Decimal,
    end: Decimal,
    method: str
) -> dict[str, str]:
    """Create one CSV row for a region and month."""

    return {
        "month": month,
        "region": REGION_TO_DATASET_NAME[region_name],
        "capacity_start_mw": decimal_text(start),
        "monthly_change_mw": decimal_text(change),
        "capacity_end_mw": decimal_text(end),
        "capacity_method": method
    }


def build_capacity_rows(
    national: dict[str, dict[str, Decimal]],
    anchor: dict[str, Decimal],
    changes: dict[str, dict[str, Decimal]]
) -> list[dict[str, str]]:
    """Build monthly regional capacities from estimates and Terna changes."""

    anchor_total = sum(anchor.values(), Decimal("0"))
    national_end_2024 = national["2024-12"]["end"]

    # The 2024 regional rows are labelled as changes, but their sum is close to
    # the national installed capacity. Round both totals to one decimal place
    # before comparing them, as in Terna's report.
    regional_total_gw = round(anchor_total / Decimal("1000"), 1)
    national_total_gw = round(national_end_2024 / Decimal("1000"), 1)

    if regional_total_gw != national_total_gw:
        raise RuntimeError(
            "The 2024 regional file does not represent total installed "
            f"capacity: regional={anchor_total} MW, "
            f"national={national_end_2024} MW."
        )

    # Calculate each region's share from the 2024 regional data.
    shares = {
        region: anchor_capacity / anchor_total
        for region, anchor_capacity in anchor.items()
    }

    # The regional and national totals match after rounding, but their exact MW
    # values may still differ. If they are exactly equal, keep the regional
    # values unchanged.
    if anchor_total == national_end_2024:
        capacity = anchor.copy()
    else:
        # Scale the regional values so their total is exactly equal to the
        # national capacity reported for the end of December 2024.
        capacity = {
            region: national_end_2024 * share
            for region, share in shares.items()
        }

    rows: list[dict[str, str]] = []
    regions = sorted(
        REGION_TO_DATASET_NAME,
        key=lambda name: REGION_TO_DATASET_NAME[name]
    )

    for month in create_month_list(2024, 1, 12):
        for region in regions:
            share = shares[region]
            start = national[month]["start"] * share
            change = national[month]["change"] * share
            end = national[month]["end"] * share

            rows.append(
                output_row(
                    month,
                    region,
                    start,
                    change,
                    end,
                    ESTIMATED_METHOD
                )
            )

    for month in create_month_list(2025, 1, 18):
        for region in regions:
            start = capacity[region]
            change = changes[month][region]
            end = start + change

            if end <= 0:
                raise RuntimeError(
                    f"Non-positive capacity for {region} at {month}."
                )

            rows.append(
                output_row(
                    month,
                    region,
                    start,
                    change,
                    end,
                    OBSERVED_METHOD
                )
            )

            capacity[region] = end


    if len(rows) != NUM_MONTHS * NUM_REGIONS:
        raise RuntimeError(f"Expected {NUM_MONTHS * NUM_REGIONS} output rows, "
                           f"found {len(rows)}.")

    return rows


def write_csv_atomic(rows: list[dict[str, str]], output: Path) -> None:
    """Save regional capacity rows to a CSV file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp" + output.suffix)

    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)

            writer.writeheader()
            writer.writerows(rows)

        temporary.replace(output)
    finally:

        if temporary.exists():
            temporary.unlink()


def main() -> int:
    """Calculate, check, and save monthly photovoltaic (PV) capacity for each region."""

    args = parse_args()
    input_dir = args.input_dir.resolve()
    national_capacity = read_national_capacity(args.national_capacity.resolve())

    annual_rows = {
        year: read_solar_rows(
            input_dir / f"installed_renewables_{year}.csv",
            year
        )
        for year in (2024, 2025, 2026)
    }

    anchor = build_anchor(annual_rows[2024])
    changes = build_monthly_changes(annual_rows)
    rows = build_capacity_rows(national_capacity, anchor, changes)

    output = args.output.resolve()

    write_csv_atomic(rows, output)

    print("Terna regional photovoltaic capacity reconstruction.")
    print(f"Saved: {output}.")
    print(f"Rows: {len(rows)} ({NUM_MONTHS} months * {NUM_REGIONS} regions).")
    print(f"Months: {FIRST_MONTH} -> {LAST_MONTH}.")
    print("2024 method: estimated from end-of-2024 regional shares.")
    print("2025-01 onward: observed regional net changes from Terna CSV files.")
    print("WARNING: capacity_method describes how each row was calculated "
          "and should not be used as a model feature.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)

        raise SystemExit(1) from exc
