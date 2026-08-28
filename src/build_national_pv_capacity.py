#!/usr/bin/env python3
"""Create a CSV with Italy's installed solar capacity for each month.

Each row contains the capacity at the start of the month, its change during
the month, and the capacity at the end of the month.

The calculation starts from the 30.3 GW capacity shown at the beginning of
2024 in Terna's April 2025 report. Changes for 2024 come from the same report.
Changes for 2025 through June 2026 come from the June 2026 report. These
reports publish all values rounded to whole MW.

Terna's monthly reports are available here:
https://www.terna.it/it/sistema-elettrico/pubblicazioni/rapporto-mensile

Examples:

    python src/build_national_pv_capacity.py

    python src/build_national_pv_capacity.py \
        --output-csv path/to/national_pv_capacity_monthly.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from decimal import Decimal
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "share" / "terna" / "national_pv_capacity_monthly.csv"

# National photovoltaic capacity at the beginning of January 2024.
CAPACITY_START_2024_MW = Decimal("30300")

# Net changes include new activations, uprates, decommissioning, and
# downrates.
MONTHLY_CHANGE_MW = {
    "2024-01": Decimal("656"),
    "2024-02": Decimal("564"),
    "2024-03": Decimal("501"),
    "2024-04": Decimal("446"),
    "2024-05": Decimal("601"),
    "2024-06": Decimal("573"),
    "2024-07": Decimal("512"),
    "2024-08": Decimal("497"),
    "2024-09": Decimal("512"),
    "2024-10": Decimal("619"),
    "2024-11": Decimal("626"),
    "2024-12": Decimal("686"),
    "2025-01": Decimal("419"),
    "2025-02": Decimal("392"),
    "2025-03": Decimal("621"),
    "2025-04": Decimal("458"),
    "2025-05": Decimal("495"),
    "2025-06": Decimal("424"),
    "2025-07": Decimal("546"),
    "2025-08": Decimal("326"),
    "2025-09": Decimal("398"),
    "2025-10": Decimal("736"),
    "2025-11": Decimal("985"),
    "2025-12": Decimal("639"),
    "2026-01": Decimal("333"),
    "2026-02": Decimal("544"),
    "2026-03": Decimal("562"),
    "2026-04": Decimal("722"),
    "2026-05": Decimal("448"),
    "2026-06": Decimal("484")
}

OUTPUT_COLUMNS = (
    "month",
    "capacity_start_mw",
    "monthly_change_mw",
    "capacity_end_mw"
)


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
    """Convert a number to plain text."""

    return format(value, "f")


def build_capacity_rows() -> list[dict[str, str]]:
    """Calculate Italy's photovoltaic (PV) capacity at the start and end of each month."""

    number_of_months = 30  # January 2024 through June 2026.
    expected_months = create_month_list(2024, 1, number_of_months)
    configured_months = list(MONTHLY_CHANGE_MW.keys())

    if configured_months != expected_months:
        raise RuntimeError(
            "Monthly capacity changes must cover every month from 2024-01 "
            "through 2026-06 in chronological order."
        )

    capacity = CAPACITY_START_2024_MW

    rows: list[dict[str, str]] = []

    for month, change in MONTHLY_CHANGE_MW.items():
        capacity_start = capacity
        capacity_end = capacity_start + change

        rows.append(
            {
                "month": month,
                "capacity_start_mw": decimal_text(capacity_start),
                "monthly_change_mw": decimal_text(change),
                "capacity_end_mw": decimal_text(capacity_end)
            }
        )

        capacity = capacity_end

    return rows


def write_csv_atomic(rows: list[dict[str, str]], output: Path) -> None:
    """Save national capacity rows to a CSV file."""

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


def parse_args() -> argparse.Namespace:
    """Read --output-csv, the CSV file that the script will create."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-csv",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Monthly national capacity CSV to create. Default: "
            "share/terna/national_pv_capacity_monthly.csv."
        )
    )

    return parser.parse_args()


def main() -> int:
    """Calculate, check, and save Italy's monthly photovoltaic (PV) capacity."""

    args = parse_args()
    rows = build_capacity_rows()
    output = args.output.resolve()

    write_csv_atomic(rows, output)

    print("Terna national photovoltaic capacity reconstruction.")
    print(f"Saved: {output}.")
    print(f"Rows: {len(rows)}.")
    print(f"Months: {rows[0]['month']} -> {rows[-1]['month']}.")
    print(
        "Start-of-2024 capacity: "
        f"{decimal_text(CAPACITY_START_2024_MW)} MW."
    )
    print(
        "WARNING: monthly-report values are published at whole-MW precision.",
        file=sys.stderr
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
