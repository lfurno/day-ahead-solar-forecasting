#!/usr/bin/env python3
"""Download 24-hour-ahead solar weather forecasts from Open-Meteo.

The script downloads weather for the capital of each Italian region. It saves
time in UTC so that every hour is unique when Italian clocks change. Italian
local time and its UTC offset are also saved to make the file easier to check.

Additional weather variables can be added to the variable lists below. The
available variables are listed at:
https://open-meteo.com/en/docs/previous-runs-api

Examples:

    python src/download_open_meteo.py --start 2024-01-20 --end 2026-07-01

    python src/download_open_meteo.py \
        --start 2024-01-20 \
        --end 2026-07-01 \
        --output-csv path/to/regional_weather.csv

Radiation and precipitation values describe the previous hour. Download one
extra day at the end so that the final dataset can align the last hour
correctly.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LOCAL_TIMEZONE = ZoneInfo("Europe/Rome")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "share" / "open-meteo" / "regional_weather.csv"

PRECEDING_HOUR_VARIABLES = (
    "shortwave_radiation_previous_day1",
    "direct_radiation_previous_day1",
    "diffuse_radiation_previous_day1",
    "direct_normal_irradiance_previous_day1",
    "global_tilted_irradiance_previous_day1",
    "precipitation_previous_day1"
)
INSTANT_VARIABLES = (
    "temperature_2m_previous_day1",
    "cloud_cover_previous_day1",
    "wind_speed_10m_previous_day1",
    "wind_direction_10m_previous_day1",
    "relative_humidity_2m_previous_day1"
)
VARIABLES = PRECEDING_HOUR_VARIABLES + INSTANT_VARIABLES

NON_NEGATIVE_VARIABLES = {
    *PRECEDING_HOUR_VARIABLES,
    "wind_speed_10m_previous_day1"
}
PERCENTAGE_VARIABLES = {
    "cloud_cover_previous_day1",
    "relative_humidity_2m_previous_day1"
}
DIRECTION_VARIABLES = {"wind_direction_10m_previous_day1"}

REGIONS = (
    ("piemonte", "Torino", 45.0703, 7.6869),
    ("valle_d_aosta", "Aosta", 45.7370, 7.3201),
    ("lombardia", "Milano", 45.4642, 9.1900),
    ("trentino_alto_adige", "Trento", 46.0748, 11.1217),
    ("veneto", "Venezia", 45.4408, 12.3155),
    ("friuli_venezia_giulia", "Trieste", 45.6495, 13.7768),
    ("liguria", "Genova", 44.4056, 8.9463),
    ("emilia_romagna", "Bologna", 44.4949, 11.3426),
    ("toscana", "Firenze", 43.7696, 11.2558),
    ("umbria", "Perugia", 43.1107, 12.3908),
    ("marche", "Ancona", 43.6158, 13.5189),
    ("lazio", "Roma", 41.9028, 12.4964),
    ("abruzzo", "L_Aquila", 42.3498, 13.3995),
    ("molise", "Campobasso", 41.5603, 14.6627),
    ("campania", "Napoli", 40.8518, 14.2681),
    ("puglia", "Bari", 41.1171, 16.8719),
    ("basilicata", "Potenza", 40.6404, 15.8056),
    ("calabria", "Catanzaro", 38.9098, 16.5877),
    ("sicilia", "Palermo", 38.1157, 13.3615),
    ("sardegna", "Cagliari", 39.2238, 9.1217)
)

TIME_COLUMNS = ("time_utc", "time_local", "utc_offset")
WEATHER_COLUMNS = tuple(
    f"{region}_{variable}"
    for region, _, _, _ in REGIONS
    for variable in VARIABLES
)
OUTPUT_COLUMNS = (*TIME_COLUMNS, *WEATHER_COLUMNS)


def parse_args() -> argparse.Namespace:
    """Read the dates, output file, panel settings, and download settings."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--start",
        required=True,
        help=(
            "First Italian day to download. Use YYYY-MM-DD, for example "
            "2025-01-01. Data for this day are included."
        )
    )
    parser.add_argument(
        "--end",
        required=True,
        help=(
            "Last Italian day to download. Use YYYY-MM-DD, for example "
            "2025-12-31. Data for this day are included."
        )
    )
    parser.add_argument(
        "--output-csv",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "CSV file to create. Default: "
            "share/open-meteo/regional_weather.csv."
        )
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=30.0,
        help=(
            "Solar-panel tilt in degrees: 0 is flat and 90 is vertical. "
            f"Default: 30.0."
        )
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help=(
            "Direction faced by the panels: 0 is south, -90 is east, and "
            f"90 is west. Default: 0.0."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Open-Meteo weather model to use. Leave this option out to let "
            "Open-Meteo choose Best Match."
        )
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Seconds to wait between monthly downloads. Default: 0.5."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help=(
            "Stop a web request if it takes more than this many seconds. "
            "Default: 180."
        )
    )

    return parser.parse_args()


def parse_iso_date(value: str, option: str) -> date:
    """Convert a YYYY-MM-DD text value into a date."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{option} must use YYYY-MM-DD: {value!r}.") from exc


def create_month_list(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into one part for each month."""

    ranges = []
    current = start

    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        chunk_end = min(end, next_month - timedelta(days=1))
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)

    return ranges


def build_session() -> requests.Session:
    """Create the connection used to call Open-Meteo."""

    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "day-ahead-solar-forecasting/2.0"
        }
    )

    return session


def utc_from_api(value: str) -> datetime:
    """Convert an Open-Meteo date and time to UTC."""

    parsed = datetime.fromisoformat(value.strip().replace(" ", "T"))

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def parse_number(value: Any, variable: str, label: str) -> str:
    """Convert one weather value and check the limits for its variable."""

    if value is None:
        raise RuntimeError(f"Open-Meteo returned null for {label}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid Open-Meteo value for {label}: {value!r}."
        ) from exc

    if not math.isfinite(number):
        raise RuntimeError(f"Open-Meteo returned a non-finite value for {label}.")

    if variable in NON_NEGATIVE_VARIABLES and number < 0.0:
        raise RuntimeError(f"Open-Meteo returned a negative value for {label}.")

    if variable in PERCENTAGE_VARIABLES and not 0.0 <= number <= 100.0:
        raise RuntimeError(
            f"Open-Meteo returned a percentage outside 0-100 for {label}."
        )

    if variable in DIRECTION_VARIABLES and not 0.0 <= number <= 360.0:
        raise RuntimeError(
            f"Open-Meteo returned a direction outside 0-360 for {label}."
        )

    return format(number, ".12g")


def request_parameters(
    local_start: date,
    local_end: date,
    tilt: float,
    azimuth: float,
    model: str | None
) -> dict[str, str | float]:
    """Create the information sent to Open-Meteo for one request."""

    # Add one day before and after to avoid missing hours after UTC/local conversion.
    request_start = local_start - timedelta(days=1)
    request_end = local_end + timedelta(days=1)
    params: dict[str, str | float] = {
        "latitude": ",".join(str(region[2]) for region in REGIONS),
        "longitude": ",".join(str(region[3]) for region in REGIONS),
        "start_date": request_start.isoformat(),
        "end_date": request_end.isoformat(),
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
        "timeformat": "iso8601",
        "cell_selection": "land",
        "tilt": tilt,
        "azimuth": azimuth
    }

    if model:
        params["models"] = model

    return params


def fetch_chunk(
    session: requests.Session,
    local_start: date,
    local_end: date,
    tilt: float,
    azimuth: float,
    model: str | None,
    timeout: float
) -> list[dict[str, str]]:
    """Download weather data for all regions and a group of dates."""

    # Send one request containing all regions and weather variables.
    response = session.get(
        API_URL,
        params=request_parameters(local_start, local_end, tilt, azimuth, model),
        timeout=timeout
    )

    if not response.ok:
        raise RuntimeError(
            f"Open-Meteo request failed with HTTP {response.status_code}: "
            f"{response.text[:800]}."
        )

    payload: Any = response.json()

    # Open-Meteo returns a list when several locations are requested.
    locations = payload if isinstance(payload, list) else [payload]

    if len(locations) != len(REGIONS):
        raise RuntimeError(
            f"Expected {len(REGIONS)} Open-Meteo locations, received "
            f"{len(locations)}."
        )

    reference_times: tuple[str, ...] | None = None
    regional_hourly_data: list[tuple[str, dict[str, Any]]] = []

    # Check every location and make sure all regions use the same timestamps.
    for location, region_spec in zip(locations, REGIONS):
        region = region_spec[0]
        hourly = location.get("hourly") if isinstance(location, dict) else None

        if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
            reason = (
                location.get("reason", "missing hourly data")
                if isinstance(location, dict)
                else "invalid location data"
            )
            raise RuntimeError(f"{region}: invalid Open-Meteo response: {reason}.")

        api_times = tuple(str(value) for value in hourly["time"])

        if not api_times:
            raise RuntimeError(f"{region}: Open-Meteo returned no hourly timestamps.")

        if reference_times is None:
            reference_times = api_times
        elif api_times != reference_times:
            raise RuntimeError(
                f"{region}: hourly timestamps differ from other regions."
            )

        regional_hourly_data.append((region, hourly))

    selected_times: list[tuple[int, str]] = []
    rows_by_time: dict[str, dict[str, str]] = {}

    # Convert the shared timestamps once and keep only the requested local dates.
    for index, api_time in enumerate(reference_times):
        instant_utc = utc_from_api(api_time)
        local = instant_utc.astimezone(LOCAL_TIMEZONE)

        if not local_start <= local.date() <= local_end:
            continue

        key = instant_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
        offset = local.strftime("%z")

        if key in rows_by_time:
            raise RuntimeError(f"Duplicate Open-Meteo timestamp: {key}.")

        rows_by_time[key] = {
            "time_utc": key,
            "time_local": local.replace(tzinfo=None).isoformat(
                timespec="seconds"
            ),
            "utc_offset": f"{offset[:3]}:{offset[3:]}"
        }
        selected_times.append((index, key))

    # Add the weather values for each region to the prepared time rows.
    for region, hourly in regional_hourly_data:
        for variable in VARIABLES:
            values = hourly.get(variable)

            if not isinstance(values, list) or len(values) != len(reference_times):
                raise RuntimeError(f"{region}: invalid or missing {variable}.")

            column = f"{region}_{variable}"

            for index, key in selected_times:
                rows_by_time[key][column] = parse_number(
                    values[index],
                    variable,
                    column
                )

    return list(rows_by_time.values())


def validate_and_sort(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Check weather rows, remove duplicates, and sort them by time."""

    by_time: dict[str, dict[str, str]] = {}

    for row in rows:
        key = row["time_utc"]
        previous = by_time.get(key)
        if previous is not None and previous != row:
            raise RuntimeError(f"Conflicting Open-Meteo rows for {key}.")
        by_time[key] = row

    result = sorted(by_time.values(), key=lambda row: row["time_utc"])

    if not result:
        raise RuntimeError("Open-Meteo returned no rows.")

    if any(set(row) != set(OUTPUT_COLUMNS) for row in result):
        raise RuntimeError("Open-Meteo output schema is inconsistent.")

    return result


def validate_continuous_utc_hours(rows: list[dict[str, str]]) -> None:
    """Check that downloaded rows form a continuous sequence of UTC hours."""

    instants = [
        datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00"))
        for row in rows
    ]

    # Find consecutive timestamps that are not exactly one hour apart.
    bad_steps = [
        (left, right)
        for left, right in zip(instants, instants[1:])
        if right - left != timedelta(hours=1)
    ]

    # Report the first gap or irregular interval found.
    if bad_steps:
        left, right = bad_steps[0]
        raise RuntimeError(
            f"Open-Meteo UTC sequence is not hourly: {left} -> {right}."
        )


def expected_local_day_hours(day: date) -> int:
    """Return how many hours an Italian calendar day contains."""

    # Create midnight at the beginning of the requested Italian day.
    start_local = datetime(
        day.year,
        day.month,
        day.day,
        tzinfo=LOCAL_TIMEZONE
    )

    # Create midnight at the beginning of the following Italian day.
    following_day = day + timedelta(days=1)
    end_local = datetime(
        following_day.year,
        following_day.month,
        following_day.day,
        tzinfo=LOCAL_TIMEZONE
    )

    # Compare the two midnights in UTC to include clock changes correctly.
    elapsed = end_local.astimezone(timezone.utc) - start_local.astimezone(
        timezone.utc
    )

    return int(elapsed.total_seconds() // 3600)


def validate_local_calendar(
    rows: list[dict[str, str]],
    start: date,
    end: date
) -> None:
    """Check every Italian date, including daylight-saving time changes."""

    # Count how many downloaded rows belong to each Italian calendar day.
    actual_counts = Counter(
        datetime.fromisoformat(row["time_local"]).date() for row in rows
    )
    current = start

    # Check the expected 23, 24, or 25 hours for every requested day.
    while current <= end:
        expected = expected_local_day_hours(current)
        actual = actual_counts.get(current, 0)

        if actual != expected:
            raise RuntimeError(
                f"Open-Meteo local day {current} has {actual} rows; "
                f"Europe/Rome requires {expected}."
            )

        current += timedelta(days=1)

    # Make sure the output does not contain dates outside the requested range.
    unexpected_dates = set(actual_counts) - {
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    }

    if unexpected_dates:
        raise RuntimeError(
            "Open-Meteo output contains dates outside the requested local "
            f"range: {sorted(unexpected_dates)}."
        )


def write_csv_atomic(rows: list[dict[str, str]], output: Path) -> None:
    """Save actual-generation rows to a CSV file."""

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
    """Download, check, and save regional weather forecasts."""

    args = parse_args()
    start = parse_iso_date(args.start, "--start")
    end = parse_iso_date(args.end, "--end")

    if end < start:
        raise ValueError("--end must not be earlier than --start.")

    if args.pause < 0.0:
        raise ValueError("--pause cannot be negative.")

    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive.")

    session = build_session()
    chunks = create_month_list(start, end)
    all_rows: list[dict[str, str]] = []

    print("Open-Meteo Previous Runs API.")
    print(f"Period: {start} -> {end} (inclusive Europe/Rome dates).")
    print(f"Regions: {len(REGIONS)} | weather columns: {len(WEATHER_COLUMNS)}.")
    print(f"Tilt: {args.tilt:g} | azimuth: {args.azimuth:g}.")
    print(f"Model: {args.model or 'Best Match'}.")

    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(
            f"[{index:02d}/{len(chunks):02d}] "
            f"{chunk_start} -> {chunk_end}.",
            flush=True
        )
        all_rows.extend(
            fetch_chunk(
                session,
                chunk_start,
                chunk_end,
                args.tilt,
                args.azimuth,
                args.model,
                args.timeout
            )
        )
        if index != len(chunks):
            time.sleep(args.pause)

    rows = validate_and_sort(all_rows)
    validate_continuous_utc_hours(rows)
    validate_local_calendar(rows, start, end)
    write_csv_atomic(rows, args.output.resolve())
    print(f"Saved: {args.output.resolve()}.")
    print(f"Rows: {len(rows):,} | columns: {len(OUTPUT_COLUMNS)}.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
