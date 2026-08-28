# Day-ahead solar forecasting

This project builds an hourly dataset and evaluates day-ahead forecasts of
Italy's national photovoltaic (PV) generation. It combines generation and
installed-capacity data published by Terna, Italy's electricity transmission
system operator, with archived regional weather forecasts from Open-Meteo.

The scripts in *src/* download the source data, reconstruct monthly PV
capacity, build the final dataset, run an exploratory data analysis, and train
a TabM neural network.

Default outputs are saved under *share/*. Default input and output paths can be
changed. Run any script with *--help* to see its available options.

## Installation

Python 3.11 or newer is recommended.

Install the data-pipeline dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the additional dependencies before running the EDA or TabM:

```bash
python -m pip install -r requirements-eda.txt
python -m pip install -r requirements-model.txt
```

## 1. Download Terna data

*download_terna.py* downloads two Terna datasets:

- *actual-generation* downloads measured national PV generation;
- *installed-renewables* downloads annual files with monthly changes in
  installed renewable capacity by region and province.

They can be downloaded separately or together with the *all* command.

### Terna credentials

Actual Generation requires a Key and Secret. Register an application at the
[Terna Developer Portal](https://developer.terna.it/), subscribe it to the
Public API package, and make the credentials available in the shell:

```bash
export TERNA_CLIENT_ID="your-client-id"
export TERNA_CLIENT_SECRET="your-client-secret"
```

### Usage examples

Download both datasets:

```bash
python src/download_terna.py all \
    --start 2024-01-20 \
    --end 2026-06-30
```

Default outputs:

```text
share/terna/actual_generation.csv
share/terna/installed_renewables_2024.csv
share/terna/installed_renewables_2025.csv
share/terna/installed_renewables_2026.csv
```

*--start* and *--end* accept either a complete date or a year. A start year
means January 1, and an end year means December 31. Actual Generation uses the
complete dates. For Installed Renewables, *2024-01-20* is read as *2024* and
*2026-06-30* as *2026*, so the script downloads the complete files for 2024,
2025, and 2026.

Download the two datasets separately:

```bash
python src/download_terna.py actual-generation \
    --start 2024-01-20 \
    --end 2026-06-30

python src/download_terna.py installed-renewables \
    --start 2024 \
    --end 2026
```

Terna provides Actual Generation every hour in some periods and every 15
minutes in others. The downloaded CSV keeps the original time interval. The
Installed Renewables files contain monthly changes, not cumulative capacity.

## 2. Build monthly national and regional PV capacity

Create the national series:

```bash
python src/build_national_pv_capacity.py
```

Default output:

```text
share/terna/national_pv_capacity_monthly.csv
```

The file contains Italy's installed PV capacity and its monthly changes.

The values are reconstructed from Terna's
[monthly reports](https://www.terna.it/it/sistema-elettrico/pubblicazioni/rapporto-mensile).

Create the regional series after downloading the installed-renewables files
and building the national series:

```bash
python src/build_regional_pv_capacity.py
```

Default output:

```text
share/terna/regional_pv_capacity_monthly.csv
```

The file contains installed PV capacity and monthly changes for each Italian
region.

Terna's installed-renewables data cover December 2024 and every month from
January 2025 onward. Based on a comparison with Terna's monthly reports, the
December 2024 data are treated as the total installed PV capacity at the end of
2024. Regional capacity from January through November 2024 is estimated using
two sources. The December 2024 regional file determines the capacity differences
between regions, while the national file determines how capacity changes during
the earlier months.

## 3. Download Open-Meteo regional weather forecasts

```bash
python src/download_open_meteo.py \
    --start 2024-01-20 \
    --end 2026-07-01
```

Default output:

```text
share/open-meteo/regional_weather.csv
```

The script requests 24-hour-ahead forecasts for the capital of each Italian
region. It downloads:

- shortwave, direct, and diffuse radiation;
- direct normal irradiance and global tilted irradiance;
- precipitation, temperature, relative humidity and cloud cover;
- wind speed and wind direction.

Additional weather variables can be added in *download_open_meteo.py*. The
available variables are listed in the
[Open-Meteo Previous Runs API documentation](https://open-meteo.com/en/docs/previous-runs-api).

The default variables produce 220 weather columns: 11 variables for 20 regions.
The file also contains *time_utc*, *time_local*, and *utc_offset*.

Open-Meteo Best Match is used by default. Use *--model* to select a specific
model. Global tilted irradiance uses a 30-degree tilt and a 0-degree azimuth by
default; both values can be changed with *--tilt* and *--azimuth*.

Open-Meteo solar radiation, irradiance, and precipitation values recorded at
01:00 describe the period from 00:00 to 01:00. In the final dataset, this period
belongs to the 00:00 row, so the values are moved from 01:00 to 00:00. For the
same reason, completing the 23:00 row of the final requested day requires the
00:00 Open-Meteo row of the following day. Since Open-Meteo downloads complete
dates, one extra day must be included. UTC timestamps keep hours unique when
Italian clocks change.

## 4. Build the final hourly dataset

The final dataset can include any combination of these four sources:

- *actual-generation*;
- *weather*;
- *national-capacity*;
- *regional-capacity*.

At least *actual-generation* or *weather* must be included because the capacity
files do not contain hourly data.

Include all four sources:

```bash
python src/build_final_dataset.py \
    --start 2024-01-20 \
    --end 2026-06-30
```

Include only selected sources:

```bash
python src/build_final_dataset.py \
    --include actual-generation weather national-capacity \
    --start 2024-01-20 \
    --end 2026-06-30
```

Default output:

```text
share/final_dataset.csv
```

The builder:

- keeps Terna's hourly values unchanged and averages each complete group of
  four 15-minute values into one hourly value;
- excludes missing or incomplete generation hours;
- moves previous-hour Open-Meteo values back by one hour so they describe the
  same hourly interval as Terna generation, while instant weather values remain
  unchanged;
- keeps one row when the autumn local hour occurs twice;
- adds the installed PV capacity at the start of each month to every hourly row
  in that month.

The final dataset determines how much training history is available. For each
month used as the test set, all earlier rows are available for training, except
the same month of the previous year that is used for validation.

## 5. Explore the final dataset

*explore_final_dataset.py* reads the final hourly dataset without changing it.
It creates CSV summaries and diagnostic plots for:

- data quality and hourly coverage;
- generation by hour, month, and year;
- relationships between weather and generation;
- national and regional installed capacity;
- PV capacity factor.

The script does not require all four data sources. It can analyze any
combination of generation, weather, national capacity, and regional capacity
included in the final dataset.

Run the analysis with the default input and output paths:

```bash
python src/explore_final_dataset.py
```

Default output directory:

```text
share/eda
```

## 6. Train and evaluate TabM

*train_tabm.py* trains and evaluates TabM on the final hourly dataset. Each
forecast month is kept separate from training and used as the test period.

The default model inputs are:

- 20 regional shortwave-radiation forecasts;
- 20 regional direct-normal-irradiance forecasts;
- 20 regional global-tilted-irradiance forecasts;
- 20 regional cloud-cover forecasts;
- national installed PV capacity.

By default, the script evaluates months from January through June 2026.
For each forecast month:

1. the same calendar month of the previous year is temporarily removed and
   used for validation;
2. the remaining earlier rows are used for training;
3. validation selects the number of epochs;
4. the validation month is added back and a new model is trained on all earlier
   rows;
5. the forecast month is used only for final evaluation.

Training begins with the first row of the final dataset. To use a different
training start date, rebuild the final dataset with the desired start date.

Monthly metrics use local hours from 06:00 through 20:59. Aggregate metrics
are also reported for all hours. The reported metrics are the Coefficient of
Determination (R²), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE),
Mean Bias Error (MBE), and Weighted Absolute Percentage Error (WAPE).

The script saves:

- metrics for each forecast month (*monthly_metrics.csv*);
- metrics calculated over all forecast months (*metrics.csv*);
- actual and predicted generation for every test row (*forecast_records.csv*);
- model input columns (*selected_features.csv*).

It also creates:

- actual generation compared with model predictions
  (*actual_vs_predicted.png*);
- RMSE for each forecast month (*monthly_rmse.png*);
- prediction residuals grouped by local hour (*residuals_by_local_hour.png*).

Run the default configuration:

```bash
python src/train_tabm.py
```

Use *--forecast-start* and *--forecast-end* to specify another test period:

```bash
python src/train_tabm.py \
    --forecast-start 2025-07 \
    --forecast-end 2025-12
```

Complete feature groups can be added or removed with *--feature-groups*. The
available groups are:

```text
shortwave-radiation
direct-radiation
diffuse-radiation
direct-normal-irradiance
global-tilted-irradiance
precipitation
temperature
cloud-cover
wind-speed
wind-direction
relative-humidity
national-capacity
regional-capacity
```

Check the selected features and time splits without training:

```bash
python src/train_tabm.py --check-only
```

Default output directory:

```text
share/tabm
```

Use *--seed* to change the random seed and *--threads* to control the number
of CPU threads used by PyTorch.
