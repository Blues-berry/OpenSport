# OpenSport

OpenSport contains tools for collecting and analyzing IMU data from ear-worn devices.

## Included tools

- `realtime_ble_imu.py`: reads WitMotion BLE IMU notifications and can write a live CSV stream.
- `imu_analysis/`: checks recordings, cleans samples, extracts window features, and trains a small L2-regularized logistic-regression baseline.
- `采集计划.md` and `IMU数据采集计划_修订标红版.docx`: data-collection plans.
- `build_collection_plan_docx.py`: rebuilds the collection-plan document.

## Setup

```powershell
python -m pip install numpy pandas bleak
```

## Run the analysis pipeline

```powershell
python imu_analysis/run_pipeline.py "path/to/data" --work-dir "imu_output/run_01"
```

The pipeline keeps raw data, cleaned samples, extracted features, and model outputs under `imu_output/`, which is intentionally excluded from version control.

## Run live BLE capture

```powershell
python realtime_ble_imu.py --csv live_imu.csv
```

Update the device name and address in `DEVICES` before capture. Close any other application that holds the BLE connection.

## Data handling

Do not commit raw sensor recordings or files containing personal information. The repository ignores data and generated outputs by default.
