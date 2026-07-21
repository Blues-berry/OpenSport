# OpenSport

OpenSport contains tools for collecting and analyzing IMU data from ear-worn devices.

## Included tools

- `realtime_ble_imu.py`: reads WitMotion BLE IMU notifications and can write a live CSV stream.
- `imu_analysis/`: checks recordings, cleans samples, extracts window features, and trains a small L2-regularized logistic-regression baseline.
- `采集计划.md` and `IMU数据采集计划_修订标红版.docx`: data-collection plans.
- `build_collection_plan_docx.py`: rebuilds the collection-plan document.

## Setup

```powershell
python -m pip install numpy pandas bleak pyserial
```

## Run the analysis pipeline

```powershell
python imu_analysis/run_pipeline.py "path/to/data" --work-dir "imu_output/run_01"
```

The pipeline keeps raw data, cleaned samples, extracted features, and model outputs under `imu_output/`, which is intentionally excluded from version control.

## Run dual-device BLE inference

The normal live workflow connects Python directly to both configured WitMotion
devices and runs the existing binary model independently for each stream. Close
WitMotion first: Windows cannot share the same BLE connection with the desktop
application.

Start the dashboard:

```powershell
python monitor_server.py
```

In a second terminal, start the receiver with an explicit model path:

```powershell
python realtime_ble_imu.py --model imu_output\run_20260721\model\l2_logistic_model.pkl
```

Open `http://127.0.0.1:8765`. Both `WT22222` and `WT901BLE11` are connected
and shown separately. The receiver writes normalized 100 Hz samples to
`imu_output/live_imu.csv` and link health to `imu_output/live_status.json`.
The browser should never be used to choose which device is captured.

Use `python realtime_ble_imu.py --scan` only to inspect nearby BLE advertising
devices. Update the fixed addresses in `DEVICES` if the hardware addresses
change; it does not change the two-device capture behavior.

The deployed classifier only reports `运动` or `非运动`. Its model accuracy is
bounded by the existing training data; the BLE integration does not turn it
into an action-recognition model.

## Legacy WitMonitor bridges

`serial_imu_bridge.py` and `witmonitor_csv_bridge.py` remain available for
recording inspection and serial-receiver experiments, but they are not part of
the direct BLE inference workflow above. Do not run either bridge together with
`realtime_ble_imu.py` for the same device.

### Reuse a WitMonitor serial receiver

When WitMonitor uses a serial BLE receiver (for example `COM3`), close
WitMonitor first and bridge the receiver directly to the dashboard stream:

```powershell
python serial_imu_bridge.py --port COM3 --baud 9600 --model imu_output\run_20260721\model\l2_logistic_model.pkl
```

Then open `http://127.0.0.1:8765`.  The bridge expects the 20-byte `55 61`
WitMotion IMU frame emitted by the receiver.

### Use WitMonitor as the data source

To keep WitMonitor connected while the web panel runs, enable its CSV recording
to a folder under `IMU数据采集` and start the file bridge. It tails only CSV rows
created after it starts, so old recordings are never replayed as live data.

```powershell
python witmonitor_csv_bridge.py --source-dir "IMU数据采集" --model imu_output\run_20260721\model\l2_logistic_model.pkl
```

## Data handling

Do not commit raw sensor recordings or files containing personal information. The repository ignores data and generated outputs by default.
