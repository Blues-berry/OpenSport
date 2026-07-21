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

## 双链路实时推理

项目提供两条彼此隔离的实时数据链路。两者都对 `WT22222` 与
`WT901BLE11` 分别推理，模型输出为 `运动` 或 `非运动`。

| 链路 | 数据来源 | 适用场景 | 网页数据源 | 输出文件 |
| --- | --- | --- | --- | --- |
| WitMotion 记录流 | WitMotion 写入的 CSV | 希望 WitMotion 保持连接和采集 | `WitMotion 记录流` | `imu_output/live_imu.csv` |
| BLE 直连 | 设备 GATT 通知帧 | 不使用 WitMotion，追求帧级直接接收 | `BLE 直连` | `imu_output/live_ble_imu.csv` |

不要让两条链路同时连接同一台设备。WitMotion 正在连接设备时，Windows
通常不会允许 Python 再建立 BLE 直连；此时应使用第一条记录流链路。

### 1. WitMotion 记录流（采集期间推荐）

该方案不扫描 BLE、不打开串口、不修改 WitMotion 的原始记录。它只尾随
WitMotion 写入的 CSV，并将新增行归一化、推理后回传到网页。

WitMotion 的记录根目录固定为：

```text
D:\download\Witmotion(V2026.6.26.0)\Record
```

启动文件桥接：

```powershell
python witmonitor_csv_bridge.py --model imu_output\run_20260721\model\l2_logistic_model.pkl
```

桥接每 0.2 秒检查新增内容。采集结束后可以重命名记录目录；桥接按 Windows
文件身份追踪，不会把同一个 `data_0.csv` 重复作为新会话导入。实际刷新延迟
取决于 WitMotion 将内存数据追加到 CSV 的频率。

### 2. BLE 直连（不经过 CSV）

该方案通过 `BleakClient` 直接订阅两台设备的 GATT 通知，解析 WitMotion
`55 61` 二进制帧，再以 100 Hz 规范化数据执行推理。它不读取 WitMotion 的
CSV，也不会经过 WitMotion 软件。

先在 WitMotion 中断开两台设备，再启动接收器：

```powershell
python realtime_ble_imu.py --model imu_output\run_20260721\model\l2_logistic_model.pkl
```

直连状态写入 `imu_output/live_ble_status.json`；数据写入
`imu_output/live_ble_imu.csv`。如需仅检查附近广播设备，可在 WitMotion
未连接设备时运行：

```powershell
python realtime_ble_imu.py --scan
```

### 网页面板

启动网页服务：

```powershell
python monitor_server.py --port 8767
```

打开 `http://127.0.0.1:8767`，在顶部“数据链路”中选择：

- `WitMotion 记录流`：显示文件桥接的实时推理结果；
- `BLE 直连（需关闭 WitMotion）`：显示直接 GATT 通知推理结果。

“设备显示”只筛选网页中显示的设备，不会改变采集对象或设备连接。

## 可选串口桥接

`serial_imu_bridge.py` 用于带串口 BLE 接收器的实验，不属于上面两条主链路。

### Reuse a WitMonitor serial receiver

When WitMonitor uses a serial BLE receiver (for example `COM3`), close
WitMonitor first and bridge the receiver directly to the dashboard stream:

```powershell
python serial_imu_bridge.py --port COM3 --baud 9600 --model imu_output\run_20260721\model\l2_logistic_model.pkl
```

Then open `http://127.0.0.1:8765`.  The bridge expects the 20-byte `55 61`
WitMotion IMU frame emitted by the receiver.

## Data handling

Do not commit raw sensor recordings or files containing personal information. The repository ignores data and generated outputs by default.
