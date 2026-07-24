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

## Integrated activity and posture application

网页运行在 `http://127.0.0.1:8000`。它有两种数据来源：上传历史 CSV，以及实时监听 `imu_output/live_imu.csv`。实时数据会每秒显示在 Figma 同步后的“运动”和“姿态健康”页面；运动状态和姿态判断都使用同一个本地模型与 IMU 姿态角数据。

### 启动网页

```powershell
python imu_analysis\app.py
```

### 实时读取耳机

仓库根目录的实时接收工具可将数据写入 `imu_output\live_imu.csv`，网页会自动读取，无需另启监控网页。

#### 方式一：直接蓝牙 BLE（推荐）

使用耳机的 WitMotion 蓝牙协议直接订阅 IMU 数据。**先退出 WitMotion 软件**，否则 Windows 无法同时占用同一个蓝牙设备。

```powershell
python `
  .\realtime_ble_imu.py `
  --model .\imu_output\headphone_all\model\l2_logistic_model.pkl `
  --csv .\imu_output\live_imu.csv `
  --status .\imu_output\live_status.json
```

#### 方式二：WitMotion CSV 实时监听

保持 WitMotion 软件连接耳机并开启 CSV 保存，然后监听它保存 CSV 的目录。桥接程序只处理启动后新写入的数据。

```powershell
python `
  .\witmonitor_csv_bridge.py `
  --source-dir '替换为 WitMotion 的 CSV 保存目录' `
  --model .\imu_output\headphone_all\model\l2_logistic_model.pkl `
  --csv .\imu_output\live_imu.csv
```

运行实时桥接前，如缺少依赖，请安装：`pip install bleak pyserial`。不要同时让两种实时读取方式连接同一只耳机。

### Demo 多动作模型

新链路采用 50 Hz 六轴输入、4 秒窗口、0.5 秒步长的 LightGBM 多分类模型。模型只输出概率；组内、组间、训练段和每日汇总由独立状态机处理。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

训练当前数据：

```powershell
python `
  imu_analysis\train_activity_model.py `
  .\data\datafortraining `
  --output-dir imu_output\demo_activity
```

混合动作和多组文件必须按 `data_collection_timeline_template.csv` 的格式提供逐段标签，再通过 `--timeline <文件>` 传入。没有时间轴的混合文件会被自动排除。

主要产物：

- `imu_output/demo_activity/model/activity_model.pkl`：Python 实时推理模型。
- `activity_model.txt`：LightGBM 原生模型。
- `metrics.json`：固定用户切分的动作与运动二分类指标，并包含 `demo_ready` 验收结果。
- `window_features.csv`：特征缓存；原始数据未变化时可通过 `--features-csv` 加快重训。

回放一段真实数据：

```powershell
python `
  imu_analysis\replay_activity.py <csv路径> `
  --model imu_output\demo_activity\model\activity_model.pkl
```

候选模型只有通过验收门槛且不低于现有冠军时才会被提升：

```powershell
python imu_analysis\promote_activity_model.py imu_output\demo_activity\model
```

#### 实时 BLE

新接收器同时支持当前的 `55 61` 20 字节帧和新的精简协议。精简协议每个样本固定 14 字节：`sequence_id uint16 + 6×int16`，批量通知带版本与 CRC16。

```powershell
python imu_analysis\realtime_activity_ble.py `
  --address 'F6:B1:93:B5:2B:23' `
  --model imu_output\demo_activity\model\activity_model.pkl `
  --database imu_output\workouts.sqlite3
```

网页增加了：

- `GET /api/daily`：当天总训练时长、有效运动时长、按时间排序的训练段、动作组数或有氧时长。
- `GET /api/live`：实时动作概率、组内/组间状态、信号质量。
- `POST /api/analyze`：上传 CSV 后走同一套多分类推理和策略链路。

当前数据训练出的首版模型会如实写入 `demo_ready`。未通过门槛时可用于链路联调，不应宣称达到 Demo 验收精度。

### 坐姿头部姿态模型

新增的独立模型检测 `正常坐姿 / 低头 / 抬头 / 歪头 / 偏头看向一侧`，并由时序状态机记录异常姿态连续时长。它不会与运动动作模型共用标签。

训练：

```powershell
python imu_analysis\train_head_posture_model.py `
  .\data\datafortraining `
  --output-dir imu_output\head_posture
```

主要产物：

- `imu_output/head_posture/model/head_posture_model.pkl`：Python 实时推理模型
- `imu_output/head_posture/model/metrics.json`：整人留出测试指标与验收状态
- `imu_output/head_posture/model/MODEL_CARD.md`：输入约束、时序规则和已知限制
- `imu_output/head_posture/window_features_with_split.csv`：可复查的窗口与受试者切分

实时使用 `StreamingHeadPostureClassifier` 前，用户需要保持自然坐姿完成 10 秒校准。相同异常稳定 3 秒后开始计时，恢复正常 2 秒后结束事件并输出本次持续秒数。实时样本需要包含加速度、角速度、姿态角和四元数。

当前候选模型未通过离线验收，只能用于链路联调。现有采集对“歪头”的动作定义在不同受试者间不一致，需要按统一角度和保持时长重新补采后再训练。

补采时使用 `head_posture_collection_template.csv`。每位受试者应在同一次佩戴中依次采正常、低头、抬头、左右歪头和左右转头；每个目标姿态稳定保持至少 20 秒，并记录准确起止时间。建议至少 20 位受试者，测试人员不能出现在训练集中。

实时测试无需修改现有耳机六轴 BLE 协议。启动 `imu_analysis/app.py` 后打开 `http://127.0.0.1:8000/test.html`，点击“启动蓝牙采集”，切换到“坐姿”，保持自然坐姿并点击“10 秒重新校准”。页面会显示当前姿态、模型置信度和本次异常连续时间。六轴相对姿态估计、姿态模型和运动模型共享同一批实时样本。
