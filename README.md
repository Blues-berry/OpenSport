# OpenSport

OpenSport contains tools for collecting and analyzing IMU data from ear-worn devices.

The canonical local layout is documented in
[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). Raw recordings and
prepared training data live under `data/`; generated features, models, logs,
and live captures live under `imu_output/`.

## Included tools

- `realtime_ble_imu.py`: reads WitMotion BLE IMU notifications and can write a live CSV stream.
- `src/opensport/`: versioned data, model, runtime, storage, device and API contracts.
- `imu_analysis/`: backward-compatible training, replay and live entry points.
- [采集执行与标注规范](docs/IMU_DATA_COLLECTION_AND_LABELING_STANDARD.md)：采集方式、文件命名、Schema v2 标注、长会话弱标签和交付验收的强制标准。
- [实现架构](docs/ARCHITECTURE.md)：严格导入、双模型、状态机和模型注册关系。
- `采集计划.md` and `IMU数据采集计划_修订标红版.docx`: data-collection plans.
- `build_collection_plan_docx.py`: rebuilds the collection-plan document.

## Setup

```powershell
python -m pip install -e .
python -m pip install lightgbm scikit-learn bleak pyserial pytest
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

新链路采用 50 Hz 六轴输入、4 秒因果窗口、1 秒步长的 LightGBM 多分类模型。一个模型同时输出具体 `activity_id`，并把所有 `motion` 类别的概率相加得到 `motion_probability`；组内、组间、训练段和每日汇总由独立状态机处理。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

训练当前数据：

```powershell
python `
  imu_analysis\train_activity_model.py `
  .\data\training\activity `
  --output-dir imu_output\runs\activity_v1
```

历史数据只能通过额外的 `--allow-legacy-training` 训练实验模型；即使指标
达到数值门槛，也不会因为缺少 `gold` 正式评估而晋升冠军。

训练前先生成或复核统一双层标签：

```powershell
python scripts\build_filename_labels.py `
  --training-dir data\training\activity `
  --labels-dir data\training\activity\labels `
  --report imu_output\dual_label_audit.json `
  --apply
```

已有 Schema v2 人工标签会被校验并默认保留；清单脚本只为尚未标注的文件生成审核骨架，不使用子串猜测。只有明确传入 `--overwrite-v2` 才会重建现有 v2 标签。正好 180 秒及以下的单动作记录可整段标注；超过 180 秒的记录一律保存为 `session_weak`，只记录动作清单、顺序、组数、次数和过程说明，不生成逐窗监督标签。timeline 仅用于有可信边界的短记录，不能覆盖长会话限制。

每个短记录段同时包含：

- `motion_state`：`motion / non_motion`；佩戴异常为 `null`。
- `activity_id`：规范的具体运动或非运动内容。
- `wear_state`：`valid / removed / asymmetric / invalid`。
- `phase`、`window_trainable`、`label_source`、`confidence` 和审核备注。

当前“运动”采用身体活动口径：走路、跑步机走路、爬坡、上下楼、坐起站起、弯腰和健身动作属于运动；说话、咀嚼、喝水、游戏及单纯头部动作属于非运动。佩戴取下、不对称佩戴、挂脖和掉落均不可训练。

主要产物：

- `imu_output/runs/<run>/model/activity_model.pkl`：Python 实时推理模型。
- `activity_model.txt`：LightGBM 原生模型。
- `metrics.json`：固定用户切分的动作与运动二分类指标，并包含 `demo_ready` 验收结果。
- `window_features.csv`：特征缓存；原始数据未变化时可通过 `--features-csv` 加快重训。
- `data_manifest.csv`：每个文件是否进入窗口训练及排除原因。
- `dual_label_audit.json`：逐文件双层标签和长会话弱目标审计。
- `weak_session_targets.json`：长会话继承受试者切分后的弱验证目标；只有 validation/test 项计入正式会话指标。

特征缓存包含标签 schema、证据等级与分类表版本；版本变化后旧缓存会被拒绝。少于 5 名 `gold` 受试者的运动类别合并为 `other_motion`；活动模型的所有非运动内容统一为 `other_non_motion`，不再区分坐姿、站姿和头部方向。

回放一段真实数据：

```powershell
python `
  imu_analysis\replay_activity.py <csv路径> `
  --model imu_output\models\activity\candidate\activity_model.pkl
```

候选模型只有通过验收门槛且不低于现有冠军时才会被提升：

```powershell
python imu_analysis\promote_activity_model.py imu_output\runs\<run>\model
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

### 头部姿态模型

独立模型输出 `normal / poor`，并附带低头、抬头、侧倾和转头方向。
坐姿和站姿只保留为评估上下文，不是产品类别，也不会与运动模型共用标签。

训练：

```powershell
python imu_analysis\train_head_posture_model.py `
  .\data\training\activity `
  --output-dir imu_output\runs\posture_v1
```

新数据必须提供独立姿态标签。`--allow-legacy-labels` 仅用于训练明确标记的
历史实验模型。

主要产物：

- `imu_output/head_posture/model/head_posture_model.pkl`：Python 实时推理模型
- `imu_output/head_posture/model/metrics.json`：整人留出测试指标与验收状态
- `imu_output/head_posture/model/MODEL_CARD.md`：输入约束、时序规则和已知限制
- `imu_output/head_posture/window_features_with_split.csv`：可复查的窗口与受试者切分

实时使用 `StreamingHeadPostureClassifier` 前，用户保持自然中立位完成 10 秒
校准。不良姿态持续 30 秒触发提醒，恢复正常 5 秒后解除。优先使用经校验的
硬件姿态；只有六轴时使用相同的相对姿态实现并把 Yaw 标记为降级。

当前候选模型未通过离线验收，只能用于链路联调。现有采集对“歪头”的动作定义在不同受试者间不一致，需要按统一角度和保持时长重新补采后再训练。

补采时使用 `head_posture_collection_template.csv`。每位受试者应在同一次佩戴中依次采正常、低头、抬头、左右歪头和左右转头；每个目标姿态稳定保持至少 20 秒，并记录准确起止时间。建议至少 20 位受试者，测试人员不能出现在训练集中。

实时测试无需修改现有耳机六轴 BLE 协议。旧 20 字节帧中的硬件
Roll/Pitch/Yaw 会被保留；精简六轴协议走相对姿态融合。应用在没有模型时也能
启动，`GET /api/models` 返回 champion/candidate/missing 状态；candidate 始终
带“实验模型/低于正式验收标准”提示。
