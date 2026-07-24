# 耳戴式 IMU 数据采集、命名与标注规范

> 版本：v1.0
> 生效日期：2026-07-24
> 适用范围：运动/非运动与具体活动识别数据；头部姿态专项采集继续使用独立姿态模板。
> 权威原则：标签文件是唯一真值，文件名只用于定位，不得直接决定训练标签。

## 0. 制定依据

本规范针对本轮 360 份记录审核中已经出现的问题制定：

- 337 份不超过 180 秒的记录可按短试次审核，23 份超过 180 秒的记录只能保留会话级弱标签；
- `0722-李思思-跑步机走路.csv` 的 231 个窗口曾因名称匹配被误标为 `run`；
- 约 59 分钟爬坡数据曾被归入 `non_motion`；
- 佩戴取下、不对称佩戴曾被错误用作正常非运动负样本；
- 文件名含“3组/6组”的记录包含真实组间休息，整段动作标签会污染监督窗口；
- 约 170 分钟连续健身记录没有可信边界，只能验证动作覆盖、顺序和总组数，不能构造逐窗真值。

因此，后续采集必须将“可训练短试次”和“会话级长记录”从采集方式上分开，并以现场标签而非文件名语义作为训练真值。

## 1. 强制规则

1. 每个短试次只允许一名受试者、一种佩戴状态、一个目标活动。
2. 短试次推荐 30–120 秒，最长不得超过 180 秒；正好 180 秒仍属于短试次。
3. 超过 180 秒一律视为长会话，只能用于会话级弱验证，不得生成或推测逐窗标签。
4. 正常动作、休息、佩戴取下、不对称佩戴、掉落等必须分开采集；不得把佩戴异常当作非运动负样本。
5. 标签必须在采集现场完成。不得在事后仅根据文件名或记忆推断动作和边界。
6. 原始 CSV 不得覆盖、裁剪或改写；清洗、合并和重采样只能生成新文件。
7. 同一 `subject_id` 的数据只能进入 train、validation、test 中的一个集合。
8. 未提供 Schema v2 标签、标签校验失败或证据不足的数据一律不可训练。

## 2. 数据类型与采集方式

### 2.1 短试次：逐窗监督训练数据

短试次用于模型训练和逐窗评估。

- 每个文件只采一个目标活动，例如跑步机走路、深蹲、坐姿或说话。
- 采集员发出口令后，受试者先保持 3 秒准备状态，再开始动作。
- 推荐在动作开始和结束各记录一次软件事件；如软件不支持，使用可被视频和 IMU 同时观察到的拍手或明显点头同步事件。
- 动作完成后立即停止当前文件。休息应在文件外进行。
- 多组动作优先采用“一组一个 trial”。例如深蹲 3 组应保存为 T001、T002、T003，而不是一个含组间休息的长文件。
- 若必须在一个不超过 180 秒的文件中记录多组，必须实时记录每组和休息的准确起止时间，并提供 timeline；没有准确边界则整文件不可训练。

推荐时长：

| 类型 | 单 trial 建议 |
| --- | --- |
| 静坐、静站、说话 | 30–60 秒 |
| 走路、跑步机走路、爬坡、椭圆机 | 60–120 秒 |
| 重复力量动作 | 8–15 次或 20–60 秒 |
| 平板支撑 | 20–60 秒 |
| 佩戴异常专项 | 每种状态 20–30 秒，单独成文件 |

### 2.2 长会话：会话级弱验证数据

长会话用于检验整段推理的动作覆盖、顺序、组数、次数和过程，不参与逐窗监督训练。

- 允许运动—休息—运动组合，但必须同步填写会话记录。
- 每个动作至少记录：规范 `activity_id`、开始顺序、组数、每组次数或持续时长。
- 器械动作额外记录重量；有氧动作记录速度、坡度、阻力和持续时长。
- 每次休息记录开始/结束钟表时间或持续秒数。
- 掉落、松动、重新佩戴、设备重连、采集中断必须记录发生顺序和钟表时间。
- 近似时长只能写入 `weak_targets.process`，不得换算成逐窗边界。
- 如果需要把长会话升级为逐窗真值，必须依靠同步事件或视频重新人工标注，并经过第二人复核；不能使用动作时长累加推算。

### 2.3 佩戴异常专项

以下状态分别采集，不得与正常活动混在一个 trial：

- `removed`：佩戴取下；
- `asymmetric`：左右佩戴深浅或松紧明显不一致；
- `invalid`：挂脖、手持、掉落、单耳状态不符合协议或位置未知；
- `valid`：符合本次协议的正常佩戴。

佩戴异常标签的 `motion_state` 必须为 `null`，`window_trainable=false`。它们用于佩戴门控模型或异常测试，不进入运动/非运动模型。

## 3. 现场标准流程

### 3.1 会话开始前

1. 分配匿名 `subject_id`，禁止在文件名中使用姓名、手机号或工号。
2. 创建唯一 `session_id`，登记采集日期、采集员、协议版本和地点。
3. 记录左右设备 ID、耳侧、固件版本、App 版本、量程、配置采样率和滤波配置。
4. 检查两侧设备时间、采样率、剩余电量和存储空间。
5. 记录耳塞尺寸、佩戴深浅、松紧、眼镜/帽子等条件。
6. 正常佩戴后静止 10 秒完成校准；校准失败不得开始正式 trial。
7. 先采 30 秒正常佩戴静止基线。

### 3.2 每个短 trial

1. 采集员确认待采 `activity_id` 和 trial 编号。
2. 启动两侧设备和可选参考视频。
3. 记录开始同步事件。
4. 口头确认目标动作、速度/重量、次数或持续时长。
5. 受试者完成唯一目标活动；出现计划外动作时立即记录异常。
6. 记录结束同步事件并停止文件。
7. 当场填写标签，检查实际时长不超过 180 秒。
8. 如果发生掉落、重戴、无边界动作切换或严重中断，将整 trial 标为不可训练并重新采集。

### 3.3 会话结束前

1. 核对 CSV、标签和备注文件一一对应。
2. 打开每个文件抽查首尾时间、设备 ID、非空数据行和持续时长。
3. 检查左右设备是否缺一侧、时间是否倒序、是否存在明显长缺口。
4. 运行标签审计；存在 missing label、duplicate label 或 schema error 时，受试者不得离场。
5. 对照采集清单确认计划动作、组数和次数均已完成。
6. 只读归档原始数据，并生成文件大小和 SHA-256 校验值。

## 4. 目录和文件命名

### 4.1 原始归档目录

```text
data/raw/YYYYMMDD/session_id/subject_id/
  T0001/
    MMDD-S001-T0001-D01-left.csv
    MMDD-S001-T0001-D02-right.csv
    MMDD-S001-T0001.labels.json
    MMDD-S001-T0001.notes.txt
```

字段定义：

| 字段 | 格式 | 示例 |
| --- | --- | --- |
| 日期 | `MMDD` | `0725` |
| `subject_id` | `S` + 3 位数字 | `S001` |
| `trial_id` | `T` + 4 位数字，会话内唯一 | `T0007` |
| `device_id` | `D` + 2 位数字，映射表另存 | `D01` |
| 耳侧 | `left` / `right` | `left` |

命名要求：

- 仅使用 ASCII 字母、数字和连字符，禁止空格、中文标点、`+`、括号和自由文本备注。
- 动作名称不作为文件名真值；动作写入标签文件。
- 同一 trial 的左右设备必须具有相同日期、subject、trial，只允许 device 和 ear side 不同。
- 补采不得覆盖旧文件；使用新的 trial ID，并在标签中填写 `replaces_trial_id`。
- 文件内容与文件名不一致时，以现场证据和审核后的标签为准，同时记录冲突。

### 4.2 训练导入

训练目录中每个受试者 trial 应对应一份统一时基 CSV 和一个同名标签：

```text
data/training/activity/0725-S001-T0007.csv
data/training/activity/labels/0725-S001-T0007.labels.json
```

训练脚本从标签读取 `participant`、活动和边界，不从 `T0007` 推断语义。左右设备如分别建模，应生成两个明确的派生 capture，并保留原始 trial 与 device 映射。

## 5. Schema v2 标签规范

### 5.1 短试次必填字段

记录级必填：

- `schema_version`: 固定为 `"2.0"`；
- `taxonomy_version`；
- `date`、`participant`、`device`、`csv_file`；
- `annotation_scope`: `"full_recording"`；
- `recording.start_time/end_time/duration_seconds/row_count`；
- `window_trainable`；
- `annotation_quality.status/reason`。

每个 segment 必填：

| 字段 | 允许值或要求 |
| --- | --- |
| `start_s`、`end_s` | 相对 CSV 开始的秒数，`start_s < end_s` |
| `activity_id` | 规范类别 ID |
| `motion_state` | `motion` / `non_motion`；佩戴异常为 `null` |
| `wear_state` | `valid` / `removed` / `asymmetric` / `invalid` |
| `phase` | `active` / `rest` / `transition` / `artifact` |
| `window_trainable` | 只有可信且正常佩戴的段才为 `true` |
| `label_source` | `operator_event` / `video_review` / `manual_timeline` 等 |
| `confidence` | `high` / `medium` / `low` |
| `review_note` | 异常、协议偏离和复核结论 |

短试次示例：

```json
{
  "schema_version": "2.0",
  "annotation_scope": "full_recording",
  "participant": "S001",
  "csv_file": "../0725-S001-T0007.csv",
  "window_trainable": true,
  "recording": {
    "start_time": "2026-07-25T10:01:00.000+08:00",
    "end_time": "2026-07-25T10:02:00.000+08:00",
    "duration_seconds": 60.0,
    "row_count": 6001
  },
  "segments": [{
    "start_s": 3.0,
    "end_s": 57.0,
    "activity_id": "treadmill_walk",
    "motion_state": "motion",
    "wear_state": "valid",
    "phase": "active",
    "window_trainable": true,
    "label_source": "operator_event",
    "confidence": "high",
    "review_note": "速度3.0 km/h，全程正常佩戴"
  }]
}
```

### 5.2 长会话必填字段

- `annotation_scope`: `"session_weak"`；
- `window_trainable`: `false`；
- `segments`: 空数组；
- `weak_targets.ordered_activities`：按实际顺序记录活动；
- `weak_targets.total_sets`：能确认时填写；
- `weak_targets.process`：休息、重量变化、重戴、掉落、中断等；
- `weak_targets.window_boundaries_available`: `false`；
- `weak_targets.usage`: `"session_level_validation_only"`。

长会话示例：

```json
{
  "schema_version": "2.0",
  "annotation_scope": "session_weak",
  "window_trainable": false,
  "segments": [],
  "weak_targets": {
    "ordered_activities": [
      {"activity_id": "lat_pulldown", "sets": 4, "repetitions_per_set": 12, "weight_kg": 30},
      {"activity_id": "shoulder_press", "sets": 5, "repetitions_per_set": 12, "weight_kg": 10}
    ],
    "total_sets": 9,
    "process": [
      {"phase": "inter_set", "rest_minutes_range": [2, 3]},
      {"phase": "artifact", "clock_time": "10:24:31", "description": "右耳重新佩戴"}
    ],
    "window_boundaries_available": false,
    "usage": "session_level_validation_only"
  }
}
```

## 6. 运动与具体内容标签

二层标签必须同时填写：

- `motion_state` 表示身体活动状态；
- `activity_id` 表示具体内容。

主要映射：

| `motion_state` | 典型 `activity_id` |
| --- | --- |
| `motion` | `run`、`treadmill_walk`、`incline_walk`、`free_walk`、`stairs_up`、`stairs_down`、`sit_to_stand`、`bending`、`squat`、`plank` 等 |
| `non_motion` | `sitting`、`standing`、`speaking`、`chewing`、`drinking`、`gaming`、单纯头部动作 |
| `null` | `removed_wear`、`asymmetric_wear` 或其他无效佩戴 |

禁止使用以下模糊标签作为最终真值：

- “跑步机”：必须区分 `treadmill_walk`、`run` 或明确为未知且不可训练；
- “走路”：必须区分自由走路、跑步机走路和爬坡；
- “运动”“健身”“其他”：必须记录具体已知活动；确实无法确认时不可训练；
- “3组/6组”：组数是数量字段，不能替代动作和休息边界；
- “正常”：必须明确是正常佩戴、正常姿态还是目标动作执行正常。

新增活动必须先更新分类表并分配稳定 `activity_id`，不得临时拼写新名称。

## 7. Timeline 使用规范

只有满足以下条件时才能使用 timeline 生成监督窗口：

1. 整个文件不超过 180 秒；
2. 起止来自软件事件、同步视频或现场精确记录；
3. 每个区间同时填写 `motion_state`、`activity_id`、`wear_state`、`phase`；
4. 区间不重叠、不倒序，所有训练区间至少覆盖一个完整模型窗口；
5. 校准、转换、休息和佩戴异常段必须显式不可训练。

禁止：

- 按“每组约 1 分钟”累加推测边界；
- 根据模型预测反向生成训练真值；
- 将没有标注的间隙自动填成运动或非运动；
- 使用 timeline 绕过超过 180 秒的长会话限制。

## 8. 质量验收和退回标准

### 8.1 文件级验收

- CSV、标签、备注数量一致且文件名一一对应；
- 时间单调，持续时长与标签一致；
- 设备 ID、耳侧和受试者 ID 完整；
- 无重复 trial ID、空 CSV、缺失标签或孤立标签；
- 实测采样率在目标值 ±5% 内，长缺口和重连均有记录；
- 原始文件校验值已保存。

### 8.2 标签级验收

- 所有可训练段都有合法 `motion_state + activity_id`；
- 佩戴异常可训练窗口数必须为 0；
- 超过 180 秒的记录可训练窗口数必须为 0；
- 跑步机走路不得标成 `run`，爬坡不得标成 `non_motion`；
- 多组长记录不得整段标成 `active`；
- 至少 10% trial 由第二名标注者复核；
- 标签冲突必须保留双方结论和最终裁决人。

### 8.3 必须整段退回或重采

- 文件中出现未记录的动作切换且无可信边界；
- 掉落、挂脖、取下后继续采集但没有事件时间；
- 受试者或设备身份无法确认；
- 两侧数据无法对应同一 trial；
- 时间倒序、严重中断或采样率异常影响主要区间；
- 文件名、现场记录和视频互相冲突且无法裁决。

## 9. 采集交付清单

每个批次必须交付：

1. 原始 CSV/TXT，保持只读；
2. 每个 trial 的 Schema v2 标签；
3. `session_manifest`：subject、session、trial、device、ear side、协议版本；
4. 长会话 `weak_targets`；
5. 异常和补采清单；
6. 文件大小与 SHA-256 清单；
7. 标签审计报告；
8. 第二标注者复核记录；
9. 采集完成数量与计划数量对照表。

只有审计报告中 `missing_labels=[]`、`duplicate_labels=[]`、`violation_count=0` 的批次才能进入训练目录。
