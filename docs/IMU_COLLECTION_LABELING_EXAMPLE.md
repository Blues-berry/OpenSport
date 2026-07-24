# IMU 采集与标注示例

本示例说明一次“跑步机走路”短试次应如何采集、记录和标注。示例中的动作真值来自现场事件，不来自文件名。

## 1. 示例任务

| 项目 | 记录值 |
| --- | --- |
| 日期 | 2026-07-25 |
| 受试者 | `S001` |
| 会话 | `SESSION-20260725-01` |
| trial | `T0007` |
| 目标活动 | 跑步机走路 |
| 规范活动 ID | `treadmill_walk` |
| 速度 | 3.0 km/h |
| 计划时长 | 60 秒 |
| 佩戴 | 双耳正常、无松动 |
| 采集员 | `OP01` |

文件名只记录身份和 trial，不写“跑步机走路”：

```text
0725-S001-T0007-D01-left.csv
0725-S001-T0007-D02-right.csv
0725-S001-T0007.events.csv
0725-S001-T0007.labels.json
0725-S001-T0007.notes.txt
```

## 2. 如何记录原始数据

开始采集后，CSV 每一行保存同一设备的一次 IMU 采样。至少保留：

```csv
时间,设备,加速度X(g),加速度Y(g),加速度Z(g),角速度X(°/s),角速度Y(°/s),角速度Z(°/s)
2026-07-25 10:01:00.000,D01,0.012,-0.018,0.998,0.12,-0.08,0.03
2026-07-25 10:01:00.010,D01,0.013,-0.019,1.001,0.15,-0.07,0.02
```

记录要求：

1. 原始时间戳、设备 ID 和传感器值必须直接来自采集设备，不手工补值。
2. 左右设备分别保存原始文件；不得为了对齐而修改原始 CSV。
3. 设备配置、目标采样率、量程、固件版本和左右耳映射写入会话清单。
4. 如果出现丢包、重连、掉落或重新佩戴，原始数据继续保留，同时记录事件。

## 3. 如何记录现场事件

采集员在事件发生时按键记录，不在采集结束后凭记忆填写：

```csv
clock_time,elapsed_s,event,activity_id,operator_id,notes
2026-07-25T10:01:00.000+08:00,0.0,recording_start,,OP01,开始记录
2026-07-25T10:01:03.000+08:00,3.0,activity_start,treadmill_walk,OP01,速度3.0 km/h
2026-07-25T10:01:57.000+08:00,57.0,activity_end,treadmill_walk,OP01,动作正常完成
2026-07-25T10:02:00.000+08:00,60.0,recording_end,,OP01,停止记录
```

这四个事件形成三个区间：

| 区间 | 含义 | 是否训练 |
| --- | --- | --- |
| 0–3 秒 | 开始准备 | 否 |
| 3–57 秒 | 跑步机走路 | 是 |
| 57–60 秒 | 结束缓冲 | 否 |

如果没有 `activity_start/activity_end` 的可信时间，就不能把 60 秒整段假定为动作。

## 4. 如何填写标签

正确的 `0725-S001-T0007.labels.json`：

```json
{
  "schema_version": "2.0",
  "taxonomy_version": "e8d5031ca3a207fe",
  "date": "2026-07-25",
  "participant": "S001",
  "device": "D01+D02",
  "csv_file": "../0725-S001-T0007.csv",
  "raw_action": "跑步机走路",
  "annotation_scope": "full_recording",
  "window_trainable": true,
  "recording": {
    "start_time": "2026-07-25T10:01:00.000+08:00",
    "end_time": "2026-07-25T10:02:00.000+08:00",
    "duration_seconds": 60.0,
    "row_count": 6001
  },
  "annotation_quality": {
    "status": "reviewed",
    "trainable": true,
    "reason": "现场开始/结束事件完整，正常佩戴，无动作切换"
  },
  "evidence": {
    "event_file": "0725-S001-T0007.events.csv",
    "operator_id": "OP01",
    "reviewer_id": "RV01",
    "reviewed_at": "2026-07-25T11:20:00+08:00"
  },
  "segments": [
    {
      "start_s": 0.0,
      "end_s": 3.0,
      "activity_id": "standing",
      "motion_state": "non_motion",
      "wear_state": "valid",
      "phase": "transition",
      "window_trainable": false,
      "label_source": "operator_event",
      "confidence": "high",
      "review_note": "动作开始前准备"
    },
    {
      "start_s": 3.0,
      "end_s": 57.0,
      "activity_id": "treadmill_walk",
      "motion_state": "motion",
      "wear_state": "valid",
      "phase": "active",
      "window_trainable": true,
      "label_source": "operator_event",
      "confidence": "high",
      "review_note": "跑步机速度3.0 km/h，现场事件边界"
    },
    {
      "start_s": 57.0,
      "end_s": 60.0,
      "activity_id": "standing",
      "motion_state": "non_motion",
      "wear_state": "valid",
      "phase": "transition",
      "window_trainable": false,
      "label_source": "operator_event",
      "confidence": "high",
      "review_note": "动作结束后缓冲"
    }
  ],
  "weak_targets": {}
}
```

标签判断理由：

- 跑步机上是走路，因此是 `treadmill_walk`，不能写成 `run`。
- 走路属于身体活动，因此 `motion_state=motion`。
- 佩戴正常，因此 `wear_state=valid`。
- 只有 3–57 秒同时满足动作明确、边界可信和正常佩戴，所以只有该段可训练。
- `label_source=operator_event` 表示边界来自现场事件；如边界由同步视频复核，应使用 `video_review` 并记录视频或复核证据。

## 5. 哪些标签不能进入正式验收集

Schema v2 只说明字段结构正确，不代表标签一定具有真值质量。数据用途必须根据证据来源判断：

| 标签来源 | 可用于训练 | 可进入正式 validation/test |
| --- | --- | --- |
| 现场事件，且完成复核 | 是 | 是 |
| 同步视频逐段复核 | 是 | 是 |
| 有准确时钟记录的人工 timeline，且完成复核 | 是 | 是 |
| 旧文件名完整名称人工映射 | 仅作为历史引导数据 | 否 |
| 文件名子串自动匹配 | 否 | 否 |
| 根据预计时长推算边界 | 否 | 否 |
| 无证据、依靠记忆补标 | 否 | 否 |

现有 `reviewed_exact_action_table` 或 `reviewed_exact_action_compatibility` 来源，即使已经转换为 Schema v2，也应标为历史引导数据，不得进入新模型的正式验收集。仍为 Schema v1 的标签必须先进入待审核清单，不能训练或验收。

## 6. 常见错误

错误一：根据文件名写标签。

```text
文件名包含“跑步机” → 自动标成 run
```

正确做法：现场确认是走路还是跑步，并填写 `treadmill_walk` 或 `run`。

错误二：把佩戴取下标成非运动。

```json
{"motion_state": "non_motion", "wear_state": "valid", "window_trainable": true}
```

正确做法：

```json
{
  "activity_id": "removed_wear",
  "motion_state": null,
  "wear_state": "removed",
  "phase": "artifact",
  "window_trainable": false
}
```

错误三：把含休息的多组记录整段标成动作。

正确做法：优先一组一个 trial；如果使用一个短文件，必须通过现场事件或视频给每组和休息提供准确边界。超过 180 秒时只记录会话级 `weak_targets`。

## 7. 交付前检查

- 原始 CSV、事件 CSV、标签 JSON 和备注文件能够按 trial 一一对应；
- `activity_id` 与实际动作一致，且同时填写正确的 `motion_state`；
- 所有训练段均为 `wear_state=valid`；
- 准备、休息、转换和异常段均设为不可训练；
- 标签证据能够追溯到现场事件、同步视频或准确 timeline；
- 旧文件名映射标签未进入正式 validation/test；
- 标签通过 Schema v2 校验后，再进入训练目录。
