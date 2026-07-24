# 耳机 IMU 数据检查与动作分析

本目录包含质量检查、清洗、特征分析和逻辑回归训练脚本，也可用 `run_pipeline.py` 一次完成：

```powershell
python imu_analysis/run_pipeline.py "data" --work-dir "imu_output/all_data"
```

若系统 `python` 不可用，请换成实际 Python 可执行文件。仅依赖 `numpy` 和 `pandas`。

权威标签为 `label_schema.py` 定义的 Schema v2。每个可训练段同时具有
`motion_state` 和 `activity_id`；旧逻辑回归管线只作为回滚保留，读取到
Schema v2 特征时优先使用 `motion_state`，不再重新按中文名称猜标签。

多动作 Demo 使用独立的新管线：

- `label_schema.py`：双层标签 schema、完整动作名称审核表、佩戴规则和旧格式兼容。
- `activity_taxonomy.py`：运行时类别名称及运动/非运动类别集合。
- `activity_features.py`：训练/实时共用的 50 Hz、4 秒窗口特征合同。
- `train_activity_model.py`：按用户切分、LightGBM 训练、温度校准和验收报告。
- `activity_runtime.py`：只负责流式概率输出。
- `workout_strategy.py`：组内/组间、组数、动作切换和 10 分钟训练段规则。
- `workout_store.py`：SQLite 事件审计和前端日汇总。
- `ble_protocol.py`：14 字节精简六轴样本及批量 CRC16 协议。

多动作训练只读取同目录 `labels/<CSV文件名>.labels.json` 中经过校验的
Schema v2 段。未知完整名称、佩戴异常、污染短记录和所有超过 180 秒的
`session_weak` 记录均不产生训练窗口。长会话的 `weak_targets` 只用于动作
覆盖、顺序、总组数和次数等会话级验证。旧 timeline 只兼容 180 秒以内的
人工短记录，不能为长记录制造逐窗标签。

## 输出

- `reports/quality/quality_report.md`：数据量、时长、采样率、缺失、零值、重复和候选异常概览。
- `reports/quality/quality_files.csv`：每次采集一行的质量信息。
- `reports/quality/quality_channels.csv`：每个通道、每次采集的详细质量统计。
- `processed/cleaned/<采集名>/data.csv`：清洗后的副本，原始数据不会被覆盖。
- `processed/cleaned/cleaning_log.csv`：每通道修复数量。
- `reports/features/window_features.csv`：2 秒、50% 重叠窗口的特征，可直接用于建模。
- `reports/features/activity_feature_means.csv`：各动作特征均值。
- `reports/features/state_effect_sizes.csv`：运动/非运动标准化差异。
- `reports/features/action_pairwise_distances.csv`：各动作两两之间的标准化距离。
- `reports/features/action_loro_predictions.csv`：按整段留一的动作初步验证明细。
- `reports/features/analysis_report.md`：动作差异、初步验证和策略建议。
- `model/model_report.md`：L2 正则化逻辑回归的嵌套分组验证结果。
- `model/l2_logistic_model.pkl`：全量训练后的模型参数、特征名和阈值。
- `model/label_audit.csv`：输入标签与当前动作分类表的逐动作差异。

## 清洗原则

1. 只把两侧同号且数值接近的单点零值视为掉点，避免误删量化零值和真实零交叉。
2. 使用保守的 Hampel 滚动中位数/MAD 加相邻点连续性条件标记孤立尖峰，不按固定幅值粗暴裁剪运动峰值。
3. 只插值不超过约 0.25 秒的短缺口，长时间掉数继续保留为空。
4. 默认平滑窗约 0.06 秒，以减少噪声但保留人体动作的主要频率成分。
5. 只对原始加速度/角速度做上述清洗；不独立平滑四元数，也不把冻结的欧拉角“修”成伪数据。
6. 始终另存清洗数据，并在日志里保留修复计数。
