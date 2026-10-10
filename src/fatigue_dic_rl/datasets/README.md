# Datasets — 尚未实现

未来 loader 的正式观测通道返回 DICObservation；SyntheticGroundTruth 通道另行提供
role=GROUND_TRUTH 的 MechanicsState，只由受控 synthetic 数据制作与评价读取。
不得把 truth/未来观测/RUL 标签封装进观测 metadata。按 specimen/run 分割，记录 provenance。
本轮没有生成疲劳 dataset 或实现 truth loader。见 [验证计划](../../../docs/06_TRAINING_AND_VALIDATION_PLAN.md)。
