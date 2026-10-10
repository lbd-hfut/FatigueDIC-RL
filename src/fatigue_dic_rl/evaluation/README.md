# Evaluation — 科学验证与后续评价边界

Phase 1A 的 tri3_meshes/tri3_verification/tri3_plots 提供有解析依据的 FE 科学验证、
数据导出和静态收敛图。四类配置与 verify_phase1a.py 可以复现；没有真实图像相关或疲劳演化。
通用 dataset truth 评价器仍未实现。

未来评价模块可以读取独立 synthetic truth，用于 damage 误差与可识别性研究；
正式策略与 state estimator 不消费该通道。真实实验以位移/反力等独立证据评价，
RUL 需单独的失效定义和校准。当前科学证据位于 tests/scientific，
没有未经验证的疲劳指标。见 [验收范围](../../../docs/PHASE0_ACCEPTANCE.md)。
