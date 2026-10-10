# Phase 0 — 验收范围与矩阵

## 资格定义

`PHASE0_ARCHITECTURE_QUALIFIED`：A–H 核心模块/接口落实，所需科学文档齐全，
核心依赖保持轻量，规定的解析 smoke 与科学约束测试真实通过，无现有资产破坏。
它只证明架构与已实现模块符合本阶段要求，不证明 FE solver、疲劳材料模型、
damage 反演、RUL 或 RL 已经科学验证。

`PHASE0_ARCHITECTURE_PARTIALLY_QUALIFIED`：存在明确缺项或验证限制，仍有可运行基础。
`PHASE0_ARCHITECTURE_FAILED`：基础模块不能运行、关键约束失败或结果不可核验。
资格必须随实际测试变化，不得仅因文档完整而认定通过。

## 模块验收

| 必需项 | 实现 | 主要证据 |
|---|---|---|
| A MeshData | tri3/convex quad4/mixed、材料区域、边界标签、验证 | test_mesh.py |
| B FieldData | 四种 location、空间引用、units、mask | test_fields_and_state.py |
| C MechanicsState | damage/response/internal fields/cycle/load/role | test_fields_and_state.py |
| D DICObservation | 观测坐标/位移/质量/协方差，无 truth 通道 | test_dic.py |
| E DamageLaw | ABC、显式 registry、常 hazard 示例 | test_damage.py |
| F DamageTransition | 冻结 hazard 解析更新、硬约束、future correction | test_damage.py、test_analytic_invariants.py、test_correction_and_inference.py |
| G ObservationOperator | 不规则点、tri3 形函数、外点 mask、缓存 | test_observation_operator.py、test_analytic_invariants.py |
| H MechanicsSolver | 分区材料、damage、BC、response、错误契约 | test_solver_contract.py（契约 double） |

## 用户要求的十项验证

| 要求 | 测试依据 |
|---|---|
| 1. 连接关系合法 | test_invalid_connectivity、test_valid_connectivity_and_boundary |
| 2. shape validation | field/DIC/mesh/state/request 的非法形状测试 |
| 3. 位置与单位不隐混 | test_support_and_unit_mixing_rejected、test_bad_units_frames_shapes_and_same_count_reference |
| 4. 不规则点插值 | test_irregular_affine_interpolation、跨尺度/朝向解析测试 |
| 5. 网格外 mask | test_outside_mask_and_no_extrapolation |
| 6. D 不降低 | test_hazard_bound_irreversibility_over_wide_scales |
| 7. D 不超过 Dmax | 同上、test_saturated_damage_is_fixed_point |
| 8. 零驱动不变 | test_zero_hazard_and_zero_step_exact |
| 9. 输入相同确定性 | test_transition_deterministic、test_cache_determinism_and_immutability、test_analytic_module_smoke |
| 10. 非法参数/步长拒绝 | law/transition/constitutive/cycle/request 参数化测试 |

## 集成与依赖

test_phase0_smoke.py 验证 V=4、E=2、Q=8、六个有效点和两个外点。
解析误差容限固定为机器 epsilon 倍数，没有放宽物理约束来通过测试。
CLI 集成测试验证配置加载与输出，并在新进程确认 physics-only 路径不加载 torch、scipy 或 rl。
shape/support 数量测试独立覆盖 V=4、E=2、G=6、Q=9，未假设 V=G=Q。
wheel 构建/安装和非仓库目录导入的真实记录见实施报告。

## 科学文档与禁止事项

00–07、本验收文件与 PHASE0_IMPLEMENTATION_REPORT 必須存在。
文档区分真值/估计、hazard/rate、硬约束/软损失、belief update/evolution、
Critic/RUL predictor、监督/RL，并列出 RL 尚需验证的原因。
本轮不安装重型运行依赖、不进行 GPU/网络训练、不生成疲劳数据集，
不把 FE 契约 double 或解析 smoke 描述为 FE 验证。
不删除原始文件、不 push、不 merge、不 release。

最终资格、测试计数、命令输出与未解决问题，以
[实施报告](PHASE0_IMPLEMENTATION_REPORT.md) 的实际证据为准。
