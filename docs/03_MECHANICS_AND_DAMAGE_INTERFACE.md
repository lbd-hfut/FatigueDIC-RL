# 03 — 本构、力学和损伤接口

## 本构与材料分区

`ConstitutiveModel.evaluate(strain, damage, parameters, internal_variables=...)` 独立于 solver 与网络。
Phase 0 示例 `IsotropicPlaneStress` 在同一支撑上计算 stress 与 tangent。
参数 E 使用 Pa 且 E>0，-1<nu<0.5；应变采用工程剪切约定。
其 tangent `[n,3,3]` 与 stress `[n,3]` 均使用 Pa。
\[
C_0=\frac{E}{1-\nu^2}
\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&(1-\nu)/2\end{bmatrix},\quad
C(D)=(1-D)C_0.
\]
本轮提供解析张量/能量正值验证，没有材料实验校准。
通用 ConstitutiveResponse 保存 stress、tangent、tangent_units 与更新后的 local internal_variables。
tangent 使用 `[n,c,c]`，c 为模型定义的扁平张量分量数；张量顺序由模型明确声明。
历史变量与 stress 显式同支撑，纯弹性示例明确拒绝非空历史变量，而不忽略它们。
更丰富模型可以使用自己的 response/context 扩展；时间增量、非局部相场等仍需独立设计，
不能仅凭这个接口声称弹塑性/相场已实现。

MaterialAssignment 将每个 material-region ID 映射到模型与参数。
SolverRequest 必须覆盖所有网格区域，backend 决定如何读取区域刚度、
积分点值与内部变量，不允许静默把不同位置的 damage 当作相同数组。

## MechanicsSolver

调用 `solve(SolverRequest)`，adapter 实现 `_solve`。
request 包含 mesh、damage、按区域的 materials、loading_state、cycle_count、
displacement_bcs、nodal_force_bcs、traction_bcs、thickness_m 和 internal_variables。
所有输入契约在 request 构造时验证，公共 solve 验证 backend 返回结果。

DisplacementBC 指定逐 DOF 的 node_ids/components/values，单位与 mesh 长度相同。
NodalForceBC 为 `[B,2]` 的 N；同节点多个载荷按 backend 文档明确求和。
TractionBC 为外边界标签上的常值 `[2]` 向量，单位 Pa，不是直接的边界节点力。
thickness_m 严格为正且用 m，2D backend 必须显式转换网格长度并处理厚度。
当前没有体力、接触、周期边界或面内变厚；需要后续明确扩展。

result 包含 `MechanicsState(role=PREDICTION)` 与 SolverDiagnostics。
成功响应必须具有节点位移，以及显式支撑的 strain/stress。
检查返回 mesh/cycle/load 一致、damage 未被 equilibrium solver 改写、
displacement 的节点支撑与单位正确、strain/stress 支撑和分量形状一致。
backend 还需承担相容性、平衡、边界满足、积分布局和本构历史一致性验证。

ContractError：数据/物理输入或输出契约不合法。
MechanicsConvergenceError：携带未收敛的 SolverDiagnostics。
MechanicsBackendError：封装线性代数或浮点数值异常并保留异常原因。
backend 不支持的单元/材料/BC 应抛出明确的 NotImplementedError，不能退回假结果。
未约束刚体模态、奇异刚度、病态矩阵应明确报告；残差单位与容差由 backend 定义。
当前所有 solver 测试均为契约 double，没有求解有限元方程。

D=1 可使示例刚度完全为零，可能导致全局刚度奇异。
Dmax<1 可以是数值试验设定，但不是自动有效的物理失效模型；
任何残余刚度/失效删单元机制必须作为材料/solver 的显式选择。

## DamageLaw 与 Transition

`DamageLaw.hazard(state, material_parameters, CycleLoading) -> FieldData`。
统一返回与 damage 完全同支撑的有限、非负、scalar hazard，单位 cycle^-1。
registry 为每个调用方独立的实例，拒绝重名和未知模型；没有 import-time 全局副作用。
`ConstantHazardLaw` 只读取 `hazard_per_cycle >= 0`，并广播到显式支撑。
它演示接受状态/参数/加载的接口，忽略加载幅值和内部历史，不是材料疲劳定律。
更复杂模型必须说明驱动力、参数单位、加载相位和校准范围。

`DamageTransition(dmax).apply(damage, hazard, delta_cycles)` 返回新的 damage 字段。
非负、非恢复、界限等由 DamageAdmissibility 硬检查，输入不被裁剪修复。
实际公式和冻结 hazard 的限定见 [00](00_SCIENTIFIC_FORMULATION.md)。
不重求解力学、不推进所有内部变量、不暗改 cycle_count。
smoke 的 orchestration 只在已知无其他内部变量的例子中显式更新循环数。

## 未来纯物理循环

1. 在当前 damage 下求解平衡并保留响应对应的加载相位。
2. 用状态、参数与下一循环区间计算物理 hazard。
3. 使用解析 transition 更新 damage；验证硬约束。
4. 在新 damage 下重新求解，并按本构规则推进历史变量。
5. 对指定载荷相位投影至 DIC 点，计算位移残差。

如果驱动力随 damage/应力迅速变化，缩小或自适应选择循环步长，并比较细步长结果。
该流程的 FE backend、应力驱动疲劳定律与能量收支尚未在 Phase 0 实现。
