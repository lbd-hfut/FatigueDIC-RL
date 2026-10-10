# 08 — Phase 1A Tri3 力学契约

本文件补充 Phase 0 历史文档，不改写当时的验收结论。

## 入口与单位

`Tri3MechanicsSolver.solve(request)` 实现已有 MechanicsSolver。
新请求推荐使用 `Tri3SolverRequest.from_mm(..., thickness_mm=..., damage=...)`。
沿用旧 SolverRequest 时，厚度字段仍是 `thickness_m`，backend 显式乘 1000 转换为 mm。
mesh 必须为 mm、二维、Tri3；不会猜测或转换未声明的坐标单位。

`IsotropicPlaneStressMM` 接受 `young_modulus_n_per_mm2` 与 poisson_ratio，
通过旧 Pa 模型显式换算，再返回 N/mm^2。旧 IsotropicPlaneStress 的 `young_modulus_pa`
保持原含义；backend 将 Pa tangent 乘 1e-6。TractionBC 新增 N/mm^2 选项，旧 Pa 转换同理。
不把单位不同的数值直接装入同一个矩阵。常规单位为长度/位移/厚度 mm、stress/E N/mm^2、
体力密度 N/mm^3、节点力/反力 N、刚度 N/mm、能量 N*mm、strain/damage 无量纲。

## 张量与位置

strain 的 Voigt 次序严格为 `[exx,eyy,gamma_xy]`，gamma_xy 为工程剪切。
stress 为 `[sxx,syy,tau_xy]`，不对 tau_xy 额外乘 2。
能量密度为 `0.5*(exx*sxx+eyy*syy+gamma_xy*tau_xy)`。
strain tensor Frobenius 误差使用 `exx^2+eyy^2+0.5*gamma_xy^2`。
输出 metadata 标明分量和剪切约定；MM 本构拒绝显式声明 tensor shear 的输入。

state.displacement：node `[V,2]` mm。
state.strain/stress：quadrature `[G,3]`，单位 1 / N/mm^2。
result.element_strain/stress：显式积分加权投影的 element `[E,3]`。
reaction_forces、applied_forces、equilibrium_residual：node `[V,2]` N。
element_energy：element `[E]` N*mm，elastic_strain_energy 为其和。
equilibrium_residual 保留完整 `K*u-f`，包含约束 DOF 的反力；
reaction_forces 只在约束 DOF 存储反力，自由 DOF 置零。
constrained_dofs 明确采用 `[u0,v0,u1,v1,...]` 的编号方式。

## QuadratureLayout

材料积分布局保存每个点的 reference_points、reference_weights、物理坐标、Jacobian、
signed determinant、quadrature_to_element、element_to_quadrature 和 material_state_indices。
参考三角形面积为 1/2；物理权重为 reference_weight × abs(det J)，不含厚度。
默认 centroid 是 `(1/3,1/3)`、权重 1/2，每单元一个点；当前 G=E 是规则选择，
不是空间身份相同。布局 reference 包括 mesh、rule 和内容哈希。

支持通过完整显式布局提供多点正权规则；G≠E 的独立测试实际进入装配。
本轮 material_state_indices 使用 identity storage；非 identity 表会明确拒绝，
留待下一阶段设计跨布局历史状态存储。积分内变量仍是有位置/单位的 FieldData。
当前线弹性本构无历史变量，明确拒绝非空历史，不能据此声称弹塑性已实现。

damage=None 时，from_mm 请求建立该布局上的零损伤。
quadrature damage 必须引用完全一致的布局。element damage 通过显式命名的
`project_element_field` 作分片常值 lift，projection 记录在 metadata；node damage 被拒绝，
必须由调用方选择并验证映射。输出 state.damage 保留输入字段、位置、值和 cycle_count。
固定损伤要求 `D<=config.damage_max<1`，不会调用 DamageLaw 或 DamageTransition。

## 载荷、约束和错误

牵引按每条外边界线段一致分配 `t*L*traction/2` 给两个节点。
BodyForceDensity 是常向量或纯坐标函数，`evaluate([Q,2])->[Q,2]`，单位 N/mm^3；
采用独立的 Duffy Gaussian 载荷积分，不改变材料积分布局。
默认载荷积分为 5×5 点，制造解误差积分为独立的 7×7 点。
载荷规则最低阶数为 2，保证常量/线性体力的一致节点载荷不被一阶 Duffy 欠积分破坏。
重复节点力、重复/重叠牵引线段、重复位移 DOF 均拒绝；有意叠加须由调用方先明确合并。

Dirichlet 使用 `K_ff*u_f=f_f-K_fc*u_c`，保留原 K/f 计算约束反力。
每个全边连通分量验证约束后的两个平移和一个旋转模态秩。
不同域必须分别充分约束；单点铰接拓扑和孤立节点明确拒绝。
稀疏 LU 奇异错误、过小相对 pivot 或超限自由 DOF 残差返回显式失败，无人工稳定刚度。

诊断 residual_norm 是自由 DOF 无穷范数，单位 N；相对值除以内部/外部节点力的最大绝对值。
几乎零载荷/刚体算例应优先参考绝对残差；相对值在此时不稳定。
约束预检查失败时 iterations=0、残差尚未计算，旧契约中的 residual_norm=0 是占位，
不是成功收敛证据。异常中 converged=False 和 message 明确标识失败原因。
LU pivot ratio 是数值诊断，不是严格的条件数估计；patch 验证另行计算实际 reduced cond_2。
result.assembly 暴露 CSR 矩阵与单元矩阵用于科学核验，不是长期可变状态存储。

## 扩展与限制

仅支持 named isotropic linear plane-stress models，不支持任意非线性 tangent、plane strain、
quad4、高阶单元、3D、接触、变厚度或疲劳更新。正定性来自正本构、非退化相容网格与充分约束；
全域重叠/T junction 等完整网格质量检查仍需后续补充。
配置、原始 fields/matrices、误差表与图的证据见
[Phase 1A 验证报告](PHASE1A_MECHANICS_VERIFICATION_REPORT.md)。
