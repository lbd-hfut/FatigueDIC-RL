# 09 — Phase 1B fatigue evolution contracts

## Scope and physical units

二维小应变、平面应力、各向同性线弹性、局部积分点标量损伤、准静态比例循环。
`σ=(1-D)C0 ε`，`0<=D<Dmax<1`；应变为 `[exx,eyy,gamma_xy]` 工程剪切。
沿用 Phase 1A 单位：mm、N、N/mm²、N·mm；N 为循环数时单位明确标为 `cycle`。
不研究塑性、接触、大变形、不连续裂纹或一般非比例路径。

## Material history

`MaterialHistoryState` 存储 layout、damage FieldData `[G]`、累计损伤增量 `[G]`、
累计循环数 `[G]`、可选无量纲被动 history `[G,H]` 和名称，以及 quadrature IDs、
material region IDs、material state indices `[G]`、Dmax 和 committed/trial 标志。
`initialize()` 的累计损伤增量为零，N 可显式非零。
累计增量表示从该初值开始的 `D-D_initial`，不会误将初始损伤算成演化增量。
所有点共享一个物理循环数；不会用数据帧索引充当 N。

数值数组经现有 `array()` / `FieldData` 复制至不可写的 byte-backed storage，无法
`setflags(write=True)`。材料状态与力学 pattern 使用**同一 QuadratureLayout 对象**，
layout 与 pattern 使用同一 MeshData 对象，防止相同 mesh_id 掩盖几何/材料区域变化。
绝不按数组长度判定场位置。node/element damage 不能直接进入历史状态；已有显式
`layout.project_element_field()` 可由调用方选择使用，并保留投影 metadata。
积分点编号与区域来自 `quadrature_to_element`；只接受 identity state mapping。

`history.trial(new_damage, delta_N)` 返回新的 trial，不修改旧对象；`trial.commit()`
返回新 committed 对象。coupled solver 只有全部检查通过后才替换其 committed 指针。
rollback 是丢弃未提交 trial，保持原对象及循环数；不需要反向演化损伤。
可选被动 history 原样保留，不声称已经实现其他内部变量本构演化。
无跨网格状态转移接口/实现，本阶段直接拒绝绑定变化；后续若需要，另行定义并验证。
极大 exposure 在浮点上舍入到 Dmax 时也拒绝，不用截断绕过严格上界。

## Proportional cycles

`FatigueLoadCycle(N_start, delta_N, load_max, load_min, nominal_stress_ratio,
boundary_condition_refs, loading_mode, ...)` 的 max/min 是**一套有物理单位 BC pattern
的无量纲比例系数**，max>0，min<=max，R=min/max；loading_path 只接受 proportional。
边界引用必须是 pattern mesh 中的标签。load_id、ratio_definition 和标量 cycle metadata
单独保存，不以 metadata 驱动数值计算。

load_controlled：位移约束必须为零，力/牵引 pattern 必须非零，按比例缩放所有力。
displacement_controlled：非零位移 pattern 按比例缩放，禁止独立外力/牵引。
两种模式都保留相同约束 DOF；允许 R=-1 和 R=1（零振幅）。
位移控制下 R 是未损伤弹性参考的**名义**应力比，首先由位移比例定义；当前标量线弹性
同一 D 下的响应确实比例，不能将这一约定外推到塑性或非比例局部应力历史。
body force、constitutive history 和混合控制暂不支持，明确拒绝。

## Driver and damage law

`FatigueDrivingForce.evaluate(minimum, maximum)` 校验同一损伤、N、布局与 load ID，
返回 strain_amplitude、strain_mean、energy_amplitude、Ymax、Ymin，各场显式引用 layout。
首版 `EnergyAmplitudeDriver` 使用

\[
\varepsilon_a=(\varepsilon_{max}-\varepsilon_{min})/2,\quad
\varepsilon_m=(\varepsilon_{max}+\varepsilon_{min})/2,
\quad Y_a=\tfrac12\varepsilon_a^T C_0\varepsilon_a.
\]

支持的 scalar elastic backend 给出 C(D)，显式用 `C0=C(D)/(1-D)` 恢复未损伤 tangent；
应变必须携带 engineering shear metadata。纯剪切独立测试检验 gamma 和 tau 的配对。
Ymax/Ymin 分别为 `0.5 ε^T C0 ε`，与 `(1-D)` 退化后的自由能不同。
不使用 Ymax-Ymin 作振幅驱动力；完全反向时后者为零但 Ya 非零。

`FatigueDamageLaw.hazard(driving, cycle, history)` 是新循环上下文接口，区别于 Phase 0
仅接收单个 MechanicsState 的 `DamageLaw`；Phase 0 接口和原测试保持原样。
`SyntheticFatigueDamageLaw` 按 material region 读取 `FatigueLawParameters`：

\[
\lambda=A\max(Y_a/Y_{ref}-\eta,0)^m\exp(\beta_R R).
\]

A>=0 [cycle⁻¹]、Yref>0 [N/mm²]、eta>=0 [1]、m>0 [1]、beta_R [1] 有限可正可负，
其指数修正非负；默认 beta_R=0。log-domain 计算减小中间溢出，最终非有限值明确拒绝。
零振幅、阈值以下或 A=0 给出零 hazard。该模型是**合成可校准基准**，不是
Lemaitre/Chaboche 文献原式，不是已校准的真实材料模型。
`ConstantFatigueHazardLaw` 只用于数值验证，故意独立于振幅。

## Coupled advance and failure semantics

`FatigueEvolutionSolver(pattern, initial_state, damage_law, ...)` 接收 Tri3 backend、
driver、FatigueSolverConfig 和可选 ObservationMapping。
`advance(FatigueStepRequest(cycle))` 对一个请求做原子更新：

1. 读 committed D，分别求 min/max 平衡；不递增 N。
2. 由当前 strain amplitude 得到 Ya、hazard。
3. 调用原有 DamageTransition 冻结 hazard 的解析更新，得到候选 D。
4. 检查最大 ΔD；超限则拒绝或二分，重新求每个子步力学/hazard，不截断增量。
5. 构造 trial 历史；使用新 D 在 N_end 分别重新求 min/max 平衡。
6. 重新计算终点 Ya 和 endpoint hazard，并检查不可逆、有界、数值有限、能量与观测。
7. 所有内部子步通过后，一次性发布结果并替换 committed_state。

损伤变化限制不是嵌入式时间误差估计，不能代替步长收敛试验。
默认 ΔDmax=.03、二分最大深度8、minimum_cycle_increment=1e-8 cycle。
solver/material/nonfinite 错误不假装通过，不重试力学错误；整个请求 rollback。
`FatigueStepError.failures` 和 solver.failure_log 保留每次拒绝的 N、ΔN、阶段、深度、
异常类型/原因与可能的 MechanicsDiagnostics；成功细分也在 result.rejected_attempts 保留。
上一个完整请求已经成功提交时，随后请求失败不会撤销该已提交请求。
所有子步记录只随完整请求成功返回，失败不产生可保存的部分 StepResult。

冻结 hazard 更新对步内常 hazard 精确；非线性力学—损伤耦合整体是一阶时间离散。
终点 response 来自新 D 的实际重求解，不能把 start response 当成终点。

## Energy diagnostics

每个相位/起终点记录 U、`0.5 u^T K u`、`u^T f`、`u^T reaction`、
`2U-u^T f-u^T reaction`、力/力矩平衡、自由 DOF 残差。
work 产品是静态代数恒等式，不自动等于历程累计外力功。
每个相位的损伤耗散估计为 `Σ t*w_phys*Y_phase(start)*ΔD>=0`，量纲 N·mm。
记录固定相位随 D 变化的梯形机械功 `0.5*(f+R)_start/end · Δu` 和
`work_estimate-ΔU-dissipation_estimate`；该余额有离散误差，不强行置零。
循环内加载/卸载功、塑性滞回和全部疲劳损耗不在本模型内。
载荷控制下 U 可增大，位移控制下 U 可减小，均不违背非负损伤耗散。

## Trajectory storage and observation isolation

`run_trajectory(solver, cycles, random_seed=..., code_version=...)` 先检查完整载荷历史连续，
保存初始状态与每个真实接受的子步；物理 N 单独存储。
`FatigueTrajectory.save(new_directory)` 暂存完整文件后以目录 rename 发布；
目标已存在则拒绝，写出失败则删除本次临时目录，不动旧成果。

`truth/trajectory.npz`：N、D、历史、u/strain/stress 的 min/max、终点 Ya/hazard、
步内冻结 Ya/hazard、ΔD、reaction/applied force/residual、elastic energy、完整网格/积分信息。
`truth/diagnostics.json`：每步能量/耗散/平衡、solver diagnostics、拒绝/细分记录。
`truth/metadata.json`：显式单位、布局与状态含义。
`provenance.json`：材料、BC、载荷历史、初值、配置、种子、code version。
科学 CLI additionally 保存配置原文/hash、Python/NumPy/SciPy 版本、Git HEAD 与实际源文件 SHA256。
Git HEAD 不是未提交代码的唯一版本标识。

`save_observations(tuple[DICObservation], directory)` 是独立接口，仅接受位移观测。
`observations/displacement.npz` 只有 N、坐标、位移、mask、confidence。
`load_observations(directory)` 只读取该 observation 目录，不访问 truth。
FE-to-DIC 是合成位移采样；没有实际图像相关。
接口隔离减少标签泄漏机会，但不宣称操作系统权限隔离或已验证训练流程。
