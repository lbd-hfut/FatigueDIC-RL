# Phase 1B — Coupled Fatigue Damage Evolution

实施日期：2026-10-08。最终资格：**PHASE1B_FATIGUE_EVOLUTION_QUALIFIED**。

资格限定于本报告实际验证的二维小应变、平面应力、比例循环、局部标量损伤及
physics-only 合成基准模型。**不包含真实材料标定、网格客观局部化、裂纹几何、
完整疲劳每周期耗散、真实图像相关、damage 反演、RUL 或任何学习模型的科学资格。**
局部化的网格敏感性已量化并保留，不通过平滑掩盖；本轮停止于 Phase 1B。

## A. Git baseline and code changes

实际检查的 branch 为 `feature/phase0-architecture`，HEAD 为
`dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9`，只有当前一个 worktree。
Phase 0 和 Phase 1A 仍是 README 修改及 `.github/configs/docs/results/scripts/src/tests`
等未跟踪资产，没有先前阶段的完整 Git commit。没有 reset、clean、restore、commit、
push、merge、rebase 或分支切换。

首先读取 Phase 0 实施/验收报告、Phase 1A 验证报告、08 契约以及全部已有测试，
实际审计 MeshData、FieldData、MechanicsState、QuadratureLayout、Tri3MechanicsSolver、
ConstitutiveModel、DamageLaw、DamageTransition、SolverResult、DICObservation 和 FE-to-DIC。
开发前 `python -m pytest -q -W error`：**165 passed in 1.93s，exit 0**。

开发前将 **240 个文件** 的完整内容保存至
[可恢复快照](../results/phase1b/baseline/phase01a_worktree.tar.gz)，并逐文件检查 tar 内容的 SHA256。
同时保存 [manifest](../results/phase1b/baseline/manifest.json)、
[tracked.patch](../results/phase1b/baseline/tracked.patch)、
[初始 Git 状态](../results/phase1b/baseline/git_status.txt) 和
[基线 pytest](../results/phase1b/baseline/pytest.txt)。快照涵盖旧代码、未提交配置/文档/测试及旧科学结果。
若要人工恢复，应先把 tar 解至独立 `/tmp` 目录，再按 manifest 选择文件核对；本轮未执行恢复。
这不依赖新 Git checkpoint，因此没有请求或创建 checkpoint commit。

新增 `fatigue/{history,loading,models,solver,trajectory}.py` 与公开 package 入口，
`evaluation/fatigue_verification.py`、`fatigue_plots.py`、科学 CLI、8 个 TOML 配置、
独立单元/耦合/科学/CLI 测试、09 契约和本报告。旧文件只增量修改 README 和 CI；
原 Phase 0/1A 数值实现、测试、报告和科学资产逐文件保持原内容。
完整相对快照的改动、原资产保留检查和文本 patch 见
[delivery_audit.json](../results/phase1b/delivery_audit.json)、
[phase1b_delta.patch](../results/phase1b/phase1b_delta.patch)、
[最终 Git 状态](../results/phase1b/git_status.txt)。

最终科学运行的 code version 同时包括 HEAD 和未提交源码 SHA256：
`dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9+worktree-source-sha256:c9df1139de54488d2a874f4cbdf251dd12da38a7c6ee6ff8ffb69146cd7f2b24`。每例 summary 保留全部源文件哈希和配置哈希，实际核验与当前源码一致。

## B. History-state architecture

`MaterialHistoryState` 明确绑定一个 QuadratureLayout：D `[G]`、累计增量 `[G]`、
累计 N `[G]`、被动可选 history `[G,H]`、history_names、材料区域 ID、积分点 ID、
material state indices 和 committed/trial 状态。初始累计增量为零；演化后等于 `D-D_initial`。

历史与 mechanics pattern 共用**同一个 layout 对象及 mesh 对象**，避免同长度或同 mesh_id
掩盖材料区域/几何变化。只接受 identity state mapping；非 identity、node/element 输入、
错误引用、错区域 ID、错积分点顺序、不同布局和单位均拒绝，不做隐式投影。
多点独立耦合测试实际覆盖 `E=1,G=3`；分区定律测试实际检验不同区域 hazard。

数组复制到不可写 byte-backed storage，不能意外原地修改或重新开启 write flag。
trial 构造新状态，commit 只在全部子步终点求解/物理检查通过后替换 solver 的 committed 指针。
任一子步失败，整个外部请求回滚，保持旧历史对象、D、累计增量和 N。
H>0 仅支持有名称的无量纲被动变量保留，没有假装实现塑性等额外历史演化。
严格要求 `0<=D<Dmax<1`；浮点饱和到 Dmax 也失败，不截断掩盖。
未实现或声称验证跨网格状态转移。

## C. Load-cycle representation

`FatigueLoadCycle` 保存 N_start、delta_N、load_max/min、nominal_stress_ratio、
boundary_condition_refs、loading_mode、load_id、路径/幅值/比例约定及只含标量的 cycle metadata。
max/min 是有物理单位的 BC pattern 的无量纲系数；max>0、min<=max，显式检查 R=min/max。

载荷控制只允许零位移约束，非零力/牵引 pattern 按同一个系数缩放。
位移控制缩放非零位移 pattern，禁止独立力/牵引；R 是比例弹性参考的名义应力比。
在同一冻结 D 下，min/max 都单独调用实际 Tri3 solver，不根据端点能量推测振幅。
body force、混合控制和一般非比例多轴路径暂时明确拒绝。

delta_N 必须正、有限、可表示，N_start 与 committed N 连续；重复请求不会重复施加增量。
物理循环数与保存帧数独立；本次短轨迹 0–40 cycle，保存9帧，不是9个物理循环。

## D. Fatigue driving-force formulation

自由能 `psi=0.5*(1-D)*epsilon^T*C0*epsilon`，Y=`0.5*epsilon^T*C0*epsilon>=0`。
`EnergyAmplitudeDriver` 返回 epsilon_a、epsilon_m、Ya、Ymax、Ymin，全部为显式 quadrature fields：

\[
\varepsilon_a=(\varepsilon_{max}-\varepsilon_{min})/2,\quad
\varepsilon_m=(\varepsilon_{max}+\varepsilon_{min})/2,\quad
Y_a=\tfrac12\varepsilon_a^T C_0\varepsilon_a.
\]

支持的 scalar elastic backend 用 `C0=C(D)/(1-D)` 得到未退化 tangent；driver 检查工程剪切 metadata。
纯剪切独立测试检查 `Ya=0.5*G*gamma_a^2`，不混淆 tensor epsilon_xy 与 gamma_xy。
Ya 和 Y 的单位为 N/mm²，即能量密度；Ya 是当前模型输入，不是完整材料疲劳耗散。

Test B：R=1、相同 max/min，无论载荷还是位移控制，epsilon_a 和 Ya 精确为零，D 不增长。
Test C：R=-1、方向相反，端点 Ymax-Ymin 为零，但初始
Ya=0.0277008310 N/mm²，D 严格增长。该回归明确防止用端点能量差替代振幅。
证据：[reversed summary](../results/phase1b/final/reversed_loading/summary.json)。

## E. Damage-law assumptions and parameter units

采用**可校准的合成基准模型**：

\[
\lambda_{phys}=A\langle Y_a/Y_{ref}-\eta\rangle_+^m\exp(\beta_R R).
\]

| 参数 | 量纲与限制 | 本次配置 |
|---|---|---:|
| A | cycle⁻¹，有限非负 | 0.002 |
| Yref | N/mm²，有限正值 | 0.025 |
| eta | 1，有限非负 | 0 |
| m | 1，有限正值 | 1 |
| beta_R | 1，有限可正可负；指数修正非负 | 0 |
| Dmax | 1，0<Dmax<1 | 0.95 |

初始基准参数只是可复现试验数值，没有真实材料校准。
不是标准 Lemaitre、Chaboche 或其他文献原式，不作实验验证声明。
阈值/零驱动力/A=0 给出零 hazard；输出必须有限非负且匹配 layout。
log-domain 计算降低中间溢出；非有限 hazard 明确报错并回滚。
分区 parameters 以材料区域 ID 映射，错误 ID 和缺项拒绝。
新的 `FatigueDamageLaw` 显式接收完整循环驱动力上下文，旧 Phase 0 DamageLaw 接口未改写。
未来具体文献模型可替换该接口，但需独立核对公式与验证，本轮不建立复杂模型集合。

## F. Mechanics–damage coupling algorithm

实现一阶显式 staggered：committed D → 两端点 FE → amplitude/Ya → hazard →
原 DamageTransition → trial D → 两端点 FE **重求解** → 新 Ya/endpoint hazard →
约束、能量与观测检查 → 原子 commit。

\[
D_{k+1}=D_{max}-(D_{max}-D_k)\exp(-\Delta N_k\lambda_k).
\]

该更新**仅对步内冻结 hazard 精确**；随着 D 变化而变化的耦合响应不是此解析解的精确情形。
终点位移/应变/应力来自新 D，独立测试逐数组对照直接终点 FE 求解。
新损伤增量只施加一次；每次 FE 调用只标注 N，不累积 N。

默认 ΔD 上限0.03、最大二分深度8、最小 delta_N=1e-8 cycle。
科学时间精度试验采用上限0.2，实际没有触发细分，确保比较的是规定 ΔN；
synthetic/mesh 配置上限0.03。上限超出则显式拒绝或二分重新计算，不截断 ΔD。
二分是损伤变化控制，**不是时间误差估计器**；独立步长验证仍不可替代。
非有限状态、材料非法、求解失败与小于最小步长均拒绝；不静默重试力学失败。

Test G 保存五类预期失败：超限步、E<0、真实欠约束刚体模式、hazard 溢出、
已有内部子步成功后第二半步的注入终点求解失败。
五类均验证同一个 committed 历史对象保持不变，N/D/累计增量不变，没有部分 StepResult/checkpoint。
独立测试还覆盖 NaN/Inf/负 hazard、非法/过小 delta_N、布局绑定变更、二分深度耗尽与
存储写出失败后的临时目录清理。
失败阶段、类型、原因、N、delta_N、深度及 backend diagnostics 保存在
[rollback summary](../results/phase1b/final/rollback/summary.json) 和 solver.log。
正常细分的拒绝记录也保留在 StepResult；预期失败不伪装成正常力学收敛。

## G. Constant-hazard verification — Test A

冻结 lambda=0.001 cycle⁻¹，D0=0.05，Dmax=0.95，N=40。
完整 FE 耦合在 ΔN=10、5、2.5、1.25 下与独立指数解析式比较。
final D=0.085289504763；最大绝对误差均小于 9.714e-17。
D 有界且不可逆；零 hazard 精确保留 D；循环细分满足半群一致性。

[constant-hazard summary](../results/phase1b/final/constant_hazard/summary.json)
保存四组误差和全部场；单元测试另外用较紧 ΔD 上限强制二分，并对照整步解析解。
非线性 hazard 二分测试与显式执行同一组真实子步逐数组一致，且 hazard 随 D 重新计算。

## H. Load/displacement-controlled verification — Test D

均匀 2×1 mm 矩形，4 个 Tri3，E=2000 N/mm²、nu=0.3、t=0.8 mm、D0=0.05。
R=-1；载荷控制 sigma_max=10 N/mm²，位移控制右端 u_max=0.01 mm，横向自由收缩。
对每个终点独立检查均匀 D、Eeff=(1-D)E、轴向位移和反力。

均匀位移控制 epsilon_a 恒定，hazard 真正恒定，解析 D(N) 为指数式。
均匀载荷控制 lambda=k/(1-D)²，令 a=1-Dmax、s=Dmax-D、s0=Dmax-D0，独立积分为：

\[
kN=a^2\log(s_0/s)+2a(s_0-s)+(s_0^2-s^2)/2.
\]

用标量 brentq 解该单调公式作参考，不引用 staggered 更新作标准答案。
载荷控制精确 final D=0.133220241067；位移控制精确 final D=0.119195288252。

| control | ΔN (cycle) | final mean D | analytic D error | right u (mm) | left reaction (N) | elastic U (N·mm) |
|---|---:|---:|---:|---:|---:|---:|
| load | 10 | 0.1313052220 | 1.915019e-03 | 0.0115115231 | -8.0000000000 | 0.0460460924 |
| load | 5 | 0.1322436342 | 9.766069e-04 | 0.0115239719 | -8.0000000000 | 0.0460958877 |
| load | 2.5 | 0.1327269860 | 4.932550e-04 | 0.0115303945 | -8.0000000000 | 0.0461215780 |
| load | 1.25 | 0.1329723524 | 2.478886e-04 | 0.0115336576 | -8.0000000000 | 0.0461346303 |
| displacement | 10 | 0.1191952883 | 5.551115e-17 | 0.0100000000 | -7.0464376940 | 0.0352321885 |
| displacement | 5 | 0.1191952883 | 4.163336e-17 | 0.0100000000 | -7.0464376940 | 0.0352321885 |
| displacement | 2.5 | 0.1191952883 | 4.163336e-17 | 0.0100000000 | -7.0464376940 | 0.0352321885 |
| displacement | 1.25 | 0.1191952883 | 2.775558e-17 | 0.0100000000 | -7.0464376940 | 0.0352321885 |

load-control 最后时间阶0.992642；displacement-control 解析误差仅舍入量级。
载荷控制外力8 N保持固定，位移增加、弹性能增加；位移控制位移固定，反力和弹性能降低。
[完整表与解析检查](../results/phase1b/final/uniform_damage/summary.json)。

![Control energy comparison](../results/phase1b/final/uniform_damage/control_energy.png)

## I. Temporal convergence

固定同一非均匀 16-cell 网格和同一积分布局，不作跨网格插值。
比较 ΔN、ΔN/2、ΔN/4、ΔN/8，并额外采用 ΔN/32 的更细 staggered 解。
高精度参考使用独立 DOP853 对**同一 FE 空间离散的连续 ODE**积分，
每次 RHS 实际解 min/max 力学并计算 hazard，不调用 DamageTransition 或 advance。
同时积分 `Σ t*w_phys*Ymax*dD/dN`，作为最大相位损伤耗散诊断的参考。

参考 rtol=1e-10、atol=1e-12；再收紧至2e-12/2e-14，damage RMS差
6.684e-14、累计耗散差 1.840e-13 N·mm；分别62/74次 RHS。
这是独立**时间积分算法**核对，仍共享本构/FE/driver 定义，不证明真实材料模型正确。

Damage RMS=`sqrt(Σw*(D-Dref)^2/Σw)`；相应积分 L2 为 RMS×sqrt(面积)，
面积2 mm²，故最后四级/更细 RMS 序列与 L2 具有同一收敛阶。
从保存的空间场独立重算的 final damage L2 误差依次为
3.195532e-03, 1.634110e-03, 8.265317e-04, 4.156866e-04, 1.043813e-04 mm；
[temporal_l2_error_table.csv](../results/phase1b/temporal_l2_error_table.csv)
同时保存 final/reference damage L2 范数。该重算与 scientific summary 的 RMS 逐项一致。
位移 RMS 用 P1 一致质量矩阵精确面积积分，不用节点个数加权代替面积。
strain tensor norm 使用 gamma²/2，stress tensor norm 使用2*tau²。
所有完整空间分布和参考数组另行保存。

| ΔN (cycle) | Damage RMS | max D error | u RMS (mm) | stress RMS (N/mm²) | ΣYmaxΔD error (N·mm) | D order |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 2.259583e-03 | 2.475577e-03 | 2.015963e-05 | 1.225273e-03 | 2.502327e-04 | — |
| 5 | 1.155490e-03 | 1.268277e-03 | 1.032471e-05 | 6.327896e-04 | 1.287750e-04 | 0.967551 |
| 2.5 | 5.844462e-04 | 6.421109e-04 | 5.226320e-06 | 3.217109e-04 | 6.534538e-05 | 0.983363 |
| 1.25 | 2.939348e-04 | 3.230950e-04 | 2.629514e-06 | 1.622216e-04 | 3.291788e-05 | 0.991574 |
| 0.3125 | 7.380875e-05 | 8.116140e-05 | 6.604860e-07 | 4.081574e-05 | 8.276087e-06 | 0.996816 |

最后 ΔN/8 和额外更细级的 D 阶分别0.991574、0.996816，满足预先固定的>0.9要求。
终点位移、应力重分布、strain amplitude、弹性能和累计耗散误差均随减小 ΔN 下降。
[error_table.csv](../results/phase1b/final/temporal_convergence/error_table.csv)、
[参考/配置/结果](../results/phase1b/final/temporal_convergence/summary.json)。

![Time convergence](../results/phase1b/final/temporal_convergence/temporal_convergence.png)

该试验只证明固定空间离散下的时间误差；不等于 FEM 空间收敛，
更不等于本构模型与真实材料的 discrepancy 已消除。

## J. Spatial damage localization — Test E

同质矩形上给定**初始**光滑高斯损伤缺陷：
`D0=0.05+0.22*exp(-0.5*((x-1)/0.28)^2-0.5*((y-0.5)/0.22)^2)`，坐标与长度均为 mm。
最终 D、ΔD、strain amplitude 和应力重分布由实际 FE–damage 演化求得，不预设最终热点。

初始 Ya 范围0.0271681183–0.0346562669 N/mm²；实际最大演化增量0.0913188112。
初始缺陷较高区域平均增量0.0879890，远区0.0827362；演化增量本身非均匀。
最大增量位置约(1.308398,0.320027) mm，未直接等同于初始缺陷中心。
终点 strain amplitude 数组变化范数0.00242775，stress 数组变化范数0.0888137 N/mm²，
这些数组范数只用于证明响应变化，不称为空间连续 L2 收敛指标。
N=40 final Dmax=0.2865520442。

[localized summary](../results/phase1b/final/localized_damage/summary.json)、
[raw trajectory](../results/phase1b/final/localized_damage/trajectory/truth/trajectory.npz)。
图用原 Tri3 单元的平片值，不作 node 插值、卷积或损伤平滑。
展示的是缺陷条件下的局部标量损伤非均匀演化，没有声称已经发生不连续裂纹。

![Local damage](../results/phase1b/final/localized_damage/damage_fields.png)

## K. Mesh sensitivity — Test F

同一物理域、材料、解析初值缺陷、比例载荷与100 cycle历程；nx=4/8/16、ny=nx/2。
基准 ΔN=2，并在**每一网格**上独立重复 ΔN=1，量化时间误差贡献。
D 不跨网格插值；共同的非节点 DIC 坐标用同一 ObservationOperator 采样 FE 位移。

| V / E | hmax (mm) | initial Dmax | final Dmax (ΔN=2) | Dmax change at ΔN=1 | right u (mm) | area D>0.5 (mm², ΔN=2) |
|---|---:|---:|---:|---:|---:|---:|
| 15 / 16 | 0.707107 | 0.201797 | 0.439347 | 9.606507e-04 | 0.0149603748 | 0.000000 |
| 45 / 64 | 0.382705 | 0.249565 | 0.499400 | 1.154723e-03 | 0.0150042729 | 0.000000 |
| 153 / 256 | 0.208248 | 0.266170 | 0.521828 | 1.265747e-03 | 0.0150158374 | 0.051740 |

final Dmax 跨网格范围0.0824815105，明显大于各网格半步长 Dmax 变化0.00096–0.00127。
峰值位置、阈值面积与 damage evolution curves 随空间离散改变。
同时初始 Dmax 为0.2018/0.2496/0.2662：同一高斯函数的 centroid 表示也存在空间采样误差。
因此不能把全部差异单独归因于软化的病态局部化，也不能凭这三层声称不收敛的数学证明。
对位移，时间步影响约2e-5 mm，与末两网格变化量同量级，故不宣称位移已空间收敛。

**没有获得网格客观局部化预测资格。**当前局部模型没有内在非局部/梯度长度尺度，
未来更强软化、几何缺陷和更细网格下的敏感性仍需研究；本轮未加正则化。
所有差异按原始值保留，没有通过平滑改善结果。
[mesh summary](../results/phase1b/final/mesh_sensitivity/summary.json)
包含均值、增量、峰值位置、共同 DIC 位移差；六个网格/步长组合保留原始轨迹。

![Mesh fields](../results/phase1b/final/mesh_sensitivity/mesh_damage_fields.png)

## L. Energy and dissipation analysis

各相位/起终点分别保存 elastic U、矩阵能量、`u^T f`、`u^T reaction`、
能量恒等式 gap、自由 DOF residual、总力/力矩平衡和 reaction fields。
非零位移边界的反力做功项包含在代数恒等式中。

\[
\Delta W_D^{max}\approx\sum_q t\,w_q^{phys}\,Y_q^{max}(D_k)\Delta D_q\ge0.
\]

同样保存 min-phase 估计；对 R=-1 两相位估计相同，不把它们相加冒充循环总耗散。
Y 是共轭驱动力，Ya 是疲劳 hazard 输入，二者用途不同。
另记固定相位损伤变化过程的梯形机械功和 `work-ΔU-ΔWD`。
该余额有一阶损伤积分与做功近似误差；不强制等于零，也不隐藏其值。

最终29条验证轨迹全部原始数值有限、N严格递增、D不可逆且<Dmax。
损伤耗散估计范围 0.0–1.249388e-03 N·mm，全部非负；
最大反力参与的 work identity gap 1.623701e-15 N·mm，
最大自由 DOF residual 2.736353e-14 N，
最大单步固定相位 balance gap 3.404075e-05 N·mm。
完整核验见 [numerical_audit.json](../results/phase1b/numerical_audit.json)。

非均匀40-cycle参考累计最大相位损伤耗散诊断0.004749450606 N·mm。
ΔN=10的误差2.502327e-4，ΔN=1.25的误差3.291788e-5，ΔN=.3125的误差8.276087e-6 N·mm。
这支持离散诊断随时间细化收敛，不代表完整真实疲劳耗散。
载荷控制 U 增加/位移控制 U 减少在 H 图和反力/位移标度中独立验证，
没有错误地要求全部工况 U 单调减小。
无塑性滞回、不可逆微观机制或循环内真实损耗，本轮不声称满足所有真实疲劳能量机制。

## M. Full synthetic fatigue trajectory — Test H

Geometry → seeded jittered Delaunay Tri3 mesh → proportional fatigue loading → FE solve →
Ya/hazard → damage update → 新 D 的 FE re-solve → FE-to-DIC → trajectory storage 实际完整运行。
seed=19，V=15/E=G=16，N=0–40、delta_N=5，含初值共9帧。
final Dmax=0.2877593446，面积均值0.1786277481，右端均值位移0.01219056846 mm。

[trajectory/truth](../results/phase1b/final/synthetic_trajectory/trajectory/truth/trajectory.npz)
保存 N、D、累计增量/周期、u/strain/stress 的 min/max、strain amplitude/mean、
Ya/Ymax/Ymin、endpoint hazard、用于推进的 frozen hazard/Ya/Y、ΔD、reaction/applied force、
residual、能量以及 mesh/完整 quadrature geometry/IDs。
[diagnostics](../results/phase1b/final/synthetic_trajectory/trajectory/truth/diagnostics.json)
保留能量/耗散/求解及拒绝记录。
[provenance](../results/phase1b/final/synthetic_trajectory/trajectory/provenance.json)
保存构成参数、BC 节点与边段、加载历史、种子、配置、单位/坐标系与 code version。

[trajectory/observations](../results/phase1b/final/synthetic_trajectory/trajectory/observations/displacement.npz)
只有 N、coords、displacement、mask、confidence，经单独 save/load API 保存。
四个非节点域内点有效，一个域外点为 NaN/False；观测接口没有 D 或其他 truth 标签。
没有图像相关、散斑生成或噪声校准，称为 DIC-like FE 位移观测。
同配置重新生成 mesh 并独立执行完整耦合，D/u/stress/hazard/DIC 数组逐元素 bitwise 相等
（允许无效域外点的 NaN 同位置一致）。不是仅比较相同内存对象。
存储通过完整临时目录→rename 发布；目标已存在拒绝覆盖，写出失败不留部分 checkpoint。

## N. Full pytest results and reproducibility

| 项目 | 实际证据 |
|---|---|
| 初始全量测试 | 165 passed in 1.93s；baseline/pytest.txt |
| 最终 `python -m pytest -q -W error` | **237 passed in 21.61s，exit 0**；[pytest_final.txt](../results/phase1b/pytest_final.txt) |
| 新增测试 | 72 项；原165项源文件不变，包含单元、耦合、科学、CLI/存储回滚 |
| 单独科学脚本 | 8类全部 PASS，均加 `-W error`；[commands.json](../results/phase1b/final/commands.json) 和各 *_command.txt |
| wheel 构建 | PASS；[build_final.txt](../results/phase1b/build_final.txt) |
| 离线临时 venv 安装 | PASS；在 `/tmp` 移除 PYTHONPATH 后运行安装包 constant/synthetic；[installation_check.json](../results/phase1b/installation_check.json) |
| 原资产、源哈希、shape/数值/输出审核 | PASS；delivery_audit.json 与 numerical_audit.json |
| compileall / Git/text/link 检查 | PASS；delivery_audit.json |

本地 Python3.13.12、NumPy2.4.4、SciPy1.17.1、Matplotlib3.10.9；未验证本地其他版本。
核心 Phase 0 路径仍仅 NumPy；新增 fatigue/FE 需要 mechanics extra，图表需要 verification extra。
明确测试新增路径不加载 torch、RL 或 networks；没有训练、GPU、模型权重或复杂 FE 库安装。
CI 增加八类 Phase 1B CLI，原回归保留；未 push，**远程 CI 尚未运行**。

图表采用 visualize-data 规范、用户指定的本地 Markdown 与独立科研 PNG/SVG，
原始值/量纲/尺度可追溯；关键图全尺寸检查并核对所有 PNG 的缩略预览，不伪称交互网页验证。
开发期新增测试的两个构造错误曾先触发布局/材料缺项检查，已修正测试输入，未放宽物理标准。
首次图/结果和开发日志保留在 development/、verified/；最终权威数值在 **final/**。

复现（输出须为新目录，不覆盖当前证据）：

```bash
python -m pytest -q -W error
PYTHONPATH=src MPLCONFIGDIR=/tmp/fatigue-phase1b-mpl python -W error scripts/verify_phase1b.py --config configs/phase1b/constant_hazard.toml --output results/phase1b/reproduce-constant
PYTHONPATH=src MPLCONFIGDIR=/tmp/fatigue-phase1b-mpl python -W error scripts/verify_phase1b.py --config configs/phase1b/temporal_convergence.toml --output results/phase1b/reproduce-temporal
PYTHONPATH=src MPLCONFIGDIR=/tmp/fatigue-phase1b-mpl python -W error scripts/verify_phase1b.py --config configs/phase1b/mesh_sensitivity.toml --output results/phase1b/reproduce-mesh
PYTHONPATH=src MPLCONFIGDIR=/tmp/fatigue-phase1b-mpl python -W error scripts/verify_phase1b.py --config configs/phase1b/synthetic_trajectory.toml --output results/phase1b/reproduce-synthetic
```

另四类配置为 reversed_loading、uniform_damage、localized_damage、rollback，使用同一 CLI。
科学 CLI 若有非预期失败写 FAILED summary、traceback/diagnostics 与 solver.log 并返回非零。
29条短数值验证轨迹包括不同步长/网格和重复运行，未批量生成数千条训练轨迹。

## O. Scientific limitations and unresolved risks

1. 合成 hazard 参数未校准，未核对任何经典疲劳定律原式；没有真实材料预测资格。
2. 局部 softening 未正则化。三个网格的损伤峰值/空间区域明显敏感，初值采样也参与差异；
   未验证网格客观局部化。不能将标量 D 称为真实裂纹面或路径。
3. 仅一阶 staggered，ΔD 上限是变化限制，不是误差控制；接近饱和或强耦合仍需独立时间研究。
   最小步长、最大二分深度耗尽或浮点饱和会失败，保留原因并回滚。
4. energy diagnostics 是相位弹性/损伤热力学核验；没有塑性、滞回或完整每周期损耗。
5. 不支持非比例多轴、混合控制、body-force 循环、非线性材料、3D/plane strain、接触或大变形。
6. history 当前 identity mapping；附加 history 仅被动无量纲存储，没有跨网格转移或其他历史变量演化。
7. 延续 Phase 1A 的全域 overlap/T junction 等网格诊断限制；pivot ratio 不是严格条件数。
8. 高精度时间参考共用 FE 和本构，提供算法对照而非独立成熟 FE 平台或真实材料对照。
9. synthetic DIC 是无噪声位移采样，不验证实验标定、图像相关、置信度/空间相关噪声或损伤可识别性。
10. 接口分离不是文件访问权限隔离；未来数据加载和训练仍需明确防标签泄漏。
11. 只验证本地软件组合，没有远程 CI、长期大规模稳定性/性能或真正材料破坏阈值验证。

据此，本轮给出**有限范围 PHASE1B_FATIGUE_EVOLUTION_QUALIFIED**：
积分点内部状态、循环驱动力、严格 physics-only FE耦合、冻结 hazard、零/反向振幅、
两种控制模式、固定网格时间一阶收敛、非负损伤耗散、原子回滚、端到端轨迹及确定性均有实际证据。
Mesh Test F 的资格是“敏感性已正确测量/披露”，不是“网格客观局部化已验证”。

## P. Recommended Phase 1C plan

1. 保留当前 physics-only baseline 与数据隔离；首先研究 DIC 配准、实际位移/应变噪声与质量模型。
2. 用该 forward model 做受控参数可识别性/混淆试验，先检验边界、E、疲劳参数与 D 的影响，
   再考虑传统参数估计/数据同化 baseline，不能从单条无噪声轨迹宣称可识别。
3. 为真实材料另行选取并核对原始文献、参数量纲与实验范围，校准与独立验证分开。
4. 若目标需要局部化预测，先设网格客观性判据、光滑/缺口等算例及初值离散误差检查，
   再单独评估非局部/梯度正则化；本轮不擅自引入。
5. 需要更高效的演化时，再以当前误差参考评估 predictor-corrector 或带真实误差估计的步长控制。
6. 后续任何学习方法另行立项，先定义观测-only loader 和严格数据拆分。

**Phase 1B 已停止；没有自动开展 Phase 1C、RL、GNN、GRU 或神经网络。**
