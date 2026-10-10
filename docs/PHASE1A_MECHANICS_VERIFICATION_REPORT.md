# Phase 1A — Tri3 Mechanics Scientific Verification

实施日期：2026-10-08。最终资格：**PHASE1A_MECHANICS_QUALIFIED**。

已实际实现并验证非结构 Tri3 平面应力稀疏 FE 正问题；本阶段没有疲劳演化、
damage 反演、真实图像相关、神经网络或 RL 训练。资格限定于本文列出的模型和算例。

## A. Git state and implementation summary

初始/最终分支：`feature/phase0-architecture`。
HEAD 保持 `dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9`（Initial commit）。
实施前工作区为 README 修改与 63 个 Phase 0 未跟踪文件，没有提交。
没有创建新 checkpoint commit、push、merge、rebase、reset、clean 或 main 修改。

首先实际阅读 Phase 0 报告、MeshData、FieldData、MechanicsState、solver/material 接口、
测试及文档；原有 123 项测试在修改前通过。未只依赖上一轮摘要。
实施前保存 **66 个文件** 的完整文件快照、SHA256 清单、tracked diff 与全部 untracked diff，
见 [baseline](../results/phase1a/baseline/manifest.json)、
[tar.gz](../results/phase1a/baseline/phase0_worktree.tar.gz)、
[tracked.patch](../results/phase1a/baseline/tracked.patch)、
[untracked.patch](../results/phase1a/baseline/untracked.patch)。
这些是人工核验/选择性恢复的依据，本轮不执行恢复，因此无需创建 Git checkpoint。

本轮相对基线的修改清单、原测试/原结果保留核验与完整最终状态见
[delivery_audit.json](../results/phase1a/delivery_audit.json)、
[phase1a_delta.patch](../results/phase1a/phase1a_delta.patch)、
[git_status.txt](../results/phase1a/git_status.txt)。
delta patch 覆盖代码、配置、文档和测试的文本差异；二进制科学结果单独保存并记录哈希。
Phase 0 报告、科学文档、physics/damage 核心及既有测试不重写，结果目录不删除。

主要新增：`geometry/triangles.py`、`geometry/quadrature.py`、
`materials/plane_stress_mm.py`、`mechanics/tri3_contracts.py`、`mechanics/tri3.py`、
`evaluation/tri3_meshes.py`、`evaluation/tri3_verification.py`、`evaluation/tri3_plots.py`、
`scripts/verify_phase1a.py`、四个 TOML 配置、独立单元/集成/科学测试。
最小兼容调整为新增显式 stress/traction 单位 N/mm^2、可选依赖及延迟 backend 入口。
细节见 [08_TRI3_MECHANICS_CONTRACTS](08_TRI3_MECHANICS_CONTRACTS.md)。

## B. Solver architecture

Tri3MechanicsSolver 继承现有 MechanicsSolver，实现 `_solve`，保持公共 solve 的输出检查。
Tri3SolverRequest 是 SolverRequest 的增量扩展，加入可选 damage、quadrature_layout 与 body_force。
`from_mm` 入口将 mm 厚度显式转换后存入原 thickness_m，不改变旧字段含义。
零损伤请求建立合法 quadrature FieldData；一般旧 SolverRequest 仍可通过明确单位适配使用。

流程：验证 mesh/layout/material → 取得 quadrature damage → 计算局部 C(D) →
Tri3 B 和稀疏 COO/CSR 装配 → 一致载荷 → 检查约束刚体模态 →
Dirichlet 消元 → CSC/SuperLU 解自由 DOF → reaction/residual → quadrature response/energy。
没有引用 DamageLaw 或 DamageTransition，没有隐式推进循环数或损伤。

输出继承 SolverResult：prediction MechanicsState、数值 diagnostics，以及 node reaction/applied force/
full equilibrium residual、积分 strain energy、element energy、显式 element strain/stress、
constrained DOF IDs 和可审查的 sparse/element assembly。
state.strain/stress 位于 quadrature，state.displacement 位于 node；无跨空间 reshape。

核心仍仅依赖 NumPy。mechanics extra 使用 SciPy，verification extra 增加 Matplotlib。
延迟导入保证 Phase 0 physics-only smoke 不加载 SciPy；不加载 torch/RL。
稀疏解法遵循 CSC 输入与 SuperLU solve 的 API，参见
[SciPy splu 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)。
本地实际运行版本为 SciPy 1.17.1，没有接入 FEniCSx。

## C. Tri3 element formulation

参考三角形 `(0,0),(1,0),(0,1)`：
\[
N_0=1-\xi-\eta,\quad N_1=\xi,\quad N_2=\eta,\qquad
J=[x_1-x_0\quad x_2-x_0].
\]
形函数定义参考 [DefElement 的一次三角形](https://defelement.org/elements/examples/triangle-lagrange-equispaced-1.html)。
物理梯度为参考梯度行向量乘 `J^-1`，各单元内恒定。
\[
B=\begin{bmatrix}
N_{0,x}&0&N_{1,x}&0&N_{2,x}&0\\
0&N_{0,y}&0&N_{1,y}&0&N_{2,y}\\
N_{0,y}&N_{0,x}&N_{1,y}&N_{1,x}&N_{2,y}&N_{2,x}
\end{bmatrix},\quad
K_e=tA_e B^TC_eB.
\]
多积分点布局下实际使用 \(K_e=t\sum_q w_q^{phys}B^TC_qB\)，而不是忽略局部 D。
DOF 编号 `[u0,v0,u1,v1,...]`，通过 node connectivity 形成每单元六个全局索引。

以单元尺度归一化后检查 determinant，拒绝零面积/数值退化；面积取 `abs(det J)/2`，
梯度保留有向 J。顺逆时针交换会相应重排 DOF，不生成负刚度。
测试验证方向等价、分片线性再现、单元三个刚体零模态、C 对称/正定与 K 对称。
全局装配使用 SciPy COO→CSR，不为求解建立稠密全局矩阵；小型科学检查单独转换检查 cond/eigenvalues。

## D. Constitutive model and units

模型为 2D small-strain isotropic linear elasticity，plane stress。
\[
C_0=\frac{E}{1-\nu^2}
\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&(1-\nu)/2\end{bmatrix},\quad C(D)=(1-D)C_0.
\]
该关系留在具体材料模型，core 不决定 damage 的刚度关系。
E>0、-1<nu<0.5、`0<=D<=damage_max<1`，默认 damage_max=0.999。
本轮没有修改 Phase 0 DamageTransition 的物理约束；旧本构允许 D=1 的历史语义保留，
新 backend 额外拒绝会破坏正刚度的 D=1。

| 量 | 单位/约定 |
|---|---|
| coordinates/displacement/thickness | mm |
| E/stress/traction | N/mm^2 |
| body-force density | N/mm^3 |
| nodal/applied/reaction force | N |
| stiffness | N/mm |
| energy | N*mm |
| strain | 1，`[exx,eyy,gamma_xy]` |
| stress components | `[sxx,syy,tau_xy]` |
| damage | 1 |

gamma_xy=`du/dy+dv/dx`；tensor epsilon_xy=gamma_xy/2。纯剪切单元测试明确验证两者差别。
能量内积用 engineering gamma_xy*tau_xy；应变误差的张量 Frobenius 范数用 0.5*gamma_xy²。
MM adapter 对显式声明 tensor shear 的输入报错。旧 E[Pa] 与 traction[Pa] ×1e-6 后装配；
旧 thickness_m ×1000。没有将不同单位的相同数值偷偷当作同一个物理量。

## E. Quadrature layout

默认材料规则为中心 `(1/3,1/3)`、参考权重 1/2，物理权重 `abs(det J)/2`。
保存 reference points/weights、物理坐标、Jacobian、有向 determinant、
element_to_quadrature、quadrature_to_element 与 material_state_indices。
layout reference 包括 mesh_id、规则和内容哈希；同长度 element/node/quadrature 不视为同支撑。

quadrature damage 必须匹配布局；element damage 通过具名分片常值 lift，投影记录在 metadata；
node damage 明确拒绝。state.damage 保留原输入字段，无意外改写。
多点正权布局也实际参与装配：独立测试用 **E=1、G=3** 和 D=[0.1,0.2,0.4]，
验证权重积分后的刚度、三个积分点 stress 与 element average。
未来积分点内部变量的空间引用已建立；当前 stateless 弹性材料拒绝非空 history，
且 backend 仅接受 identity material-state storage indices。未实现塑性历史更新或任意高阶单元。

载荷积分与材料状态布局分离，默认 5×5 Duffy Gaussian；误差积分独立使用 7×7。
最低允许载荷阶数为 2，以保证常量/线性体力的一致节点积分，不接受一阶欠积分设置。

## F. Boundary condition implementation

Dirichlet 支持节点集合、单方向和非零位移；消元为
\(K_{ff}u_f=f_f-K_{fc}u_c\)。保留原 K/f，约束反力 `R_c=(K*u-f)_c`，自由 DOF 的 reaction=0。
不将整块非零边界行直接清零而遗漏右端贡献。

外边界牵引作为全局向量给出，每段等效节点力为 `t*L*traction/2`，支持直接节点力。
体力按 `t*integral(N_i*b dA)` 组装，使用与解析解一致的 BodyForceDensity。
重复/冲突位移、重复节点力、重叠牵引线段会明确拒绝；有意叠加需调用方先合并。
厚度缩放测试验证 traction/body force/K 均正确缩放；固定节点力时厚度翻倍导致位移减半。

每个全边连通域检查两个平移和一个旋转模态的约束秩。分离域分别约束；
铰接式单点相连组件、孤立节点明确拒绝。欠约束产生 MechanicsConvergenceError。
稀疏 LU 奇异/相对 pivot 过小或自由 DOF 残差超限会失败，未添加人工小刚度。
极端材料对比的病态矩阵测试实际触发失败诊断，而非伪装成功。

默认自由 DOF 残差要求：`r_inf <= 1e-9 N + 1e-10*force_scale`，
force_scale 是内部/外部节点力的最大绝对值。relative residual 在近零力刚体算例中不宜单独解释。
预检查失败 iterations=0 时，旧接口 residual_norm=0 仅表示尚未求残差，converged=False。
统一 logging 与已有异常机制保存到每例 solver.log。
patch 的无约束矩阵检出三个刚体模态，约束后两个 reduced matrix 的最小特征值均为正。

## G. Patch test evidence

配置：[tri3_patch_test.toml](../configs/phase1/tri3_patch_test.toml)。
E=70000 N/mm²、nu=0.3、t=0.8 mm，矩形域 2.3×1.1 mm；两组 jittered Delaunay 网格，
第二组交替反向节点顺序。仅在外边界施加仿射位移，内部 DOF 真实求解。
\(u=0.017+0.0012x-0.0007y\)，\(v=-0.023+0.0004x-0.0003y\)。
理论 strain=[0.0012,-0.0003,-0.0003]，stress 由独立解析式计算，不用 solver 返回值当标准答案。

<!-- PATCH_TABLE_START -->

| V / E / free DOFs | cond₂ | max u error (mm) | max strain error | max stress error (N/mm²) | min Kff eigenvalue (N/mm) |
|---|---:|---:|---:|---:|---:|
| 16 / 18 / 8 | 5.289583 | 6.938894e-18 | 2.596664e-17 | 1.849187e-12 | 76858.171189 |
| 30 / 40 / 24 | 12.436301 | 1.734723e-17 | 6.413056e-17 | 4.776624e-12 | 32543.548011 |

<!-- PATCH_TABLE_END -->

位移容差固定为 `64*eps*cond_2*max(1,max_abs(u_exact))`，strain 为 `256*eps*cond_2`，
stress 为上述 strain 容差 ×E，反力平衡用 force scale 与 cond_2。首次实际运行即通过，未放宽标准。
反力力/力矩均平衡；原始字段、矩阵与误差在
[patch summary](../results/phase1a/verified/patch/summary.json) 与 error_table.csv。

## H. Rigid body and uniaxial tests

小转角线性刚体：u=0.017-0.002*y，v=-0.023+0.002*x。
最大 strain **2.7755575615628914e-17**，最大 stress **2.135044278125301e-12 N/mm²**，
积分能量 **1.2884259238385783e-29 N·mm**。这不是有限转角非线性旋转验证。

单轴配置：[uniaxial_tension.toml](../configs/phase1/uniaxial_tension.toml)。
10×4 mm 板、t=0.8 mm、E=70000 N/mm²、nu=0.3，右边 σxx=10 N/mm²，
左边仅约束 u，左下角锚定一个 v DOF，其他横向自由度自由。
因此不会通过全部约束横向位移误造受限收缩。

理论 exx=1.4285714285714286e-4、eyy=-4.285714285714286e-5、gamma=0，
右端 u=0.0014285714285714286 mm、上沿 v=-0.00017142857142857143 mm。
实际节点位移最大误差 **4.1129522344339426e-18 mm**；strain 最大误差
**5.802175688691957e-19**；反力总和约 [-32.00000000000006, 1.2878587085651816e-14] N，
与外力 [32,0] N 平衡。
原始证据：[uniaxial summary](../results/phase1a/verified/uniaxial/summary.json)。

能量 Test E：每例均比较 `0.5*u^T*K*u` 与积分能量。
另用非零 Dirichlet 加内部节点力验证 `2U=u^T*f+u^T*R` 和自由虚功平衡。
混合算例 U=0.10474215073809577 N·mm，force work=0.004814606624031708，
reaction work=0.2046696948520937 N·mm；能量矩阵/积分差 **6.8695e-15 N·mm**，
完整 work identity 差 **6.6142e-14 N·mm**。
`u^T*f` 是当前力与位移的代数乘积，不自动等于任意加载历程的累计外力功；
线性比例加载才可用相应 1/2 因子。非零约束反力贡献不能忽略。

## I. Damage stiffness verification

配置：[damage_stiffness.toml](../configs/phase1/damage_stiffness.toml)。
同一几何/材料/固定损伤分别进行载荷控制与位移控制，D=0、0.2、0.5。
逐单元矩阵与全局 K 验证 `(1-D)*K0`，D=0 恢复弹性。

<!-- DAMAGE_TABLE_START -->

| D | E_eff (N/mm²) | load-control right u (mm) | displacement-control right R (N) |
|---:|---:|---:|---:|
| 0 | 70000 | 0.001428571429 | 224.000000000 |
| 0.2 | 56000 | 0.001785714286 | 179.200000000 |
| 0.5 | 35000 | 0.002857142857 | 112.000000000 |

<!-- DAMAGE_TABLE_END -->

载荷控制的右端力保持 32 N，位移按 `1/(1-D)` 增大；位移控制右端 u=0.01 mm，
反力按 `(1-D)` 减小。不能把载荷控制下右边自由 DOF 的零 reaction 当作没有施加载荷。

非均匀场：左/右 centroid 区域 D=0.1/0.5。每个 Ke 与该处 `(1-D_e)*Ke0` 对照，
最小 tangent eigenvalue **13461.53846153846 N/mm² >0**；节点位移向量变化的欧氏范数
**0.0022733257886007126 mm**，自由残差正常、输入 damage 不变。
另有独立不同材料区域测试，验证局部 E 与 damage 通过区域和积分布局进入刚度。
这全部是 **fixed-damage mechanics**，没有 fatigue damage update。

数据与图：[damage summary](../results/phase1a/verified/damage/summary.json)、
[control plot](../results/phase1a/verified/damage/damage_control.png)。

## J. Mesh convergence results

配置：[mesh_convergence.toml](../configs/phase1/mesh_convergence.toml)。
域 `[0,1]^2 mm`，E=2000 N/mm²、nu=0.3、t=0.8 mm、A=0.001 mm。
制造解 u=A*sin(pi*x)*sin(pi*y)，v=0，所有外边界施加精确位移。
这里 pi 对应波数 π/(1 mm)，而非对有量纲坐标任意取正弦。

令 M=E/(1-nu²)、G=E/[2(1+nu)]，体力严格为
\[
b_x=A\pi^2(M+G)\sin(\pi x)\sin(\pi y),\quad
b_y=-A\pi^2(\nu M+G)\cos(\pi x)\cos(\pi y),
\]
在代码中使用固定 1 mm 长度尺度的数值坐标。
另外通过独立有限差分 stress divergence 检验 `div(sigma)+b=0` 的符号和分量。
体力/边界匹配制造解，不以“更平滑”作为收敛依据。

四组 seed=19 的加密非结构网格；h 取最大单元边长，误差用独立高阶面积积分。
位移 L2 是 `sqrt(integral(|u_h-u|² dA))`，在二维单位为 mm²；
strain tensor L2 单位为 mm。图显示除以精确解范数后的无量纲相对误差。

<!-- CONVERGENCE_TABLE_START -->

| V / E | h_max (mm) | u L2 (mm²) | strain L2 (mm) | relative u L2 | relative strain L2 | u / strain rate |
|---|---:|---:|---:|---:|---:|---:|
| 25 / 32 | 0.366312713 | 7.422241985e-05 | 6.904870319e-04 | 0.148444840 | 0.358913687 | — |
| 81 / 128 | 0.191352444 | 1.806805788e-05 | 3.430811461e-04 | 0.036136116 | 0.178332848 | 2.175832 / 1.077090 |
| 289 / 512 | 0.104124081 | 4.735395518e-06 | 1.746320943e-04 | 0.009470791 | 0.090773390 | 2.200503 / 1.109693 |
| 1089 / 2048 | 0.053596275 | 1.151233760e-06 | 8.665507710e-05 | 0.002302468 | 0.045043124 | 2.129533 / 1.055175 |

<!-- CONVERGENCE_TABLE_END -->

误差随加密严格下降；最后两层位移阶 **2.1295334233**，应变阶 **1.0551753016**，
达到预先固定的 >1.8 / >0.9 判定。全四点 log–log fit 为 2.171 / 1.083；
fit 与最后两层阶的定义不同，图中已明确标注。
该结果支持本光滑问题的 P1 预期趋势，不外推到裂纹、接触或材料界面问题。

![Tri3 manufactured-solution convergence](../results/phase1a/verified/convergence/convergence.png)

精确表：[error_table.csv](../results/phase1a/verified/convergence/error_table.csv)；
配置/误差/自由残差：[summary.json](../results/phase1a/verified/convergence/summary.json)。
PNG 与 SVG 都由程序生成，经过视觉检查。首次图有拥挤次刻度，最终改为显示实际 h 值，
保留首次输出和日志，未修改任何数值数据。

## K. FE-to-DIC end-to-end test

闭环为：不规则 mesh → plane-stress material → traction/约束 → **真实 Tri3 solve** →
node displacement → Phase 0 ObservationOperator → DICObservation。
单轴例使用四个不与 FE 节点重合的域内采样点与一个网格外点。
域内预测与同坐标解析位移最大误差 **3.537209587733958e-18 mm**，
mask=[True,True,True,True,False]，外点 NaN，位移单位保持 mm。
输出 `synthetic_displacement_observation.npz` 只有坐标、位移、mask 和质量，未混入 damage。

Synthetic DIC 仅指 FE 位移采样生成的合成位移观测，**image_correlation_performed=False**。
没有散斑图像生成、真实图像相关匹配或真实实验校准。

## L. Full pytest results and reproducibility

本地：Python 3.13.12、NumPy 2.4.4、SciPy 1.17.1、pytest 9.0.3、Matplotlib 3.10.9。
核心 NumPy-only，SciPy/Matplotlib 为可选 extra；未安装重型 FE 或 RL 依赖。

| 检查 | 实际结果/证据 |
|---|---|
| 修改前 `python -m pytest -q -W error` | 123 passed in 0.28s；[baseline log](../results/phase1a/baseline/pytest.txt) |
| 新 backend 接入后的原测试 | 123 passed in 0.28s（开发终端记录，原测试未修改） |
| 开发中全量测试 | 160 passed in 3.63s；[development log](../results/phase1a/development_tests_01.txt) |
| 进一步拓扑/剪切检查 | 164 passed in 1.89s；[intermediate log](../results/phase1a/pytest.txt) |
| 原十个 test 文件独立回归 | **123 passed in 0.25s**；[regression log](../results/phase1a/phase0_regression.txt) |
| 最终 `python -m pytest -q -W error` | **165 passed in 1.89s，exit 0**；[final log](../results/phase1a/pytest_final.txt) |
| 四个 scientific CLI | **全部 PASS、exit 0**；verified/*_command.txt 保存命令/stdout/stderr |
| compileall、Git diff/text/link/原资产检查 | PASS；delivery_audit.json |
| wheel 构建 | 成功；[build_final.txt](../results/phase1a/build_final.txt) |
| /tmp venv 离线安装、取消 PYTHONPATH 后验证 | 四类 scientific cases 均 PASS；[installation_check.json](../results/phase1a/installation_check.json) |

新增 **42 项**测试，包括 element/quadrature、完整 solver、科学验证与真实 CLI 集成。
没有改写 Phase 0 标准答案，没有放宽容差；最后审查增加了对一阶 Duffy 载荷欠积分的拒绝。
日志中的无约束 ERROR 属于主动检验的 **EXPECTED_FAILURE**，summary 明确保存 converged=False，
不是隐藏失败。科学脚本若发生非预期异常会写 status=FAILED、traceback 并返回非零，
CLI 拒绝覆盖已有输出目录，集成测试验证此保护。

复现命令（已有依赖时不需安装；输出须选新目录）：

```bash
python -m pip install -e '.[test,verification]'
python -m pytest -q -W error
PYTHONPATH=src python scripts/verify_phase1a.py --config configs/phase1/tri3_patch_test.toml --output results/reproduce-patch
PYTHONPATH=src python scripts/verify_phase1a.py --config configs/phase1/uniaxial_tension.toml --output results/reproduce-uniaxial
PYTHONPATH=src python scripts/verify_phase1a.py --config configs/phase1/damage_stiffness.toml --output results/reproduce-damage
PYTHONPATH=src python scripts/verify_phase1a.py --config configs/phase1/mesh_convergence.toml --output results/reproduce-convergence
```

记录包括 mesh、E/nu/t、BC 与合成 force、solver defaults、Jacobian/weights、状态场、
reaction、energy、残差、误差表及版本/Git 信息。图表规范使用 visualize-data；
交付形式遵循本任务要求的本地 Markdown 与独立科研图。
CI 已配置 Python 3.11/3.13 全量测试与四类 CLI；没有 push，故**远程 CI 尚未执行**。
本地没有宣称验证其他 Python/SciPy 组合。

## M. Unresolved limitations and qualification scope

仅验证二维 small-strain、plane stress、Tri3、常厚度、两种具名 isotropic material。
不支持 plane strain、3D、quad4、高阶曲边、接触、弹塑性、各向异性或大变形。
未进行独立成熟 FE backend 对照；当前资格来自可复现解析验证与矩阵/能量核验。
完整全域 overlap/T junction/mesh quality 诊断未实现，极瘦单元或极端材料对比需谨慎，
后者可能被配置 pivot 阈值拒绝。pivot ratio 不是完整条件数估计。

material-state storage 当前 identity mapping；历史变量演化、跨 layout 转移、
node damage projection、变厚度与复杂载荷函数的自适应积分仍需另行验证。
高阶体力积分不是对任意函数的自动精确保证；制造解已验证指定阶数与加密范围。
CSR inspection assembly 可由调用方修改，不可拿它代替不可变 MechanicsState 作长期状态。

尚未校准真实材料/DIC 参数或噪声模型，也未验证 damage 可识别性。
固定损伤的正问题不证明疲劳定律、失效阈值、RUL 或 RL 可用。
不声称该 backend 已具备成熟工业 FE 平台的全部稳健性或性能。

这些限制均在允许的 Phase 1A 范围之外；用户要求的科学 Test A–H 与 FE→DIC 闭环
已经真实完成，故资格为 **PHASE1A_MECHANICS_QUALIFIED**。

## N. Recommended Phase 1B implementation plan

1. 先完善积分点历史变量、材料状态索引与载荷相位/工况数据契约，增加更完整网格诊断。
2. 接入有来源且可校准的 stress/energy-driven DamageLaw，坚持 physics-only。
3. 构建“当前 D 求解→hazard→解析更新→新 D 重求解”的显式耦合，检查步长收敛、
   不可逆、边界/能量/耗散与内部变量一致性，避免旧 response 被误用。
4. 用小规模受控算例研究 DIC 配准、质量/协方差、damage 可识别性和参数/同化 baseline。
5. 这些前提满足后再另行评价神经估计器或 RL 假设，当前不把它们当作下一阶段必选方案。

本轮停止于 Phase 1A，未开始 Phase 1B。
