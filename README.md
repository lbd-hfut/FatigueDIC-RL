# FatigueDIC-RL
Physics-Informed Reinforcement Learning for DIC-Observed Full-Field Fatigue Damage Evolution

当前交付推进到 Phase 2C 的低维疲劳演化修正；未开展 RL。
Phase 2A 保留 `PHASE2A_BELIEF_ESTIMATION_PARTIALLY_QUALIFIED` 及原89/100覆盖率证据。
Phase 2B 资格、全部训练seed、校准与独立时间留出评价见正式报告。

## Phase 2C：非RL物理演化修正

资格为 `PHASE2C_PHYSICS_CORRECTION_PARTIALLY_QUALIFIED`。复用原不可逆 DamageTransition 与
Tri3/FatigueEvolutionSolver，实现有界乘性 hazard、K=2/4/8 固定物理/空间basis、
DIC-only 多步 TRF 与合成 positive-hazard 残差监督。候选参数从自身N0重演；
N≤20同化，N30–220完全开环，参数失配、结构失配和状态估计误差分开评价。

结构失配的高SNR受控分布中K2修正改善长期预测；仅参数失配时A/m重标定更准确。
matched负对照、弱可辨识方向、未收敛OOD和未知载荷失败均保留，不授予概率coverage资格。
Graph-conditioned extension与Phase2B state替代未执行；当前RL建议为NO-GO。

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -W error
PYTHONPATH=src python scripts/train_phase2c.py --stage dataset --output results/reproduce2c/dataset
PYTHONPATH=src python scripts/train_phase2c.py --stage supervised --dataset results/reproduce2c/dataset --output results/reproduce2c/training
```

完整实验、时间收敛、原阶段科学回归和源码恢复证据见
[Phase2C报告](docs/PHASE2C_PHYSICS_CORRECTION_REPORT.md)、
[修正契约](docs/13_PHYSICS_CORRECTION_CONTRACTS.md)及
[Git冻结建议](docs/GIT_PHASE2C_BASELINE_RECOMMENDATION.md)。结果写入新目录，不自动创建Git提交。

## Phase 2B：非规则双图与循环 belief estimator

DIC坐标kNN图、Tri3 FE图、global/形函数Pᵀ融合、因果GRU、低维参数与完整Cholesky
不确定性头，共用Phase2A协议。候选参数更新后从N0重演物理history；N≤20观测同化，
N30/40开环physics rollout。torch仅为可选 `.[neural]` 依赖，数值核心保持独立。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
PYTHONPATH=src python scripts/generate_phase2b_dataset.py --config configs/phase2b/uniform.toml --output results/reproduce-2b-data
PYTHONPATH=src python scripts/train_phase2b.py --stage initial --dataset results/reproduce-2b-data --output results/reproduce-2b-initial
```

训练、causal条件诊断、五seed checkpoint、calibration/test、OOD及完整历史科学CLI都保存于
`results/phase2b/20261008T114713Z/`。所有输出需新目录；正式执行命令及配置见
[Phase2B报告](docs/PHASE2B_GRAPH_RECURRENT_ESTIMATOR_REPORT.md) 和
[神经接口契约](docs/12_NEURAL_BELIEF_CONTRACTS.md)。原共享配准的只读复核见
[Phase2A uncertainty follow-up](docs/PHASE2A_UNCERTAINTY_FOLLOWUP.md)。

## Phase 2A：物理序贯 belief estimation

在 Phase 1B/1C 上新增固定先验的 prefix MAP、warm/cold TRF 对照、候选独立疲劳重演、
新损伤 FEM 预测、秩亏感知的局部 Gaussian 不确定性，以及 N≤20 同化、N=30/40 时间留出。
保留 irregular DIC、共享配准 nuisance、prediction/filtered/forecast 标签和完整优化诊断。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -W error
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -W error scripts/verify_phase2a.py --all --output results/reproduce-phase2a
```

14 类科学配置在 `configs/phase2a/`；完整 matrix 含两组每情景100个种子的 CPU 校准实验。
输出必须是新目录。见 [序贯 belief 契约](docs/11_SEQUENTIAL_BELIEF_CONTRACTS.md) 和
[Phase 2A 报告](docs/PHASE2A_SEQUENTIAL_BELIEF_REPORT.md)。资格限于已验证的低维物理参数空间；
模型失配导致的偏差和区间失效单独报告。该历史阶段未实现神经网络、RL 或 RUL。

## Phase 1C：观测与低维反演

新增显式 pixel/mm 标定、各向异性及空间协方差、配准 nuisance、连续损伤参数化、
经过多步长检查的 FE 有限差分灵敏度、Fisher 零空间诊断和有界多初值 TRF 反演。
观测接口没有损伤标签；真值只进入独立科学评价。未知 E0、边界幅值和稀疏采样
造成的混淆均作为验证结果保存，不把低 DIC 误差解释为唯一恢复损伤。

```bash
python -m pip install -e '.[test,verification]'
python -m pytest -q -W error
PYTHONPATH=src python scripts/verify_phase1c.py --config configs/phase1c/uniform_confounding.toml --output results/reproduce-phase1c-confounding
PYTHONPATH=src python scripts/verify_phase1c.py --config configs/phase1c/inverse_recovery.toml --output results/reproduce-phase1c-inverse
```

全部十类科学实验配置在 `configs/phase1c/`。结果必须写入新目录。
见 [观测与反演契约](docs/10_DIC_INVERSE_CONTRACTS.md) 和
[Phase 1C 报告](docs/PHASE1C_IDENTIFIABILITY_REPORT.md)。资格不表示任意损伤场可唯一恢复，
不包含真实图像相关、实验噪声标定或网格客观局部化。本阶段完成后停止。

## Phase 1A（历史基线）

已增加真实的 SciPy sparse Tri3 平面应力求解器：常应变单元、独立积分布局、
分区材料、固定损伤刚度、非零位移消元、牵引/节点力/体力、反力、能量与失败诊断。
统一采用 mm、N、N/mm²、N/mm³、N·mm；旧 Pa 本构和 thickness_m 通过显式适配保留。
工程剪切为 gamma_xy，不把 tensor epsilon_xy 当作工程剪切输入。

```bash
python -m pip install -e '.[test,verification]'
python -m pytest -q -W error
PYTHONPATH=src python scripts/verify_phase1a.py --config configs/phase1/tri3_patch_test.toml --output results/my-patch-run
PYTHONPATH=src python scripts/verify_phase1a.py --config configs/phase1/uniaxial_tension.toml --output results/my-tension-run
PYTHONPATH=src python scripts/verify_phase1a.py --config configs/phase1/damage_stiffness.toml --output results/my-fixed-damage-run
PYTHONPATH=src python scripts/verify_phase1a.py --config configs/phase1/mesh_convergence.toml --output results/my-convergence-run
```

输出目录必须为新目录；脚本拒绝覆盖已有结果。核心仍只依赖 NumPy，只有 FE backend
额外使用 SciPy，静态验证图额外使用 Matplotlib；不安装 CUDA、FEniCSx 或 RL 库。
仅需 FE 时安装 `.[mechanics]`，需要完整测试与图表时安装 `.[test,verification]`。

科学证据与限定见 [Phase 1A 报告](docs/PHASE1A_MECHANICS_VERIFICATION_REPORT.md)。
Phase 1A 范围不包含疲劳损伤演化、真实 damage 反演、图像相关或 RL 训练。
Phase 0 待提交工作区的完整快照、diff 与哈希保存在 `results/phase1a/baseline/`，未创建 Git commit。

## Phase 1B：physics-only 疲劳演化

新增积分点 `MaterialHistoryState`、比例 `FatigueLoadCycle`、振幅能量驱动力、
可校准的合成 hazard 定律，以及带终点力学重求解和原子回滚的 `FatigueEvolutionSolver`。
循环数 N 与保存帧数独立；synthetic truth 和 DIC 位移观测分别保存。
定律未做真实材料标定；局部软化的网格敏感性显式记录，不解释为真实裂纹几何。

```bash
python -m pytest -q -W error
PYTHONPATH=src python scripts/verify_phase1b.py --config configs/phase1b/synthetic_trajectory.toml --output results/reproduce-phase1b-synthetic
PYTHONPATH=src python scripts/verify_phase1b.py --config configs/phase1b/temporal_convergence.toml --output results/reproduce-phase1b-temporal
```

输出目录必须不存在。配置在 `configs/phase1b/`；契约和验证证据见
[疲劳耦合契约](docs/09_FATIGUE_EVOLUTION_CONTRACTS.md) 与
[Phase 1B 报告](docs/PHASE1B_FATIGUE_EVOLUTION_REPORT.md)。本轮停止于 Phase 1B。

## Phase 0（历史基线）

本仓库目前提供可测试的科研架构基础：非规则二维网格、带位置和单位的物理场、
真值/belief/预测状态区分、DIC 位移观测、三角形形函数映射、非负 hazard 损伤更新，
以及独立的本构、求解器、估计器和 latent correction 接口。

**截至 Phase 0，尚未实现 FE 求解器、损伤反演或 RL 训练。** Phase 0 smoke test 使用已知仿射位移场，
验证解析插值与模块接口，不构成完整疲劳仿真或真实材料验证。

## 运行

核心运行依赖只有 NumPy，测试依赖 pytest；Python >= 3.11。
使用独立虚拟环境，不需要 CUDA、PyTorch、FEniCSx 或强化学习库。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q -W error
python scripts/phase0_smoke.py --config configs/phase0.toml
```

环境已有 NumPy 和 pytest 时，无需安装即可运行源码：

```bash
python -m pytest -q -W error
PYTHONPATH=src python scripts/phase0_smoke.py --config configs/phase0.toml
```

## 结构

```text
configs/                 解析 smoke 配置
src/fatigue_dic_rl/
  core/                  FieldData、MechanicsState、载荷与位置契约
  geometry/              tri3/quad4/混合网格拓扑和边界标签
  materials/             本构、DamageLaw 和显式注册表
  mechanics/             求解器契约和边界条件；无 FE backend
  observations/          DICObservation、tri3 ObservationOperator
  inference/             StateEstimator 接口
  physics/               DamageTransition 和硬物理约束
  rl/                    latent action 与 hazard correction 接口
  networks/              DIC/FE图、形函数融合、GRU和参数/协方差头
  environments/          真实物理环境与模型预测环境设计说明
  world_models/          未来代理模型设计说明
  datasets/              观测与 synthetic truth 隔离规范
  training/              隔离synthetic数据、训练循环、loss与checkpoint
  evaluation/            验证和资格边界说明
  visualization/         网格/场可视化设计说明
  smoke.py               确定性的解析模块 smoke
scripts/                 smoke 命令行入口
tests/{unit,integration,scientific}/
docs/                    科学设计、验收矩阵与实施报告
results/                 结果保存约定；本轮没有疲劳数据
```

尚未开发的目录只有设计说明，没有虚假的空实现。

## 数据与物理边界

V、E、G、Q 分别表示节点、单元、积分点和观测点，数量独立。
所有数值操作必须显式检查空间引用、位置、单位、坐标系和有效性。
DICObservation 没有 damage 真值通道；观测元数据只允许采集信息。

DamageLaw 返回 hazard `lambda [cycle^-1]`；DamageTransition 在每个区间冻结 hazard，
使用解析更新 `D_new = Dmax - (Dmax-D)*exp(-delta_cycles*lambda)`。
损伤不下降且保持在 `[0,Dmax]`，非法输入会报错。该数值性质不证明疲劳模型正确。

`C(D)=(1-D)C0` 仅属于示例 `IsotropicPlaneStress`，没有写入 core。
未来策略修正估计模型的 hazard，不直接控制真实试件。

从 [科学表述](docs/00_SCIENTIFIC_FORMULATION.md) 和
[系统架构](docs/01_SYSTEM_ARCHITECTURE.md) 开始；
验收证据见 [PHASE0_ACCEPTANCE](docs/PHASE0_ACCEPTANCE.md) 与
[实施报告](docs/PHASE0_IMPLEMENTATION_REPORT.md)。

## 扩展

用 `DamageLaw._hazard` 增加物理模型，通过实例注册表注册；用
`MechanicsSolver._solve` 增加经过独立验证的 backend。quad4 和混合单元能被 MeshData
保存，但当前 ObservationOperator 会明确拒绝这些网格，需增加形函数适配器。

MIT License，保留原有 [LICENSE](LICENSE)。
