# Phase 2A — Physics-Based Sequential Belief Estimation

实施日期：2026-10-08（Asia/Shanghai）。最终资格：**PHASE2A_BELIEF_ESTIMATION_PARTIALLY_QUALIFIED**。

资格限定于已定义的低维初始损伤、fatigue A 和显式 nuisance 参数空间、固定 Tri3 mesh、
已支持的局部不确定性近似与严格时间留出协议。参数后验、当前损伤估计、当前观测拟合和
未来开环预测分别保存；不将优化收敛、先验约束或小 DIC 残差解释为唯一恢复真实损伤。
本阶段不实现任意高维损伤 posterior、真实材料校准、模型失配修正、神经状态估计器、RL、RUL，
完成后停止于 Phase 2A。

实现与正式验证已完成：357项测试通过，14类正式科学配置通过预声明数值检查，原阶段23例CLI/解析smoke回归通过。
1,000次带噪拟合全部保留且没有优化失败；共享配准fatigue情景的d0局部95%区间实测覆盖率89%，
其Wilson区间不包含95%，因此**不授予全部不确定性情景的完整校准资格**。
部分资格的原因是校准精度限制，不是遗漏科学脚本或提前停止实现。

## A. Git and environment audit

实际分支：`feature/phase0-architecture`；HEAD：
`dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9`，仍是 Initial commit；只有一个 worktree。
科研基线主要是未提交/未跟踪文件。HEAD 单独不能复现 Phase 0–1C 或本次交付。
本轮未 commit、push、merge、reset、clean、rebase、分支切换或覆盖历史结果。

开发前审计 Phase 0–1C 报告、Static/FatigueForwardAdapter、ParameterSpace、TRF、
InverseObjective、centered_jacobian、fisher_diagnostics、SyntheticTruth、DICMeasurement.load、
FatigueEvolutionSolver、MaterialHistoryState、loading 与 covariance/registration 接口。
检查父目录及项目适用 AGENTS.md，未发现额外规范。基线实跑：
**303 passed in 210.10s，exit 0**，见 [开发前测试](../results/phase2a/20261008T102400Z/baseline/pytest.txt)。

在修改任何源码前保存 **137 个**源码、测试、配置、文档和工程文件，并逐文件核对 tar 内 SHA256。
[开发前源码快照](../results/phase2a/20261008T102400Z/baseline/phase01c_source.tar.gz)
大小 191,740 bytes，SHA256：
`b3e8393d0a849cf2bb86ba55ff8ba2a26d431dcddef78763d4bdd90db34f7d3c`。
[原始 source manifest](../results/phase2a/20261008T102400Z/baseline/source_manifest.json)
的规范 JSON 聚合 SHA256：`5ba381de05c4442fe5445fab175ec553e7173d35f49ace4d2dc8298adb0faf33`。
快照排除 results、cache 和旧 tar，避免递归打包；旧科学资产通过独立清单保持原位。

开发前 [旧结果 manifest](../results/phase2a/20261008T102400Z/baseline/existing_results_manifest.json)
记录 **3,341 个**结果文件。最终 3,341/3,341文件逐项SHA256完全相同，没有删除/覆盖旧证据。
恢复应在独立目录解压并按 manifest 选择文件；本轮没有恢复 live 工作树。

环境：Python 3.13.12、NumPy 2.4.4、SciPy 1.17.1、pytest 9.0.3、Matplotlib 3.10.9，Linux。
使用 CPU、OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1；正式 Monte Carlo 每配置最多4个进程。
没有引入训练、GPU 或 RL 依赖。Python 3.11 仅由既有 CI matrix 声明，本机未运行3.11；远程 CI 未执行。

全部正式 CLI 的 Python 源码 SHA256：`cc6884eb24d2ab72851c5ac117427ab5bcafee2541b5f1ba4dbd93da4a9f7451`。每例保存 HEAD、branch、
src/scripts 全部文件 hash、配置 hash、命令、seed 与依赖版本。最终完整源码/测试/配置/文档
清单、可恢复快照、相对开发前快照的增量 patch 和资产审计见：
[final source manifest](../results/phase2a/20261008T102400Z/final_source_manifest.json)、
[最终快照](../results/phase2a/20261008T102400Z/phase2a_source.tar.gz)、
[增量 patch](../results/phase2a/20261008T102400Z/phase2a_delta.patch)、
[delivery audit](../results/phase2a/20261008T102400Z/delivery_audit.json)。

## B. State and belief mathematical definition

真实隐藏状态 `s*_k={D*_k,theta*_m,h*_k,...}`。估计器只接收 DIC 位移、协方差、
mask 和公开载荷；初版参数 `theta=[theta_D,theta_f,theta_n]`：
Uniform d0 或六参数 initial Gaussian、fatigue A，以及 tx/ty/theta、load_factor；
Unknown E0 实验额外检查材料尺度歧义。其余 E0、nu、BC、Yref、m、eta、beta_R 按实验契约已知。

\[
b_k(\theta)=p(\theta\mid o_{0:k}),\qquad
\hat\theta_k=\arg\max_\theta b_k(\theta),\qquad
\hat D_k=\mathcal F_{fatigue}(\hat\theta_k,\ell_{0:k}).
\]

`MechanicsBelief`（别名 BeliefState）保存 parameter MAP/局部 Gaussian 中心、physical/normalized
坐标、prior、information rank/弱零方向、协方差状态、当前积分点 damage、N、innovation 与 provenance。
`parameter_mean` 是近似后验中心，而不是数值计算的完整 nonlinear posterior mean。
`current_damage` 是 MAP 处的 physics plug-in，通常不等于 `E[D_k | o_0:k]`。
不估计任意独立 G 维 damage posterior 或一般内部 history 后验。
damage RMSE 在同一固定 mesh 的 G 点上等权计算；DIC RMSE在active点的ux/uy分量上等权计算，
不是精确面积积分，也不把不同mesh的raw damage vector直接比较。

| 对象/标签 | 条件信息与意义 |
|---|---|
| prior MechanicsBelief | 尚未同化 DIC 的声明参数/先验 |
| prediction_before_observation | 新观测到来之前，用之前 prefix 的 physics state 与 predicted DIC |
| filtered_after_observation | 当前观测进入 prefix 后的参数 MAP 和当前 damage |
| post_fit_residual | 观测已用于拟合，不能充当独立 innovation |
| future_open_loop_forecast | 固定 prefix posterior 和公开未来载荷，未加入未来观测 |

新增接口和单位契约见 [11_SEQUENTIAL_BELIEF_CONTRACTS](11_SEQUENTIAL_BELIEF_CONTRACTS.md)。

## C. Physics prediction and observation update

预测实际调用 Phase 1B `FatigueEvolutionSolver`。每个候选 theta 建立与之匹配的 N=0
MaterialHistoryState；不在候选之间转移历史。D 为 quadrature field，V/G/Q 空间保持独立。
缓存键包含完整候选向量和目标时刻；history 还绑定 candidate key，显式传入别的候选 history 会报错。
失败 rollout 不进入 cache。已有 Tri3、DamageLaw、ObservationOperator 与疲劳 solver 的物理实现没有改变。

在每个目标时刻用新 damage 重新求解 `K(D)u=f`，再用 Phase 1C FE→DIC mapping 求 predicted DIC。
单元测试独立记录 FEM request 的 D，验证不是旧位移；与原 FatigueForwardAdapter 以及独立
Phase 1B truth rollout 的位移在数值精度内一致。

积分 anchors 由公开 load segments 和 cycle_step 决定，与观测列表无关。非网格时刻从上一
anchor 建立临时 branch；查询 [1.5,7.2,20] 与只查询20得到完全相同的后续状态。
这是确定性 replay 和 prefix/batch 对照的必要条件，避免观测保存时刻改变时间离散误差。
public diagnostic pattern 描述该观测的实际工况，独立静态 load modes 可在同一 frozen D 下观测。
未来 load_history 是公开输入计划，未来 ObservationDesign 不含实测位移或反力值。

\[
D^-_{k+1}=\mathcal F_{physics}(D_k,\theta_f,\ell_k,\Delta N_k),\quad
\hat o^-_{k+1}=P_{F\to D}u(D^-_{k+1}),\quad
e_{k+1}=o_{k+1}-\hat o^-_{k+1}.
\]

## D. Sequential MAP algorithm

重新优化完整已观测 prefix：

\[
\mathcal J_k(z)=\frac12\sum_{j\le k}\|L_j^{-1}(\hat o_j(z)-o_j)\|^2
+\frac12\|(z-z_{prior})/\sigma_{prior}\|^2,\quad z\in[-1,1]^K.
\]

z 是 Phase 1C 参数箱的 dimensionless 坐标。直接复用 InverseObjective 和原有 SciPy least_squares
TRF、三点数值差分，linear loss，最大120次函数评价，ftol/xtol/gtol=1e-9。
科学实验 cold 初始化统一为 z=0（参数箱中点），不读取真值；公共 API 默认还支持固定 ±.35 的多初值。
Gaussian evolution 的初始化来自已完成的 static inversion，不是真参数。

warm-start 使用上一次 MAP；上一次 posterior 不作为新的 prior。原始 prior 每次只出现一次，
prefix 内每个 evidence ID 只出现一次；重复 ID 拒绝。integration test 验证带 prior 的两次 update
与直接两帧 batch 目标一致，prior Fisher 不累乘。重算旧 likelihood 是数值重优化，不是重复计入信息。

每次优化保存所有 start、cost、nfev、optimality、message、残差评价 history；history 包含试探评价，
不冒充只含接受迭代。只按观测/prior cost 选择已收敛候选；全未收敛但 cost 有限时保留为
UNCONVERGED_TENTATIVE，禁用可信 Gaussian propagation。全 start 失败则 FAILED_ROLLED_BACK，
旧 belief/prefix 不变。相近目标但分离的多 start 解会标记 Laplace 不可靠；单 start 不证明单峰。

Prefix 与原 Phase 1C 独立 batch MAP 对照通过。warm/cold 在本次无噪声情景几乎达到同一目标，
并不保证所有非线性问题相同。带噪 baseline 最后 warm−cold cost 为 −5.684e−13，实际差异完整保留。
没有实现额外 recursive Gaussian filter；优先建立可独立审计的 prefix baseline。

## E. Uncertainty representation

复用已验证的双步长中心灵敏度与 Fisher 诊断。J 是 whitened prediction 对 z 的 Jacobian：

\[
F_{data}=J_w^TJ_w,\quad F_{prior}=\operatorname{diag}(\sigma_{prior}^{-2}),\quad
F_{post}=F_{data}+F_{prior}.
\]

分别保存 data rank、posterior rank、各自奇异值和 null directions、weak directions、
FD error/rank threshold、prior Fisher、posterior covariance 近似状态。
在 square-root information `[Jw; prior_root]` 上 SVD，避免直接对病态 F 普通求逆。
物理协方差通过 ParameterSpace.scales 从 normalized 坐标转换；弱方向也明确在 normalized 坐标中解释。

这是局部 Gauss-Newton/Laplace：忽略 residual-weighted second derivatives，没有全局 Gaussian/单峰保证。
active/near-boundary、未收敛、导数检查失败、分离局部解会标记不可靠；不做 clipping 后声称 Gaussian calibration。
边界用独立标记的二阶单侧、双尺度导数进行诊断，保留 Fisher，但不提供可信 Gaussian propagation。

pre-update/heldout innovation 的 NIS 使用局部预测协方差
`S=R+J C_parameter J.T`；仅 R 白化的残差 RMS 独立保存。缺少 proper covariance 时预测 NIS 为 N/A。
post-fit residual 无 predictive NIS，避免已用于拟合的数据伪装独立检验。

forecast 可用 delta method 传播到 D 和 predicted DIC，或用固定局部 Gaussian 参数样本做完整
nonlinear physics pushforward。采样不创建独立 G 维 posterior；界外参数拒绝、物理失败逐样本记录。
发生拒绝后统计是有界条件 Gaussian 的 diagnostic，不是完整真实 bounded nonlinear posterior。
本次 完成128个参数Gaussian样本，bounds rejection=0，
physical failure=0；D30/40的sample mean−MAP plug-in为
-0.000253029/-0.000393812，
对应sample-mean Monte Carlo SE≈0.000200193/0.000298668。
有限样本均值−plug-in 差还包含 Monte Carlo 误差，不能据此精确量化 Jensen 偏差。

## F. Rank-deficient posterior handling

对于数值秩亏的局部 Gaussian，`covariance_normalized/physical=None`；只保留明确命名的
identifiable-subspace covariance，不能把 pseudo-inverse 当作完整参数 covariance。
任何受到 null space 影响的 marginal Gaussian 标准差为 infinity（JSON 中转换为 null，
同时保存 rank/null directions/status，不表示零或有限小方差）。可以有限估计可辨识坐标组合，
但当前版本不在 null space 上采样或生成虚假场区间。

参数箱可使真实 bounded posterior 有限；上述 infinity 指局部未约束 Gaussian，而不是断言
实际有界 posterior 无限方差。本版没有计算秩亏有界的非 Gaussian posterior。

Uniform E0–damage 的已知解析等价：`E0*(1−d0)` 相同即同样的静态 DIC。
`(E0,d0)=(2000,.2)` 与 `(2400,1−1600/2400)` 位移一致。

| 实验 | Data rank / 2 | Posterior rank / 2 | 解释 |
|---|---:|---:|---|
| 单/多静态 load modes，unknown E0 | 1 | 1 | 优化收敛仍不可唯一辨识；无 full covariance |
| 同一数据 + Gaussian prior | 1 | 2 | prior 提供约束，不是 DIC 新信息 |
| 独立 known-healthy baseline | 2 | 2 | healthy 模态对 E0 提供独立信息 |
| 加 imposed-displacement reaction，未有 healthy baseline | 1 | 1 | 反力仍只测 E0*(1−D)，不消除全局尺度歧义 |

未知载荷幅值与均匀 static damage 则可由独立反力测量约束载荷，再识别 D。
不能把两类反力实验的作用混淆。见 [rank deficiency](../results/phase2a/20261008T102400Z/verified/rank_deficiency/summary.json)。

在固定线性化点、独立 frame noise 下累计 Fisher increment 为 PSD，解析测试通过；
不同 prefix 重新线性化后的 F 或 covariance 不要求逐项单调。

## G. Uniform and Gaussian experiments

Uniform 已知 E0/BC，d0=.2；N=0、10、20 是 frozen damage 的独立观测。单帧及 prefix
恢复误差约8.33e−17，data rank=1。单/两/三帧 SD=.000251385/.000177756/.000145137，
符合独立 likelihood 的 1/sqrt(n)。这些 SD 使用 noiseless likelihood floor，仅是数值例；
真实无噪声的 uncertainty calibration 不从此推断。

六参数 initial Gaussian 使用 Phase 1C 原参数化：d_base=.05、amplitude=.3、xc=1.05 mm、
yc=.47 mm、sx=.35 mm、sy=.22 mm。6×3 mesh（36 Tri3）、64个 arbitrary random DIC points，
在 frozen D 下依次加入 x/y/shear 独立工况。各 prefix rank=6；damage RMSE≈
1.17e−11、1.03e−11、6.86e−12。最弱方向仍报告，满秩不等于任意噪声下稳定恢复。

完成 static prerequisite 后，在4×2 mesh（16 Tri3）演化 initial Gaussian，A=.0015 已知。
仅同化0/10/20；当前 D20 RMSE=2.34e−13，future D30/40 RMSE≈2.32e−13/2.31e−13。
当前场与 initial Gaussian 参数函数的 RMS 差=.0330031；演化结果一直由逐积分点 physics rollout
给出，没有假设形状维持 Gaussian。这个受控 evolving Gaussian 只验证已知 A 的六参数初态，
其未来 uncertainty coverage 未做统计资格，不扩展为同时未知任意 fatigue 参数的恢复声明。

另外 8个Gaussian独立noise seeds（sigma=1e−5 mm，三个静态工况）全部收敛且局部covariance proper；
damage RMSE范围0.000525085–0.00186021。
参数coverage：d_base=100.0%, amplitude=100.0%, xc=75.0%, yc=87.5%, sx=87.5%, sy=87.5%。
该样本量不足以给 Gaussian coverage 独立统计资格。不同工况的多个观测点也不能当作独立随机种子。
见 [static Gaussian](../results/phase2a/20261008T102400Z/verified/gaussian_static/summary.json)、
[Gaussian evolution](../results/phase2a/20261008T102400Z/verified/gaussian_evolution/summary.json)、
[Gaussian noise statistics](../results/phase2a/20261008T102400Z/verified/gaussian_coverage/summary.json)。

## H. Multi-frame fatigue calibration

复用 Phase 1C 验证设置：2×1 mesh（4 Tri3）、16 irregular points，E0=2000 N/mm²、nu=.3、
厚度.8 mm、traction10 N/mm²，R=−1，Yref=.025 N/mm²、m=1、eta=beta_R=0。
actual d0=.12、A=.0015 cycle⁻¹；内部 step=2 cycle，N=[0,10,20,30,40]。

| 已观测 prefix | rank(d0,A) | warm MAP [d0,A] | 当前 damage mean |
|---|---:|---|---:|
| 0 | 1 | [.1200000000009,.00205] | .1200000000009 |
| 0,10 | 2 | [.12,.0015] | .1361569035 |
| 0,10,20 | 2 | [.12,.0015] | .1526002653 |
| 0,10,20,30 | 2 | [.12,.0015] | .1693457867 |
| 0,10,20,30,40 | 2 | [.12,.0015] | .1864106516 |

N=0 的 A=.00205 只是固定初始化代表，SD 为未辨识；不能声称单帧恢复 A。
N>0 的 noiseless parameter error 为数值精度量级。独立 Phase 1C batch 与新 prefix 一致。
最后两行是持续同化之后的 filtered estimates，绝不称为从20的长程预测。
独立 N≤20 estimator 的 D30/40 forecast RMSE≈4.10e−16/6.25e−16，DIC≈4.95e−18/6.37e−18 mm。
真实匹配同一离散模型是受控软件验证，存在 inverse-crime 特性，不是实验材料预测资格。
见 [fatigue sequential](../results/phase2a/20261008T102400Z/verified/fatigue_sequential/summary.json)。

## I. Observation noise and uncertainty coverage

每个模型/情景使用 **100 个**预先固定独立 noise seeds（1000–1099），全部执行，不按结果筛选。
Uniform 六个情景共600次，fatigue 四个情景共400次，另各一个 noiseless case。
不同情景复用 seed 列表便于对照；统计按情景分组，不将这些重复 seed 汇总为1000个独立样本。
统计只对最终0/10/20 prefix 做一次 batch-equivalent MAP，避免每个 seed 重做所有中间优化；
序贯 correctness/history 已由独立实验验证。完整测试不内嵌100-seed重复 FEM。

sigma_x=5e−5 mm、sigma_y=1.25 sigma_x；空间噪声使用 ell=.22 mm、
`B=sigma²[[1,.25],[.25,1.6]]` 的完整 covariance。各 frame 噪声在共享物理/nuisance 给定后独立。
配准偏差=[.0005,−.0003] mm、rotation=.0004 rad，通过全 prefix 共用 tx/ty/theta 估计。
未知 load_factor actual1.1，结合独立反力 uncertainty .01 N 估计；不使用真载荷作为未知参数初值。

下表 coverage 为95%局部边际参数/逐时刻 damage intervals，含失败和不可靠结果的 operational
denominator=100；同时原始 JSON 保存 conditional coverage、Wilson95%精度及 excluded counts。
field points 不计作独立重复。predicted mean observation 的 pointwise coverage 先逐 seed 汇总，
MC standard error 按独立 seed 计算。predictive NIS 使用 R+parameter covariance。

| 模型 | 情景 | d0 coverage | A coverage | nuisance coverage | D30 / D40 coverage | NIS30 / NIS40 |
|---|---|---:|---:|---|---|---|
| Uniform | independent | 93% | N/A | N/A | 93% / 93% | 1.033 / 0.957 |
| Uniform | spatial | 94% | N/A | N/A | 94% / 94% | 1.030 / 0.960 |
| Uniform | registration | 91% | N/A | tx 92%, ty 95%, theta 92% | 91% / 91% | 1.038 / 0.964 |
| Uniform | loading | 96% | N/A | load_factor 92% | 96% / 96% | 1.037 / 0.958 |
| d0+A fatigue | independent | 96% | 94% | N/A | 96% / 95% | 1.022 / 0.951 |
| d0+A fatigue | spatial | 97% | 97% | N/A | 98% / 98% | 1.026 / 0.960 |
| d0+A fatigue | registration | 89% | 94% | tx 92%, ty 95%, theta 92% | 95% / 95% | 1.027 / 0.959 |
| d0+A fatigue | loading | 95% | 94% | load_factor 92% | 98% / 97% | 1.026 / 0.952 |

匹配模型800次拟合全部 converged，proper/reliable local intervals均可计算；错误prior/BC的200次也全部converged，
但科学精度/coverage失效。**fatigue + shared registration 的d0 coverage=89/100，Wilson95%区间
[81.369%,93.746%]未包含标称95%。**该情景不能确认95%校准；因此最终给部分资格。
这不从单个100-seed结果断言实现错误，也未忽略多情景比较的抽样不确定性，但不会用宽gate通过
代替标称校准验证。各参数原始误差、SD和全部种子保留，未追加/筛选seed追求更高coverage。

| 模型 / 情景 | d0 bias | d0 RMSE | A bias (cycle⁻¹) | A RMSE (cycle⁻¹) |
|---|---:|---:|---:|---:|
| Uniform / independent | -0.000189042 | 0.000885923 | N/A | N/A |
| Uniform / spatial | 1.31386e-05 | 0.00128804 | N/A | N/A |
| Uniform / registration | -9.69775e-05 | 0.00232369 | N/A | N/A |
| Uniform / loading | -0.000275701 | 0.00104691 | N/A | N/A |
| fatigue / independent | -0.000185329 | 0.00136667 | 5.81012e-07 | 8.77781e-05 |
| fatigue / spatial | -1.69738e-05 | 0.0020216 | 2.63072e-06 | 0.000130122 |
| fatigue / registration | -9.34054e-05 | 0.00255862 | 1.62646e-07 | 8.86575e-05 |
| fatigue / loading | -0.000272592 | 0.0014123 | 5.92845e-07 | 6.53683e-05 |

正常情景未来normalized NIS均值范围约.951–1.038；这是本noise/模型条件下的检查，不能证明
模型正确。名义95%单个100-seed interval的Wilson精度为数个百分点，其他coverage/Wilson见JSON。

预声明宽 MC gate：匹配模型的 optimizer failure frequency≤.05，参数与forecast damage 的 operational
coverage≥.84；不会把100个种子的有限统计量严格要求为.95。这个宽 gate 不替代原始coverage/Wilson
检查，不证明所有 noise level 都校准。wrong prior/BC 情景不施加“必须覆盖真值”的通过标准。

真实 noise sigma=0 的独立两个例达到精确 recovery；其 finite covariance 来自数值 floor，
不计入 noisy coverage。所有 seed 的全部优化 history、J/F、参数误差、当前 damage、future intervals、
NIS、失败/边界状态及 observation/truth 分离数组保存。
见 [Uniform statistics](../results/phase2a/20261008T102400Z/verified/coverage_uniform/summary.json)、
[fatigue statistics](../results/phase2a/20261008T102400Z/verified/coverage_fatigue/summary.json)。

## J. Boundary/material model discrepancy

共享 registration 不是每帧独立噪声。1点、共享 tx/ty prior 的10次重复例，data rank仍2/3，
posterior rank3/3；d0 SD从.0396036到.0406161，未错误缩至1/sqrt(10)。这里每次重新线性化，
轻微增加不违反固定点 Fisher PSD increment。完整 ROI+validity 选择后的 spatial quadratic
whitened norm²=276.45627105764027，与独立 dense Sigma inverse 计算一致。
见 [likelihood](../results/phase2a/20261008T102400Z/verified/likelihood/summary.json)。

错误 prior 采用 normalized box midpoint、SD=.002 的刻意过强错误 Gaussian；错误 BC truth 的
load_factor=1.1 而估计器固定1，未引入吸收物理误差的任意 field 参数。
100-seed错误prior：d0 bias=0.120437、RMSE=0.120437，
参数/未来damage coverage均0，prefix whitened RMS均值=16.2167。
错误BC：bias=0.0798438、RMSE=0.079847，
coverage也均0；但prefix RMS≈0.988843，未来NIS≈1，
与正确观测模型一样好。这是damage吸收BC尺度错误的解析歧义，innovation不能单独发现它；
需要独立反力、载荷标定或其他外部信息，而不是任意field correction。

material discrepancy truth A=.0015；approximate fixed A=.0005。m_true=2 vs m_estimator=1
代表 hazard 对能量的函数形式从平方到线性，A 可调整但 fixed-m 家族不能精确表示真实演化。
只保留合法 Uniform initial parameter和A，不添加随帧任意损伤 correction。测量 sigma=1e−6 mm。

| 情景 | Prefix whitened RMS | D40 forecast RMSE | DIC40 RMSE (mm) | D30/40 damage interval coverage |
|---|---:|---:|---:|---|
| A 可重新标定，正确 m=1 | 1.22110 | 2.59496e−5 | 1.76441e−7 | 1 / 1 |
| 错误固定 A=.0005 | 53.3662 | .0333633 | .000217906 | 0 / 0 |
| m_true=2，m_fit=1，A 可调 | 1.97771 | .00585290 | 4.26000e−5 | 0 / 0 |

可校准 A 得到 A≈.001500881；错误函数形式则得到 A≈.002029911 的补偿值，仍残留 innovation 和
future bias。两种失配的局部 covariance 数值很稳定甚至 status=local_gaussian，但真值区间失效。
“reliable_for_propagation”只判断局部计算/秩/边界条件，不证明物理模型正确。
上述coverage是一个 realization 的逐时刻诊断，不是100-seed失配概率估计。
本次资格包括识别并保留这种失效证据，**不表示模型失配已解决**。
见 [model discrepancy](../results/phase2a/20261008T102400Z/verified/model_discrepancy/summary.json)。

## K. Forecast evaluation and time leakage audit

固定 split：assimilation N=0/10/20，forecast N=30/40。prefix loader 按公开 schedule 筛选之后
才读 observation files；estimator 的 cutoff 在构建目标前拒绝未来帧。未来接口只接受
ObservationDesign，没有 future DIC、reaction observation value、truth 或 residual 通道。

正式 audit 记录：只打开前3个 observation-only目录→估计→forecast→之后 evaluation 才打开
未来两目录。传入未来 frame 会报错且旧 belief 完全不变。unit test 将 future 路径设为不存在/
禁止访问，prefix 仍可读取；额外 damage 通道使 observation-only loader 报错。
AST audit 确认 inference 无 evaluation、truth、networks、rl、training import。
这是科研接口/流程隔离，不是操作系统 sandbox 或恶意代码安全隔离。

带噪单例，从20预测：D30/40 RMSE=.000777819/.00129753，DIC RMSE=5.07821e−6/8.83623e−6 mm。
其 parameter/posterior 在 forecast 中固定，未来场来自 physics-only rollout；正式100-seed结果见 I。
filtered-current accuracy 和未来开环误差分别保存，不用持续更新的30/40 filtered accuracy代替长期预测。
见 [holdout summary](../results/phase2a/20261008T102400Z/verified/time_holdout/summary.json)、
[time access audit](../results/phase2a/20261008T102400Z/verified/time_holdout/time_access_audit.json)。

## L. Baseline comparison

A 为观察前 z=0 的 physics-only nominal rollout；B 对每帧独立反演 current uniform damage，
不估计 initial d0/A；C cold-start full prefix joint MAP；D previous-MAP warm-start full prefix。
所有参数初始化来自相同声明 box，中点不是 truth。baseline 的噪声是一个固定 realization；
不以此单例估计 coverage，coverage 使用独立 MC 情景。

| Baseline | D20 current RMSE | DIC20 fitted RMSE (mm) | D40 forecast RMSE | DIC40 forecast RMSE (mm) |
|---|---:|---:|---:|---:|
| A physics-only | .290473 | N/A（未拟合） | .326749 | .00371269 |
| B independent current inverse | 3.17359e−6 | 7.35127e−5 | N/A | N/A |
| C prefix cold | .000277864 | 7.35291e−5 | .00129754 | 8.83624e−6 |
| D sequential warm | .000277864 | 7.35291e−5 | .00129753 | 8.83623e−6 |

B 在该单例 D20 比 C/D 更准，不能声称 prefix 在每个 seed 必然改善 current state。
B 没有已识别的 fatigue law，initial-parameter RMSE 和 fatigue forecast 为 N/A；没有制造未来结果。
A 未估计参数或 proper posterior，parameter error/coverage 指标为 N/A。
C/D N20 normalized parameter RMSE≈.016026；空间 held-out 坐标 seed909 的 DIC RMSE≈1.61245e−6 mm。
该 holdout 的实测位移不进入拟合；time holdout和spatial holdout分别标记。

C/D 最终目标≈71.57245654，difference−5.684e−13；cold与warm不是不同后验定义，只是不同初始化。
从N20冻结posterior后才预测30/40。成本、参数与不适用说明逐行保存；额外 recursive estimator未实现。
见 [baseline summary](../results/phase2a/20261008T102400Z/verified/baseline_comparison/summary.json) 和
[比较图](../results/phase2a/20261008T102400Z/verified/baseline_comparison/baseline_current_and_forecast.png)。

## M. Numerical stability and computational cost

boundary d0=0 即使 optimizer converged，也为 unreliable_local_approximation；max_nfev=1保留
UNCONVERGED_TENTATIVE。真实移除所有 Dirichlet BC 引起 Tri3 rigid-body failure，保留
mechanics_diagnostics 和所有 failed history，旧 belief/prefix 原子回滚。另对中途 fatigue endpoint
失败检查 rollout 不缓存，修复后 fresh replay 与未失败独立 rollout 一致。

解析线性Gaussian covariance对照、矩形 Jacobian、多 null directions、prior closure、finite difference
未验证拒绝fullcovariance、固定点PSD信息递增、共享nuisance、确定性 seed/replay 均通过。
边界、未收敛和多峰可疑状态没有伪装成唯一可识别 Gaussian。

noiseless 五前缀的cold耗时约.172/1.470/3.348/5.077/6.726s，warm约.156/1.106/1.484/1.730/2.413s；
对应 forward evaluations cold=122/58/53/54/54，warm=122/39/18/13/14。
这些是该次运行和cache条件的实测成本，不是独立硬件benchmark或在线速度保证。
不同科学配置与pytest同时运行会影响时钟结果，全部命令/线程配置和wall time均保留。
14类正式科学运行总case wall time=716.930s；
Uniform600-fit统计wall=39.572s，fatigue400-fit统计wall=578.493s。
每seed含truth生成、MAP、Fisher与future传播：Uniform正常情景平均约.179–.544s；
fatigue独立/空间/共享配准/未知载荷平均4.203/
4.086/9.166/
5.511s。最多4进程执行，每个BLAS为1线程。

大规模速度不是本轮目标；没有假装一次参数重演相当于在线滤波。缓存不跨参数历史复用。
统计放到独立 CLI；完整pytest只做小seed的严格流程验证，维持可接受回归时间。
见 [stability](../results/phase2a/20261008T102400Z/verified/numerical_stability/summary.json)。

## N. Full tests, scientific matrix and assets

最终完整 `python -m pytest -q -W error`：**357 passed in 329.49s，pytest子进程exit 0**。
54项新增focused tests：54 passed in 66.54s。
另外实际运行Phase 1A四个、Phase 1B八个、Phase 1C十个scientific CLI以及Phase 0 smoke，
**23例全部exit 0**，全部写入新的regression_cli目录。
工具外层pytest会话返回143，与已经完成的pytest子进程exit 0分别记录在
[session observation](../results/phase2a/20261008T102400Z/pytest_session_observation.json)；未推断其原因，
测试结果由完整pytest日志和直接subprocess returncode确认。

科学检查覆盖：无噪声 recovery；单/多帧rank；解析等价不可辨识；prefix/batch一致性；warm/cold差异；
rank-aware covariance稳定性；mask/covariance；future holdout无泄漏；失配diagnostic；100-seed coverage；
边界和未收敛；FE/fatigue rollback；observation-only隔离；deterministic replay；Phase 0–1C全部回归。
原303项测试文件没有改写；新增54项unit/integration/scientific/CLI测试。开发测试首次有两个新测试
断言失败（测试路径包含future字符串、浮点25相差3.55e−15），修正路径判断与舍入容差后54项通过；
该失败日志保留，没有删掉测试或放宽科学恢复/coverage要求。

正式 matrix 全部配置和结果：

| 配置 | 实际状态 | Wall time (s) |
|---|---|---:|
| [uniform](../results/phase2a/20261008T102400Z/verified/uniform/summary.json) | PASSED | 0.235 |
| [fatigue_sequential](../results/phase2a/20261008T102400Z/verified/fatigue_sequential/summary.json) | PASSED | 30.397 |
| [rank_deficiency](../results/phase2a/20261008T102400Z/verified/rank_deficiency/summary.json) | PASSED | 1.109 |
| [gaussian_static](../results/phase2a/20261008T102400Z/verified/gaussian_static/summary.json) | PASSED | 1.621 |
| [gaussian_evolution](../results/phase2a/20261008T102400Z/verified/gaussian_evolution/summary.json) | PASSED | 9.662 |
| [likelihood](../results/phase2a/20261008T102400Z/verified/likelihood/summary.json) | PASSED | 3.703 |
| [time_holdout](../results/phase2a/20261008T102400Z/verified/time_holdout/summary.json) | PASSED | 4.756 |
| [baseline_comparison](../results/phase2a/20261008T102400Z/verified/baseline_comparison/summary.json) | PASSED | 11.068 |
| [model_discrepancy](../results/phase2a/20261008T102400Z/verified/model_discrepancy/summary.json) | PASSED | 10.519 |
| [numerical_stability](../results/phase2a/20261008T102400Z/verified/numerical_stability/summary.json) | PASSED | 0.229 |
| [coverage_uniform](../results/phase2a/20261008T102400Z/verified/coverage_uniform/summary.json) | PASSED | 39.572 |
| [coverage_fatigue](../results/phase2a/20261008T102400Z/verified/coverage_fatigue/summary.json) | PASSED | 578.493 |
| [gaussian_coverage](../results/phase2a/20261008T102400Z/verified/gaussian_coverage/summary.json) | PASSED | 10.255 |
| [uncertainty_pushforward](../results/phase2a/20261008T102400Z/verified/uncertainty_pushforward/summary.json) | PASSED | 15.311 |

[full pytest](../results/phase2a/20261008T102400Z/pytest_full.txt)、
[pytest command metadata](../results/phase2a/20261008T102400Z/pytest_full_metadata.json)、
[Phase 2A matrix](../results/phase2a/20261008T102400Z/verified/matrix_summary.json)、
[原阶段 CLI matrix](../results/phase2a/20261008T102400Z/regression_cli/regression_matrix.json)。
每个输出目录拒绝已有路径，histories、J/F、seeds、配置、summaries、CSV 和独立 PNG/PDF 全部保留。
Qualification 依据正式全预算CLI，不依据pytest的小seed子集。

复现（必须使用新的结果目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fatiguedic-mpl python -m pytest -q -W error
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/fatiguedic-mpl python -W error scripts/verify_phase2a.py --all --output results/reproduce-phase2a
```

Git基线整理建议：经人工review后，将 src、tests、configs、scripts、pyproject、README、
.github、docs（含原阶段报告/契约）纳入版本控制，并按阶段建立清楚的reviewable baseline。
本轮不执行这些Git写操作。大型npz、plots、每seedhistory、tar和派生科学结果保留为外部科研资产，
用 immutable run ID、source/config SHA256、artifact manifest 关联；少量summary/manifest可选择入库。
不要删除旧报告、数值结果或重新写历史；不要把“initial commit HEAD”称为已验证完整基线。

## O. Limitations and unresolved scientific questions

资格只覆盖本文的低维参数模型、noise levels、固定FE离散化与短时间窗口。
高维damage posterior、active internal history、跨mesh传移、真实材料/实验DIC、图像相关、
mesh-objective localization、长时间失效/RUL均未验证。Gaussian evolution只验证已知A，
8-seed Gaussian uncertainty是探索性，不能声称其覆盖率具有100-seed统计精度。
共享配准fatigue情景d0的95%校准未得到确认，这是最终部分资格的具体限制；
完整序贯算法、physics rollout和严格时间留出已完成。deterministic replay只验证相同依赖版本/配置，
未宣称不同数值库版本之间bitwise一致。

局部Gauss-Newton没有完整 nonlinear Hessian、全局多峰搜索或有界 rank-deficient posterior积分。
强 prior、边界、结构歧义、模型/BC误差可使后验过度自信；良好的当前DIC拟合也可能隐藏状态错误。
共享 registration支持显式nuisance，但没有通用跨帧 covariance或未知时间相关noise辨识。
不同观测frame必须是真实独立条件噪声的acquisition，改evidence ID不能把重复同一measurement变成新证据。

matching synthetic truth使用相同离散模型是controlled inverse crime；不能外推到真实疲劳常数、
未知E0+未知law全部组合、所有加载/ROI或噪声范围。未来loads假定公开已知，输入计划本身不确定性
没有统一概率模型。prediction/forecast covariance只有参数近似，未纳入未知模型/时间离散误差。
历史Phase 1B局部软化mesh sensitivity依然存在，本轮没有用平滑或任意损伤参数掩盖。

需要后续研究的是更一般的观测相关性、实验噪声标定、模型discrepancy与状态可辨识性，以及
在这些相同严格protocol下学习估计器是否有效。它们不改变本轮已完成基线的有限资格。

## P. Phase 2B Graph + GRU recommendations

未来 estimator 实现同一 BeliefEstimator Protocol，复用 ObservationBatch、MechanicsBelief、
PredictedObservation、Innovation、ForecastResult 和 TemporalSplit/evaluator。
输入 irregular DIC graph：物理坐标、ux/uy、mask、uncertainty、公开load/time；不能强制H×W。
Observation Graph Encoder处理Q节点，显式FE mapping/feature fusion连接V/G支持，Temporal GRU
维护因果prefix memory；parameter/belief heads首先输出同样的低维初态/疲劳/nuisance和不确定性。
可选等效刚度/条件damage/下一时刻state输出仍需明确空间引用、角色和conditioning。

输出parameter MAP后通过已验证physics decoder获得场；不要把任意G维neural covariance冒充可识别。
neural uncertainty也必须保留数据歧义、边界/失配diagnostics，不能把强training prior称为DIC新信息。
train/validation/test specimen、noise seed和时间split必须预声明；GRU不能读取未来frame、truth或future residual。
用本轮A–D基线、相同observation/noise/holdout、全部失败率与coverage比较，不只比较in-sample位移。

本报告是建议接口和评估边界；**没有实现网络、训练或RL，本轮完成后不自动进入Phase 2B**。
