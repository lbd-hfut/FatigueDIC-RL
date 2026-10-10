# Phase 1C — DIC Observation, Damage Identifiability and Inverse Mechanics

实施日期：2026-10-08。最终资格：**PHASE1C_IDENTIFIABILITY_QUALIFIED**。

资格严格限定为：观测建模、局部可辨识性诊断、受控可辨识参数的经典反演、
以及已定义的数据隔离机制得到验证。**不表示所有损伤参数可辨识，不表示任意高维损伤场
可唯一恢复，不包含真实图像相关、实验噪声标定、真实材料疲劳参数标定、网格客观
局部化、真实裂纹几何、RUL 或学习模型资格。**

303 项测试通过；Phase 1C A–J 十类科学实验独立执行通过。不可辨识的参数组合被正确
检测并解释，属于科学验证成功。带噪反演中明显的损伤恢复误差、边界估计和四个
未收敛初值全部保留，未将优化器收敛解释为真实损伤恢复成功。本阶段完成后停止。

## A. Git baseline and implementation audit

实际 branch：`feature/phase0-architecture`；HEAD：
`dc239d1b7b6dd1e44d93e71bdcccccc0f12e8bc9`；仅一个 worktree。
HEAD 仍是初始提交，**不包含已经验证的 Phase 0–1B 全部工作区代码**。
开始时 README 已修改，`.github/configs/docs/results/scripts/src/tests/pyproject.toml`
等资产仍未提交/未跟踪。没有 commit、push、merge、reset、clean、rebase 或分支切换。

开发前实际读取 Phase 0 验收/实施报告、Phase 1A 验证报告、Phase 1B 全报告与 08/09
契约，审计 Tri3MechanicsSolver、FatigueEvolutionSolver、DICObservation、ObservationOperator、
MaterialHistoryState、QuadratureLayout、field/state/material 契约及 synthetic trajectory storage。
开发前 `python -m pytest -q -W error`：**237 passed in 20.67s，exit 0**。

在任何 Phase 1C 修改前保存 **978 个文件**，逐文件校验 tar 内 SHA256：
[开发前快照](../results/phase1c/baseline/phase01abc_worktree.tar.gz)、
[manifest](../results/phase1c/baseline/manifest.json)、
[Git 状态](../results/phase1c/baseline/git_status.txt)、
[tracked patch](../results/phase1c/baseline/tracked.patch)、
[基线测试](../results/phase1c/baseline/pytest.txt)。人工恢复时应先解压到独立临时目录，
依据 manifest 核对后选择文件；本轮未恢复或建立 Git checkpoint。

新增 observations/{covariance,registration,noise}、inverse/{parameterizations,forward_adapter,
fatigue_adapter,sensitivity,identifiability,objectives,optimizer}、独立 evaluation、科学 CLI、
10 个配置、66 项测试、10 契约及本报告。既有数值实现、19 个测试文件、历史报告均未修改；
**869 个 Phase 1A/1B 结果文件保留原哈希**。旧文件仅增量改 README 与 CI。
见 [交付审计](../results/phase1c/delivery_audit.json)、
[开发增量 patch](../results/phase1c/phase1c_delta.patch) 和
[新增/修改文件清单](../results/phase1c/changed_files.json)。

十个最终科学运行与当前源码一致的 SHA256：
`4d2c228bb95b5f7322a511eeec20cccc52748c846ec00e90697e1e29ab9a8f56`。
每例 summary 保留全部 Python 源文件哈希、配置哈希、HEAD/branch 和运行版本。
旧 HEAD 与未提交源码 hash 共同定义版本，不能仅以 HEAD 复现。

## B. DIC observation model

保留原 DICObservation 的物理坐标/位移接口与 metadata allowlist，以组合方式增量扩展
`DICMeasurement`。任意有限 `[Q,2]` 物理坐标，经 Tri3 形函数映射得到位移：

\[
u_{obs}=P u_{FE}+e_{measurement}+e_{registration}.
\]

不要求 H×W grid；支持 irregular points、ROI、validity、confidence、逐点各向异性误差、
完整空间协方差。active = ROI AND validity。域外点保留 invalid/NaN，不外插；
若将域外点声明为 active，adapter 显式拒绝。已有 Phase 0 m/mm 接口未改写；新反演
边界要求统一 mm，pixel 输入必须经 PixelCalibration 显式转换。

本轮标定比例是**合成试验设置** `0.025 mm/pixel`，不是实验校准结果。
坐标转换包含已知物理原点，位移乘比例，协方差乘比例平方。假定位移轴与物理轴对齐；
未实现透视、相机畸变、双目或真实图像相关。单位不符时报错，不直接相减。

confidence 仅表示采集置信信息；未建立统计校准，因此不作为方差倒数。
独立测试证明改变 confidence 不改变似然权重。数组不可写，并拒绝重启 write flag。
Test A 无噪声 FE→DIC 一致性、pixel/mm 往返、ROI、域外 mask 及存储隔离通过。
证据：[A summary](../results/phase1c/verified/observation_consistency/summary.json)。

## C. Noise, covariance and likelihood verification

协方差按 `[ux0,uy0,ux1,uy1,...]` 排列，单位 mm²。实现：

1. 独立点 Gaussian，支持分量不同方差及逐点 2×2 分量协方差。
2. 不规则坐标空间 Gaussian kernel，`Kij=exp(-||xi-xj||²/(2 ell²))`，
   总协方差 `Sigma=K⊗B`，本例 ell=0.22 mm。
3. 独立记录的平移/小角度旋转配准偏差，不将其偷偷吸收入随机协方差。

采样前检查对称与 PSD。eigen root 只处理通过 PSD 检查后的舍入负特征值。
零噪声得到精确零样本；坏协方差、非法长度、非法标定单位与负噪声显式拒绝。

| sigma (pixel) | sigma_x (mm) | 本轮意义 |
|---:|---:|---|
| 0 | 0 | 无噪声一致性 |
| 0.01 | 0.00025 | 合成测量误差 |
| 0.03 | 0.00075 | 合成测量误差 |
| 0.05 | 0.00125 | 合成测量误差 |

统计试验使用16个 irregular points、`B=sigma²[[1,.3],[.3,2.25]]`，
三个独立种子 101/202/303，每种子 2000 次重复，总计 **6000 次/噪声配置**。
每个种子的误差单独保留，未挑选抽样。相同种子在不同强度下采用 common random
numbers，因此相对协方差误差相同；这不是修改统计生成模型或挑选最优种子。

| 模型 | 合并经验协方差相对 Frobenius 误差 | 白化协方差相对误差 | 最大均值/标准误差 |
|---|---:|---:|---:|
| 独立点/各向异性分量 | 0.069949 | 0.075827 | 2.3753 |
| 空间相关/各向异性分量 | 0.060141 | 0.075827 | 2.3894 |

预先定义的统计检查为协方差误差<0.12、白化误差<0.12、均值标准误差绝对值<5；
全部非零强度通过。零强度保持零。不能从这项合成验证推断实验 DIC 误差统计。
见 [B summary](../results/phase1c/verified/noise_covariance/summary.json)、
[协方差图](../results/phase1c/verified/noise_covariance/covariance_spatial.png)。

对于有效点的 covariance principal submatrix，Cholesky `Sigma=L L^T`，
`z=L^-1(u_pred-u_obs)`，`L_DIC=.5 z^T z`。
完整 Gaussian NLL 加 `.5 logdet(Sigma)+M/2 log(2 pi)`，未显式求大型矩阵逆。
删点后选择协方差子矩阵，不能选择 precision 的子矩阵替代它。
噪声为零时协方差是奇异的，不能定义通常的非奇异 Gaussian 似然；无噪声反演
显式使用 **1e-10 mm² 的算法方差下限**，不称为物理测量噪声。带噪试验使用真实生成
协方差，不加隐式 jitter。Reaction N 残差以自己的 N 标准差独立白化后拼接。

分别保存 unweighted displacement RMSE、DIC weighted half-square、whitened RMSE、完整
DIC NLL、reaction residual、最近邻空间相关诊断、独立 held-out 误差。最近邻指标是
描述性残差相关性，不是经校准的统计检验；强相关/大观测规模近似算法未实现。

## D. Continuous damage parameterizations

所有模型通过物理坐标评价，再采样到该 mesh 的 QuadratureLayout，不按 raw vector 索引
认定位置一致。D 为连续标量损伤；Gaussian center/width 不是裂纹几何参数。

| 模型 | 参数及限制 |
|---|---|
| Uniform | d0∈[0,0.76]，Dmax=0.95 |
| Gaussian | d_base∈[0,.15]、A∈[0,.6]；xc∈[.3,1.7] mm、yc∈[.1,.9] mm；sx∈[.12,.7] mm、sy∈[.08,.45] mm |
| Low-rank | 4/8/16 个坐标定义的 Gaussian basis；base=.25、width=.35 mm；每个系数∈[-.012,.012] |

Gaussian 全局上界 `.15+.6=.75<.95`，矩形参数箱内处处合法。Low-rank 用
`base±K*coefficient_bound` 的最坏界保证处处合法，允许有符号系数。非法候选拒绝，
没有损伤 clipping、不可微投影或未经验证的跨网格历史插值。

基准真值：`[d_base,A,xc,yc,sx,sy]=[.05,.3,1.05,.47,.35,.22]`。
这是合成评估设置，无实验材料含义。低维模型本身是一项强先验，不等同于 G 个自由
积分点损伤，也不能由其恢复成功推断任意高维场可以恢复。
详细物理/数值设置另见 [experiment contract](../results/phase1c/experiment_contract.json)。

## E. Forward sensitivity computation

参数归一化为物理中点加 `p_j * half_bound_width_j`，p∈[-1,1]。
在真实 FE prediction 上采用中心差分，默认 h=1e-3 与 h/2 对比；必要时最多继续细化
三次。每个正负扰动保持合法，边界点不能伪装成中心差分。保存 coarse/fine Jacobian、
实际步长、每列差异、raw mm sensitivity 及 whitened sensitivity。

Gaussian 基准最大列相对差异 **6.285e-7**，中心/宽度扰动稳定。
同时记录保守舍入分辨率估计 `64 eps ||y|| sqrt(K)/min(h/2)`；这是数值启发式，
不是严格 FE 误差上界。它阻止把位移控制下约 1.58e-8 的伪灵敏度当成可识别信号。
所有正式 Fisher 点是受控内点，均通过差分检查。

优化器采用 SciPy 三点数值差分，并对收敛的内点做独立两步长中心检查。
活跃边界上的估计记录为中心检查不适用，而非宣称梯度已验证。
未使用自动微分或任何学习模型。见
[D sensitivity raw](../results/phase1c/verified/gaussian_sensitivity/gaussian_sensitivity.npz)、
[空间敏感度图](../results/phase1c/verified/gaussian_sensitivity/damage_parameter_sensitivity_maps.png)。

## F. Fisher information, rank and uncertainty

`Jw=L^-1 J`，`F=Jw^T Jw`。SVD 分解的是 Jw；图中 singular values 指 Jw 的奇异值，
F 的特征值为它们的平方。默认数值秩阈值：
`max(1e-6*smax,5*FD_error_resolution,1e-9)`。
summary 的 condition_number 为 Jw 条件数；F 的条件数是其平方，秩不足时均为无穷。
JSON 用 null 表示无穷，并保留 rank、阈值及零空间，不能误读成有限条件数。

| Gaussian，已知 E0/BC，Q=64，sigma=.00025 mm | 数值 |
|---|---|
| Jw singular values | 45.3284, 35.9655, 10.7702, 3.3434, 0.9055, 0.6840 |
| numerical rank | 6/6 |
| condition(Jw) | 66.2697 |
| condition(F) | 约 4391.68 |
| singular values>1 的方向数 | 4/6 |
| amplitude–sx sensitivity cosine | 0.966002 |
| amplitude–sy sensitivity cosine | 0.995786 |

最弱方向混合 baseline、amplitude、sx、sy，约为
`[-.423,-.376,-.014,.018,.337,.752]`。这说明中心较易区分，而幅值/宽度/基线
容易相互补偿。局部维度化标准差分别约
`[.867,.691,.107,.067,.885,1.195]`；例如 sy 的物理局部标准差约 .221 mm，
接近真实 sy=.22 mm，说明满数值秩仍不等于稳定恢复。

协方差通过 SVD 在 identifiable subspace 组装，不普通逆秩亏 F。
秩亏时明确报告 null directions，不提供虚假的完整参数边际方差。
相关热图是敏感度列 cosine，非 posterior correlation。标准差是局部线性噪声近似，
不是含边界/非线性/模型误差的完整置信区域；本轮未证明全局唯一性。
见 [相关图](../results/phase1c/verified/gaussian_sensitivity/identifiability_correlation_heatmap.png)。

## G. E0–damage confounding

真实 FE 矩形板，E0=2000 N/mm²、均匀 D=.2，`Eeff=1600 N/mm²`。
等价 `(E0,D)` 包括 `(1800,.111111)`、`(2000,.2)`、`(2500,.36)`、
`(2800,.428571)`；DIC prediction 最大差仅 **9.823e-17 mm**。
联合未知参数 Fisher rank=1/2；数值零方向与解析 `d[(1-D)E0]=0` 对齐程度
0.9999999999999999。

优化器返回的不同等价解例如 `(2287.894,.300667)` 与 `(2243.430,.286806)`，
均有近零残差。这些不是唯一的真实参数。已知 E0 时恢复 D=.2；增加独立、已知健康
D=0 的基线后 rank=2/2，可恢复 E0 与 D。**健康状态 D=0 必须有外部依据**，不是用同一
DIC 数据自行证明健康后再当作已知信息。

| 工况/未知量 | rank | 科学结论 |
|---|---:|---|
| 载荷控制，E0/D 都未知 | 1/2 | 只能辨认等效刚度尺度 |
| 已知 E0，载荷控制 | 1/1 | 受控 Uniform D 可恢复 |
| 健康 baseline + damaged，E0/D 未知 | 2/2 | 增加外部健康信息后可分离 |
| 位移控制，已知 E0，无反力 | 0/1 | 均匀全局损伤尺度不可观测 |
| 位移控制，已知 E0，加反力 | 1/1 | 反力给出全局刚度尺度 |
| 位移控制，E0/D 未知，加反力 | 1/2 | 仍只能辨认 Eeff |

Gaussian 的未知 E0 也有共同 stiffness scale 退化：改变 base/amplitude，使
`E0'(1-D'(x))=E0(1-D(x))`，中心与宽度不变。三独立载荷仍 rank=6/7。
未知 E0 反演例 D field RMSE=.024074，而 DIC RMSE≈9.995e-18 mm；明确验证了
低位移误差不能保证真实损伤恢复。见
[C summary](../results/phase1c/verified/uniform_confounding/summary.json)、
[等价解图](../results/phase1c/verified/uniform_confounding/equivalent_parameter_solutions.png)。

## H. Observation density and distribution

完整执行 `16/64/256 points × regular/random/clustered × missing=0/.1/.3 × 3 seeds`，
共 **81 组**。missing 是逐点 Bernoulli 概率，实际缺失数量有随机波动，未选最优 mask。
所有组保存 map、J、Fisher、零空间/弱方向和实际 active 数。

无缺失时最小 whitened singular value 的三种子中位数：

| Q | regular | uniform-random | clustered |
|---:|---:|---:|---:|
| 16 | .34477 | .26735 | .00636 |
| 64 | .70099 | .63653 | .01251 |
| 256 | 1.39686 | 1.39529 | .02855 |

六参数 Gaussian 本例 81 组均 rank=6，但 restricted cluster 的实用分辨率显著差；
Q=64 clustered 只有1个 singular value>1，regular 有5个。增加重复/近邻点数不能替代
空间覆盖。4/8/16 basis 的 clustered 试验 numerical rank 分别 4/7/7，且最大 singular
value 全部小于1；其弱方向并不能推广为高维可靠恢复。局部 Tri3 分片线性采样还限制
了同一小 ROI 内的独立位移信息。
见 [E summary](../results/phase1c/verified/sampling_density/summary.json)、
[采样图](../results/phase1c/verified/sampling_density/observation_sampling_maps.png)。

## I. Multiple loading conditions

同一冻结损伤的静态 x tension、y tension、top shear 诊断，不是 Phase 1B 的非比例循环。
单模式与其幅值×2 组合满足 `Jstack=[J;2J]`，差分检查相对偏差精确为0；
相同绝对噪声下 F 增至5F，信噪比提高，但 rank 和右敏感方向不增加。

| 六参数 Gaussian | rank | condition(Jw) | 最小 singular value |
|---|---:|---:|---:|
| single | 6 | 66.2697 | .6840 |
| same mode amplitude scaling | 6 | 66.2697 | 1.5295 |
| three independent modes | 6 | 约29.129 | 2.4894 |

稀疏 Q=4、16-coefficient basis：单工况 rank=8/16，独立三工况 rank=16/16。
但该三工况最小 singular value≈4.106e-6，仅2个方向>1；**增加数值秩没有验证
16个系数稳定恢复**。Gaussian+未知 E0 在多工况下依然 rank=6/7。
Reaction 对均匀尺度/边界幅值的作用已单独在 C/H 检验，不把反力等同于新的全场位移。
见 [F summary](../results/phase1c/verified/multiple_loading/summary.json)。

## J. Classical inversion baseline and truth isolation

采用 SciPy least_squares/TRF，有界 dimensionless parameters、weighted residual、可选
Gaussian prior、三组固定初值。默认 linear loss；接口另支持 Huber/soft-L1，本报告未对
robust loss 建立实验统计资格。最大120次函数评价，ftol/xtol/gtol=1e-9。
history 保存每次残差评价，不冒充只含接受的迭代。FEM failure 中止该初值并保留后端
diagnostics；best 由观测/先验目标选择，不由真值误差选择，不证明 global minimum。

独立 held-out 坐标 seed=909，不进入 optimization。损伤误差在公共256个物理坐标上
评价同一连续函数，非不同 mesh 的 raw vector 索引差。RMSE 是等权点均方根，不称为
精确面积积分。固定 mesh 的无噪声16/64点六参数恢复：damage RMSE≈2.34e-12/1.18e-12，
归一化参数 RMSE≈1.80e-11/9.42e-12，三个初值均收敛。

所有三种子带噪 damage field RMSE 范围：

| Noise (pixel) | single mode | three modes |
|---:|---:|---:|
| .01 | .02704–.06678 | .02173–.02632 |
| .03 | .08231–.15020 | .03809–.06773 |
| .05 | .12236–.20087 | .05992–.13235 |

这些结果是**参数恢复失败/不稳定的实际证据**，未设置为必须低参数误差的通过条件。
noise-aware whitened residual 符合生成量级仍不能消除弱方向。独立空间模式在此多数
配置中改善恢复，.05 pixel/seed101 并未改善 damage RMSE，不能声称对所有种子必然改善。
三种子的 spatially correlated .03 pixel 多工况 RMSE 为 .03828/.04706/.02529，
使用完整协方差，不将相关点作为独立证据计数。

共 **25 组反演、75 个初值**：71收敛、4达到评价上限未收敛；45个候选触及参数边界。
30个内点全部通过独立中心差分检查；边界候选如实标记不适用。所有初值、失败消息、
objective、参数、history 均保留。未知 E0 的等价解不作为唯一恢复成功。
一例 bounds-midpoint prior（无真值输入）令 damage RMSE=.03165；先验增加模型假设，
不增加 DIC 的信息秩，也可能引入偏差。

Observation-only loader 仅返回 coordinates/displacement/validity/ROI/confidence/covariance、
公开 load reference 与采集信息；拒绝额外 damage/parameters/stress/strain 通道。
`SyntheticTruth` 位于独立 evaluation package，存真实 damage/parameters、exact nodal
displacement、strain/stress/reaction 和 mesh/layout；不由 inverse package 导入。
optimizer 的 objective 只持有公开 forward adapter，不接受 truth。公开 pattern 必须为
零 damage，无 history；未知 E0 时保留的 E0 被替换为参数箱中点2100，而不是true2000，
每次解算再用 candidate E0 替换。已知 E0 任务明确把2000作为已知输入。

自动化检查覆盖严格存储 schema、恶意额外标签、import graph、nonzero hidden pattern
拒绝、unknown E metadata 清除和失败状态不变。该机制是科研接口隔离，不是操作系统
安全隔离。科学评价可以在 optimizer 返回后读 truth。
见 [G summary](../results/phase1c/verified/inverse_recovery/summary.json)、
[参数恢复误差图](../results/phase1c/verified/inverse_recovery/parameter_recovery_errors.png)、
[全部初值图](../results/phase1c/verified/inverse_recovery/multistart_optimization_distribution.png)。

## K. Single-frame and multi-frame comparison

单帧受控恢复完成后，采用小网格2×1、4个Tri3、16个DIC点，识别初始均匀损伤 d0
及合成 fatigue A。E0、BC、Yref=.025 N/mm²、m=1、eta=beta_R=0 均已知。
真值 d0=.12、A=.0015 cycle⁻¹，R=-1；每次 candidate 从新 committed history 开始，
实际调用 Phase 1B solver，固定内部 increment=2 cycle，保存观测 N=[0,10,20,30,40]。

| 观测 | rank(d0,A) | 解释 |
|---|---:|---|
| N=0 single frame | 1/2 | A 不影响初态位移 |
| N=20 single frame | 1/2 | 单一当前刚度不能分离初态与演化参数 |
| N=0,10,20 prefix joint | 2/2 | 在已知模型/输入下新增时间信息 |

**optimizer 只持有 N≤20 的三个 calibration cases**。N=30/40 位于另一 evaluation adapter，
严格按时间留出，未用来选择参数或初值。无噪声 prefix 恢复 d0/A，未来位移 RMSE
4.216e-18 mm；三个 .01 pixel 种子分别约7.038e-5、1.757e-4、1.837e-5 mm。
A 的噪声估计为 .001768/.000797/.001527，说明局部满秩依然有有限精度。

这里区分 current-state inversion、initial/law calibration、withheld future-observation
evaluation。未来是同一已知合成模型下的严格留出检查，不是实验预测或普适寿命预测；
不把已看到的后续帧叫未来。未知 E0/未知全部 fatigue 参数的多帧问题未验证。
内部 ΔN=2 与1在真参数下的位移 RMS 差5.246e-7 mm，单独报告时间离散影响。

另保存独立 Phase 1B 完整轨迹 [multiframe_forward_truth](../results/phase1c/multiframe_forward_truth/)，
包括21帧的 D/u/strain/stress/Ya/hazard/reaction/energy/internal history；仅科学评价访问。
其 N=0/10/20/30/40 观测与 multi-frame adapter exact predictions 完全一致，见
[fixture audit](../results/phase1c/multiframe_fixture_audit.json)。
见 [I summary](../results/phase1c/verified/multiframe/summary.json)、
[时间留出图](../results/phase1c/verified/multiframe/multiframe_future_holdout.png)。

## L. Boundary uncertainty and nuisance parameters

载荷真实 factor=1.1、名义pattern=1、真实D=.2。强制认为载荷无偏时，反演得到
`Dfit=1-.8/1.1=.272727`，位移 RMSE≈2.84e-17 mm；这是错误损伤吸收 BC error。
显式同时未知 load_factor/D 时 rank=1/2。增加独立 N 单位力观测后 rank=2/2，恢复
load_factor≈1.1、D≈.2，不强制全部误差归因于损伤。

模拟 translation=[.0005,-.0003] mm、theta=.0004 rad；忽略 registration 得 D≈.225369，
RMSE≈.0002314 mm。将 tx/ty/theta 作为显式 nuisance，rank=4/4，无噪声恢复其与D。
本例空间点覆盖较广；不能推断在小ROI/所有载荷模式下 nuisance 都可分离。
生成配准值只进入评估设置，不作为未知 optimizer 的初始化真值。
见 [H summary](../results/phase1c/verified/boundary_uncertainty/summary.json)。

## M. Fixed-mesh and cross-mesh sensitivity

在各 mesh 积分点独立采样同一连续 Gaussian。reference 为16×8、G=256；
识别 mesh 为4×2/8×4/12×6、G=16/64/144。先比较 centroid initial-field quadrature
积分与解析 Gaussian 积分，再比较公共DIC坐标 forward response，再分析反演偏差。

| G | 初始 damage 积分绝对误差 (mm²) | FE观测差 (mm) | matched-mesh D RMSE | cross-mesh D RMSE |
|---:|---:|---:|---:|---:|
| 16 | .0034104 | 6.2091e-5 | 6.38e-13 | .026339 |
| 64 | .0007107 | 1.9353e-5 | 3.28e-12 | .016666 |
| 144 | .0002810 | 5.3157e-6 | 5.26e-12 | .003951 |

coarse inverse 得到幅值.378985，fine inverse .314855，真实.3；model mismatch 会被
参数补偿，不能将其归因于 optimizer 错误或 DIC 噪声。本例误差随细化下降，只有三档
mesh 和有限 reference，不证明普遍空间收敛。固定mesh恢复近精确属于受控软件验证，
具有相同离散模型的 inverse-crime 特性；跨mesh结果单独披露。

本轮冻结初态 Gaussian 的空间诊断不消除 Phase 1B 局部软化的演化 mesh sensitivity。
未做插值平滑、非局部/梯度正则化或历史状态转移；不宣称网格客观损伤局部化。
见 [J summary](../results/phase1c/verified/mesh_diagnostics/summary.json)。

## N. Tests, scientific matrix, artifacts and reproducibility

最终执行 `python -m pytest -q -W error`：**303 passed in 215.22s，exit 0**。
开发前237项全部保留，新增66项独立 unit/integration/scientific 测试。
包括真实 singular FE failure diagnostics、参数/单位拒绝、storage isolation、中心差分、
零空间、confidence、rollback/public-state invariance、CLI拒绝覆盖，以及完整A–J实验。
见 [完整测试日志](../results/phase1c/pytest.txt)。未修改旧测试标准。
wheel 以 no-deps/no-build-isolation 构建通过，核验新增 observation/inverse/evaluation
模块包含在安装包中；没有安装或升级运行依赖。见 [package audit](../results/phase1c/package_audit.json)。
故意欠约束 FE 的失败、完整 backend diagnostics、公开 damage/N 不变及无部分保存的
检查另保存在 [failure probe](../results/phase1c/failure_probe.json)。

独立执行 Phase1C 全部10个配置，均exit0；另外 Phase0 smoke、Phase1A全部4例、Phase1B
全部8例也独立复验exit0，结果写入新目录，未覆盖原资产。
见 [1C run index](../results/phase1c/verified/run_index.json)、
[历史阶段复验](../results/phase1c/regression_cli/run_index.json)。

| 实验 | 程序/诊断成功 | 恢复失败或不能唯一恢复的情形 | 不适用/未验证 |
|---|---|---|---|
| A observation consistency | PASS | active域外点按契约拒绝 | 真实图像相关 |
| B noise covariance | PASS，重复统计 | 奇异cov不能定义通常likelihood | 实验噪声校准 |
| C E0–D | PASS，检测解析退化 | 未知E0/D；位移控制无反力 | 全局非线性唯一性 |
| D Gaussian/basis sensitivity | PASS，FD稳定 | A=0的中心/宽度；clustered高维弱/零方向 | 任意高维恢复 |
| E density/distribution | PASS，81组完整 | clustered及缺失造成弱可观测 | 任意ROI的恢复保证 |
| F multiple loading | PASS，模式/rank区分 | unknownE共尺度退化；满数值秩仍弱 | 非比例fatigue路径 |
| G classical inverse | PASS，无噪声受控恢复 | 多组带噪损伤不稳定，4初值未收敛，unknownE等价解 | 所有噪声水平可靠恢复 |
| H boundary uncertainty | PASS，nuisance与力观测 | 忽略BC/配准得到错误损伤；load/D混淆 | 任意BC误差模型 |
| I single/multi-frame | PASS，prefix/time holdout | 单帧d0/A；噪声A估计误差 | 未知E0及完整law联合识别 |
| J mesh diagnostics | PASS，物理坐标比较 | crossmesh参数偏差 | mesh-objective局部化 |

正式 `verified/` 含 **525 个NPZ、3201个数值数组、29张PNG及对应29张SVG**。
raw数值全部检查：有效状态/参数/位移无NaN/Inf；仅两个观测数组含声明的域外NaN。
sampled D 范围0–.687466，均在合法范围。五张关键图实际打开核验轴、单位、可读性和
共有色标；图为Matplotlib独立科研文件，未依赖网站。见
[numerical audit](../results/phase1c/numerical_audit.json)。

独立重放 noise covariance 的9个NPZ和multi-frame的28个NPZ，所有数值数组完全一致。
再经 observation-only loader 读取固定带噪六参数case，重复三个初值反演，最大
dimensionless parameter difference=0。见 [replay audit](../results/phase1c/replay/replay_audit.json)。
这证明本机/记录版本确定性，不宣称跨BLAS/平台逐bit一致。

开发中的 MaterialAssignment 字段适配错误以及舍入零灵敏度被误判为rank的失败，
保留在 `development/`、`development2/` summary/traceback；前者已改为真实
constitutive_model 字段，后者以记录的FD/舍入分辨率修正，不通过放宽旧测试消除。
最终所有summary源码hash一致，原始失败结果未删除。

复现（输出必须是新目录；OPENBLAS线程数只是确定性/小矩阵性能设置）：

```bash
python -m pip install -e '.[test,verification]'
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fatiguedic-mpl python -m pytest -q -W error
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fatiguedic-mpl \
  python -W error scripts/verify_phase1c.py \
  --config configs/phase1c/uniform_confounding.toml --output results/my-1c-confounding
```

全部配置与参数边界、载荷/单位、known/unknown设定、优化与FD配置、seed、内部cycle
increments在源码、各例summary、TOML、experiment_contract及10契约中保存。
源码和配置hash均存于结果，图表结论以误差表和rawNPZ为依据。

## O. Unresolved scientific risks

1. Fisher仅局部线性分析；有限差分误差/数值阈值影响effective rank，没有非线性全局唯一性证明。
2. 真正的E0/D共尺度退化不能靠优化器、重复帧数量或同模式幅值缩放自动消除。
3. Gaussian/basis先验很强；低位移误差、满数值秩、optimizer收敛都不足以保证损伤场准确。
4. 高噪声/clustered ROI 的估计触及边界，非线性/有界posterior与局部Fisher不同；未给出完整置信区域。
5. BC幅值、配准、材料模型、Poisson ratio及相机模型的真实不确定性比本轮参数模型更复杂。
6. spatial covariance为小规模dense算法；未验证大规模block/low-rank近似及实验残差相关检验。
7. matching mesh/time model属于受控软件实验，不能证明真实模型正确；crossmesh偏差显著。
8. Phase1B局部损伤演化仍有mesh dependence；Gaussian连续评价不等于非局部正则化。
9. 多帧只识别known E0/BC下的d0和合成A，未联合识别完整fatigue law或非比例循环。
10. 这是synthetic DIC-like displacement，不是实验校准噪声、材料参数或实际裂纹测量。

## P. Recommendations for Phase 2 state estimator design

未来状态估计器应输出受观测支持的低维参数/等效刚度及不确定性，保留弱方向和nuisance，
避免从单一DIC frame强制给出唯一G维损伤标签。E0未知时优先估计Eeff，或引入独立健康
基线、材料试验/力测量；位移控制需要反力信息。观测设计优先空间覆盖和独立load mode，
幅值缩放只能提高SNR。多帧只使用已观测prefix校准，未来时间保持严格留出。

继续保留observation-only边界，真值只用于隔离的评价；任何学习模型都应与本轮TRF、
Fisher及等价解集合比较，不能以更低DIC误差当作更准确damage的证据。真实数据接入前
需要实验噪声/配准/BC标定、covariance validation及模型差异研究。高维局部化估计前
应研究物理regularization与mesh dependence，再定义可验证的状态/标签空间。

本阶段没有开展 Phase 2、GNN、GRU、Actor-Critic 或 RL training。
